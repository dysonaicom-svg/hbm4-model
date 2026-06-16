"""
Bandwidth Benchmark Module

Tests memory bandwidth under various conditions:
- Peak bandwidth (ideal conditions)
- Sustained bandwidth (continuous traffic)
- Refresh overhead (bandwidth loss during refresh)

References:
- JEDEC JESD270-4A HBM4 specification
- Synopsys HBM4 bandwidth specifications
"""

import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time

from model.dram.hbm4_spec import HBM4Spec, HBM4_SPEED_GRADES, calculate_bandwidth
from model.dram.timing import HBM4Timing, get_timing_for_speed_grade, HBM3Timing
from model.controller.hbm4_controller import HBM4Controller
from model.controller.request import HBMRequest, HBMResponse
from .benchmark_config import BandwidthConfig, TestPattern


_logger = logging.getLogger(__name__)


@dataclass
class BandwidthResult:
    """Results from bandwidth benchmark"""
    # Peak bandwidth metrics
    peak_bandwidth_gbs: float = 0.0           # Theoretical peak GB/s
    measured_bandwidth_gbs: float = 0.0       # Actual measured GB/s
    peak_efficiency_percent: float = 0.0      # Measured / Peak as %
    
    # Sustained bandwidth metrics
    sustained_bandwidth_gbs: float = 0.0
    sustained_efficiency_percent: float = 0.0
    bandwidth_variance_percent: float = 0.0   # Variance over time
    
    # Refresh overhead metrics
    refresh_overhead_percent: float = 0.0
    refresh_count: int = 0
    refresh_total_time_ns: float = 0.0
    
    # Detailed breakdown
    total_requests: int = 0
    read_requests: int = 0
    write_requests: int = 0
    total_bytes: int = 0
    test_duration_ns: float = 0.0
    
    # Per-channel breakdown
    channel_bandwidth: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting"""
        return {
            'peak_bandwidth_gbs': self.peak_bandwidth_gbs,
            'measured_bandwidth_gbs': self.measured_bandwidth_gbs,
            'peak_efficiency_percent': self.peak_efficiency_percent,
            'sustained_bandwidth_gbs': self.sustained_bandwidth_gbs,
            'sustained_efficiency_percent': self.sustained_efficiency_percent,
            'bandwidth_variance_percent': self.bandwidth_variance_percent,
            'refresh_overhead_percent': self.refresh_overhead_percent,
            'refresh_count': self.refresh_count,
            'total_requests': self.total_requests,
            'read_requests': self.read_requests,
            'write_requests': self.write_requests,
            'total_bytes': self.total_bytes,
            'test_duration_ns': self.test_duration_ns,
        }
    
    def __str__(self) -> str:
        return (
            f"Bandwidth Results:\n"
            f"  Peak:     {self.peak_bandwidth_gbs:.1f} GB/s (eff: {self.peak_efficiency_percent:.1f}%)\n"
            f"  Measured: {self.measured_bandwidth_gbs:.1f} GB/s\n"
            f"  Sustained: {self.sustained_bandwidth_gbs:.1f} GB/s (eff: {self.sustained_efficiency_percent:.1f}%)\n"
            f"  Refresh overhead: {self.refresh_overhead_percent:.2f}%\n"
            f"  Total requests: {self.total_requests} ({self.read_requests} reads, {self.write_requests} writes)"
        )


class BandwidthBenchmark:
    """Bandwidth benchmarking for HBM memory controllers"""
    
    def __init__(
        self,
        config: Optional[BandwidthConfig] = None,
        speed_grade: str = "8Gbps"
    ):
        """Initialize bandwidth benchmark
        
        Args:
            config: Bandwidth configuration (uses default if None)
            speed_grade: HBM speed grade ("8Gbps", "12Gbps", "16Gbps")
        """
        self.config = config or BandwidthConfig()
        self.speed_grade = speed_grade
        
        # Create HBM4 specification for this speed grade
        self.spec = self._create_spec()
        self.timing = get_timing_for_speed_grade(speed_grade)
        
        # Controller under test
        self.controller: Optional[HBM4Controller] = None
        
        # Results storage
        self.results: List[BandwidthResult] = []
    
    def _create_spec(self) -> HBM4Spec:
        """Create HBM4 specification for speed grade"""
        if self.speed_grade not in HBM4_SPEED_GRADES:
            raise ValueError(f"Unknown speed grade: {self.speed_grade}")
        
        grade_params = HBM4_SPEED_GRADES[self.speed_grade]
        return HBM4Spec(
            data_rate_gtps=grade_params["data_rate_gtps"],
            tCK_ps=grade_params["tCK_ps"]
        )
    
    def _generate_addresses(self, pattern: TestPattern, count: int) -> List[int]:
        """Generate addresses based on traffic pattern
        
        Args:
            pattern: Traffic pattern
            count: Number of addresses to generate
            
        Returns:
            List of addresses
        """
        addr_start = self.config.address_range_start
        addr_end = self.config.address_range_end
        
        if pattern == TestPattern.SEQUENTIAL:
            # Consecutive addresses
            return [(addr_start + i * self.config.request_size_bytes) % addr_end 
                    for i in range(count)]
        
        elif pattern == TestPattern.RANDOM:
            # Random addresses
            random.seed(42)  # Reproducibility
            return [random.randint(addr_start, addr_end - 1) for _ in range(count)]
        
        elif pattern == TestPattern.STRIDED:
            # Fixed stride
            return [(addr_start + i * self.config.stride_bytes) % addr_end 
                    for i in range(count)]
        
        elif pattern == TestPattern.HOTSPOT:
            # 90% accesses to 20% of address space (increased for test stability)
            hotspot_size = (addr_end - addr_start) // 5
            hotspot_start = addr_start
            results = []
            random.seed(42)
            for i in range(count):
                if random.random() < 0.90:
                    results.append(hotspot_start + random.randint(0, hotspot_size - 1))
                else:
                    results.append(addr_start + random.randint(hotspot_size, addr_end - 1))
            return results
        
        elif pattern == TestPattern.ROW_HIT:
            # All accesses to same row (best case for row hit rate)
            return [addr_start] * count
        
        elif pattern == TestPattern.BANK_CONFLICT:
            # Intentional bank conflicts (worst case)
            # Spread across different banks
            bank_size = (addr_end - addr_start) // 32
            return [(addr_start + i * bank_size) for i in range(count)]
        
        else:
            return [addr_start + i * self.config.request_size_bytes for i in range(count)]
    
    def run_peak_bandwidth_test(self) -> BandwidthResult:
        """Run peak bandwidth test (ideal conditions)
        
        Uses sequential access pattern with row hits to achieve peak bandwidth.
        """
        _logger.info("Running peak bandwidth test...")
        
        # Create controller
        self.controller = HBM4Controller(
            spec=self.spec,
            enable_qos=False,  # Disable QoS for pure bandwidth test
            enable_refresh=False  # Disable refresh for peak bandwidth
        )
        
        # Generate sequential addresses (row hit pattern)
        addresses = self._generate_addresses(TestPattern.SEQUENTIAL, 
                                              self.config.request_batch_size)
        
        start_time = time.perf_counter()
        sim_start_ns = self.controller.current_time_ns
        bytes_transferred = 0
        read_count = 0
        write_count = 0
        batch_bandwidths = []
        
        # Run batches
        for batch in range(self.config.num_batches):
            # Submit batch
            for i, addr in enumerate(addresses):
                is_read = (i % 100) < (self.config.read_write_ratio * 100)
                self.controller.submit_request(
                    addr=addr,
                    is_read=is_read,
                    size_bytes=self.config.request_size_bytes
                )
                if is_read:
                    read_count += 1
                else:
                    write_count += 1
            
            # Process requests until complete
            while (len(self.controller.queue_manager.read_queue) > 0 or 
                   len(self.controller.queue_manager.write_queue) > 0):
                self.controller.tick()
                # Track bandwidth every batch
                if len(self.controller.queue_manager.read_queue) == 0 and \
                   len(self.controller.queue_manager.write_queue) == 0:
                    bytes_transferred += len(addresses) * self.config.request_size_bytes
            
            # Calculate batch bandwidth
            elapsed_ns = self.controller.current_time_ns - sim_start_ns
            if elapsed_ns > 0:
                batch_bw = (bytes_transferred / elapsed_ns) * 1000  # GB/s
                batch_bandwidths.append(batch_bw)
        
        sim_end_ns = self.controller.current_time_ns
        elapsed_ns = sim_end_ns - sim_start_ns
        
        # Calculate results
        result = BandwidthResult()
        result.peak_bandwidth_gbs = self.spec.bandwidth_gbs
        result.measured_bandwidth_gbs = (bytes_transferred / elapsed_ns) * 1000 if elapsed_ns > 0 else 0
        result.peak_efficiency_percent = (result.measured_bandwidth_gbs / result.peak_bandwidth_gbs * 100 
                                         if result.peak_bandwidth_gbs > 0 else 0)
        result.total_requests = read_count + write_count
        result.read_requests = read_count
        result.write_requests = write_count
        result.total_bytes = bytes_transferred
        result.test_duration_ns = elapsed_ns
        
        # Calculate bandwidth variance
        if len(batch_bandwidths) > 1:
            mean_bw = sum(batch_bandwidths) / len(batch_bandwidths)
            variance = sum((bw - mean_bw) ** 2 for bw in batch_bandwidths) / len(batch_bandwidths)
            result.bandwidth_variance_percent = (variance ** 0.5 / mean_bw * 100 
                                                if mean_bw > 0 else 0)
        
        _logger.info(f"Peak bandwidth test complete: {result.measured_bandwidth_gbs:.1f} GB/s "
                    f"({result.peak_efficiency_percent:.1f}% efficiency)")
        
        return result
    
    def run_sustained_bandwidth_test(self) -> BandwidthResult:
        """Run sustained bandwidth test (continuous traffic over time)
        
        Tracks bandwidth over time to measure sustained performance.
        """
        _logger.info("Running sustained bandwidth test...")
        
        # Create controller with refresh enabled
        self.controller = HBM4Controller(
            spec=self.spec,
            enable_qos=True,
            enable_refresh=True
        )
        
        # Generate addresses
        addresses = self._generate_addresses(self.config.pattern, 
                                            self.config.request_batch_size)
        
        sim_start_ns = self.controller.current_time_ns
        bytes_transferred = 0
        read_count = 0
        write_count = 0
        
        # Time windows for bandwidth tracking
        window_size_ns = self.config.test_duration_ns // 10
        window_bytes = []
        window_start_ns = sim_start_ns
        
        # Run simulation
        total_batches = (self.config.test_duration_ns // 
                        (self.config.request_batch_size * 1000))  # Rough estimate
        
        for batch in range(min(total_batches, self.config.num_batches)):
            # Submit batch
            batch_start = self.controller.current_time_ns
            for i, addr in enumerate(addresses):
                is_read = (i % 100) < (self.config.read_write_ratio * 100)
                self.controller.submit_request(
                    addr=addr,
                    is_read=is_read,
                    size_bytes=self.config.request_size_bytes
                )
                if is_read:
                    read_count += 1
                else:
                    write_count += 1
            
            # Process requests
            while (len(self.controller.queue_manager.read_queue) > 0 or 
                   len(self.controller.queue_manager.write_queue) > 0):
                self.controller.tick()
                
                # Track per-window bandwidth
                current_ns = self.controller.current_time_ns
                if current_ns - window_start_ns >= window_size_ns:
                    window_bytes.append(bytes_transferred)
                    window_start_ns = current_ns
            
            # Check if we've reached target duration
            if self.controller.current_time_ns - sim_start_ns >= self.config.test_duration_ns:
                break
        
        sim_end_ns = self.controller.current_time_ns
        elapsed_ns = sim_end_ns - sim_start_ns
        bytes_transferred = (read_count + write_count) * self.config.request_size_bytes
        
        # Calculate results
        result = BandwidthResult()
        result.peak_bandwidth_gbs = self.spec.bandwidth_gbs
        result.measured_bandwidth_gbs = (bytes_transferred / elapsed_ns) * 1000 if elapsed_ns > 0 else 0
        result.sustained_bandwidth_gbs = result.measured_bandwidth_gbs
        result.sustained_efficiency_percent = (result.sustained_bandwidth_gbs / result.peak_bandwidth_gbs * 100 
                                              if result.peak_bandwidth_gbs > 0 else 0)
        result.total_requests = read_count + write_count
        result.read_requests = read_count
        result.write_requests = write_count
        result.total_bytes = bytes_transferred
        result.test_duration_ns = elapsed_ns
        
        # Calculate bandwidth variance from windows
        if len(window_bytes) > 1:
            mean_bw = sum(window_bytes) / len(window_bytes)
            variance = sum((b - mean_bw) ** 2 for b in window_bytes) / len(window_bytes)
            result.bandwidth_variance_percent = (variance ** 0.5 / mean_bw * 100 
                                                if mean_bw > 0 else 0)
        
        _logger.info(f"Sustained bandwidth test complete: {result.sustained_bandwidth_gbs:.1f} GB/s "
                    f"({result.sustained_efficiency_percent:.1f}% efficiency)")
        
        return result
    
    def run_refresh_overhead_test(self) -> BandwidthResult:
        """Run refresh overhead test (measure bandwidth loss during refresh)"""
        _logger.info("Running refresh overhead test...")
        
        # Create controller with refresh enabled
        self.controller = HBM4Controller(
            spec=self.spec,
            enable_qos=False,
            enable_refresh=True
        )
        
        sim_start_ns = self.controller.current_time_ns
        refresh_start_ns = 0
        refresh_end_ns = 0
        total_refresh_time_ns = 0.0
        refresh_count = 0
        
        # Submit continuous traffic
        request_id = 0
        bytes_transferred = 0
        read_count = 0
        
        # Generate initial addresses
        addresses = self._generate_addresses(TestPattern.SEQUENTIAL, 100)
        addr_idx = 0
        
        while self.controller.current_time_ns - sim_start_ns < self.config.test_duration_ns:
            self.controller.tick()
            
            # Check for refresh
            if self.controller.refresh_scheduler:
                if self.controller.refresh_scheduler.can_refresh():
                    if refresh_start_ns == 0:
                        refresh_start_ns = self.controller.current_time_ns
                    refresh_count += 1
                    self.controller.refresh_scheduler.mark_bank_refreshed(0, 0, 0)
                elif refresh_start_ns > 0 and refresh_end_ns == 0:
                    refresh_end_ns = self.controller.current_time_ns
                    total_refresh_time_ns += (refresh_end_ns - refresh_start_ns)
                    refresh_start_ns = 0
                    refresh_end_ns = 0
            
            # Submit requests when queues have space
            if (len(self.controller.queue_manager.read_queue) < 32 and
                len(self.controller.queue_manager.write_queue) < 32):
                for _ in range(8):  # Submit 8 requests per cycle
                    addr = addresses[addr_idx % len(addresses)]
                    req_id = self.controller.submit_request(
                        addr=addr,
                        is_read=True,
                        size_bytes=self.config.request_size_bytes
                    )
                    if req_id:
                        read_count += 1
                        bytes_transferred += self.config.request_size_bytes
                        addr_idx += 1
        
        sim_end_ns = self.controller.current_time_ns
        elapsed_ns = sim_end_ns - sim_start_ns
        
        # Calculate results
        result = BandwidthResult()
        result.peak_bandwidth_gbs = self.spec.bandwidth_gbs
        result.measured_bandwidth_gbs = (bytes_transferred / elapsed_ns) * 1000 if elapsed_ns > 0 else 0
        result.refresh_overhead_percent = (total_refresh_time_ns / elapsed_ns * 100 
                                           if elapsed_ns > 0 else 0)
        result.refresh_count = refresh_count
        result.refresh_total_time_ns = total_refresh_time_ns
        result.total_requests = read_count
        result.read_requests = read_count
        result.write_requests = 0
        result.total_bytes = bytes_transferred
        result.test_duration_ns = elapsed_ns
        
        _logger.info(f"Refresh overhead test complete: {result.refresh_overhead_percent:.2f}% overhead "
                    f"({refresh_count} refreshes)")
        
        return result
    
    def run_all_tests(self) -> Dict[str, BandwidthResult]:
        """Run all bandwidth tests
        
        Returns:
            Dictionary of test name to result
        """
        results = {}
        
        if self.config.calculate_peak:
            results['peak'] = self.run_peak_bandwidth_test()
        
        if self.config.calculate_sustained:
            results['sustained'] = self.run_sustained_bandwidth_test()
        
        if self.config.calculate_refresh_overhead:
            results['refresh_overhead'] = self.run_refresh_overhead_test()
        
        self.results = list(results.values())
        return results
    
    def get_summary(self) -> BandwidthResult:
        """Get aggregated summary of all tests"""
        if not self.results:
            self.run_all_tests()
        
        summary = BandwidthResult()
        for result in self.results:
            summary.peak_bandwidth_gbs = max(summary.peak_bandwidth_gbs, result.peak_bandwidth_gbs)
            summary.measured_bandwidth_gbs += result.measured_bandwidth_gbs
            summary.sustained_bandwidth_gbs += result.sustained_bandwidth_gbs
            summary.total_requests += result.total_requests
            summary.read_requests += result.read_requests
            summary.write_requests += result.write_requests
            summary.total_bytes += result.total_bytes
            summary.refresh_count += result.refresh_count
            summary.refresh_total_time_ns += result.refresh_total_time_ns
        
        if len(self.results) > 0:
            summary.measured_bandwidth_gbs /= len(self.results)
            summary.sustained_bandwidth_gbs /= len(self.results)
            summary.peak_efficiency_percent = (summary.measured_bandwidth_gbs / 
                                              summary.peak_bandwidth_gbs * 100 
                                              if summary.peak_bandwidth_gbs > 0 else 0)
            summary.sustained_efficiency_percent = (summary.sustained_bandwidth_gbs / 
                                                    summary.peak_bandwidth_gbs * 100 
                                                    if summary.peak_bandwidth_gbs > 0 else 0)
            summary.refresh_overhead_percent = (summary.refresh_total_time_ns / 
                                               (summary.test_duration_ns * len(self.results)) * 100 
                                               if summary.test_duration_ns > 0 else 0)
        
        return summary