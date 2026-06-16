"""
Latency Benchmark Module

Tests memory latency under various conditions:
- Average latency
- Percentile latency (P50, P90, P95, P99, P99.9)
- Latency distribution histogram
- Read vs Write latency separation

References:
- JEDEC JESD270-4A HBM4 specification
- Academic latency characterization papers
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import statistics

from model.dram.hbm4_spec import HBM4Spec, HBM4_SPEED_GRADES
from model.dram.timing import HBM4Timing, get_timing_for_speed_grade
from model.controller.hbm4_controller import HBM4Controller
from model.controller.request import HBMRequest, HBMResponse
from .benchmark_config import LatencyConfig, TestPattern


@dataclass
class LatencyResult:
    """Results from latency benchmark"""
    # Average latency metrics
    average_latency_ns: float = 0.0
    median_latency_ns: float = 0.0
    
    # Percentile latency metrics
    p50_latency_ns: float = 0.0
    p90_latency_ns: float = 0.0
    p95_latency_ns: float = 0.0
    p99_latency_ns: float = 0.0
    p999_latency_ns: float = 0.0
    
    # Separate read/write latency
    read_avg_latency_ns: float = 0.0
    write_avg_latency_ns: float = 0.0
    read_p99_latency_ns: float = 0.0
    write_p99_latency_ns: float = 0.0
    
    # Statistical metrics
    min_latency_ns: float = float('inf')
    max_latency_ns: float = 0.0
    std_dev_ns: float = 0.0
    
    # Distribution (histogram bins)
    latency_histogram: Dict[str, int] = field(default_factory=dict)
    latency_histogram_bins: List[int] = field(default_factory=lambda: [0, 50, 100, 150, 200, 300, 500, 1000])
    
    # Request counts
    total_requests: int = 0
    warmup_requests: int = 0
    measured_requests: int = 0
    
    # Detailed metrics
    row_hit_latency_ns: float = 0.0
    row_miss_latency_ns: float = 0.0
    
    # Timing breakdown
    queue_delay_ns: float = 0.0
    command_delay_ns: float = 0.0
    data_delay_ns: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting"""
        return {
            'average_latency_ns': self.average_latency_ns,
            'median_latency_ns': self.median_latency_ns,
            'p50_latency_ns': self.p50_latency_ns,
            'p90_latency_ns': self.p90_latency_ns,
            'p95_latency_ns': self.p95_latency_ns,
            'p99_latency_ns': self.p99_latency_ns,
            'p999_latency_ns': self.p999_latency_ns,
            'read_avg_latency_ns': self.read_avg_latency_ns,
            'write_avg_latency_ns': self.write_avg_latency_ns,
            'min_latency_ns': self.min_latency_ns,
            'max_latency_ns': self.max_latency_ns,
            'std_dev_ns': self.std_dev_ns,
            'total_requests': self.total_requests,
            'measured_requests': self.measured_requests,
        }
    
    def __str__(self) -> str:
        return (
            f"Latency Results:\n"
            f"  Average: {self.average_latency_ns:.1f} ns\n"
            f"  Median: {self.median_latency_ns:.1f} ns\n"
            f"  P50: {self.p50_latency_ns:.1f} ns\n"
            f"  P90: {self.p90_latency_ns:.1f} ns\n"
            f"  P95: {self.p95_latency_ns:.1f} ns\n"
            f"  P99: {self.p99_latency_ns:.1f} ns\n"
            f"  P99.9: {self.p999_latency_ns:.1f} ns\n"
            f"  Read avg: {self.read_avg_latency_ns:.1f} ns\n"
            f"  Write avg: {self.write_avg_latency_ns:.1f} ns\n"
            f"  Min: {self.min_latency_ns:.1f} ns\n"
            f"  Max: {self.max_latency_ns:.1f} ns\n"
            f"  StdDev: {self.std_dev_ns:.1f} ns\n"
            f"  Requests: {self.total_requests} ({self.measured_requests} measured)"
        )


class LatencyBenchmark:
    """Latency benchmarking for HBM memory controllers"""
    
    def __init__(
        self,
        config: Optional[LatencyConfig] = None,
        speed_grade: str = "8Gbps"
    ):
        """Initialize latency benchmark
        
        Args:
            config: Latency configuration (uses default if None)
            speed_grade: HBM speed grade ("8Gbps", "12Gbps", "16Gbps")
        """
        self.config = config or LatencyConfig()
        self.speed_grade = speed_grade
        
        # Create HBM4 specification
        self.spec = self._create_spec()
        self.timing = get_timing_for_speed_grade(speed_grade)
        
        # Controller under test
        self.controller: Optional[HBM4Controller] = None
        
        # Latency storage
        self.all_latencies: List[float] = []
        self.read_latencies: List[float] = []
        self.write_latencies: List[float] = []
        self.measured_latencies: List[float] = []
        
        # Results
        self.result: Optional[LatencyResult] = None
    
    def _create_spec(self) -> HBM4Spec:
        """Create HBM4 specification"""
        if self.speed_grade not in HBM4_SPEED_GRADES:
            raise ValueError(f"Unknown speed grade: {self.speed_grade}")
        
        grade_params = HBM4_SPEED_GRADES[self.speed_grade]
        return HBM4Spec(
            data_rate_gtps=grade_params["data_rate_gtps"],
            tCK_ps=grade_params["tCK_ps"]
        )
    
    def _generate_addresses(self, pattern: TestPattern, count: int) -> List[int]:
        """Generate addresses based on traffic pattern"""
        random.seed(self.config.num_requests)  # Reproducibility
        
        addr_start = 0
        addr_end = 0x100_000_000  # 4GB
        
        if pattern == TestPattern.SEQUENTIAL:
            return [(addr_start + i * self.config.request_size_bytes) % addr_end 
                    for i in range(count)]
        
        elif pattern == TestPattern.RANDOM:
            return [random.randint(addr_start, addr_end - 1) for _ in range(count)]
        
        elif pattern == TestPattern.STRIDED:
            stride = 1024  # 1KB stride
            return [(addr_start + i * stride) % addr_end for i in range(count)]
        
        elif pattern == TestPattern.HOTSPOT:
            hotspot_size = addr_end // 5
            results = []
            for i in range(count):
                if random.random() < 0.90:
                    results.append(random.randint(0, hotspot_size - 1))
                else:
                    results.append(hotspot_size + random.randint(0, addr_end - hotspot_size - 1))
            return results
        
        elif pattern == TestPattern.ROW_HIT:
            # All same address for row hits
            return [addr_start] * count
        
        else:
            return [random.randint(addr_start, addr_end - 1) for _ in range(count)]
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * percentile / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]
    
    def _build_histogram(self, latencies: List[float]) -> Dict[str, int]:
        """Build latency histogram from bins"""
        bins = self.result.latency_histogram_bins if self.result else [0, 50, 100, 150, 200, 300, 500, 1000]
        
        histogram = {}
        for i in range(len(bins) - 1):
            bin_name = f"{bins[i]}-{bins[i+1]}"
            histogram[bin_name] = 0
        
        bin_name = f">{bins[-1]}"
        histogram[bin_name] = 0
        
        for lat in latencies:
            for i in range(len(bins) - 1):
                if bins[i] <= lat < bins[i+1]:
                    histogram[f"{bins[i]}-{bins[i+1]}"] += 1
                    break
            else:
                histogram[f">{bins[-1]}"] += 1
        
        return histogram
    
    def run_latency_test(self) -> LatencyResult:
        """Run complete latency benchmark"""
        _logger = __import__('logging').getLogger(__name__)
        _logger.info(f"Running latency test with {self.config.num_requests} requests...")
        
        # Create controller
        self.controller = HBM4Controller(
            spec=self.spec,
            enable_qos=True,
            enable_refresh=False  # Disable refresh for cleaner latency test
        )
        
        # Generate addresses
        addresses = self._generate_addresses(self.config.pattern, self.config.num_requests)
        
        # Reset storage
        self.all_latencies = []
        self.read_latencies = []
        self.write_latencies = []
        
        # Submit all requests and track latencies
        request_map: Dict[str, Tuple[int, bool, int]] = {}  # request_id -> (addr, is_read, submit_time)
        
        for i, addr in enumerate(addresses):
            is_read = random.random() < 0.7  # 70% reads
            
            request_id = self.controller.submit_request(
                addr=addr,
                is_read=is_read,
                qos_level=self.config.qos_level,
                size_bytes=self.config.request_size_bytes
            )
            
            if request_id:
                request_map[request_id] = (addr, is_read, self.controller.current_time_ns)
        
        # Process requests and collect latencies
        while (len(self.controller.queue_manager.read_queue) > 0 or 
               len(self.controller.queue_manager.write_queue) > 0):
            responses = self.controller.tick()
            
            for resp in responses:
                if resp.request_id in request_map:
                    submit_time = request_map[resp.request_id][2]
                    latency = self.controller.current_time_ns - submit_time
                    
                    self.all_latencies.append(latency)
                    
                    if request_map[resp.request_id][1]:  # is_read
                        self.read_latencies.append(latency)
                    else:
                        self.write_latencies.append(latency)
        
        # Calculate statistics
        result = LatencyResult()
        
        if self.all_latencies:
            # Basic statistics
            result.average_latency_ns = statistics.mean(self.all_latencies)
            result.median_latency_ns = statistics.median(self.all_latencies)
            result.min_latency_ns = min(self.all_latencies)
            result.max_latency_ns = max(self.all_latencies)
            
            if len(self.all_latencies) > 1:
                result.std_dev_ns = statistics.stdev(self.all_latencies)
            
            # Percentiles
            sorted_latencies = sorted(self.all_latencies)
            result.p50_latency_ns = self._calculate_percentile(sorted_latencies, 50)
            result.p90_latency_ns = self._calculate_percentile(sorted_latencies, 90)
            result.p95_latency_ns = self._calculate_percentile(sorted_latencies, 95)
            result.p99_latency_ns = self._calculate_percentile(sorted_latencies, 99)
            result.p999_latency_ns = self._calculate_percentile(sorted_latencies, 99.9)
            
            # Read/Write separation
            if self.read_latencies:
                result.read_avg_latency_ns = statistics.mean(self.read_latencies)
                result.read_p99_latency_ns = self._calculate_percentile(
                    sorted(self.read_latencies), 99)
            
            if self.write_latencies:
                result.write_avg_latency_ns = statistics.mean(self.write_latencies)
                result.write_p99_latency_ns = self._calculate_percentile(
                    sorted(self.write_latencies), 99)
            
            # Build histogram
            result.latency_histogram = self._build_histogram(self.all_latencies)
        
        result.total_requests = len(addresses)
        result.measured_requests = len(self.all_latencies)
        
        self.result = result
        
        _logger.info(f"Latency test complete: avg={result.average_latency_ns:.1f}ns, "
                    f"p99={result.p99_latency_ns:.1f}ns")
        
        return result
    
    def run_row_hit_vs_miss_test(self) -> Tuple[float, float]:
        """Run test comparing row hit vs row miss latency
        
        Returns:
            Tuple of (row_hit_latency, row_miss_latency)
        """
        _logger = __import__('logging').getLogger(__name__)
        _logger.info("Running row hit vs miss latency test...")
        
        # Test row hit latency
        self.controller = HBM4Controller(spec=self.spec, enable_qos=False, enable_refresh=False)
        
        base_addr = 0x1000  # Fixed address for row hit
        
        row_hit_latencies = []
        for _ in range(100):
            request_id = self.controller.submit_request(
                addr=base_addr,
                is_read=True,
                size_bytes=self.config.request_size_bytes
            )
            if request_id:
                submit_time = self.controller.current_time_ns
                while len(self.controller.queue_manager.read_queue) > 0:
                    self.controller.tick()
                latency = self.controller.current_time_ns - submit_time
                row_hit_latencies.append(latency)
        
        # Test row miss latency
        self.controller = HBM4Controller(spec=self.spec, enable_qos=False, enable_refresh=False)
        
        row_miss_latencies = []
        for i in range(100):
            # Different row each time
            addr = (base_addr + i * 0x10000)
            request_id = self.controller.submit_request(
                addr=addr,
                is_read=True,
                size_bytes=self.config.request_size_bytes
            )
            if request_id:
                submit_time = self.controller.current_time_ns
                while len(self.controller.queue_manager.read_queue) > 0:
                    self.controller.tick()
                latency = self.controller.current_time_ns - submit_time
                row_miss_latencies.append(latency)
        
        row_hit_avg = statistics.mean(row_hit_latencies) if row_hit_latencies else 0
        row_miss_avg = statistics.mean(row_miss_latencies) if row_miss_latencies else 0
        
        if self.result:
            self.result.row_hit_latency_ns = row_hit_avg
            self.result.row_miss_latency_ns = row_miss_avg
        
        _logger.info(f"Row hit vs miss: {row_hit_avg:.1f}ns vs {row_miss_avg:.1f}ns")
        
        return row_hit_avg, row_miss_avg
    
    def run_concurrent_latency_test(self) -> LatencyResult:
        """Run latency test with controlled concurrency level
        
        Tests how latency scales with outstanding requests.
        """
        _logger = __import__('logging').getLogger(__name__)
        _logger.info(f"Running concurrent latency test (max={self.config.max_concurrent_requests})...")
        
        # Create controller
        self.controller = HBM4Controller(
            spec=self.spec,
            enable_qos=True,
            enable_refresh=False
        )
        
        # Generate addresses
        addresses = self._generate_addresses(TestPattern.RANDOM, self.config.num_requests)
        
        # Submit requests with controlled concurrency
        pending_requests: Dict[str, Tuple[int, bool, int]] = {}
        all_latencies = []
        read_latencies = []
        write_latencies = []
        
        for addr in addresses:
            # Wait if we've reached max concurrency
            while len(pending_requests) >= self.config.max_concurrent_requests:
                responses = self.controller.tick()
                self._process_responses(responses, pending_requests, all_latencies, 
                                      read_latencies, write_latencies)
            
            is_read = random.random() < 0.7
            request_id = self.controller.submit_request(
                addr=addr,
                is_read=is_read,
                qos_level=random.randint(0, 15),
                size_bytes=self.config.request_size_bytes
            )
            
            if request_id:
                pending_requests[request_id] = (addr, is_read, self.controller.current_time_ns)
        
        # Complete remaining requests
        while pending_requests:
            responses = self.controller.tick()
            self._process_responses(responses, pending_requests, all_latencies,
                                  read_latencies, write_latencies)
        
        # Calculate results
        result = LatencyResult()
        
        if all_latencies:
            result.average_latency_ns = statistics.mean(all_latencies)
            result.median_latency_ns = statistics.median(all_latencies)
            result.min_latency_ns = min(all_latencies)
            result.max_latency_ns = max(all_latencies)
            
            if len(all_latencies) > 1:
                result.std_dev_ns = statistics.stdev(all_latencies)
            
            sorted_lat = sorted(all_latencies)
            result.p50_latency_ns = self._calculate_percentile(sorted_lat, 50)
            result.p90_latency_ns = self._calculate_percentile(sorted_lat, 90)
            result.p95_latency_ns = self._calculate_percentile(sorted_lat, 95)
            result.p99_latency_ns = self._calculate_percentile(sorted_lat, 99)
            result.p999_latency_ns = self._calculate_percentile(sorted_lat, 99.9)
            
            if read_latencies:
                result.read_avg_latency_ns = statistics.mean(read_latencies)
            if write_latencies:
                result.write_avg_latency_ns = statistics.mean(write_latencies)
        
        result.total_requests = len(addresses)
        result.measured_requests = len(all_latencies)
        
        _logger.info(f"Concurrent latency test complete: avg={result.average_latency_ns:.1f}ns")
        
        return result
    
    def _process_responses(
        self,
        responses: List[HBMResponse],
        pending: Dict[str, Tuple[int, bool, int]],
        all_lat: List[float],
        read_lat: List[float],
        write_lat: List[float]
    ) -> None:
        """Process completed responses and calculate latencies"""
        for resp in responses:
            if resp.request_id in pending:
                submit_time = pending[resp.request_id][2]
                is_read = pending[resp.request_id][1]
                latency = self.controller.current_time_ns - submit_time
                
                all_lat.append(latency)
                if is_read:
                    read_lat.append(latency)
                else:
                    write_lat.append(latency)
                
                del pending[resp.request_id]
    
    def run_all_tests(self) -> Dict[str, LatencyResult]:
        """Run all latency tests"""
        results = {}
        
        # Main latency test
        results['basic'] = self.run_latency_test()
        
        # Row hit vs miss test
        row_hit, row_miss = self.run_row_hit_vs_miss_test()
        results['row_hit_vs_miss'] = self.result
        
        # Concurrent latency test
        results['concurrent'] = self.run_concurrent_latency_test()
        
        return results
    
    def get_summary(self) -> LatencyResult:
        """Get latency summary"""
        if not self.result:
            self.run_latency_test()
        return self.result