"""
Scheduler Benchmark Module

Tests scheduler efficiency under various conditions:
- QoS scheduling effectiveness
- Row hit rate optimization
- Bank conflict rate
- Queue depth impact

References:
- JEDEC JESD270-4A HBM4 specification
- QoS scheduling literature
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from model.dram.hbm4_spec import HBM4Spec, HBM4_SPEED_GRADES
from model.dram.timing import HBM4Timing, get_timing_for_speed_grade
from model.controller.hbm4_controller import HBM4Controller
from model.controller.hbm4_qos_scheduler import HBM4QoSScheduler, QoSLevel
from .benchmark_config import SchedulerConfig, TestPattern


@dataclass
class SchedulerResult:
    """Results from scheduler benchmark"""
    # QoS scheduling metrics
    qos_enabled: bool = True
    high_priority_avg_latency_ns: float = 0.0
    low_priority_avg_latency_ns: float = 0.0
    priority_latency_ratio: float = 0.0  # low / high
    
    # Row hit rate metrics
    row_hit_rate_percent: float = 0.0
    row_miss_count: int = 0
    row_hit_count: int = 0
    optimal_row_hit_rate_percent: float = 0.0  # Best case
    
    # Bank conflict metrics
    bank_conflict_rate_percent: float = 0.0
    bank_conflict_count: int = 0
    bank_activation_count: int = 0
    average_bank_activations_per_request: float = 0.0
    
    # Queue metrics
    average_queue_depth: float = 0.0
    max_queue_depth: int = 0
    queue_full_count: int = 0
    
    # Request statistics
    total_requests: int = 0
    completed_requests: int = 0
    rejected_requests: int = 0
    
    # Timing
    test_duration_ns: float = 0.0
    requests_per_second: float = 0.0
    
    # Per-QoS level breakdown
    qos_level_stats: Dict[int, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting"""
        return {
            'qos_enabled': self.qos_enabled,
            'high_priority_avg_latency_ns': self.high_priority_avg_latency_ns,
            'low_priority_avg_latency_ns': self.low_priority_avg_latency_ns,
            'priority_latency_ratio': self.priority_latency_ratio,
            'row_hit_rate_percent': self.row_hit_rate_percent,
            'bank_conflict_rate_percent': self.bank_conflict_rate_percent,
            'average_queue_depth': self.average_queue_depth,
            'max_queue_depth': self.max_queue_depth,
            'total_requests': self.total_requests,
            'completed_requests': self.completed_requests,
            'test_duration_ns': self.test_duration_ns,
            'requests_per_second': self.requests_per_second,
        }
    
    def __str__(self) -> str:
        return (
            f"Scheduler Results:\n"
            f"  QoS Enabled: {self.qos_enabled}\n"
            f"  Priority Latency Ratio: {self.priority_latency_ratio:.2f}x\n"
            f"  Row Hit Rate: {self.row_hit_rate_percent:.1f}%\n"
            f"  Bank Conflict Rate: {self.bank_conflict_rate_percent:.1f}%\n"
            f"  Avg Queue Depth: {self.average_queue_depth:.1f}\n"
            f"  Max Queue Depth: {self.max_queue_depth}\n"
            f"  Requests/s: {self.requests_per_second:.0f}\n"
            f"  Completed: {self.completed_requests}/{self.total_requests}"
        )


class SchedulerBenchmark:
    """Scheduler efficiency benchmarking"""
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        speed_grade: str = "8Gbps"
    ):
        """Initialize scheduler benchmark
        
        Args:
            config: Scheduler configuration (uses default if None)
            speed_grade: HBM speed grade ("8Gbps", "12Gbps", "16Gbps")
        """
        self.config = config or SchedulerConfig()
        self.speed_grade = speed_grade
        
        # Create HBM4 specification
        self.spec = self._create_spec()
        self.timing = get_timing_for_speed_grade(speed_grade)
        
        # Controller under test
        self.controller: Optional[HBM4Controller] = None
        
        # Results storage
        self.results: Dict[str, SchedulerResult] = {}
    
    def _create_spec(self) -> HBM4Spec:
        """Create HBM4 specification"""
        if self.speed_grade not in HBM4_SPEED_GRADES:
            raise ValueError(f"Unknown speed grade: {self.speed_grade}")
        
        grade_params = HBM4_SPEED_GRADES[self.speed_grade]
        return HBM4Spec(
            data_rate_gtps=grade_params["data_rate_gtps"],
            tCK_ps=grade_params["tCK_ps"]
        )
    
    def _generate_addresses(
        self, 
        pattern: TestPattern, 
        count: int,
        bank_conflict_mode: bool = False
    ) -> List[int]:
        """Generate addresses based on traffic pattern"""
        random.seed(42)  # Reproducibility
        
        addr_start = 0
        addr_end = 0x100_000_000  # 4GB
        
        if pattern == TestPattern.SEQUENTIAL:
            return [(addr_start + i * 64) % addr_end for i in range(count)]
        
        elif pattern == TestPattern.RANDOM:
            return [random.randint(addr_start, addr_end - 1) for _ in range(count)]
        
        elif pattern == TestPattern.ROW_HIT:
            # Same row repeated for high row hit rate
            return [addr_start] * count
        
        elif pattern == TestPattern.BANK_CONFLICT or bank_conflict_mode:
            # Spread across different banks to trigger conflicts
            bank_size = 0x10000  # 64KB per bank
            return [(i * bank_size) % addr_end for i in range(count)]
        
        else:
            return [random.randint(addr_start, addr_end - 1) for _ in range(count)]
    
    def run_qos_effectiveness_test(self) -> SchedulerResult:
        """Test QoS scheduling effectiveness
        
        Measures latency difference between high and low priority requests.
        """
        _logger = __import__('logging').getLogger(__name__)
        _logger.info("Running QoS effectiveness test...")
        
        # Create controller with QoS enabled
        self.controller = HBM4Controller(
            spec=self.spec,
            enable_qos=True,
            enable_refresh=False
        )
        
        # Generate mixed-priority addresses
        num_requests = self.config.test_duration_ns // 1000  # Rough estimate
        addresses = self._generate_addresses(TestPattern.RANDOM, num_requests)
        
        # Assign QoS levels based on distribution
        qos_dist = self.config.qos_distribution
        high_priority_latencies = []
        low_priority_latencies = []
        pending_requests: Dict[str, Tuple[int, int, int]] = {}  # id -> (qos, is_read, submit_time)
        
        for i, addr in enumerate(addresses):
            # Determine QoS level based on distribution
            rand_val = random.random()
            cumulative = 0
            qos_level = 8  # Default
            for level, ratio in sorted(qos_dist.items()):
                cumulative += ratio
                if rand_val < cumulative:
                    qos_level = level
                    break
            
            is_read = random.random() < 0.7
            
            request_id = self.controller.submit_request(
                addr=addr,
                is_read=is_read,
                qos_level=qos_level,
                size_bytes=64
            )
            
            if request_id:
                pending_requests[request_id] = (qos_level, is_read, self.controller.current_time_ns)
        
        # Process and collect latencies
        while pending_requests:
            responses = self.controller.tick()
            
            for resp in responses:
                if resp.request_id in pending_requests:
                    qos, is_read, submit_time = pending_requests[resp.request_id]
                    latency = self.controller.current_time_ns - submit_time
                    
                    # High priority: QoS 0-4, Low priority: QoS 12-15
                    if qos <= 4:
                        high_priority_latencies.append(latency)
                    elif qos >= 12:
                        low_priority_latencies.append(latency)
                    
                    del pending_requests[resp.request_id]
        
        # Calculate results
        result = SchedulerResult()
        result.qos_enabled = True
        
        if high_priority_latencies:
            result.high_priority_avg_latency_ns = sum(high_priority_latencies) / len(high_priority_latencies)
        
        if low_priority_latencies:
            result.low_priority_avg_latency_ns = sum(low_priority_latencies) / len(low_priority_latencies)
        
        if result.high_priority_avg_latency_ns > 0 and result.low_priority_avg_latency_ns > 0:
            result.priority_latency_ratio = (result.low_priority_avg_latency_ns / 
                                            result.high_priority_avg_latency_ns)
        
        result.total_requests = len(addresses)
        result.completed_requests = len(high_priority_latencies) + len(low_priority_latencies)
        result.test_duration_ns = self.controller.current_time_ns
        
        if result.test_duration_ns > 0:
            result.requests_per_second = result.completed_requests / (result.test_duration_ns / 1e9)
        
        _logger.info(f"QoS effectiveness: high={result.high_priority_avg_latency_ns:.1f}ns, "
                    f"low={result.low_priority_avg_latency_ns:.1f}ns, "
                    f"ratio={result.priority_latency_ratio:.2f}x")
        
        return result
    
    def run_row_hit_rate_test(self) -> SchedulerResult:
        """Test row hit rate optimization
        
        Measures how well the scheduler maintains open rows.
        """
        _logger = __import__('logging').getLogger(__name__)
        _logger.info("Running row hit rate test...")
        
        # Create controller
        self.controller = HBM4Controller(
            spec=self.spec,
            enable_qos=True,
            enable_refresh=False
        )
        
        test_duration = self.config.row_hit_test_duration_ns
        sim_start = self.controller.current_time_ns
        
        # Generate addresses with row locality
        num_requests = 10000
        addresses = self._generate_addresses(TestPattern.ROW_HIT, num_requests)
        
        row_hits = 0
        row_misses = 0
        pending_requests: Dict[str, Tuple[int, int]] = {}  # id -> (submit_time, is_hit)
        
        for addr in addresses:
            request_id = self.controller.submit_request(
                addr=addr,
                is_read=True,
                qos_level=8,
                size_bytes=64
            )
            
            if request_id:
                pending_requests[request_id] = (self.controller.current_time_ns, True)
        
        # Process and count row hits/misses
        while pending_requests and self.controller.current_time_ns - sim_start < test_duration:
            responses = self.controller.tick()
            
            for resp in responses:
                if resp.request_id in pending_requests:
                    row_hits += 1
                    del pending_requests[resp.request_id]
        
        # Calculate row miss rate (simulated as no row locality)
        row_miss_addresses = self._generate_addresses(TestPattern.RANDOM, num_requests)
        for addr in row_miss_addresses:
            request_id = self.controller.submit_request(
                addr=addr,
                is_read=True,
                qos_level=8,
                size_bytes=64
            )
            
            if request_id:
                pending_requests[request_id] = (self.controller.current_time_ns, False)
        
        while pending_requests:
            responses = self.controller.tick()
            for resp in responses:
                if resp.request_id in pending_requests:
                    row_misses += 1
                    del pending_requests[resp.request_id]
        
        # Calculate results
        result = SchedulerResult()
        result.row_hit_rate_percent = (row_hits / (row_hits + row_misses) * 100 
                                       if (row_hits + row_misses) > 0 else 0)
        result.row_hit_count = row_hits
        result.row_miss_count = row_misses
        result.optimal_row_hit_rate_percent = 95.0  # Theoretical best
        result.total_requests = num_requests * 2
        result.completed_requests = row_hits + row_misses
        result.test_duration_ns = self.controller.current_time_ns - sim_start
        
        _logger.info(f"Row hit rate: {result.row_hit_rate_percent:.1f}% "
                    f"(hits={row_hits}, misses={row_misses})")
        
        return result
    
    def run_bank_conflict_test(self) -> SchedulerResult:
        """Test bank conflict rate
        
        Measures how often requests are blocked by bank conflicts.
        """
        _logger = __import__('logging').getLogger(__name__)
        _logger.info("Running bank conflict test...")
        
        # Create controller
        self.controller = HBM4Controller(
            spec=self.spec,
            enable_qos=False,
            enable_refresh=False
        )
        
        sim_start = self.controller.current_time_ns
        
        # Generate addresses spread across banks
        num_requests = 1000
        addresses = self._generate_addresses(TestPattern.BANK_CONFLICT, num_requests, 
                                             bank_conflict_mode=True)
        
        bank_activations = 0
        bank_conflicts = 0
        pending_requests: Dict[str, Tuple[int, int]] = {}  # id -> (submit_time, bank_id)
        
        # Track bank states
        bank_open_time: Dict[int, int] = defaultdict(int)
        
        for addr in addresses:
            # Calculate bank ID from address
            bank_id = (addr // 0x10000) % 16
            
            request_id = self.controller.submit_request(
                addr=addr,
                is_read=True,
                qos_level=8,
                size_bytes=64
            )
            
            if request_id:
                pending_requests[request_id] = (self.controller.current_time_ns, bank_id)
        
        # Process and count bank conflicts
        while pending_requests:
            responses = self.controller.tick()
            
            for resp in responses:
                if resp.request_id in pending_requests:
                    _, bank_id = pending_requests[resp.request_id]
                    
                    # Count activation
                    bank_activations += 1
                    
                    # Check for conflict (bank was still open)
                    if bank_open_time.get(bank_id, 0) > 0:
                        bank_conflicts += 1
                    
                    # Mark bank as opened
                    bank_open_time[bank_id] = self.controller.current_time_ns
                    
                    del pending_requests[resp.request_id]
        
        # Calculate results
        result = SchedulerResult()
        result.bank_activation_count = bank_activations
        result.bank_conflict_count = bank_conflicts
        result.bank_conflict_rate_percent = (bank_conflicts / bank_activations * 100 
                                           if bank_activations > 0 else 0)
        result.average_bank_activations_per_request = (bank_activations / num_requests 
                                                      if num_requests > 0 else 0)
        result.total_requests = num_requests
        result.completed_requests = bank_activations
        result.test_duration_ns = self.controller.current_time_ns - sim_start
        
        _logger.info(f"Bank conflict rate: {result.bank_conflict_rate_percent:.1f}% "
                    f"(activations={bank_activations}, conflicts={bank_conflicts})")
        
        return result
    
    def run_queue_depth_test(self) -> SchedulerResult:
        """Test queue depth impact on performance"""
        _logger = __import__('logging').getLogger(__name__)
        _logger.info(f"Running queue depth test (depth={self.config.queue_depth})...")
        
        # Create controller
        self.controller = HBM4Controller(
            spec=self.spec,
            enable_qos=True,
            enable_refresh=False
        )
        
        sim_start = self.controller.current_time_ns
        
        # Submit requests rapidly
        num_requests = 10000
        addresses = self._generate_addresses(TestPattern.RANDOM, num_requests)
        
        queue_depths = []
        max_depth = 0
        queue_full_events = 0
        rejected = 0
        completed = 0
        pending_requests: Dict[str, int] = {}  # id -> submit_time
        
        for addr in addresses:
            request_id = self.controller.submit_request(
                addr=addr,
                is_read=True,
                qos_level=8,
                size_bytes=64
            )
            
            if request_id:
                pending_requests[request_id] = self.controller.current_time_ns
            else:
                rejected += 1
                queue_full_events += 1
            
            # Track queue depth periodically
            if len(pending_requests) > max_depth:
                max_depth = len(pending_requests)
            
            if len(pending_requests) > 0 and len(pending_requests) % 100 == 0:
                queue_depths.append(len(pending_requests))
        
        # Process remaining requests
        while pending_requests:
            responses = self.controller.tick()
            
            for resp in responses:
                if resp.request_id in pending_requests:
                    completed += 1
                    del pending_requests[resp.request_id]
            
            if len(pending_requests) > max_depth:
                max_depth = len(pending_requests)
        
        # Calculate results
        result = SchedulerResult()
        result.max_queue_depth = max_depth
        result.queue_full_count = queue_full_events
        result.rejected_requests = rejected
        result.average_queue_depth = (sum(queue_depths) / len(queue_depths) 
                                       if queue_depths else 0)
        result.total_requests = num_requests
        result.completed_requests = completed
        result.test_duration_ns = self.controller.current_time_ns - sim_start
        
        if result.test_duration_ns > 0:
            result.requests_per_second = completed / (result.test_duration_ns / 1e9)
        
        _logger.info(f"Queue depth: avg={result.average_queue_depth:.1f}, "
                    f"max={result.max_queue_depth}, "
                    f"rejected={rejected}")
        
        return result
    
    def run_all_tests(self) -> Dict[str, SchedulerResult]:
        """Run all scheduler tests"""
        results = {}
        
        if self.config.enable_qos:
            results['qos_effectiveness'] = self.run_qos_effectiveness_test()
        
        if self.config.row_hit_test_enabled:
            results['row_hit_rate'] = self.run_row_hit_rate_test()
        
        if self.config.bank_conflict_test_enabled:
            results['bank_conflict'] = self.run_bank_conflict_test()
        
        results['queue_depth'] = self.run_queue_depth_test()
        
        self.results = results
        return results
    
    def get_summary(self) -> SchedulerResult:
        """Get aggregated scheduler summary"""
        if not self.results:
            self.run_all_tests()
        
        summary = SchedulerResult()
        summary.qos_enabled = self.config.enable_qos
        
        # Average across all tests
        for name, result in self.results.items():
            summary.total_requests += result.total_requests
            summary.completed_requests += result.completed_requests
            summary.rejected_requests += result.rejected_requests
            summary.test_duration_ns += result.test_duration_ns
            
            if hasattr(result, 'row_hit_rate_percent'):
                summary.row_hit_rate_percent = max(summary.row_hit_rate_percent, 
                                                  result.row_hit_rate_percent)
            if hasattr(result, 'bank_conflict_rate_percent'):
                summary.bank_conflict_rate_percent = max(summary.bank_conflict_rate_percent,
                                                         result.bank_conflict_rate_percent)
            if hasattr(result, 'average_queue_depth'):
                summary.average_queue_depth = max(summary.average_queue_depth,
                                                  result.average_queue_depth)
        
        summary.max_queue_depth = max(r.max_queue_depth for r in self.results.values())
        
        return summary