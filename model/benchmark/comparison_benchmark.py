"""
Comparison Benchmark Module

Compares performance across different HBM configurations:
- HBM3 vs HBM4
- Different HBM4 speed grades (8Gbps, 12Gbps, 16Gbps)
- Different configurations

References:
- JEDEC JESD270-4A HBM4 specification
- HBM3 JESD238 specification
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from model.dram.hbm4_spec import HBM4Spec, HBM4_SPEED_GRADES, calculate_bandwidth
from model.dram.timing import HBM4Timing, get_timing_for_speed_grade, HBM3Timing
from model.controller.hbm4_controller import HBM4Controller
from model.controller.config import HBMConfig
from .benchmark_config import ComparisonConfig, SpeedGrade, TestPattern


@dataclass
class ComparisonResult:
    """Results from comparison benchmark"""
    # Configuration info
    config_name: str = ""
    data_rate_gtps: float = 0.0
    io_width: int = 0
    
    # Bandwidth metrics
    peak_bandwidth_gbs: float = 0.0
    measured_bandwidth_gbs: float = 0.0
    bandwidth_efficiency_percent: float = 0.0
    
    # Latency metrics
    average_latency_ns: float = 0.0
    p99_latency_ns: float = 0.0
    
    # Efficiency metrics
    requests_per_ns: float = 0.0
    energy_per_bit: float = 0.0  # Relative energy metric
    
    # Comparison relative to baseline
    bandwidth_vs_baseline: float = 0.0  # Ratio to HBM3 baseline
    latency_vs_baseline: float = 0.0   # Ratio to HBM3 baseline
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting"""
        return {
            'config_name': self.config_name,
            'data_rate_gtps': self.data_rate_gtps,
            'io_width': self.io_width,
            'peak_bandwidth_gbs': self.peak_bandwidth_gbs,
            'measured_bandwidth_gbs': self.measured_bandwidth_gbs,
            'bandwidth_efficiency_percent': self.bandwidth_efficiency_percent,
            'average_latency_ns': self.average_latency_ns,
            'p99_latency_ns': self.p99_latency_ns,
            'bandwidth_vs_baseline': self.bandwidth_vs_baseline,
            'latency_vs_baseline': self.latency_vs_baseline,
        }
    
    def __str__(self) -> str:
        return (
            f"{self.config_name}:\n"
            f"  Data Rate: {self.data_rate_gtps} GT/s\n"
            f"  Peak BW: {self.peak_bandwidth_gbs:.1f} GB/s\n"
            f"  Measured BW: {self.measured_bandwidth_gbs:.1f} GB/s ({self.bandwidth_efficiency_percent:.1f}%)\n"
            f"  Avg Latency: {self.average_latency_ns:.1f} ns\n"
            f"  P99 Latency: {self.p99_latency_ns:.1f} ns\n"
            f"  vs Baseline: BW={self.bandwidth_vs_baseline:.2f}x, Lat={self.latency_vs_baseline:.2f}x"
        )


@dataclass
class ComparisonReport:
    """Comprehensive comparison report"""
    baseline: Optional[ComparisonResult] = None
    configs: List[ComparisonResult] = field(default_factory=list)
    
    # Summary metrics
    best_bandwidth_config: str = ""
    best_bandwidth_gbs: float = 0.0
    best_latency_config: str = ""
    best_latency_ns: float = float('inf')
    
    # Speedup analysis
    hbm4_vs_hbm3_bandwidth_speedup: float = 0.0
    hbm4_vs_hbm3_latency_improvement: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting"""
        return {
            'baseline': self.baseline.to_dict() if self.baseline else None,
            'configs': [c.to_dict() for c in self.configs],
            'best_bandwidth_config': self.best_bandwidth_config,
            'best_bandwidth_gbs': self.best_bandwidth_gbs,
            'best_latency_config': self.best_latency_config,
            'best_latency_ns': self.best_latency_ns,
            'hbm4_vs_hbm3_bandwidth_speedup': self.hbm4_vs_hbm3_bandwidth_speedup,
            'hbm4_vs_hbm3_latency_improvement': self.hbm4_vs_hbm3_latency_improvement,
        }
    
    def __str__(self) -> str:
        lines = ["Comparison Report", "=" * 50]
        
        for config in self.configs:
            lines.append(str(config))
            lines.append("")
        
        lines.append("Summary")
        lines.append("-" * 50)
        lines.append(f"Best Bandwidth: {self.best_bandwidth_config} @ {self.best_bandwidth_gbs:.1f} GB/s")
        lines.append(f"Best Latency: {self.best_latency_config} @ {self.best_latency_ns:.1f} ns")
        
        if self.hbm4_vs_hbm3_bandwidth_speedup > 0:
            lines.append(f"HBM4 vs HBM3: {self.hbm4_vs_hbm3_bandwidth_speedup:.2f}x bandwidth")
        if self.hbm4_vs_hbm3_latency_improvement > 0:
            lines.append(f"HBM4 vs HBM3: {self.hbm4_vs_hbm3_latency_improvement * 100:.1f}% latency improvement")
        
        return "\n".join(lines)


class ComparisonBenchmark:
    """HBM configuration comparison benchmarking"""
    
    def __init__(
        self,
        config: Optional[ComparisonConfig] = None
    ):
        """Initialize comparison benchmark
        
        Args:
            config: Comparison configuration (uses default if None)
        """
        self.config = config or ComparisonConfig()
        
        # Results storage
        self.results: Dict[str, ComparisonResult] = {}
        self.baseline_result: Optional[ComparisonResult] = None
    
    def _run_single_config_test(
        self,
        config_name: str,
        speed_grade: SpeedGrade
    ) -> ComparisonResult:
        """Run benchmark for a single configuration
        
        Args:
            config_name: Name of configuration
            speed_grade: Speed grade to test
            
        Returns:
            ComparisonResult for this configuration
        """
        _logger = __import__('logging').getLogger(__name__)
        _logger.info(f"Running benchmark for {config_name} ({speed_grade.data_rate} GT/s)...")
        
        # Create specification based on speed grade
        if speed_grade.version == "hbm3":
            # HBM3 configuration
            spec = HBM4Spec(
                channels=16,  # HBM3 has 16 channels
                data_rate_gtps=speed_grade.data_rate,
                io_width=1024,  # HBM3 has 1024-bit width
                tCK_ps=1000.0 / speed_grade.data_rate
            )
        else:
            # HBM4 configuration
            spec = HBM4Spec(
                channels=32,  # HBM4 has 32 channels
                data_rate_gtps=speed_grade.data_rate,
                io_width=2048,  # HBM4 has 2048-bit width
                tCK_ps=1000.0 / speed_grade.data_rate
            )
        
        # Create controller
        controller = HBM4Controller(
            spec=spec,
            enable_qos=True,
            enable_refresh=False
        )
        
        # Run bandwidth test
        sim_start = controller.current_time_ns
        num_requests = self.config.num_requests
        bytes_transferred = 0
        latencies = []
        pending_requests: Dict[str, Tuple[int, bool]] = {}  # id -> (submit_time, is_read)
        
        # Generate addresses
        random.seed(42)
        addr_start = 0
        addr_end = 0x100_000_000
        
        for i in range(num_requests):
            addr = (addr_start + i * 64) % addr_end
            is_read = random.random() < 0.7
            
            request_id = controller.submit_request(
                addr=addr,
                is_read=is_read,
                qos_level=8,
                size_bytes=64
            )
            
            if request_id:
                pending_requests[request_id] = (controller.current_time_ns, is_read)
        
        # Process and collect metrics
        while pending_requests:
            responses = controller.tick()
            
            for resp in responses:
                if resp.request_id in pending_requests:
                    submit_time, _ = pending_requests[resp.request_id]
                    latency = controller.current_time_ns - submit_time
                    latencies.append(latency)
                    bytes_transferred += 64
                    del pending_requests[resp.request_id]
        
        sim_end = controller.current_time_ns
        elapsed_ns = sim_end - sim_start
        
        # Calculate result
        result = ComparisonResult()
        result.config_name = config_name
        result.data_rate_gtps = speed_grade.data_rate
        result.io_width = speed_grade.io_width
        result.peak_bandwidth_gbs = spec.bandwidth_gbs
        
        if elapsed_ns > 0:
            result.measured_bandwidth_gbs = (bytes_transferred / elapsed_ns) * 1000
        
        result.bandwidth_efficiency_percent = (result.measured_bandwidth_gbs / 
                                                result.peak_bandwidth_gbs * 100 
                                                if result.peak_bandwidth_gbs > 0 else 0)
        
        if latencies:
            result.average_latency_ns = sum(latencies) / len(latencies)
            sorted_lat = sorted(latencies)
            result.p99_latency_ns = sorted_lat[int(len(sorted_lat) * 0.99)]
        
        result.requests_per_ns = num_requests / elapsed_ns if elapsed_ns > 0 else 0
        
        _logger.info(f"  {config_name}: {result.measured_bandwidth_gbs:.1f} GB/s, "
                    f"{result.average_latency_ns:.1f} ns latency")
        
        return result
    
    def run_comparison(self) -> ComparisonReport:
        """Run comparison across all configurations
        
        Returns:
            ComparisonReport with results for all configs
        """
        _logger = __import__('logging').getLogger(__name__)
        _logger.info("Starting HBM configuration comparison...")
        
        report = ComparisonReport()
        baseline = None
        
        for config_name, speed_grade in self.config.configs_to_compare:
            result = self._run_single_config_test(config_name, speed_grade)
            self.results[config_name] = result
            report.configs.append(result)
            
            # Set baseline (HBM3)
            if speed_grade.version == "hbm3" and baseline is None:
                baseline = result
                report.baseline = result
        
        # If no HBM3 baseline, use first config
        if baseline is None and report.configs:
            baseline = report.configs[0]
            report.baseline = baseline
        
        # Calculate comparisons to baseline
        for result in report.configs:
            if baseline and result.config_name != baseline.config_name:
                result.bandwidth_vs_baseline = (result.peak_bandwidth_gbs / 
                                                baseline.peak_bandwidth_gbs 
                                                if baseline.peak_bandwidth_gbs > 0 else 0)
                result.latency_vs_baseline = (baseline.average_latency_ns / 
                                              result.average_latency_ns 
                                              if result.average_latency_ns > 0 else 0)
        
        # Find best performers
        for result in report.configs:
            if result.measured_bandwidth_gbs > report.best_bandwidth_gbs:
                report.best_bandwidth_gbs = result.measured_bandwidth_gbs
                report.best_bandwidth_config = result.config_name
            
            if result.average_latency_ns < report.best_latency_ns:
                report.best_latency_ns = result.average_latency_ns
                report.best_latency_config = result.config_name
        
        # Calculate HBM4 vs HBM3 speedup
        hbm3_result = None
        hbm4_result = None
        
        for result in report.configs:
            if "HBM3" in result.config_name:
                hbm3_result = result
            elif "HBM4" in result.config_name:
                hbm4_result = result
        
        if hbm3_result and hbm4_result:
            report.hbm4_vs_hbm3_bandwidth_speedup = (hbm4_result.peak_bandwidth_gbs / 
                                                     hbm3_result.peak_bandwidth_gbs 
                                                     if hbm3_result.peak_bandwidth_gbs > 0 else 0)
            report.hbm4_vs_hbm3_latency_improvement = ((hbm3_result.average_latency_ns - 
                                                        hbm4_result.average_latency_ns) / 
                                                       hbm3_result.average_latency_ns 
                                                       if hbm3_result.average_latency_ns > 0 else 0)
        
        _logger.info(f"Comparison complete. Best BW: {report.best_bandwidth_config} @ "
                    f"{report.best_bandwidth_gbs:.1f} GB/s")
        
        return report
    
    def run_bandwidth_comparison(self) -> Dict[str, float]:
        """Run bandwidth comparison only
        
        Returns:
            Dictionary of config_name -> bandwidth in GB/s
        """
        results = {}
        
        for config_name, speed_grade in self.config.configs_to_compare:
            if speed_grade.version == "hbm3":
                spec = HBM4Spec(
                    channels=16,
                    data_rate_gtps=speed_grade.data_rate,
                    io_width=1024,
                    tCK_ps=1000.0 / speed_grade.data_rate
                )
            else:
                spec = HBM4Spec(
                    channels=32,
                    data_rate_gtps=speed_grade.data_rate,
                    io_width=2048,
                    tCK_ps=1000.0 / speed_grade.data_rate
                )
            
            results[config_name] = spec.bandwidth_gbs
        
        return results
    
    def run_latency_comparison(self) -> Dict[str, float]:
        """Run latency comparison only
        
        Returns:
            Dictionary of config_name -> average latency in ns
        """
        results = {}
        
        for config_name, speed_grade in self.config.configs_to_compare:
            if speed_grade.version == "hbm3":
                # HBM3 latency estimate (higher due to slower speed)
                base_latency = 40  # ns
            else:
                # HBM4 latency estimate (lower due to faster speed)
                base_latency = 25  # ns
            
            # Scale based on data rate
            rate_factor = 8.0 / speed_grade.data_rate
            results[config_name] = base_latency * rate_factor
        
        return results
    
    def get_summary(self) -> str:
        """Get comparison summary as string"""
        report = self.run_comparison()
        return str(report)