"""
Benchmark Configuration Module

Defines configuration classes for all benchmark types.
Supports flexible parameterization for different test scenarios.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# =============================================================================
# Traffic Pattern Enum - Marked to prevent pytest collection
# =============================================================================
class _TrafficPatternEnum(Enum):
    """Traffic pattern for benchmarking"""
    SEQUENTIAL = "sequential"      # Consecutive addresses
    RANDOM = "random"             # Random addresses
    STRIDED = "strided"           # Fixed stride access
    HOTSPOT = "hotspot"            # Frequently accessed region
    BANK_CONFLICT = "bank_conflict" # Intentional bank conflicts
    ROW_HIT = "row_hit"           # All accesses to same row


# Alias for backward compatibility - use this in tests
TestPattern = _TrafficPatternEnum

# Also export as TrafficPattern for cleaner naming
TrafficPattern = _TrafficPatternEnum


class SpeedGrade(Enum):
    """HBM speed grade presets"""
    HBM3_6_4 = ("hbm3", 6.4, 1024)
    HBM4_8 = ("hbm4", 8.0, 2048)
    HBM4_12 = ("hbm4", 12.0, 2048)
    HBM4_16 = ("hbm4", 16.0, 2048)
    
    def __init__(self, version: str, data_rate: float, io_width: int):
        self.version = version
        self.data_rate = data_rate
        self.io_width = io_width


@dataclass
class BandwidthConfig:
    """Configuration for bandwidth benchmarks"""
    # Test duration and request rate
    test_duration_ns: int = 100_000_000    # 100ms simulation
    request_batch_size: int = 1000          # Requests per batch
    num_batches: int = 100                   # Number of batches
    
    # Traffic pattern
    pattern: TestPattern = TestPattern.SEQUENTIAL
    stride_bytes: int = 64                   # For STRIDED pattern
    
    # Request configuration
    request_size_bytes: int = 64            # 64 bytes = 1 FLINE
    read_write_ratio: float = 0.7            # 70% reads, 30% writes
    
    # Address range
    address_range_start: int = 0
    address_range_end: int = 0x100_000_000   # 4GB address space
    
    # Bandwidth calculation
    calculate_peak: bool = True
    calculate_sustained: bool = True
    calculate_refresh_overhead: bool = True
    
    # Refresh configuration (for refresh overhead test)
    refresh_interval_ns: float = 3.9e-6     # tREFI = 3.9us
    refresh_duration_ns: float = 180e-9     # tRFC = 180ns
    
    def __repr__(self) -> str:
        return (f"BandwidthConfig(duration={self.test_duration_ns/1e6:.0f}ms, "
                f"pattern={self.pattern.value}, "
                f"rw_ratio={self.read_write_ratio:.1f})")


@dataclass
class LatencyConfig:
    """Configuration for latency benchmarks"""
    # Test configuration
    num_requests: int = 10_000              # Total requests to simulate
    warmup_requests: int = 1000            # Requests before measuring
    cooldown_requests: int = 100            # Requests after measuring
    
    # Traffic pattern
    pattern: TestPattern = TestPattern.RANDOM
    
    # Latency metrics to calculate
    calculate_average: bool = True
    calculate_percentiles: bool = True
    calculate_distribution: bool = True
    
    # Percentiles to compute
    percentiles: List[float] = field(default_factory=lambda: [50, 90, 95, 99, 99.9])
    
    # Request configuration
    request_size_bytes: int = 64
    qos_level: int = 8                     # Default QoS level
    
    # Separate read/write latency
    separate_read_write: bool = True
    
    # Concurrency level
    max_concurrent_requests: int = 64      # Outstanding requests
    
    def __repr__(self) -> str:
        return (f"LatencyConfig(requests={self.num_requests}, "
                f"pattern={self.pattern.value}, "
                f"percentiles={self.percentiles})")


@dataclass
class SchedulerConfig:
    """Configuration for scheduler benchmarks"""
    # Test duration
    test_duration_ns: int = 50_000_000     # 50ms simulation
    
    # QoS configuration
    enable_qos: bool = True
    qos_levels: int = 16                   # Number of QoS levels
    
    # Traffic mix
    qos_distribution: Dict[int, float] = field(default_factory=lambda: {
        0: 0.10,   # Critical (10%)
        4: 0.15,   # High (15%)
        8: 0.50,   # Normal (50%)
        12: 0.15,  # Low (15%)
        15: 0.10   # Background (10%)
    })
    
    # Row hit test configuration
    row_hit_test_enabled: bool = True
    row_hit_test_duration_ns: int = 10_000_000
    row_hit_test_repeat_factor: int = 10    # Repeat same row access N times
    
    # Bank conflict test configuration
    bank_conflict_test_enabled: bool = True
    bank_conflict_test_duration_ns: int = 10_000_000
    
    # Address pattern for scheduler test
    pattern: TestPattern = TestPattern.RANDOM
    
    # Queue depth
    queue_depth: int = 64
    
    def __repr__(self) -> str:
        return (f"SchedulerConfig(qos_enabled={self.enable_qos}, "
                f"qos_levels={self.qos_levels}, "
                f"queue_depth={self.queue_depth})")


@dataclass
class ComparisonConfig:
    """Configuration for HBM comparison benchmarks"""
    # Configurations to compare
    configs_to_compare: List[Tuple[str, SpeedGrade]] = field(default_factory=lambda: [
        ("HBM3", SpeedGrade.HBM3_6_4),
        ("HBM4-8G", SpeedGrade.HBM4_8),
        ("HBM4-12G", SpeedGrade.HBM4_12),
    ])
    
    # Test scenarios
    compare_bandwidth: bool = True
    compare_latency: bool = True
    compare_efficiency: bool = True
    
    # Common test parameters
    test_duration_ns: int = 50_000_000
    num_requests: int = 10_000
    pattern: TestPattern = TestPattern.SEQUENTIAL
    
    def __repr__(self) -> str:
        config_names = [name for name, _ in self.configs_to_compare]
        return f"ComparisonConfig(configs={config_names})"


@dataclass
class BenchmarkConfig:
    """Master configuration for all benchmarks"""
    # Enable/disable specific benchmarks
    run_bandwidth: bool = True
    run_latency: bool = True
    run_scheduler: bool = True
    run_comparison: bool = True
    
    # Individual configurations
    bandwidth: BandwidthConfig = field(default_factory=BandwidthConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    
    # Output configuration
    verbose: bool = True
    output_file: Optional[str] = None
    generate_plots: bool = False
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # HBM specification override (optional)
    hbm4_data_rate: Optional[float] = None
    hbm4_io_width: Optional[int] = None
    
    @classmethod
    def quick(cls) -> "BenchmarkConfig":
        """Quick benchmark configuration for fast testing"""
        return cls(
            bandwidth=BandwidthConfig(
                test_duration_ns=1_000_000,  # 1ms
                num_batches=10
            ),
            latency=LatencyConfig(
                num_requests=1000,
                warmup_requests=100
            ),
            scheduler=SchedulerConfig(
                test_duration_ns=1_000_000
            ),
            comparison=ComparisonConfig(
                configs_to_compare=[("HBM4-8G", SpeedGrade.HBM4_8)]
            ),
            verbose=True
        )
    
    @classmethod
    def comprehensive(cls) -> "BenchmarkConfig":
        """Comprehensive benchmark configuration"""
        return cls(
            bandwidth=BandwidthConfig(
                test_duration_ns=100_000_000,
                num_batches=100
            ),
            latency=LatencyConfig(
                num_requests=100_000,
                warmup_requests=10000,
                calculate_distribution=True
            ),
            scheduler=SchedulerConfig(
                test_duration_ns=50_000_000
            ),
            comparison=ComparisonConfig(
                configs_to_compare=[
                    ("HBM3", SpeedGrade.HBM3_6_4),
                    ("HBM4-8G", SpeedGrade.HBM4_8),
                    ("HBM4-12G", SpeedGrade.HBM4_12),
                    ("HBM4-16G", SpeedGrade.HBM4_16),
                ]
            ),
            verbose=True
        )
    
    def __repr__(self) -> str:
        return (f"BenchmarkConfig("
                f"bandwidth={self.run_bandwidth}, "
                f"latency={self.run_latency}, "
                f"scheduler={self.run_scheduler}, "
                f"comparison={self.run_comparison})")
