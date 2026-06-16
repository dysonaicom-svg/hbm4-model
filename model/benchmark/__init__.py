"""
HBM Performance Benchmark Suite

This package provides comprehensive performance benchmarking for HBM memory controllers.

Modules:
- benchmark_config: Configuration and test parameters
- bandwidth_benchmark: Bandwidth tests (peak, sustained, refresh overhead)
- latency_benchmark: Latency tests (average, P99, distribution)
- scheduler_benchmark: Scheduler efficiency tests (QoS, row hit, bank conflicts)
- comparison_benchmark: HBM4 vs HBM3 comparison
- benchmark_runner: Main runner that orchestrates all tests

Usage:
    from model.benchmark import BenchmarkRunner
    
    runner = BenchmarkRunner()
    report = runner.run_all_benchmarks()
    print(report)
"""

from .benchmark_runner import BenchmarkRunner, BenchmarkReport
from .benchmark_config import (
    BenchmarkConfig,
    BandwidthConfig,
    LatencyConfig,
    SchedulerConfig,
    ComparisonConfig,
)
from .bandwidth_benchmark import BandwidthBenchmark
from .latency_benchmark import LatencyBenchmark
from .scheduler_benchmark import SchedulerBenchmark
from .comparison_benchmark import ComparisonBenchmark

__all__ = [
    'BenchmarkRunner',
    'BenchmarkReport',
    'BenchmarkConfig',
    'BandwidthConfig',
    'LatencyConfig',
    'SchedulerConfig',
    'ComparisonConfig',
    'BandwidthBenchmark',
    'LatencyBenchmark',
    'SchedulerBenchmark',
    'ComparisonBenchmark',
]