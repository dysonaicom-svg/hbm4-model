"""
Benchmark Runner Module

Main orchestrator for all benchmark tests.
Generates comprehensive performance reports.

Usage:
    from model.benchmark import BenchmarkRunner
    
    runner = BenchmarkRunner()
    report = runner.run_all_benchmarks()
    print(report.to_markdown())
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

from model.dram.hbm4_spec import HBM4Spec, HBM4_SPEED_GRADES, calculate_bandwidth
from .benchmark_config import BenchmarkConfig
from .bandwidth_benchmark import BandwidthBenchmark, BandwidthResult
from .latency_benchmark import LatencyBenchmark, LatencyResult
from .scheduler_benchmark import SchedulerBenchmark, SchedulerResult
from .comparison_benchmark import ComparisonBenchmark, ComparisonReport, ComparisonResult


_logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    # Metadata
    timestamp: str = ""
    duration_seconds: float = 0.0
    config: str = ""
    
    # Individual results
    bandwidth: Optional[BandwidthResult] = None
    latency: Optional[LatencyResult] = None
    scheduler: Optional[SchedulerResult] = None
    comparison: Optional[ComparisonReport] = None
    
    # Summary metrics
    peak_bandwidth_gbs: float = 0.0
    sustained_bandwidth_gbs: float = 0.0
    average_latency_ns: float = 0.0
    p99_latency_ns: float = 0.0
    row_hit_rate_percent: float = 0.0
    bank_conflict_rate_percent: float = 0.0
    
    # Key findings
    findings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'duration_seconds': self.duration_seconds,
            'config': self.config,
            'peak_bandwidth_gbs': self.peak_bandwidth_gbs,
            'sustained_bandwidth_gbs': self.sustained_bandwidth_gbs,
            'average_latency_ns': self.average_latency_ns,
            'p99_latency_ns': self.p99_latency_ns,
            'row_hit_rate_percent': self.row_hit_rate_percent,
            'bank_conflict_rate_percent': self.bank_conflict_rate_percent,
            'findings': self.findings,
            'warnings': self.warnings,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Generate markdown report"""
        lines = [
            "# HBM Performance Benchmark Report",
            "",
            f"**Generated:** {self.timestamp}",
            f"**Duration:** {self.duration_seconds:.2f} seconds",
            f"**Configuration:** {self.config}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
        ]
        
        # Summary table
        lines.extend([
            "| Metric | Value |",
            "|--------|-------|",
            f"| Peak Bandwidth | {self.peak_bandwidth_gbs:.1f} GB/s |",
            f"| Sustained Bandwidth | {self.sustained_bandwidth_gbs:.1f} GB/s |",
            f"| Average Latency | {self.average_latency_ns:.1f} ns |",
            f"| P99 Latency | {self.p99_latency_ns:.1f} ns |",
            f"| Row Hit Rate | {self.row_hit_rate_percent:.1f}% |",
            f"| Bank Conflict Rate | {self.bank_conflict_rate_percent:.1f}% |",
            "",
        ])
        
        # Key findings
        if self.findings:
            lines.extend([
                "## Key Findings",
                "",
            ])
            for finding in self.findings:
                lines.append(f"- {finding}")
            lines.append("")
        
        # Bandwidth results
        if self.bandwidth:
            lines.extend([
                "## Bandwidth Results",
                "",
                f"- **Peak Bandwidth:** {self.bandwidth.peak_bandwidth_gbs:.1f} GB/s",
                f"- **Measured Bandwidth:** {self.bandwidth.measured_bandwidth_gbs:.1f} GB/s",
                f"- **Efficiency:** {self.bandwidth.peak_efficiency_percent:.1f}%",
                f"- **Refresh Overhead:** {self.bandwidth.refresh_overhead_percent:.2f}%",
                "",
            ])
        
        # Latency results
        if self.latency:
            lines.extend([
                "## Latency Results",
                "",
                f"- **Average:** {self.latency.average_latency_ns:.1f} ns",
                f"- **Median:** {self.latency.median_latency_ns:.1f} ns",
                f"- **P50:** {self.latency.p50_latency_ns:.1f} ns",
                f"- **P90:** {self.latency.p90_latency_ns:.1f} ns",
                f"- **P95:** {self.latency.p95_latency_ns:.1f} ns",
                f"- **P99:** {self.latency.p99_latency_ns:.1f} ns",
                f"- **P99.9:** {self.latency.p999_latency_ns:.1f} ns",
                f"- **Read Latency:** {self.latency.read_avg_latency_ns:.1f} ns",
                f"- **Write Latency:** {self.latency.write_avg_latency_ns:.1f} ns",
                "",
            ])
        
        # Scheduler results
        if self.scheduler:
            lines.extend([
                "## Scheduler Results",
                "",
                f"- **QoS Enabled:** {self.scheduler.qos_enabled}",
                f"- **Row Hit Rate:** {self.scheduler.row_hit_rate_percent:.1f}%",
                f"- **Bank Conflict Rate:** {self.scheduler.bank_conflict_rate_percent:.1f}%",
                f"- **Priority Latency Ratio:** {self.scheduler.priority_latency_ratio:.2f}x",
                f"- **Avg Queue Depth:** {self.scheduler.average_queue_depth:.1f}",
                "",
            ])
        
        # Comparison results
        if self.comparison:
            lines.extend([
                "## Configuration Comparison",
                "",
                "| Configuration | Peak BW | Measured BW | Efficiency | Avg Latency | vs Baseline |",
                "|--------------|---------|-------------|------------|-------------|-------------|",
            ])
            for config in self.comparison.configs:
                vs_base = f"{config.bandwidth_vs_baseline:.2f}x" if config.bandwidth_vs_baseline else "baseline"
                lines.append(
                    f"| {config.config_name} | {config.peak_bandwidth_gbs:.1f} GB/s | "
                    f"{config.measured_bandwidth_gbs:.1f} GB/s | {config.bandwidth_efficiency_percent:.1f}% | "
                    f"{config.average_latency_ns:.1f} ns | {vs_base} |"
                )
            lines.append("")
            
            if self.comparison.hbm4_vs_hbm3_bandwidth_speedup > 0:
                lines.extend([
                    "### HBM4 vs HBM3 Analysis",
                    "",
                    f"- **Bandwidth Speedup:** {self.comparison.hbm4_vs_hbm3_bandwidth_speedup:.2f}x",
                    f"- **Latency Improvement:** {self.comparison.hbm4_vs_hbm3_latency_improvement * 100:.1f}%",
                    "",
                ])
        
        # Warnings
        if self.warnings:
            lines.extend([
                "## Warnings",
                "",
            ])
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        lines.append("---")
        lines.append("*Report generated by HBM Performance Benchmark Suite*")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.to_markdown()


class BenchmarkRunner:
    """Main benchmark runner orchestrator"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark runner
        
        Args:
            config: Benchmark configuration (uses default if None)
        """
        self.config = config or BenchmarkConfig()
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        
        # Individual benchmarks
        self.bandwidth_benchmark: Optional[BandwidthBenchmark] = None
        self.latency_benchmark: Optional[LatencyBenchmark] = None
        self.scheduler_benchmark: Optional[SchedulerBenchmark] = None
        self.comparison_benchmark: Optional[ComparisonBenchmark] = None
        
        # Results
        self.report: Optional[BenchmarkReport] = None
    
    def run_bandwidth_benchmarks(self) -> BandwidthResult:
        """Run bandwidth benchmarks
        
        Returns:
            Aggregated bandwidth result
        """
        _logger.info("Running bandwidth benchmarks...")
        
        self.bandwidth_benchmark = BandwidthBenchmark(
            config=self.config.bandwidth,
            speed_grade="8Gbps"  # Default to 8Gbps
        )
        
        results = self.bandwidth_benchmark.run_all_tests()
        summary = self.bandwidth_benchmark.get_summary()
        
        _logger.info(f"Bandwidth benchmarks complete: {summary.measured_bandwidth_gbs:.1f} GB/s")
        
        return summary
    
    def run_latency_benchmarks(self) -> LatencyResult:
        """Run latency benchmarks
        
        Returns:
            Latency result
        """
        _logger.info("Running latency benchmarks...")
        
        self.latency_benchmark = LatencyBenchmark(
            config=self.config.latency,
            speed_grade="8Gbps"
        )
        
        result = self.latency_benchmark.run_latency_test()
        
        _logger.info(f"Latency benchmarks complete: avg={result.average_latency_ns:.1f}ns, "
                    f"p99={result.p99_latency_ns:.1f}ns")
        
        return result
    
    def run_scheduler_benchmarks(self) -> SchedulerResult:
        """Run scheduler benchmarks
        
        Returns:
            Scheduler result
        """
        _logger.info("Running scheduler benchmarks...")
        
        self.scheduler_benchmark = SchedulerBenchmark(
            config=self.config.scheduler,
            speed_grade="8Gbps"
        )
        
        results = self.scheduler_benchmark.run_all_tests()
        summary = self.scheduler_benchmark.get_summary()
        
        _logger.info(f"Scheduler benchmarks complete: row_hit={summary.row_hit_rate_percent:.1f}%, "
                    f"conflicts={summary.bank_conflict_rate_percent:.1f}%")
        
        return summary
    
    def run_comparison_benchmarks(self) -> ComparisonReport:
        """Run configuration comparison benchmarks
        
        Returns:
            Comparison report
        """
        _logger.info("Running comparison benchmarks...")
        
        self.comparison_benchmark = ComparisonBenchmark(
            config=self.config.comparison
        )
        
        report = self.comparison_benchmark.run_comparison()
        
        _logger.info(f"Comparison benchmarks complete: best={report.best_bandwidth_config} @ "
                    f"{report.best_bandwidth_gbs:.1f} GB/s")
        
        return report
    
    def run_all_benchmarks(self) -> BenchmarkReport:
        """Run all enabled benchmarks
        
        Returns:
            Comprehensive benchmark report
        """
        _logger.info("=" * 60)
        _logger.info("Starting HBM Performance Benchmark Suite")
        _logger.info("=" * 60)
        
        self.start_time = time.time()
        
        # Create report
        report = BenchmarkReport()
        report.timestamp = datetime.now().isoformat()
        report.config = str(self.config)
        
        # Run enabled benchmarks
        if self.config.run_bandwidth:
            _logger.info("-" * 40)
            _logger.info("Bandwidth Tests")
            _logger.info("-" * 40)
            report.bandwidth = self.run_bandwidth_benchmarks()
            report.peak_bandwidth_gbs = report.bandwidth.peak_bandwidth_gbs
            report.sustained_bandwidth_gbs = report.bandwidth.sustained_bandwidth_gbs
        
        if self.config.run_latency:
            _logger.info("-" * 40)
            _logger.info("Latency Tests")
            _logger.info("-" * 40)
            report.latency = self.run_latency_benchmarks()
            report.average_latency_ns = report.latency.average_latency_ns
            report.p99_latency_ns = report.latency.p99_latency_ns
        
        if self.config.run_scheduler:
            _logger.info("-" * 40)
            _logger.info("Scheduler Tests")
            _logger.info("-" * 40)
            report.scheduler = self.run_scheduler_benchmarks()
            report.row_hit_rate_percent = report.scheduler.row_hit_rate_percent
            report.bank_conflict_rate_percent = report.scheduler.bank_conflict_rate_percent
        
        if self.config.run_comparison:
            _logger.info("-" * 40)
            _logger.info("Configuration Comparison")
            _logger.info("-" * 40)
            report.comparison = self.run_comparison_benchmarks()
        
        self.end_time = time.time()
        report.duration_seconds = self.end_time - self.start_time
        
        # Generate findings
        report.findings = self._generate_findings(report)
        report.warnings = self._generate_warnings(report)
        
        self.report = report
        
        _logger.info("=" * 60)
        _logger.info(f"Benchmark Suite Complete ({report.duration_seconds:.2f}s)")
        _logger.info("=" * 60)
        
        return report
    
    def _generate_findings(self, report: BenchmarkReport) -> List[str]:
        """Generate key findings from results"""
        findings = []
        
        # Bandwidth findings
        if report.bandwidth:
            efficiency = report.bandwidth.peak_efficiency_percent
            if efficiency > 90:
                findings.append(f"Excellent bandwidth efficiency at {efficiency:.1f}% of theoretical peak")
            elif efficiency > 70:
                findings.append(f"Good bandwidth efficiency at {efficiency:.1f}% of theoretical peak")
            else:
                findings.append(f"Bandwidth efficiency below target at {efficiency:.1f}%")
            
            if report.bandwidth.refresh_overhead_percent > 2:
                findings.append(f"Refresh overhead significant at {report.bandwidth.refresh_overhead_percent:.2f}%")
        
        # Latency findings
        if report.latency:
            p99_to_avg = report.latency.p99_latency_ns / report.latency.average_latency_ns if report.latency.average_latency_ns > 0 else 0
            if p99_to_avg > 3:
                findings.append(f"High latency tail (P99/Avg = {p99_to_avg:.2f}x), consider QoS tuning")
            else:
                findings.append(f"Consistent latency distribution (P99/Avg = {p99_to_avg:.2f}x)")
        
        # Scheduler findings
        if report.scheduler:
            if report.scheduler.row_hit_rate_percent > 80:
                findings.append(f"Excellent row locality with {report.scheduler.row_hit_rate_percent:.1f}% hit rate")
            elif report.scheduler.row_hit_rate_percent < 50:
                findings.append(f"Row locality needs improvement ({report.scheduler.row_hit_rate_percent:.1f}% hit rate)")
            
            if report.scheduler.bank_conflict_rate_percent > 20:
                findings.append(f"High bank conflict rate ({report.scheduler.bank_conflict_rate_percent:.1f}%), consider address mapping")
        
        # Comparison findings
        if report.comparison and report.comparison.hbm4_vs_hbm3_bandwidth_speedup > 0:
            findings.append(f"HBM4 provides {report.comparison.hbm4_vs_hbm3_bandwidth_speedup:.2f}x bandwidth vs HBM3")
        
        if report.comparison and report.comparison.hbm4_vs_hbm3_latency_improvement > 0:
            findings.append(f"HBM4 improves latency by {report.comparison.hbm4_vs_hbm3_latency_improvement * 100:.1f}% vs HBM3")
        
        return findings
    
    def _generate_warnings(self, report: BenchmarkReport) -> List[str]:
        """Generate warnings from results"""
        warnings = []
        
        if report.bandwidth:
            if report.bandwidth.peak_efficiency_percent < 50:
                warnings.append("Bandwidth efficiency critically low - investigate bottlenecks")
            
            if report.bandwidth.refresh_overhead_percent > 5:
                warnings.append("Refresh overhead exceeds 5% - consider refresh optimization")
        
        if report.latency:
            if report.latency.average_latency_ns > 100:
                warnings.append("Average latency exceeds 100ns - investigate timing or queue delays")
        
        if report.scheduler:
            if report.scheduler.average_queue_depth > 50:
                warnings.append("High average queue depth - may indicate throughput bottleneck")
            
            if report.scheduler.queue_full_count > 0:
                warnings.append("Queue overflow events detected - requests rejected")
        
        return warnings
    
    def run_quick_benchmark(self) -> BenchmarkReport:
        """Run quick benchmark for fast iteration
        
        Returns:
            Benchmark report with reduced test scope
        """
        self.config = BenchmarkConfig.quick()
        return self.run_all_benchmarks()
    
    def run_comprehensive_benchmark(self) -> BenchmarkReport:
        """Run comprehensive benchmark for full validation
        
        Returns:
            Benchmark report with full test scope
        """
        self.config = BenchmarkConfig.comprehensive()
        return self.run_all_benchmarks()
    
    def save_report(self, filename: str, format: str = "markdown") -> str:
        """Save report to file
        
        Args:
            filename: Output filename
            format: Format ("markdown", "json", "text")
            
        Returns:
            Path to saved file
        """
        if not self.report:
            raise ValueError("No report available. Run benchmarks first.")
        
        if format == "markdown":
            content = self.report.to_markdown()
        elif format == "json":
            content = self.report.to_json()
        else:
            content = str(self.report)
        
        with open(filename, 'w') as f:
            f.write(content)
        
        _logger.info(f"Report saved to {filename}")
        return filename


# Convenience function for quick benchmarks
def run_quick_benchmark() -> BenchmarkReport:
    """Run a quick benchmark
    
    Returns:
        Benchmark report
    """
    runner = BenchmarkRunner(BenchmarkConfig.quick())
    return runner.run_all_benchmarks()


def run_comprehensive_benchmark() -> BenchmarkReport:
    """Run a comprehensive benchmark
    
    Returns:
        Benchmark report
    """
    runner = BenchmarkRunner(BenchmarkConfig.comprehensive())
    return runner.run_all_benchmarks()
