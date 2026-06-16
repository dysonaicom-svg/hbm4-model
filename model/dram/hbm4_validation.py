"""
HBM4 Silicon Validation and Margin Analysis

Provides production validation checks for HBM4 DRAM silicon including:
- Silicon validation checks
- Margin analysis tools
- Performance guardbanding
- Reliability qualification support

Based on:
- JEDEC JESD238B HBM4 specification
- JEDEC JESD47 Stress Test Qualification
- Automotive AEC-Q100 standards
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics


class ValidationResult(Enum):
    """Validation result status"""
    PASS = "pass"
    FAIL = "fail"
    MARGINAL = "marginal"
    NOT_TESTED = "not_tested"


class TemperatureCorner(Enum):
    """Temperature test corners"""
    COLD = "cold"           # -40C
    ROOM = "room"           # 25C
    HOT = "hot"             # 105C
    HOT_EXTREME = "hot_extreme"  # 125C


class VoltageCorner(Enum):
    """Voltage test corners"""
    NOMINAL = "nominal"     # 1.0V
    MIN = "min"            # 0.88V (5% low)
    MAX = "max"            # 1.2V (20% high)
    MARGIN_LOW = "margin_low"  # 0.94V (production margin low)
    MARGIN_HIGH = "margin_high"  # 1.06V (production margin high)


@dataclass
class MarginResult:
    """Result of a margin test"""
    parameter: str
    measured_value: float
    spec_min: float
    spec_max: float
    margin_low: float  # How much below spec min before failure
    margin_high: float  # How much above spec max before failure
    status: ValidationResult

    @property
    def margin_percent(self) -> float:
        """Calculate margin as percentage of spec window"""
        spec_window = self.spec_max - self.spec_min
        if spec_window == 0:
            return 0.0
        min_margin = self.margin_low / spec_window * 100
        max_margin = self.margin_high / spec_window * 100
        return min(min_margin, max_margin)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "parameter": self.parameter,
            "measured_value": self.measured_value,
            "spec_range": f"{self.spec_min}-{self.spec_max}",
            "margin_low": f"{self.margin_low:.3f}",
            "margin_high": f"{self.margin_high:.3f}",
            "margin_percent": f"{self.margin_percent:.1f}%",
            "status": self.status.value,
        }


@dataclass
class SiliconValidationReport:
    """Complete silicon validation report"""
    speed_grade: str
    lot_id: str
    die_id: str
    test_temperature: float
    test_voltage: float
    timestamp: str

    # Per-parameter results
    timing_results: List[MarginResult] = field(default_factory=list)
    voltage_results: List[MarginResult] = field(default_factory=list)
    thermal_results: List[MarginResult] = field(default_factory=list)
    reliability_results: List[MarginResult] = field(default_factory=list)

    # Summary
    overall_status: ValidationResult = ValidationResult.NOT_TESTED
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    marginal_tests: int = 0

    def add_result(self, result: MarginResult) -> None:
        """Add a margin result to the appropriate category"""
        self.total_tests += 1

        if result.status == ValidationResult.PASS:
            self.passed_tests += 1
        elif result.status == ValidationResult.FAIL:
            self.failed_tests += 1
        elif result.status == ValidationResult.MARGINAL:
            self.marginal_tests += 1

        # Categorize by parameter type
        param_lower = result.parameter.lower()
        if "temp" in param_lower or "thermal" in param_lower:
            self.thermal_results.append(result)
        elif "v" in param_lower or "voltage" in param_lower or "ddq" in param_lower:
            self.voltage_results.append(result)
        elif any(x in param_lower for x in ["tCL", "tRCD", "tRP", "tRAS", "tRC", "tCK", "latency"]):
            self.timing_results.append(result)
        else:
            self.reliability_results.append(result)

        # Update overall status
        if self.overall_status == ValidationResult.NOT_TESTED:
            self.overall_status = result.status
        elif result.status == ValidationResult.FAIL:
            self.overall_status = ValidationResult.FAIL
        elif result.status == ValidationResult.MARGINAL and self.overall_status != ValidationResult.FAIL:
            self.overall_status = ValidationResult.MARGINAL

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage"""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "device_info": {
                "speed_grade": self.speed_grade,
                "lot_id": self.lot_id,
                "die_id": self.die_id,
                "temperature_C": self.test_temperature,
                "voltage_mV": self.test_voltage,
                "timestamp": self.timestamp,
            },
            "summary": {
                "status": self.overall_status.value,
                "total_tests": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "marginal": self.marginal_tests,
                "pass_rate": f"{self.pass_rate:.1f}%",
            },
            "timing": [r.to_dict() for r in self.timing_results],
            "voltage": [r.to_dict() for r in self.voltage_results],
            "thermal": [r.to_dict() for r in self.thermal_results],
            "reliability": [r.to_dict() for r in self.reliability_results],
        }


class SiliconValidator:
    """HBM4 Silicon validation and margin analysis"""

    # Spec limits for margin analysis
    TIMING_SPECS = {
        "tCK": (125.0, 125.0),  # (min, max) in ps - exact for DDR
        "tCL": (6, 12),         # CAS latency cycles
        "tRCD": (6, 12),        # RAS to CAS delay
        "tRP": (6, 12),         # Precharge time
        "tRAS": (16, 28),        # Row active time
        "tRC": (18, 30),         # Row cycle time
        "tRRD": (4, 8),         # Row to row delay
        "tFAW": (16, 32),       # Four activate window
    }

    VOLTAGE_SPECS = {
        "VDDQ": (880, 1200),    # Core voltage in mV
        "VDD": (880, 1200),
    }

    THERMAL_SPECS = {
        "Tj": (-40, 125),       # Junction temp in C
        "Tcase": (-40, 105),     # Case temp in C
    }

    # Production margin requirements (percentage)
    REQUIRED_MARGINS = {
        "timing": 10.0,          # 10% margin on timing
        "voltage": 5.0,          # 5% margin on voltage
        "thermal": 15.0,         # 15C margin on temperature
    }

    # Minimum acceptable margins for production
    MIN_ACCEPTABLE_MARGINS = {
        "timing": 5.0,           # 5% minimum timing margin
        "voltage": 3.0,           # 3% minimum voltage margin
        "thermal": 10.0,          # 10C minimum thermal margin
    }

    def __init__(self, speed_grade: str = "8Gbps"):
        """Initialize validator for a speed grade

        Args:
            speed_grade: Target speed grade ("8Gbps", "12Gbps", "16Gbps")
        """
        self.speed_grade = speed_grade
        self.results: List[SiliconValidationReport] = []

    def validate_timing_margin(self,
                              param: str,
                              measured_cycles: int,
                              tCK_ps: float,
                              test_corner: str = "nominal"
                              ) -> MarginResult:
        """Validate timing parameter margin

        Args:
            param: Parameter name (e.g., "tCL", "tRCD")
            measured_cycles: Measured value in cycles
            tCK_ps: Clock period in picoseconds
            test_corner: Test corner description

        Returns:
            MarginResult with margin analysis
        """
        spec_range = self.TIMING_SPECS.get(param, (0, 999))
        measured_ps = measured_cycles * tCK_ps

        # Calculate margins
        margin_low = measured_ps - spec_range[0] * tCK_ps
        margin_high = spec_range[1] * tCK_ps - measured_ps

        # Determine status based on margins
        required_margin = self.REQUIRED_MARGINS["timing"]
        min_margin = self.MIN_ACCEPTABLE_MARGINS["timing"]

        margin_pct = min(abs(margin_low), abs(margin_high)) / (spec_range[1] - spec_range[0]) / tCK_ps * 100

        if margin_pct >= required_margin:
            status = ValidationResult.PASS
        elif margin_pct >= min_margin:
            status = ValidationResult.MARGINAL
        else:
            status = ValidationResult.FAIL

        return MarginResult(
            parameter=f"{param}_{test_corner}",
            measured_value=measured_cycles,
            spec_min=spec_range[0],
            spec_max=spec_range[1],
            margin_low=margin_low / tCK_ps,  # Convert back to cycles
            margin_high=margin_high / tCK_ps,
            status=status,
        )

    def validate_voltage_margin(self,
                                voltage_mV: float,
                                param: str = "VDDQ"
                                ) -> MarginResult:
        """Validate voltage margin

        Args:
            voltage_mV: Measured voltage in millivolts
            param: Voltage parameter name

        Returns:
            MarginResult with margin analysis
        """
        spec_range = self.VOLTAGE_SPECS.get(param, (850, 1250))
        required_margin = self.REQUIRED_MARGINS["voltage"]
        min_margin = self.MIN_ACCEPTABLE_MARGINS["voltage"]

        margin_low = voltage_mV - spec_range[0]
        margin_high = spec_range[1] - voltage_mV

        margin_pct = min(margin_low, margin_high) / (spec_range[1] - spec_range[0]) * 100

        if margin_pct >= required_margin:
            status = ValidationResult.PASS
        elif margin_pct >= min_margin:
            status = ValidationResult.MARGINAL
        else:
            status = ValidationResult.FAIL

        return MarginResult(
            parameter=f"{param}_voltage",
            measured_value=voltage_mV,
            spec_min=spec_range[0],
            spec_max=spec_range[1],
            margin_low=margin_low,
            margin_high=margin_high,
            status=status,
        )

    def validate_thermal_margin(self,
                                temperature_C: float,
                                param: str = "Tj"
                                ) -> MarginResult:
        """Validate thermal margin

        Args:
            temperature_C: Measured temperature in Celsius
            param: Temperature parameter name

        Returns:
            MarginResult with margin analysis
        """
        spec_range = self.THERMAL_SPECS.get(param, (-55, 130))
        required_margin = self.REQUIRED_MARGINS["thermal"]
        min_margin = self.MIN_ACCEPTABLE_MARGINS["thermal"]

        margin_low = temperature_C - spec_range[0]
        margin_high = spec_range[1] - temperature_C

        margin_C = min(margin_low, margin_high)

        if margin_C >= required_margin:
            status = ValidationResult.PASS
        elif margin_C >= min_margin:
            status = ValidationResult.MARGINAL
        else:
            status = ValidationResult.FAIL

        return MarginResult(
            parameter=f"{param}_thermal",
            measured_value=temperature_C,
            spec_min=spec_range[0],
            spec_max=spec_range[1],
            margin_low=margin_low,
            margin_high=margin_high,
            status=status,
        )

    def analyze_DQ_eye(self,
                       eye_height_mV: float,
                       eye_width_ui: float,
                       speed_grade: str
                       ) -> Dict[str, Any]:
        """Analyze DQ data eye margins

        Args:
            eye_height_mV: Eye opening height in millivolts
            eye_width_ui: Eye opening width in UI
            speed_grade: Speed grade for context

        Returns:
            Dictionary with eye analysis
        """
        # Target margins
        target_height_mV = 50.0   # Minimum eye height
        target_width_ui = 0.3      # Minimum eye width (30% UI)

        height_ok = eye_height_mV >= target_height_mV
        width_ok = eye_width_ui >= target_width_ui

        # Calculate margins
        height_margin = eye_height_mV - target_height_mV
        width_margin = eye_width_ui - target_width_ui

        return {
            "eye_height_mV": eye_height_mV,
            "eye_width_ui": eye_width_ui,
            "height_margin_mV": height_margin,
            "width_margin_ui": width_margin,
            "height_pass": height_ok,
            "width_pass": width_ok,
            "overall_pass": height_ok and width_ok,
        }

    def analyze_DQS_eye(self,
                        eye_height_mV: float,
                        eye_width_ui: float
                        ) -> Dict[str, Any]:
        """Analyze DQS strobe eye margins

        Args:
            eye_height_mV: Eye opening height in millivolts
            eye_width_ui: Eye opening width in UI

        Returns:
            Dictionary with eye analysis
        """
        target_height_mV = 40.0
        target_width_ui = 0.25

        height_ok = eye_height_mV >= target_height_mV
        width_ok = eye_width_ui >= target_width_ui

        return {
            "eye_height_mV": eye_height_mV,
            "eye_width_ui": eye_width_ui,
            "height_margin_mV": eye_height_mV - target_height_mV,
            "width_margin_ui": eye_width_ui - target_width_ui,
            "height_pass": height_ok,
            "width_pass": width_ok,
            "overall_pass": height_ok and width_ok,
        }

    def run_full_validation(self,
                             lot_id: str = "TEST_LOT",
                             die_id: str = "TEST_DIE",
                             temperature_C: float = 25.0,
                             voltage_mV: float = 1000.0
                             ) -> SiliconValidationReport:
        """Run full silicon validation suite

        Args:
            lot_id: Lot identifier
            die_id: Die identifier
            temperature_C: Test temperature
            voltage_mV: Test voltage

        Returns:
            Complete validation report
        """
        from datetime import datetime

        report = SiliconValidationReport(
            speed_grade=self.speed_grade,
            lot_id=lot_id,
            die_id=die_id,
            test_temperature=temperature_C,
            test_voltage=voltage_mV,
            timestamp=datetime.now().isoformat(),
        )

        # Add thermal validation
        thermal_result = self.validate_thermal_margin(temperature_C)
        report.add_result(thermal_result)

        # Add voltage validation
        voltage_result = self.validate_voltage_margin(voltage_mV)
        report.add_result(voltage_result)

        return report

    def batch_analyze(self,
                      measurements: List[Dict[str, Any]]
                      ) -> List[Dict[str, Any]]:
        """Analyze batch of measurements statistically

        Args:
            measurements: List of measurement dictionaries with:
                - param: parameter name
                - value: measured value
                - unit: unit of measurement

        Returns:
            List of analysis results with statistics
        """
        # Group by parameter
        param_groups: Dict[str, List[float]] = {}
        for m in measurements:
            param = m["param"]
            if param not in param_groups:
                param_groups[param] = []
            param_groups[param].append(m["value"])

        results = []
        for param, values in param_groups.items():
            if len(values) < 2:
                continue

            mean_val = statistics.mean(values)
            stdev_val = statistics.stdev(values)
            min_val = min(values)
            max_val = max(values)

            results.append({
                "parameter": param,
                "count": len(values),
                "mean": mean_val,
                "stdev": stdev_val,
                "min": min_val,
                "max": max_val,
                "range": max_val - min_val,
                "cv_percent": (stdev_val / mean_val * 100) if mean_val > 0 else 0,
            })

        return results


class MarginAnalyzer:
    """Production margin analysis tools"""

    def __init__(self, target_yield_percent: float = 99.0):
        """Initialize margin analyzer

        Args:
            target_yield_percent: Target production yield percentage
        """
        self.target_yield = target_yield_percent / 100.0
        self.margin_budget: Dict[str, float] = {}

    def calculate_guardband(self,
                            distribution_mean: float,
                            distribution_stdev: float,
                            spec_limit: float,
                            direction: str = "upper"
                            ) -> float:
        """Calculate guardband for a spec limit given distribution

        Args:
            distribution_mean: Mean of measured distribution
            distribution_stdev: Standard deviation
            spec_limit: Spec limit value
            direction: "upper" or "lower"

        Returns:
            Guardband value to add to spec limit
        """
        import math
        # Z-score for target yield (99% = 2.33 sigma)
        z_score = 2.33

        if direction == "upper":
            # Guardband = mean + z*sigma - spec_limit
            guardband = distribution_mean + z_score * distribution_stdev - spec_limit
        else:
            # Guardband = spec_limit - (mean - z*sigma)
            guardband = spec_limit - (distribution_mean - z_score * distribution_stdev)

        return max(0, guardband)

    def analyze_margin_trend(self,
                               historical_data: List[Dict[str, float]]
                               ) -> Dict[str, Any]:
        """Analyze margin trends over time

        Args:
            historical_data: List of dicts with "timestamp" and margin values

        Returns:
            Trend analysis dictionary
        """
        if len(historical_data) < 2:
            return {"status": "insufficient_data"}

        margins = [d["margin"] for d in historical_data]
        times = [d["timestamp"] for d in historical_data]

        # Simple linear regression for trend
        n = len(margins)
        mean_margin = statistics.mean(margins)
        mean_time = sum(range(n)) / n

        numerator = sum((i - mean_time) * (m - mean_margin) for i, m in enumerate(margins))
        denominator = sum((i - mean_time) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Determine trend direction
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "improving"
        else:
            trend = "degrading"

        return {
            "trend": trend,
            "slope_per_lot": slope,
            "current_margin": margins[-1],
            "initial_margin": margins[0],
            "margin_change": margins[-1] - margins[0],
            "stability": "stable" if statistics.stdev(margins) < 5 else "variable",
        }

    def calculate_screening_threshold(self,
                                        distribution: List[float],
                                        fallout_percent: float = 0.1
                                        ) -> Tuple[float, float]:
        """Calculate screening thresholds for production

        Args:
            distribution: List of measured values
            fallout_percent: Acceptable fallout percentage (0.1 = 0.1%)

        Returns:
            Tuple of (lower_threshold, upper_threshold)
        """
        import math
        mean = statistics.mean(distribution)
        stdev = statistics.stdev(distribution)

        # Z-score for fallout
        z_score = 3.0  # ~0.135% fallout for 3 sigma

        lower = mean - z_score * stdev
        upper = mean + z_score * stdev

        return (lower, upper)


def create_validator(speed_grade: str) -> SiliconValidator:
    """Create silicon validator for a speed grade

    Args:
        speed_grade: One of "8Gbps", "12Gbps", "16Gbps"

    Returns:
        SiliconValidator instance
    """
    return SiliconValidator(speed_grade=speed_grade)


def run_production_validation(speed_grade: str = "8Gbps",
                               lot_id: str = "PROD_001",
                               temperature_C: float = 85.0,
                               voltage_mV: float = 1000.0
                               ) -> SiliconValidationReport:
    """Run production validation with standard settings

    Args:
        speed_grade: Target speed grade
        lot_id: Lot identifier
        temperature_C: Test temperature (default hot)
        voltage_mV: Test voltage (default nominal)

    Returns:
        Validation report
    """
    validator = create_validator(speed_grade)
    return validator.run_full_validation(
        lot_id=lot_id,
        die_id=f"{lot_id}_DIE",
        temperature_C=temperature_C,
        voltage_mV=voltage_mV,
    )