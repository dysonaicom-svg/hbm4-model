"""
HBM4 Production Specification Updates

Based on:
- JEDEC JESD238B HBM4 specification (final release)
- Production silicon validation requirements
- Speed grade characterization data

Key changes from draft spec:
- Production validation margins
- Silicon guardbanding parameters
- Speed grade validation ranges
- Reliability qualification limits
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SpeedGrade(Enum):
    """HBM4 speed grade identifiers"""
    SG_8G = "8Gbps"      # JEDEC baseline: 8 GT/s
    SG_12G = "12Gbps"    # Extended rate: 12 GT/s
    SG_16G = "16Gbps"    # Maximum rate: 16 GT/s (HBM4E)


class ValidationLevel(Enum):
    """Production validation levels"""
    ENGINEERING = "engineering"      # Engineering samples
    QUALIFICATION = "qualification"  # Full qualification
    PRODUCTION = "production"        # Production release
    AUTO_QUAL = "auto_qual"          # Automotive qualified


@dataclass
class HBM4ProductionSpec:
    """HBM4 production specification with validation parameters

    This class extends the base HBM4 spec with production-ready
    parameters including margins, guardbands, and validation limits.
    """

    # === Base Architecture (mirrors HBM4Spec) ===
    channels: int = 32
    pseudo_channels_per_channel: int = 2
    banks_per_pseudo_channel: int = 16
    bank_groups_per_channel: int = 8
    io_width: int = 2048

    # === Speed Grade ===
    speed_grade: SpeedGrade = SpeedGrade.SG_8G
    data_rate_gtps: float = 8.0
    tCK_ps: float = 125.0

    # === Production Margins (expressed as percentage of spec) ===
    # Timing margins for production silicon
    timing_margin_percent: float = 10.0      # 10% margin on all timing
    voltage_margin_percent: float = 5.0       # 5% margin on VDD
    temperature_margin_celsius: float = 15.0  # 15C margin from spec limits

    # Read margin (how much earlier/later sampling can work)
    read_margin_ui: float = 0.15              # 0.15 UI read DQ margin
    write_margin_ui: float = 0.15            # 0.15 UI write DQ margin
    DQS_margin_ui: float = 0.20               # 0.20 UI DQS margin

    # === Validation Limits ===
    max_read_latency_cycles: int = 20         # Maximum read latency (tAA max)
    max_write_latency_cycles: int = 12        # Maximum write latency
    min_tRP_cycles: int = 6                    # Minimum precharge time
    min_tRAS_cycles: int = 16                  # Minimum row active time
    min_tRC_cycles: int = 18                  # Minimum row cycle time

    # === Speed Grade Validation Ranges ===
    # These define the valid range for each speed grade
    speed_grade_validation: Dict[SpeedGrade, Dict] = field(default_factory=lambda: {
        SpeedGrade.SG_8G: {
            "data_rate_range": (7.6, 8.4),         # 8 GT/s +/- 5%
            "tCK_range": (119.0, 131.6),           # ps
            "tCL_range": (6, 12),                   # cycles
            "tRCD_range": (6, 12),                 # cycles
            "tRP_range": (6, 12),                  # cycles
            "tRAS_range": (16, 28),                 # cycles
            "tRC_range": (18, 30),                  # cycles
            "valid_voltage_mV": (880, 1200),        # VDDQ in mV
            "valid_temp_C": (-40, 105),              # Commercial
            "characterization_temp_C": (0, 85),     # Test temperature
        },
        SpeedGrade.SG_12G: {
            "data_rate_range": (11.4, 12.6),       # 12 GT/s +/- 5%
            "tCK_range": (79.4, 87.7),              # ps
            "tCL_range": (8, 14),                   # cycles
            "tRCD_range": (8, 14),                  # cycles
            "tRP_range": (8, 14),                   # cycles
            "tRAS_range": (20, 32),                 # cycles
            "tRC_range": (22, 36),                  # cycles
            "valid_voltage_mV": (880, 1200),
            "valid_temp_C": (-40, 105),
            "characterization_temp_C": (0, 85),
        },
        SpeedGrade.SG_16G: {
            "data_rate_range": (15.2, 16.8),       # 16 GT/s +/- 5%
            "tCK_range": (59.5, 65.8),              # ps
            "tCL_range": (10, 18),                  # cycles
            "tRCD_range": (10, 18),                 # cycles
            "tRP_range": (10, 18),                  # cycles
            "tRAS_range": (24, 40),                 # cycles
            "tRC_range": (26, 44),                  # cycles
            "valid_voltage_mV": (880, 1200),
            "valid_temp_C": (-40, 105),
            "characterization_temp_C": (0, 85),
        },
    })

    # === Reliability Qualification Limits ===
    refresh_temp_threshold_C: float = 85.0     # Temp threshold for extended refresh
    operating_voltage_max_mV: int = 1200        # Maximum operating voltage
    operating_voltage_nom_mV: int = 1000        # Nominal operating voltage
    operating_voltage_min_mV: int = 880        # Minimum operating voltage
    junction_temp_max_C: int = 105             # Maximum junction temperature
    junction_temp_hot_C: int = 115            # Hot temperature for refresh
    thermal_resistance_C_per_W: float = 2.5    # Theta-JA estimate

    # === Error Detection ===
    ecc_enabled: bool = True
    crc_enabled: bool = True
    crc_polynomial: int = 0x1D    # CRC-8 polynomial for data integrity
    ecc_scrub_interval_cycles: int = 1000000  # Background ECC scrub

    # === DRAM Array Margins ===
    sense_amp_offset_mV: float = 30.0     # Sense amp mismatch tolerance
    wordline_margin_mV: float = 50.0      # Wordline programming margin
    bitline_margin_mV: float = 40.0       # Bitline sensing margin
    ref_vref_tolerance_mV: float = 20.0   # VREF tolerance

    def get_timing_with_margin(self, base_value: int, margin_type: str = "timing") -> int:
        """Calculate timing value with production margin

        Args:
            base_value: Base timing value in cycles
            margin_type: Type of margin ("timing", "voltage", or "temp")

        Returns:
            Timing value with margin applied
        """
        if margin_type == "timing":
            margin = self.timing_margin_percent / 100.0
            return int(base_value * (1 - margin))  # Margin means spec is tighter
        elif margin_type == "voltage":
            margin = self.voltage_margin_percent / 100.0
            return int(base_value * (1 - margin))
        else:
            return base_value

    def get_valid_timing_range(self, parameter: str, base_value: int) -> Tuple[int, int]:
        """Get valid range for a timing parameter with production margins

        Args:
            parameter: Parameter name (e.g., "tCL", "tRCD")
            base_value: Base/spec value for the parameter

        Returns:
            Tuple of (min_valid, max_valid) values
        """
        # Production must work with margin applied
        min_val = self.get_timing_with_margin(base_value, "timing")
        max_val = base_value  # Max is spec value
        return (min_val, max_val)

    def get_DQ_margin_ps(self) -> float:
        """Get DQ data eye margin in picoseconds

        Returns:
            Margin in ps based on UI at current speed grade
        """
        ui_ps = self.tCK_ps / 2  # DDR, so UI is half period
        return ui_ps * self.read_margin_ui

    def get_DQS_margin_ps(self) -> float:
        """Get DQS strobe margin in picoseconds

        Returns:
            Margin in ps based on UI at current speed grade
        """
        ui_ps = self.tCK_ps / 2
        return ui_ps * self.DQS_margin_ui

    def get_voltage_margin_mV(self) -> float:
        """Get voltage margin in millivolts

        Returns:
            Voltage margin in mV
        """
        return self.operating_voltage_nom_mV * self.voltage_margin_percent / 100.0


# Production speed grade configurations with validation
HBM4_PRODUCTION_GRADES = {
    "8Gbps": {
        "speed_grade": SpeedGrade.SG_8G,
        "data_rate_gtps": 8.0,
        "tCK_ps": 125.0,
        "tCL_cycles": 8,
        "tRCD_cycles": 8,
        "tRP_cycles": 8,
        "tRAS_cycles": 20,
        "tRC_cycles": 22,
        "voltage_mV": 1000,
        "description": "JEDEC HBM4 Production Baseline",
        "qualification_required": True,
        " AEC-Q100": False,
    },
    "12Gbps": {
        "speed_grade": SpeedGrade.SG_12G,
        "data_rate_gtps": 12.0,
        "tCK_ps": 83.33,
        "tCL_cycles": 10,
        "tRCD_cycles": 10,
        "tRP_cycles": 10,
        "tRAS_cycles": 24,
        "tRC_cycles": 26,
        "voltage_mV": 1000,
        "description": "HBM4 Extended Rate Production",
        "qualification_required": True,
        "AEC-Q100": False,
    },
    "16Gbps": {
        "speed_grade": SpeedGrade.SG_16G,
        "data_rate_gtps": 16.0,
        "tCK_ps": 62.5,
        "tCL_cycles": 12,
        "tRCD_cycles": 12,
        "tRP_cycles": 12,
        "tRAS_cycles": 28,
        "tRC_cycles": 30,
        "voltage_mV": 1000,
        "description": "HBM4E Maximum Rate Production",
        "qualification_required": True,
        "AEC-Q100": False,
    },
}


def create_production_spec(speed_grade: str,
                           validation_level: ValidationLevel = ValidationLevel.PRODUCTION
                           ) -> HBM4ProductionSpec:
    """Create production specification for a speed grade

    Args:
        speed_grade: One of "8Gbps", "12Gbps", "16Gbps"
        validation_level: Production validation level required

    Returns:
        HBM4ProductionSpec configured for the speed grade and validation level
    """
    if speed_grade not in HBM4_PRODUCTION_GRADES:
        raise ValueError(f"Unknown speed grade: {speed_grade}")

    grade = HBM4_PRODUCTION_GRADES[speed_grade]

    # Adjust margins based on validation level
    margin_mult = {
        ValidationLevel.ENGINEERING: 1.5,
        ValidationLevel.QUALIFICATION: 1.2,
        ValidationLevel.PRODUCTION: 1.0,
        ValidationLevel.AUTO_QUAL: 0.9,
    }[validation_level]

    spec = HBM4ProductionSpec(
        speed_grade=grade["speed_grade"],
        data_rate_gtps=grade["data_rate_gtps"],
        tCK_ps=grade["tCK_ps"],
        timing_margin_percent=10.0 * margin_mult,
        voltage_margin_percent=5.0 * margin_mult,
    )

    return spec


def get_speed_grade_limits(speed_grade: str) -> Dict:
    """Get validation limits for a speed grade

    Args:
        speed_grade: Speed grade string

    Returns:
        Dictionary of validation limits
    """
    spec = HBM4ProductionSpec()
    sg = SpeedGrade[speed_grade_enum_name(speed_grade)]
    return spec.speed_grade_validation[sg]


def validate_timing_parameter(param_name: str,
                              value: int,
                              speed_grade: str) -> Tuple[bool, str]:
    """Validate a timing parameter against production limits

    Args:
        param_name: Parameter name (e.g., "tCL", "tRCD")
        value: Measured value in cycles
        speed_grade: Speed grade string

    Returns:
        Tuple of (is_valid, reason)
    """
    limits = get_speed_grade_limits(speed_grade)

    # Map parameter names to limit keys
    param_map = {
        "tCL": "tCL_range",
        "tRCD": "tRCD_range",
        "tRP": "tRP_range",
        "tRAS": "tRAS_range",
        "tRC": "tRC_range",
    }

    limit_key = param_map.get(param_name)
    if limit_key is None:
        return False, f"Unknown parameter: {param_name}"

    limit_range = limits.get(limit_key)
    if limit_range is None:
        return False, f"No validation range for {param_name}"

    if limit_range[0] <= value <= limit_range[1]:
        return True, f"{param_name}={value} is within valid range {limit_range}"
    else:
        return False, f"{param_name}={value} is outside valid range {limit_range}"


def validate_voltage(voltage_mV: float, speed_grade: str) -> Tuple[bool, str]:
    """Validate voltage against production limits

    Args:
        voltage_mV: Measured voltage in millivolts
        speed_grade: Speed grade string

    Returns:
        Tuple of (is_valid, reason)
    """
    limits = get_speed_grade_limits(speed_grade)
    voltage_range = limits.get("valid_voltage_mV", (850, 1250))

    if voltage_range[0] <= voltage_mV <= voltage_range[1]:
        return True, f"Voltage {voltage_mV}mV is within valid range"
    else:
        return False, f"Voltage {voltage_mV}mV is outside valid range {voltage_range}"


def validate_temperature(temp_C: float, speed_grade: str) -> Tuple[bool, str]:
    """Validate temperature against production limits

    Args:
        temp_C: Temperature in Celsius
        speed_grade: Speed grade string

    Returns:
        Tuple of (is_valid, reason)
    """
    limits = get_speed_grade_limits(speed_grade)
    temp_range = limits.get("valid_temp_C", (-55, 125))

    if temp_range[0] <= temp_C <= temp_range[1]:
        return True, f"Temperature {temp_C}C is within valid range"
    else:
        return False, f"Temperature {temp_C}C is outside valid range {temp_range}"


def speed_grade_enum_name(grade: str) -> str:
    """Convert speed grade string to enum name

    Args:
        grade: Speed grade string like "8Gbps"

    Returns:
        Enum member name like "SG_8G"
    """
    mapping = {
        "8Gbps": "SG_8G",
        "12Gbps": "SG_12G",
        "16Gbps": "SG_16G",
    }
    return mapping.get(grade, grade)