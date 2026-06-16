"""
HBM4 JEDEC Compliance Checker

Provides compliance verification for:
- JEDEC JESD238B HBM4 specification compliance
- Protocol compliance tests
- Timing compliance tests
- Signal integrity compliance

Based on:
- JEDEC JESD238B HBM4 Base Specification
- JEDEC JESD235 HBM3 Specification (for HBM3 compatibility modes)
- JEDEC JESD79-4 DDR5 SDRAM (for command protocol reference)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from dataclasses import dataclass


class ComplianceLevel(Enum):
    """Compliance check severity levels"""
    MANDATORY = "mandatory"      # Must pass for any compliance
    RECOMMENDED = "recommended"  # Should pass for full compliance
    OPTIONAL = "optional"        # Nice to have


class ComplianceStatus(Enum):
    """Compliance check result status"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


@dataclass
class ComplianceCheck:
    """Individual compliance check result"""
    check_id: str
    description: str
    level: ComplianceLevel
    status: ComplianceStatus
    details: str
    spec_reference: str
    measured_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "check_id": self.check_id,
            "description": self.description,
            "level": self.level.value,
            "status": self.status.value,
            "details": self.details,
            "spec_reference": self.spec_reference,
            "measured_value": str(self.measured_value) if self.measured_value is not None else None,
            "expected_value": str(self.expected_value) if self.expected_value is not None else None,
            "remediation": self.remediation,
        }


@dataclass
class ComplianceReport:
    """Complete compliance test report"""
    spec_version: str
    test_date: str
    device_info: Dict[str, str]
    checks: List[ComplianceCheck] = field(default_factory=list)

    # Summary by level
    mandatory_passed: int = 0
    mandatory_failed: int = 0
    recommended_passed: int = 0
    recommended_failed: int = 0
    optional_passed: int = 0
    optional_failed: int = 0

    def add_check(self, check: ComplianceCheck) -> None:
        """Add a compliance check result"""
        self.checks.append(check)

        if check.level == ComplianceLevel.MANDATORY:
            if check.status == ComplianceStatus.PASS:
                self.mandatory_passed += 1
            else:
                self.mandatory_failed += 1
        elif check.level == ComplianceLevel.RECOMMENDED:
            if check.status == ComplianceStatus.PASS:
                self.recommended_passed += 1
            else:
                self.recommended_failed += 1
        else:
            if check.status == ComplianceStatus.PASS:
                self.optional_passed += 1
            else:
                self.optional_failed += 1

    @property
    def overall_pass(self) -> bool:
        """Check if all mandatory tests passed"""
        return self.mandatory_failed == 0

    @property
    def compliance_percentage(self) -> float:
        """Calculate overall compliance percentage"""
        total = len(self.checks)
        if total == 0:
            return 0.0
        passed = sum(1 for c in self.checks if c.status == ComplianceStatus.PASS)
        return passed / total * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "spec_version": self.spec_version,
            "test_date": self.test_date,
            "device_info": self.device_info,
            "summary": {
                "overall_pass": self.overall_pass,
                "compliance_percentage": f"{self.compliance_percentage:.1f}%",
                "mandatory": f"{self.mandatory_passed}/{self.mandatory_passed + self.mandatory_failed}",
                "recommended": f"{self.recommended_passed}/{self.recommended_passed + self.recommended_failed}",
                "optional": f"{self.optional_passed}/{self.optional_passed + self.optional_failed}",
            },
            "checks": [c.to_dict() for c in self.checks],
        }


class HBM4ComplianceChecker:
    """JEDEC JESD238B HBM4 Compliance Checker"""

    # Spec compliance constants from JESD238B
    # Interface and signaling
    INTERFACE_WIDTH_OPTIONS = [1024, 2048]  # bits
    DATA_RATE_OPTIONS = [8.0, 12.0, 16.0]  # GT/s
    BURST_LENGTH = 4                        # FLINE burst
    DQ_BITS_PER_DQS = 8                     # 8-bit prefetch

    # Channel configuration
    CHANNELS_HBM4 = 32
    CHANNELS_HBM3_COMPAT = 16
    PSEUDO_CHANNELS_PER_CHANNEL = 2
    BANKS_PER_PSEUDO_CHANNEL = 16
    BANK_GROUPS = 8

    # Timing parameters (8 GT/s baseline)
    tCK_ps = 125.0
    tCL_cycles = 8
    tCWL_cycles = 3
    tRCD_cycles = 8
    tRP_cycles = 8
    tRAS_cycles = 20
    tRC_cycles = 22

    # Voltage levels
    VDDQ_NOMINAL_mV = 1000
    VDDQ_RANGE_mV = (880, 1200)
    VDD_NOMINAL_mV = 1000
    VDD_RANGE_mV = (880, 1200)

    # Temperature
    Tj_RANGE_C = (-40, 125)
    Tcase_RANGE_C = (-40, 105)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize compliance checker

        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        self.spec_version = "JESD238B"
        self.report: Optional[ComplianceReport] = None

    def _create_check(self,
                      check_id: str,
                      description: str,
                      level: ComplianceLevel,
                      status: ComplianceStatus,
                      details: str,
                      spec_ref: str,
                      measured: Any = None,
                      expected: Any = None,
                      remediation: str = None
                      ) -> ComplianceCheck:
        """Create a compliance check result"""
        return ComplianceCheck(
            check_id=check_id,
            description=description,
            level=level,
            status=status,
            details=details,
            spec_reference=spec_ref,
            measured_value=measured,
            expected_value=expected,
            remediation=remediation,
        )

    def check_interface_width(self, width: int) -> ComplianceCheck:
        """Check interface width compliance

        Args:
            width: Interface width in bits

        Returns:
            ComplianceCheck result
        """
        check_id = "IF_WIDTH_001"
        spec_ref = "JESD238B Section 4.1"

        if width in self.INTERFACE_WIDTH_OPTIONS:
            return self._create_check(
                check_id, "Interface width must be 1024 or 2048 bits",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"Width {width} is compliant", spec_ref, width, "1024 or 2048"
            )
        else:
            return self._create_check(
                check_id, "Interface width must be 1024 or 2048 bits",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"Width {width} is not a valid option", spec_ref, width, "1024 or 2048",
                "Use standard HBM4 interface width (1024 or 2048 bits)"
            )

    def check_data_rate(self, rate_gtps: float) -> ComplianceCheck:
        """Check data rate compliance

        Args:
            rate_gtps: Data rate in GT/s

        Returns:
            ComplianceCheck result
        """
        check_id = "DR_001"
        spec_ref = "JESD238B Section 4.2"

        if rate_gtps in self.DATA_RATE_OPTIONS:
            return self._create_check(
                check_id, "Data rate must be 8, 12, or 16 GT/s",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"Rate {rate_gtps} GT/s is compliant", spec_ref, rate_gtps, "8.0, 12.0, or 16.0"
            )
        else:
            return self._create_check(
                check_id, "Data rate must be 8, 12, or 16 GT/s",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"Rate {rate_gtps} GT/s is not a valid option", spec_ref, rate_gtps, "8.0, 12.0, or 16.0",
                "Use standard HBM4 data rate"
            )

    def check_channel_count(self, channels: int, hbm3_compat: bool = False) -> ComplianceCheck:
        """Check channel count compliance

        Args:
            channels: Number of channels
            hbm3_compat: Whether in HBM3 compatibility mode

        Returns:
            ComplianceCheck result
        """
        check_id = "CH_001"
        spec_ref = "JESD238B Section 4.3"

        expected = self.CHANNELS_HBM4 if not hbm3_compat else self.CHANNELS_HBM3_COMPAT
        valid = [self.CHANNELS_HBM4, self.CHANNELS_HBM3_COMPAT] if hbm3_compat else [self.CHANNELS_HBM4]


        if channels == expected or channels in valid:
            mode = "HBM4 native" if not hbm3_compat else "HBM3 compatible"
            return self._create_check(
                check_id, f"Channel count must be {expected} for {mode}",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"{channels} channels is compliant for {mode}", spec_ref, channels, expected
            )
        else:
            return self._create_check(
                check_id, f"Channel count must be {expected} for {'HBM4 native' if not hbm3_compat else 'HBM3 compatible'}",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"{channels} channels is not valid", spec_ref, channels, expected,
                f"Configure for {expected} channels"
            )

    def check_burst_length(self, bl: int) -> ComplianceCheck:
        """Check burst length compliance

        Args:
            bl: Burst length

        Returns:
            ComplianceCheck result
        """
        check_id = "BL_001"
        spec_ref = "JESD238B Section 5.2"

        if bl == self.BURST_LENGTH:
            return self._create_check(
                check_id, "Burst length must be 4 (FLINE)",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"BL={bl} is compliant", spec_ref, bl, self.BURST_LENGTH
            )
        else:
            return self._create_check(
                check_id, "Burst length must be 4 (FLINE)",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"BL={bl} is not compliant", spec_ref, bl, self.BURST_LENGTH,
                "Use burst length 4 for HBM4"
            )

    def check_cas_latency(self, cl: int, rate_gtps: float) -> ComplianceCheck:
        """Check CAS latency compliance

        Args:
            cl: CAS latency in cycles
            rate_gtps: Data rate for context

        Returns:
            ComplianceCheck result
        """
        check_id = "CL_001"
        spec_ref = "JESD238B Section 6.2"

        # CL scales inversely with data rate, typical values:
        # 8 GT/s: CL=8
        # 12 GT/s: CL=10-12
        # 16 GT/s: CL=12-14
        base_cl = int(8 * 8.0 / rate_gtps) if rate_gtps > 0 else 8
        # Add offset to account for higher latency needed at higher rates
        rate_offset = int((rate_gtps - 8.0) / 4.0) if rate_gtps > 0 else 0
        min_cl = base_cl + rate_offset
        max_cl = min_cl + 6  # Allow more margin for higher rates

        if min_cl <= cl <= max_cl:
            return self._create_check(
                check_id, f"CAS latency must be {min_cl}-{max_cl} cycles at {rate_gtps} GT/s",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"CL={cl} is compliant", spec_ref, cl, f"{min_cl}-{max_cl}"
            )
        else:
            return self._create_check(
                check_id, f"CAS latency must be {min_cl}-{max_cl} cycles at {rate_gtps} GT/s",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"CL={cl} is outside valid range", spec_ref, cl, f"{min_cl}-{max_cl}",
                "Adjust CAS latency to meet spec"
            )

    def check_voltage(self, vddq_mV: float) -> ComplianceCheck:
        """Check voltage compliance

        Args:
            vddq_mV: VDDQ voltage in millivolts

        Returns:
            ComplianceCheck result
        """
        check_id = "V_001"
        spec_ref = "JESD238B Section 3.1"

        if self.VDDQ_RANGE_mV[0] <= vddq_mV <= self.VDDQ_RANGE_mV[1]:
            return self._create_check(
                check_id, "VDDQ must be 880-1200 mV",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"VDDQ={vddq_mV}mV is compliant", spec_ref, vddq_mV, "880-1200"
            )
        else:
            return self._create_check(
                check_id, "VDDQ must be 880-1200 mV",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"VDDQ={vddq_mV}mV is outside valid range", spec_ref, vddq_mV, "880-1200",
                "Adjust VDDQ to 880-1200 mV range"
            )

    def check_temperature(self, tj_C: float) -> ComplianceCheck:
        """Check temperature compliance

        Args:
            tj_C: Junction temperature in Celsius

        Returns:
            ComplianceCheck result
        """
        check_id = "T_001"
        spec_ref = "JESD238B Section 3.3"

        if self.Tj_RANGE_C[0] <= tj_C <= self.Tj_RANGE_C[1]:
            return self._create_check(
                check_id, "Junction temperature must be -40C to 125C",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"Tj={tj_C}C is compliant", spec_ref, tj_C, "-40 to 125"
            )
        else:
            return self._create_check(
                check_id, "Junction temperature must be -40C to 125C",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"Tj={tj_C}C is outside valid range", spec_ref, tj_C, "-40 to 125",
                "Ensure thermal management keeps Tj in valid range"
            )

    def check_timing_parameter(self,
                                param_name: str,
                                measured_cycles: int,
                                expected_cycles: int,
                                tolerance_percent: float = 10.0
                                ) -> ComplianceCheck:
        """Check timing parameter compliance

        Args:
            param_name: Parameter name
            measured_cycles: Measured value
            expected_cycles: Expected/spec value
            tolerance_percent: Allowed tolerance

        Returns:
            ComplianceCheck result
        """
        check_id = f"TIM_{param_name}_001"
        spec_ref = "JESD238B Section 6"

        tolerance = expected_cycles * tolerance_percent / 100
        min_val = expected_cycles - tolerance
        max_val = expected_cycles + tolerance

        if min_val <= measured_cycles <= max_val:
            return self._create_check(
                check_id, f"{param_name} must be within {tolerance_percent}% of {expected_cycles}",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"{param_name}={measured_cycles} is compliant", spec_ref,
                measured_cycles, f"{expected_cycles} +/-{tolerance_percent}%"
            )
        else:
            return self._create_check(
                check_id, f"{param_name} must be within {tolerance_percent}% of {expected_cycles}",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"{param_name}={measured_cycles} is outside valid range", spec_ref,
                measured_cycles, f"{expected_cycles} +/-{tolerance_percent}%",
                f"Adjust {param_name} timing"
            )

    def check_ecc_enabled(self, ecc_enabled: bool) -> ComplianceCheck:
        """Check ECC enablement

        Args:
            ecc_enabled: Whether ECC is enabled

        Returns:
            ComplianceCheck result
        """
        check_id = "ECC_001"
        spec_ref = "JESD238B Section 8.3"

        if ecc_enabled:
            return self._create_check(
                check_id, "ECC should be enabled for production",
                ComplianceLevel.RECOMMENDED, ComplianceStatus.PASS,
                "ECC is enabled", spec_ref, True, True
            )
        else:
            return self._create_check(
                check_id, "ECC should be enabled for production",
                ComplianceLevel.RECOMMENDED, ComplianceStatus.WARN,
                "ECC is not enabled - consider enabling for reliability", spec_ref, False, True,
                "Enable ECC for improved reliability"
            )

    def check_crc_enabled(self, crc_enabled: bool) -> ComplianceCheck:
        """Check CRC enablement

        Args:
            crc_enabled: Whether CRC is enabled

        Returns:
            ComplianceCheck result
        """
        check_id = "CRC_001"
        spec_ref = "JESD238B Section 8.4"

        if crc_enabled:
            return self._create_check(
                check_id, "CRC should be enabled for production",
                ComplianceLevel.RECOMMENDED, ComplianceStatus.PASS,
                "CRC is enabled", spec_ref, True, True
            )
        else:
            return self._create_check(
                check_id, "CRC should be enabled for production",
                ComplianceLevel.RECOMMENDED, ComplianceStatus.WARN,
                "CRC is not enabled - consider enabling for data integrity", spec_ref, False, True,
                "Enable CRC for improved data integrity"
            )

    def check_refresh_timing(self,
                             tREFI_cycles: int,
                             tRFC_cycles: int,
                             rate_gtps: float
                             ) -> ComplianceCheck:
        """Check refresh timing compliance

        Args:
            tREFI_cycles: Refresh interval in cycles
            tRFC_cycles: Refresh cycle time in cycles
            rate_gtps: Data rate for context

        Returns:
            ComplianceCheck result
        """
        check_id = "REF_001"
        spec_ref = "JESD238B Section 7.2"

        # tREFI should be approximately 3900 cycles at 8 GT/s
        base_tREFI = 3900 * 8.0 / rate_gtps
        base_tRFC = 180 * 8.0 / rate_gtps

        # Allow 10% tolerance
        tREFI_ok = abs(tREFI_cycles - base_tREFI) / base_tREFI < 0.1
        tRFC_ok = abs(tRFC_cycles - base_tRFC) / base_tRFC < 0.1

        if tREFI_ok and tRFC_ok:
            return self._create_check(
                check_id, "Refresh timing must comply with JESD238B",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"tREFI={tREFI_cycles}, tRFC={tRFC_cycles} compliant", spec_ref,
                f"tREFI={tREFI_cycles}, tRFC={tRFC_cycles}",
                f"tREFI~{base_tREFI:.0f}, tRFC~{base_tRFC:.0f}"
            )
        else:
            return self._create_check(
                check_id, "Refresh timing must comply with JESD238B",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"Refresh timing out of spec: tREFI={tREFI_cycles}, tRFC={tRFC_cycles}", spec_ref,
                f"tREFI={tREFI_cycles}, tRFC={tRFC_cycles}",
                f"tREFI~{base_tREFI:.0f}, tRFC~{base_tRFC:.0f}",
                "Adjust refresh timing parameters"
            )

    def check_bank_group_timing(self,
                                 tCCD_cycles: int,
                                 tRRD_cycles: int,
                                 same_bank_group: bool = True
                                 ) -> ComplianceCheck:
        """Check bank group timing compliance

        Args:
            tCCD_cycles: CAS-to-CAS delay
            tRRD_cycles: RAS-to-RAS delay
            same_bank_group: Whether commands are to same bank group

        Returns:
            ComplianceCheck result
        """
        check_id = "BG_001"
        spec_ref = "JESD238B Section 6.4"

        if same_bank_group:
            expected_ccd = 4
            expected_rrd = 4
        else:
            expected_ccd = 6
            expected_rrd = 6

        ccd_ok = abs(tCCD_cycles - expected_ccd) <= 2
        rrd_ok = abs(tRRD_cycles - expected_rrd) <= 2

        if ccd_ok and rrd_ok:
            return self._create_check(
                check_id, "Bank group timing must comply with JESD238B",
                ComplianceLevel.MANDATORY, ComplianceStatus.PASS,
                f"tCCD={tCCD_cycles}, tRRD={tRRD_cycles} compliant", spec_ref,
                f"tCCD={tCCD_cycles}, tRRD={tRRD_cycles}",
                f"tCCD~{expected_ccd}, tRRD~{expected_rrd}"
            )
        else:
            return self._create_check(
                check_id, "Bank group timing must comply with JESD238B",
                ComplianceLevel.MANDATORY, ComplianceStatus.FAIL,
                f"Bank group timing out of spec", spec_ref,
                f"tCCD={tCCD_cycles}, tRRD={tRRD_cycles}",
                f"tCCD~{expected_ccd}, tRRD~{expected_rrd}",
                "Adjust bank group timing"
            )

    def run_protocol_compliance(self,
                                 device_config: Dict[str, Any]
                                 ) -> List[ComplianceCheck]:
        """Run all protocol compliance checks

        Args:
            device_config: Device configuration dictionary

        Returns:
            List of compliance check results
        """
        checks = []

        # Interface checks
        checks.append(self.check_interface_width(device_config.get("interface_width", 2048)))
        checks.append(self.check_data_rate(device_config.get("data_rate", 8.0)))
        checks.append(self.check_channel_count(
            device_config.get("channels", 32),
            device_config.get("hbm3_compat", False)
        ))
        checks.append(self.check_burst_length(device_config.get("burst_length", 4)))

        return checks

    def run_timing_compliance(self,
                               timing_config: Dict[str, Any]
                               ) -> List[ComplianceCheck]:
        """Run all timing compliance checks

        Args:
            timing_config: Timing configuration dictionary

        Returns:
            List of compliance check results
        """
        checks = []

        rate = timing_config.get("data_rate", 8.0)

        # CAS latency
        checks.append(self.check_cas_latency(
            timing_config.get("tCL", 8),
            rate
        ))

        # Voltage
        checks.append(self.check_voltage(timing_config.get("VDDQ", 1000)))

        # Temperature
        checks.append(self.check_temperature(timing_config.get("Tj", 25)))

        # Timing parameters
        checks.append(self.check_timing_parameter(
            "tRCD", timing_config.get("tRCD", 8), 8
        ))
        checks.append(self.check_timing_parameter(
            "tRP", timing_config.get("tRP", 8), 8
        ))
        checks.append(self.check_timing_parameter(
            "tRAS", timing_config.get("tRAS", 20), 20
        ))
        checks.append(self.check_timing_parameter(
            "tRC", timing_config.get("tRC", 22), 22
        ))

        # Refresh
        checks.append(self.check_refresh_timing(
            timing_config.get("tREFI", 3900),
            timing_config.get("tRFC", 180),
            rate
        ))

        # Bank group
        checks.append(self.check_bank_group_timing(
            timing_config.get("tCCD", 4),
            timing_config.get("tRRD", 4),
            True
        ))

        return checks

    def run_reliability_compliance(self,
                                    reliability_config: Dict[str, Any]
                                    ) -> List[ComplianceCheck]:
        """Run all reliability compliance checks

        Args:
            reliability_config: Reliability configuration dictionary

        Returns:
            List of compliance check results
        """
        checks = []

        checks.append(self.check_ecc_enabled(reliability_config.get("ecc_enabled", True)))
        checks.append(self.check_crc_enabled(reliability_config.get("crc_enabled", True)))

        return checks

    def run_full_compliance(self,
                            device_config: Dict[str, Any],
                            timing_config: Dict[str, Any],
                            reliability_config: Dict[str, Any],
                            device_info: Optional[Dict[str, str]] = None
                            ) -> ComplianceReport:
        """Run full compliance suite

        Args:
            device_config: Device configuration
            timing_config: Timing configuration
            reliability_config: Reliability configuration
            device_info: Optional device identification

        Returns:
            Complete compliance report
        """
        from datetime import datetime

        self.report = ComplianceReport(
            spec_version=self.spec_version,
            test_date=datetime.now().isoformat(),
            device_info=device_info or {},
        )

        # Run all check categories
        for check in self.run_protocol_compliance(device_config):
            self.report.add_check(check)

        for check in self.run_timing_compliance(timing_config):
            self.report.add_check(check)

        for check in self.run_reliability_compliance(reliability_config):
            self.report.add_check(check)

        return self.report


def run_jedec_compliance(device_config: Dict[str, Any],
                          timing_config: Dict[str, Any],
                          reliability_config: Optional[Dict[str, Any]] = None,
                          device_info: Optional[Dict[str, str]] = None
                          ) -> ComplianceReport:
    """Run JEDEC compliance with standard configuration

    Args:
        device_config: Device configuration
        timing_config: Timing configuration
        reliability_config: Optional reliability configuration
        device_info: Optional device identification info

    Returns:
        Compliance report
    """
    checker = HBM4ComplianceChecker()
    return checker.run_full_compliance(
        device_config=device_config,
        timing_config=timing_config,
        reliability_config=reliability_config or {"ecc_enabled": True, "crc_enabled": True},
        device_info=device_info,
    )


def validate_hbm4_device(config: Dict[str, Any]) -> Tuple[bool, ComplianceReport]:
    """Validate HBM4 device configuration for compliance

    Args:
        config: Complete device configuration

    Returns:
        Tuple of (is_compliant, report)
    """
    report = run_jedec_compliance(
        device_config=config.get("device", {}),
        timing_config=config.get("timing", {}),
        reliability_config=config.get("reliability", {}),
        device_info=config.get("info", {}),
    )

    return report.overall_pass, report