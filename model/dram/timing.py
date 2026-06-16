"""
HBM3 DRAM Timing Parameters
参考设计文档 2026-06-15-hbm-system-model-design.md 的 5.2.4 节

HBM3 时序参数定义
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HBM3Timing:
    """HBM3 时序参数

    所有参数以 cycles 为单位，基于 tCK = 781 ps (1.28 GHz)

    时序约束:
    - tRCD: RAS to CAS delay (激活行到可以发起读写)
    - tRP: Precharge time (关闭行的时间)
    - tRAS: Active to precharge (行打开的最短时间)
    - tRC: Row cycle (连续激活同一 bank 的最小间隔)
    - tCCD: CAS to CAS (连续突发最小间隔)
    - tRRD: Rank row to rank delay (不同 bank 激活间隔)
    - tFAW: Four bank activation window (4 个 bank 激活的时间窗口)
    - tRFC: Refresh cycle (刷新一个 bank group 的时间)
    - tREFI: Refresh interval (刷新间隔)

    Performance optimizations:
    - Pre-computed clock period in seconds for O(1) cycles_to_seconds conversion
    """

    # 时钟周期 (ps)
    tCK_ps: float = 781.25  # 1.28 GHz
    tCK_cycles: int = 1

    # 时序参数 (cycles)
    tRCD: int = 17       # RAS to CAS delay
    tRP: int = 17        # Precharge time
    tRAS: int = 42       # Active to precharge minimum
    tRC: int = 59        # Row cycle time
    tCCD: int = 5        # CAS to CAS delay
    tRRD: int = 5        # Rank row to rank delay
    tFAW: int = 26       # Four bank activation window
    tRFC: int = 295      # Refresh cycle (16Gb)
    tREFI: int = 5000    # Refresh interval (cycles)

    # Data timing
    tDQSCK: int = 3      # DQS output access time from CK
    tDQSQ: int = 2       # DQS-DQ skew
    tQHS: int = 2        # DQ hold DQS

    # Command timing
    tCMD: int = 1        # Command period

    # Pre-computed constants for performance
    _clock_period_ns: float = 0.78125  # Pre-computed clock period in ns
    _clock_period_s: float = 0.78125e-9  # Pre-computed clock period in seconds

    @property
    def clock_freq(self) -> float:
        """时钟频率 (Hz)"""
        return 1e12 / self.tCK_ps

    @property
    def clock_period_ns(self) -> float:
        """时钟周期 (ns)"""
        return self._clock_period_ns

    # HBM4-compatible aliases (n-prefix)
    @property
    def nRCD(self) -> int:
        """RAS to CAS delay - HBM4 compatible alias"""
        return self.tRCD

    @property
    def nRCDRD(self) -> int:
        """RAS to CAS delay (read)"""
        return self.tRCD

    @property
    def nRCDWR(self) -> int:
        """RAS to CAS delay (write)"""
        return self.tRCD

    @property
    def nRP(self) -> int:
        """Precharge time - HBM4 compatible alias"""
        return self.tRP

    @property
    def nRAS(self) -> int:
        """Active to precharge - HBM4 compatible alias"""
        return self.tRAS

    @property
    def nRC(self) -> int:
        """Row cycle time - HBM4 compatible alias"""
        return self.tRC

    @property
    def nCCD(self) -> int:
        """CAS to CAS delay - HBM4 compatible alias"""
        return self.tCCD

    @property
    def nRRD(self) -> int:
        """Rank row to rank delay - HBM4 compatible alias"""
        return self.tRRD

    @property
    def nFAW(self) -> int:
        """Four bank activation window - HBM4 compatible alias"""
        return self.tFAW

    @property
    def nRFC(self) -> int:
        """Refresh cycle - HBM4 compatible alias"""
        return self.tRFC

    @property
    def nREFI(self) -> int:
        """Refresh interval - HBM4 compatible alias"""
        return self.tREFI

    @property
    def nBL(self) -> int:
        """Burst length"""
        return 4

    @property
    def nRTW(self) -> int:
        """Read to write turnaround"""
        return 4

    @property
    def nWTRS(self) -> int:
        """Write to read (same BG)"""
        return 4

    @property
    def nWTRL(self) -> int:
        """Write to read (different BG)"""
        return 5

    @property
    def nRRDS(self) -> int:
        """RAS to RAS delay (same BG)"""
        return 3

    @property
    def nRRDL(self) -> int:
        """RAS to RAS delay (different BG)"""
        return 4

    # Additional HBM4-compatible aliases needed by CommandSequencer
    @property
    def nCCDS(self) -> int:
        """Column command delay (same BG)"""
        return self.tCCD

    @property
    def nCCDL(self) -> int:
        """Column command delay (different BG)"""
        return self.tCCD

    @property
    def nRTW(self) -> int:
        """Read to write turnaround"""
        return 4

    @property
    def nWTRS(self) -> int:
        """Write to read (same BG)"""
        return 4

    @property
    def nWTRL(self) -> int:
        """Write to read (different BG)"""
        return 5

    def cycles_to_ns(self, cycles: int) -> float:
        """Cycles 转换为 ns - optimized with pre-computed value"""
        return cycles * self._clock_period_ns

    def cycles_to_seconds(self, cycles: int) -> float:
        """Cycles 转换为 seconds - optimized with pre-computed value"""
        return cycles * self._clock_period_s

    def cycles_to_s(self, cycles: int) -> float:
        """Cycles 转换为 seconds - alias for cycles_to_seconds"""
        return cycles * self._clock_period_s

    def ns_to_cycles(self, ns: float) -> int:
        """ns 转换为 cycles"""
        return int(ns * 1000 / self.tCK_ps + 0.5)

    def __repr__(self) -> str:
        return (f"HBM3Timing(tCK={self.tCK_ps}ps, "
                f"tRCD={self.tRCD}, tRP={self.tRP}, tRAS={self.tRAS})")


# HBM2 时序参数 (对比参考)
@dataclass
class HBM2Timing:
    """HBM2 时序参数 (参考)"""
    tCK_ps: float = 1250.0  # 800 MHz
    tRCD: int = 14
    tRP: int = 14
    tRAS: int = 34
    tRC: int = 48
    tCCD: int = 4
    tRRD: int = 4
    tFAW: int = 20
    tRFC: int = 160  # 8Gb
    tREFI: int = 7800
    
    @property
    def clock_freq(self) -> float:
        return 1e12 / self.tCK_ps
    
    @property
    def clock_period_ns(self) -> float:
        return self.tCK_ps / 1000.0

    def cycles_to_ns(self, cycles: int) -> float:
        return cycles * self.clock_period_ns

    def cycles_to_seconds(self, cycles: int) -> float:
        return self.cycles_to_ns(cycles) * 1e-9

    def cycles_to_s(self, cycles: int) -> float:
        return self.cycles_to_seconds(cycles)


@dataclass
class HBM4Timing:
    """HBM4 时序参数 (基于 JEDEC JESD270-4A)

    使用 n-prefix 命名 (nRCD, nRP 等) 与 HBM4Spec 对齐。
    基于 tCK = 125 ps (8 GHz DDR)。

    Reference: JEDEC JESD270-4A HBM4 specification
    """
    tCK_ps: float = 125.0  # 125 ps = 8 GHz DDR

    # Row command timing
    nRCD: int = 8       # RAS to CAS delay (激活到读写)
    nRP: int = 8        # Precharge time
    nRAS: int = 20      # Row active time minimum
    nRC: int = 22       # Row cycle time (same bank)

    # Column command timing
    nCL: int = 8        # CAS latency
    nCWL: int = 3       # CAS write latency
    nCCD: int = 4       # CAS to CAS delay
    nCCDS: int = 2      # CAS to CAS delay (same BG)
    nCCDL: int = 3      # CAS to CAS delay (different BG)

    # Write recovery
    nWR: int = 8        # Write recovery
    nRTPS: int = 2      # Read to precharge
    nRTPL: int = 3      # Read to precharge (last data)

    # Bank timing
    nRRD: int = 4       # RAS to RAS delay
    nRRDS: int = 3      # RAS to RAS delay (same BG)
    nRRDL: int = 4      # RAS to RAS delay (different BG)
    nFAW: int = 16      # Four-activate window

    # Turnaround timing
    nWTRS: int = 4      # Write to read (same BG)
    nWTRL: int = 5      # Write to read (different BG)
    nRTW: int = 4       # Read to write

    # Refresh timing
    nRFC: int = 180     # Refresh cycle time
    nREFI: int = 3900   # Refresh interval

    # Burst length (FLINE = 4 beats = 32 bytes)
    nBL: int = 4

    @classmethod
    def for_speed_grade(cls, speed_gbps: float) -> "HBM4Timing":
        """Create HBM4Timing for a specific speed grade

        Args:
            speed_gbps: Data rate in GT/s (e.g., 8.0, 12.0, 16.0)

        Returns:
            HBM4Timing instance configured for the speed grade
        """
        tCK_ps = 1000.0 / speed_gbps

        # Scale timing parameters proportionally for higher speeds
        # Higher speeds may require tighter timing (fewer cycles for same absolute time)
        # For 16Gbps: tCK halves, so we can keep cycles constant for same ns timing
        timing = cls(tCK_ps=tCK_ps)
        return timing

    @classmethod
    def for_8gbps(cls) -> "HBM4Timing":
        """Create HBM4Timing for 8 GT/s (JEDEC baseline)"""
        return cls(tCK_ps=125.0)

    @classmethod
    def for_12gbps(cls) -> "HBM4Timing":
        """Create HBM4Timing for 12 GT/s (extended rate)"""
        return cls(tCK_ps=83.33)

    @classmethod
    def for_16gbps(cls) -> "HBM4Timing":
        """Create HBM4Timing for 16 GT/s (maximum rate / HBM4E)"""
        return cls(tCK_ps=62.5)

    @property
    def clock_freq(self) -> float:
        """时钟频率 (Hz)"""
        return 1e12 / self.tCK_ps

    @property
    def clock_period_ns(self) -> float:
        """时钟周期 (ns)"""
        return self.tCK_ps / 1000.0

    def cycles_to_ns(self, cycles: int) -> float:
        """Cycles 转换为 ns"""
        return cycles * self.clock_period_ns

    def cycles_to_seconds(self, cycles: int) -> float:
        """Cycles 转换为 seconds"""
        return self.cycles_to_ns(cycles) * 1e-9

    def cycles_to_s(self, cycles: int) -> float:
        """Cycles 转换为 seconds"""
        return self.cycles_to_seconds(cycles)

    def ns_to_cycles(self, ns: float) -> int:
        """ns 转换为 cycles"""
        return int(ns * 1000 / self.tCK_ps + 0.5)

    # Aliases for backward compatibility with HBM3 naming
    @property
    def nRCDRD(self) -> int:
        """RAS to CAS delay (read) - alias for nRCD"""
        return self.nRCD

    @property
    def nRCDWR(self) -> int:
        """RAS to CAS delay (write) - alias for nRCD"""
        return self.nRCD

    @property
    def tRCD(self) -> int:
        return self.nRCD

    @property
    def tRP(self) -> int:
        return self.nRP

    @property
    def tRAS(self) -> int:
        return self.nRAS

    @property
    def tRC(self) -> int:
        return self.nRC

    @property
    def tCCD(self) -> int:
        return self.nCCD

    @property
    def tRRD(self) -> int:
        return self.nRRD

    @property
    def tFAW(self) -> int:
        return self.nFAW

    @property
    def tRFC(self) -> int:
        return self.nRFC

    @property
    def tREFI(self) -> int:
        return self.nREFI

    def __repr__(self) -> str:
        return (f"HBM4Timing(tCK={self.tCK_ps}ps, "
                f"nRCD={self.nRCD}, nRP={self.nRP}, nRAS={self.nRAS}, nRC={self.nRC})")


def get_timing_for_hbm_version(version: str):
    """获取指定 HBM 版本的时序参数

    Args:
        version: "hbm2", "hbm3", "hbm4", or "hbm4_8gbps", "hbm4_12gbps", "hbm4_16gbps"

    Returns:
        对应版本的时序参数
    """
    versions = {
        "hbm2": HBM2Timing,
        "hbm3": HBM3Timing,
        "hbm4": HBM4Timing,
        "hbm4_8gbps": lambda: HBM4Timing.for_8gbps(),
        "hbm4_12gbps": lambda: HBM4Timing.for_12gbps(),
        "hbm4_16gbps": lambda: HBM4Timing.for_16gbps(),
    }
    timing_class_or_func = versions.get(version.lower())
    if not timing_class_or_func:
        raise ValueError(f"Unknown HBM version: {version}")
    if callable(timing_class_or_func) and not isinstance(timing_class_or_func, type):
        return timing_class_or_func()
    return timing_class_or_func()


# Speed grade mappings for convenience
SPEED_GRADE_TIMING = {
    "8Gbps": lambda: HBM4Timing.for_8gbps(),
    "12Gbps": lambda: HBM4Timing.for_12gbps(),
    "16Gbps": lambda: HBM4Timing.for_16gbps(),
}


def get_timing_for_speed_grade(speed_grade: str) -> HBM4Timing:
    """Get HBM4 timing parameters for a specific speed grade

    Args:
        speed_grade: "8Gbps", "12Gbps", or "16Gbps"

    Returns:
        HBM4Timing instance configured for the speed grade
    """
    if speed_grade not in SPEED_GRADE_TIMING:
        raise ValueError(f"Unknown speed grade: {speed_grade}. "
                        f"Available: {list(SPEED_GRADE_TIMING.keys())}")
    return SPEED_GRADE_TIMING[speed_grade]()


def timing_to_cycles(timing: HBM3Timing, time_ns: float) -> int:
    """将时间(ns)转换为周期数"""
    return int(time_ns * 1000 / timing.tCK_ps + 0.5)
