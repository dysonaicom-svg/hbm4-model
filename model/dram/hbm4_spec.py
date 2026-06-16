"""
HBM4 DRAM Specification Constants

Based on:
- JEDEC JESD270-4A HBM4 specification
- Ramulator 2.0 HBM3 timing reference
- Multi-agent research findings (2026-06-15)

Key differences from HBM3:
- 32 channels (vs 8 channels in HBM3)
- 2048-bit interface (vs 1024-bit in HBM3)
- 64 pseudo-channels (vs 16 in HBM3)
- 8 GT/s base data rate (vs 6.4 GT/s in HBM3)
- 2 TB/s bandwidth per stack
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class HBM4Spec:
    """HBM4 DRAM specification constants

    This class defines all the key parameters for HBM4 memory,
    including channel configuration, timing parameters, and
    address mapping bit fields.

    Reference:
    - JEDEC JESD270-4A HBM4 specification
    - Synopsys DesignWare HBM4/4E Controller IP
    - Ramulator 2.0 HBM3 implementation
    - Multi-agent research synthesis (34 high-confidence facts)
    """

    # === Architecture Parameters ===
    channels: int = 32                    # HBM4: 32 channels
    pseudo_channels_per_channel: int = 2    # 2 pseudo-channels per channel
    banks_per_pseudo_channel: int = 16      # 16 banks per pseudo-channel
    bank_groups_per_channel: int = 8        # 8 bank groups per channel

    # === Physical Parameters ===
    io_width: int = 2048                   # 2048-bit (doubled from HBM3)
    data_rate_gtps: float = 8.0            # GT/s (base rate per pin)
    burst_length: int = 4                  # FLINE burst length
    row_size: int = 2048                   # bytes

    # === Calculated Values ===

    @property
    def pseudo_channels(self) -> int:
        """Total pseudo-channels = channels × pseudo_channels_per_channel"""
        return self.channels * self.pseudo_channels_per_channel

    @property
    def total_banks(self) -> int:
        """Total banks across all channels and pseudo-channels"""
        return self.channels * self.pseudo_channels_per_channel * self.banks_per_pseudo_channel

    @property
    def bandwidth(self) -> float:
        """Peak bandwidth in TB/s

        Formula: data_rate (Gb/s) × io_width (bits) / 8 / 1000
        Example: 8 GT/s × 2048 bits / 8 / 1000 = 2.048 TB/s
        """
        return self.data_rate_gtps * self.io_width / 8 / 1000

    @property
    def bandwidth_gbs(self) -> float:
        """Peak bandwidth in GB/s

        Formula: data_rate (Gb/s) × io_width (bits) / 8
        Example: 8 GT/s × 2048 bits / 8 = 2048 GB/s
        """
        return self.data_rate_gtps * self.io_width / 8

    # === Timing Parameters (cycles @ tCK) ===
    # Based on JEDEC JESD270-4A HBM4 specification
    # For 8 GT/s DDR: tCK = 1000/8 = 125 ps (not 1250 ps!)
    tCK_ps: float = 125.0                    # Clock period in ps (125ps = 8 GHz)
    nBL: int = 4                          # Burst length
    nCL: int = 8                          # CAS latency
    nRCDRD: int = 8                       # RAS to CAS delay (read)
    nRCDWR: int = 8                       # RAS to CAS delay (write)
    nRP: int = 8                          # Precharge command period
    nRAS: int = 20                        # Row active time
    nRC: int = 22                         # Row cycle time
    nWR: int = 8                          # Write recovery
    nRTPS: int = 2                        # Read to precharge
    nRTPL: int = 3                        # Read to precharge (last data)
    nCWL: int = 3                         # CAS write latency
    nCCDS: int = 2                        # Column command delay (same bank group)
    nCCDL: int = 3                        # Column command delay (different bank group)
    nRRDS: int = 3                        # RAS to RAS delay (same bank group)
    nRRDL: int = 4                        # RAS to RAS delay (different bank group)
    nWTRS: int = 4                        # Write to read turnaround (same bank group)
    nWTRL: int = 5                        # Write to read turnaround (different bank group)
    nRTW: int = 4                         # Read to write turnaround
    nFAW: int = 16                        # Four-activate window
    nRFC: int = 180                       # Refresh command duration
    nREFI: int = 3900                     # Refresh interval
    nRREFD: int = 8                       # Per-bank refresh interval

    # === Address Bit Fields ===
    # Based on DRAMSys HBM2 address mapping, extended for HBM4
    # Address format: [Stack][Channel][Pch][Bg][Bank][Row][Col][Burst]
    ADDR_STACK_BITS: int = 2              # 4 stacks
    ADDR_CHANNEL_BITS: int = 5            # 32 channels (5 bits vs HBM3's 3 bits)
    ADDR_PCH_BITS: int = 1                # 2 pseudo-channels per channel
    ADDR_BG_BITS: int = 3                 # 8 bank groups
    ADDR_BANK_BITS: int = 4              # 16 banks per group
    ADDR_ROW_BITS: int = 19               # 512K rows (for 4TB capacity)
    ADDR_COL_BITS: int = 6                # 64 columns
    ADDR_BURST_BITS: int = 2              # 4-beat burst alignment

    def get_channel_bits(self) -> Tuple[int, int]:
        """Return (start_bit, num_bits) for channel field"""
        return (0, self.ADDR_CHANNEL_BITS)

    def get_pseudo_channel_bits(self) -> Tuple[int, int]:
        """Return (start_bit, num_bits) for pseudo-channel field"""
        offset = self.ADDR_CHANNEL_BITS
        return (offset, self.ADDR_PCH_BITS)

    def get_bank_group_bits(self) -> Tuple[int, int]:
        """Return (start_bit, num_bits) for bank group field"""
        offset = self.ADDR_CHANNEL_BITS + self.ADDR_PCH_BITS
        return (offset, self.ADDR_BG_BITS)

    def get_bank_bits(self) -> Tuple[int, int]:
        """Return (start_bit, num_bits) for bank field"""
        offset = self.ADDR_CHANNEL_BITS + self.ADDR_PCH_BITS + self.ADDR_BG_BITS
        return (offset, self.ADDR_BANK_BITS)

    def get_row_bits(self) -> Tuple[int, int]:
        """Return (start_bit, num_bits) for row field"""
        offset = (self.ADDR_CHANNEL_BITS + self.ADDR_PCH_BITS +
                 self.ADDR_BG_BITS + self.ADDR_BANK_BITS)
        return (offset, self.ADDR_ROW_BITS)

    def get_column_bits(self) -> Tuple[int, int]:
        """Return (start_bit, num_bits) for column field"""
        offset = (self.ADDR_CHANNEL_BITS + self.ADDR_PCH_BITS +
                 self.ADDR_BG_BITS + self.ADDR_BANK_BITS + self.ADDR_ROW_BITS)
        return (offset, self.ADDR_COL_BITS)

    def get_burst_bits(self) -> Tuple[int, int]:
        """Return (start_bit, num_bits) for burst alignment field"""
        offset = (self.ADDR_CHANNEL_BITS + self.ADDR_PCH_BITS +
                 self.ADDR_BG_BITS + self.ADDR_BANK_BITS +
                 self.ADDR_ROW_BITS + self.ADDR_COL_BITS)
        return (offset, self.ADDR_BURST_BITS)

    def get_total_addr_bits(self) -> int:
        """Total address bits"""
        return (self.ADDR_STACK_BITS + self.ADDR_CHANNEL_BITS +
                self.ADDR_PCH_BITS + self.ADDR_BG_BITS +
                self.ADDR_BANK_BITS + self.ADDR_ROW_BITS +
                self.ADDR_COL_BITS + self.ADDR_BURST_BITS)


# Default HBM4 configuration
HBM4_CONFIG = HBM4Spec()

# Default timing values aligned with RTL hbm_types.svh HBM4_TIMING_DEFAULT
# Values in clock cycles @ 8 GT/s DDR (tCK = 125 ps)
# Reference: JEDEC JESD270-4A HBM4 specification
HBM4_DEFAULT_TIMING = {
    'tRCD': 8,    # RAS to CAS delay
    'tRP': 8,     # Row precharge time
    'tRAS': 20,   # Row active time
    'tRC': 22,    # Row cycle time
    'tCCD': 4,    # CAS-to-CAS delay
    'tRRD': 4,    # Row-to-row delay
    'tFAW': 16,   # Four-activate window
    'tRFC': 180,  # Refresh cycle time
    'tREFI': 3900,# Refresh interval
    'tCL': 8,     # CAS latency
    'tCWL': 3,    # CAS write latency
}

# Speed grade presets
# These allow configuration for different vendor speed grades
HBM4_SPEED_GRADES = {
    # JEDEC baseline
    "8Gbps": {
        "data_rate_gtps": 8.0,
        "tCK_ps": 125.0,  # 1000/8 = 125 ps for 8 GT/s DDR
        "description": "JEDEC HBM4 baseline"
    },
    # Extended rate (e.g., Cadence HBM4E)
    "12Gbps": {
        "data_rate_gtps": 12.0,
        "tCK_ps": 83.33,   # 1000/12 = 83.33 ps for 12 GT/s DDR
        "description": "12 GT/s extended rate"
    },
    # Maximum rate (e.g., Synopsys/Rambus HBM4E)
    "16Gbps": {
        "data_rate_gtps": 16.0,
        "tCK_ps": 62.5,    # 1000/16 = 62.5 ps for 16 GT/s DDR
        "description": "16 GT/s maximum rate (HBM4E compatible)"
    },
}


def create_hbm4_spec_from_speed_grade(speed_grade: str) -> HBM4Spec:
    """Create HBM4Spec with parameters from a speed grade preset

    Args:
        speed_grade: One of "8Gbps", "12Gbps", "16Gbps"

    Returns:
        HBM4Spec configured for the specified speed grade
    """
    if speed_grade not in HBM4_SPEED_GRADES:
        raise ValueError(f"Unknown speed grade: {speed_grade}. "
                        f"Available: {list(HBM4_SPEED_GRADES.keys())}")

    grade_params = HBM4_SPEED_GRADES[speed_grade]

    # Calculate timing parameters that scale with tCK
    tCK_ps = grade_params["tCK_ps"]

    # Timing values are in cycles, which scale with clock period
    # For simplicity, we keep cycles constant but tCK changes
    spec = HBM4Spec(
        data_rate_gtps=grade_params["data_rate_gtps"],
        tCK_ps=tCK_ps
    )

    return spec


def create_hbm4_spec_with_timing(speed_grade: str, timing_multiplier: float = 1.0) -> HBM4Spec:
    """Create HBM4Spec with speed-grade-appropriate timing parameters

    For higher data rates, some timing parameters may need adjustment
    to maintain signal integrity at the higher frequency.

    Args:
        speed_grade: One of "8Gbps", "12Gbps", "16Gbps"
        timing_multiplier: Scale factor for timing parameters (1.0 = keep cycles constant)

    Returns:
        HBM4Spec configured for the specified speed grade with adjusted timing
    """
    spec = create_hbm4_spec_from_speed_grade(speed_grade)

    if timing_multiplier != 1.0:
        # Apply timing multiplier to cycle-based parameters
        # This allows for tighter or looser timing at higher frequencies
        int_params = ['nCL', 'nRCDRD', 'nRCDWR', 'nRP', 'nRAS', 'nRC',
                      'nWR', 'nRTPS', 'nRTPL', 'nCWL', 'nCCDS', 'nCCDL',
                      'nRRDS', 'nRRDL', 'nWTRS', 'nWTRL', 'nRTW', 'nFAW',
                      'nRFC', 'nREFI', 'nRREFD']
        for param in int_params:
            if hasattr(spec, param):
                setattr(spec, param, int(getattr(spec, param) * timing_multiplier))

    return spec


def calculate_bandwidth(data_rate_gtps: float, io_width: int) -> float:
    """Calculate peak bandwidth from data rate and interface width

    Args:
        data_rate_gtps: Data rate in GT/s (gigatransfers per second)
        io_width: Interface width in bits

    Returns:
        Bandwidth in TB/s
    """
    # Formula: data_rate (Gb/s) × io_width (bits) / 8 / 1000 = TB/s
    # Example: 8 GT/s × 2048 bits / 8 / 1000 = 2.048 TB/s
    return data_rate_gtps * io_width / 8 / 1000


def calculate_tCK_from_rate(data_rate_gtps: float) -> float:
    """Calculate clock period from data rate

    For DDR signaling with HBM4 at 8 GT/s:
    - Each GT/s = 1 billion transfers per second
    - tCK = 1000 / data_rate (ps per cycle)

    Args:
        data_rate_gtps: Data rate in GT/s

    Returns:
        Clock period in picoseconds
    """
    # HBM4 uses DDR, so tCK = 1000 / data_rate
    # Example: 8 GT/s → tCK = 1000/8 = 125 ps
    return 1000.0 / data_rate_gtps