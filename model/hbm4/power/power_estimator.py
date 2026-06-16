"""
HBM4 Power Estimator

Estimates power consumption for HBM4 logic base die components.

Power breakdown:
- Command energy per class (ACT, PRE, RD, WR, REF, training, idle)
- PHY energy (TSV PHY, D2D PHY, DFI interface)
- Controller cluster power
- ECC/RAS logic power
- Clocking energy
- Power-down modes (PDN, SREF, etc.)
- Per-channel power tracking
- Aggregate stack power calculation

Based on:
- JEDEC JESD270-4A HBM4 specification
- Synopsys HBM4 Controller IP power estimates
- Cadence HBM4E power modeling
- Multi-agent research findings (2026-06-15)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

from model.dram.hbm4_spec import HBM4Spec


class PowerDownMode(Enum):
    """HBM4 power-down modes"""
    ACTIVE = "active"
    PDN = "pdn"           # Power-down (clock stopped)
    SREF = "sref"         # Self-refresh
    DPD = "dpd"           # Deep power-down (DRAM only)


@dataclass
class CommandEnergy:
    """Energy consumption per command type (in pJ)"""
    act: float = 0.0       # Activate
    pre: float = 0.0       # Precharge
    prea: float = 0.0      # Precharge all
    rd: float = 0.0       # Read
    wr: float = 0.0        # Write
    rda: float = 0.0       # Read with auto-precharge
    wra: float = 0.0       # Write with auto-precharge
    refab: float = 0.0    # All-bank refresh
    refsb: float = 0.0     # Per-bank refresh
    rfmab: float = 0.0     # Row flash memory refresh (all-bank)
    rfmsb: float = 0.0     # Row flash memory refresh (per-bank)
    training: float = 0.0  # Training commands (WR, RD, MPR)
    idle: float = 0.0      # Idle power (leakage)

    def total(self) -> float:
        """Sum of all command energies"""
        return (self.act + self.pre + self.prea + self.rd + self.wr +
                self.rda + self.wra + self.refab + self.refsb +
                self.rfmab + self.rfmsb + self.training + self.idle)


@dataclass
class PHYPower:
    """PHY power consumption breakdown (in mW)"""
    tsv_phy: float = 0.0        # TSV (Through-Silicon Via) PHY
    d2d_phy: float = 0.0        # D2D (Die-to-Die) PHY
    dfi_interface: float = 0.0 # DFI interface power
    analog_front_end: float = 0.0  # AFE (amplifier, equalization)

    def total(self) -> float:
        """Total PHY power"""
        return (self.tsv_phy + self.d2d_phy +
                self.dfi_interface + self.analog_front_end)


@dataclass
class ControllerPower:
    """Controller cluster power (in mW)"""
    address_decoder: float = 0.0  # Address decoding
    command_queue: float = 0.0    # Command queuing
    scheduling: float = 0.0        # Command scheduling
    refresh_logic: float = 0.0     # Refresh control logic
    qos_scheduler: float = 0.0     # QoS scheduling
    arbiter: float = 0.0          # Bus arbitration

    def total(self) -> float:
        """Total controller power"""
        return (self.address_decoder + self.command_queue +
                self.scheduling + self.refresh_logic +
                self.qos_scheduler + self.arbiter)


@dataclass
class ECCPower:
    """ECC and RAS logic power (in mW)"""
    ecc_encoder: float = 0.0   # ECC encoding
    ecc_decoder: float = 0.0   # ECC decoding
    crc_checker: float = 0.0   # CRC checking
    dbi_logic: float = 0.0    # Data Bus Inversion logic
    parity_checker: float = 0.0  # Command/address parity
    error_tracking: float = 0.0  # Error logging

    def total(self) -> float:
        """Total ECC/RAS power"""
        return (self.ecc_encoder + self.ecc_decoder +
                self.crc_checker + self.dbi_logic +
                self.parity_checker + self.error_tracking)


@dataclass
class ClockingPower:
    """Clock distribution power (in mW)"""
    pll: float = 0.0              # PLL clock generation
    clock_tree: float = 0.0        # Clock tree distribution
    clock_gating: float = 0.0      # Clock gating overhead
    dll: float = 0.0              # DLL for data capture

    def total(self) -> float:
        """Total clocking power"""
        return self.pll + self.clock_tree + self.clock_gating + self.dll


@dataclass
class PowerDownPower:
    """Power consumption in power-down modes (in mW)"""
    pdn_static: float = 0.0   # Power-down static power
    pdn_dynamic: float = 0.0  # Power-down dynamic (clock on)
    sref_static: float = 0.0   # Self-refresh static
    sref_dram: float = 0.0     # Self-refresh DRAM power
    dpd: float = 0.0           # Deep power-down (DRAM only)

    def get_mode_power(self, mode: PowerDownMode) -> float:
        """Get power for specific mode"""
        if mode == PowerDownMode.PDN:
            return self.pdn_static + self.pdn_dynamic
        elif mode == PowerDownMode.SREF:
            return self.sref_static + self.sref_dram
        elif mode == PowerDownMode.DPD:
            return self.dpd
        return 0.0


@dataclass
class ChannelPower:
    """Per-channel power breakdown (in mW)"""
    channel_id: int
    static_power: float = 0.0      # Static/leakage power
    dynamic_power: float = 0.0    # Dynamic switching power
    command_energy: CommandEnergy = field(default_factory=CommandEnergy)
    phy_power: PHYPower = field(default_factory=PHYPower)

    def total(self) -> float:
        """Total channel power"""
        return self.static_power + self.dynamic_power


@dataclass
class PowerBreakdown:
    """Complete power breakdown for HBM4 system"""
    command_energy: CommandEnergy = field(default_factory=CommandEnergy)
    phy_power: PHYPower = field(default_factory=PHYPower)
    controller_power: ControllerPower = field(default_factory=ControllerPower)
    ecc_power: ECCPower = field(default_factory=ECCPower)
    clocking_power: ClockingPower = field(default_factory=ClockingPower)
    power_down: PowerDownPower = field(default_factory=PowerDownPower)

    @property
    def total_static(self) -> float:
        """Total static power (mW)"""
        return (self.controller_power.total() + self.ecc_power.total() +
                self.clocking_power.total())

    @property
    def total_dynamic(self) -> float:
        """Total dynamic power (mW)"""
        return self.phy_power.total()

    @property
    def total_power(self) -> float:
        """Total power (mW)"""
        return self.total_static + self.total_dynamic


@dataclass
class PowerStats:
    """Runtime power statistics"""
    total_commands: int = 0
    act_count: int = 0
    pre_count: int = 0
    rd_count: int = 0
    wr_count: int = 0
    ref_count: int = 0
    training_cycles: int = 0
    idle_cycles: int = 0
    pdn_cycles: int = 0
    sref_cycles: int = 0
    energy_pJ: float = 0.0
    peak_power_mW: float = 0.0
    average_power_mW: float = 0.0

    def reset(self):
        """Reset all statistics"""
        self.total_commands = 0
        self.act_count = 0
        self.pre_count = 0
        self.rd_count = 0
        self.wr_count = 0
        self.ref_count = 0
        self.training_cycles = 0
        self.idle_cycles = 0
        self.pdn_cycles = 0
        self.sref_cycles = 0
        self.energy_pJ = 0.0
        self.peak_power_mW = 0.0
        self.average_power_mW = 0.0


class HBM4PowerEstimator:
    """HBM4 Power Estimator

    Estimates power consumption for HBM4 logic base die components.
    Supports both static analysis and runtime power tracking.

    Key features:
    - Per-command energy calculation
    - PHY power breakdown (TSV, D2D, DFI)
    - Controller cluster power
    - ECC/RAS logic power
    - Clocking power
    - Power-down mode power
    - Per-channel power tracking
    - Aggregate stack power

    Reference:
    - JEDEC JESD270-4A HBM4
    - Synopsys HBM4 Controller IP power data
    - Cadence HBM4E power models
    """

    # Default technology parameters (16nm logic base die)
    DEFAULT_VDD_MV = 0.9          # Supply voltage (V)
    DEFAULT_FREQ_MHZ = 800        # Operating frequency
    DEFAULT_CHANNEL_WIDTH = 64    # Bits per channel
    DEFAULT_TSV_PITCH_NM = 50     # TSV pitch (nm)
    DEFAULT_D2D_RATE_GBPS = 8.0   # D2D interface rate

    def __init__(
        self,
        spec: Optional[HBM4Spec] = None,
        vdd_mv: float = DEFAULT_VDD_MV,
        freq_mhz: float = DEFAULT_FREQ_MHZ,
    ):
        """Initialize HBM4 Power Estimator

        Args:
            spec: HBM4 specification (uses default if None)
            vdd_mv: Supply voltage in mV
            freq_mhz: Operating frequency in MHz
        """
        self.spec = spec if spec is not None else HBM4Spec()
        self.vdd_mv = vdd_mv
        self.freq_mhz = freq_mhz

        # Initialize power breakdowns
        self._init_command_energy()
        self._init_phy_power()
        self._init_controller_power()
        self._init_ecc_power()
        self._init_clocking_power()
        self._init_power_down_power()

        # Per-channel tracking
        self.channel_powers: List[ChannelPower] = [
            ChannelPower(channel_id=i)
            for i in range(self.spec.channels)
        ]

        # Runtime statistics
        self.stats = PowerStats()
        self._current_mode = PowerDownMode.ACTIVE

    def _init_command_energy(self):
        """Initialize command energy parameters

        Energy estimates based on HBM3/HBM4 controller IP data.
        Values in pJ per command.
        """
        # Base energy scales with voltage and frequency
        v_scale = (self.vdd_mv / 900.0) ** 2
        f_scale = self.freq_mhz / 800.0

        # ACT: Activate row (highest energy due to wordline charging)
        self.command_energy = CommandEnergy(
            act=320.0 * v_scale * f_scale,          # Row activation
            pre=85.0 * v_scale * f_scale,          # Precharge
            prea=140.0 * v_scale * f_scale,       # Precharge all
            rd=180.0 * v_scale * f_scale,          # Read burst
            wr=195.0 * v_scale * f_scale,          # Write burst
            rda=220.0 * v_scale * f_scale,         # Read with AP
            wra=235.0 * v_scale * f_scale,         # Write with AP
            refab=450.0 * v_scale * f_scale,       # All-bank refresh
            refsb=60.0 * v_scale * f_scale,        # Per-bank refresh
            rfmab=480.0 * v_scale * f_scale,      # Row flash refresh
            rfmsb=75.0 * v_scale * f_scale,        # Per-bank RFM
            training=280.0 * v_scale * f_scale,     # Training
            idle=12.0 * v_scale,                   # Idle per cycle
        )

    def _init_phy_power(self):
        """Initialize PHY power parameters

        Based on TSV/D2D PHY characteristics.
        """
        # TSV PHY: Scales with number of TSVs and data rate
        tsv_count = 1024  # Approximate TSV count per channel
        tsv_cap_ff = 50   # TSV capacitance (fF)

        tsv_power = (tsv_count * tsv_cap_ff * 1e-15 *
                     (self.spec.data_rate_gtps * 1e9) *
                     (self.vdd_mv * 1e-3) *
                     self.spec.channels * 0.3)  # Activity factor

        # D2D PHY: Die-to-die interface power
        d2d_width = 256  # D2D interface width
        d2d_power = (d2d_width * 40e-15 *  # D2D capacitance
                     self.DEFAULT_D2D_RATE_GBPS * 1e9 *
                     (self.vdd_mv * 1e-3) * 0.5)

        # DFI interface: Controller to PHY interface
        dfi_power = 45.0  # Fixed DFI overhead

        # AFE: Analog front-end (TX/RX, equalization)
        afe_power = 35.0

        self.phy_power = PHYPower(
            tsv_phy=tsv_power,
            d2d_phy=d2d_power,
            dfi_interface=dfi_power,
            analog_front_end=afe_power,
        )

    def _init_controller_power(self):
        """Initialize controller cluster power

        Logic power for address decoding, scheduling, etc.
        """
        # Base controller power scales with channel count
        channel_scale = self.spec.channels / 32.0

        self.controller_power = ControllerPower(
            address_decoder=18.0 * channel_scale,
            command_queue=22.0 * channel_scale,
            scheduling=28.0 * channel_scale,
            refresh_logic=12.0 * channel_scale,
            qos_scheduler=15.0 * channel_scale,
            arbiter=20.0 * channel_scale,
        )

    def _init_ecc_power(self):
        """Initialize ECC/RAS logic power"""
        # ECC power scales with data width
        width_scale = self.spec.io_width / 2048.0

        self.ecc_power = ECCPower(
            ecc_encoder=8.0 * width_scale,
            ecc_decoder=10.0 * width_scale,
            crc_checker=5.0 * width_scale,
            dbi_logic=6.0 * width_scale,
            parity_checker=3.0 * width_scale,
            error_tracking=2.0,
        )

    def _init_clocking_power(self):
        """Initialize clocking power"""
        # Clocking scales with frequency
        freq_scale = self.freq_mhz / 800.0

        self.clocking_power = ClockingPower(
            pll=25.0 * freq_scale,
            clock_tree=30.0 * freq_scale,
            clock_gating=8.0,
            dll=15.0 * freq_scale,
        )

    def _init_power_down_power(self):
        """Initialize power-down mode power"""
        v_scale = (self.vdd_mv / 900.0) ** 2

        self.power_down = PowerDownPower(
            pdn_static=2.0 * v_scale,    # Minimal logic retention
            pdn_dynamic=8.0 * v_scale,   # Clock running at low freq
            sref_static=1.0 * v_scale,   # Minimal retention
            sref_dram=15.0,               # DRAM self-refresh
            dpd=0.5,                      # Deep power-down (DRAM only)
        )

    def get_command_energy(self, cmd: str) -> float:
        """Get energy for a specific command

        Args:
            cmd: Command name ('ACT', 'PRE', 'RD', 'WR', etc.)

        Returns:
            Energy in pJ
        """
        cmd_map = {
            'ACT': self.command_energy.act,
            'PRE': self.command_energy.pre,
            'PREA': self.command_energy.prea,
            'RD': self.command_energy.rd,
            'WR': self.command_energy.wr,
            'RDA': self.command_energy.rda,
            'WRA': self.command_energy.wra,
            'REFAB': self.command_energy.refab,
            'REFSB': self.command_energy.refsb,
            'RFMAB': self.command_energy.rfmab,
            'RFMSB': self.command_energy.rfmsb,
            'TRAINING': self.command_energy.training,
            'IDLE': self.command_energy.idle,
        }
        return cmd_map.get(cmd.upper(), 0.0)

    def calculate_static_power(self) -> float:
        """Calculate static power consumption

        Returns:
            Static power in mW
        """
        return (self.controller_power.total() +
                self.ecc_power.total() +
                self.clocking_power.total() +
                self.phy_power.total())

    def calculate_dynamic_power(
        self,
        activity_factor: float = 0.3,
    ) -> float:
        """Calculate dynamic power consumption

        Args:
            activity_factor: Signal switching activity factor (0-1)

        Returns:
            Dynamic power in mW
        """
        base_power = self.phy_power.total() * activity_factor
        return base_power

    def calculate_total_power(
        self,
        activity_factor: float = 0.3,
    ) -> float:
        """Calculate total power consumption

        Args:
            activity_factor: Signal switching activity factor (0-1)

        Returns:
            Total power in mW
        """
        static = self.calculate_static_power()
        dynamic = self.calculate_dynamic_power(activity_factor)
        return static + dynamic

    def calculate_energy_per_cycle(
        self,
        cmd: str,
        data_width: int = 64,
    ) -> float:
        """Calculate energy consumed per command cycle

        Args:
            cmd: Command name
            data_width: Data width in bits

        Returns:
            Energy in pJ
        """
        base_energy = self.get_command_energy(cmd)

        # Scale by data width
        width_scale = data_width / self.DEFAULT_CHANNEL_WIDTH
        return base_energy * width_scale

    def set_power_down_mode(self, mode: PowerDownMode):
        """Set power-down mode

        Args:
            mode: Power-down mode
        """
        self._current_mode = mode

    def get_power_down_power(self, mode: PowerDownMode) -> float:
        """Get power for a power-down mode

        Args:
            mode: Power-down mode

        Returns:
            Power in mW
        """
        return self.power_down.get_mode_power(mode)

    def get_channel_power(self, channel_id: int) -> ChannelPower:
        """Get power breakdown for a specific channel

        Args:
            channel_id: Channel index (0-31)

        Returns:
            ChannelPower breakdown
        """
        if 0 <= channel_id < len(self.channel_powers):
            return self.channel_powers[channel_id]
        raise ValueError(f"Invalid channel_id: {channel_id}")

    def update_channel_power(
        self,
        channel_id: int,
        cmd: str,
        cycles: int = 1,
    ):
        """Update power statistics for a channel

        Args:
            channel_id: Channel index
            cmd: Command issued
            cycles: Number of cycles for this command
        """
        channel = self.get_channel_power(channel_id)

        energy = self.get_command_energy(cmd)
        channel.dynamic_power += energy * cycles / 1000.0  # Convert pJ to mW*ns

        # Update statistics
        self.stats.total_commands += 1
        if cmd == 'ACT':
            self.stats.act_count += 1
        elif cmd == 'PRE':
            self.stats.pre_count += 1
        elif cmd in ['RD', 'RDA']:
            self.stats.rd_count += 1
        elif cmd in ['WR', 'WRA']:
            self.stats.wr_count += 1
        elif cmd in ['REFab', 'REFsb', 'RFMab', 'RFMsb']:
            self.stats.ref_count += 1

        self.stats.energy_pJ += energy

        # Track peak power
        current_power = self.calculate_total_power()
        if current_power > self.stats.peak_power_mW:
            self.stats.peak_power_mW = current_power

    def calculate_stack_power(
        self,
        num_channels: int = 32,
        activity_factor: float = 0.3,
    ) -> float:
        """Calculate aggregate stack power

        Args:
            num_channels: Number of active channels
            activity_factor: Signal switching activity factor

        Returns:
            Total stack power in mW
        """
        per_channel = self.calculate_total_power(activity_factor)
        return per_channel * num_channels

    def get_power_breakdown(self) -> PowerBreakdown:
        """Get complete power breakdown

        Returns:
            PowerBreakdown with all components
        """
        return PowerBreakdown(
            command_energy=self.command_energy,
            phy_power=self.phy_power,
            controller_power=self.controller_power,
            ecc_power=self.ecc_power,
            clocking_power=self.clocking_power,
            power_down=self.power_down,
        )

    def estimate_power_for_pattern(
        self,
        pattern: List[Tuple[str, int]],
    ) -> Dict:
        """Estimate power for a command pattern

        Args:
            pattern: List of (command, count) tuples

        Returns:
            Dictionary with power estimates
        """
        total_energy = 0.0
        command_counts = {}

        for cmd, count in pattern:
            energy = self.get_command_energy(cmd)
            total_energy += energy * count
            command_counts[cmd] = count

        # Estimate time (simplified: assume 1 cycle per command)
        total_cycles = sum(count for _, count in pattern)
        time_us = total_cycles / (self.freq_mhz * 1e6)

        # Calculate average power
        avg_power_mw = (total_energy * 1e-12) / (time_us * 1e-6) if time_us > 0 else 0

        return {
            'total_energy_pJ': total_energy,
            'total_cycles': total_cycles,
            'time_us': time_us,
            'average_power_mW': avg_power_mw,
            'command_counts': command_counts,
        }

    def get_summary(self) -> Dict:
        """Get power summary

        Returns:
            Dictionary with power summary
        """
        breakdown = self.get_power_breakdown()

        return {
            'spec': {
                'channels': self.spec.channels,
                'io_width': self.spec.io_width,
                'data_rate': self.spec.data_rate_gtps,
                'bandwidth_tbps': self.spec.bandwidth,  # Peak bandwidth in TB/s
            },
            'parameters': {
                'vdd_mv': self.vdd_mv,
                'freq_mhz': self.freq_mhz,
            },
            'power_mW': {
                'static': breakdown.total_static,
                'dynamic': breakdown.total_dynamic,
                'total': breakdown.total_power,
                'phy': breakdown.phy_power.total(),
                'controller': breakdown.controller_power.total(),
                'ecc': breakdown.ecc_power.total(),
                'clocking': breakdown.clocking_power.total(),
            },
            'stats': {
                'total_commands': self.stats.total_commands,
                'peak_power_mW': self.stats.peak_power_mW,
                'total_energy_pJ': self.stats.energy_pJ,
            },
            'stack_power_mW': self.calculate_stack_power(),
        }

    def reset_stats(self):
        """Reset runtime statistics"""
        self.stats.reset()
        for channel in self.channel_powers:
            channel.static_power = 0.0
            channel.dynamic_power = 0.0


# Default technology parameters (16nm logic base die)
DEFAULT_VDD_MV = 0.9          # Supply voltage (V)
DEFAULT_FREQ_MHZ = 800        # Operating frequency
DEFAULT_CHANNEL_WIDTH = 64    # Bits per channel
DEFAULT_TSV_PITCH_NM = 50     # TSV pitch (nm)
DEFAULT_D2D_RATE_GBPS = 8.0   # D2D interface rate


# Factory function
def create_power_estimator(
    speed_grade: str = "8Gbps",
    vdd_mv: Optional[float] = None,
) -> HBM4PowerEstimator:
    """Create power estimator with specified configuration

    Args:
        speed_grade: Speed grade ('8Gbps', '12Gbps', '16Gbps')
        vdd_mv: Supply voltage in mV (uses default 0.9V if None)

    Returns:
        Configured HBM4PowerEstimator
    """
    from model.dram.hbm4_spec import create_hbm4_spec_from_speed_grade

    spec = create_hbm4_spec_from_speed_grade(speed_grade)

    # Adjust frequency based on speed grade
    freq_mhz = {
        '8Gbps': 800,
        '12Gbps': 1200,
        '16Gbps': 1600,
    }.get(speed_grade, 800)

    # Use default voltage if not specified
    if vdd_mv is None:
        vdd_mv = DEFAULT_VDD_MV

    return HBM4PowerEstimator(
        spec=spec,
        vdd_mv=vdd_mv,
        freq_mhz=freq_mhz,
    )