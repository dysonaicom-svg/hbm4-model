"""
HBM4 Power Consumption Model

Estimates power consumption based on:
- Active/Idle states
- Read/Write operations
- Refresh operations
- Temperature and process corners
- Per-command energy tracking
- Dynamic power calculation with activity factors

Reference:
- JEDEC JESD270-4A HBM4 specification
- Synopsys DesignWare HBM4 Power Analysis
- DRAM power models from academic research
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math
from datetime import datetime


class PowerState(Enum):
    """Power/operational states"""
    ACTIVE = "active"       # ACT command active
    READ = "read"           # Read operation
    WRITE = "write"         # Write operation
    REFRESH = "refresh"     # Refresh operation
    IDLE = "idle"           # Bank idle, powered up
    SELF_REFRESH = "self_refresh"  # Self-refresh mode
    POWER_DOWN = "power_down"      # Power-down mode


class ProcessCorner(Enum):
    """Process corner for power estimation"""
    SS = "slow_slow"        # Slow process corner
    TT = "typical"          # Typical process corner
    FF = "fast_fast"        # Fast process corner


class CommandType(Enum):
    """HBM4 DRAM command types"""
    ACT = "act"             # Activate
    PRE = "pre"             # Precharge
    PREA = "prea"           # Precharge all
    RD = "rd"               # Read
    WR = "wr"               # Write
    RDA = "rda"             # Read with auto-precharge
    WRA = "wra"            # Write with auto-precharge
    REFAB = "refab"         # All-bank refresh
    REFSB = "refsb"         # Per-bank refresh
    RFMAB = "rfmab"        # Row flash memory refresh (all-bank)
    RFMSB = "rfmsb"         # Row flash memory refresh (per-bank)
    MRW = "mrw"             # Mode register write
    MRR = "mrr"             # Mode register read
    PDN_ENTER = "pdn_enter"    # Enter power-down
    PDN_EXIT = "pdn_exit"      # Exit power-down
    SREF_ENTER = "sref_enter"   # Enter self-refresh
    SREF_EXIT = "sref_exit"     # Exit self-refresh


@dataclass
class PowerParameters:
    """Power consumption parameters (in mW unless noted)

    Based on JEDEC HBM4 specification and vendor data sheets.
    Values are typical for HBM4 at 8 GT/s, 1.1V VDDQ, 0.95V VDDQ2.
    """
    # === Active Power (per channel) ===
    active_power_ma: float = 350.0  # Active current (mA) - row open
    read_power_ma: float = 450.0    # Read operation power (mA)
    write_power_ma: float = 420.0   # Write operation power (mA)

    # === Idle/Standby Power (per channel) ===
    idle_power_ma: float = 50.0      # Idle (CK enabled) power (mA)
    standby_power_ma: float = 15.0   # CKE low standby (mA)

    # === Refresh Power ===
    refresh_power_ma: float = 380.0   # Refresh operation power (mA)
    refresh_cycle_ns: float = 7800.0  # tREFI in ns (7.8 us)

    # === Self-Refresh Power ===
    self_refresh_power_ma: float = 8.0  # Self-refresh mode (mA)

    # === Power-Down Power ===
    power_down_power_ma: float = 5.0   # Power-down mode (mA)

    # === Voltage Rails ===
    vddq_voltage: float = 1.1          # VDDQ voltage (V)
    vddq2_voltage: float = 0.95        # VDDQ2 voltage (V)
    vpp_voltage: float = 2.5          # VPP voltage (V)

    # === Per-Command Energy (pJ) ===
    # Based on JEDEC HBM4 timing and CADENCE/Synopsys estimates
    act_energy_pj: float = 320.0      # Activate energy (pJ)
    pre_energy_pj: float = 85.0        # Precharge energy (pJ)
    prea_energy_pj: float = 140.0     # Precharge all energy (pJ)
    rd_energy_pj: float = 180.0       # Read energy (pJ per burst)
    wr_energy_pj: float = 195.0       # Write energy (pJ per burst)
    rda_energy_pj: float = 220.0     # Read with auto-precharge (pJ)
    wra_energy_pj: float = 235.0     # Write with auto-precharge (pJ)
    refab_energy_pj: float = 450.0    # All-bank refresh (pJ)
    refsb_energy_pj: float = 60.0     # Per-bank refresh (pJ)
    rfmab_energy_pj: float = 480.0   # Row flash refresh all-bank (pJ)
    rfmsb_energy_pj: float = 75.0     # Row flash refresh per-bank (pJ)
    mrw_energy_pj: float = 120.0      # Mode register write (pJ)
    mrr_energy_pj: float = 100.0      # Mode register read (pJ)
    pdn_entry_energy_pj: float = 50.0  # Power-down entry (pJ)
    pdn_exit_energy_pj: float = 80.0   # Power-down exit (pJ)
    sref_entry_energy_pj: float = 60.0  # Self-refresh entry (pJ)
    sref_exit_energy_pj: float = 90.0    # Self-refresh exit (pJ)

    # === Process/Temperature Scaling ===
    process_corner: ProcessCorner = ProcessCorner.TT
    temperature_c: float = 45.0       # Junction temperature (C)

    @property
    def active_power_mw(self) -> float:
        """Active power in mW"""
        return self.active_power_ma * self.vddq_voltage

    @property
    def read_power_mw(self) -> float:
        """Read power in mW"""
        return self.read_power_ma * self.vddq_voltage

    @property
    def write_power_mw(self) -> float:
        """Write power in mW"""
        return self.write_power_ma * self.vddq_voltage

    @property
    def idle_power_mw(self) -> float:
        """Idle power in mW"""
        return self.idle_power_ma * self.vddq_voltage

    @property
    def refresh_power_mw(self) -> float:
        """Refresh power in mW"""
        return self.refresh_power_ma * self.vddq_voltage

    def get_process_scaling_factor(self) -> float:
        """Get power scaling factor for process corner

        Returns:
            Scaling factor (0.7-1.3 range typical)
        """
        scaling = {
            ProcessCorner.SS: 0.75,  # Slow corner - higher Vt, slower
            ProcessCorner.TT: 1.0,   # Typical corner
            ProcessCorner.FF: 1.25,  # Fast corner - lower Vt, faster
        }
        return scaling.get(self.process_corner, 1.0)

    def get_temperature_scaling_factor(self) -> float:
        """Get power scaling factor for temperature

        Returns:
            Scaling factor based on temperature
        """
        # Leakage increases exponentially with temperature
        # At 85C vs 45C: roughly 2x leakage increase
        base_temp = 45.0  # Reference temperature
        temp_diff = self.temperature_c - base_temp
        if temp_diff <= 0:
            return 1.0
        # Exponential leakage model: ~10% per 10C above reference
        return 1.0 + 0.1 * (temp_diff / 10.0)

    def get_effective_power_scale(self) -> float:
        """Get combined power scaling factor

        Returns:
            Combined scaling factor for power calculation
        """
        return self.get_process_scaling_factor() * self.get_temperature_scaling_factor()


@dataclass
class CommandEnergy:
    """Per-command energy tracking"""
    act_count: int = 0
    pre_count: int = 0
    prea_count: int = 0
    rd_count: int = 0
    wr_count: int = 0
    rda_count: int = 0
    wra_count: int = 0
    refab_count: int = 0
    refsb_count: int = 0
    rfmab_count: int = 0
    rfmsb_count: int = 0
    mrw_count: int = 0
    mrr_count: int = 0
    pdn_entry_count: int = 0
    pdn_exit_count: int = 0
    sref_entry_count: int = 0
    sref_exit_count: int = 0

    # Energy accumulators (pJ)
    total_act_energy_pj: float = 0.0
    total_pre_energy_pj: float = 0.0
    total_prea_energy_pj: float = 0.0
    total_rd_energy_pj: float = 0.0
    total_wr_energy_pj: float = 0.0
    total_rda_energy_pj: float = 0.0
    total_wra_energy_pj: float = 0.0
    total_refab_energy_pj: float = 0.0
    total_refsb_energy_pj: float = 0.0
    total_rfmab_energy_pj: float = 0.0
    total_rfmsb_energy_pj: float = 0.0
    total_mrw_energy_pj: float = 0.0
    total_mrr_energy_pj: float = 0.0
    total_pdn_entry_energy_pj: float = 0.0
    total_pdn_exit_energy_pj: float = 0.0
    total_sref_entry_energy_pj: float = 0.0
    total_sref_exit_energy_pj: float = 0.0

    @property
    def total_commands(self) -> int:
        """Total number of commands issued"""
        return (self.act_count + self.pre_count + self.prea_count +
                self.rd_count + self.wr_count + self.rda_count +
                self.wra_count + self.refab_count + self.refsb_count +
                self.rfmab_count + self.rfmsb_count + self.mrw_count +
                self.mrr_count + self.pdn_entry_count + self.pdn_exit_count +
                self.sref_entry_count + self.sref_exit_count)

    @property
    def total_energy_pj(self) -> float:
        """Total energy consumed by all commands"""
        return (self.total_act_energy_pj + self.total_pre_energy_pj +
                self.total_prea_energy_pj + self.total_rd_energy_pj +
                self.total_wr_energy_pj + self.total_rda_energy_pj +
                self.total_wra_energy_pj + self.total_refab_energy_pj +
                self.total_refsb_energy_pj + self.total_rfmab_energy_pj +
                self.total_rfmsb_energy_pj + self.total_mrw_energy_pj +
                self.total_mrr_energy_pj + self.total_pdn_entry_energy_pj +
                self.total_pdn_exit_energy_pj + self.total_sref_entry_energy_pj +
                self.total_sref_exit_energy_pj)

    def get_energy_breakdown(self) -> Dict[str, float]:
        """Get energy breakdown by command type"""
        return {
            "act": self.total_act_energy_pj,
            "pre": self.total_pre_energy_pj,
            "prea": self.total_prea_energy_pj,
            "rd": self.total_rd_energy_pj,
            "wr": self.total_wr_energy_pj,
            "rda": self.total_rda_energy_pj,
            "wra": self.total_wra_energy_pj,
            "refab": self.total_refab_energy_pj,
            "refsb": self.total_refsb_energy_pj,
            "rfmab": self.total_rfmab_energy_pj,
            "rfmsb": self.total_rfmsb_energy_pj,
            "mrw": self.total_mrw_energy_pj,
            "mrr": self.total_mrr_energy_pj,
            "pdn_entry": self.total_pdn_entry_energy_pj,
            "pdn_exit": self.total_pdn_exit_energy_pj,
            "sref_entry": self.total_sref_entry_energy_pj,
            "sref_exit": self.total_sref_exit_energy_pj,
        }

    def get_count_breakdown(self) -> Dict[str, int]:
        """Get command count breakdown"""
        return {
            "act": self.act_count,
            "pre": self.pre_count,
            "prea": self.prea_count,
            "rd": self.rd_count,
            "wr": self.wr_count,
            "rda": self.rda_count,
            "wra": self.wra_count,
            "refab": self.refab_count,
            "refsb": self.refsb_count,
            "rfmab": self.rfmab_count,
            "rfmsb": self.rfmsb_count,
            "mrw": self.mrw_count,
            "mrr": self.mrr_count,
            "pdn_entry": self.pdn_entry_count,
            "pdn_exit": self.pdn_exit_count,
            "sref_entry": self.sref_entry_count,
            "sref_exit": self.sref_exit_count,
        }


@dataclass
class ChannelPower:
    """Per-channel power tracking"""
    channel_id: int
    params: PowerParameters = field(default_factory=PowerParameters)

    # State tracking
    state: PowerState = PowerState.IDLE
    active_time_cycles: int = 0
    read_time_cycles: int = 0
    write_time_cycles: int = 0
    refresh_time_cycles: int = 0
    idle_time_cycles: int = 0
    self_refresh_cycles: int = 0

    # Energy counters (pJ)
    total_energy_pj: float = 0.0

    # Per-command energy tracking
    command_energy: CommandEnergy = field(default_factory=CommandEnergy)

    # Dynamic power tracking
    instantaneous_power_mw: float = 0.0
    power_history: List[float] = field(default_factory=list)

    def update_energy(self, cycles: int, state: PowerState):
        """Update energy consumption for cycles spent in state"""
        power_ma = self._get_power_for_state(state)
        power_mw = power_ma * self.params.vddq_voltage
        power_w = power_mw / 1000.0

        # Assuming 125ps cycle time (8 GT/s)
        time_s = cycles * 125e-12
        energy_j = power_w * time_s
        self.total_energy_pj += energy_j * 1e12

        # Update instantaneous power for history
        self.instantaneous_power_mw = power_mw * self.params.get_effective_power_scale()
        self.power_history.append(self.instantaneous_power_mw)

        # Keep history bounded
        if len(self.power_history) > 10000:
            self.power_history = self.power_history[-5000:]

        # Update state counters
        if state == PowerState.ACTIVE:
            self.active_time_cycles += cycles
        elif state == PowerState.READ:
            self.read_time_cycles += cycles
        elif state == PowerState.WRITE:
            self.write_time_cycles += cycles
        elif state == PowerState.REFRESH:
            self.refresh_time_cycles += cycles
        elif state == PowerState.IDLE:
            self.idle_time_cycles += cycles
        elif state == PowerState.SELF_REFRESH:
            self.self_refresh_cycles += cycles

    def record_command(self, cmd: CommandType, params: PowerParameters):
        """Record a command and its energy

        Args:
            cmd: Command type
            params: Power parameters for energy lookup
        """
        energy_pj = params.get_command_energy_pj(cmd)
        count_attr = f"{cmd.value}_count"
        energy_attr = f"total_{cmd.value}_energy_pj"

        if hasattr(self.command_energy, count_attr):
            setattr(self.command_energy, count_attr, getattr(self.command_energy, count_attr) + 1)
        if hasattr(self.command_energy, energy_attr):
            setattr(self.command_energy, energy_attr, getattr(self.command_energy, energy_attr) + energy_pj)

    def _get_power_for_state(self, state: PowerState) -> float:
        """Get current (mA) for a given state"""
        power_map = {
            PowerState.ACTIVE: self.params.active_power_ma,
            PowerState.READ: self.params.read_power_ma,
            PowerState.WRITE: self.params.write_power_ma,
            PowerState.REFRESH: self.params.refresh_power_ma,
            PowerState.IDLE: self.params.idle_power_ma,
            PowerState.SELF_REFRESH: self.params.self_refresh_power_ma,
            PowerState.POWER_DOWN: self.params.power_down_power_ma,
        }
        return power_map.get(state, self.params.idle_power_ma)

    def get_average_power_mw(self, total_cycles: int) -> float:
        """Calculate average power over total_cycles"""
        if total_cycles == 0:
            return 0.0
        energy_nj = self.total_energy_pj / 1e6  # Convert pJ to nJ
        time_s = total_cycles * 125e-12
        power_w = (energy_nj * 1e-9) / time_s
        return power_w * 1000.0  # Convert to mW

    def get_peak_power_mw(self) -> float:
        """Get peak power from history"""
        if not self.power_history:
            return 0.0
        return max(self.power_history)

    def get_power_stats(self) -> Dict[str, float]:
        """Get power statistics for this channel"""
        if not self.power_history:
            return {
                "average_mw": 0.0,
                "peak_mw": 0.0,
                "min_mw": 0.0,
                "rms_mw": 0.0,
            }

        avg = sum(self.power_history) / len(self.power_history)
        peak = max(self.power_history)
        min_val = min(self.power_history)

        # RMS calculation
        sum_sq = sum(p * p for p in self.power_history)
        rms = math.sqrt(sum_sq / len(self.power_history))

        return {
            "average_mw": avg,
            "peak_mw": peak,
            "min_mw": min_val,
            "rms_mw": rms,
        }


@dataclass
class PowerReport:
    """Generated power report"""
    timestamp: str = ""
    simulation_time_cycles: int = 0
    data_rate_gtps: float = 8.0

    # Power summary
    total_power_mw: float = 0.0
    average_power_mw: float = 0.0
    peak_power_mw: float = 0.0
    idle_power_mw: float = 0.0

    # Energy summary
    total_energy_pj: float = 0.0
    total_energy_mj: float = 0.0

    # Per-command statistics
    command_counts: Dict[str, int] = field(default_factory=dict)
    command_energies: Dict[str, float] = field(default_factory=dict)

    # Per-channel statistics
    channel_powers: List[Dict] = field(default_factory=list)

    # Efficiency metrics
    bandwidth_efficiency: float = 0.0
    power_efficiency: float = 0.0

    # Thermal estimates
    thermal: Dict[str, float] = field(default_factory=dict)

    # Configuration
    num_channels: int = 32
    voltage_vddq: float = 1.1
    process_corner: str = "TT"
    temperature_c: float = 45.0

    def to_text(self) -> str:
        """Generate formatted text report"""
        lines = []
        lines.append("=" * 70)
        lines.append("HBM4 POWER CONSUMPTION REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {self.timestamp}")
        lines.append(f"Simulation Cycles: {self.simulation_time_cycles:,}")
        lines.append(f"Data Rate: {self.data_rate_gtps} GT/s")
        lines.append("")

        # Power Summary
        lines.append("-" * 40)
        lines.append("POWER SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Total Power:          {self.total_power_mw:>10.2f} mW")
        lines.append(f"  Average Power:        {self.average_power_mw:>10.2f} mW")
        lines.append(f"  Peak Power:           {self.peak_power_mw:>10.2f} mW")
        lines.append(f"  Idle Power:           {self.idle_power_mw:>10.2f} mW")
        lines.append("")

        # Energy Summary
        lines.append("-" * 40)
        lines.append("ENERGY SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Total Energy:         {self.total_energy_pj:>10.2f} pJ")
        lines.append(f"                        {self.total_energy_mj:>10.6f} mJ")
        lines.append("")

        # Command Statistics
        lines.append("-" * 40)
        lines.append("COMMAND STATISTICS")
        lines.append("-" * 40)
        total_cmds = sum(self.command_counts.values())
        lines.append(f"  Total Commands:       {total_cmds:>10,}")
        lines.append("")
        lines.append("  Command Breakdown:")
        for cmd, count in sorted(self.command_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                energy = self.command_energies.get(cmd, 0)
                pct = (count / total_cmds * 100) if total_cmds > 0 else 0
                lines.append(f"    {cmd:12s}: {count:>8,} ({pct:5.1f}%)  {energy:>12.2f} pJ")
        lines.append("")

        # Per-Channel Summary
        lines.append("-" * 40)
        lines.append("CHANNEL POWER SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Number of Channels:   {self.num_channels}")
        if self.channel_powers:
            ch_avg = sum(c.get("average_mw", 0) for c in self.channel_powers) / len(self.channel_powers)
            ch_peak = max(c.get("peak_mw", 0) for c in self.channel_powers)
            lines.append(f"  Average per Channel:  {ch_avg:>10.2f} mW")
            lines.append(f"  Peak per Channel:    {ch_peak:>10.2f} mW")
        lines.append("")

        # Efficiency Metrics
        lines.append("-" * 40)
        lines.append("EFFICIENCY METRICS")
        lines.append("-" * 40)
        lines.append(f"  Bandwidth Efficiency: {self.bandwidth_efficiency:>10.2f}%")
        lines.append(f"  Power Efficiency:      {self.power_efficiency:>10.2f}%")
        lines.append("")

        # Thermal
        lines.append("-" * 40)
        lines.append("THERMAL ESTIMATES")
        lines.append("-" * 40)
        lines.append(f"  Junction Temperature:  {self.thermal.get('junction_temp_c', 0):>10.1f} C")
        lines.append(f"  Ambient Temperature:   {self.thermal.get('ambient_temp_c', 0):>10.1f} C")
        lines.append(f"  Thermal Resistance:    {self.thermal.get('theta_ja', 0):>10.2f} C/W")
        lines.append("")

        # Configuration
        lines.append("-" * 40)
        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"  VDDQ Voltage:          {self.voltage_vddq:>10.2f} V")
        lines.append(f"  Process Corner:        {self.process_corner:>10s}")
        lines.append(f"  Temperature:           {self.temperature_c:>10.1f} C")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert report to dictionary"""
        return {
            "timestamp": self.timestamp,
            "simulation": {
                "cycles": self.simulation_time_cycles,
                "data_rate_gtps": self.data_rate_gtps,
            },
            "power": {
                "total_mw": self.total_power_mw,
                "average_mw": self.average_power_mw,
                "peak_mw": self.peak_power_mw,
                "idle_mw": self.idle_power_mw,
            },
            "energy": {
                "total_pj": self.total_energy_pj,
                "total_mj": self.total_energy_mj,
            },
            "commands": {
                "counts": self.command_counts,
                "energies": self.command_energies,
            },
            "channels": self.channel_powers,
            "efficiency": {
                "bandwidth": self.bandwidth_efficiency,
                "power": self.power_efficiency,
            },
            "thermal": self.thermal,
            "configuration": {
                "num_channels": self.num_channels,
                "voltage_vddq": self.voltage_vddq,
                "process_corner": self.process_corner,
                "temperature_c": self.temperature_c,
            },
        }


@dataclass
class HBM4PowerEstimator:
    """HBM4 Power Consumption Estimator

    Tracks power consumption across all 32 channels with support for:
    - Per-channel power breakdown
    - Per-command energy tracking
    - State-based power calculation
    - Dynamic power with activity factors
    - Temperature/process corner scaling
    - Average and peak power estimation
    - Thermal modeling
    - Power report generation
    """
    num_channels: int = 32
    params: PowerParameters = field(default_factory=PowerParameters)
    data_rate_gtps: float = 8.0  # Data rate for tCK calculation

    # Per-channel tracking
    channels: List[ChannelPower] = field(default_factory=list)

    # Global tracking
    current_cycle: int = 0
    peak_power_mw: float = 0.0

    # Command tracking
    total_command_energy: CommandEnergy = field(default_factory=CommandEnergy)

    # Refresh tracking
    refresh_interval_cycles: int = 62400  # tREFI @ 8 GT/s (7.8 us / 125 ps)
    cycles_since_refresh: int = 0

    # Activity tracking
    active_cycles: int = 0
    read_cycles: int = 0
    write_cycles: int = 0

    def __post_init__(self):
        """Initialize channel power trackers"""
        if not self.channels:
            self.channels = [
                ChannelPower(channel_id=i, params=self.params)
                for i in range(self.num_channels)
            ]

    def _get_tCK_ps(self) -> float:
        """Get clock period in picoseconds"""
        return 1000.0 / self.data_rate_gtps

    def tick(self, cycles: int = 1):
        """Advance time and update power counters

        Args:
            cycles: Number of cycles to advance
        """
        self.current_cycle += cycles
        self.cycles_since_refresh += cycles

        # Check for refresh
        if self.cycles_since_refresh >= self.refresh_interval_cycles:
            self._perform_refresh()

    def _perform_refresh(self):
        """Execute refresh on all channels"""
        self.cycles_since_refresh = 0
        # Refresh takes nRFC cycles (~180 cycles)
        # For power estimation, we attribute refresh energy to affected channels
        for ch in self.channels:
            ch.update_energy(1, PowerState.REFRESH)

    def set_channel_state(self, channel_id: int, state: PowerState, cycles: int = 1):
        """Set channel state for power calculation

        Args:
            channel_id: Channel index (0-31)
            state: New power state
            cycles: Duration in cycles
        """
        if 0 <= channel_id < self.num_channels:
            ch = self.channels[channel_id]
            ch.update_energy(cycles, state)
            ch.state = state

            # Track peak power
            current_power = ch._get_power_for_state(state) * self.params.vddq_voltage
            current_power *= self.params.get_effective_power_scale()
            if current_power > self.peak_power_mw:
                self.peak_power_mw = current_power

            # Track activity
            if state == PowerState.ACTIVE:
                self.active_cycles += cycles
            elif state == PowerState.READ:
                self.read_cycles += cycles
            elif state == PowerState.WRITE:
                self.write_cycles += cycles

    def set_all_channels_state(self, state: PowerState, cycles: int = 1):
        """Set all channels to the same state

        Args:
            state: New power state
            cycles: Duration in cycles
        """
        for ch in self.channels:
            ch.update_energy(cycles, state)
            ch.state = state
        # Advance time
        self.tick(cycles)

    def record_command(self, channel_id: int, cmd: CommandType):
        """Record a command on a channel

        Args:
            channel_id: Channel index (0-31)
            cmd: Command type
        """
        if 0 <= channel_id < self.num_channels:
            ch = self.channels[channel_id]
            ch.record_command(cmd, self.params)

            # Also update global tracking
            self._increment_command_energy(cmd)

    def _increment_command_energy(self, cmd: CommandType):
        """Increment global command energy counter"""
        energy_pj = self.params.get_command_energy_pj(cmd)
        cmd_name = cmd.value

        # Update count
        count_attr = f"{cmd_name}_count"
        energy_attr = f"total_{cmd_name}_energy_pj"

        if hasattr(self.total_command_energy, count_attr):
            setattr(self.total_command_energy, count_attr,
                   getattr(self.total_command_energy, count_attr) + 1)
        if hasattr(self.total_command_energy, energy_attr):
            setattr(self.total_command_energy, energy_attr,
                   getattr(self.total_command_energy, energy_attr) + energy_pj)

    def get_total_power_mw(self) -> float:
        """Get total power across all channels"""
        scale = self.params.get_effective_power_scale()
        return sum(
            ch._get_power_for_state(ch.state) * self.params.vddq_voltage * scale
            for ch in self.channels
        )

    def get_average_power_mw(self) -> float:
        """Get average power over simulation time"""
        if self.current_cycle == 0:
            return 0.0
        total_energy_pj = sum(ch.total_energy_pj for ch in self.channels)
        time_s = self.current_cycle * (self._get_tCK_ps() * 1e-12)
        power_w = (total_energy_pj * 1e-12) / time_s
        return power_w * 1000.0

    def get_channel_power_mw(self, channel_id: int) -> float:
        """Get power for specific channel"""
        if 0 <= channel_id < self.num_channels:
            ch = self.channels[channel_id]
            scale = self.params.get_effective_power_scale()
            return ch._get_power_for_state(ch.state) * self.params.vddq_voltage * scale
        return 0.0

    def get_energy_breakdown_pj(self) -> Dict[str, float]:
        """Get energy breakdown by state type

        Returns:
            Dictionary with energy (pJ) per state type
        """
        tCK_ps = self._get_tCK_ps()
        breakdown = {
            "active": 0.0,
            "read": 0.0,
            "write": 0.0,
            "refresh": 0.0,
            "idle": 0.0,
            "self_refresh": 0.0,
        }

        for ch in self.channels:
            breakdown["active"] += ch.active_time_cycles * tCK_ps * 1e-12 * self.params.active_power_mw * 1e12
            breakdown["read"] += ch.read_time_cycles * tCK_ps * 1e-12 * self.params.read_power_mw * 1e12
            breakdown["write"] += ch.write_time_cycles * tCK_ps * 1e-12 * self.params.write_power_mw * 1e12
            breakdown["refresh"] += ch.refresh_time_cycles * tCK_ps * 1e-12 * self.params.refresh_power_mw * 1e12
            breakdown["idle"] += ch.idle_time_cycles * tCK_ps * 1e-12 * self.params.idle_power_mw * 1e12
            breakdown["self_refresh"] += ch.self_refresh_cycles * tCK_ps * 1e-12 * self.params.self_refresh_power_ma * self.params.vddq_voltage * 1e12

        return breakdown

    def get_command_energy_breakdown(self) -> Dict[str, float]:
        """Get energy breakdown by command type

        Returns:
            Dictionary with energy (pJ) per command type
        """
        return self.total_command_energy.get_energy_breakdown()

    def get_command_count_breakdown(self) -> Dict[str, int]:
        """Get command count breakdown

        Returns:
            Dictionary with command counts
        """
        return self.total_command_energy.get_count_breakdown()

    def get_bandwidth_efficiency(self, active_cycles: int, total_cycles: int) -> float:
        """Calculate bandwidth efficiency

        Args:
            active_cycles: Cycles spent in read/write
            total_cycles: Total simulation cycles

        Returns:
            Efficiency (0-1)
        """
        if total_cycles == 0:
            return 0.0
        return active_cycles / total_cycles

    def estimate_thermal(self, ambient_temp_c: float = 45.0) -> Dict[str, float]:
        """Estimate thermal characteristics

        Args:
            ambient_temp_c: Ambient temperature in Celsius

        Returns:
            Dictionary with thermal estimates
        """
        avg_power_w = self.get_average_power_mw() / 1000.0
        theta_ja = 0.5  # Thermal resistance (C/W) - package dependent

        # Junction temperature
        t_junction = ambient_temp_c + (avg_power_w * theta_ja)

        return {
            "ambient_temp_c": ambient_temp_c,
            "junction_temp_c": t_junction,
            "average_power_w": avg_power_w,
            "theta_ja": theta_ja,
            "peak_power_w": self.peak_power_mw / 1000.0,
        }

    def calculate_dynamic_power(
        self,
        activity_factor: float = 0.3,
        clock_frequency_mhz: float = 800.0,
    ) -> float:
        """Calculate dynamic power based on activity factor

        Args:
            activity_factor: Signal switching activity factor (0-1)
            clock_frequency_mhz: Operating frequency in MHz

        Returns:
            Dynamic power in mW
        """
        # Base dynamic power estimate
        base_power = self.get_total_power_mw()
        # Scale by activity factor
        return base_power * activity_factor

    def calculate_power_efficiency(
        self,
        achieved_bandwidth_gbs: float,
        peak_bandwidth_gbs: float,
    ) -> float:
        """Calculate power efficiency (bandwidth per watt)

        Args:
            achieved_bandwidth_gbs: Achieved bandwidth in GB/s
            peak_bandwidth_gbs: Peak bandwidth in GB/s

        Returns:
            Power efficiency (GB/s per Watt)
        """
        avg_power_w = self.get_average_power_mw() / 1000.0
        if avg_power_w <= 0:
            return 0.0
        return achieved_bandwidth_gbs / avg_power_w

    def generate_report(self) -> PowerReport:
        """Generate comprehensive power report

        Returns:
            PowerReport with all power statistics
        """
        report = PowerReport()
        report.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report.simulation_time_cycles = self.current_cycle
        report.data_rate_gtps = self.data_rate_gtps

        # Power summary
        report.total_power_mw = self.get_total_power_mw()
        report.average_power_mw = self.get_average_power_mw()
        report.peak_power_mw = self.peak_power_mw
        report.idle_power_mw = self.params.idle_power_mw * self.num_channels

        # Energy summary
        report.total_energy_pj = sum(ch.total_energy_pj for ch in self.channels)
        report.total_energy_mj = report.total_energy_pj * 1e-9

        # Command statistics
        report.command_counts = self.get_command_count_breakdown()
        report.command_energies = self.get_command_energy_breakdown()

        # Per-channel statistics
        report.channel_powers = []
        for ch in self.channels:
            stats = ch.get_power_stats()
            report.channel_powers.append({
                "channel_id": ch.channel_id,
                "average_mw": stats["average_mw"],
                "peak_mw": stats["peak_mw"],
                "rms_mw": stats["rms_mw"],
            })

        # Efficiency metrics
        active_total = self.active_cycles + self.read_cycles + self.write_cycles
        report.bandwidth_efficiency = self.get_bandwidth_efficiency(
            active_total, self.current_cycle
        ) * 100.0

        # Power efficiency: percentage of peak power utilized
        if self.peak_power_mw > 0:
            report.power_efficiency = (report.average_power_mw / self.peak_power_mw) * 100.0
        else:
            report.power_efficiency = 0.0

        # Thermal
        report.thermal = self.estimate_thermal(self.params.temperature_c)

        # Configuration
        report.num_channels = self.num_channels
        report.voltage_vddq = self.params.vddq_voltage
        report.process_corner = self.params.process_corner.value
        report.temperature_c = self.params.temperature_c

        return report

    def get_summary(self) -> Dict:
        """Get power estimation summary

        Returns:
            Dictionary with complete power statistics
        """
        total_cycles = self.current_cycle if self.current_cycle > 0 else 1
        energy_breakdown = self.get_energy_breakdown_pj()
        total_energy = sum(energy_breakdown.values())

        return {
            "num_channels": self.num_channels,
            "current_cycle": self.current_cycle,
            "total_power_mw": self.get_total_power_mw(),
            "average_power_mw": self.get_average_power_mw(),
            "peak_power_mw": self.peak_power_mw,
            "total_energy_pj": total_energy,
            "energy_breakdown_pj": energy_breakdown,
            "command_energy_pj": self.get_command_energy_breakdown(),
            "command_counts": self.get_command_count_breakdown(),
            "efficiency": {
                "active_ratio": sum(ch.active_time_cycles for ch in self.channels) / (total_cycles * self.num_channels),
                "read_ratio": sum(ch.read_time_cycles for ch in self.channels) / (total_cycles * self.num_channels),
                "write_ratio": sum(ch.write_time_cycles for ch in self.channels) / (total_cycles * self.num_channels),
                "idle_ratio": sum(ch.idle_time_cycles for ch in self.channels) / (total_cycles * self.num_channels),
            },
            "thermal": self.estimate_thermal(),
            "process_scaling": self.params.get_process_scaling_factor(),
            "temperature_scaling": self.params.get_temperature_scaling_factor(),
        }

    def reset(self):
        """Reset power counters"""
        for ch in self.channels:
            ch.active_time_cycles = 0
            ch.read_time_cycles = 0
            ch.write_time_cycles = 0
            ch.refresh_time_cycles = 0
            ch.idle_time_cycles = 0
            ch.self_refresh_cycles = 0
            ch.total_energy_pj = 0.0
            ch.state = PowerState.IDLE
            ch.command_energy = CommandEnergy()
            ch.power_history = []
        self.current_cycle = 0
        self.peak_power_mw = 0.0
        self.cycles_since_refresh = 0
        self.active_cycles = 0
        self.read_cycles = 0
        self.write_cycles = 0
        self.total_command_energy = CommandEnergy()

    def __repr__(self) -> str:
        return (f"HBM4PowerEstimator(channels={self.num_channels}, "
                f"avg_power={self.get_average_power_mw():.1f}mW, "
                f"peak_power={self.peak_power_mw:.1f}mW)")


# Extend PowerParameters with command energy methods
def _get_command_energy_pj(self, cmd: CommandType) -> float:
    """Get energy for a specific command type

    Args:
        cmd: Command type

    Returns:
        Energy in pJ
    """
    energy_map = {
        CommandType.ACT: self.act_energy_pj,
        CommandType.PRE: self.pre_energy_pj,
        CommandType.PREA: self.prea_energy_pj,
        CommandType.RD: self.rd_energy_pj,
        CommandType.WR: self.wr_energy_pj,
        CommandType.RDA: self.rda_energy_pj,
        CommandType.WRA: self.wra_energy_pj,
        CommandType.REFAB: self.refab_energy_pj,
        CommandType.REFSB: self.refsb_energy_pj,
        CommandType.RFMAB: self.rfmab_energy_pj,
        CommandType.RFMSB: self.rfmsb_energy_pj,
        CommandType.MRW: self.mrw_energy_pj,
        CommandType.MRR: self.mrr_energy_pj,
        CommandType.PDN_ENTER: self.pdn_entry_energy_pj,
        CommandType.PDN_EXIT: self.pdn_exit_energy_pj,
        CommandType.SREF_ENTER: self.sref_entry_energy_pj,
        CommandType.SREF_EXIT: self.sref_exit_energy_pj,
    }
    return energy_map.get(cmd, 0.0)


# Add method to PowerParameters class
PowerParameters.get_command_energy_pj = _get_command_energy_pj


# Default power estimator
DEFAULT_POWER_ESTIMATOR = HBM4PowerEstimator()

# Speed grade power presets
POWER_PRESETS = {
    "8Gbps": PowerParameters(),
    "12Gbps": PowerParameters(
        active_power_ma=420.0,
        read_power_ma=540.0,
        write_power_ma=500.0,
        vddq_voltage=1.15,
        # Energy scales with voltage
        act_energy_pj=384.0,
        rd_energy_pj=216.0,
        wr_energy_pj=234.0,
    ),
    "16Gbps": PowerParameters(
        active_power_ma=500.0,
        read_power_ma=650.0,
        write_power_ma=600.0,
        vddq_voltage=1.2,
        # Energy scales with voltage
        act_energy_pj=461.0,
        rd_energy_pj=259.0,
        wr_energy_pj=281.0,
    ),
}


# =============================================================================
# HBM3 Power Presets (JEDEC JESD238)
# =============================================================================
# HBM3 specifications at 6.4 Gbps (tCK = 156.25 ps)
# Based on JEDEC JESD238 HBM3 specification
# Active power: ~120mW/channel
# Read power: ~80mW/channel
# Write power: ~95mW/channel
# Refresh power: ~150mW/channel
# Idle power: ~25mW/channel

HBM3_POWER_PRESETS = {
    "hbm3_64": PowerParameters(  # 6.4 Gbps - HBM3 baseline
        # Active Power (per channel)
        active_power_ma=109.0,   # ~120mW at 1.1V
        read_power_ma=73.0,      # ~80mW at 1.1V
        write_power_ma=86.0,     # ~95mW at 1.1V

        # Idle/Standby Power (per channel)
        idle_power_ma=23.0,      # ~25mW at 1.1V
        standby_power_ma=10.0,   # CKE low standby

        # Refresh Power
        refresh_power_ma=136.0,  # ~150mW at 1.1V

        # Self-Refresh Power
        self_refresh_power_ma=5.0,  # Self-refresh mode

        # Power-Down Power
        power_down_power_ma=3.0,    # Power-down mode

        # Voltage Rails
        vddq_voltage=1.1,           # VDDQ voltage (V)
        vddq2_voltage=1.1,          # VDDQ2 voltage (V)
        vpp_voltage=2.5,            # VPP voltage (V)

        # Per-Command Energy (pJ)
        # Based on HBM3 timing parameters
        act_energy_pj=180.0,        # Activate energy (pJ)
        pre_energy_pj=50.0,         # Precharge energy (pJ)
        prea_energy_pj=85.0,        # Precharge all energy (pJ)
        rd_energy_pj=120.0,         # Read energy (pJ per burst)
        wr_energy_pj=135.0,         # Write energy (pJ per burst)
        rda_energy_pj=145.0,        # Read with auto-precharge
        wra_energy_pj=160.0,        # Write with auto-precharge
        refab_energy_pj=320.0,      # All-bank refresh (pJ)
        refsb_energy_pj=45.0,       # Per-bank refresh (pJ)
        rfmab_energy_pj=350.0,      # Row flash refresh all-bank
        rfmsb_energy_pj=55.0,       # Row flash refresh per-bank
        mrw_energy_pj=80.0,         # Mode register write
        mrr_energy_pj=70.0,         # Mode register read
        pdn_entry_energy_pj=30.0,   # Power-down entry
        pdn_exit_energy_pj=50.0,    # Power-down exit
        sref_entry_energy_pj=40.0,  # Self-refresh entry
        sref_exit_energy_pj=60.0,   # Self-refresh exit
    ),
    "hbm3_8g": PowerParameters(  # 8.0 Gbps - HBM3 high speed
        active_power_ma=127.0,
        read_power_ma=85.0,
        write_power_ma=100.0,
        idle_power_ma=27.0,
        standby_power_ma=12.0,
        refresh_power_ma=158.0,
        self_refresh_power_ma=6.0,
        power_down_power_ma=4.0,
        vddq_voltage=1.2,
        act_energy_pj=216.0,
        pre_energy_pj=60.0,
        rd_energy_pj=144.0,
        wr_energy_pj=162.0,
    ),
    "hbm3_96": PowerParameters(  # 9.6 Gbps - HBM3 extended rate
        active_power_ma=145.0,
        read_power_ma=97.0,
        write_power_ma=115.0,
        idle_power_ma=31.0,
        standby_power_ma=14.0,
        refresh_power_ma=180.0,
        self_refresh_power_ma=7.0,
        power_down_power_ma=5.0,
        vddq_voltage=1.25,
        act_energy_pj=252.0,
        pre_energy_pj=70.0,
        rd_energy_pj=168.0,
        wr_energy_pj=189.0,
    ),
}

# Combined power presets
# HBM3_POWER_PRESETS keys already have "hbm3_" prefix, so merge directly
ALL_POWER_PRESETS = {
    **POWER_PRESETS,
    **HBM3_POWER_PRESETS
}


def create_power_estimator(speed_grade: str = "8Gbps", num_channels: int = 32) -> HBM4PowerEstimator:
    """Create power estimator with speed grade parameters

    Args:
        speed_grade: One of "8Gbps", "12Gbps", "16Gbps", or HBM3 presets
        num_channels: Number of channels (default 32 for HBM4, 16 for HBM3)

    Returns:
        HBM4PowerEstimator configured for speed grade
    """
    params = ALL_POWER_PRESETS.get(speed_grade, ALL_POWER_PRESETS["8Gbps"])

    # Set data rate based on speed grade
    data_rates = {
        "8Gbps": 8.0,
        "12Gbps": 12.0,
        "16Gbps": 16.0,
        "hbm3_64": 6.4,
        "hbm3_8g": 8.0,
        "hbm3_96": 9.6,
    }
    data_rate = data_rates.get(speed_grade, 8.0)

    return HBM4PowerEstimator(
        num_channels=num_channels,
        params=params,
        data_rate_gtps=data_rate,
    )


def create_hbm3_power_estimator(
    speed_grade: str = "hbm3_64",
    num_channels: int = 16,
    process_corner: str = "TT",
    temperature_c: float = 45.0,
) -> HBM4PowerEstimator:
    """Create HBM3 power estimator

    Args:
        speed_grade: One of "hbm3_64", "hbm3_8g", "hbm3_96"
        num_channels: Number of channels (HBM3 has 16 channels per stack)
        process_corner: Process corner ("SS", "TT", "FF")
        temperature_c: Junction temperature in Celsius

    Returns:
        HBM4PowerEstimator configured for HBM3
    """
    params = HBM3_POWER_PRESETS.get(speed_grade, HBM3_POWER_PRESETS["hbm3_64"])

    # Update process corner
    corner_map = {
        "SS": ProcessCorner.SS,
        "TT": ProcessCorner.TT,
        "FF": ProcessCorner.FF,
    }
    params.process_corner = corner_map.get(process_corner, ProcessCorner.TT)
    params.temperature_c = temperature_c

    # Set data rate based on speed grade
    data_rates = {
        "hbm3_64": 6.4,
        "hbm3_8g": 8.0,
        "hbm3_96": 9.6,
    }
    data_rate = data_rates.get(speed_grade, 6.4)

    return HBM4PowerEstimator(
        num_channels=num_channels,
        params=params,
        data_rate_gtps=data_rate,
    )


def create_power_estimator_with_config(
    speed_grade: str = "8Gbps",
    num_channels: int = 32,
    process_corner: str = "TT",
    temperature_c: float = 45.0,
) -> HBM4PowerEstimator:
    """Create power estimator with custom configuration

    Args:
        speed_grade: One of "8Gbps", "12Gbps", "16Gbps", or HBM3 presets
        num_channels: Number of channels
        process_corner: Process corner ("SS", "TT", "FF")
        temperature_c: Junction temperature in Celsius

    Returns:
        HBM4PowerEstimator configured for specified parameters
    """
    # Check if it's an HBM3 preset
    if speed_grade in HBM3_POWER_PRESETS:
        return create_hbm3_power_estimator(
            speed_grade=speed_grade,
            num_channels=num_channels,
            process_corner=process_corner,
            temperature_c=temperature_c,
        )

    params = ALL_POWER_PRESETS.get(speed_grade, ALL_POWER_PRESETS["8Gbps"])

    # Update process corner
    corner_map = {
        "SS": ProcessCorner.SS,
        "TT": ProcessCorner.TT,
        "FF": ProcessCorner.FF,
    }
    params.process_corner = corner_map.get(process_corner, ProcessCorner.TT)
    params.temperature_c = temperature_c

    # Set data rate based on speed grade
    data_rates = {
        "8Gbps": 8.0,
        "12Gbps": 12.0,
        "16Gbps": 16.0,
        "hbm3_64": 6.4,
        "hbm3_8g": 8.0,
        "hbm3_96": 9.6,
    }
    data_rate = data_rates.get(speed_grade, 8.0)

    return HBM4PowerEstimator(
        num_channels=num_channels,
        params=params,
        data_rate_gtps=data_rate,
    )


def create_power_estimator_for_version(
    hbm_version: str = "hbm3",
    speed_grade: Optional[str] = None,
    process_corner: str = "TT",
    temperature_c: float = 45.0,
) -> HBM4PowerEstimator:
    """Create power estimator for specific HBM version

    Args:
        hbm_version: "hbm2", "hbm3", or "hbm4"
        speed_grade: Optional speed grade within version
        process_corner: Process corner ("SS", "TT", "FF")
        temperature_c: Junction temperature in Celsius

    Returns:
        Configured HBM4PowerEstimator
    """
    version = hbm_version.lower()

    if version == "hbm2":
        # HBM2 at 1.2 Gbps
        return HBM4PowerEstimator(
            num_channels=8,  # HBM2 has 8 channels per stack
            params=PowerParameters(
                active_power_ma=95.0,
                read_power_ma=65.0,
                write_power_ma=75.0,
                idle_power_ma=20.0,
                refresh_power_ma=120.0,
                self_refresh_power_ma=4.0,
                power_down_power_ma=2.0,
                vddq_voltage=1.2,
                act_energy_pj=150.0,
                rd_energy_pj=100.0,
                wr_energy_pj=115.0,
                process_corner=ProcessCorner[process_corner] if process_corner in ["SS", "TT", "FF"] else ProcessCorner.TT,
                temperature_c=temperature_c,
            ),
            data_rate_gtps=1.2,
        )

    elif version == "hbm3":
        if speed_grade is None:
            speed_grade = "hbm3_64"
        return create_hbm3_power_estimator(
            speed_grade=speed_grade,
            num_channels=16,  # HBM3 has 16 channels per stack
            process_corner=process_corner,
            temperature_c=temperature_c,
        )

    elif version == "hbm4":
        if speed_grade is None:
            speed_grade = "8Gbps"
        return create_power_estimator_with_config(
            speed_grade=speed_grade,
            num_channels=32,  # HBM4 has 32 channels
            process_corner=process_corner,
            temperature_c=temperature_c,
        )

    else:
        raise ValueError(f"Unknown HBM version: {hbm_version}")