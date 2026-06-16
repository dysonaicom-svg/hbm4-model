"""
HBM4 Independent Channel Timing Model

Implements per-channel independent timing for HBM4, where each channel
operates asynchronously from others (per JEDEC JESD270-4 requirement).

Key features:
- Independent clock domains per channel
- Local timing parameter management
- Per-channel bank state tracking
- Async channel operation support

Based on:
- JEDEC JESD270-4A HBM4 specification
- "Each channel is completely independent of one another"
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math


class TimingConstraint(Enum):
    """Timing constraint violation types"""
    tRC = "tRC"           # Row cycle time
    tRCD = "tRCD"         # RAS to CAS delay
    tRP = "tRP"           # Precharge command period
    tRAS = "tRAS"         # Row active time
    tCCD = "tCCD"         # Column command delay
    tRRD = "tRRD"         # RAS to RAS delay
    tWTR = "tWTR"         # Write to read turnaround
    tRTW = "tRTW"         # Read to write turnaround
    tFAW = "tFAW"         # Four-activate window
    tRFC = "tRFC"         # Refresh command duration


@dataclass
class BankState:
    """Per-bank state machine state"""
    bank_id: int
    row_id: Optional[int] = None      # Currently open row
    state: str = "IDLE"              # IDLE, ACTIVE, PRECHARGING
    last_act_cycle: int = -1
    last_pre_cycle: int = -1
    last_rd_cycle: int = -1
    last_wr_cycle: int = -1

    @property
    def is_open(self) -> bool:
        return self.state == "ACTIVE" and self.row_id is not None


@dataclass
class TimingParameters:
    """Per-channel timing parameters

    These parameters may vary per channel for:
    - Process corners (SS/TT/FF)
    - Voltage/temperature variations
    - Training results (VREF, delays)
    """
    # Core timing (cycles @ tCK)
    tCK_ps: float = 125.0            # Clock period (125ps = 8GHz)
    nCL: int = 8                      # CAS latency
    nCWL: int = 3                    # CAS write latency
    nBL: int = 4                     # Burst length

    # Row access
    nRCDRD: int = 8                  # RAS to CAS (read)
    nRCDWR: int = 8                  # RAS to CAS (write)
    nRP: int = 8                     # Precharge
    nRAS: int = 20                   # Row active time
    nRC: int = 22                    # Row cycle time

    # Column access
    nCCD: int = 2                    # Column command delay
    nCCDS: int = 2                   # CCD same bank group
    nCCDL: int = 3                   # CCD different bank group

    # Turnaround
    nWTRS: int = 4                    # Write to read (same)
    nWTRL: int = 5                    # Write to read (last)
    nRTW: int = 4                    # Read to write

    # Activation
    nRRDS: int = 3                   # RAS to RAS (same BG)
    nRRDL: int = 4                   # RAS to RAS (diff BG)
    nFAW: int = 16                   # Four-activate window

    # Refresh
    nRFC: int = 180                  # Refresh command duration
    nREFI: int = 3900                # Refresh interval

    @property
    def frequency_mhz(self) -> float:
        """Clock frequency in MHz

        Note: tCK_ps is in picoseconds
        Example: tCK_ps=125.0 -> 1000/125 = 8 MHz (8 GT/s)
        """
        return 1000.0 / self.tCK_ps


@dataclass
class ChannelClockDomain:
    """Independent clock domain for a channel

    Each channel can have:
    - Independent clock frequency (within spec tolerance)
    - Independent phase offset
    - Independent enable/disable
    """
    channel_id: int
    base_frequency_mhz: float = 8000.0  # 8 GHz default
    phase_offset_ps: float = 0.0
    enabled: bool = True

    @property
    def tCK_ps(self) -> float:
        """Clock period in picoseconds"""
        return 1000.0 / self.base_frequency_mhz

    @property
    def frequency_mhz(self) -> float:
        return self.base_frequency_mhz


class IndependentChannelTiming:
    """Independent Channel Timing Model

    Implements per-channel independent timing as required by JEDEC:
    "Each channel is completely independent of one another.
     Channels are not necessarily synchronous to each other."

    This model allows:
    - Independent clock domains per channel
    - Per-channel timing parameter variations
    - Asynchronous command execution
    - Local bank state management

    Usage:
        >>> timing = IndependentChannelTiming(channel_id=0)
        >>> ok, msg = timing.check_timing_constraints('RD', 0x1234)
        >>> timing.execute_with_independent_timing('RD', ...)
    """

    # Maximum activations in FAW window
    MAX_ACTIVATIONS_FAW = 4
    FAW_WINDOW_CYCLES = 4

    def __init__(
        self,
        channel_id: int,
        params: Optional[TimingParameters] = None,
        clock_domain: Optional[ChannelClockDomain] = None,
    ):
        """Initialize independent channel timing

        Args:
            channel_id: Channel index (0-31)
            params: Optional per-channel timing parameters
            clock_domain: Optional per-channel clock domain
        """
        self.channel_id = channel_id
        self.params = params or TimingParameters()
        self.clock_domain = clock_domain or ChannelClockDomain(
            channel_id=channel_id,
            base_frequency_mhz=self.params.frequency_mhz,
        )

        # Per-bank state machines
        self.bank_states: Dict[int, BankState] = {}
        for bank in range(16):  # 16 banks per channel
            self.bank_states[bank] = BankState(bank_id=bank)

        # Local cycle counter (independent from global)
        self.local_cycle = 0

        # FAW tracking
        self._recent_activations: List[int] = []

        # Statistics
        self._constraint_violations = 0
        self._commands_executed = 0

    @property
    def cycle(self) -> int:
        """Current local cycle"""
        return self.local_cycle

    def tick(self):
        """Advance local cycle by one"""
        self.local_cycle += 1

        # Clean up old FAW tracking
        if self._recent_activations:
            self._recent_activations = [
                c for c in self._recent_activations
                if self.local_cycle - c < self.FAW_WINDOW_CYCLES
            ]

    def set_timing_params(self, params: TimingParameters):
        """Update timing parameters

        Args:
            params: New timing parameters
        """
        self.params = params
        # Update clock domain frequency
        self.clock_domain.base_frequency_mhz = params.frequency_mhz

    def check_timing_constraints(
        self,
        command: str,
        bank: int,
        row: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Check if command satisfies timing constraints

        Args:
            command: Command type ('ACT', 'PRE', 'RD', 'WR', 'REF')
            bank: Target bank
            row: Target row (for ACT)

        Returns:
            Tuple of (constraint_met, violation_message)
        """
        if bank not in self.bank_states:
            return False, f"Invalid bank {bank}"

        bank_state = self.bank_states[bank]
        cycle = self.local_cycle

        if command == 'ACT':
            return self._check_act_constraints(bank_state, row, cycle)
        elif command == 'PRE':
            return self._check_pre_constraints(bank_state, cycle)
        elif command in ('RD', 'WR'):
            return self._check_col_constraints(bank_state, command, cycle)
        elif command == 'REF':
            return self._check_ref_constraints(cycle)

        return True, ""

    def _check_act_constraints(
        self,
        bank_state: BankState,
        row: int,
        cycle: int,
    ) -> Tuple[bool, str]:
        """Check ACTIVATE timing constraints"""
        # tRC: Row cycle time (same bank)
        if bank_state.last_act_cycle >= 0:
            elapsed = cycle - bank_state.last_act_cycle
            if elapsed < self.params.nRC:
                return False, f"tRC violation: {elapsed} < {self.params.nRC}"

        # tRAS: Row active time
        if bank_state.is_open and bank_state.last_act_cycle >= 0:
            elapsed = cycle - bank_state.last_act_cycle
            if elapsed < self.params.nRAS:
                return False, f"tRAS violation: {elapsed} < {self.params.nRAS}"

        # tRP: Precharge must complete before ACT
        if bank_state.last_pre_cycle >= 0:
            elapsed = cycle - bank_state.last_pre_cycle
            if elapsed < self.params.nRP:
                return False, f"tRP violation: {elapsed} < {self.params.nRP}"

        # tRRD: RAS to RAS delay
        for other_bank, other_state in self.bank_states.items():
            if other_bank != bank_state.bank_id and other_state.last_act_cycle >= 0:
                elapsed = cycle - other_state.last_act_cycle
                if elapsed < self.params.nRRDS:
                    return False, f"tRRD violation with bank {other_bank}"

        # tFAW: Four-activate window
        if len(self._recent_activations) >= self.MAX_ACTIVATIONS_FAW:
            return False, f"tFAW violation: {len(self._recent_activations)} activations in window"

        return True, ""

    def _check_pre_constraints(
        self,
        bank_state: BankState,
        cycle: int,
    ) -> Tuple[bool, str]:
        """Check PRECHARGE timing constraints"""
        # Bank must be open
        if not bank_state.is_open:
            return False, "Bank not open"

        # tRAS: Minimum row active time
        if bank_state.last_act_cycle >= 0:
            elapsed = cycle - bank_state.last_act_cycle
            if elapsed < self.params.nRAS:
                return False, f"tRAS violation: {elapsed} < {self.params.nRAS}"

        return True, ""

    def _check_col_constraints(
        self,
        bank_state: BankState,
        command: str,
        cycle: int,
    ) -> Tuple[bool, str]:
        """Check column access timing constraints"""
        # Bank must be open
        if not bank_state.is_open:
            return False, "Bank not open"

        # tRCD: RAS to CAS delay
        if bank_state.last_act_cycle >= 0:
            elapsed = cycle - bank_state.last_act_cycle
            min_rcd = (self.params.nRCDRD if command == 'RD'
                      else self.params.nRCDWR)
            if elapsed < min_rcd:
                return False, f"tRCD violation: {elapsed} < {min_rcd}"

        # tCCD: Column to column delay
        last_col_cycle = max(
            bank_state.last_rd_cycle if bank_state.last_rd_cycle >= 0 else 0,
            bank_state.last_wr_cycle if bank_state.last_wr_cycle >= 0 else 0
        )
        if last_col_cycle >= 0:
            elapsed = cycle - last_col_cycle
            if elapsed < self.params.nCCDS:
                return False, f"tCCD violation: {elapsed} < {self.params.nCCDS}"

        return True, ""

    def _check_ref_constraints(self, cycle: int) -> Tuple[bool, str]:
        """Check REFRESH timing constraints"""
        # tRFC: Refresh command duration (simplified - no other banks tracked)
        return True, ""

    def execute_with_independent_timing(
        self,
        command: str,
        bank: int,
        row: Optional[int] = None,
        data: Optional[int] = None,
    ) -> Tuple[bool, str, Any]:
        """Execute command with channel-local timing

        Args:
            command: Command type
            bank: Target bank
            row: Target row
            data: Optional data

        Returns:
            Tuple of (success, message, result_data)
        """
        # Check constraints
        ok, msg = self.check_timing_constraints(command, bank, row)
        if not ok:
            self._constraint_violations += 1
            return False, msg, None

        # Execute command
        bank_state = self.bank_states[bank]

        if command == 'ACT':
            bank_state.state = "ACTIVE"
            bank_state.row_id = row
            bank_state.last_act_cycle = self.local_cycle
            self._recent_activations.append(self.local_cycle)

        elif command == 'PRE':
            bank_state.state = "IDLE"
            bank_state.row_id = None
            bank_state.last_pre_cycle = self.local_cycle

        elif command == 'RD':
            bank_state.last_rd_cycle = self.local_cycle

        elif command == 'WR':
            bank_state.last_wr_cycle = self.local_cycle

        elif command == 'REF':
            # Refresh all banks
            for bs in self.bank_states.values():
                bs.state = "IDLE"
                bs.row_id = None

        self._commands_executed += 1
        return True, "", None

    def get_bank_state(self, bank: int) -> Optional[BankState]:
        """Get state of a specific bank

        Args:
            bank: Bank index

        Returns:
            BankState or None
        """
        return self.bank_states.get(bank)

    def get_timing_status(self) -> Dict[str, Any]:
        """Get current timing status

        Returns:
            Dictionary with timing status
        """
        return {
            'channel_id': self.channel_id,
            'local_cycle': self.local_cycle,
            'frequency_mhz': self.params.frequency_mhz,
            'open_banks': sum(1 for bs in self.bank_states.values() if bs.is_open),
            'recent_activations': len(self._recent_activations),
            'constraint_violations': self._constraint_violations,
            'commands_executed': self._commands_executed,
        }


class HBM4TimingManager:
    """Manager for all 32 independent channel timing domains

    Coordinates independent timing across all HBM4 channels.
    """

    def __init__(self, num_channels: int = 32):
        """Initialize timing manager

        Args:
            num_channels: Number of channels (default 32 for HBM4)
        """
        self.num_channels = num_channels
        self.channels: List[IndependentChannelTiming] = []

        for ch in range(num_channels):
            self.channels.append(IndependentChannelTiming(channel_id=ch))

    def tick(self):
        """Advance all channel timing domains"""
        for ch_timing in self.channels:
            ch_timing.tick()

    def get_channel_timing(self, channel_id: int) -> Optional[IndependentChannelTiming]:
        """Get timing model for a channel

        Args:
            channel_id: Channel index

        Returns:
            IndependentChannelTiming or None
        """
        if 0 <= channel_id < self.num_channels:
            return self.channels[channel_id]
        return None

    def set_channel_timing_params(
        self,
        channel_id: int,
        params: TimingParameters,
    ):
        """Set timing parameters for a specific channel

        Args:
            channel_id: Channel index
            params: Timing parameters
        """
        timing = self.get_channel_timing(channel_id)
        if timing:
            timing.set_timing_params(params)

    def get_all_timing_status(self) -> List[Dict[str, Any]]:
        """Get timing status for all channels

        Returns:
            List of timing status dictionaries
        """
        return [ch.get_timing_status() for ch in self.channels]