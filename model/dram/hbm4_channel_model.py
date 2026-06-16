"""
HBM4 Channel Model - Enhanced Version

Implements 32 independent channels, each with 2 pseudo-channels and 8 bank groups.
Based on Ramulator 2.0 hierarchical node structure.

Key features:
- 32 independent memory channels
- 2 pseudo-channels per channel (64 total)
- 8 bank groups per pseudo-channel (2 banks per bank group)
- Independent bank state machines per pseudo-channel
- Bank group-aware command scheduling
- Command scheduling and timing
- Numeric command encoding for RTL interface

Command Encoding (aligned with RTL hbm_types.svh):
- 0: NOP    - No operation
- 1: ACT    - Activate command
- 2: READ   - Read command
- 3: WRITE  - Write command
- 4: PRE    - Precharge single bank
- 5: PREA   - Precharge all banks
- 6: REF    - Refresh (all banks)
- 7: RFM    - Row flash memory (refresh)

Reference:
- Ramulator 2.0: src/dram/impl/HBM3.cpp
- DRAMSys: configs/memspec/HBM2.json
- JEDEC JESD270-4A HBM4 specification
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import time

from model.dram.bank_state_machine import BankStateMachine, BankStateEnum
from model.dram.hbm4_spec import HBM4Spec, create_hbm4_spec_from_speed_grade, HBM4_SPEED_GRADES
from model.dram.timing import HBM4Timing, get_timing_for_speed_grade


# =============================================================================
# HBM4 Command Encoding (aligned with RTL hbm_types.svh)
# =============================================================================
class HBM4Command(IntEnum):
    """HBM4 command encoding for RTL interface

    Values must match RTL dram_cmd signal encoding.
    Numeric encoding (4 bits):
    - 0: NOP  - No operation
    - 1: ACT  - Activate command
    - 2: READ - Read command
    - 3: WRITE - Write command
    - 4: PRE  - Precharge single bank
    - 5: PREA - Precharge all banks
    - 6: REF  - Refresh (all banks)
    - 7: RFM  - Row flash memory (refresh)
    """
    NOP = 0
    ACT = 1
    READ = 2
    WRITE = 3
    PRE = 4
    PREA = 5
    REF = 6
    RFM = 7

    @classmethod
    def from_string(cls, cmd_str: str) -> 'HBM4Command':
        """Convert string command to numeric encoding"""
        mapping = {
            'ACT': cls.ACT,
            'PRE': cls.PRE,
            'PREA': cls.PREA,
            'RD': cls.READ,
            'RDA': cls.READ,
            'WR': cls.WRITE,
            'WRA': cls.WRITE,
            'REFab': cls.REF,
            'REFsb': cls.REF,
            'RFMab': cls.RFM,
            'RFMsb': cls.RFM,
        }
        return mapping.get(cmd_str, cls.NOP)

    @classmethod
    def to_string(cls, cmd: 'HBM4Command') -> str:
        """Convert numeric encoding to string command"""
        mapping = {
            cls.NOP: 'NOP',
            cls.ACT: 'ACT',
            cls.READ: 'RD',
            cls.WRITE: 'WR',
            cls.PRE: 'PRE',
            cls.PREA: 'PREA',
            cls.REF: 'REF',
            cls.RFM: 'RFM',
        }
        return mapping.get(cmd, 'NOP')


class HBM4ChannelState(IntEnum):
    """HBM4 Channel operational states"""
    IDLE = 0
    ACTIVE = 1
    REFRESHING = 2
    TRAINING = 3
    MAINTENANCE = 4


class PseudoChannelState(IntEnum):
    """Pseudo-channel operational states"""
    IDLE = 0
    ACTIVE = 1
    REFRESHING = 2
    READING = 3
    WRITING = 4


@dataclass
class BankGroup:
    """HBM4 Bank Group state

    Each bank group contains 2 banks in HBM4.
    Bank groups provide:
    - Reduced activation latency (tRRDS < tRRD)
    - Bank group interleaving for performance
    - Different turnaround times for same vs different BG

    Reference: JEDEC JESD270-4A HBM4 specification Section 5.4
    """
    group_id: int  # 0-7 for HBM4
    spec: HBM4Spec
    timing: HBM4Timing
    current_cycle: int = 0

    # Bank indices within this group (2 banks per group)
    bank_indices: List[int] = field(default_factory=list)

    # Timing tracking for bank group commands (in cycles)
    last_act_cycle: int = -1
    last_ref_cycle: int = -1

    @property
    def num_banks(self) -> int:
        """Number of banks in this group"""
        return len(self.bank_indices)

    def set_time(self, current_cycle: int) -> None:
        """Set current time for this bank group (in cycles)"""
        self.current_cycle = current_cycle

    def can_activate_bank_group(self, current_cycle: Optional[int] = None) -> bool:
        """Check if an activate can be issued to this bank group

        Timing constraints:
        - tRRDS: RAS-to-RAS delay (same BG) = 3 cycles
        - tRRDL: RAS-to-RAS delay (different BG) = 4 cycles

        Args:
            current_cycle: Current simulation cycle (uses self.current_cycle if None)
        """
        if self.last_act_cycle < 0:
            return True

        if current_cycle is None:
            current_cycle = self.current_cycle

        elapsed = current_cycle - self.last_act_cycle
        return elapsed >= self.timing.nRRDS

    def record_activation(self, current_cycle: Optional[int] = None):
        """Record that an activation was issued to this group

        Args:
            current_cycle: Current simulation cycle (uses self.current_cycle if None)
        """
        if current_cycle is None:
            current_cycle = self.current_cycle
        self.last_act_cycle = current_cycle

    def set_time(self, current_time: float):
        """Set current simulation time"""
        self.current_time = current_time


@dataclass
class PseudoChannel:
    """HBM4 Pseudo-Channel state

    Each physical channel has 2 pseudo-channels for doubled parallelism.
    Each pseudo-channel has:
    - 8 bank groups (each with 2 banks)
    - Independent bank state machines

    Bank group organization:
    - 8 bank groups (BG0-BG7)
    - 2 banks per bank group (per pseudo-channel)
    - Total: 8 BG × 2 banks = 16 banks per pseudo-channel

    Based on Ramulator 2.0 pseudochannel level.
    """
    channel_id: int
    pseudo_channel_id: int  # 0 or 1
    spec: HBM4Spec
    timing: HBM4Timing

    # Bank groups (8 groups, 2 banks each)
    bank_groups: List[BankGroup]

    # Flat bank list for compatibility (maps to bank_groups)
    banks: List[BankStateMachine]

    # State tracking
    state: PseudoChannelState = PseudoChannelState.IDLE
    open_row: int = -1
    current_time: float = 0.0

    # Bank group tracking for interleaving
    _last_act_bank_group: int = -1
    _bg_activation_count: int = 0

    def __init__(self, channel_id: int, pseudo_channel_id: int, spec: HBM4Spec, timing: Optional[HBM4Timing] = None):
        """Initialize pseudo-channel

        Args:
            channel_id: Channel this pseudo-channel belongs to
            pseudo_channel_id: Pseudo-channel index (0 or 1)
            spec: HBM4 specification
            timing: HBM4 timing parameters (uses default if None)
        """
        self.channel_id = channel_id
        self.pseudo_channel_id = pseudo_channel_id
        self.spec = spec
        self.timing = timing if timing is not None else HBM4Timing()

        # Calculate banks per bank group
        self.banks_per_group = spec.banks_per_pseudo_channel // spec.bank_groups_per_channel  # 2

        # Create bank groups
        self.bank_groups = [
            BankGroup(
                group_id=g,
                spec=spec,
                timing=self.timing,
                bank_indices=[g * self.banks_per_group + b for b in range(self.banks_per_group)]
            )
            for g in range(spec.bank_groups_per_channel)
        ]

        # Create flat bank list for compatibility
        self.banks = [
            BankStateMachine(bank_id, self.timing)
            for bank_id in range(spec.banks_per_pseudo_channel)
        ]

    def set_time(self, current_cycle: int) -> None:
        """Set current time for this pseudo-channel"""
        self.current_time = float(current_cycle)
        for bg in self.bank_groups:
            bg.set_time(current_cycle)
        for bank in self.banks:
            bank.set_time(current_cycle)

    def get_bank_group(self, bank_id: int) -> BankGroup:
        """Get the bank group for a given bank ID

        Args:
            bank_id: Bank index (0-15)

        Returns:
            BankGroup containing the bank
        """
        group_idx = bank_id // self.banks_per_group
        return self.bank_groups[group_idx]

    def get_bank_in_group(self, bank_group: int, index_in_group: int) -> BankStateMachine:
        """Get a specific bank within a bank group

        Args:
            bank_group: Bank group index (0-7)
            index_in_group: Index within bank group (0-1)

        Returns:
            BankStateMachine for the bank
        """
        bank_id = bank_group * self.banks_per_group + index_in_group
        return self.banks[bank_id]

    def activate_row(self, row: int, bank_group: Optional[int] = None,
                     bank_id: Optional[int] = None) -> bool:
        """Activate a row in this pseudo-channel

        Args:
            row: Row number to activate
            bank_group: Optional bank group to target (auto-select if None)
            bank_id: Optional specific bank index to activate (0-15)

        Returns:
            True if activation succeeded
        """
        if bank_id is not None:
            # Activate specific bank
            if bank_id < 0 or bank_id >= len(self.banks):
                return False
            bank = self.banks[bank_id]
            bank.set_time(self.current_time)
            if bank.can_activate():
                bank.activate(row)
                self.open_row = row
                self.state = PseudoChannelState.ACTIVE

                # Update bank group tracking
                inferred_bg = bank_id // self.banks_per_group
                self._last_act_bank_group = inferred_bg
                bg = self.bank_groups[inferred_bg]
                bg.record_activation()
                return True
            return False

        # Find an idle bank to activate
        for bank in self.banks:
            bank.set_time(self.current_time)
            if bank.can_activate():
                bank.activate(row)
                self.open_row = row
                self.state = PseudoChannelState.ACTIVE

                # Update bank group tracking
                if bank_group is not None:
                    self._last_act_bank_group = bank_group
                    bg = self.bank_groups[bank_group]
                    bg.record_activation()
                else:
                    # Infer bank group from bank index
                    inferred_bg = bank.bank.bank_id // self.banks_per_group
                    self._last_act_bank_group = inferred_bg
                    bg = self.bank_groups[inferred_bg]
                    bg.record_activation()

                return True

        # All banks busy
        return False

    def activate_row_in_bank_group(self, bank_group: int, index_in_group: int, row: int) -> bool:
        """Activate a row in a specific bank within a bank group

        Args:
            bank_group: Bank group index (0-7)
            index_in_group: Index within bank group (0-1)
            row: Row number to activate

        Returns:
            True if activation succeeded
        """
        bank = self.get_bank_in_group(bank_group, index_in_group)
        bank.set_time(self.current_time)

        if bank.can_activate():
            bank.activate(row)
            self.open_row = row
            self.state = PseudoChannelState.ACTIVE

            # Update bank group tracking
            bg = self.bank_groups[bank_group]
            bg.record_activation(self.current_time)
            self._last_act_bank_group = bank_group

            return True
        return False

    def is_row_open(self, row: int) -> bool:
        """Check if row is currently open in any bank

        Args:
            row: Row number to check

        Returns:
            True if row is open
        """
        return self.open_row == row

    def precharge_all(self) -> bool:
        """Precharge all banks in this pseudo-channel

        Returns:
            True if precharge succeeded
        """
        for bank in self.banks:
            bank.set_time(self.current_time)
            if bank.can_precharge():
                bank.precharge()

        self.open_row = -1
        self.state = PseudoChannelState.IDLE
        return True

    def precharge_bank(self, bank_id: int) -> bool:
        """Precharge a specific bank

        Args:
            bank_id: Bank index (0-15)

        Returns:
            True if precharge succeeded
        """
        if bank_id < 0 or bank_id >= len(self.banks):
            return False

        bank = self.banks[bank_id]
        bank.set_time(self.current_time)

        if bank.can_precharge():
            bank.precharge()
            # Check if all banks are now idle
            if all(b.bank.state == BankStateEnum.IDLE for b in self.banks):
                self.open_row = -1
                self.state = PseudoChannelState.IDLE
            return True
        return False

    def can_read(self) -> bool:
        """Check if a read can be issued

        Returns:
            True if any bank can accept a read
        """
        for bank in self.banks:
            bank.set_time(self.current_time)
            if bank.can_read():
                return True
        return False

    def can_write(self) -> bool:
        """Check if a write can be issued

        Returns:
            True if any bank can accept a write
        """
        for bank in self.banks:
            bank.set_time(self.current_time)
            if bank.can_write():
                return True
        return False

    def can_read_in_bank_group(self, bank_group: int) -> bool:
        """Check if a read can be issued in a specific bank group

        Args:
            bank_group: Bank group index (0-7)

        Returns:
            True if any bank in the group can accept a read
        """
        bg = self.bank_groups[bank_group]
        for bank_idx in bg.bank_indices:
            bank = self.banks[bank_idx]
            bank.set_time(self.current_time)
            if bank.can_read():
                return True
        return False

    def can_write_in_bank_group(self, bank_group: int) -> bool:
        """Check if a write can be issued in a specific bank group

        Args:
            bank_group: Bank group index (0-7)

        Returns:
            True if any bank in the group can accept a write
        """
        bg = self.bank_groups[bank_group]
        for bank_idx in bg.bank_indices:
            bank = self.banks[bank_idx]
            bank.set_time(self.current_time)
            if bank.can_write():
                return True
        return False

    def refresh(self, bank_id: Optional[int] = None) -> bool:
        """Execute refresh on banks

        Args:
            bank_id: Optional specific bank to refresh (None = all banks)

        Returns:
            True if refresh succeeded
        """
        if bank_id is not None:
            # Per-bank refresh (REFsb)
            if bank_id < 0 or bank_id >= len(self.banks):
                return False
            bank = self.banks[bank_id]
            if bank.bank.state == BankStateEnum.IDLE:
                bank.refresh()
                # Set pseudo-channel state to REFRESHING so tick() completes it
                self.state = PseudoChannelState.REFRESHING
                return True
            return False
        else:
            # All-bank refresh (REFab) - all banks must be IDLE
            all_idle = all(b.bank.state == BankStateEnum.IDLE for b in self.banks)
            if all_idle:
                for bank in self.banks:
                    bank.refresh()
                self.state = PseudoChannelState.REFRESHING
                return True
            return False

    def refresh_bank_group(self, bank_group: int) -> bool:
        """Execute refresh on a specific bank group

        Args:
            bank_group: Bank group index (0-7)

        Returns:
            True if refresh succeeded
        """
        bg = self.bank_groups[bank_group]
        # All banks in group must be IDLE
        all_idle = all(
            self.banks[idx].bank.state == BankStateEnum.IDLE
            for idx in bg.bank_indices
        )
        if all_idle:
            for idx in bg.bank_indices:
                self.banks[idx].refresh()
            self.state = PseudoChannelState.REFRESHING
            return True
        return False

    def get_bank_group_state(self, bank_group: int) -> Dict:
        """Get state information for a bank group

        Args:
            bank_group: Bank group index (0-7)

        Returns:
            Dictionary with bank group state information
        """
        bg = self.bank_groups[bank_group]
        return {
            'group_id': bg.group_id,
            'last_act_cycle': bg.last_act_cycle,
            'active_banks': sum(1 for idx in bg.bank_indices
                               if self.banks[idx].bank.state == BankStateEnum.ACTIVE)
        }

    def tick(self):
        """Advance time for this pseudo-channel"""
        self.current_time += 1.0

    def set_time(self, current_time: float):
        """Set current simulation time"""
        self.current_time = current_time


class HBM4Channel:
    """HBM4 Channel Model

    Represents one of 32 independent memory channels in HBM4.
    Each channel has 2 pseudo-channels (64 total pseudo-channels).
    Each pseudo-channel has 8 bank groups (2 banks per group).

    Reference: Ramulator 2.0 HBM3 channel node
    """

    # HBM4 commands (from JEDEC spec and Ramulator 2.0)
    COMMANDS = [
        'ACT', 'PRE', 'PREA',  # Row commands
        'RD', 'WR', 'RDA', 'WRA',  # Column commands (with auto-precharge)
        'REFab', 'REFsb',  # All-bank and per-bank refresh
        'RFMab', 'RFMsb'  # Row flash memory (refresh) commands
    ]

    # Supported speed grades
    SUPPORTED_SPEED_GRADES = list(HBM4_SPEED_GRADES.keys())

    @classmethod
    def create_with_speed_grade(cls, channel_id: int, speed_grade: str = "8Gbps",
                                timing: Optional[HBM4Timing] = None) -> "HBM4Channel":
        """Create an HBM4Channel with a specific speed grade

        Args:
            channel_id: Channel index (0-31)
            speed_grade: One of "8Gbps", "12Gbps", "16Gbps"
            timing: Optional HBM4Timing (uses default for speed grade if None)

        Returns:
            HBM4Channel configured for the specified speed grade
        """
        if speed_grade not in cls.SUPPORTED_SPEED_GRADES:
            raise ValueError(f"Unknown speed grade: {speed_grade}. "
                            f"Available: {cls.SUPPORTED_SPEED_GRADES}")

        spec = create_hbm4_spec_from_speed_grade(speed_grade)
        if timing is None:
            timing = get_timing_for_speed_grade(speed_grade)

        return cls(channel_id, spec, timing)

    def __init__(self, channel_id: int, spec: Optional[HBM4Spec] = None, timing: Optional[HBM4Timing] = None):
        """Initialize HBM4 channel

        Args:
            channel_id: Channel index (0-31)
            spec: HBM4 specification (uses default if None)
            timing: HBM4 timing parameters (uses default if None)
        """
        if spec is None:
            spec = HBM4Spec()
        if timing is None:
            timing = HBM4Timing()

        self.channel_id = channel_id
        self.spec = spec
        self.timing = timing
        self.current_cycle = 0

        # Create 2 pseudo-channels per channel
        self.pseudo_channels = [
            PseudoChannel(channel_id, pch_id, spec, timing)
            for pch_id in range(spec.pseudo_channels_per_channel)
        ]

        # Channel-level state
        self.state = HBM4ChannelState.IDLE

        # Bank group-aware command scheduling
        self._bg_scheduler = BankGroupScheduler(timing)

    def set_time(self, current_cycle: int) -> None:
        """Set current simulation cycle and propagate to pseudo-channels

        Args:
            current_cycle: Current simulation cycle
        """
        self.current_cycle = current_cycle
        for pc in self.pseudo_channels:
            pc.set_time(current_cycle)

    def issue_numeric_command(self, cmd: HBM4Command, pseudo_channel: int,
                             bank: int, row: int, col: int = 0) -> bool:
        """Issue a command using numeric encoding (RTL interface)

        Args:
            cmd: Numeric command (HBM4Command enum)
            pseudo_channel: Pseudo-channel index (0 or 1)
            bank: Bank index (0-15)
            row: Row index
            col: Column index

        Returns:
            True if command succeeded
        """
        return self.issue_command(HBM4Command.to_string(cmd), pseudo_channel, bank, row, col)

    @property
    def peak_bandwidth_gbs(self) -> float:
        """Peak bandwidth per channel in GB/s

        Each channel has 64-bit @ 8 GT/s = 64 GB/s
        Note: 8 GT/s × 64 bits / 8 = 64 GB/s per channel
        Total: 32 channels × 64 GB/s = 2048 GB/s
        """
        # Per-channel: data_rate × (io_width/32) / 8 = GB/s
        channel_width = self.spec.io_width // self.spec.channels
        return self.spec.data_rate_gtps * channel_width / 8

    @property
    def peak_bandwidth_tbs(self) -> float:
        """Peak bandwidth per channel in TB/s"""
        return self.peak_bandwidth_gbs / 1000

    @property
    def total_pseudo_channels(self) -> int:
        """Total pseudo-channels (32 channels × 2 pseudo-channels)"""
        return self.spec.pseudo_channels_per_channel * self.spec.channels

    @property
    def total_bank_groups(self) -> int:
        """Total bank groups per channel"""
        return self.spec.bank_groups_per_channel

    @property
    def banks_per_bank_group(self) -> int:
        """Banks per bank group"""
        return self.spec.banks_per_pseudo_channel // self.spec.bank_groups_per_channel

    def issue_command(self, cmd: str, pseudo_channel: int,
                     bank: int, row: int, col: int = 0) -> bool:
        """Issue a command to this channel

        Args:
            cmd: Command name ('ACT', 'PRE', 'RD', 'WR', etc.)
            pseudo_channel: Pseudo-channel index (0 or 1)
            bank: Bank index (0-15)
            row: Row index
            col: Column index

        Returns:
            True if command succeeded
        """
        if pseudo_channel not in [0, 1]:
            return False

        if bank < 0 or bank >= self.spec.banks_per_pseudo_channel:
            return False

        pc = self.pseudo_channels[pseudo_channel]

        if cmd == 'ACT':
            result = pc.activate_row(row, bank_id=bank)
            if result:
                self.state = HBM4ChannelState.ACTIVE
            return result

        elif cmd == 'PRE':
            # Precharge specific bank
            result = pc.precharge_bank(bank)
            if result:
                self.state = HBM4ChannelState.IDLE
            return result

        elif cmd == 'PREA':
            pc.precharge_all()
            self.state = HBM4ChannelState.IDLE
            return True

        elif cmd in ['RD', 'RDA']:
            # Check if row is open in the specified bank
            bank_obj = pc.banks[bank]
            if bank_obj.bank.state == BankStateEnum.IDLE:
                # Need to activate first
                if not pc.activate_row(row, bank_id=bank):
                    return False
            elif bank_obj.bank.open_row != row:
                # Different row open - need to precharge and activate
                pc.precharge_bank(bank)
                if not pc.activate_row(row, bank_id=bank):
                    return False
            pc.state = PseudoChannelState.READING
            return True

        elif cmd in ['WR', 'WRA']:
            # Check if row is open in the specified bank
            bank_obj = pc.banks[bank]
            if bank_obj.bank.state == BankStateEnum.IDLE:
                # Need to activate first
                if not pc.activate_row(row, bank_id=bank):
                    return False
            elif bank_obj.bank.open_row != row:
                # Different row open - need to precharge and activate
                pc.precharge_bank(bank)
                if not pc.activate_row(row, bank_id=bank):
                    return False
            pc.state = PseudoChannelState.WRITING
            return True

        elif cmd == 'REFab':
            # All-bank refresh - all banks must be idle
            result = pc.refresh()
            if result:
                self.state = HBM4ChannelState.REFRESHING
            return result

        elif cmd == 'REFsb':
            # Per-bank refresh (bank-specific refresh)
            result = pc.refresh(bank_id=bank)
            if result:
                self.state = HBM4ChannelState.REFRESHING
            return result

        elif cmd in ['RFMab', 'RFMsb']:
            # Row flash memory refresh - similar to REF but may have different timing
            result = pc.refresh() if cmd == 'RFMab' else pc.refresh(bank_id=bank)
            return result

        return False

    def execute_refresh(self, command: str, pseudo_channel: int = 0, bank: int = 0) -> bool:
        """Execute a refresh command from the refresh scheduler

        This is a convenience method that bridges the refresh scheduler output
        to the channel model.

        Args:
            command: 'REFab' (all-bank) or 'REFsb' (per-bank)
            pseudo_channel: Pseudo-channel index (0 or 1)
            bank: Bank index (0-15) for per-bank refresh

        Returns:
            True if refresh was executed successfully
        """
        if command == 'REFab':
            # All-bank refresh: execute on both pseudo-channels
            result0 = self.issue_command('REFab', pseudo_channel=0, bank=0, row=0)
            result1 = self.issue_command('REFab', pseudo_channel=1, bank=0, row=0)
            return result0 or result1
        elif command == 'REFsb':
            # Per-bank refresh on specified pseudo-channel and bank
            return self.issue_command('REFsb', pseudo_channel=pseudo_channel, bank=bank, row=0)
        return False

    def issue_command_with_bank_group(self, cmd: str, pseudo_channel: int,
                                       bank_group: int, bank_in_group: int,
                                       row: int, col: int = 0) -> bool:
        """Issue a command with explicit bank group targeting

        Args:
            cmd: Command name ('ACT', 'PRE', 'RD', 'WR', etc.)
            pseudo_channel: Pseudo-channel index (0 or 1)
            bank_group: Bank group index (0-7)
            bank_in_group: Bank index within group (0-1)
            row: Row index
            col: Column index

        Returns:
            True if command succeeded
        """
        if pseudo_channel not in [0, 1]:
            return False

        if bank_group < 0 or bank_group >= self.spec.bank_groups_per_channel:
            return False

        pc = self.pseudo_channels[pseudo_channel]
        bank_id = bank_group * self.banks_per_bank_group + bank_in_group

        if cmd == 'ACT':
            result = pc.activate_row_in_bank_group(bank_group, bank_in_group, row)
            if result:
                self.state = HBM4ChannelState.ACTIVE
            return result

        elif cmd in ['PRE', 'PREA']:
            pc.precharge_all()
            self.state = HBM4ChannelState.IDLE
            return True

        elif cmd in ['RD', 'RDA']:
            if not pc.is_row_open(row):
                pc.activate_row_in_bank_group(bank_group, bank_in_group, row)
            pc.state = PseudoChannelState.READING
            return True

        elif cmd in ['WR', 'WRA']:
            if not pc.is_row_open(row):
                pc.activate_row_in_bank_group(bank_group, bank_in_group, row)
            pc.state = PseudoChannelState.WRITING
            return True

        return False

    def tick(self):
        """Advance channel time by one cycle"""
        self.current_cycle += 1

        # Update all pseudo-channels
        for pc in self.pseudo_channels:
            pc.set_time(self.current_cycle)

            # Update bank state machines
            for bank in pc.banks:
                bank.set_time(self.current_cycle)

            # Update bank groups
            for bg in pc.bank_groups:
                bg.set_time(self.current_cycle)

            # Complete refresh operations - check per-bank
            if pc.state == PseudoChannelState.REFRESHING:
                # Track which banks were refreshing (before completing)
                banks_were_refreshing = [
                    bank for bank in pc.banks
                    if bank.bank.state == BankStateEnum.REFRESHING
                ]
                # Complete all refreshing banks
                for bank in banks_were_refreshing:
                    bank.complete_refresh()
                # Only transition to IDLE if no more refreshing banks remain
                if not any(b.bank.state == BankStateEnum.REFRESHING for b in pc.banks):
                    pc.state = PseudoChannelState.IDLE
                    self.state = HBM4ChannelState.IDLE

            # Complete read/write operations
            if pc.state in [PseudoChannelState.READING, PseudoChannelState.WRITING]:
                pc.state = PseudoChannelState.ACTIVE

    def reset(self):
        """Reset channel to initial state"""
        self.current_cycle = 0
        self.state = HBM4ChannelState.IDLE
        # Reset all pseudo-channels
        for pc in self.pseudo_channels:
            pc.state = PseudoChannelState.IDLE
            pc.current_cycle = 0
            # Reset all banks
            for bank in pc.banks:
                bank.bank.state = BankStateEnum.IDLE
                bank.bank.open_row = -1
                bank.bank.activate_time = 0
                bank.bank.precharge_time = 0
            # Reset bank groups
            for bg in pc.bank_groups:
                bg.last_act_cycle = -1

    def get_bank(self, pseudo_channel: int, bank: int) -> Optional[BankStateMachine]:
        """Get a specific bank state machine

        Args:
            pseudo_channel: Pseudo-channel index (0 or 1)
            bank: Bank index (0-15)

        Returns:
            BankStateMachine or None if invalid indices
        """
        if pseudo_channel not in [0, 1]:
            return None
        if bank < 0 or bank >= len(self.pseudo_channels[pseudo_channel].banks):
            return None

        return self.pseudo_channels[pseudo_channel].banks[bank]

    def get_bank_group(self, pseudo_channel: int, bank_group: int) -> Optional[BankGroup]:
        """Get a specific bank group

        Args:
            pseudo_channel: Pseudo-channel index (0 or 1)
            bank_group: Bank group index (0-7)

        Returns:
            BankGroup or None if invalid indices
        """
        if pseudo_channel not in [0, 1]:
            return None
        if bank_group < 0 or bank_group >= self.spec.bank_groups_per_channel:
            return None

        return self.pseudo_channels[pseudo_channel].bank_groups[bank_group]

    def is_row_hit(self, pseudo_channel: int, row: int) -> bool:
        """Check if row is currently open

        Args:
            pseudo_channel: Pseudo-channel index
            row: Row number

        Returns:
            True if row is open
        """
        if pseudo_channel not in [0, 1]:
            return False
        return self.pseudo_channels[pseudo_channel].is_row_open(row)

    def can_schedule_command(self, cmd: str, pseudo_channel: int, bank_group: int) -> bool:
        """Check if a command can be scheduled (bank group aware)

        Args:
            cmd: Command name
            pseudo_channel: Pseudo-channel index (0 or 1)
            bank_group: Bank group index (0-7)

        Returns:
            True if command can be issued respecting timing constraints
        """
        pc = self.pseudo_channels[pseudo_channel]

        if cmd == 'ACT':
            # Check bank group timing (tRRDS for same BG, tRRDL for different BG)
            last_bg = pc._last_act_bank_group
            if last_bg < 0:
                return True

            bg = pc.bank_groups[bank_group]
            if last_bg == bank_group:
                # Same BG: requires tRRDS
                elapsed = pc.current_time - bg.last_act_cycle
                return elapsed >= self.timing.nRRDS
            else:
                # Different BG: requires tRRDL
                elapsed = pc.current_time - bg.last_act_cycle
                return elapsed >= self.timing.nRRDL

        elif cmd in ['RD', 'WR']:
            # For RD/WR, just check if row is open
            return pc.can_read() if cmd.startswith('RD') else pc.can_write()

        return True

    def get_state_summary(self) -> dict:
        """Get channel state summary

        Returns:
            Dictionary with state information
        """
        return {
            'channel_id': self.channel_id,
            'state': self.state.name,
            'pseudo_channels': [
                {
                    'id': pc.pseudo_channel_id,
                    'state': pc.state.name,
                    'open_row': pc.open_row,
                    'active_banks': sum(1 for b in pc.banks if b.bank.state == BankStateEnum.ACTIVE),
                    'bank_groups': [
                        {
                            'id': bg.group_id,
                            'active_banks': sum(1 for idx in bg.bank_indices
                                               if pc.banks[idx].bank.state == BankStateEnum.ACTIVE)
                        }
                        for bg in pc.bank_groups
                    ]
                }
                for pc in self.pseudo_channels
            ],
            'current_cycle': self.current_cycle
        }


class BankGroupScheduler:
    """Bank group-aware command scheduler

    Manages command scheduling based on bank group organization.
    Implements:
    - tRRDS/tRRDL: RAS-to-RAS delay (same/different BG)
    - tCCDS/tCCDL: CAS-to-CAS delay (same/different BG)
    - tWTRS/tWTRL: Write-to-read turnaround (same/different BG)

    Note: All timing is tracked in cycles (integers), not seconds.

    Reference: JEDEC JESD270-4A HBM4 specification
    """

    def __init__(self, timing: HBM4Timing):
        self.timing = timing

        # Last command tracking per pseudo-channel (cycles)
        self._last_cmd: Dict[int, Tuple[str, int, int]] = {}  # pch_id -> (cmd, bg, cycles)
        self._last_col_cmd: Dict[int, Tuple[str, int, int]] = {}  # pch_id -> (cmd, bg, cycles)

        # FAW (four-bank activation window) tracking - per pseudo_channel (cycles)
        self._faw_window: Dict[int, List[int]] = {}  # pch_id -> [cycles of last 4 activations]

    def can_issue_act(self, pseudo_channel: int, bank_group: int, current_cycle: int) -> bool:
        """Check if ACT can be issued to a bank group

        Args:
            pseudo_channel: Pseudo-channel index
            bank_group: Bank group index
            current_cycle: Current simulation cycle

        Returns:
            True if ACT can be issued
        """
        # Check FAW window (per pseudo_channel)
        if pseudo_channel not in self._faw_window:
            self._faw_window[pseudo_channel] = []

        faw_window = self._faw_window[pseudo_channel]

        # Remove old entries from FAW window
        faw_window[:] = [t for t in faw_window if current_cycle - t < self.timing.nFAW]

        # FAW limits 4 activations within nFAW cycles
        if len(faw_window) >= 4:
            return False

        # Check bank group timing (per pseudo_channel)
        last_key = pseudo_channel
        if last_key in self._last_cmd:
            last_cmd, last_bg, last_cycle = self._last_cmd[last_key]
            if last_bg == bank_group:
                # Same BG: tRRDS
                elapsed = current_cycle - last_cycle
                if elapsed < self.timing.nRRDS:
                    return False
            else:
                # Different BG: tRRDL
                elapsed = current_cycle - last_cycle
                if elapsed < self.timing.nRRDL:
                    return False

        return True

    def record_act(self, pseudo_channel: int, bank_group: int, current_cycle: int):
        """Record an ACT command

        Args:
            pseudo_channel: Pseudo-channel index
            bank_group: Bank group index
            current_cycle: Current simulation cycle
        """
        self._last_cmd[pseudo_channel] = ('ACT', bank_group, current_cycle)

        if pseudo_channel not in self._faw_window:
            self._faw_window[pseudo_channel] = []
        self._faw_window[pseudo_channel].append(current_cycle)

    def can_issue_col(self, pseudo_channel: int, bank_group: int, current_cycle: int,
                     is_write: bool = False) -> bool:
        """Check if column command can be issued

        Args:
            pseudo_channel: Pseudo-channel index
            bank_group: Bank group index
            current_cycle: Current simulation cycle
            is_write: True if write, False if read

        Returns:
            True if column command can be issued
        """
        if pseudo_channel not in self._last_col_cmd:
            return True

        last_cmd, last_bg, last_cycle = self._last_col_cmd[pseudo_channel]
        elapsed = current_cycle - last_cycle

        # Check CAS-to-CAS delay (nCCDS/nCCDL)
        if last_bg == bank_group:
            # Same BG
            if is_write != (last_cmd == 'WR'):
                # Different direction: need RTW or WT
                if elapsed < self.timing.nRTW:
                    return False
            else:
                # Same BG and direction: nCCDS
                if elapsed < self.timing.nCCDS:
                    return False
        else:
            # Different BG
            if is_write != (last_cmd == 'WR'):
                # Different BG, different direction
                turnaround = self.timing.nWTRL if is_write else self.timing.nRTW
                if elapsed < turnaround:
                    return False
            else:
                # Different BG, same direction: nCCDL
                if elapsed < self.timing.nCCDL:
                    return False

        return True

    def record_col(self, pseudo_channel: int, bank_group: int, current_cycle: int,
                  is_write: bool = False):
        """Record a column command

        Args:
            pseudo_channel: Pseudo-channel index
            bank_group: Bank group index
            current_cycle: Current simulation cycle
            is_write: True if write, False if read
        """
        cmd = 'WR' if is_write else 'RD'
        self._last_col_cmd[pseudo_channel] = (cmd, bank_group, current_cycle)

    def reset(self):
        """Reset scheduler state"""
        self._last_cmd.clear()
        self._last_col_cmd.clear()
        self._faw_window.clear()


class HBM4ChannelArray:
    """Array of HBM4 channels for system-level simulation

    Manages all 32 HBM4 channels and provides system-level operations.
    """

    def __init__(self, spec: Optional[HBM4Spec] = None, timing: Optional[HBM4Timing] = None):
        """Initialize channel array

        Args:
            spec: HBM4 specification (uses default if None)
            timing: HBM4 timing parameters (uses default if None)
        """
        if spec is None:
            spec = HBM4Spec()
        if timing is None:
            timing = HBM4Timing()

        self.spec = spec
        self.timing = timing

        # Create all 32 channels
        self.channels = [
            HBM4Channel(i, spec, timing)
            for i in range(spec.channels)
        ]

        # System-level scheduler
        self.scheduler = BankGroupScheduler(timing)

    def get_channel(self, channel_id: int) -> Optional[HBM4Channel]:
        """Get a specific channel

        Args:
            channel_id: Channel index (0-31)

        Returns:
            HBM4Channel or None if invalid
        """
        if channel_id < 0 or channel_id >= self.spec.channels:
            return None
        return self.channels[channel_id]

    def get_pseudo_channel(self, channel_id: int, pch_id: int) -> Optional[PseudoChannel]:
        """Get a specific pseudo-channel

        Args:
            channel_id: Channel index (0-31)
            pch_id: Pseudo-channel index (0 or 1)

        Returns:
            PseudoChannel or None if invalid
        """
        ch = self.get_channel(channel_id)
        if ch is None or pch_id not in [0, 1]:
            return None
        return ch.pseudo_channels[pch_id]

    def tick(self):
        """Advance all channels by one cycle"""
        for ch in self.channels:
            ch.tick()

    @property
    def num_channels(self) -> int:
        """Number of channels in the array"""
        return len(self.channels)

    @property
    def total_bandwidth_gbs(self) -> float:
        """Total system bandwidth in GB/s"""
        return sum(ch.peak_bandwidth_gbs for ch in self.channels)

    @property
    def total_bandwidth_tbs(self) -> float:
        """Total system bandwidth in TB/s"""
        return self.total_bandwidth_gbs / 1000

    def get_system_state_summary(self) -> dict:
        """Get system-wide state summary"""
        return {
            'num_channels': len(self.channels),
            'total_pseudo_channels': len(self.channels) * self.spec.pseudo_channels_per_channel,
            'total_bank_groups': len(self.channels) * self.spec.pseudo_channels_per_channel * self.spec.bank_groups_per_channel,
            'total_banks': self.spec.total_banks,
            'peak_bandwidth_gbs': self.total_bandwidth_gbs,
            'peak_bandwidth_tbs': self.total_bandwidth_tbs,
            'channels': [ch.get_state_summary() for ch in self.channels]
        }