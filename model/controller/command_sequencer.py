"""
HBM Command Sequencer

Generates DRAM command sequences from HBM requests.

Command Sequences:
- Row miss: ACT -> RD/WR -> PRE
- Row hit: RD/WR -> PRE (no ACT needed)

HBM4 Features:
- 32 channels with per-channel scheduling
- 2 pseudo-channels per channel
- 8 bank groups (3-bit bank group field)
- 16 banks per pseudo-channel (4-bit bank field)
- Per-bank-group timing (nRRDS/nRRDL, nWTRS/nWTRL)

Reference: Design document 2026-06-15-hbm-system-model-design.md Section 5.2
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from model.dram.hbm4_spec import HBM4Spec
from model.dram.bank_state_machine import BankStateEnum
from model.dram.timing import HBM3Timing
from model.controller.request import HBMRequest


class DRAMCommand(Enum):
    """DRAM Command Types

    HBM commands mapped to JEDEC standard.
    """
    ACT = "ACT"     # Activate (open row)
    PRE = "PRE"     # Precharge (close row)
    RD = "RD"       # Read
    WR = "WR"       # Write
    REF = "REF"     # Refresh
    PDE = "PDE"     # Power-down entry
    PDX = "PDX"     # Power-down exit
    SRX = "SRX"     # Self-refresh exit
    SRE = "SRE"     # Self-refresh entry


@dataclass
class CommandTiming:
    """Timing information for a single command"""
    command: DRAMCommand
    cycle: int              # Absolute cycle when command is issued
    relative_cycle: int      # Relative to sequence start
    bank_id: int
    row_id: int = 0
    col_id: int = 0
    data_cycles: int = 0    # Number of cycles data is on bus
    is_row_hit: bool = False

    def __repr__(self) -> str:
        hit_str = "HIT" if self.is_row_hit else "MISS"
        return (f"CmdTiming({self.command.value}@{self.cycle}, "
                f"bank={self.bank_id}, row=0x{self.row_id:x}, "
                f"{hit_str}, data_cycles={self.data_cycles})")


@dataclass
class CommandSequence:
    """DRAM Command Sequence

    Represents a complete command sequence for a memory request.
    Contains all commands with their timing information.

    Attributes:
        request: Associated HBM request
        commands: List of commands in sequence
        start_cycle: Cycle when sequence starts
        end_cycle: Cycle when sequence completes
        total_cycles: Total duration of sequence
        is_row_hit: Whether this is a row hit sequence
        total_data_cycles: Total cycles data is transferred
    """
    request: HBMRequest
    commands: List[CommandTiming] = field(default_factory=list)
    start_cycle: int = 0
    end_cycle: int = 0
    is_row_hit: bool = False

    def __post_init__(self):
        """Calculate derived fields after initialization"""
        if self.commands:
            self.start_cycle = self.commands[0].cycle
            self.end_cycle = self.commands[-1].cycle
        self._total_data_cycles: Optional[int] = None

    @property
    def total_cycles(self) -> int:
        """Total cycles from start to end"""
        return self.end_cycle - self.start_cycle

    @property
    def total_data_cycles(self) -> int:
        """Total cycles data is transferred on bus"""
        if self._total_data_cycles is None:
            self._total_data_cycles = sum(c.data_cycles for c in self.commands)
        return self._total_data_cycles

    @property
    def command_types(self) -> List[DRAMCommand]:
        """List of command types in order"""
        return [c.command for c in self.commands]

    @property
    def has_act(self) -> bool:
        """Check if sequence contains ACT command"""
        return DRAMCommand.ACT in self.command_types

    @property
    def has_pre(self) -> bool:
        """Check if sequence contains PRE command"""
        return DRAMCommand.PRE in self.command_types

    def get_command_at_cycle(self, cycle: int) -> Optional[CommandTiming]:
        """Get command at specific cycle"""
        for cmd in self.commands:
            if cmd.cycle == cycle:
                return cmd
        return None

    def get_command_count(self, command: DRAMCommand) -> int:
        """Count occurrences of specific command"""
        return sum(1 for c in self.commands if c.command == command)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "request_id": self.request.request_id,
            "is_read": self.request.is_read,
            "is_row_hit": self.is_row_hit,
            "start_cycle": self.start_cycle,
            "end_cycle": self.end_cycle,
            "total_cycles": self.total_cycles,
            "total_data_cycles": self.total_data_cycles,
            "commands": [
                {
                    "command": c.command.value,
                    "cycle": c.cycle,
                    "relative_cycle": c.relative_cycle,
                    "bank_id": c.bank_id,
                    "row_id": c.row_id,
                    "col_id": c.col_id,
                    "data_cycles": c.data_cycles,
                    "is_row_hit": c.is_row_hit,
                }
                for c in self.commands
            ]
        }

    def __repr__(self) -> str:
        cmd_str = " -> ".join(c.command.value for c in self.commands)
        hit_str = "HIT" if self.is_row_hit else "MISS"
        return (f"CmdSeq(req={self.request.request_id}, "
                f"{hit_str}, cycles={self.total_cycles}, "
                f"[{cmd_str}])")


@dataclass
class BankState:
    """Bank state for command generation

    Simple bank state structure used by command sequencer.
    This is a simplified version for command generation,
    not the full BankStateMachine from dram module.

    Attributes:
        bank_id: Bank identifier
        state: Current bank state
        open_row: Currently open row (-1 if closed)
        activate_time: Cycle when bank was last activated
        precharge_time: Cycle when bank was last precharged
    """
    bank_id: int
    state: BankStateEnum = BankStateEnum.IDLE
    open_row: int = -1
    activate_time: int = -1  # -1 means never activated
    precharge_time: int = -1

    @property
    def is_idle(self) -> bool:
        """Check if bank is idle"""
        return self.state == BankStateEnum.IDLE

    @property
    def is_active(self) -> bool:
        """Check if bank is active"""
        return self.state == BankStateEnum.ACTIVE

    @property
    def row_open(self) -> bool:
        """Check if a row is open"""
        return self.is_active and self.open_row >= 0

    def is_row_hit(self, row: int) -> bool:
        """Check if accessing same row that's open"""
        return self.row_open and self.open_row == row

    def is_row_conflict(self, row: int) -> bool:
        """Check if different row is open (row miss)"""
        return self.row_open and self.open_row != row


class CommandSequencer:
    """DRAM Command Sequencer

    Generates DRAM command sequences from HBM memory requests.
    Handles row hit vs row miss path selection.

    Command Sequences:
    - Row miss: ACT -> [tRCD] -> RD/WR -> [tCCD] -> PRE
    - Row hit:  RD/WR -> [tCCD] -> PRE (no ACT needed)

    HBM4 Turnaround Penalties:
    - RD_TO_WR: nRTW cycles (4 cycles)
    - WR_TO_RD: nWTRL cycles (5 cycles, different BG)

    Per-bank-group timing:
    - Same BG: nCCDS, nRRDS, nWTRS
    - Different BG: nCCDL, nRRDL, nWTRL

    Reference: JEDEC JESD270-4A HBM4 Specification
    """

    # Turnaround penalties (cycles) - defaults for same BG
    TURNAROUND_RD_TO_WR: int = 4   # nRTW
    TURNAROUND_WR_TO_RD: int = 4   # nWTRS (same BG)

    def __init__(self, spec: Optional[HBM4Spec] = None):
        """Initialize command sequencer

        Args:
            spec: HBM4 specification (uses default if None)
        """
        self.spec = spec or HBM4Spec()
        self.last_command: Optional[DRAMCommand] = None
        self.last_bank: Optional[int] = None
        self.last_bank_group: Optional[int] = None

    def check_row_hit(self, request: HBMRequest, bank_state: BankState) -> bool:
        """Check if request is a row hit

        Args:
            request: HBM memory request
            bank_state: Current state of target bank

        Returns:
            True if row hit, False if row miss
        """
        # Row hit only possible if bank is active and row matches
        if not bank_state.is_active:
            return False
        return bank_state.open_row == request.row_id

    def calculate_turnaround_penalty(self, new_command: DRAMCommand,
                                     bank_group_id: int = 0,
                                     channel_id: int = None) -> int:
        """Calculate turnaround penalty between commands

        Args:
            new_command: The new command to be issued
            bank_group_id: Bank group ID for the new command (for BG-aware timing)
            channel_id: Channel ID for cross-channel optimization (no penalty for different channels)

        Returns:
            Number of cycles penalty (0 if no penalty)
        """
        if self.last_command is None:
            return 0

        # No turnaround penalty for different channels (parallel access)
        if channel_id is not None and self.last_bank != channel_id:
            return 0

        # Only RD and WR commands have turnaround penalties
        if new_command not in [DRAMCommand.RD, DRAMCommand.WR]:
            return 0

        # Check if different bank group (longer penalty)
        same_bg = (self.last_bank_group == bank_group_id)

        if self.last_command == DRAMCommand.RD and new_command == DRAMCommand.WR:
            # Read to Write: use nRTW (same for all cases)
            return self.spec.nRTW
        elif self.last_command == DRAMCommand.WR and new_command == DRAMCommand.RD:
            # Write to Read: use nWTRS (same BG) or nWTRL (different BG)
            if same_bg:
                return self.spec.nWTRS
            else:
                return self.spec.nWTRL

        return 0

    def generate_row_miss_sequence(
        self,
        request: HBMRequest,
        bank_state: BankState,
        start_cycle: int
    ) -> CommandSequence:
        """Generate command sequence for row miss

        Sequence: PRE (if needed) -> ACT -> tRCD -> RD/WR -> PRE

        HBM4 timing parameters:
        - nRP: Precharge command period (8 cycles)
        - nRCDRD: RAS to CAS delay for read (8 cycles)
        - nCCDS/nCCDL: Column command delay

        Args:
            request: HBM memory request
            bank_state: Current state of target bank
            start_cycle: Cycle to start sequence

        Returns:
            CommandSequence with all commands and timing
        """
        commands: List[CommandTiming] = []
        current_cycle = start_cycle
        seq_start_cycle = start_cycle

        # Row miss: must precharge if row is open, then activate new row
        # Check if we need to precharge first
        need_precharge = bank_state.row_open or bank_state.is_active

        if need_precharge:
            # Precharge the currently open row
            commands.append(CommandTiming(
                command=DRAMCommand.PRE,
                cycle=current_cycle,
                relative_cycle=0,
                bank_id=request.bank_id,
                row_id=bank_state.open_row if bank_state.open_row >= 0 else 0,
                is_row_hit=False
            ))
            current_cycle += 1  # PRE command takes 1 cycle

            # Add tRP delay (nRP cycles)
            current_cycle += self.spec.nRP

        # ACT command to open target row
        commands.append(CommandTiming(
            command=DRAMCommand.ACT,
            cycle=current_cycle,
            relative_cycle=current_cycle - seq_start_cycle,
            bank_id=request.bank_id,
            row_id=request.row_id,
            is_row_hit=False
        ))
        current_cycle += 1

        # Add tRCD delay before RD/WR (HBM4Spec has nRCDRD/nRCDWR)
        if request.is_read:
            current_cycle += self.spec.nRCDRD
        else:
            current_cycle += self.spec.nRCDWR

        # RD or WR command
        rd_wr_command = DRAMCommand.RD if request.is_read else DRAMCommand.WR

        # Calculate turnaround penalty with bank group awareness and cross-channel optimization
        turnaround = self.calculate_turnaround_penalty(
            rd_wr_command, request.bank_group_id, request.channel_id
        )
        current_cycle += turnaround

        # Data cycles based on burst length
        # HBM4 burst length is 4 (FLINE), takes nBL cycles
        data_cycles = self.spec.nBL

        commands.append(CommandTiming(
            command=rd_wr_command,
            cycle=current_cycle,
            relative_cycle=current_cycle - seq_start_cycle,
            bank_id=request.bank_id,
            row_id=request.row_id,
            col_id=request.col_id,
            data_cycles=data_cycles,
            is_row_hit=False
        ))
        current_cycle += self.spec.nCCDS  # tCCD same bank group

        # PRE command to close row (required for HBM)
        # Only precharge after tRAS is satisfied
        precharge_cycle = current_cycle + max(0, self.spec.nRAS - self.spec.nRCDRD - self.spec.nCCDS)

        commands.append(CommandTiming(
            command=DRAMCommand.PRE,
            cycle=precharge_cycle,
            relative_cycle=precharge_cycle - seq_start_cycle,
            bank_id=request.bank_id,
            row_id=request.row_id,
            is_row_hit=False
        ))

        # Update last command tracking
        self.last_command = DRAMCommand.PRE
        self.last_bank = request.bank_id
        self.last_bank_group = request.bank_group_id

        # Create sequence
        sequence = CommandSequence(
            request=request,
            commands=commands,
            start_cycle=seq_start_cycle,
            is_row_hit=False
        )

        return sequence

    def generate_row_hit_sequence(
        self,
        request: HBMRequest,
        bank_state: BankState,
        start_cycle: int
    ) -> CommandSequence:
        """Generate command sequence for row hit

        Sequence: RD/WR -> PRE (no ACT needed)

        HBM4 timing parameters:
        - nCCDS: Column command delay, same bank group (2 cycles)
        - nBL: Burst length (4 cycles)

        Args:
            request: HBM memory request
            bank_state: Current state of target bank
            start_cycle: Cycle to start sequence

        Returns:
            CommandSequence with all commands and timing
        """
        commands: List[CommandTiming] = []
        seq_start_cycle = start_cycle
        current_cycle = start_cycle

        # RD or WR command (no ACT needed for row hit)
        rd_wr_command = DRAMCommand.RD if request.is_read else DRAMCommand.WR

        # Calculate turnaround penalty with bank group awareness and cross-channel optimization
        turnaround = self.calculate_turnaround_penalty(
            rd_wr_command, request.bank_group_id, request.channel_id
        )
        current_cycle += turnaround

        # Data cycles based on burst length (HBM4 nBL = 4)
        data_cycles = self.spec.nBL

        commands.append(CommandTiming(
            command=rd_wr_command,
            cycle=current_cycle,
            relative_cycle=0,
            bank_id=request.bank_id,
            row_id=request.row_id,
            col_id=request.col_id,
            data_cycles=data_cycles,
            is_row_hit=True
        ))
        current_cycle += self.spec.nCCDS

        # PRE command to close row
        # For row hit, tRAS is already satisfied since row was opened earlier
        commands.append(CommandTiming(
            command=DRAMCommand.PRE,
            cycle=current_cycle,
            relative_cycle=current_cycle - seq_start_cycle,
            bank_id=request.bank_id,
            row_id=request.row_id,
            is_row_hit=True
        ))

        # Update last command tracking
        self.last_command = DRAMCommand.PRE
        self.last_bank = request.bank_id
        self.last_bank_group = request.bank_group_id

        # Create sequence
        sequence = CommandSequence(
            request=request,
            commands=commands,
            start_cycle=seq_start_cycle,
            is_row_hit=True
        )

        return sequence

    def generate_command_sequence(
        self,
        request: HBMRequest,
        bank_state: BankState,
        start_cycle: int = 0
    ) -> CommandSequence:
        """Generate DRAM command sequence from HBM request

        Automatically selects row hit or row miss path based on bank state.

        Row Miss Path:
            PRE (if needed) -> ACT -> tRCD -> RD/WR -> PRE

        Row Hit Path:
            RD/WR -> PRE

        Args:
            request: HBM memory request
            bank_state: Current state of target bank
            start_cycle: Cycle to start sequence (default 0)

        Returns:
            CommandSequence with all commands and timing info
        """
        is_row_hit = self.check_row_hit(request, bank_state)

        if is_row_hit:
            return self.generate_row_hit_sequence(request, bank_state, start_cycle)
        else:
            return self.generate_row_miss_sequence(request, bank_state, start_cycle)

    def get_command_latency(
        self,
        request: HBMRequest,
        bank_state: BankState
    ) -> int:
        """Calculate total latency for a request

        Args:
            request: HBM memory request
            bank_state: Current state of target bank

        Returns:
            Total latency in cycles
        """
        sequence = self.generate_command_sequence(request, bank_state)
        return sequence.total_cycles


def generate_command_sequence(
    request: HBMRequest,
    bank_state: BankState,
    start_cycle: int = 0,
    timing: Optional[HBM3Timing] = None
) -> CommandSequence:
    """Generate DRAM command sequence from HBM request

    Convenience function that creates a CommandSequencer and generates
    the command sequence.

    Args:
        request: HBM memory request
        bank_state: Current state of target bank
        start_cycle: Cycle to start sequence (default 0)
        timing: HBM3 timing parameters (uses default if None)

    Returns:
        CommandSequence with all commands and timing info

    Example:
        >>> from model.controller.command_sequencer import generate_command_sequence, BankState
        >>> from model.controller.request import HBMRequest
        >>> from model.dram.bank_state_machine import BankStateEnum
        >>>
        >>> request = HBMRequest(addr=0x1000, length=32, is_read=True)
        >>> request.bank_id = 0
        >>> request.row_id = 0x100
        >>> request.col_id = 0
        >>>
        >>> bank_state = BankState(bank_id=0, state=BankStateEnum.IDLE)
        >>> sequence = generate_command_sequence(request, bank_state)
        >>> print(sequence)
        CmdSeq(req=1, MISS, cycles=83, [ACT -> RD -> PRE])
    """
    sequencer = CommandSequencer(timing)
    return sequencer.generate_command_sequence(request, bank_state, start_cycle)


def estimate_bandwidth_loss(
    row_hit_rate: float,
    spec: HBM4Spec
) -> float:
    """Estimate bandwidth loss due to row misses

    Args:
        row_hit_rate: Row buffer hit rate (0.0 - 1.0)
        spec: HBM4 specification parameters

    Returns:
        Bandwidth loss percentage
    """
    # Row miss adds: nRCDRD + nRP cycles
    # Row hit saves: ACT + nRCDRD + nRP cycles (but still needs PRE)
    # Net savings from row hit: nRCDRD cycles

    # Approximate overhead from row miss
    miss_penalty = spec.nRCDRD + spec.nRP
    hit_savings = spec.nRCDRD  # ACT saved

    # Bandwidth impact calculation
    if row_hit_rate >= 1.0:
        return 0.0

    # Simplified: each request that misses adds penalty cycles
    overhead_per_request = (1.0 - row_hit_rate) * (miss_penalty - hit_savings)

    # Normalize to percentage (assuming continuous traffic)
    # This is a rough estimate
    return overhead_per_request * 100 / (spec.nRC + miss_penalty)


# Export public API
__all__ = [
    "DRAMCommand",
    "CommandTiming",
    "CommandSequence",
    "BankState",
    "CommandSequencer",
    "generate_command_sequence",
    "estimate_bandwidth_loss",
]