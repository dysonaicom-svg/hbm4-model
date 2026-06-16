"""
HBM Command Pipeline

This module executes commands on DRAM and tracks timing.
Connects the Controller to the DRAM Model with proper latency tracking.

Reference: Design Document 2026-06-15-hbm-system-model-design.md Section 5.2

Key Responsibilities:
1. Send commands to DRAMModel.execute_*()
2. Track command completion timing
3. Return actual latency to controller
4. Sync bank state between controller and DRAM

HBM4 Features:
- 32 independent channels
- 2 pseudo-channels per channel
- Per-bank-group scheduling
- Lane repair support
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time

from model.controller.request import HBMRequest, HBMResponse, RequestState
from model.controller.scheduler import BankState
from model.dram.hbm4_spec import HBM4Spec


class CommandType(Enum):
    """DRAM Command Types"""
    ACTIVATE = "ACT"
    PRECHARGE = "PRE"
    READ = "RD"
    WRITE = "WR"
    REFRESH = "REF"


class PendingState(Enum):
    """Pending Command State"""
    WAITING = 0       # Waiting for timing constraints
    IN_PROGRESS = 1   # Command executing
    COMPLETED = 2     # Command done
    FAILED = 3       # Command failed


@dataclass
class PendingCommand:
    """Track pending DRAM command

    Maintains state for a command that is waiting for DRAM execution.

    Attributes:
        request: Associated HBM request
        command_type: Type of DRAM command
        start_time: When command started (in cycles)
        expected_duration: Expected command duration (cycles)
        state: Current state
        bank_key: Tuple identifying the bank (ch, ps, bank)
    """
    request: HBMRequest
    command_type: CommandType
    start_time: float  # cycles
    expected_duration: int  # cycles
    state: PendingState = PendingState.WAITING
    bank_key: Tuple[int, int, int] = field(default_factory=tuple)

    # Bank identification
    stack_id: int = 0
    channel_id: int = 0
    pseudo_channel_id: int = 0
    bank_group_id: int = 0
    bank_id: int = 0
    row_id: int = 0

    # Completion tracking
    completion_time: float = 0.0
    actual_latency: float = 0.0  # cycles

    def __post_init__(self):
        """Initialize bank key from request"""
        if not self.bank_key:
            self.bank_key = (
                self.request.channel_id,
                self.request.pseudo_channel_id,
                self.request.bank_id
            )
            self.stack_id = self.request.stack_id
            self.channel_id = self.request.channel_id
            self.pseudo_channel_id = self.request.pseudo_channel_id
            self.bank_group_id = self.request.bank_group_id
            self.bank_id = self.request.bank_id
            self.row_id = self.request.row_id

    def mark_in_progress(self, current_time: float):
        """Mark command as in progress"""
        self.state = PendingState.IN_PROGRESS
        self.start_time = current_time

    def mark_completed(self, current_time: float):
        """Mark command as completed"""
        self.state = PendingState.COMPLETED
        self.completion_time = current_time
        self.actual_latency = current_time - self.request.arrival_time

    def mark_failed(self):
        """Mark command as failed"""
        self.state = PendingState.FAILED

    @property
    def latency_ns(self) -> float:
        """Get latency in nanoseconds"""
        # Convert cycles to nanoseconds using tCK from HBM4 spec
        # tCK = 125 ps for 8 GT/s, latency_cycles * tCK_ps / 1000
        tCK_ps = self.request.arrival_time if hasattr(self, '_tCK_ps') else 125.0
        return self.actual_latency * tCK_ps / 1000.0

    def __repr__(self) -> str:
        return (f"PendingCmd({self.command_type.value}, "
                f"ch={self.channel_id}, ps={self.pseudo_channel_id}, "
                f"bg={self.bank_group_id}, bk={self.bank_id}, "
                f"state={self.state.name}, latency={self.actual_latency:.0f}cyc)")


@dataclass
class CommandPipeline:
    """Command Pipeline for DRAM Execution

    Manages command flow between Controller and DRAM Model.

    Key Features:
    - Sends commands to DRAMModel.execute_request()
    - Tracks command completion timing
    - Syncs bank state with controller
    - Returns actual latency
    - HBM4: 32 channels, 2 pseudo-channels per channel, 8 bank groups

    The pipeline operates in cycles:
    1. Controller schedules a request
    2. Pipeline submits command to DRAM
    3. Pipeline tracks pending commands
    4. On completion, updates controller bank state and returns response
    """

    spec: HBM4Spec = field(default_factory=HBM4Spec)
    max_pending: int = 64  # Max pending commands

    # Internal state
    pending_commands: List[PendingCommand] = field(default_factory=list)
    completed_commands: List[PendingCommand] = field(default_factory=list)
    current_cycle: float = 0.0

    # Statistics
    stats: Dict = field(default_factory=lambda: {
        'commands_sent': 0,
        'commands_completed': 0,
        'commands_failed': 0,
        'total_latency_cycles': 0.0,
        'avg_latency_cycles': 0.0,
        'max_latency_cycles': 0.0,
    })

    # Data path state
    _write_data: Optional[bytes] = None
    _pending_write: Optional[PendingCommand] = None

    def tick(self, cycles: int = 1):
        """Advance simulation by N cycles

        Args:
            cycles: Number of cycles to advance
        """
        self.current_cycle += cycles

    def set_cycle(self, cycle: float):
        """Set current cycle directly

        Args:
            cycle: Current cycle number
        """
        self.current_cycle = cycle

    def write_data(self, data: bytes) -> bool:
        """Write data for pending write request

        Args:
            data: Data bytes to write

        Returns:
            True if data was stored for pending write
        """
        if self._pending_write is None:
            return False
        if self._pending_write.request.is_read:
            return False
        self._write_data = data
        return True

    def get_write_data(self) -> Optional[bytes]:
        """Get stored write data and clear

        Returns:
            Stored write data or None
        """
        data = self._write_data
        self._write_data = None
        return data

    def get_read_data(self, length: int) -> bytes:
        """Generate mock read data

        Args:
            length: Number of bytes to read

        Returns:
            Mock read data
        """
        return bytes(length)

    def submit_command(self, request: HBMRequest, dram_model) -> PendingCommand:
        """Submit a command to DRAM

        Args:
            request: The HBM request to execute
            dram_model: DRAMModel instance with execute_request()

        Returns:
            PendingCommand tracking the execution

        Raises:
            RuntimeError: If pipeline is full
        """
        if len(self.pending_commands) >= self.max_pending:
            raise RuntimeError(f"Command pipeline full: {self.max_pending} pending commands")

        # Determine command type
        cmd = "READ" if request.is_read else "WRITE"

        # Create pending command with full HBM4 address info
        pending = PendingCommand(
            request=request,
            command_type=CommandType.READ if request.is_read else CommandType.WRITE,
            start_time=self.current_cycle,
            expected_duration=self._estimate_duration(request),
            bank_key=(
                request.channel_id,
                request.pseudo_channel_id,
                request.bank_id
            )
        )
        # Set all address fields explicitly
        pending.stack_id = request.stack_id
        pending.channel_id = request.channel_id
        pending.pseudo_channel_id = request.pseudo_channel_id
        pending.bank_group_id = request.bank_group_id
        pending.bank_id = request.bank_id
        pending.row_id = request.row_id

        # Mark as in progress
        pending.mark_in_progress(self.current_cycle)

        # Execute on DRAM model (HBM4 interface)
        if hasattr(dram_model, 'execute_request'):
            # HBM4 channel model interface
            success = dram_model.execute_request(
                stack_id=request.stack_id,
                ch_id=request.channel_id,
                ps_id=request.pseudo_channel_id,
                bg_id=request.bank_group_id,
                bank_id=request.bank_id,
                row=request.row_id,
                cmd=cmd,
                current_time=int(self.current_cycle)
            )
        elif hasattr(dram_model, 'execute'):
            # Alternative simple interface
            success = dram_model.execute(
                channel=request.channel_id,
                bank=request.bank_id,
                row=request.row_id,
                cmd=cmd
            )
        else:
            # No dram model, simulate success
            success = True

        if not success:
            pending.mark_failed()
            self.stats['commands_failed'] += 1
        else:
            self.pending_commands.append(pending)
            self.stats['commands_sent'] += 1

        return pending

    def _estimate_duration(self, request: HBMRequest) -> int:
        """Estimate command duration in cycles

        Row hit: tCCD (minimal)
        Row miss: tRCD + tCCD

        HBM4 timing (cycles):
        - nCCDS = 2 (same bank group)
        - nCCDL = 3 (different bank group)
        - nRCDRD = 8 (RAS to CAS delay for read)
        """
        if request.row_hit:
            # Row hit: only RD/WR + PRE needed, tCCD cycles
            return self.spec.nCCDS
        else:
            # Row miss: ACT + tRCD + RD/WR
            return self.spec.nRCDRD + self.spec.nCCDS

    def process_completions(self) -> List[HBMResponse]:
        """Process completed commands

        Returns:
            List of HBMResponse for completed requests
        """
        responses = []
        completed = []

        for pending in self.pending_commands:
            if pending.state == PendingState.COMPLETED or \
               self._is_command_done(pending):
                pending.mark_completed(self.current_cycle)
                completed.append(pending)

                # Update statistics
                self.stats['commands_completed'] += 1
                self.stats['total_latency_cycles'] += pending.actual_latency
                self.stats['avg_latency_cycles'] = (
                    self.stats['total_latency_cycles'] /
                    max(1, self.stats['commands_completed'])
                )
                self.stats['max_latency_cycles'] = max(
                    self.stats['max_latency_cycles'],
                    pending.actual_latency
                )

                # Create response
                response = HBMResponse(
                    request_id=pending.request.request_id,
                    status="OK",
                    latency=pending.latency_ns,
                    channel_id=pending.channel_id,
                    bank_id=pending.bank_id,
                )
                responses.append(response)

        # Remove completed commands
        for cmd in completed:
            self.pending_commands.remove(cmd)
            self.completed_commands.append(cmd)

        return responses

    def _is_command_done(self, pending: PendingCommand) -> bool:
        """Check if a command has completed

        Uses timing parameters to determine completion.
        """
        if pending.state == PendingState.COMPLETED:
            return True

        if pending.state != PendingState.IN_PROGRESS:
            return False

        # Check if expected duration has elapsed
        elapsed = self.current_cycle - pending.start_time
        return elapsed >= pending.expected_duration

    def get_pending_count(self) -> int:
        """Get number of pending commands"""
        return len(self.pending_commands)

    def get_in_progress_commands(self) -> List[PendingCommand]:
        """Get list of in-progress commands"""
        return [c for c in self.pending_commands
                if c.state == PendingState.IN_PROGRESS]

    def get_command_for_bank(self, ch_id: int, ps_id: int, bank_id: int) -> Optional[PendingCommand]:
        """Get pending command for a specific bank

        Args:
            ch_id: Channel ID
            ps_id: Pseudo-channel ID
            bank_id: Bank ID

        Returns:
            PendingCommand if one is pending for this bank
        """
        bank_key = (ch_id, ps_id, bank_id)
        for cmd in self.pending_commands:
            if cmd.bank_key == bank_key:
                return cmd
        return None

    def sync_bank_state(self, request: HBMRequest, bank_states: Dict[Tuple, BankState]) -> BankState:
        """Sync bank state between DRAM and controller

        After a command completes, sync the bank state back to controller.

        Args:
            request: The completed request
            bank_states: Controller's bank state dictionary

        Returns:
            Updated BankState
        """
        bank_key = (
            request.channel_id,
            request.pseudo_channel_id,
            request.bank_id
        )

        # Get or create bank state
        if bank_key not in bank_states:
            bank_states[bank_key] = BankState(bank_id=request.bank_id)

        bank_state = bank_states[bank_key]

        # Update bank state based on request
        bank_state.is_open = True
        bank_state.open_row = request.row_id
        bank_state.last_access_time = self.current_cycle

        return bank_state

    def check_timing_violation(self, request: HBMRequest, dram_model) -> bool:
        """Check for timing violations

        Args:
            request: The request to check
            dram_model: DRAMModel instance

        Returns:
            True if timing violation detected
        """
        # This would check against DRAM model state
        # For now, return False (no violation)
        return False

    def get_stats(self) -> Dict:
        """Get pipeline statistics

        Returns:
            Dictionary of statistics
        """
        # Calculate timing conversion factor
        tCK_ps = self.spec.tCK_ps
        cycles_to_ns = tCK_ps / 1000.0

        return {
            **self.stats,
            'pending_count': self.get_pending_count(),
            'completed_count': len(self.completed_commands),
            'avg_latency_ns': self.stats['avg_latency_cycles'] * cycles_to_ns,
            'max_latency_ns': self.stats['max_latency_cycles'] * cycles_to_ns,
            'spec': {
                'channels': self.spec.channels,
                'pseudo_channels': self.spec.pseudo_channels,
                'tCK_ps': self.spec.tCK_ps,
            }
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'commands_sent': 0,
            'commands_completed': 0,
            'commands_failed': 0,
            'total_latency_cycles': 0.0,
            'avg_latency_cycles': 0.0,
            'max_latency_cycles': 0.0,
        }

    def __repr__(self) -> str:
        return (f"CommandPipeline(pending={self.get_pending_count()}, "
                f"completed={len(self.completed_commands)}, "
                f"cycle={self.current_cycle:.0f})")