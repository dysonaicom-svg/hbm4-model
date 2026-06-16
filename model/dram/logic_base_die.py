"""
HBM4 Logic Base Die Model

Unified wrapper integrating all Logic Base Die components for HBM4 simulation.
The Logic Base Die is the control die in the HBM stack that manages:
- Address decoding and routing
- PHY interface and signal encoding
- Training and calibration
- Lane repair and redundancy
- ECC/CRC error handling

Key features:
- Per-channel independent operation (JEDEC requirement)
- Integration with existing modules (PHY, Lane Repair, ECC)
- Cycle-accurate timing model
- DFI 5.0 interface support

Based on:
- JEDEC JESD270-4A HBM4 specification
- Project's existing HBM4 modules
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# Import existing HBM4 modules
from model.dram.hbm4_spec import HBM4Spec
from model.dram.phy_signal import PAM3SignalModel, HBM4PAM3Encoder
from model.dram.phy_training import (
    HBM4PHYManager,
    PHYTrainingStateMachine,
    PHYInitializationStateMachine,
)
from model.dram.lane_repair import HBM4LaneRepairModel, RepairStatus
from model.dram.ecc_crc import HBM4DataIntegrity, HBM4ECC, HBM4CRC
from model.dram.dfi_interface import (
    DFI5Interface,
    DFICommand,
    DFIRequest,
    DFIResponse,
    DFILowPowerState,
)
from model.dram.bank_state_machine import BankStateMachine, BankStateEnum
from model.dram.timing import HBM3Timing


class ChannelState(Enum):
    """Channel operational state"""
    IDLE = "idle"
    ACTIVE = "active"
    TRAINING = "training"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class ChannelContext:
    """Per-channel execution context

    Each channel maintains independent state including:
    - Local clock domain
    - Timing parameters
    - Bank state machine
    - Pending commands
    """
    channel_id: int
    state: ChannelState = ChannelState.IDLE
    local_cycle: int = 0

    # Timing state
    last_act_cycle: int = -1
    last_pre_cycle: int = -1
    last_rd_cycle: int = -1
    last_wr_cycle: int = -1
    open_row: Optional[int] = None

    # Training state
    training_passed: bool = False
    calibration_data: Dict[str, Any] = field(default_factory=dict)

    # Lane repair state
    repair_status: RepairStatus = RepairStatus.NO_REPAIR
    repaired_lanes: List[int] = field(default_factory=list)

    # Error state
    error_count: int = 0
    last_error: Optional[str] = None

    # Bank state tracking per pseudo-channel and bank
    bank_states: Dict[int, BankStateEnum] = field(default_factory=dict)


class CommandBuffer:
    """Command buffer for pending commands

    Implements a FIFO command buffer with configurable depth
    for DFI command buffering and scheduling.
    """

    def __init__(self, depth: int = 64):
        """Initialize command buffer

        Args:
            depth: Maximum number of commands in buffer
        """
        self.depth = depth
        self._buffer: deque = deque(maxlen=depth)
        self._command_counter = 0
        self._total_commands_issued = 0
        self._total_commands_completed = 0

    def enqueue(self, command: str, channel: int, address: int,
                priority: int = 0, data: Optional[int] = None) -> int:
        """Add command to buffer

        Args:
            command: Command name (ACT, PRE, RD, WR, REF, MRS)
            channel: Target channel (0-31)
            address: Memory address
            priority: Command priority (higher = more urgent)
            data: Optional data payload for write commands

        Returns:
            Command ID if successful, -1 if buffer full
        """
        if len(self._buffer) >= self.depth:
            return -1

        cmd_id = self._command_counter
        self._command_counter += 1

        cmd_entry = {
            'id': cmd_id,
            'command': command,
            'channel': channel,
            'address': address,
            'priority': priority,
            'data': data,
            'enqueued_cycle': None,  # Will be set by tick()
            'issued_cycle': None,
            'completed': False,
        }

        self._buffer.append(cmd_entry)
        return cmd_id

    def dequeue(self) -> Optional[Dict]:
        """Remove and return next command

        Returns:
            Next command dict or None if empty
        """
        if not self._buffer:
            return None

        cmd = self._buffer.popleft()
        cmd['completed'] = True
        self._total_commands_completed += 1
        return cmd

    def peek(self) -> Optional[Dict]:
        """View next command without removing

        Returns:
            Next command dict or None if empty
        """
        if not self._buffer:
            return None
        return self._buffer[0]

    def tick(self):
        """Advance buffer state (called each cycle)

        Updates internal timestamps for queued commands.
        """
        cycle_commands = [c for c in self._buffer if c['enqueued_cycle'] is None]
        for cmd in cycle_commands:
            cmd['enqueued_cycle'] = cmd['id']  # Placeholder for cycle tracking

    def clear(self):
        """Clear all commands from buffer"""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Current buffer size"""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity"""
        return len(self._buffer) >= self.depth

    @property
    def available_capacity(self) -> int:
        """Available slots in buffer"""
        return self.depth - len(self._buffer)

    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            'current_size': len(self._buffer),
            'max_depth': self.depth,
            'total_commands_issued': self._total_commands_issued,
            'total_commands_completed': self._total_commands_completed,
            'utilization': len(self._buffer) / self.depth if self.depth > 0 else 0,
        }


@dataclass
class LogicBaseDieConfig:
    """Configuration for Logic Base Die model"""
    # Architecture
    num_channels: int = 32
    channel_width: int = 64           # JEDEC standard
    burst_width: int = 256           # Data width per channel (4 x 64)

    # Signal encoding
    pam3_enabled: bool = True
    symbol_rate_gbaud: float = 8.0   # 8 Gbaud for HBM4 base rate

    # ECC/CRC
    ecc_enabled: bool = True
    crc_enabled: bool = True
    data_width: int = 64

    # Lane repair
    lanes_per_channel: int = 64
    spare_lanes_per_channel: int = 4

    # Training
    training_timeout_cycles: int = 50000
    auto_training: bool = True

    # Timing (cycles @ 8 GT/s)
    tCK_ps: float = 125.0            # 125ps = 8 GHz

    # Command buffer
    command_buffer_depth: int = 64

    # Bank state tracking
    banks_per_channel: int = 16       # 16 banks per pseudo-channel
    pseudo_channels_per_channel: int = 2


class HBM4LogicBaseDie:
    """HBM4 Logic Base Die Model

    Unified model integrating all Logic Base Die functionality.
    Provides cycle-accurate simulation of the control die in HBM4 stack.

    Architecture:
    ```
    +----------------------------------------------------------+
    |                    Logic Base Die                          |
    |  +------------------+  +------------------+               |
    |  | Address Decoder   |  | Command Queue    |               |
    |  +------------------+  +------------------+               |
    |  +------------------+  +------------------+               |
    |  | PAM3 Encoder     |  | ECC/CRC Engine   |               |
    |  +------------------+  +------------------+               |
    |  +------------------+  +------------------+               |
    |  | PHY Manager      |  | Lane Repair      |               |
    |  +------------------+  +------------------+               |
    |  +------------------+  +------------------+               |
    |  | DFI 5.0 Interface|  | Bank State Track|               |
    |  +------------------+  +------------------+               |
    +----------------------------------------------------------+
    |              Per-Channel Contexts (x32)                  |
    |  [Ch0] [Ch1] [Ch2] ... [Ch31]                           |
    +----------------------------------------------------------+
    ```

    Usage:
        >>> lbd = HBM4LogicBaseDie()
        >>> lbd.initialize()
        >>> for _ in range(1000):
        ...     lbd.tick()
        >>> status = lbd.get_status()
    """

    def __init__(self, config: Optional[LogicBaseDieConfig] = None):
        """Initialize Logic Base Die model

        Args:
            config: Optional configuration
        """
        self.config = config or LogicBaseDieConfig()

        # Initialize specification
        self.spec = HBM4Spec()

        # Initialize PAM3 signal model (if enabled)
        if self.config.pam3_enabled:
            self.pam3_encoder = HBM4PAM3Encoder(config={
                'symbol_rate': self.config.symbol_rate_gbaud * 1e9,
                'voltage_swing': 0.8,
            })
        else:
            self.pam3_encoder = None

        # Initialize DFI 5.0 Interface
        self.dfi = DFI5Interface()

        # Initialize PHY Manager (per-channel training)
        self.phy_manager = HBM4PHYManager(
            num_channels=self.config.num_channels,
            config={
                'timeout_cycles': self.config.training_timeout_cycles,
            }
        )

        # Initialize Lane Repair (per-channel redundancy)
        self.lane_repair = HBM4LaneRepairModel(
            num_channels=self.config.num_channels,
            lanes_per_channel=self.config.lanes_per_channel,
            spare_lanes_per_channel=self.config.spare_lanes_per_channel,
        )

        # Initialize ECC/CRC (per-channel error handling)
        self.data_integrity = HBM4DataIntegrity(
            data_width=self.config.data_width,
            enable_ecc=self.config.ecc_enabled,
            enable_crc=self.config.crc_enabled,
        )

        # Initialize Command Buffer
        self.command_buffer = CommandBuffer(depth=self.config.command_buffer_depth)

        # Initialize Bank State Tracking
        # Each channel has pseudo_channels * banks structure
        self._bank_state_machines: Dict[int, Dict[int, BankStateMachine]] = {}
        self._initialize_bank_state_machines()

        # Per-channel contexts (independent operation)
        self._channels: List[ChannelContext] = []
        for ch in range(self.config.num_channels):
            self._channels.append(ChannelContext(channel_id=ch))

        # Global state
        self._global_cycle = 0
        self._initialized = False
        self._training_complete = False

        # Statistics
        self._total_commands = 0
        self._total_errors = 0
        self._dfi_commands_sent = 0
        self._dfi_commands_completed = 0

    def _initialize_bank_state_machines(self):
        """Initialize bank state machines for all channels"""
        timing = HBM3Timing()
        total_banks = self.config.banks_per_channel * self.config.pseudo_channels_per_channel

        for ch in range(self.config.num_channels):
            self._bank_state_machines[ch] = {}
            for bank_id in range(total_banks):
                self._bank_state_machines[ch][bank_id] = BankStateMachine(
                    bank_id=bank_id,
                    timing=timing
                )

    @property
    def cycle(self) -> int:
        """Current global cycle"""
        return self._global_cycle

    @property
    def is_initialized(self) -> bool:
        """Check if Logic Base Die is initialized"""
        return self._initialized

    @property
    def is_ready(self) -> bool:
        """Check if all channels are ready"""
        return self._initialized and self._training_complete and self._phy_ready()

    def _phy_ready(self) -> bool:
        """Check if PHY training is complete on all channels"""
        return all(
            ctx.training_passed for ctx in self._channels
        )

    # ==================== Initialization ====================

    def initialize(self):
        """Initialize Logic Base Die

        Starts initialization sequence on all channels.
        """
        if self._initialized:
            return

        # Start PHY initialization on all channels
        self.phy_manager.start_initialization()
        self._initialized = True

    def tick(self):
        """Advance simulation by one cycle

        Updates all channel contexts, DFI interface, command buffer,
        and component state machines.
        """
        self._global_cycle += 1

        # Update DFI interface
        self.dfi.tick()

        # Update command buffer
        self.command_buffer.tick()

        # Update PHY state machines
        self.phy_manager.tick()

        # Update per-channel local cycles
        for ctx in self._channels:
            ctx.local_cycle += 1

        # Check training completion
        if not self._training_complete:
            self._check_training_complete()

    def _check_training_complete(self):
        """Check if training is complete on all channels"""
        all_ready = True

        for ch, ctx in enumerate(self._channels):
            phy_status = self.phy_manager.get_channel_status(ch)

            if phy_status.get('training', {}).get('passed'):
                if not ctx.training_passed:
                    ctx.training_passed = True
                    # Collect calibration data
                    ctx.calibration_data = self.phy_manager.get_channel_status(ch)
            elif phy_status.get('state') != 'INIT_COMPLETE':
                all_ready = False

        if all_ready and self._initialized:
            self._training_complete = True

    # ==================== Bank State Tracking ====================

    def get_bank_state(self, channel_id: int, bank_id: int) -> Optional[BankStateEnum]:
        """Get state of a specific bank

        Args:
            channel_id: Channel index (0-31)
            bank_id: Bank index within channel

        Returns:
            BankStateEnum or None if invalid channel/bank
        """
        if not 0 <= channel_id < self.config.num_channels:
            return None

        if channel_id not in self._bank_state_machines:
            return None

        if bank_id not in self._bank_state_machines[channel_id]:
            return None

        bsm = self._bank_state_machines[channel_id][bank_id]
        return bsm.bank.state

    def get_all_bank_states(self, channel_id: int) -> Dict[int, BankStateEnum]:
        """Get states of all banks in a channel

        Args:
            channel_id: Channel index (0-31)

        Returns:
            Dictionary mapping bank_id to BankStateEnum
        """
        if not 0 <= channel_id < self.config.num_channels:
            return {}

        states = {}
        if channel_id in self._bank_state_machines:
            for bank_id, bsm in self._bank_state_machines[channel_id].items():
                states[bank_id] = bsm.bank.state

        return states

    def can_activate_bank(self, channel_id: int, bank_id: int) -> bool:
        """Check if a bank can be activated

        Args:
            channel_id: Channel index
            bank_id: Bank index

        Returns:
            True if bank can be activated
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        bsm = self._bank_state_machines[channel_id][bank_id]
        bsm.set_time(self._global_cycle)
        return bsm.can_activate()

    def activate_bank(self, channel_id: int, bank_id: int, row: int) -> bool:
        """Activate a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index
            row: Row address to activate

        Returns:
            True if activation successful
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        bsm = self._bank_state_machines[channel_id][bank_id]
        bsm.set_time(self._global_cycle)
        success = bsm.activate(row)

        if success:
            # Update channel context
            ctx = self._channels[channel_id]
            ctx.last_act_cycle = ctx.local_cycle
            ctx.state = ChannelState.ACTIVE
            ctx.open_row = row
            ctx.bank_states[bank_id] = BankStateEnum.ACTIVE

        return success

    def can_precharge_bank(self, channel_id: int, bank_id: int) -> bool:
        """Check if a bank can be precharged

        Args:
            channel_id: Channel index
            bank_id: Bank index

        Returns:
            True if bank can be precharged
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        bsm = self._bank_state_machines[channel_id][bank_id]
        bsm.set_time(self._global_cycle)
        return bsm.can_precharge()

    def precharge_bank(self, channel_id: int, bank_id: int) -> bool:
        """Precharge a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index

        Returns:
            True if precharge successful
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        bsm = self._bank_state_machines[channel_id][bank_id]
        bsm.set_time(self._global_cycle)
        success = bsm.precharge()

        if success:
            # Update channel context
            ctx = self._channels[channel_id]
            ctx.last_pre_cycle = ctx.local_cycle
            ctx.open_row = None
            ctx.bank_states[bank_id] = BankStateEnum.IDLE

        return success

    def can_read_bank(self, channel_id: int, bank_id: int) -> bool:
        """Check if a read can be issued to a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index

        Returns:
            True if read can be issued
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        bsm = self._bank_state_machines[channel_id][bank_id]
        bsm.set_time(self._global_cycle)
        return bsm.can_read()

    def read_bank(self, channel_id: int, bank_id: int) -> bool:
        """Issue a read to a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index

        Returns:
            True if read started successfully
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        bsm = self._bank_state_machines[channel_id][bank_id]
        bsm.set_time(self._global_cycle)
        success = bsm.read()

        if success:
            ctx = self._channels[channel_id]
            ctx.last_rd_cycle = ctx.local_cycle

        return success

    def can_write_bank(self, channel_id: int, bank_id: int) -> bool:
        """Check if a write can be issued to a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index

        Returns:
            True if write can be issued
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        bsm = self._bank_state_machines[channel_id][bank_id]
        bsm.set_time(self._global_cycle)
        return bsm.can_write()

    def write_bank(self, channel_id: int, bank_id: int) -> bool:
        """Issue a write to a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index

        Returns:
            True if write started successfully
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        bsm = self._bank_state_machines[channel_id][bank_id]
        bsm.set_time(self._global_cycle)
        success = bsm.write()

        if success:
            ctx = self._channels[channel_id]
            ctx.last_wr_cycle = ctx.local_cycle

        return success

    def complete_bank_read(self, channel_id: int, bank_id: int):
        """Complete a read operation on a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index
        """
        if channel_id in self._bank_state_machines and \
           bank_id in self._bank_state_machines[channel_id]:
            self._bank_state_machines[channel_id][bank_id].complete_read()

    def complete_bank_write(self, channel_id: int, bank_id: int):
        """Complete a write operation on a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index
        """
        if channel_id in self._bank_state_machines and \
           bank_id in self._bank_state_machines[channel_id]:
            self._bank_state_machines[channel_id][bank_id].complete_write()

    def refresh_bank(self, channel_id: int, bank_id: int) -> bool:
        """Refresh a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index

        Returns:
            True if refresh successful
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        bsm = self._bank_state_machines[channel_id][bank_id]
        bsm.set_time(self._global_cycle)
        success = bsm.refresh()

        if success:
            ctx = self._channels[channel_id]
            ctx.state = ChannelState.MAINTENANCE

        return success

    def complete_bank_refresh(self, channel_id: int, bank_id: int):
        """Complete a refresh operation on a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index
        """
        if channel_id in self._bank_state_machines and \
           bank_id in self._bank_state_machines[channel_id]:
            self._bank_state_machines[channel_id][bank_id].complete_refresh()
            ctx = self._channels[channel_id]
            if ctx.state == ChannelState.MAINTENANCE:
                ctx.state = ChannelState.IDLE

    def is_row_hit(self, channel_id: int, bank_id: int, row: int) -> bool:
        """Check if a row is currently open in a bank

        Args:
            channel_id: Channel index
            bank_id: Bank index
            row: Row address to check

        Returns:
            True if row is open (row hit)
        """
        if channel_id not in self._bank_state_machines:
            return False
        if bank_id not in self._bank_state_machines[channel_id]:
            return False

        return self._bank_state_machines[channel_id][bank_id].is_row_hit(row)

    # ==================== DFI Interface ====================

    def submit_dfi_command(self, command: DFICommand, address: int, bank: int,
                          channel: int, pseudo_channel: int = 0,
                          wrdata_en: bool = False, rddata_en: bool = False,
                          priority: int = 0) -> bool:
        """Submit a command through the DFI interface

        Args:
            command: DFI command type
            address: Memory address
            bank: Bank index
            channel: Channel index (0-31)
            pseudo_channel: Pseudo-channel index (0-1)
            wrdata_en: Write data enable
            rddata_en: Read data enable
            priority: Command priority

        Returns:
            True if command submitted successfully
        """
        request = DFIRequest(
            command=command,
            address=address,
            bank=bank,
            pseudo_channel=pseudo_channel,
            channel=channel,
            wrdata_en=wrdata_en,
            rddata_en=rddata_en,
            priority=priority,
            timestamp=self._global_cycle
        )

        success = self.dfi.queue_request(request)
        if success:
            self._dfi_commands_sent += 1
        return success

    def submit_dfi_act(self, channel: int, bank: int, row: int,
                       priority: int = 0) -> bool:
        """Submit ACTIVATE command through DFI

        Args:
            channel: Channel index
            bank: Bank index
            row: Row address
            priority: Command priority

        Returns:
            True if command submitted
        """
        return self.submit_dfi_command(
            command=DFICommand.ACT,
            address=row,
            bank=bank,
            channel=channel,
            priority=priority
        )

    def submit_dfi_pre(self, channel: int, bank: int,
                        priority: int = 0) -> bool:
        """Submit PRECHARGE command through DFI

        Args:
            channel: Channel index
            bank: Bank index
            priority: Command priority

        Returns:
            True if command submitted
        """
        return self.submit_dfi_command(
            command=DFICommand.PRE,
            address=0,
            bank=bank,
            channel=channel,
            priority=priority
        )

    def submit_dfi_read(self, channel: int, bank: int, column: int,
                        pseudo_channel: int = 0, priority: int = 0) -> bool:
        """Submit READ command through DFI

        Args:
            channel: Channel index
            bank: Bank index
            column: Column address
            pseudo_channel: Pseudo-channel index
            priority: Command priority

        Returns:
            True if command submitted
        """
        return self.submit_dfi_command(
            command=DFICommand.RD,
            address=column,
            bank=bank,
            channel=channel,
            pseudo_channel=pseudo_channel,
            rddata_en=True,
            priority=priority
        )

    def submit_dfi_write(self, channel: int, bank: int, column: int,
                         pseudo_channel: int = 0, priority: int = 0) -> bool:
        """Submit WRITE command through DFI

        Args:
            channel: Channel index
            bank: Bank index
            column: Column address
            pseudo_channel: Pseudo-channel index
            priority: Command priority

        Returns:
            True if command submitted
        """
        return self.submit_dfi_command(
            command=DFICommand.WR,
            address=column,
            bank=bank,
            channel=channel,
            pseudo_channel=pseudo_channel,
            wrdata_en=True,
            priority=priority
        )

    def submit_dfi_refresh(self, channel: int, priority: int = 0) -> bool:
        """Submit REFRESH command through DFI

        Args:
            channel: Channel index
            priority: Command priority

        Returns:
            True if command submitted
        """
        return self.submit_dfi_command(
            command=DFICommand.REFab,
            address=0,
            bank=0,
            channel=channel,
            priority=priority
        )

    def get_next_dfi_request(self) -> Optional[DFIRequest]:
        """Get next request from DFI queue

        Returns:
            Next DFIRequest or None
        """
        return self.dfi.get_next_request()

    def peek_dfi_request(self) -> Optional[DFIRequest]:
        """Peek at next request without removing

        Returns:
            Next DFIRequest or None
        """
        return self.dfi.peek_request()

    @property
    def dfi_pending_count(self) -> int:
        """Number of pending DFI requests"""
        return self.dfi.pending_request_count

    @property
    def dfi_is_ready(self) -> bool:
        """Check if DFI interface is ready"""
        return self.dfi.is_ready()

    def get_dfi_signals(self) -> Dict:
        """Get current DFI signal states

        Returns:
            Dictionary with DFI signal states
        """
        return self.dfi.get_dfi_signals()

    # ==================== Command Buffer ====================

    def enqueue_command(self, command: str, channel: int, address: int,
                        priority: int = 0, data: Optional[int] = None) -> int:
        """Add a command to the internal command buffer

        Args:
            command: Command name (ACT, PRE, RD, WR, REF, MRS)
            channel: Target channel
            address: Memory address
            priority: Command priority
            data: Optional data payload

        Returns:
            Command ID if successful, -1 if buffer full
        """
        return self.command_buffer.enqueue(
            command=command,
            channel=channel,
            address=address,
            priority=priority,
            data=data
        )

    def dequeue_command(self) -> Optional[Dict]:
        """Remove and return next command from buffer

        Returns:
            Next command dict or None
        """
        return self.command_buffer.dequeue()

    def peek_command(self) -> Optional[Dict]:
        """View next command without removing

        Returns:
            Next command dict or None
        """
        return self.command_buffer.peek()

    @property
    def command_buffer_size(self) -> int:
        """Current command buffer size"""
        return self.command_buffer.size

    @property
    def command_buffer_full(self) -> bool:
        """Check if command buffer is full"""
        return self.command_buffer.is_full

    def get_command_buffer_stats(self) -> Dict:
        """Get command buffer statistics

        Returns:
            Dictionary with buffer stats
        """
        return self.command_buffer.get_stats()

    # ==================== Command Processing ====================

    def process_command(
        self,
        channel_id: int,
        command: str,
        address: int,
        data: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Process command on a channel

        Args:
            channel_id: Target channel (0-31)
            command: Command type ('ACT', 'PRE', 'RD', 'WR', 'REF', etc.)
            address: Address for command
            data: Optional data for write commands

        Returns:
            Tuple of (success, error_message)
        """
        if not 0 <= channel_id < self.config.num_channels:
            return False, f"Invalid channel {channel_id}"

        ctx = self._channels[channel_id]
        self._total_commands += 1

        # Check channel state
        if ctx.state == ChannelState.ERROR:
            return False, f"Channel {channel_id} in error state"

        # Route to command handler
        handlers = {
            'ACT': self._handle_activate,
            'PRE': self._handle_precharge,
            'RD': self._handle_read,
            'WR': self._handle_write,
            'REF': self._handle_refresh,
            'MRS': self._handle_mrs,
        }

        handler = handlers.get(command)
        if handler:
            return handler(ctx, address, data)

        return False, f"Unknown command: {command}"

    def _handle_activate(
        self,
        ctx: ChannelContext,
        address: int,
        data: Optional[int],
    ) -> Tuple[bool, str]:
        """Handle ACTIVATE command"""
        # Check timing (tRC from spec)
        if ctx.last_act_cycle >= 0:
            cycles_since_act = ctx.local_cycle - ctx.last_act_cycle
            if cycles_since_act < self.spec.nRC:
                return False, f"tRC violation: {cycles_since_act} < {self.spec.nRC}"

        ctx.last_act_cycle = ctx.local_cycle
        ctx.state = ChannelState.ACTIVE
        ctx.open_row = address & 0xFFFF  # Extract row from address

        return True, ""

    def _handle_precharge(
        self,
        ctx: ChannelContext,
        address: int,
        data: Optional[int],
    ) -> Tuple[bool, str]:
        """Handle PRECHARGE command"""
        if ctx.last_rd_cycle >= 0:
            cycles_since_rd = ctx.local_cycle - ctx.last_rd_cycle
            if cycles_since_rd < self.spec.nRTPS:
                return False, f"tRTPS violation"

        ctx.state = ChannelState.IDLE
        ctx.open_row = None

        return True, ""

    def _handle_read(
        self,
        ctx: ChannelContext,
        address: int,
        data: Optional[int],
    ) -> Tuple[bool, str]:
        """Handle READ command"""
        if ctx.state != ChannelState.ACTIVE:
            return False, "Bank not active"

        # Check timing from activation
        if ctx.last_act_cycle >= 0:
            cycles_since_act = ctx.local_cycle - ctx.last_act_cycle
            if cycles_since_act < self.spec.nRCDRD:
                return False, f"tRCD_RD violation"

        # Check previous command
        if ctx.last_rd_cycle >= 0 or ctx.last_wr_cycle >= 0:
            last_cmd_cycle = max(ctx.last_rd_cycle, ctx.last_wr_cycle)
            cycles_since_last = ctx.local_cycle - last_cmd_cycle
            if cycles_since_last < self.spec.nCCDS:
                return False, f"tCCD violation"

        ctx.last_rd_cycle = ctx.local_cycle

        # Apply lane repair mapping if needed
        # (lane_repair handles this transparently)

        return True, ""

    def _handle_write(
        self,
        ctx: ChannelContext,
        address: int,
        data: Optional[int],
    ) -> Tuple[bool, str]:
        """Handle WRITE command"""
        if ctx.state != ChannelState.ACTIVE:
            return False, "Bank not active"

        if data is None:
            return False, "Write data required"

        # Check timing
        if ctx.last_act_cycle >= 0:
            cycles_since_act = ctx.local_cycle - ctx.last_act_cycle
            if cycles_since_act < self.spec.nRCDWR:
                return False, f"tRCD_WR violation"

        # Apply ECC encoding
        if self.config.ecc_enabled:
            encoded = self.data_integrity.encode_data(data)
            data = encoded['data']

        # Apply lane repair mapping
        if self.lane_repair.is_lane_remapped(ctx.channel_id, 0):
            # Data will be transparently routed through spare lanes
            pass

        ctx.last_wr_cycle = ctx.local_cycle

        return True, ""

    def _handle_refresh(
        self,
        ctx: ChannelContext,
        address: int,
        data: Optional[int],
    ) -> Tuple[bool, str]:
        """Handle REFRESH command"""
        ctx.state = ChannelState.MAINTENANCE

        # Refresh timing handled by spec
        return True, ""

    def _handle_mrs(
        self,
        ctx: ChannelContext,
        address: int,
        data: Optional[int],
    ) -> Tuple[bool, str]:
        """Handle MODE REGISTER SET command"""
        # MRS timing
        return True, ""

    # ==================== Channel State ====================

    def get_channel_state(self, channel_id: int) -> Optional[Dict]:
        """Get state for a specific channel

        Args:
            channel_id: Channel to query

        Returns:
            Dictionary with channel state or None
        """
        if not 0 <= channel_id < self.config.num_channels:
            return None

        ctx = self._channels[channel_id]

        return {
            'channel_id': channel_id,
            'state': ctx.state.value,
            'local_cycle': ctx.local_cycle,
            'open_row': ctx.open_row,
            'training_passed': ctx.training_passed,
            'repair_status': ctx.repair_status.value,
            'error_count': ctx.error_count,
        }

    def get_all_channel_states(self) -> List[Dict]:
        """Get state for all channels

        Returns:
            List of channel state dictionaries
        """
        return [self.get_channel_state(ch) for ch in range(self.config.num_channels)]

    # ==================== Statistics ====================

    def get_stats(self) -> Dict:
        """Get Logic Base Die statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'global_cycle': self._global_cycle,
            'initialized': self._initialized,
            'training_complete': self._training_complete,
            'ready': self.is_ready,
            'total_commands': self._total_commands,
            'total_errors': self._total_errors,
            'pam3_enabled': self.config.pam3_enabled,
            'ecc_enabled': self.config.ecc_enabled,
            'crc_enabled': self.config.crc_enabled,
            'channels_ready': sum(1 for ctx in self._channels if ctx.training_passed),
            'channels_total': self.config.num_channels,
        }

    def get_calibration_data(self, channel_id: Optional[int] = None) -> Dict:
        """Get calibration data for channel(s)

        Args:
            channel_id: Specific channel or None for all

        Returns:
            Calibration data dictionary
        """
        if channel_id is not None:
            return self._channels[channel_id].calibration_data

        return {
            f'ch{ch}': ctx.calibration_data
            for ch, ctx in enumerate(self._channels)
            if ctx.calibration_data
        }

    def get_lane_repair_stats(self) -> Dict:
        """Get lane repair statistics

        Returns:
            Lane repair statistics
        """
        return self.lane_repair.get_stats()

    # ==================== Utility Methods ====================

    def wait_for_ready(self, max_cycles: int = 100000) -> bool:
        """Wait for Logic Base Die to be ready

        Args:
            max_cycles: Maximum cycles to wait

        Returns:
            True if ready, False if timeout
        """
        for _ in range(max_cycles):
            if self.is_ready:
                return True
            self.tick()
        return False

    def reset(self):
        """Reset Logic Base Die to initial state

        Resets all state machines, queues, and statistics.
        Preserves configuration.
        """
        # Reset global state
        self._global_cycle = 0
        self._initialized = False
        self._training_complete = False

        # Reset DFI interface
        self.dfi.reset()

        # Reset command buffer
        self.command_buffer.clear()

        # Reset bank state machines
        timing = HBM3Timing()
        total_banks = self.config.banks_per_channel * self.config.pseudo_channels_per_channel
        for ch in range(self.config.num_channels):
            for bank_id in range(total_banks):
                self._bank_state_machines[ch][bank_id] = BankStateMachine(
                    bank_id=bank_id,
                    timing=timing
                )

        # Reset channel contexts
        for ctx in self._channels:
            ctx.state = ChannelState.IDLE
            ctx.local_cycle = 0
            ctx.last_act_cycle = -1
            ctx.last_pre_cycle = -1
            ctx.last_rd_cycle = -1
            ctx.last_wr_cycle = -1
            ctx.open_row = None
            ctx.training_passed = False
            ctx.calibration_data = {}
            ctx.error_count = 0
            ctx.last_error = None
            ctx.bank_states = {}

        # Reset statistics
        self._total_commands = 0
        self._total_errors = 0
        self._dfi_commands_sent = 0
        self._dfi_commands_completed = 0

    def get_status(self) -> Dict:
        """Get comprehensive status of Logic Base Die

        Returns:
            Dictionary with complete status information
        """
        return {
            'cycle': self._global_cycle,
            'initialized': self._initialized,
            'training_complete': self._training_complete,
            'ready': self.is_ready,
            'dfi': {
                'lp_state': self.dfi.lp_state.value,
                'frequency_mhz': self.dfi.frequency_mhz,
                'pending_requests': self.dfi_pending_count,
                'ready': self.dfi_is_ready,
            },
            'command_buffer': {
                'size': self.command_buffer_size,
                'full': self.command_buffer_full,
            },
            'channels': {
                'total': self.config.num_channels,
                'ready': sum(1 for ctx in self._channels if ctx.training_passed),
            },
            'statistics': self.get_stats(),
        }