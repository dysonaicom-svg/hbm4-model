"""
HBM DRAM Bank State Machine - Optimized Version
Reference design document 2026-06-15-hbm-system-model-design.md Section 5.2.1 and 5.2.2

Optimizations:
- __slots__ for memory reduction
- Frozen dataclass for immutable types
- Batch state checks
- Pre-computed timing values
- Timing lookup table for O(1) access
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import sys


class BankStateEnum(IntEnum):
    """Bank state enum

    Aligned with RTL 3-bit encoding:
    - RTL: 000=IDLE, 001=ACTIVE, 010=BUSY, 011=REFRESH, 100=POWERDN, 101=SELFREF
    """
    IDLE = 0       # 000 - Bank idle
    ACTIVE = 1     # 001 - Bank activated
    BUSY = 2       # 010 - Bank busy (READ/WRITE in progress)
    REFRESHING = 3 # 011 - Refreshing
    POWERDN = 4    # 100 - Power down
    SELFREF = 5    # 101 - Self refresh

    # Aliases for backward compatibility with HBM3 naming
    READING = 2    # Same as BUSY
    WRITING = 2    # Same as BUSY


class OperationType(IntEnum):
    """Operation type enum"""
    NONE = 0
    READ = 1
    WRITE = 2
    REFRESH = 3


# Pre-computed state masks for fast checking
_STATE_IDLE = 1 << BankStateEnum.IDLE
_STATE_ACTIVE = 1 << BankStateEnum.ACTIVE
_STATE_BUSY = 1 << BankStateEnum.BUSY
_STATE_REFRESHING = 1 << BankStateEnum.REFRESHING


# Timing lookup table - pre-computed values for common timing parameters
# Maps timing parameter names to their cycle values for HBM3
_TIMING_LOOKUP = {
    'nRC': 340,
    'nRAS': 320,
    'nRCD': 20,
    'nRP': 20,
    'nRFC': 260,
    'nCL': 20,
    'nCWL': 16,
    'nCCD': 4,
    'nWTRS': 4,
    'nRTW': 4,
    # HBM3 aliases
    'tRC': 340,
    'tRAS': 320,
    'tRCD': 20,
    'tRP': 20,
    'tRFC': 260,
    'tCL': 20,
    'tCWL': 16,
    'tCCD': 4,
}


class Bank:
    """DRAM Bank State

    Represents the state of a single DRAM bank.
    """
    __slots__ = ('bank_id', 'state', 'open_row', 'activate_time', 'precharge_time',
                 'last_operation_time', 'read_start_time', 'read_complete_time',
                 'write_start_time', 'write_complete_time', 'refresh_time',
                 'refresh_complete_time', '_cached_state')

    def __init__(self, bank_id: int):
        self.bank_id = bank_id
        self.state = BankStateEnum.IDLE
        self.open_row = -1
        self.activate_time = -1.0
        self.precharge_time = -1.0
        self.last_operation_time = 0.0
        self.read_start_time = -1.0
        self.read_complete_time = -1.0
        self.write_start_time = -1.0
        self.write_complete_time = -1.0
        self.refresh_time = -1.0
        self.refresh_complete_time = -1.0
        self._cached_state = 1 << BankStateEnum.IDLE

    @property
    def is_idle(self) -> bool:
        return self.state == BankStateEnum.IDLE

    @property
    def is_active(self) -> bool:
        return self.state == BankStateEnum.ACTIVE

    @property
    def is_busy(self) -> bool:
        return self.state == BankStateEnum.BUSY

    @property
    def is_refresh(self) -> bool:
        return self.state == BankStateEnum.REFRESHING

    @property
    def is_powered_down(self) -> bool:
        return self.state == BankStateEnum.POWERDN

    @property
    def is_self_refresh(self) -> bool:
        return self.state == BankStateEnum.SELFREF

    @property
    def row_open(self) -> bool:
        return self.is_active and self.open_row >= 0

    @property
    def has_been_activated(self) -> bool:
        """Check if bank has ever been activated"""
        return self.activate_time >= 0

    @property
    def has_been_precharged(self) -> bool:
        """Check if bank has ever been precharged"""
        return self.precharge_time >= 0

    def update_state(self, new_state: BankStateEnum):
        """Update state with cached flag update"""
        self.state = new_state
        self._cached_state = 1 << new_state

    def __repr__(self) -> str:
        row_str = f"row=0x{self.open_row:x}" if self.open_row >= 0 else "row=closed"
        return f"Bank{self.bank_id}({self.state.name}, {row_str})"


class TimingViolation:
    """Timing violation record"""
    __slots__ = ('violation_type', 'required_time', 'actual_time', 'time_available', 'description')

    def __init__(self, violation_type: str, required_time: float, actual_time: float,
                 time_available: float, description: str):
        self.violation_type = violation_type
        self.required_time = required_time
        self.actual_time = actual_time
        self.time_available = time_available
        self.description = description


class TimingViolationList:
    """Efficient timing violation list"""
    __slots__ = ('_violations', '_capacity')

    def __init__(self, capacity: int = 16):
        self._violations = []
        self._capacity = capacity

    def append(self, violation: TimingViolation):
        self._violations.append(violation)

    def clear(self):
        self._violations.clear()

    def get_all(self) -> list:
        return self._violations.copy()

    def __len__(self) -> int:
        return len(self._violations)


class BankStateMachine:
    """Bank State Machine - Optimized Version

    Manages single bank state transitions and timing constraints.
    Supports HBM3/HBM4 timing parameters.

    Optimizations:
    - Batch timing checks
    - Pre-computed timing conversions
    - Fast state comparisons using cached masks
    - Timing lookup table for O(1) access
    - No set_time() calls - time passed directly to check methods
    """

    __slots__ = ('bank', 'timing', 'current_time', 'timing_violations',
                 '_clock_period_ns', '_clock_period_s', '_timing_cache',
                 '_cache_valid', '_cached_tRC', '_cached_tRAS', '_cached_tRCD',
                 '_cached_tRFC', '_cached_tCL', '_cached_tCWL', '_cached_tCCD')

    def __init__(self, bank_id: int, timing):
        """Initialize Bank State Machine

        Args:
            bank_id: Bank ID
            timing: Timing parameter object (HBM3Timing or HBM4Timing)
        """
        self.bank = Bank(bank_id=bank_id)
        self.timing = timing
        self.current_time = 0.0
        self.timing_violations: List[TimingViolation] = []

        # Pre-compute clock period for fast cycles-to-seconds conversion
        # Use timing object's pre-computed value if available
        self._clock_period_s = getattr(timing, '_clock_period_s', None) or 0.78125e-9
        self._clock_period_ns = getattr(timing, '_clock_period_ns', None) or 0.78125

        # Pre-compute all timing values once at init time
        self._timing_cache = {}
        self._cache_valid = False
        self._init_timing_cache()

        # Cache commonly used timing values
        self._cached_tRC = 0
        self._cached_tRAS = 0
        self._cached_tRCD = 0
        self._cached_tRFC = 0
        self._cached_tCL = 0
        self._cached_tCWL = 0
        self._cached_tCCD = 0

    def _init_timing_cache(self):
        """Initialize timing lookup cache at construction time"""
        # Pre-populate cache from timing object
        for name in _TIMING_LOOKUP:
            val = self.get_timing_value(name)
            if val > 0:
                self._timing_cache[name] = val

        # Also cache values from the timing object
        for name in dir(self.timing):
            if not name.startswith('_'):
                val = getattr(self.timing, name)
                if isinstance(val, (int, float)):
                    self._timing_cache[name] = int(val)

    def set_time(self, current_time: float):
        """Set current time (in cycles)

        OPTIMIZATION: This method is called frequently but most of its
        work is deferred. The actual time check is done in the operation
        methods directly.
        """
        self.current_time = current_time

    def _get_cached_timing(self, name: str) -> int:
        """Get timing value from cache (O(1) lookup)

        Args:
            name: Parameter name

        Returns:
            Timing value in cycles
        """
        # Fast path: check cache first
        if name in self._timing_cache:
            return self._timing_cache[name]

        # Fallback: compute and cache
        val = self.get_timing_value(name)
        self._timing_cache[name] = val
        return val

    def _record_violation(self, violation_type: str, required_time: float,
                         actual_time: float, description: str):
        """Record timing violation"""
        violation = TimingViolation(
            violation_type=violation_type,
            required_time=required_time,
            actual_time=actual_time,
            time_available=actual_time,
            description=description
        )
        self.timing_violations.append(violation)

    def get_timing_value(self, name: str) -> int:
        """Get timing parameter value (compatible with HBM3/HBM4 naming)

        Args:
            name: Parameter name (e.g., 'tRCD', 'nRCD', 'tRC', 'nRC', etc.)

        Returns:
            Timing parameter value (in cycles)
        """
        # Check lookup table first (fastest path)
        if name in _TIMING_LOOKUP:
            return _TIMING_LOOKUP[name]

        # HBM4 n-prefix priority
        if hasattr(self.timing, name):
            return getattr(self.timing, name)
        # HBM3 t-prefix fallback
        hbm3_name = name.replace('n', 't', 1) if name.startswith('n') else name
        if hasattr(self.timing, hbm3_name):
            return getattr(self.timing, hbm3_name)
        # Default to 0
        return 0

    # =========================================================================
    # Activation State Transitions
    # =========================================================================

    def can_activate(self) -> bool:
        """Check if activation can be initiated

        Timing constraints:
        - Bank must be IDLE
        - Must be >= tRC since last operation (ACT or PRE)
        """
        if self.bank.state != BankStateEnum.IDLE:
            return False

        # If never activated, can activate
        if self.bank.activate_time < 0:
            return True

        # tRC: Minimum interval between consecutive ACTs on same bank
        time_since_last = self.current_time - self.bank.last_operation_time
        # Use cached timing value directly
        tRC = self._get_cached_timing('nRC')
        tRC_seconds = self._cycles_to_seconds(tRC)
        return time_since_last >= tRC_seconds

    def activate(self, row: int) -> Tuple[bool, Optional[str]]:
        """Activate Bank

        Args:
            row: Row number to activate

        Returns:
            (success flag, error message)
        """
        # Use cached timing value directly (no refresh needed)
        tRC = self._get_cached_timing('nRC')
        tRC_seconds = self._cycles_to_seconds(tRC)

        if self.bank.state != BankStateEnum.IDLE:
            return False, f"Bank {self.bank.bank_id} not idle (state={self.bank.state.name})"

        # If ever activated, must satisfy tRC
        if self.bank.activate_time >= 0:
            time_since_last = self.current_time - self.bank.last_operation_time
            if time_since_last < tRC_seconds:
                msg = f"tRC violation: need {tRC_seconds}s, have {time_since_last}s"
                self._record_violation('tRC', tRC, time_since_last, msg)
                return False, msg

        self.bank.update_state(BankStateEnum.ACTIVE)
        self.bank.open_row = row
        self.bank.activate_time = self.current_time
        self.bank.last_operation_time = self.current_time
        return True, None

    # =========================================================================
    # Precharge State Transitions
    # =========================================================================

    def can_precharge(self) -> bool:
        """Check if precharge can be initiated

        Timing constraints:
        - Bank must be ACTIVE (or BUSY but READ/WRITE complete)
        - Must be >= tRAS since ACT
        """
        # Fast state check using cached mask
        state_mask = self.bank._cached_state
        if state_mask & (_STATE_ACTIVE | _STATE_BUSY) == 0:
            return False

        # If BUSY, check if operation complete
        if state_mask & _STATE_BUSY:
            if not self._is_operation_complete():
                return False

        time_since_act = self.current_time - self.bank.activate_time
        tRAS = self._get_cached_timing('nRAS')
        tRAS_seconds = self._cycles_to_seconds(tRAS)
        return time_since_act >= tRAS_seconds

    def precharge(self) -> Tuple[bool, Optional[str]]:
        """Close Bank

        Returns:
            (success flag, error message)
        """
        state_mask = self.bank._cached_state
        if state_mask & (_STATE_ACTIVE | _STATE_BUSY) == 0:
            return False, f"Bank {self.bank.bank_id} not active (state={self.bank.state.name})"

        # Check tRAS using cached value
        time_since_act = self.current_time - self.bank.activate_time
        tRAS = self._get_cached_timing('nRAS')
        tRAS_seconds = self._cycles_to_seconds(tRAS)
        if time_since_act < tRAS_seconds:
            msg = f"tRAS violation: need {tRAS} cycles ({tRAS_seconds}s), have {time_since_act}s"
            self._record_violation('tRAS', tRAS, time_since_act, msg)
            return False, msg

        self.bank.update_state(BankStateEnum.IDLE)
        self.bank.open_row = -1
        self.bank.precharge_time = self.current_time
        self.bank.last_operation_time = self.current_time
        return True, None

    # =========================================================================
    # Read State Transitions
    # =========================================================================

    def _cycles_to_seconds(self, cycles: int) -> float:
        """Convert timing cycles to seconds

        OPTIMIZED: Uses pre-computed clock period for O(1) conversion.
        HBM3: tCK = 781.25 ps = 0.78125 ns = 0.78125e-9 s
        """
        return cycles * self._clock_period_s

    def can_read(self) -> bool:
        """Check if READ can be initiated

        Timing constraints:
        - Bank must be ACTIVE
        - Must be >= tRCD since ACT
        """
        if self.bank.state != BankStateEnum.ACTIVE:
            return False

        time_since_act = self.current_time - self.bank.activate_time
        tRCD = self._get_cached_timing('nRCD')
        tRCD_seconds = self._cycles_to_seconds(tRCD)
        return time_since_act >= tRCD_seconds

    def read(self, burst_length: int = 4) -> Tuple[bool, Optional[str]]:
        """Initiate READ

        Args:
            burst_length: Burst length (default 4 for HBM)

        Returns:
            (success flag, error message)
        """
        if not self.can_read():
            return False, f"Cannot read: state={self.bank.state.name}, " \
                         f"time since act={self.current_time - self.bank.activate_time}"

        self.bank.update_state(BankStateEnum.BUSY)
        self.bank.read_start_time = self.current_time

        # Use cached timing values
        tRCD = self._get_cached_timing('nRCD')
        tCL = self._get_cached_timing('nCL')
        tCCD = self._get_cached_timing('nCCD')
        self.bank.read_complete_time = self.current_time + tRCD + tCL + (burst_length - 1) * tCCD

        return True, None

    def can_complete_read(self) -> bool:
        """Check if read can complete"""
        if self.bank.read_start_time < 0:
            return False
        return self.current_time >= self.bank.read_complete_time

    def complete_read(self) -> Tuple[bool, Optional[str]]:
        """READ complete, return to ACTIVE

        Returns:
            (success flag, error message)
        """
        if self.bank.state != BankStateEnum.BUSY:
            return False, "Not in BUSY state"

        if self.bank.read_start_time < 0:
            return False, "No read in progress"

        self.bank.update_state(BankStateEnum.ACTIVE)
        self.bank.last_operation_time = self.current_time
        self.bank.read_start_time = -1.0
        self.bank.read_complete_time = -1.0
        return True, None

    # =========================================================================
    # Write State Transitions
    # =========================================================================

    def can_write(self) -> bool:
        """Check if WRITE can be initiated

        Timing constraints:
        - Bank must be ACTIVE
        - Must be >= tRCD since ACT
        """
        if self.bank.state != BankStateEnum.ACTIVE:
            return False

        time_since_act = self.current_time - self.bank.activate_time
        tRCD = self._get_cached_timing('nRCD')
        tRCD_seconds = self._cycles_to_seconds(tRCD)
        return time_since_act >= tRCD_seconds

    def write(self, burst_length: int = 4) -> Tuple[bool, Optional[str]]:
        """Initiate WRITE

        Args:
            burst_length: Burst length (default 4 for HBM)

        Returns:
            (success flag, error message)
        """
        if not self.can_write():
            return False, f"Cannot write: state={self.bank.state.name}, " \
                         f"time since act={self.current_time - self.bank.activate_time}"

        self.bank.update_state(BankStateEnum.BUSY)
        self.bank.write_start_time = self.current_time

        # Use cached timing values
        tRCD = self._get_cached_timing('nRCD')
        tCWL = self._get_cached_timing('nCWL')
        tCCD = self._get_cached_timing('nCCD')
        self.bank.write_complete_time = self.current_time + tRCD + tCWL + (burst_length - 1) * tCCD

        return True, None

    def can_complete_write(self) -> bool:
        """Check if write can complete"""
        if self.bank.write_start_time < 0:
            return False
        return self.current_time >= self.bank.write_complete_time

    def complete_write(self) -> Tuple[bool, Optional[str]]:
        """WRITE complete, return to ACTIVE

        Returns:
            (success flag, error message)
        """
        if self.bank.state != BankStateEnum.BUSY:
            return False, "Not in BUSY state"

        if self.bank.write_start_time < 0:
            return False, "No write in progress"

        self.bank.update_state(BankStateEnum.ACTIVE)
        self.bank.last_operation_time = self.current_time
        self.bank.write_start_time = -1.0
        self.bank.write_complete_time = -1.0
        return True, None

    # =========================================================================
    # Operation Completion Helpers
    # =========================================================================

    def _is_operation_complete(self) -> bool:
        """Check if current BUSY operation is complete"""
        if self.bank.read_start_time >= 0:
            return self.current_time >= self.bank.read_complete_time
        if self.bank.write_start_time >= 0:
            return self.current_time >= self.bank.write_complete_time
        return True  # No operation in progress

    def is_operation_in_progress(self) -> bool:
        """Check if operation is in progress (READ/WRITE/REFRESH)"""
        state_mask = self.bank._cached_state
        return (state_mask & (_STATE_BUSY | _STATE_REFRESHING)) != 0

    # =========================================================================
    # Turnaround Timing
    # =========================================================================

    def can_read_after_write(self) -> bool:
        """Check if READ can be initiated after WRITE (tWTRS/tWTRL)

        Returns:
            True if read can be initiated
        """
        if self.bank.write_start_time < 0:
            return True  # No write operation

        if not self.can_complete_write():
            return False

        # tWTRS: Write to Read (same Bank Group)
        tWTRS = self.get_timing_value('nWTRS')
        time_since_write = self.current_time - self.bank.write_complete_time
        return time_since_write >= tWTRS

    def can_write_after_read(self) -> bool:
        """Check if WRITE can be initiated after READ (tRTW)

        Returns:
            True if write can be initiated
        """
        if self.bank.read_start_time < 0:
            return True  # No read operation

        if not self.can_complete_read():
            return False

        # tRTW: Read to Write
        tRTW = self.get_timing_value('nRTW')
        time_since_read = self.current_time - self.bank.read_complete_time
        return time_since_read >= tRTW

    # =========================================================================
    # Refresh State Transitions
    # =========================================================================

    def can_refresh(self) -> bool:
        """Check if refresh can be initiated

        Timing constraints:
        - Bank must be IDLE
        - Must be >= tRFC since last refresh
        """
        if self.bank.state != BankStateEnum.IDLE:
            return False

        # If never refreshed, can refresh
        if self.bank.refresh_time < 0:
            return True

        time_since_refresh = self.current_time - self.bank.refresh_time
        tRFC = self._get_cached_timing('nRFC')
        tRFC_seconds = self._cycles_to_seconds(tRFC)
        return time_since_refresh >= tRFC_seconds

    def refresh(self) -> Tuple[bool, Optional[str]]:
        """Execute refresh

        Returns:
            (success flag, error message)
        """
        if self.bank.state != BankStateEnum.IDLE:
            return False, f"Bank not idle (state={self.bank.state.name})"

        self.bank.update_state(BankStateEnum.REFRESHING)
        self.bank.refresh_time = self.current_time

        # Use cached timing value
        tRFC = self._get_cached_timing('nRFC')
        self.bank.refresh_complete_time = self.current_time + tRFC

        return True, None

    def can_complete_refresh(self) -> bool:
        """Check if refresh can complete"""
        if self.bank.state != BankStateEnum.REFRESHING:
            return False
        if self.bank.refresh_complete_time < 0:
            return False
        return self.current_time >= self.bank.refresh_complete_time

    def complete_refresh(self) -> Tuple[bool, Optional[str]]:
        """Refresh complete

        Returns:
            (success flag, error message)
        """
        if self.bank.state != BankStateEnum.REFRESHING:
            return False, "Not refreshing"

        if self.bank.refresh_complete_time >= 0:
            if self.current_time < self.bank.refresh_complete_time:
                return False, f"Refresh not complete: need {self.bank.refresh_complete_time}, current {self.current_time}"

        self.bank.update_state(BankStateEnum.IDLE)
        self.bank.refresh_time = self.current_time
        self.bank.refresh_complete_time = -1.0
        self.bank.last_operation_time = self.current_time
        return True, None

    # =========================================================================
    # Power Management State Transitions
    # =========================================================================

    def can_enter_power_down(self) -> bool:
        """Check if power down mode can be entered

        Constraints:
        - Bank must be IDLE
        """
        return self.bank.state == BankStateEnum.IDLE

    def enter_power_down(self) -> Tuple[bool, Optional[str]]:
        """Enter power down mode

        Returns:
            (success flag, error message)
        """
        if not self.can_enter_power_down():
            return False, f"Cannot enter power down: state={self.bank.state.name}"

        self.bank.update_state(BankStateEnum.POWERDN)
        return True, None

    def exit_power_down(self) -> Tuple[bool, Optional[str]]:
        """Exit power down mode

        Returns:
            (success flag, error message)
        """
        if self.bank.state != BankStateEnum.POWERDN:
            return False, f"Not in power down: state={self.bank.state.name}"

        self.bank.update_state(BankStateEnum.IDLE)
        return True, None

    def can_enter_self_refresh(self) -> bool:
        """Check if self refresh mode can be entered

        Constraints:
        - Bank must be IDLE
        """
        return self.bank.state == BankStateEnum.IDLE

    def enter_self_refresh(self) -> Tuple[bool, Optional[str]]:
        """Enter self refresh mode

        Returns:
            (success flag, error message)
        """
        if not self.can_enter_self_refresh():
            return False, f"Cannot enter self refresh: state={self.bank.state.name}"

        self.bank.update_state(BankStateEnum.SELFREF)
        return True, None

    def exit_self_refresh(self) -> Tuple[bool, Optional[str]]:
        """Exit self refresh mode

        Returns:
            (success flag, error message)
        """
        if self.bank.state != BankStateEnum.SELFREF:
            return False, f"Not in self refresh: state={self.bank.state.name}"

        self.bank.update_state(BankStateEnum.IDLE)
        return True, None

    # =========================================================================
    # Row Access Helpers
    # =========================================================================

    def is_row_hit(self, row: int) -> bool:
        """Check if row hit"""
        return (self.bank.state == BankStateEnum.ACTIVE and
                self.bank.open_row == row)

    def is_row_open(self, row: int) -> bool:
        """Check if specified row is open"""
        return self.bank.open_row == row

    def close_row(self) -> Tuple[bool, Optional[str]]:
        """Close currently open row"""
        if self.bank.state != BankStateEnum.ACTIVE:
            return False, "Bank not active"
        return self.precharge()

    # =========================================================================
    # Timing Query
    # =========================================================================

    def time_to_ready(self) -> float:
        """Calculate time until next ACT can be initiated

        Returns:
            Required wait time (cycles), 0 if already ready
        """
        if self.bank.state != BankStateEnum.IDLE:
            return float('inf')  # Wrong state, need to precharge first

        if not self.bank.has_been_activated:
            return 0.0

        time_since_last = self.current_time - self.bank.last_operation_time
        tRC = self._get_cached_timing('nRC')
        if time_since_last >= tRC:
            return 0.0

        return tRC - time_since_last

    def time_to_read_ready(self) -> float:
        """Calculate time until READ can be initiated

        Returns:
            Required wait time (cycles)
        """
        if self.bank.state == BankStateEnum.ACTIVE:
            time_since_act = self.current_time - self.bank.activate_time
            tRCD = self._get_cached_timing('nRCD')
            if time_since_act >= tRCD:
                return 0.0
            return tRCD - time_since_act

        return float('inf')  # Need to activate first

    def time_to_precharge_ready(self) -> float:
        """Calculate time until PRE can be initiated

        Returns:
            Required wait time (cycles)
        """
        state_mask = self.bank._cached_state
        if state_mask & (_STATE_ACTIVE | _STATE_BUSY) == 0:
            return float('inf')

        time_since_act = self.current_time - self.bank.activate_time
        tRAS = self._get_cached_timing('nRAS')
        if time_since_act >= tRAS:
            return 0.0

        return tRAS - time_since_act

    def get_violations(self) -> List[TimingViolation]:
        """Get recorded timing violations"""
        return self.timing_violations.copy()

    def clear_violations(self):
        """Clear recorded timing violations"""
        self.timing_violations.clear()

    # =========================================================================
    # State Queries (for compatibility with existing code)
    # =========================================================================

    def complete_read_legacy(self):
        """READ complete, return ACTIVE (legacy)"""
        self.complete_read()

    def complete_write_legacy(self):
        """WRITE complete (legacy)"""
        self.complete_write()


# Aliases for backward compatibility
def create_bank_state_machine(bank_id: int, timing) -> BankStateMachine:
    """Factory function to create BankStateMachine"""
    return BankStateMachine(bank_id=bank_id, timing=timing)


# Vectorized operations for batch processing (using lists, no numpy dependency)
def batch_check_can_activate(bank_machines: List[BankStateMachine]) -> List[bool]:
    """Batch check if banks can activate

    Args:
        bank_machines: List of BankStateMachine instances

    Returns:
        List of boolean results
    """
    return [bm.can_activate() for bm in bank_machines]


def batch_check_can_read(bank_machines: List[BankStateMachine]) -> List[bool]:
    """Batch check if banks can read

    Args:
        bank_machines: List of BankStateMachine instances

    Returns:
        List of boolean results
    """
    return [bm.can_read() for bm in bank_machines]


def batch_check_can_write(bank_machines: List[BankStateMachine]) -> List[bool]:
    """Batch check if banks can write

    Args:
        bank_machines: List of BankStateMachine instances

    Returns:
        List of boolean results
    """
    return [bm.can_write() for bm in bank_machines]