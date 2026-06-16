"""
HBM Request and Response Classes - Optimized Version

Optimizations:
- __slots__ for memory reduction
- optimized property access
"""

from enum import IntEnum
from typing import Optional, ClassVar


class RequestState(IntEnum):
    """Request state enum.

    Represents the state of a request in its lifecycle.
    """
    PENDING = 0      # Waiting for scheduling
    SCHEDULED = 1    # Scheduled, waiting for execution
    IN_PROGRESS = 2 # In execution
    COMPLETED = 3    # Completed
    FAILED = 4       # Failed


# Pre-computed state masks for fast checking
_STATE_COMPLETED_MASK = 1 << RequestState.COMPLETED
_STATE_FAILED_MASK = 1 << RequestState.FAILED
_STATE_PENDING_MASK = 1 << RequestState.PENDING


# Class variables for HBMRequest - defined at module level (not in __slots__)
_HBMRequest_next_id: int = 0


class HBMRequest:
    """HBM Memory Request - Optimized Version

    Uses __slots__ to reduce memory footprint and improve access speed.

    Attributes:
        addr: 64-bit memory address
        length: Request length (bytes)
        is_read: True=read request, False=write request
        qos: QoS priority (0-15, 15 highest)
        burst_length: Burst size
        request_id: Globally unique request ID
        arrival_time: Request arrival timestamp
        stack_id: Decoded stack ID
        channel_id: Decoded channel ID
        pseudo_channel_id: Decoded pseudo channel ID
        bank_group_id: Decoded bank group ID
        bank_id: Decoded bank ID
        row_id: Decoded row ID
        col_id: Decoded column ID
        row_hit: Whether row hit
        state: Current request state
        scheduled_time: Scheduling time
        completion_time: Completion time
        data: Write data (bytes, for write requests only)
    """
    __slots__ = (
        'addr', 'length', 'is_read', 'qos', 'burst_length',
        'request_id', 'arrival_time',
        'stack_id', 'channel_id', 'pseudo_channel_id',
        'bank_group_id', 'bank_id', 'row_id', 'col_id',
        'row_hit', 'state', 'scheduled_time', 'completion_time',
        'data', '_is_read_completed', '_is_read_failed', '_is_read_pending',
        'estimated_cycles'
    )

    def __init__(
        self,
        addr: int,
        length: int,
        is_read: bool,
        qos: int = 8,
        burst_length: int = 32,
        request_id: int = 0,
        arrival_time: float = 0.0,
        stack_id: int = 0,
        channel_id: int = 0,
        pseudo_channel_id: int = 0,
        bank_group_id: int = 0,
        bank_id: int = 0,
        row_id: int = 0,
        col_id: int = 0,
        row_hit: bool = False,
        state: RequestState = RequestState.PENDING,
        scheduled_time: float = 0.0,
        completion_time: float = 0.0,
        data: Optional[bytes] = None,
        estimated_cycles: float = 0.0,
    ):
        """Initialize HBM Request"""
        self.addr = addr
        self.length = length
        self.is_read = is_read
        self.qos = qos
        self.burst_length = burst_length
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.stack_id = stack_id
        self.channel_id = channel_id
        self.pseudo_channel_id = pseudo_channel_id
        self.bank_group_id = bank_group_id
        self.bank_id = bank_id
        self.row_id = row_id
        self.col_id = col_id
        self.row_hit = row_hit
        self.state = state
        self.scheduled_time = scheduled_time
        self.completion_time = completion_time
        self.data = data
        self.estimated_cycles = estimated_cycles

        # Generate unique request ID using module-level counter
        global _HBMRequest_next_id
        if self.request_id == 0:
            _HBMRequest_next_id += 1
            self.request_id = _HBMRequest_next_id

        # Initialize cached flags
        self._is_read_completed = self.state == RequestState.COMPLETED
        self._is_read_failed = self.state == RequestState.FAILED
        self._is_read_pending = self.state == RequestState.PENDING

    def set_arrival_time(self, cycle: float):
        """Set arrival time (simulation cycle)"""
        self.arrival_time = cycle

    def get_latency_cycles(self) -> float:
        """Calculate latency (cycles)"""
        if self.completion_time > 0 and self.arrival_time > 0:
            return self.completion_time - self.arrival_time
        return 0.0

    @property
    def latency(self) -> float:
        """Calculate request latency (seconds)"""
        if self.completion_time > 0:
            return self.completion_time - self.arrival_time
        return 0.0

    @property
    def is_completed(self) -> bool:
        """Check if request is completed"""
        return self._is_read_completed

    @property
    def is_failed(self) -> bool:
        """Check if request failed"""
        return self._is_read_failed

    @property
    def is_pending(self) -> bool:
        """Check if request is pending"""
        return self._is_read_pending

    def mark_scheduled(self, timestamp: float):
        """Mark request as scheduled"""
        self.state = RequestState.SCHEDULED
        self.scheduled_time = timestamp
        self._update_state_flags()

    def mark_in_progress(self):
        """Mark request as in progress"""
        self.state = RequestState.IN_PROGRESS
        self._update_state_flags()

    def mark_completed(self, timestamp: float):
        """Mark request as completed"""
        self.state = RequestState.COMPLETED
        self.completion_time = timestamp
        self._is_read_completed = True
        self._update_state_flags()

    def mark_failed(self):
        """Mark request as failed"""
        self.state = RequestState.FAILED
        self._is_read_failed = True
        self._update_state_flags()

    def _update_state_flags(self):
        """Update cached state flags"""
        self._is_read_completed = self.state == RequestState.COMPLETED
        self._is_read_failed = self.state == RequestState.FAILED
        self._is_read_pending = self.state == RequestState.PENDING

    def set_write_data(self, data: bytes):
        """Set write data

        Args:
            data: Data to write
        """
        if self.is_read:
            raise ValueError("Cannot set write data on a read request")
        self.data = data

    def get_write_data(self) -> Optional[bytes]:
        """Get write data

        Returns:
            Write data (bytes) or None
        """
        return self.data

    def __repr__(self) -> str:
        op = "READ" if self.is_read else "WRITE"
        return (f"HBMRequest(id={self.request_id}, {op}, "
                f"addr=0x{self.addr:016x}, len={self.length}, "
                f"qos={self.qos}, state={self.state.name})")


class HBMResponse:
    """HBM Response - Optimized Version

    Uses __slots__ to reduce memory footprint.

    Attributes:
        request_id: Associated request ID
        status: Status ("OK", "SLVERR", "DECERR")
        latency: Response latency (nanoseconds)
        channel_id: Response channel ID (HBM4 specific)
        bank_id: Response bank ID
        data: Read data (for read requests)
    """
    __slots__ = ('request_id', 'status', 'latency', 'channel_id', 'bank_id', 'data')

    def __init__(
        self,
        request_id: int,
        status: str = "OK",
        latency: float = 0.0,
        channel_id: int = 0,
        bank_id: int = 0,
        data: Optional[bytes] = None,
    ):
        self.request_id = request_id
        self.status = status
        self.latency = latency
        self.channel_id = channel_id
        self.bank_id = bank_id
        self.data = data

    @property
    def is_success(self) -> bool:
        """Check if response is successful"""
        return self.status == "OK"

    def __repr__(self) -> str:
        return f"HBMResponse(id={self.request_id}, status={self.status}, latency={self.latency:.2f}ns)"


# Batch request type for efficient processing
class RequestBatch:
    """Batch of requests for efficient processing"""
    __slots__ = ('requests', 'size', 'read_count', 'write_count')

    def __init__(self, requests: list):
        self.requests = requests
        self.size = len(requests)
        self.read_count = sum(1 for r in requests if r.is_read)
        self.write_count = len(requests) - self.read_count

    @classmethod
    def from_list(cls, requests: list) -> 'RequestBatch':
        """Create batch from request list"""
        return cls(requests=requests)


# Pool for reusing request objects (reduces GC pressure)
class HBMRequestPool:
    """Object pool for HBMRequest to reduce allocation overhead"""

    __slots__ = ('_pool', '_max_size', '_allocated')

    def __init__(self, max_size: int = 1024):
        self._pool = []
        self._max_size = max_size
        self._allocated = 0

    def acquire(self, addr: int, length: int, is_read: bool, **kwargs) -> HBMRequest:
        """Acquire a request from pool or create new"""
        if self._pool:
            req = self._pool.pop()
            # Reinitialize fields
            req.addr = addr
            req.length = length
            req.is_read = is_read
            for k, v in kwargs.items():
                if hasattr(req, k):
                    setattr(req, k, v)
            req._update_state_flags()
            return req
        else:
            self._allocated += 1
            return HBMRequest(addr=addr, length=length, is_read=is_read, **kwargs)

    def release(self, req: HBMRequest):
        """Return request to pool"""
        if len(self._pool) < self._max_size:
            # Reset fields to defaults
            req.state = RequestState.PENDING
            req.completion_time = 0.0
            req.scheduled_time = 0.0
            req.row_hit = False
            req.data = None
            self._pool.append(req)

    def clear(self):
        """Clear pool"""
        self._pool.clear()

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def total_allocated(self) -> int:
        return self._allocated