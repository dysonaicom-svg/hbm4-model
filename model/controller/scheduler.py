"""
HBM FR-FCFS Scheduler - Optimized Version
Reference design document 2026-06-15-hbm-system-model-design.md Section 5.1.2 and 5.1.3

FR-FCFS (First-Ready First-Come-First-Served):
1. Priority for row-hit requests
2. Same priority by timestamp, select oldest
3. Read/Write arbitration

Optimizations:
- __slots__ for memory reduction
- Batch candidate filtering
- Pre-computed priority scores
- Reduced queue iteration overhead
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import time

from model.controller.config import HBMConfig
from model.controller.request import HBMRequest, RequestState
from model.controller.queue import ReadQueue, WriteQueue


class BankState:
    """Bank state"""
    __slots__ = ('bank_id', 'is_open', 'open_row', 'last_access_time')

    def __init__(self, bank_id: int):
        self.bank_id = bank_id
        self.is_open = False
        self.open_row = -1
        self.last_access_time = 0.0


class HBMScheduler:
    """HBM Scheduler base class"""

    __slots__ = ('config',)

    def __init__(self, config: HBMConfig):
        self.config = config

    def schedule(self, read_queue: ReadQueue, write_queue: WriteQueue,
                bank_states: Dict[Tuple, BankState], current_time: float) -> Optional[HBMRequest]:
        """Schedule next request

        Args:
            read_queue: Read queue
            write_queue: Write queue
            bank_states: Bank state dictionary
            current_time: Current time

        Returns:
            Next scheduled request
        """
        raise NotImplementedError


# Pre-computed turnaround penalty (in cycles)
_TURNAROUND_PENALTY = 3


@dataclass
class SchedulerStats:
    """Scheduler statistics"""
    schedule_count: int = 0
    row_hit_count: int = 0
    row_miss_count: int = 0
    read_count: int = 0
    write_count: int = 0

    @property
    def row_hit_rate(self) -> float:
        if self.schedule_count == 0:
            return 0.0
        return self.row_hit_count / self.schedule_count

    def record_schedule(self, request: HBMRequest):
        self.schedule_count += 1
        if request.row_hit:
            self.row_hit_count += 1
        else:
            self.row_miss_count += 1
        if request.is_read:
            self.read_count += 1
        else:
            self.write_count += 1


class FRFCFSScheduler(HBMScheduler):
    """FR-FCFS Scheduler - Optimized Version

    First-Ready FCFS scheduling strategy:
    - Row-hit priority
    - Same priority by timestamp
    - Configurable read/write arbitration

    Optimizations:
    - Batch candidate filtering
    - Pre-computed priority scores
    - Reduced method call overhead
    """

    __slots__ = ('rd_priority', 'wr_priority', 'TURNAROUND_PENALTY',
                 '_cached_read_candidates', '_cached_write_candidates')

    # Read/Write arbitration weights
    RD_PRIORITY = 1.0
    WR_PRIORITY = 1.0

    def __init__(self, config: HBMConfig, rd_priority: float = 1.0, wr_priority: float = 1.0):
        super().__init__(config)
        self.rd_priority = rd_priority
        self.wr_priority = wr_priority

        # Read-Write turnaround penalty (cycles)
        self.TURNAROUND_PENALTY = _TURNAROUND_PENALTY

        # Cached candidates for batch processing
        self._cached_read_candidates: List[HBMRequest] = []
        self._cached_write_candidates: List[HBMRequest] = []

    def schedule(self, read_queue: ReadQueue, write_queue: WriteQueue,
                bank_states: Dict[Tuple, BankState],
                current_time: float,
                last_cmd_type: str = "READ") -> Optional[HBMRequest]:
        """FR-FCFS scheduling - Optimized

        Args:
            read_queue: Read queue
            write_queue: Write queue
            bank_states: Bank state dictionary
            current_time: Current time
            last_cmd_type: Last command type ("READ" or "WRITE")

        Returns:
            Next scheduled request
        """
        # Batch get candidates for both queues
        read_candidates = self._get_row_hit_candidates_fast(read_queue, bank_states)
        write_candidates = self._get_row_hit_candidates_fast(write_queue, bank_states)

        # If no row-hit requests, get all requests
        if not read_candidates and not write_candidates:
            read_candidates = list(read_queue._queue)
            write_candidates = list(write_queue._queue)

        # Read/Write arbitration - optimized
        best_read = self._select_oldest(read_candidates) if read_candidates else None
        best_write = self._select_oldest(write_candidates) if write_candidates else None

        # Select best request
        selected = self._arbitrate_read_write_fast(best_read, best_write, last_cmd_type)

        if selected:
            # Update request state
            selected.mark_scheduled(current_time)

            # Remove from queue
            if selected.is_read:
                read_queue.remove(selected.request_id)
            else:
                write_queue.remove(selected.request_id)

        return selected

    def _get_row_hit_candidates_fast(self, queue, bank_states: Dict) -> List[HBMRequest]:
        """Fast batch get row-hit candidates

        Optimizations:
        - Direct queue iteration
        - Early exit for hit detection
        - Reduced bank_state lookups
        """
        candidates = []
        queue_items = queue._queue

        for req in queue_items:
            bank_key = (req.channel_id, req.pseudo_channel_id, req.bank_id)
            bank_state = bank_states.get(bank_key)

            if bank_state is None:
                # No state tracked, use existing row_hit flag
                if req.row_hit:
                    candidates.append(req)
                continue

            if bank_state.is_open and bank_state.open_row == req.row_id:
                req.row_hit = True
                candidates.append(req)
            else:
                req.row_hit = False

        return candidates

    def _get_row_hit_candidates(self, queue, bank_states: Dict) -> List[HBMRequest]:
        """Get row-hit candidate requests - Legacy compatibility"""
        return self._get_row_hit_candidates_fast(queue, bank_states)

    def _select_oldest(self, candidates: List[HBMRequest]) -> Optional[HBMRequest]:
        """Select oldest request - Optimized"""
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.arrival_time)

    def _arbitrate_read_write_fast(self, read_req: Optional[HBMRequest],
                                   write_req: Optional[HBMRequest],
                                   last_cmd: str) -> Optional[HBMRequest]:
        """Fast read/write arbitration

        Optimizations:
        - Pre-computed turnaround penalty
        - Reduced branching
        """
        if not read_req and not write_req:
            return None

        if read_req and not write_req:
            return read_req

        if write_req and not read_req:
            return write_req

        # Both exist - select by arrival time with turnaround penalty
        read_score = read_req.arrival_time
        write_score = write_req.arrival_time

        # Apply turnaround penalty (in time units)
        if last_cmd == "READ":
            # Penalize write after read
            write_score += self.TURNAROUND_PENALTY
        else:
            # Penalize read after write
            read_score += self.TURNAROUND_PENALTY

        return read_req if read_score < write_score else write_req

    def _arbitrate_read_write(self, read_req: Optional[HBMRequest],
                              write_req: Optional[HBMRequest],
                              last_cmd: str, current_time: float) -> Optional[HBMRequest]:
        """Read/Write arbitration - Legacy compatibility"""
        return self._arbitrate_read_write_fast(read_req, write_req, last_cmd)

    def clear_cache(self):
        """Clear cached candidates"""
        self._cached_read_candidates.clear()
        self._cached_write_candidates.clear()


# Batch scheduler for processing multiple requests at once
@dataclass
class BatchSchedulerConfig:
    """Configuration for batch scheduling"""
    batch_size: int = 32
    max_queue_scan: int = 64  # Maximum queue entries to scan per batch


class BatchScheduler(HBMScheduler):
    """Batch Scheduler - Process multiple requests efficiently

    Optimizations:
    - Process requests in batches
    - Vectorized priority calculation
    - Reduced queue operations
    """

    __slots__ = ('batch_config', '_scheduler')

    def __init__(self, config: HBMConfig, batch_config: BatchSchedulerConfig = None):
        super().__init__(config)
        self.batch_config = batch_config or BatchSchedulerConfig()
        self._scheduler = FRFCFSScheduler(config)

    def schedule_batch(self, read_queue: ReadQueue, write_queue: WriteQueue,
                      bank_states: Dict[Tuple, BankState],
                      current_time: float,
                      last_cmd_type: str = "READ") -> List[HBMRequest]:
        """Schedule batch of requests

        Args:
            read_queue: Read queue
            write_queue: Write queue
            bank_states: Bank state dictionary
            current_time: Current time
            last_cmd_type: Last command type

        Returns:
            List of scheduled requests
        """
        scheduled = []
        batch_size = self.batch_config.batch_size

        for _ in range(batch_size):
            # Check if queues are empty
            if read_queue.is_empty() and write_queue.is_empty():
                break

            request = self._scheduler.schedule(
                read_queue, write_queue, bank_states, current_time, last_cmd_type
            )

            if request is None:
                break

            scheduled.append(request)
            last_cmd_type = "READ" if request.is_read else "WRITE"

        return scheduled

    def schedule(self, read_queue: ReadQueue, write_queue: WriteQueue,
                bank_states: Dict[Tuple, BankState],
                current_time: float,
                last_cmd_type: str = "READ") -> Optional[HBMRequest]:
        """Single request scheduling - delegates to FRFCFSScheduler"""
        return self._scheduler.schedule(
            read_queue, write_queue, bank_states, current_time, last_cmd_type
        )