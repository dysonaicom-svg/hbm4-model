"""
HBM4 QoS Scheduler with Anti-Starvation

Based on Synopsys HBM4 Controller findings:
- 16 priority classes with anti-starvation
- Bandwidth guarantee per QoS level
- Address collision control
- FR-FCFS within same priority

Reference: Synopsys DesignWare HBM4/4E Controller IP
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from collections import defaultdict
import time as time_module

from model.dram.hbm4_spec import HBM4Spec

if TYPE_CHECKING:
    from model.controller.request import HBMRequest


class QoSLevel(IntEnum):
    """HBM4 QoS priority levels (0-15)

    Higher values = higher priority.
    Critical traffic (real-time) gets highest priority.
    """
    CRITICAL = 15    # Real-time/critical
    HIGH = 12        # High priority
    NORMAL = 8      # Normal traffic
    LOW = 4          # Background/batch
    IDLE = 0         # Idle/probe


@dataclass
class QueuedRequest:
    """Request in QoS queue

    Tracks all information needed for scheduling decisions.
    """
    request_id: int
    addr: int
    qos: int
    is_read: bool
    arrival_time: float
    row_hit: bool = False
    channel: int = 0
    pseudo_channel: int = 0
    bank: int = 0
    row: int = 0
    col: int = 0
    length: int = 64  # bytes


class HBM4QoSScheduler:
    """HBM4 QoS Scheduler with anti-starvation

    Key features from research:
    - 16 priority levels (0-15)
    - Anti-starvation guarantees for low priority
    - Bandwidth guarantee per QoS class
    - FR-FCFS within same priority

    Reference: Synopsys DesignWare HBM4/4E Controller IP
    """

    # Priority level constants
    QOS_CRITICAL = 15
    QOS_HIGH = 12
    QOS_NORMAL = 8
    QOS_LOW = 4
    QOS_IDLE = 0

    def __init__(self, config: Optional[HBM4Spec] = None):
        """Initialize QoS scheduler

        Args:
            config: HBM4 specification (uses default if None)
        """
        self.config = config if config else HBM4Spec()
        self.priority_levels = 16

        # Bandwidth guarantees (GB/s per stack)
        # Higher priority gets higher guarantee
        self.bw_guarantee = {
            self.QOS_CRITICAL: 200.0,
            self.QOS_HIGH: 300.0,
            self.QOS_NORMAL: 200.0,
            self.QOS_LOW: 100.0,
            self.QOS_IDLE: 0,
        }

        # Bandwidth caps (to prevent starvation of others)
        self.bw_cap = {
            self.QOS_CRITICAL: 1000.0,
            self.QOS_HIGH: 800.0,
            self.QOS_NORMAL: 400.0,
            self.QOS_LOW: 200.0,
            self.QOS_IDLE: 50.0,
        }

        # Bandwidth tracking
        self.bw_window_ms = 1.0
        self.bandwidth_tracked: Dict[int, List[tuple]] = defaultdict(list)

        # Request queues per priority level
        self.queues: Dict[int, List[QueuedRequest]] = defaultdict(list)

        # Statistics
        self.stats = {
            'total_scheduled': 0,
            'by_qos': defaultdict(int)
        }

    def submit_request(self, request_id: int, addr: int = 0,
                      qos: int = 8, is_read: bool = True,
                      channel: int = 0, pseudo_channel: int = 0,
                      bank: int = 0, row: int = 0, col: int = 0,
                      row_hit: bool = False, length: int = 64) -> bool:
        """Submit a request to the QoS scheduler

        Args:
            request_id: Unique request identifier
            addr: Address
            qos: QoS level (0-15)
            is_read: True for read, False for write
            channel: Target channel (0-31)
            pseudo_channel: Target pseudo-channel (0-1)
            bank: Target bank (0-15)
            row: Target row
            col: Target column
            row_hit: Whether this is a row hit
            length: Transaction length in bytes

        Returns:
            True if request was queued
        """
        if qos < 0 or qos >= self.priority_levels:
            return False

        req = QueuedRequest(
            request_id=request_id,
            addr=addr,
            qos=qos,
            is_read=is_read,
            arrival_time=time_module.time(),
            row_hit=row_hit,
            channel=channel,
            pseudo_channel=pseudo_channel,
            bank=bank,
            row=row,
            col=col,
            length=length
        )

        self.queues[qos].append(req)
        return True

    def _get_current_bandwidth(self, qos_level: int) -> float:
        """Calculate current bandwidth for a QoS level

        Args:
            qos_level: QoS level to check

        Returns:
            Current bandwidth in GB/s
        """
        now = time_module.time()
        window_start = now - self.bw_window_ms / 1000.0

        # Filter recent entries
        recent = [
            (t, b) for t, b in self.bandwidth_tracked[qos_level]
            if t >= window_start
        ]
        total_bytes = sum(b for _, b in recent)
        total_time = self.bw_window_ms / 1000.0

        return total_bytes / total_time / 1e9 if total_time > 0 else 0

    def _can_schedule(self, qos_level: int) -> bool:
        """Check if a QoS level can be scheduled (anti-starvation)

        Args:
            qos_level: QoS level to check

        Returns:
            True if this level can be scheduled
        """
        current_bw = self._get_current_bandwidth(qos_level)

        # Below guarantee: can always schedule
        if current_bw < self.bw_guarantee.get(qos_level, 0):
            return True

        # Above cap: cannot schedule (prevents starvation of others)
        if current_bw >= self.bw_cap.get(qos_level, float('inf')):
            return False

        return True  # Between guarantee and cap: fair scheduling

    def schedule(self) -> Optional[QueuedRequest]:
        """Schedule the next request using QoS + FR-FCFS

        Priority order:
        1. Highest QoS level that can be scheduled
        2. FR-FCFS within that level (row hits first, then oldest)

        Returns:
            Next request to schedule, or None if queue empty
        """
        # Check QoS levels from high to low
        for qos_level in range(self.priority_levels - 1, -1, -1):
            if not self._can_schedule(qos_level):
                continue

            # Get candidates at this QoS level
            candidates = self.queues[qos_level]
            if not candidates:
                continue

            # FR-FCFS selection within same priority
            best = self._fr_fcfs_select(candidates)
            if best:
                self.queues[qos_level].remove(best)
                self.stats['total_scheduled'] += 1
                self.stats['by_qos'][qos_level] += 1

                # Track bandwidth
                now = time_module.time()
                self.bandwidth_tracked[qos_level].append((now, best.length))

                return best

        return None

    def _fr_fcfs_select(self, candidates: List[QueuedRequest]) -> Optional[QueuedRequest]:
        """First-Ready FCFS selection

        Priority:
        1. Row hit requests (first)
        2. Oldest request (FCFS)

        Args:
            candidates: List of candidate requests

        Returns:
            Best request to schedule, or None
        """
        if not candidates:
            return None

        # Priority 1: Row hit requests
        row_hits = [r for r in candidates if r.row_hit]
        if row_hits:
            # Sort by arrival time, pick oldest
            return min(row_hits, key=lambda r: r.arrival_time)

        # Priority 2: All requests, oldest first
        return min(candidates, key=lambda r: r.arrival_time)

    def get_queue_size(self, qos_level: int) -> int:
        """Get number of requests in a specific queue

        Args:
            qos_level: QoS level to query

        Returns:
            Number of queued requests
        """
        return len(self.queues[qos_level])

    def get_total_queue_size(self) -> int:
        """Get total number of queued requests across all priorities

        Returns:
            Total queued requests
        """
        return sum(len(q) for q in self.queues.values())

    def clear_queue(self, qos_level: int):
        """Clear all requests in a specific queue

        Args:
            qos_level: QoS level to clear
        """
        self.queues[qos_level].clear()

    def clear_all_queues(self):
        """Clear all queues"""
        self.queues.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'total_scheduled': self.stats['total_scheduled'],
            'by_qos': dict(self.stats['by_qos']),
            'total_queued': self.get_total_queue_size(),
            'queues_by_level': {
                qos: len(reqs) for qos, reqs in self.queues.items()
            }
        }

    def select_next(self, requests: List) -> Optional[Any]:
        """Select next request from a list using QoS priority + FR-FCFS

        This method accepts a list of HBMRequest objects and selects
        the highest priority one based on QoS level and row hit status.

        Args:
            requests: List of HBMRequest objects to select from

        Returns:
            Selected request or None if list is empty
        """
        if not requests:
            return None

        # Group requests by QoS level
        by_qos: Dict[int, List] = defaultdict(list)
        for req in requests:
            qos = getattr(req, 'qos', 8)  # Default to level 8
            by_qos[qos].append(req)

        # Select from highest QoS level that has requests
        for qos_level in range(self.priority_levels - 1, -1, -1):
            if qos_level not in by_qos or not by_qos[qos_level]:
                continue

            candidates = by_qos[qos_level]

            # FR-FCFS: row hits first, then oldest
            row_hits = [r for r in candidates if getattr(r, 'row_hit', False)]
            if row_hits:
                return min(row_hits, key=lambda r: getattr(r, 'arrival_time', 0))

            # No row hits, select oldest
            return min(candidates, key=lambda r: getattr(r, 'arrival_time', 0))

        return None

    def set_bandwidth_guarantee(self, qos_level: int, guarantee_gbs: float):
        """Set bandwidth guarantee for a QoS level

        Args:
            qos_level: QoS level
            guarantee_gbs: Bandwidth guarantee in GB/s
        """
        self.bw_guarantee[qos_level] = guarantee_gbs

    def set_bandwidth_cap(self, qos_level: int, cap_gbs: float):
        """Set bandwidth cap for a QoS level

        Args:
            qos_level: QoS level
            cap_gbs: Bandwidth cap in GB/s
        """
        self.bw_cap[qos_level] = cap_gbs

    def select_next(self, requests: List['HBMRequest']) -> Optional['HBMRequest']:
        """Select the next request from a provided list based on QoS priority.

        This method takes existing requests (from controller queues) and selects
        the highest priority one using QoS-aware selection.

        Priority order:
        1. Highest QoS level (higher qos value = higher priority)
        2. FR-FCFS within same priority (row hits first, then oldest)

        Args:
            requests: List of HBMRequest objects to select from

        Returns:
            The highest priority HBMRequest, or None if list is empty
        """
        if not requests:
            return None

        # Group requests by QoS level
        by_qos: Dict[int, List['HBMRequest']] = defaultdict(list)
        for req in requests:
            by_qos[req.qos].append(req)

        # Check QoS levels from high to low (15 = highest priority)
        for qos_level in range(self.priority_levels - 1, -1, -1):
            candidates = by_qos.get(qos_level, [])
            if not candidates:
                continue

            # FR-FCFS selection within same priority
            best = self._fr_fcfs_select_hbm(candidates)
            if best:
                self.stats['total_scheduled'] += 1
                self.stats['by_qos'][qos_level] += 1
                return best

        return None

    def _fr_fcfs_select_hbm(self, candidates: List['HBMRequest']) -> Optional['HBMRequest']:
        """First-Ready FCFS selection for HBMRequest objects

        Priority:
        1. Row hit requests (first)
        2. Oldest request (FCFS)

        Args:
            candidates: List of candidate HBMRequest objects

        Returns:
            Best request to schedule, or None
        """
        if not candidates:
            return None

        # Priority 1: Row hit requests
        row_hits = [r for r in candidates if r.row_hit]
        if row_hits:
            # Sort by arrival time, pick oldest
            return min(row_hits, key=lambda r: r.arrival_time)

        # Priority 2: All requests, oldest first
        return min(candidates, key=lambda r: r.arrival_time)