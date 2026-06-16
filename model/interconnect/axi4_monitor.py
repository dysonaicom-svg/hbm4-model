"""
AXI4 Protocol Monitor

This module provides monitoring and verification for AXI4 interfaces:
- Protocol compliance checker
- Transaction logger
- Performance metrics
- Error detection
- Timing analysis

Features:
- Validates AXI4 protocol rules
- Tracks all transactions and beats
- Computes bandwidth and latency metrics
- Detects protocol violations
- Generates detailed transaction logs

Based on:
- ARM AMBA AXI4 Protocol Specification
- JEDEC JESD270-4A HBM4 specification

Usage:
    >>> from model.interconnect.axi4_monitor import AXI4Monitor, ProtocolViolation
    >>> monitor = AXI4Monitor()
    >>> 
    >>> # Connect to bridge signals
    >>> monitor.connect(bridge.signals)
    >>> 
    >>> # Run simulation cycles
    >>> for cycle in range(1000):
    ...     bridge.tick()
    ...     monitor.tick()
    >>> 
    >>> # Get report
    >>> report = monitor.get_report()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from enum import IntEnum, auto
from collections import defaultdict
from datetime import datetime
import logging
import uuid

logger = logging.getLogger('hbm4.axi4_monitor')
logger.setLevel(logging.WARNING)


# ============================================================================
# Protocol Violations
# ============================================================================

class ViolationType(IntEnum):
    """Types of protocol violations"""
    # AR channel violations
    AR_VALID_WITHOUT_READY = 1
    AR_ADDRESS_ALIGNMENT = 2
    AR_BURST_LENGTH = 3
    AR_BURST_TYPE = 4
    
    # AW channel violations
    AW_VALID_WITHOUT_READY = 10
    AW_ADDRESS_ALIGNMENT = 11
    AW_BURST_LENGTH = 12
    AW_BURST_TYPE = 13
    
    # W channel violations
    W_NO_LAST = 20
    W_EXTRA_LAST = 21
    W_STROBE_INVALID = 22
    W_DATA_WITHOUT_READY = 23
    
    # R channel violations
    R_LAST_MISMATCH = 30
    R_VALID_WITHOUT_READY = 31
    
    # B channel violations
    B_VALID_WITHOUT_READY = 40
    
    # General violations
    OUTSTANDING_OVERFLOW = 50
    ID_CONFLICT = 51
    ADDRESS_OUT_OF_RANGE = 52
    TIMEOUT = 53


@dataclass
class ProtocolViolation:
    """Record of a protocol violation"""
    violation_type: ViolationType
    cycle: int
    channel: str  # AR, AW, W, R, B
    details: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO
    
    def __repr__(self) -> str:
        return (f"[{self.severity}] Cycle {self.cycle}: {self.channel} - "
                f"{self.violation_type.name}: {self.details}")


@dataclass
class TransactionLogEntry:
    """Log entry for a transaction"""
    transaction_id: str
    txn_type: str  # READ, WRITE
    addr: int
    length: int
    size: int
    burst: int
    id: int
    qos: int
    
    # Timing
    submission_cycle: int = 0
    ar_issued_cycle: Optional[int] = None
    first_data_cycle: Optional[int] = None
    last_data_cycle: Optional[int] = None
    completion_cycle: Optional[int] = None
    
    # Status
    status: str = "PENDING"
    response: int = 0
    error_message: Optional[str] = None
    
    # Beats
    r_beats: int = 0
    w_beats: int = 0
    
    @property
    def latency(self) -> int:
        """Total latency"""
        if self.completion_cycle is not None:
            return self.completion_cycle - self.submission_cycle
        return -1
    
    @property
    def address_phase_latency(self) -> int:
        """Address phase latency"""
        if self.ar_issued_cycle is not None:
            return self.ar_issued_cycle - self.submission_cycle
        return -1
    
    @property
    def data_phase_latency(self) -> int:
        """Data phase latency"""
        if self.first_data_cycle is not None and self.ar_issued_cycle is not None:
            return self.first_data_cycle - self.ar_issued_cycle
        return -1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'transaction_id': self.transaction_id,
            'type': self.txn_type,
            'addr': f"0x{self.addr:x}",
            'length': self.length,
            'size': self.size,
            'burst': self.burst,
            'id': self.id,
            'qos': self.qos,
            'submission_cycle': self.submission_cycle,
            'ar_issued_cycle': self.ar_issued_cycle,
            'first_data_cycle': self.first_data_cycle,
            'last_data_cycle': self.last_data_cycle,
            'completion_cycle': self.completion_cycle,
            'latency': self.latency,
            'status': self.status,
            'response': self.response,
        }


# ============================================================================
# Performance Metrics
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for AXI4 interface"""
    
    # Cycle counts
    total_cycles: int = 0
    active_cycles_ar: int = 0
    active_cycles_aw: int = 0
    active_cycles_w: int = 0
    active_cycles_r: int = 0
    active_cycles_b: int = 0
    
    # Transaction counts
    ar_transactions: int = 0
    aw_transactions: int = 0
    r_beats: int = 0
    w_beats: int = 0
    b_responses: int = 0
    
    # Latency statistics
    read_latency_sum: int = 0
    read_latency_count: int = 0
    write_latency_sum: int = 0
    write_latency_count: int = 0
    
    # Bandwidth (bytes per cycle)
    read_bytes_total: int = 0
    write_bytes_total: int = 0
    
    # Outstanding transaction tracking
    max_outstanding_ar: int = 0
    max_outstanding_aw: int = 0
    max_outstanding_r: int = 0
    max_outstanding_w: int = 0
    
    @property
    def average_read_latency(self) -> float:
        if self.read_latency_count == 0:
            return 0.0
        return self.read_latency_sum / self.read_latency_count
    
    @property
    def average_write_latency(self) -> float:
        if self.write_latency_count == 0:
            return 0.0
        return self.write_latency_sum / self.write_latency_count
    
    @property
    def read_bandwidth_bytes_per_cycle(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.read_bytes_total / self.total_cycles
    
    @property
    def write_bandwidth_bytes_per_cycle(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.write_bytes_total / self.total_cycles
    
    @property
    def ar_utilization(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.active_cycles_ar / self.total_cycles
    
    @property
    def r_utilization(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.active_cycles_r / self.total_cycles
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'total_cycles': self.total_cycles,
            'ar_utilization': f"{self.ar_utilization * 100:.2f}%",
            'aw_utilization': f"{self.active_cycles_aw / max(1, self.total_cycles) * 100:.2f}%",
            'r_utilization': f"{self.r_utilization * 100:.2f}%",
            'w_utilization': f"{self.active_cycles_w / max(1, self.total_cycles) * 100:.2f}%",
            'b_utilization': f"{self.active_cycles_b / max(1, self.total_cycles) * 100:.2f}%",
            'ar_transactions': self.ar_transactions,
            'aw_transactions': self.aw_transactions,
            'r_beats': self.r_beats,
            'w_beats': self.w_beats,
            'b_responses': self.b_responses,
            'avg_read_latency': f"{self.average_read_latency:.2f}",
            'avg_write_latency': f"{self.average_write_latency:.2f}",
            'read_bandwidth_Bpc': f"{self.read_bandwidth_bytes_per_cycle:.2f}",
            'write_bandwidth_Bpc': f"{self.write_bandwidth_bytes_per_cycle:.2f}",
            'max_outstanding_ar': self.max_outstanding_ar,
            'max_outstanding_aw': self.max_outstanding_aw,
        }


# ============================================================================
# AXI4 Monitor
# ============================================================================

class AXI4Monitor:
    """
    AXI4 Protocol Monitor
    
    Monitors AXI4 interface for:
    - Protocol compliance
    - Transaction tracking
    - Performance metrics
    - Error detection
    
    Usage:
        >>> monitor = AXI4Monitor(strict_protocol=True)
        >>> 
        >>> # In simulation loop
        >>> for cycle in range(1000):
        ...     # Update signals
        ...     monitor.update_signals(signals)
        ...     monitor.tick()
        >>> 
        >>> # Get results
        >>> violations = monitor.get_violations()
        >>> metrics = monitor.get_metrics()
        >>> report = monitor.get_report()
    """
    
    def __init__(
        self,
        strict_protocol: bool = True,
        enable_transaction_log: bool = True,
        max_log_entries: int = 10000,
        check_alignment: bool = True,
        check_burst_length: bool = True,
        check_timeout_cycles: int = 10000,
    ):
        """
        Initialize monitor
        
        Args:
            strict_protocol: Treat warnings as errors
            enable_transaction_log: Enable detailed transaction logging
            max_log_entries: Maximum log entries to keep
            check_alignment: Check address alignment
            check_burst_length: Check burst length limits
            check_timeout_cycles: Cycles before declaring timeout
        """
        self.strict_protocol = strict_protocol
        self.enable_transaction_log = enable_transaction_log
        self.max_log_entries = max_log_entries
        self.check_alignment = check_alignment
        self.check_burst_length = check_burst_length
        self.check_timeout_cycles = check_timeout_cycles
        
        self._cycle: int = 0
        
        # Violations
        self._violations: List[ProtocolViolation] = []
        
        # Transaction log
        self._transaction_log: Dict[str, TransactionLogEntry] = {}
        self._pending_read_txns: Dict[int, str] = {}
        self._pending_write_txns: Dict[int, str] = {}
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Outstanding tracking
        self._outstanding_ar: Set[int] = set()
        self._outstanding_aw: Set[int] = set()
        self._outstanding_r_beats: Dict[int, int] = defaultdict(int)
        self._outstanding_w_beats: Dict[int, int] = defaultdict(int)
        
        # Signal state for edge detection
        self._prev_signals = {}
        
        # Configuration
        self._data_width = 512  # Default
        self._addr_width = 64
        
        logger.info(f"AXI4Monitor initialized: strict={strict_protocol}")
    
    def set_widths(self, data_width: int, addr_width: int) -> None:
        """Set interface widths for validation"""
        self._data_width = data_width
        self._addr_width = addr_width
    
    def connect_signals(self, signals) -> None:
        """Connect to AXI4Signals object"""
        self._signals = signals
    
    def tick(self) -> None:
        """Process one clock cycle"""
        self._cycle += 1
        self.metrics.total_cycles = self._cycle
        
        if hasattr(self, '_signals'):
            self._process_cycle(self._signals)
        
        # Check for timeouts
        self._check_timeouts()
        
        # Update metrics
        self._update_metrics()
    
    def _process_cycle(self, signals) -> None:
        """Process signals for one cycle"""
        # AR channel
        if signals.arvalid:
            if signals.arready:
                self._handle_ar_valid()
            else:
                self.metrics.active_cycles_ar += 1
                self._check_ar_validity(signals)
        
        # AW channel
        if signals.awvalid:
            if signals.awready:
                self._handle_aw_valid()
            else:
                self.metrics.active_cycles_aw += 1
                self._check_aw_validity(signals)
        
        # W channel
        if signals.wvalid:
            if signals.wready:
                self.metrics.active_cycles_w += 1
                self._handle_w_beat(signals)
            else:
                self.metrics.active_cycles_w += 1
        
        # R channel
        if signals.rvalid:
            if signals.rready:
                self.metrics.active_cycles_r += 1
                self._handle_r_beat(signals)
            else:
                self.metrics.active_cycles_r += 1
        
        # B channel
        if signals.bvalid:
            if signals.bready:
                self.metrics.active_cycles_b += 1
                self._handle_b_response(signals)
            else:
                self.metrics.active_cycles_b += 1
    
    def _check_ar_validity(self, signals) -> None:
        """Check AR channel protocol validity"""
        # Check alignment
        if self.check_alignment:
            size = 1 << signals.arsize
            if signals.araddr & (size - 1):
                self._add_violation(
                    ViolationType.AR_ADDRESS_ALIGNMENT,
                    "AR",
                    f"Address 0x{signals.araddr:x} not aligned to size {size}",
                    "WARNING"
                )
        
        # Check burst length (AXI4 allows up to 256)
        if self.check_burst_length:
            if signals.arlen > 255:
                self._add_violation(
                    ViolationType.AR_BURST_LENGTH,
                    "AR",
                    f"Burst length {signals.arlen + 1} exceeds AXI4 maximum of 256",
                    "ERROR"
                )
    
    def _check_aw_validity(self, signals) -> None:
        """Check AW channel protocol validity"""
        if self.check_alignment:
            size = 1 << signals.awsize
            if signals.awaddr & (size - 1):
                self._add_violation(
                    ViolationType.AW_ADDRESS_ALIGNMENT,
                    "AW",
                    f"Address 0x{signals.awaddr:x} not aligned to size {size}",
                    "WARNING"
                )
        
        if self.check_burst_length:
            if signals.awlen > 255:
                self._add_violation(
                    ViolationType.AW_BURST_LENGTH,
                    "AW",
                    f"Burst length {signals.awlen + 1} exceeds AXI4 maximum of 256",
                    "ERROR"
                )
    
    def _handle_ar_valid(self) -> None:
        """Handle AR transaction valid"""
        if hasattr(self, '_signals'):
            signals = self._signals
            txn_id = f"ar_{signals.arid}_{self._cycle}"
            
            # Create log entry
            entry = TransactionLogEntry(
                transaction_id=txn_id,
                txn_type="READ",
                addr=signals.araddr,
                length=signals.arlen,
                size=signals.arsize,
                burst=signals.arburst,
                id=signals.arid,
                qos=signals.arqos,
                submission_cycle=self._cycle,
                ar_issued_cycle=self._cycle,
            )
            
            self._transaction_log[txn_id] = entry
            self._pending_read_txns[signals.arid] = txn_id
            self._outstanding_ar.add(signals.arid)
            
            self.metrics.ar_transactions += 1
            self.metrics.max_outstanding_ar = max(
                self.metrics.max_outstanding_ar,
                len(self._outstanding_ar)
            )
    
    def _handle_aw_valid(self) -> None:
        """Handle AW transaction valid"""
        if hasattr(self, '_signals'):
            signals = self._signals
            txn_id = f"aw_{signals.awid}_{self._cycle}"
            
            entry = TransactionLogEntry(
                transaction_id=txn_id,
                txn_type="WRITE",
                addr=signals.awaddr,
                length=signals.awlen,
                size=signals.awsize,
                burst=signals.awburst,
                id=signals.awid,
                qos=signals.awqos,
                submission_cycle=self._cycle,
            )
            
            self._transaction_log[txn_id] = entry
            self._pending_write_txns[signals.awid] = txn_id
            self._outstanding_aw.add(signals.awid)
            
            self.metrics.aw_transactions += 1
            self.metrics.max_outstanding_aw = max(
                self.metrics.max_outstanding_aw,
                len(self._outstanding_aw)
            )
    
    def _handle_w_beat(self, signals) -> None:
        """Handle W beat"""
        txn_id = self._pending_write_txns.get(signals.wid)
        if txn_id and txn_id in self._transaction_log:
            entry = self._transaction_log[txn_id]
            entry.w_beats += 1
            
            # Track outstanding W beats
            self._outstanding_w_beats[signals.wid] += 1
            self.metrics.w_beats += 1
            self.metrics.max_outstanding_w = max(
                self.metrics.max_outstanding_w,
                sum(self._outstanding_w_beats.values())
            )
            
            # Check WLAST
            if signals.wlast:
                self.metrics.w_beats -= 1  # Don't count WLAST twice
                self._outstanding_w_beats[signals.wid] = 0
    
    def _handle_r_beat(self, signals) -> None:
        """Handle R beat"""
        txn_id = self._pending_read_txns.get(signals.rid)
        if txn_id and txn_id in self._transaction_log:
            entry = self._transaction_log[txn_id]
            entry.r_beats += 1
            
            if entry.first_data_cycle is None:
                entry.first_data_cycle = self._cycle
            
            self._outstanding_r_beats[signals.rid] += 1
            self.metrics.r_beats += 1
            self.metrics.max_outstanding_r = max(
                self.metrics.max_outstanding_r,
                sum(self._outstanding_r_beats.values())
            )
            
            # Check RLAST
            if signals.rlast:
                entry.last_data_cycle = self._cycle
                entry.completion_cycle = self._cycle
                entry.status = "COMPLETED"
                entry.response = signals.rresp
                
                self._outstanding_ar.discard(signals.rid)
                if signals.rid in self._pending_read_txns:
                    del self._pending_read_txns[signals.rid]
                
                # Update latency stats
                latency = entry.latency
                self.metrics.read_latency_sum += latency
                self.metrics.read_latency_count += 1
                self.metrics.read_bytes_total += entry.length * (1 << entry.size)
    
    def _handle_b_response(self, signals) -> None:
        """Handle B response"""
        txn_id = self._pending_write_txns.get(signals.bid)
        if txn_id and txn_id in self._transaction_log:
            entry = self._transaction_log[txn_id]
            entry.completion_cycle = self._cycle
            entry.status = "COMPLETED"
            entry.response = signals.bresp
            
            self._outstanding_aw.discard(signals.bid)
            if signals.bid in self._pending_write_txns:
                del self._pending_write_txns[signals.bid]
            
            # Update latency stats
            latency = entry.latency
            self.metrics.write_latency_sum += latency
            self.metrics.write_latency_count += 1
            self.metrics.write_bytes_total += entry.length * (1 << entry.size)
            self.metrics.b_responses += 1
    
    def _add_violation(
        self,
        violation_type: ViolationType,
        channel: str,
        details: str,
        severity: str = "ERROR",
    ) -> None:
        """Add a protocol violation"""
        violation = ProtocolViolation(
            violation_type=violation_type,
            cycle=self._cycle,
            channel=channel,
            details=details,
            severity=severity,
        )
        
        self._violations.append(violation)
        
        if severity == "ERROR" or self.strict_protocol:
            logger.error(f"{violation}")
        else:
            logger.warning(f"{violation}")
    
    def _check_timeouts(self) -> None:
        """Check for transaction timeouts"""
        timeout_threshold = self._cycle - self.check_timeout_cycles
        
        for txn_id, entry in list(self._transaction_log.items()):
            if entry.status == "PENDING" and entry.submission_cycle < timeout_threshold:
                entry.status = "TIMEOUT"
                entry.error_message = f"Timeout after {self.check_timeout_cycles} cycles"
                self._add_violation(
                    ViolationType.TIMEOUT,
                    entry.txn_type[0] + "R",  # AR or AW
                    f"Transaction {txn_id} timed out",
                    "ERROR"
                )
    
    def _update_metrics(self) -> None:
        """Update performance metrics"""
        # Metrics are updated during cycle processing
        pass
    
    # ========================================================================
    # Public Interface
    # ========================================================================
    
    def get_violations(self, severity: Optional[str] = None) -> List[ProtocolViolation]:
        """Get all protocol violations
        
        Args:
            severity: Filter by severity ("ERROR", "WARNING", "INFO")
            
        Returns:
            List of violations
        """
        if severity:
            return [v for v in self._violations if v.severity == severity]
        return self._violations.copy()
    
    def get_error_count(self) -> int:
        """Get count of ERROR violations"""
        return len([v for v in self._violations if v.severity == "ERROR"])
    
    def get_warning_count(self) -> int:
        """Get count of WARNING violations"""
        return len([v for v in self._violations if v.severity == "WARNING"])
    
    def get_transaction_log(self) -> List[Dict]:
        """Get transaction log as list of dictionaries"""
        return [entry.to_dict() for entry in self._transaction_log.values()]
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    def get_report(self) -> Dict:
        """Get comprehensive report"""
        return {
            'cycle': self._cycle,
            'metrics': self.metrics.to_dict(),
            'violations': {
                'total': len(self._violations),
                'errors': self.get_error_count(),
                'warnings': self.get_warning_count(),
                'list': [str(v) for v in self._violations],
            },
            'transactions': {
                'total': len(self._transaction_log),
                'completed': len([e for e in self._transaction_log.values() if e.status == "COMPLETED"]),
                'pending': len([e for e in self._transaction_log.values() if e.status == "PENDING"]),
                'timeout': len([e for e in self._transaction_log.values() if e.status == "TIMEOUT"]),
            },
            'outstanding': {
                'ar': len(self._outstanding_ar),
                'aw': len(self._outstanding_aw),
                'r_beats': sum(self._outstanding_r_beats.values()),
                'w_beats': sum(self._outstanding_w_beats.values()),
            },
        }
    
    def is_compliant(self) -> bool:
        """Check if interface is protocol compliant"""
        return self.get_error_count() == 0
    
    def reset(self) -> None:
        """Reset monitor state"""
        self._cycle = 0
        self._violations.clear()
        self._transaction_log.clear()
        self._pending_read_txns.clear()
        self._pending_write_txns.clear()
        self.metrics = PerformanceMetrics()
        self._outstanding_ar.clear()
        self._outstanding_aw.clear()
        self._outstanding_r_beats.clear()
        self._outstanding_w_beats.clear()
        
        logger.info("AXI4Monitor reset")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_axi4_monitor(
    strict_protocol: bool = True,
    enable_log: bool = True,
) -> AXI4Monitor:
    """Create AXI4 monitor with common configuration"""
    return AXI4Monitor(
        strict_protocol=strict_protocol,
        enable_transaction_log=enable_log,
    )


def analyze_axi4_log(log_entries: List[Dict]) -> Dict:
    """Analyze transaction log for patterns"""
    if not log_entries:
        return {}
    
    read_latencies = [e['latency'] for e in log_entries if e['type'] == 'READ' and e['latency'] > 0]
    write_latencies = [e['latency'] for e in log_entries if e['type'] == 'WRITE' and e['latency'] > 0]
    
    return {
        'total_transactions': len(log_entries),
        'read_count': len([e for e in log_entries if e['type'] == 'READ']),
        'write_count': len([e for e in log_entries if e['type'] == 'WRITE']),
        'avg_read_latency': sum(read_latencies) / max(1, len(read_latencies)),
        'avg_write_latency': sum(write_latencies) / max(1, len(write_latencies)),
        'min_read_latency': min(read_latencies) if read_latencies else 0,
        'max_read_latency': max(read_latencies) if read_latencies else 0,
        'min_write_latency': min(write_latencies) if write_latencies else 0,
        'max_write_latency': max(write_latencies) if write_latencies else 0,
    }


if __name__ == "__main__":
    print("Testing AXI4 Monitor...")
    
    # Create monitor
    monitor = create_axi4_monitor(strict_protocol=False)
    
    # Create bridge and connect
    from model.interconnect.axi4_bridge import create_axi4_bridge
    bridge = create_axi4_bridge(max_pending=16)
    monitor.connect_signals(bridge.signals)
    
    # Submit transactions
    read_id = bridge.submit_read(addr=0x1000, length=7, qos=8)
    write_id = bridge.submit_write(addr=0x2000, data=[0xDEADBEEF] * 8, length=7, qos=4)
    
    print(f"Submitted read: {read_id}, write: {write_id}")
    
    # Simulate cycles
    for cycle in range(100):
        bridge.tick()
        monitor.tick()
    
    # Get report
    report = monitor.get_report()
    print(f"\n=== AXI4 Monitor Report ===")
    print(f"Total cycles: {report['cycle']}")
    print(f"Violations: {report['violations']['errors']} errors, "
          f"{report['violations']['warnings']} warnings")
    print(f"Transactions: {report['transactions']['total']} total, "
          f"{report['transactions']['completed']} completed")
    print(f"Metrics: {report['metrics']}")