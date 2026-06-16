"""
HBM4 Controller Integration

Integrates all HBM4-specific modules into a complete controller model.

Key modules:
- HBM4AddressDecoder: 32-channel address decoding
- HBM4QoSScheduler: 16-level QoS scheduling
- HBM4RefreshScheduler: Per-bank and autonomous refresh
- HBM4ChannelModel: DRAM channel timing
- DFI5Interface: Controller-PHY interface

HBM4 Features:
- 32 independent channels (5-bit channel field)
- 2 pseudo-channels per channel (1-bit pseudo-channel field)
- 8 bank groups per pseudo-channel
- 16 banks per pseudo-channel
- 64K rows per bank
- 2048-bit I/O width
- 8 GT/s data rate (125 ps tCK)
- Lane repair support
- Per-bank-group timing
- DFI 5.0 protocol support

Based on:
- JEDEC JESD270-4A HBM4 specification
- Multi-agent research findings (2026-06-15)

Debug Logging:
    The controller uses Python logging with the 'hbm4.controller' logger.
    Enable debug logging to see detailed operation traces.
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import time
import uuid

from model.dram.hbm4_spec import HBM4Spec
from model.dram.dfi_interface import (
    DFI5Interface, DFICommand, DFILowPowerState,
    DFIRequest, DFIResponse as DFIPhyResponse
)
from model.controller.config import HBMConfig
from model.controller.request import HBMRequest, HBMResponse, RequestState
from model.controller.queue import ReadQueue, WriteQueue, QueueManager
from model.controller.hbm4_address_decoder import HBM4AddressDecoder
from model.controller.hbm4_qos_scheduler import HBM4QoSScheduler, QoSLevel
from model.controller.hbm4_refresh_scheduler import HBM4RefreshScheduler, RefreshMode
from model.controller.exceptions import QueueOverflowError
from model.dram.hbm4_channel_model import HBM4ChannelArray

# Configure debug logging for HBM4 controller
_logger = logging.getLogger('hbm4.controller')
_logger.setLevel(logging.WARNING)  # Default to WARNING, set to DEBUG for tracing


@dataclass
class HBM4ControllerStats:
    """Statistics for HBM4 Controller"""
    total_requests: int = 0
    read_requests: int = 0
    write_requests: int = 0
    row_hit_count: int = 0
    refresh_count: int = 0
    training_count: int = 0
    repair_count: int = 0
    total_latency_ns: float = 0.0
    total_bandwidth_bytes: float = 0.0

    @property
    def average_latency_ns(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ns / self.total_requests

    @property
    def row_hit_rate(self) -> float:
        if self.read_requests + self.write_requests == 0:
            return 0.0
        return self.row_hit_count / (self.read_requests + self.write_requests)


class HBM4Controller:
    """HBM4 Memory Controller Integration

    This controller integrates all HBM4-specific modules:
    - 32 independent channels
    - Per-channel and pseudo-channel scheduling
    - QoS-based request prioritization
    - Per-bank and autonomous refresh
    - DFI 5.0 PHY interface
    - Lane repair and training support
    - Command generation and scheduling
    """

    def __init__(
        self,
        spec: Optional[HBM4Spec] = None,
        config: Optional[HBMConfig] = None,
        enable_qos: bool = True,
        enable_refresh: bool = True,
        enable_dfi: bool = True,
    ):
        """Initialize HBM4 Controller

        Args:
            spec: HBM4 specification (uses default if None)
            config: Optional HBMConfig for base class compatibility
            enable_qos: Enable QoS scheduling
            enable_refresh: Enable refresh scheduling
            enable_dfi: Enable DFI 5.0 interface
        """
        self.spec = spec or HBM4Spec()
        self.current_time_ns = 0
        self._cycle_count = 0

        # Configuration
        self._enable_qos = enable_qos
        self._enable_refresh = enable_refresh
        self._enable_dfi = enable_dfi

        # Initialize HBM4-specific address decoder
        self.decoder = HBM4AddressDecoder(spec=self.spec)

        # Initialize queue manager with HBM4 channel count
        # Each channel gets a portion of the total queue depth
        # Allow 8 requests per channel to ensure all channels can submit simultaneously
        per_channel_queue = 8  # Fixed capacity per channel
        self.queue_manager = QueueManager.create(queue_depth=per_channel_queue * self.spec.channels)

        # Initialize QoS scheduler
        if self._enable_qos:
            self.qos_scheduler = HBM4QoSScheduler(config=self.spec)
        else:
            self.qos_scheduler = None

        # Initialize refresh scheduler
        if self._enable_refresh:
            self.refresh_scheduler = HBM4RefreshScheduler(config=self.spec)
        else:
            self.refresh_scheduler = None

        # Initialize DFI 5.0 interface
        if self._enable_dfi:
            self.dfi = DFI5Interface()
        else:
            self.dfi = None

        # Initialize HBM4 DRAM channel model for refresh integration
        self.channel_model = HBM4ChannelArray(spec=self.spec)

        # Per-channel state tracking
        self._channel_states: Dict[int, 'ChannelState'] = {}
        for ch in range(self.spec.channels):
            self._channel_states[ch] = ChannelState(channel_id=ch)

        # Command generation state
        self._pending_commands: Dict[str, DFIRequest] = {}

        # Statistics
        self.stats = HBM4ControllerStats()

        # Request tracking
        self._pending_requests: Dict[str, HBMRequest] = {}
        self._completed_requests: List[HBMResponse] = []

    @property
    def channels(self) -> int:
        """Number of HBM4 channels"""
        return self.spec.channels

    @property
    def pseudo_channels(self) -> int:
        """Total pseudo-channels"""
        return self.spec.pseudo_channels

    @property
    def dfi_ready(self) -> bool:
        """Check if DFI interface is ready"""
        if not self.dfi:
            return True
        return self.dfi.is_ready()

    def submit_request(
        self,
        addr: int,
        is_read: bool,
        qos_level: int = 8,
        size_bytes: int = 64,
    ) -> Optional[str]:
        """Submit a request to the controller

        Args:
            addr: 64-bit physical address
            is_read: True for read, False for write
            qos_level: QoS priority level (0-15, HIGHER = higher priority)
            size_bytes: Request size in bytes

        Returns:
            Request ID if successful, None if queue full
        """
        # Decode address
        decoded = self.decoder.decode(addr)

        # Create request
        request = HBMRequest(
            addr=addr,
            length=size_bytes,
            is_read=is_read,
            qos=qos_level,
            channel_id=decoded.channel_id,
            pseudo_channel_id=decoded.pseudo_channel_id,
            bank_id=decoded.bank_id,
            row_id=decoded.row_id,
            col_id=decoded.col_id,
        )
        request.arrival_time = self.current_time_ns

        # Enqueue request - queue push returns success/failure
        if is_read:
            success = self.queue_manager.push_read(request)
        else:
            success = self.queue_manager.push_write(request)

        if not success:
            _logger.debug(f"Queue full, request rejected: addr=0x{addr:x}, ch={decoded.channel_id}")
            return None

        # Track request
        self._pending_requests[request.request_id] = request

        # Generate DFI command if DFI is enabled
        if self.dfi:
            self._generate_dfi_command(request, decoded)

        # Update statistics
        self.stats.total_requests += 1
        if is_read:
            self.stats.read_requests += 1
        else:
            self.stats.write_requests += 1

        _logger.debug(
            f"Request submitted: id={request.request_id}, "
            f"addr=0x{addr:x}, ch={decoded.channel_id}, "
            f"pch={decoded.pseudo_channel_id}, qos={qos_level}"
        )
        return request.request_id

    def _generate_dfi_command(self, request: HBMRequest, decoded) -> None:
        """Generate DFI command for a request

        Args:
            request: The HBM request
            decoded: Decoded address fields
        """
        if not self.dfi:
            return

        # Map request type to DFI command
        if request.is_read:
            cmd_type = 'RD'
        else:
            cmd_type = 'WR'

        # Create DFI request with address components
        addr_vec = {
            'row': request.row_id,
            'bank': request.bank_id,
            'pseudo_channel': request.pseudo_channel_id,
            'channel': request.channel_id,
            'address': request.addr,
        }

        # Encode command
        dfi_req = self.dfi.encode_command(
            cmd=cmd_type,
            addr_vec=addr_vec,
            priority=request.qos,
        )
        dfi_req.request_id = request.request_id

        # Queue the DFI request
        self.dfi.queue_request(dfi_req)
        self._pending_commands[request.request_id] = dfi_req

        _logger.debug(
            f"DFI command generated: req_id={request.request_id}, "
            f"cmd={cmd_type}, ch={request.channel_id}, "
            f"pch={request.pseudo_channel_id}, row={request.row_id}"
        )

    def tick(self) -> List[HBMResponse]:
        """Execute one clock cycle

        Returns:
            List of completed responses this cycle
        """
        self._cycle_count += 1
        self.current_time_ns += 1  # 1ns per cycle at 1 GHz

        responses = []

        _logger.debug(f"[Cycle {self._cycle_count}] Time={self.current_time_ns}ns")

        # Tick DFI interface if enabled
        if self.dfi:
            self.dfi.tick()

        # Handle refresh if enabled
        if self.refresh_scheduler:
            self.refresh_scheduler.tick()
            refresh_response = self._handle_refresh()
            if refresh_response:
                responses.append(refresh_response)
                _logger.debug(
                    f"Refresh completed: ch={refresh_response.channel_id}, "
                    f"bank={refresh_response.bank_id}"
                )

        # Handle per-channel scheduling
        for ch_id in range(self.spec.channels):
            response = self._schedule_channel(ch_id)
            if response:
                responses.append(response)
                _logger.debug(
                    f"Request completed: id={response.request_id}, "
                    f"ch={response.channel_id}, latency={response.latency}ns"
                )

        # Handle training/repair if needed
        self._handle_background_tasks()

        return responses

    def _handle_refresh(self) -> Optional[HBMResponse]:
        """Handle refresh scheduling and execute on channel model

        Returns:
            Refresh response if refresh completed
        """
        if not self.refresh_scheduler:
            return None

        # Check if refresh is needed
        if self.refresh_scheduler.can_refresh():
            # Get next refresh command (returns 4-tuple: cmd, channel_id, pch, bank_id)
            refresh_cmd = self.refresh_scheduler.get_refresh_command()
            if refresh_cmd:
                cmd_name, channel_id, pseudo_channel_id, bank_id = refresh_cmd

                # Execute refresh on the channel model
                if cmd_name == 'REFab':
                    # All-bank refresh
                    self.channel_model.get_channel(channel_id).execute_refresh('REFab')
                elif cmd_name == 'REFsb':
                    # Per-bank refresh
                    ch = self.channel_model.get_channel(channel_id)
                    if ch:
                        ch.execute_refresh('REFsb', pseudo_channel=pseudo_channel_id, bank=bank_id)

                # Mark bank as refreshed in scheduler
                self.refresh_scheduler.mark_bank_refreshed(
                    channel_id, pseudo_channel_id, bank_id, self._cycle_count
                )
                self.stats.refresh_count += 1

                return HBMResponse(
                    request_id=f"refresh_ch{channel_id}_pch{pseudo_channel_id}_bank{bank_id}",
                    status="REFRESH_COMPLETE",
                    latency=self.spec.nRFC,
                    channel_id=channel_id,
                    bank_id=bank_id,
                )

        return None

    def _schedule_channel(self, channel_id: int) -> Optional[HBMResponse]:
        """Schedule requests for a specific channel

        Args:
            channel_id: Channel to schedule

        Returns:
            Response if request completed
        """
        channel_state = self._channel_states[channel_id]

        # Get requests for this channel
        read_queue = self.queue_manager.read_queue
        write_queue = self.queue_manager.write_queue

        # Filter requests for this channel
        ch_reads = [r for r in read_queue if r.channel_id == channel_id]
        ch_writes = [r for r in write_queue if r.channel_id == channel_id]

        if not ch_reads and not ch_writes:
            return None

        # Select request based on QoS if enabled
        if self.qos_scheduler and self._enable_qos:
            # Use QoS scheduler to select highest priority request
            all_requests = ch_reads + ch_writes
            selected = self.qos_scheduler.select_next(all_requests)
        else:
            # Simple FCFS
            all_requests = ch_reads + ch_writes
            if all_requests:
                selected = min(all_requests, key=lambda r: r.arrival_time)
            else:
                selected = None

        if not selected:
            return None

        # Calculate latency based on request type and row state
        if selected.is_read:
            if selected.row_hit:
                # Read with row hit: CAS latency + burst
                latency = self.spec.nCL + self.spec.nBL
                _logger.debug(f"Read row hit: latency={latency}ns (CL={self.spec.nCL}, BL={self.spec.nBL})")
                self.stats.row_hit_count += 1
            else:
                # Read with row miss: ACT + READ + PRE
                latency = (
                    self.spec.nRCDRD + self.spec.nCL + self.spec.nBL +
                    self.spec.nRP + self.spec.nRAS
                )
                _logger.debug(f"Read row miss: latency={latency}ns (RCDRD={self.spec.nRCDRD})")
        else:
            if selected.row_hit:
                # Write with row hit: CWL + burst + write recovery
                latency = self.spec.nCWL + self.spec.nBL + self.spec.nWR
                _logger.debug(f"Write row hit: latency={latency}ns (CWL={self.spec.nCWL})")
                self.stats.row_hit_count += 1
            else:
                # Write with row miss: ACT + WRITE + PRE
                latency = (
                    self.spec.nRCDWR + self.spec.nCWL + self.spec.nBL +
                    self.spec.nWR + self.spec.nRP + self.spec.nRAS
                )
                _logger.debug(f"Write row miss: latency={latency}ns (RCDWR={self.spec.nRCDWR})")

        # Mark request completed
        selected.mark_completed(self.current_time_ns)

        # Update statistics
        self.stats.total_latency_ns += latency
        self.stats.total_bandwidth_bytes += selected.length

        # Remove from queue using QueueManager convenience methods
        if selected.is_read:
            self.queue_manager.remove_read(selected.request_id)
        else:
            self.queue_manager.remove_write(selected.request_id)

        # Update channel state
        channel_state.queue_depth = max(0, channel_state.queue_depth - 1)

        # Remove from pending
        if selected.request_id in self._pending_requests:
            del self._pending_requests[selected.request_id]

        return HBMResponse(
            request_id=selected.request_id,
            status="OK",
            latency=latency,
            channel_id=channel_id,
            bank_id=selected.bank_id,
        )

    def _handle_background_tasks(self) -> None:
        """Handle background tasks like training and repair

        This method processes:
        - Training sequences (WRLvl, RDDLL, etc.)
        - Lane repair mapping updates
        - ECC/CRC error tracking
        - Per-channel power state management
        """
        # Background tasks are typically managed externally
        # This is a placeholder for periodic maintenance
        # In real hardware, this would include:
        # - Periodic DQ calibration
        # - Read data eye training
        # - Write level training
        # - VREF calibration
        _logger.debug(f"[Cycle {self._cycle_count}] Background task check")

    def _handle_write_command(
        self,
        request: HBMRequest,
        channel_state: 'ChannelState'
    ) -> Optional[HBMResponse]:
        """Handle write command execution with proper timing

        Args:
            request: The write request to process
            channel_state: Current channel state

        Returns:
            Write response if completed, None otherwise
        """
        # Calculate write latency based on row state
        if request.row_hit:
            # Write data + internal write timing
            latency = self.spec.nCWL + self.spec.nBL + self.spec.nWR
            _logger.debug(f"Write row hit: latency={latency}ns")
        else:
            # Row miss - requires precharge + activate + write
            latency = (
                self.spec.nCWL + self.spec.nBL +
                self.spec.nRCDWR + self.spec.nRP +
                self.spec.nRAS + self.spec.nWR
            )
            _logger.debug(f"Write row miss: latency={latency}ns")

        # Mark request completed
        request.mark_completed(self.current_time_ns)

        # Update statistics
        self.stats.total_latency_ns += latency
        self.stats.total_bandwidth_bytes += request.length

        # Remove from pending
        if request.request_id in self._pending_requests:
            del self._pending_requests[request.request_id]

        return HBMResponse(
            request_id=request.request_id,
            status="WRITE_COMPLETE",
            latency=latency,
            channel_id=request.channel_id,
            bank_id=request.bank_id,
        )

    def _get_queue_capacity(self) -> int:
        """Get per-channel queue capacity"""
        return 8  # 8 requests per channel

    def trigger_repair(self, channel_id: int, lane_mask: int) -> bool:
        """Trigger lane repair for a channel

        Args:
            channel_id: Channel to repair
            lane_mask: Bit mask of lanes to remap

        Returns:
            True if repair successful
        """
        if channel_id not in self._channel_states:
            return False

        channel_state = self._channel_states[channel_id]
        channel_state.repair_state = lane_mask
        self.stats.repair_count += 1

        return True

    def get_stats(self) -> Dict:
        """Get comprehensive statistics

        Returns:
            Dictionary of all statistics
        """
        stats = {
            'controller': {
                'total_requests': self.stats.total_requests,
                'read_requests': self.stats.read_requests,
                'write_requests': self.stats.write_requests,
                'row_hit_rate': self.stats.row_hit_rate,
                'average_latency_ns': self.stats.average_latency_ns,
                'refresh_count': self.stats.refresh_count,
                'training_count': self.stats.training_count,
                'repair_count': self.stats.repair_count,
            },
            'spec': {
                'channels': self.spec.channels,
                'pseudo_channels': self.spec.pseudo_channels,
                'total_banks': self.spec.total_banks,
                'bandwidth_tbps': self.spec.bandwidth,
                'io_width': self.spec.io_width,
                'data_rate_gtps': self.spec.data_rate_gtps,
            },
            'queues': {
                'read_depth': len(self.queue_manager.read_queue),
                'write_depth': len(self.queue_manager.write_queue),
            },
            'qos': {
                'enabled': self._enable_qos,
                'priority_levels': 16,
            } if self.qos_scheduler else None,
            'refresh': {
                'enabled': self._enable_refresh,
                'mode': str(self.refresh_scheduler.mode) if self.refresh_scheduler else None,
            } if self.refresh_scheduler else None,
            'dfi': {
                'enabled': self._enable_dfi,
                'ready': self.dfi_ready,
                'lp_state': str(self.dfi.lp_state.name) if self.dfi else None,
                'pending_commands': len(self._pending_commands),
            } if self.dfi else None,
        }
        return stats

    def get_bandwidth_gbs(self) -> float:
        """Calculate current effective bandwidth in GB/s

        Returns:
            Effective bandwidth in GB/s (capped at peak bandwidth)
        """
        if self.current_time_ns == 0:
            return 0.0

        # Bandwidth = bytes / time
        bytes_per_ns = self.stats.total_bandwidth_bytes / self.current_time_ns
        gbs = bytes_per_ns * 1000  # Convert to GB/s

        # Cap at peak bandwidth
        return min(gbs, self.spec.bandwidth_gbs)

    def get_effective_bandwidth_tbps(self) -> float:
        """Calculate effective bandwidth after overhead

        Returns:
            Effective bandwidth in TB/s
        """
        gbs = self.get_bandwidth_gbs()
        return gbs / 1000  # Convert to TB/s

    # === DFI 5.0 Interface Methods ===

    def dfi_request_ctrlupd(self) -> bool:
        """Request a DFI control update

        Returns:
            True if request was accepted
        """
        if not self.dfi:
            return False
        return self.dfi.request_ctrlupd()

    def dfi_set_frequency(self, freq_mhz: int) -> bool:
        """Set DFI interface frequency

        Args:
            freq_mhz: Target frequency in MHz

        Returns:
            True if frequency change was accepted
        """
        if not self.dfi:
            return False
        return self.dfi.request_freq_change(freq_mhz)

    def dfi_enter_freq_change(self) -> bool:
        """Enter frequency change sequence

        Returns:
            True if transition was successful
        """
        if not self.dfi:
            return False
        return self.dfi.enter_freq_change()

    def dfi_exit_freq_change(self) -> bool:
        """Exit frequency change sequence

        Returns:
            True if transition was successful
        """
        if not self.dfi:
            return False
        return self.dfi.exit_freq_change()

    def dfi_set_low_power(self, state: DFILowPowerState) -> bool:
        """Set DFI low power state

        Args:
            state: Target low power state

        Returns:
            True if transition was successful
        """
        if not self.dfi:
            return False
        return self.dfi.request_low_power(state)

    def dfi_wakeup(self) -> None:
        """Wakeup from low power state"""
        if self.dfi:
            self.dfi.wakeup_from_low_power()

    def dfi_get_signals(self):
        """Get current DFI signal states

        Returns:
            DFISignals object or None if DFI disabled
        """
        if not self.dfi:
            return None
        return self.dfi.get_dfi_signals()

    def dfi_get_statistics(self) -> Dict[str, Any]:
        """Get DFI interface statistics

        Returns:
            Dictionary with DFI statistics
        """
        if not self.dfi:
            return {}
        return self.dfi.get_statistics()

    def trigger_training(self, channel_id: Optional[int] = None) -> str:
        """Trigger training for a channel or all channels

        Args:
            channel_id: Specific channel to train, or None for all

        Returns:
            Training command ID
        """
        training_id = f"train_{uuid.uuid4().hex[:8]}"
        self.stats.training_count += 1

        # Start DFI training if enabled
        if self.dfi:
            self.dfi.start_training()

        # Training is modeled as a blocking operation
        # In real hardware, this would take many cycles
        return training_id


@dataclass
class ChannelState:
    """State tracking for a single HBM4 channel"""
    channel_id: int
    queue_depth: int = 0
    repair_state: int = 0  # 0 = no repair needed
    last_refresh_cycle: int = 0
    training_state: str = "COMPLETE"  # IDLE, TRAINING, COMPLETE
    power_state: str = "ACTIVE"  # ACTIVE, SELF_REFRESH, POWER_DOWN

    def is_available(self) -> bool:
        """Check if channel is available for requests"""
        return (
            self.training_state == "COMPLETE" and
            self.power_state == "ACTIVE"
        )