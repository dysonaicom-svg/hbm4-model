"""
HBM4 Lane Repair (Redundancy) Model

Implements lane repair functionality for HBM4 channels, providing redundancy
to mitigate DRAM manufacturing defects and improve overall system yield.

LANE ARCHITECTURE:
================
Each HBM4 channel contains N data lanes plus S spare lanes:
  - Data lanes: indices 0 to (lanes_per_channel - 1)
  - Spare lanes: indices lanes_per_channel to (lanes_per_channel + total_spares - 1)

For example, with 64 data lanes and 4 spares:
  - Data lane indices: 0-63
  - Spare lane indices: 64-67

REPAIR TYPES:
============
  - "bit": Single bit repair within a lane (granularity: individual bit)
  - "byte": Byte-level repair (8 bits repaired as a unit)
  - "channel": Full lane/channel repair (entire lane replaced)

REPAIR WORKFLOW:
===============
1. During manufacturing test or PHY training, failed lanes are detected
2. add_failed_lane() or perform_repair() registers the failure
3. Spare lanes are allocated from the available pool
4. Traffic is transparently remapped via get_remapped_lane()

REPAIR LIMITS:
=============
Each channel has a fixed number of spare lanes (typical: 2-4 for HBM4).
When all spares are exhausted, the channel is marked UNREPAIRABLE.
The system tracks repair status per-channel and globally.

Based on:
  - JEDEC JESD270-4A HBM4 specification
  - Cadence HBM4E documentation
  - Synopsys HBM4 Controller IP
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import struct


class RepairStatus(Enum):
    """Lane repair status indicating repair coverage level.

    NO_REPAIR:      No failures detected in this channel
    PARTIAL_REPAIR: Some failures repaired, spares remain available
    FULL_REPAIR:    All spares used, channel fully repaired
    UNREPAIRABLE:  Failures exceed spare capacity
    """
    NO_REPAIR = "no_repair"
    PARTIAL_REPAIR = "partial_repair"
    FULL_REPAIR = "full_repair"
    UNREPAIRABLE = "unrepairable"


@dataclass
class LaneRepairEntry:
    """Single lane repair mapping entry.

    Attributes:
        failed_lane: Index of the defective data lane (0 to lanes-1)
        spare_lane: Index of the replacement spare lane (lanes to lanes+spares-1)
        repair_type: Granularity of repair ("bit", "byte", "channel")
        channel_id: HBM4 channel this repair applies to
    """
    failed_lane: int
    spare_lane: int
    repair_type: str  # "bit", "byte", "channel"
    channel_id: int


@dataclass
class LaneRepairMap:
    """Lane repair map for one HBM4 channel.

    Maintains the complete repair state for a single channel including:
      - List of failed lanes (detected defects)
      - List of allocated spare lanes (in use)
      - Repair entries (failed -> spare mappings)
      - Repair capacity tracking

    Lane Indexing Convention:
      - Data lanes: indices 0 to (total_lanes - 1)
      - Spare lanes: indices total_lanes to (total_lanes + total_spares - 1)
    """
    channel_id: int
    total_lanes: int
    total_spares: int
    failed_lanes: List[int] = field(default_factory=list)
    spare_lanes: List[int] = field(default_factory=list)
    repair_entries: List[LaneRepairEntry] = field(default_factory=list)

    # Repair state
    repair_count: int = 0
    max_repair_count: int = 0  # Set during initialization

    def __post_init__(self):
        if self.max_repair_count == 0:
            self.max_repair_count = self.total_spares

    @property
    def available_spares(self) -> int:
        """Number of spare lanes still available for repair."""
        return self.total_spares - len(self.repair_entries)

    @property
    def is_repairable(self) -> bool:
        """Check if channel can accept more repairs.

        A channel is repairable if:
          - Failed lane count does not exceed total spares
          - Repair count is below maximum (spares available)
        """
        return len(self.failed_lanes) <= self.total_spares and self.repair_count < self.max_repair_count

    @property
    def status(self) -> RepairStatus:
        """Get current repair status based on failed lane count vs spares."""
        if len(self.failed_lanes) == 0:
            return RepairStatus.NO_REPAIR
        if len(self.failed_lanes) < self.total_spares:
            return RepairStatus.PARTIAL_REPAIR
        if len(self.failed_lanes) == self.total_spares:
            return RepairStatus.FULL_REPAIR
        return RepairStatus.UNREPAIRABLE


class HBM4LaneRepairModel:
    """HBM4 Lane Repair (Redundancy) Model

    Manages lane repair for all channels in the HBM4 stack. This model
    implements the redundancy mechanism used in HBM devices to improve
    manufacturing yield by providing spare lanes to replace defective ones.

    KEY CAPABILITIES:
    =================
    - Per-channel repair maps: Each of N channels has independent repair tracking
    - Spare lane allocation: Automatic selection from available spare pool
    - Lane remapping: Transparent traffic redirection via get_remapped_lane()
    - Repair status tracking: NO_REPAIR -> PARTIAL_REPAIR -> FULL_REPAIR -> UNREPAIRABLE
    - Yield simulation: Simulate random failures to analyze system-level impact

    USAGE EXAMPLE:
    ==============
    ```python
    # Create model for 32-channel HBM4 (64 DQ lanes + 4 spare per channel)
    model = HBM4LaneRepairModel(num_channels=32, lanes_per_channel=64, spare_lanes_per_channel=4)

    # Detect and repair a failed lane
    spare = model.perform_repair(channel_id=0, failed_lane=42)
    if spare is not None:
        print(f"Remapped lane 42 -> spare {spare}")

    # Check remapping for data traffic
    actual_lane = model.get_remapped_lane(channel_id=0, lane_id=42)  # Returns spare index
    actual_lane = model.get_remapped_lane(channel_id=0, lane_id=10)  # Returns 10 (no remap)

    # Query system health
    stats = model.get_stats()
    print(f"Total repairs: {stats['total_repairs']}, Unrepairable channels: {stats['unrepairable_channels']}")
    ```

    INTEGRATION POINTS:
    ===================
    - PHY Training: Report failed lanes detected during margin testing
    - Memory BIST: Register defects found during manufacturing test
    - Traffic Monitor: Use get_remapped_lane() to redirect traffic through spares
    - System Simulation: simulate_yield_loss() for statistical analysis
    """

    def __init__(
        self,
        num_channels: int = 32,
        lanes_per_channel: int = 64,
        spare_lanes_per_channel: int = 4,
    ):
        """Initialize Lane Repair Model

        Args:
            num_channels: Number of HBM4 channels (default 32 for full HBM4 stack)
            lanes_per_channel: Data lanes per channel (default 64 for x64 DQ interface)
            spare_lanes_per_channel: Number of spare lanes (typical: 2-4 per JEDEC)
        """
        self.num_channels = num_channels
        self.lanes_per_channel = lanes_per_channel
        self.spare_lanes_per_channel = spare_lanes_per_channel

        # Initialize per-channel repair maps
        self._repair_maps: Dict[int, LaneRepairMap] = {}
        for ch in range(num_channels):
            self._repair_maps[ch] = LaneRepairMap(
                channel_id=ch,
                total_lanes=lanes_per_channel,
                total_spares=spare_lanes_per_channel,
            )

        # Global statistics
        self._total_repairs: int = 0
        self._total_failed_lanes: int = 0
        self._unrepairable_channels: List[int] = []

    # ==================== Configuration ====================

    def configure_channel(
        self,
        channel_id: int,
        lanes: int,
        spares: int,
    ) -> None:
        """Configure lane/spare counts for a channel

        Args:
            channel_id: Channel to configure
            lanes: Number of data lanes
            spares: Number of spare lanes
        """
        if channel_id not in self._repair_maps:
            self._repair_maps[channel_id] = LaneRepairMap(
                channel_id=channel_id,
                total_lanes=lanes,
                total_spares=spares,
            )
        else:
            rm = self._repair_maps[channel_id]
            rm.total_lanes = lanes
            rm.total_spares = spares
            rm.max_repair_count = spares

    # ==================== Repair Operations ====================

    def add_failed_lane(self, channel_id: int, lane_id: int) -> bool:
        """Add a failed lane to repair map

        Args:
            channel_id: Channel with failed lane
            lane_id: Failed lane index

        Returns:
            True if lane added successfully
        """
        if channel_id not in self._repair_maps:
            return False

        rm = self._repair_maps[channel_id]

        # Check if lane already tracked
        if lane_id in rm.failed_lanes:
            return True  # Already tracked

        # Check if repair possible
        if not rm.is_repairable:
            if channel_id not in self._unrepairable_channels:
                self._unrepairable_channels.append(channel_id)
            return False

        rm.failed_lanes.append(lane_id)
        self._total_failed_lanes += 1

        return True

    def allocate_spare(
        self,
        channel_id: int,
        failed_lane: int,
        spare_lane: int,
        repair_type: str = "bit",
    ) -> bool:
        """Allocate a spare lane for a failed lane

        Args:
            channel_id: Channel to repair
            failed_lane: Failed lane index
            spare_lane: Spare lane to use
            repair_type: Type of repair ("bit", "byte", "channel")

        Returns:
            True if spare allocated successfully
        """
        if channel_id not in self._repair_maps:
            return False

        rm = self._repair_maps[channel_id]

        # Validate spare lane
        if spare_lane in rm.spare_lanes:
            return False  # Spare already used

        # Check repair capacity
        if rm.available_spares <= 0:
            return False

        # Add repair entry
        entry = LaneRepairEntry(
            failed_lane=failed_lane,
            spare_lane=spare_lane,
            repair_type=repair_type,
            channel_id=channel_id,
        )
        rm.repair_entries.append(entry)
        rm.spare_lanes.append(spare_lane)
        rm.repair_count += 1
        self._total_repairs += 1

        return True

    def perform_repair(
        self,
        channel_id: int,
        failed_lane: int,
        repair_type: str = "bit",
    ) -> Optional[int]:
        """Perform repair by allocating first available spare lane.

        This is the main repair operation - it:
          1. Checks if lane is already remapped (return existing spare)
          2. Adds the failed lane to the track list (if not already tracked)
          3. Finds the first available spare lane
          4. Creates the repair mapping entry

        Args:
            channel_id: Channel to repair
            failed_lane: Failed lane index (0 to lanes_per_channel-1)
            repair_type: Granularity of repair ("bit", "byte", "channel")

        Returns:
            Spare lane index allocated, or None if repair failed (no spares available)
        """
        if channel_id not in self._repair_maps:
            return None

        rm = self._repair_maps[channel_id]

        # Check if lane is already remapped - return existing spare
        for entry in rm.repair_entries:
            if entry.failed_lane == failed_lane:
                return entry.spare_lane

        # Add failed lane if not already tracked
        if failed_lane not in rm.failed_lanes:
            if not self.add_failed_lane(channel_id, failed_lane):
                return None

        # Find first available spare
        spare_base = rm.total_lanes  # Spares are after data lanes
        for i in range(rm.total_spares):
            spare_lane = spare_base + i
            if spare_lane not in rm.spare_lanes:
                if self.allocate_spare(channel_id, failed_lane, spare_lane, repair_type):
                    return spare_lane

        return None

    def is_lane_remapped(self, channel_id: int, lane_id: int) -> bool:
        """Check if a lane has been remapped to a spare.

        Use this to determine if traffic for a given lane should be redirected.

        Args:
            channel_id: Channel to check
            lane_id: Lane index to query

        Returns:
            True if lane has been remapped to a spare lane
        """
        if channel_id not in self._repair_maps:
            return False
        rm = self._repair_maps[channel_id]
        return any(e.failed_lane == lane_id for e in rm.repair_entries)

    def get_remapped_lane(self, channel_id: int, lane_id: int) -> int:
        """Get the spare lane that replaces a failed lane.

        This is the primary interface for traffic redirection - use in the data path
        to transparently route traffic through spare lanes.

        Args:
            channel_id: Channel to check
            lane_id: Original (failed) lane index

        Returns:
            Spare lane index if remapped, otherwise returns original lane_id
        """
        if channel_id not in self._repair_maps:
            return lane_id
        rm = self._repair_maps[channel_id]
        for entry in rm.repair_entries:
            if entry.failed_lane == lane_id:
                return entry.spare_lane
        return lane_id

    # ==================== Query Operations ====================

    def get_channel_repair_map(self, channel_id: int) -> Optional[LaneRepairMap]:
        """Get repair map for a channel

        Args:
            channel_id: Channel to query

        Returns:
            LaneRepairMap or None if channel doesn't exist
        """
        return self._repair_maps.get(channel_id)

    def get_repair_status(self, channel_id: int) -> RepairStatus:
        """Get repair status for a channel

        Args:
            channel_id: Channel to query

        Returns:
            RepairStatus enum value
        """
        if channel_id not in self._repair_maps:
            return RepairStatus.NO_REPAIR
        return self._repair_maps[channel_id].status

    def get_all_failed_lanes(self, channel_id: int) -> List[int]:
        """Get all failed lanes for a channel

        Args:
            channel_id: Channel to query

        Returns:
            List of failed lane indices
        """
        if channel_id not in self._repair_maps:
            return []
        return list(self._repair_maps[channel_id].failed_lanes)

    # ==================== Statistics ====================

    def get_stats(self) -> Dict:
        """Get lane repair statistics

        Returns:
            Dictionary with repair statistics
        """
        total_repairs = sum(rm.repair_count for rm in self._repair_maps.values())
        total_failed = sum(len(rm.failed_lanes) for rm in self._repair_maps.values())

        return {
            'total_channels': self.num_channels,
            'lanes_per_channel': self.lanes_per_channel,
            'spares_per_channel': self.spare_lanes_per_channel,
            'total_repairs': total_repairs,
            'total_failed_lanes': total_failed,
            'unrepairable_channels': len(self._unrepairable_channels),
            'channels_with_repairs': sum(1 for rm in self._repair_maps.values() if rm.repair_count > 0),
        }

    def get_channel_stats(self, channel_id: int) -> Optional[Dict]:
        """Get statistics for a specific channel

        Args:
            channel_id: Channel to query

        Returns:
            Dictionary with channel statistics
        """
        rm = self._repair_maps.get(channel_id)
        if rm is None:
            return None

        return {
            'channel_id': channel_id,
            'failed_lanes': len(rm.failed_lanes),
            'repair_count': rm.repair_count,
            'available_spares': rm.available_spares,
            'status': rm.status.value,
            'is_repairable': rm.is_repairable,
        }

    # ==================== Simulation Support ====================

    def simulate_yield_loss(
        self,
        channel_id: int,
        failure_rate: float = 0.01,
    ) -> int:
        """Simulate random lane failures for yield analysis.

        Used for statistical analysis of system yield. Each lane has an independent
        probability of failure based on failure_rate.

        Args:
            channel_id: Channel to simulate failures on
            failure_rate: Probability of each lane failing (0.0 to 1.0).
                          Default 0.01 (1% per lane).

        Returns:
            Number of lanes that failed in this simulation run
        """
        if channel_id not in self._repair_maps:
            return 0

        rm = self._repair_maps[channel_id]
        failed_count = 0

        for lane in range(rm.total_lanes):
            if random.random() < failure_rate:
                if lane not in rm.failed_lanes:
                    rm.failed_lanes.append(lane)
                    self._total_failed_lanes += 1
                    failed_count += 1

        return failed_count

    def reset_channel(self, channel_id: int) -> None:
        """Reset repair state for a channel (e.g., for new test scenario).

        Clears all repair entries, failed lanes, and statistics for the channel.
        Does not affect other channels.

        Args:
            channel_id: Channel to reset
        """
        if channel_id in self._repair_maps:
            rm = self._repair_maps[channel_id]
            rm.failed_lanes.clear()
            rm.spare_lanes.clear()
            rm.repair_entries.clear()
            rm.repair_count = 0

            if channel_id in self._unrepairable_channels:
                self._unrepairable_channels.remove(channel_id)

    def reset_all(self) -> None:
        """Reset all repair state across all channels.

        Clears all repair maps, failed lanes, and global statistics.
        Useful for running multiple independent test scenarios.
        """
        for ch in self._repair_maps:
            self.reset_channel(ch)
        self._total_repairs = 0
        self._total_failed_lanes = 0
        self._unrepairable_channels.clear()

    # ==================== Repair Sequence Generation ====================

    def generate_repair_sequence(self, channel_id: int) -> Optional[List[Dict[str, Any]]]:
        """Generate repair programming sequence for eFuse/fuse box.

        Creates a sequence of repair entries that can be programmed into
        non-volatile storage (eFuses) for permanent lane remapping.

        Args:
            channel_id: Channel to generate sequence for

        Returns:
            List of repair entries, each containing:
              - failed_lane: Original lane index
              - spare_lane: Replacement spare lane index
              - repair_type: Type of repair ("bit", "byte", "channel")
              - encoding: Encoded value for fuse programming
            Returns None if channel doesn't exist or has no repairs
        """
        if channel_id not in self._repair_maps:
            return None

        rm = self._repair_maps[channel_id]
        if not rm.repair_entries:
            return None

        sequence = []
        for entry in rm.repair_entries:
            sequence.append({
                'failed_lane': entry.failed_lane,
                'spare_lane': entry.spare_lane,
                'repair_type': entry.repair_type,
                'encoding': self._encode_repair_entry(entry),
            })

        return sequence

    def _encode_repair_entry(self, entry: LaneRepairEntry) -> int:
        """Encode repair entry into a single integer for fuse programming.


        Encoding format (32 bits):
          [31:24] - Repair type (0=bit, 1=byte, 2=channel)
          [23:16] - Channel ID (0-255)
          [15:8]  - Failed lane (0-255)
          [7:0]   - Spare lane (0-255)

        Args:
            entry: Repair entry to encode

        Returns:
            32-bit encoded value
        """
        type_map = {'bit': 0, 'byte': 1, 'channel': 2}
        type_val = type_map.get(entry.repair_type, 0)

        encoding = (type_val << 24) | (entry.channel_id << 16) | \
                   (entry.failed_lane << 8) | entry.spare_lane
        return encoding

    def decode_repair_entry(self, encoding: int) -> Dict[str, Any]:
        """Decode a fused repair entry back to components.

        Args:
            encoding: 32-bit encoded value

        Returns:
            Dictionary with decoded fields:
              - repair_type: String ("bit", "byte", "channel")
              - channel_id: Channel number
              - failed_lane: Original lane index
              - spare_lane: Replacement spare index
        """
        type_map = {0: 'bit', 1: 'byte', 2: 'channel'}
        type_val = (encoding >> 24) & 0xFF
        channel_id = (encoding >> 16) & 0xFF
        failed_lane = (encoding >> 8) & 0xFF
        spare_lane = encoding & 0xFF

        return {
            'repair_type': type_map.get(type_val, 'bit'),
            'channel_id': channel_id,
            'failed_lane': failed_lane,
            'spare_lane': spare_lane,
        }

    def generate_bulk_repair_sequence(self) -> Dict[int, List[Dict[str, Any]]]:
        """Generate repair sequences for all channels.

        Creates a complete programming sequence for all channels with repairs.
        Useful for mass programming of eFuses during manufacturing.

        Returns:
            Dictionary mapping channel_id -> list of repair entries
        """
        bulk_sequence = {}
        for ch_id in self._repair_maps:
            seq = self.generate_repair_sequence(ch_id)
            if seq:
                bulk_sequence[ch_id] = seq
        return bulk_sequence

    def export_repair_map(self, channel_id: int) -> Optional[Dict[str, Any]]:
        """Export complete repair map for a channel.

        Creates a serializable dictionary representation of the repair state
        for persistence or transmission to other systems.

        Args:
            channel_id: Channel to export

        Returns:
            Dictionary containing:
              - channel_id: Channel number
              - total_lanes: Number of data lanes
              - total_spares: Number of spare lanes
              - repair_entries: List of repair mappings
              - status: Repair status string
              - encoding: Encoded fuse values for each repair
        """
        if channel_id not in self._repair_maps:
            return None

        rm = self._repair_maps[channel_id]
        entries = []
        for entry in rm.repair_entries:
            entries.append({
                'failed_lane': entry.failed_lane,
                'spare_lane': entry.spare_lane,
                'repair_type': entry.repair_type,
                'encoding': self._encode_repair_entry(entry),
            })

        return {
            'channel_id': channel_id,
            'total_lanes': rm.total_lanes,
            'total_spares': rm.total_spares,
            'repair_entries': entries,
            'status': rm.status.value,
            'failed_lanes': list(rm.failed_lanes),
        }

    def import_repair_map(self, data: Dict[str, Any]) -> bool:
        """Import repair map from serialized data.

        Restores repair state from a previously exported repair map.
        Useful for loading manufacturing test results.

        Args:
            data: Dictionary from export_repair_map()

        Returns:
            True if import succeeded
        """
        try:
            channel_id = data['channel_id']
            total_lanes = data['total_lanes']
            total_spares = data['total_spares']

            # Configure channel
            self.configure_channel(channel_id, total_lanes, total_spares)

            # Reset existing state
            self.reset_channel(channel_id)

            # Restore repairs
            for entry_data in data['repair_entries']:
                failed_lane = entry_data['failed_lane']
                spare_lane = entry_data['spare_lane']
                repair_type = entry_data['repair_type']

                self.add_failed_lane(channel_id, failed_lane)
                self.allocate_spare(channel_id, failed_lane, spare_lane, repair_type)

            return True
        except (KeyError, TypeError):
            return False

    def verify_repair_integrity(self, channel_id: int) -> Dict[str, Any]:
        """Verify repair state integrity for a channel.

        Checks that repair mappings are internally consistent:
          - No duplicate failed lanes
          - No duplicate spare lanes
          - All spare lanes are valid (in spare range)
          - Repair count matches entry count

        Args:
            channel_id: Channel to verify

        Returns:
            Dictionary with:
              - valid: Boolean indicating if state is valid
              - errors: List of error strings (empty if valid)
              - warnings: List of warning strings
        """
        if channel_id not in self._repair_maps:
            return {'valid': False, 'errors': ['Channel not found'], 'warnings': []}

        rm = self._repair_maps[channel_id]
        errors = []
        warnings = []

        # Check for duplicate failed lanes
        if len(rm.failed_lanes) != len(set(rm.failed_lanes)):
            errors.append('Duplicate failed lanes detected')

        # Check for duplicate spare lanes
        if len(rm.spare_lanes) != len(set(rm.spare_lanes)):
            errors.append('Duplicate spare lanes detected')

        # Check spare lane range
        spare_base = rm.total_lanes
        spare_top = rm.total_lanes + rm.total_spares
        for spare in rm.spare_lanes:
            if spare < spare_base or spare >= spare_top:
                errors.append(f'Invalid spare lane {spare} (valid range: {spare_base}-{spare_top-1})')

        # Check failed lane range
        for failed in rm.failed_lanes:
            if failed < 0 or failed >= rm.total_lanes:
                errors.append(f'Invalid failed lane {failed} (valid range: 0-{rm.total_lanes-1})')

        # Check repair count matches
        if rm.repair_count != len(rm.repair_entries):
            errors.append(f'Repair count mismatch: {rm.repair_count} != {len(rm.repair_entries)}')

        # Check entry consistency
        for entry in rm.repair_entries:
            if entry.channel_id != channel_id:
                errors.append(f'Entry channel mismatch: {entry.channel_id} != {channel_id}')

        # Warnings
        if rm.available_spares == 0 and rm.status != RepairStatus.UNREPAIRABLE:
            warnings.append('All spares used')

        if rm.status == RepairStatus.UNREPAIRABLE:
            warnings.append('Channel marked unrepairable')

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
        }