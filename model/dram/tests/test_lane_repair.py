"""
Tests for HBM4 Lane Repair Model
"""

import pytest
from model.dram.lane_repair import (
    HBM4LaneRepairModel,
    LaneRepairMap,
    LaneRepairEntry,
    RepairStatus,
)


class TestLaneRepairEntry:
    """Test LaneRepairEntry dataclass"""

    def test_entry_creation(self):
        """Test creating a repair entry"""
        entry = LaneRepairEntry(
            failed_lane=5,
            spare_lane=68,
            repair_type="bit",
            channel_id=0,
        )
        assert entry.failed_lane == 5
        assert entry.spare_lane == 68
        assert entry.repair_type == "bit"
        assert entry.channel_id == 0


class TestLaneRepairMap:
    """Test LaneRepairMap dataclass"""

    def test_map_creation(self):
        """Test creating a repair map"""
        rm = LaneRepairMap(
            channel_id=0,
            total_lanes=64,
            total_spares=4,
        )
        assert rm.channel_id == 0
        assert rm.total_lanes == 64
        assert rm.total_spares == 4
        assert rm.max_repair_count == 4

    def test_available_spares(self):
        """Test available spares calculation"""
        rm = LaneRepairMap(channel_id=0, total_lanes=64, total_spares=4)
        assert rm.available_spares == 4

        rm.failed_lanes.append(10)
        rm.failed_lanes.append(20)
        # Spares not yet allocated, just failed lanes tracked
        assert rm.available_spares == 4

    def test_is_repairable(self):
        """Test repairability check"""
        rm = LaneRepairMap(channel_id=0, total_lanes=64, total_spares=4)
        assert rm.is_repairable

        # Add 4 failed lanes (exactly at spare limit, still repairable)
        rm.failed_lanes.extend([1, 2, 3, 4])
        assert rm.is_repairable  # Still repairable, can repair all 4

        # Add 5th failed lane (exceeds spares)
        rm.failed_lanes.append(5)
        assert not rm.is_repairable  # Unrepairable

    def test_is_repairable_with_repairs(self):
        """Test repairability when repairs are performed"""
        rm = LaneRepairMap(channel_id=0, total_lanes=64, total_spares=4)
        # Simulate 3 repairs performed
        rm.repair_count = 3
        assert rm.is_repairable  # Can do 1 more repair

        rm.repair_count = 4
        assert not rm.is_repairable  # All spares used

    def test_status_no_repair(self):
        """Test status with no failures"""
        rm = LaneRepairMap(channel_id=0, total_lanes=64, total_spares=4)
        assert rm.status == RepairStatus.NO_REPAIR

    def test_status_partial_repair(self):
        """Test status with partial repair"""
        rm = LaneRepairMap(channel_id=0, total_lanes=64, total_spares=4)
        rm.failed_lanes.extend([1, 2, 3])
        assert rm.status == RepairStatus.PARTIAL_REPAIR

    def test_status_full_repair(self):
        """Test status with all spares used"""
        rm = LaneRepairMap(channel_id=0, total_lanes=64, total_spares=4)
        rm.failed_lanes.extend([1, 2, 3, 4])
        assert rm.status == RepairStatus.FULL_REPAIR

    def test_status_unrepairable(self):
        """Test status when failures exceed spares"""
        rm = LaneRepairMap(channel_id=0, total_lanes=64, total_spares=4)
        rm.failed_lanes.extend([1, 2, 3, 4, 5])
        assert rm.status == RepairStatus.UNREPAIRABLE


class TestHBM4LaneRepairModel:
    """Test HBM4LaneRepairModel"""

    def test_model_creation(self):
        """Test model creation with defaults"""
        model = HBM4LaneRepairModel()
        assert model.num_channels == 32
        assert model.lanes_per_channel == 64
        assert model.spare_lanes_per_channel == 4

    def test_model_creation_custom(self):
        """Test model creation with custom parameters"""
        model = HBM4LaneRepairModel(
            num_channels=16,
            lanes_per_channel=32,
            spare_lanes_per_channel=2,
        )
        assert model.num_channels == 16
        assert model.lanes_per_channel == 32
        assert model.spare_lanes_per_channel == 2

    def test_add_failed_lane(self):
        """Test adding a failed lane"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        result = model.add_failed_lane(channel_id=0, lane_id=3)
        assert result is True
        assert 3 in model.get_all_failed_lanes(0)

    def test_add_failed_lane_duplicate(self):
        """Test adding duplicate failed lane"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        model.add_failed_lane(channel_id=0, lane_id=3)
        result = model.add_failed_lane(channel_id=0, lane_id=3)
        assert result is True  # Already tracked

    def test_add_failed_lane_invalid_channel(self):
        """Test adding failed lane to invalid channel"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        result = model.add_failed_lane(channel_id=99, lane_id=3)
        assert result is False

    def test_add_failed_lane_unrepairable(self):
        """Test adding failed lane when unrepairable"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        # Add 2 failed lanes (exactly at spare limit)
        model.add_failed_lane(channel_id=0, lane_id=0)
        model.add_failed_lane(channel_id=0, lane_id=1)
        # Status should be FULL_REPAIR (2 failed = 2 spares)
        assert model.get_repair_status(0) == RepairStatus.FULL_REPAIR
        # Channel is still repairable (can repair both failed lanes)
        assert model.get_channel_repair_map(0).is_repairable

        # Try to add 3rd failed lane (exceeds spares, repairable becomes False)
        result = model.add_failed_lane(channel_id=0, lane_id=2)
        # The model allows adding failed lanes but marks channel as unrepairable
        assert model.get_repair_status(0) == RepairStatus.UNREPAIRABLE
        # is_repairable is now False
        assert not model.get_channel_repair_map(0).is_repairable

    def test_perform_repair(self):
        """Test performing a repair"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        spare = model.perform_repair(channel_id=0, failed_lane=3, repair_type="channel")
        assert spare == 8  # First spare after 8 data lanes
        assert model.is_lane_remapped(0, 3)

    def test_perform_repair_multiple(self):
        """Test multiple repairs"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        spare1 = model.perform_repair(channel_id=0, failed_lane=1, repair_type="bit")
        spare2 = model.perform_repair(channel_id=0, failed_lane=5, repair_type="byte")

        assert spare1 == 8  # First spare after 8 data lanes
        assert spare2 == 9  # Second spare


        # Status should be FULL_REPAIR (2 repairs = 2 spares)
        assert model.get_repair_status(0) == RepairStatus.FULL_REPAIR

        # Next repair should fail (all spares used)
        spare3 = model.perform_repair(channel_id=0, failed_lane=7, repair_type="channel")
        assert spare3 is None

    def test_get_remapped_lane(self):
        """Test lane remapping"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        model.perform_repair(channel_id=0, failed_lane=3, repair_type="channel")

        # Repaired lane maps to spare
        assert model.get_remapped_lane(0, 3) == 8
        # Normal lane returns itself
        assert model.get_remapped_lane(0, 5) == 5

    def test_get_remapped_lane_invalid_channel(self):
        """Test remapping for invalid channel"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        assert model.get_remapped_lane(99, 3) == 3

    def test_is_lane_remapped(self):
        """Test checking if lane is remapped"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        assert not model.is_lane_remapped(0, 3)
        model.perform_repair(channel_id=0, failed_lane=3, repair_type="channel")
        assert model.is_lane_remapped(0, 3)

    def test_configure_channel(self):
        """Test configuring a channel"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        model.configure_channel(channel_id=3, lanes=16, spares=4)

        rm = model.get_channel_repair_map(3)
        assert rm.total_lanes == 16
        assert rm.total_spares == 4

    def test_get_repair_status(self):
        """Test getting repair status"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        assert model.get_repair_status(0) == RepairStatus.NO_REPAIR

        model.add_failed_lane(0, 3)
        assert model.get_repair_status(0) == RepairStatus.PARTIAL_REPAIR

    def test_get_channel_stats(self):
        """Test channel statistics"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        model.perform_repair(0, 3, "channel")
        model.perform_repair(0, 5, "bit")

        stats = model.get_channel_stats(0)
        assert stats['failed_lanes'] == 2
        assert stats['repair_count'] == 2
        assert stats['available_spares'] == 0
        assert stats['status'] == 'full_repair'

    def test_get_stats(self):
        """Test global statistics"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        model.perform_repair(0, 3, "channel")
        model.perform_repair(1, 5, "byte")

        stats = model.get_stats()
        assert stats['total_channels'] == 4
        assert stats['lanes_per_channel'] == 8
        assert stats['spares_per_channel'] == 2
        assert stats['total_repairs'] == 2
        assert stats['total_failed_lanes'] == 2
        assert stats['channels_with_repairs'] == 2

    def test_reset_channel(self):
        """Test resetting a channel"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        model.perform_repair(0, 3, "channel")
        model.reset_channel(0)

        assert model.get_repair_status(0) == RepairStatus.NO_REPAIR
        assert len(model.get_all_failed_lanes(0)) == 0

    def test_reset_all(self):
        """Test resetting all channels"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        model.perform_repair(0, 3, "channel")
        model.perform_repair(1, 5, "byte")
        model.perform_repair(2, 7, "channel")

        model.reset_all()

        for ch in range(4):
            assert model.get_repair_status(ch) == RepairStatus.NO_REPAIR

    def test_simulate_yield_loss(self):
        """Test yield loss simulation"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        # Use deterministic random for testing
        import random
        random.seed(42)

        failed = model.simulate_yield_loss(0, failure_rate=0.5)
        # With 50% rate and 8 lanes, expect ~4 failures
        assert 0 <= failed <= 8


class TestLaneRemapping:
    """Test lane remapping functionality"""

    def test_remap_preserves_traffic(self):
        """Test that remapping preserves traffic integrity"""
        model = HBM4LaneRepairModel(num_channels=2, lanes_per_channel=16, spare_lanes_per_channel=4)

        # Repair lanes 3, 7, 12
        model.perform_repair(0, 3, "channel")
        model.perform_repair(0, 7, "channel")
        model.perform_repair(0, 12, "channel")

        # Verify remapping
        assert model.get_remapped_lane(0, 3) == 16
        assert model.get_remapped_lane(0, 7) == 17
        assert model.get_remapped_lane(0, 12) == 18

        # Verify non-repaired lanes unchanged
        for lane in [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15]:
            assert model.get_remapped_lane(0, lane) == lane

    def test_remap_across_channels(self):
        """Test that remapping is channel-specific"""
        model = HBM4LaneRepairModel(num_channels=2, lanes_per_channel=16, spare_lanes_per_channel=4)

        model.perform_repair(0, 5, "channel")
        model.perform_repair(1, 5, "channel")

        # Lane 5 repaired differently in each channel
        assert model.get_remapped_lane(0, 5) == 16
        assert model.get_remapped_lane(1, 5) == 16


class TestRepairSequenceGeneration:
    """Test repair sequence generation for eFuse programming"""

    def test_generate_repair_sequence(self):
        """Test generating repair sequence"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        model.perform_repair(0, 5, "bit")
        model.perform_repair(0, 10, "byte")

        sequence = model.generate_repair_sequence(0)
        assert sequence is not None
        assert len(sequence) == 2

        assert sequence[0]['failed_lane'] == 5
        assert sequence[0]['spare_lane'] == 16
        assert sequence[0]['repair_type'] == 'bit'
        assert isinstance(sequence[0]['encoding'], int)

    def test_generate_repair_sequence_no_repairs(self):
        """Test generating sequence for channel with no repairs"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        sequence = model.generate_repair_sequence(0)
        assert sequence is None

    def test_generate_repair_sequence_invalid_channel(self):
        """Test generating sequence for invalid channel"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        sequence = model.generate_repair_sequence(99)
        assert sequence is None

    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding repair entries"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)

        # Encode
        entry = LaneRepairEntry(failed_lane=7, spare_lane=20, repair_type="byte", channel_id=2)
        encoding = model._encode_repair_entry(entry)

        # Decode
        decoded = model.decode_repair_entry(encoding)

        assert decoded['failed_lane'] == 7
        assert decoded['spare_lane'] == 20
        assert decoded['repair_type'] == 'byte'
        assert decoded['channel_id'] == 2

    def test_encode_channel_type_mapping(self):
        """Test encoding for different repair types"""
        model = HBM4LaneRepairModel(num_channels=1, lanes_per_channel=8, spare_lanes_per_channel=2)

        for repair_type, expected_type_val in [('bit', 0), ('byte', 1), ('channel', 2)]:
            entry = LaneRepairEntry(failed_lane=1, spare_lane=8, repair_type=repair_type, channel_id=0)
            encoding = model._encode_repair_entry(entry)
            type_val = (encoding >> 24) & 0xFF
            assert type_val == expected_type_val

    def test_generate_bulk_repair_sequence(self):
        """Test generating bulk repair sequence"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        model.perform_repair(0, 5, "bit")
        model.perform_repair(1, 10, "channel")
        model.perform_repair(2, 3, "byte")

        bulk = model.generate_bulk_repair_sequence()
        assert len(bulk) == 3
        assert 0 in bulk
        assert 1 in bulk
        assert 2 in bulk

    def test_export_repair_map(self):
        """Test exporting repair map"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        model.perform_repair(0, 5, "bit")
        model.perform_repair(0, 10, "byte")

        exported = model.export_repair_map(0)
        assert exported is not None
        assert exported['channel_id'] == 0
        assert exported['total_lanes'] == 16
        assert exported['total_spares'] == 4
        # Status is PARTIAL_REPAIR (2 repairs out of 4 spares)
        assert exported['status'] == 'partial_repair'
        assert len(exported['repair_entries']) == 2
        assert len(exported['failed_lanes']) == 2

    def test_import_repair_map(self):
        """Test importing repair map"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)

        # Export from one model
        model.perform_repair(0, 5, "bit")
        model.perform_repair(0, 10, "byte")
        exported = model.export_repair_map(0)

        # Create new model and import
        new_model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        result = new_model.import_repair_map(exported)
        assert result is True

        # Verify imported state
        assert new_model.get_remapped_lane(0, 5) == 16
        assert new_model.get_remapped_lane(0, 10) == 17

    def test_import_repair_map_invalid(self):
        """Test importing invalid repair map"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        result = model.import_repair_map({})  # Empty dict
        assert result is False


class TestRepairIntegrity:
    """Test repair state integrity verification"""

    def test_verify_valid_state(self):
        """Test verifying valid repair state"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        model.perform_repair(0, 5, "bit")
        model.perform_repair(0, 10, "byte")

        result = model.verify_repair_integrity(0)
        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_verify_invalid_channel(self):
        """Test verifying invalid channel"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        result = model.verify_repair_integrity(99)
        assert result['valid'] is False
        assert 'Channel not found' in result['errors']

    def test_verify_unrepairable_warning(self):
        """Test warning for unrepairable channel"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=16, spare_lanes_per_channel=4)
        model.add_failed_lane(0, 0)
        model.add_failed_lane(0, 1)
        model.add_failed_lane(0, 2)
        model.add_failed_lane(0, 3)
        model.add_failed_lane(0, 4)  # 5th failure - unrepairable

        result = model.verify_repair_integrity(0)
        # Unrepairable status should trigger warning
        assert 'Channel marked unrepairable' in result['warnings']


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_all_spares_used(self):
        """Test behavior when all spares are used"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)

        # Use all spares
        model.perform_repair(0, 0, "channel")
        model.perform_repair(0, 1, "channel")

        # Status should be FULL_REPAIR
        assert model.get_repair_status(0) == RepairStatus.FULL_REPAIR

        # Next repair should fail
        spare = model.perform_repair(0, 2, "channel")
        assert spare is None

    def test_channel_boundary(self):
        """Test at channel boundaries"""
        model = HBM4LaneRepairModel(num_channels=2, lanes_per_channel=8, spare_lanes_per_channel=2)

        # First channel
        model.perform_repair(0, 0, "bit")
        assert model.get_remapped_lane(0, 0) == 8

        # Second channel
        model.perform_repair(1, 0, "bit")
        assert model.get_remapped_lane(1, 0) == 8  # Same lane index, different channel

    def test_lane_boundary(self):
        """Test at lane index boundaries"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)

        # First lane
        model.perform_repair(0, 0, "bit")
        assert model.get_remapped_lane(0, 0) == 8

        # Last data lane
        model.perform_repair(0, 7, "bit")
        assert model.get_remapped_lane(0, 7) == 9

    def test_repair_same_lane_twice(self):
        """Test repairing same lane twice returns same spare"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        spare1 = model.perform_repair(0, 5, "bit")
        # Repair same lane again - should return same spare, not allocate new one
        spare2 = model.perform_repair(0, 5, "bit")
        assert spare2 == spare1  # Same spare returned

        # Stats should show only one repair
        stats = model.get_channel_stats(0)
        assert stats['repair_count'] == 1

        # Second spare should still be available
        assert stats['available_spares'] == 1
        assert stats['repair_count'] == 1

    def test_allocate_spare_directly(self):
        """Test direct spare allocation"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        model.add_failed_lane(0, 5)
        result = model.allocate_spare(0, 5, 8, "channel")
        assert result is True
        assert model.get_remapped_lane(0, 5) == 8

    def test_allocate_spare_already_used(self):
        """Test allocating already used spare"""
        model = HBM4LaneRepairModel(num_channels=4, lanes_per_channel=8, spare_lanes_per_channel=2)
        model.add_failed_lane(0, 5)
        model.allocate_spare(0, 5, 8, "channel")

        # Try to allocate same spare again
        model.add_failed_lane(0, 6)
        result = model.allocate_spare(0, 6, 8, "channel")
        assert result is False