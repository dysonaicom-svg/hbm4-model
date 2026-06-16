"""
Tests for DRAM Channel Model
"""

import pytest
from model.dram.channel_model import Channel, ChannelArray
from model.dram.bank_state_machine import BankStateMachine, BankStateEnum
from model.dram.timing import HBM3Timing


class TestChannel:
    """Test single Channel"""

    def test_channel_creation(self):
        """测试 Channel 创建"""
        channel = Channel(channel_id=0)

        assert channel.channel_id == 0
        assert len(channel.pseudo_channels) == 2
        assert all(hasattr(ps, 'bank_groups') for ps in channel.pseudo_channels)

    def test_channel_get_pseudo_channel(self):
        """测试获取 pseudo-channel"""
        channel = Channel(channel_id=0)

        ps0 = channel.get_pseudo_channel(0)
        ps1 = channel.get_pseudo_channel(1)

        assert ps0.pseudo_id == 0
        assert ps1.pseudo_id == 1

    def test_channel_get_bank(self):
        """测试获取 bank"""
        channel = Channel(channel_id=0)

        # HBM3: 4 bank groups, 2 banks per group
        bank = channel.get_bank(ps_id=0, bg_id=0, bank_id=0)
        assert isinstance(bank, BankStateMachine)

    def test_channel_activate(self):
        """测试激活 bank"""
        channel = Channel(channel_id=0)

        channel.set_time(0.0)
        success = channel.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x100)

        assert success
        bank = channel.get_bank(0, 0, 0)
        assert bank.bank.is_active
        assert bank.bank.open_row == 0x100

    def test_channel_read(self):
        """测试读操作"""
        channel = Channel(channel_id=0)
        timing = HBM3Timing()

        channel.set_time(0.0)
        channel.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x100)

        # 等待 tRCD
        trcd_s = timing.cycles_to_s(timing.tRCD)
        channel.set_time(trcd_s + 0.000001)

        success = channel.execute_command(ps_id=0, cmd="RD", bg_id=0, bank_id=0)
        assert success

    def test_channel_write(self):
        """测试写操作"""
        channel = Channel(channel_id=0)
        timing = HBM3Timing()

        channel.set_time(0.0)
        channel.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x100)

        # 等待 tRCD
        trcd_s = timing.cycles_to_s(timing.tRCD)
        channel.set_time(trcd_s + 0.000001)

        success = channel.execute_command(ps_id=0, cmd="WR", bg_id=0, bank_id=0)
        assert success

    def test_channel_precharge(self):
        """测试预充电"""
        channel = Channel(channel_id=0)
        timing = HBM3Timing()

        channel.set_time(0.0)
        channel.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x100)

        # 等待 tRAS
        tras_s = timing.cycles_to_s(timing.tRAS)
        channel.set_time(tras_s + 0.000001)

        success = channel.execute_command(ps_id=0, cmd="PRE", bg_id=0, bank_id=0)
        assert success
        bank = channel.get_bank(0, 0, 0)
        assert bank.bank.is_idle

    def test_channel_row_hit_detection(self):
        """测试行命中检测"""
        channel = Channel(channel_id=0)

        channel.set_time(0.0)
        channel.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x100)

        # 同一行应该命中
        assert channel.is_row_hit(ps_id=0, bg_id=0, bank_id=0, row=0x100)

        # 不同行应该冲突
        assert not channel.is_row_hit(ps_id=0, bg_id=0, bank_id=0, row=0x200)


class TestChannelArray:
    """Test Channel Array (multiple channels)"""

    def test_channel_array_creation(self):
        """测试 ChannelArray 创建"""
        array = ChannelArray(num_channels=8)

        assert array.num_channels == 8
        assert len(array.channels) == 8
        assert all(isinstance(c, Channel) for c in array.channels)

    def test_channel_array_get_channel(self):
        """测试获取 channel"""
        array = ChannelArray(num_channels=8)

        ch = array.get_channel(3)
        assert ch.channel_id == 3

    def test_channel_array_activate(self):
        """测试跨 channel 激活"""
        array = ChannelArray(num_channels=8)

        array.set_time(0.0)

        # 激活不同 channel
        ch0 = array.get_channel(0)
        ch1 = array.get_channel(1)

        success0 = ch0.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x100)
        success1 = ch1.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x200)

        assert success0
        assert success1

    def test_channel_array_row_hit(self):
        """测试跨 channel 行命中"""
        array = ChannelArray(num_channels=8)

        array.set_time(0.0)

        ch0 = array.get_channel(0)
        ch0.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x100)

        # 同 channel 同行应该命中
        assert array.is_row_hit(0, 0, 0, 0, 0x100)

        # 同 channel 不同行应该冲突
        assert not array.is_row_hit(0, 0, 0, 0, 0x200)

        # 不同 channel 应该不冲突
        assert not array.is_row_hit(1, 0, 0, 0, 0x100)

    def test_channel_array_independent_channels(self):
        """测试 channel 独立性"""
        array = ChannelArray(num_channels=2)

        array.set_time(0.0)

        ch0 = array.get_channel(0)
        ch1 = array.get_channel(1)

        # Channel 0 激活
        ch0.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x100)

        # Channel 1 也可以激活（独立）
        ch1.execute_command(ps_id=0, cmd="ACT", bg_id=0, bank_id=0, row=0x200)

        ch0_bank = array.get_bank(0, 0, 0, 0)
        ch1_bank = array.get_bank(1, 0, 0, 0)

        assert ch0_bank.bank.is_active
        assert ch1_bank.bank.is_active