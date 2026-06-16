"""
Tests for Scheduler
"""

import pytest
from model.controller.scheduler import (
    HBMScheduler, FRFCFSScheduler, BankState, SchedulerStats
)
from model.controller.queue import ReadQueue, WriteQueue
from model.controller.request import HBMRequest
from model.controller.config import HBMConfig


class TestBankState:
    """Test BankState"""

    def test_bank_state_creation(self):
        """测试 Bank 状态创建"""
        bank = BankState(bank_id=0)
        assert bank.bank_id == 0
        assert not bank.is_open
        assert bank.open_row == -1


class TestFRFCFSScheduler:
    """Test FR-FCFS Scheduler"""

    def test_scheduler_creation(self):
        """测试调度器创建"""
        config = HBMConfig()
        scheduler = FRFCFSScheduler(config)
        assert scheduler.rd_priority == 1.0
        assert scheduler.wr_priority == 1.0

    def test_schedule_empty_queues(self):
        """测试空队列调度"""
        config = HBMConfig()
        scheduler = FRFCFSScheduler(config)

        read_queue = ReadQueue()
        write_queue = WriteQueue()

        result = scheduler.schedule(
            read_queue, write_queue, {}, 0.0
        )
        assert result is None

    def test_schedule_read_request(self):
        """测试读请求调度"""
        config = HBMConfig()
        scheduler = FRFCFSScheduler(config)

        read_queue = ReadQueue()
        write_queue = WriteQueue()

        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        read_queue.push(req)

        result = scheduler.schedule(
            read_queue, write_queue, {}, 0.001
        )

        assert result is not None
        assert result.request_id == req.request_id

    def test_schedule_write_request(self):
        """测试写请求调度"""
        config = HBMConfig()
        scheduler = FRFCFSScheduler(config)

        read_queue = ReadQueue()
        write_queue = WriteQueue()

        req = HBMRequest(addr=0x1000, length=64, is_read=False)
        write_queue.push(req)

        result = scheduler.schedule(
            read_queue, write_queue, {}, 0.001
        )

        assert result is not None
        assert result.request_id == req.request_id

    def test_row_hit_priority(self):
        """测试 row-hit 优先级"""
        config = HBMConfig()
        scheduler = FRFCFSScheduler(config)

        read_queue = ReadQueue()
        write_queue = WriteQueue()

        # 创建两个请求
        req1 = HBMRequest(addr=0x1000, length=64, is_read=True)
        req2 = HBMRequest(addr=0x2000, length=64, is_read=True)

        read_queue.push(req1)
        read_queue.push(req2)

        # 模拟一个 bank 打开同一行
        bank_states = {
            (0, 0, 0): BankState(bank_id=0, is_open=True, open_row=0x10)
        }

        result = scheduler.schedule(
            read_queue, write_queue, bank_states, 0.001
        )

        assert result is not None


class TestSchedulerStats:
    """Test SchedulerStats"""

    def test_stats_initialization(self):
        """测试统计初始化"""
        stats = SchedulerStats()
        assert stats.schedule_count == 0
        assert stats.row_hit_rate == 0.0

    def test_record_schedule(self):
        """测试记录调度"""
        stats = SchedulerStats()
        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        req.row_hit = True

        stats.record_schedule(req)
        assert stats.schedule_count == 1
        assert stats.row_hit_count == 1
        assert stats.read_count == 1

    def test_row_hit_rate(self):
        """测试 row-hit 率计算"""
        stats = SchedulerStats()

        for _ in range(3):
            req = HBMRequest(addr=0x1000, length=64, is_read=True)
            req.row_hit = True
            stats.record_schedule(req)

        for _ in range(7):
            req = HBMRequest(addr=0x1000, length=64, is_read=True)
            req.row_hit = False
            stats.record_schedule(req)

        assert stats.row_hit_rate == 0.3  # 3/10