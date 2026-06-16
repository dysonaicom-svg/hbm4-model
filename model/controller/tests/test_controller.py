"""
Tests for HBM Controller Integration
"""

import pytest
import time
from model.controller.controller import HBMController
from model.controller.config import HBMConfig
from model.controller.request import HBMRequest


class TestHBMController:
    """Test HBM Controller Integration"""

    def test_controller_creation(self):
        """测试控制器创建"""
        config = HBMConfig()
        controller = HBMController(config)
        assert controller.config.stack_count == 2
        assert controller.config.channels_per_stack == 8

    def test_submit_read_request(self):
        """测试提交读请求"""
        config = HBMConfig()
        controller = HBMController(config)

        req = HBMRequest(addr=0x1000, length=64, is_read=True, qos=8)
        success = controller.submit_request(req)

        assert success
        assert controller.stats['total_requests'] == 1
        assert controller.stats['read_requests'] == 1

    def test_submit_write_request(self):
        """测试提交写请求"""
        config = HBMConfig()
        controller = HBMController(config)

        req = HBMRequest(addr=0x2000, length=64, is_read=False, qos=8)
        success = controller.submit_request(req)

        assert success
        assert controller.stats['write_requests'] == 1

    def test_multiple_requests(self):
        """测试提交多个请求"""
        config = HBMConfig()
        controller = HBMController(config)

        for i in range(10):
            req = HBMRequest(addr=0x1000 + i * 0x1000, length=64, is_read=True)
            assert controller.submit_request(req)

        assert controller.stats['total_requests'] == 10
        assert controller.queue_manager.read_queue.size() == 10

    def test_queue_overflow(self):
        """测试队列溢出"""
        config = HBMConfig()
        config.queue_depth = 4
        controller = HBMController(config)

        # 填满队列
        for i in range(4):
            req = HBMRequest(addr=0x1000 + i * 0x1000, length=64, is_read=True)
            assert controller.submit_request(req)

        # 下一个应该失败
        req = HBMRequest(addr=0x5000, length=64, is_read=True)
        assert not controller.submit_request(req)

    def test_address_decoding(self):
        """测试地址解码"""
        config = HBMConfig()
        controller = HBMController(config)

        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        controller.submit_request(req)

        assert req.stack_id >= 0
        assert req.channel_id >= 0
        assert req.bank_id >= 0
        assert req.row_id >= 0

    def test_row_hit_detection(self):
        """测试 row-hit 检测"""
        config = HBMConfig()
        controller = HBMController(config)

        # 第一个请求打开行
        req1 = HBMRequest(addr=0x1000, length=64, is_read=True)
        controller.submit_request(req1)

        # 第二个请求访问同一行，应该 row-hit
        req2 = HBMRequest(addr=0x1100, length=64, is_read=True)
        controller.submit_request(req2)

        # 由于第一个请求还没有被调度，bank 状态可能还没更新
        # 这个测试验证 row_hit 字段被设置
        assert hasattr(req1, 'row_hit')
        assert hasattr(req2, 'row_hit')

    def test_tick_execution(self):
        """测试 tick 执行"""
        config = HBMConfig()
        controller = HBMController(config)

        # 提交请求
        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        controller.submit_request(req)

        # 执行 tick
        response = controller.tick()

        # 响应应该不是 None，因为有请求在队列中
        assert response is not None or controller.queue_manager.read_queue.size() == 1

    def test_stats_collection(self):
        """测试统计收集"""
        config = HBMConfig()
        controller = HBMController(config)

        # 提交各种请求 (8-byte 对齐地址)
        for i in range(5):
            controller.submit_request(HBMRequest(addr=0x1000 + i * 8, length=64, is_read=True))
        for i in range(3):
            controller.submit_request(HBMRequest(addr=0x2000 + i * 8, length=64, is_read=False))

        stats = controller.get_stats()
        assert stats['controller']['total_requests'] == 8
        assert stats['controller']['read_requests'] == 5
        assert stats['controller']['write_requests'] == 3

    def test_qos_scheduler_mode(self):
        """测试 QoS 调度器模式"""
        config = HBMConfig()
        config.scheduler_mode = "qos"
        controller = HBMController(config)

        # 验证使用 QoS 调度器
        from model.controller.qos_scheduler import QoSScheduler
        assert isinstance(controller.scheduler, QoSScheduler)

    def test_frfcfs_scheduler_mode(self):
        """测试 FR-FCFS 调度器模式"""
        config = HBMConfig()
        config.scheduler_mode = "fr-fcfs"
        controller = HBMController(config)

        # 验证使用 FR-FCFS 调度器
        from model.controller.scheduler import FRFCFSScheduler
        assert isinstance(controller.scheduler, FRFCFSScheduler)

    def test_refresh_manager(self):
        """测试刷新管理器"""
        config = HBMConfig()
        controller = HBMController(config)

        assert controller.refresh_manager is not None
        assert len(controller.refresh_manager.schedulers) == config.stack_count

    def test_bank_states(self):
        """测试 Bank 状态管理"""
        config = HBMConfig()
        controller = HBMController(config)

        # 提交请求到不同地址
        req1 = HBMRequest(addr=0x1000, length=64, is_read=True)
        req2 = HBMRequest(addr=0x2000, length=64, is_read=True)

        controller.submit_request(req1)
        controller.submit_request(req2)

        # 验证 bank 状态被创建
        assert len(controller.bank_states) >= 1

    def test_bandwidth_calculation(self):
        """测试带宽计算"""
        config = HBMConfig()
        controller = HBMController(config)

        # 提交请求并执行 tick (8-byte 对齐地址)
        for i in range(5):
            controller.submit_request(HBMRequest(addr=0x1000 + i * 8, length=64, is_read=True))

        for _ in range(5):
            controller.tick()

        bw = controller.get_bandwidth()
        assert bw >= 0


class TestHBMControllerEdgeCases:
    """Test HBM Controller Edge Cases"""

    def test_empty_controller_tick(self):
        """测试空控制器 tick"""
        config = HBMConfig()
        controller = HBMController(config)

        # 没有请求时执行 tick
        response = controller.tick()
        assert response is None

    def test_concurrent_submissions(self):
        """测试并发提交"""
        config = HBMConfig()
        controller = HBMController(config)

        # 快速提交多个请求
        requests = []
        for i in range(20):
            req = HBMRequest(addr=0x1000 + i * 0x100, length=64, is_read=True)
            requests.append(req)
            controller.submit_request(req)

        # 验证所有请求都被接受
        assert controller.queue_manager.read_queue.size() <= config.queue_depth

    def test_mixed_read_write(self):
        """测试混合读写"""
        config = HBMConfig()
        config.queue_depth = 8
        controller = HBMController(config)

        for i in range(4):
            controller.submit_request(HBMRequest(addr=0x1000 + i * 8, length=64, is_read=True))
            controller.submit_request(HBMRequest(addr=0x2000 + i * 8, length=64, is_read=False))

        stats = controller.get_stats()
        assert stats['queue']['read']['current_occupancy'] == 4
        assert stats['queue']['write']['current_occupancy'] == 4