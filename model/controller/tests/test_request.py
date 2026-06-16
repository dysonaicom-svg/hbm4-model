"""
Tests for HBM Request
"""

import pytest
import time
from model.controller.request import (
    HBMRequest, RequestState
)


class TestHBMRequest:
    """Test HBM Request"""

    def test_create_read_request(self):
        """测试创建读请求"""
        req = HBMRequest(addr=0x1000, length=64, is_read=True, qos=8)
        assert req.is_read
        assert req.addr == 0x1000
        assert req.length == 64
        assert req.qos == 8
        assert req.state == RequestState.PENDING

    def test_create_write_request(self):
        """测试创建写请求"""
        req = HBMRequest(addr=0x2000, length=64, is_read=False, qos=8)
        assert not req.is_read
        assert req.addr == 0x2000
        assert req.length == 64
        assert req.state == RequestState.PENDING

    def test_request_id_unique(self):
        """测试请求 ID 唯一性"""
        req1 = HBMRequest(addr=0x1000, length=64, is_read=True)
        req2 = HBMRequest(addr=0x2000, length=64, is_read=True)
        assert req1.request_id != req2.request_id

    def test_mark_scheduled(self):
        """测试标记为已调度"""
        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        current_time = time.time()
        req.mark_scheduled(current_time)
        assert req.state == RequestState.SCHEDULED
        assert req.scheduled_time == current_time

    def test_mark_completed(self):
        """测试标记为完成"""
        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        current_time = time.time()
        req.mark_scheduled(current_time)
        req.mark_completed(current_time + 0.001)
        assert req.state == RequestState.COMPLETED

    def test_mark_failed(self):
        """测试标记为失败"""
        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        req.mark_failed()
        assert req.state == RequestState.FAILED

    def test_latency_calculation(self):
        """测试延迟计算"""
        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        # 保存到达时间
        arrival = req.arrival_time
        req.mark_scheduled(arrival + 0.001)
        req.mark_completed(arrival + 0.006)
        latency = req.latency
        assert 0.004 <= latency <= 0.007  # ~5ms

    def test_is_completed(self):
        """测试完成检查"""
        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        assert not req.is_completed
        req.mark_completed(time.time())
        assert req.is_completed

    def test_is_pending(self):
        """测试等待检查"""
        req = HBMRequest(addr=0x1000, length=64, is_read=True)
        assert req.is_pending
        req.mark_scheduled(time.time())
        assert not req.is_pending

    def test_repr(self):
        """测试表示"""
        req = HBMRequest(addr=0x1000, length=64, is_read=True, qos=8)
        repr_str = repr(req)
        assert "READ" in repr_str
        assert "1000" in repr_str