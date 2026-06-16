"""
Tests for Request Queue
"""

import pytest
import time
from model.controller.queue import RequestQueue, ReadQueue, WriteQueue, QueueManager
from model.controller.request import HBMRequest


class TestRequestQueue:
    """Test RequestQueue base class"""

    def test_queue_creation(self):
        """测试队列创建"""
        queue = RequestQueue(max_depth=16, name="TestQueue")
        assert queue.max_depth == 16
        assert queue.size() == 0
        assert queue.is_empty()
        assert not queue.is_full()

    def test_push_pop(self):
        """测试入队出队"""
        queue = RequestQueue(max_depth=4)

        req1 = HBMRequest(addr=0x1000, length=64, is_read=True)
        req2 = HBMRequest(addr=0x2000, length=64, is_read=True)

        assert queue.push(req1)
        assert queue.size() == 1

        assert queue.push(req2)
        assert queue.size() == 2

        popped = queue.pop()
        assert popped.request_id == req1.request_id
        assert queue.size() == 1

    def test_full_queue(self):
        """测试队列满"""
        queue = RequestQueue(max_depth=2)

        req1 = HBMRequest(addr=0x1000, length=64, is_read=True)
        req2 = HBMRequest(addr=0x2000, length=64, is_read=True)
        req3 = HBMRequest(addr=0x3000, length=64, is_read=True)

        assert queue.push(req1)
        assert queue.push(req2)
        assert not queue.push(req3)  # 应该失败

    def test_empty_pop(self):
        """测试空队列出队"""
        queue = RequestQueue(max_depth=4)
        assert queue.pop() is None

    def test_remove(self):
        """测试移除请求"""
        queue = RequestQueue(max_depth=4)

        req1 = HBMRequest(addr=0x1000, length=64, is_read=True)
        req2 = HBMRequest(addr=0x2000, length=64, is_read=True)

        queue.push(req1)
        queue.push(req2)

        assert queue.remove(req1.request_id)
        assert queue.size() == 1

        assert not queue.remove(9999)  # 不存在的 ID

    def test_clear(self):
        """测试清空队列"""
        queue = RequestQueue(max_depth=4)

        for i in range(3):
            queue.push(HBMRequest(addr=0x1000 + i, length=64, is_read=True))

        assert queue.size() == 3
        queue.clear()
        assert queue.size() == 0

    def test_stats(self):
        """测试统计"""
        queue = RequestQueue(max_depth=4)

        queue.push(HBMRequest(addr=0x1000, length=64, is_read=True))
        queue.push(HBMRequest(addr=0x2000, length=64, is_read=True))
        queue.pop()

        stats = queue.get_stats()
        assert stats['push_count'] == 2
        assert stats['pop_count'] == 1
        assert stats['max_occupancy'] == 2


class TestReadQueue:
    """Test ReadQueue"""

    def test_get_best_request(self):
        """测试获取最佳请求 (FR-FCFS)"""
        queue = ReadQueue(max_depth=16)

        req1 = HBMRequest(addr=0x1000, length=64, is_read=True)
        req2 = HBMRequest(addr=0x2000, length=64, is_read=True)

        time.sleep(0.001)
        queue.push(req2)
        queue.push(req1)  # req1 更早到达

        best = queue.get_best_request()
        assert best.request_id == req1.request_id

    def test_get_oldest_request(self):
        """测试获取最老请求"""
        queue = ReadQueue(max_depth=16)

        req1 = HBMRequest(addr=0x1000, length=64, is_read=True)
        time.sleep(0.001)
        req2 = HBMRequest(addr=0x2000, length=64, is_read=True)

        queue.push(req2)
        queue.push(req1)

        oldest = queue.get_oldest_request()
        assert oldest.request_id == req1.request_id


class TestWriteQueue:
    """Test WriteQueue"""

    def test_should_drain(self):
        """测试写队列耗尽检测"""
        queue = WriteQueue(max_depth=10, drain_threshold=0.8)

        # 80% 以下不应该 drain
        for i in range(7):
            queue.push(HBMRequest(addr=0x1000 + i, length=64, is_read=False))
        assert not queue.should_drain()

        # 80% 以上应该 drain
        queue.push(HBMRequest(addr=0x2000, length=64, is_read=False))
        assert queue.should_drain()

    def test_pending_bytes(self):
        """测试待写入字节统计"""
        queue = WriteQueue(max_depth=16)

        queue.push(HBMRequest(addr=0x1000, length=64, is_read=False))
        queue.push(HBMRequest(addr=0x2000, length=128, is_read=False))

        assert queue.get_pending_bytes() == 192


class TestQueueManager:
    """Test QueueManager"""

    def test_create(self):
        """测试创建"""
        manager = QueueManager.create(queue_depth=32)
        assert manager.read_queue.max_depth == 32
        assert manager.write_queue.max_depth == 32

    def test_push_read_write(self):
        """测试入队"""
        manager = QueueManager.create()

        read_req = HBMRequest(addr=0x1000, length=64, is_read=True)
        write_req = HBMRequest(addr=0x2000, length=64, is_read=False)

        assert manager.push_read(read_req)
        assert manager.push_write(write_req)
        assert manager.total_size() == 2

    def test_is_full(self):
        """测试队列满检测"""
        manager = QueueManager.create(queue_depth=2)

        for i in range(2):
            manager.push_read(HBMRequest(addr=0x1000 + i, length=64, is_read=True))

        assert manager.is_full()

    def test_stats(self):
        """测试统计"""
        manager = QueueManager.create()

        manager.push_read(HBMRequest(addr=0x1000, length=64, is_read=True))
        manager.push_write(HBMRequest(addr=0x2000, length=64, is_read=False))

        stats = manager.get_stats()
        assert stats['total']['size'] == 2