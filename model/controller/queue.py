"""
HBM Request Queues
参考设计文档 2026-06-15-hbm-system-model-design.md 的 5.1.4 节

实现线程安全的请求队列:
- ReadQueue: 读请求队列
- WriteQueue: 写请求队列
"""

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from collections import deque
import time

from model.controller.request import HBMRequest, RequestState
from model.controller.exceptions import QueueOverflowError


class RequestQueue:
    """线程安全的请求队列基类"""
    
    def __init__(self, max_depth: int = 32, name: str = "Queue"):
        """初始化请求队列
        
        Args:
            max_depth: 最大队列深度
            name: 队列名称 (用于调试)
        """
        self.max_depth = max_depth
        self.name = name
        self._queue = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        # 统计
        self._stats = {
            'push_count': 0,
            'pop_count': 0,
            'reject_count': 0,
            'max_occupancy': 0,
        }
    
    def push(self, request: HBMRequest, timeout: float = 0.0) -> bool:
        """入队请求
        
        Args:
            request: HBM 请求
            timeout: 超时时间 (秒), 0 表示不等待
            
        Returns:
            True 如果成功入队, False 如果队列满
            
        Raises:
            QueueOverflowError: 队列满且超时
        """
        with self._not_full:
            if timeout > 0:
                # 带超时等待
                end_time = time.time() + timeout
                while len(self._queue) >= self.max_depth:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        self._stats['reject_count'] += 1
                        return False
                    if not self._not_full.wait(remaining):
                        self._stats['reject_count'] += 1
                        return False
            else:
                # 非阻塞
                if len(self._queue) >= self.max_depth:
                    self._stats['reject_count'] += 1
                    return False
            
            self._queue.append(request)
            self._stats['push_count'] += 1
            self._stats['max_occupancy'] = max(
                self._stats['max_occupancy'], 
                len(self._queue)
            )
            self._not_empty.notify()
            return True
    
    def pop(self, timeout: float = 0.0) -> Optional[HBMRequest]:
        """出队请求
        
        Args:
            timeout: 超时时间 (秒), 0 表示不等待
            
        Returns:
            HBMRequest 如果成功, None 如果队列空
        """
        with self._not_empty:
            if timeout > 0:
                # 带超时等待
                end_time = time.time() + timeout
                while len(self._queue) == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return None
                    if not self._not_empty.wait(remaining):
                        return None
            else:
                # 非阻塞
                if len(self._queue) == 0:
                    return None
            
            request = self._queue.popleft()
            self._stats['pop_count'] += 1
            self._not_full.notify()
            return request
    
    def peek(self) -> Optional[HBMRequest]:
        """查看队首请求 (不移除)"""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None
    
    def remove(self, request_id: int) -> bool:
        """移除指定请求
        
        Args:
            request_id: 请求 ID
            
        Returns:
            True 如果找到并移除
        """
        with self._lock:
            for i, req in enumerate(self._queue):
                if req.request_id == request_id:
                    del self._queue[i]
                    return True
            return False
    
    def size(self) -> int:
        """获取当前队列大小"""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        with self._lock:
            return len(self._queue) == 0
    
    def is_full(self) -> bool:
        """检查队列是否已满"""
        with self._lock:
            return len(self._queue) >= self.max_depth
    
    def clear(self):
        """清空队列"""
        with self._lock:
            self._queue.clear()
            self._not_full.notify_all()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                **self._stats,
                'current_occupancy': len(self._queue),
                'occupancy_rate': len(self._queue) / self.max_depth if self.max_depth > 0 else 0,
            }
    
    def __repr__(self) -> str:
        return f"{self.name}(size={self.size()}, max={self.max_depth})"

    def __iter__(self):
        """使队列可迭代"""
        with self._lock:
            return iter(list(self._queue))

    def __len__(self):
        """使队列可使用 len()"""
        with self._lock:
            return len(self._queue)


class ReadQueue(RequestQueue):
    """读请求队列
    
    特殊功能:
    - 按 row_hit 排序
    - 按时间戳排序 (FR-FCFS)
    """
    
    def __init__(self, max_depth: int = 32):
        super().__init__(max_depth, name="ReadQueue")
    
    def get_row_hit_requests(self) -> List[HBMRequest]:
        """获取所有 row-hit 的请求"""
        with self._lock:
            return [r for r in self._queue if r.row_hit]
    
    def get_oldest_request(self) -> Optional[HBMRequest]:
        """获取最早的请求"""
        with self._lock:
            if not self._queue:
                return None
            return min(self._queue, key=lambda r: r.arrival_time)
    
    def get_best_request(self) -> Optional[HBMRequest]:
        """获取最佳调度的请求 (FR-FCFS)
        
        优先选择:
        1. row-hit 请求中最老的
        2. 如果没有 row-hit，选择最老的请求
        """
        with self._lock:
            row_hit_requests = [r for r in self._queue if r.row_hit]
            if row_hit_requests:
                return min(row_hit_requests, key=lambda r: r.arrival_time)
            if self._queue:
                return min(self._queue, key=lambda r: r.arrival_time)
            return None


class WriteQueue(RequestQueue):
    """写请求队列
    
    特殊功能:
    - Write drain 策略支持
    """
    
    def __init__(self, max_depth: int = 32, drain_threshold: float = 0.8):
        super().__init__(max_depth, name="WriteQueue")
        self.drain_threshold = drain_threshold
    
    def should_drain(self) -> bool:
        """检查是否应该执行 write drain
        
        当写队列达到阈值时返回 True。
        """
        with self._lock:
            return len(self._queue) >= self.max_depth * self.drain_threshold
    
    def get_oldest_request(self) -> Optional[HBMRequest]:
        """获取最早的写请求"""
        with self._lock:
            if not self._queue:
                return None
            return min(self._queue, key=lambda r: r.arrival_time)
    
    def get_pending_bytes(self) -> int:
        """获取队列中待写入的总字节数"""
        with self._lock:
            return sum(r.length for r in self._queue)


@dataclass
class QueueManager:
    """队列管理器
    
    管理读/写队列和调度决策。
    """
    read_queue: ReadQueue
    write_queue: WriteQueue
    
    @classmethod
    def create(cls, queue_depth: int = 32) -> "QueueManager":
        """创建队列管理器
        
        Args:
            queue_depth: 队列深度
            
        Returns:
            QueueManager 实例
        """
        return cls(
            read_queue=ReadQueue(max_depth=queue_depth),
            write_queue=WriteQueue(max_depth=queue_depth),
        )
    
    def push_read(self, request: HBMRequest, timeout: float = 0.0) -> bool:
        """入队读请求"""
        return self.read_queue.push(request, timeout)

    def push_write(self, request: HBMRequest, timeout: float = 0.0) -> bool:
        """入队写请求"""
        return self.write_queue.push(request, timeout)

    def remove_read(self, request_id: int) -> bool:
        """从读队列移除请求"""
        return self.read_queue.remove(request_id)

    def remove_write(self, request_id: int) -> bool:
        """从写队列移除请求"""
        return self.write_queue.remove(request_id)

    def total_size(self) -> int:
        """总队列大小"""
        return self.read_queue.size() + self.write_queue.size()
    
    def is_full(self) -> bool:
        """检查是否任一队列已满"""
        return self.read_queue.is_full() or self.write_queue.is_full()
    
    def get_stats(self) -> dict:
        """获取所有队列统计"""
        return {
            'read': self.read_queue.get_stats(),
            'write': self.write_queue.get_stats(),
            'total': {
                'size': self.total_size(),
                'max_occupancy': max(
                    self.read_queue.get_stats()['max_occupancy'],
                    self.write_queue.get_stats()['max_occupancy'],
                ),
            },
        }
