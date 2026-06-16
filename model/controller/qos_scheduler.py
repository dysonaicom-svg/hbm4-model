"""
HBM QoS Scheduler
参考设计文档 2026-06-15-hbm-system-model-design.md 的 5.1.3 节

带带宽保证的 QoS 调度器:
- 16 个优先级 (0-15)
- 每个优先级有带宽保证和上限
- 带宽追踪
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time

from model.controller.config import HBMConfig
from model.controller.request import HBMRequest, RequestState
from model.controller.queue import ReadQueue, WriteQueue
from model.controller.scheduler import HBMScheduler, BankState, FRFCFSScheduler


@dataclass
class BandwidthTracker:
    """带宽追踪器
    
    追踪每个 QoS 等级的带宽使用情况。
    """
    def __init__(self, window_ms: float = 1.0, max_samples: int = 1000):
        self.window_ms = window_ms
        self.max_samples = max_samples
        self.data = defaultdict(deque)  # {qos: [(timestamp_ms, bytes), ...]}
    
    def record(self, qos: int, bytes: int, timestamp_ms: float):
        """记录带宽使用"""
        self.data[qos].append((timestamp_ms, bytes))
        
        # 清理旧数据
        cutoff = timestamp_ms - self.window_ms
        while self.data[qos] and self.data[qos][0][0] < cutoff:
            self.data[qos].popleft()
        
        # 限制样本数量
        while len(self.data[qos]) > self.max_samples:
            self.data[qos].popleft()
    
    def get_bandwidth(self, qos: int) -> float:
        """获取当前带宽 (GB/s)
        
        Args:
            qos: QoS 等级
            
        Returns:
            当前带宽 (GB/s)
        """
        if not self.data[qos]:
            return 0.0
        
        total_bytes = sum(b for _, b in self.data[qos])
        total_time_s = self.window_ms / 1000.0
        
        return total_bytes / total_time_s / 1e9


class QoSScheduler(HBMScheduler):
    """带带宽保证的 QoS 调度器
    
    设计目标:
    - 高优先级请求获得更多带宽
    - 低优先级请求不会被完全饿死
    - 带宽保证可配置
    """
    
    # QoS 优先级定义 (0=最低, 15=最高)
    QOS_CRITICAL = 15
    QOS_HIGH = 12
    QOS_NORMAL = 8
    QOS_LOW = 4
    QOS_IDLE = 0
    
    def __init__(self, config: HBMConfig):
        super().__init__(config)
        
        # 带宽保证配置 (GB/s per stack)
        self.bandwidth_guarantee = {
            self.QOS_CRITICAL: getattr(config, 'bw_guarantee_critical', 200.0),
            self.QOS_HIGH: getattr(config, 'bw_guarantee_high', 300.0),
            self.QOS_NORMAL: getattr(config, 'bw_guarantee_normal', 200.0),
            self.QOS_LOW: getattr(config, 'bw_guarantee_low', 100.0),
        }
        
        # 带宽上限 (GB/s per stack)
        self.bandwidth_cap = {
            self.QOS_CRITICAL: 1000.0,  # 无限制
            self.QOS_HIGH: 800.0,
            self.QOS_NORMAL: 400.0,
            self.QOS_LOW: 200.0,
            self.QOS_IDLE: 50.0,
        }
        
        # 带宽追踪
        self.bw_tracker = BandwidthTracker(window_ms=1.0)
        
        # FR-FCFS 调度器 (用于同优先级内调度)
        self.frfcfs = FRFCFSScheduler(config)
    
    def schedule(self, read_queue: ReadQueue, write_queue: WriteQueue,
                bank_states: Dict[Tuple, BankState],
                current_time: float,
                last_cmd_type: str = "READ") -> Optional[HBMRequest]:
        """QoS 调度
        
        1. 从高到低检查各 QoS 等级
        2. 检查带宽保证和上限
        3. 在符合条件的请求中选择 FR-FCFS 最优
        
        Args:
            read_queue: 读队列
            write_queue: 写队列
            bank_states: Bank 状态
            current_time: 当前时间
            last_cmd_type: 上次命令类型
            
        Returns:
            下一个调度的请求
        """
        timestamp_ms = current_time * 1000.0
        
        # 合并所有请求
        all_requests = list(read_queue._queue) + list(write_queue._queue)
        
        if not all_requests:
            return None
        
        # 按 QoS 分组
        qos_groups = defaultdict(list)
        for req in all_requests:
            qos_groups[req.qos].append(req)
        
        # 从高到低遍历 QoS 等级
        for qos_level in range(15, -1, -1):
            if qos_level not in qos_groups:
                continue
            
            candidates = qos_groups[qos_level]
            
            # 检查带宽保证
            if self._can_schedule(qos_level, len(candidates) * 64):
                # 可以在此优先级调度
                # 使用 FR-FCFS 选择最佳请求
                best = self.frfcfs._select_oldest(
                    self.frfcfs._get_row_hit_candidates(
                        read_queue if candidates[0].is_read else write_queue,
                        bank_states
                    ) or candidates
                )
                
                if best:
                    best.mark_scheduled(current_time)
                    
                    # 记录带宽使用
                    self.bw_tracker.record(best.qos, best.length, timestamp_ms)
                    
                    # 从队列移除
                    if best.is_read:
                        read_queue.remove(best.request_id)
                    else:
                        write_queue.remove(best.request_id)
                    
                    return best
        
        # 如果所有 QoS 都受限，降级到 FR-FCFS
        return self.frfcfs.schedule(read_queue, write_queue, bank_states, current_time, last_cmd_type)
    
    def _can_schedule(self, qos_level: int, estimated_bytes: int) -> bool:
        """检查是否可以调度该 QoS 等级
        
        Args:
            qos_level: QoS 等级
            estimated_bytes: 预估字节数
            
        Returns:
            True 如果可以调度
        """
        current_bw = self.bw_tracker.get_bandwidth(qos_level)
        guarantee = self.bandwidth_guarantee.get(qos_level, 0.0)
        cap = self.bandwidth_cap.get(qos_level, float('inf'))
        
        # 低于保证带宽，可以调度
        if current_bw < guarantee:
            return True
        
        # 超过上限，不能调度
        if current_bw >= cap:
            return False
        
        # 在保证和上限之间，竞态调度
        return True
    
    def set_bandwidth_guarantee(self, qos_level: int, guarantee: float):
        """设置带宽保证"""
        self.bandwidth_guarantee[qos_level] = guarantee
    
    def set_bandwidth_cap(self, qos_level: int, cap: float):
        """设置带宽上限"""
        self.bandwidth_cap[qos_level] = cap
    
    def get_qos_stats(self) -> Dict:
        """获取 QoS 统计"""
        return {
            qos: {
                'bandwidth': self.bw_tracker.get_bandwidth(qos),
                'guarantee': self.bandwidth_guarantee.get(qos, 0),
                'cap': self.bandwidth_cap.get(qos, float('inf')),
            }
            for qos in range(16)
        }
