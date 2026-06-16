"""
HBM Refresh Scheduler
参考设计文档 2026-06-15-hbm-system-model-design.md 的 5.2.3 节

支持 HBM3 staggered refresh:
- 每个 REFI 间隔刷新 8 个 bank group
- 交错执行减少峰值功耗
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from model.controller.config import HBMConfig


class RefreshMode(Enum):
    """刷新模式"""
    REFRESH_ALL_BANKS = "all"       # 一次刷新所有 bank
    REFRESH_PER_BANK = "per_bank"    # 逐 bank 刷新
    REFRESH_BANK_GROUP = "bank_group" # 按 bank group 刷新 (HBM3)


@dataclass
class RefreshCommand:
    """刷新命令"""
    timestamp: float
    bank_group_id: int = -1  # -1 表示所有 bank
    duration_cycles: int = 0


class RefreshScheduler:
    """HBM 刷新调度器
    
    HBM3 staggered refresh:
    - tREFI = 3.9 us (典型值)
    - tRFC = 295 cycles (16Gb)
    - 8 bank groups per pseudo-channel
    
    每个 REFI 间隔需要刷新所有 bank groups。
    使用 staggered 策略，每次只刷新一个 bank group。
    """
    
    # 刷新模式
    REFRESH_ALL_BANKS = "all"
    REFRESH_BANK_GROUP = "bank_group"
    
    def __init__(self, config: HBMConfig, mode: str = "bank_group"):
        """初始化刷新调度器
        
        Args:
            config: HBM 配置
            mode: 刷新模式 ("all" or "bank_group")
        """
        self.config = config
        self.mode = mode
        
        # 计算刷新参数 (cycles)
        # 假设 1.28 GHz 时钟频率
        self.clock_freq = 1.28e9  # Hz
        self.tREFI_cycles = int(config.refresh_interval * self.clock_freq)
        self.tRFC_cycles = int(config.refresh_penalty * self.clock_freq)
        
        # Bank group 配置
        self.bank_groups_per_channel = config.bank_groups_per_channel
        self.channels = config.channels_per_stack
        self.pseudo_channels = config.pseudo_channels_per_channel
        
        # Staggered refresh 状态
        self._last_refresh_time = -config.refresh_interval
        self._next_bank_group = 0
        self._pending_refreshes: List[RefreshCommand] = []
        
        # 统计
        self._stats = {
            'refresh_count': 0,
            'bank_group_refresh_count': 0,
            'total_refresh_cycles': 0,
        }
    
    def needs_refresh(self, current_time: float) -> bool:
        """检查是否需要执行刷新
        
        Args:
            current_time: 当前时间 (秒)
            
        Returns:
            True 如果需要刷新
        """
        time_since_refresh = current_time - self._last_refresh_time
        return time_since_refresh >= self.config.refresh_interval
    
    def get_next_refresh_time(self) -> float:
        """获取下次刷新时间"""
        return self._last_refresh_time + self.config.refresh_interval
    
    def schedule_refresh(self, current_time: float, 
                        bank_states: Dict[Tuple, 'BankState']) -> Optional[RefreshCommand]:
        """调度刷新命令
        
        Args:
            current_time: 当前时间 (秒)
            bank_states: Bank 状态字典
            
        Returns:
            RefreshCommand 或 None
        """
        if not self.needs_refresh(current_time):
            return None
        
        self._last_refresh_time = current_time
        
        if self.mode == self.REFRESH_ALL_BANKS:
            # 所有 bank 同时刷新
            cmd = RefreshCommand(
                timestamp=current_time,
                bank_group_id=-1,
                duration_cycles=self.tRFC_cycles,
            )
        else:
            # Bank group staggered refresh (HBM3)
            # 每次刷新一个 bank group
            cmd = RefreshCommand(
                timestamp=current_time,
                bank_group_id=self._next_bank_group,
                duration_cycles=self.tRFC_cycles,
            )
            self._next_bank_group = (self._next_bank_group + 1) % self.bank_groups_per_channel
        
        self._stats['refresh_count'] += 1
        self._stats['total_refresh_cycles'] += cmd.duration_cycles
        if cmd.bank_group_id >= 0:
            self._stats['bank_group_refresh_count'] += 1
        
        return cmd
    
    def calc_refresh_overhead(self, sim_duration: float) -> float:
        """计算刷新开销
        
        Args:
            sim_duration: 仿真时长 (秒)
            
        Returns:
            刷新开销比例 (0.0 - 1.0)
        """
        # 总刷新周期数
        num_refresh_intervals = sim_duration / self.config.refresh_interval
        total_refresh_cycles = num_refresh_intervals * self.tRFC_cycles
        
        # 总周期数
        total_cycles = sim_duration * self.clock_freq
        
        return total_refresh_cycles / total_cycles
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            **self._stats,
            'tREFI_cycles': self.tREFI_cycles,
            'tRFC_cycles': self.tRFC_cycles,
        }
    
    def reset(self):
        """重置刷新状态"""
        self._last_refresh_time = -self.config.refresh_interval
        self._next_bank_group = 0
        self._pending_refreshes.clear()
        self._stats = {
            'refresh_count': 0,
            'bank_group_refresh_count': 0,
            'total_refresh_cycles': 0,
        }


@dataclass
class RefreshManager:
    """刷新管理器
    
    管理多个 stack/channel 的刷新调度。
    """
    schedulers: List[RefreshScheduler] = field(default_factory=list)
    
    @classmethod
    def create(cls, config: HBMConfig) -> "RefreshManager":
        """创建刷新管理器
        
        Args:
            config: HBM 配置
            
        Returns:
            RefreshManager 实例
        """
        schedulers = []
        for _ in range(config.stack_count):
            schedulers.append(RefreshScheduler(config))
        return cls(schedulers=schedulers)
    
    def schedule_refresh(self, stack_id: int, current_time: float,
                         bank_states: Dict) -> Optional[RefreshCommand]:
        """调度指定 stack 的刷新"""
        if 0 <= stack_id < len(self.schedulers):
            return self.schedulers[stack_id].schedule_refresh(current_time, bank_states)
        return None
    
    def needs_refresh(self, stack_id: int, current_time: float) -> bool:
        """检查指定 stack 是否需要刷新"""
        if 0 <= stack_id < len(self.schedulers):
            return self.schedulers[stack_id].needs_refresh(current_time)
        return False
    
    def get_total_stats(self) -> dict:
        """获取所有 stack 的总统计"""
        total = {
            'refresh_count': 0,
            'bank_group_refresh_count': 0,
            'total_refresh_cycles': 0,
        }
        for sched in self.schedulers:
            stats = sched.get_stats()
            total['refresh_count'] += stats['refresh_count']
            total['bank_group_refresh_count'] += stats['bank_group_refresh_count']
            total['total_refresh_cycles'] += stats['total_refresh_cycles']
        return total
