"""
HBM Controller Integration
整合所有 Phase A 模块的主控制器
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import time

from model.controller.config import HBMConfig, HBM3_DEFAULT
from model.controller.request import HBMRequest, HBMResponse, RequestState
from model.controller.queue import ReadQueue, WriteQueue, QueueManager
from model.controller.address_decoder import AddressDecoder, DecodedAddress
from model.controller.scheduler import FRFCFSScheduler, BankState, SchedulerStats
from model.controller.qos_scheduler import QoSScheduler
from model.controller.refresh_scheduler import RefreshScheduler, RefreshManager
from model.controller.exceptions import QueueOverflowError
from model.controller.command_sequencer import (
    CommandSequencer,
    CommandSequence,
    BankState as SeqBankState,
    BankStateEnum,
)
from model.controller.command_pipeline import CommandPipeline
from model.dram.timing import HBM3Timing


@dataclass
class HBMController:
    """HBM 控制器整合模型
    
    整合所有 Phase A 模块的主控制器。
    """
    
    def __init__(self, config: Optional[HBMConfig] = None):
        """初始化控制器
        
        Args:
            config: HBM 配置 (默认 HBM3 配置)
        """
        self.config = config or HBM3_DEFAULT
        self.current_time = 0.0
        
        # 初始化组件
        self.decoder = AddressDecoder(self.config)
        self.queue_manager = QueueManager.create(self.config.queue_depth)
        
        # 初始化调度器
        if self.config.scheduler_mode == "qos":
            self.scheduler = QoSScheduler(self.config)
        else:
            self.scheduler = FRFCFSScheduler(self.config)
        
        # 初始化刷新调度器
        self.refresh_manager = RefreshManager.create(self.config)

        # 初始化 CommandSequencer (用于生成 DRAM 命令序列)
        # Use the timing spec from config for correct timing parameters
        if hasattr(self.config, 'timing'):
            timing_spec = self.config.timing
        else:
            timing_spec = self.config  # Fallback to config itself
        self.sequencer = CommandSequencer(timing_spec)

        # 初始化 CommandPipeline (用于 DRAM 命令执行)
        self.pipeline = CommandPipeline(timing_spec)

        # Bank 状态
        self.bank_states: Dict[Tuple, BankState] = {}
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'read_requests': 0,
            'write_requests': 0,
            'row_hit_count': 0,
            'refresh_count': 0,
        }
        
        # 调度统计
        self.scheduler_stats = SchedulerStats()

        # 最近调度的请求 (用于 CommandSequencer 集成)
        self._last_scheduled_request: Optional[HBMRequest] = None

        # 最近调度的命令类型
        self._last_cmd_type: str = "READ"
    
    def submit_request(self, request: HBMRequest) -> bool:
        """提交请求
        
        Args:
            request: HBM 请求
            
        Returns:
            True 如果成功提交
        """
        # 解码地址
        decoded = self.decoder.decode(request.addr)
        request.stack_id = decoded.stack_id
        request.channel_id = decoded.channel_id
        request.pseudo_channel_id = decoded.pseudo_channel_id
        request.bank_group_id = decoded.bank_group_id
        request.bank_id = decoded.bank_id
        request.row_id = decoded.row_id
        request.col_id = decoded.col_id
        
        # 更新 bank 状态
        bank_key = (request.channel_id, request.pseudo_channel_id, request.bank_id)
        if bank_key not in self.bank_states:
            self.bank_states[bank_key] = BankState(bank_id=request.bank_id)
        
        # 检查 row hit
        bank_state = self.bank_states[bank_key]
        request.row_hit = (bank_state.is_open and bank_state.open_row == request.row_id)
        
        # 入队
        if request.is_read:
            success = self.queue_manager.push_read(request)
        else:
            success = self.queue_manager.push_write(request)

        if success:
            # 设置到达时间（使用当前仿真周期）
            request.set_arrival_time(self.current_time)
            self.stats['total_requests'] += 1
            if request.is_read:
                self.stats['read_requests'] += 1
            else:
                self.stats['write_requests'] += 1
            if request.row_hit:
                self.stats['row_hit_count'] += 1
        
        return success
    
    def tick(self) -> Tuple[Optional[HBMRequest], Optional[HBMResponse]]:
        """执行一个时钟周期

        Returns:
            Tuple of (scheduled_request, response).
            scheduled_request is the request being scheduled this cycle.
            response is None if no request completed, or HBMResponse if completed.
            Note: In the current model, the scheduled request IS completed
            immediately (simplified model). For cycle-accurate timing,
            use tick_advanced() instead.
        """
        self.current_time += 1  # 使用周期作为时间单位

        # 检查刷新
        for stack_id in range(self.config.stack_count):
            if self.refresh_manager.needs_refresh(stack_id, self.current_time):
                cmd = self.refresh_manager.schedule_refresh(stack_id, self.current_time, self.bank_states)
                if cmd:
                    self.stats['refresh_count'] += 1

        # 调度请求
        scheduled = self.scheduler.schedule(
            self.queue_manager.read_queue,
            self.queue_manager.write_queue,
            self.bank_states,
            self.current_time,
            self._last_cmd_type
        )

        if scheduled:
            # 更新 last command type
            self._last_cmd_type = "READ" if scheduled.is_read else "WRITE"
            self._last_scheduled_request = scheduled

            # 生成 DRAM 命令序列 (使用 CommandSequencer)
            # 创建 BankState 用于命令序列生成
            seq_bank_state = SeqBankState(
                bank_id=scheduled.bank_id,
                open_row=scheduled.row_id if scheduled.row_hit else -1,
                state=BankStateEnum.ACTIVE if scheduled.row_hit else BankStateEnum.IDLE
            )
            cmd_sequence = self._generate_command_sequence(scheduled, self.current_time)

            # 通过 CommandPipeline 跟踪命令 (用于延迟估算)
            # 注意: 实际 DRAMModel 在 HBMSimulator 中, 这里只记录命令供延迟计算使用
            # 延迟已通过 cmd_sequence.total_cycles 计算，不需要实际提交到 pipeline
            # (tick() 是简化模型，tick_advanced() 会使用完整的 pipeline)

            # 计算实际延迟 (基于命令序列)
            # Row hit: RD/WR + PRE = ~5 cycles
            # Row miss: ACT + RD/WR + PRE = ~43 cycles (HBM3)
            actual_latency_cycles = cmd_sequence.total_cycles
            scheduled.estimated_cycles = actual_latency_cycles

            # 标记完成
            scheduled.mark_completed(self.current_time)

            # 记录调度统计
            self.scheduler_stats.record_schedule(scheduled)

            # 计算延迟（周期转换为 ns）
            latency_ns = actual_latency_cycles * self.config.timing.clock_period_ns

            return (scheduled, HBMResponse(
                request_id=scheduled.request_id,
                status="OK",
                latency=latency_ns,
            ))

        return (None, None)

    def _generate_command_sequence(self, request: HBMRequest, start_cycle: int = 0) -> CommandSequence:
        """生成 DRAM 命令序列

        Args:
            request: HBM 请求
            start_cycle: 序列开始的周期

        Returns:
            CommandSequence 包含所有命令和时序信息
        """
        # 获取 bank 状态
        bank_key = (request.channel_id, request.pseudo_channel_id, request.bank_id)
        bank_state = self.bank_states.get(bank_key)

        if bank_state:
            # 转换为 CommandSequencer 使用的 BankState
            seq_bank_state = SeqBankState(
                bank_id=request.bank_id,
                open_row=bank_state.open_row if bank_state.is_open else -1,
                state=BankStateEnum.ACTIVE if bank_state.is_open else BankStateEnum.IDLE
            )
        else:
            seq_bank_state = SeqBankState(bank_id=request.bank_id)

        # 生成命令序列 (使用 sequencer 实例方法)
        sequence = self.sequencer.generate_command_sequence(
            request=request,
            bank_state=seq_bank_state,
            start_cycle=start_cycle
        )

        return sequence

    def get_bandwidth(self) -> float:
        """计算当前有效带宽"""
        total_bytes = 0
        for req_id in range(1, self.scheduler_stats.schedule_count + 1):
            total_bytes += 64  # 假设每个请求 64 bytes
        if self.current_time > 0:
            return total_bytes / self.current_time / 1e9  # GB/s
        return 0.0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'controller': self.stats,
            'scheduler': {
                'schedule_count': self.scheduler_stats.schedule_count,
                'row_hit_rate': self.scheduler_stats.row_hit_rate,
                'read_count': self.scheduler_stats.read_count,
                'write_count': self.scheduler_stats.write_count,
            },
            'queue': self.queue_manager.get_stats(),
            'refresh': self.refresh_manager.get_total_stats(),
        }
