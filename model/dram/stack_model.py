"""
HBM DRAM Stack Model
参考设计文档 2026-06-15-hbm-system-model-design.md 的 5.2 节

Stack 模型:
- 包含多个 Channel
- 支持 1-8 个 Stack
- 多 Stack 互联拓扑

Performance optimizations:
- Eliminated hierarchical set_time() propagation
- Channels store time locally
- Banks track their own time
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from model.dram.timing import HBM3Timing
from model.dram.channel_model import Channel, ChannelArray


class InterconnectTopology(Enum):
    """互联拓扑"""
    MESH = "mesh"
    FULL_CROSSBAR = "full_crossbar"
    BUTTERFLY = "butterfly"


@dataclass
class Stack:
    """HBM DRAM Stack

    包含:
    - 8 个 Channel
    - 每个 Channel 2 个 Pseudo-channel
    - 每个 Pseudo-channel 8 个 Bank Groups
    - 每个 Bank Group 2 个 Banks

    Performance: No longer propagates set_time() to all banks.
    """
    stack_id: int
    num_channels: int = 8
    channels: ChannelArray = field(default=None)
    current_time: float = 0.0

    def __post_init__(self):
        if self.channels is None:
            self.channels = ChannelArray(num_channels=self.num_channels)

    def set_time(self, current_time: float):
        """Set time for this stack

        OPTIMIZATION: No longer propagates to all banks.
        Each channel stores time locally.
        """
        self.current_time = current_time
        # Only update channel times, not all banks
        self.channels.set_time(current_time)

    def get_channel(self, ch_id: int) -> Channel:
        return self.channels.get_channel(ch_id)

    def get_bank(self, ch_id: int, ps_id: int, bg_id: int, bank_id: int):
        return self.channels.get_bank(ch_id, ps_id, bg_id, bank_id)

    def is_row_hit(self, ch_id: int, ps_id: int, bg_id: int, bank_id: int, row: int) -> bool:
        return self.channels.is_row_hit(ch_id, ps_id, bg_id, bank_id, row)

    def execute_command(self, ch_id: int, ps_id: int, cmd: str,
                        bg_id: int = 0, bank_id: int = 0, row: int = 0) -> bool:
        """执行命令"""
        return self.channels.channels[ch_id].execute_command(
            ps_id, cmd, bg_id, bank_id, row
        )

    def get_total_banks(self) -> int:
        """获取总 bank 数"""
        return self.num_channels * 2 * 8 * 2  # ch * ps * bg * banks

    def get_stats(self) -> Dict:
        """获取统计"""
        active_banks = 0
        idle_banks = 0
        for ch in self.channels.channels:
            for ps in ch.pseudo_channels:
                for bg in ps.bank_groups:
                    for bank in bg.banks:
                        if bank.bank.state.value == 1:  # ACTIVE
                            active_banks += 1
                        else:
                            idle_banks += 1
        return {
            'total_banks': self.get_total_banks(),
            'active_banks': active_banks,
            'idle_banks': idle_banks,
        }


@dataclass
class StackArray:
    """Stack 数组

    管理多个 HBM Stack
    支持多 Stack 互联
    """
    num_stacks: int = 2
    stacks: List[Stack] = field(default_factory=list)
    topology: InterconnectTopology = InterconnectTopology.MESH

    def __post_init__(self):
        if not self.stacks:
            self.stacks = [
                Stack(stack_id=i)
                for i in range(self.num_stacks)
            ]

    def set_time(self, current_time: float):
        for stack in self.stacks:
            stack.current_time = current_time

    def get_stack(self, stack_id: int) -> Stack:
        return self.stacks[stack_id]

    def get_total_banks(self) -> int:
        return sum(s.get_total_banks() for s in self.stacks)

    def get_stats(self) -> Dict:
        stats = {
            'num_stacks': self.num_stacks,
            'topology': self.topology.value,
            'total_banks': self.get_total_banks(),
            'stacks': [s.get_stats() for s in self.stacks],
        }
        return stats


@dataclass
class DRAMModel:
    """完整的 DRAM 模型

    整合 Stack、Channel、Bank
    提供高层接口
    """
    num_stacks: int = 2
    num_channels: int = 8
    timing: HBM3Timing = field(default_factory=HBM3Timing)
    stack_array: StackArray = field(default=None)
    current_time: float = 0.0

    def __post_init__(self):
        if self.stack_array is None:
            self.stack_array = StackArray(num_stacks=self.num_stacks)

    def set_time(self, current_time: float):
        """Set time for all stacks

        OPTIMIZATION: No longer propagates to all banks.
        """
        self.current_time = current_time
        self.stack_array.set_time(current_time)

    def tick(self, cycles: int = 1):
        """推进时钟

        Args:
            cycles: 时钟周期数
        """
        self.current_time += cycles * self.timing.clock_period_ns * 1e-9
        self.set_time(self.current_time)

    def execute_request(self, stack_id: int, ch_id: int, ps_id: int,
                        bg_id: int, bank_id: int, row: int, cmd: str) -> bool:
        """执行内存请求

        Args:
            cmd: "READ" or "WRITE"
        """
        stack = self.stack_array.get_stack(stack_id)

        # 检查 row hit
        if stack.is_row_hit(ch_id, ps_id, bg_id, bank_id, row):
            if cmd == "READ":
                success = stack.execute_command(ch_id, ps_id, "RD", bg_id, bank_id)
                if success:
                    stack.get_bank(ch_id, ps_id, bg_id, bank_id).complete_read()
                return success
            else:
                success = stack.execute_command(ch_id, ps_id, "WR", bg_id, bank_id)
                if success:
                    stack.get_bank(ch_id, ps_id, bg_id, bank_id).complete_write()
                return success
        else:
            # Row miss: PRE -> ACT -> RD/WR
            stack.execute_command(ch_id, ps_id, "PRE", bg_id, bank_id)
            success = stack.execute_command(ch_id, ps_id, "ACT", bg_id, bank_id, row)
            if not success:
                return False
            # 等待 tRCD 后才能发起 READ/WR
            self.current_time += self.timing.cycles_to_s(self.timing.tRCD)
            stack.set_time(self.current_time)
            if cmd == "READ":
                success = stack.execute_command(ch_id, ps_id, "RD", bg_id, bank_id)
                if success:
                    stack.get_bank(ch_id, ps_id, bg_id, bank_id).complete_read()
                return success
            else:
                success = stack.execute_command(ch_id, ps_id, "WR", bg_id, bank_id)
                if success:
                    stack.get_bank(ch_id, ps_id, bg_id, bank_id).complete_write()
                return success

    def get_stats(self) -> Dict:
        return self.stack_array.get_stats()
