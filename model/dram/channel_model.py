"""
HBM DRAM Channel Model
参考设计文档 2026-06-15-hbm-system-model-design.md 的 5.2 节

Channel 模型:
- 每个 Channel 有独立的命令/地址总线
- 每个 Channel 有多个 Bank Group
- 每个 Bank Group 有多个 Bank

Performance optimizations:
- Eliminated hierarchical set_time() propagation
- Banks track their own time directly
- Batch operations for bank state queries
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from model.dram.timing import HBM3Timing
from model.dram.bank_state_machine import BankStateMachine, BankStateEnum


@dataclass
class BankGroup:
    """Bank Group

    HBM3: 每个 pseudo-channel 有 8 个 bank groups，每个 group 2 个 banks

    Attributes:
        group_id: Bank group ID
        banks: Bank 列表 (每个 group 2 个 bank)
        timing: 共享的时序参数实例
    """
    group_id: int
    banks: List[BankStateMachine] = field(default_factory=list)
    timing: HBM3Timing = field(default_factory=HBM3Timing)

    def __post_init__(self):
        if not self.banks:
            self.banks = [
                BankStateMachine(bank_id=i, timing=self.timing)
                for i in range(2)  # 2 banks per group
            ]

    def get_bank(self, bank_in_group: int) -> BankStateMachine:
        return self.banks[bank_in_group]

    def can_activate_any(self) -> bool:
        return any(bm.can_activate() for bm in self.banks)


@dataclass
class PseudoChannel:
    """Pseudo Channel

    HBM3: 每个 Channel 有 2 个 pseudo-channel
    每个 pseudo-channel 有独立的地址/命令总线
    """
    channel_id: int
    pseudo_id: int
    bank_groups: List[BankGroup] = field(default_factory=list)

    def __post_init__(self):
        if not self.bank_groups:
            self.bank_groups = [
                BankGroup(group_id=i)
                for i in range(8)  # 8 bank groups per pseudo-channel
            ]

    def get_bank(self, bg_id: int, bank_id: int) -> BankStateMachine:
        return self.bank_groups[bg_id].banks[bank_id]

    def get_bank_by_global_id(self, global_bank_id: int) -> Tuple[int, int, BankStateMachine]:
        """根据全局 bank ID 获取 bank

        Global bank ID: bg_id * 2 + bank_in_group
        """
        bg_id = global_bank_id // 2
        bank_in_group = global_bank_id % 2
        return bg_id, bank_in_group, self.bank_groups[bg_id].banks[bank_in_group]


@dataclass
class Channel:
    """HBM DRAM Channel

    每个 Channel 有:
    - 2 个 pseudo-channel
    - 独立的命令/地址总线
    - 数据总线宽度: 128-bit (HBM3)

    Performance: No longer propagates set_time() to all banks.
    Each bank tracks its own time when operations are performed.
    """
    channel_id: int
    pseudo_channels: List[PseudoChannel] = field(default_factory=list)
    current_time: float = 0.0

    def __post_init__(self):
        if not self.pseudo_channels:
            self.pseudo_channels = [
                PseudoChannel(channel_id=self.channel_id, pseudo_id=0),
                PseudoChannel(channel_id=self.channel_id, pseudo_id=1),
            ]

    def set_time(self, current_time: float):
        """Set time for this channel

        OPTIMIZATION: No longer propagates to all banks.
        Each bank will receive time when needed during operations.
        """
        self.current_time = current_time

    def get_pseudo_channel(self, ps_id: int) -> PseudoChannel:
        return self.pseudo_channels[ps_id]

    def get_bank(self, ps_id: int, bg_id: int, bank_id: int) -> BankStateMachine:
        return self.pseudo_channels[ps_id].bank_groups[bg_id].banks[bank_id]

    def is_row_hit(self, ps_id: int, bg_id: int, bank_id: int, row: int) -> bool:
        bank = self.get_bank(ps_id, bg_id, bank_id)
        return bank.is_row_hit(row)

    def execute_command(self, ps_id: int, cmd: str, bg_id: int = 0,
                        bank_id: int = 0, row: int = 0) -> bool:
        """执行命令

        Args:
            ps_id: pseudo-channel ID
            cmd: "ACT", "PRE", "RD", "WR", "REF"
            bg_id: bank group ID
            bank_id: bank ID (within group)
            row: 行号 (用于 ACT)
        """
        bank = self.get_bank(ps_id, bg_id, bank_id)

        if cmd == "ACT":
            return bank.activate(row)
        elif cmd == "PRE":
            return bank.precharge()
        elif cmd == "RD":
            return bank.read()
        elif cmd == "WR":
            return bank.write()
        elif cmd == "REF":
            return bank.refresh()

        return False


@dataclass
class ChannelArray:
    """Channel 数组

    管理多个 channel
    """
    num_channels: int = 8
    channels: List[Channel] = field(default_factory=list)

    def __post_init__(self):
        if not self.channels:
            self.channels = [
                Channel(channel_id=i)
                for i in range(self.num_channels)
            ]

    def set_time(self, current_time: float):
        """Set time for all channels

        OPTIMIZATION: No longer propagates to all banks.
        """
        for ch in self.channels:
            ch.current_time = current_time

    def get_channel(self, ch_id: int) -> Channel:
        return self.channels[ch_id]

    def get_bank(self, ch_id: int, ps_id: int, bg_id: int, bank_id: int) -> BankStateMachine:
        return self.channels[ch_id].get_bank(ps_id, bg_id, bank_id)

    def is_row_hit(self, ch_id: int, ps_id: int, bg_id: int, bank_id: int, row: int) -> bool:
        return self.channels[ch_id].is_row_hit(ps_id, bg_id, bank_id, row)
