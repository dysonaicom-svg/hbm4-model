"""
HBM DRAM Model - 完整 DRAM 模型接口
集成 bank 状态机、channel 模型、stack 模型

参考设计文档 2026-06-15-hbm-system-model-design.md 的 5.2 节

Multi-channel HBM3 支持:
- 8 channels per stack (JEDEC HBM3 spec)
- Per-channel statistics tracking
- Channel-aware command execution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from model.dram.timing import HBM3Timing, get_timing_for_hbm_version
from model.dram.bank_state_machine import BankStateMachine, BankStateEnum
from model.dram.channel_model import Channel, ChannelArray
from model.dram.stack_model import Stack


class DRAMCommand(Enum):
    """DRAM 命令枚举

    与 RTL 4-bit 命令编码对齐 (低 3 位):
    RTL: NOP=0, ACT=1, READ=2, WRITE=3, PRE=4, REF=5, MRS=6, ZQ=7
    """
    NOP = 0       # 0000 - No operation
    ACT = 1       # 0001 - Activate
    READ = 2      # 0010 - Read (与 RTL CMD_READ 对齐)
    WRITE = 3     # 0011 - Write (与 RTL CMD_WRITE 对齐)
    PRE = 4       # 0100 - Precharge (与 RTL CMD_PRE 对齐)
    REF = 5       # 0101 - Refresh (与 RTL CMD_REF 对齐)
    MRS = 6       # 0110 - Mode Register Set
    ZQ = 7        # 0111 - ZQ calibration

    # 别名兼容旧代码
    RD = 2
    WR = 3
    PREA = 8      # Precharge All (扩展)
    REFPB = 9     # Refresh per bank (扩展)


@dataclass
class DRAMResponse:
    """DRAM 响应"""
    success: bool
    data: Optional[bytes] = None
    latency_cycles: int = 0
    error: Optional[str] = None


@dataclass
class DRAMStats:
    """DRAM 统计信息"""
    total_activations: int = 0
    total_precharges: int = 0
    total_reads: int = 0
    total_writes: int = 0
    total_refreshes: int = 0
    row_hits: int = 0
    row_misses: int = 0
    row_conflicts: int = 0
    bank_busy_cycles: int = 0

    # Per-channel statistics (for multi-channel HBM3)
    per_channel_activations: Dict[int, int] = field(default_factory=dict)
    per_channel_reads: Dict[int, int] = field(default_factory=dict)
    per_channel_writes: Dict[int, int] = field(default_factory=dict)

    def add_activation(self):
        self.total_activations += 1

    def add_activation_to_channel(self, channel_id: int):
        """Add activation to specific channel"""
        self.total_activations += 1
        if channel_id not in self.per_channel_activations:
            self.per_channel_activations[channel_id] = 0
        self.per_channel_activations[channel_id] += 1

    def add_read(self):
        self.total_reads += 1

    def add_read_to_channel(self, channel_id: int):
        """Add read to specific channel"""
        self.total_reads += 1
        if channel_id not in self.per_channel_reads:
            self.per_channel_reads[channel_id] = 0
        self.per_channel_reads[channel_id] += 1

    def add_write(self):
        self.total_writes += 1

    def add_write_to_channel(self, channel_id: int):
        """Add write to specific channel"""
        self.total_writes += 1
        if channel_id not in self.per_channel_writes:
            self.per_channel_writes[channel_id] = 0
        self.per_channel_writes[channel_id] += 1

    def add_hit(self):
        self.row_hits += 1

    def add_miss(self):
        self.row_misses += 1

    def add_conflict(self):
        self.row_conflicts += 1

    def get_channel_stats(self, channel_id: int) -> Dict[str, int]:
        """Get statistics for a specific channel"""
        return {
            'activations': self.per_channel_activations.get(channel_id, 0),
            'reads': self.per_channel_reads.get(channel_id, 0),
            'writes': self.per_channel_writes.get(channel_id, 0),
        }

    def __repr__(self) -> str:
        total = self.row_hits + self.row_misses + self.row_conflicts
        hit_rate = self.row_hits / total * 100 if total > 0 else 0
        return (f"DRAMStats(acts={self.total_activations}, reads={self.total_reads}, "
                f"writes={self.total_writes}, hit_rate={hit_rate:.1f}%)")


@dataclass
class DecodedAddress:
    """解码后的地址"""
    stack_id: int
    channel_id: int
    pseudo_channel: int
    bank_group: int
    bank: int
    row: int
    col: int


class DRAMModel:
    """完整的 HBM DRAM 模型

    整合 stack、channel、bank 的完整层次结构。
    提供与控制器交互的高层接口。
    """

    def __init__(
        self,
        hbm_version: str = "hbm3",
        stack_count: int = 2,
        banks_per_channel: int = 16,
        rows_per_bank: int = 262144,
        cols_per_row: int = 128,
        bus_width: int = 64,
        burst_length: int = 4,
    ):
        """初始化 DRAM 模型

        Args:
            hbm_version: HBM 版本 ("hbm2", "hbm3", "hbm4")
            stack_count: Stack 数量
            banks_per_channel: 每个 channel 的 bank 数量 (HBM3 = 16)
            rows_per_bank: 每个 bank 的行数
            cols_per_row: 每行的列数
            bus_width: 数据总线宽度 (bits)
            burst_length: 突发长度
        """
        self.hbm_version = hbm_version
        self.config = {
            'stack_count': stack_count,
            'channels_per_stack': 8,
            'banks_per_channel': banks_per_channel,
            'rows_per_bank': rows_per_bank,
            'cols_per_row': cols_per_row,
            'bus_width': bus_width,
            'burst_length': burst_length,
        }

        # 时序参数
        self.timing = get_timing_for_hbm_version(hbm_version)

        # 创建 Stack 模型
        # HBM3: 每个 stack 8 channels, 每 channel 2 pseudo-channels,
        #        每 pseudo-channel 8 bank groups, 每 group 2 banks = 16 banks/channel
        self.stacks: List[Stack] = []
        for i in range(stack_count):
            stack = Stack(stack_id=i, num_channels=8)
            self.stacks.append(stack)

        # 统计
        self.stats = DRAMStats()

        # 内存 (可选的完整内存模型)
        self._memory: Optional[Dict] = None
        self._enable_memory = False

    @property
    def total_banks(self) -> int:
        """总 bank 数量"""
        return self.config['stack_count'] * 8 * 16  # 2 * 8 * 16 = 256

    def get_bank(self, stack_id: int, channel_id: int, bank_id: int) -> BankStateMachine:
        """获取指定 bank 的状态机

        Args:
            stack_id: Stack ID
            channel_id: Channel ID
            bank_id: 全局 Bank ID (0-31 per channel in actual impl, 16 per spec)

        Returns:
            BankStateMachine 实例
        """
        if stack_id >= len(self.stacks):
            raise ValueError(f"Invalid stack_id: {stack_id}")

        # Channel 模型: 2 pseudo-channels * 8 bank groups * 2 banks = 32 banks/channel
        # 映射: bank_id -> (ps_id, bg_id, bank_in_group)
        ps_id = bank_id // 16 if bank_id < 32 else 0  # 简化处理
        bank_in_group = bank_id % 2
        bg_id = (bank_id // 2) % 8

        # 实际模型中只有 ps_id=0,1; bg_id=0-7; bank=0,1
        ps_id = bank_id // 16  # 0 或 1
        bank_in_group = bank_id % 2
        bg_id = (bank_id // 2) % 8

        return self.stacks[stack_id].get_bank(channel_id, ps_id, bg_id, bank_in_group)

    def set_time(self, current_time: int):
        """Set current time

        OPTIMIZATION: This no longer propagates to all banks.
        Time is passed directly to bank operations when needed.

        Args:
            current_time: Current time (cycles)
        """
        # Only update the top-level time reference
        time_s = self.timing.cycles_to_s(current_time)
        for stack in self.stacks:
            stack.current_time = time_s

    def check_bank_available(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        current_time: int,
    ) -> Tuple[bool, str]:
        """检查 bank 是否可用

        Args:
            stack_id: Stack ID
            channel_id: Channel ID
            bank_id: Bank ID
            current_time: 当前时间 (cycles)

        Returns:
            (可用, 原因)
        """
        self.set_time(current_time)
        bank = self.get_bank(stack_id, channel_id, bank_id)
        available = bank.can_activate()
        return (available, "" if available else "Bank not available")

    def execute_activate(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        row_id: int,
        current_time: int,
    ) -> DRAMResponse:
        """Execute activate command

        Args:
            stack_id: Stack ID
            channel_id: Channel ID
            bank_id: Bank ID
            row_id: Row ID
            current_time: Current time (cycles)

        Returns:
            DRAMResponse
        """
        try:
            # Set time on the specific bank being accessed
            time_s = self.timing.cycles_to_s(current_time)
            bank = self.get_bank(stack_id, channel_id, bank_id)
            bank.set_time(time_s)

            success, error_msg = bank.activate(row_id)

            if success:
                self.stats.add_activation_to_channel(channel_id)
                return DRAMResponse(success=True, latency_cycles=self.timing.tRCD)
            else:
                return DRAMResponse(success=False, error=error_msg or "Activation failed")

        except Exception as e:
            return DRAMResponse(success=False, error=str(e))

    def execute_read(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        col_id: int,
        current_time: int,
        length: int = 32,
    ) -> DRAMResponse:
        """Execute read command

        Args:
            stack_id: Stack ID
            channel_id: Channel ID
            bank_id: Bank ID
            col_id: Column ID
            current_time: Current time (cycles)
            length: Read data length (bytes)

        Returns:
            DRAMResponse
        """
        try:
            # Set time on the specific bank being accessed
            time_s = self.timing.cycles_to_s(current_time)
            bank = self.get_bank(stack_id, channel_id, bank_id)
            bank.set_time(time_s)

            # Check bank state
            if bank.bank.state != BankStateEnum.ACTIVE:
                return DRAMResponse(success=False, error="Bank not activated")

            # Check timing
            if not bank.can_read():
                return DRAMResponse(success=False, error="Read timing violation")

            # Read data
            data = self._read_memory(stack_id, channel_id, bank_id, bank.bank.open_row, col_id, length)

            # Update stats (per-channel)
            self.stats.add_read_to_channel(channel_id)
            if bank.is_row_hit(col_id):
                self.stats.add_hit()
            else:
                self.stats.add_conflict()

            # Calculate latency (burst + tCCD)
            latency = self.timing.tCCD * (length // (self.config['bus_width'] // 8))

            return DRAMResponse(
                success=True,
                data=data,
                latency_cycles=latency,
            )

        except Exception as e:
            return DRAMResponse(success=False, error=str(e))

    def execute_write(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        col_id: int,
        data: bytes,
        current_time: int,
    ) -> DRAMResponse:
        """Execute write command

        Args:
            stack_id: Stack ID
            channel_id: Channel ID
            bank_id: Bank ID
            col_id: Column ID
            data: Write data
            current_time: Current time (cycles)

        Returns:
            DRAMResponse
        """
        try:
            # Set time on the specific bank being accessed
            time_s = self.timing.cycles_to_s(current_time)
            bank = self.get_bank(stack_id, channel_id, bank_id)
            bank.set_time(time_s)

            # Check bank state
            if bank.bank.state != BankStateEnum.ACTIVE:
                return DRAMResponse(success=False, error="Bank not activated")

            # Check timing
            if not bank.can_write():
                return DRAMResponse(success=False, error="Write timing violation")

            # Write data
            self._write_memory(stack_id, channel_id, bank_id, bank.bank.open_row, col_id, data)

            # Update stats (per-channel)
            self.stats.add_write_to_channel(channel_id)

            return DRAMResponse(success=True, latency_cycles=self.timing.tCCD)

        except Exception as e:
            return DRAMResponse(success=False, error=str(e))

    def execute_precharge(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        current_time: int,
    ) -> DRAMResponse:
        """Execute precharge command

        Args:
            stack_id: Stack ID
            channel_id: Channel ID
            bank_id: Bank ID
            current_time: Current time (cycles)

        Returns:
            DRAMResponse
        """
        try:
            # Set time on the specific bank being accessed
            time_s = self.timing.cycles_to_s(current_time)
            bank = self.get_bank(stack_id, channel_id, bank_id)
            bank.set_time(time_s)

            success = bank.precharge()

            if success:
                self.stats.total_precharges += 1
                return DRAMResponse(success=True, latency_cycles=self.timing.tRP)
            else:
                return DRAMResponse(success=False, error="Precharge failed")

        except Exception as e:
            return DRAMResponse(success=False, error=str(e))

    def execute_refresh(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        current_time: int,
    ) -> DRAMResponse:
        """Execute refresh command

        Args:
            stack_id: Stack ID
            channel_id: Channel ID
            bank_id: Bank ID
            current_time: Current time (cycles)

        Returns:
            DRAMResponse
        """
        try:
            # Set time on the specific bank being accessed
            time_s = self.timing.cycles_to_s(current_time)
            bank = self.get_bank(stack_id, channel_id, bank_id)
            bank.set_time(time_s)

            # During refresh, bank is unavailable
            self.stats.total_refreshes += 1

            # Simplified: after refresh, row is invalidated
            bank.bank.state = BankStateEnum.IDLE
            bank.bank.open_row = None
            bank.bank.precharge_time = time_s

            return DRAMResponse(success=True, latency_cycles=self.timing.tRFC)

        except Exception as e:
            return DRAMResponse(success=False, error=str(e))

    def execute_request(
        self,
        stack_id: int,
        ch_id: int,
        ps_id: int,
        bg_id: int,
        bank_id: int,
        row: int,
        cmd: str,
        data: Optional[bytes] = None,
        col: int = 0,
        length: int = 32,
        current_time: Optional[int] = None,
    ) -> bool:
        """Execute a read or write request on DRAM

        This is the unified interface for the command pipeline.

        Args:
            stack_id: Stack ID
            ch_id: Channel ID
            ps_id: Pseudo-channel ID
            bg_id: Bank group ID
            bank_id: Bank ID (local within bank group)
            row: Row ID
            cmd: Command type ("READ" or "WRITE")
            data: Write data (bytes, for WRITE commands)
            col: Column ID (default 0)
            length: Transfer length in bytes (default 32)
            current_time: Current time in cycles (optional, uses internal state if None)

        Returns:
            True if command was accepted
        """
        # Calculate global bank_id for internal methods
        global_bank_id = (ps_id * 16) + (bg_id * 2) + bank_id

        if current_time is None:
            current_time = 0

        if cmd.upper() == "READ":
            resp = self.execute_read(
                stack_id=stack_id,
                channel_id=ch_id,
                bank_id=global_bank_id,
                col_id=col,
                current_time=current_time,
                length=length,
            )
            return resp.success
        elif cmd.upper() == "WRITE":
            if data is None:
                # Generate dummy data for write
                data = bytes(length)
            resp = self.execute_write(
                stack_id=stack_id,
                channel_id=ch_id,
                bank_id=global_bank_id,
                col_id=col,
                data=data,
                current_time=current_time,
            )
            return resp.success
        else:
            return False

    def write(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        row_id: int,
        col_id: int,
        data: bytes,
    ) -> bool:
        """Direct write to DRAM memory

        This bypasses timing checks for direct data write access.

        Args:
            stack_id: Stack ID
            channel_id: Channel ID
            bank_id: Bank ID
            row_id: Row ID
            col_id: Column ID
            data: Data to write

        Returns:
            True if write succeeded
        """
        try:
            self._write_memory(stack_id, channel_id, bank_id, row_id, col_id, data)
            self.stats.add_write()
            return True
        except Exception:
            return False

    def read(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        row_id: int,
        col_id: int,
        length: int,
    ) -> bytes:
        """Direct read from DRAM memory

        This bypasses timing checks for direct data read access.

        Args:
            stack_id: Stack ID
            channel_id: Channel ID
            bank_id: Bank ID
            row_id: Row ID
            col_id: Column ID
            length: Number of bytes to read

        Returns:
            Data read from memory
        """
        try:
            data = self._read_memory(stack_id, channel_id, bank_id, row_id, col_id, length)
            self.stats.add_read()
            return data
        except Exception:
            return bytes(length)

    def tick(self, current_time: int):
        """更新所有 bank 状态

        Args:
            current_time: 当前时间 (cycles)
        """
        self.set_time(current_time)

    def get_utilization(self, window: int = 10000) -> float:
        """计算 bank 利用率

        Args:
            window: 统计窗口 (cycles)

        Returns:
            利用率 (0-1)
        """
        # 简化: 基于激活次数估算
        total_cycles = self.total_banks * window
        busy_cycles = self.stats.total_activations * self.timing.tRAS
        return min(1.0, busy_cycles / total_cycles)

    def _read_memory(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        row_id: int,
        col_id: int,
        length: int,
    ) -> bytes:
        """读取内存数据 (如果启用)"""
        if not self._enable_memory or self._memory is None:
            # 返回假数据
            return bytes(length)

        key = (stack_id, channel_id, bank_id, row_id)
        if key not in self._memory:
            self._memory[key] = bytearray(self.config['cols_per_row'] * self.config['bus_width'] // 8)

        data = self._memory[key]
        start = col_id * (self.config['bus_width'] // 8)
        return bytes(data[start:start + length])

    def _write_memory(
        self,
        stack_id: int,
        channel_id: int,
        bank_id: int,
        row_id: int,
        col_id: int,
        data: bytes,
    ):
        """写入内存数据 (如果启用)"""
        if not self._enable_memory or self._memory is None:
            return

        key = (stack_id, channel_id, bank_id, row_id)
        if key not in self._memory:
            self._memory[key] = bytearray(self.config['cols_per_row'] * self.config['bus_width'] // 8)

        mem = self._memory[key]
        start = col_id * (self.config['bus_width'] // 8)
        mem[start:start + len(data)] = data

    def enable_memory_model(self):
        """启用完整内存模型"""
        self._enable_memory = True
        self._memory = {}

    def reset(self):
        """重置 DRAM 模型"""
        for stack in self.stacks:
            for ch in stack.channels.channels:
                for ps in ch.pseudo_channels:
                    for bg in ps.bank_groups:
                        for bank in bg.banks:
                            bank.bank.state = BankStateEnum.IDLE
                            bank.bank.open_row = None
        self.stats = DRAMStats()
        if self._memory:
            self._memory = {}

    def get_all_channel_stats(self) -> Dict[int, Dict[str, int]]:
        """Get statistics for all channels

        Returns:
            Dict mapping channel_id to stats dict
        """
        result = {}
        for ch_id in range(self.config['channels_per_stack']):
            result[ch_id] = self.stats.get_channel_stats(ch_id)
        return result

    def get_channel_utilization(self, channel_id: int, window: int = 10000) -> float:
        """Calculate utilization for a specific channel

        Args:
            channel_id: Channel ID
            window: Statistics window (cycles)

        Returns:
            Utilization (0-1)
        """
        ch_stats = self.stats.get_channel_stats(channel_id)
        busy_cycles = ch_stats['activations'] * self.timing.tRAS
        return min(1.0, busy_cycles / window)

    def __repr__(self) -> str:
        return (f"DRAMModel(v={self.hbm_version}, stacks={len(self.stacks)}, "
                f"channels={self.config['channels_per_stack']}, "
                f"banks={self.config['banks_per_channel']})")


def create_dram_model(config: Dict) -> DRAMModel:
    """从配置创建 DRAM 模型

    Args:
        config: 配置字典

    Returns:
        DRAMModel 实例
    """
    return DRAMModel(
        hbm_version=config.get('hbm_version', 'hbm3'),
        stack_count=config.get('stack_count', 2),
        banks_per_channel=config.get('banks_per_channel', 16),
        rows_per_bank=config.get('rows_per_bank', 262144),
        cols_per_row=config.get('cols_per_row', 128),
        bus_width=config.get('bus_width', 64),
        burst_length=config.get('burst_length', 4),
    )