"""
HBM Address Decoder
参考设计文档 2026-06-15-hbm-system-model-design.md 的 5.1.4 节

支持多种地址映射方案:
- RBC (Row-Bank-Channel): 适合顺序访问
- BCR (Bank-Channel-Row): 最大化并行度
- CRB (Channel-Row-Bank): 跨 channel 随机
- Custom: 可配置矩阵

Multi-channel HBM3 支持:
- 8 channels per stack (JEDEC HBM3)
- Channel selection via Addr[45:43]
- Per-channel load balancing support
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from enum import Enum

from model.controller.config import HBMConfig
from model.controller.exceptions import AddressError


class AddressMapping(Enum):
    """地址映射方案枚举"""
    RBC = "rbc"    # Row-Bank-Channel
    BCR = "bcr"    # Bank-Channel-Row
    CRB = "crb"    # Channel-Row-Bank
    CUSTOM = "custom"


@dataclass
class DecodedAddress:
    """解码后的地址字段"""
    stack_id: int = 0
    channel_id: int = 0
    pseudo_channel_id: int = 0
    bank_group_id: int = 0
    bank_id: int = 0
    row_id: int = 0
    col_id: int = 0
    burst_id: int = 0  # Burst beat index (HBM4 specific)
    byte_offset: int = 0

    def __repr__(self) -> str:
        return (f"DecodedAddr(ch={self.channel_id}, ps={self.pseudo_channel_id}, "
                f"bg={self.bank_group_id}, bk={self.bank_id}, "
                f"row=0x{self.row_id:x}, col=0x{self.col_id:x})")

    def get_channel_key(self) -> Tuple[int, int, int]:
        """Get unique channel identifier

        Returns:
            (stack_id, channel_id, pseudo_channel_id)
        """
        return (self.stack_id, self.channel_id, self.pseudo_channel_id)

    def get_bank_key(self) -> Tuple[int, int, int, int]:
        """Get unique bank identifier

        Returns:
            (stack_id, channel_id, pseudo_channel_id, bank_id)
        """
        return (self.stack_id, self.channel_id, self.pseudo_channel_id, self.bank_id)


class AddressDecoder:
    """HBM 地址解码器

    根据配置将 64-bit 物理地址解码为 HBM 地址字段。
    支持可配置的地址映射方案。

    HBM3 默认地址映射 (JEDEC):
        Addr[47:46] = Stack ID (2-bit, 支持 4 stack)
        Addr[45:43] = Channel (3-bit, 8 channels)
        Addr[42]    = Pseudo-channel (1-bit, 2 pseudo-ch)
        Addr[41:39] = Bank group (3-bit, 8 bank groups per pseudo-ch)
        Addr[38:34] = Bank within group (5-bit, 2 banks per group)
        Addr[33:16] = Row (18-bit)
        Addr[15:3]  = Column (13-bit)
        Addr[2:0]   = Byte offset (8-byte 粒度)

    Multi-channel HBM3 features:
    - Proper 8-channel selection per JEDEC spec
    - Per-channel load tracking for scheduling
    - Channel isolation for QoS
    """

    # 默认位分配 (HBM3 JEDEC)
    DEFAULT_BIT_STACK = (47, 46, 2)      # (msb, lsb, bits)
    DEFAULT_BIT_CHANNEL = (45, 43, 3)
    DEFAULT_BIT_PSEUDO_CH = (42, 42, 1)
    DEFAULT_BIT_BANK_GROUP = (41, 39, 3)
    DEFAULT_BIT_BANK = (38, 34, 5)
    DEFAULT_BIT_ROW = (33, 16, 18)
    DEFAULT_BIT_COL = (15, 3, 13)
    DEFAULT_BIT_OFFSET = (2, 0, 3)

    def __init__(self, config: HBMConfig, custom_mapping: Optional[Dict] = None):
        """初始化地址解码器

        Args:
            config: HBM 配置
            custom_mapping: 自定义映射 (可选)
        """
        self.config = config

        if custom_mapping:
            self.mapping = custom_mapping
        else:
            self.mapping = self._get_default_mapping(config.address_mapping)

        # 预计算掩码和移位
        self._setup_bit_masks()

    def _get_default_mapping(self, mapping_name: str) -> Dict:
        """获取默认映射方案

        Args:
            mapping_name: 映射方案名称

        Returns:
            映射参数字典
        """
        mapping_name = mapping_name.lower()

        if mapping_name == "rbc":
            # Row-Bank-Channel: Row 最低位，适合顺序访问
            return {
                'stack': (47, 46, 2),
                'channel': (45, 43, 3),
                'pseudo_channel': (42, 42, 1),
                'bank_group': (41, 39, 3),
                'bank': (38, 34, 5),
                'row': (33, 16, 18),
                'col': (15, 3, 13),
                'offset': (2, 0, 3),
            }
        elif mapping_name == "bcr":
            # Bank-Channel-Row: Bank 在 Row 之前，最大化并行度
            return {
                'stack': (47, 46, 2),
                'bank_group': (45, 43, 3),  # Bank group 在 channel 位
                'bank': (42, 38, 5),        # Bank 在 pseudo-channel 位
                'channel': (37, 35, 3),      # Channel
                'pseudo_channel': (34, 34, 1),
                'row': (33, 16, 18),
                'col': (15, 3, 13),
                'offset': (2, 0, 3),
            }
        elif mapping_name == "crb":
            # Channel-Row-Bank: Channel 最高位，适合跨 channel 随机
            return {
                'channel': (47, 45, 3),
                'stack': (44, 43, 2),
                'pseudo_channel': (42, 42, 1),
                'bank_group': (41, 39, 3),
                'bank': (38, 34, 5),
                'row': (33, 16, 18),
                'col': (15, 3, 13),
                'offset': (2, 0, 3),
            }
        else:
            raise ValueError(f"Unknown mapping: {mapping_name}")

    def _setup_bit_masks(self):
        """预计算位掩码和移位"""
        self.masks = {}
        for field, (msb, lsb, _) in self.mapping.items():
            self.masks[field] = {
                'msb': msb,
                'lsb': lsb,
                'mask': ((1 << (msb - lsb + 1)) - 1) << lsb,
                'shift': lsb,
            }

    def decode(self, addr: int) -> DecodedAddress:
        """解码地址

        Args:
            addr: 64-bit 物理地址

        Returns:
            DecodedAddress 对象

        Raises:
            AddressError: 地址越界或无效
        """
        # 验证地址对齐
        if addr & 0x7:  # 必须 8-byte 对齐
            raise AddressError(f"Address 0x{addr:x} not 8-byte aligned")

        result = DecodedAddress()

        # 解码各字段
        if 'stack' in self.masks:
            m = self.masks['stack']
            result.stack_id = (addr & m['mask']) >> m['shift']
            if result.stack_id >= self.config.stack_count:
                raise AddressError(f"Stack ID {result.stack_id} exceeds stack_count {self.config.stack_count}")

        if 'channel' in self.masks:
            m = self.masks['channel']
            result.channel_id = (addr & m['mask']) >> m['shift']
            if result.channel_id >= self.config.channels_per_stack:
                raise AddressError(f"Channel ID {result.channel_id} exceeds channels_per_stack")

        if 'pseudo_channel' in self.masks:
            m = self.masks['pseudo_channel']
            result.pseudo_channel_id = (addr & m['mask']) >> m['shift']
            if result.pseudo_channel_id >= self.config.pseudo_channels_per_channel:
                raise AddressError(f"Pseudo-channel ID exceeds config")

        if 'bank_group' in self.masks:
            m = self.masks['bank_group']
            result.bank_group_id = (addr & m['mask']) >> m['shift']
            if result.bank_group_id >= self.config.bank_groups_per_channel:
                raise AddressError(f"Bank group ID exceeds config")

        if 'bank' in self.masks:
            m = self.masks['bank']
            result.bank_id = (addr & m['mask']) >> m['shift']
            # Bank 范围检查
            max_banks = self.config.banks_per_pseudo_channel * self.config.pseudo_channels_per_channel
            if result.bank_id >= max_banks:
                raise AddressError(f"Bank ID {result.bank_id} exceeds max {max_banks}")

        if 'row' in self.masks:
            m = self.masks['row']
            result.row_id = (addr & m['mask']) >> m['shift']

        if 'col' in self.masks:
            m = self.masks['col']
            result.col_id = (addr & m['mask']) >> m['shift']

        if 'offset' in self.masks:
            m = self.masks['offset']
            result.byte_offset = (addr & m['mask']) >> m['shift']

        return result

    def encode(self, decoded: DecodedAddress) -> int:
        """编码地址 (反向操作)

        Args:
            decoded: 解码后的地址

        Returns:
            64-bit 物理地址
        """
        addr = 0

        for field, (msb, lsb, _) in self.mapping.items():
            field_name = field.replace('-', '_')
            if field == 'offset':
                value = decoded.byte_offset
            else:
                value = getattr(decoded, field_name + '_id', 0)
            addr |= (value & ((1 << (msb - lsb + 1)) - 1)) << lsb

        return addr

    def get_bank_key(self, decoded: DecodedAddress) -> Tuple[int, int, int, int]:
        """获取唯一 bank 标识

        用于 bank 状态查找和调度决策。

        Returns:
            (stack_id, channel_id, pseudo_channel_id, bank_id)
        """
        return (
            decoded.stack_id,
            decoded.channel_id,
            decoded.pseudo_channel_id,
            decoded.bank_id,
        )

    def get_row_key(self, decoded: DecodedAddress) -> Tuple[int, int, int, int, int]:
        """获取唯一 row 标识

        Returns:
            (stack_id, channel_id, pseudo_channel_id, bank_id, row_id)
        """
        return (
            decoded.stack_id,
            decoded.channel_id,
            decoded.pseudo_channel_id,
            decoded.bank_id,
            decoded.row_id,
        )

    def get_channel_id_from_addr(self, addr: int) -> int:
        """从地址中提取 channel ID

        快速方法，用于不需要完整解码的场景。

        Args:
            addr: 64-bit 地址

        Returns:
            Channel ID (0-7 for HBM3)
        """
        if 'channel' in self.masks:
            m = self.masks['channel']
            return (addr & m['mask']) >> m['shift']
        return 0

    def get_total_channels(self) -> int:
        """获取总 channel 数

        Returns:
            stack_count * channels_per_stack
        """
        return self.config.stack_count * self.config.channels_per_stack
