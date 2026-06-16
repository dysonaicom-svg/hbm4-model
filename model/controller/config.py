"""
HBM Configuration Module
参考设计文档 2026-06-15-hbm-system-model-design.md 的 5.1.6 节
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
from ..dram.timing import HBM3Timing, HBM4Timing


@dataclass
class HBMConfig:
    """HBM 控制器配置类
    
    所有参数都有默认值，可以从 YAML 文件或字典加载。
    支持 HBM3 标准配置 (JEDEC JESD238)。
    
    Attributes:
        stack_count: HBM stack 数量 (1-8)
        channels_per_stack: 每个 stack 的通道数 (4-16)
        pseudo_channels_per_channel: 每个通道的伪通道数 (1-4)
        banks_per_pseudo_channel: 每个伪通道的 bank 数 (8-32)
        bank_groups_per_channel: 每个通道的 bank group 数 (4-16)
        row_size: 行大小 (bytes)
        burst_length: 突发长度 (FLINE)
        data_rate: 每引脚数据速率 (bits/s)
        io_width: 接口宽度 (bits)
        read_latency_base: 基础读延迟 (cycles)
        write_latency_base: 基础写延迟 (cycles)
        phy_latency: PHY 延迟 (cycles)
        queue_depth: 最大请求队列深度
        max_outstanding: 最大未完成请求数
        address_mapping: 地址映射方案 ("rbc", "bcr", "crb", "custom")
        scheduler_mode: 调度器模式 ("fr-fcfs", "qos")
        write_drain_policy: 写 drain 策略 ("immediate", "threshold", "interval")
        refresh_interval: 刷新间隔 (seconds, tREFI)
        refresh_penalty: 刷新惩罚 (seconds, tRFC)
    """
    # Stack 配置
    stack_count: int = 2                    # 1-8
    channels_per_stack: int = 8             # 4-16
    pseudo_channels_per_channel: int = 2   # 1-4
    banks_per_pseudo_channel: int = 16       # 8-32
    bank_groups_per_channel: int = 8        # 4-16
    
    # 存储配置
    row_size: int = 2048                    # bytes
    burst_length: int = 32                  # FLINE
    
    # 性能配置
    data_rate: float = 6.4e9               # bits/s per pin
    io_width: int = 1024                    # bits
    
    # 延迟配置 (cycles @ tCK)
    read_latency_base: int = 30
    write_latency_base: int = 10
    phy_latency: int = 20
    
    # 队列配置
    queue_depth: int = 32                   # 16-128
    max_outstanding: int = 16               # 8-64
    
    # 调度配置
    address_mapping: str = "rbc"            # "rbc", "bcr", "crb", "custom"
    scheduler_mode: str = "fr-fcfs"         # "fr-fcfs" or "qos"
    write_drain_policy: str = "threshold"   # "immediate", "threshold", "interval"
    
    # Refresh 配置 (seconds)
    refresh_interval: float = 3.9e-6        # tREFI
    refresh_penalty: float = 230e-9          # tRFC
    
    # QoS 带宽保证 (GB/s per stack)
    bw_guarantee_critical: float = 200.0
    bw_guarantee_high: float = 300.0
    bw_guarantee_normal: float = 200.0
    bw_guarantee_low: float = 100.0

    # 时序参数 (从 dram.timing 导入)
    timing: HBM3Timing = field(default_factory=HBM3Timing)
    
    @classmethod
    def from_yaml(cls, path: str) -> "HBMConfig":
        """从 YAML 文件加载配置
        
        Args:
            path: YAML 文件路径
            
        Returns:
            HBMConfig 实例
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HBMConfig":
        """从字典加载配置
        
        Args:
            data: 配置参数字典
            
        Returns:
            HBMConfig 实例
        """
        # 过滤掉 None 值和未知参数
        valid_data = {k: v for k, v in data.items() if v is not None}
        return cls(**valid_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典
        
        Returns:
            配置参数字典
        """
        return self.__dict__
    
    def calc_bandwidth(self) -> float:
        """计算理论峰值带宽 (GB/s)

        基于数据速率和接口宽度计算单 stack 带宽。

        公式: bandwidth = data_rate (Gb/s/pin) * io_width (bits) / 8

        注意: io_width = 1024 已经包含所有 8 channels 的总宽度 (8 * 128 = 1024)

        Example HBM3 (per stack):
            data_rate = 6.4e9 bits/s = 6.4 Gb/s/pin
            io_width = 1024 bits (8 channels * 128 bits)
            => bandwidth = 6.4 * 1024 / 8 = 819.2 GB/s

        Returns:
            理论峰值带宽 (GB/s) per stack
        """
        # data_rate 单位是 bits/s，需要转换为 Gb/s (除以 1e9)
        # io_width 单位是 bits
        # bandwidth = data_rate (Gb/s) * io_width (bits) / 8
        data_rate_gb = self.data_rate / 1e9  # Convert to Gb/s
        total_bw = data_rate_gb * self.io_width / 8.0
        return total_bw

    def calc_bandwidth_total(self) -> float:
        """计算所有 stack 的总带宽 (GB/s)"""
        return self.calc_bandwidth() * self.stack_count

    def __repr__(self) -> str:
        bw_per_stack = self.calc_bandwidth()
        bw_str = f"{bw_per_stack/1e3:.2f} TB/s" if bw_per_stack > 1e3 else f"{bw_per_stack:.1f} GB/s"
        return f"HBMConfig(stack={self.stack_count}, ch={self.channels_per_stack}, bw={bw_str})"


# 默认 HBM3 配置
HBM3_DEFAULT = HBMConfig(
    stack_count=2,
    channels_per_stack=8,
    pseudo_channels_per_channel=2,
    banks_per_pseudo_channel=16,
    bank_groups_per_channel=8,
    row_size=2048,
    burst_length=32,
    data_rate=6.4e9,
    io_width=1024,
    read_latency_base=30,
    write_latency_base=10,
    phy_latency=20,
    queue_depth=32,
    max_outstanding=16,
    address_mapping="rbc",
    scheduler_mode="fr-fcfs",
    write_drain_policy="threshold",
    refresh_interval=3.9e-6,
    refresh_penalty=230e-9,
)


# HBM4 默认配置 (基于 JEDEC JESD270-4A)
# 特点: 8 GT/s DDR (tCK=125ps), 32 channels per stack, 2 TB/s bandwidth
HBM4_DEFAULT = HBMConfig(
    stack_count=4,                      # HBM4 支持最多 4 stacks
    channels_per_stack=32,              # HBM4: 32 channels per stack
    pseudo_channels_per_channel=2,
    banks_per_pseudo_channel=16,
    bank_groups_per_channel=8,
    row_size=2048,
    burst_length=4,                      # FLINE burst length
    data_rate=8.0e9,                    # 8 GT/s DDR (HBM4 baseline)
    io_width=2048,                       # 32 channels * 64 bits (or 16 channels * 128 bits)
    read_latency_base=25,               # Optimized read latency
    write_latency_base=8,               # Optimized write latency
    phy_latency=15,                     # Faster PHY
    queue_depth=64,                      # Larger queue depth
    max_outstanding=32,                 # Higher concurrency
    address_mapping="rbc",
    scheduler_mode="fr-fcfs",
    write_drain_policy="threshold",
    refresh_interval=3.9e-6,
    refresh_penalty=180e-9,            # Faster refresh (tRFC=180 cycles @ 8GHz)
    timing=HBM4Timing.for_8gbps(),       # Use 8 GT/s timing
)
