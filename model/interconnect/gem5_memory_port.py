"""
gem5 Memory Port Implementation
HBM4 与 gem5 内存端口接口的完整实现

提供:
1. Gem5SlavePort - gem5 从端口实现
2. TimingSimpleMemory 端口接口
3. HBM4 内存端口适配器
4. 缓存行处理 (64/128 bytes)
5. Burst 事务支持

Usage:
    from model.interconnect.gem5_memory_port import (
        Gem5SlavePort,
        HBM4MemoryPort,
        CacheLineHandler,
    )

    # 创建端口
    port = HBM4MemoryPort(name="dram.port", cache_line_size=64)
    port.connect(master_port)

    # 发送请求
    port.send_request(addr=0x1000, size=64, is_write=False)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from enum import Enum
from collections import deque
import logging

from model.dram.hbm4_spec import HBM4Spec

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Line Configuration
# ============================================================================

class CacheLineSize(Enum):
    """缓存行大小枚举"""
    SIZE_64 = 64
    SIZE_128 = 128
    SIZE_256 = 256


@dataclass
class CacheLineConfig:
    """缓存行配置"""
    line_size: int = 64                    # 缓存行大小 (bytes)
    burst_beats: int = 4                  # 每个缓存行的 beats 数
    beat_size: int = 16                   # 每个 beat 的大小 (bytes, 128-bit = 16 bytes)
    num_lanes: int = 128                  # HBM4 数据通道数

    def __post_init__(self):
        """验证配置一致性"""
        expected_beats = self.line_size // self.beat_size
        if expected_beats != self.burst_beats:
            logger.warning(
                f"Burst beats mismatch: calculated {expected_beats}, "
                f"got {self.burst_beats}"
            )
            self.burst_beats = expected_beats

    @property
    def addr_mask(self) -> int:
        """地址掩码 (用于缓存行对齐)"""
        return self.line_size - 1


@dataclass
class CacheLineHandler:
    """缓存行处理器

    处理缓存行对齐、分块和组装
    支持 64-byte 和 128-byte 缓存行
    """

    def __init__(self, line_size: int = 64):
        self.line_size = line_size
        self.config = CacheLineConfig(line_size=line_size)

        # 地址统计
        self._cache_hits = 0
        self._cache_misses = 0
        self._burst_requests = 0
        self._split_requests = 0

    def align_address(self, addr: int) -> int:
        """对齐地址到缓存行边界

        Args:
            addr: 原始地址

        Returns:
            对齐后的地址
        """
        return addr & ~(self.config.addr_mask)

    def is_aligned(self, addr: int, size: int) -> bool:
        """检查请求是否缓存行对齐

        Args:
            addr: 地址
            size: 请求大小

        Returns:
            True if aligned
        """
        return (addr & self.config.addr_mask) == 0 and size == self.line_size

    def split_request(
        self,
        addr: int,
        size: int,
    ) -> List[Tuple[int, int]]:
        """将请求分割成缓存行对齐的块

        Args:
            addr: 起始地址
            size: 请求大小 (bytes)

        Returns:
            [(aligned_addr, aligned_size), ...] 列表
        """
        chunks = []
        current_addr = addr
        remaining_size = size

        # 对齐起始地址
        if current_addr & self.config.addr_mask:
            aligned_addr = self.align_address(current_addr)
            aligned_size = min(
                self.line_size - (current_addr - aligned_addr),
                remaining_size
            )
            chunks.append((aligned_addr, aligned_size))
            current_addr += aligned_size
            remaining_size -= aligned_size
            self._split_requests += 1

        # 处理中间完整缓存行
        while remaining_size >= self.line_size:
            chunks.append((current_addr, self.line_size))
            current_addr += self.line_size
            remaining_size -= self.line_size

        # 处理尾部
        if remaining_size > 0:
            chunks.append((current_addr, remaining_size))
            self._split_requests += 1

        return chunks

    def calculate_beats(self, size: int) -> int:
        """计算给定大小的 beats 数量

        Args:
            size: 请求大小 (bytes)

        Returns:
            beats 数量
        """
        return (size + self.config.beat_size - 1) // self.config.beat_size

    def calculate_burst_cycles(self, size: int) -> int:
        """计算突发传输所需周期数

        对于 HBM4，每 4 beats 需要 1 个 FLINE 周期

        Args:
            size: 请求大小 (bytes)

        Returns:
            传输周期数
        """
        beats = self.calculate_beats(size)
        # HBM4 FLINE = 4 beats per cycle
        return (beats + 3) // 4

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存行处理统计"""
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'burst_requests': self._burst_requests,
            'split_requests': self._split_requests,
            'hit_rate': (
                self._cache_hits / max(1, self._cache_hits + self._cache_misses)
            ),
        }

    def record_hit(self) -> None:
        """记录缓存命中"""
        self._cache_hits += 1

    def record_miss(self) -> None:
        """记录缓存未命中"""
        self._cache_misses += 1


# ============================================================================
# Port Interface Types
# ============================================================================

class PortState(Enum):
    """端口状态"""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class PortConfig:
    """端口配置"""
    name: str = "port"
    latency: int = 10                     # 端口延迟 (cycles)
    bandwidth_gbs: float = 2048.0        # 带宽 (GB/s)
    queue_depth: int = 32                  # 队列深度
    enable_backpressure: bool = True       # 启用背压


@dataclass
class PortStatistics:
    """端口统计"""
    packets_sent: int = 0
    packets_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    stalls: int = 0                       # 背压导致的停顿
    timeouts: int = 0

    @property
    def avg_latency(self) -> float:
        if self.packets_received == 0:
            return 0.0
        return self.bytes_sent / max(1, self.packets_sent) if self.packets_sent else 0.0


# ============================================================================
# Memory Port Implementation
# ============================================================================

class Gem5SlavePortBase:
    """gem5 从端口基类

    定义端口接口的基本行为
    """

    def __init__(
        self,
        name: str,
        config: Optional[PortConfig] = None,
    ):
        self.name = name
        self.config = config or PortConfig(name=name)
        self.state = PortState.IDLE

        # 连接状态
        self.peer: Optional["Gem5MasterPortBase"] = None

        # 队列
        self._request_queue: deque = deque(maxlen=self.config.queue_depth)

        # 统计
        self.stats = PortStatistics()

        # 回调
        self._on_recv: Optional[Callable] = None
        self._on_send: Optional[Callable] = None

    def connect(self, peer: "Gem5MasterPortBase") -> None:
        """连接到 master 端口

        Args:
            peer: master 端口对象
        """
        self.peer = peer
        peer.peer = self
        logger.debug(f"Port {self.name}: connected to {peer.name}")

    def send_response(self, response: Any) -> bool:
        """发送响应

        Args:
            response: 响应对象

        Returns:
            True if sent successfully
        """
        if self.peer is None:
            logger.error(f"Port {self.name}: not connected")
            return False

        self.stats.packets_sent += 1
        if response.data:
            self.stats.bytes_sent += len(response.data) * 8

        if self._on_send:
            self._on_send(response)

        return True

    def recv_request(self) -> Optional[Any]:
        """接收请求

        Returns:
            请求对象，如果没有请求则返回 None
        """
        if self._request_queue:
            return self._request_queue.popleft()
        return None

    def set_callback(self, event: str, callback: Callable) -> None:
        """设置回调函数

        Args:
            event: 事件类型 ("recv", "send")
            callback: 回调函数
        """
        if event == "recv":
            self._on_recv = callback
        elif event == "send":
            self._on_send = callback


class Gem5MasterPortBase:
    """gem5 主端口基类"""

    def __init__(
        self,
        name: str,
        config: Optional[PortConfig] = None,
    ):
        self.name = name
        self.config = config or PortConfig(name=name)
        self.state = PortState.IDLE

        # 连接状态
        self.peer: Optional[Gem5SlavePortBase] = None

        # 统计
        self.stats = PortStatistics()

        # 回调
        self._on_recv: Optional[Callable] = None
        self._on_send: Optional[Callable] = None

    def connect(self, peer: Gem5SlavePortBase) -> None:
        """连接到 slave 端口"""
        self.peer = peer
        peer.peer = self
        logger.debug(f"Port {self.name}: connected to {peer.name}")

    def send_request(self, request: Any) -> bool:
        """发送请求"""
        if self.peer is None:
            logger.error(f"Port {self.name}: not connected")
            return False

        self.stats.packets_sent += 1
        if hasattr(request, 'size'):
            self.stats.bytes_sent += request.size

        if self._on_send:
            self._on_send(request)

        return True

    def recv_response(self) -> Optional[Any]:
        """接收响应"""
        return None  # 由子类实现

    def set_callback(self, event: str, callback: Callable) -> None:
        """设置回调函数"""
        if event == "recv":
            self._on_recv = callback
        elif event == "send":
            self._on_send = callback


# ============================================================================
# HBM4 Memory Port
# ============================================================================

@dataclass
class HBM4MemoryRequest:
    """HBM4 内存请求

    扩展基本请求，添加 HBM4 特定字段
    """
    req_id: int
    addr: int
    size: int
    is_write: bool

    # HBM4 特定字段
    channel_id: int = 0
    pseudo_channel_id: int = 0
    bank_id: int = 0
    row_id: int = 0
    col_id: int = 0

    # QoS
    qos: int = 8
    master_id: int = 0

    # Burst 信息
    num_beats: int = 1
    cur_beat: int = 0

    # 数据
    data: Optional[List[int]] = None

    # 状态
    issued_cycle: int = 0
    state: str = "pending"


@dataclass
class HBM4MemoryResponse:
    """HBM4 内存响应"""
    req_id: int
    addr: int
    status: str = "OK"
    data: Optional[List[int]] = None

    # 统计
    latency_cycles: int = 0
    error_code: int = 0


class HBM4MemoryPort:
    """HBM4 内存端口实现

    实现与 gem5 TimingSimpleMemory 兼容的端口接口
    支持:
    - 32 通道地址解码
    - 伪通道调度
    - 缓存行处理 (64/128 bytes)
    - Burst 事务支持
    - QoS 优先级
    """

    def __init__(
        self,
        name: str,
        spec: Optional[HBM4Spec] = None,
        cache_line_size: int = 64,
        config: Optional[PortConfig] = None,
    ):
        self.name = name
        self.spec = spec or HBM4Spec()
        self.config = config or PortConfig(name=name)

        # 缓存行处理器
        self.cache_handler = CacheLineHandler(line_size=cache_line_size)

        # 端口状态
        self.state = PortState.IDLE
        self.peer: Optional[Gem5MasterPortBase] = None

        # 请求队列
        self._pending_requests: Dict[int, HBM4MemoryRequest] = {}
        self._completed_responses: Dict[int, HBM4MemoryResponse] = {}
        self._request_counter = 0

        # 队列
        self._request_queue: deque = deque(maxlen=self.config.queue_depth)
        self._response_queue: deque = deque(maxlen=self.config.queue_depth)

        # 统计
        self.stats = PortStatistics()
        self._channel_stats: Dict[int, Dict[str, int]] = {
            ch: {'requests': 0, 'reads': 0, 'writes': 0}
            for ch in range(self.spec.channels)
        }

        # 回调
        self._on_request: Optional[Callable] = None
        self._on_response: Optional[Callable] = None

        # 地址解码器
        from model.controller.hbm4_address_decoder import HBM4AddressDecoder
        self._decoder = HBM4AddressDecoder(spec=self.spec)

        # 当前周期
        self._current_cycle = 0

        logger.info(
            f"HBM4MemoryPort '{name}' created: "
            f"channels={self.spec.channels}, "
            f"cache_line={cache_line_size}B"
        )

    def connect(self, peer: Gem5MasterPortBase) -> None:
        """连接到 master 端口"""
        self.peer = peer
        logger.debug(f"HBM4MemoryPort '{self.name}': connected to {peer.name}")

    def send_request(
        self,
        addr: int,
        size: int,
        is_write: bool,
        data: Optional[List[int]] = None,
        qos: int = 8,
        master_id: int = 0,
    ) -> Optional[int]:
        """发送内存请求

        Args:
            addr: 目标地址
            size: 请求大小 (bytes)
            is_write: 是否为写请求
            data: 写数据
            qos: QoS 优先级
            master_id: Master ID

        Returns:
            请求 ID，失败返回 None
        """
        # 检查队列满
        if len(self._request_queue) >= self.config.queue_depth:
            logger.warning(f"Port '{self.name}': request queue full")
            self.stats.stalls += 1
            return None

        # 生成请求 ID
        req_id = self._request_counter
        self._request_counter += 1

        # 解码地址
        decoded = self._decoder.decode(addr)

        # 计算 beats
        num_beats = self.cache_handler.calculate_beats(size)

        # 创建请求
        request = HBM4MemoryRequest(
            req_id=req_id,
            addr=addr,
            size=size,
            is_write=is_write,
            channel_id=decoded.channel_id,
            pseudo_channel_id=decoded.pseudo_channel_id,
            bank_id=decoded.bank_id,
            row_id=decoded.row_id,
            col_id=decoded.col_id,
            qos=qos,
            master_id=master_id,
            num_beats=num_beats,
            data=data,
            issued_cycle=self._current_cycle,
            state="pending",
        )

        # 入队
        self._pending_requests[req_id] = request
        self._request_queue.append(request)

        # 更新统计
        self.stats.packets_sent += 1
        self.stats.bytes_sent += size
        self._channel_stats[decoded.channel_id]['requests'] += 1
        if is_write:
            self._channel_stats[decoded.channel_id]['writes'] += 1
        else:
            self._channel_stats[decoded.channel_id]['reads'] += 1

        logger.debug(
            f"HBM4 request sent: id={req_id}, addr=0x{addr:x}, "
            f"size={size}, ch={decoded.channel_id}, "
            f"pch={decoded.pseudo_channel_id}, qos={qos}"
        )

        # 回调
        if self._on_request:
            self._on_request(request)

        return req_id

    def recv_response(
        self,
        req_id: Optional[int] = None,
        timeout_cycles: int = 1000,
    ) -> Optional[HBM4MemoryResponse]:
        """接收内存响应

        Args:
            req_id: 特定请求 ID，None 表示接收任意响应
            timeout_cycles: 超时周期数

        Returns:
            响应对象，超时返回 None
        """
        start_cycle = self._current_cycle

        while (self._current_cycle - start_cycle) < timeout_cycles:
            # 检查特定请求
            if req_id is not None and req_id in self._completed_responses:
                return self._completed_responses.pop(req_id)

            # 检查队列
            if self._response_queue:
                return self._response_queue.popleft()

            # 推进周期
            self.tick()

        # 超时
        logger.warning(f"Response timeout: req_id={req_id}")
        self.stats.timeouts += 1
        return None

    def send_response(self, response: HBM4MemoryResponse) -> bool:
        """发送响应

        Args:
            response: 响应对象

        Returns:
            True if sent successfully
        """
        if self.peer is None:
            logger.error(f"Port '{self.name}': not connected")
            return False

        self.stats.packets_sent += 1
        if response.data:
            self.stats.bytes_sent += len(response.data) * 8

        # 缓存响应
        self._completed_responses[response.req_id] = response
        self._response_queue.append(response)

        # 从 pending 移除
        if response.req_id in self._pending_requests:
            del self._pending_requests[response.req_id]

        # 回调
        if self._on_response:
            self._on_response(response)

        return True

    def tick(self) -> None:
        """推进一个周期"""
        self._current_cycle += 1

        # 处理队列中的请求
        self._process_queue()

    def _process_queue(self) -> None:
        """处理请求队列"""
        if not self._request_queue:
            return

        # 按 QoS 优先级排序
        pending = sorted(
            self._request_queue,
            key=lambda r: (-r.qos, r.issued_cycle)
        )

        # 处理高优先级请求
        for request in pending[:8]:  # 每周期最多处理 8 个请求
            if request.state == "pending":
                request.state = "processing"
                # 这里应该调用实际的 HBM 控制器
                # 暂时模拟完成
                if self._current_cycle - request.issued_cycle >= self.config.latency:
                    response = HBM4MemoryResponse(
                        req_id=request.req_id,
                        addr=request.addr,
                        status="OK",
                        data=[0] * (request.size // 8) if not request.is_write else None,
                        latency_cycles=self._current_cycle - request.issued_cycle,
                    )
                    self.send_response(response)
                    self._request_queue.remove(request)

    def get_pending_count(self) -> int:
        """获取待处理请求数"""
        return len(self._pending_requests)

    def get_channel_load(self, channel_id: int) -> int:
        """获取通道负载

        Args:
            channel_id: 通道 ID

        Returns:
            待处理请求数
        """
        return sum(
            1 for req in self._pending_requests.values()
            if req.channel_id == channel_id
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'port_name': self.name,
            'packets_sent': self.stats.packets_sent,
            'packets_received': self.stats.packets_received,
            'bytes_sent': self.stats.bytes_sent,
            'bytes_received': self.stats.bytes_received,
            'stalls': self.stats.stalls,
            'timeouts': self.stats.timeouts,
            'pending_requests': len(self._pending_requests),
            'current_cycle': self._current_cycle,
            'cache_line_stats': self.cache_handler.get_stats(),
            'channel_stats': self._channel_stats.copy(),
        }

    def set_callback(self, event: str, callback: Callable) -> None:
        """设置回调函数"""
        if event == "request":
            self._on_request = callback
        elif event == "response":
            self._on_response = callback


# ============================================================================
# Traffic Generator Interface
# ============================================================================

class TrafficGeneratorInterface:
    """Traffic Generator 接口

    为 CPU/NPU/GPU 提供标准化的流量生成接口
    支持:
    - 顺序访问模式
    - 随机访问模式
    - 热点访问模式
    - stride 访问模式
    """

    class AccessPattern(Enum):
        """访问模式"""
        SEQUENTIAL = "sequential"
        RANDOM = "random"
        HOTSPOT = "hotspot"
        STRIDE = "stride"

    def __init__(
        self,
        name: str,
        port: HBM4MemoryPort,
        spec: Optional[HBM4Spec] = None,
    ):
        self.name = name
        self.port = port
        self.spec = spec or HBM4Spec()

        # 模式配置
        self.pattern = self.AccessPattern.SEQUENTIAL
        self.base_addr = 0x1000_0000
        self.access_size = 64
        self.qos = 8

        # 热点配置
        self.hotspot_base = 0x1000_0000
        self.hotspot_size = 0x1000_0000  # 256 MB hotspot
        self.hotspot_ratio = 0.8  # 80% access hotspot

        # Stride 配置
        self.stride = 64

        # 统计
        self._requests_sent = 0
        self._responses_received = 0
        self._total_latency = 0

        # 内部状态
        self._addr = self.base_addr
        import random
        self._random = random.Random(42)  # 可重现的随机

    def generate_request(self) -> Optional[int]:
        """生成下一个请求

        Returns:
            请求 ID
        """
        # 根据模式生成地址
        addr = self._generate_address()

        # 发送到端口
        req_id = self.port.send_request(
            addr=addr,
            size=self.access_size,
            is_write=False,
            qos=self.qos,
        )

        if req_id is not None:
            self._requests_sent += 1

        return req_id

    def _generate_address(self) -> int:
        """根据当前模式生成地址"""
        if self.pattern == self.AccessPattern.SEQUENTIAL:
            addr = self._addr
            self._addr += self.access_size
            # 环绕
            if self._addr >= self.base_addr + 0x1_0000_0000:
                self._addr = self.base_addr

        elif self.pattern == self.AccessPattern.RANDOM:
            addr = self.base_addr + (
                self._random.randint(0, 0xFFFF_FFFF) & ~63
            )

        elif self.pattern == self.AccessPattern.HOTSPOT:
            if self._random.random() < self.hotspot_ratio:
                addr = self.hotspot_base + (
                    self._random.randint(0, self.hotspot_size) & ~63
                )
            else:
                addr = self.base_addr + (
                    self._random.randint(0, 0xFFFF_FFFF) & ~63
                )

        elif self.pattern == self.AccessPattern.STRIDE:
            addr = self._addr
            self._addr += self.stride
            # 环绕
            if self._addr >= self.base_addr + 0x1_0000_0000:
                self._addr = self.base_addr

        else:
            addr = self._addr

        return addr

    def generate_burst(self, num_requests: int) -> List[int]:
        """生成突发请求序列

        Args:
            num_requests: 请求数量

        Returns:
            请求 ID 列表
        """
        req_ids = []
        for _ in range(num_requests):
            req_id = self.generate_request()
            if req_id is not None:
                req_ids.append(req_id)
        return req_ids

    def record_response(self, latency: int) -> None:
        """记录响应到达

        Args:
            latency: 响应延迟
        """
        self._responses_received += 1
        self._total_latency += latency

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_latency = (
            self._total_latency / self._responses_received
            if self._responses_received > 0 else 0
        )
        return {
            'pattern': self.pattern.value,
            'requests_sent': self._requests_sent,
            'responses_received': self._responses_received,
            'average_latency': avg_latency,
            'outstanding': self._requests_sent - self._responses_received,
        }

    def set_pattern(self, pattern: AccessPattern) -> None:
        """设置访问模式"""
        self.pattern = pattern

    def set_base_address(self, addr: int) -> None:
        """设置基地址"""
        self.base_addr = addr & ~63  # 对齐到缓存行
        self._addr = self.base_addr

    def set_access_size(self, size: int) -> None:
        """设置访问大小"""
        self.access_size = size & ~63  # 对齐到缓存行


# ============================================================================
# Factory Functions
# ============================================================================

def create_memory_port(
    name: str,
    cache_line_size: int = 64,
    spec: Optional[HBM4Spec] = None,
    **kwargs,
) -> HBM4MemoryPort:
    """创建 HBM4 内存端口

    Args:
        name: 端口名称
        cache_line_size: 缓存行大小
        spec: HBM4 规范
        **kwargs: 额外参数

    Returns:
        HBM4MemoryPort 实例
    """
    return HBM4MemoryPort(
        name=name,
        spec=spec,
        cache_line_size=cache_line_size,
        **kwargs,
    )


def create_traffic_generator(
    name: str,
    port: HBM4MemoryPort,
    pattern: str = "sequential",
    spec: Optional[HBM4Spec] = None,
) -> TrafficGeneratorInterface:
    """创建 Traffic Generator

    Args:
        name: 生成器名称
        port: 连接到 HBM4 端口
        pattern: 访问模式
        spec: HBM4 规范

    Returns:
        TrafficGeneratorInterface 实例
    """
    tg = TrafficGeneratorInterface(name=name, port=port, spec=spec)

    # 设置模式
    pattern_map = {
        "sequential": TrafficGeneratorInterface.AccessPattern.SEQUENTIAL,
        "random": TrafficGeneratorInterface.AccessPattern.RANDOM,
        "hotspot": TrafficGeneratorInterface.AccessPattern.HOTSPOT,
        "stride": TrafficGeneratorInterface.AccessPattern.STRIDE,
    }
    if pattern in pattern_map:
        tg.set_pattern(pattern_map[pattern])

    return tg


# ============================================================================
# __init__.py exports
# ============================================================================

__all__ = [
    # Cache line
    "CacheLineSize",
    "CacheLineConfig",
    "CacheLineHandler",

    # Port types
    "PortState",
    "PortConfig",
    "PortStatistics",

    # Port implementations
    "Gem5SlavePortBase",
    "Gem5MasterPortBase",
    "HBM4MemoryPort",
    "HBM4MemoryRequest",
    "HBM4MemoryResponse",

    # Traffic generator
    "TrafficGeneratorInterface",

    # Factory
    "create_memory_port",
    "create_traffic_generator",
]
