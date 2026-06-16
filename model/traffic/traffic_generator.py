"""
HBM4 Traffic Generator

Layer 0 of the 5-layer HBM system architecture.
Generates realistic traffic patterns for AI training/inference workloads
and synthetic traffic patterns for stress testing.

Reference:
- Design Document: docs/design/2026-06-15-hbm-system-model-design.md
- Requirements: research/hbm4-logic-base-die/requirements/hbm4_logic_base_die_requirements.md
"""

import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple, Iterator, Any
from collections import deque
import numpy as np

from model.controller.request import HBMRequest, RequestState
from model.dram.hbm4_spec import HBM4Spec


class TrafficPattern(IntEnum):
    """Traffic pattern types"""
    # AI Training Patterns
    TRAINING_WEIGHT_UPDATE = 1      # Large sequential writes (weight gradients)
    TRAINING_GRADIENT = 2           # Large sequential reads (gradient computation)
    TRAINING_FEATURE_MAP = 3        # Mixed read/write (feature maps)

    # AI Inference Patterns
    INFERENCE_BURST_READ = 10       # Short burst reads
    INFERENCE_WEIGHT_REUSE = 11     # High locality weight reads
    INFERENCE_MIXED_PRECISION = 12  # FP16/BF16/INT8 mixed precision

    # Synthetic Patterns
    SYNTHETIC_FIXED_RATE = 20       # Constant request rate
    SYNTHETIC_BURST = 21            # Bursty traffic
    SYNTHETIC_RANDOM = 22           # Random address access
    SYNTHETIC_RAMP_UP = 23          # Gradually increasing rate
    SYNTHETIC_RAMP_DOWN = 24        # Gradually decreasing rate
    SYNTHETIC_SINUSOIDAL = 25       # Sinusoidal traffic pattern

    # Special Patterns
    TRACE_REPLAY = 30              # Replay from recorded trace
    ADDRESS_PATTERN = 31            # Custom address pattern


class DataPrecision(IntEnum):
    """Data precision types for mixed precision inference"""
    FP32 = 32
    FP16 = 16
    BF16 = 16
    INT8 = 8
    INT4 = 4


@dataclass
class TrafficConfig:
    """Traffic generation configuration

    Attributes:
        read_write_ratio: Ratio of read to write requests (0.0-1.0)
        request_rate: Requests per second
        burst_size: Number of requests per burst
        address_pattern: Base address pattern
        qos_distribution: Distribution of QoS levels (0-15)
        batch_size: Batch size for AI training patterns
        precision: Data precision for inference
    """
    # Traffic mix
    read_write_ratio: float = 0.7  # 70% reads, 30% writes

    # Rate control
    request_rate: float = 1e6       # requests/second
    burst_size: int = 32            # requests per burst

    # Address configuration
    base_address: int = 0x100000000
    address_range: int = 0x10000000000  # 256 GB range
    address_stride: int = 64         # 64-byte stride

    # QoS configuration
    qos_distribution: Dict[int, float] = field(default_factory=lambda: {
        15: 0.05,  # Critical (5%)
        12: 0.15,  # High (15%)
        8: 0.50,   # Normal (50%)
        4: 0.20,   # Low (20%)
        0: 0.10,   # Idle (10%)
    })

    # AI-specific parameters
    batch_size: int = 1
    sequence_length: int = 512
    hidden_size: int = 4096

    # Precision configuration
    precision: DataPrecision = DataPrecision.FP16

    # HBM4 configuration
    channels: int = 32
    pseudo_channels: int = 64
    banks_per_channel: int = 16

    def __post_init__(self):
        """Validate traffic configuration parameters"""
        # Validate read_write_ratio
        if not 0.0 <= self.read_write_ratio <= 1.0:
            raise ValueError(
                f"read_write_ratio must be between 0.0 and 1.0, got {self.read_write_ratio}"
            )

        # Validate request_rate
        if self.request_rate <= 0:
            raise ValueError(
                f"request_rate must be positive, got {self.request_rate}"
            )

        # Validate burst_size
        if self.burst_size <= 0:
            raise ValueError(
                f"burst_size must be positive, got {self.burst_size}"
            )

        # Validate addresses
        if self.base_address < 0:
            raise ValueError(
                f"base_address must be non-negative, got {self.base_address}"
            )
        if self.address_range <= 0:
            raise ValueError(
                f"address_range must be positive, got {self.address_range}"
            )
        if self.address_stride <= 0:
            raise ValueError(
                f"address_stride must be positive, got {self.address_stride}"
            )
        if self.base_address + self.address_range > (1 << 64):
            raise ValueError(
                f"Address range exceeds 64-bit address space"
            )

        # Validate QoS distribution
        if not self.qos_distribution:
            raise ValueError("qos_distribution cannot be empty")
        total_prob = sum(self.qos_distribution.values())
        if abs(total_prob - 1.0) > 0.001:
            raise ValueError(
                f"qos_distribution probabilities must sum to 1.0, got {total_prob}"
            )
        for qos, prob in self.qos_distribution.items():
            if not 0 <= qos <= 15:
                raise ValueError(f"QoS level must be 0-15, got {qos}")
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"QoS probability must be 0.0-1.0, got {prob}")

        # Validate AI parameters
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}"
            )
        if self.sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )
        if self.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {self.hidden_size}"
            )

        # Validate HBM4 configuration
        spec = HBM4Spec()
        if self.channels <= 0 or self.channels > spec.channels:
            raise ValueError(
                f"channels must be 1-{spec.channels}, got {self.channels}"
            )
        if self.pseudo_channels <= 0 or self.pseudo_channels > spec.pseudo_channels:
            raise ValueError(
                f"pseudo_channels must be 1-{spec.pseudo_channels}, got {self.pseudo_channels}"
            )
        if self.banks_per_channel <= 0 or self.banks_per_channel > spec.banks_per_pseudo_channel:
            raise ValueError(
                f"banks_per_channel must be 1-{spec.banks_per_pseudo_channel}, got {self.banks_per_channel}"
            )


@dataclass
class AddressGenerator:
    """Configurable address generator

    Supports multiple address patterns for different traffic types.
    """
    base_address: int = 0x100000000
    address_range: int = 0x10000000000
    stride: int = 64

    # Pattern state
    _current_addr: int = 0
    _current_bank: int = 0
    _channel_round_robin: int = 0

    def sequential(self, count: int = 1) -> List[int]:
        """Generate sequential addresses"""
        addresses = []
        for _ in range(count):
            addr = self.base_address + self._current_addr
            self._current_addr = (self._current_addr + self.stride) % self.address_range
            addresses.append(addr)
        return addresses

    def random(self, count: int = 1) -> List[int]:
        """Generate random addresses"""
        return [
            self.base_address + random.randint(0, self.address_range - 1)
            for _ in range(count)
        ]

    def stride_access(self, count: int = 1, stride: Optional[int] = None) -> List[int]:
        """Generate strided addresses"""
        s = stride if stride is not None else self.stride
        addresses = []
        for _ in range(count):
            addr = self.base_address + self._current_addr
            self._current_addr = (self._current_addr + s) % self.address_range
            addresses.append(addr)
        return addresses

    def bank_round_robin(self, num_banks: int, count: int = 1) -> List[int]:
        """Generate addresses with bank-level round-robin"""
        addresses = []
        for _ in range(count):
            bank = self._current_bank % num_banks
            addr = self.base_address + (bank * self.stride)
            self._current_bank += 1
            addresses.append(addr)
        return addresses

    def channel_round_robin(self, num_channels: int, count: int = 1) -> List[int]:
        """Generate addresses with channel-level round-robin"""
        addresses = []
        for _ in range(count):
            channel = self._channel_round_robin % num_channels
            addr = self.base_address + (channel * self.address_range // num_channels)
            self._channel_round_robin += 1
            addresses.append(addr)
        return addresses

    def reset(self):
        """Reset generator state"""
        self._current_addr = 0
        self._current_bank = 0
        self._channel_round_robin = 0


class AITrainingPattern(ABC):
    """Base class for AI training traffic patterns"""

    @abstractmethod
    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate HBM requests for AI training"""
        pass


class WeightUpdatePattern(AITrainingPattern):
    """Neural network weight update pattern

    Characteristics:
    - Large sequential writes (gradient updates)
    - Low locality (scattered across weight matrix)
    - High bandwidth requirements
    """

    def __init__(self, weight_matrix_size: int = 128 * 1024 * 1024):
        """Initialize weight update pattern

        Args:
            weight_matrix_size: Size of weight matrix in bytes
        """
        self.weight_matrix_size = weight_matrix_size
        self.addr_gen = AddressGenerator()

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate weight update (write) requests"""
        requests = []
        for _ in range(count):
            addr = self.addr_gen.sequential(1)[0]
            request = HBMRequest(
                addr=addr,
                length=64,  # 64 bytes per weight update
                is_read=False,  # Write for gradient update
                qos=12,  # High QoS for training
                burst_length=config.burst_size,
            )
            requests.append(request)
        return requests


class GradientComputationPattern(AITrainingPattern):
    """Gradient computation pattern

    Characteristics:
    - Large sequential reads (forward pass data)
    - Medium locality
    - High bandwidth requirements
    """

    def __init__(self, gradient_size: int = 256 * 1024 * 1024):
        """Initialize gradient computation pattern

        Args:
            gradient_size: Size of gradient tensor in bytes
        """
        self.gradient_size = gradient_size
        self.addr_gen = AddressGenerator()

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate gradient computation (read) requests"""
        requests = []
        for _ in range(count):
            addr = self.addr_gen.sequential(1)[0]
            request = HBMRequest(
                addr=addr,
                length=64,  # 64 bytes per gradient read
                is_read=True,  # Read for gradient computation
                qos=12,  # High QoS for training
                burst_length=config.burst_size,
            )
            requests.append(request)
        return requests


class FeatureMapTransferPattern(AITrainingPattern):
    """Feature map transfer pattern

    Characteristics:
    - Mixed read/write (feature maps)
    - High locality (within batch)
    - Variable size based on layer dimensions
    """

    def __init__(self, batch_size: int = 32, feature_size: int = 64 * 1024):
        """Initialize feature map transfer pattern

        Args:
            batch_size: Training batch size
            feature_size: Size of feature map per sample
        """
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.addr_gen = AddressGenerator()
        self._is_read_next = False

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate feature map transfer requests (mixed read/write)"""
        requests = []
        for _ in range(count):
            addr = self.addr_gen.sequential(1)[0]

            # Alternate between write (activation write) and read (activation read)
            is_read = self._is_read_next
            self._is_read_next = not self._is_read_next

            request = HBMRequest(
                addr=addr,
                length=self.feature_size,
                is_read=is_read,
                qos=8,  # Normal QoS for feature maps
                burst_length=config.burst_size,
            )
            requests.append(request)
        return requests


class AIInferencePattern(ABC):
    """Base class for AI inference traffic patterns"""

    @abstractmethod
    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate HBM requests for AI inference"""
        pass


class BurstReadPattern(AIInferencePattern):
    """Short burst read pattern for inference

    Characteristics:
    - Short burst reads
    - High locality
    - Low latency requirement
    """

    def __init__(self, burst_length: int = 8):
        """Initialize burst read pattern

        Args:
            burst_length: Length of each burst
        """
        self.burst_length = burst_length
        self.addr_gen = AddressGenerator()
        self._current_offset = 0

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate short burst read requests"""
        requests = []
        for _ in range(count):
            # Inference: access nearby addresses (high locality)
            base = self.addr_gen.base_address + self._current_offset
            self._current_offset = (self._current_offset + 64) % (self.burst_length * 64)

            request = HBMRequest(
                addr=base,
                length=64,
                is_read=True,
                qos=15,  # Critical QoS for inference latency
                burst_length=self.burst_length,
            )
            requests.append(request)
        return requests


class WeightReusePattern(AIInferencePattern):
    """Weight reuse pattern for inference

    Characteristics:
    - High locality (same weights accessed multiple times)
    - Read-only (weights are not modified)
    - Large sequential reads
    """

    def __init__(self, weight_buffer_size: int = 16 * 1024 * 1024):
        """Initialize weight reuse pattern

        Args:
            weight_buffer_size: Size of active weight buffer
        """
        self.weight_buffer_size = weight_buffer_size
        self.addr_gen = AddressGenerator()
        self._reuse_count = 0
        self._reuse_threshold = 16  # Reuse same address 16 times

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate weight reuse requests"""
        requests = []
        current_addr = self.addr_gen.base_address

        for _ in range(count):
            # Reuse same address multiple times before moving
            if self._reuse_count >= self._reuse_threshold:
                current_addr = (current_addr + self.weight_buffer_size) % config.address_range
                self._reuse_count = 0

            request = HBMRequest(
                addr=current_addr,
                length=64,
                is_read=True,  # Weights are read-only
                qos=12,  # High QoS for weight access
                burst_length=config.burst_size,
            )
            requests.append(request)
            self._reuse_count += 1

        return requests


class MixedPrecisionPattern(AIInferencePattern):
    """Mixed precision inference pattern

    Characteristics:
    - Variable request sizes based on precision
    - Read-only (inference)
    - High bandwidth with FP16/BF16, higher with INT8
    """

    def __init__(self):
        """Initialize mixed precision pattern"""
        self.addr_gen = AddressGenerator()
        self.precision_map = {
            DataPrecision.FP32: 128,  # 128 bytes per request
            DataPrecision.FP16: 64,
            DataPrecision.BF16: 64,
            DataPrecision.INT8: 32,
            DataPrecision.INT4: 16,
        }

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate mixed precision requests"""
        requests = []
        request_size = self.precision_map.get(config.precision, 64)

        for _ in range(count):
            addr = self.addr_gen.sequential(1)[0]
            request = HBMRequest(
                addr=addr,
                length=request_size,
                is_read=True,
                qos=12,
                burst_length=config.burst_size,
            )
            requests.append(request)
        return requests


class SyntheticPattern(ABC):
    """Base class for synthetic traffic patterns"""

    @abstractmethod
    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate HBM requests"""
        pass


class FixedRatePattern(SyntheticPattern):
    """Fixed rate traffic pattern

    Generates constant rate traffic for baseline testing.
    """

    def __init__(self):
        """Initialize fixed rate pattern"""
        self.addr_gen = AddressGenerator()
        self._read_count = 0

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate fixed rate requests"""
        requests = []

        for i in range(count):
            addr = self.addr_gen.sequential(1)[0]
            is_read = (self._read_count % 100) < (config.read_write_ratio * 100)
            self._read_count += 1

            qos = _sample_qos(config.qos_distribution)

            request = HBMRequest(
                addr=addr,
                length=64,
                is_read=is_read,
                qos=qos,
                burst_length=config.burst_size,
            )
            requests.append(request)

        return requests


class BurstPattern(SyntheticPattern):
    """Burst traffic pattern

    Generates traffic in bursts with idle periods.
    """

    def __init__(self, burst_requests: int = 32, idle_ratio: float = 0.3):
        """Initialize burst pattern

        Args:
            burst_requests: Number of requests per burst
            idle_ratio: Ratio of idle time (0.0-1.0)
        """
        self.burst_requests = burst_requests
        self.idle_ratio = idle_ratio
        self.addr_gen = AddressGenerator()
        self._request_in_burst = 0
        self._is_burst_active = True
        # Calculate how many loop iterations needed to generate count requests
        # active_ratio = burst_requests / (burst_requests + idle_period)
        # With idle_ratio = 0.3, active_ratio = 0.7
        self._active_ratio = 1.0 - idle_ratio

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate burst requests"""
        requests = []

        # Generate enough loop iterations to get count requests
        # accounting for idle periods
        target_iterations = int(count / self._active_ratio) + 10

        for _ in range(target_iterations):
            if len(requests) >= count:
                break

            if self._request_in_burst >= self.burst_requests:
                # Start idle period - skip some iterations
                self._is_burst_active = False
                self._request_in_burst = 0
            elif not self._is_burst_active:
                # End idle period
                self._is_burst_active = True

            if self._is_burst_active:
                addr = self.addr_gen.sequential(1)[0]
                request = HBMRequest(
                    addr=addr,
                    length=64,
                    is_read=random.random() < config.read_write_ratio,
                    qos=_sample_qos(config.qos_distribution),
                    burst_length=config.burst_size,
                )
                requests.append(request)
                self._request_in_burst += 1

        return requests


class RandomPattern(SyntheticPattern):
    """Random traffic pattern

    Generates completely random address access.
    """

    def __init__(self):
        """Initialize random pattern"""
        self.addr_gen = AddressGenerator()

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate random requests"""
        requests = []

        for _ in range(count):
            addr = self.addr_gen.random(1)[0]
            request = HBMRequest(
                addr=addr,
                length=64,
                is_read=random.random() < config.read_write_ratio,
                qos=_sample_qos(config.qos_distribution),
                burst_length=config.burst_size,
            )
            requests.append(request)

        return requests


class RampPattern(SyntheticPattern):
    """Ramp traffic pattern (up or down)

    Generates gradually increasing or decreasing traffic.
    """

    def __init__(self, ramp_up: bool = True):
        """Initialize ramp pattern

        Args:
            ramp_up: True for ramp up, False for ramp down
        """
        self.ramp_up = ramp_up
        self.addr_gen = AddressGenerator()
        self._current_rate = 0.1  # Start at 10%
        self._step = 0.05  # 5% step per generation
        self._request_count = 0

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate ramp requests"""
        requests = []

        for _ in range(count):
            # Update rate based on direction
            if self.ramp_up:
                self._current_rate = min(1.0, self._current_rate + self._step)
            else:
                self._current_rate = max(0.1, self._current_rate - self._step)

            # Generate request based on current rate
            if random.random() < self._current_rate:
                addr = self.addr_gen.sequential(1)[0]
                request = HBMRequest(
                    addr=addr,
                    length=64,
                    is_read=random.random() < config.read_write_ratio,
                    qos=_sample_qos(config.qos_distribution),
                    burst_length=config.burst_size,
                )
                requests.append(request)

            self._request_count += 1

        return requests


class SinusoidalPattern(SyntheticPattern):
    """Sinusoidal traffic pattern

    Generates traffic with sinusoidal rate variation.
    """

    def __init__(self, period: int = 1000):
        """Initialize sinusoidal pattern

        Args:
            period: Period of the sine wave in requests
        """
        self.period = period
        self.addr_gen = AddressGenerator()
        self._request_count = 0

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate sinusoidal requests"""
        requests = []

        for _ in range(count):
            # Calculate sinusoidal rate
            phase = 2 * np.pi * (self._request_count % self.period) / self.period
            rate = (np.sin(phase) + 1) / 2  # Normalize to 0-1

            if random.random() < rate:
                addr = self.addr_gen.sequential(1)[0]
                request = HBMRequest(
                    addr=addr,
                    length=64,
                    is_read=random.random() < config.read_write_ratio,
                    qos=_sample_qos(config.qos_distribution),
                    burst_length=config.burst_size,
                )
                requests.append(request)

            self._request_count += 1

        return requests


class TraceReplayPattern(SyntheticPattern):
    """Trace replay pattern

    Replays requests from a recorded trace file.
    """

    def __init__(self, trace_file: Optional[str] = None):
        """Initialize trace replay pattern

        Args:
            trace_file: Path to trace file
        """
        self.trace_file = trace_file
        self._trace: List[HBMRequest] = []
        self._index = 0

    def load_trace(self, trace_file: str):
        """Load trace from file

        Args:
            trace_file: Path to trace file
        """
        # Placeholder for trace loading
        # In real implementation, this would parse CSV/JSON trace files
        self.trace_file = trace_file
        self._trace = []
        self._index = 0

    def set_trace(self, requests: List[HBMRequest]):
        """Set trace from in-memory list

        Args:
            requests: List of requests to replay
        """
        self._trace = requests
        self._index = 0

    def generate_requests(self, config: TrafficConfig, count: int) -> List[HBMRequest]:
        """Generate requests from trace"""
        requests = []

        for _ in range(count):
            if self._index >= len(self._trace):
                self._index = 0  # Loop trace

            request = self._trace[self._index]
            self._index += 1
            requests.append(request)

        return requests


def _sample_qos(distribution: Dict[int, float]) -> int:
    """Sample a QoS level from distribution

    Args:
        distribution: QoS level -> probability mapping

    Returns:
        Sampled QoS level (0-15)
    """
    r = random.random()
    cumulative = 0.0
    for qos, prob in sorted(distribution.items(), key=lambda x: -x[0]):
        cumulative += prob
        if r < cumulative:
            return qos
    return 0


class TrafficGenerator:
    """HBM4 Traffic Generator

    Layer 0 of the 5-layer HBM system architecture.
    Generates realistic traffic patterns for AI training/inference workloads
    and synthetic patterns for stress testing.

    Features:
    - AI training patterns (weight update, gradient, feature map)
    - AI inference patterns (burst read, weight reuse, mixed precision)
    - Synthetic patterns (fixed rate, burst, random, ramp)
    - Integration with HBM4Controller

    Example:
        >>> config = TrafficConfig(request_rate=1e6)
        >>> tg = TrafficGenerator(config)
        >>> requests = tg.generate(count=100)
        >>> for req in requests:
        ...     controller.submit_request(req)
    """

    def __init__(self, config: Optional[TrafficConfig] = None):
        """Initialize traffic generator

        Args:
            config: Traffic configuration

        Raises:
            ValueError: If config parameters are invalid
        """
        # Validate config before assignment
        if config is not None:
            try:
                config.__post_init__()
            except ValueError:
                raise  # Re-raise validation errors from config

        self.config = config if config else TrafficConfig()
        self.hbm_spec = HBM4Spec()

        # Validate HBM4Spec compatibility
        self._validate_hbm4_config()

        # Initialize address generator using HBM4Spec constants
        self.addr_gen = AddressGenerator(
            base_address=self.config.base_address,
            address_range=self.config.address_range,
            stride=self.hbm_spec.row_size,  # Use HBM4Spec burst length
        )

        # Initialize patterns
        self._patterns: Dict[TrafficPattern, SyntheticPattern] = {
            TrafficPattern.SYNTHETIC_FIXED_RATE: FixedRatePattern(),
            TrafficPattern.SYNTHETIC_BURST: BurstPattern(),
            TrafficPattern.SYNTHETIC_RANDOM: RandomPattern(),
            TrafficPattern.SYNTHETIC_RAMP_UP: RampPattern(ramp_up=True),
            TrafficPattern.SYNTHETIC_RAMP_DOWN: RampPattern(ramp_up=False),
            TrafficPattern.SYNTHETIC_SINUSOIDAL: SinusoidalPattern(),
            TrafficPattern.TRACE_REPLAY: TraceReplayPattern(),
        }

        # AI training patterns
        self._training_patterns: Dict[TrafficPattern, AITrainingPattern] = {
            TrafficPattern.TRAINING_WEIGHT_UPDATE: WeightUpdatePattern(),
            TrafficPattern.TRAINING_GRADIENT: GradientComputationPattern(),
            TrafficPattern.TRAINING_FEATURE_MAP: FeatureMapTransferPattern(),
        }

        # AI inference patterns
        self._inference_patterns: Dict[TrafficPattern, AIInferencePattern] = {
            TrafficPattern.INFERENCE_BURST_READ: BurstReadPattern(),
            TrafficPattern.INFERENCE_WEIGHT_REUSE: WeightReusePattern(),
            TrafficPattern.INFERENCE_MIXED_PRECISION: MixedPrecisionPattern(),
        }

        # Statistics
        self._stats = {
            'total_requests': 0,
            'read_requests': 0,
            'write_requests': 0,
            'requests_by_qos': {i: 0 for i in range(16)},
            'pattern_switches': 0,
        }

        # Current pattern
        self._current_pattern: TrafficPattern = TrafficPattern.SYNTHETIC_FIXED_RATE
        self._last_pattern = self._current_pattern

        # Thread safety
        self._lock = threading.Lock()

        # Error logging
        self._error_log: List[Dict[str, Any]] = []

    def _validate_hbm4_config(self):
        """Validate configuration against HBM4 specification

        Raises:
            ValueError: If configuration violates HBM4 spec
        """
        # Check channel count
        if self.config.channels > self.hbm_spec.channels:
            raise ValueError(
                f"Configured channels ({self.config.channels}) exceeds "
                f"HBM4 spec maximum ({self.hbm_spec.channels})"
            )

        # Check pseudo-channel count
        if self.config.pseudo_channels > self.hbm_spec.pseudo_channels:
            raise ValueError(
                f"Configured pseudo_channels ({self.config.pseudo_channels}) exceeds "
                f"HBM4 spec maximum ({self.hbm_spec.pseudo_channels})"
            )

        # Check banks per channel
        if self.config.banks_per_channel > self.hbm_spec.banks_per_pseudo_channel:
            raise ValueError(
                f"Configured banks_per_channel ({self.config.banks_per_channel}) exceeds "
                f"HBM4 spec maximum ({self.hbm_spec.banks_per_pseudo_channel})"
            )

    def set_pattern(self, pattern: TrafficPattern):
        """Set traffic pattern

        Args:
            pattern: Traffic pattern to use
        """
        with self._lock:
            if pattern != self._current_pattern:
                self._last_pattern = self._current_pattern
                self._current_pattern = pattern
                self._stats['pattern_switches'] += 1

    def generate(self, count: int = 1, pattern: Optional[TrafficPattern] = None) -> List[HBMRequest]:
        """Generate traffic requests

        Args:
            count: Number of requests to generate (must be positive)
            pattern: Optional pattern override

        Returns:
            List of HBMRequest objects

        Raises:
            ValueError: If count is not positive
        """
        if count <= 0:
            error_msg = f"count must be positive, got {count}"
            self._log_error("invalid_parameter", error_msg)
            raise ValueError(error_msg)

        with self._lock:
            p = pattern if pattern else self._current_pattern

            try:
                # AI training patterns
                if p in self._training_patterns:
                    requests = self._training_patterns[p].generate_requests(self.config, count)
                # AI inference patterns
                elif p in self._inference_patterns:
                    requests = self._inference_patterns[p].generate_requests(self.config, count)
                # Synthetic patterns
                elif p in self._patterns:
                    requests = self._patterns[p].generate_requests(self.config, count)
                else:
                    # Default to fixed rate
                    requests = self._patterns[TrafficPattern.SYNTHETIC_FIXED_RATE].generate_requests(self.config, count)

                # Update statistics
                self._update_stats(requests)

                return requests

            except Exception as e:
                # Log error instead of silently swallowing
                error_msg = f"Error generating requests for pattern {p.name}: {str(e)}"
                self._log_error("generation_error", error_msg)
                # Re-raise the exception so caller knows about the error
                raise

    def _log_error(self, error_type: str, message: str) -> None:
        """Log an error for diagnostics

        Args:
            error_type: Category of error
            message: Error message
        """
        import logging
        logger = logging.getLogger('hbm4.traffic_generator')
        logger.error(message)

        self._error_log.append({
            'type': error_type,
            'message': message,
            'timestamp': time.time(),
        })

    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all recorded errors

        Returns:
            List of error records
        """
        return list(self._error_log)

    def clear_errors(self) -> None:
        """Clear error log"""
        self._error_log = []

    def generate_stream(self, pattern: TrafficPattern = TrafficPattern.SYNTHETIC_FIXED_RATE,
                       batch_size: int = 32) -> Iterator[List[HBMRequest]]:
        """Generate traffic as a stream

        Args:
            pattern: Traffic pattern
            batch_size: Number of requests per batch

        Yields:
            Batch of requests
        """
        while True:
            requests = self.generate(count=batch_size, pattern=pattern)
            yield requests

    def _update_stats(self, requests: List[HBMRequest]):
        """Update traffic statistics

        Args:
            requests: Generated requests
        """
        self._stats['total_requests'] += len(requests)

        for req in requests:
            if req.is_read:
                self._stats['read_requests'] += 1
            else:
                self._stats['write_requests'] += 1

            self._stats['requests_by_qos'][req.qos] = self._stats['requests_by_qos'].get(req.qos, 0) + 1

    def get_stats(self) -> Dict:
        """Get traffic statistics

        Returns:
            Statistics dictionary
        """
        with self._lock:
            stats = {**self._stats}

            # Add computed metrics
            if stats['total_requests'] > 0:
                stats['read_ratio'] = stats['read_requests'] / stats['total_requests']
                stats['write_ratio'] = stats['write_requests'] / stats['total_requests']
            else:
                stats['read_ratio'] = 0.0
                stats['write_ratio'] = 0.0

            return stats

    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self._stats = {
                'total_requests': 0,
                'read_requests': 0,
                'write_requests': 0,
                'requests_by_qos': {i: 0 for i in range(16)},
                'pattern_switches': 0,
            }

    def reset(self):
        """Reset generator state"""
        with self._lock:
            self.addr_gen.reset()
            self._stats = {
                'total_requests': 0,
                'read_requests': 0,
                'write_requests': 0,
                'requests_by_qos': {i: 0 for i in range(16)},
                'pattern_switches': 0,
            }
            self._last_pattern = self._current_pattern


class TrafficGeneratorRunner:
    """Traffic Generator Runner

    Runs traffic generator in a separate thread with rate control.
    Connects to HBM4Controller for request submission.
    """

    def __init__(self, generator: TrafficGenerator, controller: Optional[Callable] = None):
        """Initialize traffic generator runner

        Args:
            generator: TrafficGenerator instance
            controller: Controller submit function
        """
        self.generator = generator
        self.controller = controller

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # Rate control
        self._target_rate = 1e6  # requests/second
        self._actual_rate = 0.0
        self._requests_generated = 0
        self._start_time = 0.0

    def set_target_rate(self, rate: float):
        """Set target request rate

        Args:
            rate: Target rate in requests/second
        """
        self._target_rate = rate

    def start(self, pattern: TrafficPattern = TrafficPattern.SYNTHETIC_FIXED_RATE,
              batch_size: int = 32):
        """Start traffic generation

        Args:
            pattern: Traffic pattern
            batch_size: Batch size
        """
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._pause_event.clear()
        self._start_time = time.time()

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(pattern, batch_size),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop traffic generation"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def pause(self):
        """Pause traffic generation"""
        self._pause_event.set()

    def resume(self):
        """Resume traffic generation"""
        self._pause_event.clear()

    def _run_loop(self, pattern: TrafficPattern, batch_size: int):
        """Run traffic generation loop

        Args:
            pattern: Traffic pattern
            batch_size: Batch size
        """
        interval = batch_size / self._target_rate if self._target_rate > 0 else 0

        while self._running:
            if self._stop_event.is_set():
                break

            if self._pause_event.is_set():
                time.sleep(0.001)
                continue

            # Generate batch
            requests = self.generator.generate(count=batch_size, pattern=pattern)

            # Submit to controller
            if self.controller:
                for req in requests:
                    try:
                        self.controller(req)
                    except Exception:
                        pass  # Handle queue full

            self._requests_generated += len(requests)

            # Rate limiting
            if interval > 0:
                time.sleep(interval)

    def get_rate(self) -> float:
        """Get actual generation rate

        Returns:
            Actual rate in requests/second
        """
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            return self._requests_generated / elapsed
        return 0.0


class AddressPatternGenerator:
    """Address Pattern Generator

    Generates addresses according to specific patterns.
    Used for custom address sequences.
    """

    def __init__(self, config: TrafficConfig):
        """Initialize address pattern generator

        Args:
            config: Traffic configuration
        """
        self.config = config
        self.addr_gen = AddressGenerator(
            base_address=config.base_address,
            address_range=config.address_range,
            stride=config.address_stride,
        )

        # Pattern-specific state
        self._pattern_type: str = "sequential"
        self._custom_sequence: List[int] = []
        self._sequence_index: int = 0

    def set_pattern(self, pattern_type: str, **kwargs):
        """Set address pattern

        Args:
            pattern_type: Pattern type (sequential, random, stride, custom)
            **kwargs: Pattern-specific parameters
        """
        self._pattern_type = pattern_type

        if pattern_type == "custom" and "addresses" in kwargs:
            self._custom_sequence = kwargs["addresses"]
            self._sequence_index = 0

    def next(self) -> int:
        """Get next address

        Returns:
            Next address
        """
        if self._pattern_type == "sequential":
            return self.addr_gen.sequential(1)[0]
        elif self._pattern_type == "random":
            return self.addr_gen.random(1)[0]
        elif self._pattern_type == "stride":
            stride = self._custom_sequence[0] if self._custom_sequence else self.config.address_stride
            return self.addr_gen.stride_access(1, stride)[0]
        elif self._pattern_type == "custom":
            if self._sequence_index >= len(self._custom_sequence):
                self._sequence_index = 0
            addr = self._custom_sequence[self._sequence_index]
            self._sequence_index += 1
            return addr
        else:
            return self.addr_gen.sequential(1)[0]

    def next_batch(self, count: int) -> List[int]:
        """Get next batch of addresses

        Args:
            count: Number of addresses

        Returns:
            List of addresses
        """
        return [self.next() for _ in range(count)]

    def reset(self):
        """Reset generator state"""
        self.addr_gen.reset()
        self._sequence_index = 0


# Factory function for creating traffic generator
def create_traffic_generator(
    pattern: TrafficPattern = TrafficPattern.SYNTHETIC_FIXED_RATE,
    read_write_ratio: float = 0.7,
    request_rate: float = 1e6,
    **kwargs
) -> TrafficGenerator:
    """Create traffic generator with configuration

    Args:
        pattern: Traffic pattern
        read_write_ratio: Read/write ratio
        request_rate: Request rate
        **kwargs: Additional configuration

    Returns:
        Configured TrafficGenerator
    """
    config = TrafficConfig(
        read_write_ratio=read_write_ratio,
        request_rate=request_rate,
        **kwargs
    )

    generator = TrafficGenerator(config)
    generator.set_pattern(pattern)

    return generator
