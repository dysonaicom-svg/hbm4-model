"""
Multi-Channel HBM3 Support Module

This module provides proper multi-channel support for HBM3 simulation:
- Channel-aware traffic generation
- Channel load balancing
- Queue-aware channel selection for adaptive load balancing
- Per-channel statistics
- Fairness metrics (Jain's fairness index)

Reference: 2026-06-15-hbm-system-model-design.md
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import statistics
import math


@dataclass
class ChannelStats:
    """Per-channel statistics"""
    channel_id: int
    total_requests: int = 0
    read_requests: int = 0
    write_requests: int = 0
    row_hits: int = 0
    row_misses: int = 0
    total_latency_cycles: int = 0
    activations: int = 0
    pending_requests: int = 0  # Track in-flight requests per channel

    @property
    def avg_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_cycles / self.total_requests

    @property
    def hit_rate(self) -> float:
        total = self.row_hits + self.row_misses
        if total == 0:
            return 0.0
        return self.row_hits / total


def calculate_jains_fairness_index(values: List[float]) -> float:
    """Calculate Jain's fairness index

    Jain's fairness index = (sum(x_i))^2 / (n * sum(x_i^2))

    Range: 0 to 1
    - 1.0 = perfect fairness (all values equal)
    - 0.0 = complete unfairness (one channel gets all)

    Args:
        values: List of values (e.g., request counts per channel)

    Returns:
        Fairness index between 0 and 1
    """
    if not values:
        return 1.0

    # Filter out zeros for calculation
    non_zero = [v for v in values if v > 0]
    if not non_zero:
        return 1.0

    n = len(non_zero)
    sum_values = sum(non_zero)
    sum_squares = sum(v * v for v in non_zero)

    if sum_squares == 0:
        return 1.0

    return (sum_values * sum_values) / (n * sum_squares)


def calculate_load_variance(values: List[float]) -> float:
    """Calculate variance of load distribution

    Args:
        values: List of load values per channel

    Returns:
        Variance of load distribution
    """
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)


def calculate_load_std_dev(values: List[float]) -> float:
    """Calculate standard deviation of load distribution

    Args:
        values: List of load values per channel

    Returns:
        Standard deviation of load distribution
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


class QueueAwareChannelSelector:
    """Queue-aware adaptive channel selector

    This selector monitors queue depth per channel and routes new requests
    to the least-loaded channel for better load balancing and fairness.

    Features:
    - Tracks pending request depth per channel
    - Selects least-loaded channel for new requests
    - Supports mixed strategies (address-based with load balancing)
    - Provides real-time load metrics

    Usage:
        selector = QueueAwareChannelSelector(num_channels=16)
        channel = selector.select_channel(addr, pending_depths)
        selector.record_request(channel)
        selector.release_channel(channel)
    """

    def __init__(
        self,
        num_channels: int = 8,
        strategy: str = "queue_aware",
        seed: Optional[int] = None,
        enable_adaptive: bool = True,
    ):
        """Initialize queue-aware channel selector

        Args:
            num_channels: Number of channels
            strategy: Selection strategy
            seed: Random seed for reproducible behavior
            enable_adaptive: Enable adaptive load balancing
        """
        self.num_channels = num_channels
        self.strategy = strategy
        self.enable_adaptive = enable_adaptive
        if seed is not None:
            random.seed(seed)

        # Track pending requests per channel (from queue depths)
        self._pending_load: Dict[int, int] = {i: 0 for i in range(num_channels)}

        # Track completed requests per channel (for fairness calculation)
        self._completed_load: Dict[int, int] = {i: 0 for i in range(num_channels)}

        # Round-robin state
        self._round_robin_index = 0

        # History for fairness calculation
        self._load_history: List[Dict[int, int]] = []

    def select_channel(
        self,
        addr: int,
        pending_depths: Optional[Dict[int, int]] = None,
    ) -> int:
        """Select channel based on queue depth and strategy

        Args:
            addr: Memory address (for address-based strategies)
            pending_depths: Current queue depth per channel (optional)

        Returns:
            Selected channel ID
        """
        # Update pending loads if provided
        if pending_depths is not None:
            self._pending_load.update(pending_depths)

        if self.strategy == "queue_aware":
            return self._select_least_loaded()
        elif self.strategy == "weighted_random":
            return self._select_weighted_random()
        elif self.strategy == "round_robin":
            return self._select_round_robin()
        elif self.strategy == "addr_with_balance":
            return self._select_addr_with_balance(addr)
        else:  # Default to queue-aware
            return self._select_least_loaded()

    def _select_least_loaded(self) -> int:
        """Select the channel with minimum pending load

        Uses pending queue depth to make routing decisions.
        This provides better load balancing than completed request count.

        Returns:
            Channel ID with lowest pending load
        """
        min_load = min(self._pending_load.values())

        # Find all channels with minimum load and pick randomly for fairness
        candidates = [ch for ch, load in self._pending_load.items() if load == min_load]

        # If adaptive balancing is enabled, add small randomization to avoid
        # always picking the same channel when loads are equal
        if self.enable_adaptive and len(candidates) > 1:
            return random.choice(candidates)

        return candidates[0] if candidates else 0

    def _select_weighted_random(self) -> int:
        """Select channel using weighted random based on inverse load

        Channels with lower load have higher probability of selection.

        Returns:
            Selected channel ID
        """
        # Calculate inverse load weights
        weights = []
        channels = []
        max_load = max(self._pending_load.values()) + 1  # Avoid division by zero

        for ch in range(self.num_channels):
            # Weight = max_load - current_load (lower load = higher weight)
            weight = max_load - self._pending_load[ch]
            weights.append(max(1, weight))  # At least weight of 1
            channels.append(ch)

        total_weight = sum(weights)
        if total_weight == 0:
            return 0

        # Weighted random selection
        r = random.random() * total_weight
        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                return channels[i]

        return channels[-1]

    def _select_round_robin(self) -> int:
        """Round-robin channel selection

        Returns:
            Next channel in round-robin order
        """
        ch = self._round_robin_index
        self._round_robin_index = (self._round_robin_index + 1) % self.num_channels
        return ch

    def _select_addr_with_balance(self, addr: int) -> int:
        """Address-based selection with load balancing offset

        Uses address bits for channel selection but applies a small
        offset to route traffic to less-loaded channels when possible.

        Args:
            addr: Memory address

        Returns:
            Selected channel ID
        """
        # Calculate address-based channel
        channel_bits_needed = (self.num_channels - 1).bit_length()
        channel_start_bit = 46 - channel_bits_needed
        base_channel = (addr >> channel_start_bit) & (self.num_channels - 1)

        # If current channel is heavily loaded, try neighbors
        if self._pending_load[base_channel] > 5:
            min_load = min(self._pending_load.values())
            if min_load < self._pending_load[base_channel] - 3:
                # Find a less-loaded channel
                for offset in range(1, self.num_channels):
                    for direction in [-1, 1]:
                        candidate = (base_channel + direction * offset) % self.num_channels
                        if self._pending_load[candidate] <= min_load + 2:
                            return candidate

        return base_channel

    def record_request(self, channel_id: int):
        """Record that a request was queued to a channel

        Args:
            channel_id: Target channel ID
        """
        if channel_id in self._pending_load:
            self._pending_load[channel_id] += 1

    def release_channel(self, channel_id: int):
        """Record that a request completed on a channel

        Args:
            channel_id: Channel that completed
        """
        if channel_id in self._pending_load:
            self._pending_load[channel_id] = max(0, self._pending_load[channel_id] - 1)
        if channel_id in self._completed_load:
            self._completed_load[channel_id] += 1

    def record_completion(self, channel_id: int):
        """Record that a request completed

        Args:
            channel_id: Channel that completed
        """
        if channel_id in self._completed_load:
            self._completed_load[channel_id] += 1

    def get_channel_load(self) -> Dict[int, int]:
        """Get current pending load for all channels

        Returns:
            Dict mapping channel_id to pending request count
        """
        return dict(self._pending_load)

    def get_completed_load(self) -> Dict[int, int]:
        """Get completed request count for all channels

        Returns:
            Dict mapping channel_id to completed request count
        """
        return dict(self._completed_load)

    def get_load_balance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive load balance metrics

        Returns:
            Dict with fairness metrics:
            - jains_fairness_index: 0-1 (1 = perfect fairness)
            - load_std_dev: Standard deviation of pending load
            - load_variance: Variance of pending load
            - min_load: Minimum pending load
            - max_load: Maximum pending load
            - load_spread: max_load - min_load
        """
        loads = list(self._pending_load.values())

        return {
            'jains_fairness_index': calculate_jains_fairness_index(loads),
            'load_std_dev': calculate_load_std_dev(loads),
            'load_variance': calculate_load_variance(loads),
            'min_load': min(loads) if loads else 0,
            'max_load': max(loads) if loads else 0,
            'load_spread': max(loads) - min(loads) if loads else 0,
            'total_pending': sum(loads),
        }

    def get_completed_fairness(self) -> float:
        """Calculate fairness based on completed requests

        Returns:
            Jain's fairness index for completed requests
        """
        completed = list(self._completed_load.values())
        return calculate_jains_fairness_index(completed)

    def record_pending_depth(self, channel_id: int, depth: int):
        """Record actual pending depth for a channel

        This allows external components (e.g., controller) to report
        actual queue depths for more accurate load balancing.

        Args:
            channel_id: Channel ID
            depth: Current pending request count
        """
        if channel_id in self._pending_load:
            self._pending_load[channel_id] = depth

    def update_pending_depths(self, depths: Dict[int, int]):
        """Update pending depths for all channels

        Args:
            depths: Dict mapping channel_id to pending count
        """
        self._pending_load.update(depths)

    def reset(self):
        """Reset channel selector state"""
        self._pending_load = {i: 0 for i in range(self.num_channels)}
        self._completed_load = {i: 0 for i in range(self.num_channels)}
        self._round_robin_index = 0
        self._load_history = []

    def snapshot(self):
        """Take a snapshot of current load state for history tracking"""
        self._load_history.append(dict(self._pending_load))
        # Keep history limited
        if len(self._load_history) > 1000:
            self._load_history = self._load_history[-500:]

    def get_load_history(self) -> List[Dict[int, int]]:
        """Get load history for analysis

        Returns:
            List of load snapshots
        """
        return list(self._load_history)


class AdaptiveLoadBalancer:
    """Adaptive load balancer for HBM multi-channel systems

    This class coordinates between the traffic generator, controller queues,
    and channel selector to achieve optimal load distribution across channels.

    Features:
    - Monitors queue depth per channel in real-time
    - Adjusts channel selection based on actual load
    - Reports comprehensive fairness metrics
    - Supports dynamic rebalancing during simulation

    Usage:
        balancer = AdaptiveLoadBalancer(num_channels=16)
        balancer.set_controller(controller)  # Link to HBMController
        channel = balancer.select_channel(addr)
        balancer.record_completion(channel)
    """

    def __init__(
        self,
        num_channels: int = 16,
        strategy: str = "queue_aware",
        enable_adaptive: bool = True,
        rebalance_threshold: float = 0.2,
        seed: Optional[int] = None,
    ):
        """Initialize adaptive load balancer

        Args:
            num_channels: Total number of channels
            strategy: Load balancing strategy
            enable_adaptive: Enable adaptive rebalancing
            rebalance_threshold: Load difference threshold for rebalancing (0-1)
            seed: Random seed for reproducible behavior
        """
        self.num_channels = num_channels
        self.strategy = strategy
        self.enable_adaptive = enable_adaptive
        self.rebalance_threshold = rebalance_threshold

        # Internal queue-aware selector
        self._queue_selector = QueueAwareChannelSelector(
            num_channels=num_channels,
            strategy=strategy,
            seed=seed,
            enable_adaptive=enable_adaptive,
        )

        # Link to controller for queue monitoring
        self._controller = None
        self._last_queue_update = 0
        self._update_interval = 10  # Update queue depths every N cycles

        # Statistics
        self._total_selections = 0
        self._selection_history: List[int] = []

        if seed is not None:
            random.seed(seed)

    def set_controller(self, controller):
        """Link to HBMController for queue monitoring

        Args:
            controller: HBMController instance
        """
        self._controller = controller

    def _update_queue_depths(self, current_cycle: int):
        """Update queue depths from controller

        Args:
            current_cycle: Current simulation cycle
        """
        if self._controller is None:
            return

        # Only update periodically to reduce overhead
        if current_cycle - self._last_queue_update < self._update_interval:
            return

        self._last_queue_update = current_cycle

        # Get queue depths from controller
        pending_depths = {}

        if hasattr(self._controller, 'queue_manager'):
            qm = self._controller.queue_manager
            if hasattr(qm, 'read_queue'):
                # Get per-channel pending counts
                for ch in range(self.num_channels):
                    pending_depths[ch] = 0

                # Count requests per channel in read queue
                if hasattr(qm, '_queue'):
                    for req in qm._queue._queue:
                        if hasattr(req, 'channel_id') and req.channel_id in pending_depths:
                            pending_depths[req.channel_id] += 1

        # Update the queue selector with current depths
        self._queue_selector.update_pending_depths(pending_depths)

    def select_channel(
        self,
        addr: int,
        current_cycle: int = 0,
    ) -> int:
        """Select channel using adaptive load balancing

        Args:
            addr: Memory address (for address-based fallback)
            current_cycle: Current simulation cycle (for queue updates)

        Returns:
            Selected channel ID
        """
        # Update queue depths periodically
        self._update_queue_depths(current_cycle)

        # Select channel using queue-aware selector
        channel = self._queue_selector.select_channel(addr)

        # Record selection
        self._total_selections += 1
        self._selection_history.append(channel)

        return channel

    def record_request(self, channel_id: int):
        """Record that a request was queued to a channel

        Args:
            channel_id: Target channel ID
        """
        self._queue_selector.record_request(channel_id)

    def record_completion(self, channel_id: int):
        """Record that a request completed on a channel

        Args:
            channel_id: Channel that completed
        """
        self._queue_selector.release_channel(channel_id)

    def get_channel_load(self) -> Dict[int, int]:
        """Get current pending load for all channels

        Returns:
            Dict mapping channel_id to pending request count
        """
        return self._queue_selector.get_channel_load()

    def get_completed_load(self) -> Dict[int, int]:
        """Get completed request count for all channels

        Returns:
            Dict mapping channel_id to completed request count
        """
        return self._queue_selector.get_completed_load()

    def get_fairness_metrics(self) -> Dict[str, float]:
        """Get comprehensive fairness metrics

        Returns:
            Dict with all fairness metrics:
            - jains_fairness_index: 0-1 (1 = perfect fairness)
            - load_std_dev: Standard deviation of pending load
            - load_variance: Variance of pending load
            - load_spread: Difference between max and min load
            - min_load: Minimum pending load
            - max_load: Maximum pending load
            - completed_fairness: Fairness based on completed requests
            - utilization_variance: Variance of channel utilization
            - active_channels: Number of channels with requests
        """
        metrics = self._queue_selector.get_load_balance_metrics()
        metrics['completed_fairness'] = self._queue_selector.get_completed_fairness()

        # Calculate utilization variance
        completed = list(self._queue_selector._completed_load.values())
        total_completed = sum(completed)
        if total_completed > 0:
            utilizations = [c / total_completed for c in completed if c > 0]
            metrics['utilization_variance'] = calculate_load_variance(utilizations)
            metrics['active_channels'] = len([c for c in completed if c > 0])
        else:
            metrics['utilization_variance'] = 0.0
            metrics['active_channels'] = 0

        return metrics

    def get_pending_depths(self) -> Dict[int, int]:
        """Get current pending queue depths

        Returns:
            Dict mapping channel_id to pending depth
        """
        return self._queue_selector.get_channel_load()

    def reset(self):
        """Reset load balancer state"""
        self._queue_selector.reset()
        self._total_selections = 0
        self._selection_history = []
        self._last_queue_update = 0

    def get_selection_distribution(self) -> Dict[int, int]:
        """Get distribution of channel selections

        Returns:
            Dict mapping channel_id to selection count
        """
        dist = {ch: 0 for ch in range(self.num_channels)}
        for ch in self._selection_history:
            dist[ch] = dist.get(ch, 0) + 1
        return dist


class ChannelSelector:
    """Channel selection strategies for multi-channel HBM3

    Supports various channel selection policies:
    - ROUND_ROBIN: Simple round-robin across channels
    - HASH: Hash-based deterministic selection
    - LOAD_BALANCED: Select least-loaded channel
    - ADDR_BASED: Direct address-based channel selection
    - QUEUE_AWARE: Queue-depth aware adaptive selection (uses QueueAwareChannelSelector)
    - ADAPTIVE: Full adaptive load balancing (uses AdaptiveLoadBalancer)
    """

    ROUND_ROBIN = "round_robin"
    HASH = "hash"
    LOAD_BALANCED = "load_balanced"
    ADDR_BASED = "addr_based"
    QUEUE_AWARE = "queue_aware"
    ADAPTIVE = "adaptive"  # Full adaptive load balancing

    def __init__(
        self,
        num_channels: int = 8,
        strategy: str = ADAPTIVE,
        seed: Optional[int] = None
    ):
        """Initialize channel selector

        Args:
            num_channels: Number of channels (default 8 for HBM3)
            strategy: Selection strategy
            seed: Random seed for reproducible behavior
        """
        self.num_channels = num_channels
        self.strategy = strategy
        if seed is not None:
            random.seed(seed)

        # Round-robin state
        self._round_robin_index = 0

        # Load balancing state
        self._channel_load: Dict[int, int] = {i: 0 for i in range(num_channels)}

        # Queue-aware selector (lazy initialization)
        self._queue_aware: Optional[QueueAwareChannelSelector] = None

        # Adaptive load balancer (for full adaptive balancing)
        self._adaptive: Optional[AdaptiveLoadBalancer] = None

    def select_channel(self, addr: int, length: int = 64, current_cycle: int = 0) -> int:
        """Select channel based on address and strategy

        Args:
            addr: Memory address
            length: Request length in bytes
            current_cycle: Current simulation cycle (for adaptive selection)

        Returns:
            Selected channel ID (0-7 for HBM3)
        """
        if self.strategy == self.ROUND_ROBIN:
            return self._round_robin()
        elif self.strategy == self.HASH:
            return self._hash_channel(addr)
        elif self.strategy == self.LOAD_BALANCED:
            return self._least_loaded_channel()
        elif self.strategy == self.QUEUE_AWARE:
            return self._queue_aware_selection(addr)
        elif self.strategy == self.ADAPTIVE:
            return self._adaptive_selection(addr, current_cycle)
        else:  # ADDR_BASED
            return self._addr_based_channel(addr)

    def _round_robin(self) -> int:
        """Round-robin channel selection"""
        ch = self._round_robin_index
        self._round_robin_index = (self._round_robin_index + 1) % self.num_channels
        return ch

    def _hash_channel(self, addr: int) -> int:
        """Hash-based deterministic channel selection

        Uses a simple hash to distribute addresses across channels
        while maintaining some locality for adjacent addresses.
        """
        # XOR-fold the address to get channel
        hash_val = addr ^ (addr >> 8) ^ (addr >> 16)
        return hash_val % self.num_channels

    def _least_loaded_channel(self) -> int:
        """Select the least loaded channel"""
        min_load = min(self._channel_load.values())
        # Find first channel with minimum load
        for ch in range(self.num_channels):
            if self._channel_load[ch] == min_load:
                return ch
        return 0

    def _queue_aware_selection(self, addr: int) -> int:
        """Queue-aware adaptive channel selection

        Uses QueueAwareChannelSelector for intelligent load balancing.

        Args:
            addr: Memory address

        Returns:
            Selected channel ID
        """
        if self._queue_aware is None:
            self._queue_aware = QueueAwareChannelSelector(
                num_channels=self.num_channels,
                seed=None
            )

        return self._queue_aware.select_channel(addr)

    def _adaptive_selection(self, addr: int, current_cycle: int = 0) -> int:
        """Full adaptive load balancing selection

        Uses AdaptiveLoadBalancer for real-time queue-aware selection
        with controller integration for optimal load distribution.

        Args:
            addr: Memory address
            current_cycle: Current simulation cycle

        Returns:
            Selected channel ID
        """
        if self._adaptive is None:
            self._adaptive = AdaptiveLoadBalancer(
                num_channels=self.num_channels,
                seed=None
            )

        return self._adaptive.select_channel(addr, current_cycle)

    def _addr_based_channel(self, addr: int) -> int:
        """Direct address-based channel selection

        For HBM3 with 8 channels (46-bit address space):
        - Addr[45:43] = Channel (3-bit, 8 channels)

        For HBM4 with 32 channels (46-bit address space):
        - Addr[45:41] = Channel (5-bit, 32 channels)

        This matches the JEDEC HBM3/HBM4 address mapping spec.
        Uses 46-bit address space (bits 0-45).
        """
        # Calculate channel bit position based on number of channels
        # HBM3: 8 channels = 3 bits, channel at bits [45:43], LSB at 43
        # HBM4: 32 channels = 5 bits, channel at bits [45:41], LSB at 41
        # Formula: channel_start_bit = 46 - channel_bits_needed
        channel_bits_needed = (self.num_channels - 1).bit_length()  # 3 for 8ch, 5 for 32ch
        channel_start_bit = 46 - channel_bits_needed

        # Extract channel bits
        channel_bits = (addr >> channel_start_bit) & (self.num_channels - 1)
        return int(channel_bits)

    def record_request(self, channel_id: int):
        """Record that a request was sent to a channel

        Args:
            channel_id: Target channel ID
        """
        if channel_id in self._channel_load:
            self._channel_load[channel_id] += 1

        # Also update queue-aware selector if present
        if self._queue_aware is not None:
            self._queue_aware.record_request(channel_id)

        # Also update adaptive balancer if present
        if self._adaptive is not None:
            self._adaptive.record_request(channel_id)

    def release_channel(self, channel_id: int):
        """Record that a request completed on a channel

        Args:
            channel_id: Channel that completed
        """
        if channel_id in self._channel_load:
            self._channel_load[channel_id] = max(0, self._channel_load[channel_id] - 1)

        # Also update queue-aware selector if present
        if self._queue_aware is not None:
            self._queue_aware.release_channel(channel_id)

        # Also update adaptive balancer if present
        if self._adaptive is not None:
            self._adaptive.record_completion(channel_id)

    def set_controller(self, controller):
        """Link to HBMController for queue monitoring

        Args:
            controller: HBMController instance
        """
        if self._adaptive is not None:
            self._adaptive.set_controller(controller)

    def get_channel_load(self) -> Dict[int, int]:
        """Get current load for all channels

        Returns:
            Dict mapping channel_id to load count
        """
        if self._adaptive is not None:
            return self._adaptive.get_channel_load()
        return dict(self._channel_load)

    def get_queue_aware_metrics(self) -> Dict[str, float]:
        """Get queue-aware load balancing metrics

        Returns:
            Dict with fairness metrics from QueueAwareChannelSelector
        """
        if self._queue_aware is None:
            return {
                'jains_fairness_index': 1.0,
                'load_std_dev': 0.0,
                'load_variance': 0.0,
            }
        return self._queue_aware.get_load_balance_metrics()

    def get_adaptive_metrics(self) -> Dict[str, float]:
        """Get adaptive load balancing metrics

        Returns:
            Dict with comprehensive fairness metrics from AdaptiveLoadBalancer
        """
        if self._adaptive is None:
            return {
                'jains_fairness_index': 1.0,
                'completed_fairness': 1.0,
                'load_std_dev': 0.0,
                'load_variance': 0.0,
                'load_spread': 0,
                'active_channels': 0,
            }
        return self._adaptive.get_fairness_metrics()

    def update_pending_depths(self, depths: Dict[int, int]):
        """Update pending depths for queue-aware selection

        Args:
            depths: Dict mapping channel_id to pending count
        """
        if self._queue_aware is not None:
            self._queue_aware.update_pending_depths(depths)

        if self._adaptive is not None:
            self._queue_aware.update_pending_depths(depths)

    def reset(self):
        """Reset channel selector state"""
        self._round_robin_index = 0
        self._channel_load = {i: 0 for i in range(self.num_channels)}
        if self._queue_aware is not None:
            self._queue_aware.reset()
        if self._adaptive is not None:
            self._adaptive.reset()


class MultiChannelTrafficGenerator:
    """Traffic generator with proper multi-channel support

    Generates traffic that properly distributes across HBM3 channels.
    """

    def __init__(
        self,
        config: 'SimulationConfig',  # Forward reference
        num_channels: int = 8,
        channel_selector: Optional[ChannelSelector] = None,
    ):
        """Initialize multi-channel traffic generator

        Args:
            config: Simulation configuration
            num_channels: Number of channels (8 for HBM3)
            channel_selector: Optional custom channel selector
        """
        self.config = config
        self.num_channels = num_channels

        if channel_selector is None:
            channel_selector = ChannelSelector(
                num_channels=num_channels,
                strategy=ChannelSelector.ADAPTIVE
            )
        self.channel_selector = channel_selector

        # Address range per channel
        self._addr_bits_per_channel = (num_channels - 1).bit_length()

        # Use seeded random for reproducibility
        if config.seed is not None:
            self._random = random.Random(config.seed)
        else:
            self._random = random.Random()
        self.current_addr = 0
        self.hot_bank = 0

        # Reference to adaptive balancer (set by simulator)
        self._adaptive_balancer: Optional[AdaptiveLoadBalancer] = None

    def generate(self) -> List['HBMRequest']:
        """Generate requests with proper channel distribution (single request mode)

        Returns:
            List of HBMRequest with properly decoded channel info
        """
        from model.controller.request import HBMRequest

        requests = []

        # According to request rate, decide whether to generate
        if self._random.random() > self.config.request_rate:
            return requests

        req = self._generate_single_request()
        if req:
            requests.append(req)
        return requests

    def generate_burst(self) -> List['HBMRequest']:
        """Generate burst of requests based on request rate

        Generates multiple requests per call to maximize throughput.
        Each request is generated with probability request_rate.

        Returns:
            List of HBMRequest with properly decoded channel info
        """
        from model.controller.request import HBMRequest

        requests = []
        max_requests = getattr(self.config, 'max_requests_per_cycle', 4)
        request_rate = self.config.request_rate
        random_func = self._random.random  # Local reference for speed
        generate_func = self._generate_single_request  # Local reference for speed

        for _ in range(max_requests):
            # Each slot generates a request with probability request_rate
            if random_func() < request_rate:
                req = generate_func()
                if req:
                    requests.append(req)

        return requests

    def _generate_single_request(self) -> Optional['HBMRequest']:
        """Generate a single request with proper channel addressing

        Returns:
            HBMRequest or None if generation fails
        """
        from model.controller.request import HBMRequest

        # Generate address based on pattern
        if self.config.traffic_pattern.value == "random":
            addr = self._random.randint(0, self.config.address_range - 1)
        elif self.config.traffic_pattern.value == "sequential":
            addr = self.current_addr
            self.current_addr = (self.current_addr + self.config.burst_size) % self.config.address_range
        elif self.config.traffic_pattern.value == "stride":
            addr = self.current_addr
            self.current_addr = (self.current_addr + self.config.stride_value) % self.config.address_range
        elif self.config.traffic_pattern.value == "hot_spot":
            if self._random.random() < 0.8:  # 80% access hot spot
                addr = self._random.randint(0, self.config.address_range // 10)
            else:
                addr = self._random.randint(0, self.config.address_range - 1)
        else:  # scatter
            addr = self._random.randint(0, self.config.address_range - 1)

        # Align address
        addr = addr & ~0x3F  # 64-byte alignment

        # Select channel using adaptive balancing if available
        if self._adaptive_balancer is not None:
            channel_id = self._adaptive_balancer.select_channel(addr)
            self._adaptive_balancer.record_request(channel_id)
        else:
            # Fallback to channel selector
            channel_id = self.channel_selector.select_channel(addr, self.config.burst_size)
            self.channel_selector.record_request(channel_id)

        # Generate read or write request
        is_read = self._random.random() < self.config.read_ratio
        req = HBMRequest(addr=addr, length=self.config.burst_size, is_read=is_read)

        # Set channel info from address decoder logic
        req.channel_id = channel_id

        return req

    def get_channel_stats(self) -> Dict[int, ChannelStats]:
        """Get statistics per channel

        Returns:
            Dict mapping channel_id to ChannelStats
        """
        loads = self.channel_selector.get_channel_load()
        stats = {}
        for ch in range(self.num_channels):
            stats[ch] = ChannelStats(
                channel_id=ch,
                total_requests=loads[ch]
            )
        return stats


class MultiChannelStats:
    """Multi-channel statistics aggregator"""

    def __init__(self, num_channels: int = 8):
        """Initialize stats aggregator

        Args:
            num_channels: Number of channels to track
        """
        self.num_channels = num_channels
        self.channel_stats: Dict[int, ChannelStats] = {
            i: ChannelStats(channel_id=i) for i in range(num_channels)
        }

    def record_request(self, channel_id: int, is_read: bool):
        """Record a request on a channel

        Args:
            channel_id: Target channel
            is_read: True for read, False for write
        """
        if channel_id not in self.channel_stats:
            self.channel_stats[channel_id] = ChannelStats(channel_id=channel_id)

        stats = self.channel_stats[channel_id]
        stats.total_requests += 1
        if is_read:
            stats.read_requests += 1
        else:
            stats.write_requests += 1

    def record_completion(self, channel_id: int, latency_cycles: int, is_row_hit: bool):
        """Record a completion on a channel

        Args:
            channel_id: Channel that completed
            latency_cycles: Completion latency
            is_row_hit: Whether it was a row hit
        """
        if channel_id not in self.channel_stats:
            self.channel_stats[channel_id] = ChannelStats(channel_id=channel_id)

        stats = self.channel_stats[channel_id]
        stats.total_latency_cycles += latency_cycles
        if is_row_hit:
            stats.row_hits += 1
        else:
            stats.row_misses += 1

    def record_activation(self, channel_id: int):
        """Record an activation on a channel

        Args:
            channel_id: Channel that activated
        """
        if channel_id not in self.channel_stats:
            self.channel_stats[channel_id] = ChannelStats(channel_id=channel_id)

        self.channel_stats[channel_id].activations += 1

    def get_per_channel_stats(self) -> Dict[int, ChannelStats]:
        """Get statistics for all channels

        Returns:
            Dict mapping channel_id to ChannelStats
        """
        return dict(self.channel_stats)

    def get_load_balance_score(self) -> float:
        """Calculate load balance score (0-1, 1=perfect balance)

        Returns:
            Load balance score
        """
        if not self.channel_stats:
            return 0.0

        requests = [s.total_requests for s in self.channel_stats.values()]
        if sum(requests) == 0:
            return 1.0

        avg = sum(requests) / len(requests)
        if avg == 0:
            return 1.0

        # Calculate coefficient of variation
        variance = sum((x - avg) ** 2 for x in requests) / len(requests)
        cv = (variance ** 0.5) / avg

        # Convert to 0-1 score (lower CV = better balance)
        # CV=0 means perfect balance, CV=1 means high imbalance
        return max(0.0, 1.0 - min(1.0, cv))

    def get_jains_fairness_index(self) -> float:
        """Calculate Jain's fairness index for completed requests

        Jain's fairness index = (sum(x_i))^2 / (n * sum(x_i^2))
        Range: 0 to 1
        - 1.0 = perfect fairness (all values equal)
        - 0.0 = complete unfairness (one channel gets all)

        Returns:
            Fairness index between 0 and 1
        """
        requests = [s.total_requests for s in self.channel_stats.values()]
        return calculate_jains_fairness_index(requests)

    def get_fairness_metrics(self) -> Dict[str, float]:
        """Get comprehensive fairness metrics

        Returns:
            Dict with all fairness metrics
        """
        requests = [s.total_requests for s in self.channel_stats.values()]

        return {
            'jains_fairness_index': calculate_jains_fairness_index(requests),
            'load_balance_score': self.get_load_balance_score(),
            'load_std_dev': calculate_load_std_dev(requests),
            'load_variance': calculate_load_variance(requests),
            'load_spread': max(requests) - min(requests) if requests else 0,
            'min_load': min(requests) if requests else 0,
            'max_load': max(requests) if requests else 0,
        }

    def get_channel_utilization(self) -> Dict[int, float]:
        """Calculate per-channel utilization as fraction of total

        Returns:
            Dict mapping channel_id to utilization (0-1)
        """
        total = sum(s.total_requests for s in self.channel_stats.values())
        if total == 0:
            return {ch: 0.0 for ch in self.channel_stats}

        return {
            ch: s.total_requests / total
            for ch, s in self.channel_stats.items()
        }

    def get_summary(self) -> Dict:
        """Get summary statistics

        Returns:
            Dict with summary stats
        """
        total_requests = sum(s.total_requests for s in self.channel_stats.values())
        total_reads = sum(s.read_requests for s in self.channel_stats.values())
        total_writes = sum(s.write_requests for s in self.channel_stats.values())
        total_activations = sum(s.activations for s in self.channel_stats.values())

        return {
            'total_requests': total_requests,
            'total_reads': total_reads,
            'total_writes': total_writes,
            'total_activations': total_activations,
            'load_balance_score': self.get_load_balance_score(),
            'per_channel': {
                ch: {
                    'requests': stats.total_requests,
                    'hit_rate': stats.hit_rate,
                    'avg_latency': stats.avg_latency,
                }
                for ch, stats in self.channel_stats.items()
            }
        }