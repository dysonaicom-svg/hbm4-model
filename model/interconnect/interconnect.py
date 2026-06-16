"""
Interconnect Model for HBM4 Multi-Stack Systems

This module provides interconnect models supporting various topologies,
routing mechanisms, and arbitration schemes for HBM4 memory systems.

Key Features:
- Multiple topology options (Crossbar, Mesh, Binary Tree)
- Flexible routing (Address-based, Load-based, Shortest Path First)
- Configurable arbitration (Round-robin, Priority-based)
- Multi-stack support (1-8 HBM4 stacks)
- Load balancing across stacks

Topologies:
1. Crossbar (NF-031): Full N×M crossbar switch
   - Best for: Small-medium scale, low latency
   - Routing complexity: O(1)
   - Scalability: Limited by port count

2. Mesh: 2D grid interconnect
   - Best for: Large scale, good locality
   - Routing complexity: O(N)
   - Scalability: Excellent for large systems

3. Binary Tree: Hierarchical tree structure
   - Best for: Broadcast-heavy workloads
   - Routing complexity: O(log N)
   - Scalability: Excellent for large systems

Routing Mechanisms:
- Address-based: Route based on address bits (channel/stack ID)
- Load-based: Route to least loaded destination
- Shortest Path First: Route to destination with fewest hops

Arbitration Mechanisms:
- Round-robin: Fair access, prevents starvation
- Priority: Higher priority requests first (QoS-aware)

Based on:
- JEDEC JESD270-4A HBM4 specification
- Multi-agent research findings (2026-06-15)

Usage Example:
    >>> # Create a crossbar interconnect
    >>> ic = CrossbarInterconnect(num_ports=32, stack_count=4)
    >>> req = InterconnectRequest(source_port=0, addr=0x123456, size=64)
    >>> resp = ic.route_request(req)
    >>> print(f"Route: port 0 -> stack {resp.dest_stack}, ch {resp.dest_channel}")

    >>> # Create a mesh interconnect for larger systems
    >>> ic = MeshInterconnect(rows=4, cols=8, stack_count=4)
    >>> ic.route_request(InterconnectRequest(source_port=0, addr=0xABC))

    >>> # Create a binary tree for broadcast support
    >>> ic = BinaryTreeInterconnect(num_leaves=32, stack_count=4)
    >>> ic.route_request(InterconnectRequest(source_port=0, addr=0x1000))
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import math

# Configure debug logging
_logger = logging.getLogger('hbm4.interconnect')
_logger.setLevel(logging.WARNING)  # Default level


class TopologyType(IntEnum):
    """Interconnect topology types (NF-031)"""
    CROSSBAR = 1      # Full crossbar switch
    MESH = 2          # 2D mesh grid
    BINARY_TREE = 3   # Hierarchical binary tree


class RoutingMode(IntEnum):
    """Routing mechanism types"""
    ADDRESS_BASED = 1    # Route based on address bits
    LOAD_BALANCED = 2    # Route to least loaded destination
    SHORTEST_PATH = 3    # Route with fewest hops


class ArbitrationMode(IntEnum):
    """Arbitration mechanism types"""
    ROUND_ROBIN = 1      # Fair round-robin arbitration
    PRIORITY = 2         # Priority-based arbitration (QoS-aware)


@dataclass
class InterconnectPort:
    """Represents a port in the interconnect

    Attributes:
        port_id: Unique port identifier (0 to num_ports-1)
        is_input: True if input port, False if output port
        is_active: Port status
        queue_depth: Current number of pending requests
        bandwidth_gbs: Port bandwidth in GB/s
        latency_ns: Port latency in nanoseconds
    """
    port_id: int
    is_input: bool = True
    is_active: bool = True
    queue_depth: int = 0
    bandwidth_gbs: float = 1024.0  # Default 1 TB/s per port
    latency_ns: float = 1.0        # Default 1ns port delay

    def __repr__(self) -> str:
        direction = "IN" if self.is_input else "OUT"
        return f"Port{self.port_id}({direction}, q={self.queue_depth})"


@dataclass
class InterconnectRequest:
    """Request to be routed through the interconnect

    Attributes:
        source_port: Source port ID
        dest_stack: Destination stack ID (0 to stack_count-1)
        dest_channel: Destination channel ID (0 to channels-1)
        addr: 64-bit memory address
        size: Request size in bytes
        is_read: True for read, False for write
        qos: QoS priority level (0-15, higher = higher priority)
        arrival_cycle: Cycle when request arrived
        id: Unique request identifier
    """
    source_port: int
    addr: int = 0
    size: int = 64
    is_read: bool = True
    qos: int = 8                    # Default QoS level
    dest_stack: Optional[int] = None  # Computed by router
    dest_channel: Optional[int] = None  # Computed by router
    arrival_cycle: int = 0
    id: int = field(default=0, init=False)

    # Class variable for ID generation
    _next_id: int = 1

    def __post_init__(self):
        if self.id == 0:
            InterconnectRequest._next_id += 1
            self.id = InterconnectRequest._next_id

    def __repr__(self) -> str:
        return (f"Request(id={self.id}, src={self.source_port}, "
                f"stack={self.dest_stack}, ch={self.dest_channel}, "
                f"qos={self.qos})")


@dataclass
class InterconnectResponse:
    """Response from the interconnect

    Attributes:
        request_id: Associated request ID
        success: Whether routing was successful
        dest_stack: Final destination stack
        dest_channel: Final destination channel
        latency: Routing latency in cycles
        arbitration_wait: Cycles spent waiting for arbitration
        congestion_level: Estimated congestion (0-1)
        error: Error message if routing failed
    """
    request_id: int
    success: bool = True
    dest_stack: int = 0
    dest_channel: int = 0
    latency: int = 1               # Cycles to traverse interconnect
    arbitration_wait: int = 0       # Cycles waiting for arbitration
    congestion_level: float = 0.0  # 0 = no congestion, 1 = saturated
    error: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return (f"Response(id={self.request_id}, "
                    f"stack={self.dest_stack}, ch={self.dest_channel}, "
                    f"lat={self.latency}, wait={self.arbitration_wait})")
        return f"Response(id={self.request_id}, ERROR: {self.error})"


@dataclass
class InterconnectStats:
    """Statistics for interconnect performance monitoring

    Attributes:
        total_requests: Total number of routed requests
        successful_requests: Requests successfully routed
        failed_requests: Requests that failed routing
        total_latency_cycles: Sum of all routing latencies
        arbitration_waits: Sum of all arbitration wait cycles
        max_congestion: Peak congestion level observed
        total_hops: Total hops traversed
        load_distribution: Requests per output port
        blocked_cycles: Cycles spent blocked
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_cycles: int = 0
    arbitration_waits: int = 0
    max_congestion: float = 0.0
    total_hops: int = 0
    load_distribution: Dict[int, int] = field(default_factory=dict)
    blocked_cycles: int = 0

    @property
    def average_latency(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_cycles / self.successful_requests

    @property
    def average_arb_wait(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.arbitration_waits / self.successful_requests

    @property
    def average_hops(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_hops / self.successful_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class InterconnectBase(ABC):
    """Base class for all interconnect implementations

    This abstract base class defines the common interface for all
    interconnect topologies. Subclasses must implement the routing
    logic specific to their topology.

    Attributes:
        num_ports: Number of input/output ports
        stack_count: Number of HBM4 stacks (1-8)
        channels_per_stack: Channels per stack (default 32 for HBM4)
        routing_mode: Current routing mode
        arbitration_mode: Current arbitration mode
        _ports: Dictionary of port states
        _arb_queues: Per-port arbitration queues
        _round_robin_ptr: Round-robin pointer state
    """

    def __init__(
        self,
        num_ports: int,
        stack_count: int = 1,
        channels_per_stack: int = 32,
        routing_mode: RoutingMode = RoutingMode.ADDRESS_BASED,
        arbitration_mode: ArbitrationMode = ArbitrationMode.ROUND_ROBIN,
    ):
        """Initialize interconnect base

        Args:
            num_ports: Number of input/output ports (must be positive)
            stack_count: Number of HBM4 stacks (1-8)
            channels_per_stack: Channels per stack (default 32 for HBM4)
            routing_mode: Routing mechanism to use
            arbitration_mode: Arbitration mechanism to use

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate num_ports
        if num_ports <= 0:
            raise ValueError(f"num_ports must be positive, got {num_ports}")

        # Validate stack_count (HBM4 supports 1-8 stacks)
        if stack_count < 1 or stack_count > 8:
            raise ValueError(f"stack_count must be 1-8 for HBM4, got {stack_count}")

        # Validate channels_per_stack (HBM4 has 32 channels per stack)
        if channels_per_stack <= 0:
            raise ValueError(f"channels_per_stack must be positive, got {channels_per_stack}")

        # Validate routing_mode
        if not isinstance(routing_mode, RoutingMode):
            raise ValueError(f"routing_mode must be RoutingMode enum, got {type(routing_mode)}")

        # Validate arbitration_mode
        if not isinstance(arbitration_mode, ArbitrationMode):
            raise ValueError(f"arbitration_mode must be ArbitrationMode enum, got {type(arbitration_mode)}")

        self.num_ports = num_ports
        self.stack_count = stack_count
        self.channels_per_stack = channels_per_stack
        self.routing_mode = routing_mode
        self.arbitration_mode = arbitration_mode

        # Initialize ports
        self._ports: Dict[int, InterconnectPort] = {}
        for i in range(num_ports):
            self._ports[i] = InterconnectPort(port_id=i, is_input=True)

        # Arbitration state
        self._arb_queues: Dict[int, deque] = {i: deque() for i in range(num_ports)}
        self._round_robin_ptr: Dict[int, int] = {i: 0 for i in range(num_ports)}

        # Output port state (for load balancing)
        self._output_load: Dict[int, int] = {}  # port_id -> pending count
        for i in range(stack_count * channels_per_stack):
            self._output_load[i] = 0

        # Statistics
        self.stats = InterconnectStats()

        # Current simulation cycle
        self._cycle: int = 0

    @abstractmethod
    def route_request(self, request: InterconnectRequest) -> InterconnectResponse:
        """Route a request through the interconnect

        Subclasses must implement topology-specific routing logic.

        Args:
            request: Request to route

        Returns:
            InterconnectResponse with routing result
        """
        pass

    @abstractmethod
    def _compute_destination(self, request: InterconnectRequest) -> Tuple[int, int]:
        """Compute destination stack and channel for a request

        Args:
            request: Request to route

        Returns:
            Tuple of (dest_stack, dest_channel)
        """
        pass

    def _compute_hops(self, source: int, dest: int) -> int:
        """Compute number of hops between source and destination

        Default implementation returns 1. Subclasses should override
        for accurate hop counting based on topology.

        Args:
            source: Source port ID
            dest: Destination port ID

        Returns:
            Number of hops
        """
        return 1

    def _arbitrate(self, port_id: int, request: InterconnectRequest) -> Tuple[bool, int]:
        """Perform arbitration for a request at a port

        Args:
            port_id: Input port ID
            request: Request to arbitrate

        Returns:
            Tuple of (granted, wait_cycles)
        """
        if self.arbitration_mode == ArbitrationMode.ROUND_ROBIN:
            return self._round_robin_arbitrate(port_id, request)
        elif self.arbitration_mode == ArbitrationMode.PRIORITY:
            return self._priority_arbitrate(port_id, request)
        else:
            return True, 0

    def _round_robin_arbitrate(self, port_id: int, request: InterconnectRequest) -> Tuple[bool, int]:
        """Round-robin arbitration

        Ensures fair access by rotating through requesters.
        Each requester gets a time slice before moving to the next.

        Args:
            port_id: Input port ID
            request: Request to arbitrate

        Returns:
            Tuple of (granted, wait_cycles)
        """
        queue = self._arb_queues[port_id]

        # Check if this request is next in line
        if queue and queue[0].id == request.id:
            # Request is at head of queue, grant immediately
            wait = 0
        else:
            # Add to queue and calculate wait
            queue.append(request)
            wait = len(queue) - 1

        # Simulate processing
        if wait == 0:
            queue.popleft()
            self._round_robin_ptr[port_id] = (port_id + 1) % self.num_ports

        return wait == 0, wait

    def _priority_arbitrate(self, port_id: int, request: InterconnectRequest) -> Tuple[bool, int]:
        """Priority-based arbitration

        Higher priority requests are served first (QoS-aware).
        Priority is determined by request.qos field.

        Args:
            port_id: Input port ID
            request: Request to arbitrate

        Returns:
            Tuple of (granted, wait_cycles)
        """
        queue = self._arb_queues[port_id]
        request_priority = request.qos

        if not queue:
            # Queue is empty, immediate grant
            return True, 0

        # Check head of queue
        head_request = queue[0]
        head_priority = head_request.qos

        if request_priority >= head_priority:
            # Insert at front (higher or equal priority)
            queue.appendleft(request)
            wait = 0
        else:
            # Insert at appropriate position
            inserted = False
            for i, queued_req in enumerate(queue):
                if request_priority > queued_req.qos:
                    queue.insert(i, request)
                    inserted = True
                    break
            if not inserted:
                queue.append(request)
            wait = len(queue) - 1

        return wait == 0, wait

    def tick(self) -> None:
        """Advance simulation by one cycle

        Should be called once per simulation cycle to update internal state.
        """
        self._cycle += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get interconnect statistics

        Returns:
            Dictionary of statistics
        """
        return {
            'total_requests': self.stats.total_requests,
            'successful_requests': self.stats.successful_requests,
            'failed_requests': self.stats.failed_requests,
            'average_latency': self.stats.average_latency,
            'average_arb_wait': self.stats.average_arb_wait,
            'max_congestion': self.stats.max_congestion,
            'average_hops': self.stats.average_hops,
            'load_distribution': self.stats.load_distribution.copy(),
            'success_rate': self.stats.success_rate,
        }

    def reset(self) -> None:
        """Reset interconnect state"""
        for queue in self._arb_queues.values():
            queue.clear()
        self._round_robin_ptr = {i: 0 for i in range(self.num_ports)}
        self._output_load = {k: 0 for k in self._output_load}
        self._cycle = 0
        self.stats = InterconnectStats()


class CrossbarInterconnect(InterconnectBase):
    """Full N×M crossbar switch interconnect (NF-031)

    A crossbar provides complete connectivity between all input and output
    ports. Each input can connect to any output simultaneously, as long as
    no two inputs connect to the same output.

    Architecture:
        Input Ports (N) -----+-----> Output Ports (M)
                           X      (one per stack/channel)
                           |
                           +----->

    Properties:
    - O(1) routing decision (just select output port)
    - Best latency for small-medium scale
    - Limited scalability (N×M switches)
    - Good for: Low-latency workloads, small systems

    Example:
        32 inputs × 128 outputs (32 channels × 4 stacks)

    Attributes:
        _crossbar_state: Crossbar switch matrix state
    """

    def __init__(
        self,
        num_ports: int,
        stack_count: int = 1,
        channels_per_stack: int = 32,
        routing_mode: RoutingMode = RoutingMode.ADDRESS_BASED,
        arbitration_mode: ArbitrationMode = ArbitrationMode.ROUND_ROBIN,
    ):
        """Initialize crossbar interconnect

        Args:
            num_ports: Number of input ports
            stack_count: Number of HBM4 stacks (1-8)
            channels_per_stack: Channels per stack
            routing_mode: Routing mechanism
            arbitration_mode: Arbitration mechanism
        """
        super().__init__(
            num_ports=num_ports,
            stack_count=stack_count,
            channels_per_stack=channels_per_stack,
            routing_mode=routing_mode,
            arbitration_mode=arbitration_mode,
        )

        # Crossbar state: output_port -> list of input ports competing
        self._crossbar_state: Dict[int, List[int]] = {
            i: [] for i in range(stack_count * channels_per_stack)
        }

    def route_request(self, request: InterconnectRequest) -> InterconnectResponse:
        """Route request through crossbar

        Crossbar routing is O(1):
        1. Compute destination from address
        2. Check for contention
        3. Grant/deny based on arbitration

        Args:
            request: Request to route

        Returns:
            InterconnectResponse with routing result
        """
        self.stats.total_requests += 1
        request.arrival_cycle = self._cycle

        # Compute destination
        dest_stack, dest_channel = self._compute_destination(request)
        dest_port = dest_stack * self.channels_per_stack + dest_channel

        # Perform arbitration
        granted, wait = self._arbitrate(request.source_port, request)
        request.arbitration_wait = wait
        self.stats.arbitration_waits += wait

        if not granted:
            return InterconnectResponse(
                request_id=request.id,
                success=False,
                error=" Arbitration blocked",
                arbitration_wait=wait,
            )

        # Update crossbar state
        self._crossbar_state[dest_port].append(request.source_port)
        self._output_load[dest_port] += 1

        # Calculate latency (crossbar is O(1))
        latency = 1  # One cycle to cross

        # Calculate congestion
        congestion = len(self._crossbar_state[dest_port]) / self.num_ports
        self.stats.max_congestion = max(self.stats.max_congestion, congestion)

        # Update statistics
        self.stats.successful_requests += 1
        self.stats.total_latency_cycles += latency
        self.stats.total_hops += 1
        self.stats.load_distribution[dest_port] = \
            self.stats.load_distribution.get(dest_port, 0) + 1

        # Release crossbar connection
        if request.source_port in self._crossbar_state[dest_port]:
            self._crossbar_state[dest_port].remove(request.source_port)
        self._output_load[dest_port] = max(0, self._output_load[dest_port] - 1)

        _logger.debug(
            f"CROSSBAR: req {request.id} port {request.source_port} -> "
            f"stack {dest_stack} ch {dest_channel} (lat={latency})"
        )

        return InterconnectResponse(
            request_id=request.id,
            success=True,
            dest_stack=dest_stack,
            dest_channel=dest_channel,
            latency=latency,
            arbitration_wait=wait,
            congestion_level=congestion,
        )

    def _compute_destination(self, request: InterconnectRequest) -> Tuple[int, int]:
        """Compute destination for crossbar routing

        Crossbar uses address-based routing to select output port.

        Args:
            request: Request to route

        Returns:
            Tuple of (dest_stack, dest_channel)
        """
        addr = request.addr

        if self.routing_mode == RoutingMode.ADDRESS_BASED:
            # Extract stack and channel from address
            # HBM4 RBC mapping: Stack in bits [47:46], Channel in bits [45:41]
            stack_id = (addr >> 46) & 0x3
            channel_id = (addr >> 41) & 0x1F
        elif self.routing_mode == RoutingMode.LOAD_BALANCED:
            # Route to least loaded output
            channel_id = (addr >> 41) & 0x1F  # Start with address-based channel
            # Find least loaded stack
            min_load = float('inf')
            best_stack = 0
            for s in range(self.stack_count):
                port = s * self.channels_per_stack + channel_id
                if self._output_load.get(port, 0) < min_load:
                    min_load = self._output_load.get(port, 0)
                    best_stack = s
            stack_id = best_stack
        else:  # SHORTEST_PATH
            # Crossbar is always 1 hop, same as address-based
            stack_id = (addr >> 46) & 0x3
            channel_id = (addr >> 41) & 0x1F

        # Validate bounds
        stack_id = stack_id % self.stack_count
        channel_id = channel_id % self.channels_per_stack

        return stack_id, channel_id


class MeshInterconnect(InterconnectBase):
    """2D Mesh interconnect

    A mesh interconnect organizes ports in a 2D grid topology. Each port
    connects to its immediate neighbors in the grid.

    Architecture:
        +---+---+---+---+
        | 0 | 1 | 2 | 3 |
        +---+---+---+---+
        | 4 | 5 | 6 | 7 |
        +---+---+---+---+
        | 8 | 9 |10 |11 |
        +---+---+---+---+
        |12 |13 |14 |15 |
        +---+---+---+---+

    Properties:
    - O(sqrt(N)) routing complexity
    - Good locality for adjacent ports
    - Excellent scalability
    - Good for: Large systems, workloads with locality

    Example:
        4×8 mesh = 32 input ports, connected to 32 HBM channels

    Attributes:
        rows: Number of rows in mesh
        cols: Number of columns in mesh
        _mesh_state: Per-node state
        _routing_table: Precomputed routing paths
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        stack_count: int = 1,
        channels_per_stack: int = 32,
        routing_mode: RoutingMode = RoutingMode.SHORTEST_PATH,
        arbitration_mode: ArbitrationMode = ArbitrationMode.ROUND_ROBIN,
    ):
        """Initialize mesh interconnect

        Args:
            rows: Number of rows in mesh
            cols: Number of columns in mesh
            stack_count: Number of HBM4 stacks
            channels_per_stack: Channels per stack
            routing_mode: Routing mechanism (SHORTEST_PATH recommended)
            arbitration_mode: Arbitration mechanism
        """
        num_ports = rows * cols
        super().__init__(
            num_ports=num_ports,
            stack_count=stack_count,
            channels_per_stack=channels_per_stack,
            routing_mode=routing_mode,
            arbitration_mode=arbitration_mode,
        )

        self.rows = rows
        self.cols = cols

        # Mesh state per node
        self._mesh_state: Dict[int, Dict[str, Any]] = {}
        for i in range(num_ports):
            r, c = divmod(i, cols)
            self._mesh_state[i] = {
                'row': r,
                'col': c,
                'connections': self._get_neighbors(i),
            }

        # Precompute routing table (XY routing)
        self._routing_table: Dict[Tuple[int, int], List[int]] = {}
        self._build_routing_table()

    def _get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbor node IDs for a mesh node

        Args:
            node_id: Node ID

        Returns:
            List of neighbor node IDs
        """
        r, c = divmod(node_id, self.cols)
        neighbors = []

        # Up
        if r > 0:
            neighbors.append((r - 1) * self.cols + c)
        # Down
        if r < self.rows - 1:
            neighbors.append((r + 1) * self.cols + c)
        # Left
        if c > 0:
            neighbors.append(r * self.cols + (c - 1))
        # Right
        if c < self.cols - 1:
            neighbors.append(r * self.cols + (c + 1))

        return neighbors

    def _build_routing_table(self) -> None:
        """Build routing table for XY routing

        XY routing: go horizontally first, then vertically.
        This guarantees no deadlocks in 2D mesh.
        """
        for src in range(self.num_ports):
            for dst in range(self.num_ports):
                if src == dst:
                    path = [src]
                else:
                    path = self._xy_route(src, dst)
                self._routing_table[(src, dst)] = path

    def _xy_route(self, src: int, dst: int) -> List[int]:
        """XY routing between two nodes

        XY routing policy:
        1. Move horizontally toward destination column
        2. Then move vertically toward destination row

        This deterministic routing prevents deadlocks.

        Args:
            src: Source node ID
            dst: Destination node ID

        Returns:
            List of node IDs in path (including src and dst)
        """
        src_r, src_c = divmod(src, self.cols)
        dst_r, dst_c = divmod(dst, self.cols)

        path = [src]

        # Horizontal first (X dimension)
        while src_c != dst_c:
            if src_c < dst_c:
                src_c += 1
            else:
                src_c -= 1
            path.append(src_r * self.cols + src_c)

        # Then vertical (Y dimension)
        while src_r != dst_r:
            if src_r < dst_r:
                src_r += 1
            else:
                src_r -= 1
            path.append(src_r * self.cols + src_c)

        return path

    def route_request(self, request: InterconnectRequest) -> InterconnectResponse:
        """Route request through mesh

        Mesh routing uses XY routing:
        1. Compute destination from address
        2. Look up routing path
        3. Reserve each node along path
        4. Forward to next hop

        Args:
            request: Request to route

        Returns:
            InterconnectResponse with routing result
        """
        self.stats.total_requests += 1
        request.arrival_cycle = self._cycle

        # Compute destination
        dest_stack, dest_channel = self._compute_destination(request)

        # Map destination to mesh node (output port)
        # For simplicity, map channels to mesh nodes
        dest_node = dest_channel % self.num_ports

        # Perform arbitration
        granted, wait = self._arbitrate(request.source_port, request)
        request.arbitration_wait = wait
        self.stats.arbitration_waits += wait

        if not granted:
            return InterconnectResponse(
                request_id=request.id,
                success=False,
                error="Arbitration blocked",
                arbitration_wait=wait,
            )

        # Get routing path
        path = self._routing_table.get((request.source_port, dest_node), [])
        hops = len(path) - 1 if path else 0

        # Calculate latency (1 cycle per hop)
        latency = hops

        # Calculate congestion
        congestion = hops / (self.rows + self.cols)  # Normalized by max distance
        self.stats.max_congestion = max(self.stats.max_congestion, congestion)

        # Update statistics
        self.stats.successful_requests += 1
        self.stats.total_latency_cycles += latency
        self.stats.total_hops += hops
        self.stats.load_distribution[dest_node] = \
            self.stats.load_distribution.get(dest_node, 0) + 1

        _logger.debug(
            f"MESH: req {request.id} port {request.source_port} -> "
            f"node {dest_node} ({hops} hops, lat={latency})"
        )

        return InterconnectResponse(
            request_id=request.id,
            success=True,
            dest_stack=dest_stack,
            dest_channel=dest_channel,
            latency=latency,
            arbitration_wait=wait,
            congestion_level=congestion,
        )

    def _compute_destination(self, request: InterconnectRequest) -> Tuple[int, int]:
        """Compute destination for mesh routing

        Args:
            request: Request to route

        Returns:
            Tuple of (dest_stack, dest_channel)
        """
        addr = request.addr

        if self.routing_mode == RoutingMode.ADDRESS_BASED:
            stack_id = (addr >> 46) & 0x3
            channel_id = (addr >> 41) & 0x1F
        elif self.routing_mode == RoutingMode.LOAD_BALANCED:
            # Route to least loaded region
            channel_id = (addr >> 41) & 0x1F
            # Find least loaded stack
            min_load = float('inf')
            best_stack = 0
            for s in range(self.stack_count):
                load = sum(
                    self._output_load.get(
                        s * self.channels_per_stack + c, 0
                    ) for c in range(self.channels_per_stack)
                )
                if load < min_load:
                    min_load = load
                    best_stack = s
            stack_id = best_stack
        else:  # SHORTEST_PATH
            stack_id = (addr >> 46) & 0x3
            channel_id = (addr >> 41) & 0x1F

        # Validate bounds
        stack_id = stack_id % self.stack_count
        channel_id = channel_id % self.channels_per_stack

        return stack_id, channel_id

    def _compute_hops(self, source: int, dest: int) -> int:
        """Compute mesh distance (hops) between two nodes

        Uses Manhattan distance for 2D mesh.

        Args:
            source: Source node ID
            dest: Destination node ID

        Returns:
            Number of hops
        """
        src_r, src_c = divmod(source, self.cols)
        dst_r, dst_c = divmod(dest, self.cols)
        return abs(src_r - dst_r) + abs(src_c - dst_c)


class BinaryTreeInterconnect(InterconnectBase):
    r"""Hierarchical Binary Tree interconnect

    A binary tree interconnect organizes ports in a tree hierarchy.
    Each internal node connects to two children, root connects to all leaves.

    Architecture:
                    [Root]
                   /      \
               [L1]        [L2]
              /  \        /  \
            [0]  [1]    [2]  [3]

    Properties:
    - O(log N) routing complexity
    - Excellent for broadcast/multicast
    - Good hierarchical load balancing
    - Good for: Large systems, broadcast-heavy workloads

    Example:
        32 leaves (input ports) + 16 internal nodes + 1 root

    Attributes:
        num_leaves: Number of leaf nodes
        _tree_state: State of each tree node
        _height: Tree height
    """

    def __init__(
        self,
        num_leaves: int,
        stack_count: int = 1,
        channels_per_stack: int = 32,
        routing_mode: RoutingMode = RoutingMode.SHORTEST_PATH,
        arbitration_mode: ArbitrationMode = ArbitrationMode.ROUND_ROBIN,
    ):
        """Initialize binary tree interconnect

        Args:
            num_leaves: Number of leaf nodes (input ports)
            stack_count: Number of HBM4 stacks
            channels_per_stack: Channels per stack
            routing_mode: Routing mechanism
            arbitration_mode: Arbitration mechanism
        """
        # Calculate tree height
        height = math.ceil(math.log2(num_leaves)) + 1
        num_ports = num_leaves

        super().__init__(
            num_ports=num_ports,
            stack_count=stack_count,
            channels_per_stack=channels_per_stack,
            routing_mode=routing_mode,
            arbitration_mode=arbitration_mode,
        )

        self.num_leaves = num_leaves
        self._height = height
        self._internal_nodes = (1 << (height - 1)) - 1

        # Tree state
        self._tree_state: Dict[int, Dict[str, Any]] = {}
        for i in range(self._internal_nodes):
            self._tree_state[i] = {
                'level': int(math.log2(i + 1)) if i > 0 else 0,
                'left': 2 * i + 1 if 2 * i + 1 < self._internal_nodes else None,
                'right': 2 * i + 2 if 2 * i + 2 < self._internal_nodes else None,
                'parent': (i - 1) // 2 if i > 0 else None,
            }

        # Output mapping: leaf index -> (stack, channel)
        self._output_map: Dict[int, Tuple[int, int]] = {}
        for i in range(num_leaves):
            stack = (i // channels_per_stack) % stack_count
            channel = i % channels_per_stack
            self._output_map[i] = (stack, channel)

    def route_request(self, request: InterconnectRequest) -> InterconnectResponse:
        """Route request through binary tree

        Tree routing:
        1. Request enters at leaf node
        2. Travels up to root
        3. Travels down to output leaf
        4. O(log N) hop count

        Args:
            request: Request to route

        Returns:
            InterconnectResponse with routing result
        """
        self.stats.total_requests += 1
        request.arrival_cycle = self._cycle

        # Compute destination
        dest_stack, dest_channel = self._compute_destination(request)

        # Perform arbitration at input
        granted, wait = self._arbitrate(request.source_port, request)
        request.arbitration_wait = wait
        self.stats.arbitration_waits += wait

        if not granted:
            return InterconnectResponse(
                request_id=request.id,
                success=False,
                error="Arbitration blocked",
                arbitration_wait=wait,
            )

        # Tree routing hops = 2 * tree height (up + down)
        hops = 2 * (self._height - 1)  # Approximate
        latency = hops

        # Calculate congestion
        congestion = hops / (2 * self._height)
        self.stats.max_congestion = max(self.stats.max_congestion, congestion)

        # Update statistics
        self.stats.successful_requests += 1
        self.stats.total_latency_cycles += latency
        self.stats.total_hops += hops
        output_idx = dest_channel % self.num_leaves
        self.stats.load_distribution[output_idx] = \
            self.stats.load_distribution.get(output_idx, 0) + 1

        _logger.debug(
            f"TREE: req {request.id} port {request.source_port} -> "
            f"stack {dest_stack} ch {dest_channel} ({hops} hops, lat={latency})"
        )

        return InterconnectResponse(
            request_id=request.id,
            success=True,
            dest_stack=dest_stack,
            dest_channel=dest_channel,
            latency=latency,
            arbitration_wait=wait,
            congestion_level=congestion,
        )

    def _compute_destination(self, request: InterconnectRequest) -> Tuple[int, int]:
        """Compute destination for tree routing

        Args:
            request: Request to route

        Returns:
            Tuple of (dest_stack, dest_channel)
        """
        addr = request.addr

        if self.routing_mode == RoutingMode.ADDRESS_BASED:
            stack_id = (addr >> 46) & 0x3
            channel_id = (addr >> 41) & 0x1F
        elif self.routing_mode == RoutingMode.LOAD_BALANCED:
            channel_id = (addr >> 41) & 0x1F
            # Find least loaded stack
            min_load = float('inf')
            best_stack = 0
            for s in range(self.stack_count):
                load = sum(
                    self._output_load.get(
                        s * self.channels_per_stack + c, 0
                    ) for c in range(self.channels_per_stack)
                )
                if load < min_load:
                    min_load = load
                    best_stack = s
            stack_id = best_stack
        else:  # SHORTEST_PATH
            stack_id = (addr >> 46) & 0x3
            channel_id = (addr >> 41) & 0x1F

        # Validate bounds
        stack_id = stack_id % self.stack_count
        channel_id = channel_id % self.channels_per_stack

        return stack_id, channel_id

    def _compute_hops(self, source: int, dest: int) -> int:
        """Compute tree distance between two nodes

        Args:
            source: Source leaf ID
            dest: Destination leaf ID

        Returns:
            Number of hops (up to root + down to destination)
        """
        # Tree path: source -> root (log N) + root -> dest (log N)
        return 2 * (self._height - 1)

    def broadcast(self, request: InterconnectRequest) -> List[InterconnectResponse]:
        """Broadcast request to all output ports

        Binary tree supports efficient broadcast/multicast.
        Request goes up to root, then floods down to all leaves.

        Args:
            request: Request to broadcast

        Returns:
            List of responses for each output port
        """
        responses = []

        for output_idx in range(self.num_leaves):
            stack, channel = self._output_map[output_idx]
            resp = InterconnectResponse(
                request_id=request.id,
                success=True,
                dest_stack=stack,
                dest_channel=channel,
                latency=self._height,  # Root to leaf
                congestion_level=0.0,
            )
            responses.append(resp)

        return responses


class InterconnectFactory:
    """Factory for creating interconnect instances

    Provides a convenient way to create interconnect instances
    with common configurations.

    Example:
        >>> # Create crossbar for small system
        >>> ic = InterconnectFactory.create_crossbar(32, stack_count=4)

        >>> # Create mesh for large system
        >>> ic = InterconnectFactory.create_mesh(rows=4, cols=8, stack_count=4)

        >>> # Create tree for broadcast workloads
        >>> ic = InterconnectFactory.create_tree(num_leaves=32, stack_count=4)
    """

    @staticmethod
    def create_crossbar(
        num_ports: int,
        stack_count: int = 1,
        channels_per_stack: int = 32,
        routing_mode: RoutingMode = RoutingMode.ADDRESS_BASED,
        arbitration_mode: ArbitrationMode = ArbitrationMode.ROUND_ROBIN,
    ) -> CrossbarInterconnect:
        """Create a crossbar interconnect

        Args:
            num_ports: Number of input ports
            stack_count: Number of HBM4 stacks
            channels_per_stack: Channels per stack
            routing_mode: Routing mechanism
            arbitration_mode: Arbitration mechanism

        Returns:
            CrossbarInterconnect instance
        """
        return CrossbarInterconnect(
            num_ports=num_ports,
            stack_count=stack_count,
            channels_per_stack=channels_per_stack,
            routing_mode=routing_mode,
            arbitration_mode=arbitration_mode,
        )

    @staticmethod
    def create_mesh(
        rows: int,
        cols: int,
        stack_count: int = 1,
        channels_per_stack: int = 32,
        routing_mode: RoutingMode = RoutingMode.SHORTEST_PATH,
        arbitration_mode: ArbitrationMode = ArbitrationMode.ROUND_ROBIN,
    ) -> MeshInterconnect:
        """Create a mesh interconnect

        Args:
            rows: Number of rows
            cols: Number of columns
            stack_count: Number of HBM4 stacks
            channels_per_stack: Channels per stack
            routing_mode: Routing mechanism
            arbitration_mode: Arbitration mechanism

        Returns:
            MeshInterconnect instance
        """
        return MeshInterconnect(
            rows=rows,
            cols=cols,
            stack_count=stack_count,
            channels_per_stack=channels_per_stack,
            routing_mode=routing_mode,
            arbitration_mode=arbitration_mode,
        )

    @staticmethod
    def create_tree(
        num_leaves: int,
        stack_count: int = 1,
        channels_per_stack: int = 32,
        routing_mode: RoutingMode = RoutingMode.SHORTEST_PATH,
        arbitration_mode: ArbitrationMode = ArbitrationMode.ROUND_ROBIN,
    ) -> BinaryTreeInterconnect:
        """Create a binary tree interconnect

        Args:
            num_leaves: Number of leaf nodes
            stack_count: Number of HBM4 stacks
            channels_per_stack: Channels per stack
            routing_mode: Routing mechanism
            arbitration_mode: Arbitration mechanism

        Returns:
            BinaryTreeInterconnect instance
        """
        return BinaryTreeInterconnect(
            num_leaves=num_leaves,
            stack_count=stack_count,
            channels_per_stack=channels_per_stack,
            routing_mode=routing_mode,
            arbitration_mode=arbitration_mode,
        )

    @staticmethod
    def create(
        topology: TopologyType,
        **kwargs,
    ) -> InterconnectBase:
        """Create interconnect by topology type

        Args:
            topology: Topology type to create
            **kwargs: Arguments passed to topology constructor

        Returns:
            InterconnectBase instance of specified type
        """
        if topology == TopologyType.CROSSBAR:
            return InterconnectFactory.create_crossbar(**kwargs)
        elif topology == TopologyType.MESH:
            return InterconnectFactory.create_mesh(**kwargs)
        elif topology == TopologyType.BINARY_TREE:
            return InterconnectFactory.create_tree(**kwargs)
        else:
            raise ValueError(f"Unknown topology: {topology}")


def create_interconnect(
    topology: str = "crossbar",
    num_ports: int = 32,
    stack_count: int = 1,
    channels_per_stack: int = 32,
    routing_mode: str = "address",
    arbitration_mode: str = "round_robin",
    **kwargs,
) -> InterconnectBase:
    """Create interconnect with string parameters

    This is a convenience function for creating interconnects
    using string parameters instead of enums.

    Args:
        topology: Topology type ("crossbar", "mesh", "tree")
        num_ports: Number of input ports
        stack_count: Number of HBM4 stacks
        channels_per_stack: Channels per stack
        routing_mode: Routing mode ("address", "load", "shortest")
        arbitration_mode: Arbitration mode ("rr", "priority")
        **kwargs: Additional arguments

    Returns:
        InterconnectBase instance

    Example:
        >>> ic = create_interconnect(
        ...     topology="crossbar",
        ...     num_ports=32,
        ...     stack_count=4,
        ...     routing_mode="load",
        ... )
    """
    # Parse topology
    topo_map = {
        "crossbar": TopologyType.CROSSBAR,
        "mesh": TopologyType.MESH,
        "tree": TopologyType.BINARY_TREE,
        "binary_tree": TopologyType.BINARY_TREE,
    }
    topo = topo_map.get(topology.lower(), TopologyType.CROSSBAR)

    # Parse routing mode
    routing_map = {
        "address": RoutingMode.ADDRESS_BASED,
        "address_based": RoutingMode.ADDRESS_BASED,
        "load": RoutingMode.LOAD_BALANCED,
        "load_balanced": RoutingMode.LOAD_BALANCED,
        "shortest": RoutingMode.SHORTEST_PATH,
        "shortest_path": RoutingMode.SHORTEST_PATH,
    }
    routing = routing_map.get(routing_mode.lower(), RoutingMode.ADDRESS_BASED)

    # Parse arbitration mode
    arb_map = {
        "rr": ArbitrationMode.ROUND_ROBIN,
        "round_robin": ArbitrationMode.ROUND_ROBIN,
        "priority": ArbitrationMode.PRIORITY,
    }
    arb = arb_map.get(arbitration_mode.lower(), ArbitrationMode.ROUND_ROBIN)

    # Create interconnect
    if topo == TopologyType.CROSSBAR:
        return InterconnectFactory.create_crossbar(
            num_ports=num_ports,
            stack_count=stack_count,
            channels_per_stack=channels_per_stack,
            routing_mode=routing,
            arbitration_mode=arb,
            **kwargs,
        )
    elif topo == TopologyType.MESH:
        rows = kwargs.get('rows', int(math.sqrt(num_ports)))
        cols = kwargs.get('cols', int(math.ceil(num_ports / rows)))
        return InterconnectFactory.create_mesh(
            rows=rows,
            cols=cols,
            stack_count=stack_count,
            channels_per_stack=channels_per_stack,
            routing_mode=routing,
            arbitration_mode=arb,
        )
    else:  # BINARY_TREE
        return InterconnectFactory.create_tree(
            num_leaves=num_ports,
            stack_count=stack_count,
            channels_per_stack=channels_per_stack,
            routing_mode=routing,
            arbitration_mode=arb,
        )