"""
Interconnect Module for HBM4 System Modeling

This module provides interconnect models for HBM4 multi-stack systems,
supporting various topologies, routing mechanisms, and arbitration schemes.

Key Features:
- Multiple topology options (Crossbar, Mesh, Binary Tree)
- Flexible routing (Address-based, Load-based, Shortest Path)
- Configurable arbitration (Round-robin, Priority)
- Multi-stack support (1-8 HBM4 stacks)
- Load balancing across stacks
- AXI4 interface bridge with full protocol support
- AXI4 to HBM request conversion
- AXI4 protocol monitoring and compliance checking

Based on:
- JEDEC JESD270-4A HBM4 specification
- ARM AMBA AXI4 Protocol Specification
- Multi-agent research findings (2026-06-15)

Architecture Overview:
    Traffic Generator / Requesters
            |
    +-------+-------+------------+
    |  AXI4 Bridge  |  Interconnect  |  <-- This module
    +-------+-------+------------+
    |  AXI4 Converter  |
    +-------------------+
    |  AXI4 Monitor    |
    +-------+-------+
    |  HBM Controller |
    +-------+-------+
    |  HBM4 Stacks   |
    +---------------+

Topologies:
1. Crossbar: Full connectivity, O(1) routing, best for small scale
2. Mesh: Grid-based, good locality, scalable
3. Binary Tree: Hierarchical, efficient for broadcast, scalable

AXI4 Interface:
- Full AXI4 protocol support (AXI4, AXI4-Lite)
- Out-of-order transaction handling
- Outstanding transaction support
- Burst support (INCR, FIXED, WRAP)
- Transaction ID tracking
- QoS-based prioritization
- Protocol compliance monitoring

Usage Example:
    >>> from model.interconnect import CrossbarInterconnect, RoutingMode, ArbitrationMode
    >>> ic = CrossbarInterconnect(num_ports=32, stack_count=4)
    >>> ic.route_request(addr=0x123456, source_port=0)
    (dest_stack=0, dest_channel=9, latency=2)

    >>> # Using AXI4 bridge
    >>> from model.interconnect.axi4_bridge import AXI4Bridge, AXI4BridgeConfig
    >>> config = AXI4BridgeConfig(enable_out_of_order=True, enable_outstanding=True)
    >>> bridge = AXI4Bridge(config)
    >>> txn_id = bridge.submit_read(addr=0x1000, length=7, qos=8)
"""

from .interconnect import (
    # Enums
    TopologyType,
    RoutingMode,
    ArbitrationMode,
    InterconnectPort,
    InterconnectRequest,
    InterconnectResponse,
    InterconnectStats,

    # Main Classes
    InterconnectBase,
    CrossbarInterconnect,
    MeshInterconnect,
    BinaryTreeInterconnect,
    InterconnectFactory,

    # Utility
    create_interconnect,
)


__all__ = [
    # Enums
    'TopologyType',
    'RoutingMode',
    'ArbitrationMode',
    'InterconnectPort',
    'InterconnectRequest',
    'InterconnectResponse',
    'InterconnectStats',

    # Main Classes
    'InterconnectBase',
    'CrossbarInterconnect',
    'MeshInterconnect',
    'BinaryTreeInterconnect',
    'InterconnectFactory',

    # Utility
    'create_interconnect',

    # AXI4 Bridge (import from submodule)
    'AXI4Bridge',
    'AXI4BridgeConfig',
    'AXI4BurstType',
    'AXI4Response',
    'AXI4Size',
    'AXI4Lock',
    'AXI4Cache',
    'AXI4Prot',
    'AXI4Signals',
    'AXI4ReadTransaction',
    'AXI4WriteTransaction',
    'AXI4TransactionResponse',
    'create_axi4_bridge',
    'create_axi4lite_bridge',

    # AXI4 Converter
    'AddressMapping',
    'AddressMappingMode',
    'AXI4ToHBMConverter',
    'HBMToAXI4Converter',
    'AXI4Converter',
    'ConversionResult',
    'create_hbm_address_mapping',
    'create_axi4_converter',

    # AXI4 Monitor
    'ViolationType',
    'ProtocolViolation',
    'TransactionLogEntry',
    'PerformanceMetrics',
    'AXI4Monitor',
    'create_axi4_monitor',
    'analyze_axi4_log',
]


# Import AXI4 components lazily to avoid import errors
def __getattr__(name):
    """Lazy import for AXI4 components"""
    if name in ('AXI4Bridge', 'AXI4BridgeConfig', 'AXI4BurstType', 'AXI4Response',
                'AXI4Size', 'AXI4Lock', 'AXI4Cache', 'AXI4Prot', 'AXI4Signals',
                'AXI4ReadTransaction', 'AXI4WriteTransaction', 'AXI4TransactionResponse',
                'create_axi4_bridge', 'create_axi4lite_bridge'):
        from . import axi4_bridge as _mod
        return getattr(_mod, name)

    if name in ('AddressMapping', 'AddressMappingMode', 'AXI4ToHBMConverter',
                'HBMToAXI4Converter', 'AXI4Converter', 'ConversionResult',
                'create_hbm_address_mapping', 'create_axi4_converter'):
        from . import axi4_converter as _mod
        return getattr(_mod, name)

    if name in ('ViolationType', 'ProtocolViolation', 'TransactionLogEntry',
                'PerformanceMetrics', 'AXI4Monitor', 'create_axi4_monitor',
                'analyze_axi4_log'):
        from . import axi4_monitor as _mod
        return getattr(_mod, name)

    raise AttributeError(f"module 'model.interconnect' has no attribute '{name}'")