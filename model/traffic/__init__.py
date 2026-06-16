"""
HBM4 Traffic Generator Module

Layer 0 of the 5-layer HBM system architecture.
Provides traffic patterns for AI training, inference, and synthetic testing.
"""

from model.traffic.traffic_generator import (
    # Enums
    TrafficPattern,
    DataPrecision,

    # Configuration
    TrafficConfig,
    AddressGenerator,

    # Traffic Patterns
    AITrainingPattern,
    WeightUpdatePattern,
    GradientComputationPattern,
    FeatureMapTransferPattern,
    AIInferencePattern,
    BurstReadPattern,
    WeightReusePattern,
    MixedPrecisionPattern,
    SyntheticPattern,
    FixedRatePattern,
    BurstPattern,
    RandomPattern,
    RampPattern,
    SinusoidalPattern,
    TraceReplayPattern,

    # Main Classes
    TrafficGenerator,
    TrafficGeneratorRunner,
    AddressPatternGenerator,

    # Factory
    create_traffic_generator,
)

__all__ = [
    # Enums
    'TrafficPattern',
    'DataPrecision',

    # Configuration
    'TrafficConfig',
    'AddressGenerator',

    # Traffic Patterns
    'AITrainingPattern',
    'WeightUpdatePattern',
    'GradientComputationPattern',
    'FeatureMapTransferPattern',
    'AIInferencePattern',
    'BurstReadPattern',
    'WeightReusePattern',
    'MixedPrecisionPattern',
    'SyntheticPattern',
    'FixedRatePattern',
    'BurstPattern',
    'RandomPattern',
    'RampPattern',
    'SinusoidalPattern',
    'TraceReplayPattern',

    # Main Classes
    'TrafficGenerator',
    'TrafficGeneratorRunner',
    'AddressPatternGenerator',

    # Factory
    'create_traffic_generator',
]