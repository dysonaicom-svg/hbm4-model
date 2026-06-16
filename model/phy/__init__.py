"""
HBM PHY Models

This package provides PHY-level models for HBM memory interface.
The actual PHY implementation is in model.dram.phy_training module.

Classes available from model.dram.phy_training:
    - HBM4PHYManager: Main PHY management controller
    - PHYTrainingStateMachine: PHY training state machine
    - PHYInitializationStateMachine: PHY initialization state machine
    - PHYInitState: PHY initialization states
    - TrainingPhase: Training phase enumeration

Signal Integrity Models (in model.phy):
    - ChannelModel: Channel model with frequency-dependent loss
    - TXPreEmphasis: TX pre-emphasis equalizer
    - RXCTLE: RX continuous time linear equalizer
    - DFEEqualizer: Decision feedback equalizer
    - EyeDiagramAnalyzer: Eye diagram analysis and BER estimation

The PHY layer handles:
    - Initialization sequences
    - Training (write leveling, read DQ calibration, etc.)
    - Signal integrity monitoring
    - Lane repair and remapping
    - Channel equalization and eye analysis

Usage:
    from model.dram.phy_training import HBM4PHYManager

    phy = HBM4PHYManager()
    phy.start_training()

Signal Integrity Usage:
    from model.phy.channel_model import ChannelModel, ChannelConfig
    from model.phy.eye_analyzer import EyeDiagramAnalyzer

    config = ChannelConfig(sample_rate=32e9, length_mm=50.0)
    channel = ChannelModel(config)
    analyzer = EyeDiagramAnalyzer()
"""

# PHY models are implemented in model.dram.phy_training
# Import from there for actual functionality
from model.dram.phy_training import (
    HBM4PHYManager,
    PHYTrainingStateMachine,
    PHYInitializationStateMachine,
    PHYInitState,
    TrainingPhase,
)

# Signal integrity models
from model.phy.channel_model import (
    ChannelModel,
    ChannelConfig,
    ChannelCrosstalkModel,
    RLGCParameters
)

from model.phy.phy_training import (
    PHYTrainingState,
    PHYTrainingType,
    TrainingPattern,
    PHYTrainingConfig,
    TrainingPhaseResult,
    PHYTapCoefficients,
    PHYTrainingStatus,
    PHYTrainingModel,
    create_phy_training_model,
)

from model.phy.training_sequences import (
    TrainingSequenceType,
    DFITrainingCommand,
    TrainingSequenceStep,
    TrainingSequenceDefinition,
    DFITrainingControl,
    TrainingCompletionStatus,
    TrainingSequenceExecutor,
    TrainingCompletionDetector,
    QUICK_BOOT_SEQUENCE,
    NORMAL_TRAINING_SEQUENCE,
    EXTENDED_TRAINING_SEQUENCE,
    MARGIN_SCAN_SEQUENCE,
    create_training_sequence,
    get_dfi_training_command,
)

from model.phy.tap_coefficient import (
    CoefficientType,
    TXCoefficients,
    RXCoefficients,
    LaneCoefficients,
    CompleteTapCoefficients,
    CoefficientOptimizer,
    CoefficientComparator,
    create_default_coefficients,
    export_coefficients_to_dict,
    import_coefficients_from_dict,
)

from model.phy.signal_integrity import (
    TXPreEmphasis,
    RXCTLE,
    DFEEqualizer,
    SignalIntegrityModel,
    PreEmphasisConfig,
    CTLEConfig,
    DFEConfig,
    SignalIntegrityConfig,
    EqualizerType
)

from model.phy.eye_analyzer import (
    EyeDiagramAnalyzer,
    EyeMeasurementConfig,
    BathtubCurveGenerator,
    EyeMetrics,
    EyeMeasurementType
)

# IBIS (I/O Buffer Information Specification) models
from model.phy.ibis_parser import (
    IBISParser,
    IBISFile,
    IBISModel,
    IBISModelType,
    IBISPackage,
    IBISPin,
    IVCurve,
    VTWaveform,
    CompositeDataTable,
    parse_ibis_file,
    parse_ibis_content
)

from model.phy.ibis_model import (
    IBISModelWrapper,
    BehavioralModel,
    WaveformMetrics,
    ChannelResponse,
    SignalIntegrityMetric,
    create_model_wrapper,
    create_model_wrapper_from_file
)

from model.phy.ibis_simulator import (
    IBISSimulator,
    ChannelParameters,
    SimulationConfig,
    SimulationMode,
    SignalDistortion,
    CrosstalkResult,
    EyeAnalysisResult,
    SimulationResult,
    create_simulator
)

__all__ = [
    # PHY training models
    'HBM4PHYManager',
    'PHYTrainingStateMachine',
    'PHYInitializationStateMachine',
    'PHYInitState',
    'TrainingPhase',
    # Channel models
    'ChannelModel',
    'ChannelConfig',
    'ChannelCrosstalkModel',
    'RLGCParameters',
    # Signal integrity
    'TXPreEmphasis',
    'RXCTLE',
    'DFEEqualizer',
    'SignalIntegrityModel',
    'PreEmphasisConfig',
    'CTLEConfig',
    'DFEConfig',
    'SignalIntegrityConfig',
    'EqualizerType',
    # Eye analysis
    'EyeDiagramAnalyzer',
    'EyeMeasurementConfig',
    'BathtubCurveGenerator',
    'EyeMetrics',
    'EyeMeasurementType',
    # IBIS parser
    'IBISParser',
    'IBISFile',
    'IBISModel',
    'IBISModelType',
    'IBISPackage',
    'IBISPin',
    'IVCurve',
    'VTWaveform',
    'CompositeDataTable',
    'parse_ibis_file',
    'parse_ibis_content',
    # IBIS model wrapper
    'IBISModelWrapper',
    'BehavioralModel',
    'WaveformMetrics',
    'ChannelResponse',
    'SignalIntegrityMetric',
    'create_model_wrapper',
    'create_model_wrapper_from_file',
    # IBIS simulator
    'IBISSimulator',
    'ChannelParameters',
    'SimulationConfig',
    'SimulationMode',
    'SignalDistortion',
    'CrosstalkResult',
    'EyeAnalysisResult',
    'SimulationResult',
    'create_simulator',
    # New PHY training models
    'PHYTrainingState',
    'PHYTrainingType',
    'TrainingPattern',
    'PHYTrainingConfig',
    'TrainingPhaseResult',
    'PHYTapCoefficients',
    'PHYTrainingStatus',
    'PHYTrainingModel',
    'create_phy_training_model',
    # Training sequences
    'TrainingSequenceType',
    'DFITrainingCommand',
    'TrainingSequenceStep',
    'TrainingSequenceDefinition',
    'DFITrainingControl',
    'TrainingCompletionStatus',
    'TrainingSequenceExecutor',
    'TrainingCompletionDetector',
    'QUICK_BOOT_SEQUENCE',
    'NORMAL_TRAINING_SEQUENCE',
    'EXTENDED_TRAINING_SEQUENCE',
    'MARGIN_SCAN_SEQUENCE',
    'create_training_sequence',
    'get_dfi_training_command',
    # Tap coefficients
    'CoefficientType',
    'TXCoefficients',
    'RXCoefficients',
    'LaneCoefficients',
    'CompleteTapCoefficients',
    'CoefficientOptimizer',
    'CoefficientComparator',
    'create_default_coefficients',
    'export_coefficients_to_dict',
    'import_coefficients_from_dict',
]