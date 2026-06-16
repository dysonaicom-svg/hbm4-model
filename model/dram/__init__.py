"""
HBM4 DRAM Model Package

This package contains all DRAM-related models for HBM4 simulation.

Modules:
- hbm4_spec: HBM4 specification constants
- hbm4_channel_model: Channel-level model
- channel_timing: Independent channel timing (NEW)
- dfi_interface: DFI 5.0 interface
- phy_signal: PAM3 signal encoding (NEW)
- phy_training: PHY training sequences
- logic_base_die: Logic Base Die unified model (NEW)
- lane_repair: Lane repair/redundancy
- ecc_crc: ECC/CRC error handling
- power_estimator: Power consumption estimation
- bank_state_machine: Per-bank state tracking
- timing: Timing parameter management
- dram_model: Generic DRAM model
- stack_model: HBM stack model
- mbist_controller: Memory BIST controller
- loopback_controller: Loopback test controller
"""

# HBM4 Specification
from model.dram.hbm4_spec import HBM4Spec

# Channel Models
from model.dram.channel_model import Channel, ChannelArray
from model.dram.hbm4_channel_model import HBM4Channel
from model.dram.channel_timing import (
    IndependentChannelTiming,
    HBM4TimingManager,
    TimingParameters,
    ChannelClockDomain,
    BankState,
)

# PHY Models
from model.dram.phy_signal import (
    PAM3SignalModel,
    PAM3Level,
    PAM3Symbol,
    PAM3EyeDiagram,
    HBM4PAM3Encoder,
)
from model.dram.phy_training import (
    PHYTrainingStateMachine,
    PHYInitializationStateMachine,
    HBM4PHYManager,
    TrainingPhase,
    TrainingResult,
)

# Logic Base Die (NEW - Core integration)
from model.dram.logic_base_die import (
    HBM4LogicBaseDie,
    LogicBaseDieConfig,
    ChannelState,
    ChannelContext,
)

# Error Handling
from model.dram.ecc_crc import (
    HBM4ECC,
    HBM4CRC,
    HBM4DataIntegrity,
    ErrorType,
    HBM4ECCMode,
    HBM4CRCMode,
)

# Reliability
from model.dram.lane_repair import (
    HBM4LaneRepairModel,
    LaneRepairMap,
    LaneRepairEntry,
    RepairStatus,
)

# Power and Thermal
from model.dram.power_estimator import HBM4PowerEstimator

# Timing
from model.dram.timing import (
    HBM3Timing,
    HBM4Timing,
    get_timing_for_speed_grade,
)

# Interfaces
from model.dram.dfi_interface import (
    DFI5Interface,
    DFICommand,
    DFIRequest,
    DFILowPowerState,
)

# State Management
from model.dram.bank_state_machine import (
    BankStateMachine,
    BankStateEnum,
)

# Full Stack Models
from model.dram.dram_model import DRAMModel
from model.dram.stack_model import Stack, StackArray
from model.dram.mbist_controller import MBISTController, MBISTConfig, MBISTResult

__all__ = [
    # Specification
    'HBM4Spec',

    # Channel Models
    'Channel',
    'ChannelArray',
    'HBM4Channel',
    'IndependentChannelTiming',
    'HBM4TimingManager',
    'TimingParameters',
    'ChannelClockDomain',
    

    # PHY Models (NEW)
    'PAM3SignalModel',
    'PAM3Level',
    'PAM3Symbol',
    'PAM3EyeDiagram',
    'HBM4PAM3Encoder',
    'PHYTrainingStateMachine',
    'PHYInitializationStateMachine',
    'HBM4PHYManager',
    'TrainingPhase',
    'TrainingResult',

    # Logic Base Die (NEW - Core)
    'HBM4LogicBaseDie',
    'LogicBaseDieConfig',
    'ChannelState',
    'ChannelContext',

    # Error Handling
    'HBM4ECC',
    'HBM4CRC',
    'HBM4DataIntegrity',
    'ErrorType',
    'HBM4ECCMode',
    'HBM4CRCMode',

    # Reliability
    'HBM4LaneRepairModel',
    'LaneRepairMap',
    'LaneRepairEntry',
    'RepairStatus',

    # Power
    'HBM4PowerEstimator',

    # Interfaces
    'DFI5Interface',
    'DFICommand',
    'DFIRequest',
    'DFILowPowerState',

    # State Management
    'BankStateMachine',
    'BankStateEnum',

    # Full Stack
    'DRAMModel',
    'Stack',
    'StackArray',
    'MBISTController',
    'MBISTConfig',
    'MBISTResult',
]

# Version info
__version__ = '1.0.0'
__hbm4_compliant__ = True