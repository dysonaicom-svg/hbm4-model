"""
Training Sequences for HBM PHY

Implements training sequence definitions, DFI command generation
for training, and training completion detection.

Reference:
- JEDEC JESD270-4A HBM4 specification
- DFI 5.0/5.1 specification
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import deque
import numpy as np


class TrainingSequenceType(Enum):
    """Training sequence types"""
    QUICK_BOOT = auto()          # Quick boot training
    NORMAL = auto()             # Normal full training
    EXTENDED = auto()           # Extended training with margin scan
    DEBUG = auto()              # Debug mode with verbose output
    MARGIN_SCAN = auto()        # Margin scan only


class DFITrainingCommand(Enum):
    """DFI training commands
    
    Encodes training operations for DFI 5.0/5.1 interface.
    """
    NOP = 0x00                  # No operation
    WRLVL_REQ = 0x01            # Write leveling request
    RDGD_REQ = 0x02             # Read gate training request
    RDDLY_REQ = 0x03            # Read delay training request
    WR_DQ_REQ = 0x04            # Write DQ training request
    RD_DQ_REQ = 0x05            # Read DQ training request
    VREF_REQ = 0x06             # VREF training request
    MGCAL_REQ = 0x07           # Margin calibration request
    DFE_REQ = 0x08             # DFE training request
    SYNC_REQ = 0x09            # Synchronization request
    ABORT = 0x0F               # Abort training


@dataclass
class TrainingSequenceStep:
    """Single step in a training sequence"""
    name: str
    command: DFITrainingCommand
    target_state: str
    iterations: int = 64
    timeout_cycles: int = 50000
    pattern: str = "PRBS7"
    verify_after: bool = True
    
    # Per-lane configuration
    per_lane: bool = True
    lanes_per_group: int = 8
    
    # Success criteria
    min_margin: float = 0.15
    min_eye_height: float = 50.0  # mV


@dataclass
class TrainingSequenceDefinition:
    """Complete training sequence definition"""
    name: str
    sequence_type: TrainingSequenceType
    steps: List[TrainingSequenceStep] = field(default_factory=list)
    total_timeout_cycles: int = 500000
    retry_count: int = 3


@dataclass
class DFITrainingControl:
    """DFI training control signals
    
    Controls PHY training via DFI 5.0/5.1 interface.
    """
    # Training request
    training_req: bool = False
    training_cmd: DFITrainingCommand = DFITrainingCommand.NOP
    training_mode: int = 0  # 0=normal, 1=quick, 2=extended
    
    # Training acknowledge
    training_ack: bool = False
    training_error: bool = False
    training_error_code: int = 0
    
    # Training data
    training_data_valid: bool = False
    training_vref: int = 32
    training_delay: int = 0
    
    # Completion signals
    training_complete: bool = False
    training_passed: bool = False
    
    # Per-lane control
    lane_enable: int = 0xFFFFFFFFFFFFFFFF  # 64 lanes
    lane_select: int = 0


@dataclass
class TrainingCompletionStatus:
    """Training completion detection status"""
    sequence_complete: bool = False
    all_phases_passed: bool = False
    total_cycles: int = 0
    failed_phases: List[str] = field(default_factory=list)
    warning_count: int = 0
    
    # Margins
    min_read_margin: float = 0.0
    min_write_margin: float = 0.0
    min_vref_margin: float = 0.0


# Pre-defined training sequences
QUICK_BOOT_SEQUENCE = TrainingSequenceDefinition(
    name="Quick Boot Training",
    sequence_type=TrainingSequenceType.QUICK_BOOT,
    steps=[
        TrainingSequenceStep(
            name="Write Leveling",
            command=DFITrainingCommand.WRLVL_REQ,
            target_state="WRLVL",
            iterations=32,
            timeout_cycles=25000,
            per_lane=False
        ),
        TrainingSequenceStep(
            name="Read Gate",
            command=DFITrainingCommand.RDGD_REQ,
            target_state="RDGD",
            iterations=32,
            timeout_cycles=25000,
            per_lane=False
        ),
        TrainingSequenceStep(
            name="VREF Training",
            command=DFITrainingCommand.VREF_REQ,
            target_state="MGCAL_VREF",
            iterations=32,
            timeout_cycles=25000,
            verify_after=False
        ),
    ],
    total_timeout_cycles=100000,
    retry_count=2
)

NORMAL_TRAINING_SEQUENCE = TrainingSequenceDefinition(
    name="Normal Training",
    sequence_type=TrainingSequenceType.NORMAL,
    steps=[
        TrainingSequenceStep(
            name="Write Leveling",
            command=DFITrainingCommand.WRLVL_REQ,
            target_state="WRLVL",
            iterations=64,
            timeout_cycles=50000
        ),
        TrainingSequenceStep(
            name="Write Leveling DQS",
            command=DFITrainingCommand.WRLVL_REQ,
            target_state="WRLVL_DQS",
            iterations=16,
            timeout_cycles=20000
        ),
        TrainingSequenceStep(
            name="Read Gate",
            command=DFITrainingCommand.RDGD_REQ,
            target_state="RDGD",
            iterations=64,
            timeout_cycles=50000
        ),
        TrainingSequenceStep(
            name="Read Gate DQS",
            command=DFITrainingCommand.RDGD_REQ,
            target_state="RDGD_DQS",
            iterations=16,
            timeout_cycles=20000
        ),
        TrainingSequenceStep(
            name="Read Delay",
            command=DFITrainingCommand.RDDLY_REQ,
            target_state="RDDLY",
            iterations=64,
            per_lane=True
        ),
        TrainingSequenceStep(
            name="Write DQ",
            command=DFITrainingCommand.WR_DQ_REQ,
            target_state="WR_DQ",
            iterations=64,
            per_lane=True
        ),
        TrainingSequenceStep(
            name="Read DQ",
            command=DFITrainingCommand.RD_DQ_REQ,
            target_state="RD_DQ",
            iterations=64,
            per_lane=True
        ),
        TrainingSequenceStep(
            name="VREF Training",
            command=DFITrainingCommand.VREF_REQ,
            target_state="MGCAL_VREF",
            iterations=64,
            timeout_cycles=50000
        ),
        TrainingSequenceStep(
            name="Margin Calibration",
            command=DFITrainingCommand.MGCAL_REQ,
            target_state="MGCAL_DQ",
            iterations=64,
            per_lane=True
        ),
        TrainingSequenceStep(
            name="DFE Training",
            command=DFITrainingCommand.DFE_REQ,
            target_state="DFE_TRAIN",
            iterations=128,
            timeout_cycles=80000
        ),
    ],
    total_timeout_cycles=500000,
    retry_count=3
)

EXTENDED_TRAINING_SEQUENCE = TrainingSequenceDefinition(
    name="Extended Training",
    sequence_type=TrainingSequenceType.EXTENDED,
    steps=[
        *NORMAL_TRAINING_SEQUENCE.steps,
        TrainingSequenceStep(
            name="DFE Adaptation",
            command=DFITrainingCommand.DFE_REQ,
            target_state="DFE_ADAPT",
            iterations=256,
            timeout_cycles=100000
        ),
    ],
    total_timeout_cycles=800000,
    retry_count=3
)

MARGIN_SCAN_SEQUENCE = TrainingSequenceDefinition(
    name="Margin Scan",
    sequence_type=TrainingSequenceType.MARGIN_SCAN,
    steps=[
        TrainingSequenceStep(
            name="Read Margin Scan",
            command=DFITrainingCommand.RDGD_REQ,
            target_state="RDGD",
            iterations=128,
            timeout_cycles=80000
        ),
        TrainingSequenceStep(
            name="Write Margin Scan",
            command=DFITrainingCommand.WRLVL_REQ,
            target_state="WRLVL",
            iterations=128,
            timeout_cycles=80000
        ),
        TrainingSequenceStep(
            name="VREF Margin Scan",
            command=DFITrainingCommand.VREF_REQ,
            target_state="MGCAL_VREF",
            iterations=128,
            timeout_cycles=80000
        ),
    ],
    total_timeout_cycles=300000,
    retry_count=1
)


class TrainingSequenceExecutor:
    """Execute training sequences
    
    Manages the execution of training sequences, generating
    appropriate DFI commands and monitoring completion.
    """
    
    def __init__(self, sequence: Optional[TrainingSequenceDefinition] = None,
                 dfi_interface=None):
        """Initialize training sequence executor
        
        Args:
            sequence: Training sequence definition
            dfi_interface: DFI 5.0/5.1 interface
        """
        self.sequence = sequence
        self.dfi = dfi_interface
        
        # Execution state
        self._current_step_index = 0
        self._step_iteration = 0
        self._step_cycle_start = 0
        self._total_cycles = 0
        self._retry_count = 0
        
        # DFI control
        self.dfi_control = DFITrainingControl()
        
        # Completion tracking
        self.completion_status = TrainingCompletionStatus()
        
        # Results
        self._step_results: List[Dict[str, Any]] = []
    
    @property
    def current_step(self) -> Optional[TrainingSequenceStep]:
        """Get current step in sequence"""
        if self.sequence and self._current_step_index < len(self.sequence.steps):
            return self.sequence.steps[self._current_step_index]
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if sequence execution is complete"""
        return self.completion_status.sequence_complete
    
    def start_sequence(self, sequence_type: TrainingSequenceType = TrainingSequenceType.NORMAL):
        """Start a training sequence
        
        Args:
            sequence_type: Type of sequence to execute
        """
        # Select sequence based on type
        sequence_map = {
            TrainingSequenceType.QUICK_BOOT: QUICK_BOOT_SEQUENCE,
            TrainingSequenceType.NORMAL: NORMAL_TRAINING_SEQUENCE,
            TrainingSequenceType.EXTENDED: EXTENDED_TRAINING_SEQUENCE,
            TrainingSequenceType.MARGIN_SCAN: MARGIN_SCAN_SEQUENCE,
        }
        
        self.sequence = sequence_map.get(sequence_type, NORMAL_TRAINING_SEQUENCE)
        self._current_step_index = 0
        self._step_iteration = 0
        self._retry_count = 0
        self._step_results.clear()
        self.completion_status = TrainingCompletionStatus()
        
        # Start first step
        self._start_step()
    
    def _start_step(self):
        """Start execution of current step"""
        step = self.current_step
        if step is None:
            self.completion_status.sequence_complete = True
            return
        
        self._step_iteration = 0
        self._step_cycle_start = self._total_cycles
        
        # Configure DFI for this step
        self.dfi_control.training_req = True
        self.dfi_control.training_cmd = step.command
        self.dfi_control.training_mode = 1 if step.per_lane else 0
        
        # Send to DFI interface
        if self.dfi:
            self.dfi.set_training_command(step.command.value)
            self.dfi.set_training_mode(self.dfi_control.training_mode)
    
    def tick(self) -> bool:
        """Process one training cycle
        
        Args:
            None
            
        Returns:
            True if step/sequence completed successfully
        """
        self._total_cycles += 1
        
        # Check for step timeout
        step = self.current_step
        if step:
            elapsed = self._total_cycles - self._step_cycle_start
            if elapsed > step.timeout_cycles:
                return self._handle_step_timeout()
        
        # Execute step iterations
        return self._execute_step_iteration()
    
    def _execute_step_iteration(self) -> bool:
        """Execute one iteration of current step"""
        step = self.current_step
        if step is None:
            self.completion_status.sequence_complete = True
            return True
        
        # Execute iteration
        self._step_iteration += 1
        
        # Perform training measurement
        success, margin, best_value = self._measure_step_iteration(step)
        
        # Check if step is complete
        if success and margin >= step.min_margin:
            return self._complete_step(success, margin, best_value)
        
        if self._step_iteration >= step.iterations:
            # Check margin
            if margin < step.min_margin:
                if self._retry_count < (self.sequence.retry_count if self.sequence else 3):
                    self._retry_count += 1
                    self._step_iteration = 0
                    return False
                else:
                    return self._complete_step(False, margin, best_value)
            else:
                return self._complete_step(True, margin, best_value)
        
        return False
    
    def _measure_step_iteration(self, step: TrainingSequenceStep) -> Tuple[bool, float, int]:
        """Measure training iteration
        
        Args:
            step: Current step configuration
            
        Returns:
            Tuple of (success, margin, best_value)
        """
        # Simulate measurement
        # In real implementation, this would interface with PHY hardware
        import random
        margin = random.uniform(0.1, 0.9)
        best_value = self._step_iteration
        
        success = margin >= step.min_margin * 0.5
        
        return success, margin, best_value
    
    def _complete_step(self, success: bool, margin: float, best_value: int) -> bool:
        """Complete current step and advance
        
        Args:
            success: Whether step passed
            margin: Achieved margin
            best_value: Best delay/vref value
            
        Returns:
            True if step passed
        """
        step = self.current_step
        if step is None:
            return True
        
        # Record result
        result = {
            'step_name': step.name,
            'success': success,
            'margin': margin,
            'best_value': best_value,
            'iterations': self._step_iteration,
            'cycles': self._total_cycles - self._step_cycle_start
        }
        self._step_results.append(result)
        
        if not success:
            self.completion_status.failed_phases.append(step.name)
        
        # Update completion status
        self._update_completion_status()
        
        # Advance to next step
        self._current_step_index += 1
        self._step_iteration = 0
        self._step_cycle_start = self._total_cycles
        
        if self._current_step_index >= len(self.sequence.steps):
            self.completion_status.sequence_complete = True
            self.completion_status.all_phases_passed = len(self.completion_status.failed_phases) == 0
            return success
        
        # Start next step
        self._start_step()
        return success
    
    def _handle_step_timeout(self) -> bool:
        """Handle step timeout"""
        step = self.current_step
        if step:
            self._step_results.append({
                'step_name': step.name,
                'success': False,
                'error': 'timeout',
                'cycles': step.timeout_cycles
            })
            self.completion_status.failed_phases.append(f"{step.name} (timeout)")
        
        # Retry or advance
        if self._retry_count < (self.sequence.retry_count if self.sequence else 3):
            self._retry_count += 1
            self._step_iteration = 0
            self._step_cycle_start = self._total_cycles
            return False
        else:
            self._retry_count = 0
            self._current_step_index += 1
            if self._current_step_index >= len(self.sequence.steps):
                self.completion_status.sequence_complete = True
            else:
                self._start_step()
            return False
    
    def _update_completion_status(self):
        """Update completion status after step"""
        self.completion_status.total_cycles = self._total_cycles
        
        # Calculate minimum margins
        read_margins = [r['margin'] for r in self._step_results 
                       if 'Read' in r.get('step_name', '')]
        write_margins = [r['margin'] for r in self._step_results 
                        if 'Write' in r.get('step_name', '')]
        vref_margins = [r['margin'] for r in self._step_results 
                       if 'VREF' in r.get('step_name', '')]
        
        self.completion_status.min_read_margin = min(read_margins) if read_margins else 0.0
        self.completion_status.min_write_margin = min(write_margins) if write_margins else 0.0
        self.completion_status.min_vref_margin = min(vref_margins) if vref_margins else 0.0
    
    def get_results(self) -> Dict[str, Any]:
        """Get sequence execution results
        
        Returns:
            Dictionary with execution results
        """
        return {
            'sequence_name': self.sequence.name if self.sequence else None,
            'sequence_type': self.sequence.sequence_type.name if self.sequence else None,
            'is_complete': self.completion_status.sequence_complete,
            'all_passed': self.completion_status.all_phases_passed,
            'total_cycles': self.completion_status.total_cycles,
            'failed_phases': self.completion_status.failed_phases,
            'min_read_margin': self.completion_status.min_read_margin,
            'min_write_margin': self.completion_status.min_write_margin,
            'min_vref_margin': self.completion_status.min_vref_margin,
            'step_results': self._step_results,
        }


class TrainingCompletionDetector:
    """Detect training completion
    
    Monitors training progress and determines when training
    is complete based on various criteria.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize completion detector
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Detection criteria
        self.min_pass_count = self.config.get('min_pass_count', 2)
        self.margin_stable_threshold = self.config.get('margin_stable_threshold', 0.02)
        self.consecutive_pass_required = self.config.get('consecutive_pass_required', 3)
        
        # State tracking
        self._pass_history: deque = deque(maxlen=10)
        self._margin_history: deque = deque(maxlen=20)
        self._last_best_margin = 0.0
    
    def update(self, phase_passed: bool, margin: float, phase_name: str) -> bool:
        """Update detector with new measurement
        
        Args:
            phase_passed: Whether phase passed
            margin: Achieved margin
            phase_name: Name of current phase
            
        Returns:
            True if training should be considered complete
        """
        self._pass_history.append(phase_passed)
        self._margin_history.append(margin)
        
        # Check consecutive passes
        consecutive = 0
        for passed in reversed(self._pass_history):
            if passed:
                consecutive += 1
            else:
                break
        
        if consecutive >= self.consecutive_pass_required:
            return True
        
        # Check margin stability
        if len(self._margin_history) >= 5:
            recent_margins = list(self._margin_history)[-5:]
            margin_std = np.std(recent_margins)
            
            if margin_std < self.margin_stable_threshold:
                return True
        
        # Check improvement plateau
        margin_improvement = margin - self._last_best_margin
        if margin_improvement < self.margin_stable_threshold:
            # No significant improvement
            if len(self._margin_history) >= self.min_pass_count:
                return True
        
        self._last_best_margin = max(self._last_best_margin, margin)
        return False
    
    def reset(self):
        """Reset detector state"""
        self._pass_history.clear()
        self._margin_history.clear()
        self._last_best_margin = 0.0


def create_training_sequence(sequence_type: TrainingSequenceType) -> TrainingSequenceDefinition:
    """Factory function to create training sequence
    
    Args:
        sequence_type: Type of sequence to create
        
    Returns:
        Training sequence definition
    """
    sequence_map = {
        TrainingSequenceType.QUICK_BOOT: QUICK_BOOT_SEQUENCE,
        TrainingSequenceType.NORMAL: NORMAL_TRAINING_SEQUENCE,
        TrainingSequenceType.EXTENDED: EXTENDED_TRAINING_SEQUENCE,
        TrainingSequenceType.MARGIN_SCAN: MARGIN_SCAN_SEQUENCE,
    }
    
    return sequence_map.get(sequence_type, NORMAL_TRAINING_SEQUENCE)


def get_dfi_training_command(phase_name: str) -> DFITrainingCommand:
    """Map phase name to DFI training command
    
    Args:
        phase_name: Name of training phase
        
    Returns:
        Corresponding DFI command
    """
    command_map = {
        'WRLVL': DFITrainingCommand.WRLVL_REQ,
        'WRLVL_DQS': DFITrainingCommand.WRLVL_REQ,
        'RDGD': DFITrainingCommand.RDGD_REQ,
        'RDGD_DQS': DFITrainingCommand.RDGD_REQ,
        'RDDLY': DFITrainingCommand.RDDLY_REQ,
        'WR_DQ': DFITrainingCommand.WR_DQ_REQ,
        'RD_DQ': DFITrainingCommand.RD_DQ_REQ,
        'MGCAL_VREF': DFITrainingCommand.VREF_REQ,
        'MGCAL_DQ': DFITrainingCommand.MGCAL_REQ,
        'DFE_TRAIN': DFITrainingCommand.DFE_REQ,
        'DFE_ADAPT': DFITrainingCommand.DFE_REQ,
    }
    
    return command_map.get(phase_name, DFITrainingCommand.NOP)