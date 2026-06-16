"""
PHY Training Model for HBM Simulation Platform

Complete PHY training implementation with:
- Training state machine (IDLE, WRLVL, RDGD, MGCAL, etc.)
- Write levelization sequence
- Read gate training sequence
- Margin calibration
- DFE training
- DFI 5.0/5.1 integration

Reference:
- JEDEC JESD270-4A HBM4 specification
- DFI 5.0/5.1 specification
- Cadence HBM4E documentation
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import deque
import numpy as np


# Training timing constants (cycles @ various speeds)
DEFAULT_TRAINING_TIMEOUT = 100000  # 100ms @ 1ns cycle
WRLVL_TIMEOUT = 50000
RDGD_TIMEOUT = 50000
MGCAL_TIMEOUT = 80000
DFE_TRAIN_TIMEOUT = 100000


class PHYTrainingState(Enum):
    """PHY Training State Machine States
    
    Implements the HBM4 PHY training state machine as defined
    in JEDEC JESD270-4A.
    """
    IDLE = auto()              # Idle state, no training in progress
    INIT = auto()              # Training initialization
    WRLVL = auto()             # Write leveling training
    WRLVL_DQS = auto()         # Write leveling DQS adjustment
    RDGD = auto()              # Read gate training
    RDGD_DQS = auto()          # Read gate DQS adjustment
    RDDLY = auto()             # Read data delay training
    WR_DQ = auto()             # Write DQ training
    RD_DQ = auto()             # Read DQ training
    MGCAL_VREF = auto()        # Margin calibration VREF training
    MGCAL_DQ = auto()          # Margin calibration DQ training
    DFE_TRAIN = auto()         # DFE tap training
    DFE_ADAPT = auto()         # DFE adaptive equalization
    VERIFY = auto()            # Verify training results
    COMPLETE = auto()           # Training complete
    FAIL = auto()             # Training failed


class PHYTrainingType(Enum):
    """Types of PHY training"""
    NORMAL = auto()            # Normal training mode
    QUICK = auto()             # Quick training (reduced iterations)
    VERIFY_ONLY = auto()       # Verify only, no training
    MARGIN_SCAN = auto()       # Margin scan mode


class TrainingPattern(Enum):
    """Training patterns for various calibration phases"""
    PRBS7 = "prbs7"
    PRBS15 = "prbs15"
    PRBS31 = "prbs31"
    WALKING_1 = "walking_1"
    WALKING_0 = "walking_0"
    ALL_ONES = "all_ones"
    ALL_ZEROS = "all_zeros"
    ALTERNATING = "alternating"
    USER_DEFINED = "user_defined"


@dataclass
class PHYTrainingConfig:
    """Configuration for PHY training"""
    # Training parameters
    enable_write_leveling: bool = True
    enable_read_gate: bool = True
    enable_margin_cal: bool = True
    enable_dfe: bool = True
    enable_per_lane: bool = True
    
    # Timing parameters
    wrlvl_iterations: int = 64      # Write leveling sweep points
    rdgd_iterations: int = 64       # Read gate sweep points
    mgcal_iterations: int = 64      # Margin calibration points
    dfe_iterations: int = 128       # DFE convergence iterations
    
    # Convergence criteria
    min_margin_ui: float = 0.15     # Minimum margin in UI
    min_eye_height_mv: float = 50.0  # Minimum eye height in mV
    convergence_threshold: float = 0.01  # DFE convergence threshold
    
    # Timeout settings
    timeout_cycles: int = DEFAULT_TRAINING_TIMEOUT
    retry_count: int = 3
    
    # Lane configuration
    num_lanes: int = 64             # HBM4: 64 lanes per channel
    lanes_per_group: int = 8        # Group lanes for parallel calibration


@dataclass
class TrainingPhaseResult:
    """Result of a single training phase"""
    phase: PHYTrainingState
    passed: bool
    start_cycle: int = 0
    end_cycle: int = 0
    best_value: int = 0
    best_margin: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def duration_cycles(self) -> int:
        return self.end_cycle - self.start_cycle


@dataclass
class PHYTapCoefficients:
    """PHY tap coefficients after training"""
    # TX coefficients
    tx_precursor: List[float] = field(default_factory=list)  # Pre-cursor taps
    tx_postcursor: List[float] = field(default_factory=list)  # Post-cursor taps
    tx_main_cursor: float = 1.0
    
    # RX coefficients  
    rx_ctle_dc_gain: float = 0.0
    rx_ctle_peaking: float = 3.0
    rx_vref: int = 32
    
    # DFE coefficients
    dfe_taps: List[float] = field(default_factory=list)
    
    # Per-lane delays
    lane_delays: Dict[int, int] = field(default_factory=dict)
    
    # Per-lane DQ delays
    lane_dq_delays: Dict[int, int] = field(default_factory=dict)
    
    def get_tx_taps(self) -> List[float]:
        """Get all TX tap values including main cursor"""
        return self.tx_precursor + [self.tx_main_cursor] + self.tx_postcursor
    
    def is_valid(self) -> bool:
        """Check if coefficients are valid"""
        if len(self.tx_precursor) == 0 or len(self.tx_postcursor) == 0:
            return False
        if self.rx_vref < 0 or self.rx_vref > 63:
            return False
        return True


@dataclass
class PHYTrainingStatus:
    """Current status of PHY training"""
    state: PHYTrainingState = PHYTrainingState.IDLE
    state_enter_cycle: int = 0
    training_type: PHYTrainingType = PHYTrainingType.NORMAL
    current_pattern: TrainingPattern = TrainingPattern.PRBS7
    current_lane: int = 0
    current_bank_group: int = 0
    
    # Statistics
    total_cycles: int = 0
    phase_results: Dict[PHYTrainingState, TrainingPhaseResult] = field(default_factory=dict)
    retry_count: int = 0
    
    # DFI interface state
    dfi_training_req: bool = False
    dfi_training_ack: bool = False
    dfi_training_complete: bool = False
    dfi_training_error: bool = False


class PHYTrainingError(Exception):
    """Exception raised for PHY training errors"""
    pass


class PHYTrainingTimeout(Exception):
    """Exception raised for training timeout"""
    pass


class PHYTrainingConvergenceError(Exception):
    """Exception raised when training fails to converge"""
    pass


class PHYTrainingModel:
    """PHY Training State Machine
    
    Implements complete PHY training for HBM4 including:
    - Write leveling
    - Read gate training
    - Margin calibration
    - DFE training
    - Per-lane calibration
    
    Integrates with DFI 5.0/5.1 interface for controller communication.
    """
    
    # Training sequence order
    TRAINING_SEQUENCE = [
        PHYTrainingState.WRLVL,
        PHYTrainingState.WRLVL_DQS,
        PHYTrainingState.RDGD,
        PHYTrainingState.RDGD_DQS,
        PHYTrainingState.RDDLY,
        PHYTrainingState.WR_DQ,
        PHYTrainingState.RD_DQ,
        PHYTrainingState.MGCAL_VREF,
        PHYTrainingState.MGCAL_DQ,
        PHYTrainingState.DFE_TRAIN,
    ]
    
    def __init__(self, channel_id: int = 0,
                 config: Optional[PHYTrainingConfig] = None,
                 dfi_interface=None):
        """Initialize PHY training model
        
        Args:
            channel_id: Channel index for this training instance
            config: Training configuration
            dfi_interface: Optional DFI 5.0/5.1 interface
        """
        self.channel_id = channel_id
        self.config = config or PHYTrainingConfig()
        self.dfi = dfi_interface
        
        # State tracking
        self.status = PHYTrainingStatus()
        self.coefficients = PHYTapCoefficients()
        
        # Cycle counter
        self._cycle = 0
        
        # Training data
        self._prng = np.random.RandomState(42)  # Deterministic seed
        self._lane_count = self.config.num_lanes
        
        # Pattern generator state
        self._pattern_state = 0
        self._pattern_buffer = deque(maxlen=256)
        
        # Phase handlers
        self._phase_handlers: Dict[PHYTrainingState, Callable[[], bool]] = {
            PHYTrainingState.WRLVL: self._train_write_leveling,
            PHYTrainingState.WRLVL_DQS: self._train_write_leveling_dqs,
            PHYTrainingState.RDGD: self._train_read_gate,
            PHYTrainingState.RDGD_DQS: self._train_read_gate_dqs,
            PHYTrainingState.RDDLY: self._train_read_delay,
            PHYTrainingState.WR_DQ: self._train_write_dq,
            PHYTrainingState.RD_DQ: self._train_read_dq,
            PHYTrainingState.MGCAL_VREF: self._train_vref,
            PHYTrainingState.MGCAL_DQ: self._train_margin_dq,
            PHYTrainingState.DFE_TRAIN: self._train_dfe,
            PHYTrainingState.VERIFY: self._verify_training,
        }
        
        # Initialize coefficients
        self._init_coefficients()
    
    def _init_coefficients(self):
        """Initialize default coefficients"""
        # TX: 2 pre-taps, main cursor, 2 post-taps
        self.coefficients.tx_precursor = [0.0, 0.0]
        self.coefficients.tx_postcursor = [0.0, 0.0]
        self.coefficients.tx_main_cursor = 1.0
        
        # RX: Default VREF (6-bit DAC, mid-range)
        self.coefficients.rx_vref = 32
        self.coefficients.rx_ctle_dc_gain = 0.0
        self.coefficients.rx_ctle_peaking = 3.0
        
        # DFE: 5 taps
        self.coefficients.dfe_taps = [0.0] * 5
        
        # Per-lane delays (all lanes start at 0)
        for lane in range(self._lane_count):
            self.coefficients.lane_delays[lane] = 0
            self.coefficients.lane_dq_delays[lane] = 0
    
    @property
    def cycle(self) -> int:
        """Current simulation cycle"""
        return self._cycle
    
    @property
    def is_training(self) -> bool:
        """Check if training is in progress"""
        return self.status.state not in [PHYTrainingState.IDLE,
                                         PHYTrainingState.COMPLETE,
                                         PHYTrainingState.FAIL]
    
    @property
    def is_complete(self) -> bool:
        """Check if training has completed"""
        return self.status.state in [PHYTrainingState.COMPLETE,
                                      PHYTrainingState.FAIL]
    
    def tick(self):
        """Advance training state machine by one cycle"""
        self._cycle += 1
        self.status.total_cycles += 1
        
        # Check for timeout
        elapsed = self._cycle - self.status.state_enter_cycle
        if elapsed > self.config.timeout_cycles:
            self._handle_timeout()
    
    def start_training(self, training_type: PHYTrainingType = PHYTrainingType.NORMAL):
        """Start training sequence
        
        Args:
            training_type: Type of training to perform
        """
        if self.is_training:
            return False
        
        # Reset state
        self.status.state = PHYTrainingState.INIT
        self.status.state_enter_cycle = self._cycle
        self.status.training_type = training_type
        self.status.phase_results.clear()
        self.status.retry_count = 0
        
        # Signal DFI
        if self.dfi:
            self.dfi.start_training()
            self.status.dfi_training_req = True
        
        return True
    
    def _handle_timeout(self):
        """Handle training phase timeout"""
        # Record failure
        result = TrainingPhaseResult(
            phase=self.status.state,
            passed=False,
            start_cycle=self.status.state_enter_cycle,
            end_cycle=self._cycle,
            errors=[f"Timeout after {self.config.timeout_cycles} cycles"]
        )
        self.status.phase_results[self.status.state] = result
        
        # Retry or fail
        if self.status.retry_count < self.config.retry_count:
            self.status.retry_count += 1
            self.status.state = PHYTrainingState.INIT
            self.status.state_enter_cycle = self._cycle
        else:
            self.status.state = PHYTrainingState.FAIL
    
    def _advance_state(self):
        """Advance to next training state"""
        try:
            idx = self.TRAINING_SEQUENCE.index(self.status.state)
            if idx < len(self.TRAINING_SEQUENCE) - 1:
                self.status.state = self.TRAINING_SEQUENCE[idx + 1]
            else:
                self.status.state = PHYTrainingState.VERIFY
        except ValueError:
            # Not in sequence, move to first
            if self.TRAINING_SEQUENCE:
                self.status.state = self.TRAINING_SEQUENCE[0]
        
        self.status.state_enter_cycle = self._cycle
        self.status.current_lane = 0
        self.status.current_bank_group = 0
    
    def _record_phase_result(self, phase: PHYTrainingState, passed: bool,
                            best_value: int = 0, best_margin: float = 0.0,
                            errors: Optional[List[str]] = None):
        """Record result of a training phase"""
        result = TrainingPhaseResult(
            phase=phase,
            passed=passed,
            start_cycle=self.status.state_enter_cycle,
            end_cycle=self._cycle,
            best_value=best_value,
            best_margin=best_margin,
            errors=errors or []
        )
        self.status.phase_results[phase] = result
    
    # === Phase Training Handlers ===
    
    def _train_write_leveling(self) -> bool:
        """Execute write leveling training
        
        Write leveling aligns the write DQS signal with the write data.
        The PHY sends DQS strobes and the DRAM reports the alignment status.
        
        Returns:
            True if training passed
        """
        best_delay = 0
        best_margin = 0.0
        
        # Sweep through delay values
        for delay in range(self.config.wrlvl_iterations):
            margin = self._measure_wrlvl_margin(delay)
            if margin > best_margin:
                best_margin = margin
                best_delay = delay
            
            # Early termination if margin is excellent
            if margin > 0.9:
                break
        
        self.coefficients.lane_delays[0] = best_delay
        
        passed = best_margin >= self.config.min_margin_ui
        self._record_phase_result(PHYTrainingState.WRLVL, passed,
                                  best_delay, best_margin)
        
        return passed
    
    def _train_write_leveling_dqs(self) -> bool:
        """Execute write leveling DQS fine adjustment"""
        best_delay = self.coefficients.lane_delays.get(0, 32)
        best_margin = 0.0
        
        # Fine sweep around best delay
        start = max(0, best_delay - 4)
        end = min(self.config.wrlvl_iterations, best_delay + 5)
        
        for delay in range(start, end):
            margin = self._measure_wrlvl_margin(delay)
            if margin > best_margin:
                best_margin = margin
                best_delay = delay
        
        self.coefficients.lane_delays[0] = best_delay
        
        passed = best_margin >= self.config.min_margin_ui * 0.9
        self._record_phase_result(PHYTrainingState.WRLVL_DQS, passed,
                                  best_delay, best_margin)
        
        return passed
    
    def _train_read_gate(self) -> bool:
        """Execute read gate training
        
        Read gate training optimizes the timing of the read enable signal
        to capture data at the optimal sampling point.
        
        Returns:
            True if training passed
        """
        best_delay = 0
        best_margin = 0.0
        
        for delay in range(self.config.rdgd_iterations):
            margin = self._measure_rdgd_margin(delay)
            if margin > best_margin:
                best_margin = margin
                best_delay = delay
            
            if margin > 0.9:
                break
        
        # Store per-lane read gate delays
        for lane in range(self._lane_count):
            self.coefficients.lane_delays[lane] = best_delay
        
        passed = best_margin >= self.config.min_margin_ui
        self._record_phase_result(PHYTrainingState.RDGD, passed,
                                  best_delay, best_margin)
        
        return passed
    
    def _train_read_gate_dqs(self) -> bool:
        """Execute read gate DQS fine adjustment"""
        best_delay = self.coefficients.lane_delays.get(0, 32)
        best_margin = 0.0
        
        start = max(0, best_delay - 4)
        end = min(self.config.rdgd_iterations, best_delay + 5)
        
        for delay in range(start, end):
            margin = self._measure_rdgd_margin(delay)
            if margin > best_margin:
                best_margin = margin
                best_delay = delay
        
        for lane in range(self._lane_count):
            self.coefficients.lane_delays[lane] = best_delay
        
        passed = best_margin >= self.config.min_margin_ui * 0.9
        self._record_phase_result(PHYTrainingState.RDGD_DQS, passed,
                                  best_delay, best_margin)
        
        return passed
    
    def _train_read_delay(self) -> bool:
        """Execute read data delay training
        
        Per-lane read DQ delay calibration.
        
        Returns:
            True if training passed
        """
        all_passed = True
        
        if self.config.enable_per_lane:
            # Per-lane calibration
            for lane in range(self._lane_count):
                best_delay = self._calibrate_lane_rd(lane)
                self.coefficients.lane_dq_delays[lane] = best_delay
        else:
            # Group calibration
            for group in range(self._lane_count // self.config.lanes_per_group):
                best_delay = self._calibrate_group_rd(group)
                for i in range(self.config.lanes_per_group):
                    lane = group * self.config.lanes_per_group + i
                    self.coefficients.lane_dq_delays[lane] = best_delay
        
        self._record_phase_result(PHYTrainingState.RDDLY, all_passed)
        return all_passed
    
    def _train_write_dq(self) -> bool:
        """Execute write DQ training
        
        Per-lane write DQ delay calibration.
        
        Returns:
            True if training passed
        """
        all_passed = True
        
        if self.config.enable_per_lane:
            for lane in range(self._lane_count):
                best_delay = self._calibrate_lane_wr(lane)
                self.coefficients.lane_delays[lane + self._lane_count] = best_delay
        else:
            for group in range(self._lane_count // self.config.lanes_per_group):
                best_delay = self._calibrate_group_wr(group)
                for i in range(self.config.lanes_per_group):
                    lane = group * self.config.lanes_per_group + i
                    self.coefficients.lane_delays[lane + self._lane_count] = best_delay
        
        self._record_phase_result(PHYTrainingState.WR_DQ, all_passed)
        return all_passed
    
    def _train_read_dq(self) -> bool:
        """Execute read DQ training
        
        Fine-tune read DQ delays for optimal eye opening.
        
        Returns:
            True if training passed
        """
        all_passed = True
        
        for lane in range(self._lane_count):
            current_delay = self.coefficients.lane_dq_delays.get(lane, 32)
            best_delay = self._optimize_rd_dq_delay(lane, current_delay)
            self.coefficients.lane_dq_delays[lane] = best_delay
        
        self._record_phase_result(PHYTrainingState.RD_DQ, all_passed)
        return all_passed
    
    def _train_vref(self) -> bool:
        """Execute VREF training
        
        Optimize VREF settings for maximum margin.
        
        Returns:
            True if training passed
        """
        best_vref = 32
        best_margin = 0.0
        
        for vref in range(64):  # 6-bit DAC
            margin = self._measure_vref_margin(vref)
            if margin > best_margin:
                best_margin = margin
                best_vref = vref
        
        self.coefficients.rx_vref = best_vref
        
        passed = best_margin >= self.config.min_margin_ui
        self._record_phase_result(PHYTrainingState.MGCAL_VREF, passed,
                                  best_vref, best_margin)
        
        return passed
    
    def _train_margin_dq(self) -> bool:
        """Execute margin calibration for DQ
        
        Fine-tune per-lane delays for margin optimization.
        
        Returns:
            True if training passed
        """
        all_passed = True
        
        for lane in range(self._lane_count):
            current_delay = self.coefficients.lane_dq_delays.get(lane, 32)
            optimized = self._optimize_margin(lane, current_delay)
            self.coefficients.lane_dq_delays[lane] = optimized
        
        self._record_phase_result(PHYTrainingState.MGCAL_DQ, all_passed)
        return all_passed
    
    def _train_dfe(self) -> bool:
        """Execute DFE (Decision Feedback Equalizer) training
        
        Train DFE taps for optimal equalization.
        
        Returns:
            True if training passed
        """
        # Initialize DFE taps
        self.coefficients.dfe_taps = [0.0] * len(self.coefficients.dfe_taps)
        
        # Generate training pattern
        pattern = self._generate_training_pattern(TrainingPattern.PRBS15)
        
        # Adaptive LMS algorithm
        mu = 0.01  # Convergence rate
        max_tap_mag = 0.3
        
        for iteration in range(self.config.dfe_iterations):
            total_error = 0.0
            
            for i in range(1, len(pattern)):
                # Decision (assume ideal sampling)
                decision = 1 if pattern[i] > 0 else -1
                
                # Calculate DFE feedback
                feedback = sum(
                    self.coefficients.dfe_taps[j] * pattern[i - j - 1]
                    for j in range(min(i, len(self.coefficients.dfe_taps)))
                )

                # Equalized sample
                equalized = pattern[i] - feedback
                
                # Error
                error = equalized - decision
                total_error += error ** 2
                
                # LMS update
                for j in range(len(self.coefficients.dfe_taps)):
                    if i - j - 1 >= 0:
                        self.coefficients.dfe_taps[j] += mu * error * pattern[i - j - 1]
                        self.coefficients.dfe_taps[j] = np.clip(
                            self.coefficients.dfe_taps[j],
                            -max_tap_mag, max_tap_mag)
            
            # Check convergence
            mse = total_error / len(pattern)
            if mse < self.config.convergence_threshold:
                break
        
        passed = True  # DFE always passes (adapts until timeout)
        self._record_phase_result(PHYTrainingState.DFE_TRAIN, passed)
        
        return passed
    
    def _verify_training(self) -> bool:
        """Verify training results
        
        Run verification pattern to ensure training succeeded.
        
        Returns:
            True if verification passed
        """
        # Generate verification pattern
        pattern = self._generate_training_pattern(TrainingPattern.PRBS31)
        
        # Measure final margin
        margin = self._measure_verification_margin(pattern)
        
        passed = margin >= self.config.min_margin_ui
        
        if not passed:
            self._record_phase_result(PHYTrainingState.VERIFY, False,
                                      errors=["Verification failed: margin below threshold"])
        
        # Signal DFI
        if self.dfi:
            self.dfi.complete_training()
            self.status.dfi_training_complete = True
        
        return passed
    
    # === Measurement Helpers ===
    
    def _measure_wrlvl_margin(self, delay: int) -> float:
        """Measure write leveling margin for given delay"""
        # Simulate margin with noise
        noise = self._prng.uniform(-0.05, 0.05)
        margin = 0.5 - abs(delay - 32) / 64 + noise
        return float(np.clip(margin, 0.0, 1.0))
    
    def _measure_rdgd_margin(self, delay: int) -> float:
        """Measure read gate margin for given delay"""
        noise = self._prng.uniform(-0.05, 0.05)
        margin = 0.5 - abs(delay - 32) / 64 + noise
        return float(np.clip(margin, 0.0, 1.0))
    
    def _measure_vref_margin(self, vref: int) -> float:
        """Measure margin at given VREF setting"""
        noise = self._prng.uniform(-0.03, 0.03)
        margin = 0.5 - abs(vref - 32) / 128 + noise
        return float(np.clip(margin, 0.0, 1.0))
    
    def _measure_verification_margin(self, pattern: np.ndarray) -> float:
        """Measure margin from verification pattern"""
        # Simulate eye opening
        eye_height = self._prng.uniform(0.1, 0.5)
        return float(eye_height)
    
    def _calibrate_lane_rd(self, lane: int) -> int:
        """Calibrate read for a single lane"""
        # Simulated optimal delay
        base_delay = 32 + self._prng.randint(-4, 5)
        return int(np.clip(base_delay, 0, 63))
    
    def _calibrate_lane_wr(self, lane: int) -> int:
        """Calibrate write for a single lane"""
        base_delay = 32 + self._prng.randint(-4, 5)
        return int(np.clip(base_delay, 0, 63))
    
    def _calibrate_group_rd(self, group: int) -> int:
        """Calibrate read for a lane group"""
        return 32 + self._prng.randint(-4, 5)
    
    def _calibrate_group_wr(self, group: int) -> int:
        """Calibrate write for a lane group"""
        return 32 + self._prng.randint(-4, 5)
    
    def _optimize_rd_dq_delay(self, lane: int, base_delay: int) -> int:
        """Optimize read DQ delay for a lane"""
        best_delay = base_delay
        best_margin = self._measure_rdgd_margin(base_delay)
        
        for offset in range(-4, 5):
            delay = base_delay + offset
            if 0 <= delay <= 63:
                margin = self._measure_rdgd_margin(delay)
                if margin > best_margin:
                    best_margin = margin
                    best_delay = delay
        
        return best_delay
    
    def _optimize_margin(self, lane: int, base_delay: int) -> int:
        """Optimize margin for a lane"""
        return self._optimize_rd_dq_delay(lane, base_delay)
    
    # === Pattern Generation ===
    
    def _generate_training_pattern(self, pattern_type: TrainingPattern) -> np.ndarray:
        """Generate training pattern
        
        Args:
            pattern_type: Type of pattern to generate
            
        Returns:
            numpy array of pattern values (-1, +1)
        """
        if pattern_type == TrainingPattern.PRBS7:
            return self._generate_prbs(7)
        elif pattern_type == TrainingPattern.PRBS15:
            return self._generate_prbs(15)
        elif pattern_type == TrainingPattern.PRBS31:
            return self._generate_prbs(31)
        elif pattern_type == TrainingPattern.WALKING_1:
            return self._generate_walking_1()
        elif pattern_type == TrainingPattern.WALKING_0:
            return self._generate_walking_0()
        elif pattern_type == TrainingPattern.ALL_ONES:
            return np.ones(128)
        elif pattern_type == TrainingPattern.ALL_ZEROS:
            return -np.ones(128)
        elif pattern_type == TrainingPattern.ALTERNATING:
            return np.array([1 if i % 2 == 0 else -1 for i in range(128)])
        else:
            return self._generate_prbs(7)
    
    def _generate_prbs(self, bits: int) -> np.ndarray:
        """Generate PRBS pattern
        
        Args:
            bits: PRBS order (7, 15, 31)
            
        Returns:
            PRBS sequence
        """
        if bits == 7:
            poly = 0x09  # x^7 + x^6 + 1
            length = 127
        elif bits == 15:
            poly = 0x4001  # x^15 + x^14 + 1
            length = 32767
        elif bits == 31:
            poly = 0x80000001  # x^31 + x^28 + 1
            length = 2147483647
        else:
            poly = 0x09
            length = 127
        
        # Limit length for efficiency
        length = min(length, 1024)
        
        lfsr = (1 << (bits - 1))  # Initialize LFSR
        
        sequence = []
        for _ in range(length):
            sequence.append((lfsr >> (bits - 1)) & 1)
            
            # Calculate feedback
            fb = 0
            p = bits
            while p > 0:
                fb ^= (lfsr >> p) & 1
                p -= 1
            
            lfsr = ((lfsr << 1) | fb) & ((1 << bits) - 1)
        
        return np.array(sequence, dtype=np.float64) * 2 - 1  # Convert to +/- 1
    
    def _generate_walking_1(self) -> np.ndarray:
        """Generate walking 1 pattern"""
        pattern = []
        for i in range(64):
            val = 1 << i
            for bit in range(64):
                pattern.append(1 if (val >> bit) & 1 else -1)
        return np.array(pattern[:128])
    
    def _generate_walking_0(self) -> np.ndarray:
        """Generate walking 0 pattern"""
        return -self._generate_walking_1()
    
    # === Main Processing ===
    
    def process_cycle(self) -> bool:
        """Process one training cycle
        
        Main state machine advancement logic.
        
        Returns:
            True if training completed successfully
        """
        current = self.status.state
        
        if current == PHYTrainingState.IDLE:
            pass
        
        elif current == PHYTrainingState.INIT:
            # Initialize and move to first training phase
            self._init_coefficients()
            if self.TRAINING_SEQUENCE:
                self.status.state = self.TRAINING_SEQUENCE[0]
                self.status.state_enter_cycle = self._cycle
        
        elif current in self._phase_handlers:
            # Execute phase handler
            handler = self._phase_handlers[current]
            success = handler()
            
            if success:
                self._advance_state()
            else:
                # Phase failed
                if self.status.retry_count < self.config.retry_count:
                    self.status.retry_count += 1
                    self.status.state_enter_cycle = self._cycle
                else:
                    self.status.state = PHYTrainingState.FAIL
        
        elif current == PHYTrainingState.VERIFY:
            success = self._verify_training()
            if success:
                self.status.state = PHYTrainingState.COMPLETE
            else:
                self.status.state = PHYTrainingState.FAIL
        
        elif current == PHYTrainingState.COMPLETE:
            return True
        
        elif current == PHYTrainingState.FAIL:
            return False
        
        return False
    
    def get_training_results(self) -> Dict[str, Any]:
        """Get training results summary
        
        Returns:
            Dictionary with training results
        """
        return {
            'channel_id': self.channel_id,
            'passed': self.status.state == PHYTrainingState.COMPLETE,
            'state': self.status.state.name,
            'total_cycles': self.status.total_cycles,
            'retry_count': self.status.retry_count,
            'coefficients': {
                'tx_precursor': self.coefficients.tx_precursor,
                'tx_postcursor': self.coefficients.tx_postcursor,
                'tx_main_cursor': self.coefficients.tx_main_cursor,
                'rx_vref': self.coefficients.rx_vref,
                'rx_ctle_dc_gain': self.coefficients.rx_ctle_dc_gain,
                'rx_ctle_peaking': self.coefficients.rx_ctle_peaking,
                'dfe_taps': self.coefficients.dfe_taps,
            },
            'phase_results': {
                phase.name: {
                    'passed': result.passed,
                    'duration_cycles': result.duration_cycles,
                    'best_value': result.best_value,
                    'best_margin': result.best_margin,
                    'errors': result.errors,
                }
                for phase, result in self.status.phase_results.items()
            }
        }
    
    def get_coefficients(self) -> PHYTapCoefficients:
        """Get trained tap coefficients
        
        Returns:
            Trained coefficients
        """
        return self.coefficients


def create_phy_training_model(channel_id: int = 0,
                              config: Optional[PHYTrainingConfig] = None,
                              dfi_interface=None) -> PHYTrainingModel:
    """Factory function to create PHY training model
    
    Args:
        channel_id: Channel index
        config: Training configuration
        dfi_interface: DFI interface
        
    Returns:
        Configured PHY training model
    """
    return PHYTrainingModel(channel_id, config, dfi_interface)