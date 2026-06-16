"""
PHY Training State Machine for HBM4

Implements PHY initialization and training sequences according to
JEDEC JESD270-4A HBM4 specification.

Key features:
- PHY initialization state machine (PH-003)
- Training sequence state machine (PH-004)
- Read/Write DQS training
- Margin training (MT)
- VREF CA training
- DFI 5.1 interface integration

Reference:
- JEDEC JESD270-4A HBM4 specification
- Cadence HBM4E documentation
- DFI 5.1 specification
- Synopsys DesignWare HBM4/4E Controller IP
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque


# HBM4 VREF range constants (JEDEC JESD270-4A)
# VREF DAC is typically 6-bit (0-63)
VREF_DAC_BITS = 6
VREF_DAC_RANGE = (0, 63)  # 6-bit DAC range
VREF_CA_MIN = VREF_DAC_RANGE[0]  # 0
VREF_CA_MAX = VREF_DAC_RANGE[1]  # 63
VREF_DQ_MIN = VREF_DAC_RANGE[0]  # 0
VREF_DQ_MAX = VREF_DAC_RANGE[1]  # 63
# VREF as percentage of VDDQ (typical range)
VREF_CA_RANGE_PERCENT = (15.0, 45.0)  # 15-45% VDDQ
VREF_DQ_RANGE_PERCENT = (15.0, 45.0)   # 15-45% VDDQ


class PHYInitState(Enum):
    """PHY Initialization State Machine (PH-003)

    Tracks the initialization sequence from power-on to ready state.
    """
    INIT_IDLE = auto()           # Initial idle state
    INIT_START = auto()          # Start initialization sequence
    INIT_POWER_UP = auto()      # Power-up sequence
    INIT_RESET = auto()          # Reset phase
    INIT_CONFIG = auto()         # Configuration loading
    INIT_CALIBRATE = auto()      # Calibration phase
    INIT_TRAINING = auto()       # Training phase
    INIT_COMPLETE = auto()       # Initialization complete


class TrainingPhase(Enum):
    """Training Sequence State Machine (PH-004)

    Defines the stages of memory training for link optimization.
    """
    # Initial states
    TRAIN_IDLE = auto()                    # Not training
    TRAIN_START = auto()                   # Start training sequence
    TRAIN_INIT = auto()                    # Training initialization

    # DQS training phases
    TRAIN_RD_DQS = auto()                  # Read DQS training
    TRAIN_WR_LEVELING = auto()             # Write leveling

    # Margin training phases
    TRAIN_RD_MT = auto()                   # Read margin training
    TRAIN_WR_MT = auto()                   # Write margin training

    # DQ training phases
    TRAIN_RD_DQ = auto()                  # Read DQ training
    TRAIN_WR_DQ = auto()                   # Write DQ training

    # VREF training
    TRAIN_VREF_CA = auto()                 # VREF CA training
    TRAIN_VREF_DQ = auto()                 # VREF DQ training

    # Completion states
    TRAIN_VERIFY = auto()                  # Verify training results
    TRAIN_COMPLETE = auto()                # Training complete
    TRAIN_FAIL = auto()                     # Training failed


class TrainingResult(Enum):
    """Training result codes"""
    SUCCESS = auto()                       # Training passed
    FAIL_TIMEOUT = auto()                  # Timeout during training
    FAIL_MARGIN = auto()                   # Margin too small
    FAIL_VERIFY = auto()                   # Verification failed
    FAIL_PARAM = auto()                    # Invalid parameters


@dataclass
class TrainingParameters:
    """Training parameters for each training phase

    Stores delay values, margins, and configuration for each
    training stage.
    """
    # Read DQS training
    rd_dqs_delay: int = 0                  # Read DQS delay (taps)
    rd_dqs_gate_delay: int = 0             # Read DQS gate delay (taps)

    # Write leveling
    wr_level_delay: int = 0               # Write leveling delay (taps)
    wr_dq_delay: int = 0                   # Write DQ delay (taps)

    # Margin training
    rd_margin: float = 0.0                 # Read margin (UI)
    wr_margin: float = 0.0                 # Write margin (UI)
    rd_vref: int = 0                       # Read VREF setting
    wr_vref: int = 0                       # Write VREF setting

    # CA training
    ca_vref: int = 0                       # CA VREF setting
    ca_delay: int = 0                      # CA delay (taps)

    # Per-lane calibration data
    lane_delays: Dict[int, int] = field(default_factory=dict)  # Lane-specific delays

    # Training status
    training_passed: bool = False
    training_errors: List[str] = field(default_factory=list)


@dataclass
class DFI5TrainingControl:
    """DFI 5.1 Training Control Signals

    According to DFI 5.1 specification for PHY Independent Mode
    training control.
    """
    # Training request signals
    tra_req: bool = False                  # Training request
    tra_mode: int = 0                       # Training mode (0-7)
    tra_type: int = 0                       # Training type selector

    # Training acknowledge
    tra_ack: bool = False                  # Training acknowledge from PHY
    tra_complete: bool = False              # Training complete

    # Training status
    tra_error: bool = False                 # Training error
    tra_fail_code: int = 0                  # Failure code

    def encode_training_cmd(self, cmd: TrainingPhase) -> Tuple[bool, int, int]:
        """Encode training command for DFI interface

        Args:
            cmd: Training phase command

        Returns:
            Tuple of (tra_req, tra_mode, tra_type)
        """
        cmd_map = {
            TrainingPhase.TRAIN_RD_DQS: (True, 1, 0),
            TrainingPhase.TRAIN_WR_LEVELING: (True, 1, 1),
            TrainingPhase.TRAIN_RD_MT: (True, 2, 0),
            TrainingPhase.TRAIN_WR_MT: (True, 2, 1),
            TrainingPhase.TRAIN_RD_DQ: (True, 3, 0),
            TrainingPhase.TRAIN_WR_DQ: (True, 3, 1),
            TrainingPhase.TRAIN_VREF_CA: (True, 4, 0),
            TrainingPhase.TRAIN_VREF_DQ: (True, 4, 1),
        }
        return cmd_map.get(cmd, (False, 0, 0))


@dataclass
class TrainingStatus:
    """Current training status and statistics"""
    current_phase: TrainingPhase = TrainingPhase.TRAIN_IDLE
    phase_start_cycle: int = 0
    phase_timeout_cycles: int = 0
    total_training_cycles: int = 0
    retry_count: int = 0
    max_retries: int = 3
    results: Dict[TrainingPhase, TrainingResult] = field(default_factory=dict)
    params: TrainingParameters = field(default_factory=TrainingParameters)


@dataclass
class PHYInitStatus:
    """PHY Initialization Status"""
    state: PHYInitState = PHYInitState.INIT_IDLE
    state_enter_cycle: int = 0
    calibration_count: int = 0
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)


class PHYTrainingError(Exception):
    """Exception raised for PHY training errors"""
    pass


class PHYInitError(Exception):
    """Exception raised for PHY initialization errors"""
    pass


class PHYTrainingStateMachine:
    """PHY Training State Machine

    Implements the training sequence state machine (PH-004)
    for HBM4 PHY calibration.

    Training sequence follows JEDEC HBM4 specification:
    1. Initialize training
    2. Read DQS training
    3. Write leveling
    4. Read margin training
    5. Write margin training
    6. Read DQ training
    7. Write DQ training
    8. VREF CA training
    9. VREF DQ training
    10. Verify and complete
    """

    # Training phase sequence order
    TRAINING_SEQUENCE = [
        TrainingPhase.TRAIN_RD_DQS,
        TrainingPhase.TRAIN_WR_LEVELING,
        TrainingPhase.TRAIN_RD_MT,
        TrainingPhase.TRAIN_WR_MT,
        TrainingPhase.TRAIN_RD_DQ,
        TrainingPhase.TRAIN_WR_DQ,
        TrainingPhase.TRAIN_VREF_CA,
        TrainingPhase.TRAIN_VREF_DQ,
    ]

    # Default timeout per training phase (cycles @ 8 GT/s)
    # Based on JEDEC HBM4 training time budget
    DEFAULT_TIMEOUT_CYCLES = 50000  # ~5ms @ 10ns cycle

    def __init__(self, channel_id: int = 0,
                 dfi_interface=None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize PHY training state machine

        Args:
            channel_id: Channel index for this training instance
            dfi_interface: Optional DFI 5.1 interface for integration
            config: Optional configuration dictionary
        """
        self.channel_id = channel_id
        self.dfi = dfi_interface

        # Configuration
        self.config = config or {}
        self.timeout_cycles = self.config.get('timeout_cycles',
                                              self.DEFAULT_TIMEOUT_CYCLES)
        self.enable_retry = self.config.get('enable_retry', True)
        self.verify_results = self.config.get('verify_results', True)

        # State tracking
        self.status = TrainingStatus()
        self.params = TrainingParameters()
        self.dfi_control = DFI5TrainingControl()

        # Cycle counter
        self._cycle = 0

        # Training patterns (PRBS, walking 1/0, etc.)
        self._training_patterns = self._init_training_patterns()

        # Lane data for per-lane calibration
        self._lane_count = 64  # HBM4: 64 lanes per channel

    def _init_training_patterns(self) -> Dict[str, List[int]]:
        """Initialize training test patterns

        Returns:
            Dictionary of training patterns
        """
        # PRBS-7 pattern
        prbs7 = []
        lfsr = 0x7F
        for _ in range(128):
            prbs7.append((lfsr >> 6) & 1)
            new_bit = ((lfsr >> 6) ^ (lfsr >> 5)) & 1
            lfsr = ((lfsr << 1) | new_bit) & 0x7F

        # Walking 1 pattern
        walking_1 = [1 << i for i in range(64)]

        # Walking 0 pattern
        walking_0 = [~(1 << i) & 0xFFFF for i in range(64)]

        # All ones / zeros
        all_ones = [0xFFFF] * 64
        all_zeros = [0x0000] * 64

        return {
            'prbs7': prbs7,
            'walking_1': walking_1,
            'walking_0': walking_0,
            'all_ones': all_ones,
            'all_zeros': all_zeros,
        }

    @property
    def cycle(self) -> int:
        """Current simulation cycle"""
        return self._cycle

    def tick(self):
        """Advance training state machine by one cycle

        Call this once per cycle to update state machine.
        """
        self._cycle += 1

        # Update phase timer
        if self.status.current_phase != TrainingPhase.TRAIN_IDLE:
            elapsed = self._cycle - self.status.phase_start_cycle
            if elapsed > self.timeout_cycles:
                self._handle_phase_timeout()

    def start_training(self) -> bool:
        """Start training sequence

        Returns:
            True if training started successfully
        """
        if self.status.current_phase not in [TrainingPhase.TRAIN_IDLE,
                                              TrainingPhase.TRAIN_COMPLETE,
                                              TrainingPhase.TRAIN_FAIL]:
            return False

        # Reset training state
        self.status.current_phase = TrainingPhase.TRAIN_START
        self.status.phase_start_cycle = self._cycle
        self.status.total_training_cycles = 0
        self.status.retry_count = 0
        self.status.results.clear()
        self.params = TrainingParameters()

        # Signal DFI interface
        if self.dfi:
            self.dfi.start_training()

        return True

    def _execute_phase(self, phase: TrainingPhase) -> bool:
        """Execute a training phase

        Args:
            phase: Training phase to execute

        Returns:
            True if phase completed successfully
        """
        phase_handlers = {
            TrainingPhase.TRAIN_RD_DQS: self._train_rd_dqs,
            TrainingPhase.TRAIN_WR_LEVELING: self._train_wr_leveling,
            TrainingPhase.TRAIN_RD_MT: self._train_rd_mt,
            TrainingPhase.TRAIN_WR_MT: self._train_wr_mt,
            TrainingPhase.TRAIN_RD_DQ: self._train_rd_dq,
            TrainingPhase.TRAIN_WR_DQ: self._train_wr_dq,
            TrainingPhase.TRAIN_VREF_CA: self._train_vref_ca,
            TrainingPhase.TRAIN_VREF_DQ: self._train_vref_dq,
        }

        handler = phase_handlers.get(phase)
        if handler:
            return handler()

        return False

    def _train_rd_dqs(self) -> bool:
        """Execute Read DQS training

        Finds optimal DQS sampling position for reads.

        Returns:
            True if training passed
        """
        # Send training request via DFI
        tra_req, tra_mode, tra_type = self.dfi_control.encode_training_cmd(
            TrainingPhase.TRAIN_RD_DQS)
        self.dfi_control.tra_req = tra_req
        self.dfi_control.tra_mode = tra_mode
        self.dfi_control.tra_type = tra_type

        # Simulate DQS delay sweep
        best_delay = 0
        best_margin = 0.0

        for delay in range(64):  # 64 tap sweep
            margin = self._measure_rd_dqs_margin(delay)
            if margin > best_margin:
                best_margin = margin
                best_delay = delay

        self.params.rd_dqs_delay = best_delay

        # Verify result
        if best_margin < 0.1:  # Minimum margin threshold
            self.params.training_errors.append("RD DQS margin too small")
            return False

        return True

    def _train_wr_leveling(self) -> bool:
        """Execute Write Leveling training

        Aligns write DQS with data.

        Returns:
            True if training passed
        """
        tra_req, tra_mode, tra_type = self.dfi_control.encode_training_cmd(
            TrainingPhase.TRAIN_WR_LEVELING)
        self.dfi_control.tra_req = tra_req
        self.dfi_control.tra_mode = tra_mode
        self.dfi_control.tra_type = tra_type

        # Find optimal write leveling delay
        best_delay = 0
        best_margin = 0.0

        for delay in range(64):
            margin = self._measure_wr_level_margin(delay)
            if margin > best_margin:
                best_margin = margin
                best_delay = delay

        self.params.wr_level_delay = best_delay

        if best_margin < 0.1:
            self.params.training_errors.append("WR leveling margin too small")
            return False

        return True

    def _train_rd_mt(self) -> bool:
        """Execute Read Margin Training

        Optimizes read VREF for maximum margin.

        Returns:
            True if training passed
        """
        tra_req, tra_mode, tra_type = self.dfi_control.encode_training_cmd(
            TrainingPhase.TRAIN_RD_MT)
        self.dfi_control.tra_req = tra_req
        self.dfi_control.tra_mode = tra_mode
        self.dfi_control.tra_type = tra_type

        # VREF sweep for read
        best_vref = 32  # Center default
        best_margin = 0.0

        for vref in range(64):  # 6-bit VREF DAC
            margin = self._measure_rd_margin(vref)
            if margin > best_margin:
                best_margin = margin
                best_vref = vref

        self.params.rd_vref = best_vref
        self.params.rd_margin = best_margin

        if best_margin < 0.15:
            self.params.training_errors.append("RD margin training failed")
            return False

        return True

    def _train_wr_mt(self) -> bool:
        """Execute Write Margin Training

        Optimizes write VREF for maximum margin.

        Returns:
            True if training passed
        """
        tra_req, tra_mode, tra_type = self.dfi_control.encode_training_cmd(
            TrainingPhase.TRAIN_WR_MT)
        self.dfi_control.tra_req = tra_req
        self.dfi_control.tra_mode = tra_mode
        self.dfi_control.tra_type = tra_type

        best_vref = 32
        best_margin = 0.0

        for vref in range(64):
            margin = self._measure_wr_margin(vref)
            if margin > best_margin:
                best_margin = margin
                best_vref = vref

        self.params.wr_vref = best_vref
        self.params.wr_margin = best_margin

        if best_margin < 0.15:
            self.params.training_errors.append("WR margin training failed")
            return False

        return True

    def _train_rd_dq(self) -> bool:
        """Execute Read DQ Training

        Per-lane DQ delay calibration.

        Returns:
            True if training passed
        """
        tra_req, tra_mode, tra_type = self.dfi_control.encode_training_cmd(
            TrainingPhase.TRAIN_RD_DQ)
        self.dfi_control.tra_req = tra_req
        self.dfi_control.tra_mode = tra_mode
        self.dfi_control.tra_type = tra_type

        # Per-lane calibration
        for lane in range(self._lane_count):
            best_delay = self._calibrate_lane_rd(lane)
            self.params.lane_delays[lane] = best_delay

        return True

    def _train_wr_dq(self) -> bool:
        """Execute Write DQ Training

        Per-lane write DQ delay calibration.

        Returns:
            True if training passed
        """
        tra_req, tra_mode, tra_type = self.dfi_control.encode_training_cmd(
            TrainingPhase.TRAIN_WR_DQ)
        self.dfi_control.tra_req = tra_req
        self.dfi_control.tra_mode = tra_mode
        self.dfi_control.tra_type = tra_type

        for lane in range(self._lane_count):
            best_delay = self._calibrate_lane_wr(lane)
            self.params.lane_delays[lane + self._lane_count] = best_delay

        return True

    def _train_vref_ca(self) -> bool:
        """Execute VREF CA Training

        Trains CA interface VREF.

        Returns:
            True if training passed
        """
        tra_req, tra_mode, tra_type = self.dfi_control.encode_training_cmd(
            TrainingPhase.TRAIN_VREF_CA)
        self.dfi_control.tra_req = tra_req
        self.dfi_control.tra_mode = tra_mode
        self.dfi_control.tra_type = tra_type

        best_vref = 32
        best_margin = 0.0

        # Sweep VREF DAC range (0-63 for 6-bit DAC)
        for vref in range(VREF_CA_MIN, VREF_CA_MAX + 1):
            margin = self._measure_ca_vref_margin(vref)
            if margin > best_margin:
                best_margin = margin
                best_vref = vref

        # Validate VREF result
        self.params.ca_vref = best_vref
        if not self._validate_vref_result(best_vref, "CA"):
            return False

        if best_margin < 0.1:
            self.params.training_errors.append("VREF CA training failed")
            return False

        return True

    def _train_vref_dq(self) -> bool:
        """Execute VREF DQ Training

        Trains DQ interface VREF.

        Returns:
            True if training passed
        """
        tra_req, tra_mode, tra_type = self.dfi_control.encode_training_cmd(
            TrainingPhase.TRAIN_VREF_DQ)
        self.dfi_control.tra_req = tra_req
        self.dfi_control.tra_mode = tra_mode
        self.dfi_control.tra_type = tra_type

        # DQ VREF calibrated in margin training - validate stored results
        if hasattr(self.params, 'rd_vref') and not self._validate_vref_result(self.params.rd_vref, "DQ"):
            return False
        if hasattr(self.params, 'wr_vref') and not self._validate_vref_result(self.params.wr_vref, "DQ"):
            return False

        return True

    # === Measurement helpers ===

    def _validate_vref(self, vref: int, vref_type: str = "DQ") -> bool:
        """Validate VREF setting is within valid range

        Args:
            vref: VREF DAC setting to validate
            vref_type: Type of VREF ("CA" or "DQ")

        Returns:
            True if VREF is valid

        Raises:
            ValueError: If VREF is out of range
        """
        if vref_type == "CA":
            vref_min = VREF_CA_MIN
            vref_max = VREF_CA_MAX
        else:
            vref_min = VREF_DQ_MIN
            vref_max = VREF_DQ_MAX

        if not (vref_min <= vref <= vref_max):
            raise ValueError(
                f"Invalid {vref_type} VREF value {vref}: "
                f"must be in range [{vref_min}, {vref_max}]"
            )
        return True

    def _validate_vref_result(self, vref: int, vref_type: str = "DQ") -> bool:
        """Validate VREF training result

        Args:
            vref: VREF DAC setting from training
            vref_type: Type of VREF ("CA" or "DQ")

        Returns:
            True if VREF is within valid range
        """
        try:
            self._validate_vref(vref, vref_type)
        except ValueError:
            self.params.training_errors.append(
                f"{vref_type} VREF training resulted in invalid value: {vref}"
            )
            return False
        return True

    def _measure_rd_dqs_margin(self, delay: int) -> float:
        """Measure read DQS margin for given delay

        Args:
            delay: DQS delay tap value

        Returns:
            Margin as fraction of UI (0.0 to 1.0)
        """
        # Simulate margin measurement
        # Real implementation would send patterns and measure errors
        import random
        noise = random.uniform(-0.05, 0.05)
        margin = 0.5 - abs(delay - 32) / 64 + noise
        return max(0.0, min(1.0, margin))

    def _measure_wr_level_margin(self, delay: int) -> float:
        """Measure write leveling margin

        Args:
            delay: Write leveling delay tap

        Returns:
            Margin as fraction of UI
        """
        import random
        noise = random.uniform(-0.05, 0.05)
        margin = 0.5 - abs(delay - 32) / 64 + noise
        return max(0.0, min(1.0, margin))

    def _measure_rd_margin(self, vref: int) -> float:
        """Measure read margin at given VREF

        Args:
            vref: VREF setting

        Returns:
            Margin as fraction of UI
        """
        import random
        # VREF centered around 32
        noise = random.uniform(-0.03, 0.03)
        margin = 0.5 - abs(vref - 32) / 128 + noise
        return max(0.0, min(1.0, margin))

    def _measure_wr_margin(self, vref: int) -> float:
        """Measure write margin at given VREF

        Args:
            vref: VREF setting

        Returns:
            Margin as fraction of UI
        """
        import random
        noise = random.uniform(-0.03, 0.03)
        margin = 0.5 - abs(vref - 32) / 128 + noise
        return max(0.0, min(1.0, margin))

    def _measure_ca_vref_margin(self, vref: int) -> float:
        """Measure CA VREF margin

        Args:
            vref: CA VREF setting

        Returns:
            Margin as fraction of UI
        """
        import random
        noise = random.uniform(-0.04, 0.04)
        margin = 0.5 - abs(vref - 32) / 128 + noise
        return max(0.0, min(1.0, margin))

    def _calibrate_lane_rd(self, lane: int) -> int:
        """Calibrate read delay for a single lane

        Args:
            lane: Lane index

        Returns:
            Optimal delay tap value
        """
        import random
        # Find best delay for this lane
        best_delay = random.randint(28, 36)  # Simulated optimal
        return best_delay

    def _calibrate_lane_wr(self, lane: int) -> int:
        """Calibrate write delay for a single lane

        Args:
            lane: Lane index

        Returns:
            Optimal delay tap value
        """
        import random
        best_delay = random.randint(28, 36)
        return best_delay

    def _handle_phase_timeout(self):
        """Handle training phase timeout"""
        self.status.results[self.status.current_phase] = TrainingResult.FAIL_TIMEOUT
        self.params.training_errors.append(
            f"Timeout in phase {self.status.current_phase.name}"
        )

        if self.enable_retry and self.status.retry_count < self.status.max_retries:
            self.status.retry_count += 1
            self.status.current_phase = TrainingPhase.TRAIN_INIT
        else:
            self.status.current_phase = TrainingPhase.TRAIN_FAIL

    def _advance_to_next_phase(self):
        """Advance to next training phase in sequence"""
        try:
            current_idx = self.TRAINING_SEQUENCE.index(self.status.current_phase)
            if current_idx < len(self.TRAINING_SEQUENCE) - 1:
                next_phase = self.TRAINING_SEQUENCE[current_idx + 1]
                self.status.current_phase = next_phase
                self.status.phase_start_cycle = self._cycle
            else:
                # Training sequence complete
                self.status.current_phase = TrainingPhase.TRAIN_VERIFY
        except ValueError:
            # Not in sequence, move to first phase
            if self.TRAINING_SEQUENCE:
                self.status.current_phase = self.TRAINING_SEQUENCE[0]
                self.status.phase_start_cycle = self._cycle

    def process_training_cycle(self) -> bool:
        """Process one training cycle

        Main state machine advancement logic.

        Returns:
            True if training completed successfully
        """
        current = self.status.current_phase

        if current == TrainingPhase.TRAIN_IDLE:
            # Waiting for training start
            pass

        elif current == TrainingPhase.TRAIN_START:
            # Initialize training
            self.status.current_phase = TrainingPhase.TRAIN_INIT
            self.status.phase_start_cycle = self._cycle

        elif current == TrainingPhase.TRAIN_INIT:
            # Move to first training phase
            if self.TRAINING_SEQUENCE:
                self.status.current_phase = self.TRAINING_SEQUENCE[0]
                self.status.phase_start_cycle = self._cycle

        elif current in self.TRAINING_SEQUENCE:
            # Execute current phase
            success = self._execute_phase(current)
            if success:
                self.status.results[current] = TrainingResult.SUCCESS
                self._advance_to_next_phase()
            else:
                self.status.results[current] = TrainingResult.FAIL_MARGIN
                if self.enable_retry and self.status.retry_count < self.status.max_retries:
                    self.status.retry_count += 1
                    # Retry current phase
                    self.status.phase_start_cycle = self._cycle
                else:
                    self.status.current_phase = TrainingPhase.TRAIN_FAIL

        elif current == TrainingPhase.TRAIN_VERIFY:
            # Verify training results
            if self.verify_results:
                verified = self._verify_training_results()
                if verified:
                    self.status.current_phase = TrainingPhase.TRAIN_COMPLETE
                    self.params.training_passed = True
                else:
                    self.status.results[current] = TrainingResult.FAIL_VERIFY
                    self.status.current_phase = TrainingPhase.TRAIN_FAIL
            else:
                self.status.current_phase = TrainingPhase.TRAIN_COMPLETE
                self.params.training_passed = True

        elif current == TrainingPhase.TRAIN_COMPLETE:
            # Training complete
            if self.dfi:
                self.dfi.complete_training()
            return True

        elif current == TrainingPhase.TRAIN_FAIL:
            # Training failed
            return False

        return False

    def _verify_training_results(self) -> bool:
        """Verify all training results

        Returns:
            True if all results meet requirements
        """
        # Check that all phases passed
        for phase in self.TRAINING_SEQUENCE:
            result = self.status.results.get(phase)
            if result != TrainingResult.SUCCESS:
                return False

        # Check parameter validity
        if self.params.rd_vref < 0 or self.params.rd_vref > 63:
            return False
        if self.params.wr_vref < 0 or self.params.wr_vref > 63:
            return False

        # Check margins
        if self.params.rd_margin < 0.1 or self.params.wr_margin < 0.1:
            return False

        return True

    def get_training_results(self) -> Dict[str, Any]:
        """Get training results summary

        Returns:
            Dictionary with training results
        """
        return {
            'channel_id': self.channel_id,
            'passed': self.params.training_passed,
            'current_phase': self.status.current_phase.name,
            'total_cycles': self._cycle,
            'retry_count': self.status.retry_count,
            'results': {p.name: r.name for p, r in self.status.results.items()},
            'parameters': {
                'rd_dqs_delay': self.params.rd_dqs_delay,
                'wr_level_delay': self.params.wr_level_delay,
                'rd_vref': self.params.rd_vref,
                'wr_vref': self.params.wr_vref,
                'ca_vref': self.params.ca_vref,
                'rd_margin': self.params.rd_margin,
                'wr_margin': self.params.wr_margin,
            },
            'errors': self.params.training_errors,
        }

    def is_training_complete(self) -> bool:
        """Check if training is complete

        Returns:
            True if training reached terminal state
        """
        return self.status.current_phase in [TrainingPhase.TRAIN_COMPLETE,
                                              TrainingPhase.TRAIN_FAIL]

    def is_training_passed(self) -> bool:
        """Check if training passed

        Returns:
            True if training completed successfully
        """
        return self.status.current_phase == TrainingPhase.TRAIN_COMPLETE


class PHYInitializationStateMachine:
    """PHY Initialization State Machine (PH-003)

    Implements the initialization sequence from power-on
    to ready state for HBM4 PHY.

    Sequence:
    1. Power-up
    2. Reset
    3. Configuration
    4. Calibration
    5. Training
    6. Complete
    """

    # State transition timeout (cycles)
    STATE_TIMEOUT = 100000  # 100ms @ 1ns cycle

    def __init__(self, training_sm: Optional[PHYTrainingStateMachine] = None,
                 dfi_interface=None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize PHY initialization state machine

        Args:
            training_sm: Training state machine instance
            dfi_interface: Optional DFI 5.1 interface
            config: Optional configuration dictionary
        """
        self.training_sm = training_sm
        self.dfi = dfi_interface
        self.config = config or {}

        # Status tracking
        self.status = PHYInitStatus()
        self.training_sm_ref = training_sm

        # Configuration loaded from config
        self._config_loaded = False
        self._calibration_data: Dict[str, Any] = {}

        # Cycle counter
        self._cycle = 0

    @property
    def cycle(self) -> int:
        """Current simulation cycle"""
        return self._cycle

    @property
    def is_initialized(self) -> bool:
        """Check if initialization is complete"""
        return self.status.state == PHYInitState.INIT_COMPLETE

    @property
    def is_ready(self) -> bool:
        """Check if PHY is ready for operation"""
        return (self.is_initialized and
                (self.training_sm is None or self.training_sm.is_training_passed()))

    def tick(self):
        """Advance initialization state machine by one cycle"""
        self._cycle += 1

        # Update training state machine if exists
        if self.training_sm:
            self.training_sm.tick()

        # Check for state timeout
        elapsed = self._cycle - self.status.state_enter_cycle
        if elapsed > self.STATE_TIMEOUT:
            self._handle_state_timeout()

    def _handle_state_timeout(self):
        """Handle state timeout"""
        self.status.error_count += 1
        self.status.warnings.append(
            f"Timeout in state {self.status.state.name} at cycle {self._cycle}"
        )

    def start_initialization(self):
        """Start PHY initialization sequence"""
        if self.status.state != PHYInitState.INIT_IDLE:
            return

        self.status.state = PHYInitState.INIT_START
        self.status.state_enter_cycle = self._cycle
        self.status.error_count = 0
        self.status.warnings.clear()

    def process_init_cycle(self):
        """Process one initialization cycle

        Main state machine advancement logic.
        """
        current = self.status.state

        if current == PHYInitState.INIT_IDLE:
            # Waiting for initialization start
            pass

        elif current == PHYInitState.INIT_START:
            # Move to power-up
            self.status.state = PHYInitState.INIT_POWER_UP
            self.status.state_enter_cycle = self._cycle

        elif current == PHYInitState.INIT_POWER_UP:
            # Simulate power-up sequence
            # In real hardware, this involves voltage ramps, etc.
            if self._cycle - self.status.state_enter_cycle > 100:
                self.status.state = PHYInitState.INIT_RESET
                self.status.state_enter_cycle = self._cycle

        elif current == PHYInitState.INIT_RESET:
            # Simulate reset sequence
            if self._cycle - self.status.state_enter_cycle > 50:
                self.status.state = PHYInitState.INIT_CONFIG
                self.status.state_enter_cycle = self._cycle

        elif current == PHYInitState.INIT_CONFIG:
            # Load configuration
            if not self._config_loaded:
                self._load_configuration()
            if self._cycle - self.status.state_enter_cycle > 20:
                self.status.state = PHYInitState.INIT_CALIBRATE
                self.status.state_enter_cycle = self._cycle

        elif current == PHYInitState.INIT_CALIBRATE:
            # Run calibration
            if self._cycle - self.status.state_enter_cycle > 1000:
                self.status.calibration_count += 1
                self.status.state = PHYInitState.INIT_TRAINING
                self.status.state_enter_cycle = self._cycle

                # Start training if state machine exists
                if self.training_sm:
                    self.training_sm.start_training()

        elif current == PHYInitState.INIT_TRAINING:
            # Run training sequence
            if self.training_sm:
                complete = self.training_sm.process_training_cycle()
                if complete:
                    self.status.state = PHYInitState.INIT_COMPLETE
                    self.status.state_enter_cycle = self._cycle
            else:
                # No training, skip to complete
                self.status.state = PHYInitState.INIT_COMPLETE
                self.status.state_enter_cycle = self._cycle

        elif current == PHYInitState.INIT_COMPLETE:
            # Initialization complete
            pass

    def _load_configuration(self):
        """Load PHY configuration"""
        # Load default calibration data
        self._calibration_data = {
            'rd_vref': 32,
            'wr_vref': 32,
            'ca_vref': 32,
            'dqs_delay': 0,
            'dq_delays': {},
        }

        # Apply any config overrides
        if 'default_rd_vref' in self.config:
            self._calibration_data['rd_vref'] = self.config['default_rd_vref']
        if 'default_wr_vref' in self.config:
            self._calibration_data['wr_vref'] = self.config['default_wr_vref']

        self._config_loaded = True

    def get_initialization_status(self) -> Dict[str, Any]:
        """Get initialization status

        Returns:
            Dictionary with status information
        """
        status = {
            'state': self.status.state.name,
            'cycle': self._cycle,
            'calibration_count': self.status.calibration_count,
            'error_count': self.status.error_count,
            'warnings': list(self.status.warnings),
            'initialized': self.is_initialized,
            'ready': self.is_ready,
        }

        if self.training_sm:
            status['training'] = self.training_sm.get_training_results()

        return status

    def get_calibration_data(self) -> Dict[str, Any]:
        """Get calibration data

        Returns:
            Dictionary with calibration values
        """
        data = dict(self._calibration_data)

        # Always include default calibration data if config was loaded
        if not data and self._config_loaded:
            data = {
                'rd_vref': 32,
                'wr_vref': 32,
                'ca_vref': 32,
                'dqs_delay': 0,
                'dq_delays': {},
            }
            if 'default_rd_vref' in self.config:
                data['rd_vref'] = self.config['default_rd_vref']
            if 'default_wr_vref' in self.config:
                data['wr_vref'] = self.config['default_wr_vref']

        # Merge with training results if training has completed
        if self.training_sm and self.training_sm.params.training_passed:
            params = self.training_sm.params
            data.update({
                'rd_vref': params.rd_vref,
                'wr_vref': params.wr_vref,
                'ca_vref': params.ca_vref,
                'rd_dqs_delay': params.rd_dqs_delay,
                'wr_level_delay': params.wr_level_delay,
                'rd_margin': params.rd_margin,
                'wr_margin': params.wr_margin,
            })

        return data


class HBM4PHYManager:
    """HBM4 PHY Manager

    Top-level manager that coordinates PHY initialization
    and training across all channels.

    This class provides the unified interface for PHY control
    and integrates with the DFI 5.1 interface.
    """

    def __init__(self, num_channels: int = 32,
                 dfi_interface=None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize HBM4 PHY Manager

        Args:
            num_channels: Number of HBM4 channels
            dfi_interface: Optional DFI 5.1 interface
            config: Optional configuration dictionary
        """
        self.num_channels = num_channels
        self.dfi = dfi_interface
        self.config = config or {}

        # Create initialization state machines per channel
        self._init_machines: List[PHYInitializationStateMachine] = []
        self._training_machines: List[PHYTrainingStateMachine] = []

        for ch in range(num_channels):
            # Create training state machine
            training_sm = PHYTrainingStateMachine(
                channel_id=ch,
                dfi_interface=dfi_interface,
                config=self.config.get('training', {})
            )
            self._training_machines.append(training_sm)

            # Create initialization state machine
            init_sm = PHYInitializationStateMachine(
                training_sm=training_sm,
                dfi_interface=dfi_interface,
                config=self.config
            )
            self._init_machines.append(init_sm)

        # Global state
        self._global_cycle = 0
        self._all_initialized = False

    @property
    def cycle(self) -> int:
        """Current global cycle"""
        return self._global_cycle

    def tick(self):
        """Advance all PHY state machines by one cycle"""
        self._global_cycle += 1

        for init_sm in self._init_machines:
            init_sm.tick()

        # Check if all initialized
        self._all_initialized = all(sm.is_initialized for sm in self._init_machines)

    def start_initialization(self):
        """Start initialization on all channels"""
        for init_sm in self._init_machines:
            init_sm.start_initialization()

    def process_cycles(self, num_cycles: int):
        """Process multiple initialization cycles

        Args:
            num_cycles: Number of cycles to process
        """
        for _ in range(num_cycles):
            # Process each channel
            for init_sm in self._init_machines:
                init_sm.process_init_cycle()

            self.tick()

    def wait_for_initialization(self, max_cycles: int = 100000) -> bool:
        """Wait for initialization to complete

        Args:
            max_cycles: Maximum cycles to wait

        Returns:
            True if initialization completed successfully
        """
        for _ in range(max_cycles):
            if self._all_initialized:
                return True
            self.process_cycles(1)
        return False

    def get_channel_status(self, channel: int) -> Dict[str, Any]:
        """Get status for a specific channel

        Args:
            channel: Channel index

        Returns:
            Channel status dictionary
        """
        if channel < 0 or channel >= self.num_channels:
            return {'error': 'Invalid channel index'}

        return self._init_machines[channel].get_initialization_status()

    def get_all_channel_status(self) -> List[Dict[str, Any]]:
        """Get status for all channels

        Returns:
            List of channel status dictionaries
        """
        return [sm.get_initialization_status() for sm in self._init_machines]

    def is_ready(self) -> bool:
        """Check if all PHYs are ready

        Returns:
            True if all channels are ready
        """
        return self._all_initialized and all(sm.is_ready for sm in self._init_machines)

    def get_aggregate_calibration_data(self) -> Dict[str, Any]:
        """Get calibration data aggregated across all channels

        Returns:
            Dictionary with calibration data
        """
        all_data = [sm.get_calibration_data() for sm in self._init_machines]

        # Aggregate statistics
        return {
            'num_channels': self.num_channels,
            'num_initialized': sum(1 for sm in self._init_machines if sm.is_initialized),
            'num_ready': sum(1 for sm in self._init_machines if sm.is_ready),
            'channel_data': all_data,
        }