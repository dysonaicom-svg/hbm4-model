"""
Tap Coefficient Management for HBM PHY

Manages TX pre-cursor/post-cursor coefficients, RX CTLE/DFE tap
coefficients, and coefficient optimization during training.

Reference:
- JEDEC JESD270-4A HBM4 specification
- DFI 5.0/5.1 specification
- IEEE 802.3 for equalizer specifications
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np


class CoefficientType(Enum):
    """Types of tap coefficients"""
    TX_PRE_CURSOR = "tx_pre_cursor"
    TX_POST_CURSOR = "tx_post_cursor"
    TX_MAIN_CURSOR = "tx_main_cursor"
    RX_CTLE_DC_GAIN = "rx_ctle_dc_gain"
    RX_CTLE_PEAKING = "rx_ctle_peaking"
    RX_VREF = "rx_vref"
    DFE_TAP = "dfe_tap"


@dataclass
class TXCoefficients:
    """TX (Transmitter) Equalizer Coefficients
    
    Implements FIR-based TX equalization with pre-cursor,
    main cursor, and post-cursor taps.
    """
    # Pre-cursor taps (before main cursor)
    pre_cursor: List[float] = field(default_factory=lambda: [0.0, 0.0])
    
    # Main cursor (data symbol)
    main_cursor: float = 1.0
    
    # Post-cursor taps (after main cursor)
    post_cursor: List[float] = field(default_factory=lambda: [0.0, 0.0])
    
    # Constraints
    max_pre_taps: int = 4
    max_post_taps: int = 4
    max_tap_magnitude: float = 0.5
    
    # Resolution
    tap_resolution_bits: int = 5  # 5-bit tap resolution
    
    @property
    def num_pre_taps(self) -> int:
        return len(self.pre_cursor)
    
    @property
    def num_post_taps(self) -> int:
        return len(self.post_cursor)
    
    @property
    def total_taps(self) -> int:
        return self.num_pre_taps + 1 + self.num_post_taps
    
    def get_all_taps(self) -> List[float]:
        """Get all taps as single list"""
        return self.pre_cursor + [self.main_cursor] + self.post_cursor
    
    def set_taps(self, pre_taps: List[float], post_taps: List[float],
                 main: float = 1.0):
        """Set all tap values with saturation
        
        Args:
            pre_taps: Pre-cursor tap values
            post_taps: Post-cursor tap values
            main: Main cursor value
        """
        # Pad or truncate pre-taps
        self.pre_cursor = self._clamp_taps(pre_taps[:self.max_pre_taps],
                                          len(self.pre_cursor))
        
        # Pad or truncate post-taps
        self.post_cursor = self._clamp_taps(post_taps[:self.max_post_taps],
                                            len(self.post_cursor))
        
        # Set main cursor
        self.main_cursor = main
    
    def _clamp_taps(self, taps: List[float], target_len: int) -> List[float]:
        """Clamp tap values and pad to target length"""
        clamped = [np.clip(t, -self.max_tap_magnitude, self.max_tap_magnitude)
                   for t in taps]
        
        # Pad to target length
        while len(clamped) < target_len:
            clamped.append(0.0)
        
        return clamped[:target_len]
    
    def normalize(self) -> float:
        """Normalize taps for unity DC gain
        
        Returns:
            Scale factor applied
        """
        total = sum(self.pre_cursor) + self.main_cursor + sum(self.post_cursor)
        if abs(total) < 1e-9:
            return 1.0
        
        scale = self.main_cursor / total
        self.pre_cursor = [t * scale for t in self.pre_cursor]
        self.post_cursor = [t * scale for t in self.post_cursor]
        
        return scale
    
    def calculate_boost_db(self) -> float:
        """Calculate high-frequency boost in dB
        
        Returns:
            Boost at Nyquist vs DC
        """
        # Simplified boost calculation
        pre_sum = sum(abs(t) for t in self.pre_cursor)
        post_sum = sum(abs(t) for t in self.post_cursor)
        
        # Boost proportional to tap magnitudes
        boost = 20 * np.log10(1 + pre_sum + post_sum)
        return float(boost)
    
    def to_fir_coeffs(self) -> np.ndarray:
        """Convert to FIR filter coefficients
        
        Returns:
            FIR coefficient array
        """
        return np.array(self.get_all_taps())
    
    def from_fir_coeffs(self, coeffs: np.ndarray):
        """Load from FIR coefficients
        
        Args:
            coeffs: FIR coefficient array
        """
        n = len(coeffs)
        mid = n // 2
        
        self.pre_cursor = list(coeffs[:mid])
        self.main_cursor = float(coeffs[mid])
        self.post_cursor = list(coeffs[mid + 1:])
    
    def copy(self) -> 'TXCoefficients':
        """Create a copy of these coefficients"""
        return TXCoefficients(
            pre_cursor=list(self.pre_cursor),
            main_cursor=self.main_cursor,
            post_cursor=list(self.post_cursor),
            max_pre_taps=self.max_pre_taps,
            max_post_taps=self.max_post_taps,
            max_tap_magnitude=self.max_tap_magnitude,
            tap_resolution_bits=self.tap_resolution_bits
        )


@dataclass
class RXCoefficients:
    """RX (Receiver) Equalizer Coefficients
    
    Manages RX CTLE (Continuous Time Linear Equalizer) and
    DFE (Decision Feedback Equalizer) coefficients.
    """
    # CTLE parameters
    ctle_dc_gain_db: float = 0.0  # DC gain in dB
    ctle_peaking_db: float = 3.0  # Peaking gain in dB
    ctle_zero_idx: int = 1       # Zero frequency index
    ctle_pole_idx: int = 1        # Pole frequency index
    
    # VREF setting (6-bit DAC, 0-63)
    vref: int = 32
    
    # CTLE configuration ranges
    dc_gain_range_db: Tuple[float, float] = (-6.0, 6.0)
    peaking_range_db: Tuple[float, float] = (0.0, 12.0)
    zero_options: List[float] = field(default_factory=lambda: [4e9, 8e9, 12e9])
    pole_options: List[float] = field(default_factory=lambda: [8e9, 16e9, 24e9])
    
    # DFE taps
    dfe_taps: List[float] = field(default_factory=lambda: [0.0] * 5)
    dfe_max_tap_magnitude: float = 0.3
    
    @property
    def ctle_zero_freq(self) -> float:
        return self.zero_options[self.ctle_zero_idx] if self.ctle_zero_idx < len(self.zero_options) else self.zero_options[0]
    
    @property
    def ctle_pole_freq(self) -> float:
        return self.pole_options[self.ctle_pole_idx] if self.ctle_pole_idx < len(self.pole_options) else self.pole_options[0]
    
    def set_ctle(self, dc_gain: float, peaking: float):
        """Set CTLE parameters
        
        Args:
            dc_gain: DC gain in dB
            peaking: Peaking in dB
        """
        self.ctle_dc_gain_db = np.clip(dc_gain, *self.dc_gain_range_db)
        self.ctle_peaking_db = np.clip(peaking, *self.peaking_range_db)
    
    def set_vref(self, vref_value: int):
        """Set VREF value
        
        Args:
            vref_value: VREF DAC value (0-63)
        """
        self.vref = int(np.clip(vref_value, 0, 63))
    
    def set_dfe_taps(self, taps: List[float]):
        """Set DFE tap values
        
        Args:
            taps: DFE tap values
        """
        self.dfe_taps = [np.clip(t, -self.dfe_max_tap_magnitude, self.dfe_max_tap_magnitude)
                        for t in taps]
    
    def update_dfe_tap(self, index: int, value: float):
        """Update a single DFE tap
        
        Args:
            index: Tap index
            value: New value
        """
        if 0 <= index < len(self.dfe_taps):
            self.dfe_taps[index] = np.clip(value, -self.dfe_max_tap_magnitude,
                                          self.dfe_max_tap_magnitude)
    
    def calculate_ctle_transfer(self, frequency: np.ndarray) -> np.ndarray:
        """Calculate CTLE transfer function
        
        Args:
            frequency: Frequency vector in Hz
            
        Returns:
            Complex transfer function
        """
        s = 2j * np.pi * frequency
        
        w_z = 2 * np.pi * self.ctle_zero_freq
        w_p = 2 * np.pi * self.ctle_pole_freq
        
        H_dc = 10 ** (self.ctle_dc_gain_db / 20)
        
        with np.errstate(divide='ignore'):
            H = H_dc * (s / w_z + 1) / (s / w_p + 1)
        
        return H
    
    def copy(self) -> 'RXCoefficients':
        """Create a copy of these coefficients"""
        return RXCoefficients(
            ctle_dc_gain_db=self.ctle_dc_gain_db,
            ctle_peaking_db=self.ctle_peaking_db,
            ctle_zero_idx=self.ctle_zero_idx,
            ctle_pole_idx=self.ctle_pole_idx,
            vref=self.vref,
            dc_gain_range_db=self.dc_gain_range_db,
            peaking_range_db=self.peaking_range_db,
            zero_options=list(self.zero_options),
            pole_options=list(self.pole_options),
            dfe_taps=list(self.dfe_taps),
            dfe_max_tap_magnitude=self.dfe_max_tap_magnitude
        )


@dataclass
class LaneCoefficients:
    """Per-lane coefficient storage
    
    Stores calibrated values for each lane.
    """
    num_lanes: int = 64
    
    # Per-lane read delays
    rd_delays: Dict[int, int] = field(default_factory=dict)
    
    # Per-lane write delays
    wr_delays: Dict[int, int] = field(default_factory=dict)
    
    # Per-lane DQ delays
    rd_dq_delays: Dict[int, int] = field(default_factory=dict)
    wr_dq_delays: Dict[int, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default values"""
        for lane in range(self.num_lanes):
            if lane not in self.rd_delays:
                self.rd_delays[lane] = 0
            if lane not in self.wr_delays:
                self.wr_delays[lane] = 0
            if lane not in self.rd_dq_delays:
                self.rd_dq_delays[lane] = 0
            if lane not in self.wr_dq_delays:
                self.wr_dq_delays[lane] = 0
    
    def set_rd_delay(self, lane: int, delay: int):
        """Set read delay for a lane"""
        self.rd_delays[lane] = int(np.clip(delay, 0, 63))
    
    def set_wr_delay(self, lane: int, delay: int):
        """Set write delay for a lane"""
        self.wr_delays[lane] = int(np.clip(delay, 0, 63))
    
    def set_rd_dq_delay(self, lane: int, delay: int):
        """Set read DQ delay for a lane"""
        self.rd_dq_delays[lane] = int(np.clip(delay, 0, 63))
    
    def set_wr_dq_delay(self, lane: int, delay: int):
        """Set write DQ delay for a lane"""
        self.wr_dq_delays[lane] = int(np.clip(delay, 0, 63))
    
    def copy(self) -> 'LaneCoefficients':
        """Create a copy"""
        return LaneCoefficients(
            num_lanes=self.num_lanes,
            rd_delays=dict(self.rd_delays),
            wr_delays=dict(self.wr_delays),
            rd_dq_delays=dict(self.rd_dq_delays),
            wr_dq_delays=dict(self.wr_dq_delays)
        )


@dataclass
class CompleteTapCoefficients:
    """Complete set of tap coefficients for HBM PHY
    
    Aggregates TX, RX, and lane-specific coefficients.
    """
    tx: TXCoefficients = field(default_factory=TXCoefficients)
    rx: RXCoefficients = field(default_factory=RXCoefficients)
    lane: LaneCoefficients = field(default_factory=LaneCoefficients)
    
    # Metadata
    channel_id: int = 0
    training_complete: bool = False
    training_timestamp: int = 0
    
    def is_valid(self) -> bool:
        """Check if coefficients are valid"""
        if self.rx.vref < 0 or self.rx.vref > 63:
            return False
        if abs(self.tx.main_cursor) < 1e-9:
            return False
        return True
    
    def copy(self) -> 'CompleteTapCoefficients':
        """Create a deep copy"""
        return CompleteTapCoefficients(
            tx=self.tx.copy(),
            rx=self.rx.copy(),
            lane=self.lane.copy(),
            channel_id=self.channel_id,
            training_complete=self.training_complete,
            training_timestamp=self.training_timestamp
        )


class CoefficientOptimizer:
    """Coefficient optimization during training
    
    Implements LMS (Least Mean Squares) and other algorithms
    for adaptive coefficient optimization.
    """
    
    def __init__(self, coefficients: Optional[CompleteTapCoefficients] = None):
        """Initialize optimizer
        
        Args:
            coefficients: Initial coefficients
        """
        self.coeffs = coefficients or CompleteTapCoefficients()
        
        # Optimization parameters
        self.lms_mu = 0.01  # LMS step size
        self.max_iterations = 1000
        self.convergence_threshold = 0.001
        
        # History
        self._error_history: List[float] = []
        self._coeff_history: List[CompleteTapCoefficients] = []
    
    def optimize_tx_taps(self, target_response: np.ndarray,
                         channel_response: np.ndarray) -> TXCoefficients:
        """Optimize TX taps for target frequency response
        
        Args:
            target_response: Target frequency response
            channel_response: Channel frequency response
            
        Returns:
            Optimized TX coefficients
        """
        tx = self.coeffs.tx
        n_taps = tx.total_taps
        
        # Initialize taps
        taps = np.array(tx.get_all_taps())
        
        for iteration in range(self.max_iterations):
            # Calculate current response
            current_response = np.convolve(taps, channel_response, mode='full')
            
            # Truncate to match length
            if len(current_response) > len(target_response):
                current_response = current_response[:len(target_response)]
            
            # Calculate error
            error = target_response - current_response[:len(target_response)]
            mse = np.mean(error ** 2)
            
            self._error_history.append(float(mse))
            
            # Check convergence
            if mse < self.convergence_threshold:
                break
            
            # LMS update
            # Simplified: adjust based on error gradient
            gradient = np.correlate(error, channel_response, mode='same')
            taps += self.lms_mu * gradient[:n_taps]
            
            # Apply constraints
            mid = len(tx.pre_cursor)
            taps[:mid] = np.clip(taps[:mid], -tx.max_tap_magnitude, tx.max_tap_magnitude)
            taps[mid + 1:] = np.clip(taps[mid + 1:], -tx.max_tap_magnitude, tx.max_tap_magnitude)
        
        # Update coefficients
        tx.from_fir_coeffs(taps)
        tx.normalize()
        
        return tx
    
    def optimize_dfe_taps(self, tx_signal: np.ndarray,
                          rx_signal: np.ndarray,
                          decisions: np.ndarray) -> List[float]:
        """Optimize DFE taps using LMS
        
        Args:
            tx_signal: Transmitted signal
            rx_signal: Received signal
            decisions: Symbol decisions
            
        Returns:
            Optimized DFE taps
        """
        n_taps = len(self.coeffs.rx.dfe_taps)
        taps = np.zeros(n_taps)
        
        samples_per_ui = 64
        
        for iteration in range(self.max_iterations):
            total_error = 0.0
            
            for i in range(1, min(len(tx_signal) // samples_per_ui, len(decisions))):
                # Calculate DFE feedback
                feedback = sum(taps[j] * decisions[i - j - 1]
                              for j in range(min(i, n_taps)))
                
                # Equalized sample
                center_idx = i * samples_per_ui + samples_per_ui // 2
                equalized = rx_signal[center_idx] - feedback
                
                # Decision
                decision = 1 if equalized > 0 else -1
                
                # Error
                error = equalized - decision
                total_error += error ** 2
                
                # LMS update
                for j in range(min(i, n_taps)):
                    taps[j] += self.lms_mu * error * decisions[i - j - 1]
                    taps[j] = np.clip(taps[j], -self.coeffs.rx.dfe_max_tap_magnitude,
                                     self.coeffs.rx.dfe_max_tap_magnitude)
            
            mse = total_error / len(tx_signal)
            self._error_history.append(float(mse))
            
            if mse < self.convergence_threshold:
                break
        
        self.coeffs.rx.set_dfe_taps(list(taps))
        return list(taps)
    
    def optimize_ctle_for_channel(self, channel_loss_db: np.ndarray,
                                  frequency: np.ndarray) -> RXCoefficients:
        """Auto-tune CTLE based on channel loss
        
        Args:
            channel_loss_db: Channel insertion loss in dB
            frequency: Frequency vector
            
        Returns:
            Optimized RX coefficients
        """
        rx = self.coeffs.rx
        
        # Find frequency with maximum loss
        max_loss_idx = np.argmax(np.abs(channel_loss_db))
        
        # Set zero just below max loss frequency
        if max_loss_idx > 0:
            rx.ctle_zero_idx = min(max_loss_idx, len(rx.zero_options) - 1)
        
        # Set pole above for high-frequency boost
        rx.ctle_pole_idx = min(rx.ctle_zero_idx + 1, len(rx.pole_options) - 1)
        
        # Calculate optimal peaking to compensate loss at Nyquist
        if max_loss_idx < len(channel_loss_db) and max_loss_idx > 0:
            nyquist_idx = len(channel_loss_db) - 1
            target_peaking = max(0, np.abs(channel_loss_db[nyquist_idx]) / 2)
            rx.ctle_peaking_db = min(target_peaking, rx.peaking_range_db[1])
        
        return rx
    
    def optimize_vref_binary_search(self, measure_func: Callable[[int], float],
                                     min_vref: int = 0,
                                     max_vref: int = 63) -> int:
        """Find optimal VREF using binary search
        
        Args:
            measure_func: Function that returns margin for given VREF
            min_vref: Minimum VREF value
            max_vref: Maximum VREF value
            
        Returns:
            Optimal VREF value
        """
        best_vref = 32
        best_margin = 0.0
        
        low, high = min_vref, max_vref
        
        while low <= high:
            mid = (low + high) // 2
            margin = measure_func(mid)
            
            if margin > best_margin:
                best_margin = margin
                best_vref = mid
            
            # Search both sides
            # Assume margin decreases as we move away from optimal
            left_margin = measure_func(low) if low != mid else 0
            right_margin = measure_func(high) if high != mid else 0
            
            if left_margin > right_margin:
                high = mid - 1
            else:
                low = mid + 1
        
        self.coeffs.rx.set_vref(best_vref)
        return best_vref
    
    def optimize_delay_sweep(self, delay_range: range,
                              measure_func: Callable[[int], float]) -> Tuple[int, float]:
        """Find optimal delay by sweeping
        
        Args:
            delay_range: Range of delays to sweep
            measure_func: Function that returns margin for given delay
            
        Returns:
            Tuple of (best_delay, best_margin)
        """
        best_delay = 32
        best_margin = 0.0
        
        for delay in delay_range:
            margin = measure_func(delay)
            if margin > best_margin:
                best_margin = margin
                best_delay = delay
        
        return best_delay, best_margin
    
    def get_convergence_history(self) -> List[float]:
        """Get MSE convergence history"""
        return list(self._error_history)
    
    def is_converged(self) -> bool:
        """Check if optimization has converged"""
        if len(self._error_history) < 10:
            return False
        
        recent = self._error_history[-10:]
        variance = np.var(recent)
        
        return variance < self.convergence_threshold


class CoefficientComparator:
    """Compare and analyze coefficient sets"""
    
    @staticmethod
    def compare(tx1: TXCoefficients, tx2: TXCoefficients) -> Dict[str, Any]:
        """Compare two TX coefficient sets
        
        Returns:
            Dictionary with comparison results
        """
        taps1 = np.array(tx1.get_all_taps())
        taps2 = np.array(tx2.get_all_taps())
        
        diff = taps1 - taps2
        
        return {
            'max_difference': float(np.max(np.abs(diff))),
            'mean_difference': float(np.mean(np.abs(diff))),
            'rms_difference': float(np.sqrt(np.mean(diff ** 2))),
            'boost_difference_db': float(tx1.calculate_boost_db() - tx2.calculate_boost_db()),
        }
    
    @staticmethod
    def compare_rx(rx1: RXCoefficients, rx2: RXCoefficients) -> Dict[str, Any]:
        """Compare two RX coefficient sets"""
        return {
            'ctle_gain_diff_db': float(rx1.ctle_dc_gain_db - rx2.ctle_dc_gain_db),
            'ctle_peaking_diff_db': float(rx1.ctle_peaking_db - rx2.ctle_peaking_db),
            'vref_diff': int(rx1.vref - rx2.vref),
            'dfe_tap_diff': [float(a - b) for a, b in zip(rx1.dfe_taps, rx2.dfe_taps)],
        }
    
    @staticmethod
    def analyze_margin_sensitivity(coeffs: CompleteTapCoefficients,
                                    measure_func: Callable[[CompleteTapCoefficients], float],
                                    perturbation: float = 0.05) -> Dict[str, float]:
        """Analyze margin sensitivity to coefficient changes
        
        Args:
            coeffs: Base coefficients
            measure_func: Function that returns margin for given coefficients
            perturbation: Fractional perturbation to apply
            
        Returns:
            Sensitivity analysis results
        """
        base_margin = measure_func(coeffs)
        
        sensitivities = {}
        
        # TX sensitivity
        for i, tap in enumerate(coeffs.tx.get_all_taps()):
            perturbed = coeffs.copy()
            perturbed.tx.from_fir_coeffs(np.array(perturbed.tx.get_all_taps()))
            taps = np.array(perturbed.tx.get_all_taps())
            taps[i] *= (1 + perturbation)
            perturbed.tx.from_fir_coeffs(taps)
            
            margin = measure_func(perturbed)
            sensitivities[f'tx_tap_{i}'] = float((base_margin - margin) / perturbation)
        
        # RX VREF sensitivity
        for delta in [-5, -2, 2, 5]:
            perturbed = coeffs.copy()
            perturbed.rx.set_vref(coeffs.rx.vref + delta)
            margin = measure_func(perturbed)
            sensitivities[f'vref_{delta:+d}'] = float((base_margin - margin) / abs(delta))
        
        return sensitivities


def create_default_coefficients(channel_id: int = 0) -> CompleteTapCoefficients:
    """Create default coefficients for a channel
    
    Args:
        channel_id: Channel index
        
    Returns:
        Default coefficients
    """
    coeffs = CompleteTapCoefficients()
    coeffs.channel_id = channel_id
    
    # TX defaults
    coeffs.tx.main_cursor = 1.0
    coeffs.tx.pre_cursor = [0.0, 0.0]
    coeffs.tx.post_cursor = [0.0, 0.0]
    
    # RX defaults
    coeffs.rx.vref = 32
    coeffs.rx.ctle_dc_gain_db = 0.0
    coeffs.rx.ctle_peaking_db = 3.0
    
    # DFE defaults
    coeffs.rx.dfe_taps = [0.0] * 5
    
    return coeffs


def export_coefficients_to_dict(coeffs: CompleteTapCoefficients) -> Dict[str, Any]:
    """Export coefficients to dictionary for serialization
    
    Args:
        coeffs: Coefficients to export
        
    Returns:
        Dictionary representation
    """
    return {
        'channel_id': coeffs.channel_id,
        'training_complete': coeffs.training_complete,
        'training_timestamp': coeffs.training_timestamp,
        'tx': {
            'pre_cursor': coeffs.tx.pre_cursor,
            'main_cursor': coeffs.tx.main_cursor,
            'post_cursor': coeffs.tx.post_cursor,
            'boost_db': coeffs.tx.calculate_boost_db(),
        },
        'rx': {
            'ctle_dc_gain_db': coeffs.rx.ctle_dc_gain_db,
            'ctle_peaking_db': coeffs.rx.ctle_peaking_db,
            'ctle_zero_idx': coeffs.rx.ctle_zero_idx,
            'ctle_pole_idx': coeffs.rx.ctle_pole_idx,
            'vref': coeffs.rx.vref,
            'dfe_taps': coeffs.rx.dfe_taps,
        },
        'lane': {
            'rd_delays': coeffs.lane.rd_delays,
            'wr_delays': coeffs.lane.wr_delays,
            'rd_dq_delays': coeffs.lane.rd_dq_delays,
            'wr_dq_delays': coeffs.lane.wr_dq_delays,
        }
    }


def import_coefficients_from_dict(data: Dict[str, Any]) -> CompleteTapCoefficients:
    """Import coefficients from dictionary
    
    Args:
        data: Dictionary representation
        
    Returns:
        Imported coefficients
    """
    coeffs = CompleteTapCoefficients()
    
    coeffs.channel_id = data.get('channel_id', 0)
    coeffs.training_complete = data.get('training_complete', False)
    coeffs.training_timestamp = data.get('training_timestamp', 0)
    
    # TX
    tx_data = data.get('tx', {})
    coeffs.tx.pre_cursor = tx_data.get('pre_cursor', [0.0, 0.0])
    coeffs.tx.main_cursor = tx_data.get('main_cursor', 1.0)
    coeffs.tx.post_cursor = tx_data.get('post_cursor', [0.0, 0.0])
    
    # RX
    rx_data = data.get('rx', {})
    coeffs.rx.ctle_dc_gain_db = rx_data.get('ctle_dc_gain_db', 0.0)
    coeffs.rx.ctle_peaking_db = rx_data.get('ctle_peaking_db', 3.0)
    coeffs.rx.ctle_zero_idx = rx_data.get('ctle_zero_idx', 1)
    coeffs.rx.ctle_pole_idx = rx_data.get('ctle_pole_idx', 1)
    coeffs.rx.vref = rx_data.get('vref', 32)
    coeffs.rx.dfe_taps = rx_data.get('dfe_taps', [0.0] * 5)
    
    # Lane
    lane_data = data.get('lane', {})
    coeffs.lane.rd_delays = lane_data.get('rd_delays', {})
    coeffs.lane.wr_delays = lane_data.get('wr_delays', {})
    coeffs.lane.rd_dq_delays = lane_data.get('rd_dq_delays', {})
    coeffs.lane.wr_dq_delays = lane_data.get('wr_dq_delays', {})
    
    return coeffs