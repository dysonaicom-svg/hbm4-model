"""
Eye Diagram Analyzer for HBM Signal Integrity

Provides eye diagram generation, metrics calculation, BER estimation,
and margin analysis for high-speed memory interfaces.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
import warnings


class EyeMeasurementType(Enum):
    """Types of eye measurements."""
    TIME_DOMAIN = "time_domain"
    VOLTAGE_DOMAIN = "voltage_domain"
    COMBINED = "combined"


@dataclass
class EyeMeasurementConfig:
    """Configuration for eye measurements."""
    # Samples per UI
    samples_per_ui: int = 64
    # Number of UI to capture
    n_ui: int = 1000
    # Voltage resolution for histogram
    v_resolution: float = 0.001
    # Time resolution for histogram (fraction of UI)
    t_resolution: float = 0.01
    # Decision threshold (normalized)
    decision_threshold: float = 0.0
    # BER target
    target_ber: float = 1e-12
    # Confidence level for BER estimation
    confidence_level: float = 0.95


@dataclass
class EyeMetrics:
    """Results of eye diagram analysis."""
    # Eye width (UI)
    eye_width: float
    # Eye height (V)
    eye_height: float
    # Eye area (normalized)
    eye_area: float
    # Vertical closure at center (%)
    vertical_closure: float
    # Horizontal closure at center (%)
    horizontal_closure: float
    # BER estimate
    ber_estimate: float
    # SNR at center (dB)
    snr_db: float
    # Jitter (RMS, UI)
    jitter_rms: float
    # Noise (RMS, V)
    noise_rms: float
    # One level mean
    one_level_mean: float
    # Zero level mean
    zero_level_mean: float
    # One level sigma
    one_level_sigma: float
    # Zero level sigma
    zero_level_sigma: float


class EyeDiagramAnalyzer:
    """
    Eye diagram analyzer for high-speed signals.

    Provides comprehensive eye diagram analysis including:
    - Eye width/height calculation
    - BER estimation via bathtub curve
    - Margin analysis with statistical bounds
    """

    def __init__(self, config: Optional[EyeMeasurementConfig] = None):
        """Initialize eye analyzer."""
        self.config = config or EyeMeasurementConfig()
        self._histogram: Optional[np.ndarray] = None
        self._time_bins: Optional[np.ndarray] = None
        self._voltage_bins: Optional[np.ndarray] = None
        self._eye_samples: Optional[np.ndarray] = None

    def generate_eye_diagram(self, signal: np.ndarray,
                              samples_per_ui: Optional[int] = None,
                              n_ui: Optional[int] = None) -> np.ndarray:
        """
        Generate eye diagram data from signal.

        Args:
            signal: Input signal (oversampled NRZ)
            samples_per_ui: Samples per unit interval
            n_ui: Number of UI to capture

        Returns:
            2D histogram matrix (time x voltage)
        """
        spui = samples_per_ui or self.config.samples_per_ui
        n = n_ui or self.config.n_ui

        # Limit samples if signal is shorter
        max_ui = min(n, len(signal) // spui)
        signal = signal[:max_ui * spui]

        # Calculate histogram dimensions
        t_bins = int(1.0 / self.config.t_resolution)
        v_bins = int(2.0 / self.config.v_resolution)  # Assume +/-1V range

        # Initialize histogram
        histogram = np.zeros((t_bins, v_bins))

        # Voltage bins - extend range to cover all signal values
        v_min, v_max = -1.0, 1.0
        # Extend edges slightly beyond signal range
        epsilon = 0.01
        v_edges = np.linspace(v_min - epsilon, v_max + epsilon, v_bins + 1)

        # Time bins
        t_edges = np.linspace(0, 1.0, t_bins + 1)

        # Build histogram
        for i in range(max_ui):
            start = i * spui
            end = start + spui
            ui_samples = signal[start:end]

            # Time bin for each sample
            t_indices = np.clip(
                ((np.arange(spui) / spui) * t_bins).astype(int),
                0, t_bins - 1
            )

            # Voltage bin for each sample
            # digitize returns index where value would be inserted
            # For values exactly at the last edge, it returns len(edges)
            v_raw = np.digitize(ui_samples, v_edges)
            # Map index len(edges) to last bin (v_bins - 1)
            v_indices = np.clip(v_raw - 1, 0, v_bins - 1)

            # Increment histogram
            for j in range(spui):
                histogram[t_indices[j], v_indices[j]] += 1

        self._histogram = histogram
        self._time_bins = t_edges[:-1]
        self._voltage_bins = v_edges[:-1]
        self._eye_samples = signal

        return histogram

    def calculate_eye_width(self, percentile: float = 0.5) -> float:
        """
        Calculate eye width at specified percentile.

        Args:
            percentile: Fraction of vertical opening (0.5 = center)

        Returns:
            Eye width in UI
        """
        if self._histogram is None:
            return 0.0

        t_bins, v_bins = self._histogram.shape
        eye_widths = []

        # For each time bin, find voltage range
        for t in range(t_bins):
            col = self._histogram[t, :]
            total = np.sum(col)

            if total > 0:
                # Find voltage range at this time
                cumsum = np.cumsum(col) / total
                low_idx = np.searchsorted(cumsum, (1 - percentile) / 2)
                high_idx = np.searchsorted(cumsum, 1 - (1 - percentile) / 2)

                # Eye width contribution
                eye_widths.append(1.0 / t_bins)

        return sum(eye_widths) if eye_widths else 0.0

    def calculate_eye_height(self, percentile: float = 0.5) -> float:
        """
        Calculate eye height at specified percentile.

        Args:
            percentile: Fraction of horizontal opening (0.5 = center)

        Returns:
            Eye height in voltage units
        """
        if self._histogram is None:
            return 0.0

        t_bins, v_bins = self._histogram.shape

        # Find the voltage range at center of time
        center_t_bin = t_bins // 2
        col = self._histogram[center_t_bin, :]

        # Find voltage range containing data at center time
        total = np.sum(col)
        if total == 0:
            return 0.0

        cumsum = np.cumsum(col) / total

        # Find low threshold (percentile/2 from bottom)
        low_thresh = (1 - percentile) / 2
        high_thresh = 1 - low_thresh

        # Find voltage bins at these thresholds
        low_idx = np.searchsorted(cumsum, low_thresh)
        high_idx = np.searchsorted(cumsum, high_thresh)

        # Calculate voltage range
        if low_idx < v_bins and high_idx < v_bins:
            low_v = self._voltage_bins[low_idx]
            high_v = self._voltage_bins[high_idx]
            return max(high_v - low_v, 0.0)

        return 0.0

    def calculate_full_metrics(self) -> EyeMetrics:
        """
        Calculate comprehensive eye metrics.

        Returns:
            EyeMetrics object with all measurements
        """
        if self._histogram is None or self._eye_samples is None:
            return EyeMetrics(
                eye_width=0.0, eye_height=0.0, eye_area=0.0,
                vertical_closure=100.0, horizontal_closure=100.0,
                ber_estimate=1.0, snr_db=0.0, jitter_rms=0.0,
                noise_rms=0.0, one_level_mean=0.0, zero_level_mean=0.0,
                one_level_sigma=0.0, zero_level_sigma=0.0
            )

        # Calculate basic metrics
        eye_width = self.calculate_eye_width(0.5)
        eye_height = self.calculate_eye_height(0.5)

        # Calculate eye area (simplified)
        eye_area = eye_width * eye_height

        # Calculate closures
        vertical_closure = (1.0 - eye_height / 2.0) * 100
        horizontal_closure = (1.0 - eye_width) * 100

        # BER estimation via bathtub model
        ber_estimate = self.estimate_ber()

        # SNR calculation
        snr_db = self.calculate_snr()

        # Jitter estimation
        jitter_rms = self.estimate_jitter()

        # Noise estimation
        noise_rms = self.estimate_noise()

        # Level statistics
        spui = self.config.samples_per_ui
        n_ui = len(self._eye_samples) // spui

        one_samples = []
        zero_samples = []

        for i in range(n_ui):
            start = i * spui
            # First half of UI is "one" if transition exists
            one_samples.extend(self._eye_samples[start:start + spui // 4])
            zero_samples.extend(self._eye_samples[start + 3 * spui // 4:start + spui])

        one_samples = np.array(one_samples)
        zero_samples = np.array(zero_samples)

        one_level_mean = np.mean(one_samples)
        zero_level_mean = np.mean(zero_samples)
        one_level_sigma = np.std(one_samples)
        zero_level_sigma = np.std(zero_samples)

        return EyeMetrics(
            eye_width=eye_width,
            eye_height=eye_height,
            eye_area=eye_area,
            vertical_closure=vertical_closure,
            horizontal_closure=horizontal_closure,
            ber_estimate=ber_estimate,
            snr_db=snr_db,
            jitter_rms=jitter_rms,
            noise_rms=noise_rms,
            one_level_mean=one_level_mean,
            zero_level_mean=zero_level_mean,
            one_level_sigma=one_level_sigma,
            zero_level_sigma=zero_level_sigma
        )

    def estimate_ber(self, method: str = "bathtub") -> float:
        """
        Estimate BER from eye diagram.

        Args:
            method: Estimation method ("bathtub", "gaussian", or "histogram")

        Returns:
            Estimated BER
        """
        if self._histogram is None:
            return 1.0

        # Calculate SNR directly without recursion
        snr_db = self.calculate_snr()

        if snr_db < 1.0:
            # SNR too low, assume worst case
            return 0.5

        # Calculate noise directly
        noise_rms = self.estimate_noise()
        eye_height = self.calculate_eye_height(0.5)

        if method == "bathtub":
            # Simplified bathtub model
            # BER ~ Q((Vopening/2) / sigma)
            Q = snr_db / (20 * np.log10(np.e))  # Convert dB to Q
            ber = 0.5 * (1 - np.math.erf(Q / np.sqrt(2)))
            return max(min(ber, 1.0), 1e-15)

        elif method == "gaussian":
            # Gaussian approximation
            V_margin = eye_height / 2
            sigma = max(noise_rms, 1e-6)
            Q = V_margin / sigma
            ber = 0.5 * np.exp(-Q**2 / 2)
            return min(ber, 1.0)

        else:  # histogram
            # Count samples outside eye opening
            t_bins, v_bins = self._histogram.shape

            # Find eye region (center 50% time, between levels)
            t_low = t_bins // 4
            t_high = 3 * t_bins // 4

            v_center = v_bins // 2
            v_bin_width = (self._voltage_bins[1] - self._voltage_bins[0]) if len(self._voltage_bins) > 1 else 0.01
            v_margin = max(1, int(eye_height / (2 * v_bin_width)))

            # Clamp margins
            v_margin = min(v_margin, v_center - 1)

            outside_eye = np.sum(
                self._histogram[t_low:t_high, :max(0, v_center - v_margin)]
            ) + np.sum(
                self._histogram[t_low:t_high, min(v_bins, v_center + v_margin):]
            )

            total = np.sum(self._histogram)
            return outside_eye / total if total > 0 else 1.0

    def calculate_snr(self) -> float:
        """
        Calculate SNR at eye center.

        Returns:
            SNR in dB
        """
        if self._eye_samples is None:
            return 0.0

        spui = self.config.samples_per_ui
        n_ui = len(self._eye_samples) // spui

        # Sample at center of each UI
        center_samples = []
        for i in range(n_ui):
            idx = i * spui + spui // 2
            if idx < len(self._eye_samples):
                center_samples.append(self._eye_samples[idx])

        center_samples = np.array(center_samples)

        # Separate into levels based on sign
        signal_mean = np.abs(np.mean(center_samples))
        noise_sigma = np.std(center_samples)

        if noise_sigma < 1e-9:
            return 40.0  # Very clean signal

        snr_linear = (signal_mean / noise_sigma) ** 2
        return 10 * np.log10(max(snr_linear, 1e-12))

    def estimate_jitter(self) -> float:
        """
        Estimate RMS jitter from eye crossings.

        Returns:
            RMS jitter in UI
        """
        if self._eye_samples is None:
            return 0.0

        spui = self.config.samples_per_ui
        n_ui = len(self._eye_samples) // spui

        crossings = []

        # Find zero crossings in each UI
        for i in range(n_ui - 1):
            start = i * spui
            ui_data = self._eye_samples[start:start + spui]
            next_ui = self._eye_samples[start + spui:start + 2 * spui]

            # Find crossing between UI i and i+1
            combined = np.concatenate([ui_data, next_ui])
            signs = np.sign(combined)
            sign_changes = np.where(np.diff(signs) != 0)[0]

            for idx in sign_changes:
                if idx < spui:
                    t_crossing = (idx + 1) / spui - 0.5
                    crossings.append(t_crossing)

        if len(crossings) < 2:
            return 0.1  # Default 10% jitter

        crossings = np.array(crossings)
        return np.std(crossings)

    def estimate_noise(self) -> float:
        """
        Estimate RMS noise from eye diagram.

        Returns:
            RMS noise in voltage units
        """
        if self._eye_samples is None:
            return 0.0

        spui = self.config.samples_per_ui
        n_ui = len(self._eye_samples) // spui

        # Collect samples from each level
        one_samples = []
        zero_samples = []

        for i in range(n_ui):
            start = i * spui
            ui_data = self._eye_samples[start:start + spui]

            # First quarter: "one" level region
            one_samples.extend(ui_data[:spui // 4])
            # Last quarter: "zero" level region
            zero_samples.extend(ui_data[3 * spui // 4:spui])

        one_samples = np.array(one_samples)
        zero_samples = np.array(zero_samples)

        # Combined noise estimate
        noise_one = np.std(one_samples)
        noise_zero = np.std(zero_samples)

        return (noise_one + noise_zero) / 2

    def bathtub_curve(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate bathtub curve (BER vs time offset).

        Args:
            n_points: Number of points to sample

        Returns:
            Tuple of (time offsets in UI, BER values)
        """
        if self._histogram is None:
            return np.array([]), np.array([])

        t_bins, v_bins = self._histogram.shape

        # Sample time axis
        t_offsets = np.linspace(0, 1.0, n_points)
        ber_values = []

        threshold = self.config.decision_threshold

        for t in t_offsets:
            t_idx = int(t * t_bins)
            if t_idx >= t_bins:
                t_idx = t_bins - 1

            col = self._histogram[t_idx, :]
            total = np.sum(col)

            if total > 0:
                # BER at this time offset
                v_center = v_bins // 2
                above = np.sum(col[v_center:])
                below = np.sum(col[:v_center])

                ber_t = (above + below) / total
                ber_values.append(ber_t)
            else:
                ber_values.append(1.0)

        return t_offsets, np.array(ber_values)

    def margin_analysis(self, target_ber: float = 1e-12) -> Dict[str, float]:
        """
        Analyze margin to target BER.

        Args:
            target_ber: Target BER specification

        Returns:
            Dictionary with margin metrics
        """
        metrics = self.calculate_full_metrics()

        # Calculate margins
        margin_voltage = metrics.eye_height / 2 - 3 * metrics.noise_rms
        margin_time = metrics.eye_width - 6 * metrics.jitter_rms

        # Voltage margin in dB (relative to signal swing)
        signal_swing = metrics.one_level_mean - metrics.zero_level_mean
        margin_voltage_db = 20 * np.log10(
            max(margin_voltage / signal_swing, 1e-6)
        ) if signal_swing > 0 else -100

        # Time margin in UI
        margin_time_ui = max(margin_time, 0)

        # Overall margin (combined voltage and time)
        margin_combined = margin_voltage * margin_time_ui

        # BER margin
        ber_margin = -np.log10(metrics.ber_estimate / target_ber) if metrics.ber_estimate > 0 else 0

        return {
            'voltage_margin': margin_voltage,
            'voltage_margin_db': margin_voltage_db,
            'time_margin_ui': margin_time_ui,
            'combined_margin': margin_combined,
            'ber_margin_orders': ber_margin,
            'meets_target_ber': metrics.ber_estimate <= target_ber
        }


class BathtubCurveGenerator:
    """
    Bathtub curve generator for BER analysis.

    Produces bathtub curves showing BER variation across
    the unit interval.
    """

    def __init__(self, n_samples_per_ui: int = 64):
        """Initialize bathtub generator."""
        self.n_samples = n_samples_per_ui

    def generate_bathtub(self, signal: np.ndarray,
                         threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate bathtub curve from signal.

        Args:
            signal: Input signal
            threshold: Decision threshold

        Returns:
            Tuple of (time offsets, BER values)
        """
        n_ui = len(signal) // self.n_samples
        t_offsets = np.linspace(0, 1.0, self.n_samples)
        ber_curve = np.zeros(self.n_samples)

        # For each time offset, count errors
        for t in range(self.n_samples):
            n_errors = 0
            n_samples_total = 0

            for ui in range(n_ui - 1):
                idx = ui * self.n_samples + t
                if idx < len(signal) - 1:
                    # Sample and determine expected value
                    sample = signal[idx]
                    next_sample = signal[idx + 1]

                    # Simple error counting (crossing threshold)
                    if sample > threshold:
                        expected = 1
                    else:
                        expected = -1

                    if expected == 1 and sample < threshold:
                        n_errors += 1
                    elif expected == -1 and sample > threshold:
                        n_errors += 1

                    n_samples_total += 1

            if n_samples_total > 0:
                ber_curve[t] = n_errors / n_samples_total

        return t_offsets, ber_curve

    def fit_bathtub_model(self, t: np.ndarray, ber: np.ndarray) -> Dict[str, float]:
        """
        Fit bathtub model to data.

        Model: BER ~ exp(-((t - t_center) / sigma_t)^2 / 2) + BER_floor

        Returns:
            Dictionary with fitted parameters
        """
        # Find center (minimum BER)
        center_idx = np.argmin(ber)
        t_center = t[center_idx]

        # Estimate sigma from width at 10x minimum BER
        min_ber = ber[center_idx]
        threshold_ber = min_ber * 10

        # Find width at threshold
        left_idx = np.where(ber[:center_idx] > threshold_ber)[0]
        right_idx = np.where(ber[center_idx:] > threshold_ber)[0]

        left_width = 0.0
        right_width = 0.0

        if len(left_idx) > 0:
            left_width = t_center - t[left_idx[-1]]
        if len(right_idx) > 0:
            right_width = t[right_idx[0] + center_idx] - t_center

        sigma_t = (left_width + right_width) / 2

        return {
            'center_ui': t_center,
            'sigma_ui': sigma_t,
            'ber_floor': min_ber
        }