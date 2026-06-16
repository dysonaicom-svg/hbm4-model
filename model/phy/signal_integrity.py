"""
Signal Integrity Models for HBM PHY Simulation

Implements TX pre-emphasis, RX CTLE (Continuous Time Linear Equalizer),
and signal conditioning for high-speed memory interfaces.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


class EqualizerType(Enum):
    """Types of equalization."""
    NONE = "none"
    TX_PRE_EMPHASIS = "tx_pre_emphasis"
    RX_CTLE = "rx_ctle"
    DFE = "dfe"


@dataclass
class PreEmphasisConfig:
    """Configuration for TX pre-emphasis."""
    # Number of pre-tap positions
    n_pre_taps: int = 2
    # Number of post-tap positions
    n_post_taps: int = 2
    # Maximum tap weight (fraction of main tap)
    max_tap_weight: float = 0.4
    # Tap resolution (bits)
    tap_resolution: int = 5
    # Main cursor weight (normalized)
    main_cursor: float = 1.0


@dataclass
class CTLEConfig:
    """Configuration for RX CTLE."""
    # Number of DC gain stages
    n_dc_gainStages: int = 4
    # DC gain range (dB)
    dc_gain_range: Tuple[float, float] = (-6.0, 6.0)
    # Zero frequency options (Hz)
    zero_options: List[float] = field(default_factory=lambda: [4e9, 8e9, 12e9])
    # Pole frequency options (Hz)
    pole_options: List[float] = field(default_factory=lambda: [8e9, 16e9, 24e9])
    # Number of peaking dB
    peaking_range: Tuple[float, float] = (0.0, 12.0)
    # Stage resolution (dB)
    stage_resolution: float = 1.0


@dataclass
class DFEConfig:
    """Configuration for DFE (Decision Feedback Equalizer)."""
    # Number of DFE taps
    n_taps: int = 5
    # Maximum tap magnitude
    max_tap_magnitude: float = 0.3
    # Convergence rate (mu)
    mu: float = 0.01
    # Decision threshold
    decision_threshold: float = 0.0


@dataclass
class SignalIntegrityConfig:
    """Complete signal integrity configuration."""
    sample_rate: float = 32e9
    ui_ns: float = 31.25e-9  # 32 Gbps
    signal_amplitude: float = 1.0
    noise_rms: float = 0.05
    jitter_rms_ps: float = 2.0
    pre_emphasis: PreEmphasisConfig = field(default_factory=PreEmphasisConfig)
    ctle: CTLEConfig = field(default_factory=CTLEConfig)
    dfe: DFEConfig = field(default_factory=DFEConfig)


class TXPreEmphasis:
    """
    TX Pre-emphasis equalizer.

    Implements FIR-based pre-emphasis to compensate for channel loss
    by boosting high-frequency components at the transmitter.
    """

    def __init__(self, config: Optional[PreEmphasisConfig] = None):
        """Initialize pre-emphasis with configuration."""
        self.config = config or PreEmphasisConfig()
        self.taps = self._initialize_taps()

    def _initialize_taps(self) -> np.ndarray:
        """Initialize tap weights to zero (flat response)."""
        n_taps = self.config.n_pre_taps + 1 + self.config.n_post_taps
        taps = np.zeros(n_taps)
        taps[self.config.n_pre_taps] = self.config.main_cursor
        return taps

    def set_taps(self, tap_values: List[float]) -> None:
        """
        Set tap values with saturation and normalization.

        Args:
            tap_values: List of tap weights
        """
        # Set all taps including main cursor
        for i, val in enumerate(tap_values):
            if i < len(self.taps):
                self.taps[i] = np.clip(
                    val,
                    -self.config.max_tap_weight,
                    self.config.max_tap_weight
                )

        # Normalize so sum of all taps equals main_cursor (unity DC gain)
        total = np.sum(self.taps)
        main_idx = self.config.n_pre_taps
        if np.abs(total) > 1e-9:
            scale = self.config.main_cursor / total
            self.taps = self.taps * scale

    def get_taps(self) -> np.ndarray:
        """Get current tap values."""
        return self.taps.copy()

    def equalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis to signal.

        Args:
            signal: Input signal (NRZ data)

        Returns:
            Equalized signal with pre-emphasis applied
        """
        return np.convolve(signal, self.taps, mode='same')

    def frequency_response(self, n_points: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency response of pre-emphasis filter.

        Returns:
            Tuple of (frequency vector, complex response)
        """
        dt = 1.0 / (32e9)  # Sample period
        t = np.arange(-len(self.taps)//2, len(self.taps)//2 + 1) * dt

        # FIR frequency response using DTFT
        omega = np.linspace(-np.pi, np.pi, n_points)
        f = omega / (2 * np.pi * dt)

        H = np.zeros(n_points, dtype=complex)
        for k, w in enumerate(omega):
            H[k] = np.sum(self.taps * np.exp(-1j * w * np.arange(len(self.taps))))

        return f, H

    def calculate_boost_db(self) -> float:
        """Calculate high-frequency boost in dB."""
        _, H = self.frequency_response()

        # Boost at Nyquist vs DC
        H_dc = np.abs(H[0])
        H_nyq = np.abs(H[-1])

        if H_dc > 0:
            return 20 * np.log10(H_nyq / H_dc)
        return 0.0


class RXCTLE:
    """
    RX Continuous Time Linear Equalizer.

    Implements analog-style CTLE with configurable DC gain,
    zero and pole frequencies for peaking response.
    """

    def __init__(self, config: Optional[CTLEConfig] = None):
        """Initialize CTLE with configuration."""
        self.config = config or CTLEConfig()
        self._dc_gain_db = 0.0
        self._peaking_db = 3.0
        self._zero_idx = 1
        self._pole_idx = 1

    def set_dc_gain(self, gain_db: float) -> None:
        """Set DC gain in dB."""
        self._dc_gain_db = np.clip(
            gain_db,
            self.config.dc_gain_range[0],
            self.config.dc_gain_range[1]
        )

    def set_peaking(self, peaking_db: float) -> None:
        """Set peaking gain in dB."""
        self._peaking_db = np.clip(
            peaking_db,
            self.config.peaking_range[0],
            self.config.peaking_range[1]
        )

    def set_zero_pole(self, zero_idx: int, pole_idx: int) -> None:
        """Set zero and pole frequency indices."""
        self._zero_idx = np.clip(zero_idx, 0, len(self.config.zero_options) - 1)
        self._pole_idx = np.clip(pole_idx, 0, len(self.config.pole_options) - 1)

    def get_zero_frequency(self) -> float:
        """Get current zero frequency."""
        return self.config.zero_options[self._zero_idx]

    def get_pole_frequency(self) -> float:
        """Get current pole frequency."""
        return self.config.pole_options[self._pole_idx]

    def transfer_function(self, frequency: np.ndarray) -> np.ndarray:
        """
        Calculate CTLE transfer function.

        H(s) = H_dc * (s/w_z + 1) / (s/w_p + 1)

        Args:
            frequency: Frequency vector in Hz

        Returns:
            Complex transfer function
        """
        s = 2j * np.pi * frequency

        w_z = 2 * np.pi * self.get_zero_frequency()
        w_p = 2 * np.pi * self.get_pole_frequency()

        H_dc = 10 ** (self._dc_gain_db / 20)

        # Transfer function
        with np.errstate(divide='ignore'):
            H = H_dc * (s / w_z + 1) / (s / w_p + 1)

        return H

    def equalize(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply CTLE equalization to signal.

        Args:
            signal: Input signal
            sample_rate: Signal sample rate

        Returns:
            Equalized signal
        """
        n_points = len(signal)

        # Generate frequency vector
        freq = np.fft.rfftfreq(n_points, 1.0 / sample_rate)

        # Get frequency response
        H_ctle = self.transfer_function(freq)

        # FFT of input signal
        X = np.fft.rfft(signal)

        # Apply CTLE
        Y = X * H_ctle

        # Inverse FFT
        y = np.fft.irfft(Y, n=n_points)

        return y

    def optimize_for_channel(self, channel_loss_db: np.ndarray,
                             frequency: np.ndarray) -> None:
        """
        Auto-tune CTLE based on channel loss curve.

        Args:
            channel_loss_db: Channel insertion loss in dB
            frequency: Corresponding frequency vector
        """
        # Find frequency with maximum loss
        max_loss_idx = np.argmax(np.abs(channel_loss_db))

        # Set zero just below the max loss frequency for peaking
        if max_loss_idx > 0:
            self._zero_idx = min(max_loss_idx, len(self.config.zero_options) - 1)

        # Set pole well above for high-frequency boost
        self._pole_idx = min(self._zero_idx + 1, len(self.config.pole_options) - 1)

        # Set peaking to compensate for loss at Nyquist
        if max_loss_idx < len(channel_loss_db) and max_loss_idx > 0:
            nyquist_idx = len(channel_loss_db) - 1
            target_peaking = max(0, np.abs(channel_loss_db[nyquist_idx]) / 2)
            self._peaking_db = min(target_peaking, self.config.peaking_range[1])


class DFEEqualizer:
    """
    Decision Feedback Equalizer (DFE).

    Implements symbol-by-symbol DFE with adaptive tap update.
    """

    def __init__(self, config: Optional[DFEConfig] = None):
        """Initialize DFE with configuration."""
        self.config = config or DFEConfig()
        self.taps = np.zeros(self.config.n_taps)
        self.samples_per_ui = 64

    def reset(self) -> None:
        """Reset DFE state and taps."""
        self.taps = np.zeros(self.config.n_taps)

    def equalize_symbol(self, samples: np.ndarray, decisions: np.ndarray,
                        symbol_idx: int) -> float:
        """
        Equalize a single symbol using DFE.

        Args:
            samples: Received samples around symbol
            decisions: Previous symbol decisions
            symbol_idx: Index of current symbol

        Returns:
            Equalized sample value
        """
        center_sample = samples[self.samples_per_ui // 2]
        feedback = 0.0

        # Calculate feedback from previous symbols
        for i in range(min(symbol_idx, self.config.n_taps)):
            feedback += self.taps[i] * decisions[symbol_idx - i - 1]

        return center_sample - feedback

    def update_taps(self, error: float, decisions: np.ndarray,
                    symbol_idx: int) -> None:
        """
        Update DFE taps using LMS algorithm.

        Args:
            error: Decision error
            decisions: Symbol decisions
            symbol_idx: Current symbol index
        """
        for i in range(min(symbol_idx, self.config.n_taps)):
            # LMS update: w = w + mu * error * d_prev
            # Standard DFE uses w += mu * error * (-d_prev)
            # where error = equalized - decision and d_prev is previous symbol
            self.taps[i] += self.config.mu * error * (-decisions[symbol_idx - i - 1])

            # Saturate taps
            self.taps[i] = np.clip(
                self.taps[i],
                -self.config.max_tap_magnitude,
                self.config.max_tap_magnitude
            )

    def train(self, tx_signal: np.ndarray, rx_signal: np.ndarray,
              n_iterations: int = 100) -> List[float]:
        """
        Train DFE taps using known training pattern.

        Args:
            tx_signal: Transmitted signal
            rx_signal: Received signal (after channel)
            n_iterations: Number of training iterations

        Returns:
            List of MSE values per iteration
        """
        mse_history = []

        # Resample to symbol rate
        n_symbols = len(tx_signal)
        decisions = np.sign(rx_signal[::self.samples_per_ui])

        for _ in range(n_iterations):
            total_error = 0.0

            for i in range(1, n_symbols):
                equalized = self.equalize_symbol(
                    rx_signal[i * self.samples_per_ui:(i + 2) * self.samples_per_ui],
                    decisions[:i],
                    i
                )

                # Decision
                decision = 1 if equalized > self.config.decision_threshold else -1
                decisions[i] = decision

                # Error
                error = equalized - decision
                total_error += error ** 2

                # Update taps
                self.update_taps(error, decisions, i)

            mse = total_error / n_symbols
            mse_history.append(mse)

        return mse_history


class SignalIntegrityModel:
    """
    Complete signal integrity model integrating TX, channel, and RX components.

    Combines pre-emphasis, channel model, and CTLE/DFE for end-to-end
    signal path simulation.
    """

    def __init__(self, config: Optional[SignalIntegrityConfig] = None):
        """Initialize signal integrity model."""
        self.config = config or SignalIntegrityConfig()

        # Initialize components
        self.tx_pre_emphasis = TXPreEmphasis(self.config.pre_emphasis)
        self.rx_ctle = RXCTLE(self.config.ctle)
        self.dfe = DFEEqualizer(self.config.dfe)

    def set_pre_emphasis_taps(self, pre_taps: List[float], post_taps: List[float]) -> None:
        """
        Set pre-emphasis tap values.

        Args:
            pre_taps: Pre-tap values (before main cursor)
            post_taps: Post-tap values (after main cursor)
        """
        all_taps = pre_taps + [1.0] + post_taps
        self.tx_pre_emphasis.set_taps(all_taps)

    def simulate_tx_to_rx(self, signal: np.ndarray,
                          channel_response: np.ndarray) -> np.ndarray:
        """
        Simulate complete TX -> channel -> RX signal path.

        Args:
            signal: Input TX signal
            channel_response: Channel impulse response

        Returns:
            Received signal after equalization
        """
        # TX pre-emphasis
        tx_out = self.tx_pre_emphasis.equalize(signal)

        # Channel convolution
        channel_out = np.convolve(tx_out, channel_response, mode='same')

        # Add noise if configured
        if self.config.noise_rms > 0:
            noise = np.random.randn(len(channel_out)) * self.config.noise_rms
            channel_out += noise

        # RX CTLE
        rx_ctle_out = self.rx_ctle.equalize(
            channel_out,
            self.config.sample_rate
        )

        return rx_ctle_out

    def apply_dfe(self, signal: np.ndarray, decisions: np.ndarray) -> np.ndarray:
        """
        Apply DFE to signal.

        Args:
            signal: Input signal
            decisions: Symbol decisions

        Returns:
            DFE-equalized signal
        """
        n_symbols = len(signal) // self.dfe.samples_per_ui
        output = np.zeros(len(signal))

        for i in range(n_symbols):
            start = i * self.dfe.samples_per_ui
            end = (i + 1) * self.dfe.samples_per_ui

            equalized = self.dfe.equalize_symbol(
                signal[start:end],
                decisions[:i],
                i
            )
            output[start:end] = equalized

        return output

    def estimate_tx_eye(self, prbs_length: int = 127) -> dict:
        """
        Estimate TX eye diagram metrics.

        Args:
            prbs_length: PRBS pattern length

        Returns:
            Dictionary of eye metrics
        """
        # Generate PRBS
        prbs = np.array([1 if i % 2 == 0 else -1 for i in range(prbs_length)])
        samples_per_ui = 64
        signal = np.repeat(prbs, samples_per_ui) * (self.config.signal_amplitude / 2)

        # Apply pre-emphasis
        tx_out = self.tx_pre_emphasis.equalize(signal)

        # Calculate metrics
        return self._calculate_eye_metrics(tx_out, samples_per_ui)

    def estimate_rx_eye(self, channel_response: np.ndarray,
                        prbs_length: int = 127) -> dict:
        """
        Estimate RX eye diagram metrics after equalization.

        Args:
            channel_response: Channel impulse response
            prbs_length: PRBS pattern length

        Returns:
            Dictionary of eye metrics
        """
        # Generate PRBS
        prbs = np.array([1 if i % 2 == 0 else -1 for i in range(prbs_length)])
        samples_per_ui = 64
        signal = np.repeat(prbs, samples_per_ui) * (self.config.signal_amplitude / 2)

        # Simulate path
        rx_out = self.simulate_tx_to_rx(signal, channel_response)

        # Calculate metrics
        return self._calculate_eye_metrics(rx_out, samples_per_ui)

    def _calculate_eye_metrics(self, signal: np.ndarray,
                               samples_per_ui: int) -> dict:
        """
        Calculate eye diagram metrics from signal.

        Args:
            signal: Signal samples
            samples_per_ui: Samples per unit interval

        Returns:
            Dictionary with eye width, height, and other metrics
        """
        n_ui = len(signal) // samples_per_ui

        # Extract eye samples at each UI crossing
        eye_samples = []
        for i in range(n_ui):
            start = i * samples_per_ui
            end = start + samples_per_ui
            eye_samples.append(signal[start:end])

        eye_samples = np.array(eye_samples)

        # Eye height: vertical opening at center
        center = samples_per_ui // 2
        one_level = eye_samples[:, :center // 2]
        zero_level = eye_samples[:, center + center // 2:]

        eye_height = np.mean(one_level.max(axis=1)) - np.mean(zero_level.min(axis=1))

        # Eye width: horizontal opening at center
        crossings = []
        for i in range(n_ui - 1):
            # Find transition
            diff = eye_samples[i + 1] - eye_samples[i]
            zero_crossings = np.where(np.diff(np.sign(diff)))[0]
            if len(zero_crossings) > 0:
                crossings.append(zero_crossings[0])

        if len(crossings) > 0:
            crossing_mean = np.mean(crossings)
            eye_width = 1.0  # Normalized UI
        else:
            eye_width = 0.5

        # SNR estimate
        signal_pwr = np.mean(eye_samples ** 2)
        noise_pwr = np.var(eye_samples)

        return {
            'eye_height': eye_height,
            'eye_width': eye_width,
            'snr_db': 10 * np.log10(signal_pwr / max(noise_pwr, 1e-12)),
            'pre_emphasis_boost_db': self.tx_pre_emphasis.calculate_boost_db(),
            'ctle_dc_gain_db': self.rx_ctle._dc_gain_db,
            'ctle_peaking_db': self.rx_ctle._peaking_db
        }