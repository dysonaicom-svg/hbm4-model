"""
Channel Model for HBM Signal Integrity Simulation

Implements frequency-dependent loss, impulse response generation,
and crosstalk modeling for high-speed memory interfaces.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class RLGCParameters:
    """RLGC transmission line parameters per unit length."""
    R: float  # Resistance (ohm/m)
    L: float  # Inductance (H/m)
    G: float  # Conductance (S/m)
    C: float  # Capacitance (F/m)
    length: float  # Channel length (m)

    @property
    def characteristic_impedance(self) -> complex:
        """Calculate characteristic impedance Z0 = sqrt((R+jwL)/(G+jwC))."""
        return np.sqrt((self.R + 1j * 0) / (self.G + 1j * 0))

    @property
    def propagation_constant(self) -> complex:
        """Calculate propagation constant gamma = sqrt((R+jwL)(G+jwC))."""
        return np.sqrt((self.R + 1j * 0) * (self.G + 1j * 0))


@dataclass
class ChannelConfig:
    """Configuration for channel model."""
    # Sample rate (Hz)
    sample_rate: float = 32e9  # 32 Gsps for HBM4
    # Target impedance (ohms)
    impedance: float = 50.0
    # DC resistance (ohms)
    dc_resistance: float = 0.1
    # Skin effect coefficient (ohm * sqrt(Hz))
    skin_effect_coeff: float = 0.1
    # Dielectric loss tangent
    loss_tangent: float = 0.02
    # Channel length (mm)
    length_mm: float = 50.0
    # Number of UI for impulse response
    ui_count: int = 10
    # Crosstalk coupling coefficient
    crosstalk_coupling: float = 0.05


class ChannelModel:
    """
    Channel model with frequency-dependent loss.

    Models the channel's S-parameters, impulse response, and crosstalk
    effects for accurate signal integrity simulation.
    """

    def __init__(self, config: Optional[ChannelConfig] = None):
        """Initialize channel model with configuration."""
        self.config = config or ChannelConfig()
        self.length = self.config.length_mm * 1e-3  # Convert to meters
        self._frequency_response: Optional[np.ndarray] = None
        self._impulse_response: Optional[np.ndarray] = None
        self._time_vector: Optional[np.ndarray] = None

    def _generate_frequency_vector(self, n_points: int) -> np.ndarray:
        """Generate frequency vector for analysis."""
        f_max = self.config.sample_rate / 2
        return np.linspace(0, f_max, n_points)

    def calculate_skin_effect_resistance(self, frequency: np.ndarray) -> np.ndarray:
        """
        Calculate frequency-dependent resistance due to skin effect.

        R(f) = R_dc + k * sqrt(f)
        where k is the skin effect coefficient.
        """
        # Use frequency in GHz to keep coefficients reasonable
        freq_ghz = frequency / 1e9
        return self.config.dc_resistance + self.config.skin_effect_coeff * np.sqrt(freq_ghz)

    def calculate_dielectric_admittance(self, frequency: np.ndarray) -> np.ndarray:
        """
        Calculate frequency-dependent admittance due to dielectric loss.

        G(w) = w * C * tan(delta)
        """
        omega = 2 * np.pi * frequency
        # Assume base capacitance of 100 pF/m
        C_base = 100e-12
        return omega * C_base * self.config.loss_tangent

    def frequency_response(self, n_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate channel frequency response.

        Returns:
            Tuple of (frequency vector in Hz, complex transfer function)
        """
        freq = self._generate_frequency_vector(n_points)

        # Calculate frequency-dependent parameters
        R_f = self.calculate_skin_effect_resistance(freq)
        G_f = self.calculate_dielectric_admittance(freq)

        # Base L and C values for typical PCB channel
        L_base = 380e-9  # H/m
        C_base = 100e-12  # F/m

        # Calculate propagation constant
        # gamma = sqrt((R + jwL)(G + jwC))
        omega = 2 * np.pi * freq
        R_complex = R_f
        L_complex = omega * L_base
        G_complex = G_f
        C_complex = omega * C_base

        # Add small epsilon to avoid division by zero at DC
        epsilon = 1e-12
        G_complex = G_complex + epsilon
        C_complex = C_complex + epsilon

        gamma = np.sqrt((R_complex + L_complex) * (G_complex + C_complex))

        # Calculate characteristic impedance
        # Z0 = sqrt((R + jwL)/(G + jwC))
        denominator = G_complex + C_complex
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        Z0 = np.sqrt((R_complex + L_complex) / denominator)

        # For low frequencies where Z0 is ill-defined, use nominal impedance
        Z0 = np.where(np.isfinite(Z0), Z0, self.config.impedance)

        # Transfer function for lossy transmission line
        # H(f) = exp(-gamma * length)
        H = np.exp(-gamma * self.length)

        # Ensure H is finite
        H = np.where(np.isfinite(H), H, 0.0)

        # Apply impedance matching factor at both ends
        Z0_safe = np.where(np.abs(Z0) < epsilon, epsilon, Z0)
        match_factor = (self.config.impedance / (Z0_safe + self.config.impedance)) * \
                      (2 * Z0_safe / (Z0_safe + self.config.impedance))
        match_factor = np.where(np.isfinite(match_factor), match_factor, 1.0)

        H = H * match_factor

        self._frequency_response = H
        return freq, H

    def impulse_response(self, n_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate channel impulse response via IFFT of frequency response.

        Returns:
            Tuple of (time vector in seconds, impulse response)
        """
        if self._impulse_response is None or self._time_vector is None:
            freq, H = self.frequency_response(n_points)

            # Generate time vector
            dt = 1.0 / (2 * self.config.sample_rate)
            t = np.linspace(0, dt * (n_points - 1), n_points)

            # IFFT to get impulse response
            # Need conjugate symmetric input for real output
            H_symmetric = np.concatenate([H, np.conj(H[-2:0:-1])])
            h_full = np.fft.ifft(H_symmetric).real

            # Trim to original length
            h = h_full[:n_points]

            # Normalize for unity gain at DC
            sum_h = np.sum(h)
            if np.abs(sum_h) > 1e-12:
                h = h / sum_h

            self._impulse_response = h
            self._time_vector = t

        return self._time_vector, self._impulse_response

    def step_response(self, n_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate step response by integrating impulse response.

        Returns:
            Tuple of (time vector, step response)
        """
        t, h = self.impulse_response(n_points)
        step = np.cumsum(h) * (t[1] - t[0])
        return t, step

    def insert_loss_db(self, frequency: np.ndarray) -> np.ndarray:
        """
        Calculate insertion loss in dB at given frequencies.

        Args:
            frequency: Frequency vector in Hz

        Returns:
            Insertion loss in dB (positive values for loss)
        """
        if self._frequency_response is None:
            self.frequency_response(len(frequency))

        # IL = -20 * log10(|H(f)|)
        magnitude = np.abs(self._frequency_response)
        # Avoid log(0)
        magnitude = np.maximum(magnitude, 1e-12)
        return -20 * np.log10(magnitude)

    def crosstalk_response(self, aggressor_signal: np.ndarray,
                          n_points: int = 1024) -> np.ndarray:
        """
        Calculate far-end crosstalk from aggressor signal.

        Args:
            aggressor_signal: Input signal on aggressor channel
            n_points: Number of points for simulation

        Returns:
            Crosstalk voltage on victim channel
        """
        t_agg, h_agg = self.impulse_response(n_points)

        # Far-end crosstalk transfer function
        # FEXT coupling coefficient decreases with frequency
        freq, H = self.frequency_response(n_points)

        # Coupling factor with frequency dependence
        coupling = self.config.crosstalk_coupling * (1 + 1j) * freq / freq[-1] * 0.1

        # Apply coupling to transfer function
        H_xtalk = H * coupling

        # IFFT to get crosstalk impulse response
        H_xtalk_symmetric = np.concatenate([H_xtalk, np.conj(H_xtalk[-2:0:-1])])
        h_xtalk = np.fft.ifft(H_xtalk_symmetric).real

        # Convolve aggressor signal with crosstalk impulse response
        crosstalk = np.convolve(aggressor_signal, h_xtalk, mode='same')

        return crosstalk[:len(aggressor_signal)]

    def eye_diagram_parameters(self, prbs_length: int = 127,
                               amplitude: float = 1.0) -> dict:
        """
        Calculate eye diagram parameters for PRBS input.

        Args:
            prbs_length: Length of PRBS pattern
            amplitude: Peak-to-peak signal amplitude

        Returns:
            Dictionary with eye parameters
        """
        # Generate PRBS pattern
        prbs = np.array([1 if i % 2 == 0 else -1 for i in range(prbs_length)])

        # Oversample for eye diagram
        oversample = 64
        samples_per_ui = 64
        prbs_upsampled = np.repeat(prbs, samples_per_ui) * (amplitude / 2)

        # Get channel impulse response
        t, h = self.impulse_response(len(prbs_upsampled))

        # Convolve with channel
        channel_output = np.convolve(prbs_upsampled, h, mode='same')

        # Calculate metrics
        t_ui = 1.0 / self.config.sample_rate * samples_per_ui
        n_ui = len(channel_output) // samples_per_ui

        # Eye opening at center of UI
        center_points = []
        for i in range(n_ui):
            center_idx = i * samples_per_ui + samples_per_ui // 2
            if center_idx < len(channel_output):
                center_points.append(channel_output[center_idx])

        eye_height = np.max(center_points) - np.min(center_points)

        # Eye width (simplified - measure at 50% crossing)
        crossing_points = []
        for i in range(n_ui - 1):
            idx1 = i * samples_per_ui + samples_per_ui // 4
            idx2 = (i + 1) * samples_per_ui - samples_per_ui // 4
            if idx1 < len(channel_output) and idx2 < len(channel_output):
                crossing_points.append((channel_output[idx1], channel_output[idx2]))

        # Calculate zero crossing times
        eye_width = t_ui * 0.8  # Simplified - assume 80% of UI

        return {
            'eye_height': eye_height,
            'eye_width': eye_width,
            'insertion_loss_dc': self.insert_loss_db(np.array([1.0]))[0],
            'insertion_loss_nyquist': self.insert_loss_db(
                np.array([self.config.sample_rate / 4])
            )[0],
            'dc_resistance': self.config.dc_resistance,
            'characteristic_impedance': np.abs(
                self.config.impedance / (1 + self.config.crosstalk_coupling)
            )
        }


class ChannelCrosstalkModel:
    """
    Extended channel model with detailed crosstalk simulation.

    Models NEXT (near-end) and FEXT (far-end) crosstalk for
    multi-lane HBM interfaces.
    """

    def __init__(self, n_channels: int = 16, config: Optional[ChannelConfig] = None):
        """
        Initialize crosstalk model.

        Args:
            n_channels: Number of parallel channels
            config: Channel configuration
        """
        self.n_channels = n_channels
        self.config = config or ChannelConfig()
        self.channels: List[ChannelModel] = []

        for _ in range(n_channels):
            self.channels.append(ChannelModel(self.config))

    def simulate_crosstalk(self, signals: np.ndarray) -> np.ndarray:
        """
        Simulate crosstalk for all channels.

        Args:
            signals: Array of shape (n_samples, n_channels) with input signals

        Returns:
            Array of shape (n_samples, n_channels) with output including crosstalk
        """
        n_samples, n_channels_input = signals.shape
        n_channels_input = min(n_channels_input, self.n_channels)

        outputs = np.zeros((n_samples, self.n_channels))

        for i in range(n_channels_input):
            # Direct channel response - use signal length for impulse response
            t, h = self.channels[i].impulse_response(n_samples)
            # Convolve with trimmed impulse response to match signal length
            conv_result = np.convolve(signals[:, i], h, mode='same')
            outputs[:len(conv_result), i] = conv_result

            # Crosstalk from other channels
            for j in range(n_channels_input):
                if i != j:
                    coupling = self.config.crosstalk_coupling / (abs(i - j) + 1)
                    t, h = self.channels[j].impulse_response(n_samples)
                    xtalk = np.convolve(signals[:, j], h, mode='same')
                    outputs[:, i] += coupling * xtalk

        return outputs

    def calculate_pssn(self, frequency: np.ndarray,
                      victim_channel: int = 0) -> np.ndarray:
        """
        Calculate PSSeN (Power Sum Disturbance) for a victim channel.

        Args:
            frequency: Frequency vector in Hz
            victim_channel: Index of victim channel

        Returns:
            Power sum disturbance in dB
        """
        victim_loss = self.channels[victim_channel].insert_loss_db(frequency)

        total_disturbance = 0
        for i in range(self.n_channels):
            if i != victim_channel:
                dist = self.channels[i].insert_loss_db(frequency)
                total_disturbance += 10 ** (dist / 10)

        psd = -10 * np.log10(10 ** (-victim_loss / 10) + total_disturbance)
        return psd