"""
IBIS-based Channel Simulator

Performs signal integrity analysis using IBIS models:
- Channel simulation with behavioral models
- Signal distortion calculation
- Crosstalk estimation
- Eye diagram generation
- Jitter analysis

Key features:
- Time-domain simulation with IBIS behavioral models
- Frequency-domain S-parameter analysis
- Crosstalk coupling simulation
- Statistical eye diagram analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import math
import numpy as np
from enum import Enum

from model.phy.ibis_parser import (
    IBISFile, IBISModel, IVCurve, VTWaveform, IBISPackage, parse_ibis_file
)
from model.phy.ibis_model import (
    IBISModelWrapper, BehavioralModel, WaveformMetrics,
    WaveformMetrics as WM, ChannelResponse, create_model_wrapper_from_file
)


class SimulationMode(Enum):
    """Simulation mode"""
    TIME_DOMAIN = "time_domain"
    FREQUENCY_DOMAIN = "frequency_domain"
    HYBRID = "hybrid"


@dataclass
class ChannelParameters:
    """Channel physical parameters"""
    length: float = 10.0  # Channel length (mm)
    impedance: float = 50.0  # Characteristic impedance (ohms)
    propagation_velocity: float = 150.0  # ps/mm (typical for PCB)
    capacitance_per_mm: float = 0.5  # pF/mm
    inductance_per_mm: float = 0.5  # nH/mm
    resistance_per_mm: float = 0.1  # ohms/mm (lossy channel)
    dielectric_loss_tan: float = 0.02  # Loss tangent
    coupling_capacitance: float = 0.1  # pF (crosstalk coupling)


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    mode: SimulationMode = SimulationMode.TIME_DOMAIN
    t_stop: float = 50.0  # Stop time (ns)
    t_start: float = -5.0  # Start time (ns)
    dt: float = 0.001  # Time step (ns)
    v_drive: float = 1.2  # Drive voltage (V)
    r_drive: float = 50.0  # Driver impedance (ohms)
    r_load: float = 50.0  # Load impedance (ohms)
    c_load: float = 0.0  # Load capacitance (pF)
    temperature: float = 25.0  # Temperature (C)
    process: str = "typical"  # Process corner

    def num_samples(self) -> int:
        """Number of simulation samples"""
        return int((self.t_stop - self.t_start) / self.dt)


@dataclass
class SignalDistortion:
    """Signal distortion metrics"""
    attenuation_db: float = 0.0  # Total attenuation at Nyquist
    dispersion_ps: float = 0.0  # Pulse broadening
    intersymbol_interference: float = 0.0  # ISI metric
    duty_cycle_distortion: float = 0.0  # DCD
    total_jitter: float = 0.0  # Tj (ps)
    deterministic_jitter: float = 0.0  # Dj (ps)
    random_jitter: float = 0.0  # Rj (ps)


@dataclass
class CrosstalkResult:
    """Crosstalk analysis results"""
    peak_aggressor_voltage: float = 0.0  # Peak voltage on victim
    rms_voltage: float = 0.0  # RMS noise
    coupling_coefficient: float = 0.0  # Mutual coupling
    far_end_crosstalk: float = 0.0  # FEXT
    near_end_crosstalk: float = 0.0  # NEXT


@dataclass
class EyeAnalysisResult:
    """Eye diagram analysis results"""
    eye_width: float = 0.0  # Eye opening width (UI)
    eye_height: float = 0.0  # Eye opening height (V)
    eye_height_percent: float = 0.0  # Eye height as percentage
    bit_error_rate_estimate: float = 0.0  # Estimated BER
    vertical_closure: float = 0.0  # Vertical eye closure
    horizontal_closure: float = 0.0  # Horizontal eye closure
    sampling_window: float = 0.0  # Valid sampling window (ps)


@dataclass
class SimulationResult:
    """Complete simulation result"""
    time: np.ndarray
    input_voltage: np.ndarray
    output_voltage: np.ndarray
    waveform_metrics: Optional[WaveformMetrics] = None
    distortion: Optional[SignalDistortion] = None
    crosstalk: Optional[CrosstalkResult] = None
    eye_analysis: Optional[EyeAnalysisResult] = None
    channel_response: Optional[ChannelResponse] = None

    def save_to_file(self, filename: str):
        """Save results to numpy file"""
        np.savez(filename,
                 time=self.time,
                 input_voltage=self.input_voltage,
                 output_voltage=self.output_voltage)


class IBISSimulator:
    """IBIS-based channel simulator

    Performs signal integrity analysis using IBIS behavioral models.
    Supports time-domain, frequency-domain, and hybrid simulation modes.
    """

    def __init__(self, ibis_file: Optional[IBISFile] = None,
                 channel_params: Optional[ChannelParameters] = None):
        """Initialize simulator

        Args:
            ibis_file: Parsed IBIS file (optional)
            channel_params: Channel parameters (optional)
        """
        self.ibis_file = ibis_file
        self.channel_params = channel_params or ChannelParameters()
        self.model_wrapper = None
        if ibis_file:
            self.model_wrapper = IBISModelWrapper(ibis_file)

    @classmethod
    def from_file(cls, ibis_file_path: str,
                  channel_params: Optional[ChannelParameters] = None) -> 'IBISSimulator':
        """Create simulator from IBIS file

        Args:
            ibis_file_path: Path to .ibs file
            channel_params: Channel parameters (optional)

        Returns:
            IBISSimulator instance
        """
        ibis_file = parse_ibis_file(ibis_file_path)
        return cls(ibis_file, channel_params)

    def simulate(self, model_name: str,
                 config: Optional[SimulationConfig] = None) -> SimulationResult:
        """Run channel simulation

        Args:
            model_name: Name of IBIS model to use
            config: Simulation configuration (optional)

        Returns:
            SimulationResult with waveforms and metrics
        """
        config = config or SimulationConfig()

        # Generate time array
        t = np.arange(config.t_start, config.t_stop, config.dt)

        # Generate input signal (simple PRBS-like pattern)
        input_v = self._generate_input_pattern(t, config)

        # Simulate output through channel
        output_v = self._simulate_channel(t, input_v, model_name, config)

        result = SimulationResult(
            time=t,
            input_voltage=input_v,
            output_voltage=output_v
        )

        # Compute metrics if model available
        if self.model_wrapper:
            model = self.model_wrapper.get_model(model_name)
            if model:
                result.waveform_metrics = self._compute_waveform_metrics(t, output_v)
                result.distortion = self._compute_distortion(t, input_v, output_v, config)

        return result

    def _generate_input_pattern(self, t: np.ndarray, config: SimulationConfig) -> np.ndarray:
        """Generate input signal pattern

        Args:
            t: Time array
            config: Simulation config

        Returns:
            Input voltage array
        """
        v = np.zeros_like(t)
        ui = 2.0  # Unit interval (ns) for 500 Mbps

        for i, ti in enumerate(t):
            # Simple clock pattern
            period_idx = int(ti / ui) % 2
            v[i] = config.v_drive if period_idx == 0 else 0.0

        return v

    def _simulate_channel(self, t: np.ndarray, input_v: np.ndarray,
                          model_name: str, config: SimulationConfig) -> np.ndarray:
        """Simulate signal propagation through channel

        Args:
            t: Time array
            input_v: Input voltage
            model_name: IBIS model name
            config: Simulation config

        Returns:
            Output voltage at far end
        """
        output_v = np.zeros_like(input_v)

        # Get behavioral model
        model = None
        if self.model_wrapper:
            model = self.model_wrapper.get_model(model_name)

        # Channel parameters
        l_ch = self.channel_params.length
        v_p = self.channel_params.propagation_velocity
        z0 = self.channel_params.impedance

        # Propagation delay
        t_prop = l_ch / v_p  # ns

        # Simple lossy transmission line model
        r_l = self.channel_params.resistance_per_mm * l_ch
        l_l = self.channel_params.inductance_per_mm * l_ch
        c_l = self.channel_params.capacitance_per_mm * l_ch

        # Frequency-dependent attenuation
        f_nyquist = 0.5 / (config.dt * 1e-9)  # Nyquist frequency

        for i, ti in enumerate(t):
            # Input delay (propagation)
            t_input = ti - t_prop
            if t_input < t[0]:
                output_v[i] = 0.0
                continue

            # Find input at propagation delay
            idx = np.searchsorted(t, t_input)
            if idx >= len(input_v):
                idx = len(input_v) - 1

            v_in = input_v[idx]

            # Apply driver output model
            if model:
                v_out = model.drive_output(ti, v_in > config.v_drive / 2, config.v_drive)
            else:
                v_out = v_in

            # Apply channel loss (simplified)
            alpha = 1.0 / (1.0 + r_l / z0)  # Loss factor
            v_out = v_out * alpha

            # Apply RC filtering effect
            rc_time = r_l * c_l * 1e-6  # Convert to ns
            if rc_time > 0:
                filter_factor = 1.0 - math.exp(-config.dt / rc_time)
                if i > 0:
                    v_out = output_v[i-1] * (1 - filter_factor) + v_out * filter_factor

            output_v[i] = v_out

        return output_v

    def _compute_waveform_metrics(self, t: np.ndarray, v: np.ndarray) -> WaveformMetrics:
        """Compute waveform quality metrics

        Args:
            t: Time array
            v: Voltage array

        Returns:
            WaveformMetrics
        """
        metrics = WaveformMetrics()

        v_min = np.min(v)
        v_max = np.max(v)
        v_mid = (v_max + v_min) / 2

        # Rise/fall times
        v_low = v_min + (v_max - v_min) * 0.1
        v_high = v_min + (v_max - v_min) * 0.9

        # Find crossing times
        t_rise_start = None
        t_rise_end = None
        t_fall_start = None
        t_fall_end = None

        for i in range(len(v) - 1):
            # Rising edge
            if v[i] <= v_low <= v[i+1]:
                t_rise_start = t[i]
            if v[i] <= v_high <= v[i+1]:
                t_rise_end = t[i]

            # Falling edge
            if v[i] >= v_high >= v[i+1]:
                t_fall_start = t[i]
            if v[i] >= v_low >= v[i+1]:
                t_fall_end = t[i]

        if t_rise_end and t_rise_start:
            metrics.rise_time = t_rise_end - t_rise_start
        if t_fall_end and t_fall_start:
            metrics.fall_time = t_fall_end - t_fall_start

        # Slew rate
        if len(t) >= 2:
            dv = np.diff(v)
            dt = np.diff(t)
            dt[dt == 0] = 1e-9
            slew_rates = dv / dt
            metrics.max_slew_rate = np.max(np.abs(slew_rates))

        return metrics

    def _compute_distortion(self, t: np.ndarray, v_in: np.ndarray,
                           v_out: np.ndarray, config: SimulationConfig) -> SignalDistortion:
        """Compute signal distortion metrics

        Args:
            t: Time array
            v_in: Input voltage
            v_out: Output voltage
            config: Simulation config

        Returns:
            SignalDistortion metrics
        """
        dist = SignalDistortion()

        # Compute attenuation
        v_in_peak = np.max(np.abs(v_in))
        v_out_peak = np.max(np.abs(v_out))
        if v_in_peak > 0:
            attenuation = 20 * math.log10(v_out_peak / v_in_peak)
            dist.attenuation_db = -attenuation if attenuation < 0 else 0

        # Compute ISI (simplified)
        # Find transitions and measure eye opening
        ui = 2.0  # Unit interval
        num_ui = int(config.t_stop / ui)

        v_levels = []
        for i in range(num_ui):
            t_start = i * ui
            idx_start = np.searchsorted(t, t_start)
            idx_end = np.searchsorted(t, t_start + ui * 0.5)
            if idx_start < len(v_out) and idx_end <= len(v_out):
                v_levels.append(np.mean(v_out[idx_start:idx_end]))

        if len(v_levels) >= 2:
            v_level_std = np.std(v_levels)
            dist.intersymbol_interference = v_level_std

        # Simplified jitter estimate
        dist.total_jitter = config.dt * 1000  # ps

        return dist

    def simulate_crosstalk(self, aggressor_config: SimulationConfig,
                          victim_model_name: str,
                          aggressor_model_name: str = None) -> CrosstalkResult:
        """Simulate crosstalk between channels

        Args:
            aggressor_config: Simulation config for aggressor
            victim_model_name: Victim model name
            aggressor_model_name: Aggressor model name (optional)

        Returns:
            CrosstalkResult with crosstalk metrics
        """
        result = CrosstalkResult()

        # Simulate aggressor
        aggressor_result = self.simulate(aggressor_model_name or "default", aggressor_config)
        v_agg = aggressor_result.output_voltage

        # Coupling model
        k_c = self.channel_params.coupling_capacitance / 10.0  # Normalized coupling

        # Compute coupling coefficient
        z0 = self.channel_params.impedance
        l = self.channel_params.length
        c_c = self.channel_params.coupling_capacitance

        result.coupling_coefficient = k_c

        # Far-end crosstalk (FEXT) - proportional to coupled length and dv/dt
        if len(v_agg) >= 2 and len(aggressor_result.time) >= 2:
            # Use time array to compute dt
            dt_array = np.diff(aggressor_result.time)
            dt_array[dt_array == 0] = 1e-9
            dv = np.diff(v_agg)
            dv_dt_max = np.max(np.abs(dv / dt_array))
            result.far_end_crosstalk = k_c * dv_dt_max * l * 1e-3  # Scale factor

        # Peak noise on victim
        result.peak_aggressor_voltage = np.max(np.abs(v_agg)) * k_c

        # RMS noise
        noise = v_agg * k_c
        result.rms_voltage = np.sqrt(np.mean(noise ** 2))

        return result

    def analyze_eye(self, waveform: np.ndarray, t: np.ndarray,
                   bit_pattern: str = "10101010",
                   ui: float = 2.0) -> EyeAnalysisResult:
        """Analyze eye diagram

        Args:
            waveform: Received voltage waveform
            t: Time array
            bit_pattern: Bit pattern for clock recovery
            ui: Unit interval (ns)

        Returns:
            EyeAnalysisResult with eye metrics
        """
        result = EyeAnalysisResult()

        if len(waveform) == 0:
            return result

        # Build eye diagram data
        num_bits = len(bit_pattern)
        t_eye = []
        v_eye = []

        # Align to bit boundaries
        for i, bit in enumerate(bit_pattern):
            t_start = i * ui
            t_end = (i + 1) * ui

            idx_start = np.searchsorted(t, t_start)
            idx_end = np.searchsorted(t, t_end)

            if idx_start < len(t) and idx_end <= len(t):
                t_eye.extend(t[idx_start:idx_end] - t_start)
                v_eye.extend(waveform[idx_start:idx_end])

        if len(t_eye) == 0 or len(v_eye) == 0:
            return result

        t_eye = np.array(t_eye)
        v_eye = np.array(v_eye)

        # Find eye opening
        # Sample at center of UI
        center_idx = t_eye < ui * 0.5

        if np.any(center_idx):
            v_center = v_eye[center_idx]
            v_low = np.percentile(v_center, 10)
            v_high = np.percentile(v_center, 90)

            result.eye_height = v_high - v_low

            # Eye width (time where voltage is between thresholds)
            v_mid = (v_high + v_low) / 2
            v_thresh = v_low + (v_high - v_low) * 0.5

            crossing_mask = np.abs(v_eye - v_thresh) < (v_high - v_low) * 0.1
            if np.any(crossing_mask):
                t_crossings = t_eye[crossing_mask]
                if len(t_crossings) >= 2:
                    t_sorted = np.sort(t_crossings)
                    # Find longest gap at each phase position
                    eye_width_samples = []
                    for phase in np.linspace(0, ui, 100):
                        phase_mask = np.abs((t_sorted % ui) - phase) < ui * 0.05
                        if np.sum(phase_mask) >= 2:
                            phase_samples = t_sorted[phase_mask]
                            max_gap = np.max(np.diff(phase_samples))
                            eye_width_samples.append(max_gap)

                    if eye_width_samples:
                        result.eye_width = np.min(eye_width_samples)

        # Eye height as percentage
        v_peak = np.max(v_eye) - np.min(v_eye)
        if v_peak > 0:
            result.eye_height_percent = (result.eye_height / v_peak) * 100

        # Estimated BER (simplified)
        noise_std = np.std(v_eye - np.mean(v_eye))
        if noise_std > 0:
            # Q-factor based BER estimate
            q_factor = result.eye_height / (2 * noise_std)
            # Complementary error function approximation
            result.bit_error_rate_estimate = 0.5 * math.erfc(q_factor / math.sqrt(2))

        return result

    def compute_channel_frequency_response(self) -> ChannelResponse:
        """Compute channel frequency response

        Returns:
            ChannelResponse with S-parameters
        """
        l = self.channel_params.length
        v_p = self.channel_params.propagation_velocity
        z0 = self.channel_params.impedance
        r_l = self.channel_params.resistance_per_mm * l
        l_l = self.channel_params.inductance_per_mm * l
        c_l = self.channel_params.capacitance_per_mm * l

        # Frequency points
        f = np.logspace(8, 11, 200)  # 100 MHz to 100 GHz
        omega = 2 * np.pi * f

        impedance = []
        transfer = []

        for w in omega:
            # Simplified transmission line model
            gamma = complex(r_l, w * l_l * 1e-9) * complex(0, w * c_l * 1e-12)
            z_in = z0 * (1 + gamma) / (1 - gamma) if abs(1 - gamma) > 1e-10 else z0

            # Transfer function (S21) with numerical safety
            # Clamp gamma to prevent exp overflow
            gamma_clamped = max(-700, min(700, gamma.real)) + 1j * max(-700, min(700, gamma.imag))
            h = np.exp(-gamma_clamped) if abs(gamma) > 1e-10 else complex(1, 0)
            # Include loss
            loss_factor = 1.0 / (1.0 + r_l / z0)

            impedance.append(z_in)
            transfer.append(h * loss_factor)

        return ChannelResponse(
            frequency=f.tolist(),
            impedance=impedance,
            transfer_function=transfer
        )

    def compute_signal_distortion(self, simulation_result: SimulationResult,
                                  config: SimulationConfig) -> SignalDistortion:
        """Compute comprehensive signal distortion

        Args:
            simulation_result: Result from simulate()
            config: Simulation configuration

        Returns:
            SignalDistortion with all metrics
        """
        dist = SignalDistortion()

        t = simulation_result.time
        v_in = simulation_result.input_voltage
        v_out = simulation_result.output_voltage

        if len(t) < 2:
            return dist

        # Attenuation
        v_in_rms = np.sqrt(np.mean(v_in ** 2))
        v_out_rms = np.sqrt(np.mean(v_out ** 2))
        if v_in_rms > 0:
            dist.attenuation_db = 20 * math.log10(v_out_rms / v_in_rms)

        # Dispersion (pulse broadening)
        # Find pulse width at half max
        v_peak = np.max(v_out)
        v_half = v_peak / 2

        width_samples = []
        for i in range(len(v_out) - 1):
            if (v_out[i] < v_half <= v_out[i+1]) or (v_out[i+1] < v_half <= v_out[i]):
                width_samples.append(t[i])

        if len(width_samples) >= 2:
            pulse_width = width_samples[1] - width_samples[0] if len(width_samples) >= 2 else 0
            dist.dispersion_ps = pulse_width * 1000  # Convert to ps

        # ISI
        ui = 2.0  # Unit interval
        num_bits = int(config.t_stop / ui)

        v_levels = []
        for i in range(num_bits):
            t_start = i * ui
            idx_start = np.searchsorted(t, t_start + ui * 0.3)
            idx_end = np.searchsorted(t, t_start + ui * 0.7)
            if idx_start < len(v_out) and idx_end <= len(v_out):
                v_levels.append(np.mean(v_out[idx_start:idx_end]))

        if len(v_levels) >= 2:
            dist.intersymbol_interference = np.std(v_levels) / np.mean(v_levels)

        # Jitter decomposition
        # Find crossing points
        v_thresh = (np.max(v_out) + np.min(v_out)) / 2
        crossing_times = []

        for i in range(len(v_out) - 1):
            if (v_out[i] < v_thresh <= v_out[i+1]) or (v_out[i+1] < v_thresh <= v_out[i]):
                # Linear interpolation
                if v_out[i+1] != v_out[i]:
                    t_cross = t[i] + (v_thresh - v_out[i]) / (v_out[i+1] - v_out[i]) * (t[i+1] - t[i])
                    crossing_times.append(t_cross)

        if len(crossing_times) >= 2:
            # Period jitter
            periods = np.diff(crossing_times)
            period_mean = np.mean(periods)
            period_std = np.std(periods)

            # Deterministic jitter (DCD + DDJ)
            dist.deterministic_jitter = period_std * 1000  # ps

            # Total jitter (simplified)
            dist.total_jitter = dist.deterministic_jitter * 3  # Approximate Tj

        return dist


def create_simulator(ibis_file_path: str,
                     channel_params: Optional[ChannelParameters] = None) -> IBISSimulator:
    """Create IBIS simulator from file

    Args:
        ibis_file_path: Path to .ibs file
        channel_params: Channel parameters (optional)

    Returns:
        IBISSimulator instance
    """
    return IBISSimulator.from_file(ibis_file_path, channel_params)