"""
IBIS Model Wrapper - Behavioral model for simulation

Provides high-level interface to IBIS model data for channel simulation.
Generates IV curves, V-T waveforms, and behavioral models for signal integrity analysis.

Key features:
- IV curve generation and interpolation
- V-T curve generation and interpolation
- Behavioral model construction for simulation
- Signal integrity metrics calculation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import math
import numpy as np
from enum import Enum

from model.phy.ibis_parser import (
    IBISFile, IBISModel, IBISModelType, IVCurve, VTWaveform,
    CompositeDataTable, IBISPackage, parse_ibis_content
)


class SignalIntegrityMetric(Enum):
    """Signal integrity analysis metrics"""
    OVERSHOOT = "overshoot"
    UNDERSHOOT = "undershoot"
    SETTLING_TIME = "settling_time"
    RINGING_FREQUENCY = "ringing_frequency"
    CROSSTALK_PEAK = "crosstalk_peak"
    EYE_WIDTH = "eye_width"
    EYE_HEIGHT = "eye_height"


@dataclass
class WaveformMetrics:
    """Computed waveform quality metrics"""
    rise_time: float = 0.0  # 20%-80% rise time (ns)
    fall_time: float = 0.0  # 20%-80% fall time (ns)
    overshoot: float = 0.0  # Maximum overshoot (V)
    undershoot: float = 0.0  # Maximum undershoot (V)
    settling_time: float = 0.0  # Time to settle within tolerance (ns)
    settling_voltage: float = 0.0  # Target settling voltage (V)
    max_slew_rate: float = 0.0  # Maximum slew rate (V/ns)
    min_slew_rate: float = 0.0  # Minimum slew rate (V/ns)


@dataclass
class ChannelResponse:
    """Channel frequency response"""
    frequency: List[float]  # Frequency points (Hz)
    impedance: List[complex]  # Impedance at each frequency
    transfer_function: List[complex]  # S21 or transfer function

    def get_insertion_loss(self, freq_hz: float) -> float:
        """Get insertion loss at given frequency (dB)"""
        for i, f in enumerate(self.frequency):
            if f >= freq_hz:
                z = self.transfer_function[i]
                mag = abs(z)
                if mag > 0:
                    return 20 * math.log10(mag)
                return -float('inf')
        return -float('inf')

    def get_phase_delay(self, freq_hz: float) -> float:
        """Get phase delay at given frequency (ns)"""
        for i, f in enumerate(self.frequency):
            if f >= freq_hz:
                z = self.transfer_function[i]
                phase_rad = math.atan2(z.imag, z.real)
                phase_deg = math.degrees(phase_rad)
                return -phase_deg / (2 * math.pi * freq_hz * 1e-9) if freq_hz > 0 else 0.0
        return 0.0


@dataclass
class BehavioralModel:
    """Behavioral model for simulation

    Encapsulates the behavioral representation of an IBIS model
    suitable for time-domain or frequency-domain simulation.
    """
    model_name: str
    v_il: float = 0.0  # Input low voltage threshold
    v_ih: float = 0.0  # Input high voltage threshold
    v_ol: float = 0.0  # Output low voltage
    v_oh: float = 0.0  # Output high voltage
    r_out: float = 50.0  # Output resistance (ohms)
    c_comp: float = 0.0  # Compensation capacitance (pF)
    i_osc: float = 0.0  # Oscillator current
    v_cc: float = 1.2  # Supply voltage

    # Timing parameters
    t_rise: float = 0.0  # Rise time (ns)
    t_fall: float = 0.0  # Fall time (ns)
    t_prop: float = 0.0  # Propagation delay (ns)

    # Waveform data
    rising_waveform: Optional[VTWaveform] = None
    falling_waveform: Optional[VTWaveform] = None

    # Composite data table
    composite_data: Optional[CompositeDataTable] = None

    def drive_output(self, time: float, input_value: bool, v_toggle: float = 1.2) -> float:
        """Compute output voltage at given time

        Args:
            time: Current time (ns)
            input_value: True for high, False for low
            v_toggle: Toggle voltage level

        Returns:
            Output voltage (V)
        """
        if self.rising_waveform is None and self.falling_waveform is None:
            # Simple step response model
            if input_value:
                return v_toggle if time >= 0 else 0.0
            else:
                return 0.0 if time >= 0 else v_toggle

        if input_value and self.rising_waveform:
            return self.rising_waveform.interpolate(time)
        elif not input_value and self.falling_waveform:
            return self.falling_waveform.interpolate(time)
        else:
            return v_toggle / 2  # Unknown state

    def compute_slew_rate(self, waveform: VTWaveform, start_pct: float = 20.0, end_pct: float = 80.0) -> float:
        """Compute slew rate from waveform

        Args:
            waveform: V-T waveform data
            start_pct: Start percentage (e.g., 20 for 20%)
            end_pct: End percentage (e.g., 80 for 80%)

        Returns:
            Slew rate in V/ns
        """
        if len(waveform.time) < 2 or len(waveform.voltage) < 2:
            return 0.0

        v_start = min(waveform.voltage) + (max(waveform.voltage) - min(waveform.voltage)) * start_pct / 100
        v_end = min(waveform.voltage) + (max(waveform.voltage) - min(waveform.voltage)) * end_pct / 100

        # Find times at these voltage levels
        t_start = None
        t_end = None

        for i in range(len(waveform.voltage) - 1):
            v1, v2 = waveform.voltage[i], waveform.voltage[i + 1]
            if v1 <= v_start <= v2 or v2 <= v_start <= v1:
                if v2 != v1:
                    t_start = waveform.time[i] + (v_start - v1) / (v2 - v1) * (waveform.time[i + 1] - waveform.time[i])
            if v1 <= v_end <= v2 or v2 <= v_end <= v1:
                if v2 != v1:
                    t_end = waveform.time[i] + (v_end - v1) / (v2 - v1) * (waveform.time[i + 1] - waveform.time[i])

        if t_start is not None and t_end is not None and t_end != t_start:
            dv = v_end - v_start
            dt = t_end - t_start
            return dv / dt if dt != 0 else 0.0

        return 0.0


class IBISModelWrapper:
    """IBIS Model Wrapper for simulation

    Wraps IBIS model data and provides high-level interface for:
    - IV curve generation and interpolation
    - V-T curve generation and interpolation
    - Behavioral model construction
    - Signal integrity analysis
    """

    def __init__(self, ibis_file: IBISFile):
        """Initialize wrapper with parsed IBIS file

        Args:
            ibis_file: Parsed IBIS file data
        """
        self.ibis_file = ibis_file
        self._models: Dict[str, BehavioralModel] = {}

    def get_model(self, model_name: str) -> Optional[BehavioralModel]:
        """Get behavioral model for named IBIS model

        Args:
            model_name: Name of IBIS model

        Returns:
            BehavioralModel or None if not found
        """
        if model_name in self._models:
            return self._models[model_name]

        # Find model in IBIS file
        if model_name not in self.ibis_file.models:
            return None

        ibis_model = self.ibis_file.models[model_name]
        behavioral = self._create_behavioral_model(ibis_model)
        self._models[model_name] = behavioral
        return behavioral

    def _create_behavioral_model(self, ibis_model: IBISModel) -> BehavioralModel:
        """Create behavioral model from IBIS model data

        Args:
            ibis_model: Parsed IBIS model

        Returns:
            BehavioralModel object
        """
        model = BehavioralModel(
            model_name=ibis_model.model_name,
            c_comp=ibis_model.c_comp,
            v_cc=ibis_model.v_meas if ibis_model.v_meas > 0 else 1.2,
            rising_waveform=ibis_model.rising_waveform,
            falling_waveform=ibis_model.falling_waveform,
            composite_data=ibis_model.composite_data
        )

        # Compute output levels from IV curves or waveforms
        # v_oh = high output level (from rising waveform final value or v_cc)
        if ibis_model.rising_waveform and len(ibis_model.rising_waveform.voltage) > 0:
            model.v_oh = max(ibis_model.rising_waveform.voltage)
        elif ibis_model.pullup:
            # Use operating point from pullup curve (midpoint current)
            mid_current = sum(ibis_model.pullup.current) / len(ibis_model.pullup.current)
            model.v_oh = self._interpolate_voltage_at_current(ibis_model.pullup, mid_current)
        else:
            model.v_oh = model.v_cc

        # v_ol = low output level (from falling waveform final value or 0)
        if ibis_model.falling_waveform and len(ibis_model.falling_waveform.voltage) > 0:
            model.v_ol = min(ibis_model.falling_waveform.voltage)
        elif ibis_model.pulldown:
            mid_current = sum(ibis_model.pulldown.current) / len(ibis_model.pulldown.current)
            model.v_ol = self._interpolate_voltage_at_current(ibis_model.pulldown, mid_current)
        else:
            model.v_ol = 0.0

        # Compute input thresholds
        model.v_il = model.v_oh * 0.3
        model.v_ih = model.v_oh * 0.7

        # Compute timing from waveforms
        if ibis_model.rising_waveform:
            model.t_rise = self._compute_transition_time(ibis_model.rising_waveform)
        if ibis_model.falling_waveform:
            model.t_fall = self._compute_transition_time(ibis_model.falling_waveform)

        # Estimate output resistance from pullup/pulldown
        if ibis_model.pulldown:
            model.r_out = self._estimate_output_resistance(ibis_model.pulldown)

        return model

    def _interpolate_voltage_at_current(self, curve: IVCurve, current: float) -> float:
        """Find voltage where IV curve produces given current

        Args:
            curve: IV curve
            current: Target current (A)

        Returns:
            Voltage (V)
        """
        if len(curve.current) < 2:
            return 0.0

        # Linear interpolation
        for i in range(len(curve.current) - 1):
            i1, i2 = curve.current[i], curve.current[i + 1]
            v1, v2 = curve.voltage[i], curve.voltage[i + 1]

            if (i1 <= current <= i2) or (i2 <= current <= i1):
                if i2 != i1:
                    t = (current - i1) / (i2 - i1)
                    return v1 + t * (v2 - v1)

        # Extrapolate
        if current < curve.current[0]:
            i1, i2 = curve.current[0], curve.current[1]
            v1, v2 = curve.voltage[0], curve.voltage[1]
            if i2 != i1:
                t = (current - i1) / (i2 - i1)
                return v1 + t * (v2 - v1)
        else:
            i1, i2 = curve.current[-2], curve.current[-1]
            v1, v2 = curve.voltage[-2], curve.voltage[-1]
            if i2 != i1:
                t = (current - i1) / (i2 - i1)
                return v1 + t * (v2 - v1)

        return 0.0

    def _estimate_output_resistance(self, curve: IVCurve) -> float:
        """Estimate output resistance from IV curve

        Args:
            curve: IV curve

        Returns:
            Resistance in ohms
        """
        if len(curve.voltage) < 2:
            return 50.0  # Default

        # Use dV/dI around operating point
        v_op = 0.5  # Operating voltage
        i_op = curve.interpolate(v_op)

        v_small = v_op * 0.9
        v_large = v_op * 1.1
        i_small = curve.interpolate(v_small)
        i_large = curve.interpolate(v_large)

        if i_large != i_small:
            r_out = (v_large - v_small) / (i_large - i_small)
            if r_out > 0 and r_out < 1000:
                return r_out

        return 50.0  # Default

    def _compute_transition_time(self, waveform: VTWaveform, pct: float = 80.0) -> float:
        """Compute transition time for waveform

        Args:
            waveform: V-T waveform
            pct: Percentage for transition (80 = 20%-80%)

        Returns:
            Transition time in ns
        """
        if len(waveform.voltage) < 2:
            return 0.0

        v_min = min(waveform.voltage)
        v_max = max(waveform.voltage)
        v_range = v_max - v_min

        if v_range == 0:
            return 0.0

        v_start = v_min + v_range * (100 - pct) / 100
        v_end = v_min + v_range * pct / 100

        t_start = None
        t_end = None

        for i in range(len(waveform.voltage) - 1):
            v1, v2 = waveform.voltage[i], waveform.voltage[i + 1]
            t1, t2 = waveform.time[i], waveform.time[i + 1]

            # Rising edge
            if v1 <= v_start <= v2 or v2 <= v_start <= v1:
                if v2 != v1:
                    t_start = t1 + (v_start - v1) / (v2 - v1) * (t2 - t1)
            if v1 <= v_end <= v2 or v2 <= v_end <= v1:
                if v2 != v1:
                    t_end = t1 + (v_end - v1) / (v2 - v1) * (t2 - t1)

        if t_start is not None and t_end is not None:
            return abs(t_end - t_start)

        return 0.0

    def generate_iv_curve(self, model_name: str, num_points: int = 100,
                          v_min: float = -0.5, v_max: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate IV curve with fine resolution

        Args:
            model_name: Name of IBIS model
            num_points: Number of points to generate
            v_min: Minimum voltage
            v_max: Maximum voltage

        Returns:
            Tuple of (voltage array, current array)
        """
        model = self.get_model(model_name)
        if model is None:
            return np.array([]), np.array([])

        ibis_model = self.ibis_file.models.get(model_name)
        if ibis_model is None:
            return np.array([]), np.array([])

        v_out = np.linspace(v_min, v_max, num_points)
        i_out = np.zeros(num_points)

        # Use pullup for high state
        if ibis_model.pullup:
            for i, v in enumerate(v_out):
                i_out[i] = ibis_model.pullup.interpolate(v)
        # Use pulldown for low state
        elif ibis_model.pulldown:
            for i, v in enumerate(v_out):
                i_out[i] = ibis_model.pulldown.interpolate(v)

        return v_out, i_out

    def generate_vt_curve(self, model_name: str, num_points: int = 100,
                          t_max: float = 10.0, is_rising: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Generate V-T curve with fine resolution

        Args:
            model_name: Name of IBIS model
            num_points: Number of points to generate
            t_max: Maximum time (ns)
            is_rising: True for rising, False for falling

        Returns:
            Tuple of (time array, voltage array)
        """
        model = self.get_model(model_name)
        if model is None:
            return np.array([]), np.array([])

        ibis_model = self.ibis_file.models.get(model_name)
        if ibis_model is None:
            return np.array([]), np.array([])

        t_out = np.linspace(0, t_max, num_points)
        v_out = np.zeros(num_points)

        waveform = ibis_model.rising_waveform if is_rising else ibis_model.falling_waveform

        if waveform and len(waveform.time) > 0:
            for i, t in enumerate(t_out):
                v_out[i] = waveform.interpolate(t)
        else:
            # Simple step response
            v_final = model.v_oh if is_rising else model.v_ol
            t_transition = model.t_rise if is_rising else model.t_fall
            if t_transition > 0:
                for i, t in enumerate(t_out):
                    if is_rising:
                        v_out[i] = v_final * (1 - math.exp(-t / (t_transition / 5)))
                    else:
                        v_out[i] = v_final * math.exp(-t / (t_transition / 5))
            else:
                v_out = np.full(num_points, v_final)

        return t_out, v_out

    def compute_waveform_metrics(self, waveform: VTWaveform,
                                 tolerance: float = 0.02) -> WaveformMetrics:
        """Compute waveform quality metrics

        Args:
            waveform: V-T waveform
            tolerance: Settling tolerance (fraction)

        Returns:
            WaveformMetrics object
        """
        metrics = WaveformMetrics()

        if len(waveform.time) < 2 or len(waveform.voltage) < 2:
            return metrics

        v_min = min(waveform.voltage)
        v_max = max(waveform.voltage)
        v_settle = v_min if v_max > 0 else 0.0  # Target is final value

        # Compute rise/fall times (20%-80%)
        metrics.rise_time = self._compute_transition_time(waveform)
        metrics.fall_time = self._compute_transition_time(waveform)

        # Compute overshoot/undershoot
        mid_point = (v_max + v_min) / 2
        metrics.overshoot = max(0, max(waveform.voltage) - v_max) if v_max > 0 else 0
        metrics.undershoot = min(0, min(waveform.voltage) - v_min)

        # Compute settling time
        v_tolerance = abs(v_max - v_min) * tolerance
        metrics.settling_voltage = v_settle

        # Find time when waveform enters tolerance band and stays
        settled = False
        settle_time = waveform.time[-1]

        for i in range(len(waveform.voltage) - 1, -1, -1):
            v = waveform.voltage[i]
            if abs(v - v_settle) > v_tolerance:
                if i + 1 < len(waveform.time):
                    settle_time = waveform.time[i + 1]
                    settled = True
                break

        metrics.settling_time = settle_time

        # Compute slew rate
        if len(waveform.time) >= 2:
            dv = np.diff(waveform.voltage)
            dt = np.diff(waveform.time)
            dt[dt == 0] = 1e-9  # Avoid division by zero
            slew_rates = dv / dt
            metrics.max_slew_rate = np.max(np.abs(slew_rates))
            metrics.min_slew_rate = np.min(np.abs(slew_rates))

        return metrics


def create_model_wrapper(ibis_content: str) -> IBISModelWrapper:
    """Create IBIS model wrapper from content string

    Args:
        ibis_content: Raw IBIS file content

    Returns:
        IBISModelWrapper object
    """
    ibis_file = parse_ibis_content(ibis_content)
    return IBISModelWrapper(ibis_file)


def create_model_wrapper_from_file(file_path: str) -> IBISModelWrapper:
    """Create IBIS model wrapper from file

    Args:
        file_path: Path to .ibs file

    Returns:
        IBISModelWrapper object
    """
    from model.phy.ibis_parser import parse_ibis_file
    ibis_file = parse_ibis_file(file_path)
    return IBISModelWrapper(ibis_file)