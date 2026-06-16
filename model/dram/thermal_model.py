"""
HBM3/4 Thermal Model

Per-channel thermal modeling with:
- Thermal coupling between channels
- Thermal time constants (transient thermal response)
- Junction temperature estimation
- Thermal emergency handling
- Package thermal model

Reference:
- JEDEC JESD238 HBM3 specification
- JEDEC JESD270-4A HBM4 specification
- JESD51-14 Thermal test method
- Semiconductor Thermal Measurement and Management Manual

Thermal Characteristics:
- HBM3: Junction-to-case thermal resistance ~0.5 C/W
- HBM4: Improved thermal design ~0.4 C/W
- Thermal time constants: 1-100ms range
- Max junction temperature: 95C (HBM3), 105C (HBM4)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math
import time


class ThermalState(Enum):
    """Thermal state of a channel or device"""
    NORMAL = "normal"              # Temperature within spec
    ELEVATED = "elevated"          # Above normal but safe
    WARNING = "warning"            # Approaching thermal limit
    CRITICAL = "critical"          # Near thermal limit
    EMERGENCY = "emergency"        # Thermal emergency shutdown


class ThermalEvent(Enum):
    """Thermal events for monitoring"""
    TEMP_THRESHOLD_EXCEEDED = "temp_threshold_exceeded"
    THERMAL_COUPLED_WARNING = "thermal_coupled_warning"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    THERMAL_RECOVERY = "thermal_recovery"


@dataclass
class ThermalParameters:
    """Thermal model parameters

    All temperatures in Celsius unless noted.
    All resistances in C/W unless noted.
    """
    # === Package Thermal Resistance ===
    # Junction-to-case (die to package top)
    theta_jc: float = 0.5       # C/W (HBM3 typical)
    theta_jc_hbm4: float = 0.4  # C/W (HBM4 improved)

    # Junction-to-ambient (with heat sink)
    theta_ja: float = 10.0      # C/W (typical with heat sink)
    theta_ja_no_hs: float = 25.0  # C/W (without heat sink)

    # Channel-to-channel thermal coupling
    theta_cc: float = 2.0       # C/W (coupling between adjacent channels)

    # === Temperature Limits ===
    max_junction_temp_c: float = 95.0    # HBM3 max junction temp
    max_junction_temp_hbm4_c: float = 105.0  # HBM4 max junction temp
    warning_threshold_c: float = 80.0    # Warning threshold
    critical_threshold_c: float = 88.0   # Critical threshold
    emergency_threshold_c: float = 92.0  # Emergency threshold

    # === Ambient Conditions ===
    ambient_temp_c: float = 45.0        # Typical ambient
    max_ambient_temp_c: float = 85.0    # Max operating ambient

    # === Thermal Time Constants ===
    # RC thermal model time constants (in seconds)
    tau_fast: float = 0.001      # 1ms - fast thermal transient (local heating)
    tau_medium: float = 0.1      # 100ms - medium thermal transient
    tau_slow: float = 1.0        # 1s - slow thermal transient (package)

    # Thermal capacitance (J/C)
    c_thermal_jc: float = 0.5    # Junction-to-case thermal cap
    c_thermal_ja: float = 5.0    # Junction-to-ambient thermal cap

    # === Power Calibration ===
    power_to_temp_factor: float = 1.0  # mW to temperature factor

    @property
    def max_junction_temp(self) -> float:
        """Get max junction temp (HBM3 default)"""
        return self.max_junction_temp_c

    def get_thermal_resistance(self, model: str = "jc") -> float:
        """Get thermal resistance for model type

        Args:
            model: "jc" (junction-case) or "ja" (junction-ambient)

        Returns:
            Thermal resistance in C/W
        """
        if model == "jc":
            return self.theta_jc
        elif model == "ja":
            return self.theta_ja
        else:
            return self.theta_jc

    def get_safe_temperature(self) -> float:
        """Get safe operating temperature (below warning)"""
        return self.warning_threshold_c - 5.0


@dataclass
class ChannelThermalState:
    """Per-channel thermal state"""
    channel_id: int

    # Temperature tracking (Celsius)
    junction_temp_c: float = 45.0      # Current junction temperature
    local_temp_c: float = 45.0         # Local hot-spot temperature
    case_temp_c: float = 42.0          # Case temperature
    neighbor_avg_temp_c: float = 45.0   # Average of neighboring channels

    # Thermal history for RC model
    temp_history: List[Tuple[float, float]] = field(default_factory=list)  # (time, temp)
    power_history: List[Tuple[float, float]] = field(default_factory=list)   # (time, power_mw)

    # Thermal state
    state: ThermalState = ThermalState.NORMAL

    # Time tracking
    last_update_time_s: float = 0.0

    def update_temperature(self, power_mw: float, dt_s: float, params: ThermalParameters):
        """Update temperature using RC thermal model

        Args:
            power_mw: Instantaneous power in mW
            dt_s: Time delta in seconds
            params: Thermal parameters
        """
        current_time = time.time()
        if self.last_update_time_s > 0:
            dt_s = current_time - self.last_update_time_s
        self.last_update_time_s = current_time

        # Simple RC thermal model:
        # dT/dt = (P * theta - T) / tau
        # Solution: T(t) = T_initial * exp(-t/tau) + P * theta * (1 - exp(-t/tau))

        # Junction temperature from power
        delta_t_junction = power_mw * params.theta_jc / 1000.0  # Convert mW to W

        # Apply thermal time constant (using average of fast and medium)
        tau = (params.tau_fast + params.tau_medium) / 2

        # Exponential moving average for temperature
        alpha = 1.0 - math.exp(-dt_s / tau)

        # Update junction temperature
        self.junction_temp_c = (self.junction_temp_c * (1 - alpha) +
                                (params.ambient_temp_c + delta_t_junction * 1000) * alpha)

        # Local hot-spot is typically higher
        self.local_temp_c = self.junction_temp_c + (power_mw * 0.05)  # 0.05 C/mW local rise

        # Case temperature follows junction with delay
        self.case_temp_c = (self.case_temp_c * 0.9 +
                           (self.junction_temp_c - 3.0) * 0.1)  # Lag behind junction

        # Update thermal state
        self._update_thermal_state(params)

        # Keep history bounded
        if len(self.temp_history) > 1000:
            self.temp_history = self.temp_history[-500:]
        self.temp_history.append((current_time, self.junction_temp_c))

        if len(self.power_history) > 1000:
            self.power_history = self.power_history[-500:]
        self.power_history.append((current_time, power_mw))

    def _update_thermal_state(self, params: ThermalParameters):
        """Update thermal state based on temperature"""
        if self.junction_temp_c >= params.emergency_threshold_c:
            self.state = ThermalState.EMERGENCY
        elif self.junction_temp_c >= params.critical_threshold_c:
            self.state = ThermalState.CRITICAL
        elif self.junction_temp_c >= params.warning_threshold_c:
            self.state = ThermalState.WARNING
        elif self.junction_temp_c >= params.get_safe_temperature():
            self.state = ThermalState.ELEVATED
        else:
            self.state = ThermalState.NORMAL

    def get_thermal_rate(self) -> float:
        """Calculate temperature change rate (C/s)

        Returns:
            Temperature change rate
        """
        if len(self.temp_history) < 2:
            return 0.0

        # Look at last few samples
        recent = self.temp_history[-min(10, len(self.temp_history)):]
        if len(recent) < 2:
            return 0.0

        t1, temp1 = recent[0]
        t2, temp2 = recent[-1]
        dt = t2 - t1

        if dt <= 0:
            return 0.0

        return (temp2 - temp1) / dt

    def get_average_power(self, window_s: float = 1.0) -> float:
        """Get average power over time window

        Args:
            window_s: Time window in seconds

        Returns:
            Average power in mW
        """
        if not self.power_history:
            return 0.0

        current_time = time.time()
        cutoff_time = current_time - window_s

        # Filter samples within window
        samples = [(t, p) for t, p in self.power_history if t >= cutoff_time]
        if not samples:
            return 0.0

        return sum(p for _, p in samples) / len(samples)

    def get_steady_state_temp(self, power_mw: float, params: ThermalParameters) -> float:
        """Calculate steady-state temperature for given power

        Args:
            power_mw: Power in mW
            params: Thermal parameters

        Returns:
            Steady-state junction temperature in Celsius
        """
        power_w = power_mw / 1000.0
        delta_t = power_w * params.theta_jc
        return params.ambient_temp_c + delta_t * 1000


@dataclass
class ThermalEventRecord:
    """Record of a thermal event"""
    event_type: ThermalEvent
    channel_id: int
    timestamp: float
    temperature_c: float
    threshold_c: float
    description: str


@dataclass
class ThermalEmergencyAction:
    """Action to take during thermal emergency"""
    action_type: str           # "throttle", "power_down", "refresh_reduce"
    duration_cycles: int       # How long to apply action
    description: str


@dataclass
class ThermalModel:
    """HBM Thermal Model

    Models thermal behavior of HBM stack with:
    - Per-channel thermal tracking
    - Thermal coupling between channels
    - Thermal time constants (transient response)
    - Junction temperature estimation
    - Emergency handling
    - Power-temperature conversion
    """
    num_channels: int = 32
    params: ThermalParameters = field(default_factory=ThermalParameters)

    # Per-channel thermal state
    channels: List[ChannelThermalState] = field(default_factory=list)

    # Thermal events
    events: List[ThermalEventRecord] = field(default_factory=list)
    max_events: int = 1000

    # Emergency state
    emergency_active: bool = False
    emergency_channels: List[int] = field(default_factory=list)

    # Coupling model
    enable_coupling: bool = True

    # Time tracking
    current_time_s: float = 0.0

    def __post_init__(self):
        """Initialize channel thermal states"""
        if not self.channels:
            self.channels = [
                ChannelThermalState(channel_id=i)
                for i in range(self.num_channels)
            ]

    def update_channel_power(
        self,
        channel_id: int,
        power_mw: float,
        dt_s: float = 0.001
    ):
        """Update thermal state for a channel based on power

        Args:
            channel_id: Channel index (0-31)
            power_mw: Instantaneous power in mW
            dt_s: Time step in seconds (default 1ms)
        """
        if 0 <= channel_id < self.num_channels:
            self.channels[channel_id].update_temperature(power_mw, dt_s, self.params)

            # Update thermal coupling
            if self.enable_coupling:
                self._update_thermal_coupling(channel_id, power_mw, dt_s)

    def _update_thermal_coupling(
        self,
        channel_id: int,
        source_power_mw: float,
        dt_s: float
    ):
        """Update neighboring channels due to thermal coupling

        Args:
            channel_id: Source channel
            source_power_mw: Power of source channel
            dt_s: Time step
        """
        # Get adjacent channels
        neighbors = self._get_neighbor_channels(channel_id)

        for neighbor_id in neighbors:
            if 0 <= neighbor_id < self.num_channels:
                neighbor = self.channels[neighbor_id]

                # Calculate coupled temperature rise
                coupled_temp_rise = (source_power_mw * self.params.theta_cc / 1000.0)

                # Apply coupling factor (temperature diff drives heat flow)
                temp_diff = self.channels[channel_id].junction_temp_c - neighbor.junction_temp_c
                if temp_diff > 0:
                    coupling_factor = temp_diff * 0.01  # Small coupling contribution
                    neighbor.junction_temp_c += coupling_factor * dt_s / self.params.tau_fast

                    # Update neighbor average
                    neighbor.neighbor_avg_temp_c = (
                        sum(self.channels[n].junction_temp_c for n in neighbors) / len(neighbors)
                    )

    def _get_neighbor_channels(self, channel_id: int) -> List[int]:
        """Get thermally coupled neighbor channels

        Args:
            channel_id: Source channel

        Returns:
            List of neighbor channel IDs
        """
        neighbors = []

        # Adjacent channels in same stack
        if channel_id > 0:
            neighbors.append(channel_id - 1)
        if channel_id < self.num_channels - 1:
            neighbors.append(channel_id + 1)

        # Channels in same bank group (for HBM3/4 architecture)
        # This is a simplified model
        bank_group_size = 8
        bg_id = channel_id % bank_group_size

        # Add same position in adjacent bank groups
        if bg_id > 0:
            neighbors.append(channel_id - 1)
        if bg_id < bank_group_size - 1:
            neighbors.append(channel_id + 1)

        return list(set(neighbors))  # Remove duplicates

    def update_all_channels_power(
        self,
        power_per_channel_mw: List[float],
        dt_s: float = 0.001
    ):
        """Update all channels with power values

        Args:
            power_per_channel_mw: List of power values per channel
            dt_s: Time step
        """
        for i, power in enumerate(power_per_channel_mw):
            if i < self.num_channels:
                self.update_channel_power(i, power, dt_s)

    def get_channel_temperature(self, channel_id: int) -> float:
        """Get junction temperature for a channel

        Args:
            channel_id: Channel index

        Returns:
            Junction temperature in Celsius
        """
        if 0 <= channel_id < self.num_channels:
            return self.channels[channel_id].junction_temp_c
        return self.params.ambient_temp_c

    def get_all_temperatures(self) -> List[float]:
        """Get temperatures for all channels

        Returns:
            List of junction temperatures
        """
        return [ch.junction_temp_c for ch in self.channels]

    def get_average_temperature(self) -> float:
        """Get average temperature across all channels

        Returns:
            Average junction temperature
        """
        if not self.channels:
            return self.params.ambient_temp_c
        return sum(ch.junction_temp_c for ch in self.channels) / len(self.channels)

    def get_max_temperature(self) -> Tuple[int, float]:
        """Get channel with maximum temperature

        Returns:
            (channel_id, temperature)
        """
        if not self.channels:
            return (-1, self.params.ambient_temp_c)

        max_ch = max(self.channels, key=lambda ch: ch.junction_temp_c)
        return (max_ch.channel_id, max_ch.junction_temp_c)

    def get_min_temperature(self) -> Tuple[int, float]:
        """Get channel with minimum temperature

        Returns:
            (channel_id, temperature)
        """
        if not self.channels:
            return (-1, self.params.ambient_temp_c)

        min_ch = min(self.channels, key=lambda ch: ch.junction_temp_c)
        return (min_ch.channel_id, min_ch.junction_temp_c)

    def get_temperature_gradient(self) -> float:
        """Get temperature gradient across channels

        Returns:
            Max - Min temperature difference
        """
        if not self.channels:
            return 0.0

        temps = [ch.junction_temp_c for ch in self.channels]
        return max(temps) - min(temps)

    def check_thermal_state(self, channel_id: int) -> ThermalState:
        """Check thermal state for a channel

        Args:
            channel_id: Channel index

        Returns:
            Thermal state
        """
        if 0 <= channel_id < self.num_channels:
            return self.channels[channel_id].state
        return ThermalState.NORMAL

    def get_emergency_actions(self) -> List[ThermalEmergencyAction]:
        """Get recommended emergency actions based on thermal state

        Returns:
            List of recommended actions
        """
        actions = []

        # Check all channels
        for ch in self.channels:
            if ch.state == ThermalState.EMERGENCY:
                actions.append(ThermalEmergencyAction(
                    action_type="throttle",
                    duration_cycles=1000,
                    description=f"Throttle channel {ch.channel_id} during thermal emergency"
                ))
            elif ch.state == ThermalState.CRITICAL:
                actions.append(ThermalEmergencyAction(
                    action_type="power_down",
                    duration_cycles=500,
                    description=f"Reduce power on channel {ch.channel_id} in critical state"
                ))
            elif ch.state == ThermalState.WARNING:
                actions.append(ThermalEmergencyAction(
                    action_type="refresh_reduce",
                    duration_cycles=200,
                    description=f"Reduce refresh rate on channel {ch.channel_id} in warning"
                ))

        return actions

    def record_event(
        self,
        event_type: ThermalEvent,
        channel_id: int,
        temperature_c: float,
        threshold_c: float,
        description: str = ""
    ):
        """Record a thermal event

        Args:
            event_type: Type of event
            channel_id: Channel where event occurred
            temperature_c: Temperature at event
            threshold_c: Threshold that was exceeded
            description: Event description
        """
        event = ThermalEventRecord(
            event_type=event_type,
            channel_id=channel_id,
            timestamp=time.time(),
            temperature_c=temperature_c,
            threshold_c=threshold_c,
            description=description
        )

        self.events.append(event)

        # Keep history bounded
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events // 2:]

    def get_recent_events(self, count: int = 10) -> List[ThermalEventRecord]:
        """Get recent thermal events

        Args:
            count: Number of events to return

        Returns:
            List of recent events
        """
        return self.events[-count:]

    def get_events_by_type(self, event_type: ThermalEvent) -> List[ThermalEventRecord]:
        """Get events of a specific type

        Args:
            event_type: Event type to filter

        Returns:
            List of matching events
        """
        return [e for e in self.events if e.event_type == event_type]

    def estimate_power_from_temperature(
        self,
        channel_id: int,
        ambient_temp_c: float = None
    ) -> float:
        """Estimate power from measured temperature

        Args:
            channel_id: Channel index
            ambient_temp_c: Ambient temperature (uses default if None)

        Returns:
            Estimated power in mW
        """
        if 0 > channel_id or channel_id >= self.num_channels:
            return 0.0

        if ambient_temp_c is None:
            ambient_temp_c = self.params.ambient_temp_c

        ch = self.channels[channel_id]
        temp_rise_c = ch.junction_temp_c - ambient_temp_c

        # P = delta_T / theta_jc (but convert to mW)
        power_w = temp_rise_c / self.params.theta_jc
        return power_w * 1000.0

    def get_thermal_resistance_network(
        self,
        channel_id: int
    ) -> Dict[str, float]:
        """Get thermal resistance network for a channel

        Args:
            channel_id: Channel index

        Returns:
            Dict of resistance paths
        """
        if 0 > channel_id or channel_id >= self.num_channels:
            return {}

        ch = self.channels[channel_id]

        return {
            "junction_to_case": self.params.theta_jc,
            "junction_to_ambient": self.params.theta_ja,
            "channel_to_channel": self.params.theta_cc,
            "neighbor_avg_temp_c": ch.neighbor_avg_temp_c,
            "case_to_ambient": self.params.theta_ja - self.params.theta_jc,
        }

    def simulate_temperature_response(
        self,
        initial_temp_c: float,
        power_mw: float,
        duration_s: float,
        num_steps: int = 100
    ) -> List[Tuple[float, float]]:
        """Simulate temperature response to step power change

        Args:
            initial_temp_c: Initial temperature
            power_mw: Step power change in mW
            duration_s: Simulation duration in seconds
            num_steps: Number of simulation steps

        Returns:
            List of (time, temperature) tuples
        """
        results = []

        dt = duration_s / num_steps
        tau = (self.params.tau_fast + self.params.tau_medium + self.params.tau_slow) / 3
        power_w = power_mw / 1000.0

        temp = initial_temp_c
        ambient = self.params.ambient_temp_c
        theta = self.params.theta_jc

        for step in range(num_steps + 1):
            t = step * dt

            # Analytical solution for RC thermal model
            # T(t) = T_ambient + (T_initial - T_ambient) * exp(-t/tau) + P * theta * (1 - exp(-t/tau))
            exp_factor = math.exp(-t / tau)
            steady_state_delta = power_w * theta * 1000  # Convert to C

            temp = ambient + (initial_temp_c - ambient) * exp_factor + steady_state_delta * (1 - exp_factor)

            results.append((t, temp))

        return results

    def get_temperature_stats(self) -> Dict[str, float]:
        """Get temperature statistics

        Returns:
            Dict with temperature statistics
        """
        if not self.channels:
            return {
                "avg_temp_c": self.params.ambient_temp_c,
                "max_temp_c": self.params.ambient_temp_c,
                "min_temp_c": self.params.ambient_temp_c,
                "gradient_c": 0.0,
            }

        temps = [ch.junction_temp_c for ch in self.channels]

        return {
            "avg_temp_c": sum(temps) / len(temps),
            "max_temp_c": max(temps),
            "min_temp_c": min(temps),
            "gradient_c": max(temps) - min(temps),
        }

    def get_safe_power_budget(self, channel_id: int) -> float:
        """Calculate safe power budget for a channel

        Args:
            channel_id: Channel index

        Returns:
            Safe power in mW to stay below warning threshold
        """
        if 0 > channel_id or channel_id >= self.num_channels:
            return 0.0

        safe_temp = self.params.get_safe_temperature()
        current_temp = self.channels[channel_id].junction_temp_c

        temp_margin = safe_temp - current_temp
        if temp_margin <= 0:
            return 0.0

        # P = delta_T / theta_jc
        power_w = temp_margin / self.params.theta_jc
        return power_w * 1000.0

    def reset(self):
        """Reset thermal model state"""
        for ch in self.channels:
            ch.junction_temp_c = self.params.ambient_temp_c
            ch.local_temp_c = self.params.ambient_temp_c
            ch.case_temp_c = self.params.ambient_temp_c - 3.0
            ch.state = ThermalState.NORMAL
            ch.temp_history = []
            ch.power_history = []

        self.events = []
        self.emergency_active = False
        self.emergency_channels = []
        self.current_time_s = 0.0

    def __repr__(self) -> str:
        avg_temp = self.get_average_temperature()
        max_id, max_temp = self.get_max_temperature()
        return (f"ThermalModel(channels={self.num_channels}, "
                f"avg_temp={avg_temp:.1f}C, max_temp={max_temp:.1f}C @ ch{max_id})")


# Factory functions

def create_thermal_model(
    num_channels: int = 32,
    ambient_temp_c: float = 45.0,
    hbm_version: str = "hbm3"
) -> ThermalModel:
    """Create thermal model for HBM version

    Args:
        num_channels: Number of channels
        ambient_temp_c: Ambient temperature
        hbm_version: "hbm3" or "hbm4"

    Returns:
        Configured ThermalModel
    """
    params = ThermalParameters()
    params.ambient_temp_c = ambient_temp_c

    if hbm_version.lower() == "hbm4":
        params.theta_jc = params.theta_jc_hbm4
        params.max_junction_temp_c = params.max_junction_temp_hbm4_c

    return ThermalModel(
        num_channels=num_channels,
        params=params,
    )


def create_thermal_model_with_power_estimator(
    power_estimator,
    ambient_temp_c: float = 45.0
) -> Tuple[ThermalModel, List[float]]:
    """Create thermal model with initial power from estimator

    Args:
        power_estimator: HBM4PowerEstimator instance
        ambient_temp_c: Ambient temperature

    Returns:
        (ThermalModel, initial_power_list)
    """
    model = create_thermal_model(
        num_channels=power_estimator.num_channels,
        ambient_temp_c=ambient_temp_c,
    )

    # Get initial power per channel
    initial_powers = [
        power_estimator.get_channel_power_mw(i)
        for i in range(power_estimator.num_channels)
    ]

    return model, initial_powers


# Default thermal model
DEFAULT_THERMAL_MODEL = create_thermal_model()