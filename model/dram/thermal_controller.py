"""
HBM4 Thermal Controller

Provides thermal throttling and emergency handling:
- Temperature-based request throttling
- Thermal emergency handling
- Thermal history tracking
- Adaptive throttling based on thermal trends

Reference:
- JEDEC JESD270-4A HBM4 specification
- Thermal management best practices for high-bandwidth memory
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import math


class ThrottleLevel(Enum):
    """Thermal throttle levels"""
    NONE = 0           # No throttling
    LIGHT = 1          # Light throttling (10-25% reduction)
    MODERATE = 2       # Moderate throttling (25-50% reduction)
    HEAVY = 3          # Heavy throttling (50-75% reduction)
    CRITICAL = 4       # Critical throttling (75-90% reduction)
    EMERGENCY = 5      # Emergency mode (90-100% reduction)


class ThermalState(Enum):
    """Thermal state machine states"""
    NORMAL = "normal"              # Temperature within limits
    CAUTION = "caution"            # Temperature approaching threshold
    THROTTLING = "throttling"      # Actively throttling
    CRITICAL = "critical"          # Critical temperature
    EMERGENCY = "emergency"        # Emergency shutdown
    RECOVERING = "recovering"      # Returning to normal


@dataclass
class ThrottleProfile:
    """Throttling profile configuration"""
    level: ThrottleLevel

    # Bandwidth reduction factors
    bandwidth_reduction: float = 0.0      # Fraction to reduce (0-1)
    priority_reduction: float = 0.0       # Priority queue reduction

    # Timing
    min_duration_cycles: int = 100        # Minimum throttle duration
    cooldown_cycles: int = 50             # Cooldown after throttle

    # HBM4-specific throttling
    disable_channels: int = 0             # Number of channels to disable
    reduce_data_rate: bool = False        # Reduce data rate
    increase_precharge_gap: bool = False  # Insert gaps between commands

    @classmethod
    def get_profile(cls, level: ThrottleLevel) -> 'ThrottleProfile':
        """Get predefined throttle profile

        Args:
            level: Throttle level

        Returns:
            ThrottleProfile for the level
        """
        profiles = {
            ThrottleLevel.NONE: cls(
                level=ThrottleLevel.NONE,
                bandwidth_reduction=0.0,
                priority_reduction=0.0,
            ),
            ThrottleLevel.LIGHT: cls(
                level=ThrottleLevel.LIGHT,
                bandwidth_reduction=0.15,
                priority_reduction=0.1,
            ),
            ThrottleLevel.MODERATE: cls(
                level=ThrottleLevel.MODERATE,
                bandwidth_reduction=0.35,
                priority_reduction=0.25,
            ),
            ThrottleLevel.HEAVY: cls(
                level=ThrottleLevel.HEAVY,
                bandwidth_reduction=0.60,
                priority_reduction=0.50,
                disable_channels=8,
            ),
            ThrottleLevel.CRITICAL: cls(
                level=ThrottleLevel.CRITICAL,
                bandwidth_reduction=0.80,
                priority_reduction=0.75,
                disable_channels=16,
                increase_precharge_gap=True,
            ),
            ThrottleLevel.EMERGENCY: cls(
                level=ThrottleLevel.EMERGENCY,
                bandwidth_reduction=0.95,
                priority_reduction=0.90,
                disable_channels=24,
                reduce_data_rate=True,
                increase_precharge_gap=True,
                min_duration_cycles=500,
            ),
        }
        return profiles.get(level, profiles[ThrottleLevel.NONE])


@dataclass
class ThermalHistoryEntry:
    """Single entry in thermal history"""
    timestamp_ns: int
    temperature_c: float
    throttle_level: ThrottleLevel
    state: ThermalState
    power_mw: float
    bandwidth_reduction: float


@dataclass
class ThermalController:
    """HBM4 Thermal Throttling Controller

    Implements temperature-based throttling with:
    - Multi-level throttling based on temperature
    - Thermal history tracking
    - Emergency handling
    - Adaptive throttling based on trends
    """
    # Configuration
    max_junction_temp_c: float = 85.0
    throttle_threshold_c: float = 75.0
    caution_threshold_c: float = 65.0
    critical_threshold_c: float = 80.0
    emergency_threshold_c: float = 90.0
    shutdown_temp_c: float = 95.0

    # Thermal parameters
    thermal_resistance_c_per_w: float = 20.0  # C/W
    thermal_time_constant_ms: float = 100.0   # Thermal time constant

    # Throttling configuration
    throttle_hysteresis_c: float = 2.0        # Hysteresis for throttle release
    state_change_cooldown_cycles: int = 100   # Cooldown between state changes

    # State
    current_state: ThermalState = ThermalState.NORMAL
    current_throttle_level: ThrottleLevel = ThrottleLevel.NONE
    current_temperature_c: float = 45.0

    # History tracking
    thermal_history: List[ThermalHistoryEntry] = field(default_factory=list)
    throttle_events: List[Dict] = field(default_factory=list)
    state_change_count: int = 0

    # Throttle timing
    throttle_start_time_ns: int = 0
    throttle_duration_cycles: int = 0
    last_state_change_ns: int = 0

    # Callbacks
    on_throttle_start: Optional[Callable] = None
    on_throttle_end: Optional[Callable] = None
    on_emergency: Optional[Callable] = None

    # Statistics
    total_throttle_time_cycles: int = 0
    throttle_cycles_by_level: Dict[ThrottleLevel, int] = field(default_factory=lambda: {
        level: 0 for level in ThrottleLevel
    })

    def _calculate_throttle_level(self, temperature_c: float) -> ThrottleLevel:
        """Calculate throttle level based on temperature

        Args:
            temperature_c: Current temperature (C)

        Returns:
            ThrottleLevel for the temperature
        """
        # Hysteresis: use different thresholds for increasing vs decreasing temp
        margin = temperature_c - self.throttle_threshold_c

        if temperature_c >= self.emergency_threshold_c:
            return ThrottleLevel.EMERGENCY
        elif temperature_c >= self.critical_threshold_c:
            return ThrottleLevel.CRITICAL
        elif temperature_c >= self.throttle_threshold_c:
            return ThrottleLevel.HEAVY if temperature_c >= 78.0 else ThrottleLevel.MODERATE
        elif temperature_c >= self.caution_threshold_c:
            # Check if we're rising or falling
            if self.current_state in [ThermalState.THROTTLING, ThermalState.CAUTION]:
                return ThrottleLevel.LIGHT
            return ThrottleLevel.NONE
        else:
            return ThrottleLevel.NONE

    def _calculate_throttle_level_with_hysteresis(
        self,
        temperature_c: float,
        prev_temp_c: float
    ) -> Tuple[ThrottleLevel, bool]:
        """Calculate throttle level with hysteresis

        Args:
            temperature_c: Current temperature
            prev_temp_c: Previous temperature

        Returns:
            Tuple of (throttle_level, is_rising)
        """
        is_rising = temperature_c > prev_temp_c
        level = self._calculate_throttle_level(temperature_c)

        # Apply hysteresis for throttle release
        if not is_rising and level == ThrottleLevel.NONE:
            # Check if we should maintain throttle due to hysteresis
            if self.current_throttle_level != ThrottleLevel.NONE:
                if temperature_c < (self.throttle_threshold_c - self.throttle_hysteresis_c):
                    # Safe to release throttle
                    return level, is_rising
                else:
                    # Maintain at least light throttle
                    return ThrottleLevel.LIGHT, is_rising

        return level, is_rising

    def update(self, temperature_c: float, time_ns: int, power_mw: float = 0.0) -> ThrottleLevel:
        """Update thermal state and return throttle level

        Args:
            temperature_c: Current temperature (C)
            time_ns: Current simulation time (ns)
            power_mw: Current power consumption (mW)

        Returns:
            Current throttle level
        """
        prev_temp = self.current_temperature_c
        prev_state = self.current_state
        prev_level = self.current_throttle_level

        self.current_temperature_c = temperature_c

        # Check for emergency conditions
        if temperature_c >= self.shutdown_temp_c:
            self.current_state = ThermalState.EMERGENCY
            self.current_throttle_level = ThrottleLevel.EMERGENCY
            if self.on_emergency:
                self.on_emergency(temperature_c, time_ns)

        # Calculate new throttle level with hysteresis
        new_level, is_rising = self._calculate_throttle_level_with_hysteresis(
            temperature_c, prev_temp
        )

        # Update state machine
        self._update_state_machine(temperature_c, new_level, is_rising)

        # Check for state transitions
        if self.current_state != prev_state or self.current_throttle_level != prev_level:
            self._handle_state_transition(prev_state, prev_level, time_ns)

        # Update history
        self._record_history(time_ns, temperature_c, power_mw)

        return self.current_throttle_level

    def _update_state_machine(self, temperature_c: float, level: ThrottleLevel, is_rising: bool):
        """Update internal state machine

        Args:
            temperature_c: Current temperature
            level: Calculated throttle level
            is_rising: Whether temperature is rising
        """
        level_val = level.value
        # State transitions based on temperature and throttle level
        if level_val == ThrottleLevel.EMERGENCY.value:
            self.current_state = ThermalState.EMERGENCY
            self.current_throttle_level = ThrottleLevel.EMERGENCY
        elif level_val == ThrottleLevel.CRITICAL.value:
            self.current_state = ThermalState.CRITICAL
            self.current_throttle_level = level
        elif level_val >= ThrottleLevel.MODERATE.value:
            self.current_state = ThermalState.THROTTLING
            self.current_throttle_level = level
        elif level_val == ThrottleLevel.LIGHT.value:
            self.current_state = ThermalState.CAUTION
            self.current_throttle_level = level
        elif level_val == ThrottleLevel.NONE.value:
            if self.current_state in [ThermalState.THROTTLING, ThermalState.CAUTION]:
                self.current_state = ThermalState.RECOVERING
                self.current_throttle_level = ThrottleLevel.NONE
            elif self.current_state == ThermalState.RECOVERING:
                # Keep recovering until we hit caution threshold
                if temperature_c < self.caution_threshold_c - self.throttle_hysteresis_c:
                    self.current_state = ThermalState.NORMAL
            else:
                self.current_state = ThermalState.NORMAL
                self.current_throttle_level = ThrottleLevel.NONE

    def _handle_state_transition(
        self,
        prev_state: ThermalState,
        prev_level: ThrottleLevel,
        time_ns: int
    ):
        """Handle state transition events

        Args:
            prev_state: Previous state
            prev_level: Previous throttle level
            time_ns: Current time (ns)
        """
        self.state_change_count += 1
        self.last_state_change_ns = time_ns

        # Record throttle event
        event = {
            "time_ns": time_ns,
            "from_state": prev_state.value,
            "to_state": self.current_state.value,
            "from_level": prev_level.value,
            "to_level": self.current_throttle_level.value,
            "temperature_c": self.current_temperature_c,
        }
        self.throttle_events.append(event)

        # Trigger callbacks
        if (prev_level == ThrottleLevel.NONE and
            self.current_throttle_level != ThrottleLevel.NONE):
            if self.on_throttle_start:
                self.on_throttle_start(
                    self.current_throttle_level,
                    self.current_temperature_c,
                    time_ns
                )
        elif (prev_level != ThrottleLevel.NONE and
              self.current_throttle_level == ThrottleLevel.NONE):
            if self.on_throttle_end:
                self.on_throttle_end(time_ns, self.total_throttle_time_cycles)

    def _record_history(self, time_ns: int, temperature_c: float, power_mw: float):
        """Record thermal history entry

        Args:
            time_ns: Current time (ns)
            temperature_c: Temperature (C)
            power_mw: Power consumption (mW)
        """
        entry = ThermalHistoryEntry(
            timestamp_ns=time_ns,
            temperature_c=temperature_c,
            throttle_level=self.current_throttle_level,
            state=self.current_state,
            power_mw=power_mw,
            bandwidth_reduction=ThrottleProfile.get_profile(
                self.current_throttle_level
            ).bandwidth_reduction,
        )
        self.thermal_history.append(entry)

        # Keep history bounded
        if len(self.thermal_history) > MAX_THERMAL_HISTORY:
            self.thermal_history = self.thermal_history[-MAX_THERMAL_HISTORY:]

    def should_throttle_request(self, request_priority: int = 0) -> bool:
        """Determine if a request should be throttled

        Args:
            request_priority: Request priority (higher = more important)

        Returns:
            True if request should be throttled/delayed
        """
        if self.current_throttle_level == ThrottleLevel.NONE:
            return False

        profile = ThrottleProfile.get_profile(self.current_throttle_level)

        # Higher priority requests are throttled less
        priority_threshold = int(10 * profile.priority_reduction)
        if request_priority >= priority_threshold:
            return False

        # Random throttle based on bandwidth reduction
        import random
        if random.random() < profile.bandwidth_reduction:
            return True

        return False

    def get_allowed_bandwidth_fraction(self) -> float:
        """Get fraction of bandwidth that is allowed

        Returns:
            Fraction (0-1) of bandwidth to allow
        """
        profile = ThrottleProfile.get_profile(self.current_throttle_level)
        return 1.0 - profile.bandwidth_reduction

    def get_throttle_statistics(self) -> Dict:
        """Get throttle statistics

        Returns:
            Dictionary with throttle statistics
        """
        total_events = len(self.throttle_events)
        throttle_events = sum(
            1 for e in self.throttle_events
            if e["from_level"] == 0 and e["to_level"] > 0
        )

        return {
            "total_state_changes": self.state_change_count,
            "total_throttle_events": throttle_events,
            "total_throttle_time_cycles": self.total_throttle_time_cycles,
            "throttle_cycles_by_level": {
                level.value: count
                for level, count in self.throttle_cycles_by_level.items()
            },
            "current_state": self.current_state.value,
            "current_throttle_level": self.current_throttle_level.value,
            "current_temperature_c": self.current_temperature_c,
            "thermal_margin_c": self.max_junction_temp_c - self.current_temperature_c,
            "history_length": len(self.thermal_history),
        }

    def get_temperature_prediction(self, cycles_ahead: int, power_mw: float) -> float:
        """Predict temperature N cycles ahead

        Uses thermal RC model for prediction:
        T(t) = T_ambient + P * R * (1 - exp(-t/tau))

        Args:
            cycles_ahead: Cycles to predict ahead
            power_mw: Assumed power consumption (mW)

        Returns:
            Predicted temperature (C)
        """
        # Convert cycles to time (assuming 125ps cycle = 8 GT/s)
        tCK_ps = 125.0
        time_s = cycles_ahead * tCK_ps * 1e-12

        # Thermal model
        P_w = power_mw / 1000.0
        R = self.thermal_resistance_c_per_w
        tau = self.thermal_time_constant_ms / 1000.0  # Convert to seconds

        # Current excess temperature
        T_excess = self.current_temperature_c - 45.0  # Assume ambient 45C

        # Predicted temperature
        if tau > 0:
            T_pred = 45.0 + T_excess * math.exp(-time_s / tau) + P_w * R * (1 - math.exp(-time_s / tau))
        else:
            T_pred = self.current_temperature_c + P_w * R * (time_s / tau if tau > 0 else 0)

        return T_pred

    def get_safe_power_level(self, target_temp_c: float) -> float:
        """Calculate safe power level for target temperature

        Args:
            target_temp_c: Target temperature (C)

        Returns:
            Safe power level (mW)
        """
        if target_temp_c >= self.max_junction_temp_c:
            return 0.0

        T_excess = target_temp_c - 45.0  # Assume ambient 45C
        R = self.thermal_resistance_c_per_w

        if R <= 0:
            return 10000.0  # Return arbitrary high value

        return (T_excess / R) * 1000.0  # Convert to mW

    def get_recommended_throttle_level(self) -> ThrottleLevel:
        """Get recommended throttle level based on thermal trend

        Returns:
            Recommended ThrottleLevel
        """
        if len(self.thermal_history) < 2:
            return self.current_throttle_level

        # Calculate temperature rate
        recent = self.thermal_history[-10:]
        if len(recent) < 2:
            return self.current_throttle_level

        time_delta_s = (recent[-1].timestamp_ns - recent[0].timestamp_ns) * 1e-9
        temp_delta_c = recent[-1].temperature_c - recent[0].temperature_c

        if time_delta_s <= 0:
            return self.current_throttle_level

        rate_c_per_sec = temp_delta_c / time_delta_s

        # Predict temperature in 1 second
        cycles_per_sec = 8e9  # 8 GT/s
        cycles_1sec = int(cycles_per_sec)
        predicted_temp = self.get_temperature_prediction(
            cycles_1sec,
            500.0  # Assume moderate power
        )

        # Recommend throttle based on prediction
        if predicted_temp >= self.emergency_threshold_c:
            return ThrottleLevel.EMERGENCY
        elif predicted_temp >= self.critical_threshold_c:
            return ThrottleLevel.CRITICAL
        elif predicted_temp >= self.throttle_threshold_c:
            return ThrottleLevel.HEAVY
        elif predicted_temp >= self.caution_threshold_c:
            return ThrottleLevel.LIGHT
        else:
            return ThrottleLevel.NONE

    def tick(self, cycles: int = 1):
        """Advance time and update throttle timing

        Args:
            cycles: Number of cycles to advance
        """
        if self.current_throttle_level != ThrottleLevel.NONE:
            self.total_throttle_time_cycles += cycles
            self.throttle_cycles_by_level[self.current_throttle_level] += cycles

    def reset(self):
        """Reset thermal controller state"""
        self.current_state = ThermalState.NORMAL
        self.current_throttle_level = ThrottleLevel.NONE
        self.current_temperature_c = 45.0
        self.thermal_history = []
        self.throttle_events = []
        self.state_change_count = 0
        self.throttle_start_time_ns = 0
        self.throttle_duration_cycles = 0
        self.last_state_change_ns = 0
        self.total_throttle_time_cycles = 0
        self.throttle_cycles_by_level = {level: 0 for level in ThrottleLevel}


# Constants
MAX_THERMAL_HISTORY = 10000


def create_default_thermal_controller() -> ThermalController:
    """Create thermal controller with default settings

    Returns:
        Configured ThermalController
    """
    return ThermalController(
        max_junction_temp_c=85.0,
        throttle_threshold_c=75.0,
        caution_threshold_c=65.0,
        critical_threshold_c=80.0,
        emergency_threshold_c=90.0,
        shutdown_temp_c=95.0,
    )


def create_aggressive_controller() -> ThermalController:
    """Create aggressive thermal controller with lower thresholds

    Returns:
        Configured ThermalController
    """
    return ThermalController(
        max_junction_temp_c=85.0,
        throttle_threshold_c=70.0,
        caution_threshold_c=60.0,
        critical_threshold_c=75.0,
        emergency_threshold_c=85.0,
        shutdown_temp_c=90.0,
    )


def create_conservative_controller() -> ThermalController:
    """Create conservative thermal controller with higher thresholds

    Returns:
        Configured ThermalController
    """
    return ThermalController(
        max_junction_temp_c=85.0,
        throttle_threshold_c=78.0,
        caution_threshold_c=68.0,
        critical_threshold_c=82.0,
        emergency_threshold_c=92.0,
        shutdown_temp_c=97.0,
    )