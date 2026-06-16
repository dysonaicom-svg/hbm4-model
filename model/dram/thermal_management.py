"""
HBM4 Thermal Management System

Global thermal management with:
- Power budget allocation based on thermal state
- Thermal-aware scheduling hints
- Multi-zone thermal coordination
- Thermal emergency protocols

Reference:
- JEDEC JESD270-4A HBM4 specification
- Thermal design for high-bandwidth memory systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, TYPE_CHECKING
from enum import Enum
import math
import random

# Import thermal components
from .thermal_sensor import (
    SensorArray,
    SensorConfiguration,
    SensorCalibration,
    SensorReading,
    SensorType,
    ThermalZone,
    ThermalSensor,
)
from .thermal_controller import (
    ThermalController,
    ThrottleLevel,
    ThermalState,
    ThrottleProfile,
    ThermalHistoryEntry,
)

if TYPE_CHECKING:
    pass


class PowerBudgetState(Enum):
    """Power budget allocation states"""
    NORMAL = "normal"              # Full power budget available
    LIMITED = "limited"            # Reduced power budget
    CRITICAL = "critical"          # Minimal power budget
    EMERGENCY = "emergency"        # Emergency power budget


@dataclass
class PowerBudget:
    """Power budget configuration and state"""
    # Budget limits (mW)
    peak_budget_mw: float = 5000.0      # Maximum power budget
    sustained_budget_mw: float = 4000.0  # Sustained power budget
    idle_budget_mw: float = 500.0       # Idle power budget

    # Current allocation
    allocated_mw: float = 0.0            # Currently allocated power
    available_mw: float = 5000.0        # Available power

    # Budget state
    state: PowerBudgetState = PowerBudgetState.NORMAL

    # Per-channel budget
    per_channel_budget_mw: float = 156.0  # Budget per channel (5W / 32 channels)

    # Reserve for thermal headroom
    thermal_reserve_mw: float = 500.0    # Reserve for temperature control

    @property
    def effective_budget_mw(self) -> float:
        """Get effective budget after reserves"""
        return self.peak_budget_mw - self.thermal_reserve_mw

    @property
    def budget_utilization(self) -> float:
        """Get budget utilization (0-1)"""
        if self.peak_budget_mw <= 0:
            return 0.0
        return self.allocated_mw / self.peak_budget_mw

    def allocate_power(self, channel_id: int, power_mw: float) -> bool:
        """Allocate power to a channel

        Args:
            channel_id: Channel index
            power_mw: Power to allocate (mW)

        Returns:
            True if allocation successful
        """
        if power_mw > self.available_mw:
            return False
        self.allocated_mw += power_mw
        self.available_mw -= power_mw
        return True

    def release_power(self, power_mw: float):
        """Release allocated power

        Args:
            power_mw: Power to release (mW)
        """
        self.allocated_mw = max(0, self.allocated_mw - power_mw)
        self.available_mw = min(self.peak_budget_mw, self.available_mw + power_mw)

    def set_budget_state(self, state: PowerBudgetState):
        """Set budget state and adjust limits

        Args:
            state: New budget state
        """
        self.state = state
        if state == PowerBudgetState.EMERGENCY:
            self.available_mw = min(self.available_mw, 1000.0)
        elif state == PowerBudgetState.CRITICAL:
            self.available_mw = min(self.available_mw, 2000.0)
        elif state == PowerBudgetState.LIMITED:
            self.available_mw = min(self.available_mw, 3500.0)
        else:
            self.available_mw = self.peak_budget_mw - self.thermal_reserve_mw


@dataclass
class SchedulingHint:
    """Scheduling hint for memory controller"""
    action: str                      # "throttle", "defer", "allow", "priority"
    reason: str                     # Reason for the hint
    priority_adjustment: float = 0.0  # Priority adjustment (-1 to 1)
    bandwidth_fraction: float = 1.0   # Allowed bandwidth fraction
    channels_to_disable: List[int] = field(default_factory=list)
    temperature_c: float = 0.0        # Current temperature
    timestamp_ns: int = 0             # Hint timestamp


@dataclass
class ThermalZoneStatus:
    """Status of a thermal zone"""
    zone_id: int
    zone_name: str
    temperature_c: float
    max_temperature_c: float
    thermal_margin_c: float
    power_mw: float
    power_budget_mw: float
    is_throttling: bool
    throttle_level: int  # 0-5


@dataclass
class ThermalManagementConfig:
    """Configuration for thermal management"""
    # Thermal thresholds
    max_junction_temp_c: float = 85.0
    throttle_threshold_c: float = 75.0
    caution_threshold_c: float = 65.0
    critical_threshold_c: float = 80.0
    emergency_threshold_c: float = 90.0

    # Power budgets
    peak_power_budget_mw: float = 5000.0
    sustained_power_budget_mw: float = 4000.0

    # Thermal parameters
    thermal_resistance_c_per_w: float = 20.0
    ambient_temperature_c: float = 45.0

    # Control parameters
    throttle_hysteresis_c: float = 2.0
    power_adjustment_rate_mw_per_ms: float = 100.0  # Max rate of power change
    update_interval_ns: int = 1000  # Management update interval

    # Number of channels
    num_channels: int = 32


class ThermalManagementSystem:
    """HBM4 Global Thermal Management System

    Coordinates thermal behavior across all memory channels with:
    - Global power budget allocation
    - Thermal-aware scheduling hints
    - Multi-zone thermal monitoring
    - Emergency protocols
    """
    def __init__(
        self,
        config: Optional[ThermalManagementConfig] = None,
        sensors: Optional[SensorArray] = None,
        controller: Optional[ThermalController] = None,
    ):
        """Initialize thermal management system

        Args:
            config: Configuration for thermal management
            sensors: Pre-configured sensor array (optional)
            controller: Pre-configured controller (optional)
        """
        self.config = config or ThermalManagementConfig()
        self.sensors = sensors
        self.controller = controller

        # Power budget
        self.power_budget = PowerBudget(
            peak_budget_mw=self.config.peak_power_budget_mw,
            sustained_budget_mw=self.config.sustained_power_budget_mw,
        )

        # Zone temperatures
        self.zone_temperatures: Dict[ThermalZone, float] = {}
        self.zone_powers: Dict[ThermalZone, float] = {}

        # State tracking
        self.current_time_ns: int = 0
        self.thermal_emergency_active: bool = False
        self.emergency_start_ns: int = 0

        # Scheduling hints history
        self.hint_history: List[SchedulingHint] = []

        # Statistics
        self.total_throttle_events: int = 0
        self.emergency_events: int = 0
        self.power_adjustments: int = 0

        # Callbacks
        self.on_emergency: Optional[Callable] = None
        self.on_budget_change: Optional[Callable] = None

        # Initialize default components if not provided
        if self.sensors is None:
            self.sensors = SensorArray(sensor_count=8)
        if self.controller is None:
            self.controller = ThermalController(
                max_junction_temp_c=self.config.max_junction_temp_c,
                throttle_threshold_c=self.config.throttle_threshold_c,
                caution_threshold_c=self.config.caution_threshold_c,
                critical_threshold_c=self.config.critical_threshold_c,
                emergency_threshold_c=self.config.emergency_threshold_c,
            )

        # Initialize zone tracking
        for zone in ThermalZone:
            if zone not in self.zone_temperatures:
                self.zone_temperatures[zone] = self.config.ambient_temperature_c
            if zone not in self.zone_powers:
                self.zone_powers[zone] = 100.0  # Default 100mW per zone

        # Set up controller callbacks
        self.controller.on_emergency = self._handle_emergency
        self.controller.on_throttle_start = self._handle_throttle_start
        self.controller.on_throttle_end = self._handle_throttle_end

    def _handle_emergency(self, temperature_c: float, time_ns: int):
        """Handle emergency thermal condition

        Args:
            temperature_c: Emergency temperature
            time_ns: Current time
        """
        self.thermal_emergency_active = True
        self.emergency_start_ns = time_ns
        self.emergency_events += 1
        self.power_budget.set_budget_state(PowerBudgetState.EMERGENCY)

        if self.on_emergency:
            self.on_emergency(temperature_c, time_ns)

    def _handle_throttle_start(self, level: ThrottleLevel, temp_c: float, time_ns: int):
        """Handle throttle start event

        Args:
            level: Throttle level
            temp_c: Temperature at throttle start
            time_ns: Current time
        """
        self.total_throttle_events += 1

        # Adjust power budget
        if level.value >= 3:  # HEAVY or above
            self.power_budget.set_budget_state(PowerBudgetState.LIMITED)
            self.power_adjustments += 1

    def _handle_throttle_end(self, time_ns: int, duration_cycles: int):
        """Handle throttle end event

        Args:
            time_ns: Current time
            duration_cycles: Duration of throttle
        """
        if not self.thermal_emergency_active:
            self.power_budget.set_budget_state(PowerBudgetState.NORMAL)

    def update(self, zone_powers: Dict[ThermalZone, float], time_ns: int) -> SchedulingHint:
        """Update thermal management and generate scheduling hint

        Args:
            zone_powers: Power consumption per zone (mW)
            time_ns: Current simulation time (ns)

        Returns:
            SchedulingHint for memory controller
        """
        self.current_time_ns = time_ns
        self.zone_powers = zone_powers

        # Update zone temperatures
        for zone, power in zone_powers.items():
            self.zone_temperatures[zone] = self._calculate_zone_temperature(zone, power)

        # Get max temperature
        max_temp = max(self.zone_temperatures.values())

        # Update thermal controller
        throttle_level = self.controller.update(max_temp, time_ns, sum(zone_powers.values()))

        # Generate scheduling hint
        hint = self._generate_scheduling_hint(throttle_level, max_temp, time_ns)

        # Update power budget based on thermal state
        self._update_power_budget(throttle_level)

        # Record hint
        self.hint_history.append(hint)
        if len(self.hint_history) > MAX_HINT_HISTORY:
            self.hint_history = self.hint_history[-MAX_HINT_HISTORY:]

        return hint

    def _calculate_zone_temperature(self, zone: ThermalZone, power_mw: float) -> float:
        """Calculate temperature for a zone

        Args:
            zone: Thermal zone
            power_mw: Power consumption (mW)

        Returns:
            Temperature (C)
        """
        # Use thermal model
        power_w = power_mw / 1000.0
        ambient = self.config.ambient_temperature_c
        R = self.config.thermal_resistance_c_per_w

        # Temperature rise from power
        T_rise = power_w * R

        # Zone-specific adjustments
        zone_adjustments = {
            ThermalZone.PACKAGE_CORE: 5.0,       # Higher in core
            ThermalZone.LOGIC_BASE_DIE: 3.0,       # Logic die runs hot
            ThermalZone.DRAM_BANK_0: 0.0,
            ThermalZone.DRAM_BANK_1: 0.0,
            ThermalZone.DRAM_BANK_2: 0.0,
            ThermalZone.DRAM_BANK_3: 0.0,
            ThermalZone.PACKAGE_EDGE: -2.0,       # Cooler at edges
            ThermalZone.SUBSTRATE: -5.0,          # Cooler in substrate
        }

        return ambient + T_rise + zone_adjustments.get(zone, 0.0)

    def _generate_scheduling_hint(
        self,
        throttle_level: ThrottleLevel,
        temperature_c: float,
        time_ns: int
    ) -> SchedulingHint:
        """Generate scheduling hint based on thermal state

        Args:
            throttle_level: Current throttle level
            temperature_c: Current temperature
            time_ns: Current time

        Returns:
            SchedulingHint for controller
        """
        if throttle_level == ThrottleLevel.NONE:
            action = "allow"
            reason = "thermal_conditions_normal"
            bandwidth = 1.0
            priority_adj = 0.0
            channels = []
        elif throttle_level == ThrottleLevel.LIGHT:
            action = "throttle"
            reason = "thermal_caution"
            bandwidth = 0.85
            priority_adj = 0.1
            channels = []
        elif throttle_level == ThrottleLevel.MODERATE:
            action = "throttle"
            reason = "thermal_throttling_active"
            bandwidth = 0.65
            priority_adj = 0.25
            channels = []
        elif throttle_level == ThrottleLevel.HEAVY:
            action = "throttle"
            reason = "heavy_thermal_throttling"
            bandwidth = 0.40
            priority_adj = 0.5
            channels = list(range(8))  # Disable 8 channels
        elif throttle_level == ThrottleLevel.CRITICAL:
            action = "defer"
            reason = "critical_thermal_condition"
            bandwidth = 0.20
            priority_adj = 0.75
            channels = list(range(16))  # Disable 16 channels
        else:  # EMERGENCY
            action = "defer"
            reason = "emergency_thermal_shutdown"
            bandwidth = 0.05
            priority_adj = 0.9
            channels = list(range(24))  # Disable 24 channels

        return SchedulingHint(
            action=action,
            reason=reason,
            priority_adjustment=priority_adj,
            bandwidth_fraction=bandwidth,
            channels_to_disable=channels,
            temperature_c=temperature_c,
            timestamp_ns=time_ns,
        )

    def _update_power_budget(self, throttle_level: ThrottleLevel):
        """Update power budget based on thermal state

        Args:
            throttle_level: Current throttle level
        """
        prev_state = self.power_budget.state

        if throttle_level == ThrottleLevel.NONE:
            self.power_budget.set_budget_state(PowerBudgetState.NORMAL)
        elif throttle_level in [ThrottleLevel.LIGHT, ThrottleLevel.MODERATE]:
            self.power_budget.set_budget_state(PowerBudgetState.LIMITED)
        elif throttle_level in [ThrottleLevel.HEAVY, ThrottleLevel.CRITICAL]:
            self.power_budget.set_budget_state(PowerBudgetState.CRITICAL)
        else:  # EMERGENCY
            self.power_budget.set_budget_state(PowerBudgetState.EMERGENCY)

        if prev_state != self.power_budget.state and self.on_budget_change:
            self.on_budget_change(self.power_budget.state, self.power_budget)

    def get_power_allocation(self, num_active_channels: int) -> Dict[int, float]:
        """Get power allocation per channel

        Args:
            num_active_channels: Number of active channels

        Returns:
            Dictionary mapping channel ID to power allocation (mW)
        """
        if num_active_channels <= 0:
            return {}

        # Calculate per-channel allocation
        available = self.power_budget.available_mw
        per_channel = min(
            available / num_active_channels,
            self.power_budget.per_channel_budget_mw
        )

        return {i: per_channel for i in range(num_active_channels)}

    def should_defer_request(
        self,
        request_power_mw: float,
        request_priority: int
    ) -> bool:
        """Determine if request should be deferred

        Args:
            request_power_mw: Power required for request
            request_priority: Request priority (higher = more important)

        Returns:
            True if request should be deferred
        """
        # Check if we have budget
        if request_power_mw > self.power_budget.available_mw:
            return True

        # Check thermal state
        if self.thermal_emergency_active:
            # Only allow critical requests
            return request_priority < 8

        # Check throttle level
        hint = self._get_latest_hint()
        if hint:
            if hint.action == "defer":
                return request_priority < 7
            elif hint.action == "throttle":
                return random.random() > hint.bandwidth_fraction

        return False

    def _get_latest_hint(self) -> Optional[SchedulingHint]:
        """Get latest scheduling hint"""
        if self.hint_history:
            return self.hint_history[-1]
        return None

    def get_zone_status(self) -> List[ThermalZoneStatus]:
        """Get status of all thermal zones

        Returns:
            List of ThermalZoneStatus
        """
        statuses = []
        throttle_level = self.controller.current_throttle_level.value

        for zone, temp in self.zone_temperatures.items():
            margin = self.config.max_junction_temp_c - temp
            power = self.zone_powers.get(zone, 0.0)
            is_throttling = temp >= self.config.throttle_threshold_c

            statuses.append(ThermalZoneStatus(
                zone_id=zone.value,
                zone_name=zone.value,
                temperature_c=temp,
                max_temperature_c=self.config.max_junction_temp_c,
                thermal_margin_c=margin,
                power_mw=power,
                power_budget_mw=self.power_budget.per_channel_budget_mw,
                is_throttling=is_throttling,
                throttle_level=throttle_level if is_throttling else 0,
            ))

        return statuses

    def get_max_temperature(self) -> float:
        """Get maximum temperature across all zones"""
        if not self.zone_temperatures:
            return self.config.ambient_temperature_c
        return max(self.zone_temperatures.values())

    def get_min_thermal_margin(self) -> float:
        """Get minimum thermal margin across all zones"""
        return self.config.max_junction_temp_c - self.get_max_temperature()

    def get_system_summary(self) -> Dict:
        """Get thermal management system summary

        Returns:
            Dictionary with system state summary
        """
        return {
            "current_time_ns": self.current_time_ns,
            "max_temperature_c": self.get_max_temperature(),
            "min_thermal_margin_c": self.get_min_thermal_margin(),
            "power_budget": {
                "state": self.power_budget.state.value,
                "allocated_mw": self.power_budget.allocated_mw,
                "available_mw": self.power_budget.available_mw,
                "utilization": self.power_budget.budget_utilization,
            },
            "thermal_state": self.controller.current_state.value,
            "throttle_level": self.controller.current_throttle_level.value,
            "emergency_active": self.thermal_emergency_active,
            "statistics": {
                "total_throttle_events": self.total_throttle_events,
                "emergency_events": self.emergency_events,
                "power_adjustments": self.power_adjustments,
            },
            "latest_hint": {
                "action": self.hint_history[-1].action if self.hint_history else None,
                "reason": self.hint_history[-1].reason if self.hint_history else None,
                "bandwidth_fraction": self.hint_history[-1].bandwidth_fraction if self.hint_history else 1.0,
            } if self.hint_history else None,
        }

    def reset(self):
        """Reset thermal management system"""
        self.current_time_ns = 0
        self.thermal_emergency_active = False
        self.emergency_start_ns = 0
        self.hint_history = []
        self.total_throttle_events = 0
        self.emergency_events = 0
        self.power_adjustments = 0

        # Reset components
        if self.sensors:
            self.sensors.reset_all()
        if self.controller:
            self.controller.reset()

        # Reset zone temperatures
        for zone in self.zone_temperatures:
            self.zone_temperatures[zone] = self.config.ambient_temperature_c
            self.zone_powers[zone] = 100.0

        # Reset power budget
        self.power_budget = PowerBudget(
            peak_budget_mw=self.config.peak_power_budget_mw,
            sustained_budget_mw=self.config.sustained_power_budget_mw,
        )


# Constants
MAX_HINT_HISTORY = 1000


def create_default_thermal_management() -> ThermalManagementSystem:
    """Create thermal management system with default settings

    Returns:
        Configured ThermalManagementSystem
    """
    config = ThermalManagementConfig()
    return ThermalManagementSystem(config=config)


def create_thermal_management_for_hbm4() -> ThermalManagementSystem:
    """Create thermal management system configured for HBM4

    Returns:
        ThermalManagementSystem for HBM4
    """
    config = ThermalManagementConfig(
        max_junction_temp_c=85.0,
        throttle_threshold_c=75.0,
        caution_threshold_c=65.0,
        critical_threshold_c=80.0,
        emergency_threshold_c=90.0,
        peak_power_budget_mw=5000.0,
        sustained_power_budget_mw=4000.0,
        thermal_resistance_c_per_w=20.0,
        ambient_temperature_c=45.0,
        num_channels=32,
    )
    return ThermalManagementSystem(config=config)


def create_thermal_management_for_hbm3() -> ThermalManagementSystem:
    """Create thermal management system configured for HBM3

    Returns:
        ThermalManagementSystem for HBM3
    """
    config = ThermalManagementConfig(
        max_junction_temp_c=85.0,
        throttle_threshold_c=72.0,
        caution_threshold_c=62.0,
        critical_threshold_c=78.0,
        emergency_threshold_c=88.0,
        peak_power_budget_mw=4000.0,
        sustained_power_budget_mw=3200.0,
        thermal_resistance_c_per_w=18.0,
        ambient_temperature_c=45.0,
        num_channels=16,
    )
    return ThermalManagementSystem(config=config)