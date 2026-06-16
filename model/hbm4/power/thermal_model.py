"""
HBM4 Enhanced Thermal Model

Provides comprehensive thermal modeling for the HBM4 logic base die components
with advanced throttling policies and temperature-based performance adjustment.

Key features:
- Hotspot proxies (controller cluster, D2D PHY, TSV PHY, ECC/RAS, clocking)
- Thermal resistance modeling (R_theta_jc, R_theta_ca)
- Advanced thermal throttling policy with hysteresis
- Adaptive throttling based on thermal history
- PDN voltage and frequency scaling
- Temperature-based performance adjustment
- Multi-component temperature tracking
- Integration with PowerEstimator for dynamic power

Based on:
- JEDEC JESD270-4A HBM4 specification
- Hotspot thermal simulation models
- Synopsys HBM4 Controller IP thermal data
- Multi-agent research findings (2026-06-15)

Thermal model overview:
    Power consumption from each component (P_i)
        |
        v
    [Thermal Resistance R_th] --> Temperature rise (delta_T = P * R_th)
        |
        v
    [Heat spreading in base die]
        |
        v
    [Package thermal resistance]
        |
        v
    [Ambient temperature]

Advanced Throttling Policy:
- Warning threshold: ~85C (begin monitoring)
- Throttle threshold: ~95C (reduce frequency/voltage)
- Critical threshold: ~105C (aggressive throttling)
- Shutdown threshold: ~110C (emergency thermal shutdown)
- Hysteresis: Prevent oscillation around thresholds
- Adaptive throttling: Adjust based on thermal history
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import math
import time


class ThrottleLevel(Enum):
    """Thermal throttling levels"""
    NONE = "none"           # Normal operation
    WARNING = "warning"     # Temperature approaching throttle
    THROTTLE = "throttle"  # Frequency/voltage reduction active
    CRITICAL = "critical"   # Aggressive throttling
    SHUTDOWN = "shutdown"  # Emergency thermal shutdown


class PDNVoltageMode(Enum):
    """PDN voltage operating points"""
    NOMINAL = "nominal"      # 0.9V nominal
    PERFORMANCE = "perf"     # 1.0V boosted performance
    LOW_POWER = "low_pwr"    # 0.8V reduced power
    ULTRA_LOW = "ultra_low"  # 0.65V ultra low power


class ThrottleDirection(Enum):
    """Direction of throttle change"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


@dataclass
class TemperatureThresholds:
    """Temperature thresholds for thermal management (in Celsius)"""
    warning: float = 85.0      # Begin monitoring
    throttle: float = 95.0      # Start throttling
    critical: float = 105.0     # Aggressive throttling
    shutdown: float = 110.0     # Emergency shutdown

    # Hysteresis margins to prevent oscillation
    hysteresis_warning: float = 3.0
    hysteresis_throttle: float = 5.0
    hysteresis_critical: float = 5.0
    hysteresis_shutdown: float = 3.0

    def get_throttle_level(self, temperature: float,
                          current_level: Optional[ThrottleLevel] = None,
                          direction: ThrottleDirection = ThrottleDirection.STABLE) -> ThrottleLevel:
        """Determine throttle level based on temperature with hysteresis

        Args:
            temperature: Current temperature
            current_level: Current throttle level (for hysteresis)
            direction: Direction of temperature change

        Returns:
            ThrottleLevel enum value
        """
        if temperature >= self.shutdown:
            return ThrottleLevel.SHUTDOWN
        elif temperature >= self.critical:
            # For decreasing temp, require more hysteresis to drop level
            if current_level == ThrottleLevel.CRITICAL and direction == ThrottleDirection.DECREASING:
                if temperature >= self.critical - self.hysteresis_critical:
                    return ThrottleLevel.CRITICAL
            return ThrottleLevel.CRITICAL
        elif temperature >= self.throttle:
            if current_level in [ThrottleLevel.CRITICAL, ThrottleLevel.THROTTLE] and direction == ThrottleDirection.DECREASING:
                if temperature >= self.throttle - self.hysteresis_throttle:
                    return ThrottleLevel.THROTTLE
            return ThrottleLevel.THROTTLE
        elif temperature >= self.warning:
            if current_level in [ThrottleLevel.THROTTLE, ThrottleLevel.WARNING] and direction == ThrottleDirection.DECREASING:
                if temperature >= self.warning - self.hysteresis_warning:
                    return ThrottleLevel.WARNING
            return ThrottleLevel.WARNING
        return ThrottleLevel.NONE


@dataclass
class ThermalResistance:
    """Thermal resistance parameters (in C/W)"""
    r_jc: float = 0.5      # Junction-to-case resistance
    r_jb: float = 1.0      # Junction-to-board resistance
    r_ca: float = 10.0    # Case-to-ambient resistance (package level)
    r_sp: float = 2.0     # Spreading resistance within die

    @property
    def total(self) -> float:
        """Total thermal resistance"""
        return self.r_jc + self.r_jb + self.r_ca

    def get_temperature_rise(self, power_mw: float) -> float:
        """Calculate temperature rise from power dissipation

        Args:
            power_mw: Power in milliwatts

        Returns:
            Temperature rise in Celsius
        """
        return power_mw * self.total / 1000.0


@dataclass
class ComponentTemperatures:
    """Temperature readings for each component (in Celsius)"""
    timestamp_ns: int = 0
    ambient: float = 25.0          # Ambient temperature
    case: float = 25.0              # Package case temperature
    die: float = 25.0               # Die temperature
    controller_cluster: float = 25.0
    d2d_phy: float = 25.0
    tsv_phy: float = 25.0
    ecc_ras: float = 25.0
    clocking: float = 25.0
    phy_interface: float = 25.0

    # Temperature rate of change (C/s)
    rate_of_change: float = 0.0
    # Thermal gradient across die
    thermal_gradient: float = 0.0

    @property
    def max_temperature(self) -> float:
        """Maximum temperature across all components"""
        return max(
            self.die,
            self.controller_cluster,
            self.d2d_phy,
            self.tsv_phy,
            self.ecc_ras,
            self.clocking,
            self.phy_interface,
        )

    @property
    def min_temperature(self) -> float:
        """Minimum temperature across all components"""
        return min(
            self.controller_cluster,
            self.d2d_phy,
            self.tsv_phy,
            self.ecc_ras,
            self.clocking,
            self.phy_interface,
        )

    @property
    def average_temperature(self) -> float:
        """Average temperature across all components"""
        return (
            self.controller_cluster +
            self.d2d_phy +
            self.tsv_phy +
            self.ecc_ras +
            self.clocking +
            self.phy_interface
        ) / 6.0

    @property
    def hotspot_temperature(self) -> float:
        """Hottest component temperature"""
        return max(
            self.controller_cluster,
            self.d2d_phy,
            self.tsv_phy,
        )


@dataclass
class HotspotConfig:
    """Configuration for each hotspot component"""
    # Thermal resistance from junction to die surface (C/W)
    r_junction: float = 1.0
    # Size factor (relative to full die)
    size_factor: float = 0.1
    # Power density factor (higher for dense logic)
    power_density: float = 1.0
    # Thermal coupling to adjacent hotspots
    coupling_factor: float = 0.05
    # Maximum allowed temperature
    max_temp_c: float = 110.0


@dataclass
class PDNOperatingPoint:
    """PDN voltage operating point configuration"""
    mode: PDNVoltageMode
    voltage_mv: float
    max_current_ma: float
    max_power_mw: float
    thermal_limit_c: float = 95.0
    frequency_scale: float = 1.0  # Frequency multiplier
    voltage_scale: float = 1.0    # Voltage multiplier from nominal


@dataclass
class ThrottleState:
    """Current thermal throttling state"""
    level: ThrottleLevel = ThrottleLevel.NONE
    active: bool = False
    throttle_factor: float = 1.0     # Frequency/voltage throttle multiplier
    pdn_mode: PDNVoltageMode = PDNVoltageMode.NOMINAL
    time_in_throttle_ns: int = 0
    throttle_count: int = 0
    max_temperature_reached: float = 0.0
    # Hysteresis state
    hysteresis_locked: bool = False
    hysteresis_lock_temp: float = 0.0
    # Adaptive throttling state
    adaptive_count: int = 0
    thermal_trend: float = 0.0  # Rate of temperature change


@dataclass
class PerformanceAdjustment:
    """Temperature-based performance adjustment parameters"""
    frequency_scale: float = 1.0     # Current frequency scale (0-1)
    voltage_scale: float = 1.0       # Current voltage scale (0-1)
    bandwidth_scale: float = 1.0      # Effective bandwidth scale
    latency_penalty: float = 0.0       # Latency increase factor

    # Performance degradation thresholds
    freq_degrade_start_c: float = 90.0   # Start frequency degradation
    volt_degrade_start_c: float = 95.0    # Start voltage reduction

    # Degradation rates (% per C above threshold)
    freq_degrade_rate: float = 2.0    # 2% per C
    volt_degrade_rate: float = 1.5     # 1.5% per C

    @property
    def effective_bandwidth(self) -> float:
        """Effective bandwidth considering throttling"""
        return self.bandwidth_scale

    def apply_temperature(self, temperature: float) -> 'PerformanceAdjustment':
        """Calculate performance adjustment based on temperature

        Args:
            temperature: Current temperature in Celsius

        Returns:
            New PerformanceAdjustment with adjusted values
        """
        adjusted = PerformanceAdjustment(
            frequency_scale=self.frequency_scale,
            voltage_scale=self.voltage_scale,
            bandwidth_scale=self.bandwidth_scale,
            latency_penalty=self.latency_penalty,
        )

        # Frequency degradation
        if temperature > self.freq_degrade_start_c:
            delta_t = temperature - self.freq_degrade_start_c
            adjusted.frequency_scale = max(0.5, 1.0 - (delta_t * self.freq_degrade_rate / 100.0))

        # Voltage degradation
        if temperature > self.volt_degrade_start_c:
            delta_t = temperature - self.volt_degrade_start_c
            adjusted.voltage_scale = max(0.6, 1.0 - (delta_t * self.volt_degrade_rate / 100.0))

        # Bandwidth scales with frequency
        adjusted.bandwidth_scale = adjusted.frequency_scale

        # Latency penalty increases as voltage drops
        if adjusted.voltage_scale < 1.0:
            adjusted.latency_penalty = (1.0 - adjusted.voltage_scale) * 0.5

        return adjusted


@dataclass
class ThrottlePolicy:
    """Configurable thermal throttling policy"""
    # Throttle factor per level
    throttle_factors: Dict[ThrottleLevel, float] = field(default_factory=lambda: {
        ThrottleLevel.NONE: 1.0,
        ThrottleLevel.WARNING: 0.95,
        ThrottleLevel.THROTTLE: 0.75,
        ThrottleLevel.CRITICAL: 0.5,
        ThrottleLevel.SHUTDOWN: 0.0,
    })

    # PDN mode per throttle level
    pdn_modes: Dict[ThrottleLevel, PDNVoltageMode] = field(default_factory=lambda: {
        ThrottleLevel.NONE: PDNVoltageMode.PERFORMANCE,
        ThrottleLevel.WARNING: PDNVoltageMode.NOMINAL,
        ThrottleLevel.THROTTLE: PDNVoltageMode.LOW_POWER,
        ThrottleLevel.CRITICAL: PDNVoltageMode.ULTRA_LOW,
        ThrottleLevel.SHUTDOWN: PDNVoltageMode.ULTRA_LOW,
    })

    # Enable adaptive throttling based on thermal history
    enable_adaptive: bool = True
    # Minimum time in throttle before considering recovery (ns)
    min_throttle_time_ns: int = 1000000
    # Temperature rate threshold for accelerated throttling (C/s)
    rapid_rise_threshold_cps: float = 10.0

    def get_throttle_factor(self, level: ThrottleLevel,
                          thermal_rate: float = 0.0) -> float:
        """Get throttle factor with optional adaptive adjustment

        Args:
            level: Current throttle level
            thermal_rate: Rate of temperature change (C/s)

        Returns:
            Throttle factor (0-1)
        """
        factor = self.throttle_factors.get(level, 1.0)

        if self.enable_adaptive and thermal_rate > self.rapid_rise_threshold_cps:
            # Accelerate throttling if temperature rising rapidly
            factor *= 0.9  # Additional 10% reduction

        return factor

    def get_pdn_mode(self, level: ThrottleLevel) -> PDNVoltageMode:
        """Get appropriate PDN mode for throttle level"""
        return self.pdn_modes.get(level, PDNVoltageMode.NOMINAL)


@dataclass
class ThermalStatistics:
    """Runtime thermal statistics"""
    samples: int = 0
    peak_temperature_c: float = 0.0
    average_temperature_c: float = 0.0
    throttle_events: int = 0
    warning_events: int = 0
    critical_events: int = 0
    shutdown_events: int = 0
    total_throttle_time_ns: int = 0
    time_in_warning_ns: int = 0
    time_in_throttle_ns: int = 0
    time_in_critical_ns: int = 0

    # Extended statistics
    max_thermal_gradient_c: float = 0.0
    time_at_max_temp_ns: int = 0
    thermal_recovery_count: int = 0  # Times recovered from throttle
    adaptive_throttle_count: int = 0

    def reset(self):
        """Reset all statistics"""
        self.samples = 0
        self.peak_temperature_c = 0.0
        self.average_temperature_c = 0.0
        self.throttle_events = 0
        self.warning_events = 0
        self.critical_events = 0
        self.shutdown_events = 0
        self.total_throttle_time_ns = 0
        self.time_in_warning_ns = 0
        self.time_in_throttle_ns = 0
        self.time_in_critical_ns = 0
        self.max_thermal_gradient_c = 0.0
        self.time_at_max_temp_ns = 0
        self.thermal_recovery_count = 0
        self.adaptive_throttle_count = 0


class HBM4ThermalModel:
    """HBM4 Enhanced Thermal Model for Logic Base Die

    Provides comprehensive thermal modeling for HBM4 components with:
    - Per-component hotspot temperature tracking
    - Thermal resistance modeling
    - Advanced throttling policy with hysteresis
    - Adaptive throttling based on thermal history
    - PDN voltage management
    - Temperature-based performance adjustment
    - Power-to-temperature conversion
    - Integration with HBM4PowerEstimator

    The thermal model uses a lumped RC model for each hotspot:
        T_junction = T_ambient + P * R_total

    where:
        R_total = R_junction + R_spreading + R_case + R_ambient

    Temperature rise is computed at each simulation step:
        delta_T = P * R * tau / C
        C = thermal capacitance

    Key features:
    - Thermal coupling between adjacent hotspots
    - Time-averaged temperature with exponential decay
    - Configurable thresholds for throttling
    - Hysteresis to prevent oscillation
    - Adaptive throttling for rapid temperature rise
    - PDN-aware power limiting
    - Performance adjustment based on temperature

    Reference:
    - JEDEC JESD270-4A HBM4
    - Hotspot thermal simulator
    - Synopsys HBM4 Controller IP thermal management
    """

    # Default thermal parameters (16nm logic base die)
    DEFAULT_AMBIENT_TEMP_C = 25.0       # Ambient temperature
    DEFAULT_INITIAL_TEMP_C = 35.0       # Initial die temperature
    DEFAULT_THERMAL_TAU_NS = 1000.0    # Thermal time constant (ns)

    # Thermal resistance defaults (C/W)
    DEFAULT_R_JUNCTION = 0.5           # Junction to die surface
    DEFAULT_R_SPREADING = 2.0           # Spreading in die
    DEFAULT_R_CASE = 8.0               # Case to ambient

    # Hotspot size factors (relative to full die)
    DEFAULT_CONTROLLER_SIZE = 0.15      # Controller cluster: 15% of die
    DEFAULT_D2D_PHY_SIZE = 0.08          # D2D PHY: 8% of die
    DEFAULT_TSV_PHY_SIZE = 0.12         # TSV PHY: 12% of die
    DEFAULT_ECC_SIZE = 0.05             # ECC/RAS: 5% of die
    DEFAULT_CLOCKING_SIZE = 0.03        # Clocking: 3% of die

    def __init__(
        self,
        ambient_temp_c: float = DEFAULT_AMBIENT_TEMP_C,
        initial_temp_c: float = DEFAULT_INITIAL_TEMP_C,
        thermal_tau_ns: float = DEFAULT_THERMAL_TAU_NS,
        thresholds: Optional[TemperatureThresholds] = None,
        throttle_policy: Optional[ThrottlePolicy] = None,
    ):
        """Initialize HBM4 Enhanced Thermal Model

        Args:
            ambient_temp_c: Ambient temperature in Celsius
            initial_temp_c: Initial die temperature in Celsius
            thermal_tau_ns: Thermal time constant in nanoseconds
            thresholds: Temperature thresholds for throttling
            throttle_policy: Configurable throttling policy
        """
        self.ambient_temp_c = ambient_temp_c
        self.thermal_tau_ns = thermal_tau_ns

        # Temperature thresholds
        self.thresholds = thresholds or TemperatureThresholds()

        # Throttle policy
        self.throttle_policy = throttle_policy or ThrottlePolicy()

        # Initialize hotspot configurations
        self._init_hotspot_configs()

        # Initialize temperature state
        self.temperatures = ComponentTemperatures(
            ambient=ambient_temp_c,
            case=initial_temp_c,
            die=initial_temp_c,
            controller_cluster=initial_temp_c,
            d2d_phy=initial_temp_c,
            tsv_phy=initial_temp_c,
            ecc_ras=initial_temp_c,
            clocking=initial_temp_c,
            phy_interface=initial_temp_c,
        )

        # Initialize throttling state
        self.throttle_state = ThrottleState()

        # Initialize PDN operating points
        self._init_pdn_operating_points()

        # Initialize performance adjustment
        self.performance = PerformanceAdjustment()

        # Initialize thermal statistics
        self.stats = ThermalStatistics()

        # External power estimator reference
        self._power_estimator = None

        # Performance callback for throttling
        self._throttle_callback: Optional[Callable[[ThrottleLevel, float], None]] = None

        # Last update timestamp
        self._last_update_ns = 0
        self._last_temp_c = initial_temp_c
        self._last_update_time = time.time()

        # Temperature history for trend analysis
        self._temp_history: List[Tuple[int, float]] = []
        self._max_history_size = 100

    def _init_hotspot_configs(self):
        """Initialize hotspot configurations"""
        self.hotspot_configs = {
            'controller_cluster': HotspotConfig(
                r_junction=self.DEFAULT_R_JUNCTION,
                size_factor=self.DEFAULT_CONTROLLER_SIZE,
                power_density=1.5,  # High activity controller
                coupling_factor=0.08,
                max_temp_c=110.0,
            ),
            'd2d_phy': HotspotConfig(
                r_junction=self.DEFAULT_R_JUNCTION * 0.8,
                size_factor=self.DEFAULT_D2D_PHY_SIZE,
                power_density=2.0,  # High-speed SerDes, dense
                coupling_factor=0.05,
                max_temp_c=105.0,  # More sensitive to heat
            ),
            'tsv_phy': HotspotConfig(
                r_junction=self.DEFAULT_R_JUNCTION * 0.6,
                size_factor=self.DEFAULT_TSV_PHY_SIZE,
                power_density=1.2,  # TSV drivers
                coupling_factor=0.06,
                max_temp_c=110.0,
            ),
            'ecc_ras': HotspotConfig(
                r_junction=self.DEFAULT_R_JUNCTION * 0.5,
                size_factor=self.DEFAULT_ECC_SIZE,
                power_density=0.8,  # ECC logic, lower activity
                coupling_factor=0.03,
                max_temp_c=115.0,  # Can tolerate higher temp
            ),
            'clocking': HotspotConfig(
                r_junction=self.DEFAULT_R_JUNCTION * 0.7,
                size_factor=self.DEFAULT_CLOCKING_SIZE,
                power_density=1.0,  # PLL, DLL
                coupling_factor=0.02,
                max_temp_c=100.0,  # Clocking sensitive
            ),
            'phy_interface': HotspotConfig(
                r_junction=self.DEFAULT_R_JUNCTION * 0.9,
                size_factor=0.10,
                power_density=1.3,  # DFI, TX/RX
                coupling_factor=0.07,
                max_temp_c=105.0,
            ),
        }

    def _init_pdn_operating_points(self):
        """Initialize PDN voltage operating points"""
        self.pdn_operating_points = {
            PDNVoltageMode.NOMINAL: PDNOperatingPoint(
                mode=PDNVoltageMode.NOMINAL,
                voltage_mv=900,
                max_current_ma=5000,
                max_power_mw=4500,
                thermal_limit_c=95.0,
                frequency_scale=1.0,
                voltage_scale=1.0,
            ),
            PDNVoltageMode.PERFORMANCE: PDNOperatingPoint(
                mode=PDNVoltageMode.PERFORMANCE,
                voltage_mv=1000,
                max_current_ma=6000,
                max_power_mw=6000,
                thermal_limit_c=90.0,  # Stricter at higher voltage
                frequency_scale=1.1,
                voltage_scale=1.11,
            ),
            PDNVoltageMode.LOW_POWER: PDNOperatingPoint(
                mode=PDNVoltageMode.LOW_POWER,
                voltage_mv=800,
                max_current_ma=4000,
                max_power_mw=3200,
                thermal_limit_c=100.0,  # Can tolerate higher temp
                frequency_scale=0.9,
                voltage_scale=0.89,
            ),
            PDNVoltageMode.ULTRA_LOW: PDNOperatingPoint(
                mode=PDNVoltageMode.ULTRA_LOW,
                voltage_mv=650,
                max_current_ma=2500,
                max_power_mw=1625,
                thermal_limit_c=105.0,  # Maximum thermal margin
                frequency_scale=0.7,
                voltage_scale=0.72,
            ),
        }

    def set_power_estimator(self, estimator):
        """Set reference to power estimator for dynamic power tracking

        Args:
            estimator: HBM4PowerEstimator instance
        """
        self._power_estimator = estimator

    def set_throttle_callback(self, callback: Callable[[ThrottleLevel, float], None]):
        """Set callback for throttle state changes

        Args:
            callback: Function called with (throttle_level, throttle_factor)
        """
        self._throttle_callback = callback

    def update_temperature(
        self,
        timestamp_ns: int,
        power_breakdown: Optional[Dict[str, float]] = None,
    ):
        """Update temperatures based on power consumption

        Uses exponential moving average for temperature tracking:
            T_new = T_old + (P * R / C) * (1 - exp(-dt / tau))

        Args:
            timestamp_ns: Current simulation time in nanoseconds
            power_breakdown: Optional dict of component powers (mW)
                             If None, uses power estimator if available
        """
        # Calculate delta time
        dt = timestamp_ns - self._last_update_ns
        self._last_update_ns = timestamp_ns

        # Get power breakdown
        if power_breakdown is None and self._power_estimator is not None:
            power_breakdown = self._get_power_from_estimator()
        elif power_breakdown is None:
            power_breakdown = self._default_power_breakdown()

        # Update temperatures with thermal dynamics
        self._update_hotspot_temperatures(power_breakdown, dt, timestamp_ns)

        # Update die and case temperatures
        self._update_die_temperature(power_breakdown)

        # Update thermal gradient
        self._update_thermal_gradient()

        # Update temperature rate of change
        self._update_temperature_rate(timestamp_ns)

        # Update thermal throttling
        self._update_throttling(timestamp_ns)

        # Update performance adjustment
        self._update_performance_adjustment()

        # Update statistics
        self._update_statistics(timestamp_ns)

        # Call throttle callback if set
        if self._throttle_callback and self.throttle_state.active:
            self._throttle_callback(self.throttle_state.level, self.throttle_state.throttle_factor)

    def _get_power_from_estimator(self) -> Dict[str, float]:
        """Get power breakdown from power estimator"""
        if self._power_estimator is None:
            return self._default_power_breakdown()

        breakdown = self._power_estimator.get_power_breakdown()

        return {
            'controller_cluster': breakdown.controller_power.total(),
            'd2d_phy': breakdown.phy_power.d2d_phy,
            'tsv_phy': breakdown.phy_power.tsv_phy,
            'ecc_ras': breakdown.ecc_power.total(),
            'clocking': breakdown.clocking_power.total(),
            'phy_interface': breakdown.phy_power.dfi_interface,
        }

    def _default_power_breakdown(self) -> Dict[str, float]:
        """Return default power breakdown when no estimator available"""
        return {
            'controller_cluster': 115.0,
            'd2d_phy': 80.0,
            'tsv_phy': 120.0,
            'ecc_ras': 34.0,
            'clocking': 78.0,
            'phy_interface': 45.0,
        }

    def _update_hotspot_temperatures(
        self,
        power_breakdown: Dict[str, float],
        dt_ns: float,
        timestamp_ns: int,
    ):
        """Update individual hotspot temperatures"""
        # Exponential decay factor for thermal time constant
        decay = math.exp(-dt_ns / self.thermal_tau_ns) if dt_ns > 0 else 0.0

        for hotspot_name, config in self.hotspot_configs.items():
            power = power_breakdown.get(hotspot_name, 0.0)

            # Calculate temperature rise
            r_total = config.r_junction + self.DEFAULT_R_SPREADING
            delta_t = power * r_total / 1000.0

            # Get current temperature
            current_temp = getattr(self.temperatures, hotspot_name)

            # Apply thermal dynamics with exponential settling
            steady_state = self.ambient_temp_c + delta_t
            new_temp = current_temp + (steady_state - current_temp) * (1.0 - decay)

            # Update temperature
            setattr(self.temperatures, hotspot_name, new_temp)

            # Track temperature history
            self._temp_history.append((timestamp_ns, new_temp))
            if len(self._temp_history) > self._max_history_size:
                self._temp_history.pop(0)

        # Apply thermal coupling between hotspots
        self._apply_thermal_coupling(decay)

    def _apply_thermal_coupling(self, decay: float):
        """Apply thermal coupling between adjacent hotspots"""
        hotspots = ['controller_cluster', 'd2d_phy', 'tsv_phy',
                    'ecc_ras', 'clocking', 'phy_interface']

        temp_deltas = {}

        for i, name in enumerate(hotspots):
            config = self.hotspot_configs[name]
            if config.coupling_factor <= 0:
                continue

            # Get average of adjacent temperatures
            adjacent_temps = []
            if i > 0:
                adjacent_temps.append(getattr(self.temperatures, hotspots[i-1]))
            if i < len(hotspots) - 1:
                adjacent_temps.append(getattr(self.temperatures, hotspots[i+1]))

            if adjacent_temps:
                avg_adjacent = sum(adjacent_temps) / len(adjacent_temps)
                current = getattr(self.temperatures, name)
                coupling_delta = (avg_adjacent - current) * config.coupling_factor
                temp_deltas[name] = coupling_delta * (1.0 - decay)

        # Apply deltas
        for name, delta in temp_deltas.items():
            current = getattr(self.temperatures, name)
            setattr(self.temperatures, name, current + delta)

    def _update_die_temperature(self, power_breakdown: Dict[str, float]):
        """Update die-level temperature from hotspot average"""
        total_power = sum(power_breakdown.values())
        r_total = self.DEFAULT_R_JUNCTION + self.DEFAULT_R_SPREADING + self.DEFAULT_R_CASE
        delta_t = total_power * r_total / 1000.0

        # Die temperature is weighted average of hotspots
        hotspot_temps = [
            self.temperatures.controller_cluster,
            self.temperatures.d2d_phy,
            self.temperatures.tsv_phy,
            self.temperatures.ecc_ras,
            self.temperatures.clocking,
            self.temperatures.phy_interface,
        ]
        avg_hotspot = sum(hotspot_temps) / len(hotspot_temps)

        # Case temperature
        self.temperatures.case = self.ambient_temp_c + delta_t * 0.8

        # Die temperature with hotspot influence
        self.temperatures.die = avg_hotspot + delta_t * 0.2

    def _update_thermal_gradient(self):
        """Update thermal gradient across the die"""
        max_temp = self.temperatures.max_temperature
        min_temp = self.temperatures.min_temperature
        self.temperatures.thermal_gradient = max_temp - min_temp

        if self.temperatures.thermal_gradient > self.stats.max_thermal_gradient_c:
            self.stats.max_thermal_gradient_c = self.temperatures.thermal_gradient

    def _update_temperature_rate(self, timestamp_ns: int):
        """Update rate of temperature change"""
        current_time = time.time()
        dt_real = current_time - self._last_update_time
        self._last_update_time = current_time

        if dt_real > 0 and len(self._temp_history) >= 2:
            _, last_temp = self._temp_history[-1]
            _, prev_temp = self._temp_history[-2]
            self.temperatures.rate_of_change = (last_temp - prev_temp) / dt_real
            self.throttle_state.thermal_trend = self.temperatures.rate_of_change

    def _update_throttling(self, timestamp_ns: int):
        """Update thermal throttling state with hysteresis and adaptive throttling"""
        max_temp = self.temperatures.max_temperature

        # Determine direction of temperature change
        if len(self._temp_history) >= 2:
            _, last_temp = self._temp_history[-1]
            _, prev_temp = self._temp_history[-2]
            if last_temp > prev_temp:
                direction = ThrottleDirection.INCREASING
            elif last_temp < prev_temp:
                direction = ThrottleDirection.DECREASING
            else:
                direction = ThrottleDirection.STABLE
        else:
            direction = ThrottleDirection.STABLE

        # Get new throttle level with hysteresis
        new_level = self.thresholds.get_throttle_level(
            max_temp,
            current_level=self.throttle_state.level,
            direction=direction
        )

        # Track level transitions
        previous_active = self.throttle_state.active
        if new_level != self.throttle_state.level:
            if new_level in [ThrottleLevel.WARNING, ThrottleLevel.THROTTLE,
                             ThrottleLevel.CRITICAL, ThrottleLevel.SHUTDOWN]:
                self.throttle_state.throttle_count += 1

            # Track recovery events
            if (self.throttle_state.level in [ThrottleLevel.THROTTLE, ThrottleLevel.CRITICAL] and
                new_level in [ThrottleLevel.NONE, ThrottleLevel.WARNING]):
                self.stats.thermal_recovery_count += 1

        self.throttle_state.level = new_level
        self.throttle_state.max_temperature_reached = max(
            self.throttle_state.max_temperature_reached,
            max_temp
        )

        # Calculate thermal rate for adaptive throttling
        thermal_rate = self.throttle_state.thermal_trend

        # Update throttle factor based on level and adaptive policy
        if new_level == ThrottleLevel.SHUTDOWN:
            self.throttle_state.throttle_factor = 0.0
            self.throttle_state.active = True
        elif new_level == ThrottleLevel.CRITICAL:
            self.throttle_state.throttle_factor = self.throttle_policy.get_throttle_factor(
                ThrottleLevel.CRITICAL, thermal_rate
            )
            self.throttle_state.active = True
            if thermal_rate > self.throttle_policy.rapid_rise_threshold_cps:
                self.stats.adaptive_throttle_count += 1
        elif new_level == ThrottleLevel.THROTTLE:
            self.throttle_state.throttle_factor = self.throttle_policy.get_throttle_factor(
                ThrottleLevel.THROTTLE, thermal_rate
            )
            self.throttle_state.active = True
            if thermal_rate > self.throttle_policy.rapid_rise_threshold_cps:
                self.stats.adaptive_throttle_count += 1
        elif new_level == ThrottleLevel.WARNING:
            self.throttle_state.throttle_factor = self.throttle_policy.get_throttle_factor(
                ThrottleLevel.WARNING, thermal_rate
            )
            self.throttle_state.active = False  # Monitoring only
        else:
            self.throttle_state.throttle_factor = 1.0
            self.throttle_state.active = False

        # Update PDN mode based on throttle policy
        self.throttle_state.pdn_mode = self.throttle_policy.get_pdn_mode(new_level)

        # Update throttle time tracking
        if self.throttle_state.active:
            self.throttle_state.time_in_throttle_ns = timestamp_ns

        # Update time at max temperature
        if max_temp >= self.stats.peak_temperature_c - 0.1:
            self.stats.time_at_max_temp_ns = timestamp_ns

    def _update_performance_adjustment(self):
        """Update performance adjustment based on temperature and throttle state"""
        max_temp = self.temperatures.max_temperature

        # Start with temperature-based degradation
        self.performance = self.performance.apply_temperature(max_temp)

        # Apply throttle factor
        self.performance.frequency_scale *= self.throttle_state.throttle_factor

        # Bandwidth scales with frequency
        self.performance.bandwidth_scale = self.performance.frequency_scale

        # Get PDN operating point for additional scaling
        op_point = self.pdn_operating_points.get(
            self.throttle_state.pdn_mode,
            self.pdn_operating_points[PDNVoltageMode.NOMINAL]
        )
        self.performance.voltage_scale = op_point.voltage_scale

    def _update_statistics(self, timestamp_ns: int):
        """Update thermal statistics"""
        self.stats.samples += 1

        max_temp = self.temperatures.max_temperature
        avg_temp = self.temperatures.average_temperature

        # Running average
        if self.stats.samples == 1:
            self.stats.average_temperature_c = avg_temp
        else:
            self.stats.average_temperature_c = (
                (self.stats.average_temperature_c * (self.stats.samples - 1) + avg_temp)
                / self.stats.samples
            )

        # Peak tracking
        if max_temp > self.stats.peak_temperature_c:
            self.stats.peak_temperature_c = max_temp

        # Event counting
        level = self.throttle_state.level
        if level == ThrottleLevel.WARNING:
            self.stats.warning_events += 1
            self.stats.time_in_warning_ns += 1
        elif level == ThrottleLevel.THROTTLE:
            self.stats.throttle_events += 1
            self.stats.time_in_throttle_ns += 1
        elif level == ThrottleLevel.CRITICAL:
            self.stats.critical_events += 1
            self.stats.time_in_critical_ns += 1
        elif level == ThrottleLevel.SHUTDOWN:
            self.stats.shutdown_events += 1

    def get_component_temperature(self, component: str) -> float:
        """Get temperature for a specific component

        Args:
            component: Component name ('controller_cluster', 'd2d_phy', etc.)

        Returns:
            Temperature in Celsius
        """
        return getattr(self.temperatures, component, self.temperatures.die)

    def get_die_temperature(self) -> float:
        """Get die temperature"""
        return self.temperatures.die

    def get_max_temperature(self) -> float:
        """Get maximum temperature across all components"""
        return self.temperatures.max_temperature

    def get_throttle_factor(self) -> float:
        """Get current throttle factor for frequency/voltage adjustment"""
        return self.throttle_state.throttle_factor

    def get_throttle_level(self) -> ThrottleLevel:
        """Get current throttle level"""
        return self.throttle_state.level

    def is_throttling_active(self) -> bool:
        """Check if throttling is active"""
        return self.throttle_state.active

    def get_pdn_mode(self) -> PDNVoltageMode:
        """Get current PDN voltage mode"""
        return self.throttle_state.pdn_mode

    def get_pdn_voltage(self) -> float:
        """Get PDN voltage for current operating point"""
        op_point = self.pdn_operating_points.get(
            self.throttle_state.pdn_mode,
            self.pdn_operating_points[PDNVoltageMode.NOMINAL]
        )
        return op_point.voltage_mv

    def get_performance_adjustment(self) -> PerformanceAdjustment:
        """Get current performance adjustment"""
        return self.performance

    def get_effective_bandwidth(self, nominal_bandwidth_gbs: float) -> float:
        """Calculate effective bandwidth after throttling

        Args:
            nominal_bandwidth_gbs: Nominal bandwidth in GB/s

        Returns:
            Effective bandwidth in GB/s
        """
        return nominal_bandwidth_gbs * self.performance.bandwidth_scale

    def get_thermal_resistance(self, component: str) -> float:
        """Get thermal resistance for a component

        Args:
            component: Component name

        Returns:
            Thermal resistance in C/W
        """
        config = self.hotspot_configs.get(component)
        if config:
            return config.r_junction + self.DEFAULT_R_SPREADING
        return self.DEFAULT_R_JUNCTION + self.DEFAULT_R_SPREADING

    def calculate_power_limit(
        self,
        target_temp_c: float,
        ambient_temp_c: Optional[float] = None,
    ) -> float:
        """Calculate maximum power for a target temperature

        Args:
            target_temp_c: Target maximum temperature in Celsius
            ambient_temp_c: Ambient temperature (uses default if None)

        Returns:
            Maximum power in mW
        """
        if ambient_temp_c is None:
            ambient_temp_c = self.ambient_temp_c

        delta_t = target_temp_c - ambient_temp_c
        r_total = self.DEFAULT_R_JUNCTION + self.DEFAULT_R_SPREADING + self.DEFAULT_R_CASE

        return delta_t * 1000.0 / r_total

    def get_temperature_rise(self, power_mw: float, component: Optional[str] = None) -> float:
        """Calculate temperature rise from power

        Args:
            power_mw: Power in milliwatts
            component: Component name (uses die-level if None)

        Returns:
            Temperature rise in Celsius
        """
        if component:
            r = self.get_thermal_resistance(component)
        else:
            r = self.DEFAULT_R_JUNCTION + self.DEFAULT_R_SPREADING + self.DEFAULT_R_CASE

        return power_mw * r / 1000.0

    def get_temperature_trend(self) -> float:
        """Get rate of temperature change (C/s)"""
        return self.throttle_state.thermal_trend

    def get_throttle_summary(self) -> Dict:
        """Get throttling state summary

        Returns:
            Dictionary with throttling information
        """
        return {
            'level': self.throttle_state.level.value,
            'active': self.throttle_state.active,
            'throttle_factor': self.throttle_state.throttle_factor,
            'pdn_mode': self.throttle_state.pdn_mode.value,
            'pdn_voltage_mv': self.get_pdn_voltage(),
            'max_temp_reached': self.throttle_state.max_temperature_reached,
            'throttle_count': self.throttle_state.throttle_count,
            'time_in_throttle_ns': self.throttle_state.time_in_throttle_ns,
            'thermal_trend_cps': self.throttle_state.thermal_trend,
            'adaptive_count': self.throttle_state.adaptive_count,
        }

    def get_summary(self) -> Dict:
        """Get complete thermal model summary

        Returns:
            Dictionary with thermal model state
        """
        return {
            'temperatures': {
                'ambient_c': self.temperatures.ambient,
                'case_c': self.temperatures.case,
                'die_c': self.temperatures.die,
                'max_c': self.temperatures.max_temperature,
                'min_c': self.temperatures.min_temperature,
                'average_c': self.temperatures.average_temperature,
                'hotspot_c': self.temperatures.hotspot_temperature,
                'rate_of_change_cps': self.temperatures.rate_of_change,
                'thermal_gradient_c': self.temperatures.thermal_gradient,
                'controller_cluster_c': self.temperatures.controller_cluster,
                'd2d_phy_c': self.temperatures.d2d_phy,
                'tsv_phy_c': self.temperatures.tsv_phy,
                'ecc_ras_c': self.temperatures.ecc_ras,
                'clocking_c': self.temperatures.clocking,
                'phy_interface_c': self.temperatures.phy_interface,
            },
            'throttle': self.get_throttle_summary(),
            'performance': {
                'frequency_scale': self.performance.frequency_scale,
                'voltage_scale': self.performance.voltage_scale,
                'bandwidth_scale': self.performance.bandwidth_scale,
                'latency_penalty': self.performance.latency_penalty,
            },
            'pdn': {
                mode.value: {
                    'voltage_mv': op.voltage_mv,
                    'max_power_mw': op.max_power_mw,
                    'frequency_scale': op.frequency_scale,
                }
                for mode, op in self.pdn_operating_points.items()
            },
            'thresholds': {
                'warning_c': self.thresholds.warning,
                'throttle_c': self.thresholds.throttle,
                'critical_c': self.thresholds.critical,
                'shutdown_c': self.thresholds.shutdown,
                'hysteresis_warning_c': self.thresholds.hysteresis_warning,
                'hysteresis_throttle_c': self.thresholds.hysteresis_throttle,
            },
            'stats': {
                'samples': self.stats.samples,
                'peak_temperature_c': self.stats.peak_temperature_c,
                'average_temperature_c': self.stats.average_temperature_c,
                'throttle_events': self.stats.throttle_events,
                'warning_events': self.stats.warning_events,
                'critical_events': self.stats.critical_events,
                'shutdown_events': self.stats.shutdown_events,
                'max_thermal_gradient_c': self.stats.max_thermal_gradient_c,
                'thermal_recovery_count': self.stats.thermal_recovery_count,
                'adaptive_throttle_count': self.stats.adaptive_throttle_count,
            },
            'thermal_resistance': {
                'r_junction_c_w': self.DEFAULT_R_JUNCTION,
                'r_spreading_c_w': self.DEFAULT_R_SPREADING,
                'r_case_c_w': self.DEFAULT_R_CASE,
                'r_total_c_w': self.DEFAULT_R_JUNCTION + self.DEFAULT_R_SPREADING + self.DEFAULT_R_CASE,
            },
        }

    def reset(self):
        """Reset thermal model state"""
        self.temperatures = ComponentTemperatures(
            ambient=self.ambient_temp_c,
            case=self.ambient_temp_c,
            die=self.ambient_temp_c,
            controller_cluster=self.ambient_temp_c,
            d2d_phy=self.ambient_temp_c,
            tsv_phy=self.ambient_temp_c,
            ecc_ras=self.ambient_temp_c,
            clocking=self.ambient_temp_c,
            phy_interface=self.ambient_temp_c,
        )

        self.throttle_state = ThrottleState()
        self.performance = PerformanceAdjustment()
        self.stats = ThermalStatistics()
        self._last_update_ns = 0
        self._last_temp_c = self.ambient_temp_c
        self._temp_history.clear()


# Factory function
def create_thermal_model(
    ambient_temp_c: float = 25.0,
    speed_grade: str = "8Gbps",
) -> HBM4ThermalModel:
    """Create thermal model with specified configuration

    Args:
        ambient_temp_c: Ambient temperature in Celsius
        speed_grade: Speed grade ('8Gbps', '12Gbps', '16Gbps')

    Returns:
        Configured HBM4ThermalModel
    """
    # Adjust thresholds based on speed grade (higher speed = tighter thermal)
    if speed_grade == "16Gbps":
        thresholds = TemperatureThresholds(
            warning=80.0,
            throttle=90.0,
            critical=100.0,
            shutdown=105.0,
            hysteresis_warning=3.0,
            hysteresis_throttle=5.0,
            hysteresis_critical=5.0,
            hysteresis_shutdown=3.0,
        )
    elif speed_grade == "12Gbps":
        thresholds = TemperatureThresholds(
            warning=82.0,
            throttle=92.0,
            critical=102.0,
            shutdown=108.0,
        )
    else:  # 8Gbps
        thresholds = TemperatureThresholds()

    return HBM4ThermalModel(
        ambient_temp_c=ambient_temp_c,
        thresholds=thresholds,
    )


def create_thermal_model_with_policy(
    ambient_temp_c: float = 25.0,
    enable_adaptive: bool = True,
    rapid_rise_threshold_cps: float = 10.0,
) -> HBM4ThermalModel:
    """Create thermal model with custom throttling policy

    Args:
        ambient_temp_c: Ambient temperature in Celsius
        enable_adaptive: Enable adaptive throttling
        rapid_rise_threshold_cps: Temperature rise rate threshold for accelerated throttling

    Returns:
        Configured HBM4ThermalModel with custom policy
    """
    policy = ThrottlePolicy(
        enable_adaptive=enable_adaptive,
        rapid_rise_threshold_cps=rapid_rise_threshold_cps,
    )

    return HBM4ThermalModel(
        ambient_temp_c=ambient_temp_c,
        throttle_policy=policy,
    )