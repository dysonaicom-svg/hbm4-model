"""
HBM4 Thermal Sensor Model

Provides thermal sensor abstraction with:
- Temperature reading simulation
- Sensor calibration model
- Thermal margin calculation
- Multi-sensor support for HBM4 stacking

Reference:
- JEDEC JESD270-4A HBM4 specification
- JESD51-14 Thermal test method
- On-die thermal sensing literature
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math
import random


class ThermalZone(Enum):
    """Thermal zones in HBM4 package"""
    PACKAGE_CORE = "package_core"       # Core logic die
    DRAM_BANK_0 = "dram_bank_0"         # DRAM bank group 0
    DRAM_BANK_1 = "dram_bank_1"         # DRAM bank group 1
    DRAM_BANK_2 = "dram_bank_2"         # DRAM bank group 2
    DRAM_BANK_3 = "dram_bank_3"         # DRAM bank group 3
    LOGIC_BASE_DIE = "logic_base_die"   # Logic base die (HBM4)
    PACKAGE_EDGE = "package_edge"       # Package edge
    SUBSTRATE = "substrate"             # Package substrate


class SensorType(Enum):
    """Types of thermal sensors"""
    ON_DIE_BANDGAP = "on_die_bandgap"   # Bandgap-based temperature sensor
    THERMOCOUPLE = "thermocouple"      # Thermocouple-based
    RESISTANCE = "resistance"           # Resistance temperature detector (RTD)
    DIGITAL = "digital"                 # Digital thermal sensor


@dataclass
class SensorCalibration:
    """Sensor calibration parameters

    Calibration model: T_measured = offset + gain * T_raw + (noise_factor * rand)
    """
    offset_c: float = 0.0           # Temperature offset (C)
    gain: float = 1.0               # Gain factor
    noise_factor_c: float = 0.5     # Random noise factor (C RMS)
    drift_rate_c_per_hour: float = 0.01  # Temperature drift rate

    # Temperature-dependent accuracy
    accuracy_at_25c: float = 1.0    # Accuracy at 25C (C)
    accuracy_at_85c: float = 2.0     # Accuracy at 85C (C)

    # Calibration history
    last_calibrated: float = 0.0     # Simulation time of last calibration
    calibration_count: int = 0       # Number of calibrations performed


@dataclass
class SensorReading:
    """Single sensor reading with metadata"""
    timestamp_ns: int                # Simulation time (ns)
    raw_temperature_c: float         # Raw sensor reading (C)
    calibrated_temperature_c: float  # Calibrated temperature (C)
    confidence: float = 1.0         # Confidence level (0-1)
    sensor_id: int = 0               # Sensor identifier
    zone: ThermalZone = ThermalZone.PACKAGE_CORE

    @property
    def thermal_margin_c(self) -> float:
        """Calculate margin to max junction temperature"""
        return MAX_JUNCTION_TEMP_C - self.calibrated_temperature_c

    @property
    def is_safe(self) -> bool:
        """Check if temperature is within safe limits"""
        return self.calibrated_temperature_c < THERMAL_THROTTLE_THRESHOLD_C


@dataclass
class SensorConfiguration:
    """Configuration for thermal sensor"""
    sensor_id: int = 0
    sensor_type: SensorType = SensorType.ON_DIE_BANDGAP
    zone: ThermalZone = ThermalZone.PACKAGE_CORE

    # Sampling configuration
    sampling_interval_ns: int = 1000     # Sampling interval (ns)
    averaging_samples: int = 4           # Number of samples for averaging

    # Range configuration
    min_temperature_c: float = -40.0      # Minimum measurable temperature (C)
    max_temperature_c: float = 125.0      # Maximum measurable temperature (C)

    # Update latency
    update_latency_ns: int = 100          # Time from measurement to output (ns)

    # Calibration
    calibration: SensorCalibration = field(default_factory=SensorCalibration)


@dataclass
class ThermalSensor:
    """HBM4 Thermal Sensor Model

    Simulates on-die thermal sensors with:
    - Temperature-dependent characteristics
    - Calibration drift over time
    - Measurement noise
    - Multi-zone support
    """
    config: SensorConfiguration

    # State tracking
    current_temperature_c: float = 45.0    # Current temperature (C)
    temperature_history: List[float] = field(default_factory=list)
    reading_history: List[SensorReading] = field(default_factory=list)

    # Simulation state
    current_time_ns: int = 0
    samples_since_calibration: int = 0

    # Sensor characteristics
    response_time_constant_ns: int = 100000  # Thermal response time (100 us)

    def __post_init__(self):
        """Initialize sensor state"""
        self.temperature_history = []
        self.reading_history = []
        self.samples_since_calibration = 0

    def _simulate_raw_temperature(self, true_temp_c: float) -> float:
        """Simulate raw sensor reading

        Args:
            true_temp_c: True temperature to measure

        Returns:
            Raw sensor reading with noise
        """
        cal = self.config.calibration

        # Add measurement noise
        noise = random.gauss(0, cal.noise_factor_c)

        # Apply calibration
        raw = true_temp_c * cal.gain + cal.offset_c + noise

        # Clamp to sensor range
        raw = max(self.config.min_temperature_c,
                  min(self.config.max_temperature_c, raw))

        return raw

    def _calibrate_reading(self, raw_temp_c: float) -> float:
        """Apply calibration to raw reading

        Args:
            raw_temp_c: Raw temperature reading

        Returns:
            Calibrated temperature
        """
        cal = self.config.calibration

        # Temperature-dependent accuracy model
        base_accuracy = cal.accuracy_at_25c
        high_temp_accuracy = cal.accuracy_at_85c

        # Interpolate based on temperature
        temp_range = 85.0 - 25.0
        temp_norm = max(0, min(1, (raw_temp_c - 25.0) / temp_range))
        accuracy = base_accuracy + (high_temp_accuracy - base_accuracy) * temp_norm

        # Apply drift if calibration is old
        drift = 0.0
        if self.samples_since_calibration > 0:
            hours_elapsed = (self.current_time_ns * 1e-9) / 3600.0
            drift = hours_elapsed * cal.drift_rate_c_per_hour

        calibrated = raw_temp_c - drift

        return calibrated

    def _calculate_confidence(self, raw_temp_c: float, calibrated_temp_c: float) -> float:
        """Calculate confidence level for reading

        Args:
            raw_temp_c: Raw temperature
            calibrated_temp_c: Calibrated temperature

        Returns:
            Confidence level (0-1)
        """
        cal = self.config.calibration

        # Base confidence
        confidence = 1.0

        # Reduce confidence if far from reference temperature
        if abs(calibrated_temp_c - 45.0) > 30.0:
            confidence *= 0.8

        # Reduce confidence if calibration is old
        if self.samples_since_calibration > 1000:
            confidence *= 0.9

        # Reduce confidence at extreme temperatures
        if calibrated_temp_c > 80.0 or calibrated_temp_c < 10.0:
            confidence *= 0.85

        return confidence

    def measure(self, ambient_temp_c: float, power_dissipation_mw: float,
                time_ns: int) -> SensorReading:
        """Take a temperature measurement

        Args:
            ambient_temp_c: Ambient/package temperature (C)
            power_dissipation_mw: Power being dissipated (mW)
            time_ns: Current simulation time (ns)

        Returns:
            SensorReading with calibrated temperature
        """
        self.current_time_ns = time_ns

        # Calculate self-heating effect
        # P = I^2 * R, self-heating depends on current
        thermal_resistance = THERMAL_RESISTANCE_C_PER_W  # C/W
        self_heating_c = (power_dissipation_mw / 1000.0) * thermal_resistance

        # True temperature is ambient + self-heating
        true_temp_c = ambient_temp_c + self_heating_c

        # Apply thermal time constant for sensor response
        if self.temperature_history:
            prev_temp = self.temperature_history[-1]
            tau = self.response_time_constant_ns
            alpha = 1.0 - math.exp(-(time_ns - (self.current_time_ns - self.config.sampling_interval_ns)) / tau)
            effective_temp = prev_temp + alpha * (true_temp_c - prev_temp)
        else:
            effective_temp = true_temp_c

        # Simulate sensor reading
        raw_temp = self._simulate_raw_temperature(effective_temp)
        calibrated_temp = self._calibrate_reading(raw_temp)
        confidence = self._calculate_confidence(raw_temp, calibrated_temp)

        # Update state
        self.current_temperature_c = calibrated_temp
        self.temperature_history.append(calibrated_temp)
        self.samples_since_calibration += 1

        # Create reading
        reading = SensorReading(
            timestamp_ns=time_ns,
            raw_temperature_c=raw_temp,
            calibrated_temperature_c=calibrated_temp,
            confidence=confidence,
            sensor_id=self.config.sensor_id,
            zone=self.config.zone,
        )
        self.reading_history.append(reading)

        # Keep history bounded
        if len(self.temperature_history) > MAX_HISTORY_SAMPLES:
            self.temperature_history = self.temperature_history[-MAX_HISTORY_SAMPLES:]
        if len(self.reading_history) > MAX_HISTORY_SAMPLES:
            self.reading_history = self.reading_history[-MAX_HISTORY_SAMPLES:]

        return reading

    def calibrate(self, reference_temp_c: float, time_ns: int):
        """Perform sensor calibration

        Args:
            reference_temp_c: Known reference temperature (C)
            time_ns: Current simulation time (ns)
        """
        cal = self.config.calibration
        averaging = self.config.averaging_samples

        # Calculate current offset
        if self.temperature_history:
            current_avg = sum(self.temperature_history[-averaging:]) / averaging
            measured_offset = reference_temp_c - current_avg

            # Update calibration parameters (exponential moving average)
            alpha = 0.3  # Learning rate
            cal.offset_c = cal.offset_c * (1 - alpha) + measured_offset * alpha

        cal.last_calibrated = time_ns
        cal.calibration_count += 1
        self.samples_since_calibration = 0

    def get_average_temperature_c(self, num_samples: int = 10) -> float:
        """Get rolling average temperature

        Args:
            num_samples: Number of recent samples to average

        Returns:
            Average temperature (C)
        """
        if not self.temperature_history:
            return self.current_temperature_c

        samples = self.temperature_history[-num_samples:]
        return sum(samples) / len(samples)

    def get_temperature_rate_c_per_sec(self, num_samples: int = 10) -> float:
        """Calculate temperature change rate using linear regression

        Args:
            num_samples: Number of recent samples for rate calculation

        Returns:
            Temperature rate (C/s)
        """
        if len(self.temperature_history) < 2:
            return 0.0

        recent = self.temperature_history[-num_samples:]
        if len(recent) < 2:
            return 0.0

        # Use linear regression for robust rate estimation
        n = len(recent)
        # x values are sample indices (0, 1, 2, ...)
        sum_x = n * (n - 1) / 2
        sum_xx = n * (n - 1) * (2 * n - 1) / 6
        sum_y = sum(recent)
        sum_xy = sum(i * recent[i] for i in range(n))

        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Convert from per-sample to per-second
        samples_per_sec = 1e9 / self.config.sampling_interval_ns
        return slope * samples_per_sec

    def get_thermal_trend(self) -> str:
        """Get thermal trend description

        Returns:
            Trend string: "rising", "falling", "stable", "unknown"
        """
        rate = self.get_temperature_rate_c_per_sec()

        if abs(rate) < 0.1:
            return "stable"
        elif rate > 1.0:
            return "rising"
        elif rate < -1.0:
            return "falling"
        else:
            return "stable"

    def get_statistics(self) -> Dict[str, float]:
        """Get temperature statistics

        Returns:
            Dictionary with temperature statistics
        """
        if not self.temperature_history:
            return {
                "current_c": self.current_temperature_c,
                "average_c": self.current_temperature_c,
                "min_c": self.current_temperature_c,
                "max_c": self.current_temperature_c,
                "std_dev_c": 0.0,
                "thermal_margin_c": MAX_JUNCTION_TEMP_C - self.current_temperature_c,
            }

        temps = self.temperature_history
        avg = sum(temps) / len(temps)
        variance = sum((t - avg) ** 2 for t in temps) / len(temps)
        std_dev = math.sqrt(variance)

        return {
            "current_c": self.current_temperature_c,
            "average_c": avg,
            "min_c": min(temps),
            "max_c": max(temps),
            "std_dev_c": std_dev,
            "thermal_margin_c": MAX_JUNCTION_TEMP_C - self.current_temperature_c,
            "samples_count": len(temps),
        }

    def reset(self):
        """Reset sensor state"""
        self.current_temperature_c = 45.0
        self.temperature_history = []
        self.reading_history = []
        self.current_time_ns = 0
        self.samples_since_calibration = 0
        self.config.calibration.last_calibrated = 0.0
        self.config.calibration.calibration_count = 0


@dataclass
class SensorArray:
    """Array of thermal sensors for multi-zone monitoring"""
    sensors: List[ThermalSensor] = field(default_factory=list)

    # Configuration
    sensor_count: int = 8
    include_zones: List[ThermalZone] = field(default_factory=list)

    def __post_init__(self):
        """Initialize sensor array"""
        if not self.sensors and self.sensor_count > 0:
            # Default: create sensors for main zones
            zones = [
                ThermalZone.PACKAGE_CORE,
                ThermalZone.LOGIC_BASE_DIE,
                ThermalZone.DRAM_BANK_0,
                ThermalZone.DRAM_BANK_1,
                ThermalZone.DRAM_BANK_2,
                ThermalZone.DRAM_BANK_3,
                ThermalZone.PACKAGE_EDGE,
                ThermalZone.SUBSTRATE,
            ]

            for i in range(min(self.sensor_count, len(zones))):
                config = SensorConfiguration(
                    sensor_id=i,
                    zone=zones[i],
                )
                self.sensors.append(ThermalSensor(config))

    def measure_all(self, ambient_temp_c: float, power_per_zone: Dict[ThermalZone, float],
                    time_ns: int) -> List[SensorReading]:
        """Measure temperature across all sensors

        Args:
            ambient_temp_c: Ambient temperature (C)
            power_per_zone: Power dissipation per zone (mW)
            time_ns: Current simulation time (ns)

        Returns:
            List of sensor readings
        """
        readings = []

        for sensor in self.sensors:
            power = power_per_zone.get(sensor.config.zone, 100.0)  # Default 100mW
            reading = sensor.measure(ambient_temp_c, power, time_ns)
            readings.append(reading)

        return readings

    def get_max_temperature_c(self) -> float:
        """Get maximum temperature across all sensors"""
        if not self.sensors:
            return self.sensors[0].current_temperature_c if self.sensors else 45.0
        return max(s.current_temperature_c for s in self.sensors)

    def get_min_temperature_c(self) -> float:
        """Get minimum temperature across all sensors"""
        if not self.sensors:
            return self.sensors[0].current_temperature_c if self.sensors else 45.0
        return min(s.current_temperature_c for s in self.sensors)

    def get_average_temperature_c(self) -> float:
        """Get average temperature across all sensors"""
        if not self.sensors:
            return 45.0
        return sum(s.current_temperature_c for s in self.sensors) / len(self.sensors)

    def get_min_thermal_margin_c(self) -> float:
        """Get minimum thermal margin across all sensors"""
        if not self.sensors:
            return MAX_JUNCTION_TEMP_C - 45.0
        return min(MAX_JUNCTION_TEMP_C - s.current_temperature_c for s in self.sensors)

    def get_zone_temperature(self, zone: ThermalZone) -> Optional[float]:
        """Get temperature for specific zone

        Args:
            zone: Thermal zone to query

        Returns:
            Temperature or None if zone not found
        """
        for sensor in self.sensors:
            if sensor.config.zone == zone:
                return sensor.current_temperature_c
        return None

    def reset_all(self):
        """Reset all sensors"""
        for sensor in self.sensors:
            sensor.reset()


# Thermal constants
MAX_JUNCTION_TEMP_C = 85.0          # Maximum junction temperature (C)
THERMAL_THROTTLE_THRESHOLD_C = 75.0 # Throttle threshold (C)
THERMAL_SHUTDOWN_TEMP_C = 95.0      # Shutdown temperature (C)
THERMAL_RESISTANCE_C_PER_W = 20.0   # Thermal resistance (C/W)
THERMAL_TIME_CONSTANT_MS = 100.0    # Thermal time constant (ms)
MAX_HISTORY_SAMPLES = 10000          # Maximum samples in history


def create_default_sensor_array(sensor_count: int = 8) -> SensorArray:
    """Create default sensor array configuration

    Args:
        sensor_count: Number of sensors to create

    Returns:
        Configured SensorArray
    """
    return SensorArray(sensor_count=sensor_count)


def create_sensor_for_zone(zone: ThermalZone, sensor_id: int = 0) -> ThermalSensor:
    """Create sensor configured for specific zone

    Args:
        zone: Thermal zone
        sensor_id: Sensor identifier

    Returns:
        Configured ThermalSensor
    """
    config = SensorConfiguration(
        sensor_id=sensor_id,
        zone=zone,
        sensor_type=SensorType.ON_DIE_BANDGAP,
    )
    return ThermalSensor(config)


def simulate_temperature_reading(
    true_temp_c: float,
    sensor_type: SensorType = SensorType.ON_DIE_BANDGAP,
    noise_level: float = 0.5
) -> Tuple[float, float]:
    """Simulate temperature reading from sensor

    Args:
        true_temp_c: True temperature
        sensor_type: Type of sensor
        noise_level: Noise level (C RMS)

    Returns:
        Tuple of (raw_reading, calibrated_reading)
    """
    # Add sensor-type-specific characteristics
    type_noise = {
        SensorType.ON_DIE_BANDGAP: 0.3,
        SensorType.THERMOCOUPLE: 0.5,
        SensorType.RESISTANCE: 0.4,
        SensorType.DIGITAL: 0.2,
    }

    effective_noise = max(noise_level, type_noise.get(sensor_type, 0.5))
    raw = true_temp_c + random.gauss(0, effective_noise)

    # Simple calibration (offset correction)
    calibrated = raw  # In practice, would apply more complex calibration

    return raw, calibrated