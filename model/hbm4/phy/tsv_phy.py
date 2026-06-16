"""
TSV PHY Abstraction for HBM4 Logic Base Die

Models Through-Silicon Via (TSV) PHY for HBM4, including:
- TSV group mapping to channels
- Signal integrity proxy (BER estimation)
- Latency modeling (fixed + variability)
- Power estimation for TSV PHY
- Training state machine abstraction
- Lane mapping abstraction

Based on:
- JEDEC JESD270-4A HBM4 specification
- TSV technology parameters for 2.5D/3D integration
- Cadence/Synopsys HBM4E PHY documentation
- Multi-agent research findings (2026-06-15)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import random
import math


class TSVGroupType(Enum):
    """TSV group types in HBM4"""
    DATA = "data"           # DQ, DQS signals
    ADDRESS = "address"     # Command/address signals
    CONTROL = "control"     # Clock, reset, control
    POWER = "power"          # Power/ground TSVs
    RESERVED = "reserved"    # Reserved spares


class TrainingState(Enum):
    """PHY training state machine states"""
    NOT_STARTED = "not_started"
    INIT = "init"
    WRITE_LEVELING = "write_leveling"
    READ_GATE_TRAINING = "read_gate_training"
    READ_DQ_TRAINING = "read_dq_training"
    WRITE_DQ_TRAINING = "write_dq_training"
    VREF_CALIBRATION = "vref_calibration"
    MARGIN_CHECK = "margin_check"
    COMPLETE = "complete"
    FAILED = "failed"


class BEREstimate(Enum):
    """Bit error rate estimate categories"""
    EXCELLENT = "excellent"      # < 1e-15
    GOOD = "good"               # 1e-15 to 1e-12
    ACCEPTABLE = "acceptable"   # 1e-12 to 1e-9
    POOR = "poor"               # 1e-9 to 1e-6
    FAILED = "failed"           # > 1e-6


@dataclass
class TSVGroup:
    """TSV group configuration"""
    group_id: int
    group_type: TSVGroupType
    channel_id: int
    tsv_count: int
    pitch_nm: float
    resistance_ohm: float = 0.1      # TSV resistance
    capacitance_ff: float = 50.0     # TSV capacitance (fF)
    inductance_ph: float = 0.5       # TSV inductance (pH)
    is_active: bool = True

    @property
    def impedance(self) -> complex:
        """Characteristic impedance of TSV"""
        # Z = sqrt(L/C) for transmission line approximation
        l = self.inductance_ph * 1e-12
        c = self.capacitance_ff * 1e-15
        return complex(math.sqrt(l / c), 0)

    @property
    def delay_ps(self) -> float:
        """Propagation delay through TSV"""
        # tpd = sqrt(L*C) approximation
        l = self.inductance_ph * 1e-12
        c = self.capacitance_ff * 1e-15
        return math.sqrt(l * c) * 1e12


@dataclass
class LaneMapping:
    """Lane to physical TSV mapping"""
    lane_id: int
    channel_id: int
    tsv_indices: List[int]       # TSVs for this lane
    is_remapped: bool = False
    remapped_to: Optional[int] = None


@dataclass
class SignalIntegrityMetrics:
    """Signal integrity metrics for a TSV group"""
    group_id: int
    ber_estimate: float          # Bit error rate
    eye_width_ps: float           # valid data window
    eye_height_mv: float          # voltage margin
    jitter_ps: float              # jitter contribution
    noise_floor_dbm: float        # noise floor
    crosstalk_db: float           # crosstalk isolation
    reflections_db: float        # reflection coefficient

    @property
    def ber_category(self) -> BEREstimate:
        """Categorize BER estimate"""
        if self.ber_estimate < 1e-15:
            return BEREstimate.EXCELLENT
        elif self.ber_estimate < 1e-12:
            return BEREstimate.GOOD
        elif self.ber_estimate < 1e-9:
            return BEREstimate.ACCEPTABLE
        elif self.ber_estimate < 1e-6:
            return BEREstimate.POOR
        return BEREstimate.FAILED


@dataclass
class LatencyComponent:
    """Latency component breakdown"""
    fixed_latency_ps: float       # Fixed propagation delay
    variability_ps: float       # Process/voltage/temperature variation
    interconnect_delay_ps: float # Wire interconnect delay
    serializer_delay_ps: float    # SerDes overhead

    @property
    def total_latency_ps(self) -> float:
        """Total latency including variability"""
        return self.fixed_latency_ps + self.variability_ps

    @property
    def worst_case_ps(self) -> float:
        """Worst-case latency (3-sigma)"""
        return self.fixed_latency_ps + 3 * self.variability_ps

    @property
    def best_case_ps(self) -> float:
        """Best-case latency (minus 3-sigma)"""
        return max(0, self.fixed_latency_ps - 3 * self.variability_ps)


@dataclass
class TrainingResult:
    """Training sequence result"""
    state: TrainingState
    group_id: int
    delay_codes: List[int]
    vref_code: int
    eye_width_ps: float
    eye_height_mv: float
    pass_: bool  # Training passed flag (renamed from 'pass' which is reserved)
    message: str = ""

    @property
    def passed(self) -> bool:
        """Alias for pass_ for cleaner access"""
        return self.pass_


@dataclass
class TSVPowerBreakdown:
    """TSV PHY power breakdown"""
    transmitter_mW: float = 0.0
    receiver_mW: float = 0.0
    serializer_mW: float = 0.0
    deserializer_mW: float = 0.0
    clock_recovery_mW: float = 0.0
    calibration_mW: float = 0.0
    leakage_mW: float = 0.0

    @property
    def total_mW(self) -> float:
        """Total TSV PHY power"""
        return (self.transmitter_mW + self.receiver_mW +
                self.serializer_mW + self.deserializer_mW +
                self.clock_recovery_mW + self.calibration_mW +
                self.leakage_mW)


class HBM4TSVPHY:
    """HBM4 TSV PHY Abstraction

    Models the TSV (Through-Silicon Via) physical layer for HBM4
    logic base die to DRAM stack interconnection.

    Key features:
    - TSV group management per channel
    - Signal integrity estimation
    - Latency modeling with variability
    - Training state machine
    - Lane mapping with repair integration
    - Power estimation

    Reference:
    - JEDEC JESD270-4A HBM4
    - TSV technology: 50nm pitch, ~100nm diameter
    - Cadence HBM4E PHY methodology
    """

    # Default TSV parameters
    DEFAULT_TSV_PITCH_NM = 50.0
    DEFAULT_TSV_DIAMETER_NM = 100.0
    DEFAULT_TSV_COUNT_PER_CHANNEL = 1024
    DEFAULT_DATA_TSV_RATIO = 0.7  # 70% data, 30% overhead

    # Latency parameters (ps)
    DEFAULT_TSV_DELAY_PS = 5.0
    DEFAULT_VARIABILITY_PS = 1.0
    DEFAULT_INTERCONNECT_PS = 10.0

    def __init__(
        self,
        num_channels: int = 32,
        tsv_pitch_nm: float = DEFAULT_TSV_PITCH_NM,
        data_rate_gtps: float = 8.0,
        vdd_mv: float = 0.9,
        lane_repair_model=None,
    ):
        """Initialize TSV PHY

        Args:
            num_channels: Number of HBM4 channels (default 32)
            tsv_pitch_nm: TSV pitch in nanometers
            data_rate_gtps: Data rate in GT/s
            vdd_mv: Supply voltage in mV
            lane_repair_model: Optional LaneRepairModel for integration
        """
        self.num_channels = num_channels
        self.tsv_pitch_nm = tsv_pitch_nm
        self.data_rate_gtps = data_rate_gtps
        self.vdd_mv = vdd_mv
        self.lane_repair_model = lane_repair_model

        # TSV group storage
        self._tsv_groups: Dict[int, TSVGroup] = {}
        self._channel_groups: Dict[int, List[int]] = {}

        # Lane mapping
        self._lane_mappings: Dict[int, LaneMapping] = {}

        # Signal integrity tracking
        self._signal_integrity: Dict[int, SignalIntegrityMetrics] = {}

        # Training state machine
        self._training_state = TrainingState.NOT_STARTED
        self._training_results: List[TrainingResult] = []

        # Statistics
        self._init_stats()

        # Initialize TSV groups
        self._init_tsv_groups()

    def _init_stats(self):
        """Initialize statistics tracking"""
        self.stats = {
            'total_tsvs': 0,
            'active_tsvs': 0,
            'failed_tsvs': 0,
            'training_cycles': 0,
            'repaired_lanes': 0,
            'bit_errors': 0,
        }

    def _init_tsv_groups(self):
        """Initialize TSV groups for all channels"""
        # Calculate TSV counts
        total_tsvs = self.num_channels * self.DEFAULT_TSV_COUNT_PER_CHANNEL
        data_tsvs = int(total_tsvs * self.DEFAULT_DATA_TSV_RATIO)
        overhead_tsvs = total_tsvs - data_tsvs

        tsvs_per_channel = self.DEFAULT_TSV_COUNT_PER_CHANNEL
        data_tsvs_per_channel = int(tsvs_per_channel * self.DEFAULT_DATA_TSV_RATIO)

        # Create TSV groups per channel
        group_id = 0
        for ch in range(self.num_channels):
            channel_group_ids = []

            # Data group
            data_group = TSVGroup(
                group_id=group_id,
                group_type=TSVGroupType.DATA,
                channel_id=ch,
                tsv_count=data_tsvs_per_channel,
                pitch_nm=self.tsv_pitch_nm,
            )
            self._tsv_groups[group_id] = data_group
            channel_group_ids.append(group_id)
            group_id += 1

            # Address group
            addr_group = TSVGroup(
                group_id=group_id,
                group_type=TSVGroupType.ADDRESS,
                channel_id=ch,
                tsv_count=64,  # Reduced for address/command
                pitch_nm=self.tsv_pitch_nm,
            )
            self._tsv_groups[group_id] = addr_group
            channel_group_ids.append(group_id)
            group_id += 1

            # Control group
            ctrl_group = TSVGroup(
                group_id=group_id,
                group_type=TSVGroupType.CONTROL,
                channel_id=ch,
                tsv_count=16,  # Clock, reset, etc.
                pitch_nm=self.tsv_pitch_nm,
            )
            self._tsv_groups[group_id] = ctrl_group
            channel_group_ids.append(group_id)
            group_id += 1

            # Power group
            power_group = TSVGroup(
                group_id=group_id,
                group_type=TSVGroupType.POWER,
                channel_id=ch,
                tsv_count=tsvs_per_channel - data_tsvs_per_channel - 64 - 16,
                pitch_nm=self.tsv_pitch_nm,
            )
            self._tsv_groups[group_id] = power_group
            channel_group_ids.append(group_id)
            group_id += 1

            self._channel_groups[ch] = channel_group_ids

        self.stats['total_tsvs'] = sum(g.tsv_count for g in self._tsv_groups.values())
        self.stats['active_tsvs'] = self.stats['total_tsvs']

    # ==================== TSV Group Management ====================

    def get_channel_groups(self, channel_id: int) -> List[TSVGroup]:
        """Get all TSV groups for a channel

        Args:
            channel_id: Channel index (0-31)

        Returns:
            List of TSVGroup objects
        """
        if channel_id not in self._channel_groups:
            return []
        return [self._tsv_groups[gid] for gid in self._channel_groups[channel_id]]

    def get_group_by_id(self, group_id: int) -> Optional[TSVGroup]:
        """Get TSV group by ID

        Args:
            group_id: TSV group index

        Returns:
            TSVGroup or None
        """
        return self._tsv_groups.get(group_id)

    def get_groups_by_type(self, group_type: TSVGroupType) -> List[TSVGroup]:
        """Get all TSV groups of a specific type

        Args:
            group_type: Type filter

        Returns:
            List of matching TSVGroup objects
        """
        return [g for g in self._tsv_groups.values() if g.group_type == group_type]

    def set_group_active(self, group_id: int, active: bool):
        """Set TSV group active state

        Args:
            group_id: TSV group to modify
            active: True to activate, False to deactivate
        """
        if group_id in self._tsv_groups:
            self._tsv_groups[group_id].is_active = active
            if active:
                self.stats['active_tsvs'] += self._tsv_groups[group_id].tsv_count
            else:
                self.stats['active_tsvs'] -= self._tsv_groups[group_id].tsv_count

    # ==================== Signal Integrity ====================

    def estimate_ber(
        self,
        group_id: int,
        data_rate_gtps: Optional[float] = None,
        vdd_mv: Optional[float] = None,
        temperature_c: float = 85.0,
    ) -> SignalIntegrityMetrics:
        """Estimate signal integrity metrics for a TSV group

        Args:
            group_id: TSV group to analyze
            data_rate_gtps: Override data rate
            vdd_mv: Override supply voltage
            temperature_c: Operating temperature (Celsius)

        Returns:
            SignalIntegrityMetrics with BER estimate
        """
        if group_id not in self._tsv_groups:
            raise ValueError(f"Invalid group_id: {group_id}")

        group = self._tsv_groups[group_id]
        rate = data_rate_gtps or self.data_rate_gtps
        vdd = vdd_mv or self.vdd_mv

        # TSV noise model (simplified)
        # Thermal noise: kT/C
        kT = 1.38e-23 * (temperature_c + 273.15)
        c_tsv = group.capacitance_ff * 1e-15
        thermal_noise_v = math.sqrt(kT / c_tsv)

        # Supply noise coupling (simplified)
        vdd_noise = vdd * 0.02 * math.sin(2 * math.pi * 100e6 * 1e-9)  # 100MHz ripple

        # Crosstalk (from adjacent TSVs)
        coupling_factor = 0.05  # 5% coupling
        crosstalk_v = coupling_factor * vdd

        # Total noise
        total_noise = math.sqrt(thermal_noise_v**2 + vdd_noise**2 + crosstalk_v**2)

        # Signal swing
        signal_swing = vdd * 0.8  # 80% of supply

        # Eye opening estimation
        eye_height_mv = (signal_swing - 3 * total_noise) * 1000  # 3-sigma margin
        eye_width_ps = (1.0 / (rate * 2)) * 1e12 * 0.6  # 60% UI

        # Jitter estimation
        t_jitter = 0.02 * (1.0 / rate) * 1e12  # 2% of UI

        # BER calculation using Gaussian approximation
        Q = eye_height_mv / (total_noise * 1000 * 6)  # 6-sigma for BER
        if Q > 0:
            # Q-to-BER conversion (approximation for high Q)
            ber_estimate = 0.5 * math.exp(-Q**2 / 2) / (Q * math.sqrt(2 * math.pi))
        else:
            ber_estimate = 1e-3  # Failed

        # Reflections
        reflection_coeff = 0.02  # 2% reflection from impedance mismatch

        metrics = SignalIntegrityMetrics(
            group_id=group_id,
            ber_estimate=ber_estimate,
            eye_width_ps=max(0, eye_width_ps),
            eye_height_mv=max(0, eye_height_mv),
            jitter_ps=t_jitter,
            noise_floor_dbm=-70.0,
            crosstalk_db=20 * math.log10(1 / coupling_factor),
            reflections_db=20 * math.log10(reflection_coeff),
        )

        self._signal_integrity[group_id] = metrics
        return metrics

    def get_signal_integrity(self, group_id: int) -> Optional[SignalIntegrityMetrics]:
        """Get signal integrity metrics for a group

        Args:
            group_id: TSV group

        Returns:
            SignalIntegrityMetrics or None if not computed
        """
        return self._signal_integrity.get(group_id)

    def update_signal_integrity_from_measurement(
        self,
        group_id: int,
        measured_ber: float,
        eye_width_ps: float,
        eye_height_mv: float,
    ):
        """Update signal integrity from actual measurements

        Args:
            group_id: TSV group
            measured_ber: Measured bit error rate
            eye_width_ps: Measured eye width
            eye_height_mv: Measured eye height
        """
        if group_id not in self._tsv_groups:
            return

        # Create metrics from measurements
        metrics = SignalIntegrityMetrics(
            group_id=group_id,
            ber_estimate=measured_ber,
            eye_width_ps=eye_width_ps,
            eye_height_mv=eye_height_mv,
            jitter_ps=0.0,  # Unknown from BER alone
            noise_floor_dbm=-70.0,
            crosstalk_db=20.0,
            reflections_db=-40.0,
        )
        self._signal_integrity[group_id] = metrics

    # ==================== Latency Modeling ====================

    def get_latency(
        self,
        group_id: int,
        include_variability: bool = True,
    ) -> LatencyComponent:
        """Get latency components for a TSV group

        Args:
            group_id: TSV group
            include_variability: Include PVT variation

        Returns:
            LatencyComponent with breakdown
        """
        if group_id not in self._tsv_groups:
            raise ValueError(f"Invalid group_id: {group_id}")

        group = self._tsv_groups[group_id]

        # Fixed TSV delay
        fixed_latency = group.delay_ps

        # Process variation (simplified)
        process_variation = self.DEFAULT_VARIABILITY_PS * random.uniform(0.5, 1.5)

        # Voltage scaling
        vdd_scale = self.vdd_mv / 0.9
        voltage_variation = self.DEFAULT_VARIABILITY_PS * (1.0 / vdd_scale)

        # Temperature effect
        temp_variation = self.DEFAULT_VARIABILITY_PS * 0.2

        total_variability = process_variation + voltage_variation + temp_variation
        if not include_variability:
            total_variability = 0

        # Interconnect delay
        interconnect_delay = self.DEFAULT_INTERCONNECT_PS * (group.tsv_count / 1000)

        # Serializer overhead (8b/10b encoding)
        serializer_delay = (1.0 / self.data_rate_gtps) * 1e6 * 0.25  # 25% overhead

        return LatencyComponent(
            fixed_latency_ps=fixed_latency,
            variability_ps=total_variability,
            interconnect_delay_ps=interconnect_delay,
            serializer_delay_ps=serializer_delay,
        )

    def get_worst_case_latency_ps(self, channel_id: int) -> float:
        """Get worst-case latency for all groups in a channel

        Args:
            channel_id: Channel to analyze

        Returns:
            Worst-case latency in ps
        """
        groups = self.get_channel_groups(channel_id)
        if not groups:
            return 0

        worst = 0
        for group in groups:
            latency = self.get_latency(group.group_id)
            if latency.worst_case_ps > worst:
                worst = latency.worst_case_ps

        return worst

    def get_best_case_latency_ps(self, channel_id: int) -> float:
        """Get best-case latency for all groups in a channel

        Args:
            channel_id: Channel to analyze

        Returns:
            Best-case latency in ps
        """
        groups = self.get_channel_groups(channel_id)
        if not groups:
            return 0

        best = float('inf')
        for group in groups:
            latency = self.get_latency(group.group_id)
            if latency.best_case_ps < best:
                best = latency.best_case_ps

        return best if best < float('inf') else 0

    # ==================== Lane Mapping ====================

    def init_lane_mappings(self, lanes_per_channel: int = 64):
        """Initialize lane mappings for all channels

        Args:
            lanes_per_channel: Number of data lanes per channel
        """
        lane_id = 0
        for ch in range(self.num_channels):
            for lane in range(lanes_per_channel):
                mapping = LaneMapping(
                    lane_id=lane_id,
                    channel_id=ch,
                    tsv_indices=[lane],  # Simplified 1:1 mapping
                )
                self._lane_mappings[lane_id] = mapping
                lane_id += 1

    def get_lane_mapping(self, lane_id: int) -> Optional[LaneMapping]:
        """Get lane mapping by lane ID

        Args:
            lane_id: Lane index

        Returns:
            LaneMapping or None
        """
        return self._lane_mappings.get(lane_id)

    def remap_lane(
        self,
        lane_id: int,
        spare_tsv_index: int,
    ) -> bool:
        """Remap a lane to a spare TSV

        Args:
            lane_id: Lane to remap
            spare_tsv_index: Spare TSV index to use

        Returns:
            True if remap successful
        """
        if lane_id not in self._lane_mappings:
            return False

        mapping = self._lane_mappings[lane_id]
        mapping.is_remapped = True
        mapping.remapped_to = spare_tsv_index

        self.stats['repaired_lanes'] += 1
        return True

    def is_lane_active(self, lane_id: int) -> bool:
        """Check if a lane is active (not remapped to failed TSV)

        Args:
            lane_id: Lane to check

        Returns:
            True if lane is active
        """
        if lane_id not in self._lane_mappings:
            return False

        mapping = self._lane_mappings[lane_id]

        # Check if lane repair model says it's remapped
        if self.lane_repair_model is not None:
            channel_id = mapping.channel_id
            if self.lane_repair_model.is_lane_remapped(channel_id, mapping.tsv_indices[0]):
                return False

        return True

    # ==================== Training State Machine ====================

    def start_training(self):
        """Start PHY training sequence"""
        self._training_state = TrainingState.INIT
        self._training_results.clear()
        self.stats['training_cycles'] = 0

    def advance_training(self) -> TrainingState:
        """Advance training state machine by one step

        Returns:
            Current training state
        """
        state = self._training_state

        if state == TrainingState.NOT_STARTED:
            self._training_state = TrainingState.INIT
        elif state == TrainingState.INIT:
            self._training_state = TrainingState.WRITE_LEVELING
        elif state == TrainingState.WRITE_LEVELING:
            self._training_state = TrainingState.READ_GATE_TRAINING
        elif state == TrainingState.READ_GATE_TRAINING:
            self._training_state = TrainingState.READ_DQ_TRAINING
        elif state == TrainingState.READ_DQ_TRAINING:
            self._training_state = TrainingState.WRITE_DQ_TRAINING
        elif state == TrainingState.WRITE_DQ_TRAINING:
            self._training_state = TrainingState.VREF_CALIBRATION
        elif state == TrainingState.VREF_CALIBRATION:
            self._training_state = TrainingState.MARGIN_CHECK
        elif state == TrainingState.MARGIN_CHECK:
            self._training_state = TrainingState.COMPLETE
        elif state == TrainingState.COMPLETE:
            pass  # Stay complete
        elif state == TrainingState.FAILED:
            pass  # Stay failed

        self.stats['training_cycles'] += 1
        return self._training_state

    def run_training_sequence(
        self,
        target_groups: Optional[List[int]] = None,
        max_iterations: int = 100,
    ) -> List[TrainingResult]:
        """Run complete training sequence for TSV groups

        Args:
            target_groups: List of group IDs to train (all if None)
            max_iterations: Maximum training iterations per group

        Returns:
            List of TrainingResult for each group
        """
        if target_groups is None:
            target_groups = list(self._tsv_groups.keys())

        results = []

        for group_id in target_groups:
            self.start_training()

            for _ in range(max_iterations):
                state = self.advance_training()

                if state == TrainingState.COMPLETE:
                    break
                elif state == TrainingState.FAILED:
                    break

            # Generate training result
            metrics = self.estimate_ber(group_id)
            result = TrainingResult(
                state=self._training_state,
                group_id=group_id,
                delay_codes=[random.randint(0, 63) for _ in range(8)],
                vref_code=random.randint(0, 127),
                eye_width_ps=metrics.eye_width_ps,
                eye_height_mv=metrics.eye_height_mv,
                pass_=self._training_state == TrainingState.COMPLETE,
                message="Training completed" if self._training_state == TrainingState.COMPLETE else "Training failed",
            )
            results.append(result)
            self._training_results.append(result)

        return results

    def set_training_failed(self, message: str = ""):
        """Mark training as failed

        Args:
            message: Failure reason
        """
        self._training_state = TrainingState.FAILED
        if message:
            # Add to last result if exists
            if self._training_results:
                self._training_results[-1].message = message

    def get_training_state(self) -> TrainingState:
        """Get current training state

        Returns:
            Current TrainingState
        """
        return self._training_state

    def get_training_results(self) -> List[TrainingResult]:
        """Get all training results

        Returns:
            List of TrainingResult
        """
        return self._training_results

    # ==================== Power Estimation ====================

    def estimate_power(self) -> TSVPowerBreakdown:
        """Estimate TSV PHY power consumption

        Returns:
            TSVPowerBreakdown with power components
        """
        # TSV transmitter power (current * voltage * activity)
        i_tx = 0.5e-3  # 0.5mA per TSV @ 0.9V
        total_tsvs = self.stats['active_tsvs']
        activity = 0.3  # 30% activity factor
        tx_power = total_tsvs * i_tx * self.vdd_mv * activity / 1000

        # Receiver power (typically less than transmitter)
        rx_power = tx_power * 0.6

        # Serializer/deserializer power
        serializer_power = self.num_channels * 2.5  # mW per channel
        deserializer_power = self.num_channels * 3.0

        # Clock recovery (CDR) power
        cdr_power = self.num_channels * 1.5  # mW per channel

        # Calibration power (periodic)
        calibration_power = self.num_channels * 0.5

        # Leakage power
        leakage_power = total_tsvs * 0.001  # 1uA per TSV

        return TSVPowerBreakdown(
            transmitter_mW=tx_power,
            receiver_mW=rx_power,
            serializer_mW=serializer_power,
            deserializer_mW=deserializer_power,
            clock_recovery_mW=cdr_power,
            calibration_mW=calibration_power,
            leakage_mW=leakage_power,
        )

    def get_power_mW(self) -> float:
        """Get total TSV PHY power in mW

        Returns:
            Total power in mW
        """
        return self.estimate_power().total_mW

    # ==================== Statistics ====================

    def get_stats(self) -> Dict:
        """Get TSV PHY statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'num_channels': self.num_channels,
            'total_tsvs': self.stats['total_tsvs'],
            'active_tsvs': self.stats['active_tsvs'],
            'failed_tsvs': self.stats['failed_tsvs'],
            'training_cycles': self.stats['training_cycles'],
            'repaired_lanes': self.stats['repaired_lanes'],
            'bit_errors': self.stats['bit_errors'],
            'total_groups': len(self._tsv_groups),
        }

    def get_channel_stats(self, channel_id: int) -> Optional[Dict]:
        """Get statistics for a specific channel

        Args:
            channel_id: Channel to query

        Returns:
            Dictionary with channel statistics
        """
        groups = self.get_channel_groups(channel_id)
        if not groups:
            return None

        total_tsvs = sum(g.tsv_count for g in groups)
        active_tsvs = sum(g.tsv_count for g in groups if g.is_active)

        return {
            'channel_id': channel_id,
            'num_groups': len(groups),
            'total_tsvs': total_tsvs,
            'active_tsvs': active_tsvs,
            'worst_latency_ps': self.get_worst_case_latency_ps(channel_id),
            'best_latency_ps': self.get_best_case_latency_ps(channel_id),
        }

    def get_group_stats(self, group_id: int) -> Optional[Dict]:
        """Get statistics for a specific TSV group

        Args:
            group_id: TSV group to query

        Returns:
            Dictionary with group statistics
        """
        group = self._tsv_groups.get(group_id)
        if group is None:
            return None

        si_metrics = self._signal_integrity.get(group_id)

        return {
            'group_id': group_id,
            'type': group.group_type.value,
            'channel_id': group.channel_id,
            'tsv_count': group.tsv_count,
            'is_active': group.is_active,
            'latency_ps': self.get_latency(group_id).total_latency_ps,
            'ber_estimate': si_metrics.ber_estimate if si_metrics else None,
            'eye_width_ps': si_metrics.eye_width_ps if si_metrics else None,
            'eye_height_mv': si_metrics.eye_height_mv if si_metrics else None,
        }

    def reset_stats(self):
        """Reset statistics"""
        self._init_stats()
        self._training_results.clear()
        self._training_state = TrainingState.NOT_STARTED


# Factory function
def create_tsv_phy(
    num_channels: int = 32,
    speed_grade: str = "8Gbps",
    lane_repair_model=None,
) -> HBM4TSVPHY:
    """Create TSV PHY with specified configuration

    Args:
        num_channels: Number of HBM4 channels
        speed_grade: Speed grade ('8Gbps', '12Gbps', '16Gbps')
        lane_repair_model: Optional LaneRepairModel for integration

    Returns:
        Configured HBM4TSVPHY
    """
    from model.dram.hbm4_spec import HBM4_SPEED_GRADES

    data_rate = HBM4_SPEED_GRADES[speed_grade]['data_rate_gtps']

    return HBM4TSVPHY(
        num_channels=num_channels,
        data_rate_gtps=data_rate,
        lane_repair_model=lane_repair_model,
    )