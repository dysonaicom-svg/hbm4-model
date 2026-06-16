"""
HBM4 PAM3 Signal Model

Implements 3-level Pulse Amplitude Modulation (PAM3) for HBM4 data transmission.
PAM3 allows higher data rates by encoding 2 bits per symbol using 3 voltage levels.

Key features:
- PAM3 encode/decode with -1, 0, +1 levels
- Eye diagram computation for signal integrity analysis
- SNR estimation
- Bit error rate modeling

Based on:
- JEDEC JESD270-4A HBM4 specification
- IEEE 802.3 PAM3 encoding standards
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class PAM3Level(Enum):
    """PAM3 signal levels"""
    NEGATIVE = -1
    ZERO = 0
    POSITIVE = 1


@dataclass
class PAM3EyeDiagram:
    """Eye diagram metrics for PAM3 signal"""
    eye_height: float          # Vertical eye opening (V)
    eye_width: float           # Horizontal eye opening (UI)
    center_level: float        # Center level voltage
    level_spacing: float        # Voltage spacing between levels
    snr_db: float              # Signal-to-noise ratio in dB
    ber_estimate: float        # Estimated bit error rate


@dataclass
class PAM3Symbol:
    """Single PAM3 symbol representation"""
    level: int                 # -1, 0, or +1
    ui_position: float         # Position in unit interval
    amplitude: float           # Signal amplitude at sample time


class PAM3SignalModel:
    """HBM4 PAM3 Signal Model

    PAM3 (3-level Pulse Amplitude Modulation) encodes data using three
    voltage levels: -1, 0, +1. This allows ~1.585 bits per symbol
    compared to 1 bit per symbol for NRZ.

    Symbol encoding (2 bits per symbol):
        00 -> -1 (negative level)
        01 ->  0 (zero level)
        10 ->  0 (zero level)
        11 -> +1 (positive level)

    HBM4 uses PAM3 at higher data rates (8+ Gb/s) for improved bandwidth
    efficiency while maintaining signal integrity.
    """

    # PAM3 level mapping
    LEVELS = [-1, 0, 1]
    LEVEL_TO_BITS = {
        -1: (0, 0),  # 00 -> -1
         0: (0, 1),  # 01 -> 0
         1: (1, 1),  # 11 -> +1
    }
    BITS_TO_LEVEL = {
        (0, 0): -1,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 1,
    }

    def __init__(
        self,
        symbol_rate: float = 8e9,      # 8 Gbaud/s for HBM4 base rate
        voltage_swing: float = 1.0,     # Vdiff swing in volts
        noise_std: float = 0.05,         # Noise standard deviation
        transition_time: float = 0.1,   # UI transition time
    ):
        """Initialize PAM3 Signal Model

        Args:
            symbol_rate: Symbol rate in baud/s (default 8 Gbaud for HBM4)
            voltage_swing: Signal voltage swing (Vdiff)
            noise_std: Standard deviation of noise
            transition_time: Transition time as fraction of UI
        """
        self.symbol_rate = symbol_rate
        self.ui_ps = 1e12 / symbol_rate  # Unit interval in picoseconds
        self.voltage_swing = voltage_swing
        self.noise_std = noise_std
        self.transition_time = transition_time

        # Calculate level voltages
        self.level_voltage = {
            -1: -voltage_swing / 2,
             0: 0,
             1: voltage_swing / 2,
        }
        self.level_spacing = voltage_swing / 2

        # Statistics
        self.symbol_count = 0
        self.error_count = 0

    def encode(self, data_bits: int, num_bits: int) -> List[PAM3Symbol]:
        """Encode data bits into PAM3 symbols

        Args:
            data_bits: Input data (as integer)
            num_bits: Number of bits to encode (must be even)

        Returns:
            List of PAM3Symbol objects
        """
        if num_bits % 2 != 0:
            raise ValueError("num_bits must be even for PAM3 encoding")

        symbols = []
        for i in range(0, num_bits, 2):
            bit1 = (data_bits >> (i + 1)) & 1
            bit0 = (data_bits >> i) & 1
            level = self.BITS_TO_LEVEL[(bit1, bit0)]
            symbols.append(PAM3Symbol(
                level=level,
                ui_position=float(i // 2),
                amplitude=self.level_voltage[level],
            ))

        self.symbol_count += len(symbols)
        return symbols

    def decode(self, symbols: List[PAM3Symbol]) -> Tuple[int, int]:
        """Decode PAM3 symbols back to data bits

        Args:
            symbols: List of PAM3Symbol objects

        Returns:
            Tuple of (decoded_data, num_bits)
        """
        data_bits = 0
        for i, sym in enumerate(symbols):
            level = sym.level
            if level not in self.LEVEL_TO_BITS:
                level = 0  # Default on error
            bits = self.LEVEL_TO_BITS[level]
            bit_pos = i * 2
            if bits[0]:
                data_bits |= (1 << (bit_pos + 1))
            if bits[1]:
                data_bits |= (1 << bit_pos)

        return data_bits, len(symbols) * 2

    def apply_noise(self, symbol: PAM3Symbol, seed: Optional[int] = None) -> PAM3Symbol:
        """Apply simulated noise to a symbol

        Args:
            symbol: Input symbol
            seed: Optional random seed for reproducibility

        Returns:
            Symbol with noise applied
        """
        import random
        if seed is not None:
            random.seed(seed)

        noise = random.gauss(0, self.noise_std)
        noisy_amplitude = symbol.amplitude + noise

        # Determine received level (nearest level decision)
        distances = [(abs(noisy_amplitude - self.level_voltage[l]), l) for l in self.LEVELS]
        distances.sort()
        received_level = distances[0][1]

        return PAM3Symbol(
            level=received_level,
            ui_position=symbol.ui_position,
            amplitude=noisy_amplitude,
        )

    def compute_eye_diagram(
        self,
        num_symbols: int = 1000,
        samples_per_ui: int = 64,
    ) -> PAM3EyeDiagram:
        """Compute eye diagram metrics for PAM3 signal

        Simulates PAM3 transmission and computes eye opening metrics
        to assess signal integrity.

        Args:
            num_symbols: Number of symbols to simulate
            samples_per_ui: Samples per unit interval for resolution

        Returns:
            PAM3EyeDiagram with computed metrics
        """
        import random
        random.seed(42)  # Deterministic for reproducibility

        # Generate random data
        data = random.getrandbits(num_symbols * 2)
        symbols = self.encode(data, num_symbols * 2)

        # Collect samples for eye diagram
        all_samples = []
        for sym in symbols:
            base_v = self.level_voltage[sym.level]
            # Sample at multiple points in UI
            for s in range(samples_per_ui):
                t_fraction = s / samples_per_ui
                # Interpolate between levels
                if t_fraction < self.transition_time:
                    # Transition region - interpolate
                    prev_level = sym.level
                    # Simplified: just use current level with noise
                    sample = base_v + random.gauss(0, self.noise_std)
                else:
                    # Steady state
                    sample = base_v + random.gauss(0, self.noise_std)
                all_samples.append(sample)

        # Compute statistics
        v_neg = [s for s in all_samples if s < -self.level_spacing / 2]
        v_zero = [s for s in all_samples if abs(s) <= self.level_spacing / 2]
        v_pos = [s for s in all_samples if s > self.level_spacing / 2]

        # Eye height: minimum separation between level distributions
        if v_neg and v_zero and v_pos:
            neg_max = max(v_neg) if v_neg else -float('inf')
            zero_min = min(v_zero) if v_zero else float('inf')
            zero_max = max(v_zero) if v_zero else float('inf')
            pos_min = min(v_pos) if v_pos else float('inf')

            eye1 = zero_min - neg_max  # Between -1 and 0
            eye2 = pos_min - zero_max  # Between 0 and +1
            eye_height = min(eye1, eye2) if eye1 > 0 and eye2 > 0 else 0
        else:
            eye_height = self.level_spacing - 2 * self.noise_std * 3

        # Eye width: based on transition time
        eye_width = 1.0 - self.transition_time * 2

        # SNR estimation
        signal_power = (self.voltage_swing / 2) ** 2
        noise_power = self.noise_std ** 2
        snr_linear = signal_power / noise_power if noise_power > 0 else float('inf')
        snr_db = 10 * math.log10(snr_linear) if snr_linear > 0 else 30

        # BER estimate from SNR
        # Approximate Q-function based BER for PAM3
        # Each level decision has 2 boundaries
        Q = math.sqrt(2 * snr_linear) * 0.707  # Approximate for PAM3
        ber_estimate = 0.5 * math.erfc(Q / math.sqrt(2))

        return PAM3EyeDiagram(
            eye_height=max(0, eye_height),
            eye_width=max(0, eye_width),
            center_level=0,
            level_spacing=self.level_spacing,
            snr_db=snr_db,
            ber_estimate=ber_estimate,
        )

    def get_snr_estimate(self) -> float:
        """Estimate signal-to-noise ratio

        Returns:
            SNR in dB
        """
        signal_power = (self.voltage_swing / 2) ** 2
        noise_power = self.noise_std ** 2
        snr_linear = signal_power / noise_power if noise_power > 0 else float('inf')
        return 10 * math.log10(snr_linear) if snr_linear > 0 else 30

    def get_bandwidth_efficiency(self) -> float:
        """Calculate bandwidth efficiency

        Returns:
            Bits per symbol (theoretical maximum ~1.585 for PAM3)
        """
        # PAM3 with Gray-like coding can achieve ~1.585 bits/symbol
        # But practically limited by the 0 level encoding
        return math.log2(3)  # ~1.585 bits per symbol

    def get_stats(self) -> Dict:
        """Get model statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'symbol_count': self.symbol_count,
            'error_count': self.error_count,
            'symbol_rate_gbaud': self.symbol_rate / 1e9,
            'snr_estimate_db': self.get_snr_estimate(),
            'bandwidth_efficiency': self.get_bandwidth_efficiency(),
        }


class HBM4PAM3Encoder:
    """HBM4-specific PAM3 encoder with protocol-specific features

    Handles HBM4-specific encoding including:
    - Command/address encoding
    - Data burst encoding
    - Training pattern insertion
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize HBM4 PAM3 Encoder

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.signal_model = PAM3SignalModel(
            symbol_rate=self.config.get('symbol_rate', 8e9),
            voltage_swing=self.config.get('voltage_swing', 0.8),
            noise_std=self.config.get('noise_std', 0.05),
        )

        # Training patterns for PAM3
        self._training_patterns = self._init_training_patterns()

    def _init_training_patterns(self) -> Dict[str, List[int]]:
        """Initialize training patterns

        Returns:
            Dictionary of training patterns
        """
        patterns = {}

        # PAM3-friendly pattern: balanced levels
        # 32 symbols alternating through all 3 levels
        balanced = []
        for i in range(32):
            level = self.signal_model.LEVELS[i % 3]
            balanced.append(level)
        patterns['balanced'] = balanced

        # All ones (positive level)
        patterns['all_positive'] = [1] * 32

        # All zeros (negative level)
        patterns['all_negative'] = [-1] * 32

        # PRBS-like pattern (pseudo-random)
        import random
        random.seed(0xABCD)
        prbs = []
        lfsr = 0xFFFF
        for _ in range(32):
            bit = (lfsr >> 14) & 1
            prbs.append(self.signal_model.LEVELS[bit] if bit else 0)
            new_bit = ((lfsr >> 13) ^ (lfsr >> 12) ^ (lfsr >> 10) ^ (lfsr >> 9)) & 1
            lfsr = ((lfsr << 1) | new_bit) & 0xFFFF
        patterns['prbs'] = prbs

        return patterns

    def encode_command(self, command: int, cmd_bits: int) -> List[PAM3Symbol]:
        """Encode command/address bits

        Args:
            command: Command bits
            cmd_bits: Number of command bits

        Returns:
            List of PAM3Symbol
        """
        return self.signal_model.encode(command, cmd_bits)

    def encode_data_burst(
        self,
        data: int,
        dq_width: int = 128,
    ) -> List[PAM3Symbol]:
        """Encode data burst

        Args:
            data: Data to encode
            dq_width: DQ width per channel

        Returns:
            List of PAM3Symbol
        """
        return self.signal_model.encode(data, dq_width)

    def insert_training_pattern(
        self,
        pattern_name: str,
        length: int = 32,
    ) -> List[PAM3Symbol]:
        """Insert training pattern

        Args:
            pattern_name: Name of pattern ('balanced', 'prbs', etc.)
            length: Pattern repeat length

        Returns:
            List of PAM3Symbol
        """
        base_pattern = self._training_patterns.get(pattern_name, [0] * 32)
        symbols = []
        for i in range(length):
            level = base_pattern[i % len(base_pattern)]
            symbols.append(PAM3Symbol(
                level=level,
                ui_position=float(i),
                amplitude=self.signal_model.level_voltage[level],
            ))
        return symbols

    def verify_training_pattern(
        self,
        received: List[PAM3Symbol],
        expected_name: str,
    ) -> Tuple[bool, float]:
        """Verify received training pattern

        Args:
            received: Received symbols
            expected_name: Expected pattern name

        Returns:
            Tuple of (verified, error_rate)
        """
        expected = self._training_patterns.get(expected_name, [])
        if not expected:
            return False, 1.0

        errors = 0
        for i, sym in enumerate(received):
            expected_level = expected[i % len(expected)]
            if sym.level != expected_level:
                errors += 1

        error_rate = errors / len(received) if received else 1.0
        return error_rate < 0.01, error_rate  # Pass if < 1% errors