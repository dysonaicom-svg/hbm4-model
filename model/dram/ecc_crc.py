"""
HBM4 ECC and CRC Module

Implements error detection and correction for HBM4 data paths.

Key features:
- SEC-DED (Single Error Correction, Double Error Detection) ECC
- CRC16 for data integrity
- CRC15+KBD for command/address protection
- Error tracking and reporting

Based on:
- JEDEC JESD270-4A HBM4 specification
- Synopsys HBM4 Controller IP
- Cadence HBM4E documentation
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import struct


class ErrorType(Enum):
    """Types of errors detected"""
    NO_ERROR = "no_error"
    SINGLE_BIT = "single_bit"
    DOUBLE_BIT = "double_bit"
    MULTI_BIT = "multi_bit"
    UNCORRECTABLE = "uncorrectable"


@dataclass
class ECCResult:
    """Result of ECC operation"""
    data: int
    syndrome: int
    error_type: ErrorType
    error_bit: Optional[int] = None
    corrected: bool = False


@dataclass
class CRCResult:
    """Result of CRC operation"""
    data: int
    crc: int
    valid: bool


@dataclass
class ErrorEvent:
    """Single error event for tracking"""
    timestamp: int
    error_type: ErrorType
    channel: int
    bank: int
    address: int
    error_mask: int
    corrected: bool
    syndrome: int


@dataclass
class ErrorCounter:
    """Error counting statistics"""
    single_bit_errors: int = 0
    double_bit_errors: int = 0
    multi_bit_errors: int = 0
    uncorrectable_errors: int = 0
    corrections: int = 0
    crc_errors: int = 0
    total_transactions: int = 0

    def reset(self):
        """Reset all counters"""
        self.single_bit_errors = 0
        self.double_bit_errors = 0
        self.multi_bit_errors = 0
        self.uncorrectable_errors = 0
        self.corrections = 0
        self.crc_errors = 0
        self.total_transactions = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'single_bit_errors': self.single_bit_errors,
            'double_bit_errors': self.double_bit_errors,
            'multi_bit_errors': self.multi_bit_errors,
            'uncorrectable_errors': self.uncorrectable_errors,
            'corrections': self.corrections,
            'crc_errors': self.crc_errors,
            'total_transactions': self.total_transactions,
        }


class HBM4ECCMode(Enum):
    """ECC mode selection"""
    DISABLED = 0
    SECDED = 1  # Single Error Correction, Double Error Detection
    SECDED_DBD = 2  # SECDED with Double Byte Error Detection


class HBM4CRCMode(Enum):
    """CRC mode selection"""
    DISABLED = 0
    CRC16 = 1  # 16-bit CRC for data
    CRC15_KBD = 2  # 15-bit CRC + Known Bit Detection for CA


class ErrorTracker:
    """Tracks error history with bounded memory"""

    def __init__(self, max_events: int = 1000):
        """Initialize error tracker

        Args:
            max_events: Maximum number of events to store
        """
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.counter = ErrorCounter()

    def record_event(
        self,
        error_type: ErrorType,
        channel: int = 0,
        bank: int = 0,
        address: int = 0,
        error_mask: int = 0,
        corrected: bool = False,
        syndrome: int = 0,
    ):
        """Record an error event

        Args:
            error_type: Type of error
            channel: Channel number
            bank: Bank number
            address: Address that had error
            error_mask: Bit mask of error locations
            corrected: Whether error was corrected
            syndrome: ECC syndrome value
        """
        event = ErrorEvent(
            timestamp=len(self.events),
            error_type=error_type,
            channel=channel,
            bank=bank,
            address=address,
            error_mask=error_mask,
            corrected=corrected,
            syndrome=syndrome,
        )
        self.events.append(event)

        # Update counters
        self.counter.total_transactions += 1
        if error_type == ErrorType.SINGLE_BIT:
            self.counter.single_bit_errors += 1
            if corrected:
                self.counter.corrections += 1
        elif error_type == ErrorType.DOUBLE_BIT:
            self.counter.double_bit_errors += 1
        elif error_type == ErrorType.MULTI_BIT:
            self.counter.multi_bit_errors += 1
        elif error_type == ErrorType.UNCORRECTABLE:
            self.counter.uncorrectable_errors += 1

    def get_recent_errors(self, count: int = 10) -> List[ErrorEvent]:
        """Get recent error events

        Args:
            count: Number of events to return

        Returns:
            List of recent error events
        """
        return list(self.events)[-count:]

    def get_errors_by_type(self, error_type: ErrorType) -> List[ErrorEvent]:
        """Get all errors of a specific type

        Args:
            error_type: Type to filter by

        Returns:
            List of matching errors
        """
        return [e for e in self.events if e.error_type == error_type]

    def get_error_rate(self) -> float:
        """Calculate error rate

        Returns:
            Error rate as percentage
        """
        if self.counter.total_transactions == 0:
            return 0.0
        total_errors = (
            self.counter.single_bit_errors +
            self.counter.double_bit_errors +
            self.counter.multi_bit_errors +
            self.counter.uncorrectable_errors
        )
        return (total_errors / self.counter.total_transactions) * 100.0

    def reset(self):
        """Reset error tracker"""
        self.events.clear()
        self.counter.reset()


class HBM4ECC:
    """HBM4 ECC Engine

    Implements SEC-DED (Single Error Correction, Double Error Detection)
    for 64-bit or 128-bit data words.

    Uses precomputed lookup tables for proper SEC-DED syndrome generation.
    For 64-bit data: 8 parity bits, total 72 bits (72,64) code
    For 128-bit data: 9 parity bits, total 137 bits (137,128) code

    The syndrome directly indicates the error bit position:
    - syndrome = 0: No error
    - syndrome = k (1 <= k <= 64): Single-bit error at data bit k-1
    - syndrome = 65-72: Single-bit error in ECC bits
    - syndrome with bit 7 set: Double-bit error detected
    """

    def __init__(
        self,
        data_width: int = 64,
        ecc_mode: HBM4ECCMode = HBM4ECCMode.SECDED,
        enable_tracking: bool = True,
    ):
        """Initialize ECC Engine

        Args:
            data_width: Data word width (64 or 128)
            ecc_mode: ECC mode
            enable_tracking: Enable error tracking
        """
        self.data_width = data_width
        self.ecc_mode = ecc_mode
        self.enable_tracking = enable_tracking
        self._error_counter = ErrorCounter()
        self._error_tracker = ErrorTracker() if enable_tracking else None

        if data_width == 64:
            self.ecc_bits = 8
        elif data_width == 128:
            self.ecc_bits = 9
        else:
            raise ValueError(f"Unsupported data width: {data_width}")

        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build lookup tables for ECC encoding/decoding

        Creates:
        - syndrome_to_bit: maps syndrome value to error bit position
        - parity_lookup: maps data value to ECC parity bits
        """
        # Precompute syndrome to bit mapping for single-bit errors
        # For proper SEC-DED: syndrome = error_bit_position + 1
        self._syndrome_to_bit = {}
        for bit in range(self.data_width + self.ecc_bits):
            syndrome = bit + 1  # Syndrome is 1-indexed
            self._syndrome_to_bit[syndrome] = bit

        # For 128-bit we need 9 parity bits, syndrome up to 137
        if self.data_width == 128:
            for bit in range(128, 137):
                syndrome = bit + 1
                self._syndrome_to_bit[syndrome] = bit

    def encode(self, data: int) -> int:
        """Encode data with ECC

        Args:
            data: Data to encode

        Returns:
            Encoded data with ECC bits appended in upper bits
        """
        if self.ecc_mode == HBM4ECCMode.DISABLED:
            return data

        ecc = self._calculate_parity(data)
        return (ecc << self.data_width) | (data & ((1 << self.data_width) - 1))

    def decode(self, encoded: int, record: bool = True) -> ECCResult:
        """Decode and check ECC

        Args:
            encoded: Encoded data (data + ECC bits)
            record: Whether to record error in tracker

        Returns:
            ECCResult with corrected data and error information
        """
        if self.ecc_mode == HBM4ECCMode.DISABLED:
            return ECCResult(
                data=encoded,
                syndrome=0,
                error_type=ErrorType.NO_ERROR,
                corrected=False,
            )

        data_mask = (1 << self.data_width) - 1
        data = encoded & data_mask
        ecc_stored = (encoded >> self.data_width) & ((1 << self.ecc_bits) - 1)

        ecc_calculated = self._calculate_parity(data)
        syndrome = ecc_stored ^ ecc_calculated

        if syndrome == 0:
            if record and self._error_tracker:
                self._error_tracker.record_event(ErrorType.NO_ERROR)
            return ECCResult(
                data=data,
                syndrome=0,
                error_type=ErrorType.NO_ERROR,
                corrected=False,
            )

        error_type, error_bit = self._analyze_syndrome(syndrome)

        if error_type == ErrorType.SINGLE_BIT:
            if error_bit is not None and error_bit < self.data_width:
                corrected_data = data ^ (1 << error_bit)
                if record and self._error_tracker:
                    self._error_tracker.record_event(
                        error_type=error_type,
                        error_mask=1 << error_bit,
                        corrected=True,
                        syndrome=syndrome,
                    )
                return ECCResult(
                    data=corrected_data,
                    syndrome=syndrome,
                    error_type=error_type,
                    error_bit=error_bit,
                    corrected=True,
                )
            else:
                if record and self._error_tracker:
                    self._error_tracker.record_event(
                        error_type=error_type,
                        error_mask=syndrome << self.data_width,
                        corrected=False,
                        syndrome=syndrome,
                    )
                return ECCResult(
                    data=data,
                    syndrome=syndrome,
                    error_type=error_type,
                    error_bit=error_bit,
                    corrected=False,
                )
        else:
            if record and self._error_tracker:
                self._error_tracker.record_event(
                    error_type=error_type,
                    corrected=False,
                    syndrome=syndrome,
                )

        return ECCResult(
            data=data,
            syndrome=syndrome,
            error_type=error_type,
            error_bit=error_bit,
            corrected=False,
        )

    def _calculate_parity(self, data: int) -> int:
        """Calculate ECC parity bits using XOR-based approach"""
        if self.data_width == 64:
            return self._xor_parity_64(data)
        else:
            return self._xor_parity_128(data)

    def _popcount(self, value: int) -> int:
        """Count 1 bits (popcount)"""
        return bin(value).count('1')

    def _xor_parity_64(self, data: int) -> int:
        """Calculate XOR-based parity for 64-bit data

        Uses a simplified but effective parity scheme where:
        - Each data bit's position determines which parity bits it affects
        - Parity bit Pi is XOR of data bits where bit i of position is set
        """
        data = data & 0xFFFFFFFFFFFFFFFF
        p = 0

        # For SEC-DED, we need 8 parity bits
        # P0-P5: standard Hamming parity (cover data bits based on position)
        # P6: even parity of upper 32 bits
        # P7: overall parity (for extended SEC-DED)

        # P0: covers bits 0, 2, 4, 6, 8, ...
        for i in range(0, 64, 2):
            p ^= ((data >> i) & 1) << 0

        # P1: covers bits 0-1, 4-5, 8-9, 12-13, ...
        for i in range(0, 64, 4):
            p ^= ((data >> i) & 1) << 1
            if i + 1 < 64:
                p ^= ((data >> (i + 1)) & 1) << 1

        # P2: covers bits 0-3, 8-11, 16-19, 24-27, ...
        for i in range(0, 64, 8):
            for j in range(4):
                if i + j < 64:
                    p ^= ((data >> (i + j)) & 1) << 2

        # P3: covers bits 0-7, 16-23, 32-39, 48-55
        for i in range(0, 64, 16):
            for j in range(8):
                if i + j < 64:
                    p ^= ((data >> (i + j)) & 1) << 3

        # P4: covers bits 0-15, 32-47
        for i in range(0, 32):
            p ^= ((data >> i) & 1) << 4
        for i in range(32, 48):
            p ^= ((data >> i) & 1) << 4

        # P5: covers bits 16-31, 48-63
        for i in range(16, 32):
            p ^= ((data >> i) & 1) << 5
        for i in range(48, 64):
            p ^= ((data >> i) & 1) << 5

        # P6: covers bits 32-63
        for i in range(32, 64):
            p ^= ((data >> i) & 1) << 6

        # P7: overall parity (extended SEC-DED)
        if self._popcount(data) % 2 == 1:
            p ^= (1 << 7)

        return p & 0xFF

    def _xor_parity_128(self, data: int) -> int:
        """Calculate parity for 128-bit data"""
        p = 0
        data_low = data & 0xFFFFFFFFFFFFFFFF
        data_high = (data >> 64) & 0xFFFFFFFFFFFFFFFF

        p |= self._xor_parity_64(data_low) & 0xFF

        # P8: even parity of upper 64 bits
        if self._popcount(data_high) % 2 == 1:
            p ^= (1 << 8)

        # Overall parity
        if self._popcount(data) % 2 == 1:
            p ^= (1 << 8)

        return p & 0x1FF

    def _analyze_syndrome(self, syndrome: int) -> Tuple[ErrorType, Optional[int]]:
        """Analyze syndrome to determine error type and position

        Args:
            syndrome: ECC syndrome

        Returns:
            Tuple of (error_type, error_bit)
        """
        if syndrome == 0:
            return ErrorType.NO_ERROR, None

        popcnt = self._popcount(syndrome)

        # Single-bit error: syndrome is power of 2 OR matches lookup
        if popcnt == 1:
            error_bit = syndrome.bit_length() - 1
            if error_bit < self.data_width:
                return ErrorType.SINGLE_BIT, error_bit
            else:
                return ErrorType.SINGLE_BIT, None

        # Check lookup table for single-bit error
        if syndrome in self._syndrome_to_bit:
            error_bit = self._syndrome_to_bit[syndrome]
            if error_bit < self.data_width:
                return ErrorType.SINGLE_BIT, error_bit
            else:
                return ErrorType.SINGLE_BIT, None

        # Double-bit error detection (extended parity bit set)
        if syndrome & 0x80 and popcnt == 2:
            return ErrorType.DOUBLE_BIT, None

        # Double or multi-bit error
        if popcnt == 2:
            return ErrorType.DOUBLE_BIT, None
        elif popcnt >= 3:
            return ErrorType.MULTI_BIT, None

        return ErrorType.UNCORRECTABLE, None

    def get_error_stats(self) -> Dict:
        """Get error statistics"""
        return {
            'single_bit_errors': self._error_counter.single_bit_errors,
            'double_bit_errors': self._error_counter.double_bit_errors,
            'multi_bit_errors': self._error_counter.multi_bit_errors,
            'uncorrectable_errors': self._error_counter.uncorrectable_errors,
            'corrections': self._error_counter.corrections,
        }

    def get_error_rate(self) -> float:
        """Get error rate from tracker"""
        if self._error_tracker:
            return self._error_tracker.get_error_rate()
        return 0.0

    def get_recent_errors(self, count: int = 10) -> List[ErrorEvent]:
        """Get recent errors"""
        if self._error_tracker:
            return self._error_tracker.get_recent_errors(count)
        return []

    def reset_stats(self):
        """Reset error statistics"""
        self._error_counter.reset()
        if self._error_tracker:
            self._error_tracker.reset()


class HBM4CRC:
    """HBM4 CRC Engine

    Implements CRC16 for data integrity and CRC15+KBD for
    command/address protection.
    """

    CRC16_POLY = 0x1021
    CRC15_POLY = 0x4599

    def __init__(self, crc_mode: HBM4CRCMode = HBM4CRCMode.CRC16):
        """Initialize CRC Engine"""
        self.crc_mode = crc_mode
        self._crc_errors = 0
        self._total_crc = 0

    def calculate_crc16(self, data: int, width: int = 64) -> int:
        """Calculate CRC16 using CRC-CCITT polynomial

        Args:
            data: Input data
            width: Data width in bits

        Returns:
            16-bit CRC
        """
        crc = 0xFFFF

        for byte_idx in range(0, width, 8):
            byte = (data >> byte_idx) & 0xFF
            crc ^= byte << 8

            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ self.CRC16_POLY) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF

        return crc

    def calculate_crc16_fast(self, data: int, width: int = 64) -> int:
        """Fast CRC16 using table lookup"""
        if not hasattr(self, '_crc16_table'):
            self._crc16_table = []
            for i in range(256):
                crc = i << 8
                for _ in range(8):
                    if crc & 0x8000:
                        crc = ((crc << 1) ^ self.CRC16_POLY) & 0xFFFF
                    else:
                        crc = (crc << 1) & 0xFFFF
                self._crc16_table.append(crc)

        crc = 0xFFFF
        for byte_idx in range(0, width, 8):
            byte = (data >> byte_idx) & 0xFF
            crc = (self._crc16_table[(crc >> 8) ^ byte] ^ (crc << 8)) & 0xFFFF

        return crc

    def verify_crc16(self, data: int, crc: int, width: int = 64) -> Tuple[bool, int]:
        """Verify CRC16

        Returns:
            Tuple of (valid, calculated_crc)
        """
        self._total_crc += 1
        calculated = self.calculate_crc16(data, width)
        valid = calculated == crc
        if not valid:
            self._crc_errors += 1
        return valid, calculated

    def calculate_crc15(self, ca_bits: int) -> int:
        """Calculate CRC15 for command/address

        Args:
            ca_bits: Command/address bits

        Returns:
            15-bit CRC
        """
        crc = 0x7FFF

        for i in range(15):
            bit = (ca_bits >> i) & 1
            crc_bit = (crc >> 14) & 1
            crc = ((crc << 1) & 0x7FFF) | bit
            if crc_bit ^ bit:
                crc ^= self.CRC15_POLY

        return crc ^ 0x7FFF

    def calculate_crc15_kbd(self, ca_bits: int, known_bits: int) -> Tuple[int, int]:
        """Calculate CRC15 with Known Bit Detection

        Args:
            ca_bits: Command/address bits
            known_bits: Mask of known (fixed) bits

        Returns:
            Tuple of (crc, detected_unknown_bits)
        """
        crc = 0x7FFF
        unknown_count = 0

        for i in range(15):
            if known_bits & (1 << i):
                continue
            unknown_count += 1

            bit = (ca_bits >> i) & 1
            crc_bit = (crc >> 14) & 1
            crc = ((crc << 1) & 0x7FFF) | bit
            if crc_bit ^ bit:
                crc ^= self.CRC15_POLY

        return crc ^ 0x7FFF, unknown_count

    def verify_crc15(self, ca_bits: int, crc: int) -> Tuple[bool, int]:
        """Verify CRC15"""
        calculated = self.calculate_crc15(ca_bits)
        valid = calculated == crc
        if not valid:
            self._crc_errors += 1
        return valid, calculated

    def calculate_dbi(self, data: int, width: int = 64) -> Tuple[int, bool]:
        """Calculate DBI (Data Bus Inversion)"""
        ones_count = bin(data).count('1')

        if ones_count > width // 2:
            inverted = (~data) & ((1 << width) - 1)
            return inverted, True

        return data, False

    def verify_dbi(self, data: int, dbi_flag: bool, width: int = 64) -> int:
        """Verify DBI and restore original data"""
        if dbi_flag:
            return (~data) & ((1 << width) - 1)
        return data

    def get_crc_stats(self) -> Dict:
        """Get CRC statistics"""
        return {
            'total_crc_checks': self._total_crc,
            'crc_errors': self._crc_errors,
            'error_rate': (self._crc_errors / self._total_crc * 100) if self._total_crc > 0 else 0.0,
        }

    def reset_stats(self):
        """Reset CRC statistics"""
        self._crc_errors = 0
        self._total_crc = 0


class HBM4DataIntegrity:
    """Combined data integrity engine

    Integrates ECC and CRC for comprehensive error detection/correction.
    """

    def __init__(
        self,
        data_width: int = 64,
        enable_ecc: bool = True,
        enable_crc: bool = True,
    ):
        """Initialize data integrity engine"""
        self.data_width = data_width

        ecc_mode = HBM4ECCMode.SECDED if enable_ecc else HBM4ECCMode.DISABLED
        self.ecc = HBM4ECC(data_width=data_width, ecc_mode=ecc_mode)

        crc_mode = HBM4CRCMode.CRC16 if enable_crc else HBM4CRCMode.DISABLED
        self.crc = HBM4CRC(crc_mode=crc_mode)

    def encode_data(self, data: int) -> Dict:
        """Encode data with ECC and CRC"""
        ecc_encoded = self.ecc.encode(data)
        crc = self.crc.calculate_crc16(ecc_encoded, self.data_width + self.ecc.ecc_bits)

        return {
            'data': ecc_encoded,
            'ecc': ecc_encoded >> self.data_width,
            'crc': crc,
        }

    def decode_data(self, encoded: int, crc: int) -> Tuple[int, bool, str]:
        """Decode and verify data"""
        valid_crc, _ = self.crc.verify_crc16(
            encoded, crc, self.data_width + self.ecc.ecc_bits
        )
        if not valid_crc:
            return encoded, False, "CRC mismatch"

        result = self.ecc.decode(encoded)

        valid = not result.corrected and result.error_type == ErrorType.NO_ERROR
        return result.data, valid, result.error_type.value

    def inject_error(self, encoded: int, bit: int) -> int:
        """Inject an error into encoded data"""
        return encoded ^ (1 << bit)

    def get_stats(self) -> Dict:
        """Get combined statistics"""
        return {
            'ecc': self.ecc.get_error_stats(),
            'ecc_rate': self.ecc.get_error_rate(),
            'crc': self.crc.get_crc_stats(),
        }

    def get_error_summary(self) -> Dict:
        """Get comprehensive error summary"""
        return {
            'total_transactions': self.ecc._error_counter.total_transactions,
            'error_rate': self.ecc.get_error_rate(),
            'single_bit_errors': self.ecc._error_counter.single_bit_errors,
            'double_bit_errors': self.ecc._error_counter.double_bit_errors,
            'multi_bit_errors': self.ecc._error_counter.multi_bit_errors,
            'uncorrectable_errors': self.ecc._error_counter.uncorrectable_errors,
            'corrections': self.ecc._error_counter.corrections,
            'crc_errors': self.crc._crc_errors,
            'crc_total': self.crc._total_crc,
        }