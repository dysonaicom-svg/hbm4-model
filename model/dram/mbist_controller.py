"""
MBIST Controller for HBM4 DRAM

Implements Memory Built-In Self-Test (MBIST) for HBM4 DRAM verification.
Supports multiple test algorithms including March tests, Walking patterns,
Address tests, and Data retention tests.

Key features:
- Multiple MBIST algorithms (March-C, March-L, March-U, Walking Ones/Zeros)
- Address pattern testing
- Data retention testing
- Fault detection and classification
- Integration with HBM4ChannelModel

Based on:
- JEDEC JESD270-4A HBM4 specification
- Cadence HBM4E documentation
- Standard MBIST patterns (March-M, March-S, etc.)

Reference:
- HBM4 Controller Integration: model/controller/hbm4_controller.py
- HBM4 Channel Model: model/dram/hbm4_channel_model.py
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from typing import Generator
import copy


class MBISTState(Enum):
    """MBIST Controller States"""
    IDLE = 0
    SETUP = 1
    RUNNING = 2
    COMPLETE = 3
    FAIL = 4


class MBISTAlgorithm(Enum):
    """Supported MBIST Algorithms"""
    MARCH_C = "march_c"       # March-C: compact fault detection
    MARCH_L = "march_l"       # March-L: linked fault detection
    MARCH_U = "march_u"       # March-U: unlinked fault detection
    MARCH_MINUS = "march_minus"  # March-: simplified pattern
    MARCH_PLUS = "march_plus"   # March+: extended pattern
    WALKING_ONES = "walking_ones"  # Walking 1s pattern
    WALKING_ZEROS = "walking_zeros"  # Walking 0s pattern
    ADDRESS_TEST = "address_test"    # Address decoder test
    DATA_RETENTION = "data_retention"  # Data retention test
    GALPAT = "galpat"           # Galloping pattern (column/row)


class FaultType(Enum):
    """Detected Fault Types"""
    STUCK_AT_0 = "stuck_at_0"      # Cell always reads 0
    STUCK_AT_1 = "stuck_at_1"      # Cell always reads 1
    TRANSITION = "transition"      # Failed to transition
    ADDRESS_DECODE = "address_decode"  # Address decoder fault
    COUPLING = "coupling"          # Coupling fault
    DATA_RETENTION = "data_retention"  # Data retention fault
    READ_DISTURB = "read_disturb"   # Read disturb fault


@dataclass
class MBISTConfig:
    """MBIST Test Configuration"""
    algorithm: MBISTAlgorithm = MBISTAlgorithm.MARCH_C
    start_address: int = 0
    end_address: int = 0xFFFFFFFF  # Full 32-bit address range
    channel_mask: int = 0xFFFFFFFF  # All 32 channels
    bank_mask: int = 0xFFFF        # All 16 banks per pseudo-channel
    row_start: int = 0
    row_end: int = 0xFFFF          # Full row range
    timeout_cycles: int = 1000000   # Safety timeout
    retention_time_cycles: int = 10000  # For data retention test
    fail_stop: bool = True          # Stop on first failure
    verify_mode: bool = True        # Verify expected vs actual data


@dataclass
class MBISTFault:
    """Detected MBIST Fault"""
    fault_type: FaultType
    address: int
    expected: int
    actual: int
    cycle: int
    algorithm: MBISTAlgorithm
    channel: int
    bank: int
    row: int
    column: int
    bit_position: Optional[int] = None  # For bit-level faults


@dataclass
class MBISTResult:
    """MBIST Test Result"""
    test_name: str
    algorithm: MBISTAlgorithm
    start_time: int
    end_time: int
    cycles_executed: int
    addresses_tested: int
    faults_found: List[MBISTFault] = field(default_factory=list)
    status: str = "UNKNOWN"
    pass_count: int = 0
    fail_count: int = 0

    @property
    def total_cycles(self) -> int:
        return self.end_time - self.start_time

    @property
    def passed(self) -> bool:
        return len(self.faults_found) == 0

    @property
    def fault_rate(self) -> float:
        if self.addresses_tested == 0:
            return 0.0
        return len(self.faults_found) / self.addresses_tested


@dataclass
class MBISTStats:
    """MBIST Statistics"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    total_cycles: int = 0
    total_addresses_tested: int = 0
    total_faults: int = 0
    stuck_at_0_count: int = 0
    stuck_at_1_count: int = 0
    transition_count: int = 0
    coupling_count: int = 0
    address_decode_count: int = 0
    data_retention_count: int = 0
    read_disturb_count: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    @property
    def fault_coverage(self) -> Dict[str, float]:
        """Fault type distribution"""
        total = self.total_faults
        if total == 0:
            return {}
        return {
            "stuck_at_0": self.stuck_at_0_count / total,
            "stuck_at_1": self.stuck_at_1_count / total,
            "transition": self.transition_count / total,
            "coupling": self.coupling_count / total,
            "address_decode": self.address_decode_count / total,
            "data_retention": self.data_retention_count / total,
            "read_disturb": self.read_disturb_count / total,
        }


class MarchPattern:
    """March Pattern Element"""

    def __init__(self, operation: str, address_order: str = "up"):
        """
        Args:
            operation: Operation string (e.g., "w0", "r0", "w1", "ra", "wa")
                      w = write, r = read, ra = read and expect 0, wa = read and expect 1
            address_order: "up" for ascending, "down" for descending
        """
        self.operation = operation
        self.address_order = address_order

    def __repr__(self) -> str:
        return f"March({self.operation}, {self.address_order})"


class MBISTController:
    """MBIST Controller for HBM4 DRAM

    Implements comprehensive memory BIST for HBM4 verification.
    Supports multiple algorithms and integrates with HBM4ChannelModel.
    """

    # March algorithm patterns
    MARCH_PATTERNS = {
        MBISTAlgorithm.MARCH_C: [
            MarchPattern("w0"),
            MarchPattern("r0", "up"),
            MarchPattern("w1", "up"),
            MarchPattern("r1", "up"),
            MarchPattern("w0", "up"),
            MarchPattern("r0", "down"),
        ],
        MBISTAlgorithm.MARCH_L: [
            MarchPattern("w0"),
            MarchPattern("r0", "up"),
            MarchPattern("w1", "up"),
            MarchPattern("r1", "down"),
            MarchPattern("w0", "down"),
            MarchPattern("r0", "up"),
        ],
        MBISTAlgorithm.MARCH_U: [
            MarchPattern("w0"),
            MarchPattern("r0", "up"),
            MarchPattern("w1", "up"),
            MarchPattern("r1", "up"),
            MarchPattern("w0", "up"),
            MarchPattern("r0", "up"),
            MarchPattern("w1", "up"),
            MarchPattern("r1", "up"),
        ],
        MBISTAlgorithm.MARCH_MINUS: [
            MarchPattern("w0"),
            MarchPattern("r0", "up"),
            MarchPattern("w1", "up"),
            MarchPattern("r1", "up"),
            MarchPattern("w0", "up"),
            MarchPattern("r0", "up"),
        ],
        MBISTAlgorithm.MARCH_PLUS: [
            MarchPattern("w0"),
            MarchPattern("r0", "up"),
            MarchPattern("w1", "up"),
            MarchPattern("r1", "up"),
            MarchPattern("w0", "up"),
            MarchPattern("r0", "up"),
            MarchPattern("w1", "up"),
            MarchPattern("r1", "up"),
            MarchPattern("w0", "up"),
            MarchPattern("r0", "up"),
        ],
    }

    def __init__(
        self,
        channel_model: Optional[Any] = None,
        spec: Optional[Any] = None,
    ):
        """Initialize MBIST Controller

        Args:
            channel_model: HBM4ChannelModel for DRAM access
            spec: HBM4 specification
        """
        self.channel_model = channel_model
        self.spec = spec

        # State machine
        self.state = MBISTState.IDLE
        self.current_cycle = 0
        self.config = MBISTConfig()
        self.algorithm_patterns: List[MarchPattern] = []

        # Current test state
        self.current_address = 0
        self.current_pattern_index = 0
        self.test_data: Dict[int, int] = {}  # address -> data
        self.expected_data: int = 0

        # Results
        self.current_result: Optional[MBISTResult] = None
        self.results: List[MBISTResult] = []
        self.stats = MBISTStats()

        # Memory interface (for standalone mode without channel model)
        self.memory_array: Dict[int, int] = {}  # address -> data
        # Fault injection map - injected faults take precedence
        self.fault_map: Dict[int, int] = {}  # address -> fault value (for stuck-at faults)

    def configure(self, config: MBISTConfig) -> None:
        """Configure MBIST test parameters

        Args:
            config: MBIST configuration
        """
        self.config = config
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load test patterns based on algorithm"""
        if self.config.algorithm in self.MARCH_PATTERNS:
            self.algorithm_patterns = self.MARCH_PATTERNS[self.config.algorithm]
        else:
            # For non-March algorithms, patterns are generated on-the-fly
            self.algorithm_patterns = []

    def start_test(self, test_name: str = "") -> bool:
        """Start MBIST test

        Args:
            test_name: Optional test name for result tracking

        Returns:
            True if test started successfully
        """
        if self.state != MBISTState.IDLE:
            return False

        self.state = MBISTState.SETUP
        self.current_cycle = 0
        self.current_address = self.config.start_address
        self.current_pattern_index = 0
        self.test_data = {}

        name = test_name or f"MBIST_{self.config.algorithm.value}"
        self.current_result = MBISTResult(
            test_name=name,
            algorithm=self.config.algorithm,
            start_time=self.current_cycle,
            end_time=0,
            cycles_executed=0,
            addresses_tested=0,
        )

        self.state = MBISTState.RUNNING
        return True

    def tick(self) -> bool:
        """Execute one MBIST cycle

        Returns:
            True if test still running, False if test completed or failed
        """
        self.current_cycle += 1

        if self.state == MBISTState.IDLE:
            return False

        if self.state == MBISTState.SETUP:
            self.state = MBISTState.RUNNING
            return True

        if self.state == MBISTState.RUNNING:
            return self._run_test_cycle()

        if self.state == MBISTState.COMPLETE:
            return False

        if self.state == MBISTState.FAIL:
            return False

        return False

    def _run_test_cycle(self) -> bool:
        """Execute one test cycle

        Returns:
            True if test continues, False if test finished
        """
        # Check timeout
        if self.current_cycle > self.config.timeout_cycles:
            self._complete_test("TIMEOUT")
            return False

        # Check if all patterns executed
        if self.current_pattern_index >= len(self.algorithm_patterns):
            self._complete_test("PASS")
            return False

        # Execute current pattern element
        pattern = self.algorithm_patterns[self.current_pattern_index]
        done = self._execute_pattern_element(pattern)

        if done:
            self.current_pattern_index += 1
            self.current_address = self.config.start_address  # Reset address for next pattern

        return self.state == MBISTState.RUNNING

    def _execute_pattern_element(self, pattern: MarchPattern) -> bool:
        """Execute a single March pattern element

        Args:
            pattern: March pattern to execute

        Returns:
            True if pattern element completed
        """
        # Get address range
        addresses = self._generate_addresses(pattern.address_order)

        for addr in addresses:
            self.current_address = addr
            op = pattern.operation.lower()

            # Execute operation
            if op == "w0":
                self._write_data(addr, 0)
            elif op == "w1":
                self._write_data(addr, 0xFFFFFFFFFFFFFFFF)
            elif op == "r0":
                self._read_and_verify(addr, 0)
            elif op == "r1":
                self._read_and_verify(addr, 0xFFFFFFFFFFFFFFFF)
            elif op == "ra":
                self._read_and_verify(addr, 0, fail_on_mismatch=True)
            elif op == "wa":
                self._read_and_verify(addr, 0xFFFFFFFFFFFFFFFF, fail_on_mismatch=True)
            else:
                # Unknown operation - skip
                pass

            # Check if we should stop on fail (fail_stop=True only)
            if self.current_result and len(self.current_result.faults_found) > 0:
                if self.config.fail_stop:
                    self._complete_test("FAIL")
                    return False

            # Check timeout
            if self.current_cycle > self.config.timeout_cycles:
                self._complete_test("TIMEOUT")
                return False

        # Pattern element completed (all addresses processed)
        return True

    def _generate_addresses(self, order: str) -> List[int]:
        """Generate address list for test

        Args:
            order: "up" or "down"

        Returns:
            List of addresses to test
        """
        addresses = []
        addr = self.current_address

        if order == "up":
            while addr <= self.config.end_address:
                addresses.append(addr)
                addr += 1
        else:  # "down"
            # Start from end_address and go down to start_address
            addr = self.config.end_address
            while addr >= self.config.start_address:
                addresses.append(addr)
                addr -= 1

        return addresses

    def _write_data(self, address: int, data: int) -> None:
        """Write data to memory

        Args:
            address: Target address
            data: Data to write
        """
        self.test_data[address] = data

        # Don't write to memory if there's an injected fault at this address
        if address in self.fault_map:
            return

        if self.channel_model:
            # Use channel model for actual DRAM access
            self._write_to_channel(address, data)
        else:
            # Standalone mode - use local memory
            self.memory_array[address] = data

    def _read_data(self, address: int) -> int:
        """Read data from memory

        Args:
            address: Source address

        Returns:
            Read data (checks fault_map first for injected faults)
        """
        # Check fault_map first (for injected faults - takes highest precedence)
        # This simulates hardware faults where memory cell returns wrong value
        if address in self.fault_map:
            return self.fault_map[address]

        # Then check memory_array
        if address in self.memory_array:
            return self.memory_array[address]

        # Then check test_data (for normal test operations)
        if address in self.test_data:
            return self.test_data[address]

        if self.channel_model:
            return self._read_from_channel(address)
        else:
            return 0

    def _write_to_channel(self, address: int, data: int) -> None:
        """Write to HBM4 channel

        Args:
            address: Physical address
            data: Data to write
        """
        if not self.channel_model:
            return

        # Decode address for channel model
        from model.controller.hbm4_address_decoder import HBM4AddressDecoder
        decoder = HBM4AddressDecoder(spec=self.spec)

        decoded = decoder.decode(address)
        channel = self.channel_model.channels[decoded.channel_id]

        # Issue write command
        channel.issue_command(
            'WR',
            pseudo_channel=decoded.pseudo_channel_id,
            bank=decoded.bank_id,
            row=decoded.row_id,
            col=decoded.col_id
        )

    def _read_from_channel(self, address: int) -> int:
        """Read from HBM4 channel

        Args:
            address: Physical address

        Returns:
            Read data
        """
        if not self.channel_model:
            return 0

        from model.controller.hbm4_address_decoder import HBM4AddressDecoder
        decoder = HBM4AddressDecoder(spec=self.spec)

        decoded = decoder.decode(address)
        channel = self.channel_model.channels[decoded.channel_id]

        # Issue read command
        channel.issue_command(
            'RD',
            pseudo_channel=decoded.pseudo_channel_id,
            bank=decoded.bank_id,
            row=decoded.row_id,
            col=decoded.col_id
        )

        return 0  # Would need async response handling in real model

    def _read_and_verify(
        self,
        address: int,
        expected: int,
        fail_on_mismatch: bool = False
    ) -> bool:
        """Read and verify data

        Args:
            address: Address to read
            expected: Expected data value
            fail_on_mismatch: If True, stop test on mismatch

        Returns:
            True if data matches expected
        """
        actual = self._read_data(address)
        match = actual == expected

        if not match and self.current_result:
            fault = self._create_fault(address, expected, actual)
            self.current_result.faults_found.append(fault)
            self.current_result.fail_count += 1
            self._update_stats(fault)

            if fail_on_mismatch:
                self._complete_test("FAIL")
        elif match:
            if self.current_result:
                self.current_result.pass_count += 1

        if self.current_result:
            self.current_result.addresses_tested += 1

        return match

    def _create_fault(
        self,
        address: int,
        expected: int,
        actual: int
    ) -> MBISTFault:
        """Create a fault record

        Args:
            address: Fault address
            expected: Expected data
            actual: Actual data

        Returns:
            MBISTFault record
        """
        # Determine fault type
        fault_type = self._classify_fault(expected, actual)

        # Decode address
        from model.controller.hbm4_address_decoder import HBM4AddressDecoder
        if self.spec:
            decoder = HBM4AddressDecoder(spec=self.spec)
            decoded = decoder.decode(address)
            channel = decoded.channel_id
            bank = decoded.bank_id
            row = decoded.row_id
            col = decoded.col_id
        else:
            channel = bank = row = col = 0

        return MBISTFault(
            fault_type=fault_type,
            address=address,
            expected=expected,
            actual=actual,
            cycle=self.current_cycle,
            algorithm=self.config.algorithm,
            channel=channel,
            bank=bank,
            row=row,
            column=col,
        )

    def _classify_fault(self, expected: int, actual: int) -> FaultType:
        """Classify fault type from data comparison

        Args:
            expected: Expected data
            actual: Actual data

        Returns:
            FaultType classification
        """
        if expected == 0 and actual != 0:
            return FaultType.STUCK_AT_0
        if expected != 0 and actual == 0:
            return FaultType.STUCK_AT_1
        if expected != actual:
            # Check if it's a transition fault
            if (expected & actual) == 0 or ((~expected) & actual) == 0:
                return FaultType.TRANSITION
            return FaultType.DATA_RETENTION

        return FaultType.STUCK_AT_0  # Default

    def _update_stats(self, fault: MBISTFault) -> None:
        """Update statistics with fault

        Args:
            fault: Detected fault
        """
        self.stats.total_faults += 1

        if fault.fault_type == FaultType.STUCK_AT_0:
            self.stats.stuck_at_0_count += 1
        elif fault.fault_type == FaultType.STUCK_AT_1:
            self.stats.stuck_at_1_count += 1
        elif fault.fault_type == FaultType.TRANSITION:
            self.stats.transition_count += 1
        elif fault.fault_type == FaultType.COUPLING:
            self.stats.coupling_count += 1
        elif fault.fault_type == FaultType.ADDRESS_DECODE:
            self.stats.address_decode_count += 1
        elif fault.fault_type == FaultType.DATA_RETENTION:
            self.stats.data_retention_count += 1
        elif fault.fault_type == FaultType.READ_DISTURB:
            self.stats.read_disturb_count += 1

    def _complete_test(self, status: str) -> None:
        """Complete current test

        Args:
            status: Final status ("PASS", "FAIL", "TIMEOUT")
        """
        if self.current_result:
            # If completing normally (PASS) but faults were found, mark as FAIL
            if status == "PASS" and len(self.current_result.faults_found) > 0:
                status = "FAIL"

            self.current_result.end_time = self.current_cycle
            self.current_result.cycles_executed = self.current_result.total_cycles
            self.current_result.status = status
            self.results.append(self.current_result)

            # Update stats
            self.stats.total_tests += 1
            self.stats.total_cycles += self.current_result.cycles_executed
            self.stats.total_addresses_tested += self.current_result.addresses_tested

            if status == "PASS":
                self.stats.passed_tests += 1
            else:
                self.stats.failed_tests += 1

        self.state = MBISTState.COMPLETE if status == "PASS" else MBISTState.FAIL

    def run_test(self, config: Optional[MBISTConfig] = None) -> MBISTResult:
        """Run complete MBIST test

        Args:
            config: Optional configuration override

        Returns:
            Test result
        """
        # Ensure we're in IDLE state before starting
        self.reset()

        if config:
            self.configure(config)

        self.start_test()

        while self.tick():
            pass

        # Reset to IDLE for next test
        self.state = MBISTState.IDLE

        return self.current_result or MBISTResult(
            test_name="",
            algorithm=self.config.algorithm,
            start_time=0,
            end_time=0,
            cycles_executed=0,
            addresses_tested=0,
        )

    def run_all_algorithms(self) -> List[MBISTResult]:
        """Run all supported MBIST algorithms

        Returns:
            List of test results
        """
        results = []

        for algo in MBISTAlgorithm:
            config = MBISTConfig(algorithm=algo)
            result = self.run_test(config)
            results.append(result)

        return results

    # === Walking Ones/Zeros Implementation ===

    def run_walking_ones(self) -> MBISTResult:
        """Run Walking Ones test pattern

        Tests each bit position by writing 1 to one bit at a time
        and verifying all other bits remain 0.

        Returns:
            Test result
        """
        config = MBISTConfig(algorithm=MBISTAlgorithm.WALKING_ONES)
        self.configure(config)

        test_name = "WalkingOnes"
        self.current_result = MBISTResult(
            test_name=test_name,
            algorithm=MBISTAlgorithm.WALKING_ONES,
            start_time=self.current_cycle,
            end_time=0,
            cycles_executed=0,
            addresses_tested=0,
        )

        self.start_test(test_name)

        # Walking ones: 64-bit word
        word_size = 64
        addresses = list(range(self.config.start_address,
                             min(self.config.end_address + 1, 256)))

        for addr in addresses:
            for bit in range(word_size):
                # Write pattern: 1 at position 'bit'
                pattern = 1 << bit
                self._write_data(addr, pattern)

                # Verify: read back and check
                read_back = self._read_data(addr)
                if read_back != pattern:
                    fault = self._create_fault(addr, pattern, read_back)
                    self.current_result.faults_found.append(fault)
                    self._update_stats(fault)

                    if self.config.fail_stop:
                        self._complete_test("FAIL")
                        return self.current_result

                # Verify other bits in same word are 0
                for other_bit in range(word_size):
                    if other_bit != bit:
                        # Read other address
                        other_addr = addr + (1 << other_bit)
                        if other_addr < self.config.end_address:
                            data = self._read_data(other_addr)
                            if data != 0:
                                fault = self._create_fault(other_addr, 0, data)
                                self.current_result.faults_found.append(fault)
                                self._update_stats(fault)

                self.current_result.addresses_tested += 1
                self.current_cycle += 1

        self._complete_test("PASS" if len(self.current_result.faults_found) == 0 else "FAIL")
        return self.current_result

    def run_walking_zeros(self) -> MBISTResult:
        """Run Walking Zeros test pattern

        Tests each bit position by writing 0 to one bit at a time
        (with all others as 1) and verifying.

        Returns:
            Test result
        """
        config = MBISTConfig(algorithm=MBISTAlgorithm.WALKING_ZEROS)
        self.configure(config)

        test_name = "WalkingZeros"
        self.current_result = MBISTResult(
            test_name=test_name,
            algorithm=MBISTAlgorithm.WALKING_ZEROS,
            start_time=self.current_cycle,
            end_time=0,
            cycles_executed=0,
            addresses_tested=0,
        )

        self.start_test(test_name)

        word_size = 64
        addresses = list(range(self.config.start_address,
                             min(self.config.end_address + 1, 256)))

        for addr in addresses:
            for bit in range(word_size):
                # Write pattern: 0 at position 'bit', 1 everywhere else
                pattern = ~(1 << bit) & ((1 << word_size) - 1)
                self._write_data(addr, pattern)

                # Verify
                read_back = self._read_data(addr)
                if read_back != pattern:
                    fault = self._create_fault(addr, pattern, read_back)
                    self.current_result.faults_found.append(fault)
                    self._update_stats(fault)

                    if self.config.fail_stop:
                        self._complete_test("FAIL")
                        return self.current_result

                self.current_result.addresses_tested += 1
                self.current_cycle += 1

        self._complete_test("PASS" if len(self.current_result.faults_found) == 0 else "FAIL")
        return self.current_result

    # === Address Test Implementation ===

    def run_address_test(self) -> MBISTResult:
        """Run Address decoder test

        Tests address decoding by:
        1. Writing unique values to consecutive addresses
        2. Reading and verifying each address contains its unique value

        Returns:
            Test result
        """
        config = MBISTConfig(algorithm=MBISTAlgorithm.ADDRESS_TEST)
        self.configure(config)

        test_name = "AddressTest"
        self.current_result = MBISTResult(
            test_name=test_name,
            algorithm=MBISTAlgorithm.ADDRESS_TEST,
            start_time=self.current_cycle,
            end_time=0,
            cycles_executed=0,
            addresses_tested=0,
        )

        self.start_test(test_name)

        # Use address itself as test data (unique per address)
        max_addrs = min(1024, self.config.end_address - self.config.start_address + 1)
        addresses = [self.config.start_address + i for i in range(max_addrs)]

        # Phase 1: Write address as data
        for addr in addresses:
            self._write_data(addr, addr & 0xFFFFFFFFFFFFFFFF)
            self.current_result.addresses_tested += 1

        # Phase 2: Read and verify
        for addr in addresses:
            expected = addr & 0xFFFFFFFFFFFFFFFF
            actual = self._read_data(addr)

            if actual != expected:
                fault = self._create_fault(addr, expected, actual)
                fault.fault_type = FaultType.ADDRESS_DECODE
                self.current_result.faults_found.append(fault)
                self._update_stats(fault)

                if self.config.fail_stop:
                    self._complete_test("FAIL")
                    return self.current_result

            self.current_cycle += 1

        self._complete_test("PASS" if len(self.current_result.faults_found) == 0 else "FAIL")
        return self.current_result

    # === Data Retention Test Implementation ===

    def run_data_retention_test(self) -> MBISTResult:
        """Run Data Retention test

        Tests memory retention by:
        1. Writing pattern to all addresses
        2. Waiting for retention_time_cycles
        3. Reading and verifying data persisted

        Returns:
            Test result
        """
        config = MBISTConfig(algorithm=MBISTAlgorithm.DATA_RETENTION)
        self.configure(config)

        test_name = "DataRetention"
        self.current_result = MBISTResult(
            test_name=test_name,
            algorithm=MBISTAlgorithm.DATA_RETENTION,
            start_time=self.current_cycle,
            end_time=0,
            cycles_executed=0,
            addresses_tested=0,
        )

        self.start_test(test_name)

        retention_patterns = [0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA]
        max_addrs = min(256, self.config.end_address - self.config.start_address + 1)
        addresses = [self.config.start_address + i for i in range(max_addrs)]

        for pattern in retention_patterns:
            # Phase 1: Write pattern
            for addr in addresses:
                self._write_data(addr, pattern)
                self.current_result.addresses_tested += 1

            # Phase 2: Wait retention time
            for _ in range(self.config.retention_time_cycles):
                self.current_cycle += 1

            # Phase 3: Read and verify
            for addr in addresses:
                actual = self._read_data(addr)

                if actual != pattern:
                    fault = self._create_fault(addr, pattern, actual)
                    fault.fault_type = FaultType.DATA_RETENTION
                    self.current_result.faults_found.append(fault)
                    self._update_stats(fault)

                    if self.config.fail_stop:
                        self._complete_test("FAIL")
                        return self.current_result

        self._complete_test("PASS" if len(self.current_result.faults_found) == 0 else "FAIL")
        return self.current_result

    # === GalPat (Galloping Pattern) Test ===

    def run_galpat_test(self) -> MBISTResult:
        """Run Galloping Pattern test

        Tests coupling faults by checking if operations on one address
        affect neighboring addresses.

        Returns:
            Test result
        """
        config = MBISTConfig(algorithm=MBISTAlgorithm.GALPAT)
        self.configure(config)

        test_name = "GalPat"
        self.current_result = MBISTResult(
            test_name=test_name,
            algorithm=MBISTAlgorithm.GALPAT,
            start_time=self.current_cycle,
            end_time=0,
            cycles_executed=0,
            addresses_tested=0,
        )

        self.start_test(test_name)

        # Test a limited number of addresses for GalPat
        max_addrs = min(64, self.config.end_address - self.config.start_address + 1)
        addresses = [self.config.start_address + i for i in range(max_addrs)]

        for base_addr in addresses:
            # Write 0 to base address
            self._write_data(base_addr, 0)

            # Write 1s to all other addresses
            for other_addr in addresses:
                if other_addr != base_addr:
                    self._write_data(other_addr, 0xFFFFFFFFFFFFFFFF)

            # Read base address - should still be 0 (no coupling)
            actual = self._read_data(base_addr)
            if actual != 0:
                fault = self._create_fault(base_addr, 0, actual)
                fault.fault_type = FaultType.COUPLING
                self.current_result.faults_found.append(fault)
                self._update_stats(fault)

                if self.config.fail_stop:
                    self._complete_test("FAIL")
                    return self.current_result

            # Now write 1 to base and 0 to others
            self._write_data(base_addr, 0xFFFFFFFFFFFFFFFF)

            # Read base - should be 1
            actual = self._read_data(base_addr)
            if actual != 0xFFFFFFFFFFFFFFFF:
                fault = self._create_fault(base_addr, 0xFFFFFFFFFFFFFFFF, actual)
                fault.fault_type = FaultType.COUPLING
                self.current_result.faults_found.append(fault)
                self._update_stats(fault)

                if self.config.fail_stop:
                    self._complete_test("FAIL")
                    return self.current_result

            self.current_result.addresses_tested += 1
            self.current_cycle += 1

        self._complete_test("PASS" if len(self.current_result.faults_found) == 0 else "FAIL")
        return self.current_result

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary

        Returns:
            Dictionary with test summary
        """
        return {
            'stats': {
                'total_tests': self.stats.total_tests,
                'passed_tests': self.stats.passed_tests,
                'failed_tests': self.stats.failed_tests,
                'pass_rate': f"{self.stats.pass_rate:.1%}",
                'total_cycles': self.stats.total_cycles,
                'total_addresses': self.stats.total_addresses_tested,
                'total_faults': self.stats.total_faults,
                'fault_coverage': self.stats.fault_coverage,
            },
            'results': [
                {
                    'name': r.test_name,
                    'algorithm': r.algorithm.value,
                    'status': r.status,
                    'cycles': r.cycles_executed,
                    'addresses': r.addresses_tested,
                    'faults': len(r.faults_found),
                }
                for r in self.results
            ],
            'current_state': self.state.name,
        }

    def reset(self) -> None:
        """Reset MBIST controller to IDLE state

        Clears test state but preserves injected faults.
        """
        self.state = MBISTState.IDLE
        self.current_cycle = 0
        self.current_address = 0
        self.current_pattern_index = 0
        self.test_data = {}
        self.current_result = None
        # Note: fault_map is intentionally NOT cleared here because injected
        # faults should persist across resets - they represent permanent hardware
        # faults that need to be detected throughout the test session.

    def inject_fault(
        self,
        address: int,
        fault_type: FaultType,
        value: int = 0
    ) -> None:
        """Inject a fault for testing

        Args:
            address: Address to inject fault at
            fault_type: Type of fault to inject
            value: Value to inject (for stuck-at faults)
        """
        if fault_type == FaultType.STUCK_AT_0:
            self.fault_map[address] = 0
        elif fault_type == FaultType.STUCK_AT_1:
            self.fault_map[address] = 0xFFFFFFFFFFFFFFFF
        elif fault_type == FaultType.ADDRESS_DECODE:
            # Map this address to another location using fault_map
            self.fault_map[address] = value
            self.fault_map[(address + 1) % 256] = value  # Alias
        else:
            self.fault_map[address] = value


# === Factory Functions ===

def create_mbist_controller(
    channel_model: Optional[Any] = None,
    spec: Optional[Any] = None
) -> MBISTController:
    """Create MBIST controller with default configuration

    Args:
        channel_model: Optional HBM4 channel model
        spec: Optional HBM4 specification

    Returns:
        Configured MBIST controller
    """
    return MBISTController(channel_model=channel_model, spec=spec)


def create_mbist_config(
    algorithm: MBISTAlgorithm = MBISTAlgorithm.MARCH_C,
    start_addr: int = 0,
    end_addr: int = 0xFFFF,
    fail_stop: bool = True
) -> MBISTConfig:
    """Create MBIST configuration

    Args:
        algorithm: MBIST algorithm to use
        start_addr: Start address for test
        end_addr: End address for test
        fail_stop: Stop on first failure

    Returns:
        MBIST configuration
    """
    return MBISTConfig(
        algorithm=algorithm,
        start_address=start_addr,
        end_address=end_addr,
        fail_stop=fail_stop,
    )
