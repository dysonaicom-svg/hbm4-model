"""
Loopback Controller for HBM4 PHY Verification

Implements loopback testing for PHY validation and debugging according to
JEDEC JESD270-4A HBM4 specification.

Key features:
- Multiple loopback modes (PRBS-7, PRBS-15, PRBS-31, fixed patterns, 8N)
- Loopback controller state machine
- PRBS sequence generation
- Error detection and BER calculation
- Lane-level and channel-level loopback support
- Integration with PHY training state machine

Reference:
- JEDEC JESD270-4A HBM4 specification
- Cadence HBM4E documentation
- Synopsys DesignWare HBM4/4E Controller IP
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import random


class LoopbackMode(Enum):
    """Loopback mode enumeration (PH-006)
    
    Defines the available loopback test patterns.
    """
    # PRBS (Pseudo-Random Binary Sequence) modes
    PRBS_7 = auto()       # PRBS-7: 2^7 - 1 = 127 bit pattern
    PRBS_15 = auto()      # PRBS-15: 2^15 - 1 = 32767 bit pattern
    PRBS_31 = auto()      # PRBS-31: 2^31 - 1 = 2147483647 bit pattern
    
    # Fixed pattern modes
    FIXED_ALL_ZEROS = auto()    # All zeros
    FIXED_ALL_ONES = auto()     # All ones
    FIXED_ALTERNATING = auto()  # Alternating 0/1
    
    # 8N mode (8-bit cycling pattern)
    MODE_8N = auto()       # 8-bit cycling through 0x00 to 0xFF


class LoopbackLevel(Enum):
    """Loopback level (where the loopback is performed)"""
    LANE = auto()       # Per-lane loopback
    CHANNEL = auto()    # Per-channel loopback
    STACK = auto()      # Full stack loopback


class LoopbackState(Enum):
    """Loopback Controller State Machine
    
    States:
    - IDLE:空闲状态
    - CONFIGURE: 配置loopback参数
    - RUNNING: 执行loopback测试
    - VERIFY: 验证结果
    - COMPLETE: 完成
    """
    IDLE = auto()
    CONFIGURE = auto()
    RUNNING = auto()
    VERIFY = auto()
    COMPLETE = auto()


class LoopbackResult(Enum):
    """Loopback test result codes"""
    SUCCESS = auto()
    FAIL_TIMEOUT = auto()
    FAIL_ERRORS = auto()
    FAIL_MISMATCH = auto()
    FAIL_CONFIG = auto()


@dataclass
class LoopbackConfig:
    """Loopback configuration parameters"""
    mode: LoopbackMode = LoopbackMode.PRBS_7
    level: LoopbackLevel = LoopbackLevel.LANE
    channel_mask: int = 0xFFFFFFFF  # 32-bit mask for 32 channels
    lane_mask: int = 0xFFFFFFFFFFFFFFFF  # 64-bit mask for 64 lanes per channel
    test_length: int = 10000  # Number of symbols to test
    timeout_cycles: int = 100000  # Timeout for test completion
    enable_error_injection: bool = False  # For testing: inject synthetic errors
    error_injection_rate: float = 0.0  # Rate for error injection (0.0 to 1.0)


@dataclass
class LaneResult:
    """Per-lane loopback test result"""
    lane_id: int
    channel_id: int
    total_bits: int = 0
    error_bits: int = 0
    ber: float = 0.0
    passed: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Alias for error_bits for compatibility"""
        return self.error_bits


@dataclass
class ChannelResult:
    """Per-channel loopback test result"""
    channel_id: int
    num_lanes: int = 64  # HBM4: 64 lanes per channel
    lane_results: List[LaneResult] = field(default_factory=list)
    total_bits: int = 0
    total_errors: int = 0
    ber: float = 0.0
    passed: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class LoopbackStatus:
    """Loopback test status tracking"""
    state: LoopbackState = LoopbackState.IDLE
    state_enter_cycle: int = 0
    current_channel: int = 0
    current_lane: int = 0
    bits_transmitted: int = 0
    bits_received: int = 0
    total_errors: int = 0
    retry_count: int = 0
    max_retries: int = 3


class PRBSGenerator:
    """PRBS (Pseudo-Random Binary Sequence) Generator
    
    Generates PRBS patterns for loopback testing.
    
    PRBS-7:  Polynomial: x^7 + x^6 + 1, length = 127
    PRBS-15: Polynomial: x^15 + x^14 + 1, length = 32767
    PRBS-31: Polynomial: x^31 + x^28 + 1, length = 2147483647
    """
    
    # PRBS polynomial coefficients (tap positions)
    PRBS7_TAPS = (6, 5)     # x^7 + x^6 + 1
    PRBS15_TAPS = (14, 13)  # x^15 + x^14 + 1
    PRBS31_TAPS = (30, 27)  # x^31 + x^28 + 1
    
    def __init__(self, mode: LoopbackMode = LoopbackMode.PRBS_7, seed: Optional[int] = None):
        """Initialize PRBS generator
        
        Args:
            mode: PRBS mode (PRBS_7, PRBS_15, PRBS_31)
            seed: Optional seed value (None for default)
        """
        self.mode = mode
        self.lfsr = self._init_lfsr(seed)
        self._length = self._get_length()
        self._counter = 0
    
    def _init_lfsr(self, seed: Optional[int]) -> int:
        """Initialize LFSR with appropriate seed"""
        if self.mode == LoopbackMode.PRBS_7:
            max_val = 0x7F
        elif self.mode == LoopbackMode.PRBS_15:
            max_val = 0x7FFF
        elif self.mode == LoopbackMode.PRBS_31:
            max_val = 0x7FFFFFFF
        else:
            max_val = 0xFF
        
        if seed is None:
            return max_val
        return seed & max_val
    
    def _get_length(self) -> int:
        """Get pattern length for current mode"""
        if self.mode == LoopbackMode.PRBS_7:
            return 127  # 2^7 - 1
        elif self.mode == LoopbackMode.PRBS_15:
            return 32767  # 2^15 - 1
        elif self.mode == LoopbackMode.PRBS_31:
            return 2147483647  # 2^31 - 1
        return 256
    
    @property
    def length(self) -> int:
        """Get pattern length"""
        return self._length
    
    def reset(self, seed: Optional[int] = None):
        """Reset generator with optional new seed"""
        self.lfsr = self._init_lfsr(seed)
        self._counter = 0
    
    def next(self) -> int:
        """Generate next bit
        
        Returns:
            Next bit (0 or 1)
        """
        self._counter += 1
        
        if self.mode == LoopbackMode.PRBS_7:
            return self._next_prbs7()
        elif self.mode == LoopbackMode.PRBS_15:
            return self._next_prbs15()
        elif self.mode == LoopbackMode.PRBS_31:
            return self._next_prbs31()
        else:
            return 0
    
    def _next_prbs7(self) -> int:
        """Generate next PRBS-7 bit"""
        tap1, tap2 = self.PRBS7_TAPS
        bit = ((self.lfsr >> tap1) ^ (self.lfsr >> tap2)) & 1
        self.lfsr = ((self.lfsr << 1) | bit) & 0x7F
        return (self.lfsr >> 6) & 1
    
    def _next_prbs15(self) -> int:
        """Generate next PRBS-15 bit"""
        tap1, tap2 = self.PRBS15_TAPS
        bit = ((self.lfsr >> tap1) ^ (self.lfsr >> tap2)) & 1
        self.lfsr = ((self.lfsr << 1) | bit) & 0x7FFF
        return (self.lfsr >> 14) & 1
    
    def _next_prbs31(self) -> int:
        """Generate next PRBS-31 bit"""
        tap1, tap2 = self.PRBS31_TAPS
        bit = ((self.lfsr >> tap1) ^ (self.lfsr >> tap2)) & 1
        self.lfsr = ((self.lfsr << 1) | bit) & 0x7FFFFFFF
        return (self.lfsr >> 30) & 1
    
    def generate_byte(self) -> int:
        """Generate a full byte
        
        Returns:
            8-bit value
        """
        result = 0
        for i in range(8):
            result |= (self.next() << i)
        return result
    
    def generate_n_bytes(self, n: int) -> List[int]:
        """Generate N bytes
        
        Args:
            n: Number of bytes to generate
            
        Returns:
            List of byte values
        """
        return [self.generate_byte() for _ in range(n)]


class FixedPatternGenerator:
    """Fixed pattern generator for loopback testing
    
    Generates fixed patterns: all zeros, all ones, alternating, 8N.
    """
    
    def __init__(self, mode: LoopbackMode):
        """Initialize fixed pattern generator
        
        Args:
            mode: Pattern mode (FIXED_ALL_ZEROS, FIXED_ALL_ONES, etc.)
        """
        self.mode = mode
        self._counter = 0
    
    def reset(self):
        """Reset generator"""
        self._counter = 0
    
    def next(self) -> int:
        """Generate next bit
        
        Returns:
            Next bit (0 or 1)
        """
        self._counter += 1
        
        if self.mode == LoopbackMode.FIXED_ALL_ZEROS:
            return 0
        elif self.mode == LoopbackMode.FIXED_ALL_ONES:
            return 1
        elif self.mode == LoopbackMode.FIXED_ALTERNATING:
            return self._counter & 1
        elif self.mode == LoopbackMode.MODE_8N:
            # 8N mode: cycling through 0x00-0xFF
            return (self._counter // 8) & 1
        return 0
    
    def generate_byte(self) -> int:
        """Generate a full byte
        
        Returns:
            8-bit value
        """
        if self.mode == LoopbackMode.FIXED_ALL_ZEROS:
            return 0x00
        elif self.mode == LoopbackMode.FIXED_ALL_ONES:
            return 0xFF
        elif self.mode == LoopbackMode.FIXED_ALTERNATING:
            return 0xAA if (self._counter // 8) % 2 == 0 else 0x55
        elif self.mode == LoopbackMode.MODE_8N:
            # 8N mode: Generate byte by cycling through next() 8 times
            # Each call to next() increments _counter
            byte_val = 0
            for i in range(8):
                bit = self.next()
                byte_val |= (bit << i)
            return byte_val
        return 0x00


class LoopbackController:
    """Loopback Controller
    
    Implements loopback testing for HBM4 PHY verification.
    
    State Machine:
    - IDLE: Waiting for loopback test start
    - CONFIGURE: Configuring loopback parameters
    - RUNNING: Executing loopback test
    - VERIFY: Verifying results
    - COMPLETE: Test complete
    
    Reference: PH-006 in HBM4 specification
    """
    
    def __init__(self, 
                 num_channels: int = 32,
                 num_lanes_per_channel: int = 64,
                 phy_training_sm: Optional[Any] = None,
                 config: Optional[LoopbackConfig] = None):
        """Initialize loopback controller
        
        Args:
            num_channels: Number of HBM4 channels (default: 32)
            num_lanes_per_channel: Lanes per channel (default: 64 for HBM4)
            phy_training_sm: Optional PHY training state machine for integration
            config: Optional loopback configuration
        """
        self.num_channels = num_channels
        self.num_lanes = num_lanes_per_channel
        self.phy_training_sm = phy_training_sm
        
        # Configuration
        self.config = config or LoopbackConfig()
        
        # Status tracking
        self.status = LoopbackStatus()
        
        # Per-channel results
        self.channel_results: Dict[int, ChannelResult] = {}
        
        # Data generators per channel
        self._generators: Dict[int, Any] = {}
        
        # Expected data buffer (for verification)
        self._expected_data: Dict[int, List[int]] = {}
        
        # Received data buffer
        self._received_data: Dict[int, List[int]] = {}
        
        # Cycle counter
        self._cycle = 0
        
        # Initialize generators
        self._init_generators()
    
    def _init_generators(self):
        """Initialize data generators for all channels"""
        for ch in range(self.num_channels):
            if self.config.mode in [LoopbackMode.PRBS_7, LoopbackMode.PRBS_15, LoopbackMode.PRBS_31]:
                self._generators[ch] = PRBSGenerator(mode=self.config.mode)
            else:
                self._generators[ch] = FixedPatternGenerator(mode=self.config.mode)
            self._expected_data[ch] = []
            self._received_data[ch] = []
    
    @property
    def cycle(self) -> int:
        """Current simulation cycle"""
        return self._cycle
    
    @property
    def state(self) -> LoopbackState:
        """Current loopback state"""
        return self.status.state
    
    def configure(self, config: LoopbackConfig) -> bool:
        """Configure loopback parameters
        
        Args:
            config: Loopback configuration
            
        Returns:
            True if configuration successful
        """
        if self.status.state not in [LoopbackState.IDLE, LoopbackState.COMPLETE]:
            return False
        
        self.config = config
        self._init_generators()
        
        # Reset results
        self.channel_results.clear()
        
        return True
    
    def start(self) -> bool:
        """Start loopback test
        
        Returns:
            True if test started successfully
        """
        if self.status.state != LoopbackState.IDLE:
            return False
        
        # Transition to CONFIGURE state
        self.status.state = LoopbackState.CONFIGURE
        self.status.state_enter_cycle = self._cycle
        self.status.bits_transmitted = 0
        self.status.bits_received = 0
        self.status.total_errors = 0
        self.status.retry_count = 0
        
        # Initialize results storage
        for ch in range(self.num_channels):
            if self._is_channel_enabled(ch):
                self.channel_results[ch] = ChannelResult(channel_id=ch, num_lanes=self.num_lanes)
                # Create lane results
                for lane in range(self.num_lanes):
                    if self._is_lane_enabled(ch, lane):
                        lane_result = LaneResult(lane_id=lane, channel_id=ch)
                        self.channel_results[ch].lane_results.append(lane_result)
        
        # Signal PHY training state machine
        if self.phy_training_sm:
            self.phy_training_sm.start_loopback_test()
        
        return True
    
    def _is_channel_enabled(self, channel: int) -> bool:
        """Check if channel is enabled in mask"""
        return (self.config.channel_mask >> channel) & 1 == 1
    
    def _is_lane_enabled(self, channel: int, lane: int) -> bool:
        """Check if lane is enabled in mask"""
        return (self.config.lane_mask >> lane) & 1 == 1
    
    def tick(self):
        """Advance loopback controller by one cycle"""
        self._cycle += 1
        
        # Check for timeout
        elapsed = self._cycle - self.status.state_enter_cycle
        if elapsed > self.config.timeout_cycles:
            self._handle_timeout()
    
    def _handle_timeout(self):
        """Handle test timeout"""
        self.status.state = LoopbackState.COMPLETE
        
        # Mark all channels as failed
        for ch_result in self.channel_results.values():
            ch_result.passed = False
            ch_result.errors.append("Timeout")
    
    def process_cycle(self) -> bool:
        """Process one loopback cycle
        
        Main state machine advancement logic.
        
        Returns:
            True if loopback test completed
        """
        current = self.status.state
        
        if current == LoopbackState.IDLE:
            # Waiting for start
            pass
        
        elif current == LoopbackState.CONFIGURE:
            # Configure complete, move to running
            self.status.state = LoopbackState.RUNNING
            self.status.state_enter_cycle = self._cycle
            self.status.current_channel = 0
            self.status.current_lane = 0
        
        elif current == LoopbackState.RUNNING:
            # Execute loopback test
            self._execute_loopback()
            
            # Check if test complete
            if self.status.bits_transmitted >= self.config.test_length:
                self.status.state = LoopbackState.VERIFY
                self.status.state_enter_cycle = self._cycle
        
        elif current == LoopbackState.VERIFY:
            # Verify results
            self._verify_results()
            self.status.state = LoopbackState.COMPLETE
        
        elif current == LoopbackState.COMPLETE:
            # Test complete
            return True
        
        return False
    
    def _execute_loopback(self):
        """Execute loopback test for current position"""
        ch = self.status.current_channel
        lane = self.status.current_lane
        
        # Skip disabled channels/lanes
        while not self._is_channel_enabled(ch) or not self._is_lane_enabled(ch, lane):
            self._advance_position()
            ch = self.status.current_channel
            lane = self.status.current_lane
            if self.status.bits_transmitted >= self.config.test_length:
                return
        
        # Generate expected data
        gen = self._generators.get(ch)
        if gen:
            expected_bit = gen.next()
            self._expected_data[ch].append(expected_bit)
            
            # Simulate received data (with possible errors)
            received_bit = expected_bit
            
            # Inject errors if enabled
            if self.config.enable_error_injection:
                if random.random() < self.config.error_injection_rate:
                    received_bit ^= 1  # Flip bit
                    self.status.total_errors += 1
                    self._record_error(ch, lane)
            
            self._received_data[ch].append(received_bit)
            
            # Update counters
            self.status.bits_transmitted += 1
            self.status.bits_received += 1
        
        # Advance to next position
        self._advance_position()
    
    def _advance_position(self):
        """Advance to next channel/lane position"""
        self.status.current_lane += 1
        if self.status.current_lane >= self.num_lanes:
            self.status.current_lane = 0
            self.status.current_channel += 1
            if self.status.current_channel >= self.num_channels:
                self.status.current_channel = 0
    
    def _record_error(self, channel: int, lane: int):
        """Record error for a specific lane"""
        if channel in self.channel_results:
            ch_result = self.channel_results[channel]
            ch_result.total_errors += 1
            
            # Find lane result
            for lane_result in ch_result.lane_results:
                if lane_result.lane_id == lane:
                    lane_result.error_bits += 1
                    break
    
    def _verify_results(self):
        """Verify loopback test results"""
        total_bits = 0
        total_errors = 0
        
        for ch, ch_result in self.channel_results.items():
            ch_result.total_bits = len(self._expected_data.get(ch, []))
            ch_result.total_errors = 0
            ch_result.passed = True
            
            expected = self._expected_data.get(ch, [])
            received = self._received_data.get(ch, [])
            
            for i, (exp, rec) in enumerate(zip(expected, received)):
                if exp != rec:
                    ch_result.total_errors += 1
                    # Update lane error count
                    lane = i // 8  # Rough lane assignment
                    for lane_result in ch_result.lane_results:
                        if lane_result.lane_id == lane:
                            lane_result.error_bits += 1
                            lane_result.total_bits += 1
                            break
            
            total_bits += ch_result.total_bits
            total_errors += ch_result.total_errors
            
            # Calculate BER for channel
            if ch_result.total_bits > 0:
                ch_result.ber = ch_result.total_errors / ch_result.total_bits
                ch_result.passed = ch_result.ber < 1e-6  # BER threshold: 1e-6
                
                # Update individual lane BER
                for lane_result in ch_result.lane_results:
                    if lane_result.total_bits > 0:
                        lane_result.ber = lane_result.error_bits / lane_result.total_bits
                        lane_result.passed = lane_result.ber < 1e-6
        
        # Overall status
        if total_bits > 0:
            overall_ber = total_errors / total_bits
            self.status.total_errors = total_errors
    
    def get_channel_result(self, channel: int) -> Optional[ChannelResult]:
        """Get result for a specific channel
        
        Args:
            channel: Channel index
            
        Returns:
            Channel result or None
        """
        return self.channel_results.get(channel)
    
    def get_lane_result(self, channel: int, lane: int) -> Optional[LaneResult]:
        """Get result for a specific lane
        
        Args:
            channel: Channel index
            lane: Lane index
            
        Returns:
            Lane result or None
        """
        ch_result = self.channel_results.get(channel)
        if ch_result:
            for lane_result in ch_result.lane_results:
                if lane_result.lane_id == lane:
                    return lane_result
        return None
    
    def get_overall_ber(self) -> float:
        """Calculate overall BER across all channels
        
        Returns:
            Overall BER
        """
        total_bits = sum(r.total_bits for r in self.channel_results.values())
        total_errors = sum(r.total_errors for r in self.channel_results.values())
        
        if total_bits > 0:
            return total_errors / total_bits
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get loopback test summary
        
        Returns:
            Dictionary with test summary
        """
        passed_channels = sum(1 for r in self.channel_results.values() if r.passed)
        failed_channels = len(self.channel_results) - passed_channels
        
        return {
            'state': self.status.state.name,
            'cycle': self._cycle,
            'mode': self.config.mode.name,
            'level': self.config.level.name,
            'total_channels': self.num_channels,
            'tested_channels': len(self.channel_results),
            'passed_channels': passed_channels,
            'failed_channels': failed_channels,
            'total_bits': self.status.bits_transmitted,
            'total_errors': self.status.total_errors,
            'overall_ber': self.get_overall_ber(),
            'complete': self.status.state == LoopbackState.COMPLETE,
        }
    
    def is_complete(self) -> bool:
        """Check if loopback test is complete
        
        Returns:
            True if test is complete
        """
        return self.status.state == LoopbackState.COMPLETE
    
    def is_passed(self) -> bool:
        """Check if loopback test passed
        
        Returns:
            True if all channels passed
        """
        if not self.is_complete():
            return False
        
        return all(r.passed for r in self.channel_results.values())


class HBM4LoopbackManager:
    """HBM4 Loopback Manager
    
    Top-level manager for coordinating loopback tests
    across all HBM4 channels.
    """
    
    def __init__(self, 
                 num_channels: int = 32,
                 num_lanes_per_channel: int = 64,
                 phy_training_sm: Optional[Any] = None,
                 config: Optional[LoopbackConfig] = None):
        """Initialize HBM4 loopback manager
        
        Args:
            num_channels: Number of HBM4 channels
            num_lanes_per_channel: Lanes per channel
            phy_training_sm: Optional PHY training state machine
            config: Optional loopback configuration
        """
        self.num_channels = num_channels
        self.config = config or LoopbackConfig()
        
        # Create loopback controllers per channel if level is LANE
        if self.config.level == LoopbackLevel.LANE:
            self._controllers: List[LoopbackController] = []
            for ch in range(num_channels):
                ch_config = LoopbackConfig(
                    mode=self.config.mode,
                    level=self.config.level,
                    channel_mask=1 << ch,
                    lane_mask=self.config.lane_mask,
                    test_length=self.config.test_length,
                    timeout_cycles=self.config.timeout_cycles,
                    enable_error_injection=self.config.enable_error_injection,
                    error_injection_rate=self.config.error_injection_rate,
                )
                controller = LoopbackController(
                    num_channels=1,
                    num_lanes_per_channel=num_lanes_per_channel,
                    phy_training_sm=phy_training_sm,
                    config=ch_config,
                )
                self._controllers.append(controller)
        else:
            # Single controller for CHANNEL or STACK level
            self._controllers = [
                LoopbackController(
                    num_channels=num_channels,
                    num_lanes_per_channel=num_lanes_per_channel,
                    phy_training_sm=phy_training_sm,
                    config=self.config,
                )
            ]
        
        self._global_cycle = 0
    
    @property
    def cycle(self) -> int:
        """Current global cycle"""
        return self._global_cycle
    
    def tick(self):
        """Advance all loopback controllers"""
        self._global_cycle += 1
        for ctrl in self._controllers:
            ctrl.tick()
    
    def start_all(self):
        """Start loopback tests on all controllers"""
        for ctrl in self._controllers:
            ctrl.start()
    
    def process_cycles(self, num_cycles: int):
        """Process multiple cycles
        
        Args:
            num_cycles: Number of cycles to process
        """
        for _ in range(num_cycles):
            for ctrl in self._controllers:
                ctrl.process_cycle()
            self.tick()
    
    def wait_for_completion(self, max_cycles: int = 100000) -> bool:
        """Wait for all loopback tests to complete
        
        Args:
            max_cycles: Maximum cycles to wait
            
        Returns:
            True if all tests completed successfully
        """
        for _ in range(max_cycles):
            if all(ctrl.is_complete() for ctrl in self._controllers):
                return True
            self.process_cycles(1)
        return False
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get results from all controllers
        
        Returns:
            List of result dictionaries
        """
        results = []
        for ctrl in self._controllers:
            for ch, ch_result in ctrl.channel_results.items():
                results.append({
                    'channel_id': ch,
                    'total_bits': ch_result.total_bits,
                    'total_errors': ch_result.total_errors,
                    'ber': ch_result.ber,
                    'passed': ch_result.passed,
                })
        return results
    
    def is_all_passed(self) -> bool:
        """Check if all loopback tests passed
        
        Returns:
            True if all controllers passed
        """
        return all(ctrl.is_passed() for ctrl in self._controllers)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate summary
        
        Returns:
            Summary dictionary
        """
        total_bits = sum(ctrl.status.bits_transmitted for ctrl in self._controllers)
        total_errors = sum(ctrl.status.total_errors for ctrl in self._controllers)
        passed = sum(1 for ctrl in self._controllers if ctrl.is_passed())
        
        return {
            'num_controllers': len(self._controllers),
            'num_passed': passed,
            'num_failed': len(self._controllers) - passed,
            'total_bits': total_bits,
            'total_errors': total_errors,
            'overall_ber': total_errors / total_bits if total_bits > 0 else 0.0,
            'all_passed': self.is_all_passed(),
        }