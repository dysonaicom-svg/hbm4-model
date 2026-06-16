"""
AXI4 to HBM Request Converter

This module provides conversion between AXI4 transactions and HBM requests:
- AXI4 to HBM request conversion
- HBM response to AXI4 conversion
- Address mapping for AXI address space to HBM channel/bank
- Burst handling and beat expansion
- Transaction ordering and tracking

Based on:
- ARM AMBA AXI4 Protocol Specification
- JEDEC JESD270-4A HBM4 specification

Usage:
    >>> from model.interconnect.axi4_converter import AXI4Converter, AddressMapping
    >>> converter = AXI4Converter()
    >>> 
    >>> # Convert AXI4 read to HBM request
    >>> hbm_req = converter.axi4_to_hbm_read(axi_txn)
    >>> 
    >>> # Convert HBM response to AXI4
    >>> axi_resp = converter.hbm_to_axi4_response(hbm_resp, txn_id)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import IntEnum, auto
from collections import defaultdict
import logging

from model.interconnect.axi4_bridge import (
    AXI4Bridge, AXI4BridgeConfig, AXI4ReadTransaction, AXI4WriteTransaction,
    AXI4TransactionResponse, AXI4BurstType, AXI4Response,
    AXI4Size, create_axi4_bridge
)
from model.controller.request import HBMRequest, HBMResponse, RequestState

logger = logging.getLogger('hbm4.axi4_converter')
logger.setLevel(logging.WARNING)


# ============================================================================
# Address Mapping
# ============================================================================

class AddressMappingMode(IntEnum):
    """Address mapping modes"""
    LINEAR = 0        # Linear mapping
    CHANNEL_INTERLEAVED = 1  # Channel interleaving
    BANK_INTERLEAVED = 2    # Bank interleaving
    ROW_BANK_CHANNEL = 3     # Row-Bank-Channel (JEDEC RBC)


@dataclass
class AddressMapping:
    """
    AXI to HBM Address Mapping
    
    Maps AXI address space to HBM address space:
    - AXI address: flat 64-bit address space
    - HBM address: Stack/Channel/PseudoChannel/Bank/Row/Column
    
    Default HBM4 address map (JEDEC RBC - Row Bank Column):
    - Bits [63:48]: Reserved
    - Bits [47:46]: Stack ID (4 stacks max)
    - Bits [45:41]: Channel ID (32 channels)
    - Bit [40]: Pseudo-channel (0 or 1)
    - Bits [39:27]: Row (13 bits, 8K rows)
    - Bits [26:24]: Bank Group (8 groups)
    - Bits [23:20]: Bank (16 banks)
    - Bits [19:6]: Column (14 bits, 16K columns)
    - Bits [5:3]: Burst position
    - Bits [2:0]: Byte offset
    """
    
    # AXI address space configuration
    axi_addr_width: int = 64
    axi_data_width: int = 512  # 512 bits = 64 bytes per beat
    
    # HBM address space configuration
    hbm_channels: int = 32
    hbm_pseudo_channels: int = 2
    hbm_bank_groups: int = 8
    hbm_banks: int = 16
    hbm_rows: int = 65536  # 64K rows
    hbm_cols: int = 2048   # 2K columns
    hbm_stacks: int = 4
    
    # Mapping configuration
    mapping_mode: AddressMappingMode = AddressMappingMode.ROW_BANK_CHANNEL
    
    # Interleaving parameters
    channel_interleave_size: int = 256    # Bytes between channel switches
    bank_interleave_depth: int = 4        # Number of banks in interleaving
    channel_hash_enable: bool = True      # Enable channel hashing
    
    # Address region configuration
    regions: List[Tuple[int, int, int]] = field(default_factory=list)  # (base, size, stack_id)
    
    def __post_init__(self):
        """Compute derived parameters"""
        # Compute HBM address space
        self.hbm_total_banks = self.hbm_bank_groups * self.hbm_banks
        self.hbm_banks_per_channel = self.hbm_total_banks // self.hbm_pseudo_channels
        
        # Compute total address space
        self.total_hbm_bytes = (
            self.hbm_stacks * 
            self.hbm_channels * 
            self.hbm_pseudo_channels *
            self.hbm_total_banks * 
            self.hbm_rows * 
            self.hbm_cols * 
            32  # 256-bit column width
        )
        
        # Byte address bits
        self.byte_offset_bits = 3  # 8 bytes per burst for 256-bit interface
        self.burst_offset_bits = 3
        self.col_offset_bits = 3   # 8 prefetch beats
        self.row_bits = 13          # 8K rows
        self.bank_group_bits = 3   # 8 bank groups
        self.bank_bits = 4         # 16 banks
        self.pch_bits = 1           # 2 pseudo-channels
        self.channel_bits = 5       # 32 channels
        self.stack_bits = 2         # 4 stacks
        
        # Computed boundaries
        self.col_lsb = self.byte_offset_bits
        self.bank_lsb = self.col_lsb + self.col_offset_bits
        self.bg_lsb = self.bank_lsb + self.bank_bits
        self.row_lsb = self.bg_lsb + self.bank_group_bits
        self.pch_lsb = self.row_lsb + self.row_bits
        self.chan_lsb = self.pch_lsb + self.pch_bits
        self.stack_lsb = self.chan_lsb + self.channel_bits
    
    def decode_axi_addr(self, axi_addr: int) -> Dict:
        """
        Decode AXI address to components
        
        Returns:
            Dictionary with decoded address fields
        """
        # Extract components based on mapping mode
        byte_offset = axi_addr & ((1 << self.byte_offset_bits) - 1)
        col = (axi_addr >> self.col_lsb) & ((1 << self.col_offset_bits) - 1)
        bank = (axi_addr >> self.bank_lsb) & ((1 << self.bank_bits) - 1)
        bank_group = (axi_addr >> self.bg_lsb) & ((1 << self.bank_group_bits) - 1)
        row = (axi_addr >> self.row_lsb) & ((1 << self.row_bits) - 1)
        pch = (axi_addr >> self.pch_lsb) & ((1 << self.pch_bits) - 1)
        channel = (axi_addr >> self.chan_lsb) & ((1 << self.channel_bits) - 1)
        stack = (axi_addr >> self.stack_lsb) & ((1 << self.stack_bits) - 1)
        
        return {
            'axi_addr': axi_addr,
            'byte_offset': byte_offset,
            'col': col,
            'bank': bank,
            'bank_group': bank_group,
            'row': row,
            'pseudo_channel': pch,
            'channel': channel,
            'stack': stack,
        }
    
    def encode_hbm_addr(
        self,
        stack: int,
        channel: int,
        pseudo_channel: int,
        bank_group: int,
        bank: int,
        row: int,
        col: int,
        byte_offset: int = 0,
    ) -> int:
        """
        Encode HBM address from components
        
        Args:
            stack: Stack ID (0-3)
            channel: Channel ID (0-31)
            pseudo_channel: Pseudo-channel (0-1)
            bank_group: Bank group (0-7)
            bank: Bank (0-15)
            row: Row (0-65535)
            col: Column (0-7)
            byte_offset: Byte offset within column
            
        Returns:
            64-bit HBM address
        """
        addr = byte_offset
        addr |= col << self.col_lsb
        addr |= bank << self.bank_lsb
        addr |= bank_group << self.bg_lsb
        addr |= row << self.row_lsb
        addr |= pseudo_channel << self.pch_lsb
        addr |= channel << self.chan_lsb
        addr |= stack << self.stack_lsb
        
        return addr
    
    def get_stack_channel(self, axi_addr: int) -> Tuple[int, int]:
        """
        Get stack and channel for an AXI address
        
        Returns:
            Tuple of (stack_id, channel_id)
        """
        decoded = self.decode_axi_addr(axi_addr)
        return (decoded['stack'], decoded['channel'])
    
    def get_hbm_address_components(self, axi_addr: int) -> Dict:
        """
        Get full HBM address components
        
        Returns:
            Dictionary with stack/channel/pch/bank/row/col
        """
        return self.decode_axi_addr(axi_addr)
    
    def axi_to_hbm_addr(self, axi_addr: int) -> int:
        """
        Convert AXI address to HBM address
        
        For flat address spaces, this is a pass-through.
        For interleaved spaces, this maps to the appropriate HBM location.
        """
        if self.mapping_mode == AddressMappingMode.LINEAR:
            # Linear: pass through
            return axi_addr
        else:
            # RBC mapping
            return axi_addr
    
    def hbm_to_axi_addr(self, hbm_addr: int) -> int:
        """Convert HBM address to AXI address"""
        return hbm_addr


@dataclass
class ConversionResult:
    """Result of AXI4 to HBM conversion"""
    success: bool
    hbm_requests: List[HBMRequest] = field(default_factory=list)
    error_message: Optional[str] = None
    beats_generated: int = 0
    bytes_converted: int = 0


# ============================================================================
# AXI4 to HBM Converter
# ============================================================================

class AXI4ToHBMConverter:
    """
    Converter from AXI4 transactions to HBM requests
    
    Handles:
    - Burst expansion (AXI4 bursts -> multiple HBM requests)
    - Address translation
    - Transaction ordering
    - Request batching for efficiency
    
    Usage:
        >>> from model.interconnect.axi4_converter import AXI4ToHBMConverter
        >>> converter = AXI4ToHBMConverter()
        >>> 
        >>> # Convert an AXI4 read
        >>> result = converter.convert_read(axi_txn)
        >>> for req in result.hbm_requests:
        >>>     controller.submit_request(req)
    """
    
    def __init__(
        self,
        address_mapping: Optional[AddressMapping] = None,
        max_requests_per_burst: int = 32,
        enable_burst_splitting: bool = True,
        hbm_burst_size: int = 64,  # HBM burst size in bytes
    ):
        """
        Initialize converter
        
        Args:
            address_mapping: Address mapping configuration
            max_requests_per_burst: Maximum HBM requests per burst
            enable_burst_splitting: Enable splitting large bursts
            hbm_burst_size: HBM burst size in bytes
        """
        self.address_mapping = address_mapping or AddressMapping()
        self.max_requests_per_burst = max_requests_per_burst
        self.enable_burst_splitting = enable_burst_splitting
        self.hbm_burst_size = hbm_burst_size
        
        # Transaction tracking
        self._pending_conversions: Dict[int, AXI4ReadTransaction] = {}
        self._pending_write_conversions: Dict[int, AXI4WriteTransaction] = {}
        
        # Statistics
        self.stats = {
            'reads_converted': 0,
            'writes_converted': 0,
            'requests_generated': 0,
            'beats_generated': 0,
            'bytes_converted': 0,
            'bursts_split': 0,
            'errors': 0,
        }
    
    def convert_read(self, txn: AXI4ReadTransaction) -> ConversionResult:
        """
        Convert AXI4 read transaction to HBM requests
        
        Args:
            txn: AXI4 read transaction
            
        Returns:
            ConversionResult with HBM requests
        """
        hbm_requests = []
        
        try:
            # Validate transaction
            if txn is None:
                return ConversionResult(False, error_message="Null transaction")
            
            # Get beat addresses
            beat_addrs = txn.get_beat_addresses()
            num_beats = len(beat_addrs)
            
            # Calculate bytes per beat
            bytes_per_beat = 1 << txn.size
            
            # For HBM, we typically submit one request per burst
            # The HBM controller handles the burst internally
            for beat_idx, addr in enumerate(beat_addrs):
                # Decode address
                decoded = self.address_mapping.decode_axi_addr(addr)
                
                # Create HBM request
                hbm_req = HBMRequest(
                    addr=addr,
                    length=bytes_per_beat,
                    is_read=True,
                    qos=txn.qos,
                    stack_id=decoded['stack'],
                    channel_id=decoded['channel'],
                    pseudo_channel_id=decoded['pseudo_channel'],
                    bank_group_id=decoded['bank_group'],
                    bank_id=decoded['bank'],
                    row_id=decoded['row'],
                    col_id=decoded['col'],
                )
                hbm_req.set_arrival_time(txn.submission_cycle)
                
                hbm_requests.append(hbm_req)
                
                # Limit requests per burst
                if len(hbm_requests) >= self.max_requests_per_burst:
                    if self.enable_burst_splitting:
                        self.stats['bursts_split'] += 1
                    break
            
            self.stats['reads_converted'] += 1
            self.stats['requests_generated'] += len(hbm_requests)
            self.stats['beats_generated'] += num_beats
            self.stats['bytes_converted'] += txn.total_bytes
            
            return ConversionResult(
                success=True,
                hbm_requests=hbm_requests,
                beats_generated=num_beats,
                bytes_converted=txn.total_bytes,
            )
            
        except Exception as e:
            logger.error(f"Read conversion error: {e}")
            self.stats['errors'] += 1
            return ConversionResult(False, error_message=str(e))
    
    def convert_write(self, txn: AXI4WriteTransaction) -> ConversionResult:
        """
        Convert AXI4 write transaction to HBM requests
        
        Args:
            txn: AXI4 write transaction
            
        Returns:
            ConversionResult with HBM requests
        """
        hbm_requests = []
        
        try:
            if txn is None:
                return ConversionResult(False, error_message="Null transaction")
            
            beat_addrs = txn.get_beat_addresses()
            num_beats = len(beat_addrs)
            bytes_per_beat = 1 << txn.size
            
            for beat_idx, addr in enumerate(beat_addrs):
                decoded = self.address_mapping.decode_axi_addr(addr)
                
                # Get data for this beat
                data = None
                if beat_idx < len(txn.data):
                    data_bytes = txn.data[beat_idx].to_bytes(
                        bytes_per_beat, 'little'
                    )
                    data = bytes(data_bytes)
                
                hbm_req = HBMRequest(
                    addr=addr,
                    length=bytes_per_beat,
                    is_read=False,
                    qos=txn.qos,
                    stack_id=decoded['stack'],
                    channel_id=decoded['channel'],
                    pseudo_channel_id=decoded['pseudo_channel'],
                    bank_group_id=decoded['bank_group'],
                    bank_id=decoded['bank'],
                    row_id=decoded['row'],
                    col_id=decoded['col'],
                    data=data,
                )
                hbm_req.set_arrival_time(txn.submission_cycle)
                
                hbm_requests.append(hbm_req)
                
                if len(hbm_requests) >= self.max_requests_per_burst:
                    if self.enable_burst_splitting:
                        self.stats['bursts_split'] += 1
                    break
            
            self.stats['writes_converted'] += 1
            self.stats['requests_generated'] += len(hbm_requests)
            self.stats['beats_generated'] += num_beats
            self.stats['bytes_converted'] += txn.total_bytes
            
            return ConversionResult(
                success=True,
                hbm_requests=hbm_requests,
                beats_generated=num_beats,
                bytes_converted=txn.total_bytes,
            )
            
        except Exception as e:
            logger.error(f"Write conversion error: {e}")
            self.stats['errors'] += 1
            return ConversionResult(False, error_message=str(e))
    
    def convert_burst_addresses(
        self,
        start_addr: int,
        size: int,
        length: int,
        burst_type: AXI4BurstType,
    ) -> List[int]:
        """
        Convert burst addresses (for address generation)
        
        Args:
            start_addr: Starting address
            size: Bytes per beat
            length: Number of beats - 1
            burst_type: Burst type
            
        Returns:
            List of beat addresses
        """
        addrs = []
        num_beats = length + 1
        
        for i in range(num_beats):
            if burst_type == AXI4BurstType.FIXED:
                addr = start_addr
            elif burst_type == AXI4BurstType.INCR:
                addr = start_addr + (i << size)
            elif burst_type == AXI4BurstType.WRAP:
                wrap_boundary = start_addr & ~((num_beats << size) - 1)
                addr = wrap_boundary + ((start_addr + (i << size) - wrap_boundary) % (num_beats << size))
            else:
                addr = start_addr
            addrs.append(addr)
        
        return addrs
    
    def get_stats(self) -> Dict:
        """Get conversion statistics"""
        return self.stats.copy()
    
    def reset(self) -> None:
        """Reset converter state"""
        self._pending_conversions.clear()
        self._pending_write_conversions.clear()
        for key in self.stats:
            self.stats[key] = 0


# ============================================================================
# HBM to AXI4 Converter
# ============================================================================

class HBMToAXI4Converter:
    """
    Converter from HBM responses to AXI4 responses
    
    Handles:
    - Response assembly (HBM responses -> AXI4 R/B channel data)
    - Error propagation
    - Completion ordering
    - Data formatting
    """
    
    def __init__(
        self,
        address_mapping: Optional[AddressMapping] = None,
        data_width: int = 512,
    ):
        """
        Initialize HBM to AXI4 converter
        
        Args:
            address_mapping: Address mapping configuration
            data_width: AXI data width in bits
        """
        self.address_mapping = address_mapping or AddressMapping()
        self.data_width = data_width
        self._data_bytes = data_width // 8
        
        # Response tracking
        self._pending_read_responses: Dict[int, List[HBMResponse]] = {}
        self._pending_write_responses: Dict[int, HBMResponse] = {}
        
        # Statistics
        self.stats = {
            'read_responses': 0,
            'write_responses': 0,
            'r_beats_generated': 0,
            'errors': 0,
        }
    
    def convert_read_response(
        self,
        hbm_responses: List[HBMResponse],
        txn_id: int,
        beat_count: int,
    ) -> List[Tuple[int, int, int, bool]]:
        """
        Convert HBM read responses to AXI4 R channel beats
        
        Args:
            hbm_responses: List of HBM responses
            txn_id: Transaction ID
            beat_count: Number of expected beats
            
        Returns:
            List of (rdata, rresp, rid, rlast) tuples
        """
        beats = []
        
        try:
            for i, resp in enumerate(hbm_responses):
                is_last = (i == len(hbm_responses) - 1)
                
                # Pack data into AXI width
                rdata = 0
                if resp.data:
                    data_int = int.from_bytes(resp.data[:self._data_bytes], 'little')
                    rdata = data_int
                
                # Convert HBM status to AXI response
                if resp.status == "OK":
                    rresp = 0b00  # OKAY
                elif resp.status == "EXOKAY":
                    rresp = 0b01  # EXOKAY
                elif resp.status == "SLVERR":
                    rresp = 0b10  # SLVERR
                else:
                    rresp = 0b11  # DECERR
                
                beats.append((rdata, rresp, txn_id, is_last))
                self.stats['r_beats_generated'] += 1
            
            self.stats['read_responses'] += 1
            
        except Exception as e:
            logger.error(f"Read response conversion error: {e}")
            self.stats['errors'] += 1
        
        return beats
    
    def convert_write_response(
        self,
        hbm_resp: HBMResponse,
        txn_id: int,
    ) -> Tuple[int, int]:
        """
        Convert HBM write response to AXI4 B channel response
        
        Args:
            hbm_resp: HBM response
            txn_id: Transaction ID
            
        Returns:
            Tuple of (bresp, bid)
        """
        try:
            if hbm_resp.status == "OK":
                bresp = 0b00  # OKAY
            elif hbm_resp.status == "EXOKAY":
                bresp = 0b01  # EXOKAY
            elif hbm_resp.status == "SLVERR":
                bresp = 0b10  # SLVERR
            else:
                bresp = 0b11  # DECERR
            
            self.stats['write_responses'] += 1
            
            return (bresp, txn_id)
            
        except Exception as e:
            logger.error(f"Write response conversion error: {e}")
            self.stats['errors'] += 1
            return (0b11, txn_id)  # DECERR on error
    
    def format_read_data(self, data: bytes, size: int) -> int:
        """
        Format read data for AXI4 R channel
        
        Args:
            data: Data bytes
            size: Transfer size
            
        Returns:
            Integer data value
        """
        if len(data) < size:
            data = data + b'\x00' * (size - len(data))
        
        # Truncate to size
        data = data[:size]
        
        return int.from_bytes(data, 'little')
    
    def get_stats(self) -> Dict:
        """Get conversion statistics"""
        return self.stats.copy()
    
    def reset(self) -> None:
        """Reset converter state"""
        self._pending_read_responses.clear()
        self._pending_write_responses.clear()
        for key in self.stats:
            self.stats[key] = 0


# ============================================================================
# Unified Converter
# ============================================================================

class AXI4Converter:
    """
    Unified AXI4 to HBM Converter
    
    Combines AXI4 to HBM and HBM to AXI4 conversion in both directions.
    Provides a complete interface for connecting AXI4 masters to HBM memory.
    
    Usage:
        >>> from model.interconnect.axi4_converter import AXI4Converter
        >>> converter = AXI4Converter()
        >>> 
        >>> # From AXI4 master to HBM controller
        >>> hbm_reqs = converter.to_hbm(axi_txn)
        >>> 
        >>> # From HBM controller to AXI4 slave
        >>> axi_beats = converter.to_axi4(hbm_resp, txn_id)
    """
    
    def __init__(
        self,
        address_mapping: Optional[AddressMapping] = None,
        data_width: int = 512,
        enable_burst_splitting: bool = True,
    ):
        """
        Initialize unified converter
        
        Args:
            address_mapping: Address mapping configuration
            data_width: AXI data width
            enable_burst_splitting: Enable splitting large bursts
        """
        self.address_mapping = address_mapping or AddressMapping()
        self.data_width = data_width
        
        # Initialize sub-converters
        self.axi4_to_hbm = AXI4ToHBMConverter(
            address_mapping=address_mapping,
            enable_burst_splitting=enable_burst_splitting,
            hbm_burst_size=data_width // 8,
        )
        
        self.hbm_to_axi4 = HBMToAXI4Converter(
            address_mapping=address_mapping,
            data_width=data_width,
        )
    
    def to_hbm(self, txn) -> ConversionResult:
        """
        Convert AXI4 transaction to HBM requests
        
        Args:
            txn: AXI4ReadTransaction or AXI4WriteTransaction
            
        Returns:
            ConversionResult with HBM requests
        """
        if isinstance(txn, AXI4ReadTransaction):
            return self.axi4_to_hbm.convert_read(txn)
        elif isinstance(txn, AXI4WriteTransaction):
            return self.axi4_to_hbm.convert_write(txn)
        else:
            return ConversionResult(False, error_message=f"Unknown transaction type: {type(txn)}")
    
    def to_axi4(self, hbm_resp, txn_id: int) -> any:
        """
        Convert HBM response to AXI4 format
        
        Args:
            hbm_resp: HBMResponse
            txn_id: Transaction ID
            
        Returns:
            AXI4 response data
        """
        # Check if response is successful (read responses typically have data)
        if hbm_resp.data is not None:
            return self.hbm_to_axi4.convert_read_response([hbm_resp], txn_id, 1)
        else:
            return self.hbm_to_axi4.convert_write_response(hbm_resp, txn_id)
    
    def get_stats(self) -> Dict:
        """Get combined statistics"""
        return {
            'axi4_to_hbm': self.axi4_to_hbm.get_stats(),
            'hbm_to_axi4': self.hbm_to_axi4.get_stats(),
        }
    
    def reset(self) -> None:
        """Reset all converter state"""
        self.axi4_to_hbm.reset()
        self.hbm_to_axi4.reset()


# ============================================================================
# Factory Functions
# ============================================================================

def create_hbm_address_mapping(
    mode: str = "row_bank_channel",
    channels: int = 32,
    stacks: int = 4,
) -> AddressMapping:
    """
    Create HBM address mapping
    
    Args:
        mode: Mapping mode ("linear", "channel_interleaved", "row_bank_channel")
        channels: Number of channels
        stacks: Number of stacks
        
    Returns:
        Configured AddressMapping
    """
    mode_map = {
        "linear": AddressMappingMode.LINEAR,
        "channel_interleaved": AddressMappingMode.CHANNEL_INTERLEAVED,
        "bank_interleaved": AddressMappingMode.BANK_INTERLEAVED,
        "row_bank_channel": AddressMappingMode.ROW_BANK_CHANNEL,
    }
    
    mapping_mode = mode_map.get(mode, AddressMappingMode.ROW_BANK_CHANNEL)
    
    return AddressMapping(
        mapping_mode=mapping_mode,
        hbm_channels=channels,
        hbm_stacks=stacks,
    )


def create_axi4_converter(
    address_mapping: Optional[AddressMapping] = None,
    data_width: int = 512,
) -> AXI4Converter:
    """
    Create AXI4 converter with common configuration
    
    Args:
        address_mapping: Optional address mapping
        data_width: AXI data width
        
    Returns:
        Configured AXI4Converter
    """
    mapping = address_mapping or AddressMapping()
    return AXI4Converter(
        address_mapping=mapping,
        data_width=data_width,
    )


if __name__ == "__main__":
    print("Testing AXI4 Converter...")
    
    # Create converter
    converter = create_axi4_converter()
    
    # Test address mapping
    mapping = converter.address_mapping
    test_addr = 0x0001_0000_0000_1234
    
    decoded = mapping.decode_axi_addr(test_addr)
    print(f"\nAddress 0x{test_addr:x} decoded:")
    for key, value in decoded.items():
        print(f"  {key}: {value}")
    
    # Test AXI4 to HBM conversion
    print("\nTesting AXI4 read conversion...")
    from model.interconnect.axi4_bridge import AXI4ReadTransaction, AXI4BurstType
    
    txn = AXI4ReadTransaction(
        addr=0x1000,
        size=6,  # 64 bytes
        length=7,  # 8 beats
        burst=AXI4BurstType.INCR,
        id=1,
        qos=8,
    )
    
    result = converter.to_hbm(txn)
    print(f"Conversion result: success={result.success}")
    print(f"HBM requests generated: {len(result.hbm_requests)}")
    print(f"Bytes converted: {result.bytes_converted}")
    
    for i, req in enumerate(result.hbm_requests):
        print(f"  Request {i}: addr=0x{req.addr:x}, ch={req.channel_id}, bank={req.bank_id}")
    
    print(f"\nStats: {converter.get_stats()}")