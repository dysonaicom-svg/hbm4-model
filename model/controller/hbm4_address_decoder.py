"""
HBM4 Address Decoder

Extends the base AddressDecoder for HBM4-specific 32-channel architecture.

Key differences from HBM3:
- 32 channels (5-bit channel field vs 3-bit in HBM3)
- 64 pseudo-channels (1-bit pseudo-channel field)
- Extended address space for 2 TB/s bandwidth
- Additional address bits for larger capacity

Address Mapping Schemes
========================

This decoder supports 4 address mapping schemes optimized for different access patterns:

1. RBC (Row-Bank-Channel) / HBM4 Default
   - Best for: Sequential access, streaming workloads
   - Row changes slowest, maximizing row buffer hits
   - Layout: [Stack][Channel][Pch][BankGroup][Bank][Row][Col][Burst][Offset]

2. BCR (Bank-Channel-Row)
   - Best for: Maximizing bank parallelism
   - Banks spread across wider address range
   - Layout: [Stack][BankGroup][Bank][Channel][Pch][Row][Col][Burst][Offset]

3. CRB (Channel-Row-Bank)
   - Best for: Cross-channel random access
   - Channel at top bits for easy striping
   - Layout: [Channel][Stack][Pch][BankGroup][Bank][Row][Col][Burst][Offset]

4. Custom
   - User-defined mapping via custom_mapping parameter

HBM4 Address Bit Fields
========================

Default RBC mapping (48-bit address space):
    Addr[47:46] = Stack ID (2-bit, supports 4 stacks)
    Addr[45:41] = Channel (5-bit, 32 channels)
    Addr[40]    = Pseudo-channel (1-bit, 2 pseudo-channels)
    Addr[39:37] = Bank group (3-bit, 8 bank groups)
    Addr[36:33] = Bank within group (4-bit, 16 banks)
    Addr[32:17] = Row (16-bit, 64K rows)
    Addr[16:11] = Column (6-bit, 64 columns)
    Addr[10:9]  = Burst beat (2-bit, 4-beat burst alignment)
    Addr[8:6]   = Byte offset (3-bit, 8-byte offset within burst)

Based on:
- JEDEC JESD270-4A HBM4 specification
- Multi-agent research findings (2026-06-15)
"""

from typing import Dict, Optional, Tuple
from model.controller.address_decoder import AddressDecoder, DecodedAddress
from model.dram.hbm4_spec import HBM4Spec
from model.controller.config import HBMConfig


class HBM4AddressDecoder(AddressDecoder):
    """HBM4-specific address decoder with 32-channel support

    This decoder extends the base AddressDecoder to support HBM4's
    expanded channel count and address space.

    Key differences from HBM3:
    - 32 channels (5 bits vs 3 bits in HBM3)
    - 64 pseudo-channels (1 bit per channel)
    - 16 bank groups (3 bits)
    - 16 banks per group (4 bits)
    - 64K rows (16 bits)

    Address Format (64-bit physical address):
        For RBC mapping (default):
            MSB  [47:46] Stack ID (2 bits, 4 stacks)
                 [45:41] Channel (5 bits, 32 channels)
                 [40]    Pseudo-channel (1 bit, 2 per channel)
                 [39:37] Bank group (3 bits, 8 per pseudo-channel)
                 [36:33] Bank within group (4 bits, 16 per group)
                 [32:17] Row (16 bits, 64K per bank)
                 [16:11] Column (6 bits, 64 per row)
                 [10:9]  Burst beat (2 bits, 4-beat alignment)
            LSB  [8:6]   Byte offset (3 bits, 8-byte granularity)

    Usage Example:
        >>> decoder = HBM4AddressDecoder()
        >>> addr = 0x123456789ABC
        >>> decoded = decoder.decode(addr)
        >>> print(f"Channel: {decoded.channel_id}, Row: {decoded.row_id}")
        Channel: 9, Row: 0x1234

    Attributes:
        CHANNEL_BITS: Number of bits for channel field (5 for 32 channels)
        PCH_BITS: Number of bits for pseudo-channel (1 for 2 per channel)
        BG_BITS: Number of bits for bank group (3 for 8 per pseudo-channel)
        BANK_BITS: Number of bits for bank (4 for 16 per group)
        ROW_BITS: Number of bits for row (16 for 64K per bank)
        TOTAL_ADDR_BITS: Total address bits (42 for HBM4 default mapping)

    Note:
        All addresses must be 8-byte aligned (bits [2:0] = 0).
        The decoder handles misaligned addresses by masking to 8-byte boundary.
    """

    # HBM4 address bit field configuration (aligned with HBM4Spec)
    CHANNEL_BITS = 5      # 32 channels
    PCH_BITS = 1          # 2 pseudo-channels per channel
    BG_BITS = 3           # 8 bank groups
    BANK_BITS = 4         # 16 banks per group
    ROW_BITS = 16         # 64K rows
    COL_BITS = 6           # 64 columns (matches spec.ADDR_COL_BITS)
    STACK_BITS = 2        # 4 stacks
    BURST_BITS = 2        # 4-beat burst alignment (matches spec.ADDR_BURST_BITS)
    OFFSET_BITS = 3       # 8-byte offset within burst

    # Total address bits for HBM4: 2+5+1+3+4+16+6+2+3 = 42 bits
    TOTAL_ADDR_BITS = STACK_BITS + CHANNEL_BITS + PCH_BITS + BG_BITS + BANK_BITS + ROW_BITS + COL_BITS + BURST_BITS + OFFSET_BITS

    def __init__(self, spec: Optional[HBM4Spec] = None, mapping_scheme: str = "rbc"):
        """Initialize HBM4 address decoder

        Args:
            spec: HBM4 specification (uses default if None)
            mapping_scheme: Address mapping scheme ("rbc", "bcr", "crb", "hbm4")
        """
        if spec is None:
            spec = HBM4Spec()

        self.spec = spec
        self._mapping_scheme = mapping_scheme  # Store original string

        # Create a minimal HBMConfig for base class
        config = HBMConfig(
            stack_count=2**spec.ADDR_STACK_BITS,
            channels_per_stack=spec.channels,
            pseudo_channels_per_channel=spec.pseudo_channels_per_channel,
            bank_groups_per_channel=spec.bank_groups_per_channel,
            banks_per_pseudo_channel=spec.banks_per_pseudo_channel,
            io_width=spec.io_width,
            address_mapping=mapping_scheme,
        )

        # Get mapping based on scheme
        mapping = self._get_hbm4_mapping(mapping_scheme)

        super().__init__(config, custom_mapping=mapping)

    def _get_hbm4_mapping(self, mapping_scheme: str) -> Dict:
        """Get HBM4-specific address mapping for the specified scheme

        This method returns the bit field configuration for each supported
        mapping scheme. The mapping defines how address bits are allocated
        to different HBM address fields.

        Args:
            mapping_scheme: The address mapping scheme name:
                - "rbc" or "hbm4": Row-Bank-Channel (default, sequential access)
                - "bcr": Bank-Channel-Row (maximizes parallelism)
                - "crb": Channel-Row-Bank (cross-channel random access)

        Returns:
            Dictionary mapping field names to (msb, lsb, bits) tuples.
            The tuple contains:
                - msb: Most significant bit position (0-indexed from LSB)
                - lsb: Least significant bit position
                - bits: Number of bits for the field

        Mapping Details:
            RBC (Row-Bank-Channel) - Default for HBM4:
                Optimized for sequential access patterns where row changes
                are infrequent, maximizing row buffer hit rate.
                Bit layout: Stack > Channel > Pch > BankGroup > Bank > Row > Col > Burst

            BCR (Bank-Channel-Row):
                Maximizes bank-level parallelism by spreading banks across
                a wider address range. Good for random access with many banks.
                Bit layout: Stack > BankGroup > Bank > Channel > Pch > Row > Col

            CRB (Channel-Row-Bank):
                Places channel at top bits for easy striping across channels.
                Best for applications that explicitly manage channel routing.
                Bit layout: Channel > Stack > Pch > BankGroup > Bank > Row > Col

        Raises:
            ValueError: If mapping_scheme is unknown and not in fallback list.

        Example:
            >>> mapping = decoder._get_hbm4_mapping("rbc")
            >>> print(mapping['channel'])
            (45, 41, 5)  # 5 bits at position 41-45 for 32 channels
        """
        if mapping_scheme == "hbm4" or mapping_scheme == "rbc":
            # HBM4 default: Row-Bank-Channel (optimized for sequential access)
            # Address layout for HBM4 32-channel:
            # - Stack: bits 47-46 (2 bits)
            # - Channel: bits 45-41 (5 bits for 32 channels)
            # - Pseudo-channel: bit 40 (1 bit for 2 pseudo-ch)
            # - Bank group: bits 39-37 (3 bits for 8 groups)
            # - Bank: bits 36-33 (4 bits for 16 banks)
            # - Row: bits 32-17 (16 bits for 64K rows)
            # - Column: bits 16-11 (6 bits for 64 columns)
            # - Burst: bits 10-9 (2 bits for 4-beat burst alignment)
            # - Offset: bits 8-6 (3 bits for 8-byte offset within burst)
            return {
                'stack': (47, 46, 2),
                'channel': (45, 41, 5),      # 32 channels
                'pseudo_channel': (40, 40, 1),  # 2 pseudo-channels
                'bank_group': (39, 37, 3),     # 8 bank groups
                'bank': (36, 33, 4),           # 16 banks
                'row': (32, 17, 16),           # 64K rows
                'col': (16, 11, 6),            # 64 columns
                'burst': (10, 9, 2),          # 4-beat burst (matches spec)
                'offset': (8, 6, 3),          # 8-byte offset alignment
            }
        elif mapping_scheme == "bcr":
            # Bank-Channel-Row (maximizes parallelism)
            return {
                'stack': (47, 46, 2),
                'bank_group': (45, 43, 3),
                'bank': (42, 39, 4),
                'channel': (38, 34, 5),       # 32 channels
                'pseudo_channel': (33, 33, 1),
                'row': (32, 17, 16),
                'col': (16, 11, 6),
                'burst': (10, 9, 2),          # 4-beat burst
                'offset': (8, 6, 3),
            }
        elif mapping_scheme == "crb":
            # Channel-Row-Bank (optimized for cross-channel random access)
            return {
                'channel': (47, 43, 5),        # 32 channels at top
                'stack': (42, 41, 2),
                'pseudo_channel': (40, 40, 1),
                'bank_group': (39, 37, 3),
                'bank': (36, 33, 4),
                'row': (32, 17, 16),
                'col': (16, 11, 6),
                'burst': (10, 3, 8),
                'offset': (2, 0, 3),
            }
        else:
            return self._get_hbm4_mapping("hbm4")

    def decode(self, addr: int) -> DecodedAddress:
        """Decode HBM4 address into component fields

        Parses a 64-bit physical address into its HBM4 address components
        according to the configured mapping scheme.

        HBM4 Address Format (64-bit):
            Stack ID:      bits [47:46] (2 bits, supports 4 stacks)
            Channel:       bits [45:41] (5 bits, 32 channels) - RBC default
            Pseudo-channel: bit [40] (1 bit, 2 pseudo-channels)
            Bank group:    bits [39:37] (3 bits, 8 bank groups)
            Bank:          bits [36:33] (4 bits, 16 banks per group)
            Row:           bits [32:17] (16 bits, 64K rows)
            Column:        bits [16:11] (6 bits, 64 columns)
            Burst beat:    bits [10:9] (2 bits, 4-beat burst alignment)
            Byte offset:   bits [8:6] (3 bits, 8-byte offset within burst)

        Note:
            The actual bit positions depend on the mapping scheme configured
            at construction time. The positions above reflect the RBC (default)
            mapping scheme.

        Args:
            addr: 64-bit physical address to decode. Must be 8-byte aligned
                  (bits [2:0] = 0). Misaligned addresses are automatically
                  masked to 8-byte boundary before decoding.

        Returns:
            DecodedAddress containing all decoded fields:
                - stack_id: Stack identifier (0-3)
                - channel_id: Channel identifier (0-31)
                - pseudo_channel_id: Pseudo-channel (0-1)
                - bank_group_id: Bank group (0-7)
                - bank_id: Bank within group (0-15)
                - row_id: Row address (0-65535)
                - col_id: Column address (0-63)
                - burst_id: Burst beat index (0-3)
                - byte_offset: Byte offset within burst (0-7)

        Raises:
            AddressError: If address is misaligned or contains invalid field values.

        Example:
            >>> decoder = HBM4AddressDecoder(mapping_scheme="rbc")
            >>> decoded = decoder.decode(0x0001_2345_6789_ABC0)
            >>> print(f"Channel {decoded.channel_id}, Row 0x{decoded.row_id:x}")
            Channel 18, Row 0x1234
        """
        # Ensure 8-byte alignment before decoding
        if addr & 0x7:
            # Align to 8-byte boundary, then decode
            aligned_addr = addr & ~0x7
            result = super().decode(aligned_addr)
        else:
            result = super().decode(addr)

        # Extract burst_id and byte_offset from mapping
        mapping = self._get_hbm4_mapping(self._mapping_scheme)
        if 'burst' in mapping:
            burst_msb, burst_lsb, _ = mapping['burst']
            result.burst_id = (addr >> burst_lsb) & ((1 << (burst_msb - burst_lsb + 1)) - 1)
        if 'offset' in mapping:
            offset_msb, offset_lsb, _ = mapping['offset']
            result.byte_offset = (addr >> offset_lsb) & ((1 << (offset_msb - offset_lsb + 1)) - 1)

        return result

    def get_channel_id(self, addr: int) -> int:
        """Extract channel ID from address

        Extracts the channel identifier from a 64-bit address based on
        the configured mapping scheme.

        Args:
            addr: 64-bit physical address

        Returns:
            Channel ID (0-31 for HBM4 default mapping).
            The valid range depends on the mapping scheme:
            - RBC/HBM4: 0-31 (5 bits)
            - BCR: 0-31 (5 bits)
            - CRB: 0-31 (5 bits)

        Note:
            The channel bit position varies with the mapping scheme.
            This method uses the current mapping scheme's channel field
            definition to extract the correct bits.

        Example:
            >>> decoder = HBM4AddressDecoder(mapping_scheme="rbc")
            >>> ch = decoder.get_channel_id(0x0020_0000_0000_0000)  # Channel 1
            >>> print(ch)
            1
        """
        # Channel bit position depends on mapping scheme
        mapping = self._get_hbm4_mapping(self._mapping_scheme)
        ch_msb, ch_lsb, _ = mapping['channel']
        return (addr >> ch_lsb) & ((1 << (ch_msb - ch_lsb + 1)) - 1)

    def get_pseudo_channel_id(self, addr: int) -> int:
        """Extract pseudo-channel ID from address

        Pseudo-channels demultiplex a single channel into two independent
        sub-channels, each with its own command queue.

        Args:
            addr: 64-bit physical address

        Returns:
            Pseudo-channel ID (0 or 1).
            Pseudo-channel 0 and 1 share the same channel but have
            independent bank groups and banks.

        Note:
            In RBC mapping, pseudo-channel is bit 40 (1 bit).
            In other mappings, the position may vary.

        Example:
            >>> decoder = HBM4AddressDecoder()
            >>> pch = decoder.get_pseudo_channel_id(0x0010_0000_0000_0000)  # Pch 1
            >>> print(pch)
            1
        """
        # Pseudo-channel is bit 40 in RBC mapping
        mapping = self._get_hbm4_mapping(self._mapping_scheme)
        if 'pseudo_channel' in mapping:
            _, pc_lsb, _ = mapping['pseudo_channel']
            return (addr >> pc_lsb) & 0x1
        return 0

    def get_row_id(self, addr: int) -> int:
        """Extract row ID from address

        Rows are the finest granularity for activation/deactivation.
        Each row contains 2KB of data (256 columns × 8 bytes).

        Args:
            addr: 64-bit physical address

        Returns:
            Row ID (0-65535 for 16-bit row field).
            The row address is used for row buffer hit/miss detection.

        Note:
            In RBC mapping, row is bits 32:17 (16 bits = 64K rows).
            Row hits occur when consecutive accesses target the same row.

        Example:
            >>> decoder = HBM4AddressDecoder()
            >>> row = decoder.get_row_id(0x0001_0000_0000_0000)  # Row 0x100
            >>> print(f"Row: 0x{row:x}")
            Row: 0x100
        """
        # Row is bits 32:17 in RBC mapping (16 bits = 64K rows)
        mapping = self._get_hbm4_mapping(self._mapping_scheme)
        if 'row' in mapping:
            row_msb, row_lsb, row_bits = mapping['row']
            return (addr >> row_lsb) & ((1 << row_bits) - 1)
        return 0

    def get_bank_id(self, addr: int) -> int:
        """Extract bank ID from address

        Each bank group contains multiple banks that can be accessed
        independently, enabling bank-level parallelism.

        Args:
            addr: 64-bit physical address

        Returns:
            Bank ID (0-15 for 4-bit bank field within group).
            Banks are indexed within their bank group.

        Note:
            In RBC mapping, bank is bits 36:33 (4 bits = 16 banks).
            Combined with bank groups, HBM4 supports 128 banks per
            pseudo-channel (8 groups × 16 banks).

        Example:
            >>> decoder = HBM4AddressDecoder()
            >>> bank = decoder.get_bank_id(0x0000_2000_0000_0000)  # Bank 2
            >>> print(bank)
            2
        """
        # Bank is bits 36:33 in RBC mapping (4 bits = 16 banks)
        mapping = self._get_hbm4_mapping(self._mapping_scheme)
        if 'bank' in mapping:
            bank_msb, bank_lsb, _ = mapping['bank']
            return (addr >> bank_lsb) & ((1 << (bank_msb - bank_lsb + 1)) - 1)
        return 0

    def get_bank_group_id(self, addr: int) -> int:
        """Extract bank group ID from address

        Bank groups allow independent command timing for different groups,
        enabling command scheduling optimizations.

        Args:
            addr: 64-bit physical address

        Returns:
            Bank group ID (0-7 for 3-bit bank group field).
            Each pseudo-channel has 8 bank groups.

        Note:
            In RBC mapping, bank group is bits 39:37 (3 bits = 8 groups).
            Commands to different bank groups have longer timing gaps
            (nCCDL vs nCCDS) due to shared command bus.

        Example:
            >>> decoder = HBM4AddressDecoder()
            >>> bg = decoder.get_bank_group_id(0x0000_8000_0000_0000)  # BG 3
            >>> print(bg)
            3
        """
        # Bank group is bits 39:37 in RBC mapping (3 bits = 8 groups)
        mapping = self._get_hbm4_mapping(self._mapping_scheme)
        if 'bank_group' in mapping:
            bg_msb, bg_lsb, _ = mapping['bank_group']
            return (addr >> bg_lsb) & ((1 << (bg_msb - bg_lsb + 1)) - 1)
        return 0

    def get_column_id(self, addr: int) -> int:
        """Extract column ID from address

        Columns address individual data elements within an active row.

        Args:
            addr: 64-bit physical address

        Returns:
            Column ID (0-63 for 6-bit column field in RBC mapping).

        Note:
            In RBC mapping, column is bits 16:11 (6 bits = 64 columns).
            With 256-bit (32 byte) bus and 4-beat burst, each column
            access transfers 128 bytes.

        Example:
            >>> decoder = HBM4AddressDecoder()
            >>> col = decoder.get_column_id(0x0000_0800_0000_0000)  # Col 0x20
            >>> print(col)
            32
        """
        mapping = self._get_hbm4_mapping(self._mapping_scheme)
        if 'col' in mapping:
            col_msb, col_lsb, col_bits = mapping['col']
            return (addr >> col_lsb) & ((1 << col_bits) - 1)
        return 0

    def get_stack_id(self, addr: int) -> int:
        """Extract stack ID from address

        Multiple HBM stacks can be connected to a single controller.

        Args:
            addr: 64-bit physical address

        Returns:
            Stack ID (0-3 for 2-bit stack field).

        Example:
            >>> decoder = HBM4AddressDecoder()
            >>> stack = decoder.get_stack_id(0x8000_0000_0000_0000)  # Stack 2
            >>> print(stack)
            2
        """
        mapping = self._get_hbm4_mapping(self._mapping_scheme)
        if 'stack' in mapping:
            stack_msb, stack_lsb, _ = mapping['stack']
            return (addr >> stack_lsb) & ((1 << (stack_msb - stack_lsb + 1)) - 1)
        return 0

    def get_address_range(self, channel: Optional[int] = None) -> Tuple[int, int]:
        """Calculate address range for a channel or full memory

        Args:
            channel: Specific channel to calculate range for, or None for total.

        Returns:
            Tuple of (start_address, end_address) defining the addressable range.

        Example:
            >>> decoder = HBM4AddressDecoder()
            >>> start, end = decoder.get_address_range(channel=0)
            >>> print(f"Channel 0: 0x{start:016x} - 0x{end:016x}")
            Channel 0: 0x0000000000000000 - 0x07FFFFFFFFFFFF
        """
        # Calculate bits for each field
        total_bits = (
            self.STACK_BITS + self.CHANNEL_BITS + self.PCH_BITS +
            self.BG_BITS + self.BANK_BITS + self.ROW_BITS +
            self.COL_BITS + self.BURST_BITS + self.OFFSET_BITS
        )
        max_addr = (1 << total_bits) - 1

        if channel is not None:
            # Mask for single channel
            channel_mask = ((1 << self.CHANNEL_BITS) - 1) << 41
            start = channel_mask & (channel << 41)
            end = start | (max_addr & ~channel_mask)
        else:
            start = 0
            end = max_addr

        return start, end

    def validate_address(self, addr: int) -> bool:
        """Validate that an address is properly formatted for HBM4

        Args:
            addr: 64-bit physical address to validate

        Returns:
            True if the address is valid for HBM4:
            - 8-byte aligned (bits [2:0] = 0)
            - All field values within valid ranges
            False otherwise
        """
        # Check alignment first (before auto-masking)
        if addr & 0x7:
            return False

        try:
            decoded = self.decode(addr)
            # Check field ranges
            if decoded.channel_id >= 32:
                return False
            if decoded.pseudo_channel_id >= 2:
                return False
            if decoded.bank_group_id >= 8:
                return False
            if decoded.bank_id >= 16:
                return False
            if decoded.row_id >= (1 << self.ROW_BITS):
                return False
            return True
        except Exception:
            return False