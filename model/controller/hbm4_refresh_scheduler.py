"""
HBM4 Refresh Scheduler with Autonomous Per-Bank Refresh

Based on research findings:
- Per-bank and all-bank refresh modes
- Autonomous refresh management
- DRFM (Direct Refresh Management) for row-hammer mitigation
- Staggered refresh for reduced peak power

Reference:
- Synopsys DesignWare HBM4/4E Controller IP
- JEDEC JESD270-4A HBM4 specification
- Ramulator 2.0 HBM3 refresh implementation
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from model.dram.hbm4_spec import HBM4Spec


class RefreshMode(Enum):
    """Refresh operating modes

    HBM4 supports multiple refresh modes for different power/performance tradeoffs.
    """
    ALL_BANKS = "all"         # Refresh all banks at once
    PER_BANK = "per_bank"     # Staggered per-bank refresh (default for HBM4)
    BANK_GROUP = "bank_group"  # Refresh by bank group


@dataclass
class RefreshBankStatus:
    """Status tracking for per-bank refresh

    Tracks when each bank was last refreshed and if it needs refresh.
    """
    bank_id: int
    last_refresh_cycle: int = 0
    needs_refresh: bool = False


class HBM4RefreshScheduler:
    """HBM4 Refresh Scheduler

    Manages DRAM refresh operations with support for:
    - All-bank refresh (HBM2 style)
    - Per-bank refresh (staggered, HBM3/HBM4 style)
    - Bank group refresh
    - Autonomous refresh scheduling
    - DRFM (Direct Refresh Management) for row-hammer mitigation

    Reference: Synopsys DesignWare HBM4/4E Controller IP
    """

    def __init__(self, config: Optional[HBM4Spec] = None):
        """Initialize refresh scheduler

        Args:
            config: HBM4 specification (uses default if None)
        """
        if config is None:
            config = HBM4Spec()

        self.spec = config
        self.mode = RefreshMode.PER_BANK  # Default to per-bank for HBM4

        # Timing parameters from spec
        self.tREFI = config.nREFI  # Refresh interval (cycles)
        self.tRFC = config.nRFC    # Refresh command duration (cycles)
        self.nRREFD = config.nRREFD  # Per-bank refresh interval

        # Refresh state tracking
        self.cycles_since_refresh = 0
        self.current_refresh_bank = 0
        self.total_refresh_count = 0
        self.current_cycle = 0

        # Per-bank refresh tracking
        self.bank_status: List[RefreshBankStatus] = [
            RefreshBankStatus(bank_id=i)
            for i in range(config.total_banks)
        ]

        # Bank group refresh tracking (8 groups × 16 banks)
        self.bank_groups_per_channel = config.bank_groups_per_channel

        self.supported_modes = [
            RefreshMode.ALL_BANKS,
            RefreshMode.PER_BANK,
            RefreshMode.BANK_GROUP
        ]

        # DRFM (Direct Refresh Management) for row-hammer mitigation
        self.drfm_enabled = False
        self.drfm_rowhammer_threshold = 1000  # cycles before refresh needed

        # Statistics
        self.stats = {
            'total_refreshes': 0,
            'all_bank_refreshes': 0,
            'per_bank_refreshes': 0,
            'bank_group_refreshes': 0
        }

    def tick(self):
        """Advance refresh timer by one cycle"""
        self.cycles_since_refresh += 1
        self.current_cycle += 1

    def can_refresh(self) -> bool:
        """Check if refresh is needed

        Returns:
            True if enough cycles have passed since last refresh
        """
        return self.cycles_since_refresh >= self.tREFI

    def get_next_refresh_bank(self) -> Optional[Tuple[int, int, int]]:
        """Get next bank to refresh (wrapper for backward compatibility)

        Returns:
            Tuple of (channel_id, pseudo_channel_id, bank_id) or None if no refresh needed
        """
        result = self.get_refresh_command()
        if result is None:
            return None

        command_name, channel_id, pseudo_channel_id, bank_id = result
        return (channel_id, pseudo_channel_id, bank_id)

    def get_refresh_command(self) -> Optional[tuple]:
        """Get the next refresh command to execute

        Returns:
            Tuple of (command_name, channel_id, pseudo_channel_id, bank_id) or None

        Note:
            - channel_id: 0-31 for 32 channels
            - pseudo_channel_id: 0 or 1 (within channel)
            - bank_id: 0-15 (within pseudo-channel)
        """
        if not self.can_refresh():
            return None

        if self.mode == RefreshMode.ALL_BANKS:
            self.total_refresh_count += 1
            self.cycles_since_refresh = 0
            self.stats['total_refreshes'] += 1
            self.stats['all_bank_refreshes'] += 1
            return ('REFab', None, None, None)

        elif self.mode == RefreshMode.PER_BANK:
            # Rotate through banks (global bank index 0-511 for HBM4)
            # 32 channels × 2 pseudo-channels × 16 banks = 1024 global banks (but spec uses 512)
            bank_to_refresh = self.current_refresh_bank

            # Calculate channel, pseudo-channel, and bank indices
            banks_per_pch = self.spec.banks_per_pseudo_channel  # 16
            pch_idx = bank_to_refresh // banks_per_pch  # 0-63 (pseudo-channel within array)
            bank_idx = bank_to_refresh % banks_per_pch  # 0-15

            # Map pseudo-channel index to channel and pseudo-channel
            # Each physical channel has 2 pseudo-channels
            channel_id = pch_idx // 2  # 0-31
            pseudo_channel_id = pch_idx % 2  # 0 or 1

            self.current_refresh_bank = (self.current_refresh_bank + 1) % self.spec.total_banks
            self.cycles_since_refresh = 0
            self.total_refresh_count += 1
            self.stats['total_refreshes'] += 1
            self.stats['per_bank_refreshes'] += 1

            # Update bank status using global bank index for channel model integration
            self.mark_bank_refreshed(channel_id, pseudo_channel_id, bank_idx, self.current_cycle)

            return ('REFsb', channel_id, pseudo_channel_id, bank_idx)

        elif self.mode == RefreshMode.BANK_GROUP:
            # Refresh one bank group per interval
            group_to_refresh = (self.total_refresh_count //
                               self.spec.banks_per_pseudo_channel) % self.bank_groups_per_channel
            self.cycles_since_refresh = 0
            self.total_refresh_count += 1
            self.stats['total_refreshes'] += 1
            self.stats['bank_group_refreshes'] += 1

            # Bank group refresh targets pseudo-channel 0, starting bank of the group
            return ('REFsb', 0, 0, group_to_refresh * self.spec.banks_per_pseudo_channel)

        return None

    def set_mode(self, mode: RefreshMode):
        """Set refresh operating mode

        Args:
            mode: New refresh mode
        """
        if mode in self.supported_modes:
            self.mode = mode

    def mark_bank_refreshed(self, channel_id: int, pseudo_channel_id: int, bank_id: int, cycle: int):
        """Mark a specific bank as refreshed

        Args:
            channel_id: Channel index (0-31)
            pseudo_channel_id: Pseudo-channel index (0 or 1)
            bank_id: Bank index within pseudo-channel (0-15)
            cycle: Current cycle when refresh occurred
        """
        # Convert to global bank index: channels have 2 pseudo-channels each
        # Global bank index = channel_id * (2 * banks_per_pch) + pseudo_channel_id * banks_per_pch + bank_id
        global_bank_id = (
            channel_id * self.spec.pseudo_channels_per_channel * self.spec.banks_per_pseudo_channel +
            pseudo_channel_id * self.spec.banks_per_pseudo_channel +
            bank_id
        )
        if 0 <= global_bank_id < len(self.bank_status):
            self.bank_status[global_bank_id].last_refresh_cycle = cycle
            self.bank_status[global_bank_id].needs_refresh = False

    def enable_drfm(self, enabled: bool = True, threshold: int = None):
        """Enable/disable DRFM (Direct Refresh Management)

        DRFM provides row-hammer mitigation by tracking access counts
        and triggering targeted refreshes.

        Args:
            enabled: True to enable DRFM
            threshold: Optional threshold for row-hammer detection
        """
        self.drfm_enabled = enabled
        if threshold is not None:
            self.drfm_rowhammer_threshold = threshold

    def get_banks_needing_refresh(self) -> List[int]:
        """Get list of banks that need refresh (DRFM)

        Returns:
            List of bank IDs that need refresh due to row-hammer
        """
        if not self.drfm_enabled:
            return []

        # Find banks that haven't been refreshed recently
        threshold = self.current_cycle - self.drfm_rowhammer_threshold
        return [
            bs.bank_id for bs in self.bank_status
            if bs.last_refresh_cycle < threshold
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get refresh scheduler statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'total_refreshes': self.stats['total_refreshes'],
            'all_bank_refreshes': self.stats['all_bank_refreshes'],
            'per_bank_refreshes': self.stats['per_bank_refreshes'],
            'bank_group_refreshes': self.stats['bank_group_refreshes'],
            'cycles_since_refresh': self.cycles_since_refresh,
            'current_cycle': self.current_cycle,
            'mode': self.mode.value,
            'drfm_enabled': self.drfm_enabled
        }

    def set_refresh_interval(self, cycles: int):
        """Set refresh interval (tREFI)

        Args:
            cycles: New refresh interval in cycles
        """
        self.tREFI = cycles

    def reset(self):
        """Reset scheduler state"""
        self.cycles_since_refresh = 0
        self.current_refresh_bank = 0
        self.total_refresh_count = 0
        self.current_cycle = 0

        for bs in self.bank_status:
            bs.last_refresh_cycle = 0
            bs.needs_refresh = False