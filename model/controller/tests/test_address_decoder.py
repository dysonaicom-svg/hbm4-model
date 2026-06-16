"""
Tests for Address Decoder
"""

import pytest
from model.controller.address_decoder import AddressDecoder, DecodedAddress
from model.controller.config import HBMConfig


class TestAddressDecoder:
    """Test Address Decoder"""

    def test_default_config(self):
        """测试默认配置"""
        config = HBMConfig()
        decoder = AddressDecoder(config)
        assert decoder.config.stack_count == 2
        assert decoder.config.channels_per_stack == 8
        assert decoder.config.banks_per_pseudo_channel == 16

    def test_decode_row_addr(self):
        """测试解码行地址"""
        config = HBMConfig()
        decoder = AddressDecoder(config)
        addr = 0x100000  # 1MB address
        decoded = decoder.decode(addr)

        assert isinstance(decoded, DecodedAddress)
        assert 0 <= decoded.stack_id < decoder.config.stack_count
        assert 0 <= decoded.channel_id < decoder.config.channels_per_stack
        assert 0 <= decoded.bank_id < decoder.config.banks_per_pseudo_channel

    def test_encode_decode_consistency(self):
        """测试编码解码一致性"""
        config = HBMConfig()
        decoder = AddressDecoder(config)

        original_addr = 0x100000
        decoded = decoder.decode(original_addr)
        reencoded = decoder.encode(decoded)

        # 重新编码后的地址应该映射到相同的 bank
        decoded2 = decoder.decode(reencoded)
        assert decoded.bank_id == decoded2.bank_id
        assert decoded.row_id == decoded2.row_id

    def test_custom_mapping_rbc(self):
        """测试 RBC 映射"""
        config = HBMConfig()
        config.address_mapping = "rbc"
        decoder = AddressDecoder(config)
        assert 'row' in decoder.masks

    def test_custom_mapping_bcr(self):
        """测试 BCR 映射"""
        config = HBMConfig()
        config.address_mapping = "bcr"
        decoder = AddressDecoder(config)
        assert 'bank' in decoder.masks

    def test_decode_address_bits(self):
        """测试地址位解析"""
        config = HBMConfig()
        decoder = AddressDecoder(config)
        addr = 0x10000000000  # 任意地址

        decoded = decoder.decode(addr)

        # 验证各字段在有效范围内
        assert decoded.stack_id < decoder.config.stack_count
        assert decoded.channel_id < decoder.config.channels_per_stack
        assert decoded.bank_id < decoder.config.banks_per_pseudo_channel * decoder.config.pseudo_channels_per_channel

    def test_get_bank_key(self):
        """测试获取 bank key"""
        config = HBMConfig()
        decoder = AddressDecoder(config)
        decoded = DecodedAddress(
            stack_id=0, channel_id=1, pseudo_channel_id=0, bank_id=2
        )

        key = decoder.get_bank_key(decoded)
        assert key == (0, 1, 0, 2)

    def test_get_row_key(self):
        """测试获取 row key"""
        config = HBMConfig()
        decoder = AddressDecoder(config)
        decoded = DecodedAddress(
            stack_id=0, channel_id=1, pseudo_channel_id=0, bank_id=2, row_id=100
        )

        key = decoder.get_row_key(decoded)
        assert key == (0, 1, 0, 2, 100)


class TestDecodedAddress:
    """Test DecodedAddress dataclass"""

    def test_address_creation(self):
        """测试地址创建"""
        addr = DecodedAddress(
            stack_id=0,
            channel_id=0,
            pseudo_channel_id=0,
            bank_id=0,
            row_id=100,
            col_id=0,
        )
        assert addr.stack_id == 0
        assert addr.row_id == 100

    def test_address_repr(self):
        """测试地址表示"""
        addr = DecodedAddress(
            stack_id=1,
            channel_id=2,
            pseudo_channel_id=0,
            bank_id=3,
            row_id=0x1234,
            col_id=0,
        )
        repr_str = repr(addr)
        assert "ch=2" in repr_str
        assert "bk=3" in repr_str