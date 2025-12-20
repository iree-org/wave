"""Unit tests for DS (LDS) instruction classes with offset support."""

import pytest
from wave_lang.kernel.wave.asm.instructions import (
    DSReadB64,
    DSWriteB32,
    DSWriteB64,
    DSWriteB128,
)


class TestDSInstructionOffset:
    """Tests for DS instruction offset parameter."""

    def test_ds_read_b64_no_offset(self):
        """DSReadB64 without offset should not include offset modifier."""
        instr = DSReadB64((10, 11), 5)
        assert str(instr) == "    ds_read_b64 v[10:11], v5"

    def test_ds_read_b64_with_offset(self):
        """DSReadB64 with offset should include offset modifier."""
        instr = DSReadB64((10, 11), 5, offset=128)
        assert str(instr) == "    ds_read_b64 v[10:11], v5 offset:128"

    def test_ds_read_b64_offset_zero(self):
        """DSReadB64 with offset=0 should not include modifier."""
        instr = DSReadB64((10, 11), 5, offset=0)
        assert str(instr) == "    ds_read_b64 v[10:11], v5"

    def test_ds_read_b64_max_offset(self):
        """DSReadB64 should accept max offset (65535)."""
        instr = DSReadB64((10, 11), 5, offset=65535)
        assert "offset:65535" in str(instr)

    def test_ds_read_b64_negative_offset_raises(self):
        """DSReadB64 should raise for negative offset."""
        with pytest.raises(ValueError, match="must be 0-65535"):
            DSReadB64((10, 11), 5, offset=-1)

    def test_ds_read_b64_overflow_offset_raises(self):
        """DSReadB64 should raise for offset > 65535."""
        with pytest.raises(ValueError, match="must be 0-65535"):
            DSReadB64((10, 11), 5, offset=65536)

    def test_ds_write_b32_with_offset(self):
        """DSWriteB32 with offset should include offset modifier."""
        instr = DSWriteB32(5, 10, offset=64)
        assert str(instr) == "    ds_write_b32 v5, v10 offset:64"

    def test_ds_write_b64_with_offset(self):
        """DSWriteB64 with offset should include offset modifier."""
        instr = DSWriteB64(5, (10, 11), offset=256)
        assert str(instr) == "    ds_write_b64 v5, v[10:11] offset:256"

    def test_ds_write_b128_with_offset(self):
        """DSWriteB128 with offset should include offset modifier."""
        instr = DSWriteB128(5, (10, 11, 12, 13), offset=512)
        assert str(instr) == "    ds_write_b128 v5, v[10:13] offset:512"


class TestDSInstructionAlignment:
    """Tests for DS instruction offset alignment considerations."""

    def test_ds_read_b64_8byte_aligned_offsets(self):
        """Common 8-byte aligned offsets for ds_read_b64."""
        for offset in [0, 8, 16, 64, 128, 256, 512, 1024, 2040]:
            instr = DSReadB64((0, 1), 0, offset=offset)
            if offset == 0:
                assert "offset" not in str(instr)
            else:
                assert f"offset:{offset}" in str(instr)
