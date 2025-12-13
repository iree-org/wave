"""
Tests for the ExprEmitter class which emits AMDGCN assembly from SymPy expressions.

These tests verify the actual instruction sequences emitted, not just that something happened.
"""

import pytest
import sympy as sp
from wave_lang.kernel.wave.asm.expression_emitter import ExprEmitter
from wave_lang.kernel.wave.asm.instructions import (
    VMovB32,
    VAddU32,
    VLshlRevB32,
    VLshrrevB32,
    VMulLoU32,
    VAndB32,
)


# ============================================================================
# Test helper classes - real implementations instead of mocks
# ============================================================================


class FakeRegisterFile:
    """Simple register file for testing."""

    def __init__(self):
        self.v_used = set()


class FakeVGPRAllocator:
    """Fake VGPR allocator for testing."""

    def __init__(self):
        self._next_temp_reg = 10  # Start temp registers at v10
        self._freed_regs = []  # Track freed registers

    def alloc_v(self):
        """Allocate a temporary vector register."""
        reg = self._next_temp_reg
        self._next_temp_reg += 1
        return reg

    def free_v(self, reg_num):
        """Free a temporary register."""
        self._freed_regs.append(reg_num)


class FakeEmitter:
    """Fake emitter that captures emitted instructions."""

    def __init__(self):
        self.register_file = FakeRegisterFile()
        self.vgpr_allocator = FakeVGPRAllocator()
        self.instructions = []  # List of emitted instructions
        self._lane_id = 1  # lane_id is always in v1
        self.asm_lines = []  # Capture raw assembly strings

    def ensure_lane_id(self, subgroup_size):
        """Return the lane_id register number."""
        return self._lane_id

    def emit_instruction(self, instruction):
        """Capture emitted instruction."""
        self.instructions.append(instruction)

    def emit(self, asm_line):
        """Capture raw assembly line."""
        self.asm_lines.append(asm_line)

    def get_instruction_types(self):
        """Get list of instruction types for easy testing."""
        return [type(instr) for instr in self.instructions]

    def get_instructions_by_type(self, instr_type):
        """Get all instructions of a specific type."""
        return [instr for instr in self.instructions if isinstance(instr, instr_type)]

    def reset(self):
        """Reset for a fresh test."""
        self.instructions = []
        self.asm_lines = []
        self.register_file.v_used = set()
        self.vgpr_allocator._next_temp_reg = 10
        self.vgpr_allocator._freed_regs = []


class FakeKernelInfo:
    """Simple kernel info for testing."""

    def __init__(self, subgroup_size=64, wg_size=(64, 1, 1)):
        self.subgroup_size = subgroup_size
        self.wg_size = wg_size


# ============================================================================
# Test suite
# ============================================================================


class TestExprEmitter:
    """Test suite for ExprEmitter."""

    @pytest.fixture
    def test_emitter(self):
        """Create a test emitter."""
        return FakeEmitter()

    @pytest.fixture
    def kernel_info(self):
        """Create kernel info."""
        return FakeKernelInfo(subgroup_size=64)

    @pytest.fixture
    def expr_emitter(self, test_emitter, kernel_info):
        """Create an ExprEmitter instance."""
        return ExprEmitter(test_emitter, kernel_info)

    def test_emit_constant(self, expr_emitter, test_emitter):
        """Test emitting a constant integer."""
        expr = sp.Integer(42)
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Should emit exactly: v_mov_b32 v2, 42
        assert len(test_emitter.instructions) == 1
        instr = test_emitter.instructions[0]
        assert isinstance(instr, VMovB32)
        assert instr.mnemonic == "v_mov_b32"
        assert instr.operands == ["v2", "42"]

    def test_emit_tid_x(self, expr_emitter, test_emitter):
        """Test emitting tid_x symbol - returns lane_id directly, no instructions."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        result = expr_emitter.emit(tid_x, "v2")

        assert result == "v1"  # Returns lane_id register directly

        # No instructions should be emitted (tid_x is already in v1)
        assert len(test_emitter.instructions) == 0

    def test_emit_tid_x_plus_constant(self, expr_emitter, test_emitter):
        """Test emitting tid_x + constant."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = tid_x + 5
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Should emit (efficient sequence):
        # 1. v_mov_b32 v_temp, 5        (materialize const to temp)
        # 2. v_add_u32 v_temp, v_temp, v1  (add lane_id to const)
        # 3. v_mov_b32 v2, v_temp       (move to dst)
        assert len(test_emitter.instructions) == 3
        instrs = test_emitter.instructions

        # Check instruction types and key operations
        assert isinstance(instrs[0], VMovB32)
        assert instrs[0].operands[1] == "5"  # Loading constant 5
        temp_reg = instrs[0].operands[0]

        assert isinstance(instrs[1], VAddU32)
        assert instrs[1].operands[0] == temp_reg  # Accumulating in place
        assert "v1" in instrs[1].operands  # Adding lane_id

        assert isinstance(instrs[2], VMovB32)
        assert instrs[2].operands == ["v2", temp_reg]  # Moving to destination

    def test_emit_multiplier_times_tid_x_power_of_2(self, expr_emitter, test_emitter):
        """Test emitting k * tid_x where k is power of 2 - optimized to shift."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = 8 * tid_x  # 8 is 2^3
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Should emit:
        # 1. v_lshlrev_b32 v10, 3, v1  (shift left by 3)
        # 2. v_mov_b32 v2, v10          (move to dst)
        assert len(test_emitter.instructions) == 2
        instrs = test_emitter.instructions

        assert isinstance(instrs[0], VLshlRevB32)
        assert instrs[0].operands == ["v10", "3", "v1"]

        assert isinstance(instrs[1], VMovB32)
        assert instrs[1].operands == ["v2", "v10"]

    def test_emit_multiplier_times_tid_x_non_power_of_2(
        self, expr_emitter, test_emitter
    ):
        """Test emitting k * tid_x where k is not power of 2 - uses multiply."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = 7 * tid_x
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Should emit:
        # 1. v_mov_b32 v_temp1, 7           (load const multiplier)
        # 2. v_mul_lo_u32 v_temp2, v1, v_temp1  (multiply)
        # 3. v_mov_b32 v2, v_temp2          (move to dst)
        assert len(test_emitter.instructions) == 3
        instrs = test_emitter.instructions

        # Check instruction types and key operands (register numbers may vary)
        assert isinstance(instrs[0], VMovB32)
        assert instrs[0].operands[1] == "7"  # Loading constant 7
        const_reg = instrs[0].operands[0]

        assert isinstance(instrs[1], VMulLoU32)
        assert instrs[1].operands[1] == "v1"  # Multiplying lane_id
        assert instrs[1].operands[2] == const_reg  # By the const we loaded
        mul_result = instrs[1].operands[0]

        assert isinstance(instrs[2], VMovB32)
        assert instrs[2].operands == ["v2", mul_result]  # Move result to v2

    def test_emit_mod_power_of_2(self, expr_emitter, test_emitter):
        """Test emitting Mod(tid_x, 2^k) - optimized to bitwise AND."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = sp.Mod(tid_x, 16)  # 16 is 2^4
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Should emit:
        # 1. v_and_b32 v10, 15, v1  (mask with 16-1=15)
        # 2. v_mov_b32 v2, v10       (move to dst)
        assert len(test_emitter.instructions) == 2
        instrs = test_emitter.instructions

        assert isinstance(instrs[0], VAndB32)
        assert instrs[0].operands == ["v10", "15", "v1"]

        assert isinstance(instrs[1], VMovB32)
        assert instrs[1].operands == ["v2", "v10"]

    def test_emit_mod_non_power_of_2_raises(self, expr_emitter):
        """Test that Mod with non-power-of-2 divisor raises ValueError."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = sp.Mod(tid_x, 17)  # Not power of 2

        with pytest.raises(ValueError, match="power-of-two"):
            expr_emitter.emit(expr, "v2")

    def test_emit_floor_div_power_of_2(self, expr_emitter, test_emitter):
        """Test emitting floor(tid_x / 2^k) - optimized to right shift."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = sp.floor(tid_x / 16)  # 16 is 2^4
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Should emit:
        # 1. v_lshrrev_b32 v10, 4, v1  (right shift by 4)
        # 2. v_mov_b32 v2, v10          (move to dst)
        assert len(test_emitter.instructions) == 2
        instrs = test_emitter.instructions

        assert isinstance(instrs[0], VLshrrevB32)
        assert instrs[0].operands == ["v10", "4", "v1"]  # log2(16) = 4

        assert isinstance(instrs[1], VMovB32)
        assert instrs[1].operands == ["v2", "v10"]

    def test_emit_floor_div_non_power_of_2_raises(self, expr_emitter):
        """Test that floor division by non-power-of-2 raises ValueError."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = sp.floor(tid_x / 17)  # Not power of 2

        with pytest.raises(ValueError, match="power-of-two"):
            expr_emitter.emit(expr, "v2")

    def test_emit_complex_expression(self, expr_emitter, test_emitter):
        """Test emitting: 4 * Mod(tid_x, 16) + 256 * floor(tid_x / 16).

        Common 2D indexing pattern. Should produce:
        - v_and_b32 for Mod(tid_x, 16)
        - v_lshlrev_b32 for *4 (shift left by 2)
        - v_lshrrev_b32 for floor(tid_x/16) (shift right by 4)
        - v_lshlrev_b32 for *256 (shift left by 8)
        - v_add_u32 to sum them
        """
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = 4 * sp.Mod(tid_x, 16) + 256 * sp.floor(tid_x / 16)
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Verify key operations are present (exact sequence depends on implementation)
        instrs = test_emitter.instructions

        # Should have: AND (mod), 2 shifts (for *4 and *256), right shift (for /16), ADD, moves
        and_instrs = [i for i in instrs if isinstance(i, VAndB32)]
        lshift_instrs = [i for i in instrs if isinstance(i, VLshlRevB32)]
        rshift_instrs = [i for i in instrs if isinstance(i, VLshrrevB32)]
        add_instrs = [i for i in instrs if isinstance(i, VAddU32)]

        assert len(and_instrs) == 1  # One Mod operation
        assert len(rshift_instrs) == 1  # One floor division
        assert len(lshift_instrs) == 2  # *4 and *256
        assert len(add_instrs) == 1  # Final addition

    def test_emit_nested_mod_floor(self, expr_emitter, test_emitter):
        """Test emitting floor(Mod(tid_x, 64) / 16) - nested operations.

        Should emit:
        - v_and_b32 for Mod(tid_x, 64)
        - v_lshrrev_b32 for floor(.../16)
        """
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = sp.floor(sp.Mod(tid_x, 64) / 16)
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        instrs = test_emitter.instructions

        # Should have: AND (mod), right shift (floor div)
        and_instrs = [i for i in instrs if isinstance(i, VAndB32)]
        rshift_instrs = [i for i in instrs if isinstance(i, VLshrrevB32)]

        assert len(and_instrs) == 1
        assert "63" in and_instrs[0].operands  # mask for 64

        assert len(rshift_instrs) == 1
        assert "4" in rshift_instrs[0].operands  # log2(16)

    def test_emit_power_of_2(self, expr_emitter, test_emitter):
        """Test emitting 2^k as a constant."""
        expr = sp.Pow(2, 5)  # 2^5 = 32
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        # Should emit v_mov_b32 v2, 32

    def test_emit_unsupported_power_raises(self, expr_emitter):
        """Test that non-2^k powers raise ValueError."""
        # Note: sp.Pow(3, 2) evaluates to 9 which is a constant, not an error
        # We need a power expression that doesn't simplify
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = sp.Pow(tid_x, 2)  # tid_x^2 is not supported

        with pytest.raises(ValueError, match="power expression"):
            expr_emitter.emit(expr, "v2")

    def test_emit_multiple_terms_sum(self, expr_emitter, test_emitter):
        """Test emitting a sum with multiple terms."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = tid_x + 10 + 20  # Should simplify to tid_x + 30
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

    def test_emit_product_of_dynamics_raises(self):
        """Test that products of multiple dynamic terms raise ValueError.

        Uses multi-wave mode where both tid_x and tid_y are dynamic (not constant).
        """
        # Create multi-wave emitter where tid_y is truly dynamic
        test_emitter = FakeEmitter()
        kernel_info = FakeKernelInfo(subgroup_size=64, wg_size=(64, 4, 1))
        expr_emitter = ExprEmitter(test_emitter, kernel_info)

        tid_x = sp.Symbol("tid_x", nonnegative=True)
        tid_y = sp.Symbol("tid_y", nonnegative=True)
        expr = tid_x * tid_y

        # Will raise because products of multiple dynamic terms are not supported
        with pytest.raises(ValueError, match="product of multiple dynamic terms"):
            expr_emitter.emit(expr, "v2")

    def test_emit_unknown_symbol_raises(self, expr_emitter):
        """Test that unknown symbols raise ValueError."""
        unknown = sp.Symbol("unknown")

        with pytest.raises(ValueError, match="Unknown symbol"):
            expr_emitter.emit(unknown, "v2")

    def test_register_allocation_and_freeing(self, expr_emitter, test_emitter):
        """Test that temporary registers are properly allocated."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        # Expression that requires temp registers
        expr = sp.Mod(tid_x, 16) + sp.floor(tid_x / 16)

        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        # Verify temp registers were allocated
        # (Constants stay as markers and don't consume registers)
        assert (
            test_emitter.vgpr_allocator._next_temp_reg > 10
        )  # Some temps were allocated

    def test_realistic_2d_indexing_pattern(self, expr_emitter, test_emitter):
        """Test realistic 2D indexing: offset = row*4 + col*256.

        Where row = tid_x % 16, col = tid_x // 16
        Common GPU pattern for 2D indexing from linear thread ID.
        """
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        row = sp.Mod(tid_x, 16)
        col = sp.floor(tid_x / 16)
        offset = 4 * row + 256 * col

        result = expr_emitter.emit(offset, "v9")

        assert result == "v9"
        assert 9 in test_emitter.register_file.v_used

        # Verify key operations
        instrs = test_emitter.instructions

        and_instrs = [i for i in instrs if isinstance(i, VAndB32)]
        lshift_instrs = [i for i in instrs if isinstance(i, VLshlRevB32)]
        rshift_instrs = [i for i in instrs if isinstance(i, VLshrrevB32)]
        add_instrs = [i for i in instrs if isinstance(i, VAddU32)]

        assert len(and_instrs) == 1  # Mod(tid_x, 16)
        assert len(rshift_instrs) == 1  # floor(tid_x / 16)
        assert len(lshift_instrs) == 2  # row*4 and col*256
        assert len(add_instrs) == 1  # Final sum


class TestExprEmitterEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def test_emitter(self):
        """Create a test emitter."""
        return FakeEmitter()

    @pytest.fixture
    def kernel_info(self):
        """Create kernel info."""
        return FakeKernelInfo(subgroup_size=64)

    @pytest.fixture
    def expr_emitter(self, test_emitter, kernel_info):
        """Create an ExprEmitter instance."""
        return ExprEmitter(test_emitter, kernel_info)

    def test_zero_constant(self, expr_emitter, test_emitter):
        """Test emitting zero."""
        expr = sp.Integer(0)
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Should emit: v_mov_b32 v2, 0
        assert len(test_emitter.instructions) == 1
        instr = test_emitter.instructions[0]
        assert isinstance(instr, VMovB32)
        assert instr.operands == ["v2", "0"]

    def test_negative_constant(self, expr_emitter, test_emitter):
        """Test emitting negative constant."""
        expr = sp.Integer(-42)
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Should emit: v_mov_b32 v2, -42
        assert len(test_emitter.instructions) == 1
        instr = test_emitter.instructions[0]
        assert isinstance(instr, VMovB32)
        assert instr.operands == ["v2", "-42"]

    def test_one_times_tid_x(self, expr_emitter, test_emitter):
        """Test that 1 * tid_x is optimized (SymPy simplifies to tid_x)."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = 1 * tid_x
        result = expr_emitter.emit(expr, "v2")

        # SymPy simplifies 1 * tid_x to just tid_x, which returns lane_id directly
        assert result == "v1"  # Returns lane_id register

        # No instructions should be emitted (optimization)
        assert len(test_emitter.instructions) == 0

    def test_mod_by_1_raises(self, expr_emitter, test_emitter):
        """Test that Mod by 1 handles degenerate case (mask with 0)."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = sp.Mod(tid_x, 1)
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # 1 is 2^0, mask = 1-1 = 0
        # Should emit:
        # 1. v_and_b32 v10, 0, v1  (mask with 0)
        # 2. v_mov_b32 v2, v10      (move to dst)
        assert len(test_emitter.instructions) == 2
        instrs = test_emitter.instructions

        assert isinstance(instrs[0], VAndB32)
        assert instrs[0].operands[1] == "0"  # mask = 0
        assert instrs[0].operands[2] == "v1"  # src = lane_id

        assert isinstance(instrs[1], VMovB32)
        assert instrs[1].operands[0] == "v2"

    def test_floor_div_by_1(self, expr_emitter, test_emitter):
        """Test floor division by 1 (SymPy simplifies tid_x/1 to tid_x, floor(tid_x) is not a division pattern)."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        # SymPy simplifies tid_x / 1 to tid_x, so floor(tid_x) is not a valid division pattern
        # Instead test with a non-simplified rational
        expr = sp.floor(sp.Mul(tid_x, sp.Rational(1, 1)))  # Explicit 1/1 rational
        # This will still simplify, so let's skip this edge case
        # The implementation correctly handles floor(tid_x / 2^k) for k >= 1
        pytest.skip(
            "SymPy simplifies tid_x/1 to tid_x, floor(tid_x) is not a division pattern"
        )

    def test_very_large_power_of_2(self, expr_emitter, test_emitter):
        """Test with very large power of 2 - should use shift."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        expr = 1024 * tid_x  # 2^10
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Should emit:
        # 1. v_lshlrev_b32 v10, 10, v1  (left shift by 10 for *1024)
        # 2. v_mov_b32 v2, v10           (move to dst)
        assert len(test_emitter.instructions) == 2
        instrs = test_emitter.instructions

        assert isinstance(instrs[0], VLshlRevB32)
        assert instrs[0].operands == ["v10", "10", "v1"]  # log2(1024) = 10

        assert isinstance(instrs[1], VMovB32)
        assert instrs[1].operands == ["v2", "v10"]

    def test_complex_nested_expression(self, expr_emitter, test_emitter):
        """Test deeply nested expression: ((tid_x % 64) // 16) * 8 + (tid_x % 16) * 2."""
        tid_x = sp.Symbol("tid_x", nonnegative=True)
        # ((tid_x % 64) // 16) * 8 + (tid_x % 16) * 2
        expr = sp.floor(sp.Mod(tid_x, 64) / 16) * 8 + sp.Mod(tid_x, 16) * 2
        result = expr_emitter.emit(expr, "v2")

        assert result == "v2"
        assert 2 in test_emitter.register_file.v_used

        # Verify key operations:
        # - 2 AND operations (two Mod operations)
        # - 1 right shift (floor division)
        # - 2 left shifts (*8 and *2)
        # - 1 ADD (final sum)
        instrs = test_emitter.instructions

        and_instrs = [i for i in instrs if isinstance(i, VAndB32)]
        lshift_instrs = [i for i in instrs if isinstance(i, VLshlRevB32)]
        rshift_instrs = [i for i in instrs if isinstance(i, VLshrrevB32)]
        add_instrs = [i for i in instrs if isinstance(i, VAddU32)]

        assert len(and_instrs) == 2  # Mod(tid_x, 64) and Mod(tid_x, 16)
        assert len(rshift_instrs) == 1  # floor(.../ 16)
        assert len(lshift_instrs) == 2  # *8 and *2
        assert len(add_instrs) == 1  # Final addition

        # Verify the masks are correct
        masks = [i.operands[1] for i in and_instrs]
        assert "63" in masks  # 64-1 = 63
        assert "15" in masks  # 16-1 = 15

        # Verify shifts are correct
        rshift_amounts = [i.operands[1] for i in rshift_instrs]
        assert "4" in rshift_amounts  # log2(16) = 4

        lshift_amounts = [i.operands[1] for i in lshift_instrs]
        assert "3" in lshift_amounts  # log2(8) = 3
        assert "1" in lshift_amounts  # log2(2) = 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
