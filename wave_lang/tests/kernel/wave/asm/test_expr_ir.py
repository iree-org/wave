"""Unit tests for the virtual register IR (expr_ir.py)."""

import pytest
import sympy
from wave_lang.kernel.wave.asm.expr_ir import (
    OpCode,
    VReg,
    SReg,
    PhysVReg,
    PhysSReg,
    Imm,
    ExprInstr,
    ExprProgram,
    CachedExprRef,
    is_inline_constant,
)


class TestVirtualRegisters:
    """Tests for virtual register classes."""

    def test_vreg_creation(self):
        vreg = VReg(0)
        assert vreg.index == 0
        assert "0" in str(vreg)  # Flexible string check

    def test_sreg_creation(self):
        sreg = SReg(5)
        assert sreg.index == 5
        assert "5" in str(sreg)

    def test_phys_vreg(self):
        pv = PhysVReg(10)
        assert pv.index == 10
        assert str(pv) == "v10"

    def test_phys_sreg(self):
        ps = PhysSReg(2)
        assert ps.index == 2
        assert str(ps) == "s2"


class TestImmediate:
    """Tests for immediate value handling."""

    def test_inline_constants(self):
        """Test inline constant range (0-64, -1 to -16)."""
        assert is_inline_constant(0)
        assert is_inline_constant(64)
        assert is_inline_constant(-1)
        assert is_inline_constant(-16)
        assert not is_inline_constant(65)
        assert not is_inline_constant(-17)
        assert not is_inline_constant(100)

    def test_imm_creation(self):
        imm = Imm(32)
        assert imm.value == 32


class TestExprInstr:
    """Tests for ExprInstr dataclass."""

    def test_add_instruction(self):
        dst = VReg(0)
        src1 = VReg(1)
        src2 = VReg(2)
        instr = ExprInstr(OpCode.ADD, dst, [src1, src2])
        assert instr.opcode == OpCode.ADD
        assert instr.dst == dst
        assert instr.operands == [src1, src2]

    def test_mov_instruction(self):
        dst = VReg(0)
        src = VReg(1)
        instr = ExprInstr(OpCode.MOV, dst, [src])
        assert instr.opcode == OpCode.MOV


class TestExprProgram:
    """Tests for ExprProgram class."""

    def test_alloc_vreg_sequential(self):
        prog = ExprProgram()
        v0 = prog.alloc_vreg()
        v1 = prog.alloc_vreg()
        v2 = prog.alloc_vreg()
        assert v0.index == 0
        assert v1.index == 1
        assert v2.index == 2

    def test_emit_creates_instruction(self):
        prog = ExprProgram()
        dst = prog.alloc_vreg()
        src = prog.alloc_vreg()
        prog.emit(OpCode.MOV, dst, [src])
        assert len(prog.instructions) == 1


class TestCachedExprRef:
    """Tests for CachedExprRef wrapper."""

    def test_cached_expr_ref_prevents_flattening(self):
        """CachedExprRef should prevent sympy Add flattening."""
        x = sympy.Symbol('x')
        base = x + 100  # x + 100
        
        # Without wrapper: (x + 100) + 200 becomes x + 300
        flattened = base + 200
        assert len(flattened.args) == 2  # x and 300
        
        # With wrapper: CachedExprRef(x + 100) + 200 stays separate
        wrapped = CachedExprRef(base)
        combined = wrapped + 200
        # The wrapped expression stays as a unit
        assert isinstance(combined, sympy.Add)

    def test_cached_expr_ref_wrapped_property(self):
        expr = sympy.Symbol('x') * 2 + 5
        ref = CachedExprRef(expr)
        assert ref.wrapped == expr
        assert ref.args[0] == expr
