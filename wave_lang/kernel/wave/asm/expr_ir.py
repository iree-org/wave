# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Expression IR for AMDGCN assembly generation.

This module defines a simple SSA-style intermediate representation for
arithmetic expressions. The key insight is that using virtual registers
(VReg/SReg) decouples the order of computation from physical register 
allocation, enabling:

1. Global subexpression CSE without allocation order dependencies
2. Deterministic register allocation via a separate pass
3. Better instruction scheduling opportunities

The IR consists of:
- VReg: Virtual vector register (placeholder for physical vN)
- SReg: Virtual scalar register (placeholder for physical sN)
- Imm: Immediate constant value
- PhysVReg: A physical VGPR (after allocation)
- PhysSReg: A physical SGPR (after allocation)
- ExprInstr: A single instruction operating on virtual registers
- ExprProgram: A sequence of instructions with CSE cache

Example flow:
    1. ExprEmitter emits to ExprProgram using VReg/SReg
    2. CSE pass eliminates duplicate subexpressions 
    3. Linear scan allocator assigns VReg -> physical vN
    4. Final code generation emits physical instructions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum, auto
import sympy


class CachedExprRef(sympy.Expr):
    """
    A wrapper for sympy expressions that prevents flattening when combined with other terms.
    
    When you add a constant to a sympy.Add expression like `base_expr + 2048`, sympy flattens
    it so that `base_expr` is no longer a single argument - instead, the args of `base_expr`
    get merged with the new constant. This breaks CSE because the emitter can't recognize
    that `base_expr` was already computed.
    
    CachedExprRef solves this by:
    1. Wrapping a sympy expression in a special Function
    2. When the emitter sees CachedExprRef, it looks up the wrapped expression in the cache
    3. The wrapped expression is preserved as a single unit in Add/Mul operations
    
    Example:
        >>> base = 136*Mod(tid_x, 16) + 8*floor(Mod(tid_x, 64)/16)
        >>> wrapped = CachedExprRef(base)
        >>> expr = wrapped + 2048
        >>> expr.args  # [CachedExprRef(base), 2048] - base is preserved!
    """
    
    def __new__(cls, wrapped_expr):
        """Create a new CachedExprRef wrapping the given expression."""
        obj = sympy.Expr.__new__(cls, wrapped_expr)
        return obj
    
    @property
    def wrapped(self):
        """Get the wrapped sympy expression."""
        return self.args[0]
    
    def _sympystr(self, printer):
        """String representation for debugging."""
        return f"CachedExprRef({self.wrapped})"
    
    def _eval_is_real(self):
        """Propagate realness from wrapped expression."""
        return self.wrapped.is_real
    
    def _eval_is_positive(self):
        """Propagate positivity from wrapped expression."""
        return self.wrapped.is_positive


def expr_key(expr: sympy.Expr):
    """
    Build a structural, hashable key for an expression.
    
    Uses a direct structural walk instead of canonicalization to avoid
    sympy.simplify which can incorrectly simplify floor expressions.
    The key is commutative-aware (Add and Mul args are sorted).
    """
    def to_key(e):
        if e is None:
            return ("none",)
        # Handle CachedExprRef - key is based on the wrapped expression
        if isinstance(e, CachedExprRef):
            return ("cached_ref", to_key(e.wrapped))
        if isinstance(e, sympy.Integer):
            return ("int", int(e))
        if isinstance(e, sympy.Rational) and not isinstance(e, sympy.Integer):
            # Rationals like 1/64 - preserve structure
            return ("rat", int(e.p), int(e.q))
        if isinstance(e, sympy.Symbol):
            return ("sym", str(e))
        if isinstance(e, sympy.Add):
            # Sort args by their string keys for commutativity
            arg_keys = tuple(sorted([to_key(a) for a in e.args], key=str))
            return ("add", arg_keys)
        if isinstance(e, sympy.Mul):
            # Sort args by their string keys for commutativity
            arg_keys = tuple(sorted([to_key(a) for a in e.args], key=str))
            return ("mul", arg_keys)
        if isinstance(e, sympy.Mod):
            return ("mod", to_key(e.args[0]), to_key(e.args[1]))
        if getattr(e, "func", None) == sympy.floor:
            return ("floor", to_key(e.args[0]))
        if isinstance(e, sympy.Pow):
            return ("pow", to_key(e.args[0]), to_key(e.args[1]))
        # Generic fallback - use srepr for structural representation
        return ("raw", sympy.srepr(e))

    return to_key(expr)


class OpCode(Enum):
    """Opcodes for expression IR instructions."""
    # Register moves
    MOV = auto()           # dst = src
    MOV_IMM = auto()       # dst = immediate
    
    # Arithmetic
    ADD = auto()           # dst = src1 + src2
    ADD_IMM = auto()       # dst = src + immediate
    MUL = auto()           # dst = src1 * src2
    MUL_IMM = auto()       # dst = src * immediate
    
    # Shifts
    LSHL = auto()          # dst = src << shift_amt
    LSHR = auto()          # dst = src >> shift_amt
    
    # Bitwise
    AND = auto()           # dst = src1 & src2
    AND_IMM = auto()       # dst = src & immediate
    OR = auto()            # dst = src1 | src2
    BFE = auto()           # dst = (src >> offset) & mask
    
    # Fused
    LSHL_ADD = auto()      # dst = (src1 << shift) + src2
    LSHL_OR = auto()       # dst = (src1 << shift) | src2
    
    # SGPR -> VGPR
    READFIRSTLANE = auto() # sgpr_dst = vgpr_src (uniform)
    BROADCAST = auto()     # vgpr_dst = sgpr_src (replicate)


# ============================================================================
# AMDGCN Immediate Encoding Constraints
# ============================================================================
#
# AMDGCN instructions have specific rules for immediate operands:
#
# 1. VOP2 instructions (v_add_u32, v_and_b32, v_or_b32, v_mul_lo_u32):
#    - Can have ONE 32-bit literal constant as src0
#    - src1 must be a VGPR
#
# 2. VOP3 instructions (v_lshl_add_u32, v_mad_u32_u24, etc.):
#    - Can have ONE literal constant total across all operands
#    - The shift amount in v_lshl_add_u32 uses the literal slot if not inline
#
# 3. Inline constants (encoded in instruction, no literal slot needed):
#    - Integers: 0-64, -1 to -16
#    - Special: 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0
#
# This means: if an instruction already has a literal, adding another 
# immediate requires materializing that constant into a VGPR first.

def is_inline_constant(value: int) -> bool:
    """
    Check if an integer value can be encoded as an inline constant.
    
    Inline constants don't consume the literal slot, so they're "free"
    in terms of instruction encoding.
    
    Returns True for: 0-64, -1 to -16
    """
    if isinstance(value, int):
        return (0 <= value <= 64) or (-16 <= value <= -1)
    return False


def get_immediate_info(opcode: "OpCode") -> dict:
    """
    Get information about immediate operand support for an opcode.
    
    Returns a dict with:
    - 'can_use_literal': True if this instruction can use a 32-bit literal
    - 'literal_operand_idx': Which operand index can be a literal (or None)
    - 'max_literals': Maximum number of literal constants (usually 0 or 1)
    - 'notes': Human-readable description
    """
    # Instructions that can take a literal as src0 (VOP2-style)
    vop2_literal_src0 = {
        OpCode.ADD_IMM: {'can_use_literal': True, 'literal_operand_idx': 1, 'max_literals': 1,
                        'notes': 'v_add_u32 can use literal as src0'},
        OpCode.AND_IMM: {'can_use_literal': True, 'literal_operand_idx': 1, 'max_literals': 1,
                        'notes': 'v_and_b32 can use literal as src0'},
        OpCode.MOV_IMM: {'can_use_literal': True, 'literal_operand_idx': 0, 'max_literals': 1,
                        'notes': 'v_mov_b32 can use literal'},
    }
    
    # Instructions where the immediate is typically small (shift amounts)
    # These usually fit in inline constants
    shift_ops = {
        OpCode.LSHL: {'can_use_literal': True, 'literal_operand_idx': 1, 'max_literals': 1,
                     'notes': 'shift amount usually inline (0-31)'},
        OpCode.LSHR: {'can_use_literal': True, 'literal_operand_idx': 1, 'max_literals': 1,
                     'notes': 'shift amount usually inline (0-31)'},
        OpCode.BFE: {'can_use_literal': True, 'literal_operand_idx': None, 'max_literals': 1,
                   'notes': 'offset/width typically inline'},
    }
    
    # Instructions that require materializing constants (no literal support)
    no_literal = {
        OpCode.MUL_IMM: {'can_use_literal': False, 'literal_operand_idx': None, 'max_literals': 0,
                        'notes': 'v_mul_lo_u32 requires VGPR operands'},
        OpCode.MUL: {'can_use_literal': False, 'literal_operand_idx': None, 'max_literals': 0,
                   'notes': 'v_mul_lo_u32 requires VGPR operands'},
    }
    
    # Fused ops - can use ONE literal total
    fused_ops = {
        OpCode.LSHL_ADD: {'can_use_literal': True, 'literal_operand_idx': 1, 'max_literals': 1,
                         'notes': 'v_lshl_add_u32 shift often inline; can use 1 literal'},
        OpCode.LSHL_OR: {'can_use_literal': True, 'literal_operand_idx': 1, 'max_literals': 1,
                        'notes': 'v_lshl_or_b32 shift often inline; can use 1 literal'},
    }
    
    # Combine all
    all_info = {**vop2_literal_src0, **shift_ops, **no_literal, **fused_ops}
    
    return all_info.get(opcode, {
        'can_use_literal': False,
        'literal_operand_idx': None,
        'max_literals': 0,
        'notes': 'Unknown opcode - assume no literal support'
    })


@dataclass(frozen=True, eq=True)
class VReg:
    """
    Virtual vector register.
    
    Represents a placeholder for a physical VGPR that will be assigned
    during register allocation. The index is unique within an ExprProgram.
    """
    index: int
    
    def __str__(self) -> str:
        return f"vr{self.index}"
    
    def __hash__(self) -> int:
        return hash(("VReg", self.index))


@dataclass(frozen=True, eq=True)
class SReg:
    """
    Virtual scalar register.
    
    Represents a placeholder for a physical SGPR that will be assigned
    during register allocation.
    """
    index: int
    
    def __str__(self) -> str:
        return f"sr{self.index}"
    
    def __hash__(self) -> int:
        return hash(("SReg", self.index))


@dataclass(frozen=True, eq=True)
class PhysVReg:
    """
    Physical vector register.
    
    Represents an actual VGPR (v0, v1, ..., v255).
    Used after register allocation or for ABI-mandated registers.
    """
    index: int
    
    def __str__(self) -> str:
        return f"v{self.index}"
    
    def __hash__(self) -> int:
        return hash(("PhysVReg", self.index))


@dataclass(frozen=True, eq=True)
class PhysSReg:
    """
    Physical scalar register.
    
    Represents an actual SGPR (s0, s1, ..., s103).
    Used after register allocation or for ABI-mandated registers.
    """
    index: int
    
    def __str__(self) -> str:
        return f"s{self.index}"
    
    def __hash__(self) -> int:
        return hash(("PhysSReg", self.index))


@dataclass(frozen=True, eq=True)
class Imm:
    """
    Immediate constant value.
    
    Represents a literal integer that can be used as an instruction operand.
    Some instructions can encode small immediates directly; larger values
    may need to be materialized into a register.
    """
    value: int
    
    def __str__(self) -> str:
        if self.value >= 0x100:
            return f"0x{self.value:x}"
        return str(self.value)
    
    def __hash__(self) -> int:
        return hash(("Imm", self.value))


# Type aliases for operands
Operand = Union[VReg, SReg, PhysVReg, PhysSReg, Imm]
VirtualReg = Union[VReg, SReg]
PhysicalReg = Union[PhysVReg, PhysSReg]
AnyReg = Union[VReg, SReg, PhysVReg, PhysSReg]


@dataclass
class ExprInstr:
    """
    A single instruction in the expression IR.
    
    Each instruction has:
    - opcode: The operation to perform
    - dst: The destination register (VReg or SReg)
    - operands: Source operands (can be VReg, SReg, PhysVReg, PhysSReg, or Imm)
    - comment: Optional comment for debugging
    
    After register allocation, dst and operands are replaced with physical registers.
    """
    opcode: OpCode
    dst: AnyReg
    operands: Tuple[Operand, ...]
    comment: str = ""
    
    # Source SymPy expression that generated this instruction (for CSE)
    source_expr: Optional[sympy.Expr] = None
    
    def __str__(self) -> str:
        ops_str = ", ".join(str(op) for op in self.operands)
        base = f"{self.opcode.name.lower()} {self.dst}, {ops_str}"
        if self.comment:
            return f"{base}  // {self.comment}"
        return base
    
    def get_uses(self) -> List[AnyReg]:
        """Get all registers used (read) by this instruction."""
        return [op for op in self.operands if isinstance(op, (VReg, SReg, PhysVReg, PhysSReg))]
    
    def get_def(self) -> AnyReg:
        """Get the register defined (written) by this instruction."""
        return self.dst
    
    def replace_operands(self, mapping: Dict[VirtualReg, PhysicalReg]) -> "ExprInstr":
        """
        Replace virtual registers with physical registers using the given mapping.
        
        Returns a new ExprInstr with replaced operands.
        """
        new_dst = mapping.get(self.dst, self.dst) if isinstance(self.dst, (VReg, SReg)) else self.dst
        new_ops = tuple(
            mapping.get(op, op) if isinstance(op, (VReg, SReg)) else op
            for op in self.operands
        )
        return ExprInstr(
            opcode=self.opcode,
            dst=new_dst,
            operands=new_ops,
            comment=self.comment,
            source_expr=self.source_expr,
        )


@dataclass
class ExprProgram:
    """
    A sequence of expression IR instructions with CSE support.
    
    This is the main container for expression IR. It maintains:
    - instructions: Ordered list of instructions
    - _next_vreg: Counter for allocating new VRegs
    - _next_sreg: Counter for allocating new SRegs
    - _expr_cache: Maps SymPy expressions to their result registers (for CSE)
    - _cse_enabled: Whether CSE is active
    
    Usage:
        prog = ExprProgram()
        v0 = prog.alloc_vreg()
        v1 = prog.alloc_vreg()
        prog.emit(OpCode.ADD, v1, (v0, Imm(1)))
    """
    instructions: List[ExprInstr] = field(default_factory=list)
    
    # Virtual register counters
    _next_vreg: int = 0
    _next_sreg: int = 0
    
    # CSE cache: maps expr_key -> result register
    _expr_cache: Dict[tuple, VirtualReg] = field(default_factory=dict)
    
    # Enable/disable CSE
    _cse_enabled: bool = True
    
    # Debug: log all CSE hits
    _cse_log: List[str] = field(default_factory=list)
    
    def alloc_vreg(self) -> VReg:
        """Allocate a new virtual VGPR."""
        vreg = VReg(self._next_vreg)
        self._next_vreg += 1
        return vreg
    
    def alloc_sreg(self) -> SReg:
        """Allocate a new virtual SGPR."""
        sreg = SReg(self._next_sreg)
        self._next_sreg += 1
        return sreg
    
    def emit(
        self,
        opcode: OpCode,
        dst: AnyReg,
        operands: Tuple[Operand, ...],
        comment: str = "",
        source_expr: Optional[sympy.Expr] = None,
    ) -> None:
        """
        Emit an instruction to the program.
        
        Args:
            opcode: The operation to perform
            dst: Destination register
            operands: Source operands
            comment: Optional comment
            source_expr: The SymPy expression that generated this (for CSE tracking)
        """
        instr = ExprInstr(
            opcode=opcode,
            dst=dst,
            operands=operands,
            comment=comment,
            source_expr=source_expr,
        )
        self.instructions.append(instr)
    
    def cache_expr(self, expr_key: tuple, result_reg: VirtualReg) -> None:
        """
        Cache an expression result for CSE.
        
        Args:
            expr_key: Canonical key for the expression (from expr_key())
            result_reg: The virtual register holding the result
        """
        if self._cse_enabled:
            self._expr_cache[expr_key] = result_reg
            self._cse_log.append(f"CACHE: {expr_key} -> {result_reg}")
    
    def get_cached_expr(self, expr_key: tuple) -> Optional[VirtualReg]:
        """
        Look up a cached expression result.
        
        Args:
            expr_key: Canonical key for the expression
            
        Returns:
            The cached result register, or None if not cached
        """
        if not self._cse_enabled:
            return None
        result = self._expr_cache.get(expr_key)
        if result is not None:
            self._cse_log.append(f"HIT: {expr_key} -> {result}")
        return result
    
    def clear_cache(self) -> None:
        """Clear the CSE cache."""
        self._expr_cache.clear()
        self._cse_log.clear()
    
    def get_cse_stats(self) -> Dict[str, int]:
        """Get CSE statistics."""
        hits = sum(1 for log in self._cse_log if log.startswith("HIT:"))
        caches = sum(1 for log in self._cse_log if log.startswith("CACHE:"))
        return {
            "cache_entries": len(self._expr_cache),
            "cache_hits": hits,
            "expressions_cached": caches,
        }
    
    def dump(self) -> str:
        """Dump the program as a string for debugging."""
        lines = [
            "=" * 60,
            "ExprProgram",
            f"  VRegs allocated: {self._next_vreg}",
            f"  SRegs allocated: {self._next_sreg}",
            f"  CSE cache entries: {len(self._expr_cache)}",
            "-" * 60,
        ]
        for i, instr in enumerate(self.instructions):
            lines.append(f"  {i:3d}: {instr}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def get_vreg_count(self) -> int:
        """Get the total number of VRegs allocated."""
        return self._next_vreg
    
    def get_sreg_count(self) -> int:
        """Get the total number of SRegs allocated."""
        return self._next_sreg
    
    def get_live_ranges(self) -> Dict[VirtualReg, Tuple[int, int]]:
        """
        Compute live ranges for all virtual registers.
        
        Returns:
            Dict mapping each virtual register to (first_def, last_use) instruction indices
        """
        first_def: Dict[VirtualReg, int] = {}
        last_use: Dict[VirtualReg, int] = {}
        
        for i, instr in enumerate(self.instructions):
            # Record definition
            dst = instr.get_def()
            if isinstance(dst, (VReg, SReg)):
                if dst not in first_def:
                    first_def[dst] = i
            
            # Record uses
            for use in instr.get_uses():
                if isinstance(use, (VReg, SReg)):
                    last_use[use] = i
        
        # Combine into ranges
        ranges: Dict[VirtualReg, Tuple[int, int]] = {}
        all_regs = set(first_def.keys()) | set(last_use.keys())
        for reg in all_regs:
            start = first_def.get(reg, 0)
            end = last_use.get(reg, start)
            ranges[reg] = (start, end)
        
        return ranges


def emit_mov_imm(prog: ExprProgram, value: int, comment: str = "") -> VReg:
    """Helper: emit a MOV_IMM and return the destination VReg."""
    dst = prog.alloc_vreg()
    prog.emit(OpCode.MOV_IMM, dst, (Imm(value),), comment)
    return dst


def emit_mov(prog: ExprProgram, src: Operand, comment: str = "") -> VReg:
    """Helper: emit a MOV and return the destination VReg."""
    dst = prog.alloc_vreg()
    prog.emit(OpCode.MOV, dst, (src,), comment)
    return dst


def emit_add(prog: ExprProgram, src1: Operand, src2: Operand, comment: str = "") -> VReg:
    """Helper: emit an ADD and return the destination VReg."""
    dst = prog.alloc_vreg()
    if isinstance(src2, Imm):
        prog.emit(OpCode.ADD_IMM, dst, (src1, src2), comment)
    else:
        prog.emit(OpCode.ADD, dst, (src1, src2), comment)
    return dst


def emit_lshl(prog: ExprProgram, src: Operand, shift: int, comment: str = "") -> VReg:
    """Helper: emit a LSHL and return the destination VReg."""
    dst = prog.alloc_vreg()
    prog.emit(OpCode.LSHL, dst, (src, Imm(shift)), comment)
    return dst


def emit_lshr(prog: ExprProgram, src: Operand, shift: int, comment: str = "") -> VReg:
    """Helper: emit a LSHR and return the destination VReg."""
    dst = prog.alloc_vreg()
    prog.emit(OpCode.LSHR, dst, (src, Imm(shift)), comment)
    return dst


def emit_and(prog: ExprProgram, src: Operand, mask: int, comment: str = "") -> VReg:
    """Helper: emit an AND and return the destination VReg."""
    dst = prog.alloc_vreg()
    prog.emit(OpCode.AND_IMM, dst, (src, Imm(mask)), comment)
    return dst


def emit_bfe(prog: ExprProgram, src: Operand, offset: int, width: int, comment: str = "") -> VReg:
    """Helper: emit a BFE (bit field extract) and return the destination VReg."""
    dst = prog.alloc_vreg()
    prog.emit(OpCode.BFE, dst, (src, Imm(offset), Imm(width)), comment)
    return dst

