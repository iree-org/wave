# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Kernel-level expression emitter using whole-program register allocation.

This module provides KernelEmitter, which uses a single kernel-wide program
and allocator for all expression emission. Key benefits:

1. Global CSE across all expressions in the kernel
2. Better register reuse through kernel-wide allocation
3. Reduced register pressure through kernel-wide optimization

Algebraic simplification is enabled by default (WAVE_EXPR_SIMPLIFY=0 to disable).
"""

import os
import sympy
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Set, Union

from .expr_simplify import simplify_for_emission, SimplifyStats


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


# Debug flags
DEBUG_KERNEL_EMITTER = os.environ.get("WAVE_KERNEL_EMITTER_DEBUG", "0") == "1"
DEBUG_CSE = os.environ.get("WAVE_KERNEL_CSE_DEBUG", "0") == "1"

# Enable algebraic simplification (default: enabled)
# The simplification transforms patterns like 16*tid_x - 64*floor(tid_x/4) to 16*Mod(tid_x, 4)
# which reduces instruction count. Use WAVE_EXPR_SIMPLIFY=0 to disable for debugging.
ENABLE_SIMPLIFY = os.environ.get("WAVE_EXPR_SIMPLIFY", "1") == "1"

# Marker for rational values (expr / const)
_RationalReg = namedtuple("_RationalReg", ["numerator_vreg", "denominator"])


class KernelEmitter:
    """
    Kernel-level expression emitter with whole-program register allocation.
    
    Uses kernel-wide virtual registers and a single allocator for better
    register reuse. Immediately allocates and emits instructions (streaming mode)
    but tracks liveness kernel-wide for optimal reuse.
    """
    
    def __init__(self, asm_emitter, kernel_info):
        """Initialize the kernel emitter."""
        self.asm_emitter = asm_emitter
        self.kernel_info = kernel_info
        
        # Symbol setup
        self.tid_x_symbol = sympy.Symbol("tid_x", nonnegative=True)
        self.tid_y_symbol = sympy.Symbol("tid_y", nonnegative=True)
        self.tid_z_symbol = sympy.Symbol("tid_z", nonnegative=True)
        
        # Symbol bindings: SymPy symbol -> physical register string
        self.symbol_bindings: Dict[sympy.Symbol, str] = {}
        
        # CSE cache: expr_key -> physical register string
        self._cache: Dict[tuple, str] = {}
        
        # Pinned output registers
        self._pinned_outputs: Set[int] = set()
        
        # ABI registers (reserved)
        self._abi_vgprs: Set[int] = set()
        self._abi_sgprs: Set[int] = set()
        
        # Cached tid extractions
        self._tid_x_phys: Optional[str] = None
        self._tid_y_phys: Optional[str] = None
        
        # Constant cache
        self._const_cache: Dict[int, int] = {}
        self._const_cache_pinned: Set[int] = set()
        
        # Statistics
        self._cse_hits = 0
        self._cse_misses = 0
        self._subexpr_cse_hits = 0
        self._simplify_stats = SimplifyStats()
        
        # Initialize
        self._bind_abi_registers()
        self._init_pools()
    
    def _emit(self, line: str) -> None:
        """Emit an assembly line."""
        self.asm_emitter.emit(line)
    
    def _init_pools(self):
        """Initialize register allocation state."""
        # Start after already-used registers
        initial_reserved = self._get_reserved_vgprs()
        self._next_vgpr = max(initial_reserved) + 1 if initial_reserved else 1
        # Free list for reuse
        self._free_vgprs: List[int] = []
    
    def _bind_abi_registers(self):
        """Bind ABI/system registers."""
        emitter = self.asm_emitter
        
        # Workgroup ID SGPRs
        if emitter.sgpr_workgroup_id_x is not None:
            sgpr = emitter.sgpr_workgroup_id_x
            self._abi_sgprs.add(sgpr)
            wgid_x = sympy.Symbol("wgid_x", nonnegative=True)
            self.symbol_bindings[wgid_x] = f"s{sgpr}"
        
        if emitter.sgpr_workgroup_id_y is not None:
            sgpr = emitter.sgpr_workgroup_id_y
            self._abi_sgprs.add(sgpr)
            wgid_y = sympy.Symbol("wgid_y", nonnegative=True)
            self.symbol_bindings[wgid_y] = f"s{sgpr}"
        
        if emitter.sgpr_workgroup_id_z is not None:
            sgpr = emitter.sgpr_workgroup_id_z
            self._abi_sgprs.add(sgpr)
            wgid_z = sympy.Symbol("wgid_z", nonnegative=True)
            self.symbol_bindings[wgid_z] = f"s{sgpr}"
        
        # Flat workitem ID VGPR (v0)
        if emitter.special_regs.has_flat_tid():
            vgpr = emitter.special_regs.get_flat_tid_vgpr()
            self._abi_vgprs.add(vgpr)
    
    def bind_symbol(self, symbol_name: str, register: str) -> None:
        """Bind a symbol name to a register."""
        symbol = sympy.Symbol(symbol_name, nonnegative=True)
        self.symbol_bindings[symbol] = register
        
        if register.startswith("v"):
            idx = int(register[1:])
            self._abi_vgprs.add(idx)
        elif register.startswith("s"):
            idx = int(register[1:])
            self._abi_sgprs.add(idx)
    
    def _get_symbol_bounds(self) -> Dict[sympy.Symbol, Tuple[int, int]]:
        """Get bounds for symbols."""
        bounds = {}
        threads_per_wave = 64
        if hasattr(self.kernel_info, 'threads_per_wave'):
            threads_per_wave = self.kernel_info.threads_per_wave
        max_tid = threads_per_wave * 4 - 1
        
        bounds[self.tid_x_symbol] = (0, max_tid)
        bounds[self.tid_y_symbol] = (0, max_tid)
        bounds[self.tid_z_symbol] = (0, max_tid)
        
        for sym_name in ["wgid_x", "wgid_y", "wgid_z"]:
            sym = sympy.Symbol(sym_name, nonnegative=True)
            bounds[sym] = (0, 65535)
        
        return bounds
    
    def _alloc_vgpr(self) -> int:
        """Allocate a VGPR."""
        # Get all reserved registers
        reserved = self._get_reserved_vgprs()
        
        # Try free list first
        for i, v in enumerate(self._free_vgprs):
            if v not in reserved:
                self._free_vgprs.pop(i)
                self.asm_emitter.register_file.v_used.add(v)
                return v
        
        # Allocate new
        while self._next_vgpr < 256:
            v = self._next_vgpr
            self._next_vgpr += 1
            if v not in reserved:
                self.asm_emitter.register_file.v_used.add(v)
                return v
        
        raise RuntimeError("Out of VGPRs")
    
    def _get_reserved_vgprs(self) -> Set[int]:
        """Get all reserved VGPRs."""
        reserved = set()
        reserved.update(self._abi_vgprs)
        reserved.update(self.asm_emitter.register_file.v_used)
        reserved.update(self._pinned_outputs)
        reserved.update(self._const_cache_pinned)
        return reserved
    
    def get_or_emit(self, expr: sympy.Expr, dst_hint: Optional[str] = None) -> str:
        """
        Get cached register or emit expression.
        
        Returns physical register string (e.g., "v5").
        """
        # Algebraic simplification
        if ENABLE_SIMPLIFY:
            original = expr
            expr = simplify_for_emission(expr, self._get_symbol_bounds())
            if expr != original:
                if DEBUG_KERNEL_EMITTER:
                    print(f"[SIMPLIFY] {original}")
                    print(f"        -> {expr}")
                self._simplify_stats.record(original, expr)
        
        key = expr_key(expr)
        
        # Check CSE cache
        if key in self._cache:
            self._cse_hits += 1
            result = self._cache[key]
            if DEBUG_CSE:
                print(f"[CSE HIT] {expr} -> {result}")
            return result
        
        self._cse_misses += 1
        
        # Emit the expression
        result = self._emit_expr(expr)
        
        # Cache the result
        self._cache[key] = result
        
        # Pin the output register
        if result.startswith("v"):
            idx = int(result[1:])
            self._pinned_outputs.add(idx)
        
        if DEBUG_CSE:
            print(f"[CSE MISS] {expr} -> {result}")
        
        return result
    
    def _emit_expr(self, expr: sympy.Expr) -> str:
        """Emit an expression and return physical register string."""
        # Check cache for subexpression
        key = expr_key(expr)
        if key in self._cache:
            self._subexpr_cse_hits += 1
            if DEBUG_CSE:
                print(f"[SUBEXPR CSE HIT] {expr} -> {self._cache[key]}")
            return self._cache[key]
        
        # Emit based on type
        if isinstance(expr, sympy.Integer):
            result = self._emit_constant(int(expr))
        elif isinstance(expr, sympy.Symbol):
            result = self._emit_symbol(expr)
        elif isinstance(expr, sympy.Rational) and not isinstance(expr, sympy.Integer):
            result = self._emit_rational(expr)
        elif isinstance(expr, sympy.Add):
            result = self._emit_add(expr)
        elif isinstance(expr, sympy.Mul):
            result = self._emit_mul(expr)
        elif isinstance(expr, sympy.Mod):
            result = self._emit_mod(expr)
        elif getattr(expr, "func", None) == sympy.floor:
            result = self._emit_floor(expr)
        elif isinstance(expr, sympy.Pow):
            result = self._emit_pow(expr)
        else:
            raise ValueError(f"Unsupported expression: {type(expr).__name__}: {expr}")
        
        # Cache the result for subexpression CSE
        self._cache[key] = result
        
        # Pin the register
        if result.startswith("v"):
            idx = int(result[1:])
            self._pinned_outputs.add(idx)
        
        return result
    
    def _emit_constant(self, value: int) -> str:
        """Emit a constant value."""
        phys = self._alloc_vgpr()
        if -16 <= value <= 64:
            self._emit(f"    v_mov_b32 v{phys}, {value}")
        else:
            self._emit(f"    v_mov_b32 v{phys}, 0x{value & 0xffffffff:x}")
        return f"v{phys}"
    
    def _emit_symbol(self, sym: sympy.Symbol) -> str:
        """Emit a symbol reference."""
        # Check bindings
        if sym in self.symbol_bindings:
            bound_reg = self.symbol_bindings[sym]
            
            # Multi-wave tid_x/tid_y extraction
            if self._is_multi_wave():
                if sym == self.tid_x_symbol:
                    return self._emit_tid_x()
                elif sym == self.tid_y_symbol:
                    return self._emit_tid_y()
            
            return bound_reg
        
        # Handle tid_x/tid_y
        if sym == self.tid_x_symbol:
            if self._is_multi_wave():
                return self._emit_tid_x()
            else:
                return self._emit_lane_id()
        
        if sym == self.tid_y_symbol:
            if self._is_multi_wave():
                return self._emit_tid_y()
            else:
                return self._emit_constant(0)
        
        if sym == self.tid_z_symbol:
            raise ValueError("tid_z not yet supported")
        
        # Direct SGPR reference
        sym_str = str(sym)
        if sym_str.startswith("s") and sym_str[1:].isdigit():
            return sym_str
        
        raise ValueError(f"Unknown symbol: {sym}")
    
    def _emit_tid_x(self) -> str:
        """Emit tid_x extraction from flat_tid."""
        if self._tid_x_phys is not None:
            return self._tid_x_phys
        
        flat_tid = self._get_flat_tid_vgpr()
        phys = self._alloc_vgpr()
        
        # v_bfe_u32 extracts bits [offset, offset+width)
        self._emit(f"    v_bfe_u32 v{phys}, v{flat_tid}, 0, 10")
        
        self._tid_x_phys = f"v{phys}"
        return self._tid_x_phys
    
    def _emit_tid_y(self) -> str:
        """Emit tid_y extraction from flat_tid."""
        if self._tid_y_phys is not None:
            return self._tid_y_phys
        
        flat_tid = self._get_flat_tid_vgpr()
        phys = self._alloc_vgpr()
        
        self._emit(f"    v_bfe_u32 v{phys}, v{flat_tid}, 10, 10")
        
        self._tid_y_phys = f"v{phys}"
        return self._tid_y_phys
    
    def _emit_lane_id(self) -> str:
        """Emit lane_id for single-wave mode."""
        lane_id_v = self.asm_emitter.ensure_lane_id(self.kernel_info.subgroup_size)
        return f"v{lane_id_v}"
    
    def _emit_rational(self, expr: sympy.Rational) -> str:
        """Handle rational (p/q)."""
        # Rationals like 1/64 can appear in Add expressions
        # For integer arithmetic, floor(p/q) is the result
        result = int(expr.p) // int(expr.q)
        return self._emit_constant(result)
    
    def _emit_add(self, expr: sympy.Add) -> str:
        """Emit an addition."""
        args = list(expr.args)
        
        # Collect constants and register operands
        const_sum = 0
        reg_args = []
        
        for arg in args:
            if isinstance(arg, sympy.Integer):
                const_sum += int(arg)
            else:
                reg = self._emit_expr(arg)
                reg_args.append(reg)
        
        if not reg_args:
            return self._emit_constant(const_sum)
        
        # Start with first register
        result = reg_args[0]
        
        # Add remaining registers
        for reg in reg_args[1:]:
            new_phys = self._alloc_vgpr()
            self._emit(f"    v_add_u32 v{new_phys}, {result}, {reg}")
            result = f"v{new_phys}"
        
        # Add constant if non-zero
        if const_sum != 0:
            new_phys = self._alloc_vgpr()
            if -16 <= const_sum <= 64:
                self._emit(f"    v_add_u32 v{new_phys}, {result}, {const_sum}")
            else:
                # Need to materialize constant first
                const_reg = self._get_or_materialize_const(const_sum)
                self._emit(f"    v_add_u32 v{new_phys}, {result}, v{const_reg}")
            result = f"v{new_phys}"
        
        return result
    
    def _emit_mul(self, expr: sympy.Mul) -> str:
        """Emit a multiplication."""
        args = list(expr.args)
        
        const_product = 1
        divisor = None
        reg_args = []
        
        for arg in args:
            if isinstance(arg, sympy.Integer):
                const_product *= int(arg)
            elif isinstance(arg, sympy.Rational) and not isinstance(arg, sympy.Integer):
                # Rational like 1/64 indicates division
                if arg.p != 1:
                    const_product *= int(arg.p)
                divisor = int(arg.q)
            elif isinstance(arg, sympy.Pow) and arg.args[1] == -1:
                # x^(-1) indicates division
                divisor = arg.args[0]
                if isinstance(divisor, sympy.Integer):
                    divisor = int(divisor)
            else:
                reg = self._emit_expr(arg)
                reg_args.append(reg)
        
        if not reg_args:
            # Pure constant (possibly with division marker)
            if divisor is not None:
                return self._emit_constant(const_product // divisor)
            return self._emit_constant(const_product)
        
        if len(reg_args) > 1:
            raise ValueError(f"Multiple register operands in mul: {expr}")
        
        src_reg = reg_args[0]
        
        # Copy SGPR to VGPR if needed (SGPRs can't be used directly in some VOP instructions)
        if src_reg.startswith("s"):
            new_phys = self._alloc_vgpr()
            self._emit(f"    v_mov_b32 v{new_phys}, {src_reg}")
            src_reg = f"v{new_phys}"
        
        # Apply constant multiplication
        if const_product == 1:
            result = src_reg
        elif const_product > 0 and (const_product & (const_product - 1)) == 0:
            # Power of 2 - use shift
            shift = const_product.bit_length() - 1
            new_phys = self._alloc_vgpr()
            self._emit(f"    v_lshlrev_b32 v{new_phys}, {shift}, {src_reg}")
            result = f"v{new_phys}"
        elif const_product == -1:
            # Negate
            new_phys = self._alloc_vgpr()
            self._emit(f"    v_sub_u32 v{new_phys}, 0, {src_reg}")
            result = f"v{new_phys}"
        else:
            # General multiplication
            new_phys = self._alloc_vgpr()
            if -16 <= const_product <= 64:
                self._emit(f"    v_mul_lo_u32 v{new_phys}, {src_reg}, {const_product}")
            else:
                const_reg = self._get_or_materialize_const(const_product)
                self._emit(f"    v_mul_lo_u32 v{new_phys}, {src_reg}, v{const_reg}")
            result = f"v{new_phys}"
        
        # Apply division if present (floor division = right shift for power-of-2)
        if divisor is not None:
            if isinstance(divisor, int) and divisor > 0 and (divisor & (divisor - 1)) == 0:
                shift = divisor.bit_length() - 1
                if shift > 0:
                    new_phys = self._alloc_vgpr()
                    self._emit(f"    v_lshrrev_b32 v{new_phys}, {shift}, {result}")
                    result = f"v{new_phys}"
            else:
                raise ValueError(f"Non-power-of-2 division: {divisor}")
        
        return result
    
    def _emit_mod(self, expr: sympy.Mod) -> str:
        """Emit a modulo operation."""
        
        dividend_expr, divisor_expr = expr.args
        
        if not isinstance(divisor_expr, (int, sympy.Integer)):
            raise ValueError(f"Mod divisor must be constant: {divisor_expr}")
        
        divisor = int(divisor_expr)
        if divisor <= 0 or (divisor & (divisor - 1)) != 0:
            raise ValueError(f"Mod divisor must be power-of-2: {divisor}")
        
        # Handle constant dividend
        if isinstance(dividend_expr, sympy.Integer):
            return self._emit_constant(int(dividend_expr) % divisor)
        
        dividend = self._emit_expr(dividend_expr)
        
        # Power of 2: use AND with mask or BFE
        mask = divisor - 1
        new_phys = self._alloc_vgpr()
        
        if mask <= 64:
            # Inline constant in src0 position
            self._emit(f"    v_and_b32 v{new_phys}, {mask}, {dividend}")
        else:
            # Use BFE for larger masks
            width = divisor.bit_length() - 1
            self._emit(f"    v_bfe_u32 v{new_phys}, {dividend}, 0, {width}")
        
        return f"v{new_phys}"
    
    def _emit_floor(self, expr) -> str:
        """Emit a floor operation."""
        arg = expr.args[0]
        
        # floor(x/n) where n is power of 2 -> x >> log2(n)
        if isinstance(arg, sympy.Mul):
            return self._emit_floor_div(arg)
        
        # floor(Add(...)) - emit the add, floor is no-op for integers
        if isinstance(arg, sympy.Add):
            return self._emit_add(arg)
        
        # floor(constant) = constant
        if isinstance(arg, sympy.Integer):
            return self._emit_constant(int(arg))
        
        # floor(Mod(...)) - emit the mod
        if isinstance(arg, sympy.Mod):
            return self._emit_mod(arg)
        
        # floor(floor(...)) - just emit inner floor
        if getattr(arg, "func", None) == sympy.floor:
            return self._emit_floor(arg)
        
        # floor(symbol) - emit symbol (already integer)
        if isinstance(arg, sympy.Symbol):
            return self._emit_symbol(arg)
        
        raise ValueError(f"Cannot emit floor: {expr}")
    
    def _emit_floor_div(self, mul_expr: sympy.Mul) -> str:
        """Emit floor(x * 1/n) = floor(x/n)."""
        # Parse the multiplication to find x and 1/n
        x_part = None
        divisor = None
        const_coeff = 1
        
        for factor in mul_expr.args:
            if isinstance(factor, sympy.Pow) and factor.args[1] == -1:
                # x^(-1) = 1/x
                divisor = factor.args[0]
            elif isinstance(factor, sympy.Rational) and not isinstance(factor, sympy.Integer):
                # p/q
                if factor.p != 1:
                    const_coeff *= int(factor.p)
                divisor = sympy.Integer(factor.q)
            elif isinstance(factor, sympy.Integer):
                const_coeff *= int(factor)
            else:
                if x_part is None:
                    x_part = factor
                else:
                    x_part = x_part * factor
        
        if x_part is None:
            # Pure constant division
            if divisor is not None and isinstance(divisor, sympy.Integer):
                return self._emit_constant(const_coeff // int(divisor))
            return self._emit_constant(const_coeff)
        
        if divisor is None:
            # No division, just emit the expression
            return self._emit_expr(x_part * const_coeff if const_coeff != 1 else x_part)
        
        if not isinstance(divisor, sympy.Integer):
            raise ValueError(f"Divisor must be integer: {divisor}")
        
        div_val = int(divisor)
        if div_val <= 0 or (div_val & (div_val - 1)) != 0:
            raise ValueError(f"Floor divisor must be power-of-2: {div_val}")
        
        # Emit x * const_coeff first
        if const_coeff == 1:
            src = self._emit_expr(x_part)
        else:
            src = self._emit_expr(x_part * sympy.Integer(const_coeff))
        
        # Then shift right
        shift = div_val.bit_length() - 1
        if shift > 0:
            new_phys = self._alloc_vgpr()
            self._emit(f"    v_lshrrev_b32 v{new_phys}, {shift}, {src}")
            return f"v{new_phys}"
        
        return src
    
    def _emit_pow(self, expr: sympy.Pow) -> str:
        """Emit a power expression."""
        base, exp = expr.args
        
        if base == 2 and isinstance(exp, (int, sympy.Integer)):
            return self._emit_constant(2 ** int(exp))
        
        if isinstance(exp, sympy.Integer) and int(exp) == -1:
            # x^(-1) should be handled by floor_div
            raise ValueError(f"Bare inverse not supported: {expr}")
        
        raise ValueError(f"Only 2^k powers supported: {expr}")
    
    def _get_or_materialize_const(self, value: int) -> int:
        """Get or materialize a constant into a VGPR."""
        if value in self._const_cache:
            return self._const_cache[value]
        
        const_v = self._alloc_vgpr()
        if -16 <= value <= 64:
            self._emit(f"    v_mov_b32 v{const_v}, {value}")
        else:
            self._emit(f"    v_mov_b32 v{const_v}, 0x{value & 0xffffffff:x}")
        
        self._const_cache[value] = const_v
        self._const_cache_pinned.add(const_v)
        self._pinned_outputs.add(const_v)
        
        return const_v
    
    def _is_multi_wave(self) -> bool:
        """Check if kernel uses multi-wave workgroups."""
        from .utils import normalize_wg_size
        wg_size_x, wg_size_y, wg_size_z = normalize_wg_size(self.kernel_info.wg_size)
        return wg_size_y > 1 or wg_size_z > 1
    
    def _get_flat_tid_vgpr(self) -> int:
        """Get the VGPR containing flat thread ID."""
        if self.asm_emitter.special_regs.has_flat_tid():
            return self.asm_emitter.special_regs.get_flat_tid_vgpr()
        return 0
    
    def emit(self, expr: sympy.Expr, dst_reg: str) -> str:
        """
        Emit instructions for an expression into a specific register.
        
        Lower-level than get_or_emit - always emits, but uses caching internally.
        
        Args:
            expr: SymPy expression to emit
            dst_reg: Destination register (e.g., "v2")
        
        Returns:
            The register containing the result (usually dst_reg).
        """
        # Use get_or_emit which handles CSE
        result = self.get_or_emit(expr, dst_hint=dst_reg)
        return result
    
    def clear_cache(self) -> None:
        """Clear the expression cache."""
        self._cache.clear()


def create_emitter(asm_emitter, kernel_info):
    """
    Create a KernelEmitter for the given kernel.
    
    KernelEmitter manages registers across the entire kernel, providing
    global CSE and better register reuse than per-expression allocation.
    """
    emitter = KernelEmitter(asm_emitter, kernel_info)
    if DEBUG_KERNEL_EMITTER:
        print(f"[EMITTER] Created KernelEmitter id={id(emitter)}")
    return emitter
