# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Expression emitter using virtual register IR for allocation-order independence.

This module provides ExprEmitterV2, which uses virtual registers internally
for expression emission, enabling safe global CSE without allocation-order
correctness coupling.

Key benefits over the original ExprEmitter:
1. Global subexpression CSE is safe and allocation-order-independent
2. Register allocation for temporaries is deterministic
3. Easy to test with different allocation policies

Streaming mode (default):
    emitter = ExprEmitterV2(asm_emitter, kernel_info)
    reg = emitter.get_or_emit(sympy_expr)  # Returns "v5" - physical
    # Instructions are emitted immediately, result is cached
    reg2 = emitter.get_or_emit(sympy_expr)  # Returns "v5" - CSE hit
"""

import os
import sympy
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Set, Union

from .expr_ir import (
    ExprProgram, ExprInstr, VReg, SReg, PhysVReg, PhysSReg, Imm, OpCode,
    VirtualReg, PhysicalReg, Operand, CachedExprRef, is_inline_constant,
)
from .expr_regalloc import ExprRegAlloc, AllocPolicy, allocate_program
from .expr_opt import optimize_program, OptimizationStats
from .expr_simplify import simplify_for_emission, SimplifyStats
from .expression_emitter import expr_key


# Debug flag for CSE logging
DEBUG_EXPR_V2_CSE = os.environ.get("WAVE_EXPR_V2_CSE_LOG", "0") == "1"

# Debug flag for optimization pass logging
DEBUG_EXPR_OPT = os.environ.get("WAVE_EXPR_OPT_LOG", "0") == "1"

# Debug flag for algebraic simplification logging
DEBUG_SIMPLIFY = os.environ.get("WAVE_EXPR_SIMPLIFY_LOG", "0") == "1"

# Enable/disable algebraic simplification (default: enabled)
ENABLE_SIMPLIFY = os.environ.get("WAVE_EXPR_SIMPLIFY", "1") == "1"

# Environment flag for cache summary dump at kernel end
DUMP_CSE_SUMMARY = os.environ.get("WAVE_EXPR_V2_CSE_SUMMARY", "0") == "1"

# Allocation policy from env (for testing)
ALLOC_POLICY_ENV = os.environ.get("WAVE_EXPR_V2_ALLOC_POLICY", "lifo").lower()
ALLOC_SEED_ENV = os.environ.get("WAVE_EXPR_V2_ALLOC_SEED", None)


def _get_alloc_policy() -> AllocPolicy:
    """Get allocation policy from environment."""
    policy_map = {
        "sequential": AllocPolicy.SEQUENTIAL,
        "lifo": AllocPolicy.LIFO_REUSE,
        "fifo": AllocPolicy.FIFO_REUSE,
        "random": AllocPolicy.RANDOM,
    }
    return policy_map.get(ALLOC_POLICY_ENV, AllocPolicy.LIFO_REUSE)


def _get_alloc_seed() -> Optional[int]:
    """Get random seed from environment."""
    if ALLOC_SEED_ENV is not None:
        try:
            return int(ALLOC_SEED_ENV)
        except ValueError:
            pass
    return None


# Marker for rational values (expr / const)
_RationalReg = namedtuple("_RationalReg", ["numerator_vreg", "denominator"])


class ExprEmitterV2:
    """
    Expression emitter using virtual register IR with streaming emission.
    
    Implements the ExprEmitterProtocol interface:
    - bind_symbol(name, reg)
    - get_or_emit(expr, dst_hint) -> str
    - emit(expr, dst_reg) -> str
    - clear_cache()
    
    On each get_or_emit call:
    1. Check if expression is cached (CSE hit)
    2. If not, build a virtual ExprProgram for the expression
    3. Allocate physical registers avoiding live/ABI registers
    4. Emit instructions immediately
    5. Cache the output physical register
    """
    
    def __init__(self, asm_emitter, kernel_info):
        """
        Initialize the expression emitter.
        
        Args:
            asm_emitter: The AsmEmitter for emitting physical instructions
            kernel_info: Kernel configuration info
        """
        self.asm_emitter = asm_emitter
        self.kernel_info = kernel_info
        
        # Symbol setup
        self.tid_x_symbol = sympy.Symbol("tid_x", nonnegative=True)
        self.tid_y_symbol = sympy.Symbol("tid_y", nonnegative=True)
        self.tid_z_symbol = sympy.Symbol("tid_z", nonnegative=True)
        
        # Symbol bindings: maps SymPy symbol -> register string (e.g., "s2", "v0")
        self.symbol_bindings: Dict[sympy.Symbol, str] = {}
        
        # CSE cache: maps expr_key -> physical register string ("vN")
        # This is the main cache for global subexpression elimination
        self._cache: Dict[tuple, str] = {}
        
        # Pinned output registers - physical VGPRs that hold cached results
        # These must be excluded from temp allocation to prevent clobbering
        self._pinned_outputs: Set[int] = set()
        
        # Track which physical ABI registers are reserved
        self._abi_vgprs: Set[int] = set()
        self._abi_sgprs: Set[int] = set()
        
        # CSE for tid_x/tid_y extraction (physical registers)
        self._tid_x_phys: Optional[str] = None
        self._tid_y_phys: Optional[str] = None
        
        # Constant cache: maps constant value -> VGPR index
        # This avoids materializing the same constant multiple times
        self._const_cache: Dict[int, int] = {}
        self._const_cache_pinned: Set[int] = set()  # VGPRs holding constants
        
        # Statistics
        self._cse_hits = 0
        self._cse_misses = 0
        self._subexpr_cse_hits = 0
        self._const_cache_hits = 0
        self._const_cache_misses = 0
        
        # Optimization pass statistics
        self._total_opt_stats = OptimizationStats()
        
        # Simplification statistics
        self._simplify_stats = SimplifyStats()
        
        # Bind ABI registers
        self._bind_abi_registers()
    
    def _bind_abi_registers(self):
        """Bind ABI/system registers from asm_emitter."""
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
        """
        Bind a symbol name to a register.
        
        Args:
            symbol_name: Symbol name (e.g., "wgid_x", "tid_x")
            register: Register name (e.g., "s2", "v0")
        """
        symbol = sympy.Symbol(symbol_name, nonnegative=True)
        self.symbol_bindings[symbol] = register
        
        # Track as ABI register
        if register.startswith("v"):
            idx = int(register[1:])
            self._abi_vgprs.add(idx)
        elif register.startswith("s"):
            idx = int(register[1:])
            self._abi_sgprs.add(idx)
    
    def _get_symbol_bounds(self) -> Dict[sympy.Symbol, Tuple[int, int]]:
        """
        Get bounds for symbols based on kernel configuration.
        
        Returns a dict mapping symbols to (min, max) bounds.
        """
        bounds = {}
        
        # Get thread/wave info from kernel_info if available
        threads_per_wave = 64  # Default for CDNA
        if hasattr(self.kernel_info, 'threads_per_wave'):
            threads_per_wave = self.kernel_info.threads_per_wave
        
        # For multi-wave kernels, tid_x/tid_y can be larger
        # Default conservative bounds
        max_tid = threads_per_wave * 4 - 1  # Assume max 4 waves per WG
        
        bounds[self.tid_x_symbol] = (0, max_tid)
        bounds[self.tid_y_symbol] = (0, max_tid)
        bounds[self.tid_z_symbol] = (0, max_tid)
        
        # Workgroup IDs can be large but are typically < 65536
        wgid_x = sympy.Symbol("wgid_x", nonnegative=True)
        wgid_y = sympy.Symbol("wgid_y", nonnegative=True)
        wgid_z = sympy.Symbol("wgid_z", nonnegative=True)
        bounds[wgid_x] = (0, 65535)
        bounds[wgid_y] = (0, 65535)
        bounds[wgid_z] = (0, 65535)
        
        return bounds
    
    def get_or_emit(self, expr: sympy.Expr, dst_hint: Optional[str] = None) -> str:
        """
        Get cached register for expression or emit and cache it.
        
        This is the primary entry point for expression emission.
        
        Args:
            expr: SymPy expression to emit
            dst_hint: Optional destination register hint (e.g., "v2")
        
        Returns:
            Physical register string (e.g., "v5") containing the result.
        """
        # Step 1: Algebraic simplification before anything else
        if ENABLE_SIMPLIFY:
            original_expr = expr
            expr = simplify_for_emission(expr, self._get_symbol_bounds())
            if DEBUG_SIMPLIFY and expr != original_expr:
                print(f"[SIMPLIFY] {original_expr} → {expr}")
                self._simplify_stats.record(original_expr, expr)
        
        key = expr_key(expr)
        
        # Check CSE cache
        if key in self._cache:
            self._cse_hits += 1
            result = self._cache[key]
            if DEBUG_EXPR_V2_CSE:
                print(f"[CSE HIT] {expr} -> {result}")
            return result
        
        self._cse_misses += 1
        
        # First pass: emit any cacheable subexpressions that aren't already cached
        # This enables sharing subexpressions across different top-level expressions
        self._cache_subexpressions(expr)
        
        # Build virtual IR program for this expression
        program = ExprProgram()
        result_vreg = self._emit_to_program(program, expr)
        
        # Handle constant results
        if isinstance(result_vreg, tuple) and result_vreg[0] == "_const":
            # Materialize constant
            vreg = program.alloc_vreg()
            program.emit(OpCode.MOV_IMM, vreg, (Imm(result_vreg[1]),), f"const {result_vreg[1]}")
            result_vreg = vreg
        
        # Run optimization passes (copy-prop, DCE, coalescing)
        program, opt_stats = optimize_program(program)
        self._total_opt_stats.copies_propagated += opt_stats.copies_propagated
        self._total_opt_stats.movs_eliminated += opt_stats.movs_eliminated
        self._total_opt_stats.vregs_coalesced += opt_stats.vregs_coalesced
        
        if DEBUG_EXPR_OPT and opt_stats.movs_eliminated > 0:
            print(f"[EXPR OPT] Eliminated {opt_stats.movs_eliminated} MOVs, "
                  f"propagated {opt_stats.copies_propagated} copies, "
                  f"coalesced {opt_stats.vregs_coalesced} vregs")
        
        # Compute reserved set for allocation
        reserved_vgprs = self._get_reserved_vgprs()
        reserved_sgprs = self._abi_sgprs.copy()
        
        # Allocate physical registers
        policy = _get_alloc_policy()
        seed = _get_alloc_seed()
        
        phys_instrs, mapping = allocate_program(
            program,
            reserved_vgprs=reserved_vgprs,
            reserved_sgprs=reserved_sgprs,
            policy=policy,
            random_seed=seed,
        )
        
        # Track all allocated physical VGPRs to update main allocator
        allocated_phys_vgprs = set()
        for vreg, phys in mapping.items():
            if isinstance(phys, PhysVReg):
                allocated_phys_vgprs.add(phys.index)
        
        # Reserve all allocated VGPRs in the main allocator to prevent conflicts
        # This is critical: the main allocator doesn't know about our allocations
        for vgpr_idx in allocated_phys_vgprs:
            if vgpr_idx not in self.asm_emitter.register_file.v_used:
                self.asm_emitter.register_file.v_used.add(vgpr_idx)
        
        # Emit physical instructions immediately
        for instr in phys_instrs:
            self._emit_physical_instr(instr)
        
        # Get the physical register for the result
        if isinstance(result_vreg, VReg):
            phys_reg = mapping[result_vreg]
            result_str = f"v{phys_reg.index}"
        else:
            # Physical register passed through (e.g., from symbol binding)
            result_str = str(result_vreg)
        
        # Pin the output register to prevent clobbering
        if result_str.startswith("v"):
            idx = int(result_str[1:])
            self._pinned_outputs.add(idx)
        
        # Cache the result
        self._cache[key] = result_str
        
        if DEBUG_EXPR_V2_CSE:
            print(f"[CSE MISS] {expr} -> {result_str} (emitted {len(phys_instrs)} instrs)")
        
        return result_str
    
    def _cache_subexpressions(self, expr: sympy.Expr) -> None:
        """
        Pre-emit and cache common subexpressions to enable global CSE.
        
        This walks the expression tree and identifies subexpressions that:
        1. Are non-trivial (floor, mod, shifts)
        2. Are not already cached
        3. Are worth caching (contain thread/workgroup IDs)
        
        By emitting these first, subsequent expressions that use the same
        subexpressions can reference the cached results instead of recomputing.
        """
        # Collect all cacheable subexpressions
        cacheable = self._find_cacheable_subexprs(expr)
        
        # Emit each uncached subexpression (smallest first for better sharing)
        for subexpr in sorted(cacheable, key=lambda e: len(str(e))):
            key = expr_key(subexpr)
            if key not in self._cache:
                # Recursively emit this subexpression
                # Note: This will call _cache_subexpressions again, but for smaller subexprs
                # We check the cache first to avoid infinite recursion
                self._emit_subexpression(subexpr)
    
    def _find_cacheable_subexprs(self, expr: sympy.Expr) -> Set[sympy.Expr]:
        """
        Find subexpressions worth caching.
        
        Returns a set of subexpressions that are:
        - Non-trivial operations (floor, mod, mul with shift)
        - Contain dynamic symbols (tid_x, tid_y, wgid_x, etc.)
        """
        cacheable = set()
        
        def is_dynamic(e):
            """Check if expression contains dynamic (thread/wg) symbols."""
            return len(e.free_symbols) > 0
        
        def is_cacheable_op(e):
            """Check if this operation is worth caching."""
            # Floor division is worth caching
            if getattr(e, "func", None) == sympy.floor:
                return True
            # Mod is worth caching
            if isinstance(e, sympy.Mod):
                return True
            # Mul with power-of-2 shift is worth caching
            if isinstance(e, sympy.Mul) and len(e.args) > 1:
                # Check if any arg is a power of 2 constant
                for arg in e.args:
                    if isinstance(arg, sympy.Integer):
                        val = int(arg)
                        if val > 0 and (val & (val - 1)) == 0:
                            return True
            return False
        
        def walk(e):
            """Recursively find cacheable subexpressions."""
            if not is_dynamic(e):
                return
            
            # Check if this expression is cacheable
            if is_cacheable_op(e):
                cacheable.add(e)
            
            # Recurse into arguments
            if hasattr(e, 'args'):
                for arg in e.args:
                    walk(arg)
        
        walk(expr)
        return cacheable
    
    def _emit_subexpression(self, subexpr: sympy.Expr) -> str:
        """
        Emit a subexpression and cache it.
        
        This is a lightweight version of get_or_emit that avoids
        recursively calling _cache_subexpressions.
        """
        key = expr_key(subexpr)
        
        # Already cached?
        if key in self._cache:
            return self._cache[key]
        
        # Build program for this subexpression
        program = ExprProgram()
        result_vreg = self._emit_to_program(program, subexpr)
        
        # Handle constant results
        if isinstance(result_vreg, tuple) and result_vreg[0] == "_const":
            vreg = program.alloc_vreg()
            program.emit(OpCode.MOV_IMM, vreg, (Imm(result_vreg[1]),), f"const {result_vreg[1]}")
            result_vreg = vreg
        
        # If it's already a physical register, just cache and return
        if isinstance(result_vreg, PhysVReg):
            result_str = f"v{result_vreg.index}"
            self._cache[key] = result_str
            return result_str
        if isinstance(result_vreg, PhysSReg):
            result_str = f"s{result_vreg.index}"
            self._cache[key] = result_str
            return result_str
        
        # Run optimization passes (copy-prop, DCE, coalescing)
        program, opt_stats = optimize_program(program)
        self._total_opt_stats.copies_propagated += opt_stats.copies_propagated
        self._total_opt_stats.movs_eliminated += opt_stats.movs_eliminated
        self._total_opt_stats.vregs_coalesced += opt_stats.vregs_coalesced
        
        # Compute reserved set for allocation
        reserved_vgprs = self._get_reserved_vgprs()
        reserved_sgprs = self._abi_sgprs.copy()
        
        # Allocate physical registers
        policy = _get_alloc_policy()
        seed = _get_alloc_seed()
        
        phys_instrs, mapping = allocate_program(
            program,
            reserved_vgprs=reserved_vgprs,
            reserved_sgprs=reserved_sgprs,
            policy=policy,
            random_seed=seed,
        )
        
        # Track allocated VGPRs
        for vreg, phys in mapping.items():
            if isinstance(phys, PhysVReg):
                if phys.index not in self.asm_emitter.register_file.v_used:
                    self.asm_emitter.register_file.v_used.add(phys.index)
        
        # Emit physical instructions
        for instr in phys_instrs:
            self._emit_physical_instr(instr)
        
        # Get result register
        if isinstance(result_vreg, VReg):
            phys_reg = mapping[result_vreg]
            result_str = f"v{phys_reg.index}"
        else:
            result_str = str(result_vreg)
        
        # Pin and cache
        if result_str.startswith("v"):
            self._pinned_outputs.add(int(result_str[1:]))
        
        self._cache[key] = result_str
        
        if DEBUG_EXPR_V2_CSE:
            print(f"[SUBEXPR CACHE] {subexpr} -> {result_str}")
        
        return result_str

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
        
        # If result differs from dst_reg and dst_reg was specified, move it
        if result != dst_reg and dst_reg is not None:
            # The result is in a different register - move if needed
            # (In streaming mode, the allocator may have chosen a different reg)
            # For compatibility, we could emit a move, but typically callers
            # should use the returned register
            pass
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the expression cache."""
        self._cache.clear()
        self._pinned_outputs.clear()
        self._tid_x_phys = None
        self._tid_y_phys = None
        self._const_cache.clear()
        self._const_cache_pinned.clear()
        self._cse_hits = 0
        self._cse_misses = 0
        self._subexpr_cse_hits = 0
        self._const_cache_hits = 0
        self._const_cache_misses = 0
        self._total_opt_stats = OptimizationStats()
        self._simplify_stats = SimplifyStats()
    
    def _get_reserved_vgprs(self) -> Set[int]:
        """Get the set of VGPRs that must not be used for temps."""
        reserved = set()
        # ABI registers
        reserved.update(self._abi_vgprs)
        # Currently allocated (live) VGPRs
        reserved.update(self.asm_emitter.register_file.v_used)
        # Pinned output registers from CSE cache
        reserved.update(self._pinned_outputs)
        return reserved
    
    def _emit_to_program(self, program: ExprProgram, expr: sympy.Expr):
        """
        Emit expression to a virtual ExprProgram.
        
        Implements global subexpression CSE: if this expression (or subexpression)
        has already been computed and cached, return a reference to the cached
        physical register instead of recomputing.
        
        Returns:
            VReg, PhysVReg, PhysSReg, ("_const", value), or _RationalReg
        """
        # Global subexpression CSE: check if this expression is already cached
        # This enables sharing subexpressions across different top-level expressions
        key = expr_key(expr)
        if key in self._cache:
            cached_reg = self._cache[key]
            if cached_reg.startswith("v"):
                # Return reference to cached physical VGPR
                self._subexpr_cse_hits += 1
                if DEBUG_EXPR_V2_CSE:
                    print(f"[SUBEXPR CSE HIT] {expr} -> {cached_reg}")
                return PhysVReg(int(cached_reg[1:]))
            elif cached_reg.startswith("s"):
                return PhysSReg(int(cached_reg[1:]))
        
        # Not cached - proceed with emission
        if isinstance(expr, CachedExprRef):
            # CachedExprRef is a wrapper that preserves a subexpression as a unit.
            # Look up the wrapped expression in the cache - it should already be computed.
            wrapped = expr.wrapped
            wrapped_key = expr_key(wrapped)
            if wrapped_key in self._cache:
                cached_reg = self._cache[wrapped_key]
                if DEBUG_EXPR_V2_CSE:
                    print(f"[CACHED_EXPR_REF] {wrapped} -> {cached_reg}")
                if cached_reg.startswith("v"):
                    return PhysVReg(int(cached_reg[1:]))
                elif cached_reg.startswith("s"):
                    return PhysSReg(int(cached_reg[1:]))
            # Not in cache - emit the wrapped expression
            if DEBUG_EXPR_V2_CSE:
                print(f"[CACHED_EXPR_REF] {wrapped} not in cache, emitting")
            return self._emit_to_program(program, wrapped)
        elif isinstance(expr, sympy.Symbol):
            return self._emit_symbol_to_program(program, expr)
        elif isinstance(expr, sympy.Integer):
            return ("_const", int(expr))
        elif isinstance(expr, sympy.Rational) and not isinstance(expr, sympy.Integer):
            return self._emit_rational_to_program(program, expr)
        elif isinstance(expr, sympy.Add):
            return self._emit_add_to_program(program, expr)
        elif isinstance(expr, sympy.Mul):
            return self._emit_mul_to_program(program, expr)
        elif isinstance(expr, sympy.Mod):
            return self._emit_mod_to_program(program, expr)
        elif getattr(expr, "func", None) == sympy.floor:
            return self._emit_floor_to_program(program, expr)
        elif isinstance(expr, sympy.Pow):
            return self._emit_pow_to_program(program, expr)
        else:
            raise ValueError(
                f"ExprEmitterV2: Unsupported expression type: {type(expr).__name__}: {expr}. "
                "Please add support or use WAVE_EXPR_EMITTER=legacy to fall back."
            )
    
    def _emit_symbol_to_program(self, program: ExprProgram, sym: sympy.Symbol):
        """Emit a symbol reference."""
        # Check bindings
        if sym in self.symbol_bindings:
            bound_reg = self.symbol_bindings[sym]
            
            # Special handling for tid_x/tid_y in multi-wave mode:
            # In multi-wave, vgpr_tid_x is set to v0 (flat_tid), not the extracted tid_x.
            # We need to extract the actual tid_x/tid_y from the flat thread ID.
            if self._is_multi_wave():
                if sym == self.tid_x_symbol:
                    # Extract tid_x (bits 0-9) from flat_tid
                    return self._emit_tid_x_to_program(program)
                elif sym == self.tid_y_symbol:
                    # Extract tid_y (bits 10-19) from flat_tid
                    return self._emit_tid_y_to_program(program)
            
            # For other symbols or single-wave tid_x/tid_y, use binding directly
            if bound_reg.startswith("v"):
                return PhysVReg(int(bound_reg[1:]))
            elif bound_reg.startswith("s"):
                return PhysSReg(int(bound_reg[1:]))
        
        # Handle tid_x - fallback if not bound
        if sym == self.tid_x_symbol:
            if self._is_multi_wave():
                return self._emit_tid_x_to_program(program)
            else:
                return self._emit_lane_id_to_program(program)
        
        # Handle tid_y - fallback if not bound
        if sym == self.tid_y_symbol:
            if self._is_multi_wave():
                return self._emit_tid_y_to_program(program)
            else:
                return ("_const", 0)
        
        # Handle tid_z
        if sym == self.tid_z_symbol:
            raise ValueError("tid_z not yet supported in ExprEmitterV2")
        
        # Direct SGPR reference like s4
        sym_str = str(sym)
        if sym_str.startswith("s") and sym_str[1:].isdigit():
            return PhysSReg(int(sym_str[1:]))
        
        raise ValueError(f"ExprEmitterV2: Unknown symbol: {sym}")
    
    def _emit_tid_x_to_program(self, program: ExprProgram) -> VReg:
        """Emit tid_x extraction to program."""
        # Check if we already have it cached (physical)
        if self._tid_x_phys is not None:
            # Already extracted - reference the physical register
            idx = int(self._tid_x_phys[1:])
            return PhysVReg(idx)
        
        # Extract from flat tid
        flat_tid = self._get_flat_tid_phys()
        vreg = program.alloc_vreg()
        program.emit(OpCode.AND_IMM, vreg, (flat_tid, Imm(0x3ff)), "tid_x = flat_tid & 0x3ff")
        return vreg
    
    def _emit_tid_y_to_program(self, program: ExprProgram) -> VReg:
        """Emit tid_y extraction to program."""
        if self._tid_y_phys is not None:
            idx = int(self._tid_y_phys[1:])
            return PhysVReg(idx)
        
        flat_tid = self._get_flat_tid_phys()
        vreg = program.alloc_vreg()
        program.emit(OpCode.BFE, vreg, (flat_tid, Imm(10), Imm(10)), "tid_y = bfe(flat_tid, 10, 10)")
        return vreg
    
    def _emit_lane_id_to_program(self, program: ExprProgram) -> VReg:
        """Emit lane_id for single-wave mode."""
        # Get lane_id from asm_emitter
        lane_id_v = self.asm_emitter.ensure_lane_id(self.kernel_info.subgroup_size)
        return PhysVReg(lane_id_v)
    
    def _emit_rational_to_program(self, program: ExprProgram, expr: sympy.Rational):
        """Emit a rational constant (division marker)."""
        if expr.p == 1:
            return _RationalReg(None, int(expr.q))
        else:
            raise ValueError(f"Only 1/N rationals supported: {expr}")
    
    def _emit_add_to_program(self, program: ExprProgram, expr: sympy.Add):
        """Emit addition expression to program."""
        args = [self._emit_to_program(program, arg) for arg in expr.args]
        
        const_sum = 0
        reg_args = []
        
        for arg in args:
            if isinstance(arg, tuple) and arg[0] == "_const":
                const_sum += arg[1]
            elif isinstance(arg, _RationalReg):
                reg_args.append(self._materialize_rational_to_program(program, arg))
            else:
                reg_args.append(arg)
        
        if not reg_args:
            return ("_const", const_sum)
        
        result = reg_args[0]
        if not isinstance(result, VReg):
            result = self._copy_to_vreg(program, result)
        
        for arg in reg_args[1:]:
            if not isinstance(arg, VReg):
                arg = self._copy_to_vreg(program, arg)
            new_result = program.alloc_vreg()
            program.emit(OpCode.ADD, new_result, (result, arg))
            result = new_result
        
        if const_sum != 0:
            new_result = program.alloc_vreg()
            program.emit(OpCode.ADD_IMM, new_result, (result, Imm(const_sum)))
            result = new_result
        
        return result
    
    def _emit_mul_to_program(self, program: ExprProgram, expr: sympy.Mul):
        """Emit multiplication expression to program."""
        args = [self._emit_to_program(program, arg) for arg in expr.args]
        
        const_product = 1
        rational_arg = None
        reg_args = []
        
        for arg in args:
            if isinstance(arg, tuple) and arg[0] == "_const":
                const_product *= arg[1]
            elif isinstance(arg, _RationalReg):
                if rational_arg is not None:
                    raise ValueError("Multiple rationals in multiplication")
                rational_arg = arg
            else:
                reg_args.append(arg)
        
        if rational_arg is not None:
            if len(reg_args) != 1:
                raise ValueError("Rational multiplication needs exactly one register")
            return _RationalReg(reg_args[0], rational_arg.denominator)
        
        if not reg_args:
            return ("_const", const_product)
        
        if len(reg_args) == 1:
            src = reg_args[0]
            if not isinstance(src, VReg):
                src = self._copy_to_vreg(program, src)
            
            if const_product == 1:
                return src
            elif const_product > 0 and (const_product & (const_product - 1)) == 0:
                shift = const_product.bit_length() - 1
                result = program.alloc_vreg()
                program.emit(OpCode.LSHL, result, (src, Imm(shift)))
                return result
            else:
                result = program.alloc_vreg()
                program.emit(OpCode.MUL_IMM, result, (src, Imm(const_product)))
                return result
        
        raise ValueError(f"Product of multiple dynamic terms not supported: {expr}")
    
    def _emit_mod_to_program(self, program: ExprProgram, expr: sympy.Mod):
        """Emit modulo expression to program."""
        dividend = self._emit_to_program(program, expr.args[0])
        divisor_expr = expr.args[1]
        
        if not isinstance(divisor_expr, (int, sympy.Integer)):
            raise ValueError(f"Mod divisor must be constant: {divisor_expr}")
        
        divisor = int(divisor_expr)
        if divisor <= 0 or (divisor & (divisor - 1)) != 0:
            raise ValueError(f"Mod divisor must be power-of-2: {divisor}")
        
        if isinstance(dividend, tuple) and dividend[0] == "_const":
            return ("_const", dividend[1] % divisor)
        
        if not isinstance(dividend, VReg):
            dividend = self._copy_to_vreg(program, dividend)
        
        mask = divisor - 1
        result = program.alloc_vreg()
        program.emit(OpCode.AND_IMM, result, (dividend, Imm(mask)))
        return result
    
    def _emit_floor_to_program(self, program: ExprProgram, expr):
        """Emit floor expression to program."""
        operand = self._emit_to_program(program, expr.args[0])
        
        if isinstance(operand, _RationalReg):
            divisor = operand.denominator
            if divisor <= 0 or (divisor & (divisor - 1)) != 0:
                raise ValueError(f"Floor divisor must be power-of-2: {divisor}")
            
            numerator = operand.numerator_vreg
            if numerator is None:
                return ("_const", 0)
            
            if isinstance(numerator, tuple) and numerator[0] == "_const":
                return ("_const", numerator[1] // divisor)
            
            if not isinstance(numerator, VReg):
                numerator = self._copy_to_vreg(program, numerator)
            
            shift = divisor.bit_length() - 1
            if shift > 0:
                result = program.alloc_vreg()
                program.emit(OpCode.LSHR, result, (numerator, Imm(shift)))
                return result
            else:
                return numerator
        
        if isinstance(operand, tuple) and operand[0] == "_const":
            return operand
        
        return operand
    
    def _emit_pow_to_program(self, program: ExprProgram, expr: sympy.Pow):
        """Emit power expression to program."""
        base, exp = expr.args
        if base == 2 and isinstance(exp, (int, sympy.Integer)):
            return ("_const", 2 ** int(exp))
        raise ValueError(f"Only 2^k powers supported: {expr}")
    
    def _is_multi_wave(self) -> bool:
        """Check if kernel uses multi-wave workgroups."""
        from .utils import normalize_wg_size
        wg_size_x, wg_size_y, wg_size_z = normalize_wg_size(self.kernel_info.wg_size)
        return wg_size_y > 1 or wg_size_z > 1
    
    def _get_flat_tid_phys(self) -> PhysVReg:
        """Get the physical VGPR containing flat thread ID."""
        if self.asm_emitter.special_regs.has_flat_tid():
            return PhysVReg(self.asm_emitter.special_regs.get_flat_tid_vgpr())
        return PhysVReg(0)
    
    def _copy_to_vreg(self, program: ExprProgram, operand) -> VReg:
        """Copy a physical register or constant to a VReg."""
        if isinstance(operand, VReg):
            return operand
        
        if isinstance(operand, (PhysVReg, PhysSReg)):
            vreg = program.alloc_vreg()
            program.emit(OpCode.MOV, vreg, (operand,), f"copy from {operand}")
            return vreg
        
        if isinstance(operand, tuple) and operand[0] == "_const":
            vreg = program.alloc_vreg()
            program.emit(OpCode.MOV_IMM, vreg, (Imm(operand[1]),), f"const {operand[1]}")
            return vreg
        
        raise ValueError(f"Cannot copy to VReg: {operand}")
    
    def _materialize_rational_to_program(self, program: ExprProgram, rational: _RationalReg) -> VReg:
        """Materialize a rational (floor division) to a VReg."""
        if rational.numerator_vreg is None:
            vreg = program.alloc_vreg()
            program.emit(OpCode.MOV_IMM, vreg, (Imm(0),), "const 0")
            return vreg
        
        numerator = rational.numerator_vreg
        if not isinstance(numerator, VReg):
            numerator = self._copy_to_vreg(program, numerator)
        
        divisor = rational.denominator
        if divisor <= 0 or (divisor & (divisor - 1)) != 0:
            raise ValueError(f"Rational divisor must be power-of-2: {divisor}")
        
        shift = divisor.bit_length() - 1
        if shift > 0:
            result = program.alloc_vreg()
            program.emit(OpCode.LSHR, result, (numerator, Imm(shift)))
            return result
        else:
            return numerator
    
    def _get_or_materialize_const(self, value: int) -> int:
        """
        Get a VGPR containing a constant value, using cache to avoid duplicates.
        
        This is a key optimization: when multiple MUL_IMM instructions need
        the same constant, we materialize it once and reuse the VGPR.
        
        Args:
            value: The constant value to materialize
            
        Returns:
            VGPR index containing the constant
        """
        from .instructions import VMovB32
        
        if value in self._const_cache:
            self._const_cache_hits += 1
            return self._const_cache[value]
        
        self._const_cache_misses += 1
        
        # Allocate a new VGPR for this constant
        const_v = self.asm_emitter.vgpr_allocator.alloc_v()
        self.asm_emitter.emit_instruction(VMovB32(const_v, value))
        
        # Cache it for future use
        self._const_cache[value] = const_v
        self._const_cache_pinned.add(const_v)
        
        # Also pin in our output set so it's not clobbered
        self._pinned_outputs.add(const_v)
        
        return const_v
    
    def _emit_physical_instr(self, instr: ExprInstr) -> None:
        """Emit a physical instruction to the asm_emitter."""
        from .instructions import (
            VMovB32, VAddU32, VMulLoU32, VLshlRevB32, VLshrrevB32,
            VAndB32,
        )
        
        dst = instr.dst
        ops = instr.operands
        
        if not isinstance(dst, PhysVReg):
            raise ValueError(f"Expected PhysVReg dst, got {type(dst)}: {dst}")
        
        dst_idx = dst.index
        
        if instr.opcode == OpCode.MOV_IMM:
            self.asm_emitter.emit_instruction(VMovB32(dst_idx, ops[0].value))
        
        elif instr.opcode == OpCode.MOV:
            src = ops[0]
            if isinstance(src, PhysVReg):
                self.asm_emitter.emit_instruction(VMovB32(dst_idx, f"v{src.index}"))
            elif isinstance(src, PhysSReg):
                self.asm_emitter.emit_instruction(VMovB32(dst_idx, f"s{src.index}"))
            else:
                raise ValueError(f"MOV source must be register: {src}")
        
        elif instr.opcode == OpCode.ADD:
            src1, src2 = ops
            self.asm_emitter.emit_instruction(VAddU32(dst_idx, src1.index, src2.index))
        
        elif instr.opcode == OpCode.ADD_IMM:
            src, imm = ops
            # v_add_u32 can take inline constants (0-64, -1 to -16) as src0
            # For larger constants, we need to materialize into an SGPR
            if is_inline_constant(imm.value):
                self.asm_emitter.emit(f"    v_add_u32 v{dst_idx}, {imm.value}, v{src.index}")
            else:
                # Large constant - materialize into SGPR and use VAddU32Any
                from .instructions import SMovB32, VAddU32Any
                const_sgpr = self.asm_emitter.sgpr_allocator.alloc_s()
                if DEBUG_EXPR_V2_CSE:
                    print(f"[ADD_IMM] v{dst_idx} = v{src.index} + {imm.value} (via s{const_sgpr})")
                self.asm_emitter.emit_instruction(SMovB32(const_sgpr, imm.value))
                self.asm_emitter.emit_instruction(VAddU32Any(dst_idx, src.index, f"s{const_sgpr}"))
        
        elif instr.opcode == OpCode.MUL_IMM:
            src, imm = ops
            # v_mul_lo_u32 requires the constant in a VGPR
            # Use constant cache to avoid materializing the same constant multiple times
            const_v = self._get_or_materialize_const(imm.value)
            self.asm_emitter.emit_instruction(VMulLoU32(dst_idx, src.index, const_v))
        
        elif instr.opcode == OpCode.LSHL:
            src, shift = ops
            self.asm_emitter.emit_instruction(VLshlRevB32(dst_idx, shift.value, src.index))
        
        elif instr.opcode == OpCode.LSHR:
            src, shift = ops
            self.asm_emitter.emit_instruction(VLshrrevB32(dst_idx, shift.value, src.index))
        
        elif instr.opcode == OpCode.AND_IMM:
            src, mask = ops
            self.asm_emitter.emit_instruction(VAndB32(dst_idx, mask.value, src.index))
        
        elif instr.opcode == OpCode.BFE:
            src, offset, width = ops
            self.asm_emitter.emit(
                f"    v_bfe_u32 v{dst_idx}, v{src.index}, {offset.value}, {width.value}"
            )
        
        else:
            raise ValueError(
                f"ExprEmitterV2: Unsupported opcode: {instr.opcode}. "
                "Please add support or use WAVE_EXPR_EMITTER=legacy to fall back."
            )
    
    def get_cse_stats(self) -> Dict[str, int]:
        """Get CSE statistics."""
        return {
            "cache_entries": len(self._cache),
            "cse_hits": self._cse_hits,
            "cse_misses": self._cse_misses,
            "subexpr_cse_hits": self._subexpr_cse_hits,
            "const_cache_entries": len(self._const_cache),
            "const_cache_hits": self._const_cache_hits,
            "const_cache_misses": self._const_cache_misses,
            "pinned_outputs": len(self._pinned_outputs),
            "copies_propagated": self._total_opt_stats.copies_propagated,
            "movs_eliminated": self._total_opt_stats.movs_eliminated,
            "vregs_coalesced": self._total_opt_stats.vregs_coalesced,
        }
    
    def dump_cache(self) -> str:
        """Dump the CSE cache for debugging."""
        lines = [
            "=" * 60,
            "ExprEmitterV2 CSE Cache",
            f"  Entries: {len(self._cache)}",
            f"  Hits: {self._cse_hits}, Misses: {self._cse_misses}",
            f"  Pinned outputs: {sorted(self._pinned_outputs)}",
            "-" * 60,
        ]
        for key, reg in sorted(self._cache.items(), key=lambda x: str(x[0])):
            key_str = str(key)[:50]
            lines.append(f"  {reg} <- {key_str}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def maybe_dump_summary(self) -> None:
        """Dump CSE summary if WAVE_EXPR_V2_CSE_SUMMARY=1."""
        if DUMP_CSE_SUMMARY:
            print(self.dump_cache())
