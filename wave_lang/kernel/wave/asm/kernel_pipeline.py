# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Kernel-level compilation pipeline with whole-program register allocation.

This module provides the integration point for the new kernel-level linear scan
register allocator. It can be enabled via the WAVE_KERNEL_LSRA environment variable.

When enabled:
1. Expression emission goes to KernelProgram with virtual registers
2. After all instructions are emitted, liveness is computed
3. Linear scan allocator assigns physical registers
4. Renderer generates final assembly

When disabled (default):
- The original per-expression allocation path is used

Usage:
    from kernel_pipeline import use_kernel_lsra, KernelCompilationContext
    
    if use_kernel_lsra():
        ctx = KernelCompilationContext(kernel_info)
        # Emit instructions via dynamic dispatch
        v1 = ctx.v_mov_b32(42)
        v2 = ctx.v_add_u32(v1, 100)
        # Finalize and get assembly
        asm = ctx.finalize()
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .kernel_model import KernelInfo

from .kernel_ir import (
    KernelProgram, KernelBuilder, KInstr,
    KVReg, KSReg, KPhysVReg, KPhysSReg, KSpecialReg,
    KReg, KRegRange, KImm, KMemOffset,
    KernelABI, RegClass, M0,
)
from .kernel_liveness import compute_liveness, LivenessInfo
from .kernel_regalloc import KernelRegAlloc, allocate_kernel, AllocationStats, AllocationError
from .kernel_generator import KernelGenerator, PhysicalMapping, generate_program
from .unified_emitter import UnifiedEmitter, EmissionMode
from .hazards import HazardDetector
from .instruction_categories import InstructionCategory, categorize_instruction
from .instruction_registry import (
    InstructionRegistry,
    InstructionDef,
    OperandType,
    get_registry,
)


# Environment variable to enable kernel-level LSRA (advanced allocation)
# Default is "0" (disabled) during development
WAVE_KERNEL_LSRA_ENV = "WAVE_KERNEL_LSRA"

# Environment variable to enable kernel IR path
# Default is "0" (disabled) - GEMM/MFMA support still being debugged
# Set to "1" to use kernel IR path for simple kernels (copy works)
WAVE_KERNEL_IR_ENV = "WAVE_KERNEL_IR"

# Flag to enable algebraic simplification in kernel IR mode
# Enabled by default - the depth-tracking fix prevents O(n^2) behavior
_ENABLE_KERNEL_IR_SIMPLIFY = os.environ.get("WAVE_KERNEL_IR_SIMPLIFY", "1") == "1"


def use_kernel_lsra() -> bool:
    """Check if kernel-level LSRA is enabled."""
    return os.environ.get(WAVE_KERNEL_LSRA_ENV, "0") == "1"


def use_kernel_ir() -> bool:
    """
    Check if full kernel IR path is enabled (default: False).
    
    The kernel IR path supports simple kernels like copy.
    GEMM/MFMA support is still being debugged.
    Set WAVE_KERNEL_IR=1 to opt-in.
    """
    return os.environ.get(WAVE_KERNEL_IR_ENV, "0") == "1"


# =============================================================================
# Structural Expression Key (for CSE)
# =============================================================================


def _expr_key(expr) -> tuple:
    """
    Build a structural, hashable key for an expression.
    
    Uses a direct structural walk to create a hashable key that identifies
    semantically identical expressions, enabling proper CSE caching.
    The key is commutative-aware (Add and Mul args are sorted).
    """
    import sympy
    
    def to_key(e):
        if e is None:
            return ("none",)
        if isinstance(e, sympy.Integer):
            return ("int", int(e))
        if isinstance(e, sympy.Rational) and not isinstance(e, sympy.Integer):
            return ("rat", int(e.p), int(e.q))
        if isinstance(e, sympy.Symbol):
            return ("sym", str(e))
        if isinstance(e, sympy.Add):
            arg_keys = tuple(sorted([to_key(a) for a in e.args], key=str))
            return ("add", arg_keys)
        if isinstance(e, sympy.Mul):
            arg_keys = tuple(sorted([to_key(a) for a in e.args], key=str))
            return ("mul", arg_keys)
        if isinstance(e, sympy.Mod):
            return ("mod", to_key(e.args[0]), to_key(e.args[1]))
        if getattr(e, "func", None) == sympy.floor:
            return ("floor", to_key(e.args[0]))
        if isinstance(e, sympy.Pow):
            return ("pow", to_key(e.args[0]), to_key(e.args[1]))
        return ("raw", repr(e))
    
    return to_key(expr)


def _contains_loop_sgpr(expr) -> bool:
    """
    Check if expression contains an SGPR symbol (loop counter like s24).
    
    Expressions containing loop counters must not be cached because the
    loop counter changes each iteration.
    """
    if not hasattr(expr, 'free_symbols'):
        return False
    for sym in expr.free_symbols:
        name = str(sym)
        if name.startswith('s') and len(name) > 1 and name[1:].isdigit():
            return True
    return False


class _ScopeContext:
    """Context manager for scoped CSE regions."""
    
    def __init__(self, emitter: 'KernelIRExprEmitter', name: str):
        self._emitter = emitter
        self._name = name
    
    def __enter__(self):
        self._emitter.push_scope(self._name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._emitter.pop_scope()
        return False


# =============================================================================
# Kernel IR Expression Emitter with Scoped CSE
# =============================================================================


class KernelIRExprEmitter:
    """
    Expression emitter for kernel IR that can emit sympy expressions.
    
    Uses scoped CSE (Common Subexpression Elimination) to avoid redundant
    computation while respecting MLIR structured control flow.
    
    Key features:
    - Global scope: Values that dominate all uses (prologue-defined tid/wgid)
    - Region scopes: Pushed/popped for structured regions (loops, ifs)
    - Structural expression keys: Enable CSE across identical expressions
    """
    
    def __init__(self, ctx: 'KernelCompilationContext'):
        self.ctx = ctx
        # Scope stack: list of dicts, index 0 is global scope
        # Each scope maps expression_key -> KVReg
        self._scope_stack: List[Dict[tuple, KVReg]] = [{}]  # Start with global scope
        # Symbol bindings: name -> register (string like "v0" or KVReg)
        self._bindings: Dict[str, Any] = {}
        # Recursion depth counter for get_or_emit
        # Used to only run simplification at top-level (depth=0)
        self._emit_depth: int = 0
    
    @property
    def _global_scope(self) -> Dict[tuple, KVReg]:
        """Get the global (prologue) scope."""
        return self._scope_stack[0]
    
    @property
    def _current_scope(self) -> Dict[tuple, KVReg]:
        """Get the current (innermost) scope."""
        return self._scope_stack[-1]
    
    def bind_symbol(self, name: str, reg):
        """Bind a symbol name to a register."""
        self._bindings[name] = reg
    
    def push_scope(self, name: str = "region"):
        """Push a new CSE scope for a structured region."""
        self._scope_stack.append({})
    
    def pop_scope(self):
        """Pop the current CSE scope (leaving at least global scope)."""
        if len(self._scope_stack) > 1:
            self._scope_stack.pop()
    
    def scope(self, name: str = "region"):
        """Context manager for scoped CSE regions."""
        return _ScopeContext(self, name)
    
    def clear_cache(self):
        """
        Clear the expression cache completely.
        
        This forces fresh register allocation for subsequent expressions,
        preventing cached virtual registers from being reused. This is important
        when there's a risk of register clobbering between separate operations
        (e.g., multiple store operations where address computation for later
        stores might clobber registers holding data for earlier stores).
        """
        # Clear everything including global scope
        self._scope_stack = [{}]
    
    def _lookup_cache(self, key: tuple) -> Optional[KVReg]:
        """Look up a key in all active scopes (innermost to outermost)."""
        for scope in reversed(self._scope_stack):
            if key in scope:
                return scope[key]
        return None
    
    def _insert_cache(self, key: tuple, reg: KVReg, global_scope: bool = False):
        """Insert into the appropriate scope."""
        if global_scope:
            self._global_scope[key] = reg
        else:
            self._current_scope[key] = reg
    
    # Known loop-invariant symbols (thread IDs and workgroup IDs)
    _LOOP_INVARIANT_SYMBOLS = frozenset([
        'tid_x', 'tid_y', 'tid_z',
        'wgid_x', 'wgid_y', 'wgid_z',
        'lane_id',
    ])
    
    def _is_loop_invariant(self, expr) -> bool:
        """
        Check if an expression is loop-invariant.
        
        An expression is loop-invariant if ALL its free symbols are known
        loop-invariant values (tid_x, tid_y, wgid_x, etc.).
        
        Loop-varying values (like SGPR loop counters: s24, s25) make
        an expression loop-varying, so it should NOT be cached.
        
        Returns:
            True if the expression is loop-invariant and safe to cache globally
        """
        import sympy
        from sympy import Integer
        
        # Constants are always loop-invariant
        if isinstance(expr, (int, Integer)):
            return True
        
        # Check all free symbols in the expression
        if hasattr(expr, 'free_symbols'):
            for sym in expr.free_symbols:
                name = str(sym)
                # Check if it's a known invariant symbol
                if name in self._LOOP_INVARIANT_SYMBOLS:
                    continue
                # Check if it's bound to a physical SGPR (loop counter)
                # SGPR references like s24, s25 are loop counters - NOT invariant
                if name.startswith('s') and name[1:].isdigit():
                    return False
                # Check bindings
                if name in self._bindings:
                    binding = self._bindings[name]
                    # If bound to a physical SGPR reference, it's loop-varying
                    if isinstance(binding, str) and binding.startswith('s'):
                        return False
                    # If bound to a KVReg or physical VGPR string, treat as invariant
                    # (VGPRs are per-thread, not loop-varying)
                    continue
                # Unknown symbols - conservatively treat as NOT loop-invariant
                return False
            return True
        
        # Single symbol
        if isinstance(expr, sympy.Symbol):
            name = str(expr)
            if name in self._LOOP_INVARIANT_SYMBOLS:
                return True
            if name.startswith('s') and name[1:].isdigit():
                return False
            # Unknown - conservative
            return False
        
        # Default: conservative
        return False
    
    def _expr_key(self, expr) -> tuple:
        """
        Create a structural cache key for an expression.
        
        This enables CSE across structurally identical expressions.
        """
        import sympy
        from sympy import Symbol, Mul, Add, Integer, floor, Mod
        
        if isinstance(expr, (int, Integer)):
            return ('imm', int(expr))
        
        if isinstance(expr, Symbol):
            return ('sym', str(expr))
        
        if isinstance(expr, Mul):
            # Sort args for canonical ordering
            return ('mul',) + tuple(sorted(self._expr_key(a) for a in expr.args))
        
        if isinstance(expr, Add):
            return ('add',) + tuple(sorted(self._expr_key(a) for a in expr.args))
        
        if isinstance(expr, Mod):
            return ('mod', self._expr_key(expr.args[0]), self._expr_key(expr.args[1]))
        
        if getattr(expr, 'func', None) == floor:
            return ('floor', self._expr_key(expr.args[0]))
        
        # Fallback: use string representation (less efficient but always works)
        return ('expr', str(expr))
    
    def _get_symbol_bounds(self) -> Dict[Any, Tuple[int, int]]:
        """
        Get bounds for thread/workgroup ID symbols.
        
        These bounds enable algebraic simplifications like:
        - floor(tid_x / 64) → 0 when tid_x < 64
        - Mod(tid_x, 64) → tid_x when tid_x < 64
        
        Bounds are derived from the kernel's configuration:
        - tid_x/tid_y/tid_z: Upper bounded by workgroup dimensions
        - lane_id: Bounded by subgroup_size
        - wgid_*: Large upper bound (grid dimensions unknown at compile time)
        """
        import sympy
        bounds = {}
        
        # Use actual bounds from kernel configuration
        # tid_ub_* are exclusive upper bounds, so max value is tid_ub_* - 1
        tid_ub_x = getattr(self.ctx, 'tid_ub_x', 64)
        tid_ub_y = getattr(self.ctx, 'tid_ub_y', 1)
        tid_ub_z = getattr(self.ctx, 'tid_ub_z', 1)
        subgroup_size = getattr(self.ctx, 'subgroup_size', 64)
        
        # Thread IDs - bounded by workgroup dimensions
        bounds[sympy.Symbol('tid_x')] = (0, tid_ub_x - 1)
        bounds[sympy.Symbol('tid_y')] = (0, max(0, tid_ub_y - 1))
        bounds[sympy.Symbol('tid_z')] = (0, max(0, tid_ub_z - 1))
        
        # Lane ID bounded by subgroup size
        bounds[sympy.Symbol('lane_id')] = (0, subgroup_size - 1)
        
        # Workgroup IDs - bounded by grid dimensions (large upper bound)
        bounds[sympy.Symbol('wgid_x')] = (0, 65535)
        bounds[sympy.Symbol('wgid_y')] = (0, 65535)
        bounds[sympy.Symbol('wgid_z')] = (0, 65535)
        
        return bounds
    
    def get_or_emit(self, expr, cache_in_global: bool = False) -> KVReg:
        """
        Get a VGPR containing the value of expr, emitting instructions if needed.
        
        Uses scoped CSE to avoid recomputing the same expression.
        
        Args:
            expr: The expression to emit
            cache_in_global: If True, cache in global scope (for prologue values)
        """
        import sympy
        from sympy import Symbol, Mul, Add, Integer, floor, Mod
        
        # Track recursion depth
        self._emit_depth += 1
        try:
            return self._get_or_emit_impl(expr, cache_in_global)
        finally:
            self._emit_depth -= 1
    
    def _get_or_emit_impl(self, expr, cache_in_global: bool = False) -> KVReg:
        """Internal implementation of get_or_emit."""
        import sympy
        from sympy import Symbol, Mul, Add, Integer, floor, Mod
        
        # Algebraic simplification (disabled by default)
        # Enable via WAVE_KERNEL_IR_SIMPLIFY=1 or _ENABLE_KERNEL_IR_SIMPLIFY=True
        # IMPORTANT: Only simplify at top-level (depth=1), not during recursive calls.
        # This prevents the O(n^2) behavior where each sub-expression is simplified.
        if _ENABLE_KERNEL_IR_SIMPLIFY and self._emit_depth == 1:
            from .expr_simplify import simplify_for_emission
            bounds = self._get_symbol_bounds()
            expr = simplify_for_emission(expr, bounds)
        
        # Check cache for ALL expressions (after simplification)
        # This avoids re-emitting the same expression multiple times
        cache_key = self._expr_key(expr)
        cached = self._lookup_cache(cache_key)
        if cached is not None:
            return cached
        
        # Determine if this expression is loop-invariant (for global caching)
        is_invariant = self._is_loop_invariant(expr)
        
        # Handle immediate integers
        if isinstance(expr, (int, Integer)):
            val = int(expr)
            result = self.ctx.vreg()
            self.ctx.program.emit(KInstr(
                "v_mov_b32", (result,), (KImm(val),),
                comment=f"imm = {val}"
            ))
            self._insert_cache(cache_key, result, global_scope=is_invariant)
            return result
        
        # Handle symbols
        if isinstance(expr, Symbol):
            name = str(expr)
            
            # Check bindings first
            if name in self._bindings:
                reg = self._bindings[name]
                if isinstance(reg, KVReg):
                    return reg
                # String like "v0" - need to copy to virtual reg
                if isinstance(reg, str) and reg.startswith('v'):
                    phys_idx = int(reg[1:])
                    result = self.ctx.vreg()
                    self.ctx.program.emit(KInstr(
                        "v_mov_b32", (result,), (KPhysVReg(phys_idx),),
                        comment=f"copy {name} from {reg}"
                    ))
                    self._insert_cache(cache_key, result, global_scope=is_invariant)
                    return result
            
            # Common thread ID symbols - emit inline on first use, cache in GLOBAL scope
            # (Cache check already done at top of function)
            if name == 'tid_x':
                if self.ctx.use_flat_tid:
                    # Multi-wave: extract tid_x from flat_tid (v0[0:9])
                    result = self.ctx.vreg()
                    self.ctx.program.emit(KInstr(
                        "v_bfe_u32", (result,), (KPhysVReg(0), KImm(0), KImm(10)),
                        comment="extract tid_x from flat_tid"
                    ))
                else:
                    # Single-wave: compute lane_id using v_mbcnt
                    lo_result = self.ctx.vreg()
                    result = self.ctx.vreg()
                    self.ctx.program.emit(KInstr(
                        "v_mbcnt_lo_u32_b32", (lo_result,), (KImm(-1), KImm(0)),
                        comment="lane_id low"
                    ))
                    self.ctx.program.emit(KInstr(
                        "v_mbcnt_hi_u32_b32", (result,), (KImm(-1), lo_result),
                        comment="lane_id = tid_x for single-wave"
                    ))
                
                # Cache in GLOBAL scope so it survives clear_cache()
                self._insert_cache(cache_key, result, global_scope=True)
                return result
            
            if name == 'tid_y':
                if self.ctx.use_flat_tid:
                    # Multi-wave: extract tid_y from flat_tid (v0[10:19])
                    result = self.ctx.vreg()
                    self.ctx.program.emit(KInstr(
                        "v_bfe_u32", (result,), (KPhysVReg(0), KImm(10), KImm(10)),
                        comment="extract tid_y from flat_tid"
                    ))
                else:
                    # Single-wave: tid_y is 0
                    result = self.ctx.vreg()
                    self.ctx.program.emit(KInstr(
                        "v_mov_b32", (result,), (KImm(0),),
                        comment="tid_y = 0 for single-wave"
                    ))
                
                # Cache in GLOBAL scope (loop-invariant)
                self._insert_cache(cache_key, result, global_scope=True)
                return result
            
            # Handle workgroup ID symbols - also cache in global scope
            if name == 'wgid_x':
                result = self.ctx.vreg()
                self.ctx.program.emit(KInstr(
                    "v_mov_b32", (result,), (KPhysSReg(2),),
                    comment="wgid_x from s2"
                ))
                self._insert_cache(cache_key, result, global_scope=True)
                return result
            
            if name == 'wgid_y':
                result = self.ctx.vreg()
                self.ctx.program.emit(KInstr(
                    "v_mov_b32", (result,), (KPhysSReg(3),),
                    comment="wgid_y from s3"
                ))
                self._insert_cache(cache_key, result, global_scope=True)
                return result
            
            if name == 'wgid_z':
                result = self.ctx.vreg()
                self.ctx.program.emit(KInstr(
                    "v_mov_b32", (result,), (KPhysSReg(4),),
                    comment="wgid_z from s4"
                ))
                self._insert_cache(cache_key, result, global_scope=True)
                return result
            
            # Handle SGPR references (like loop counters: s8, s9, etc.)
            # NEVER cache these - loop counters change each iteration
            if name.startswith('s') and name[1:].isdigit():
                sgpr_idx = int(name[1:])
                result = self.ctx.vreg()
                self.ctx.program.emit(KInstr(
                    "v_mov_b32", (result,), (KPhysSReg(sgpr_idx),),
                    comment=f"broadcast {name} to VGPR"
                ))
                return result
            
            raise ValueError(f"Unknown symbol: {name}")
        
        # Handle multiplication
        if isinstance(expr, Mul):
            # Separate constant, rational, and variable parts
            const_part = 1
            divisor = 1  # For handling rational coefficients like 1/2
            var_parts = []
            for arg in expr.args:
                if isinstance(arg, Integer):
                    const_part *= int(arg)
                elif isinstance(arg, sympy.Rational) and not isinstance(arg, Integer):
                    # Handle rational like 1/2, 1/4, etc.
                    const_part *= int(arg.p)  # numerator
                    divisor *= int(arg.q)     # denominator
                elif arg.is_number and isinstance(arg, (int, float)):
                    const_part *= int(arg)
                else:
                    var_parts.append(arg)
            
            if len(var_parts) == 0:
                # Pure constant (possibly with division)
                if divisor > 1:
                    return self.get_or_emit(Integer(const_part // divisor))
                return self.get_or_emit(Integer(const_part))
            
            if len(var_parts) == 1:
                # const * var / divisor - common case
                var_reg = self.get_or_emit(var_parts[0])
                
                # Handle divisor first if present (represents rational coefficient)
                if divisor > 1:
                    # This is a division like var/2 -> shift right
                    if divisor > 0 and (divisor & (divisor - 1)) == 0:
                        shift = divisor.bit_length() - 1
                        div_result = self.ctx.vreg()
                        self.ctx.program.emit(KInstr(
                            "v_lshrrev_b32", (div_result,), (KImm(shift), var_reg),
                            comment=f"{var_parts[0]} >> {shift} (div by {divisor})"
                        ))
                        var_reg = div_result
                    else:
                        raise NotImplementedError(f"Non-power-of-2 divisor in Mul: {divisor}")
                
                if const_part == 1:
                    # No multiplication needed, just return the (possibly divided) result
                    if divisor > 1:
                        self._insert_cache(cache_key, var_reg, global_scope=is_invariant)
                    return var_reg
                
                result = self.ctx.vreg()
                
                # Check if const is power of 2 for shift
                if const_part > 0 and (const_part & (const_part - 1)) == 0:
                    shift = const_part.bit_length() - 1
                    self.ctx.program.emit(KInstr(
                        "v_lshlrev_b32", (result,), (KImm(shift), var_reg),
                        comment=f"{var_parts[0]} << {shift}"
                    ))
                elif -16 <= const_part <= 64:
                    # Inline constant range - can use immediate
                    self.ctx.program.emit(KInstr(
                        "v_mul_lo_u32", (result,), (var_reg, KImm(const_part)),
                        comment=f"{var_parts[0]} * {const_part}"
                    ))
                else:
                    # Large constant - need to materialize in a register first
                    const_reg = self.ctx.vreg()
                    self.ctx.program.emit(KInstr(
                        "v_mov_b32", (const_reg,), (KImm(const_part),),
                        comment=f"materialize {const_part}"
                    ))
                    self.ctx.program.emit(KInstr(
                        "v_mul_lo_u32", (result,), (var_reg, const_reg),
                        comment=f"{var_parts[0]} * {const_part}"
                    ))
                
                # Cache result (global scope if loop-invariant)
                self._insert_cache(cache_key, result, global_scope=is_invariant)
                return result
            
            # Multiple variable parts - emit sequentially
            result = self.get_or_emit(var_parts[0])
            for v in var_parts[1:]:
                v_reg = self.get_or_emit(v)
                new_result = self.ctx.vreg()
                self.ctx.program.emit(KInstr(
                    "v_mul_lo_u32", (new_result,), (result, v_reg),
                    comment="mul"
                ))
                result = new_result
            
            if const_part != 1:
                final = self.ctx.vreg()
                if -16 <= const_part <= 64:
                    self.ctx.program.emit(KInstr(
                        "v_mul_lo_u32", (final,), (result, KImm(const_part)),
                        comment=f"* {const_part}"
                    ))
                else:
                    const_reg = self.ctx.vreg()
                    self.ctx.program.emit(KInstr(
                        "v_mov_b32", (const_reg,), (KImm(const_part),),
                        comment=f"materialize {const_part}"
                    ))
                    self.ctx.program.emit(KInstr(
                        "v_mul_lo_u32", (final,), (result, const_reg),
                        comment=f"* {const_part}"
                    ))
                result = final
            
            # Cache result (global scope if loop-invariant)
            self._insert_cache(cache_key, result, global_scope=is_invariant)
            return result
        
        # Handle addition
        if isinstance(expr, Add):
            # Emit first term
            result = self.get_or_emit(expr.args[0])
            
            # Add remaining terms
            for arg in expr.args[1:]:
                arg_reg = self.get_or_emit(arg)
                new_result = self.ctx.vreg()
                self.ctx.program.emit(KInstr(
                    "v_add_u32", (new_result,), (result, arg_reg),
                    comment="add"
                ))
                result = new_result
            
            # Cache result (global scope if loop-invariant)
            self._insert_cache(cache_key, result, global_scope=is_invariant)
            return result
        
        # Handle floor expressions
        if isinstance(expr, floor):
            result = self._emit_floor(expr)
            
            # Cache result (global scope if loop-invariant)
            self._insert_cache(cache_key, result, global_scope=is_invariant)
            return result
        
        # Handle modulo
        if isinstance(expr, Mod):
            x, n = expr.args
            x_reg = self.get_or_emit(x)
            n_val = int(n)
            
            result = self.ctx.vreg()
            
            # Check if n is power of 2 for AND
            if n_val > 0 and (n_val & (n_val - 1)) == 0:
                mask = n_val - 1
                self.ctx.program.emit(KInstr(
                    "v_and_b32", (result,), (KImm(mask), x_reg),
                    comment=f"mod {n_val} (and)"
                ))
            else:
                # TODO: Implement general modulo
                raise NotImplementedError(f"modulo by {n_val} not yet implemented")
            
            # Cache result (global scope if loop-invariant)
            self._insert_cache(cache_key, result, global_scope=is_invariant)
            return result
        
        # Handle rational numbers (like Half = 1/2)
        # These can appear when simplification extracts common factors from expressions
        # In our integer-only arithmetic, rationals are handled via floor semantics:
        # - Standalone rational like 1/2: floor(1/2) = 0
        # - Multiplication like tid_y * 1/2: handled in Mul case as tid_y >> 1
        if isinstance(expr, sympy.Rational) and not isinstance(expr, Integer):
            if expr.q == 1:  # Integer in disguise (like 3/1)
                return self.get_or_emit(Integer(expr.p))
            # Pure fractional rational: use floor semantics
            # This is correct because all our intermediate values are integers
            floor_val = int(expr.p) // int(expr.q)
            return self.get_or_emit(Integer(floor_val))
        
        raise NotImplementedError(f"Expression type not supported: {type(expr).__name__}: {expr}")
    
    def _emit_floor(self, expr) -> KVReg:
        """
        Emit a floor expression.
        
        Handles:
        - floor(x / n) where n is power-of-2 -> shift right
        - floor(Add(...)) with fractions -> find common denominator and shift
        - floor(floor(...)) -> emit inner floor (nested floors are no-op)
        - floor(integer) -> no-op
        - floor(symbol) -> no-op (symbols are integers in our context)
        """
        import sympy
        from sympy import floor, Mul, Add, Integer, Rational
        from math import gcd
        
        inner = expr.args[0]
        
        # floor(Mul(...)) - floor division pattern
        if isinstance(inner, Mul):
            return self._emit_floor_div(inner)
        
        # floor(Add(...)) - need to handle terms with fractions
        if isinstance(inner, Add):
            return self._emit_floor_add(inner)
        
        # floor(floor(...)) - just emit the inner floor
        if getattr(inner, 'func', None) == floor:
            return self._emit_floor(inner)
        
        # floor(Integer) - just return the integer
        if isinstance(inner, Integer):
            return self.get_or_emit(int(inner))
        
        # floor(Symbol) - symbols are integers, no-op
        if isinstance(inner, sympy.Symbol):
            return self.get_or_emit(inner)
        
        # floor(Mod) - emit the mod (Mod produces integers)
        if isinstance(inner, sympy.Mod):
            return self.get_or_emit(inner)
        
        raise NotImplementedError(f"floor expression not supported: {expr}")
    
    def _emit_floor_div(self, mul_expr) -> KVReg:
        """
        Emit floor(x * 1/n) = floor(x/n).
        
        Parses a Mul expression to find the numerator and divisor,
        then emits a right shift if divisor is power-of-2.
        """
        import sympy
        from sympy import Integer, Rational, Pow
        
        x_part = None
        divisor = None
        const_coeff = 1
        
        for factor in mul_expr.args:
            if isinstance(factor, Pow) and factor.args[1] == -1:
                # x^(-1) = 1/x
                divisor = factor.args[0]
            elif isinstance(factor, Rational) and not isinstance(factor, Integer):
                # p/q rational
                if factor.p != 1:
                    const_coeff *= int(factor.p)
                divisor = Integer(factor.q)
            elif isinstance(factor, Integer):
                const_coeff *= int(factor)
            else:
                if x_part is None:
                    x_part = factor
                else:
                    x_part = x_part * factor
        
        if x_part is None:
            # Pure constant division
            if divisor is not None and isinstance(divisor, Integer):
                return self.get_or_emit(const_coeff // int(divisor))
            return self.get_or_emit(const_coeff)
        
        if divisor is None:
            # No division, just emit the expression
            if const_coeff != 1:
                return self.get_or_emit(x_part * Integer(const_coeff))
            return self.get_or_emit(x_part)
        
        if not isinstance(divisor, Integer):
            raise NotImplementedError(f"Non-integer divisor not supported: {divisor}")
        
        div_val = int(divisor)
        if div_val <= 0 or (div_val & (div_val - 1)) != 0:
            raise NotImplementedError(f"Floor divisor must be power-of-2: {div_val}")
        
        # Emit x * const_coeff first
        if const_coeff == 1:
            num_reg = self.get_or_emit(x_part)
        else:
            num_reg = self.get_or_emit(x_part * Integer(const_coeff))
        
        # Then shift right
        shift = div_val.bit_length() - 1
        if shift > 0:
            result = self.ctx.vreg()
            self.ctx.program.emit(KInstr(
                "v_lshrrev_b32", (result,), (KImm(shift), num_reg),
                comment=f"floor div by {div_val} (shift)"
            ))
            return result
        
        return num_reg
    
    def _emit_floor_add(self, add_expr) -> KVReg:
        """
        Emit floor(Add(...)) with fractional terms.
        
        Algorithm:
        1. Parse each term to find numerator and denominator
        2. Find LCM of all denominators
        3. Scale each numerator by LCM/denominator
        4. Sum all scaled numerators
        5. Shift right by log2(LCM)
        
        Examples:
        - floor(a/16 + 1/2) = floor((a + 8)/16) = (a + 8) >> 4
        - floor(tid_y/2 + floor(tid_x/8)/32) = (16*tid_y + floor(tid_x/8)) >> 5
        """
        import sympy
        from sympy import Integer, Rational, Mul, floor
        from math import gcd
        
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        
        def get_num_denom(term):
            """Extract (numerator_expr, denominator) from a term."""
            # Pure integer
            if isinstance(term, Integer):
                return (Integer(int(term)), 1)
            
            # Pure rational like 1/2
            if isinstance(term, Rational) and not isinstance(term, Integer):
                return (Integer(int(term.p)), int(term.q))
            
            # Symbol (integer)
            if isinstance(term, sympy.Symbol):
                return (term, 1)
            
            # floor(...) - produces integer
            if getattr(term, 'func', None) == floor:
                return (term, 1)
            
            # Mod - produces integer
            if isinstance(term, sympy.Mod):
                return (term, 1)
            
            # Mul - look for rational factor
            if isinstance(term, Mul):
                divisor = 1
                numerator_parts = []
                
                for factor in term.args:
                    if isinstance(factor, Rational) and not isinstance(factor, Integer):
                        # p/q rational
                        if factor.p != 1:
                            numerator_parts.append(Integer(int(factor.p)))
                        divisor = int(factor.q)
                    elif isinstance(factor, sympy.Pow) and factor.args[1] == -1:
                        # x^(-1) = 1/x
                        if isinstance(factor.args[0], Integer):
                            divisor = int(factor.args[0])
                        else:
                            raise NotImplementedError(f"Non-integer inverse: {factor}")
                    else:
                        numerator_parts.append(factor)
                
                if numerator_parts:
                    if len(numerator_parts) == 1:
                        return (numerator_parts[0], divisor)
                    else:
                        return (Mul(*numerator_parts), divisor)
                else:
                    return (Integer(1), divisor)
            
            # Default: treat as integer
            return (term, 1)
        
        # Parse all terms
        terms = []
        for term in add_expr.args:
            num, denom = get_num_denom(term)
            terms.append((num, denom))
        
        # Find LCM of all denominators
        common_denom = 1
        for _, denom in terms:
            common_denom = lcm(common_denom, denom)
        
        # If all denominators are 1, just emit the add (no floor needed)
        if common_denom == 1:
            return self.get_or_emit(add_expr)
        
        # Check LCM is power of 2
        if common_denom <= 0 or (common_denom & (common_denom - 1)) != 0:
            raise NotImplementedError(f"Floor add requires power-of-2 LCM: {common_denom}")
        
        # Build scaled sum: sum of (num * (common_denom / denom))
        scaled_sum = None
        for num, denom in terms:
            scale = common_denom // denom
            if scale == 1:
                scaled_term = num
            elif isinstance(num, Integer):
                scaled_term = Integer(int(num) * scale)
            else:
                scaled_term = num * Integer(scale)
            
            if scaled_sum is None:
                scaled_sum = scaled_term
            else:
                scaled_sum = scaled_sum + scaled_term
        
        # Emit the scaled sum
        sum_reg = self.get_or_emit(scaled_sum)
        
        # Shift right by log2(common_denom)
        shift = common_denom.bit_length() - 1
        result = self.ctx.vreg()
        self.ctx.program.emit(KInstr(
            "v_lshrrev_b32", (result,), (KImm(shift), sum_reg),
            comment=f"floor add by {common_denom} (shift)"
        ))
        
        return result


# =============================================================================
# Operand type to register allocation info
# =============================================================================

def _get_def_info(operand_types: Tuple[OperandType, ...]) -> Tuple[str, int, int]:
    """
    Get destination register info from operand types.
    
    Returns: (class, count, alignment) where:
        - class: 'v' for VGPR, 's' for SGPR, None for no destination
        - count: number of registers (1, 2, 4, 16)
        - alignment: alignment requirement
    """
    for ot in operand_types:
        if ot == OperandType.VGPR:
            return ('v', 1, 1)
        elif ot == OperandType.VGPR_PAIR:
            return ('v', 2, 2)
        elif ot == OperandType.VGPR_QUAD:
            return ('v', 4, 4)
        elif ot == OperandType.VGPR_16:
            return ('v', 16, 4)
        elif ot == OperandType.SGPR:
            return ('s', 1, 1)
        elif ot == OperandType.SGPR_PAIR:
            return ('s', 2, 2)
        elif ot == OperandType.SGPR_QUAD:
            return ('s', 4, 4)
    return (None, 0, 1)


@dataclass
class KernelCompilationContext:
    """
    Context for kernel compilation with whole-program register allocation.
    
    This context manages:
    - The KernelProgram being built
    - Symbol bindings (MLIR SSA values to virtual registers)
    - ABI register configuration
    - CSE cache for expression deduplication
    
    Instructions are emitted via dynamic dispatch:
        ctx.v_add_u32(src0, src1)  # Calls _emit_instruction("v_add_u32", ...)
    
    Usage:
        ctx = KernelCompilationContext(max_vgprs=256, max_sgprs=104)
        
        # Emit instructions - methods generated dynamically
        v1 = ctx.v_mov_b32(42)
        v2 = ctx.v_add_u32(v1, 100)
        ctx.ds_read_b64(addr)
        
        # Finalize and get assembly
        asm_lines = ctx.finalize()
    """
    
    # Configuration
    max_vgprs: int = 256
    max_sgprs: int = 104
    
    # ABI configuration
    use_flat_tid: bool = True
    use_workgroup_ids: Tuple[bool, bool, bool] = (True, True, True)  # x, y, z
    
    # Thread ID bounds for algebraic simplification
    # These are set from kernel_info when the context is created
    tid_ub_x: int = 64  # Upper bound for tid_x (exclusive)
    tid_ub_y: int = 1   # Upper bound for tid_y (exclusive)
    tid_ub_z: int = 1   # Upper bound for tid_z (exclusive)
    subgroup_size: int = 64
    wg_size: Tuple[int, int, int] = (64, 1, 1)
    
    # Internal state
    program: KernelProgram = field(init=False)
    builder: KernelBuilder = field(init=False)
    _unified: UnifiedEmitter = field(init=False)
    _registry: InstructionRegistry = field(init=False)
    
    # Symbol bindings: MLIR SSA value string -> virtual register
    _symbol_bindings: Dict[str, KReg] = field(default_factory=dict, init=False)
    
    # SSA to register mapping for MLIR SSA values
    # Maps SSA value string to tuple of virtual registers (for multi-reg results like loads)
    # This is the source of truth for all SSA-to-register tracking
    ssa_to_reg: Dict[str, Tuple[KReg, ...]] = field(default_factory=dict, init=False)
    
    # CSE cache: expression key -> virtual register
    _cse_cache: Dict[tuple, KVReg] = field(default_factory=dict, init=False)
    
    # SRD ranges: memref_ssa -> SGPR quad range for buffer descriptor
    srd_ranges: Dict[str, KRegRange] = field(default_factory=dict, init=False)
    
    # Pending SRD setups: list of (srd_range, arg_idx, limit_bytes)
    # These are queued during IR emission and inserted at program start in finalize()
    _pending_srd_setups: List[Tuple[KRegRange, int, int]] = field(default_factory=list, init=False)
    
    # Pending ABI prologue instructions: list of KInstr
    # These define tid_x, tid_y, wgid_* etc in the prologue for proper dominance
    _pending_abi_prologue: List[KInstr] = field(default_factory=list, init=False)
    
    # Prologue-defined ABI VGPRs (allocated early, defined in prologue)
    _prologue_tid_x: Optional[KVReg] = field(default=None, init=False)
    _prologue_tid_y: Optional[KVReg] = field(default=None, init=False)
    _prologue_wgid_x: Optional[KVReg] = field(default=None, init=False)
    _prologue_wgid_y: Optional[KVReg] = field(default=None, init=False)
    _prologue_wgid_z: Optional[KVReg] = field(default=None, init=False)
    
    # Loop management
    _loop_counter: int = field(default=0, init=False)
    _loop_stack: List[dict] = field(default_factory=list, init=False)
    
    # Ticketing for memory operations
    _vmem_ticket: int = field(default=0, init=False)
    _lgkm_ticket: int = field(default=0, init=False)
    
    # Statistics
    _cse_hits: int = field(default=0, init=False)
    
    def __post_init__(self):
        # Initialize ABI
        abi = KernelABI()
        if self.use_flat_tid:
            abi.flat_tid_vreg = KPhysVReg(0)
        if self.use_workgroup_ids[0]:
            abi.workgroup_id_x_sreg = KPhysSReg(2)
        if self.use_workgroup_ids[1]:
            abi.workgroup_id_y_sreg = KPhysSReg(3)
        if self.use_workgroup_ids[2]:
            abi.workgroup_id_z_sreg = KPhysSReg(4)
        
        # Create program
        self.program = KernelProgram(abi=abi, max_vgprs=self.max_vgprs, max_sgprs=self.max_sgprs)
        self.builder = KernelBuilder(self.program)
        
        # Load instruction registry
        self._registry = get_registry("common")
        
        # Create unified emitter in KERNEL_IR mode
        # This allows callers to use kernel_ctx.unified.v_add_u32(...) syntax
        self._unified = UnifiedEmitter(
            architecture="common",
            mode=EmissionMode.KERNEL_IR,
            context=self,
        )
        
        # Expression emitter for sympy expressions (lazy init)
        self._expr_emitter = None
    
    @property
    def expr_emitter(self) -> 'KernelIRExprEmitter':
        """Get the expression emitter for this context."""
        if self._expr_emitter is None:
            self._expr_emitter = KernelIRExprEmitter(self)
        return self._expr_emitter
    
    def update_bounds_from_kernel_info(self, kernel_info: "KernelInfo") -> None:
        """
        Update thread ID bounds from kernel_info after MLIR parsing.
        
        This should be called after interpret_func() returns, as that's when
        we know the actual workgroup_size and subgroup_size from the MLIR.
        
        Args:
            kernel_info: KernelInfo populated by interpret_func()
        """
        wg_size = getattr(kernel_info, 'wg_size', (64, 1, 1))
        subgroup_size = getattr(kernel_info, 'subgroup_size', 64)
        
        self.wg_size = wg_size
        self.subgroup_size = subgroup_size
        
        # Update use_flat_tid based on actual num_waves
        num_waves = max(1, wg_size[0] * wg_size[1] * wg_size[2] // subgroup_size)
        self.use_flat_tid = (num_waves > 1)
        
        # In multi-wave mode, tid_x/tid_y/tid_z are extracted from flat_tid
        # and bounded by the workgroup dimensions wg_size = (x, y, z)
        # In single-wave mode, tid_x = lane_id (0 to subgroup_size-1) and tid_y/z = 0
        if self.use_flat_tid:
            # Multi-wave: unpack wg_size tuple for tid bounds
            self.tid_ub_x = wg_size[0]
            self.tid_ub_y = wg_size[1]
            self.tid_ub_z = wg_size[2] if len(wg_size) > 2 else 1
        else:
            # Single-wave: tid_x = lane_id (bounded by subgroup_size), tid_y/z = 0
            self.tid_ub_x = subgroup_size
            self.tid_ub_y = 1
            self.tid_ub_z = 1
    
    # =========================================================================
    # Dynamic instruction dispatch
    # =========================================================================
    
    def __getattr__(self, name: str) -> Any:
        """
        Dynamic dispatch for instruction methods.
        
        When ctx.v_add_u32(...) is called and v_add_u32 isn't explicitly defined,
        this method handles it by:
        1. Looking up the instruction in the registry
        2. Allocating destination registers based on operand types
        3. Emitting a KInstr with the instruction name
        """
        # Avoid recursion on internal attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Check if it's an instruction in the registry
        instr_def = self._registry.get(name)
        if instr_def is not None:
            # Create and return an emission method
            def emit_method(*args, comment: str = None, **kwargs):
                return self._emit_instruction(name, instr_def, args, kwargs, comment)
            return emit_method
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def _emit_instruction(
        self,
        name: str,
        instr_def: InstructionDef,
        args: tuple,
        kwargs: dict,
        comment: str,
    ) -> Optional[Any]:
        """
        Emit an instruction with automatic register allocation.
        
        This method:
        1. Allocates destination registers based on operand types
        2. Emits a KInstr to the program with the instruction name
        3. Returns the destination register(s) if any
        """
        # Determine destination type and allocate
        dst = None
        defs = ()
        
        if instr_def.defs:
            # Get the first def's type info
            def_op = instr_def.defs[0]
            reg_class, count, alignment = _get_def_info(def_op.types)
            
            if reg_class == 'v':
                if count == 1:
                    dst = self.vreg()
                else:
                    dst = self.program.alloc_vreg_range(count, alignment=alignment)
                defs = (dst,)
            elif reg_class == 's':
                if count == 1:
                    dst = self.sreg()
                else:
                    dst = self.program.alloc_sreg_range(count, alignment=alignment)
                defs = (dst,)
        
        # Build uses from args and kwargs
        # Parse positional args first, then fill in missing operands from kwargs
        uses = []
        arg_idx = 0
        
        for use_op in instr_def.uses:
            if arg_idx < len(args):
                # Use positional arg
                value = args[arg_idx]
                arg_idx += 1
            elif use_op.name in kwargs:
                # Use kwarg
                value = kwargs[use_op.name]
            elif use_op.optional:
                # Skip optional operands with no value
                continue
            else:
                # Required operand missing
                value = None
            
            if value is not None:
                # Handle special operand types
                from .instruction_registry import OperandType
                if OperandType.OFFSET in use_op.types and isinstance(value, int):
                    uses.append(KMemOffset(value))
                elif isinstance(value, int):
                    # Let kernel_generator handle raw ints
                    uses.append(value)
                else:
                    uses.append(value)
        
        # Also add any remaining positional args (for instructions not fully defined)
        while arg_idx < len(args):
            value = args[arg_idx]
            uses.append(value)
            arg_idx += 1
        
        # Issue tickets for memory operations
        mnemonic = instr_def.mnemonic if instr_def else name
        category = categorize_instruction(mnemonic)
        if category == InstructionCategory.VMEM:
            self._vmem_ticket += 1
        elif category == InstructionCategory.LGKM:
            self._lgkm_ticket += 1
        
        # Emit the instruction using name string
        self.program.emit(KInstr(name, defs, tuple(uses), comment=comment))
        
        return dst
    
    # =========================================================================
    # Ticketing for memory operations
    # =========================================================================
    
    def get_vmem_ticket(self) -> int:
        """Get the current VMEM ticket (number of outstanding VMEM ops)."""
        return self._vmem_ticket
    
    def get_lgkm_ticket(self) -> int:
        """Get the current LGKM ticket (number of outstanding LDS/scalar ops)."""
        return self._lgkm_ticket
    
    def wait_vmem(self, count: int = 0):
        """Emit s_waitcnt vmcnt(count) to wait for VMEM operations."""
        self.program.emit(KInstr(
            "s_waitcnt", (), (f"vmcnt({count})",),
            comment="wait for VMEM"
        ))
    
    def wait_lgkm(self, count: int = 0):
        """Emit s_waitcnt lgkmcnt(count) to wait for LDS/scalar operations."""
        self.program.emit(KInstr(
            "s_waitcnt", (), (f"lgkmcnt({count})",),
            comment="wait for LGKM"
        ))
    
    # =========================================================================
    # Symbol binding (for MLIR SSA value tracking)
    # =========================================================================
    
    def bind_symbol(self, symbol: str, reg: KReg) -> None:
        """Bind an MLIR SSA value name to a virtual register."""
        self._symbol_bindings[symbol] = reg
    
    def get_binding(self, symbol: str) -> Optional[KReg]:
        """Get the virtual register bound to an MLIR SSA value."""
        return self._symbol_bindings.get(symbol)
    
    def require_binding(self, symbol: str) -> KReg:
        """Get the virtual register bound to an MLIR SSA value, or raise."""
        if symbol not in self._symbol_bindings:
            raise ValueError(f"Symbol '{symbol}' not bound to any register")
        return self._symbol_bindings[symbol]
    
    # =========================================================================
    # SSA-to-register mapping (for vector results)
    # =========================================================================
    
    def bind_ssa(self, ssa_value: str, regs: Tuple[KReg, ...]) -> None:
        """
        Bind an MLIR SSA value to a tuple of virtual registers.
        
        Used for operations that produce multi-register results like:
        - vector loads (dwordx2/x4 → 2/4 regs)
        - MFMA results (4 regs)
        
        Args:
            ssa_value: The SSA value string (e.g., "%12")
            regs: Tuple of virtual registers
        """
        self.ssa_to_reg[ssa_value] = regs
    
    def bind_ssa_single(self, ssa_value: str, reg: KReg) -> None:
        """Bind an SSA value to a single register (convenience method)."""
        self.ssa_to_reg[ssa_value] = (reg,)
    
    def bind_ssa_range(self, ssa_value: str, reg_range: KRegRange) -> None:
        """
        Bind an SSA value to a register range.
        
        Extracts individual registers from the range for proper tracking.
        """
        # For ranges, we track the base_reg and count
        # The actual tuple will be derived during allocation
        base = reg_range.base_reg
        if isinstance(base, KVReg):
            regs = tuple(KVReg(base.id + i) for i in range(reg_range.count))
        elif isinstance(base, KSReg):
            regs = tuple(KSReg(base.id + i) for i in range(reg_range.count))
        else:
            # Physical regs - store as-is
            regs = (reg_range,)
        self.ssa_to_reg[ssa_value] = regs
    
    def get_ssa_regs(self, ssa_value: str) -> Optional[Tuple[KReg, ...]]:
        """Get the virtual registers bound to an SSA value."""
        return self.ssa_to_reg.get(ssa_value)
    
    def require_ssa_regs(self, ssa_value: str) -> Tuple[KReg, ...]:
        """Get the virtual registers bound to an SSA value, or raise."""
        regs = self.ssa_to_reg.get(ssa_value)
        if regs is None:
            raise ValueError(
                f"SSA value '{ssa_value}' not bound to any registers. "
                f"Available: {list(self.ssa_to_reg.keys())}"
            )
        return regs
    
    # =========================================================================
    # CSE support
    # =========================================================================
    
    def cse_lookup(self, key: tuple) -> Optional[KVReg]:
        """Look up a value in the CSE cache."""
        return self._cse_cache.get(key)
    
    def cse_insert(self, key: tuple, reg: KVReg) -> None:
        """Insert a value into the CSE cache."""
        self._cse_cache[key] = reg
    
    def cse_get_or_emit(self, key: tuple, emit_fn) -> KVReg:
        """Get from CSE cache or emit using the provided function."""
        if key in self._cse_cache:
            self._cse_hits += 1
            return self._cse_cache[key]
        result = emit_fn()
        self._cse_cache[key] = result
        return result
    
    # =========================================================================
    # Register allocation helpers
    # =========================================================================
    
    def vreg(self) -> KVReg:
        """Allocate a new virtual VGPR."""
        return self.builder.vreg()
    
    def sreg(self) -> KSReg:
        """Allocate a new virtual SGPR."""
        return self.builder.sreg()
    
    def vreg_pair(self) -> KRegRange:
        """Allocate a pair of virtual VGPRs."""
        return self.program.alloc_vreg_range(2, alignment=2)
    
    def vreg_quad(self) -> KRegRange:
        """Allocate a quad of virtual VGPRs."""
        return self.program.alloc_vreg_range(4, alignment=4)
    
    def sreg_pair(self) -> KRegRange:
        """Allocate a pair of virtual SGPRs."""
        return self.program.alloc_sreg_range(2, alignment=2)
    
    def sreg_quad(self) -> KRegRange:
        """Allocate a quad of virtual SGPRs."""
        return self.program.alloc_sreg_range(4, alignment=4)
    
    # =========================================================================
    # Special emission methods (not auto-generated)
    # =========================================================================
    
    def emit(self, instr: KInstr):
        """Emit a raw instruction."""
        self.program.emit(instr)
    
    def emit_raw(self, asm_line: str):
        """Emit a raw assembly line (escape hatch)."""
        self.program.emit(KInstr("_raw_asm", (), (), comment=asm_line))
    
    def emit_label(self, label: str):
        """Emit a label."""
        self.program.emit(KInstr("_label", (), (), comment=label))
    
    def comment(self, text: str):
        """Emit a comment."""
        self.builder.comment(text)
    
    def s_mov_b32_to_m0(self, src, comment: str = None):
        """Emit s_mov_b32 m0, src (special: destination is M0)."""
        self.program.emit(KInstr("s_mov_b32", (M0,), (src,), comment=comment))
    
    def s_cbranch_scc1(self, label: str, comment: str = None):
        """Emit s_cbranch_scc1 (label stored in comment)."""
        self.program.emit(KInstr("s_cbranch_scc1", (), (), comment=label))
    
    def s_branch(self, label: str, comment: str = None):
        """Emit s_branch (label stored in comment)."""
        self.program.emit(KInstr("s_branch", (), (), comment=label))
    
    # =========================================================================
    # SRD Management (for kernel IR path)
    # =========================================================================
    
    def ensure_srd(self, memref_ssa: str, arg_idx: int, limit_bytes: int) -> KRegRange:
        """
        Ensure SRD is set up for a memref and return the SGPR range.

        If already set up, returns the cached range.
        Otherwise, allocates a new SGPR quad and queues SRD setup for the program prologue.
        
        SRD setup instructions are queued and emitted at the START of the program
        (during finalize) to ensure they're not inside loops.
        """
        if memref_ssa in self.srd_ranges:
            return self.srd_ranges[memref_ssa]

        # Allocate SGPR quad for SRD
        srd_range = self.sreg_quad()
        self.srd_ranges[memref_ssa] = srd_range

        # Queue the SRD setup for prologue emission
        self._pending_srd_setups.append((srd_range, arg_idx, limit_bytes))

        return srd_range
    
    def get_srd(self, memref_ssa: str) -> Optional[KRegRange]:
        """Get the SRD range for a memref if it exists."""
        return self.srd_ranges.get(memref_ssa)
    
    # =========================================================================
    # Prologue-Hoisted ABI Values
    # =========================================================================
    
    def ensure_tid_x(self) -> KVReg:
        """
        Get a VGPR containing tid_x, defined in the prologue.
        
        The VGPR is allocated immediately but the defining instruction
        is queued for prologue emission, ensuring it dominates all uses.
        """
        if self._prologue_tid_x is not None:
            return self._prologue_tid_x
        
        result = self.vreg()
        self._prologue_tid_x = result
        
        if self.use_flat_tid:
            # Multi-wave: extract tid_x from flat_tid (v0[0:9])
            self._pending_abi_prologue.append(KInstr(
                "v_bfe_u32", (result,), (KPhysVReg(0), KImm(0), KImm(10)),
                comment="extract tid_x from flat_tid"
            ))
        else:
            # Single-wave: compute lane_id using v_mbcnt
            lo_result = self.vreg()
            self._pending_abi_prologue.append(KInstr(
                "v_mbcnt_lo_u32_b32", (lo_result,), (KImm(-1), KImm(0)),
                comment="lane_id low"
            ))
            self._pending_abi_prologue.append(KInstr(
                "v_mbcnt_hi_u32_b32", (result,), (KImm(-1), lo_result),
                comment="lane_id = tid_x for single-wave"
            ))
        
        return result
    
    def ensure_tid_y(self) -> KVReg:
        """Get a VGPR containing tid_y, defined in the prologue."""
        if self._prologue_tid_y is not None:
            return self._prologue_tid_y
        
        result = self.vreg()
        self._prologue_tid_y = result
        
        if self.use_flat_tid:
            # Multi-wave: extract tid_y from flat_tid (v0[10:19])
            self._pending_abi_prologue.append(KInstr(
                "v_bfe_u32", (result,), (KPhysVReg(0), KImm(10), KImm(10)),
                comment="extract tid_y from flat_tid"
            ))
        else:
            # Single-wave: tid_y is 0
            self._pending_abi_prologue.append(KInstr(
                "v_mov_b32", (result,), (KImm(0),),
                comment="tid_y = 0 for single-wave"
            ))
        
        return result
    
    def ensure_wgid_x(self) -> KVReg:
        """Get a VGPR containing wgid_x, defined in the prologue."""
        if self._prologue_wgid_x is not None:
            return self._prologue_wgid_x
        
        result = self.vreg()
        self._prologue_wgid_x = result
        self._pending_abi_prologue.append(KInstr(
            "v_mov_b32", (result,), (KPhysSReg(2),),
            comment="wgid_x from s2"
        ))
        return result
    
    def ensure_wgid_y(self) -> KVReg:
        """Get a VGPR containing wgid_y, defined in the prologue."""
        if self._prologue_wgid_y is not None:
            return self._prologue_wgid_y
        
        result = self.vreg()
        self._prologue_wgid_y = result
        self._pending_abi_prologue.append(KInstr(
            "v_mov_b32", (result,), (KPhysSReg(3),),
            comment="wgid_y from s3"
        ))
        return result
    
    def ensure_wgid_z(self) -> KVReg:
        """Get a VGPR containing wgid_z, defined in the prologue."""
        if self._prologue_wgid_z is not None:
            return self._prologue_wgid_z
        
        result = self.vreg()
        self._prologue_wgid_z = result
        self._pending_abi_prologue.append(KInstr(
            "v_mov_b32", (result,), (KPhysSReg(4),),
            comment="wgid_z from s4"
        ))
        return result
    
    def _emit_srd_prologue(self):
        """
        Emit all queued SRD setup and ABI prologue instructions at the start of the program.
        
        This should be called at the beginning of finalize() to ensure
        SRD setup and ABI value definitions happen before any loops.
        """
        prologue_instrs = []
        
        # First, emit ABI prologue (tid_x, tid_y, wgid_x, etc.)
        # These must come first since they define values used throughout
        if self._pending_abi_prologue:
            prologue_instrs.extend(self._pending_abi_prologue)
            self._pending_abi_prologue = []
        
        # Then, emit SRD setup if any
        if self._pending_srd_setups:
            kernarg_pair = KRegRange(self.program.abi.kernarg_ptr_sreg_lo, 2)  # s[0:1]
            
            for srd_range, arg_idx, limit_bytes in self._pending_srd_setups:
                kernarg_offset = arg_idx * 8  # Each pointer is 8 bytes
                
                # Define the full SRD range (ensures 4-alignment for allocation)
                prologue_instrs.append(KInstr(
                    "_srd_define",  # Pseudo: defines the range for allocation purposes
                    (srd_range,),
                    (),
                    comment=f"Define SRD range for arg{arg_idx}"
                ))
                
                # Load base address into first 2 regs of the range
                prologue_instrs.append(KInstr(
                    "_srd_load_base",  # Pseudo: renders as s_load_dwordx2
                    (),
                    (srd_range, kernarg_pair, KImm(kernarg_offset)),
                    comment=f"Load base addr for arg{arg_idx}"
                ))
            
            # Wait for all SRD loads
            prologue_instrs.append(KInstr(
                "s_waitcnt", (), ("lgkmcnt(0)",), comment="wait for all SRD loads"
            ))
            
            # Fill SRD[2] and SRD[3] for each
            for srd_range, arg_idx, limit_bytes in self._pending_srd_setups:
                prologue_instrs.append(KInstr(
                    "_srd_fill_size",
                    (),
                    (srd_range, KImm(limit_bytes)),
                    comment=f"SRD size for arg{arg_idx}"
                ))
                prologue_instrs.append(KInstr(
                    "_srd_fill_stride",
                    (),
                    (srd_range, KImm(0x20000)),
                    comment=f"SRD stride for arg{arg_idx}"
                ))
            
            # Clear the pending list
            self._pending_srd_setups = []
        
        if not prologue_instrs:
            return
        
        # Insert prologue at the start of the program (after any comments)
        insert_pos = 0
        while insert_pos < len(self.program.instructions):
            instr = self.program.instructions[insert_pos]
            if instr.name != "_comment":
                break
            insert_pos += 1
        
        self.program.instructions[insert_pos:insert_pos] = prologue_instrs

    def emit_buffer_load(
        self,
        memref_ssa: str,
        vector_bytes: int,
        voffset: KVReg,
        inst_offset: int,
    ) -> Tuple[KRegRange, ...]:
        """
        Emit buffer load instruction(s).

        Returns tuple of destination register ranges.
        """
        srd_range = self.srd_ranges.get(memref_ssa)
        if srd_range is None:
            raise RuntimeError(f"SRD not set up for {memref_ssa}")

        result_ranges = []
        current_offset = inst_offset

        # Handle different vector sizes
        bytes_remaining = vector_bytes
        while bytes_remaining > 0:
            if bytes_remaining >= 16:
                # Use buffer_load_dwordx4
                dst_range = self.vreg_quad()
                self.program.emit(KInstr(
                    "buffer_load_dwordx4",
                    (dst_range,),
                    (voffset, srd_range, KImm(0), KMemOffset(current_offset)),
                    comment=f"load 16B @ offset {current_offset}"
                ))
                result_ranges.append(dst_range)
                bytes_remaining -= 16
                current_offset += 16
            elif bytes_remaining >= 8:
                # Use buffer_load_dwordx2
                dst_range = self.vreg_pair()
                self.program.emit(KInstr(
                    "buffer_load_dwordx2",
                    (dst_range,),
                    (voffset, srd_range, KImm(0), KMemOffset(current_offset)),
                    comment=f"load 8B @ offset {current_offset}"
                ))
                result_ranges.append(dst_range)
                bytes_remaining -= 8
                current_offset += 8
            else:
                # Use buffer_load_dword
                dst = self.vreg()
                self.program.emit(KInstr(
                    "buffer_load_dword",
                    (dst,),
                    (voffset, srd_range, KImm(0), KMemOffset(current_offset)),
                    comment=f"load 4B @ offset {current_offset}"
                ))
                result_ranges.append(KRegRange(dst, 1))
                bytes_remaining -= 4
                current_offset += 4

        return tuple(result_ranges)

    def emit_buffer_store(
        self,
        memref_ssa: str,
        src_ranges: Tuple[KRegRange, ...],
        voffset: KVReg,
        inst_offset: int,
    ):
        """Emit buffer store instruction(s)."""
        srd_range = self.srd_ranges.get(memref_ssa)
        if srd_range is None:
            raise RuntimeError(f"SRD not set up for {memref_ssa}")

        current_offset = inst_offset
        for src_range in src_ranges:
            count = src_range.count
            if count == 4:
                self.program.emit(KInstr(
                    "buffer_store_dwordx4",
                    (),
                    (src_range, voffset, srd_range, KImm(0), KMemOffset(current_offset)),
                    comment=f"store 16B @ offset {current_offset}"
                ))
                current_offset += 16
            elif count == 2:
                self.program.emit(KInstr(
                    "buffer_store_dwordx2",
                    (),
                    (src_range, voffset, srd_range, KImm(0), KMemOffset(current_offset)),
                    comment=f"store 8B @ offset {current_offset}"
                ))
                current_offset += 8
            else:
                self.program.emit(KInstr(
                    "buffer_store_dword",
                    (),
                    (src_range, voffset, srd_range, KImm(0), KMemOffset(current_offset)),
                    comment=f"store 4B @ offset {current_offset}"
                ))
                current_offset += 4

    # =========================================================================
    # LDS Read/Write Operations
    # =========================================================================
    
    def emit_lds_read_b64(
        self,
        dst_range: KRegRange,
        addr_vreg: KVReg,
        offset: int = 0,
    ):
        """Emit ds_read_b64 (LDS load of 8 bytes)."""
        self.program.emit(KInstr(
            "ds_read_b64",
            (dst_range,),
            (addr_vreg, KMemOffset(offset)),
            comment=f"LDS load 8B @ offset {offset}"
        ))
    
    def emit_lds_read_b128(
        self,
        dst_range: KRegRange,
        addr_vreg: KVReg,
        offset: int = 0,
    ):
        """Emit ds_read_b128 (LDS load of 16 bytes)."""
        self.program.emit(KInstr(
            "ds_read_b128",
            (dst_range,),
            (addr_vreg, KMemOffset(offset)),
            comment=f"LDS load 16B @ offset {offset}"
        ))
    
    def emit_lds_write_b64(
        self,
        addr_vreg: KVReg,
        src_range: KRegRange,
        offset: int = 0,
    ):
        """Emit ds_write_b64 (LDS store of 8 bytes)."""
        self.program.emit(KInstr(
            "ds_write_b64",
            (),
            (addr_vreg, src_range, KMemOffset(offset)),
            comment=f"LDS store 8B @ offset {offset}"
        ))
    
    def emit_lds_write_b128(
        self,
        addr_vreg: KVReg,
        src_range: KRegRange,
        offset: int = 0,
    ):
        """Emit ds_write_b128 (LDS store of 16 bytes)."""
        self.program.emit(KInstr(
            "ds_write_b128",
            (),
            (addr_vreg, src_range, KMemOffset(offset)),
            comment=f"LDS store 16B @ offset {offset}"
        ))

    # =========================================================================
    # Loop Support
    # =========================================================================
    
    def begin_loop(self, lower_bound: int, upper_bound: int, step: int) -> dict:
        """
        Begin a loop structure with physical registers for loop control.
        
        Loop control registers (counter, step, upper_bound) use physical SGPRs
        to avoid SSA violations from the loop counter update in the latch.
        
        Returns a loop context dict for use with emit_loop_header/latch/end.
        """
        loop_id = self._loop_counter
        self._loop_counter += 1
        
        # Use physical SGPRs for loop control to avoid SSA violations
        # Reserve SGPRs starting at 24 to avoid conflicts with SRDs (which use s4+)
        # Legacy mode uses s24 for loop counter, which is known to be safe.
        # Each loop needs 3 SGPRs: counter, step, upper_bound
        base_sgpr = 24 + loop_id * 3
        counter_phys = KPhysSReg(base_sgpr)
        step_phys = KPhysSReg(base_sgpr + 1)
        upper_bound_phys = KPhysSReg(base_sgpr + 2)
        
        # Initialize loop counter and bounds using physical registers
        self.comment(f"Initialize loop {loop_id}")
        self.program.emit(KInstr(
            "s_mov_b32", (counter_phys,), (KImm(lower_bound),),
            comment=f"loop {loop_id} counter = {lower_bound}"
        ))
        self.program.emit(KInstr(
            "s_mov_b32", (step_phys,), (KImm(step),),
            comment=f"loop {loop_id} step = {step}"
        ))
        self.program.emit(KInstr(
            "s_mov_b32", (upper_bound_phys,), (KImm(upper_bound),),
            comment=f"loop {loop_id} upper = {upper_bound}"
        ))
        
        loop_ctx = {
            "loop_id": loop_id,
            "counter_sreg": counter_phys,
            "step_sreg": step_phys,
            "upper_bound_sreg": upper_bound_phys,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "step": step,
        }
        
        self._loop_stack.append(loop_ctx)
        return loop_ctx
    
    def emit_loop_header(self, loop_ctx: dict):
        """Emit loop header with comparison and conditional branch."""
        loop_id = loop_ctx["loop_id"]
        counter = loop_ctx["counter_sreg"]
        upper = loop_ctx["upper_bound_sreg"]
        
        # Emit label
        self.emit_label(f"loop_{loop_id}_header")
        
        # Compare counter < upper (sets SCC)
        self.program.emit(KInstr(
            "s_cmp_lt_u32", (), (counter, upper),
            comment=f"compare loop {loop_id} counter < upper"
        ))
        
        # Branch to body if SCC=1
        self.program.emit(KInstr(
            "s_cbranch_scc1", (), (),
            comment=f"loop_{loop_id}_body"
        ))
        
        # Branch to exit if not taken
        self.program.emit(KInstr(
            "s_branch", (), (),
            comment=f"loop_{loop_id}_exit"
        ))
        
        # Body label
        self.emit_label(f"loop_{loop_id}_body")
    
    def emit_loop_latch(self, loop_ctx: dict):
        """Emit loop latch (increment counter and branch back)."""
        loop_id = loop_ctx["loop_id"]
        counter = loop_ctx["counter_sreg"]
        step = loop_ctx["step_sreg"]
        
        # Latch label
        self.emit_label(f"loop_{loop_id}_latch")
        
        # Increment counter - uses physical regs so no defs to track
        # s_add_u32 with physical regs: counter += step
        self.program.emit(KInstr(
            "_loop_inc", (), (counter, step),
            comment=f"loop {loop_id} counter += step"
        ))
        
        # Branch back to header
        self.program.emit(KInstr(
            "s_branch", (), (),
            comment=f"loop_{loop_id}_header"
        ))
    
    def end_loop(self):
        """End loop and emit exit label."""
        loop_ctx = self._loop_stack.pop()
        loop_id = loop_ctx["loop_id"]
        self.emit_label(f"loop_{loop_id}_exit")
    
    def alloc_accumulators(self, count: int) -> List[KRegRange]:
        """
        Allocate accumulator VGPR quads for loop iter_args.
        
        Returns list of KRegRange quads, each initialized to 0.
        """
        accumulators = []
        for i in range(count):
            quad = self.vreg_quad()
            accumulators.append(quad)
            
            # Initialize to 0 using a pseudo-instruction that defines the whole range
            # This ensures liveness analysis sees the quad with proper alignment
            self.program.emit(KInstr(
                "_init_acc_quad", (quad,), (KImm(0),),
                comment=f"Initialize accumulator {i} to 0.0"
            ))
        
        return accumulators
    
    # =========================================================================
    # MFMA Support
    # =========================================================================
    
    def emit_mfma_f32_16x16x16_f16(
        self,
        a_regs: Tuple[KReg, ...],
        b_regs: Tuple[KReg, ...],
        acc_regs: Optional[Tuple[KReg, ...]] = None,
    ) -> Tuple[KReg, ...]:
        """
        Emit MFMA instruction with virtual register tracking.
        
        Args:
            a_regs: Tuple of 2 VGPRs for A operand (f16x2)
            b_regs: Tuple of 2 VGPRs for B operand (f16x2)
            acc_regs: Optional tuple of 4 VGPRs for accumulator (f32x4)
                      If None, allocates new result registers
                      
        Returns:
            Tuple of 4 VGPRs containing the result
        """
        # Wait for LDS reads to complete before MFMA
        self.wait_lgkm(0)
        
        # Build operand ranges
        a_range = KRegRange(a_regs[0], 2, alignment=2) if len(a_regs) >= 2 else None
        b_range = KRegRange(b_regs[0], 2, alignment=2) if len(b_regs) >= 2 else None
        
        # Determine result/accumulator
        if acc_regs is not None and len(acc_regs) == 4:
            # Use provided accumulator as both input and output
            result_regs = acc_regs
            acc_range = KRegRange(acc_regs[0], 4, alignment=4)
            
            # MFMA with accumulator: v_mfma dst, a, b, acc
            # Note: When reusing accumulator, we emit a pseudo-op that doesn't 
            # define new registers (result == accumulator)
            self.program.emit(KInstr(
                "_mfma_acc",  # Pseudo: uses accumulator, doesn't define new regs
                (),  # No new defs - reusing accumulator
                (acc_range, a_range, b_range),
                comment="MFMA with accumulator (in-place)"
            ))
        else:
            # Allocate new quad for result, use 0 as accumulator
            result_range = self.vreg_quad()
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))
            
            # MFMA with zero accumulator: v_mfma dst, a, b, 0
            self.program.emit(KInstr(
                "v_mfma_f32_16x16x16_f16",
                (result_range,),
                (a_range, b_range, KImm(0)),
                comment="MFMA with zero accumulator"
            ))
        
        return result_regs

    @property
    def unified(self) -> UnifiedEmitter:
        """
        Get the unified emitter for this context.
        
        This provides a consistent API with AsmEmitter.unified, allowing
        callers to use kernel_ctx.unified.v_add_u32(...) syntax.
        
        When using the unified emitter:
        - Methods that exist on KernelCompilationContext are called directly
        - Methods that don't exist fall back to emit_raw()
        - Virtual registers are returned for instructions with destinations
        
        Example:
            result = kernel_ctx.unified.v_add_u32(src0, src1, comment="add")
        """
        return self._unified
    
    # =========================================================================
    # Finalization
    # =========================================================================
    
    def finalize(self) -> Tuple[List[str], AllocationStats]:
        """
        Finalize the kernel program and generate assembly.
        
        This:
        1. Emits SRD prologue (all SRD setup at program start)
        2. Applies hazard mitigation (inserts s_nop where needed)
        3. Computes liveness for all virtual registers
        4. Runs linear scan allocation
        5. Renders to assembly
        
        Returns:
            Tuple of (assembly lines, allocation statistics)
        """
        # Emit SRD prologue - moves all SRD setup to program start
        self._emit_srd_prologue()
        
        # Apply peephole optimizations (fuse lshl+add, lshl+or, etc.)
        self._apply_peephole_optimizations()
        
        # Apply hazard mitigation pass
        self._apply_hazard_mitigation()
        
        # Get reserved registers from ABI
        reserved_vgprs = self.program.abi.get_reserved_vgprs()
        reserved_sgprs = self.program.abi.get_reserved_sgprs()

        # IMPORTANT: Reserve physical SGPRs used for loop control.
        #
        # Loop emission intentionally uses *physical* SGPRs (s24+) for the loop
        # counter/step/upper bound to avoid SSA violations from counter updates
        # in the latch. These physical regs are not part of the ABI reserved set,
        # so we must explicitly reserve them from allocation. Otherwise, increased
        # SGPR pressure (e.g. enabling G2S, extra SRD copies, M0 temps) can cause
        # the allocator to assign a virtual SGPR to s24/s25/s26, clobbering the
        # loop counter and producing incorrect memory addresses at runtime.
        if self._loop_counter > 0:
            loop_sgpr_count = self._loop_counter * 3  # counter, step, upper per loop
            reserved_sgprs = set(reserved_sgprs)  # copy in case it's not a set
            reserved_sgprs.update(range(24, 24 + loop_sgpr_count))
        
        # Allocate
        mapping, stats = allocate_kernel(
            self.program,
            reserved_vgprs=reserved_vgprs,
            reserved_sgprs=reserved_sgprs,
        )
        
        # Account for loop control SGPRs which use physical registers
        # Loop control uses s24+ (3 SGPRs per loop: counter, step, upper_bound)
        if self._loop_counter > 0:
            max_loop_sgpr = 24 + self._loop_counter * 3  # s24,s25,s26 for first loop, etc.
            stats.peak_sgprs = max(stats.peak_sgprs, max_loop_sgpr)
        
        # Render
        generator = KernelGenerator(self.program, mapping)
        asm_lines = generator.generate()
        
        return asm_lines, stats
    
    def _apply_hazard_mitigation(self):
        """
        Apply precise hazard mitigation to the program.
        
        On gfx940+ (CDNA3/4), there's a hazard when v_readfirstlane_b32
        immediately reads a VGPR that was just written by a VALU instruction.
        This requires 1 wait state (s_nop 0) between them.
        
        This implementation is precise: it only inserts s_nop when:
        1. Current instruction is a VALU that writes a VGPR
        2. Next instruction is v_readfirstlane_b32
        3. v_readfirstlane reads the VGPR that the VALU just wrote
        """
        instructions = self.program.instructions
        if not instructions:
            return
        
        # Find positions where we need to insert s_nop
        insertions = []
        
        for i in range(len(instructions) - 1):
            instr = instructions[i]
            next_instr = instructions[i + 1]
            
            # Check if next instruction is v_readfirstlane_b32
            if next_instr.name != "v_readfirstlane_b32":
                continue
            
            # Check if current instruction is a VALU that writes a VGPR
            if not self._is_valu_vgpr_write(instr):
                continue
            
            # Check if the VALU writes to a register that readfirstlane reads
            if self._writes_to_readfirstlane_src(instr, next_instr):
                insertions.append(i + 1)
        
        # Insert s_nop instructions in reverse order to preserve indices
        for idx in reversed(insertions):
            instructions.insert(idx, KInstr(
                "s_nop", (), (KImm(0),),
                comment="hazard mitigation"
            ))
    
    def _is_valu_vgpr_write(self, instr: KInstr) -> bool:
        """Check if instruction is a VALU that writes a VGPR."""
        # Must be a vector instruction
        if not instr.name.startswith("v_"):
            return False
        # Must have at least one def (destination)
        if not instr.defs:
            return False
        # Exclude v_readfirstlane (reads VGPR, writes SGPR)
        if instr.name.startswith("v_readfirstlane"):
            return False
        return True
    
    def _writes_to_readfirstlane_src(self, valu_instr: KInstr, readfirstlane_instr: KInstr) -> bool:
        """Check if VALU writes to a VGPR that v_readfirstlane reads."""
        if not valu_instr.defs or not readfirstlane_instr.uses:
            return False
        
        # Get the VGPR(s) written by the VALU
        written_regs = set()
        for def_reg in valu_instr.defs:
            if isinstance(def_reg, KVReg):
                written_regs.add(def_reg.id)
            elif isinstance(def_reg, KRegRange) and isinstance(def_reg.base_reg, KVReg):
                for i in range(def_reg.count):
                    written_regs.add(def_reg.base_reg.id + i)
        
        # Get the VGPR read by v_readfirstlane (first use operand)
        for use_op in readfirstlane_instr.uses:
            if isinstance(use_op, KVReg):
                if use_op.id in written_regs:
                    return True
            elif isinstance(use_op, KPhysVReg):
                # Physical VGPR - would need physical mapping to check
                # Conservative: return True if any VGPR was written
                if written_regs:
                    return True
        
        return False
    
    def _apply_peephole_optimizations(self):
        """
        Apply peephole optimizations to fuse instruction sequences.
        
        Fuses patterns like:
        - v_lshlrev_b32 + v_add_u32 -> v_lshl_add_u32
        - v_lshlrev_b32 + v_or_b32 -> v_lshl_or_b32
        
        These fused instructions are supported on gfx9+ and save VALU cycles.
        """
        instructions = self.program.instructions
        if not instructions:
            return
        
        # Track which registers are written by which instruction index
        # This helps us find the producer of a register
        reg_writers: Dict[int, int] = {}  # vreg_id -> instruction_index
        
        # First pass: build def map
        for i, instr in enumerate(instructions):
            for def_reg in instr.defs:
                if isinstance(def_reg, KVReg):
                    reg_writers[def_reg.id] = i
        
        # Second pass: find fusion opportunities
        # We'll mark instructions to delete and create replacements
        to_delete = set()
        replacements = []  # (index, new_instr)
        
        for i, instr in enumerate(instructions):
            if i in to_delete:
                continue
            
            # Pattern: v_add_u32 vD, vA, vB where vA was produced by v_lshlrev_b32
            # Fuse to: v_lshl_add_u32 vD, src, shift, vB
            if instr.name == "v_add_u32" and len(instr.uses) == 2 and len(instr.defs) == 1:
                dst = instr.defs[0]
                src_a, src_b = instr.uses
                
                # Check if src_a is a VGPR produced by a v_lshlrev_b32
                if isinstance(src_a, KVReg) and src_a.id in reg_writers:
                    shift_idx = reg_writers[src_a.id]
                    shift_instr = instructions[shift_idx]
                    
                    if (shift_instr.name == "v_lshlrev_b32" and 
                        len(shift_instr.uses) == 2 and
                        isinstance(shift_instr.uses[0], KImm) and
                        shift_idx not in to_delete):
                        
                        shift_amt = shift_instr.uses[0]
                        shift_src = shift_instr.uses[1]
                        
                        # Check that the shift result isn't used elsewhere
                        # (for simplicity, we only fuse if the shift result is used once)
                        shift_result = shift_instr.defs[0]
                        uses_of_shift = sum(
                            1 for j, other in enumerate(instructions) 
                            if j != i and j not in to_delete
                            for u in other.uses 
                            if isinstance(u, KVReg) and u.id == shift_result.id
                        )
                        
                        if uses_of_shift == 0:
                            # Can fuse!
                            # v_lshl_add_u32 vD, src, shift, addend
                            fused = KInstr(
                                "v_lshl_add_u32",
                                (dst,),
                                (shift_src, shift_amt, src_b),
                                comment=f"fused: ({shift_src} << {shift_amt.value}) + {src_b}"
                            )
                            to_delete.add(shift_idx)
                            replacements.append((i, fused))
                            continue
                
                # Check if src_b is a VGPR produced by a v_lshlrev_b32 (commutative)
                if isinstance(src_b, KVReg) and src_b.id in reg_writers:
                    shift_idx = reg_writers[src_b.id]
                    shift_instr = instructions[shift_idx]
                    
                    if (shift_instr.name == "v_lshlrev_b32" and 
                        len(shift_instr.uses) == 2 and
                        isinstance(shift_instr.uses[0], KImm) and
                        shift_idx not in to_delete):
                        
                        shift_amt = shift_instr.uses[0]
                        shift_src = shift_instr.uses[1]
                        
                        shift_result = shift_instr.defs[0]
                        uses_of_shift = sum(
                            1 for j, other in enumerate(instructions) 
                            if j != i and j not in to_delete
                            for u in other.uses 
                            if isinstance(u, KVReg) and u.id == shift_result.id
                        )
                        
                        if uses_of_shift == 0:
                            # Can fuse!
                            fused = KInstr(
                                "v_lshl_add_u32",
                                (dst,),
                                (shift_src, shift_amt, src_a),
                                comment=f"fused: ({shift_src} << {shift_amt.value}) + {src_a}"
                            )
                            to_delete.add(shift_idx)
                            replacements.append((i, fused))
                            continue
            
            # Pattern: v_or_b32 vD, vA, vB where vA was produced by v_lshlrev_b32
            # Fuse to: v_lshl_or_b32 vD, src, shift, vB
            if instr.name == "v_or_b32" and len(instr.uses) == 2 and len(instr.defs) == 1:
                dst = instr.defs[0]
                src_a, src_b = instr.uses
                
                # Check if src_a is a VGPR produced by a v_lshlrev_b32
                if isinstance(src_a, KVReg) and src_a.id in reg_writers:
                    shift_idx = reg_writers[src_a.id]
                    shift_instr = instructions[shift_idx]
                    
                    if (shift_instr.name == "v_lshlrev_b32" and 
                        len(shift_instr.uses) == 2 and
                        isinstance(shift_instr.uses[0], KImm) and
                        shift_idx not in to_delete):
                        
                        shift_amt = shift_instr.uses[0]
                        shift_src = shift_instr.uses[1]
                        
                        shift_result = shift_instr.defs[0]
                        uses_of_shift = sum(
                            1 for j, other in enumerate(instructions) 
                            if j != i and j not in to_delete
                            for u in other.uses 
                            if isinstance(u, KVReg) and u.id == shift_result.id
                        )
                        
                        if uses_of_shift == 0:
                            fused = KInstr(
                                "v_lshl_or_b32",
                                (dst,),
                                (shift_src, shift_amt, src_b),
                                comment=f"fused: ({shift_src} << {shift_amt.value}) | {src_b}"
                            )
                            to_delete.add(shift_idx)
                            replacements.append((i, fused))
                            continue
        
        # Apply replacements and deletions
        if replacements or to_delete:
            # Build new instruction list
            new_instructions = []
            replace_map = {idx: instr for idx, instr in replacements}
            
            for i, instr in enumerate(instructions):
                if i in to_delete:
                    continue
                if i in replace_map:
                    new_instructions.append(replace_map[i])
                else:
                    new_instructions.append(instr)
            
            self.program.instructions = new_instructions
    
    def finalize_to_string(self) -> str:
        """Finalize and return assembly as a single string."""
        lines, _ = self.finalize()
        return "\n".join(lines)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    @property
    def num_instructions(self) -> int:
        return len(self.program)
    
    @property
    def num_virtual_vregs(self) -> int:
        return self.program._next_vreg_id
    
    @property
    def num_virtual_sregs(self) -> int:
        return self.program._next_sreg_id
    
    @property
    def cse_hit_count(self) -> int:
        return self._cse_hits


# =============================================================================
# Integration helpers
# =============================================================================

def create_kernel_context(
    kernel_info: "KernelInfo",
    max_vgprs: int = 256,
    max_sgprs: int = 104,
) -> KernelCompilationContext:
    """
    Create a kernel compilation context configured for the given kernel.
    
    Args:
        kernel_info: KernelInfo describing the kernel
        max_vgprs: Maximum VGPRs available
        max_sgprs: Maximum SGPRs available
        
    Returns:
        Configured KernelCompilationContext
    """
    # Determine ABI configuration from kernel_info
    wg_size = getattr(kernel_info, 'wg_size', (256, 1, 1))
    subgroup_size = getattr(kernel_info, 'subgroup_size', 64)
    num_waves = max(1, wg_size[0] * wg_size[1] * wg_size[2] // subgroup_size)
    
    # Get thread ID bounds from kernel_info
    tid_ub_x = getattr(kernel_info, 'tid_ub_x', wg_size[0])
    tid_ub_y = getattr(kernel_info, 'tid_ub_y', wg_size[1])
    tid_ub_z = getattr(kernel_info, 'tid_ub_z', wg_size[2]) if len(wg_size) > 2 else 1
    
    ctx = KernelCompilationContext(
        max_vgprs=max_vgprs,
        max_sgprs=max_sgprs,
        use_flat_tid=(num_waves > 1),
        use_workgroup_ids=(True, True, True),
        tid_ub_x=tid_ub_x,
        tid_ub_y=tid_ub_y,
        tid_ub_z=tid_ub_z,
        subgroup_size=subgroup_size,
        wg_size=wg_size,
    )
    
    return ctx


# =============================================================================
# Module-level Kernel Compiler
# =============================================================================

@dataclass
class KernelModuleCompiler:
    """
    Module-level kernel compiler that generates complete .s assembly files.
    
    This class handles the full compilation pipeline from MLIR to assembly:
    1. Parse MLIR and extract kernel metadata
    2. Create KernelCompilationContext
    3. Walk MLIR operations and emit to kernel IR
    4. Run register allocation
    5. Generate complete assembly (prologue + body + epilogue + metadata)
    
    Usage:
        compiler = KernelModuleCompiler(targetid="gfx942", codeobj="5")
        asm = compiler.compile_mlir_string(mlir_text)
    """
    
    targetid: str = "gfx942"
    codeobj: str = "5"
    
    def compile_mlir_string(self, mlir_text: str) -> str:
        """
        Compile MLIR text to complete AMDGCN assembly.
        
        Args:
            mlir_text: MLIR module text
            
        Returns:
            Complete assembly text ready for assembler
        """
        from wave_lang.support.ir_imports import Context, Module, func_d
        from .mlir_walker import IRWalker
        from .utils import parse_wg_and_subgroup
        
        lines: List[str] = []
        
        with Context() as ctx:
            ctx.allow_unregistered_dialects = True
            module = Module.parse(mlir_text)
            
            for fn in module.body:
                if not isinstance(fn, func_d.FuncOp):
                    continue
                
                kernel_name = fn.sym_name.value
                
                # Extract kernel metadata
                wg_size, subgroup_size = self._extract_kernel_metadata(fn)
                num_args = len(list(fn.entry_block.arguments))
                
                # Detect workgroup ID needs
                needs_wgid_x, needs_wgid_y, needs_wgid_z = self._detect_workgroup_ids(fn)
                
                # Create kernel context with proper thread ID bounds
                num_waves = max(1, wg_size[0] * wg_size[1] * wg_size[2] // subgroup_size)
                kernel_ctx = KernelCompilationContext(
                    use_flat_tid=(num_waves > 1),
                    use_workgroup_ids=(needs_wgid_x, needs_wgid_y, needs_wgid_z),
                    tid_ub_x=wg_size[0],
                    tid_ub_y=wg_size[1],
                    tid_ub_z=wg_size[2] if len(wg_size) > 2 else 1,
                    subgroup_size=subgroup_size,
                    wg_size=wg_size,
                )
                
                # Emit kernarg loading
                self._emit_kernarg_loads(kernel_ctx, num_args)
                
                # Walk MLIR and emit to kernel IR
                # Note: We still need the AsmEmitter for some legacy operations
                # Create a minimal emitter just for those cases
                from .asm_emitter import AsmEmitter
                emitter = AsmEmitter(targetid=self.targetid, codeobj=self.codeobj)
                emitter.needs_wgid_x = needs_wgid_x
                emitter.needs_wgid_y = needs_wgid_y
                emitter.needs_wgid_z = needs_wgid_z
                
                walker = IRWalker(emitter, kernel_ctx=kernel_ctx)
                kernel_info = walker.interpret_func(fn)
                
                # Finalize kernel IR and get body lines
                body_lines, allocation_stats = kernel_ctx.finalize()
                
                # Get LDS size from kernel_info
                lds_size_bytes = getattr(kernel_info, 'lds_size_bytes', 0)
                
                # Generate complete assembly
                asm = self._generate_complete_asm(
                    kernel_name=kernel_name,
                    wg_size=wg_size,
                    subgroup_size=subgroup_size,
                    num_args=num_args,
                    body_lines=body_lines,
                    allocation_stats=allocation_stats,
                    lds_size_bytes=lds_size_bytes,
                    needs_wgid=(needs_wgid_x, needs_wgid_y, needs_wgid_z),
                )
                
                lines.append(asm)
        
        return "\n".join(lines)
    
    def _extract_kernel_metadata(self, fn) -> Tuple[Tuple[int, int, int], int]:
        """Extract workgroup size and subgroup size from function attributes."""
        from wave_lang.support.ir_imports import OpAttributeMap
        from .utils import parse_wg_and_subgroup
        
        wg_size = (64, 1, 1)
        subgroup_size = 64
        
        function_attributes = (
            dict(fn.attributes)
            if isinstance(fn.attributes, OpAttributeMap)
            else {}
        )
        translation_info = function_attributes.get("translation_info")
        if translation_info is not None:
            workgroup_size_tuple, sg_size = parse_wg_and_subgroup(translation_info)
            if workgroup_size_tuple:
                wg_size = workgroup_size_tuple
            if sg_size:
                subgroup_size = sg_size
        
        return wg_size, subgroup_size
    
    def _detect_workgroup_ids(self, fn) -> Tuple[bool, bool, bool]:
        """Detect which workgroup ID dimensions are needed."""
        needs_x = False
        needs_y = False
        needs_z = False
        
        for region in fn.regions:
            for block in region.blocks:
                for op in block.operations:
                    op_name = op.operation.name
                    if "workgroup_id" in op_name:
                        # Check dimension from operation attributes if available
                        str_repr = str(op)
                        if "dim = 0" in str_repr or "x" in str_repr.lower():
                            needs_x = True
                        elif "dim = 1" in str_repr or "y" in str_repr.lower():
                            needs_y = True
                        elif "dim = 2" in str_repr or "z" in str_repr.lower():
                            needs_z = True
        
        return needs_x, needs_y, needs_z
    
    def _emit_kernarg_loads(self, ctx: KernelCompilationContext, num_args: int):
        """Emit kernarg pointer loading instructions."""
        # Kernarg pointer is in s[0:1]
        # For now, we assume handlers will use ensure_srd which handles kernarg loading
        pass
    
    def _generate_complete_asm(
        self,
        kernel_name: str,
        wg_size: Tuple[int, int, int],
        subgroup_size: int,
        num_args: int,
        body_lines: List[str],
        allocation_stats: AllocationStats,
        lds_size_bytes: int,
        needs_wgid: Tuple[bool, bool, bool],
    ) -> str:
        """Generate complete assembly with prologue, body, and epilogue."""
        lines = []
        
        # Normalize workgroup size
        wg_x = wg_size[0] if len(wg_size) > 0 else 64
        wg_y = wg_size[1] if len(wg_size) > 1 else 1
        wg_z = wg_size[2] if len(wg_size) > 2 else 1
        
        # === Prologue ===
        lines.append(f'.amdgcn_target "amdgcn-amd-amdhsa--{self.targetid}"')
        lines.append(".text")
        lines.append(f".protected {kernel_name}")
        lines.append(f".globl {kernel_name}")
        lines.append(".p2align 8")
        lines.append(f".type {kernel_name},@function\n")
        lines.append(".section .rodata,#alloc")
        lines.append(".p2align 6")
        lines.append(f".amdhsa_kernel {kernel_name}")
        lines.append("  .amdhsa_user_sgpr_kernarg_segment_ptr 1")
        
        # Workgroup ID SGPRs
        if needs_wgid[0]:
            lines.append("  .amdhsa_system_sgpr_workgroup_id_x 1")
        if needs_wgid[1]:
            lines.append("  .amdhsa_system_sgpr_workgroup_id_y 1")
        if needs_wgid[2]:
            lines.append("  .amdhsa_system_sgpr_workgroup_id_z 1")
        
        lines.append("  .amdhsa_user_sgpr_count 2")
        
        # Compute register usage
        vgpr_granularity, sgpr_granularity = self._get_register_granularity()
        
        vgprs_used = allocation_stats.peak_vgprs
        sgprs_used = allocation_stats.peak_sgprs
        
        # Round up to granularity
        vgprs_used = ((vgprs_used + vgpr_granularity - 1) // vgpr_granularity) * vgpr_granularity
        sgprs_used = ((sgprs_used + sgpr_granularity - 1) // sgpr_granularity) * sgpr_granularity
        
        # Ensure minimums
        vgprs_used = max(4, vgprs_used)  # Minimum 4 VGPRs
        sgprs_used = max(2, sgprs_used)  # Minimum 2 SGPRs (kernarg ptr)
        
        accum_offset = max(4, vgprs_used)
        
        lines.append(f"  .amdhsa_accum_offset {accum_offset}")
        lines.append(f"  .amdhsa_next_free_vgpr {vgprs_used}")
        lines.append(f"  .amdhsa_next_free_sgpr {sgprs_used}")
        lines.append(f"  .amdhsa_group_segment_fixed_size {lds_size_bytes}")
        lines.append("  .amdhsa_private_segment_fixed_size 0")
        lines.append(f"  .amdhsa_wavefront_size{subgroup_size} 1")
        lines.append(".end_amdhsa_kernel")
        lines.append(f".size {kernel_name}, .Lfunc_end0-{kernel_name}\n")
        
        # === Body ===
        lines.append(f"{kernel_name}:")
        lines.extend(body_lines)
        lines.append("    s_endpgm")
        lines.append("")
        lines.append(".Lfunc_end0:")
        
        # === Metadata ===
        amdhsa_minor = "2" if self.codeobj == "5" else "1"
        
        args_yaml = []
        for i in range(num_args):
            args_yaml.append(
                f"""      - .name: arg{i}_ptr
        .size: 8
        .offset: {i*8}
        .value_kind: global_buffer
        .value_type: i8*"""
            )
        args_yaml_string = "\n".join(args_yaml)
        
        metadata = f"""
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - {amdhsa_minor}
amdhsa.kernels:
  - .name: {kernel_name}
    .symbol: '{kernel_name}.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
{args_yaml_string}
    .group_segment_fixed_size:   {lds_size_bytes}
    .kernarg_segment_align:      8
    .kernarg_segment_size:       {num_args*8}
    .max_flat_workgroup_size:    {wg_x * wg_y * wg_z}
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - {wg_x}
      - {wg_y}
      - {wg_z}
    .sgpr_count:                 {sgprs_used}
    .sgpr_spill_count:           0
    .vgpr_count:                 {vgprs_used}
    .vgpr_spill_count:           0
    .wavefront_size:             {subgroup_size}
...
.end_amdgpu_metadata
"""
        lines.append(metadata)
        
        return "\n".join(lines)
    
    def _get_register_granularity(self) -> Tuple[int, int]:
        """Get VGPR and SGPR allocation granularity for target."""
        # gfx940+ has different granularity
        if "gfx94" in self.targetid or "gfx95" in self.targetid:
            return 8, 8  # VGPRs and SGPRs allocated in blocks of 8
        elif "gfx90" in self.targetid:
            return 4, 8  # gfx90x: VGPRs in 4s, SGPRs in 8s
        else:
            return 4, 8  # Default


def compile_kernel_ir(mlir_text: str, targetid: str = "gfx942", codeobj: str = "5") -> str:
    """
    Compile MLIR text to AMDGCN assembly using the kernel IR path.
    
    This is the main entry point for kernel IR compilation.
    
    Args:
        mlir_text: MLIR module text
        targetid: Target GPU (e.g., "gfx942")
        codeobj: Code object version (e.g., "5")
        
    Returns:
        Complete assembly text
    """
    compiler = KernelModuleCompiler(targetid=targetid, codeobj=codeobj)
    return compiler.compile_mlir_string(mlir_text)
