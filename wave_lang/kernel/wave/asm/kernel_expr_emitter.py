# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

from .kernel_pipeline_shared import (
    KVReg,
    KInstr,
    KImm,
    KPhysSReg,
    KPhysVReg,
    _ENABLE_KERNEL_IR_SIMPLIFY,
)
from .kernel_expr_floor_ops import _FloorExpressionOps
from .instruction_registry import Instruction


class _ScopeContext:
    """Context manager for scoped CSE regions."""

    def __init__(self, emitter: "KernelIRExprEmitter", name: str):
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


class KernelIRExprEmitter(_FloorExpressionOps):
    """
    Expression emitter for kernel IR that can emit sympy expressions.

    Uses scoped CSE (Common Subexpression Elimination) to avoid redundant
    computation while respecting MLIR structured control flow.

    Key features:
    - Global scope: Values that dominate all uses (prologue-defined tid/wgid)
    - Region scopes: Pushed/popped for structured regions (loops, ifs)
    - Structural expression keys: Enable CSE across identical expressions
    """

    def __init__(self, ctx: "KernelCompilationContext"):
        self.ctx = ctx
        # Scope stack: list of dicts, index 0 is global scope
        # Each scope maps expression_key -> KVReg
        self._scope_stack: List[Dict[tuple, KVReg]] = [{}]  # Start with global scope
        # Symbol bindings: name -> register (string like "v0" or KVReg)
        self._bindings: Dict[str, Any] = {}
        # Recursion depth counter for get_or_emit
        # Used to only run simplification at top-level (depth=0)
        self._emit_depth: int = 0
        # Constant cache: maps large constant value -> KVReg
        # This avoids materializing the same constant multiple times
        self._const_cache: Dict[int, KVReg] = {}

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

    def _get_or_materialize_const(self, value: int) -> KVReg:
        """
        Get or materialize a large constant, using cache to avoid duplicates.

        For small constants (-16 to 64), just creates a new mov each time.
        For large constants, caches the result for reuse.

        Returns:
            KVReg containing the constant value
        """
        # Small constants don't need caching - inline is fine
        if -16 <= value <= 64:
            result = self.ctx.vreg()
            self.ctx.program.emit(
                KInstr(
                    Instruction.V_MOV_B32,
                    (result,),
                    (KImm(value),),
                    comment=f"imm = {value}",
                )
            )
            return result

        # Large constants - check cache first
        if value in self._const_cache:
            return self._const_cache[value]

        # Materialize and cache
        result = self.ctx.vreg()
        self.ctx.program.emit(
            KInstr(
                "v_mov_b32", (result,), (KImm(value),), comment=f"materialize {value}"
            )
        )
        self._const_cache[value] = result
        return result

    # Known loop-invariant symbols (thread IDs and workgroup IDs)
    _LOOP_INVARIANT_SYMBOLS = frozenset(
        [
            "tid_x",
            "tid_y",
            "tid_z",
            "wgid_x",
            "wgid_y",
            "wgid_z",
            "lane_id",
        ]
    )

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
        if hasattr(expr, "free_symbols"):
            for sym in expr.free_symbols:
                name = str(sym)
                # Check if it's a known invariant symbol
                if name in self._LOOP_INVARIANT_SYMBOLS:
                    continue
                # Check if it's bound to an SGPR (loop counter)
                # Physical SGPR references like s24, s25 are loop counters - NOT invariant
                if name.startswith("s") and name[1:].isdigit():
                    return False
                # Virtual SGPR references like ks5, ks6 are also loop counters - NOT invariant
                if name.startswith("ks") and name[2:].isdigit():
                    return False
                # Check bindings
                if name in self._bindings:
                    binding = self._bindings[name]
                    # If bound to a physical SGPR reference, it's loop-varying
                    if isinstance(binding, str) and binding.startswith("s"):
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
            # Physical SGPR references like s24, s25 are loop counters - NOT invariant
            if name.startswith("s") and name[1:].isdigit():
                return False
            # Virtual SGPR references like ks5, ks6 are also loop counters - NOT invariant
            if name.startswith("ks") and name[2:].isdigit():
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
        from sympy import Symbol, Mul, Add, Integer, floor, Mod

        if isinstance(expr, (int, Integer)):
            return ("imm", int(expr))

        if isinstance(expr, Symbol):
            return ("sym", str(expr))

        if isinstance(expr, Mul):
            # Sort args for canonical ordering
            return ("mul",) + tuple(sorted(self._expr_key(a) for a in expr.args))

        if isinstance(expr, Add):
            return ("add",) + tuple(sorted(self._expr_key(a) for a in expr.args))

        if isinstance(expr, Mod):
            return ("mod", self._expr_key(expr.args[0]), self._expr_key(expr.args[1]))

        if getattr(expr, "func", None) == floor:
            return ("floor", self._expr_key(expr.args[0]))

        # Fallback: use string representation (less efficient but always works)
        return ("expr", str(expr))

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
        # These must be set by update_bounds_from_kernel_info() before expression emission
        if not hasattr(self.ctx, "tid_ub_x") or self.ctx.tid_ub_x is None:
            raise ValueError(
                "ctx.tid_ub_x is required but not set. "
                "Call update_bounds_from_kernel_info() before emitting expressions."
            )
        tid_ub_x = self.ctx.tid_ub_x
        tid_ub_y = getattr(
            self.ctx, "tid_ub_y", 1
        )  # tid_y/z can default to 1 (1D workgroup)
        tid_ub_z = getattr(self.ctx, "tid_ub_z", 1)
        subgroup_size = self.ctx.subgroup_size

        # Thread IDs - bounded by workgroup dimensions
        bounds[sympy.Symbol("tid_x")] = (0, tid_ub_x - 1)
        bounds[sympy.Symbol("tid_y")] = (0, max(0, tid_ub_y - 1))
        bounds[sympy.Symbol("tid_z")] = (0, max(0, tid_ub_z - 1))

        # Lane ID bounded by subgroup size
        bounds[sympy.Symbol("lane_id")] = (0, subgroup_size - 1)

        # Workgroup IDs - bounded by grid dimensions (large upper bound)
        bounds[sympy.Symbol("wgid_x")] = (0, 65535)
        bounds[sympy.Symbol("wgid_y")] = (0, 65535)
        bounds[sympy.Symbol("wgid_z")] = (0, 65535)

        return bounds

    def _get_bit_range(self, expr) -> Tuple[int, int]:
        """
        Compute the bit range [min_bit, max_bit] of an expression.

        Returns (min_bit, max_bit) where the expression's value uses bits
        min_bit through max_bit (inclusive). This enables detecting when
        two expressions can be combined with OR instead of ADD.

        For example:
        - tid_x (0-127) uses bits 0-6
        - tid_x * 16 (= tid_x << 4) uses bits 4-10
        - tid_y * 4096 (= tid_y << 12) uses bit 12 only (when tid_y ∈ [0,1])

        Returns (0, 31) as a conservative fallback when range cannot be determined.
        """
        import sympy
        from sympy import Symbol, Mul, Add, Integer, floor, Mod

        bounds = self._get_symbol_bounds()

        def get_max_value(e) -> int:
            """Get the maximum value of an expression given symbol bounds."""
            if isinstance(e, (int, Integer)):
                return abs(int(e))

            if isinstance(e, Symbol):
                if e in bounds:
                    return bounds[e][1]
                return 65535  # Conservative default

            if isinstance(e, Mul):
                result = 1
                for arg in e.args:
                    result *= get_max_value(arg)
                return result

            if isinstance(e, Add):
                return sum(get_max_value(arg) for arg in e.args)

            if isinstance(e, Mod):
                # Mod(x, n) is in [0, n-1]
                return int(e.args[1]) - 1

            if getattr(e, "func", None) == floor:
                inner = e.args[0]
                # floor(x/n) where x has max M gives floor(M/n)
                if isinstance(inner, Mul):
                    # Look for pattern like x * (1/n) = x/n
                    divisor = 1
                    var_max = 1
                    for arg in inner.args:
                        if isinstance(arg, sympy.Rational) and not isinstance(
                            arg, Integer
                        ):
                            divisor = int(arg.q)
                        else:
                            var_max *= get_max_value(arg)
                    return var_max // divisor if divisor > 1 else var_max
                return get_max_value(inner)

            return 65535  # Conservative fallback

        def get_shift_amount(e) -> int:
            """
            Get the left shift amount if expression is of form (base << N).
            Returns 0 if not a pure left-shifted expression.
            """
            if isinstance(e, Mul):
                # Look for power-of-2 multiplier
                for arg in e.args:
                    if isinstance(arg, (int, Integer)):
                        val = int(arg)
                        if val > 0 and (val & (val - 1)) == 0:
                            return val.bit_length() - 1
            return 0

        def get_base_max(e, shift: int) -> int:
            """Get max value of base when expression is base << shift."""
            if shift == 0:
                return get_max_value(e)

            if isinstance(e, Mul):
                # Remove the shift multiplier and compute max of remaining
                base_max = 1
                shift_val = 1 << shift
                found_shift = False
                for arg in e.args:
                    if (
                        isinstance(arg, (int, Integer))
                        and int(arg) == shift_val
                        and not found_shift
                    ):
                        found_shift = True
                        continue
                    base_max *= get_max_value(arg)
                return base_max

            return get_max_value(e) >> shift

        # Compute bit range
        shift = get_shift_amount(expr)
        if shift > 0:
            base_max = get_base_max(expr, shift)
            if base_max == 0:
                return (shift, shift)
            max_bit = shift + (base_max.bit_length() - 1)
            return (shift, max_bit)

        # No shift - starts at bit 0
        max_val = get_max_value(expr)
        if max_val == 0:
            return (0, 0)
        max_bit = max_val.bit_length() - 1
        return (0, max_bit)

    def _bits_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
        """Check if two bit ranges overlap."""
        return range1[1] >= range2[0] and range2[1] >= range1[0]

    def get_or_emit(self, expr, cache_in_global: bool = False) -> KVReg:
        """
        Get a VGPR containing the value of expr, emitting instructions if needed.

        Uses scoped CSE to avoid recomputing the same expression.

        Args:
            expr: The expression to emit
            cache_in_global: If True, cache in global scope (for prologue values)
        """

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
            # For large constants, use the constant cache for better reuse
            if val < -16 or val > 64:
                result = self._get_or_materialize_const(val)
            else:
                result = self.ctx.vreg()
                self.ctx.program.emit(
                    KInstr(
                        Instruction.V_MOV_B32,
                        (result,),
                        (KImm(val),),
                        comment=f"imm = {val}",
                    )
                )
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
                if isinstance(reg, str) and reg.startswith("v"):
                    phys_idx = int(reg[1:])
                    result = self.ctx.vreg()
                    self.ctx.program.emit(
                        KInstr(
                            "v_mov_b32",
                            (result,),
                            (KPhysVReg(phys_idx),),
                            comment=f"copy {name} from {reg}",
                        )
                    )
                    self._insert_cache(cache_key, result, global_scope=is_invariant)
                    return result

            # Common thread ID symbols - emit inline on first use, cache in GLOBAL scope
            # (Cache check already done at top of function)
            if name == "tid_x":
                if self.ctx.use_flat_tid:
                    # Multi-wave: extract tid_x from flat_tid (v0[0:9])
                    result = self.ctx.vreg()
                    self.ctx.program.emit(
                        KInstr(
                            "v_bfe_u32",
                            (result,),
                            (KPhysVReg(0), KImm(0), KImm(10)),
                            comment="extract tid_x from flat_tid",
                        )
                    )
                else:
                    # Single-wave: compute lane_id using v_mbcnt
                    lo_result = self.ctx.vreg()
                    result = self.ctx.vreg()
                    self.ctx.program.emit(
                        KInstr(
                            "v_mbcnt_lo_u32_b32",
                            (lo_result,),
                            (KImm(-1), KImm(0)),
                            comment="lane_id low",
                        )
                    )
                    self.ctx.program.emit(
                        KInstr(
                            "v_mbcnt_hi_u32_b32",
                            (result,),
                            (KImm(-1), lo_result),
                            comment="lane_id = tid_x for single-wave",
                        )
                    )

                # Cache in GLOBAL scope so it survives clear_cache()
                self._insert_cache(cache_key, result, global_scope=True)
                return result

            if name == "tid_y":
                if self.ctx.use_flat_tid:
                    # Multi-wave: extract tid_y from flat_tid (v0[10:19])
                    result = self.ctx.vreg()
                    self.ctx.program.emit(
                        KInstr(
                            "v_bfe_u32",
                            (result,),
                            (KPhysVReg(0), KImm(10), KImm(10)),
                            comment="extract tid_y from flat_tid",
                        )
                    )
                else:
                    # Single-wave: tid_y is 0
                    result = self.ctx.vreg()
                    self.ctx.program.emit(
                        KInstr(
                            "v_mov_b32",
                            (result,),
                            (KImm(0),),
                            comment="tid_y = 0 for single-wave",
                        )
                    )

                # Cache in GLOBAL scope (loop-invariant)
                self._insert_cache(cache_key, result, global_scope=True)
                return result

            # Handle workgroup ID symbols - also cache in global scope
            if name == "wgid_x":
                result = self.ctx.vreg()
                self.ctx.program.emit(
                    KInstr(
                        "v_mov_b32",
                        (result,),
                        (KPhysSReg(2),),
                        comment="wgid_x from s2",
                    )
                )
                self._insert_cache(cache_key, result, global_scope=True)
                return result

            if name == "wgid_y":
                result = self.ctx.vreg()
                self.ctx.program.emit(
                    KInstr(
                        "v_mov_b32",
                        (result,),
                        (KPhysSReg(3),),
                        comment="wgid_y from s3",
                    )
                )
                self._insert_cache(cache_key, result, global_scope=True)
                return result

            if name == "wgid_z":
                result = self.ctx.vreg()
                self.ctx.program.emit(
                    KInstr(
                        "v_mov_b32",
                        (result,),
                        (KPhysSReg(4),),
                        comment="wgid_z from s4",
                    )
                )
                self._insert_cache(cache_key, result, global_scope=True)
                return result

            # Handle SGPR references (like loop counters: s8, s9, etc.)
            # NEVER cache these - loop counters change each iteration
            if name.startswith("s") and name[1:].isdigit():
                sgpr_idx = int(name[1:])
                result = self.ctx.vreg()
                self.ctx.program.emit(
                    KInstr(
                        "v_mov_b32",
                        (result,),
                        (KPhysSReg(sgpr_idx),),
                        comment=f"broadcast {name} to VGPR",
                    )
                )
                return result

            # Handle virtual SGPR references (like loop counters: ks5, ks6, etc.)
            # These are virtual registers that will be allocated to physical SGPRs.
            # NEVER cache these - loop counters change each iteration.
            if name.startswith("ks") and name[2:].isdigit():
                from .kernel_ir import KSReg

                sreg_id = int(name[2:])
                sreg = KSReg(sreg_id)
                result = self.ctx.vreg()
                self.ctx.program.emit(
                    KInstr(
                        "v_mov_b32",
                        (result,),
                        (sreg,),
                        comment=f"broadcast loop counter {name} to VGPR",
                    )
                )
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
                    divisor *= int(arg.q)  # denominator
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
                        self.ctx.program.emit(
                            KInstr(
                                "v_lshrrev_b32",
                                (div_result,),
                                (KImm(shift), var_reg),
                                comment=f"{var_parts[0]} >> {shift} (div by {divisor})",
                            )
                        )
                        var_reg = div_result
                    else:
                        raise NotImplementedError(
                            f"Non-power-of-2 divisor in Mul: {divisor}"
                        )

                if const_part == 1:
                    # No multiplication needed, just return the (possibly divided) result
                    if divisor > 1:
                        self._insert_cache(
                            cache_key, var_reg, global_scope=is_invariant
                        )
                    return var_reg

                result = self.ctx.vreg()
                abs_const = abs(const_part)
                is_negative = const_part < 0

                # Check if |const| is power of 2 for shift optimization
                if abs_const > 0 and (abs_const & (abs_const - 1)) == 0:
                    shift = abs_const.bit_length() - 1
                    if is_negative:
                        # Negative power of 2: shift then negate
                        # var * -2^n = -(var << n) = 0 - (var << n)
                        shifted = self.ctx.vreg()
                        self.ctx.program.emit(
                            KInstr(
                                "v_lshlrev_b32",
                                (shifted,),
                                (KImm(shift), var_reg),
                                comment=f"{var_parts[0]} << {shift}",
                            )
                        )
                        self.ctx.program.emit(
                            KInstr(
                                "v_sub_u32",
                                (result,),
                                (KImm(0), shifted),
                                comment=f"negate (multiply by {const_part})",
                            )
                        )
                    else:
                        self.ctx.program.emit(
                            KInstr(
                                "v_lshlrev_b32",
                                (result,),
                                (KImm(shift), var_reg),
                                comment=f"{var_parts[0]} << {shift}",
                            )
                        )
                elif -16 <= const_part <= 64:
                    # Inline constant range - can use immediate
                    self.ctx.program.emit(
                        KInstr(
                            "v_mul_lo_u32",
                            (result,),
                            (var_reg, KImm(const_part)),
                            comment=f"{var_parts[0]} * {const_part}",
                        )
                    )
                else:
                    # Large constant - use cached materialization
                    const_reg = self._get_or_materialize_const(const_part)
                    self.ctx.program.emit(
                        KInstr(
                            "v_mul_lo_u32",
                            (result,),
                            (var_reg, const_reg),
                            comment=f"{var_parts[0]} * {const_part}",
                        )
                    )

                # Cache result (global scope if loop-invariant)
                self._insert_cache(cache_key, result, global_scope=is_invariant)
                return result

            # Multiple variable parts - emit sequentially
            result = self.get_or_emit(var_parts[0])
            for v in var_parts[1:]:
                v_reg = self.get_or_emit(v)
                new_result = self.ctx.vreg()
                self.ctx.program.emit(
                    KInstr(
                        "v_mul_lo_u32", (new_result,), (result, v_reg), comment="mul"
                    )
                )
                result = new_result

            if const_part != 1:
                final = self.ctx.vreg()
                abs_const = abs(const_part)
                is_negative = const_part < 0

                # Check if |const| is power of 2 for shift optimization
                if abs_const > 0 and (abs_const & (abs_const - 1)) == 0:
                    shift = abs_const.bit_length() - 1
                    if is_negative:
                        # Negative power of 2: shift then negate
                        shifted = self.ctx.vreg()
                        self.ctx.program.emit(
                            KInstr(
                                "v_lshlrev_b32",
                                (shifted,),
                                (KImm(shift), result),
                                comment=f"<< {shift}",
                            )
                        )
                        self.ctx.program.emit(
                            KInstr(
                                "v_sub_u32",
                                (final,),
                                (KImm(0), shifted),
                                comment=f"negate (multiply by {const_part})",
                            )
                        )
                    else:
                        self.ctx.program.emit(
                            KInstr(
                                "v_lshlrev_b32",
                                (final,),
                                (KImm(shift), result),
                                comment=f"<< {shift}",
                            )
                        )
                elif -16 <= const_part <= 64:
                    self.ctx.program.emit(
                        KInstr(
                            "v_mul_lo_u32",
                            (final,),
                            (result, KImm(const_part)),
                            comment=f"* {const_part}",
                        )
                    )
                else:
                    # Large constant - use cached materialization
                    const_reg = self._get_or_materialize_const(const_part)
                    self.ctx.program.emit(
                        KInstr(
                            "v_mul_lo_u32",
                            (final,),
                            (result, const_reg),
                            comment=f"* {const_part}",
                        )
                    )
                result = final

            # Cache result (global scope if loop-invariant)
            self._insert_cache(cache_key, result, global_scope=is_invariant)
            return result

        # Handle addition
        if isinstance(expr, Add):
            # Separate constant and non-constant terms
            const_sum = 0
            var_args = []
            for arg in expr.args:
                if isinstance(arg, (int, Integer)):
                    const_sum += int(arg)
                else:
                    var_args.append(arg)

            # Handle pure constant case
            if not var_args:
                return self.get_or_emit(Integer(const_sum))

            # Emit first variable term
            result = self.get_or_emit(var_args[0])
            result_range = self._get_bit_range(var_args[0])

            # Add remaining variable terms based on bit overlap
            for arg in var_args[1:]:
                arg_reg = self.get_or_emit(arg)
                arg_range = self._get_bit_range(arg)
                new_result = self.ctx.vreg()

                # Check if we can use OR instead of ADD
                # OR is valid when bit ranges don't overlap
                if not self._bits_overlap(result_range, arg_range):
                    self.ctx.program.emit(
                        KInstr(
                            "v_or_b32",
                            (new_result,),
                            (result, arg_reg),
                            comment=f"or (bits {result_range[0]}-{result_range[1]} + {arg_range[0]}-{arg_range[1]})",
                        )
                    )
                else:
                    self.ctx.program.emit(
                        KInstr(
                            "v_add_u32", (new_result,), (result, arg_reg), comment="add"
                        )
                    )

                result = new_result
                # Update result range (conservative: union of ranges for OR, expanded for ADD)
                if not self._bits_overlap(result_range, arg_range):
                    result_range = (
                        min(result_range[0], arg_range[0]),
                        max(result_range[1], arg_range[1]),
                    )
                else:
                    result_range = (
                        min(result_range[0], arg_range[0]),
                        max(result_range[1], arg_range[1]) + 1,
                    )

            # Add the constant sum if non-zero, using inline literal
            if const_sum != 0:
                new_result = self.ctx.vreg()
                # Use inline literal for the constant (v_add_u32 supports 32-bit literals)
                self.ctx.program.emit(
                    KInstr(
                        "v_add_u32",
                        (new_result,),
                        (KImm(const_sum), result),
                        comment=f"+ {const_sum} (inline literal)",
                    )
                )
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
                self.ctx.program.emit(
                    KInstr(
                        "v_and_b32",
                        (result,),
                        (KImm(mask), x_reg),
                        comment=f"mod {n_val} (and)",
                    )
                )
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

        raise NotImplementedError(
            f"Expression type not supported: {type(expr).__name__}: {expr}"
        )
