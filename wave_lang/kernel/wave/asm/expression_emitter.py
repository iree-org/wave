"""
Generic SymPy expression visitor for emitting AMDGCN assembly instructions.

This module provides the ExprEmitter class which walks SymPy expression trees
and generates appropriate AMDGCN instructions for GPU execution.
"""

import sympy
from collections import namedtuple
from typing import Dict, Optional

# Marker for rational values during expression evaluation
# Similar to _Rational in emitter.py: tracks expr/const patterns
_RationalReg = namedtuple("_RationalReg", ["numerator_reg", "denominator"])

# Canonicalization helpers for CSE
_TID_SYMBOL_NAMES = {"tid_x", "tid_y", "tid_z"}


def _flatten_commutative(op, args):
    """Flatten nested associative nodes (Add/Mul)."""
    flat = []
    for a in args:
        if isinstance(a, op):
            flat.extend(a.args)
        else:
            flat.append(a)
    return flat


def canonicalize_expr(expr: sympy.Expr) -> sympy.Expr:
    """
    Return a canonicalized SymPy expression to maximize CSE hits.

    - Flatten Add/Mul
    - Sort commutative operands
    - Fold integer constants
    - Keep symbols as-is (tid_x, tid_y, tid_z)
    """
    if expr is None:
        return None

    # Bottom-up canonicalization
    if isinstance(expr, sympy.Add):
        args = [_ for _ in map(canonicalize_expr, expr.args)]
        args = _flatten_commutative(sympy.Add, args)
        # Separate constants
        const = 0
        nonconst = []
        for a in args:
            if a.is_number:
                const += int(a)
            else:
                nonconst.append(a)
        # Sort for commutativity
        nonconst_sorted = sorted(nonconst, key=lambda x: str(x))
        if const != 0:
            nonconst_sorted.append(sympy.Integer(const))
        return (
            sympy.Add(*nonconst_sorted)
            if len(nonconst_sorted) > 1
            else (nonconst_sorted[0] if nonconst_sorted else sympy.Integer(0))
        )

    if isinstance(expr, sympy.Mul):
        args = [_ for _ in map(canonicalize_expr, expr.args)]
        args = _flatten_commutative(sympy.Mul, args)
        # Combine integer factors
        const = 1
        nonconst = []
        for a in args:
            if a.is_number:
                const *= int(a)
            else:
                nonconst.append(a)
        nonconst_sorted = sorted(nonconst, key=lambda x: str(x))
        result_args = nonconst_sorted
        if const != 1 or not result_args:
            result_args = [sympy.Integer(const)] + result_args
        return sympy.Mul(*result_args) if len(result_args) > 1 else result_args[0]

    if isinstance(expr, sympy.Mod):
        return sympy.Mod(
            canonicalize_expr(expr.args[0]), canonicalize_expr(expr.args[1])
        )

    if getattr(expr, "func", None) == sympy.floor:
        return sympy.floor(canonicalize_expr(expr.args[0]))

    if isinstance(expr, (sympy.Integer, sympy.Symbol)):
        return expr

    # Fallback: try sympy.simplify to normalize
    try:
        return sympy.simplify(expr)
    except Exception:
        return expr


def expr_key(expr: sympy.Expr):
    """Build a structural, hashable key for an expression."""
    expr = canonicalize_expr(expr)

    def to_key(e):
        if isinstance(e, sympy.Integer):
            return ("int", int(e))
        if isinstance(e, sympy.Symbol):
            return ("sym", str(e))
        if isinstance(e, sympy.Add):
            return ("add", tuple(to_key(a) for a in e.args))
        if isinstance(e, sympy.Mul):
            return ("mul", tuple(to_key(a) for a in e.args))
        if isinstance(e, sympy.Mod):
            return ("mod", to_key(e.args[0]), to_key(e.args[1]))
        if getattr(e, "func", None) == sympy.floor:
            return ("floor", to_key(e.args[0]))
        # Generic fallback
        return ("raw", str(e))

    return to_key(expr)


class ExprEmitter:
    """
    Generic SymPy expression visitor that emits AMDGCN instructions with CSE.

    Automatically caches emitted expressions to avoid redundant computation.

    Supports:
    - Constants and tid_x
    - Additive chains (left-to-right accumulation)
    - Multiplication by integer (power-of-two -> shift, else mul)
    - Mod(expr, 2^k) -> VAndB32
    - floor(expr / 2^k) -> VLshrrevB32
    - Pow(2, k) in constant folding only

    Raises ValueError on:
    - Non-power-of-two mod/div
    - Products of two dynamic sub-expressions
    """

    def __init__(self, emitter, kernel_info):
        self.emitter = emitter
        self.kernel_info = kernel_info
        self.tid_x_symbol = sympy.Symbol("tid_x", nonnegative=True)
        self.tid_y_symbol = sympy.Symbol("tid_y", nonnegative=True)
        self.tid_z_symbol = sympy.Symbol("tid_z", nonnegative=True)
        self._cache: Dict[tuple, str] = {}

        # Symbol bindings for workgroup IDs (maps symbol -> SGPR index or register name)
        self.symbol_bindings: Dict[sympy.Symbol, str] = {}

    def bind_symbol(self, symbol_name: str, register: str) -> None:
        """
        Bind a symbol name to a register.

        Args:
            symbol_name: Symbol name (e.g., "wgid_x", "wgid_y")
            register: Register name (e.g., "s2", "s3", "v5")
        """
        symbol = sympy.Symbol(symbol_name, nonnegative=True)
        self.symbol_bindings[symbol] = register

    def get_or_emit(self, expr: sympy.Expr, dst_hint: Optional[str] = None) -> str:
        """
        Get cached register for expression or emit and cache it.

        Args:
            expr: SymPy expression to emit
            dst_hint: Optional destination register hint (e.g., "v2")

        Returns:
            Register string (e.g., "v5") containing the expression result
        """
        key = expr_key(expr)
        if key in self._cache:
            return self._cache[key]

        # Materialize expression
        if dst_hint is not None:
            dst_reg = dst_hint
            self.emit(expr, dst_reg)
            reg_str = dst_reg
        else:
            v = self.emitter.vgpr_allocator.alloc_v()
            dst_reg = f"v{v}"
            self.emit(expr, dst_reg)
            reg_str = dst_reg

        self._cache[key] = reg_str
        return reg_str

    def clear_cache(self):
        """Clear the expression cache."""
        self._cache.clear()

    def emit(self, expr: sympy.Expr, dst_reg: str) -> str:
        """
        Main entry point: emit ASM for expr into dst_reg.

        Uses iterative postorder traversal of the SymPy expression tree,
        similar to gen_sympy_index in emitter.py.

        Args:
            expr: SymPy expression to emit
            dst_reg: Destination register (e.g., "v2")

        Returns:
            The register containing the result (usually dst_reg)
        """
        assert dst_reg.startswith("v")
        dst_v = int(dst_reg[1:])

        # Cache lane_id to avoid multiple calls
        lane_id_v = self.emitter.ensure_lane_id(self.kernel_info.subgroup_size)

        # Stack holds register names (strings like "v2") during traversal
        stack = []

        # Iterate through expression in postorder (children before parents)
        for term in sympy.postorder_traversal(expr):
            if isinstance(term, sympy.Symbol):
                self._handle_symbol(term, lane_id_v, stack)

            elif isinstance(term, sympy.Integer):
                self._handle_integer(term, stack)

            elif isinstance(term, sympy.Add):
                self._handle_add(term, dst_v, lane_id_v, stack)

            elif isinstance(term, sympy.Mul):
                self._handle_mul(term, stack)

            elif isinstance(term, sympy.Mod):
                self._handle_mod(term, stack)

            elif getattr(term, "func", None) == sympy.floor:
                self._handle_floor(term, stack)

            elif isinstance(term, sympy.Pow):
                self._handle_pow(term, stack)

            elif isinstance(term, sympy.Rational):
                self._handle_rational(term, stack)

            else:
                raise ValueError(
                    f"Cannot emit ASM for expression: {term}. "
                    f"Unsupported expression type: {type(term).__name__}"
                )

        # Finalize result and move to destination register if necessary
        return self._finalize_result(stack, dst_reg, dst_v, lane_id_v)

    # ===============================================================
    # Stack-based handler methods for iterative postorder traversal
    # ===============================================================

    def _is_multi_wave_mode(self) -> bool:
        """Check if kernel is in multi-wave mode (wg_size_y > 1 or wg_size_z > 1)."""
        from .utils import normalize_wg_size

        wg_size_x, wg_size_y, wg_size_z = normalize_wg_size(self.kernel_info.wg_size)
        return wg_size_y > 1 or wg_size_z > 1

    def _get_flat_thread_id_vgpr(self) -> int:
        """
        Get the VGPR index containing the flat thread ID.

        Returns:
            VGPR index (typically 0 for v0)
        """
        if self.tid_x_symbol in self.symbol_bindings:
            flat_tid_reg = self.symbol_bindings[self.tid_x_symbol]
            return int(flat_tid_reg[1:])
        else:
            return 0  # Default to v0

    def _extract_tid_x_from_flat_id(self, flat_tid_v: int, stack: list) -> None:
        """
        Extract tid_x (bits 0-9) from flat thread ID register.

        Args:
            flat_tid_v: VGPR index containing flat thread ID
            stack: Expression stack to push result onto
        """
        temp_v = self.emitter.vgpr_allocator.alloc_v()
        self.emitter.emit(
            f"    v_and_b32 v{temp_v}, 0x3ff, v{flat_tid_v}  // Extract tid_x: mask 0x3ff = bits 0-9"
        )
        stack.append(f"v{temp_v}")

    def _extract_tid_y_from_flat_id(self, flat_tid_v: int, stack: list) -> None:
        """
        Extract tid_y (bits 10-19) from flat thread ID register.

        Args:
            flat_tid_v: VGPR index containing flat thread ID
            stack: Expression stack to push result onto
        """
        temp_v = self.emitter.vgpr_allocator.alloc_v()
        self.emitter.emit(
            f"    v_bfe_u32 v{temp_v}, v{flat_tid_v}, 10, 10  // Extract tid_y from bits 10-19"
        )
        stack.append(f"v{temp_v}")

    def _handle_bound_symbol(
        self, term: sympy.Symbol, bound_reg: str, stack: list
    ) -> None:
        """
        Handle a symbol that has been bound to a register.

        Args:
            term: The symbol being handled
            bound_reg: Register name (e.g., "s2", "v0")
            stack: Expression stack to push result onto
        """
        if bound_reg.startswith("s"):
            # SGPR - need to move to VGPR for arithmetic
            temp_v = self._sgpr_to_vgpr(bound_reg)
            stack.append(f"v{temp_v}")
        else:
            # VGPR binding - but for tid_x, check if we need to extract from flat thread ID
            if term == self.tid_x_symbol:
                if self._is_multi_wave_mode():
                    # Multi-wave: bound register contains flat thread ID, extract bits 0-9 for tid_x
                    # Note: bound_reg is typically v0 (system VGPR for flat workitem ID)
                    # AMD GPU flat thread ID encoding (fixed by hardware):
                    #   Bits 0-9:   tid_x (max 1024, mask 0x3ff)
                    #   Bits 10-19: tid_y (max 1024, mask 0x3ff << 10)
                    #   Bits 20-29: tid_z (max 1024, mask 0x3ff << 20)
                    bound_v = int(bound_reg[1:])
                    self._extract_tid_x_from_flat_id(bound_v, stack)
                else:
                    # Single-wave: bound register contains tid_x directly
                    stack.append(bound_reg)
            else:
                # Other VGPR binding - use directly
                stack.append(bound_reg)

    def _handle_unbound_tid_x(self, lane_id_v: int, stack: list) -> None:
        """
        Handle tid_x when it's not bound (legacy fallback).

        Args:
            lane_id_v: VGPR index containing lane ID
            stack: Expression stack to push result onto
        """
        # Legacy fallback: use lane_id for single-wave scenarios where tid_x wasn't bound
        stack.append(f"v{lane_id_v}")

    def _handle_unbound_tid_y(self, stack: list) -> None:
        """
        Handle tid_y when it's not bound.

        Args:
            stack: Expression stack to push result onto
        """
        if self._is_multi_wave_mode():
            # Multi-wave: extract tid_y from flat thread ID
            # AMD GPU runtime uses a fixed encoding for thread IDs:
            # - Bits 0-9: thread_id_x (max 1024, mask 0x3ff)
            # - Bits 10-19: thread_id_y (max 1024, mask 0x3ff << 10)
            # - Bits 20-29: thread_id_z (max 1024, mask 0x3ff << 20)
            # This is independent of the actual workgroup dimensions.
            flat_tid_v = self._get_flat_thread_id_vgpr()
            self._extract_tid_y_from_flat_id(flat_tid_v, stack)
        else:
            # Single-wave: tid_y is always 0 (wg_size_y=1 means only one row of threads)
            stack.append(("_const", 0))

    def _handle_direct_sgpr_reference(self, term: sympy.Symbol, stack: list) -> None:
        """
        Handle direct SGPR references like "s0", "s1".

        Args:
            term: Symbol representing SGPR (e.g., Symbol("s0"))
            stack: Expression stack to push result onto
        """
        sgpr_num = int(str(term)[1:])
        temp_v = self._sgpr_to_vgpr(f"s{sgpr_num}")
        stack.append(f"v{temp_v}")

    def _handle_symbol(self, term: sympy.Symbol, lane_id_v: int, stack: list) -> None:
        """
        Handle Symbol node - push appropriate register to stack.

        Dispatches to specialized helper methods based on symbol type:
        - Bound symbols: Calls _handle_bound_symbol
        - Unbound tid_x: Calls _handle_unbound_tid_x (legacy fallback)
        - Unbound tid_y: Calls _handle_unbound_tid_y
        - Direct SGPR references: Calls _handle_direct_sgpr_reference
        - tid_z: Raises ValueError (not yet supported)
        - Unknown symbols: Raises ValueError
        """
        if term in self.symbol_bindings:
            # Symbol is bound to a register (SGPR or VGPR)
            bound_reg = self.symbol_bindings[term]
            self._handle_bound_symbol(term, bound_reg, stack)
        elif term == self.tid_x_symbol:
            # Legacy fallback for unbound tid_x
            self._handle_unbound_tid_x(lane_id_v, stack)
        elif term == self.tid_y_symbol:
            # Handle unbound tid_y (extract from flat thread ID or return 0)
            self._handle_unbound_tid_y(stack)
        elif term == self.tid_z_symbol:
            # tid_z not yet supported
            raise ValueError(
                f"Thread ID symbol {term} (tid_z) not supported - only tid_x and tid_y"
            )
        elif str(term).startswith("s") and str(term)[1:].isdigit():
            # Direct SGPR reference like "s0", "s1"
            self._handle_direct_sgpr_reference(term, stack)
        else:
            raise ValueError(f"Unknown symbol: {term}")

    def _sgpr_to_vgpr(self, sgpr: str) -> int:
        """
        Move an SGPR value to a VGPR.

        Args:
            sgpr: SGPR name (e.g., "s2")

        Returns:
            VGPR index containing the SGPR value
        """
        from .instructions import VMovB32

        temp_v = self.emitter.vgpr_allocator.alloc_v()
        self.emitter.emit_instruction(VMovB32(temp_v, sgpr))
        self.emitter.register_file.v_used.add(temp_v)
        return temp_v

    def _handle_integer(self, term: sympy.Integer, stack: list) -> None:
        """Handle Integer node - push const marker to stack."""
        stack.append(("_const", int(term)))

    def _handle_add(
        self, term: sympy.Add, dst_v: int, lane_id_v: int, stack: list
    ) -> None:
        """Handle Add node - materialize args and emit additions."""
        from .instructions import VMovB32, VAddU32

        num_args = len(term.args)
        args = [stack.pop() for _ in range(num_args)]
        args.reverse()  # Restore original order

        # Materialize any const markers to registers
        materialized_args = []
        for arg in args:
            if isinstance(arg, tuple) and arg[0] == "_const":
                temp_v = self.emitter.vgpr_allocator.alloc_v()
                self.emitter.emit_instruction(VMovB32(temp_v, arg[1]))
                self.emitter.register_file.v_used.add(temp_v)
                materialized_args.append(f"v{temp_v}")
            elif isinstance(arg, _RationalReg):
                # Rational in Add - materialize the numerator and perform division
                # This handles expressions like `x / 8 + y`
                if arg.numerator_reg is None:
                    # Bare rational like 1/8 - treat as constant 0 (since 1/8 < 1, floor(1/8) = 0)
                    # This can appear in address calculations where floor(1/divisor) is always 0
                    temp_v = self.emitter.vgpr_allocator.alloc_v()
                    self.emitter.emit_instruction(VMovB32(temp_v, 0))
                    self.emitter.register_file.v_used.add(temp_v)
                    materialized_args.append(f"v{temp_v}")
                    continue

                # Materialize the numerator first if needed
                numerator = arg.numerator_reg
                if isinstance(numerator, tuple) and numerator[0] == "_const":
                    numerator_v = self.emitter.vgpr_allocator.alloc_v()
                    self.emitter.emit_instruction(VMovB32(numerator_v, numerator[1]))
                    self.emitter.register_file.v_used.add(numerator_v)
                else:
                    numerator_v = int(numerator[1:])

                # Perform the division (must be power of 2)
                from .instructions import VLshrrevB32

                divisor = arg.denominator
                if divisor <= 0 or (divisor & (divisor - 1)) != 0:
                    raise ValueError(
                        f"Division in Add requires power-of-2 divisor, got {divisor}"
                    )
                shift = divisor.bit_length() - 1
                temp_v = self.emitter.vgpr_allocator.alloc_v()
                self.emitter.emit_instruction(VLshrrevB32(temp_v, shift, numerator_v))
                self.emitter.register_file.v_used.add(temp_v)
                materialized_args.append(f"v{temp_v}")
            elif isinstance(arg, tuple):
                # Handle any other tuple types (shouldn't happen, but be defensive)
                raise ValueError(f"Unexpected tuple type in _handle_add: {arg}")
            else:
                materialized_args.append(arg)

        # Use first arg as accumulator
        result_reg = materialized_args[0]
        if not isinstance(result_reg, str):
            raise ValueError(
                f"Expected register string, got {type(result_reg)}: {result_reg}"
            )
        result_v = int(result_reg[1:])

        # If result is lane_id or dst, allocate a temp
        if result_v == dst_v or result_v == lane_id_v:
            temp_v = self.emitter.vgpr_allocator.alloc_v()
            self.emitter.emit_instruction(VMovB32(temp_v, result_reg))
            self.emitter.register_file.v_used.add(temp_v)
            result_reg = f"v{temp_v}"
            result_v = temp_v

        # Add remaining args
        for arg_reg in materialized_args[1:]:
            arg_v = int(arg_reg[1:])
            self.emitter.emit_instruction(VAddU32(result_v, result_v, arg_v))

        stack.append(result_reg)

    def _handle_mul(self, term: sympy.Mul, stack: list) -> None:
        """Handle Mul node - separate consts/rationals/regs and emit mult/shift."""
        from .instructions import VMovB32, VLshlRevB32, VMulLoU32

        num_args = len(term.args)
        args = [stack.pop() for _ in range(num_args)]
        args.reverse()

        # Separate const markers, rationals, and register operands
        rational_arg = None
        const_factors = []
        reg_args = []

        for arg in args:
            if isinstance(arg, _RationalReg):
                if rational_arg is not None:
                    raise ValueError(
                        "Multiple rationals in multiplication not supported"
                    )
                rational_arg = arg
            elif isinstance(arg, tuple) and arg[0] == "_const":
                const_factors.append(arg[1])
            else:
                reg_args.append(arg)

        # If we have a rational (1/c), create rational result
        if rational_arg is not None:
            if len(reg_args) != 1:
                raise ValueError(
                    "Rational multiplication requires exactly one register operand"
                )
            if const_factors:
                raise ValueError(
                    "Rational multiplication with extra constants not supported"
                )
            stack.append(_RationalReg(reg_args[0], rational_arg.denominator))
            return

        # No rationals: regular multiplication
        multiplier = 1
        for c in const_factors:
            multiplier *= c

        if len(reg_args) == 0:
            # Pure constant multiplication
            stack.append(("_const", multiplier))
        elif len(reg_args) == 1:
            # multiplier * single_term
            src_reg = reg_args[0]
            src_v = int(src_reg[1:])

            temp_v = self.emitter.vgpr_allocator.alloc_v()

            if multiplier == 1:
                self.emitter.emit_instruction(VMovB32(temp_v, src_reg))
            elif multiplier > 0 and (multiplier & (multiplier - 1)) == 0:
                # Power of 2: use shift
                shift = multiplier.bit_length() - 1
                self.emitter.emit_instruction(VLshlRevB32(temp_v, shift, src_v))
            else:
                # Use multiply
                const_v = self.emitter.vgpr_allocator.alloc_v()
                self.emitter.emit_instruction(VMovB32(const_v, multiplier))
                self.emitter.emit_instruction(VMulLoU32(temp_v, src_v, const_v))
                self.emitter.vgpr_allocator.free_v(const_v)

            self.emitter.register_file.v_used.add(temp_v)
            stack.append(f"v{temp_v}")
        else:
            raise ValueError(
                f"Cannot emit ASM for product of multiple dynamic terms: {term}"
            )

    def _handle_mod(self, term: sympy.Mod, stack: list) -> None:
        """Handle Mod node - validate power-of-2 divisor and emit mask."""
        from .instructions import VAndB32

        divisor_val_or_reg = stack.pop()
        dividend = stack.pop()

        # Get divisor value
        if isinstance(divisor_val_or_reg, tuple) and divisor_val_or_reg[0] == "_const":
            divisor_val = divisor_val_or_reg[1]
        else:
            divisor_expr = term.args[1]
            if not isinstance(divisor_expr, (int, sympy.Integer)):
                raise ValueError(
                    f"Mod divisor must be a constant integer: {divisor_expr}"
                )
            divisor_val = int(divisor_expr)

        if divisor_val <= 0 or (divisor_val & (divisor_val - 1)) != 0:
            raise ValueError(f"Mod divisor must be power-of-two, got: {divisor_val}")

        # Handle dividend
        if isinstance(dividend, tuple) and dividend[0] == "_const":
            result_val = dividend[1] % divisor_val
            stack.append(("_const", result_val))
        else:
            temp_v = self.emitter.vgpr_allocator.alloc_v()
            src_v = int(dividend[1:])
            mask = divisor_val - 1
            self.emitter.emit_instruction(VAndB32(temp_v, mask, src_v))
            self.emitter.register_file.v_used.add(temp_v)
            stack.append(f"v{temp_v}")

    def _handle_floor(self, term, stack: list) -> None:
        """Handle floor node - check for rational (division) pattern and emit shift."""
        from .instructions import VMovB32, VLshrrevB32

        operand = stack.pop()

        # Check if operand is a rational marker (expr / const)
        if isinstance(operand, _RationalReg):
            divisor_val = operand.denominator

            if divisor_val <= 0 or (divisor_val & (divisor_val - 1)) != 0:
                raise ValueError(
                    f"floor division divisor must be power-of-two, got: {divisor_val}"
                )

            numerator = operand.numerator_reg
            if isinstance(numerator, tuple) and numerator[0] == "_const":
                result_val = numerator[1] // divisor_val
                stack.append(("_const", result_val))
            else:
                temp_v = self.emitter.vgpr_allocator.alloc_v()
                src_reg = numerator
                src_v = int(src_reg[1:])
                shift = divisor_val.bit_length() - 1

                if shift > 0:
                    self.emitter.emit_instruction(VLshrrevB32(temp_v, shift, src_v))
                else:
                    self.emitter.emit_instruction(VMovB32(temp_v, src_reg))

                self.emitter.register_file.v_used.add(temp_v)
                stack.append(f"v{temp_v}")
        elif isinstance(operand, tuple) and operand[0] == "_const":
            # floor(const) -> const (no-op for integers)
            stack.append(operand)
        else:
            # floor(register_expr) -> just the expr itself (no-op for integers)
            stack.append(operand)

    def _handle_pow(self, term: sympy.Pow, stack: list) -> None:
        """Handle Pow node - only support 2^k as constants."""
        base, exponent = term.args

        if base == 2 and isinstance(exponent, (int, sympy.Integer)):
            result = 2 ** int(exponent)
            stack.append(("_const", result))
        else:
            raise ValueError(f"Cannot emit ASM for power expression: {term}")

    def _handle_rational(self, term: sympy.Rational, stack: list) -> None:
        """Handle Rational node - only support 1/c for division patterns."""
        if term.p == 1:
            # This is 1/denominator - marker for division
            stack.append(_RationalReg(None, int(term.q)))
        else:
            raise ValueError(f"Cannot emit ASM for general rational: {term}")

    def _finalize_result(
        self, stack: list, dst_reg: str, dst_v: int, lane_id_v: int
    ) -> str:
        """Finalize the result - move to dst_reg if necessary."""
        from .instructions import VMovB32

        # After traversal, stack should have exactly one result
        if len(stack) != 1:
            raise ValueError(
                f"Expression evaluation error: expected 1 result, got {len(stack)}"
            )

        result = stack[0]

        # Handle _RationalReg - shouldn't appear at top level
        if isinstance(result, _RationalReg):
            raise ValueError(
                "Unexpected rational result - should be wrapped in floor()"
            )

        # Handle const marker - materialize to register
        if isinstance(result, tuple) and result[0] == "_const":
            const_val = result[1]
            self.emitter.emit_instruction(VMovB32(dst_v, const_val))
            self.emitter.register_file.v_used.add(dst_v)
            return dst_reg

        result_reg = result
        result_v = int(result_reg[1:])

        # If result is lane_id (plain tid_x), return it directly
        if result_v == lane_id_v:
            return result_reg

        # If result already in destination, we're done
        if result_v == dst_v:
            self.emitter.register_file.v_used.add(dst_v)
            return dst_reg

        # Otherwise, move result to destination
        self.emitter.emit_instruction(VMovB32(dst_v, result_reg))
        self.emitter.register_file.v_used.add(dst_v)
        return dst_reg
