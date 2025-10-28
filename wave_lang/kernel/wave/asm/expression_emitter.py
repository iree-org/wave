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
        self._cache: Dict[tuple, str] = {}

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

    def _handle_symbol(self, term: sympy.Symbol, lane_id_v: int, stack: list) -> None:
        """Handle Symbol node - push tid_x register to stack."""
        if term == self.tid_x_symbol:
            stack.append(f"v{lane_id_v}")
        else:
            raise ValueError(f"Unknown symbol: {term}")

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
            else:
                materialized_args.append(arg)

        # Use first arg as accumulator
        result_reg = materialized_args[0]
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
