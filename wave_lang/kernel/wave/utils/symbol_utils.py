# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from copy import deepcopy
from functools import lru_cache
import math
import operator as op
from typing import Optional

import sympy

# Reexport symbols from indexing.py
from ..._support.indexing import (
    IndexExpr,
    IndexingContext,  # noqa
    IndexSequence,  # noqa
    IndexSymbol,  # noqa
    safe_subs,  # noqa
    subs_idxc,  # noqa
    is_literal,  # noqa
)


# Candidate values for numeric probing.  Straddle power-of-2 boundaries
# (e.g. 31/32/33) to catch floor/Mod discontinuities that would expose
# non-constant expressions.
_PROBE_POOL = (
    # fmt: off
    0,   1,   2,   3,   4,   5,              # small values
    7,   8,   9,  13,  15,  16,  17,         # near 8/16 boundaries
   23,  24,  25,  31,  32,  33,  37,         # near 32 boundary
   47,  48,  63,  64,  65,  96,  97,         # near 64 boundary
  127, 128, 129, 191, 255, 256, 257,         # near 128/256 boundaries
  383, 511, 512, 513, 767, 1023, 1024, 1025,
    # near 512/1024 boundaries
    # fmt: on
)

# Co-prime strides used to generate linearly independent probe combinations
# from _PROBE_POOL, reducing the chance of accidental collisions.
_STRIDES = (1, 3, 7, 11, 13, 17, 19, 23)

# Sentinel values that indicate a probe produced an undefined result.
_BAD_ATOMS = (sympy.zoo, sympy.nan, sympy.oo, -sympy.oo)

# Mapping from sympy function names to Python callables, used by
# sympy.lambdify to numerically evaluate expressions with floor/Mod/etc.
_LAMBDIFY_MODULES = {
    "Mod": op.mod,
    "floor": math.floor,
    "ceiling": math.ceil,
    "xor": op.xor,
    "Min": min,
    "Max": max,
    "Abs": abs,
}


####################################################################
# Interval-arithmetic simplification for floor/Mod expressions.
####################################################################


# Ranges type: tuple of (symbol, (lo, hi)) pairs.  Hashable for lru_cache.
SymbolRanges = tuple[tuple[sympy.Symbol, tuple[sympy.Expr, sympy.Expr]], ...]


def _lookup_range(
    sym: sympy.Symbol, ranges: SymbolRanges = ()
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Find bounds for *sym* in *ranges*, or return None."""
    for s, bounds in ranges:
        if s == sym:
            return bounds
    return None


@lru_cache(maxsize=1024)
def expr_bounds(
    expr: sympy.Expr,
    ranges: SymbolRanges = (),
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Compute (lo, hi) bounds for a sympy expression via interval arithmetic.

    Free symbols default to [0, ∞) (hardware indices).  Pass *ranges* as a
    tuple of ``(symbol, (lo, hi))`` pairs to supply tighter bounds.
    Returns ``(lo, hi)`` or ``None`` if bounds cannot be determined.
    """
    if expr.is_Integer or expr.is_Rational:
        return (expr, expr)
    if expr.is_Symbol:
        r = _lookup_range(expr, ranges)
        if r is not None:
            return r
        return (sympy.Integer(0), sympy.oo) if expr.is_nonnegative else None
    if isinstance(expr, sympy.Mod):
        p, q = expr.args
        if q.is_positive and q.is_number:
            p_bounds = expr_bounds(p, ranges)
            if p_bounds and p_bounds[0] >= 0 and p_bounds[1] < q:
                return p_bounds
            return (sympy.Integer(0), q - 1)
        return None
    if isinstance(expr, sympy.floor):
        inner_bounds = expr_bounds(expr.args[0], ranges)
        if inner_bounds:
            return (sympy.floor(inner_bounds[0]), sympy.floor(inner_bounds[1]))
        return None
    if isinstance(expr, sympy.ceiling):
        inner_bounds = expr_bounds(expr.args[0], ranges)
        if inner_bounds:
            return (sympy.ceiling(inner_bounds[0]), sympy.ceiling(inner_bounds[1]))
        return None
    if isinstance(expr, sympy.Piecewise):
        # Envelope of all branches — any branch could be active.
        branch_bounds = [expr_bounds(val, ranges) for val, _ in expr.args]
        if all(b is not None for b in branch_bounds):
            return (min(b[0] for b in branch_bounds), max(b[1] for b in branch_bounds))
        return None
    if isinstance(expr, sympy.Max):
        bounds = [expr_bounds(a, ranges) for a in expr.args]
        if all(b is not None for b in bounds):
            return (max(b[0] for b in bounds), max(b[1] for b in bounds))
        return None
    if isinstance(expr, sympy.Min):
        bounds = [expr_bounds(a, ranges) for a in expr.args]
        if all(b is not None for b in bounds):
            return (min(b[0] for b in bounds), min(b[1] for b in bounds))
        return None
    if isinstance(expr, sympy.Add):
        bounds = [expr_bounds(a, ranges) for a in expr.args]
        if all(b is not None for b in bounds):
            return (sum(b[0] for b in bounds), sum(b[1] for b in bounds))
        return None
    if isinstance(expr, sympy.Mul):
        if not expr.args:
            return (sympy.Integer(1), sympy.Integer(1))
        bounds = [expr_bounds(a, ranges) for a in expr.args]
        if all(b is not None for b in bounds):
            # Bail out if any bound is infinite (0 * oo = NaN).
            if any(sympy.oo in b or -sympy.oo in b for b in bounds):
                return None
            lo, hi = bounds[0]
            for b in bounds[1:]:
                corners = [lo * b[0], lo * b[1], hi * b[0], hi * b[1]]
                lo, hi = min(corners), max(corners)
            return (lo, hi)
        return None
    return None


def simplify(
    expr: sympy.Expr,
    ranges: dict[sympy.Symbol, tuple[sympy.Expr, sympy.Expr]] | None = None,
) -> sympy.Expr:
    """Simplify a sympy expression using interval arithmetic and sympy.simplify.

    Extends sympy.simplify with bounds-based reasoning that can resolve
    floor/Mod sub-expressions (e.g. floor(Mod(x,16)/16) -> 0) that standard
    sympy cannot handle.  Iterates to a fixed point.

    Pass *ranges* as ``{symbol: (lo, hi)}`` to supply tighter bounds than
    the default [0, ∞).
    """
    print(f"simplify: {expr}")
    frozen: SymbolRanges = tuple(ranges.items()) if ranges else ()
    return _simplify_impl(subs_idxc(expr), frozen)


@lru_cache(maxsize=1024)
def _simplify_impl(expr: sympy.Expr, ranges: SymbolRanges) -> sympy.Expr:
    if not isinstance(expr, sympy.Basic):
        return expr
    for _ in range(5):
        new_expr = _algebraic_simplify(expr)
        new_expr = _bounds_simplify_once(new_expr, ranges)
        new_expr = sympy.simplify(new_expr)
        if new_expr == expr:
            break
        expr = new_expr
    return expr


@lru_cache(maxsize=1024)
def _algebraic_simplify(expr: sympy.Expr) -> sympy.Expr:
    """Algebraic rewrites for Mod and floor that sympy misses.

    - ``(floor(a)*k + c) % m  ->  (floor(a)*k) % m + c``
      when ``k | m`` and ``0 <= c < k`` (pulls constant out of Mod).
    - ``floor(floor(a)/q + c)  ->  floor(floor(a)/q)``
      when ``c < 1/q`` (drops negligible rational offset from floor).
    """

    def _check_mul_nonneg_int(mul):
        """Return the numeric factor of *mul* if all factors are nonneg integer-ish."""
        ret = None
        for arg in mul.args:
            if arg.is_number:
                if arg < 0:
                    return None
                if ret is not None:
                    return None
                ret = arg
                continue
            if not (isinstance(arg, (sympy.floor, sympy.Mod)) or arg.is_integer):
                return None
            if not arg.is_nonnegative:
                return None
        return ret

    def _transform_mod(e):
        if not isinstance(e, sympy.Mod):
            return None
        p, q = e.args
        if not q.is_number or q < 0 or not isinstance(p, sympy.Add):
            return None
        c = None
        terms = []
        mult = None
        for arg in p.args:
            if arg.is_number:
                if c is not None:
                    return None
                c = arg
                continue
            if not isinstance(arg, sympy.Mul):
                return None
            m = _check_mul_nonneg_int(arg)
            if m is None or q % m != 0:
                return None
            mult = m if mult is None or m < mult else mult
            terms.append(arg)
        if c is None or c >= mult:
            return None
        return (sum(terms) % q) + c

    def _check_mul_rational(mul):
        ret = None
        for arg in mul.args:
            if isinstance(arg, sympy.Rational):
                if ret is not None:
                    return None
                if arg.p < 0 or arg.q < 0:
                    return None
                ret = arg
                continue
            if not (isinstance(arg, (sympy.floor, sympy.Mod)) or arg.is_integer):
                return None
            if not arg.is_nonnegative:
                return None
        return ret

    def _transform_floor(e):
        if not isinstance(e, sympy.floor):
            return None
        inner = e.args[0]
        if not isinstance(inner, sympy.Add):
            return None
        c = None
        for arg in inner.args:
            if isinstance(arg, sympy.Rational):
                if c is not None:
                    return None
                c = arg
        if c is None:
            return None
        terms = []
        for arg in inner.args:
            if isinstance(arg, sympy.Rational):
                continue
            if not isinstance(arg, sympy.Mul):
                return None
            r = _check_mul_rational(arg)
            if r is None or r.p != 1 or r <= c:
                return None
            terms.append(arg)
        return sympy.floor(sum(terms))

    expr = expr.replace(lambda e: _transform_mod(e) is not None, _transform_mod)
    expr = expr.replace(lambda e: _transform_floor(e) is not None, _transform_floor)
    return expr


@lru_cache(maxsize=1024)
def _bounds_simplify_once(expr: sympy.Expr, ranges: SymbolRanges) -> sympy.Expr:
    """Single bottom-up pass of bounds-based simplification.

    Mod nodes are handled specially to avoid a sympy auto-evaluation bug
    where Mod(k*Mod(x,n), m) produces incorrect symbolic results.
    See https://github.com/sympy/sympy/issues/28744.
    """
    if not isinstance(expr, sympy.Basic) or expr.is_Atom:
        return expr

    simplified_args = [_bounds_simplify_once(a, ranges) for a in expr.args]

    # Handle Mod before reconstruction to avoid triggering the sympy bug.
    if isinstance(expr, sympy.Mod):
        p, q = simplified_args
        if q.is_positive and q.is_number:
            p_bounds = expr_bounds(p, ranges)
            if p_bounds and p_bounds[0] >= 0 and p_bounds[1] < q:
                return p
        # Keep Mod but prevent buggy auto-evaluation.
        return sympy.Mod(p, q, evaluate=False)

    # Reconstruct (safe for non-Mod nodes).
    expr = expr.func(*simplified_args)

    if isinstance(expr, sympy.floor):
        bounds = expr_bounds(expr.args[0], ranges)
        if (
            bounds
            and bounds[0] != sympy.oo
            and bounds[1] != sympy.oo
            and sympy.floor(bounds[0]) == sympy.floor(bounds[1])
        ):
            return sympy.Integer(int(sympy.floor(bounds[0])))
    if isinstance(expr, sympy.ceiling):
        bounds = expr_bounds(expr.args[0], ranges)
        if (
            bounds
            and bounds[0] != sympy.oo
            and bounds[1] != sympy.oo
            and sympy.ceiling(bounds[0]) == sympy.ceiling(bounds[1])
        ):
            return sympy.Integer(int(sympy.ceiling(bounds[0])))
    return expr


def get_min_expr(
    expr1: Optional[IndexExpr], expr2: Optional[IndexExpr]
) -> Optional[IndexExpr]:
    """
    Get minimum expression of two expressions.
    """
    if expr1 is None:
        return expr2
    if expr2 is None:
        return expr1

    return sympy.Min(expr1, expr2)


def get_induction_symbol(axis: IndexSymbol):
    return IndexSymbol("$ARG" + str(axis), integer=True, nonnegative=True)


_INDUCTION_SYMBOL_PREFIX = "$ARG"


def collect_allowed_induction_symbols(fx_node) -> set[IndexSymbol]:
    """Walk parent graphs from `fx_node` to collect in-scope induction symbols.

    Each `Iterate` ancestor contributes an induction symbol derived from its
    axis.  Symbols not in the returned set are out-of-scope for this node.
    """
    # Lazy import to avoid circular dependency (symbol_utils is imported
    # during wave_ops module initialisation via constraints.py).
    from ...ops.wave_ops import Iterate, get_custom

    allowed: set[IndexSymbol] = set()
    parent = getattr(fx_node.graph, "parent_op", None) if fx_node else None
    while parent is not None:
        parent_custom = get_custom(parent)
        if isinstance(parent_custom, Iterate):
            allowed.add(get_induction_symbol(parent_custom.axis))
        parent = getattr(parent.graph, "parent_op", None)
    return allowed


def strip_out_of_scope_induction_symbols(
    index: dict[IndexSymbol, IndexSequence],
    allowed_induction_symbols: set[IndexSymbol],
) -> dict[IndexSymbol, IndexSequence]:
    """Return a copy of `index` with out-of-scope induction symbols set to 0.

    Backward index propagation (`set_derived_index`) can place induction
    symbols on nodes that live outside the corresponding `Iterate` loop.
    This function substitutes any `$ARG`-prefixed symbol not present in
    `allowed_induction_symbols` with 0.
    """
    cleaned = deepcopy(index)
    for _dim, seq in cleaned.items():
        all_symbols: set[sympy.Symbol] = set()
        for component in (seq.start, seq.size, seq.stride):
            if isinstance(component, sympy.Expr):
                all_symbols |= component.free_symbols
        to_remove = {
            s
            for s in all_symbols
            if s.name.startswith(_INDUCTION_SYMBOL_PREFIX)
            and s not in allowed_induction_symbols
        }
        if to_remove:
            zero_subs = {s: sympy.Integer(0) for s in to_remove}
            seq.start = safe_subs(seq.start, zero_subs)
            seq.size = safe_subs(seq.size, zero_subs)
            seq.stride = safe_subs(seq.stride, zero_subs)
    return cleaned


def _compile_evaluator(expr):
    """Compile a sympy expression into a fast Python callable.

    Returns (free_symbols_tuple, callable) or ((), None) if the expression
    has no free symbols.  The callable maps integer arguments to an integer
    result using pure Python arithmetic (operator.mod, math.floor).
    """
    free = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
    if not free:
        return free, None
    f = sympy.lambdify(free, expr, modules=[_LAMBDIFY_MODULES])
    return free, f


# TODO: Presburger arithmetic (MLIR's IntegerRelation) for a formal proof.
# correct by construction but much heavier
@lru_cache(maxsize=4096)
def _numeric_eval_constant(expr, num_samples: int = 48):
    """Check if a symbolic expression is a constant via numeric probing.

    Uses lambdify to compile the expression into fast Python arithmetic,
    then probes with diverse integer values to check constancy.
    Returns that integer when consistent, ``None`` otherwise.

    This is useful as a fallback when ``sympy.simplify`` is too slow or
    unable to reduce complex floor/Mod expressions (e.g. preshuffle
    index formulas).

    Not a formal proof (but rational design choice) a non-constant expression that
    coincidentally returns the same value for all probes would be a false positive.
    For the domain this compiler operates in (thread IDs 0-63, tile sizes that are
    powers of 2 up to 1024, workgroup sizes up to 1024)

    The probe pool straddles powers-of-2 boundaries (15/16/17,
    63/64/65, 127/128/129, 255/256/257, 511/512/513, 1023/1024/1025)
    to catch floor/Mod discontinuities at common tile sizes up to 1024.

    Results are cached via ``@lru_cache`` (sympy expressions are immutable).
    """
    try:
        free, evaluator = _compile_evaluator(expr)
    except Exception:
        free, evaluator = (), None

    if not free:
        if expr.has(*_BAD_ATOMS):
            return None
        if expr.is_integer is not True:
            return None
        try:
            return int(expr)
        except (TypeError, ValueError):
            return None

    n_probes = len(_PROBE_POOL)
    n_strides = len(_STRIDES)
    strides = [_STRIDES[i % n_strides] for i in range(len(free))]

    first_val = None
    for trial in range(num_samples):
        args = [
            _PROBE_POOL[(trial * strides[i] + i) % n_probes] for i in range(len(free))
        ]

        if evaluator is not None:
            try:
                result = evaluator(*args)
            except (ZeroDivisionError, ValueError, OverflowError, TypeError):
                return None
            except NameError:
                evaluator = None

            if isinstance(result, bool):
                return None
            if isinstance(result, float):
                if not result.is_integer():
                    return None
                val = int(result)
            elif isinstance(result, int):
                val = result
            else:
                evaluator = None

        if evaluator is None:
            subs = dict(zip(free, args))
            result = expr.subs(subs)
            if result.free_symbols:
                return None
            if result.has(*_BAD_ATOMS):
                return None
            if result.is_integer is not True:
                return None
            try:
                val = int(result)
            except (TypeError, ValueError):
                return None

        if first_val is None:
            first_val = val
        elif val != first_val:
            return None

    return first_val
