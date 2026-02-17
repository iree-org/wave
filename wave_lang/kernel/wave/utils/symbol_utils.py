# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from copy import deepcopy
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


####################################################################
# Interval-arithmetic simplification for floor/Mod expressions.
####################################################################


def expr_bounds(expr: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Compute (lo, hi) bounds for a sympy expression via interval arithmetic.

    Free symbols are assumed to be non-negative integers (hardware indices).
    Returns (lo, hi) or None if bounds cannot be determined.
    """
    if expr.is_Integer or expr.is_Rational:
        return (expr, expr)
    if expr.is_Symbol:
        return (sympy.Integer(0), sympy.oo) if expr.is_nonnegative else None
    if isinstance(expr, sympy.Mod):
        p, q = expr.args
        if q.is_positive and q.is_number:
            p_bounds = expr_bounds(p)
            if p_bounds and p_bounds[0] >= 0 and p_bounds[1] < q:
                return p_bounds
            return (sympy.Integer(0), q - 1)
        return None
    if isinstance(expr, sympy.floor):
        inner_bounds = expr_bounds(expr.args[0])
        if inner_bounds:
            return (sympy.floor(inner_bounds[0]), sympy.floor(inner_bounds[1]))
        return None
    if isinstance(expr, sympy.Add):
        bounds = [expr_bounds(a) for a in expr.args]
        if all(b is not None for b in bounds):
            return (sum(b[0] for b in bounds), sum(b[1] for b in bounds))
        return None
    if isinstance(expr, sympy.Mul):
        if not expr.args:
            return (sympy.Integer(1), sympy.Integer(1))
        bounds = [expr_bounds(a) for a in expr.args]
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


def simplify_floor_mod(expr: sympy.Expr) -> sympy.Expr:
    """Simplify floor/Mod expressions using interval arithmetic.

    Standard sympy.simplify cannot handle expressions like floor(Mod(x,16)/16)
    because it lacks range information. This function uses expr_bounds to
    determine when floor() collapses to a constant or Mod() is a no-op.
    Iterates to a fixed point to handle cascading simplifications.
    """
    if not isinstance(expr, sympy.Basic):
        return expr
    for _ in range(5):
        new_expr = _simplify_floor_mod_once(expr)
        new_expr = sympy.simplify(new_expr)
        if new_expr == expr:
            break
        expr = new_expr
    return expr


def _simplify_floor_mod_once(expr: sympy.Expr) -> sympy.Expr:
    """Single pass of bounds-based simplification (bottom-up).

    Mod nodes are handled specially to avoid a sympy auto-evaluation bug
    where Mod(k*Mod(x,n), m) produces incorrect symbolic results.
    See https://github.com/sympy/sympy/issues/28744.
    """
    if not isinstance(expr, sympy.Basic) or expr.is_Atom:
        return expr

    simplified_args = [_simplify_floor_mod_once(a) for a in expr.args]

    # Handle Mod before reconstruction to avoid triggering the sympy bug.
    if isinstance(expr, sympy.Mod):
        p, q = simplified_args
        if q.is_positive and q.is_number:
            p_bounds = expr_bounds(p)
            if p_bounds and p_bounds[0] >= 0 and p_bounds[1] < q:
                return p
        # Keep Mod but prevent buggy auto-evaluation.
        return sympy.Mod(p, q, evaluate=False)

    # Reconstruct (safe for non-Mod nodes).
    expr = expr.func(*simplified_args)

    if isinstance(expr, sympy.floor):
        bounds = expr_bounds(expr.args[0])
        if (
            bounds
            and bounds[0] != sympy.oo
            and bounds[1] != sympy.oo
            and sympy.floor(bounds[0]) == sympy.floor(bounds[1])
        ):
            return sympy.Integer(int(sympy.floor(bounds[0])))
    return expr


def check_symbolic_equals_int(expr, value: int) -> bool:
    """Check if a symbolic expression equals a constant integer.

    Adds non-negative integer assumptions to free symbols and simplifies
    floor/Mod sub-expressions via interval arithmetic.
    """
    if expr == value:
        return True
    if not isinstance(expr, sympy.Basic):
        return expr == value
    return simplify_floor_mod(expr) == value


####################################################################


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
