# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from copy import deepcopy
from functools import lru_cache
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
# Affine-expression uniformity decomposition
####################################################################


def decompose_affine_by_uniformity(
    expr: IndexExpr | int,
    symbol_classes: list[set],
) -> list[IndexExpr]:
    """Decompose a polynomial expression into additive components by symbol uniformity.

    Given symbol classes ``[C0, C1, ..., Cn]`` ordered from most-uniform
    (e.g. workgroup IDs) to least-uniform (e.g. induction variables),
    produces ``n + 1`` components ``[part_0, ..., part_n, remainder]``
    such that ``sum(parts) == expr``.

    *  ``part_k`` ideally depends only on symbols from ``C_k``.
    *  ``remainder`` contains thread-dependent terms and constants.
    *  Cross-class terms (e.g. ``WG * IV``) are cascaded to the next
       less-uniform class during validation.

    This generalises two-way (workgroup / thread) and three-way
    (workgroup / induction-var / thread) index splits into a single
    N-way routine.

    Args:
        expr: Sympy expression (or plain int) to decompose.
        symbol_classes: Ordered list of symbol sets, most-uniform first.

    Returns:
        List of ``n + 1`` IndexExprs (one per class + remainder).
    """
    zero = sympy.sympify(0)
    n = len(symbol_classes)

    if isinstance(expr, (int, float)):
        return [zero] * n + [sympy.sympify(expr)]

    if not isinstance(expr, sympy.Basic):
        expr = sympy.sympify(expr)

    # Progressive remainders: r[k] = expr with classes 0..k-1 zeroed out.
    zero_subs: dict = {}
    remainders = [expr]
    for cls in symbol_classes:
        zero_subs.update({s: 0 for s in cls})
        remainders.append(safe_subs(expr, zero_subs))

    # component[k] = r[k] - r[k+1]: the part attributable to class k.
    components: list[IndexExpr] = []
    for k in range(n):
        components.append(sympy.simplify(remainders[k] - remainders[k + 1]))
    components.append(remainders[n])  # remainder

    # Cascade validation: component[k] must only contain symbols from
    # classes 0..k.  If it has symbols from "below", merge into the next
    # less-uniform component.
    allowed: set = set()
    for k in range(n):
        allowed = allowed | symbol_classes[k]
        if components[k] == zero:
            continue
        actual = (
            components[k].free_symbols
            if isinstance(components[k], sympy.Basic)
            else set()
        )
        if actual - allowed:
            components[k + 1] = sympy.simplify(components[k + 1] + components[k])
            components[k] = zero

    return components


####################################################################
# Interval-arithmetic simplification for floor/Mod expressions.
####################################################################


@lru_cache(maxsize=1024)
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


@lru_cache(maxsize=1024)
def simplify(expr: sympy.Expr) -> sympy.Expr:
    """Simplify a sympy expression using interval arithmetic and sympy.simplify.

    Extends sympy.simplify with bounds-based reasoning that can resolve
    floor/Mod sub-expressions (e.g. floor(Mod(x,16)/16) -> 0) that standard
    sympy cannot handle.  Iterates to a fixed point.
    """
    if not isinstance(expr, sympy.Basic):
        return expr
    for _ in range(5):
        new_expr = _bounds_simplify_once(expr)
        new_expr = sympy.simplify(new_expr)
        if new_expr == expr:
            break
        expr = new_expr
    return expr


def _bounds_simplify_once(expr: sympy.Expr) -> sympy.Expr:
    """Single bottom-up pass of bounds-based simplification.

    Mod nodes are handled specially to avoid a sympy auto-evaluation bug
    where Mod(k*Mod(x,n), m) produces incorrect symbolic results.
    See https://github.com/sympy/sympy/issues/28744.
    """
    if not isinstance(expr, sympy.Basic) or expr.is_Atom:
        return expr

    simplified_args = [_bounds_simplify_once(a) for a in expr.args]

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
