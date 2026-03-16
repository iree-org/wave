# Copyright 2026 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Simplify IndexMapping expressions by decomposing flat // D and flat % D.

Given an IndexMapping whose input_mapping contains paired expressions:
    dim1: flat_expr // D
    dim2: flat_expr % D

This pass tries to decompose flat_expr = quotient * D + remainder where
0 <= remainder < D, allowing the rewrite:
    dim1: quotient
    dim2: remainder

The decomposition uses:
  1. Algebraic factoring: terms in flat_expr that are provably multiples of D
     are separated into the quotient.
  2. Bounds analysis: if the remaining terms are provably bounded in [0, D),
     the dynamic floordiv/mod is eliminated entirely.
"""

import sympy
from collections.abc import Sequence

from ..lang.wave_types import IndexMapping
from .utils.symbol_utils import (
    _split_sum_by_divisibility,
    IndexExpr,
    IndexSymbol,
    simplify,
    subs_idxc,
)


def _get_symbol_lower_bounds(
    constraints: Sequence,
) -> dict[sympy.Symbol, sympy.Expr]:
    """Extract lower bounds on symbols from Assumption constraints.

    Handles ``Assumption(S >= c)`` and ``Assumption(S > c)`` patterns,
    returning ``{S: lower_bound}`` where lower_bound is the tightest
    bound found.
    """
    from .assumptions import Assumption

    bounds: dict[sympy.Symbol, sympy.Expr] = {}
    for c in constraints:
        if not isinstance(c, Assumption):
            continue
        expr = c.expr
        # Match S >= c  (GreaterThan)  or  S > c  (StrictGreaterThan).
        if isinstance(expr, sympy.GreaterThan):  # S >= c
            lhs, rhs = expr.args
            if lhs.is_Symbol and rhs.is_number:
                cur = bounds.get(lhs)
                if cur is None or rhs > cur:
                    bounds[lhs] = rhs
        elif isinstance(expr, sympy.StrictGreaterThan):  # S > c
            lhs, rhs = expr.args
            if lhs.is_Symbol and rhs.is_number:
                lb = rhs + 1  # S > c implies S >= c+1 for integers.
                cur = bounds.get(lhs)
                if cur is None or lb > cur:
                    bounds[lhs] = lb
    return bounds


def _get_iterator_bounds(
    mapping: IndexMapping,
    tile_sizes: dict[IndexSymbol, IndexExpr] | None = None,
) -> dict[sympy.Symbol, tuple[sympy.Expr, sympy.Expr]]:
    """Extract iterator bounds from tile sizes or the iteration_shape.

    When *tile_sizes* is provided (from the node's index sequences),
    the bounds use the concrete tile size for each iterator, which is
    typically much tighter than the full dimension from iteration_shape.

    Returns {iterator_symbol: (0, upper_bound - 1)} for each iterator.
    """
    bounds = {}
    for sym, idx in mapping.iters.items():
        dim = mapping.iteration_shape[idx]
        if dim is None:
            continue
        # Prefer tile_sizes (concrete) over iteration_shape (symbolic).
        if tile_sizes and dim in tile_sizes:
            upper = tile_sizes[dim] - 1
        else:
            upper = dim - 1
        bounds[sym] = (sympy.Integer(0), upper)
    return bounds


def get_tile_sizes_from_index(
    mapping: IndexMapping,
    index: dict,
) -> dict[IndexSymbol, IndexExpr]:
    """Extract tile sizes for each iteration dimension from the node's index.

    The node's index dict maps dimension symbols (M, N, K) to
    IndexSequences with start/size/stride.  The mapping's iteration_shape
    lists which dimension each iterator spans.  The size of the
    corresponding IndexSequence is the tile size for that dimension.
    """
    from .utils.symbol_utils import IndexSequence

    tile_sizes = {}
    # output_mapping: {dim_sym: iterator_sym}
    # iteration_shape maps iterator ordinal -> dim_sym
    for dim in mapping.iteration_shape:
        if dim is None:
            continue
        seq = index.get(dim)
        if isinstance(seq, IndexSequence):
            tile_sizes[dim] = seq.size
    return tile_sizes


def _expr_bounds_with_iters(
    expr: sympy.Expr,
    iter_bounds: dict[sympy.Symbol, tuple[sympy.Expr, sympy.Expr]],
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Compute [lo, hi] bounds for *expr* via structural interval arithmetic.

    Unlike the module-level ``expr_bounds`` (which only handles sympy
    assumptions), this function uses concrete iterator ranges from
    *iter_bounds* to produce tighter results for tile-level expressions.
    """
    if expr.is_Integer or expr.is_Rational:
        return (expr, expr)
    if expr.is_Symbol:
        if expr in iter_bounds:
            return iter_bounds[expr]
        return (sympy.Integer(0), sympy.oo) if expr.is_nonnegative else None

    # For floor/Mod/Add/Mul, delegate to structural recursion.
    if isinstance(expr, sympy.Mod):
        p, q = expr.args
        if q.is_positive and q.is_number:
            return (sympy.Integer(0), q - 1)
        # Symbolic modulus: Mod(p, q) is in [0, q-1] when q > 0.
        # Use the upper bound of q as a conservative ceiling.
        q_bounds = _expr_bounds_with_iters(q, iter_bounds)
        if q_bounds and q_bounds[0].is_positive:
            return (sympy.Integer(0), q_bounds[1] - 1)
        return None

    if isinstance(expr, sympy.floor):
        inner_bounds = _expr_bounds_with_iters(expr.args[0], iter_bounds)
        if inner_bounds:
            return (sympy.floor(inner_bounds[0]), sympy.floor(inner_bounds[1]))
        return None

    if isinstance(expr, sympy.Add):
        bounds = [_expr_bounds_with_iters(a, iter_bounds) for a in expr.args]
        if all(b is not None for b in bounds):
            return (sum(b[0] for b in bounds), sum(b[1] for b in bounds))
        return None

    if isinstance(expr, sympy.Mul):
        if not expr.args:
            return (sympy.Integer(1), sympy.Integer(1))
        bounds = [_expr_bounds_with_iters(a, iter_bounds) for a in expr.args]
        if all(b is not None for b in bounds):
            if any(sympy.oo in b or -sympy.oo in b for b in bounds):
                return None
            lo, hi = bounds[0]
            for b in bounds[1:]:
                corners = [lo * b[0], lo * b[1], hi * b[0], hi * b[1]]
                try:
                    lo, hi = min(corners), max(corners)
                except TypeError:
                    return None
            return (lo, hi)
        return None

    return None


def _find_floordiv_mod_pairs(
    input_mapping: dict[IndexSymbol, IndexExpr],
) -> list[tuple]:
    """Find paired floor/Mod expressions sharing the same divisor.

    Returns list of (dim_q, dim_r, numerator, divisor, addend) tuples where:
      dim_q has expression: addend + floor(numerator / divisor)
      dim_r has expression: Mod(something, divisor)

    Note: sympy auto-evaluates Mod(A*D + B, D) → Mod(B, D), so the Mod's
    first arg may not match the floor's numerator exactly.  We match on
    the divisor and verify compatibility.
    """
    floor_info: list[tuple] = []  # (dim, numerator, divisor, addend)
    mod_info: list[tuple] = []  # (dim, arg, divisor)

    for dim, expr in input_mapping.items():
        # Top-level Mod(E, D).
        if isinstance(expr, sympy.Mod):
            mod_info.append((dim, expr.args[0], expr.args[1]))
            continue

        # Top-level floor(E / D).
        if isinstance(expr, sympy.floor):
            inner = expr.args[0]
            numer, denom = inner.as_numer_denom()
            if denom != 1:
                floor_info.append((dim, numer, denom, sympy.Integer(0)))
                continue

        # A + floor(E / D) pattern.
        if isinstance(expr, sympy.Add):
            for arg in expr.args:
                if isinstance(arg, sympy.floor):
                    inner = arg.args[0]
                    numer, denom = inner.as_numer_denom()
                    if denom != 1:
                        addend = expr - arg
                        floor_info.append((dim, numer, denom, addend))
                        break

    # Match on divisor.
    pairs = []
    for dim_q, numer, divisor, addend in floor_info:
        for dim_r, mod_arg, mod_divisor in mod_info:
            if divisor == mod_divisor:
                pairs.append((dim_q, dim_r, numer, divisor, addend))
                break

    return pairs


def simplify_index_mapping(
    mapping: IndexMapping,
    constraints=(),
    tile_sizes: dict[IndexSymbol, IndexExpr] | None = None,
) -> tuple[IndexMapping, bool]:
    """Simplify flat // D and flat % D patterns in an IndexMapping.

    When *tile_sizes* is provided (from a node's index sequences), uses
    the concrete tile dimensions as iterator upper bounds.  This enables
    proving remainder < divisor for tile-level expressions that would be
    unprovable with the full-dimension iteration_shape.

    Returns (new_mapping, changed).
    """
    iter_bounds = _get_iterator_bounds(mapping, tile_sizes)
    sym_lower_bounds = _get_symbol_lower_bounds(constraints)
    input_mapping = dict(mapping.input_mapping)
    changed = False

    pairs = _find_floordiv_mod_pairs(input_mapping)
    for dim_q, dim_r, flat_expr, divisor, addend in pairs:
        # Step 1: Factor out D-multiples from flat_expr.
        split = _split_sum_by_divisibility(flat_expr, divisor)
        if split is None:
            quotient = sympy.Integer(0)
            remainder = flat_expr
        else:
            quotient, remainder = split

        # Step 2: Check if remainder is bounded in [0, D).
        rem_bounds = _expr_bounds_with_iters(remainder, iter_bounds)
        if rem_bounds is None:
            continue

        lo, hi = rem_bounds
        if hi == sympy.oo:
            continue

        # Check lo >= 0.
        lo_nonneg = lo.is_nonnegative if hasattr(lo, "is_nonnegative") else None
        if lo_nonneg is None:
            lo_simplified = simplify(lo)
            lo_nonneg = lo_simplified.is_nonnegative
        if not lo_nonneg:
            continue

        # Check hi < divisor.  When the divisor is symbolic, resolve
        # it via IndexingContext.subs (e.g. K_PACKED -> K//2) and then
        # substitute known lower bounds from constraints to obtain a
        # concrete lower bound for the divisor.
        try:
            divisor_lb = subs_idxc(divisor)
        except (IndexError, KeyError):
            divisor_lb = divisor
        if sym_lower_bounds:
            divisor_lb = divisor_lb.subs({s: lb for s, lb in sym_lower_bounds.items()})
        # Evaluate floor/ceiling after substitution.
        try:
            divisor_lb = sympy.Integer(int(divisor_lb))
        except (TypeError, ValueError):
            pass

        diff = simplify(hi - divisor_lb)
        if diff.is_negative is not True:
            continue

        # Remainder < D proven! Eliminate the dynamic floordiv/mod.
        input_mapping[dim_q] = addend + quotient
        input_mapping[dim_r] = remainder
        changed = True

    if not changed:
        return mapping, False

    return (
        IndexMapping(
            mapping.num_iterators,
            input_mapping,
            dict(mapping.output_mapping),
            dynamic_val_mappings=tuple(
                dict(dvm) for dvm in (mapping.dynamic_val_mappings or ())
            ),
        ),
        True,
    )
