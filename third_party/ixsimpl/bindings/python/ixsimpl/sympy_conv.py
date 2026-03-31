# SPDX-FileCopyrightText: 2026 ixsimpl contributors
# SPDX-License-Identifier: Apache-2.0
"""Structural converters between ixsimpl Expr trees and SymPy expressions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy

import ixsimpl

_MAX_POW_EXPONENT = 1000

_CMP_TO_SYMPY: dict[int, type[sympy.core.relational.Relational]] = {
    ixsimpl.CMP_GT: sympy.Gt,
    ixsimpl.CMP_GE: sympy.Ge,
    ixsimpl.CMP_LT: sympy.Lt,
    ixsimpl.CMP_LE: sympy.Le,
    ixsimpl.CMP_EQ: sympy.Eq,
    ixsimpl.CMP_NE: sympy.Ne,
}


def to_sympy(
    expr: ixsimpl.Expr,
    *,
    symbols: dict[str, sympy.Symbol] | None = None,
    xor_fn: Callable[..., sympy.Basic] | None = None,
) -> sympy.Basic:
    """Convert an ixsimpl Expr to an equivalent sympy expression.

    Walks the ixsimpl node tree structurally and builds the SymPy
    counterpart.  No simplification is performed on the SymPy side.

    Parameters
    ----------
    symbols:
        Optional mapping from symbol name to a pre-built ``sympy.Symbol``.
        Use this to preserve SymPy assumptions (``integer``, ``nonnegative``,
        etc.) that ixsimpl does not track.  Symbols not in the map fall back
        to ``sympy.Symbol(name, integer=True)``.
    xor_fn:
        Callable used for ``IXS_XOR`` nodes.  Defaults to ``sympy.Xor``
        (boolean XOR).  Pass a custom ``sympy.Function`` subclass for
        integer bitwise XOR instead.
    """

    def _convert(node: ixsimpl.Expr) -> sympy.Basic:
        return to_sympy(node, symbols=symbols, xor_fn=xor_fn)

    tag = expr.tag

    if tag == ixsimpl.INT:
        return sympy.Integer(int(expr))

    if tag == ixsimpl.RAT:
        return sympy.Rational(expr.rat_num, expr.rat_den)

    if tag == ixsimpl.SYM:
        name = expr.sym_name
        if symbols and name in symbols:
            return symbols[name]
        return sympy.Symbol(name, integer=True)

    if tag == ixsimpl.ADD:
        result: sympy.Basic = _convert(expr.add_coeff)
        for i in range(expr.add_nterms):
            result = result + _convert(expr.add_term_coeff(i)) * _convert(expr.add_term(i))
        return result

    if tag == ixsimpl.MUL:
        result = _convert(expr.mul_coeff)
        for i in range(expr.mul_nfactors):
            base = _convert(expr.mul_factor_base(i))
            exp = expr.mul_factor_exp(i)
            result = result * base ** sympy.Integer(exp)
        return result

    if tag == ixsimpl.FLOOR:
        # evaluate=False: SymPy incorrectly drops floor() on some
        # Max/Min-containing arguments, e.g. floor(Max(0, 2*x)/6) -> Max(0, 2*x)/6.
        return sympy.floor(_convert(expr.child(0)), evaluate=False)

    if tag == ixsimpl.CEIL:
        # Same SymPy evaluation bug as floor; see above.
        return sympy.ceiling(_convert(expr.child(0)), evaluate=False)

    if tag == ixsimpl.MOD:
        # evaluate=False: SymPy 1.14 Mod evaluation is buggy on some
        # factored forms, e.g. Mod(x*(2*x+2*y), 6) -> 0.
        return sympy.Mod(_convert(expr.child(0)), _convert(expr.child(1)), evaluate=False)

    if tag == ixsimpl.MAX:
        return sympy.Max(_convert(expr.child(0)), _convert(expr.child(1)))

    if tag == ixsimpl.MIN:
        return sympy.Min(_convert(expr.child(0)), _convert(expr.child(1)))

    if tag == ixsimpl.XOR:
        fn = xor_fn if xor_fn is not None else sympy.Xor
        return fn(_convert(expr.child(0)), _convert(expr.child(1)))

    if tag == ixsimpl.CMP:
        rel = _CMP_TO_SYMPY.get(expr.cmp_op)
        if rel is None:
            raise ValueError(f"unsupported cmp_op: {expr.cmp_op}")
        return rel(_convert(expr.child(0)), _convert(expr.child(1)))

    if tag == ixsimpl.AND:
        args = [_convert(expr.child(i)) for i in range(expr.nchildren)]
        return sympy.And(*args)

    if tag == ixsimpl.OR:
        args = [_convert(expr.child(i)) for i in range(expr.nchildren)]
        return sympy.Or(*args)

    if tag == ixsimpl.NOT:
        return sympy.Not(_convert(expr.child(0)))

    if tag == ixsimpl.PIECEWISE:
        pieces: list[tuple[Any, Any]] = []
        for i in range(expr.pw_ncases):
            val = _convert(expr.pw_value(i))
            cond = _convert(expr.pw_cond(i))
            pieces.append((val, cond))
        return sympy.Piecewise(*pieces)

    if tag == ixsimpl.TRUE:
        return sympy.true

    if tag == ixsimpl.FALSE:
        return sympy.false

    raise ValueError(f"unsupported ixsimpl tag: {tag}")


def from_sympy(ctx: ixsimpl.Context, expr: sympy.Basic) -> ixsimpl.Expr:
    """Convert a SymPy expression to an ixsimpl Expr.

    Walks the SymPy tree and builds ixsimpl nodes using the Python API.
    Only the integer-arithmetic subset supported by ixsimpl is handled.
    """
    if isinstance(expr, sympy.Integer):
        return ctx.int_(int(expr))

    if isinstance(expr, sympy.Rational):
        return ctx.rat(int(expr.p), int(expr.q))

    if isinstance(expr, sympy.Symbol):
        return ctx.sym(str(expr))

    if isinstance(expr, sympy.Add):
        if not expr.args:
            return ctx.int_(0)
        result: ixsimpl.Expr = from_sympy(ctx, expr.args[0])
        for arg in expr.args[1:]:
            result = result + from_sympy(ctx, arg)
        return result

    if isinstance(expr, sympy.Mul):
        if not expr.args:
            return ctx.int_(1)
        result = from_sympy(ctx, expr.args[0])
        for arg in expr.args[1:]:
            result = result * from_sympy(ctx, arg)
        return result

    if isinstance(expr, sympy.Pow):
        base = from_sympy(ctx, expr.args[0])
        exp = expr.args[1]
        if not isinstance(exp, sympy.Integer):
            raise ValueError(f"non-integer exponent: {exp}")
        e = int(exp)
        if abs(e) > _MAX_POW_EXPONENT:
            raise ValueError(f"exponent too large: {e}")
        if e == 0:
            return ctx.int_(1)
        if e > 0:
            result = base
            for _ in range(e - 1):
                result = result * base
            return result
        pos = base
        for _ in range(-e - 1):
            pos = pos * base
        return ctx.int_(1) / pos

    if isinstance(expr, sympy.floor):
        return ixsimpl.floor(from_sympy(ctx, expr.args[0]))

    if isinstance(expr, sympy.ceiling):
        return ixsimpl.ceil(from_sympy(ctx, expr.args[0]))

    if isinstance(expr, sympy.Mod):
        return ixsimpl.mod(from_sympy(ctx, expr.args[0]), from_sympy(ctx, expr.args[1]))

    if isinstance(expr, sympy.Max):
        result = from_sympy(ctx, expr.args[0])
        for arg in expr.args[1:]:
            result = ixsimpl.max_(result, from_sympy(ctx, arg))
        return result

    if isinstance(expr, sympy.Min):
        result = from_sympy(ctx, expr.args[0])
        for arg in expr.args[1:]:
            result = ixsimpl.min_(result, from_sympy(ctx, arg))
        return result

    if isinstance(expr, sympy.Xor):
        result = from_sympy(ctx, expr.args[0])
        for arg in expr.args[1:]:
            result = ixsimpl.xor_(result, from_sympy(ctx, arg))
        return result

    if isinstance(expr, sympy.Piecewise):
        branches: list[tuple[ixsimpl.Expr, ixsimpl.Expr]] = []
        for val, cond in expr.args:
            branches.append((from_sympy(ctx, val), from_sympy(ctx, cond)))
        return ixsimpl.pw(*branches)

    if isinstance(expr, sympy.Ge):
        return from_sympy(ctx, expr.args[0]) >= from_sympy(ctx, expr.args[1])

    if isinstance(expr, sympy.Gt):
        return from_sympy(ctx, expr.args[0]) > from_sympy(ctx, expr.args[1])

    if isinstance(expr, sympy.Le):
        return from_sympy(ctx, expr.args[0]) <= from_sympy(ctx, expr.args[1])

    if isinstance(expr, sympy.Lt):
        return from_sympy(ctx, expr.args[0]) < from_sympy(ctx, expr.args[1])

    if isinstance(expr, sympy.Eq):
        return ctx.eq(from_sympy(ctx, expr.args[0]), from_sympy(ctx, expr.args[1]))

    if isinstance(expr, sympy.Ne):
        return ctx.ne(from_sympy(ctx, expr.args[0]), from_sympy(ctx, expr.args[1]))

    if isinstance(expr, sympy.And):
        result = from_sympy(ctx, expr.args[0])
        for arg in expr.args[1:]:
            result = ixsimpl.and_(result, from_sympy(ctx, arg))
        return result

    if isinstance(expr, sympy.Or):
        result = from_sympy(ctx, expr.args[0])
        for arg in expr.args[1:]:
            result = ixsimpl.or_(result, from_sympy(ctx, arg))
        return result

    if isinstance(expr, sympy.Not):
        return ixsimpl.not_(from_sympy(ctx, expr.args[0]))

    if expr is sympy.true:
        return ctx.true_()

    if expr is sympy.false:
        return ctx.false_()

    # Custom sympy.Function subclasses matched by name.
    if isinstance(expr, sympy.Function):
        name = type(expr).__name__
        if name == "xor":
            args = [from_sympy(ctx, a) for a in expr.args]
            if len(args) < 2:
                raise ValueError(f"xor requires at least 2 arguments, got {len(args)}")
            result = ixsimpl.xor_(args[0], args[1])
            for a in args[2:]:
                result = ixsimpl.xor_(result, a)
            return result

    raise ValueError(f"unsupported sympy expression type: {type(expr).__name__}: {expr}")


def extract_assumptions(
    ctx: ixsimpl.Context,
    expr: sympy.Basic,
) -> list[ixsimpl.Expr]:
    """Extract ixsimpl assumption nodes from SymPy symbol properties.

    Walks *expr*, finds every ``sympy.Symbol``, and converts its SymPy
    assumption flags into ixsimpl comparison nodes suitable for passing
    to ``Expr.simplify(assumptions=...)``.

    Recognized flags (checked via ``sym.is_<flag>``):

    * ``nonnegative`` -- emits ``sym >= 0``
    * ``positive``    -- emits ``sym >= 1`` (symbols are integer-valued)
    * ``nonpositive`` -- emits ``sym <= 0``
    * ``negative``    -- emits ``sym <= -1``

    Symbols without any of these flags produce no assumptions.
    """
    seen: set[str] = set()
    result: list[ixsimpl.Expr] = []
    for sym in expr.free_symbols:
        if not isinstance(sym, sympy.Symbol):
            continue
        name = sym.name
        if name in seen:
            continue
        seen.add(name)
        ix = ctx.sym(name)
        if sym.is_positive:
            result.append(ix >= 1)
        elif sym.is_nonnegative:
            result.append(ix >= 0)
        if sym.is_negative:
            result.append(ix <= ctx.int_(-1))
        elif sym.is_nonpositive:
            result.append(ix <= 0)
    return result
