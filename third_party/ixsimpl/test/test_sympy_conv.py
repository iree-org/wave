# SPDX-FileCopyrightText: 2026 ixsimpl contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ixsimpl.sympy_conv: to_sympy / from_sympy roundtrip."""

from __future__ import annotations

import ixsimpl
import pytest
import sympy
from ixsimpl.sympy_conv import extract_assumptions, from_sympy, to_sympy


@pytest.fixture()
def ctx() -> ixsimpl.Context:
    return ixsimpl.Context()


@pytest.fixture()
def syms(ctx: ixsimpl.Context) -> dict[str, ixsimpl.Expr]:
    return {name: ctx.sym(name) for name in ("x", "y", "z")}


@pytest.fixture()
def sp_syms() -> dict[str, sympy.Symbol]:
    return {name: sympy.Symbol(name, integer=True) for name in ("x", "y", "z")}


# ---------------------------------------------------------------------------
#  to_sympy tests
# ---------------------------------------------------------------------------


def test_to_sympy_int(ctx: ixsimpl.Context) -> None:
    assert to_sympy(ctx.int_(42)) == sympy.Integer(42)
    assert to_sympy(ctx.int_(-7)) == sympy.Integer(-7)
    assert to_sympy(ctx.int_(0)) == sympy.Integer(0)


def test_to_sympy_rat(ctx: ixsimpl.Context) -> None:
    assert to_sympy(ctx.rat(3, 7)) == sympy.Rational(3, 7)
    assert to_sympy(ctx.rat(-5, 3)) == sympy.Rational(-5, 3)


def test_to_sympy_sym(ctx: ixsimpl.Context) -> None:
    sp = to_sympy(ctx.sym("x"))
    assert isinstance(sp, sympy.Symbol)
    assert sp.name == "x"
    assert sp.is_integer is True


def test_to_sympy_sym_with_symbol_map(ctx: ixsimpl.Context) -> None:
    rich = sympy.Symbol("x", integer=True, nonnegative=True)
    sp = to_sympy(ctx.sym("x"), symbols={"x": rich})
    assert sp is rich
    assert sp.is_nonnegative is True


def test_to_sympy_sym_map_fallback(ctx: ixsimpl.Context) -> None:
    """Symbols not in the map get the default integer=True."""
    sp = to_sympy(ctx.sym("y"), symbols={"x": sympy.Symbol("x")})
    assert sp.name == "y"
    assert sp.is_integer is True


def test_to_sympy_add(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr], sp_syms: dict[str, sympy.Symbol]
) -> None:
    e = syms["x"] + syms["y"] + 3
    sp = to_sympy(e)
    expected = sp_syms["x"] + sp_syms["y"] + 3
    assert sympy.simplify(sp - expected) == 0


def test_to_sympy_mul(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr], sp_syms: dict[str, sympy.Symbol]
) -> None:
    e = syms["x"] * syms["y"]
    sp = to_sympy(e)
    expected = sp_syms["x"] * sp_syms["y"]
    assert sympy.simplify(sp - expected) == 0


def test_to_sympy_div(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr], sp_syms: dict[str, sympy.Symbol]
) -> None:
    e = syms["x"] / 3
    sp = to_sympy(e)
    expected = sp_syms["x"] / 3
    assert sympy.simplify(sp - expected) == 0


def test_to_sympy_floor_ceil(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr], sp_syms: dict[str, sympy.Symbol]
) -> None:
    assert to_sympy(ixsimpl.floor(syms["x"])) == sympy.floor(sp_syms["x"])
    assert to_sympy(ixsimpl.ceil(syms["x"])) == sympy.ceiling(sp_syms["x"])


def test_to_sympy_floor_ceil_max_min(ctx: ixsimpl.Context) -> None:
    """Regression: SymPy drops floor/ceiling on Max/Min-containing args.

    sympy.ceiling(Max(0, 2*x)/6) incorrectly simplifies to Max(0, 2*x)/6.
    to_sympy must use evaluate=False to prevent this.
    """
    x = ctx.sym("x")
    sx = sympy.Symbol("x", integer=True)

    def _sp_eval(ixs_expr: ixsimpl.Expr, val: int) -> int:
        sp = to_sympy(ixs_expr)
        return int(sp.xreplace({sx: sympy.Integer(val)}))

    # ceiling(Max(0, 2*x)/6) at x=1: ceiling(1/3) = 1
    # Without ceiling: int(1/3) = 0 — detects the bug.
    ceil_max = ixsimpl.ceil(ixsimpl.max_(ctx.int_(0), x + x) / ctx.int_(6))
    assert _sp_eval(ceil_max, 1) == 1, "ceiling(Max) lost"

    # -ceiling(Max(0, 2*x)/6) at x=1: -1 (the original failing case)
    assert _sp_eval(-ceil_max, 1) == -1, "-ceiling(Max) lost"

    # floor(Min(0, 2*x)/6) at x=-1: floor(-1/3) = -1
    # Without floor: int(-1/3) = 0 — detects the bug.
    floor_min = ixsimpl.floor(ixsimpl.min_(ctx.int_(0), x + x) / ctx.int_(6))
    assert _sp_eval(floor_min, -1) == -1, "floor(Min) lost"


def test_to_sympy_mod(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr], sp_syms: dict[str, sympy.Symbol]
) -> None:
    sp = to_sympy(ixsimpl.mod(syms["x"], syms["y"]))
    assert sp == sympy.Mod(sp_syms["x"], sp_syms["y"], evaluate=False)


def test_to_sympy_max_min(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr], sp_syms: dict[str, sympy.Symbol]
) -> None:
    assert to_sympy(ixsimpl.max_(syms["x"], syms["y"])) == sympy.Max(sp_syms["x"], sp_syms["y"])
    assert to_sympy(ixsimpl.min_(syms["x"], syms["y"])) == sympy.Min(sp_syms["x"], sp_syms["y"])


def test_to_sympy_cmp(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr], sp_syms: dict[str, sympy.Symbol]
) -> None:
    x, y = syms["x"], syms["y"]
    sx, sy = sp_syms["x"], sp_syms["y"]
    assert to_sympy(x >= y) == sympy.Ge(sx - sy, 0)
    assert to_sympy(x > y) == sympy.Gt(sx - sy, 0)
    assert to_sympy(x <= y) == sympy.Le(sx - sy, 0)
    assert to_sympy(x < y) == sympy.Lt(sx - sy, 0)
    assert to_sympy(ctx.eq(x, y)) == sympy.Eq(sx - sy, 0)
    assert to_sympy(ctx.ne(x, y)) == sympy.Ne(sx - sy, 0)


def test_to_sympy_piecewise(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr], sp_syms: dict[str, sympy.Symbol]
) -> None:
    pw = ixsimpl.pw((syms["x"], syms["x"] > syms["y"]), (syms["y"], ctx.true_()))
    sp = to_sympy(pw)
    assert isinstance(sp, sympy.Piecewise)
    assert len(sp.args) == 2


def test_to_sympy_xor_default(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr], sp_syms: dict[str, sympy.Symbol]
) -> None:
    sp = to_sympy(ixsimpl.xor_(syms["x"], syms["y"]))
    assert isinstance(sp, sympy.Xor)


def test_to_sympy_xor_custom_fn(ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr]) -> None:
    class bitwise_xor(sympy.Function):  # type: ignore[misc]
        pass

    sp = to_sympy(ixsimpl.xor_(syms["x"], syms["y"]), xor_fn=bitwise_xor)
    assert isinstance(sp, bitwise_xor)


def test_to_sympy_symbol_map_propagates(
    ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr]
) -> None:
    """symbol_map is used for symbols nested inside compound expressions."""
    rich_x = sympy.Symbol("x", integer=True, nonnegative=True)
    sp = to_sympy(syms["x"] + syms["y"] + 1, symbols={"x": rich_x})
    x_atoms = [a for a in sp.free_symbols if a.name == "x"]
    assert len(x_atoms) == 1
    assert x_atoms[0].is_nonnegative is True


def test_to_sympy_bool(ctx: ixsimpl.Context) -> None:
    assert to_sympy(ctx.true_()) is sympy.true
    assert to_sympy(ctx.false_()) is sympy.false


# ---------------------------------------------------------------------------
#  from_sympy tests
# ---------------------------------------------------------------------------


def test_from_sympy_int(ctx: ixsimpl.Context) -> None:
    e = from_sympy(ctx, sympy.Integer(42))
    assert e.tag == ixsimpl.INT
    assert int(e) == 42


def test_from_sympy_rat(ctx: ixsimpl.Context) -> None:
    e = from_sympy(ctx, sympy.Rational(3, 7))
    assert e.tag == ixsimpl.RAT
    assert e.rat_num == 3
    assert e.rat_den == 7


def test_from_sympy_sym(ctx: ixsimpl.Context) -> None:
    e = from_sympy(ctx, sympy.Symbol("x", integer=True))
    assert e.tag == ixsimpl.SYM
    assert e.sym_name == "x"


def test_from_sympy_add(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    sp = sp_syms["x"] + sp_syms["y"] + 3
    e = from_sympy(ctx, sp)
    assert e.tag == ixsimpl.ADD


def test_from_sympy_mul(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    sp = 3 * sp_syms["x"]
    e = from_sympy(ctx, sp)
    assert e.tag == ixsimpl.MUL


def test_from_sympy_pow(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    sp = sp_syms["x"] ** 3
    e = from_sympy(ctx, sp)
    assert e.tag == ixsimpl.MUL


def test_from_sympy_neg_pow(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    sp = sp_syms["x"] ** sympy.Integer(-1)
    e = from_sympy(ctx, sp)
    assert e.tag == ixsimpl.MUL


def test_from_sympy_floor_ceil(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    e_fl = from_sympy(ctx, sympy.floor(sp_syms["x"] / 3))
    assert e_fl.tag == ixsimpl.FLOOR
    e_ce = from_sympy(ctx, sympy.ceiling(sp_syms["x"] / 3))
    assert e_ce.tag == ixsimpl.CEIL


def test_from_sympy_mod(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    e = from_sympy(ctx, sympy.Mod(sp_syms["x"], sp_syms["y"], evaluate=False))
    assert e.tag == ixsimpl.MOD


def test_from_sympy_piecewise(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    sp = sympy.Piecewise((sp_syms["x"], sp_syms["x"] > 0), (sp_syms["y"], True))
    e = from_sympy(ctx, sp)
    assert e.tag == ixsimpl.PIECEWISE


def test_from_sympy_relational(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    sx, sy = sp_syms["x"], sp_syms["y"]
    assert from_sympy(ctx, sympy.Ge(sx, sy)).tag == ixsimpl.CMP
    assert from_sympy(ctx, sympy.Gt(sx, sy)).tag == ixsimpl.CMP
    assert from_sympy(ctx, sympy.Le(sx, sy)).tag == ixsimpl.CMP
    assert from_sympy(ctx, sympy.Lt(sx, sy)).tag == ixsimpl.CMP
    assert from_sympy(ctx, sympy.Eq(sx, sy)).tag == ixsimpl.CMP
    assert from_sympy(ctx, sympy.Ne(sx, sy)).tag == ixsimpl.CMP


def test_from_sympy_logic(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    cond = sympy.Gt(sp_syms["x"], 0)
    a = from_sympy(ctx, sympy.And(cond, sympy.Gt(sp_syms["y"], 0)))
    assert a.tag == ixsimpl.AND
    o = from_sympy(ctx, sympy.Or(cond, sympy.Gt(sp_syms["y"], 0)))
    assert o.tag == ixsimpl.OR


def test_from_sympy_bool(ctx: ixsimpl.Context) -> None:
    assert from_sympy(ctx, sympy.true).tag == ixsimpl.TRUE
    assert from_sympy(ctx, sympy.false).tag == ixsimpl.FALSE


def test_from_sympy_huge_exponent(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    huge = sympy.Pow(sp_syms["x"], sympy.Integer(10**9), evaluate=False)
    with pytest.raises(ValueError, match="exponent too large"):
        from_sympy(ctx, huge)


def test_from_sympy_custom_xor(ctx: ixsimpl.Context, sp_syms: dict[str, sympy.Symbol]) -> None:
    """Custom sympy.Function subclass named 'xor' maps to ixsimpl.xor_."""

    class xor(sympy.Function):  # type: ignore[misc]
        pass

    e = from_sympy(ctx, xor(sp_syms["x"], sp_syms["y"]))
    assert e.tag == ixsimpl.XOR


def test_from_sympy_unsupported(ctx: ixsimpl.Context) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        from_sympy(ctx, sympy.sin(sympy.Symbol("x")))


# ---------------------------------------------------------------------------
#  Roundtrip tests
# ---------------------------------------------------------------------------


def test_roundtrip_arithmetic(ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr]) -> None:
    x, y = syms["x"], syms["y"]
    e = (x * 3 + y + 1).simplify()
    sp = to_sympy(e)
    e2 = from_sympy(ctx, sp).simplify()
    sp2 = to_sympy(e2)
    assert sympy.simplify(sp - sp2) == 0


def test_roundtrip_floor(ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr]) -> None:
    e = ixsimpl.floor((syms["x"] + syms["y"]) / 3)
    sp = to_sympy(e)
    e2 = from_sympy(ctx, sp)
    sp2 = to_sympy(e2)
    assert sympy.simplify(sp - sp2) == 0


def test_roundtrip_piecewise(ctx: ixsimpl.Context, syms: dict[str, ixsimpl.Expr]) -> None:
    pw_expr = ixsimpl.pw((syms["x"], syms["x"] > syms["y"]), (syms["y"], ctx.true_()))
    sp = to_sympy(pw_expr)
    e2 = from_sympy(ctx, sp)
    assert e2.tag == ixsimpl.PIECEWISE


# ---------------------------------------------------------------------------
#  extract_assumptions tests
# ---------------------------------------------------------------------------


def test_extract_assumptions_nonnegative(ctx: ixsimpl.Context) -> None:
    x = sympy.Symbol("x", integer=True, nonnegative=True)
    assumptions = extract_assumptions(ctx, x + 1)
    assert len(assumptions) == 1
    assert assumptions[0].tag == ixsimpl.CMP
    assert str(assumptions[0]) == "x >= 0"


def test_extract_assumptions_positive(ctx: ixsimpl.Context) -> None:
    x = sympy.Symbol("x", integer=True, positive=True)
    assumptions = extract_assumptions(ctx, x + 1)
    assert len(assumptions) == 1
    assert assumptions[0].tag == ixsimpl.CMP
    assert str(assumptions[0]) == "-1 + x >= 0"


def test_extract_assumptions_negative(ctx: ixsimpl.Context) -> None:
    x = sympy.Symbol("x", integer=True, negative=True)
    assumptions = extract_assumptions(ctx, x + 1)
    assert len(assumptions) == 1
    assert assumptions[0].tag == ixsimpl.CMP
    assert str(assumptions[0]) == "1 + x <= 0"


def test_extract_assumptions_nonpositive(ctx: ixsimpl.Context) -> None:
    x = sympy.Symbol("x", integer=True, nonpositive=True)
    assumptions = extract_assumptions(ctx, x + 1)
    assert len(assumptions) == 1
    assert str(assumptions[0]) == "x <= 0"


def test_extract_assumptions_no_info(ctx: ixsimpl.Context) -> None:
    x = sympy.Symbol("x", integer=True)
    assert extract_assumptions(ctx, x + 1) == []


def test_extract_assumptions_multiple_symbols(ctx: ixsimpl.Context) -> None:
    x = sympy.Symbol("x", integer=True, nonnegative=True)
    y = sympy.Symbol("y", integer=True, positive=True)
    z = sympy.Symbol("z", integer=True)
    assumptions = extract_assumptions(ctx, x + y + z)
    strs = {str(a) for a in assumptions}
    assert "x >= 0" in strs
    assert "-1 + y >= 0" in strs
    assert len(assumptions) == 2


def test_extract_assumptions_dedup(ctx: ixsimpl.Context) -> None:
    """Same symbol appearing twice produces one assumption."""
    x = sympy.Symbol("x", integer=True, nonnegative=True)
    assumptions = extract_assumptions(ctx, x + x * 2)
    assert len(assumptions) == 1
