# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Smoke tests for the vendored ixsimpl package."""

import sympy
import ixsimpl
from ixsimpl import floor, mod
from ixsimpl.sympy_conv import from_sympy, to_sympy, extract_assumptions


def test_create_context():
    ctx = ixsimpl.Context()
    assert ctx is not None


def test_symbol_and_constants():
    ctx = ixsimpl.Context()
    assert ctx.sym("x").sym_name == "x"
    assert str(ctx.int_(42)) == "42"
    r = ctx.rat(3, 7)
    assert r.rat_num == 3
    assert r.rat_den == 7


def test_arithmetic_simplify():
    ctx = ixsimpl.Context()
    x = ctx.sym("x")
    assert str((x + x).simplify()) == "2*x"
    assert str((x * ctx.int_(3)).simplify()) == "3*x"


def test_floor_mod_cancel():
    ctx = ixsimpl.Context()
    x = ctx.sym("x")
    expr = floor(mod(x, ctx.int_(16)) / ctx.int_(16))
    result = expr.simplify(assumptions=[x >= ctx.int_(0)])
    assert str(result) == "0"


def test_expand_cancel():
    ctx = ixsimpl.Context()
    x, y = ctx.sym("x"), ctx.sym("y")
    expr = (x + y) * (x + y) - x * x - ctx.int_(2) * x * y - y * y
    assert str(expr.expand().simplify()) == "0"


def test_sympy_roundtrip():
    ctx = ixsimpl.Context()
    x = sympy.Symbol("x", nonnegative=True)
    sympy_expr = x**2 + 2 * x + 1

    ixs_expr = from_sympy(ctx, sympy_expr)
    back = to_sympy(ixs_expr, symbols={"x": x})
    assert sympy.expand(back - sympy_expr) == 0


def test_sympy_simplify_roundtrip():
    ctx = ixsimpl.Context()
    x = sympy.Symbol("x", nonnegative=True)
    sympy_expr = sympy.floor(sympy.Mod(x, 16) / 16)

    ixs_expr = from_sympy(ctx, sympy_expr)
    assumptions = extract_assumptions(ctx, sympy_expr)
    result = to_sympy(ixs_expr.simplify(assumptions=assumptions), symbols={"x": x})
    assert result == 0
