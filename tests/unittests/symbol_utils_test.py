# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for interval-arithmetic simplification in symbol_utils."""

import pytest
import sympy

from wave_lang.kernel.wave.utils.symbol_utils import expr_bounds, simplify


def _sym(name: str) -> sympy.Symbol:
    """Create a non-negative integer symbol (mimics hardware indices)."""
    return sympy.Symbol(name, integer=True, nonnegative=True)


# Parametrize simplify tests with both our implementation and plain sympy.
# The sympy variant is xfail(strict=True) so that if upstream sympy learns
# bounds-based simplification, the unexpected pass alerts us.
_SIMPLIFY_PARAMS = [
    pytest.param(simplify, id="wave"),
    pytest.param(
        sympy.simplify,
        id="sympy",
        marks=pytest.mark.xfail(
            strict=True,
            reason="sympy lacks bounds-based floor/Mod simplification",
        ),
    ),
]


# ---- expr_bounds tests ----


def test_bounds_integer():
    assert expr_bounds(sympy.Integer(5)) == (5, 5)
    assert expr_bounds(sympy.Integer(0)) == (0, 0)
    assert expr_bounds(sympy.Integer(-3)) == (-3, -3)


def test_bounds_rational():
    r = sympy.Rational(7, 2)
    assert expr_bounds(r) == (r, r)


def test_bounds_nonneg_symbol():
    x = _sym("x")
    lo, hi = expr_bounds(x)
    assert lo == 0
    assert hi == sympy.oo


def test_bounds_negative_symbol_returns_none():
    x = sympy.Symbol("x")
    assert expr_bounds(x) is None


def test_bounds_mod_basic():
    x = _sym("x")
    assert expr_bounds(sympy.Mod(x, 16)) == (0, 15)


def test_bounds_mod_tightens_when_range_fits():
    # Mod(x, 64) is in [0, 63]; Mod(Mod(x,64), 128) should keep [0, 63].
    x = _sym("x")
    inner = sympy.Mod(x, 64, evaluate=False)
    outer = sympy.Mod(inner, 128, evaluate=False)
    assert expr_bounds(outer) == (0, 63)


def test_bounds_floor():
    x = _sym("x")
    # floor(Mod(x, 16) / 16) -> bounds [0, 0].
    inner = sympy.Mod(x, 16, evaluate=False) / 16
    assert expr_bounds(sympy.floor(inner)) == (0, 0)


def test_bounds_add():
    x, y = _sym("x"), _sym("y")
    mx = sympy.Mod(x, 4, evaluate=False)
    my = sympy.Mod(y, 8, evaluate=False)
    assert expr_bounds(mx + my) == (0, 10)


def test_bounds_mul_positive():
    x = _sym("x")
    m = sympy.Mod(x, 4, evaluate=False)
    assert expr_bounds(3 * m) == (0, 9)


def test_bounds_mul_with_negative():
    x = _sym("x")
    m = sympy.Mod(x, 4, evaluate=False)
    assert expr_bounds(-2 * m) == (-6, 0)


def test_bounds_mul_with_infinity_returns_none():
    x = _sym("x")
    assert expr_bounds(x * x) is None


def test_bounds_unsupported_returns_none():
    x = _sym("x")
    assert expr_bounds(x**2) is None


# ---- simplify tests ----


def test_simplify_integer_passthrough():
    assert simplify(sympy.Integer(42)) == 42


def test_simplify_plain_python_int():
    assert simplify(3) == 3


@pytest.mark.parametrize("simplify_fn", _SIMPLIFY_PARAMS)
def test_simplify_floor_of_bounded_fraction(simplify_fn):
    # floor(Mod(x,16)/16) should resolve to 0.
    x = _sym("x")
    expr = sympy.floor(sympy.Mod(x, 16, evaluate=False) / 16)
    assert simplify_fn(expr) == 0


def test_simplify_mod_elimination():
    # Mod(Mod(x,8), 16) -> Mod(x,8) since range [0,7] < 16.
    x = _sym("x")
    inner = sympy.Mod(x, 8, evaluate=False)
    outer = sympy.Mod(inner, 16, evaluate=False)
    assert simplify(outer) == inner


@pytest.mark.parametrize("simplify_fn", _SIMPLIFY_PARAMS)
def test_simplify_sympy_bug_workaround(simplify_fn):
    # Mod(2*Mod(x, 4), 64) should NOT be auto-evaluated incorrectly.
    # See https://github.com/sympy/sympy/issues/28744.
    # Range of 2*Mod(x,4) is [0,6] < 64, so Mod should be eliminated.
    x = _sym("x")
    inner = sympy.Mod(x, 4, evaluate=False)
    expr = sympy.Mod(2 * inner, 64, evaluate=False)
    assert simplify_fn(expr) == 2 * sympy.Mod(x, 4, evaluate=False)


# ---- Real expressions from the MXFP4 e8m0_shuffle merge pass ----


def _shuffle_base(t0):
    """Common sub-expression from e8m0_shuffle mapping.

    Computes: 4*Mod(t0, 16) + floor(Mod(t0, 16)/16)
              + 2*floor(Mod(floor(Mod(t0, 64)/16), 8)/4)
    """
    mod16 = sympy.Mod(t0, 16, evaluate=False)
    mod64 = sympy.Mod(t0, 64, evaluate=False)
    return (
        4 * mod16
        + sympy.floor(mod16 / 16)
        + 2 * sympy.floor(sympy.Mod(sympy.floor(mod64 / 16), 8, evaluate=False) / 4)
    )


@pytest.mark.parametrize("simplify_fn", _SIMPLIFY_PARAMS)
def test_shuffle_offset_diff_equals_1(simplify_fn):
    """Adjacent shuffle positions differ by 1 in physical offset."""
    t0 = _sym("$T0")
    base = _shuffle_base(t0)
    mod16 = sympy.Mod(t0, 16, evaluate=False)
    mod64 = sympy.Mod(t0, 64, evaluate=False)
    shifted = (
        4 * mod16
        + sympy.floor(sympy.Mod(mod16 + 16, 32, evaluate=False) / 16)
        + 2 * sympy.floor(sympy.Mod(sympy.floor(mod64 / 16), 8, evaluate=False) / 4)
    )
    expr = sympy.Mod(shifted, 64, evaluate=False) - sympy.Mod(base, 64, evaluate=False)
    assert simplify_fn(expr) == 1


@pytest.mark.parametrize("simplify_fn", _SIMPLIFY_PARAMS)
def test_shuffle_offset_diff_equals_2(simplify_fn):
    """Stride-2 shuffle positions differ by 2 in physical offset."""
    t0 = _sym("$T0")
    base = _shuffle_base(t0)
    mod16 = sympy.Mod(t0, 16, evaluate=False)
    mod64 = sympy.Mod(t0, 64, evaluate=False)
    shifted = (
        4 * mod16
        + sympy.floor(mod16 / 16)
        + 2 * sympy.floor(sympy.Mod(sympy.floor(mod64 / 16) + 4, 8, evaluate=False) / 4)
    )
    expr = sympy.Mod(shifted, 64, evaluate=False) - sympy.Mod(base, 64, evaluate=False)
    assert simplify_fn(expr) == 2


@pytest.mark.parametrize("simplify_fn", _SIMPLIFY_PARAMS)
def test_shuffle_floor_resolves_to_constant(simplify_fn):
    """Sub-expressions from the shuffle mapping resolve to known constants.

    floor(Mod(Mod(t0,16)+16, 32)/16) == 1 for any nonneg t0, because
    Mod(t0,16) in [0,15], +16 -> [16,31], Mod(.,32) -> [16,31],
    /16 -> [1, 31/16], floor -> 1.
    """
    t0 = _sym("$T0")
    mod16 = sympy.Mod(t0, 16, evaluate=False)
    expr = sympy.floor(sympy.Mod(mod16 + 16, 32, evaluate=False) / 16)
    assert simplify_fn(expr) == 1


def test_shuffle_floor_mod_nesting_bounds():
    """Deeply nested floor(Mod(...)/N) has expected bounds."""
    x = _sym("x")
    mod64 = sympy.Mod(x, 64, evaluate=False)
    inner = sympy.Mod(mod64, 8, evaluate=False)
    expr = sympy.floor(inner / 4)
    assert expr_bounds(expr) == (0, 1)
