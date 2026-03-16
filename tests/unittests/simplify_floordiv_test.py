# Copyright 2026 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the floordiv/mod divisor-splitting simplification in symbol_utils."""

import pytest
import sympy

from wave_lang.kernel.wave.utils.symbol_utils import (
    simplify,
    _split_sum_by_divisibility,
    _is_provably_divisible,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

D = sympy.Symbol("D", integer=True, positive=True)
x = sympy.Symbol("x", integer=True, nonnegative=True)
y = sympy.Symbol("y", integer=True, nonnegative=True)
z = sympy.Symbol("z", integer=True, nonnegative=True)


# ── _is_provably_divisible ───────────────────────────────────────────────────


class TestIsProvablyDivisible:
    def test_zero(self):
        assert _is_provably_divisible(sympy.Integer(0), D)

    def test_constant_divisible(self):
        assert _is_provably_divisible(sympy.Integer(12), sympy.Integer(4))

    def test_constant_not_divisible(self):
        assert not _is_provably_divisible(sympy.Integer(13), sympy.Integer(4))

    def test_symbolic_mul_factor(self):
        assert _is_provably_divisible(3 * D * x, D)

    def test_symbolic_single_factor(self):
        assert _is_provably_divisible(D * x, D)

    def test_sum_not_divisible(self):
        # x + D is a sum, not a product — not detected.
        assert not _is_provably_divisible(x + D, D)

    def test_no_divisor_factor(self):
        assert not _is_provably_divisible(x * y, D)

    def test_numeric_mul_factor(self):
        # 256*x is divisible by 8 because 256 % 8 == 0.
        assert _is_provably_divisible(256 * x, sympy.Integer(8))

    def test_numeric_mul_not_divisible(self):
        assert not _is_provably_divisible(7 * x, sympy.Integer(8))

    def test_compound_symbolic_divisor(self):
        """256*f(s)*g(s) is divisible by 8*f(s) because 256/8=32."""
        f = sympy.floor(y / 8)
        assert _is_provably_divisible(256 * f * x, 8 * f)

    def test_compound_symbolic_divisor_not_divisible(self):
        f = sympy.floor(y / 8)
        assert not _is_provably_divisible(7 * f * x, 8 * f)


# ── _split_sum_by_divisibility ───────────────────────────────────────────────


class TestSplitSumByDivisibility:
    def test_basic_split(self):
        q, r = _split_sum_by_divisibility(3 * D * x + y, D)
        assert q == 3 * x
        assert r == y

    def test_no_divisible_terms(self):
        assert _split_sum_by_divisibility(x + y, D) is None

    def test_all_divisible(self):
        q, r = _split_sum_by_divisibility(6 * x + 3, sympy.Integer(3))
        assert q == 2 * x + 1
        assert r == sympy.Integer(0)

    def test_multiple_divisible_terms(self):
        q, r = _split_sum_by_divisibility(D * x + D * y + z, D)
        assert simplify(q - (x + y)) == 0
        assert r == z


# ── simplify: floor/Mod with divisor splitting ──────────────────────────────


class TestSimplifyFloorDiv:
    def test_basic_floordiv(self):
        expr = sympy.floor((3 * D * x + y) / D)
        result = simplify(expr)
        assert result == 3 * x + sympy.floor(y / D)

    def test_floordiv_no_remainder(self):
        expr = sympy.floor(6 * D / D)
        assert simplify(expr) == 6

    def test_floordiv_bounded_remainder(self):
        # x in [0, oo) so floor(x/D) can't be eliminated without tighter
        # bounds. But the D*y term is factored out.
        expr = sympy.floor((D * y + x) / D)
        result = simplify(expr)
        assert result == y + sympy.floor(x / D)

    def test_mod_basic(self):
        expr = sympy.Mod(3 * D * x + y, D)
        result = simplify(expr)
        assert result == sympy.Mod(y, D, evaluate=False) or result == sympy.Mod(y, D)

    def test_mod_all_divisible(self):
        expr = sympy.Mod(D * x + D * y, D)
        result = simplify(expr)
        assert result == 0

    def test_scale_flat_pattern(self):
        """The actual MXFP4 scale preshuffle pattern with D%8==0 assumption."""
        k_s = sympy.Symbol("k_s", integer=True, nonnegative=True)
        n_s = sympy.Symbol("n_s", integer=True, nonnegative=True)
        m = sympy.Symbol("m", integer=True, positive=True)

        # flat with D=8*m (after divisibility substitution D%8==0)
        flat = (
            sympy.floor(n_s / 32) * m * 256
            + sympy.floor(k_s / 8) * 256
            + sympy.Mod(sympy.Mod(k_s, 8), 4) * 64
            + sympy.Mod(sympy.Mod(n_s, 32), 16) * 4
            + sympy.floor(sympy.Mod(k_s, 8) / 4) * 2
            + sympy.floor(sympy.Mod(n_s, 32) / 16)
        )

        div_expr = sympy.floor(flat / (8 * m))
        result = simplify(div_expr)
        # The 256*m*floor(n_s/32) term should be factored out as 32*floor(n_s/32).
        # The remaining floor(rest/(8*m)) should have no m in its numerator.
        assert 32 * sympy.floor(n_s / 32) in result.as_ordered_terms() or str(
            result
        ).startswith("32*floor")
        # At minimum, the result should contain 32*floor(n_s/32) as a summand.
        result_str = str(result)
        assert "32*floor(n_s/32)" in result_str, f"Expected factored term in {result}"
