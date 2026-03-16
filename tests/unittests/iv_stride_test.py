# Copyright 2026 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for IV stride extraction, mem_simplify, probe depth, and chained subs."""

import warnings

import pytest
import sympy

from wave_lang.kernel.wave.utils.symbol_utils import extract_iv
from wave_lang.kernel.wave.utils.mapping_utils import (
    mem_simplify,
    linearize_dims,
    _expand_mod,
    _eval_concrete_floor_mod,
    _compute_probe_depth,
    _extract_integer_divisors,
    _probe_iv_stride,
    _MAX_PROBE_DEPTH,
)
from wave_lang.kernel._support.indexing import IndexingContext, _resolve_chained_subs


# -- Symbols used throughout --------------------------------------------------

iv = sympy.Symbol("_j", integer=True, nonnegative=True)
x = sympy.Symbol("x", integer=True, nonnegative=True)
y = sympy.Symbol("y", integer=True, nonnegative=True)
K = sympy.Symbol("K", integer=True, positive=True)
D = sympy.Symbol("D", integer=True, positive=True)


# -- extract_iv ---------------------------------------------------------------


class TestExtractIv:
    def test_linear(self):
        """3*iv + x -> (3, x)."""
        result = extract_iv(3 * iv + x, iv)
        assert result is not None
        coeff, base = result
        assert coeff == 3
        assert base == x

    def test_no_iv(self):
        """Expression without iv -> (0, expr)."""
        result = extract_iv(x + y, iv)
        assert result is not None
        coeff, base = result
        assert coeff == 0
        assert sympy.expand(base - (x + y)) == 0

    def test_floor_divisible(self):
        """floor((16*iv + r) / 4) -> 4*iv + floor(r/4) since 4 | 16."""
        expr = sympy.floor((16 * iv + x) / 4)
        result = extract_iv(expr, iv)
        assert result is not None
        coeff, base = result
        assert coeff == 4
        assert iv not in base.free_symbols

    def test_mod_divisible(self):
        """Mod(16*iv + r, 4) -> Mod(r, 4) since 4 | 16."""
        expr = sympy.Mod(16 * iv + x, 4)
        result = extract_iv(expr, iv)
        assert result is not None
        coeff, base = result
        assert coeff == 0
        assert iv not in base.free_symbols

    def test_floor_not_divisible_returns_none(self):
        """floor((3*iv + r) / 5) -- iv cannot be separated since gcd(3,5)=1."""
        expr = sympy.floor((3 * iv + x) / 5)
        result = extract_iv(expr, iv)
        # May return None or a valid decomposition; either is acceptable.
        if result is not None:
            _, base = result
            assert iv not in base.free_symbols

    def test_combined_linear_and_floor(self):
        """2*iv + floor(8*iv/4) = 2*iv + 2*iv = 4*iv."""
        expr = 2 * iv + sympy.floor(8 * iv / 4)
        result = extract_iv(expr, iv)
        assert result is not None
        coeff, base = result
        assert coeff == 4
        assert iv not in base.free_symbols


# -- mem_simplify --------------------------------------------------------------


class TestMemSimplify:
    def test_floor_mod_roundtrip(self):
        """floor(E/D)*D + Mod(E, D) -> E."""
        E = 3 * x + 7
        expr = sympy.floor(E / D) * D + sympy.Mod(E, D)
        assert sympy.expand(mem_simplify(expr) - E) == 0

    def test_floor_mod_with_concrete_divisor(self):
        """floor(x/8)*8 + Mod(x, 8) -> x."""
        expr = sympy.floor(x / 8) * 8 + sympy.Mod(x, 8)
        assert mem_simplify(expr) == x

    def test_passthrough_integer(self):
        assert mem_simplify(sympy.Integer(42)) == 42

    def test_passthrough_symbol(self):
        assert mem_simplify(x) == x

    def test_passthrough_linear(self):
        assert sympy.expand(mem_simplify(3 * x + 5) - (3 * x + 5)) == 0

    def test_concrete_floor_evaluation(self):
        """floor(7/2) should evaluate to 3."""
        expr = sympy.floor(sympy.Rational(7, 2))
        assert mem_simplify(expr) == 3

    def test_concrete_mod_evaluation(self):
        """Mod(10, 3) should evaluate to 1."""
        expr = sympy.Mod(sympy.Integer(10), sympy.Integer(3))
        assert mem_simplify(expr) == 1


# -- linearize_dims ------------------------------------------------------------


class TestLinearizeDims:
    def test_simple_2d(self):
        """[row, col] * [stride, 1] -> row*stride + col."""
        row, col, stride = sympy.symbols(
            "row col stride", integer=True, nonnegative=True
        )
        result = linearize_dims([row, col], [stride, 1])
        assert sympy.expand(result - (row * stride + col)) == 0

    def test_floor_mod_cancellation(self):
        """[floor(flat/D), Mod(flat, D)] * [D, 1] -> flat."""
        flat = 5 * x + 3
        dims = [sympy.floor(flat / D), sympy.Mod(flat, D)]
        strides = [D, sympy.Integer(1)]
        result = linearize_dims(dims, strides)
        assert sympy.expand(result - flat) == 0


# -- _expand_mod / _eval_concrete_floor_mod ------------------------------------


class TestExpandMod:
    def test_basic(self):
        """Mod(x, D) -> x - D*floor(x/D)."""
        result = _expand_mod(sympy.Mod(x, D))
        expected = x - D * sympy.floor(x / D)
        assert sympy.expand(result - expected) == 0

    def test_no_mod_passthrough(self):
        assert _expand_mod(x + 1) == x + 1


class TestEvalConcreteFloorMod:
    def test_floor_of_integer(self):
        assert _eval_concrete_floor_mod(sympy.floor(sympy.Integer(4))) == 4

    def test_floor_of_rational(self):
        assert _eval_concrete_floor_mod(sympy.floor(sympy.Rational(7, 2))) == 3

    def test_mod_zero(self):
        assert _eval_concrete_floor_mod(sympy.Mod(sympy.Integer(0), K)) == 0

    def test_mod_concrete(self):
        assert (
            _eval_concrete_floor_mod(sympy.Mod(sympy.Integer(10), sympy.Integer(3)))
            == 1
        )

    def test_symbolic_passthrough(self):
        """floor(x/D) with symbolic args is left alone."""
        expr = sympy.floor(x / D)
        assert _eval_concrete_floor_mod(expr) == expr


# -- _compute_probe_depth / _extract_integer_divisors --------------------------


class TestComputeProbeDepth:
    def test_no_divisors(self):
        """Linear expression with no floor/Mod -> depth 1."""
        assert _compute_probe_depth(3 * iv + x, 1) == 1

    def test_single_divisor_coprime(self):
        """floor(iv/7) with coeff=1 -> period = 7/gcd(1,7) = 7."""
        expr = sympy.floor(iv / 7)
        assert _compute_probe_depth(expr, 1) == 7

    def test_single_divisor_divisible(self):
        """floor(expr/8) with coeff=4 -> period = 8/gcd(4,8) = 2."""
        # Use Rational to prevent sympy from auto-simplifying 4*iv/8 -> iv/2.
        expr = sympy.floor(iv * sympy.Rational(1, 8))
        # Divisor is 8, extracted from the denominator.
        assert _compute_probe_depth(expr, 4) == 2

    def test_multiple_divisors_lcm(self):
        """floor(iv/4) + Mod(iv, 3) with coeff=1 -> lcm(4, 3) = 12."""
        expr = sympy.floor(iv / 4) + sympy.Mod(iv, 3)
        assert _compute_probe_depth(expr, 1) == 12

    def test_zero_coeff(self):
        """Zero coefficient -> depth 1 (no IV contribution)."""
        assert _compute_probe_depth(sympy.floor(x / 8), 0) == 1

    def test_overflow_raises(self):
        """Huge coprime divisors should raise ValueError."""
        # Build an expression with many coprime divisors to exceed _MAX_PROBE_DEPTH.
        import functools
        import operator

        primes = [997, 991, 983]
        expr = sum(sympy.floor(iv / p) for p in primes)
        expected_depth = functools.reduce(operator.mul, primes, 1)
        assert expected_depth > _MAX_PROBE_DEPTH
        with pytest.raises(ValueError, match="exceeds maximum"):
            _compute_probe_depth(expr, 1)


class TestExtractIntegerDivisors:
    def test_floor_and_mod(self):
        expr = sympy.floor(x / 8) + sympy.Mod(x, 16)
        assert _extract_integer_divisors(expr) == {8, 16}

    def test_no_divisors(self):
        assert _extract_integer_divisors(x + y) == set()


# -- _probe_iv_stride ----------------------------------------------------------


class TestProbeIvStride:
    """Test the numerical probing on synthetic dim_exprs/strides."""

    def test_constant_stride(self):
        """Simple 2D layout: [iv, col] * [16, 1] -> stride = 16."""
        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        i1 = sympy.Symbol("i1", integer=True, nonnegative=True)
        dim_exprs = [i0, i1]
        strides = [sympy.Integer(16), sympy.Integer(1)]
        iters = {i0: 0, i1: 1}
        with IndexingContext():
            result = _probe_iv_stride(
                dim_exprs, strides, iters, iv_iter=i0, concrete_coeff=1
            )
        assert result == 16

    def test_scaled_coeff(self):
        """[2*iv, col] * [8, 1] -> stride = 2*8 = 16."""
        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        i1 = sympy.Symbol("i1", integer=True, nonnegative=True)
        dim_exprs = [i0, i1]
        strides = [sympy.Integer(8), sympy.Integer(1)]
        iters = {i0: 0, i1: 1}
        with IndexingContext():
            result = _probe_iv_stride(
                dim_exprs, strides, iters, iv_iter=i0, concrete_coeff=2
            )
        assert result == 16

    def test_floor_mod_cancellation(self):
        """Preshuffle pattern: [floor(flat/D), Mod(flat, D)] * [D, 1].

        flat = i0*stride + i1, so IV on i0 produces constant stride.
        """
        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        i1 = sympy.Symbol("i1", integer=True, nonnegative=True)
        flat = i0 * 128 + i1
        dim_exprs = [sympy.floor(flat / 128), sympy.Mod(flat, 128)]
        strides = [sympy.Integer(128), sympy.Integer(1)]
        iters = {i0: 0, i1: 1}
        with IndexingContext():
            result = _probe_iv_stride(
                dim_exprs, strides, iters, iv_iter=i0, concrete_coeff=1
            )
        # floor((iv*128+i1)/128)*128 + Mod(iv*128+i1, 128) = iv*128 + i1.
        # Stride on i0 with coeff=1 -> 128.
        assert result == 128

    def test_cyclic_stride(self):
        """dim = [Mod(iv, 2)] * [1] -> diffs cycle [1, -1]."""
        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        dim_exprs = [sympy.Mod(i0, 2)]
        strides = [sympy.Integer(1)]
        iters = {i0: 0}
        with IndexingContext():
            result = _probe_iv_stride(
                dim_exprs, strides, iters, iv_iter=i0, concrete_coeff=1
            )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == 1
        assert result[1] == -1


# -- _resolve_chained_subs ----------------------------------------------------


class TestResolveChainedSubs:
    def test_simple_chain(self):
        """K=8192, K_SCALE=K//32 -> K_SCALE=256."""
        K_sym = sympy.Symbol("K", integer=True, positive=True)
        KS = sympy.Symbol("K_SCALE", integer=True, positive=True)
        subs = {K_sym: 8192, KS: sympy.floor(K_sym / 32)}
        result = _resolve_chained_subs(subs)
        assert result[K_sym] == 8192
        assert result[KS] == 256

    def test_no_dependencies(self):
        """Independent entries are returned as-is."""
        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        subs = {a: 10, b: 20}
        result = _resolve_chained_subs(subs)
        assert result == {a: 10, b: 20}

    def test_circular_warns(self):
        """Circular dependencies should emit a warning."""
        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        subs = {a: b + 1, b: a + 1}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _resolve_chained_subs(subs)
            assert len(w) == 1
            assert "circular" in str(w[0].message).lower()

    def test_multi_level_chain(self):
        """A=1, B=A+1, C=B*2 -> C=4."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        C = sympy.Symbol("C")
        subs = {A: 1, B: A + 1, C: B * 2}
        result = _resolve_chained_subs(subs)
        assert result[A] == 1
        assert result[B] == 2
        assert result[C] == 4
