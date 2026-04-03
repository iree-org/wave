# Copyright 2026 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for linearize_dims and mem_simplify.

``linearize_dims`` computes ``sum(dim_i * stride_i)`` with automatic
cancellation of paired ``floor/Mod`` terms that arise from preshuffle
index mappings.  It delegates to ``mem_simplify``, which rewrites
``Mod(x, d)`` as ``x - d*floor(x/d)`` and then ``expand()`` cancels
the paired floor terms, exploiting the identity::

    floor(E / D) * D + Mod(E, D)  ==  E

SymPy cannot simplify this on its own because ``Mod`` is an opaque node.
"""

import pytest
import sympy

from wave_lang.kernel.wave.utils.mapping_utils import linearize_dims, mem_simplify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sym(name, **kw):
    """Non-negative integer symbol (matches hardware index convention)."""
    kw.setdefault("integer", True)
    kw.setdefault("nonnegative", True)
    return sympy.Symbol(name, **kw)


def _pos(name):
    """Positive integer symbol."""
    return sympy.Symbol(name, integer=True, positive=True)


def _verify_numerically(result, expected, free_syms, n_probes=20):
    """Check that result == expected for many concrete values.

    Substitutes co-prime probe values into all free symbols and checks
    integer equality.  This catches simplification bugs where the
    symbolic form looks different but is numerically equivalent.
    """
    diff = sympy.expand(result - expected)
    if diff == 0:
        return
    all_syms = sorted(free_syms, key=str)
    for trial in range(n_probes):
        probe = {s: 31 + trial * 17 + i * 7 for i, s in enumerate(all_syms)}
        val = diff.subs(probe)
        assert val == 0, (
            f"Numerical mismatch at probe {probe}: "
            f"result={result.subs(probe)}, expected={expected.subs(probe)}"
        )


# ===========================================================================
# mem_simplify: floor/Mod cancellation
# ===========================================================================


class TestMemSimplify:
    """mem_simplify should cancel paired floor(E/D)*D + Mod(E, D) -> E."""

    def test_floor_mod_roundtrip(self):
        """The fundamental identity: floor(E/D)*D + Mod(E,D) == E."""
        E = _sym("E")
        D = _pos("D")
        expr = sympy.floor(E / D) * D + sympy.Mod(E, D)
        assert mem_simplify(expr) == E

    def test_already_simple(self):
        x = _sym("x")
        assert mem_simplify(x + 3) == x + 3

    def test_pure_number(self):
        assert mem_simplify(sympy.Integer(42)) == 42

    def test_nested_floor_concrete(self):
        """floor(4) left over from symbolic substitution should collapse."""
        assert mem_simplify(sympy.floor(sympy.Integer(4))) == 4

    def test_concrete_mod_zero(self):
        """Mod(0, K) should simplify to 0."""
        K = _pos("K")
        expr = sympy.Mod(sympy.Integer(0), K)
        assert mem_simplify(expr) == 0

    def test_concrete_floor(self):
        """floor(7) should collapse to 7."""
        assert mem_simplify(sympy.floor(sympy.Integer(7))) == 7

    def test_no_mod_passthrough(self):
        """Expressions without Mod should not be damaged."""
        x, y = _sym("x"), _sym("y")
        expr = 3 * x + 2 * y + 1
        assert mem_simplify(expr) == expr


# ===========================================================================
# Basic linearization (no floor/Mod)
# ===========================================================================


class TestBasicLinearization:
    """Simple row-major linearization without floor/Mod cancellation."""

    def test_1d_identity(self):
        """Single dimension with stride 1 is identity."""
        x = _sym("x")
        assert linearize_dims([x], [sympy.Integer(1)]) == x

    def test_2d_row_major(self):
        """row * cols + col."""
        row, col = _sym("row"), _sym("col")
        cols = _pos("cols")
        result = linearize_dims([row, col], [cols, sympy.Integer(1)])
        expected = row * cols + col
        assert sympy.expand(result - expected) == 0

    def test_2d_concrete_strides(self):
        """Concrete integer strides: 256 and 1."""
        row, col = _sym("row"), _sym("col")
        result = linearize_dims([row, col], [sympy.Integer(256), sympy.Integer(1)])
        assert sympy.expand(result - (row * 256 + col)) == 0

    def test_3d_row_major(self):
        """d0 * s1*s2 + d1 * s2 + d2."""
        d0, d1, d2 = _sym("d0"), _sym("d1"), _sym("d2")
        s1, s2 = _pos("s1"), _pos("s2")
        result = linearize_dims([d0, d1, d2], [s1 * s2, s2, sympy.Integer(1)])
        expected = d0 * s1 * s2 + d1 * s2 + d2
        assert sympy.expand(result - expected) == 0

    def test_4d_strides(self):
        """4-D tensor: batch * (C*H*W) + c * (H*W) + h * W + w."""
        b, c, h, w = _sym("b"), _sym("c"), _sym("h"), _sym("w")
        C, H, W = _pos("C"), _pos("H"), _pos("W")
        strides = [C * H * W, H * W, W, sympy.Integer(1)]
        result = linearize_dims([b, c, h, w], strides)
        expected = b * C * H * W + c * H * W + h * W + w
        assert sympy.expand(result - expected) == 0

    def test_all_zero_dims(self):
        """Zero-valued dimensions produce zero offset."""
        zero = sympy.Integer(0)
        result = linearize_dims([zero, zero], [sympy.Integer(128), sympy.Integer(1)])
        assert result == 0


# ===========================================================================
# 2-D preshuffle round-trip
# ===========================================================================


class TestPreshuffle2D:
    """The canonical preshuffle pattern: [floor(E/D), Mod(E,D)] * [D, 1] -> E."""

    def test_symbolic_divisor(self):
        """floor(x/D)*D + Mod(x,D) -> x with symbolic D."""
        x = _sym("x")
        D = _pos("D")
        dims = [sympy.floor(x / D), sympy.Mod(x, D)]
        strides = [D, sympy.Integer(1)]
        assert linearize_dims(dims, strides) == x

    def test_concrete_divisor(self):
        """floor(x/16)*16 + Mod(x,16) -> x."""
        x = _sym("x")
        dims = [sympy.floor(x / 16), sympy.Mod(x, 16)]
        strides = [sympy.Integer(16), sympy.Integer(1)]
        assert linearize_dims(dims, strides) == x

    def test_with_offset(self):
        """floor((x+c)/D)*D + Mod((x+c),D) -> x+c."""
        x = _sym("x")
        D = _pos("D")
        inner = x + sympy.Integer(42)
        dims = [sympy.floor(inner / D), sympy.Mod(inner, D)]
        strides = [D, sympy.Integer(1)]
        assert linearize_dims(dims, strides) == inner

    def test_complex_inner_expression(self):
        """Preshuffle of a compound expression: thread scramble + iv*stride."""
        t = _sym("t")
        iv = _sym("iv")
        D = _pos("D")
        inner = t * 16 + iv * 256
        dims = [sympy.floor(inner / D), sympy.Mod(inner, D)]
        strides = [D, sympy.Integer(1)]
        assert linearize_dims(dims, strides) == inner

    def test_K_half_symbolic(self):
        """Preshuffle with D = K/2 (common in MXFP4 B-data)."""
        x = _sym("x")
        K = _pos("K")
        half_K = K / 2
        dims = [sympy.floor(x / half_K), sympy.Mod(x, half_K)]
        strides = [half_K, sympy.Integer(1)]
        assert linearize_dims(dims, strides) == x

    def test_K_half_concrete(self):
        """Preshuffle with D = 4096 (K=8192 resolved)."""
        x = _sym("x")
        D = sympy.Integer(4096)
        dims = [sympy.floor(x / D), sympy.Mod(x, D)]
        strides = [D, sympy.Integer(1)]
        assert linearize_dims(dims, strides) == x

    def test_reversed_dim_order(self):
        """Mod first, floor second: Mod(x,D)*1 + floor(x/D)*D -> x."""
        x = _sym("x")
        D = _pos("D")
        dims = [sympy.Mod(x, D), sympy.floor(x / D)]
        strides = [sympy.Integer(1), D]
        assert linearize_dims(dims, strides) == x


# ===========================================================================
# Mismatched strides (no cancellation expected)
# ===========================================================================


class TestMismatchedStrides:
    """When strides don't match the Mod divisor, no cancellation occurs.

    The result is still correct (a valid linearized expression) but it
    retains floor/Mod terms.
    """

    def test_stride_double_divisor(self):
        """row * 2*D + Mod(x, D): stride != D, so floor/Mod persist."""
        x = _sym("x")
        D = _pos("D")
        dims = [sympy.floor(x / D), sympy.Mod(x, D)]
        strides = [2 * D, sympy.Integer(1)]
        result = linearize_dims(dims, strides)
        # Should NOT simplify to x (stride is 2D, not D).
        # But it should be numerically correct.
        expected = sympy.floor(x / D) * 2 * D + sympy.Mod(x, D)
        _verify_numerically(result, expected, {x, D})

    def test_non_unit_col_stride(self):
        """floor(x/D)*D + Mod(x,D)*2: col stride is 2, not 1."""
        x = _sym("x")
        D = _pos("D")
        dims = [sympy.floor(x / D), sympy.Mod(x, D)]
        strides = [D, sympy.Integer(2)]
        result = linearize_dims(dims, strides)
        # floor(x/D)*D + 2*Mod(x,D) != x in general
        expected = sympy.floor(x / D) * D + 2 * sympy.Mod(x, D)
        _verify_numerically(result, expected, {x, D})


# ===========================================================================
# Nested and multi-level preshuffle
# ===========================================================================


class TestNestedPreshuffle:
    """Higher-dimensional and nested floor/Mod decompositions."""

    def test_3d_two_level_preshuffle(self):
        """Two-level decomposition: delinearize then re-linearize.

        E -> (floor(E/B), Mod(E, B)) is 2D.
        For 3D: E -> (floor(E/(S1*S2)), floor(Mod(E, S1*S2) / S2), Mod(E, S2)).
        strides = [S1*S2, S2, 1] should reconstruct E.
        """
        E = _sym("E")
        S1 = _pos("S1")
        S2 = _pos("S2")
        d0 = sympy.floor(E / (S1 * S2))
        d1 = sympy.floor(sympy.Mod(E, S1 * S2) / S2)
        d2 = sympy.Mod(E, S2)
        strides = [S1 * S2, S2, sympy.Integer(1)]
        result = linearize_dims([d0, d1, d2], strides)
        _verify_numerically(result, E, {E, S1, S2})

    def test_mixed_preshuffle_and_plain(self):
        """One dim is a preshuffle pair, another is a plain index.

        batch * (D) + floor(x/D)*D + Mod(x,D) -> batch*D + x.
        """
        batch = _sym("batch")
        x = _sym("x")
        D = _pos("D")
        dims = [batch, sympy.floor(x / D), sympy.Mod(x, D)]
        strides = [D, D, sympy.Integer(1)]
        result = linearize_dims(dims, strides)
        # batch*D + floor(x/D)*D + Mod(x,D) -> batch*D + x
        # But the first two dims both have stride D, so:
        # result = batch*D + floor(x/D)*D + Mod(x,D)
        # mem_simplify cancels the floor/Mod pair:
        expected = batch * D + x
        assert sympy.expand(result - expected) == 0


# ===========================================================================
# MXFP4 preshuffle thread-ID scrambling
# ===========================================================================


class TestMXFP4Patterns:
    """Realistic expressions from MXFP4 preshuffle GEMM codegen."""

    @staticmethod
    def _preshuffle_thread_offset(t):
        """Thread scramble from MXFP4 B-data preshuffle mapping.

        For t in [0, 63], this equals t*16, but the symbolic form has
        floor/Mod terms that sympy can't simplify.
        """
        return (
            t * 16
            + sympy.floor(sympy.Mod(t, 64) / 16) * 256
            - sympy.floor(t / 16) * 256
        )

    def test_b_data_linearize_K8192(self):
        """B-data with K=8192: [floor(E/4096), Mod(E,4096)] * [4096, 1] -> E.

        The thread offset has its own floor/Mod structure which
        mem_simplify may further simplify.  The result is numerically
        equal to the original inner expression.
        """
        t = _sym("t")
        iv = _sym("iv")
        D = sympy.Integer(4096)
        inner = self._preshuffle_thread_offset(t) + iv * D
        dims = [sympy.floor(inner / D), sympy.Mod(inner, D)]
        strides = [D, sympy.Integer(1)]
        result = linearize_dims(dims, strides)
        _verify_numerically(result, inner, {t, iv})

    def test_b_data_linearize_symbolic_K(self):
        """B-data with symbolic K: [floor(E/(K/2)), Mod(E,K/2)] * [K/2, 1] -> E."""
        t = _sym("t")
        iv = _sym("iv")
        K = _pos("K")
        half_K = K / 2
        inner = self._preshuffle_thread_offset(t) + iv * half_K
        dims = [sympy.floor(inner / half_K), sympy.Mod(inner, half_K)]
        strides = [half_K, sympy.Integer(1)]
        result = linearize_dims(dims, strides)
        _verify_numerically(result, inner, {t, iv, K})

    def test_b_scale_linearize_K8192(self):
        """B-scale with K=8192: [floor(E/256), Mod(E,256)] * [256, 1] -> E.

        The thread part has Mod(t,16)*4 + floor(t/16)*64, which
        mem_simplify may further reduce.  Numerically equivalent.
        """
        t = _sym("t")
        iv = _sym("iv")
        D = sympy.Integer(256)
        thread_part = sympy.Mod(t, 16) * 4 + sympy.floor(t / 16) * 64
        inner = thread_part + iv * D
        dims = [sympy.floor(inner / D), sympy.Mod(inner, D)]
        strides = [D, sympy.Integer(1)]
        result = linearize_dims(dims, strides)
        _verify_numerically(result, inner, {t, iv})

    def test_preserves_iv_linearity(self):
        """After linearization, the IV still appears linearly.

        If linearize_dims correctly cancels floor/Mod, the result is
        base(t) + iv*stride, and (result at iv+1) - (result at iv) is
        a constant.
        """
        t = _sym("t")
        iv = _sym("iv")
        D = sympy.Integer(4096)
        inner = self._preshuffle_thread_offset(t) + iv * D
        dims = [sympy.floor(inner / D), sympy.Mod(inner, D)]
        strides = [D, sympy.Integer(1)]
        flat = linearize_dims(dims, strides)
        diff = flat.subs({iv: iv + 1}) - flat
        assert sympy.expand(diff) == D


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:

    def test_single_element(self):
        """Single dim, stride 1: identity."""
        e = _sym("e")
        assert linearize_dims([e], [sympy.Integer(1)]) == e

    def test_single_element_with_stride(self):
        """Single dim with stride > 1."""
        e = _sym("e")
        s = _pos("s")
        result = linearize_dims([e], [s])
        assert sympy.expand(result - e * s) == 0

    def test_all_concrete(self):
        """All dims and strides are concrete integers."""
        result = linearize_dims(
            [sympy.Integer(3), sympy.Integer(7)],
            [sympy.Integer(10), sympy.Integer(1)],
        )
        assert result == 37

    def test_zero_stride_dim(self):
        """A dimension with stride 0 contributes nothing."""
        x, y = _sym("x"), _sym("y")
        result = linearize_dims([x, y], [sympy.Integer(0), sympy.Integer(1)])
        assert result == y

    def test_large_concrete_preshuffle(self):
        """Concrete preshuffle with large D=65536."""
        x = _sym("x")
        D = sympy.Integer(65536)
        dims = [sympy.floor(x / D), sympy.Mod(x, D)]
        strides = [D, sympy.Integer(1)]
        assert linearize_dims(dims, strides) == x

    def test_empty_lists(self):
        """Empty dim/stride lists produce 0."""
        result = linearize_dims([], [])
        assert result == 0
