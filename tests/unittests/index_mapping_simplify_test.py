# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for IndexMapping flat//D and flat%D simplification."""

import sympy
import pytest

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.wave_types import IndexMapping
from wave_lang.kernel.wave.index_mapping_simplify import (
    simplify_index_mapping,
    _get_iterator_bounds,
    _expr_bounds_with_iters,
)

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K


class TestSimplifyIndexMapping:
    def test_flat_div_mod_same_dim(self):
        """flat = i0*K + i1, inputs={M: flat//K, K: flat%K} -> {M: i0, K: i1}."""
        i0 = IndexMapping.iterator(0)
        i1 = IndexMapping.iterator(1)

        flat = i0 * K + i1
        m = IndexMapping(
            num_iterators=2,
            inputs={M: flat // K, K: flat % K},
            outputs={M: i0, K: i1},
        )

        m_new, changed = simplify_index_mapping(m)
        assert changed
        assert m_new.input_mapping[M] == i0
        assert m_new.input_mapping[K] == i1

    def test_flat_div_mod_with_addend(self):
        """flat//K added to another term: N: (i0//16)*16 + flat//K."""
        i0 = IndexMapping.iterator(0)
        i1 = IndexMapping.iterator(1)

        flat = i0 * K + i1
        m = IndexMapping(
            num_iterators=2,
            inputs={N: (i0 // 16) * 16 + flat // K, K: flat % K},
            outputs={N: i0, K: i1},
        )

        m_new, changed = simplify_index_mapping(m)
        assert changed
        # N should be (i0//16)*16 + i0  (the flat//K simplified to i0)
        expected_n = (i0 // 16) * 16 + i0
        assert sympy.simplify(m_new.input_mapping[N] - expected_n) == 0
        assert m_new.input_mapping[K] == i1

    def test_no_simplification_when_bounds_unknown(self):
        """When divisor doesn't match iteration dimension, no simplification."""
        i0 = IndexMapping.iterator(0)
        i1 = IndexMapping.iterator(1)
        D = sympy.Symbol("D", integer=True, positive=True)

        flat = i0 * D + i1
        m = IndexMapping(
            num_iterators=2,
            inputs={M: flat // D, K: flat % D},
            outputs={M: i0, K: i1},
        )

        m_new, changed = simplify_index_mapping(m)
        # D doesn't match any iteration dimension (M or K), so we can't
        # prove i1 < D.
        assert not changed

    def test_no_simplification_b_data_preshuffle(self):
        """B-data preshuffle: within_nblk can exceed K_PACKED for general K."""
        n_it = IndexMapping.iterator(0)
        k_it = IndexMapping.iterator(1)

        within_nblk = (
            (k_it // 32) * 512
            + ((k_it // 16) % 2) * 256
            + (n_it % 16) * 16
            + k_it % 16
        )
        K_PACKED = K // 2

        m = IndexMapping(
            num_iterators=2,
            inputs={
                N: (n_it // 16) * 16 + within_nblk // K_PACKED,
                K: within_nblk % K_PACKED,
            },
            outputs={N: n_it, K: k_it},
        )

        m_new, changed = simplify_index_mapping(m)
        assert not changed


class TestExprBoundsWithIters:
    def test_iterator_bounds(self):
        i0 = IndexMapping.iterator(0)
        i1 = IndexMapping.iterator(1)
        bounds = {i0: (sympy.Integer(0), sympy.Integer(15)),
                  i1: (sympy.Integer(0), sympy.Integer(63))}

        assert _expr_bounds_with_iters(i0, bounds) == (0, 15)
        assert _expr_bounds_with_iters(i1, bounds) == (0, 63)

    def test_within_nblk_bounds(self):
        """within_nblk for tile [0,15]x[0,63] is bounded to [0,1023]."""
        n_it = IndexMapping.iterator(0)
        k_it = IndexMapping.iterator(1)
        bounds = {n_it: (sympy.Integer(0), sympy.Integer(15)),
                  k_it: (sympy.Integer(0), sympy.Integer(63))}

        within_nblk = (
            (k_it // 32) * 512
            + ((k_it // 16) % 2) * 256
            + (n_it % 16) * 16
            + k_it % 16
        )
        result = _expr_bounds_with_iters(within_nblk, bounds)
        assert result is not None
        assert result[0] == 0
        assert result[1] == 1023
