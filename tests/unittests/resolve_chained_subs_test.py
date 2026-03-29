# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for _resolve_chained_subs and its integration into IndexingContext."""

import logging
import unittest
import warnings

import sympy

from wave_lang.kernel.lang import sym
from wave_lang.kernel._support.indexing import (
    IndexingContext,
    _resolve_chained_subs,
)

M = sym.M
N = sym.N
K = sym.K


class ResolveChainedSubsTest(unittest.TestCase):
    """Direct unit tests for _resolve_chained_subs."""

    def test_no_chaining(self):
        """All values are concrete — nothing to resolve."""
        result = _resolve_chained_subs({M: 256, N: 128, K: 8192})
        assert result[M] == 256
        assert result[N] == 128
        assert result[K] == 8192

    def test_single_chain(self):
        """K_SCALE depends on K via K // 32."""
        K_SCALE = sympy.Symbol("K_SCALE")
        subs = {K: 8192, K_SCALE: K // 32}
        result = _resolve_chained_subs(subs)
        assert result[K] == 8192
        assert result[K_SCALE] == 256

    def test_multi_level_chain(self):
        """A -> B -> C: three-level dependency chain."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        C = sympy.Symbol("C")
        subs = {A: 1024, B: A * 2, C: B + 1}
        result = _resolve_chained_subs(subs)
        assert result[A] == 1024
        assert result[B] == 2048
        assert result[C] == 2049

    def test_independent_symbolic_values(self):
        """Values reference a symbol not in the key set — left as-is."""
        X = sympy.Symbol("X")
        subs = {M: X + 1}
        result = _resolve_chained_subs(subs)
        assert result[M] == X + 1

    def test_self_referencing_key_ignored(self):
        """A value referencing its own key doesn't count as a dependency."""
        A = sympy.Symbol("A")
        subs = {A: A + 1, M: 64}
        result = _resolve_chained_subs(subs)
        assert result[A] == A + 1
        assert result[M] == 64

    def test_circular_dependency_warns(self):
        """Two symbols that depend on each other should warn."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        subs = {A: B + 1, B: A + 1}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_chained_subs(subs)
            assert any("circular dependency" in str(x.message).lower() for x in w)
        assert A in result and B in result

    def test_floor_div_chain(self):
        """Floor-division chain resolves correctly."""
        TILE = sympy.Symbol("TILE")
        NUM_TILES = sympy.Symbol("NUM_TILES")
        subs = {K: 4096, TILE: 128, NUM_TILES: K // TILE}
        result = _resolve_chained_subs(subs)
        assert result[NUM_TILES] == 32

    def test_piecewise_chain(self):
        """Piecewise expressions in values are handled."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        pw = sympy.Piecewise((A, A > 0), (0, True))
        subs = {A: 42, B: pw}
        result = _resolve_chained_subs(subs)
        assert result[A] == 42
        assert result[B] == 42

    def test_empty_dict(self):
        """Vacuous input returns an empty dict."""
        result = _resolve_chained_subs({})
        assert result == {}

    def test_diamond_dependency(self):
        """Diamond: D depends on B and C, both depend on A."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        C = sympy.Symbol("C")
        D = sympy.Symbol("D")
        subs = {A: 10, B: A * 2, C: A + 3, D: B + C}
        result = _resolve_chained_subs(subs)
        assert result[A] == 10
        assert result[B] == 20
        assert result[C] == 13
        assert result[D] == 33

    def test_order_independence(self):
        """Result is the same regardless of insertion order."""
        K_SCALE = sympy.Symbol("K_SCALE")
        subs_a = {K_SCALE: K // 32, K: 8192}
        subs_b = {K: 8192, K_SCALE: K // 32}
        assert _resolve_chained_subs(subs_a) == _resolve_chained_subs(subs_b)


class IndexingContextSetSubsTest(unittest.TestCase):
    """Integration tests: set_subs should resolve chains before storing."""

    def test_set_subs_resolves_chain(self):
        K_SCALE = sympy.Symbol("K_SCALE")
        with IndexingContext() as idxc:
            idxc.set_subs({K: 8192, K_SCALE: K // 32})
            assert idxc.subs[K_SCALE] == 256

    def test_subs_expr_uses_resolved_values(self):
        K_SCALE = sympy.Symbol("K_SCALE")
        expr = K_SCALE * 2
        with IndexingContext() as idxc:
            idxc.set_subs({K: 8192, K_SCALE: K // 32})
            result = idxc.subs_expr(expr)
            assert result == 512

    def test_set_subs_does_not_mutate_original(self):
        K_SCALE = sympy.Symbol("K_SCALE")
        original = {K: 8192, K_SCALE: K // 32}
        original_copy = dict(original)
        with IndexingContext() as idxc:
            idxc.set_subs(original)
        assert original == original_copy


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
