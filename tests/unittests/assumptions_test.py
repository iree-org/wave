# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import sympy
import unittest
from wave_lang.kernel.lang import sym
from wave_lang.kernel.wave.assumptions import Assumption
from wave_lang.kernel.wave.utils.general_utils import evaluate_with_assumptions
from wave_lang.kernel.wave.analysis.partition_strided_operators import (
    _get_divisibility_subs,
    _simplify_expr,
)
from wave_lang.kernel._support.indexing import IndexingContext

M = sym.M
N = sym.N
K = sym.K
BLOCK_N = sym.BLOCK_N
BLOCK_M = sym.BLOCK_K
I = sym.I


class AssumptionsTest(unittest.TestCase):
    def testAssumption(self):
        with IndexingContext():
            constraints: list[Assumption] = []
            constraints.append(Assumption(M < 64))
            assert evaluate_with_assumptions(constraints, M > 70) == False
            assert evaluate_with_assumptions(constraints, M < 70) == True
            assert evaluate_with_assumptions(constraints, M < 32) is None


class DivisibilityAssumptionTest(unittest.TestCase):
    """Test that divisibility assumptions simplify floor/Mod expressions."""

    def setUp(self):
        self.idxc = IndexingContext()
        self.idxc.__enter__()
        self.constraints = [Assumption(sympy.Eq(K % 32, 0))]
        self.fwd, self.bwd = _get_divisibility_subs(self.constraints)

    def tearDown(self):
        self.idxc.__exit__(None, None, None)

    def _simplify(self, expr):
        return _simplify_expr(expr, self.fwd, self.bwd)

    def testFloorExact(self):
        # floor(K/32) -> K/32 when 32 | K.
        assert self._simplify(sympy.floor(K / 32)) == K / 32

    def testModExact(self):
        # Mod(K, 32) -> 0 when 32 | K.
        assert self._simplify(sympy.Mod(K, 32)) == 0

    def testFloorSubMultiple(self):
        # floor(K/16) -> K/16 since 32 | K implies 16 | K.
        assert self._simplify(sympy.floor(K / 16)) == K / 16

    def testModSuperMultiple(self):
        # Mod(K, 64) should NOT simplify to 0 (32 | K does not imply 64 | K).
        result = self._simplify(sympy.Mod(K, 64))
        assert result != 0

    def testScaledMod(self):
        # Mod(3*K, 32) -> 0 when 32 | K.
        assert self._simplify(sympy.Mod(3 * K, 32)) == 0

    def testNoAssumptions(self):
        # Without assumptions, expressions pass through unchanged.
        fwd, bwd = _get_divisibility_subs([])
        expr = sympy.floor(K / 32)
        result = _simplify_expr(expr, fwd, bwd)
        assert result == expr

    def testMultipleAssumptions(self):
        # Multiple divisibility assumptions on different symbols.
        constraints = [
            Assumption(sympy.Eq(K % 32, 0)),
            Assumption(sympy.Eq(N % 16, 0)),
        ]
        fwd, bwd = _get_divisibility_subs(constraints)
        assert _simplify_expr(sympy.Mod(K, 32), fwd, bwd) == 0
        assert _simplify_expr(sympy.Mod(N, 16), fwd, bwd) == 0
        assert _simplify_expr(sympy.floor(N / 16), fwd, bwd) == N / 16

    def testMixedExpression(self):
        # floor(K/32) + Mod(K, 32) -> K/32 when 32 | K.
        expr = sympy.floor(K / 32) + sympy.Mod(K, 32)
        assert self._simplify(expr) == K / 32


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
