# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import unittest
from wave_lang.kernel.lang import sym
from wave_lang.kernel.wave.assumptions import Assumption
from wave_lang.kernel.wave.utils.general_utils import evaluate_with_assumptions
from wave_lang.kernel._support.indexing import IndexingContext

M = sym.M
N = sym.N
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
