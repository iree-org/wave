# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Sequence

import sympy

from .._support.indexing import IndexExpr, subs_idxc


@dataclass
class Assumption:
    """
    Assumptions are sympy assumptions that can be used to
    make decisions during code generation. These can be
    statements such as bounds on sympy variables. For example,
    we can state that

    Assumption(M < 64)

    and then later make queries based on this assumption, such as

    evaluate(M > 70) -> False
    evaluate(M < 32) -> None (because we cannot say one way or the other)
    evaluate(M < 70) -> True

    """

    expr: IndexExpr


# Type alias for substitution lists used by divisibility reasoning.
SubstList = list[tuple[sympy.Symbol, sympy.Expr]]


def get_divisibility_subs(
    constraints: Sequence,
) -> tuple[SubstList, SubstList]:
    """Extract divisibility assumptions into forward/backward substitution lists.

    For each ``Assumption(Eq(Mod(S, d), 0))`` we introduce a fresh integer
    symbol ``S_div_d`` and build:
      forward:  S  -> d * S_div_d
      backward: S_div_d -> S / d
    Applying forward subs lets sympy resolve ``Mod(S, d)`` to 0 and
    ``floor(S / d)`` to ``S_div_d``, then backward subs restores the
    original symbols.
    """
    forward: SubstList = []
    backward: SubstList = []
    for c in constraints:
        if not isinstance(c, Assumption):
            continue
        expr = c.expr
        if not isinstance(expr, sympy.Eq):
            continue
        lhs, rhs = expr.args
        # Match Eq(Mod(S, d), 0).
        if rhs != 0 or not isinstance(lhs, sympy.Mod):
            continue
        sym, divisor = lhs.args
        if not sym.is_Symbol:
            continue
        # Resolve symbolic divisors (e.g. BLOCK_M -> 32) via IndexingContext.
        divisor = sympy.sympify(subs_idxc(divisor))
        if not divisor.is_Integer:
            continue
        # Bail if the symbol may be negative: floor/Mod semantics differ
        # between Euclidean and truncated division for negative operands.
        if not sym.is_nonnegative:
            continue
        div_sym = sympy.Symbol(
            f"_{sym.name}_div_{divisor}",
            integer=True,
            nonnegative=sym.is_nonnegative,
            positive=sym.is_positive,
        )
        forward.append((sym, divisor * div_sym))
        backward.append((div_sym, sym / divisor))
    return forward, backward
