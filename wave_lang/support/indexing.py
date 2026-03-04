# Copyright 2025, The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, TypeAlias

import sympy  # type: ignore

__all__ = [
    "sym",
    "IndexExpr",
    "IndexSequence",
    "IndexSymbol",
    "index_symbol",
    "index_expr",
    "piecewise_aware_subs",
    "MMA_ACC_SYMBOL_NAME",
    "THREAD_SYMBOL_NAMES",
    "WORKGROUP_SYMBOL_NAMES",
    "DEVICE_SYMBOL_NAMES",
    "GPR_SYMBOL_NAME",
]

MMA_ACC_SYMBOL_NAME = "$MMA_ACC"
THREAD_SYMBOL_NAMES = ("$T0", "$T1", "$T2")
WORKGROUP_SYMBOL_NAMES = ("$WG0", "$WG1", "$WG2")
DEVICE_SYMBOL_NAMES = ("$DD0", "$DD1", "$DD2")
GPR_SYMBOL_NAME = "$GPR_NUM"

###############################################################################
# Index symbols and expressions
# These are just light-weight helpers around sympy symbols and expressions.
###############################################################################

IndexSymbol: TypeAlias = sympy.Symbol
IndexExpr: TypeAlias = sympy.Expr


def index_symbol(name: str) -> IndexSymbol:
    """Returns a named symbol, assumed to be a non-negative integer."""
    return sympy.Symbol(name, integer=True, nonnegative=True)


def index_expr(value: Any) -> IndexExpr:
    expr = sympy.sympify(value)
    return expr


class _IndexSymbolExpando:
    def __getattr__(self, n) -> IndexSymbol:
        return index_symbol(n)


sym = _IndexSymbolExpando()


def _subs_piecewise_walk(expr, subs_dict):
    if isinstance(expr, sympy.Piecewise):
        return sympy.Piecewise(
            *[(e.subs(subs_dict), c.subs(subs_dict)) for e, c in expr.args]
        )

    if not isinstance(expr, sympy.Basic) or expr.is_Atom:
        if isinstance(expr, sympy.Symbol) and expr in subs_dict:
            return subs_dict[expr]
        return expr

    if not expr.has(sympy.Piecewise):
        return expr.subs(subs_dict)

    new_args = [_subs_piecewise_walk(arg, subs_dict) for arg in expr.args]
    return expr.func(*new_args)


def piecewise_aware_subs(
    expr: "IndexExpr",
    subs_dict,
    simultaneous: bool = False,
) -> "IndexExpr":
    """Substitute into expr, handling Piecewise nodes efficiently.

    Avoids sympy's expensive recursive boolean simplification in
    Piecewise._eval_subs by substituting into each (value, condition) pair of a
    Piecewise independently.

    For expressions without Piecewise, delegates to sympy's normal subs().

    subs_dict can be a dict or a list of (old, new) pairs, matching sympy convention.
    """
    if not isinstance(expr, sympy.Basic):
        return expr

    if simultaneous:
        return expr.subs(subs_dict, simultaneous=True)

    if isinstance(subs_dict, (list, tuple)):
        subs_dict = dict(subs_dict)

    expr_syms = expr.free_symbols
    subs_keys = set(subs_dict.keys())
    matching = expr_syms & subs_keys
    if not matching:
        return expr

    filtered = {k: v for k, v in subs_dict.items() if k in matching}

    if isinstance(expr, sympy.Piecewise):
        return sympy.Piecewise(
            *[(e.subs(filtered), c.subs(filtered)) for e, c in expr.args]
        )
    elif expr.has(sympy.Piecewise):
        return _subs_piecewise_walk(expr, filtered)
    else:
        return expr.subs(filtered)


@dataclass
class IndexSequence:
    start: IndexExpr | int
    size: IndexExpr | int
    stride: IndexExpr | int = 1

    @staticmethod
    def _subs(
        value: int | IndexExpr,
        map: dict[IndexExpr, IndexExpr],
        simultaneous: bool = False,
    ) -> int | IndexExpr:
        if isinstance(value, IndexSequence):
            return value.subs(map, simultaneous=simultaneous)
        if isinstance(value, sympy.Basic):
            return piecewise_aware_subs(value, map, simultaneous=simultaneous)
        return value

    def has(self, symbol: IndexSymbol) -> bool:
        return (
            sympy.sympify(self.start).has(symbol)
            or sympy.sympify(self.size).has(symbol)
            or sympy.sympify(self.stride).has(symbol)
        )

    def subs(self, map: dict[IndexExpr, IndexExpr], simultaneous: bool = False):
        start = self._subs(self.start, map, simultaneous)
        size = self._subs(self.size, map, simultaneous)
        stride = self._subs(self.stride, map, simultaneous)
        return IndexSequence(start, size, stride)

    @staticmethod
    def from_expr(expr: IndexExpr, subs: dict[IndexExpr, Any]):
        start_subs = {k: v.start for k, v in subs.items()}
        size_subs = {k: v.size for k, v in subs.items()}
        stride_subs = {k: v.stride for k, v in subs.items()}
        start = IndexSequence._subs(expr, start_subs)
        size = IndexSequence._subs(expr, size_subs)
        stride = IndexSequence._subs(expr, stride_subs)
        return IndexSequence(start, size, stride)

    def __repr__(self) -> str:
        return f"{self.start} : {self.size} : {self.stride}"

    def __hash__(self):
        return hash((self.start, self.size, self.stride))
