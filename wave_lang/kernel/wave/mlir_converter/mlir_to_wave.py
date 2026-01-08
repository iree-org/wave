# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

if "iree" in sys.modules:
    raise ImportError(
        "Must not import this module when IREE is loaded. This leads to clashes between copies of MLIR bindings."
    )

import sympy  # type: ignore
from typing import Sequence
from water_mlir.water_mlir import ir
from water_mlir.water_mlir.dialects import wave

# This is fine since it doesn't depend on IREE transitively.
from wave_lang.support.indexing import (
    MMA_ACC_SYMBOL_NAME,
    IndexSequence,
    index_symbol,
    IndexSymbol,
)

assert (
    "iree" not in sys.modules
), "IREE was loaded transitively by modules. This should not happen."


ITER_SYMBOL_NAME_WAVE_PREFIX = "$ARG"
ITER_SYMBOL_NAME_WATER_PREFIX = "_Iter_"

# Mapping of special symbol names to WaveIndexSymbol enum values.
INDEX_SYMBOL_MAP: dict[str, wave.WaveIndexSymbol] = {
    "$WG0": wave.WaveIndexSymbol.WORKGROUP_0,
    "$WG1": wave.WaveIndexSymbol.WORKGROUP_1,
    "$WG2": wave.WaveIndexSymbol.WORKGROUP_2,
    "$T0": wave.WaveIndexSymbol.THREAD_0,
    "$T1": wave.WaveIndexSymbol.THREAD_1,
    "$T2": wave.WaveIndexSymbol.THREAD_2,
    "$DD0": wave.WaveIndexSymbol.DEVICE_DIM_0,
    "$DD1": wave.WaveIndexSymbol.DEVICE_DIM_1,
    "$DD2": wave.WaveIndexSymbol.DEVICE_DIM_2,
    "$GPR_NUM": wave.WaveIndexSymbol.GPR_NUMBER,
}
INDEX_SYMBOL_REVERSE_MAP: dict[wave.WaveIndexSymbol, str] = {
    value: key for key, value in INDEX_SYMBOL_MAP.items()
}


def _convert_affine_expr_to_sympy_expr(
    expr: ir.AffineExpr,
    symbol_mapping: Sequence[sympy.Symbol],
) -> sympy.Expr:
    """Convert an MLIR AffineExpr to a sympy expression.

    Args:
        expr: The MLIR AffineExpr to convert.
        symbol_mapping: A list of sympy symbols co-indexed with the positional
            affine symbols in the MLIR AffineExpr.

    Returns:
        The sympy expression corresponding to the MLIR AffineExpr.

    Raises:
        ValueError: If the expression is not supported.
    """
    if isinstance(expr, ir.AffineConstantExpr):
        return sympy.Integer(ir.AffineConstantExpr(expr).value)
    if isinstance(expr, ir.AffineSymbolExpr):
        return symbol_mapping[ir.AffineSymbolExpr(expr).position]
    if isinstance(expr, ir.AffineAddExpr):
        add_expr = ir.AffineAddExpr(expr)
        return _convert_affine_expr_to_sympy_expr(
            add_expr.lhs, symbol_mapping
        ) + _convert_affine_expr_to_sympy_expr(add_expr.rhs, symbol_mapping)
    if isinstance(expr, ir.AffineMulExpr):
        mul_expr = ir.AffineMulExpr(expr)
        return _convert_affine_expr_to_sympy_expr(
            mul_expr.lhs, symbol_mapping
        ) * _convert_affine_expr_to_sympy_expr(mul_expr.rhs, symbol_mapping)
    if isinstance(expr, ir.AffineFloorDivExpr):
        floor_div_expr = ir.AffineFloorDivExpr(expr)
        return sympy.floor(
            _convert_affine_expr_to_sympy_expr(floor_div_expr.lhs, symbol_mapping)
            / _convert_affine_expr_to_sympy_expr(floor_div_expr.rhs, symbol_mapping)
        )
    if isinstance(expr, ir.AffineCeilDivExpr):
        ceil_div_expr = ir.AffineCeilDivExpr(expr)
        return sympy.ceiling(
            _convert_affine_expr_to_sympy_expr(ceil_div_expr.lhs, symbol_mapping)
            / _convert_affine_expr_to_sympy_expr(ceil_div_expr.rhs, symbol_mapping)
        )
    if isinstance(expr, ir.AffineModExpr):
        mod_expr = ir.AffineModExpr(expr)
        return _convert_affine_expr_to_sympy_expr(
            mod_expr.lhs, symbol_mapping
        ) % _convert_affine_expr_to_sympy_expr(mod_expr.rhs, symbol_mapping)
    raise ValueError(f"Unsupported affine expression: {expr} of type {type(expr)}")


def _convert_index_mapping_attr_to_sympy(
    attr: wave.WaveIndexMappingAttr,
) -> IndexSequence:
    """Convert a WaveIndexMappingAttr to a Wave IndexSequence.

    Args:
        attr: The WaveIndexMappingAttr to convert.

    Returns:
        The Wave IndexSequence corresponding to the WaveIndexMappingAttr.

    Raises:
        ValueError: If any subexpression in the mapping is not supported.
    """

    def wrap_symbol(symbol_name: ir.Attribute) -> sympy.Symbol:
        if isinstance(symbol_name, wave.WaveSymbolAttr):
            return index_symbol(symbol_name.name)
        elif isinstance(symbol_name, wave.WaveIterSymbolAttr):
            return index_symbol(ITER_SYMBOL_NAME_WAVE_PREFIX + symbol_name.name)
        elif isinstance(symbol_name, wave.WaveIndexSymbolAttr):
            index_symbol_var = INDEX_SYMBOL_REVERSE_MAP.get(symbol_name.value, None)
            if index_symbol_var is None:
                raise ValueError(f"Unsupported index symbol: {symbol_name.value}")
            return index_symbol(index_symbol_var)
        else:
            raise ValueError(f"Unsupported symbol attribute: {symbol_name}")

    symbols = list(map(wrap_symbol, attr.symbols))
    assert (
        len(attr.start.results) == 1
    ), f"Expected start map to have one expression, got {attr.start}"
    assert (
        len(attr.step.results) == 1
    ), f"Expected step map to have one expression, got {attr.step}"
    assert (
        len(attr.stride.results) == 1
    ), f"Expected stride map to have one expression, got {attr.stride}"
    start = _convert_affine_expr_to_sympy_expr(attr.start.results[0], symbols)
    step = _convert_affine_expr_to_sympy_expr(attr.step.results[0], symbols)
    stride = _convert_affine_expr_to_sympy_expr(attr.stride.results[0], symbols)
    return IndexSequence(start, step, stride)


def _convert_index_mapping_dict_to_sympy(
    dict_attr: ir.DictAttr,
) -> dict[IndexSymbol, IndexSequence]:
    """Convert a dictionary attribute containing WaveIndexMappingAttr to a dictionary of Wave IndexSequences."""
    result = {}
    for named_attr in dict_attr:
        key = named_attr.name
        value = named_attr.attr
        assert isinstance(
            value, wave.WaveIndexMappingAttr
        ), f"Unsupported index mapping attribute: {value}"
        result[index_symbol(key)] = _convert_index_mapping_attr_to_sympy(value)
    return result


def _make_piecewise_sequence(
    *components: tuple[IndexSequence, sympy.Expr]
) -> IndexSequence:
    """Create a Piecewise IndexSequence from a list of components.

    Args:
        *components: A list of tuples: (subexpression, condition).

    Returns:
        The Piecewise IndexSequence corresponding to the list of components.
    """
    return IndexSequence(
        start=sympy.Piecewise(
            *[(component[0].start, component[1]) for component in components]
        ),
        size=sympy.Piecewise(
            *[(component[0].size, component[1]) for component in components]
        ),
        stride=sympy.Piecewise(
            *[(component[0].stride, component[1]) for component in components]
        ),
    )


def convert_index_mapping_array_to_sympy(
    op: ir.Operation, array_attr: ir.ArrayAttr
) -> dict[IndexSymbol, IndexSequence]:
    # TODO: for some reason, isinstance(op.opview, MmaOp) is not working. Something is off with dialect loading/registration.
    if op.name != "wave.mma":
        assert (
            len(array_attr) == 1
        ), f"Expected exactly one index mapping attribute for non-MMA op: {op}"
        return _convert_index_mapping_dict_to_sympy(array_attr[0])

    assert (
        len(array_attr) == 4
    ), f"Expected exactly four index mapping attributes for MMA op: {op}"
    lhs_index = _convert_index_mapping_dict_to_sympy(array_attr[0])
    rhs_index = _convert_index_mapping_dict_to_sympy(array_attr[1])
    acc_index = _convert_index_mapping_dict_to_sympy(array_attr[2])
    result_index = _convert_index_mapping_dict_to_sympy(array_attr[3])
    mk_symbols = set(lhs_index.keys())
    nk_symbols = set(rhs_index.keys())
    m_symbol = (mk_symbols - nk_symbols).pop()
    n_symbol = (nk_symbols - mk_symbols).pop()
    k_symbol = (mk_symbols.intersection(nk_symbols)).pop()
    assert lhs_index[k_symbol] == rhs_index[k_symbol]
    assert rhs_index[n_symbol] == acc_index[n_symbol]
    assert acc_index[m_symbol] == result_index[m_symbol]
    assert acc_index[n_symbol] == result_index[n_symbol]
    return {
        m_symbol: _make_piecewise_sequence(
            (lhs_index[m_symbol], ~index_symbol(MMA_ACC_SYMBOL_NAME)),
            (acc_index[m_symbol], index_symbol(MMA_ACC_SYMBOL_NAME)),
        ),
        n_symbol: rhs_index[n_symbol],
        k_symbol: lhs_index[k_symbol],
    }
