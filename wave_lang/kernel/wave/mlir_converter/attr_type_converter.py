# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Shared utilities for converting between Wave MLIR attributes and Python types.

Used by both the FX-to-MLIR emitter (water_emitter.py) and the MLIR-to-FX
converter (fx_emitter.py). Provides element-type mapping, symbol attribute
conversion, and AffineExpr-to-sympy translation.
"""

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
from wave_lang.kernel._support import dtype

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

_ELEMENT_TYPE_HANDLERS: list[tuple[type[ir.Type], dtype.DataType]] = [
    (ir.BF16Type, dtype.bf16),
    (ir.F16Type, dtype.f16),
    (ir.F32Type, dtype.f32),
    (ir.F64Type, dtype.f64),
    (ir.Float8E5M2Type, dtype.f8e5m2),
    (ir.Float8E5M2FNUZType, dtype.f8e5m2fnuz),
    (ir.Float8E4M3FNType, dtype.f8e4m3fn),
    (ir.Float8E4M3FNUZType, dtype.f8e4m3fnuz),
    (ir.Float8E8M0FNUType, dtype.f8e8m0fnu),
    (ir.Float6E2M3FNType, dtype.f6e2m3fn),
    (ir.Float4E2M1FNType, dtype.f4e2m1fn),
    (ir.IndexType, dtype.index),
]

_INTEGER_TYPE_WIDTH_TO_dtype: dict[int, dtype.DataType] = {
    1: dtype.i1,
    4: dtype.i4,
    8: dtype.i8,
    16: dtype.i16,
    32: dtype.i32,
    64: dtype.i64,
}


def mlir_element_type_to_dtype(element_type: ir.Type) -> dtype.DataType:
    """Map an MLIR element type to the corresponding Wave DataType."""
    for type_cls, dtype_value in _ELEMENT_TYPE_HANDLERS:
        if isinstance(element_type, type_cls):
            return dtype_value
    if isinstance(element_type, ir.IntegerType):
        width = int(element_type.width)
        if width in _INTEGER_TYPE_WIDTH_TO_dtype:
            return _INTEGER_TYPE_WIDTH_TO_dtype[width]
    raise ValueError(f"Unsupported element type: {element_type}")


def dtype_to_mlir_scalar_type(t: dtype.DataType) -> ir.Type:
    """Map a Wave DataType to the corresponding MLIR scalar type."""
    for type_cls, dtype_value in _ELEMENT_TYPE_HANDLERS:
        if dtype_value == t:
            return type_cls.get()
    for width, dtype_value in _INTEGER_TYPE_WIDTH_TO_dtype.items():
        if dtype_value == t:
            return ir.IntegerType.get_signless(width)
    raise RuntimeError(f"Unsupported scalar dtype: {t}")


def symbol_attr_to_name(symbol_attr: ir.Attribute) -> str:
    """Extract the Python-side symbol name from a Wave symbol attribute.

    Handles WaveSymbolAttr (plain names), WaveIterSymbolAttr (prefixed with
    ``$ARG``), and WaveIndexSymbolAttr (mapped via INDEX_SYMBOL_REVERSE_MAP).
    """
    if isinstance(symbol_attr, wave.WaveSymbolAttr):
        return symbol_attr.name
    if isinstance(symbol_attr, wave.WaveIterSymbolAttr):
        return ITER_SYMBOL_NAME_WAVE_PREFIX + symbol_attr.name
    if isinstance(symbol_attr, wave.WaveIndexSymbolAttr):
        name = INDEX_SYMBOL_REVERSE_MAP.get(symbol_attr.value, None)
        if name is None:
            raise ValueError(f"Unsupported index symbol: {symbol_attr.value}")
        return name
    raise ValueError(f"Unsupported symbol attribute: {symbol_attr}")


def symbol_attr_to_sympy(symbol_attr: ir.Attribute) -> sympy.Symbol:
    """Convert a Wave symbol attribute to a sympy Symbol."""
    return index_symbol(symbol_attr_to_name(symbol_attr))


def preprocess_symbols(
    symbols: Sequence[sympy.Symbol],
) -> dict[sympy.Symbol, sympy.Symbol]:
    """
    Preprocess symbols by:

      1. adding assumptions about all symbols being positive to later enable
         more simplifications.
      2. replacing ITER_SYMBOL_NAME_WAVE_PREFIX (`$ARG`) prefix of argument
         symbols (e.g. `ARG0`) by ITER_SYMBOL_NAME_WATER_PREFIX (`_Iter_`) to
         match dialect expectations.
      3. sorting symbols by name for deterministic symbol order as they may be
         coming from, e.g., a set.
    """
    result = {}
    for sym in sorted(symbols, key=lambda s: s.name):
        if sym.name.startswith(ITER_SYMBOL_NAME_WAVE_PREFIX):
            new_name = sym.name.replace(
                ITER_SYMBOL_NAME_WAVE_PREFIX, ITER_SYMBOL_NAME_WATER_PREFIX
            )
            result[sym] = sympy.Symbol(new_name, positive=True)
        else:
            result[sym] = sympy.Symbol(sym.name, positive=True)
    return result


def symbol_name_to_attribute(name: str) -> ir.Attribute:
    """
    Convert a symbol name to either a WaveSymbolAttr or WaveIndexSymbolAttr.

    Special symbols starting with $ are converted to WaveIndexSymbolAttr,
    while regular symbols are converted to WaveSymbolAttr.
    """
    if name in INDEX_SYMBOL_MAP:
        return wave.WaveIndexSymbolAttr.get(INDEX_SYMBOL_MAP[name])
    if name.startswith(ITER_SYMBOL_NAME_WATER_PREFIX):
        return wave.WaveIterSymbolAttr.get(
            name.replace(ITER_SYMBOL_NAME_WATER_PREFIX, "")
        )
    return wave.WaveSymbolAttr.get(name)


def expr_list_attr_to_exprs(
    attr: wave.WaveExprListAttr,
) -> list[sympy.Expr | int]:
    """Convert a WaveExprListAttr to a list of sympy expressions."""
    symbols = [symbol_attr_to_sympy(s) for s in list(attr.symbols)]
    return [
        _convert_affine_expr_to_sympy_expr(expr, symbols) for expr in attr.map.results
    ]


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

    symbols = [symbol_attr_to_sympy(s) for s in attr.symbols]

    def convert_map(map: ir.AffineMap | None) -> sympy.Expr | None:
        if map is None:
            return None
        assert len(map.results) == 1, f"Expected map to have one expression, got {map}"
        return _convert_affine_expr_to_sympy_expr(map.results[0], symbols)

    start = convert_map(attr.start)
    step = convert_map(attr.step)
    stride = convert_map(attr.stride)
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
