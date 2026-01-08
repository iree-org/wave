# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sympy
import importlib.util
import os
import sys

# These are fine since there is no transitive dependency on IREE.
from wave_lang.support.indexing import (
    MMA_ACC_SYMBOL_NAME,
    IndexSequence,
    index_symbol,
    sym,
)
from wave_lang.support.detect_water import is_water_available

# XXX: This ugly hack adds the directory wave_lang/kernel/wave directory to
# import paths to we can import water_mlir without prefixing it and therefore
# bypass initializers of the wave_lang.kernel.wave package that may import IREE,
# which would clash with Water MLIR bindings. Get the path to `wave_lang/`
# since that does not import IREE, then manually concatenate `wave` to that path
# so avoid touching wave_lang/kernel/wave/__init__.py that will transitively
# import IREE.
# TODO: Remove this hack once either (1) the main wave package doesn't
# systematically import IREE until needed or (2) there's no dependency on IREE
# anymore.
__wave_lang_spec = importlib.util.find_spec("wave_lang")
assert __wave_lang_spec is not None and __wave_lang_spec.origin is not None
__wave_lang_wave_path = os.path.join(
    os.path.dirname(__wave_lang_spec.origin), "kernel", "wave"
)
if __wave_lang_wave_path not in sys.path:
    sys.path.append(__wave_lang_wave_path)


# Only import water_mlir components if water_mlir is available, skip testing otherwise.
if is_water_available():
    from water_mlir.water_mlir import ir
    from water_mlir.water_mlir.dialects import wave

    from mlir_converter.mlir_to_wave import (
        _convert_affine_expr_to_sympy_expr,
        _convert_index_mapping_attr_to_sympy,
        _convert_index_mapping_dict_to_sympy,
        convert_index_mapping_array_to_sympy,
        _make_piecewise_sequence,
        ITER_SYMBOL_NAME_WAVE_PREFIX,
    )

pytestmark = pytest.mark.skipif(
    not is_water_available(), reason="water_mlir not available"
)


@pytest.fixture(autouse=True)
def ir_context():
    """Fixture to create and manage IR context for all tests."""
    if is_water_available():
        with ir.Context() as ctx:
            wave.register_dialect(ctx)
            yield ctx
    else:
        yield None


class TestConvertAffineExprToSympyExpr:
    """Tests for _convert_affine_expr_to_sympy_expr function."""

    def test_constant_expr(self):
        """Test conversion of constant affine expressions."""
        expr = ir.AffineConstantExpr.get(42)
        result = _convert_affine_expr_to_sympy_expr(expr, [])
        assert result == sympy.Integer(42)

    def test_symbol_expr(self):
        """Test conversion of symbol affine expressions."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        z = sympy.Symbol("z")
        symbol_mapping = [x, y, z]

        # Test s0
        expr = ir.AffineSymbolExpr.get(0)
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == x

        # Test s1
        expr = ir.AffineSymbolExpr.get(1)
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == y

        # Test s2
        expr = ir.AffineSymbolExpr.get(2)
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == z

    def test_add_expr(self):
        """Test conversion of addition affine expressions."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        symbol_mapping = [x, y]

        # Test s0 + s1
        s0 = ir.AffineSymbolExpr.get(0)
        s1 = ir.AffineSymbolExpr.get(1)
        expr = s0 + s1
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == x + y

        # Test s0 + 5
        expr = s0 + 5
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == x + 5

    def test_mul_expr(self):
        """Test conversion of multiplication affine expressions."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        symbol_mapping = [x, y]

        # Test s0 * 3
        s0 = ir.AffineSymbolExpr.get(0)
        expr = s0 * 3
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == 3 * x

        # Test s0 * s1
        s1 = ir.AffineSymbolExpr.get(1)
        expr = s0 * s1
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == x * y

    def test_floor_div_expr(self):
        """Test conversion of floor division affine expressions."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        symbol_mapping = [x, y]

        s0 = ir.AffineSymbolExpr.get(0)
        s1 = ir.AffineSymbolExpr.get(1)

        # Test s0 floordiv 2
        expr = ir.AffineExpr.get_floor_div(s0, ir.AffineConstantExpr.get(2))
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == sympy.floor(x / 2)

        # Test s0 floordiv s1
        expr = ir.AffineExpr.get_floor_div(s0, s1)
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == sympy.floor(x / y)

    def test_ceil_div_expr(self):
        """Test conversion of ceiling division affine expressions."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        symbol_mapping = [x, y]

        s0 = ir.AffineSymbolExpr.get(0)
        s1 = ir.AffineSymbolExpr.get(1)

        # Test s0 ceildiv 4
        expr = ir.AffineExpr.get_ceil_div(s0, ir.AffineConstantExpr.get(4))
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == sympy.ceiling(x / 4)

        # Test s0 ceildiv s1
        expr = ir.AffineExpr.get_ceil_div(s0, s1)
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == sympy.ceiling(x / y)

    def test_mod_expr(self):
        """Test conversion of modulo affine expressions."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        symbol_mapping = [x, y]

        s0 = ir.AffineSymbolExpr.get(0)
        s1 = ir.AffineSymbolExpr.get(1)

        # Test s0 mod s1
        expr = s0 % s1
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == x % y

        # Test s0 mod 8
        expr = s0 % 8
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == x % 8

    def test_complex_expr(self):
        """Test conversion of complex nested affine expressions."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        z = sympy.Symbol("z")
        symbol_mapping = [x, y, z]

        s0 = ir.AffineSymbolExpr.get(0)
        s1 = ir.AffineSymbolExpr.get(1)
        s2 = ir.AffineSymbolExpr.get(2)

        # Test (s0 * 2 + s1) floordiv s2
        expr = ir.AffineExpr.get_floor_div(s0 * 2 + s1, s2)
        result = _convert_affine_expr_to_sympy_expr(expr, symbol_mapping)
        assert result == sympy.floor((2 * x + y) / z)

    def test_unsupported_expr_raises_error(self):
        """Test that unsupported expression types raise ValueError."""
        # Create a dimension expression (not supported by the function)
        expr = ir.AffineDimExpr.get(0)

        with pytest.raises(ValueError, match="Unsupported affine expression"):
            _convert_affine_expr_to_sympy_expr(expr, [])


class TestConvertIndexMappingAttrToSympy:
    """Tests for _convert_index_mapping_attr_to_sympy function."""

    def test_basic_index_mapping_with_symbol_attr(self):
        """Test conversion of basic index mapping with WaveSymbolAttr."""
        # Create symbols
        symbols = [
            wave.WaveSymbolAttr.get("M"),
            wave.WaveSymbolAttr.get("N"),
        ]

        # Create affine maps
        s0 = ir.AffineSymbolExpr.get(0)
        s1 = ir.AffineSymbolExpr.get(1)
        start_map = ir.AffineMap.get(0, 2, [s0])
        step_map = ir.AffineMap.get(0, 2, [s1])
        stride_map = ir.AffineMap.get(0, 2, [ir.AffineConstantExpr.get(1)])

        # Create index mapping attribute
        attr = wave.WaveIndexMappingAttr.get(symbols, start_map, step_map, stride_map)

        # Convert to sympy
        result = _convert_index_mapping_attr_to_sympy(attr)

        # Check the result
        assert isinstance(result, IndexSequence)
        assert result.start == index_symbol("M")
        assert result.size == index_symbol("N")
        assert result.stride == 1

    def test_index_mapping_with_index_symbol_attr(self):
        """Test conversion with WaveIndexSymbolAttr (special symbols like $WG0)."""
        # Create symbols including WaveIndexSymbolAttr
        symbols = [
            wave.WaveIndexSymbolAttr.get(wave.WaveIndexSymbol.WORKGROUP_0),
            wave.WaveSymbolAttr.get("BLOCK_M"),
        ]

        s0 = ir.AffineSymbolExpr.get(0)
        s1 = ir.AffineSymbolExpr.get(1)
        start_map = ir.AffineMap.get(0, 2, [s0 * 3])
        step_map = ir.AffineMap.get(0, 2, [s1])
        stride_map = ir.AffineMap.get(0, 2, [s0 + s1])

        attr = wave.WaveIndexMappingAttr.get(symbols, start_map, step_map, stride_map)
        result = _convert_index_mapping_attr_to_sympy(attr)

        assert isinstance(result, IndexSequence)
        # $WG0 should be converted to index_symbol("$WG0")
        assert result.start == index_symbol("$WG0") * 3
        assert result.size == index_symbol("BLOCK_M")
        assert result.stride == index_symbol("$WG0") + index_symbol("BLOCK_M")

    def test_index_mapping_with_iter_symbol_attr(self):
        """Test conversion with WaveIterSymbolAttr (iteration symbols)."""
        # Create symbols including WaveIterSymbolAttr
        symbols = [
            wave.WaveIterSymbolAttr.get("i"),
        ]

        s0 = ir.AffineSymbolExpr.get(0)
        start_map = ir.AffineMap.get(0, 1, [s0 * 2])
        step_map = ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(16)])
        stride_map = ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(1)])

        attr = wave.WaveIndexMappingAttr.get(symbols, start_map, step_map, stride_map)
        result = _convert_index_mapping_attr_to_sympy(attr)

        assert isinstance(result, IndexSequence)
        # Iter symbol should be prefixed with ITER_SYMBOL_NAME_WAVE_PREFIX ($ARG)
        assert result.start == index_symbol(ITER_SYMBOL_NAME_WAVE_PREFIX + "i") * 2
        assert result.size == 16
        assert result.stride == 1

    def test_index_mapping_with_null_start(self):
        attr = ir.Attribute.parse("#wave<index_mapping[] -> (<NULL>, 1, 1)>")
        result = _convert_index_mapping_attr_to_sympy(attr)
        assert isinstance(result, IndexSequence)
        assert result.start is None
        assert result.size == 1
        assert result.stride == 1

    def test_index_mapping_with_null_step_stride(self):
        attr = ir.Attribute.parse("#wave<index_mapping[] -> (1, <NULL>, <NULL>)>")
        result = _convert_index_mapping_attr_to_sympy(attr)
        assert isinstance(result, IndexSequence)
        assert result.start == 1
        assert result.size is None
        assert result.stride is None


class TestConvertIndexMappingDictToSympy:
    """Tests for _convert_index_mapping_dict_to_sympy function."""

    def test_single_mapping(self):
        """Test conversion of dict with single index mapping."""
        # Create a simple index mapping
        symbols = [wave.WaveSymbolAttr.get("M")]
        s0 = ir.AffineSymbolExpr.get(0)
        start_map = ir.AffineMap.get(0, 1, [s0])
        step_map = ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(16)])
        stride_map = ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(1)])
        mapping_attr = wave.WaveIndexMappingAttr.get(
            symbols, start_map, step_map, stride_map
        )

        # Create dict attribute
        dict_attr = ir.DictAttr.get({"dim0": mapping_attr})

        result = _convert_index_mapping_dict_to_sympy(dict_attr)

        assert isinstance(result, dict)
        assert index_symbol("dim0") in result
        assert isinstance(result[index_symbol("dim0")], IndexSequence)
        assert result[index_symbol("dim0")].start == index_symbol("M")
        assert result[index_symbol("dim0")].size == 16

    def test_multiple_mappings(self):
        """Test conversion of dict with multiple index mappings."""
        # Create first mapping
        symbols1 = [wave.WaveSymbolAttr.get("M")]
        s0 = ir.AffineSymbolExpr.get(0)
        mapping1 = wave.WaveIndexMappingAttr.get(
            symbols1,
            ir.AffineMap.get(0, 1, [s0]),
            ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(16)]),
            ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(1)]),
        )

        # Create second mapping
        symbols2 = [wave.WaveSymbolAttr.get("N")]
        mapping2 = wave.WaveIndexMappingAttr.get(
            symbols2,
            ir.AffineMap.get(0, 1, [s0 * 2]),
            ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(32)]),
            ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(2)]),
        )

        dict_attr = ir.DictAttr.get({"m": mapping1, "n": mapping2})
        result = _convert_index_mapping_dict_to_sympy(dict_attr)

        assert len(result) == 2
        assert index_symbol("m") in result
        assert index_symbol("n") in result
        assert result[index_symbol("m")].size == 16
        assert result[index_symbol("n")].size == 32


class TestMakePiecewiseSequence:
    """Tests for _make_piecewise_sequence function."""

    def test_two_component_piecewise(self):
        """Test piecewise sequence with two components."""
        # Create two index sequences
        seq1 = IndexSequence(start=0, size=10, stride=1)
        seq2 = IndexSequence(start=100, size=20, stride=2)

        # Create conditions
        cond1 = sympy.Symbol("x") < 5
        cond2 = sympy.Symbol("x") >= 5

        # Create piecewise sequence
        result = _make_piecewise_sequence((seq1, cond1), (seq2, cond2))

        assert isinstance(result, IndexSequence)
        assert isinstance(result.start, sympy.Piecewise)
        assert isinstance(result.size, sympy.Piecewise)
        assert isinstance(result.stride, sympy.Piecewise)

    def test_single_component_piecewise(self):
        """Test piecewise sequence with single component."""
        seq = IndexSequence(start=5, size=15, stride=3)
        cond = sympy.Symbol("flag")

        result = _make_piecewise_sequence((seq, cond))

        assert isinstance(result, IndexSequence)
        assert isinstance(result.start, sympy.Piecewise)
        assert isinstance(result.size, sympy.Piecewise)
        assert isinstance(result.stride, sympy.Piecewise)


class TestConvertIndexMappingArrayToSympy:
    """Tests for convert_index_mapping_array_to_sympy function."""

    def test_non_mma_op_single_mapping(self):
        """Test conversion for non-MMA operations (expects single mapping)."""
        # Create a simple mapping
        symbols = [wave.WaveSymbolAttr.get("M")]
        s0 = ir.AffineSymbolExpr.get(0)
        mapping = wave.WaveIndexMappingAttr.get(
            symbols,
            ir.AffineMap.get(0, 1, [s0]),
            ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(16)]),
            ir.AffineMap.get(0, 1, [ir.AffineConstantExpr.get(1)]),
        )

        dict_attr = ir.DictAttr.get({"dim": mapping})
        array_attr = ir.ArrayAttr.get([dict_attr])

        # We don't need anything from the operation except its name, so use an empty module.
        dummy_op = ir.Operation.create("builtin.module", loc=ir.Location.unknown())
        result = convert_index_mapping_array_to_sympy(dummy_op, array_attr)

        assert isinstance(result, dict)
        assert index_symbol("dim") in result

    def test_mma_op_with_valid_four_mappings(self):
        """Test MMA operation with correct 4 mappings creates piecewise sequence."""
        # Create symbols for M, N, K dimensions
        m_sym = wave.WaveSymbolAttr.get("M")
        n_sym = wave.WaveSymbolAttr.get("N")
        k_sym = wave.WaveSymbolAttr.get("K")

        s0 = ir.AffineSymbolExpr.get(0)
        c16 = ir.AffineConstantExpr.get(16)
        c1 = ir.AffineConstantExpr.get(1)

        # LHS mapping: M and K
        lhs_m_mapping = wave.WaveIndexMappingAttr.get(
            [m_sym],
            ir.AffineMap.get(0, 1, [s0]),
            ir.AffineMap.get(0, 1, [c16]),
            ir.AffineMap.get(0, 1, [c1]),
        )
        lhs_k_mapping = wave.WaveIndexMappingAttr.get(
            [k_sym],
            ir.AffineMap.get(0, 1, [s0]),
            ir.AffineMap.get(0, 1, [c16]),
            ir.AffineMap.get(0, 1, [c1]),
        )

        # RHS mapping: N and K (K must match LHS K)
        rhs_n_mapping = wave.WaveIndexMappingAttr.get(
            [n_sym],
            ir.AffineMap.get(0, 1, [s0]),
            ir.AffineMap.get(0, 1, [c16]),
            ir.AffineMap.get(0, 1, [c1]),
        )
        rhs_k_mapping = wave.WaveIndexMappingAttr.get(
            [k_sym],
            ir.AffineMap.get(0, 1, [s0]),
            ir.AffineMap.get(0, 1, [c16]),
            ir.AffineMap.get(0, 1, [c1]),
        )

        # Accumulator mapping: M and N
        acc_m_mapping = wave.WaveIndexMappingAttr.get(
            [m_sym],
            ir.AffineMap.get(0, 1, [c16 - s0]),
            ir.AffineMap.get(0, 1, [c1]),
            ir.AffineMap.get(0, 1, [c16]),
        )
        acc_n_mapping = wave.WaveIndexMappingAttr.get(
            [n_sym],
            ir.AffineMap.get(0, 1, [s0]),
            ir.AffineMap.get(0, 1, [c16]),
            ir.AffineMap.get(0, 1, [c1]),
        )

        # Note that result mapping is the same as the accumulator mapping.
        lhs_dict = ir.DictAttr.get({"M": lhs_m_mapping, "K": lhs_k_mapping})
        rhs_dict = ir.DictAttr.get({"N": rhs_n_mapping, "K": rhs_k_mapping})
        acc_dict = ir.DictAttr.get({"M": acc_m_mapping, "N": acc_n_mapping})
        result_dict = ir.DictAttr.get({"M": acc_m_mapping, "N": acc_n_mapping})

        array_attr = ir.ArrayAttr.get([lhs_dict, rhs_dict, acc_dict, result_dict])

        # Create a mock MMA operation, we only need the name, it doesn't even need to verify correctly.
        dummy_mma_op = ir.Operation.create("wave.mma", loc=ir.Location.unknown())
        result = convert_index_mapping_array_to_sympy(dummy_mma_op, array_attr)

        assert isinstance(result, dict)
        assert len(result) == 3  # M, N, K
        assert index_symbol("M") in result
        assert index_symbol("N") in result
        assert index_symbol("K") in result

        # M should be piecewise (combining LHS and ACC)
        m_seq = result[index_symbol("M")]
        assert isinstance(m_seq, IndexSequence)
        # The start/size/stride should be Piecewise expressions
        assert isinstance(m_seq.start, sympy.Piecewise)
        assert isinstance(m_seq.size, sympy.Piecewise)
        assert isinstance(m_seq.stride, sympy.Piecewise)

        assert m_seq.start == sympy.Piecewise(
            (sym.M, ~index_symbol(MMA_ACC_SYMBOL_NAME)),
            (sympy.sympify(16 - sym.M), index_symbol(MMA_ACC_SYMBOL_NAME)),
        )
        assert m_seq.size == sympy.Piecewise(
            (16, ~index_symbol(MMA_ACC_SYMBOL_NAME)),
            (1, index_symbol(MMA_ACC_SYMBOL_NAME)),
        )
        assert m_seq.stride == sympy.Piecewise(
            (1, ~index_symbol(MMA_ACC_SYMBOL_NAME)),
            (16, index_symbol(MMA_ACC_SYMBOL_NAME)),
        )
