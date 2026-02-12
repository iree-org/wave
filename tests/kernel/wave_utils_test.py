# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch.fx as fx
from wave_lang.kernel.lang import sym
from wave_lang.kernel.wave.utils.general_utils import (
    delinearize_index,
    divide_shape_into_chunks,
)
from wave_lang.kernel.wave.utils.mapping_utils import (
    _simplify_sympy_expr,
)
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.utils.graph_utils import assert_traces_equivalent
from wave_lang.kernel.ops.wave_ops import Allocate, MMA, get_custom
from wave_lang.kernel.wave.utils.symbol_utils import (
    collect_allowed_induction_symbols,
    get_induction_symbol,
    strip_out_of_scope_induction_symbols,
)
from wave_lang.support.indexing import IndexSequence
import sympy
import numpy as np

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.wave_types import Memory
from wave_lang.kernel.lang import sym
from wave_lang.kernel.wave.utils.graph_utils import (
    _check_result_types_equivalent,
    _check_payloads_equivalent,
    _check_expr_equivalent,
    _check_index_mapping_equivalent,
)
from wave_lang.support.indexing import index_symbol


M = sym.M


def test_delinearize_index():
    shape = [5, 4, 3]
    nd_index = delinearize_index(M, shape)
    np_nd_index = np.unravel_index(23, shape)
    assert np.equal([x.subs({M: 23}) for x in nd_index], np_nd_index).all()


def test_divide_shape_into_chunks():
    # Test case 1: All chunks fit in first dimension
    chunks_per_dim, chunk_shape = divide_shape_into_chunks([128, 256], 8)
    assert chunks_per_dim == [8, 1]
    assert chunk_shape == [16, 256]

    # Test case 2: Single chunk (no division)
    chunks_per_dim, chunk_shape = divide_shape_into_chunks([128, 256], 1)
    assert chunks_per_dim == [1, 1]
    assert chunk_shape == [128, 256]

    # Test case 3: Division only in first dimension (larger num_chunks)
    chunks_per_dim, chunk_shape = divide_shape_into_chunks([128, 256], 16)
    assert chunks_per_dim == [16, 1]
    assert chunk_shape == [8, 256]

    # Test case 4: Division only in first dimension (3D)
    chunks_per_dim, chunk_shape = divide_shape_into_chunks([64, 128, 256], 8)
    assert chunks_per_dim == [8, 1, 1]
    assert chunk_shape == [8, 128, 256]

    # Test case 5: Division across TWO dimensions
    chunks_per_dim, chunk_shape = divide_shape_into_chunks([6, 10], 15)
    assert chunks_per_dim == [3, 5]
    assert chunk_shape == [2, 2]

    # Test case 6: Division across TWO dimensions (different factorization)
    chunks_per_dim, chunk_shape = divide_shape_into_chunks([4, 8, 16], 8)
    assert chunks_per_dim == [4, 2, 1]
    assert chunk_shape == [1, 4, 16]

    # Test case 7: Division across THREE dimensions
    chunks_per_dim, chunk_shape = divide_shape_into_chunks([6, 4, 2], 48)
    assert chunks_per_dim == [6, 4, 2]
    assert chunk_shape == [1, 1, 1]

    # Test case 8: Prime number of chunks
    chunks_per_dim, chunk_shape = divide_shape_into_chunks([35, 100], 5)
    assert chunks_per_dim == [5, 1]
    assert chunk_shape == [7, 100]

    # Test case 9: Error case - cannot evenly divide
    with pytest.raises(ValueError, match="Cannot evenly divide"):
        divide_shape_into_chunks([128, 256], 12)

    # Test case 10: Error case - dimension too small
    with pytest.raises(ValueError, match="Cannot evenly divide"):
        divide_shape_into_chunks([10, 20], 7)

    # Test case 11: Error case - invalid num_chunks
    with pytest.raises(ValueError, match="num_chunks must be positive"):
        divide_shape_into_chunks([128, 256], 0)

    # Test case 12: Error case - empty shape
    with pytest.raises(ValueError, match="shape must be non-empty"):
        divide_shape_into_chunks([], 4)


def test_custom_sympy_simplifications():
    a = sympy.Symbol("a", integer=True, nonnegative=True)
    mod_expr = (sympy.floor(a) * 4 + 3) % 16
    assert str(_simplify_sympy_expr(mod_expr)) == "4*(Mod(a, 4)) + 3"

    floor_expr = sympy.floor(sympy.floor(a) / 3 + sympy.sympify(1) / 6)
    assert str(_simplify_sympy_expr(floor_expr)) == "floor(a/3)"


@pytest.mark.skip("Too slow")
def test_fuzz_custom_sympy_simplifications_mod():
    x = sympy.Symbol("x", integer=True, nonnegative=True)
    a = sympy.Symbol("a")
    b = sympy.Symbol("b")
    c = sympy.Symbol("c")

    import random

    expr = (sympy.floor(x) * a + b) % c
    total = 0
    outer_num_iters = 1000
    for i in range(outer_num_iters):

        a_val = random.randint(2, 50)
        b_val = random.randint(1, a_val - 1)
        c_val = a_val * random.randint(1, 10)

        vals = [a_val, b_val, c_val]
        expr = expr.subs({a: vals[0], b: vals[1], c: vals[2]})
        expr = sympy.simplify(expr)

        expr2 = _simplify_sympy_expr(expr)

        if i % 50 == 0 and i > 0:
            print(f"{100*i/outer_num_iters}%")

        if expr == expr2:
            print("skip", vals)
            continue

        vals2 = vals + [0, 1]
        for j in range(100):
            val = vals2[j] if j < len(vals2) else random.randint(0, c_val * 2)
            if expr.subs({x: val}) != expr2.subs({x: val}):
                print(f"Failed: {vals}, {val}")

            assert expr.subs({x: val}) == expr2.subs({x: val})
            total += 1

    print(f"Sucess: {total} checks")


def _trace_gemm_kernel():
    gemm, hyperparams, _ = get_gemm_kernel(
        shape=(128, 128, 32),
        dynamic_dims=False,
        mfma_variant=MMAType.F32_16x16x16_F16,
    )
    options = WaveCompileOptions(subs=hyperparams, compile_to_mlir=True)
    compiled_kernel = wave_compile(options, gemm)
    trace = compiled_kernel.get_compiled_graph()
    # assert_traces_equivalent only strips the LHS (reference) Allocate
    # indices because the RHS is expected to come from MLIR import.
    # When comparing two Python traces (as these unit tests do), both
    # sides carry out-of-scope induction symbols from backward index
    # propagation, so we clean them up here.
    for node in trace.walk():
        if isinstance(get_custom(node), Allocate):
            index = getattr(node, "index", None)
            if isinstance(index, dict):
                allowed = collect_allowed_induction_symbols(node)
                node.index = strip_out_of_scope_induction_symbols(index, allowed)
    return trace, options


def _get_first_mma_node(trace):
    """Helper to find first MMA node in trace."""
    for node in trace.walk():
        if isinstance(get_custom(node), MMA):
            return node
    raise AssertionError("No MMA node found in trace")


def test_traced_equivalent_to_self():
    """Test that a trace is equivalent to itself."""
    trace, options = _trace_gemm_kernel()
    assert_traces_equivalent(trace, trace, subs=options.subs)


def test_traced_noop_index_equivalent():
    """Test that semantically equivalent index rewrites (e.g., *1, +0) are detected as equivalent."""
    trace_a, options = _trace_gemm_kernel()
    trace_b, _ = _trace_gemm_kernel()

    # Apply no-op transformations to indices: multiply by 1, add 0
    one, zero = sympy.Integer(1), sympy.Integer(0)
    for node in trace_b.walk():
        index = getattr(node, "index", None)
        if isinstance(index, dict) and index:
            node.index = {
                dim: IndexSequence(seq.start * one, seq.size + zero, seq.stride * one)
                for dim, seq in index.items()
            }

    assert_traces_equivalent(trace_a, trace_b, subs=options.subs)


def test_traced_index_change_detected():
    """Test that an index modification is detected."""
    trace_a, options = _trace_gemm_kernel()
    trace_b, _ = _trace_gemm_kernel()

    for node in trace_b.walk():
        index = getattr(node, "index", None)
        if not index or not isinstance(index, dict):
            continue
        # Modify the first index we find
        new_index = {
            dim: IndexSequence(seq.start, seq.size + 1, seq.stride)
            for dim, seq in index.items()
        }
        node.index = new_index
        break

    with pytest.raises(AssertionError, match="IndexSequence.size mismatch"):
        assert_traces_equivalent(trace_a, trace_b, subs=options.subs)


def test_traced_op_removed_detected():
    trace_a, options = _trace_gemm_kernel()
    trace_b, _ = _trace_gemm_kernel()

    mma_node = _get_first_mma_node(trace_b)
    args = list(mma_node.args)
    lhs, rhs, acc = args[0], args[1], args[2]
    lhs.replace_all_uses_with(rhs)
    mma_node.graph.erase_node(lhs)

    with pytest.raises(AssertionError, match="node count mismatch"):
        assert_traces_equivalent(trace_a, trace_b, subs=options.subs)


def test_traced_attr_removed_detected():
    trace_a, options = _trace_gemm_kernel()
    trace_b, _ = _trace_gemm_kernel()

    mma_node = _get_first_mma_node(trace_b)
    assert hasattr(mma_node, "index"), "MMA node missing index attribute"
    delattr(mma_node, "index")

    with pytest.raises(AssertionError, match="index lost"):
        assert_traces_equivalent(trace_a, trace_b, subs=options.subs)


def test_directional_lhs_missing_attrs_ok():
    """Directional comparison: lhs (reference) missing semantic attrs is OK.

    The comparison is directional -- the reference trace may come from an
    earlier pipeline stage where vector_shapes or indices have not been
    populated yet. This should not fail even if the rhs has them.
    """
    trace_a, options = _trace_gemm_kernel()
    trace_b, _ = _trace_gemm_kernel()

    # Strip vector_shapes from every node in the reference (lhs).
    for node in trace_a.walk():
        if hasattr(node, "vector_shapes"):
            delattr(node, "vector_shapes")

    # lhs missing, rhs present → should still pass.
    assert_traces_equivalent(trace_a, trace_b, subs=options.subs)


def test_directional_rhs_missing_attrs_detected():
    """Directional comparison: rhs (actual) missing semantic attrs is detected.

    If the reference has vector_shapes but the actual lost them, this is an
    error -- information was lost during the roundtrip.
    """
    trace_a, options = _trace_gemm_kernel()
    trace_b, _ = _trace_gemm_kernel()

    # Strip vector_shapes from every node in the actual (rhs).
    for node in trace_b.walk():
        if hasattr(node, "vector_shapes"):
            delattr(node, "vector_shapes")

    with pytest.raises(AssertionError, match="attr 'vector_shapes' lost"):
        assert_traces_equivalent(trace_a, trace_b, subs=options.subs)


def _get_first_allocate_node(trace):
    """Helper to find first Allocate node in trace."""
    for node in trace.walk():
        if isinstance(get_custom(node), Allocate):
            return node
    raise AssertionError("No Allocate node found in trace")


def test_allocate_out_of_scope_induction_symbols_stripped():
    """Allocate index with out-of-scope induction symbols still matches.

    Backward index propagation (set_derived_index) can place $ARG-prefixed
    induction symbols on Allocate nodes that live outside the corresponding
    Iterate loop.  The comparison strips these before checking.
    """
    trace_a, options = _trace_gemm_kernel()
    trace_b, _ = _trace_gemm_kernel()

    alloc_a = _get_first_allocate_node(trace_a)
    alloc_b = _get_first_allocate_node(trace_b)

    # Give both Allocate nodes a clean index, so only one will be bugged.
    clean_index = {sym.M: IndexSequence(0, 16, 1), sym.N: IndexSequence(0, 16, 1)}
    alloc_a.index = {
        k: IndexSequence(v.start, v.size, v.stride) for k, v in clean_index.items()
    }
    alloc_b.index = {
        k: IndexSequence(v.start, v.size, v.stride) for k, v in clean_index.items()
    }

    # Inject a bogus out-of-scope induction symbol into the LHS (reference).
    # The Allocate lives in the root graph (outside any Iterate), so $ARGK
    # is out of scope and should be stripped before comparison.
    # assert_traces_equivalent strips the LHS Allocate indices (since the
    # source trace carries these from backward index propagation), so the
    # injected symbol should be removed and the comparison should pass.
    bogus_sym = get_induction_symbol(sym.K)
    seq = alloc_a.index[sym.M]
    alloc_a.index[sym.M] = IndexSequence(
        seq.start + bogus_sym * sympy.Symbol("BLOCK_K"), seq.size, seq.stride
    )

    # Should still pass because the comparison strips out-of-scope $ARG symbols
    # from the LHS.
    assert_traces_equivalent(trace_a, trace_b, subs=options.subs)


@pytest.mark.parametrize(
    "shape1,shape2,dtype1,dtype2,expected_error",
    [
        ((sym.M, sym.N), (sym.M, sym.M), tkl.f16, tkl.f16, "symbolic_shape"),
        ((sym.M, sym.M), (sym.M, sym.M), tkl.f16, tkl.f32, "dtype"),
        ((sym.M, sym.M), (sym.M, sym.M), tkl.f16, tkl.f16, None),  # same - should pass
    ],
)
def test_result_type_comparisons(shape1, shape2, dtype1, dtype2, expected_error):
    """Test Memory type comparison detects shape and dtype differences."""
    mem1 = Memory[shape1, tkl.AddressSpace.GLOBAL_MEMORY, dtype1]
    mem2 = Memory[shape2, tkl.AddressSpace.GLOBAL_MEMORY, dtype2]
    result = _check_result_types_equivalent(mem1, mem2)

    if expected_error:
        assert not result.success and expected_error in result.reason
    else:
        assert result.success


@pytest.mark.parametrize(
    "lhs,rhs,expected_error",
    [
        (tkl.f16, tkl.f16, None),
        (tkl.f16, tkl.f32, "dtype mismatch"),
        ([1, 2, 3], [1, 2], "sequence length mismatch"),
        ([1, 2, 3], [1, 5, 3], "sequence mismatch at 1"),
        ({"key": 10}, {"key": 20}, "dict value mismatch"),
    ],
)
def test_payload_comparisons(lhs, rhs, expected_error):
    """Test payload comparison for dtypes, sequences, and dicts."""
    result = _check_payloads_equivalent(lhs, rhs, None, {})
    assert (
        result.success
        if not expected_error
        else not result.success and expected_error in result.reason
    )


@pytest.mark.parametrize(
    "seq1,seq2,expected_field",
    [
        (IndexSequence(0, 10, 1), IndexSequence(5, 10, 1), "start"),
        (IndexSequence(0, 10, 1), IndexSequence(0, 20, 1), "size"),
        (IndexSequence(0, 10, 1), IndexSequence(0, 10, 2), "stride"),
    ],
)
def test_index_sequence_field_mismatches(seq1, seq2, expected_field):
    """Test that IndexSequence field differences are detected."""
    result = _check_expr_equivalent(seq1, seq2, None)
    assert not result.success
    assert f"IndexSequence.{expected_field} mismatch" in result.reason


@pytest.mark.parametrize(
    "expr1,expr2,expected_error",
    [
        (sympy.Symbol("x") + 1, sympy.Symbol("x") + 1 + sympy.Integer(0), None),
        (sympy.Symbol("x") + 1, sympy.Symbol("x") + 2, "expr mismatch"),
        (sympy.Symbol("x") + 1, sympy.Symbol("y") + 1, "symbol mismatch"),
    ],
)
def test_symbolic_expression_equivalence(expr1, expr2, expected_error):
    """Test symbolic expression comparison and simplification."""
    result = _check_expr_equivalent(expr1, expr2, None)
    assert (
        result.success
        if not expected_error
        else not result.success and expected_error in result.reason
    )


@pytest.mark.parametrize(
    "dict1,dict2,expected_error",
    [
        (
            {index_symbol("M"): IndexSequence(0, 10, 1)},
            {index_symbol("N"): IndexSequence(0, 10, 1)},
            "index keys mismatch",
        ),
        (
            {index_symbol("M"): IndexSequence(0, 10, 1)},
            {index_symbol("M"): IndexSequence(0, 20, 1)},
            "index expr mismatch",
        ),
    ],
)
def test_index_dict_mismatches(dict1, dict2, expected_error):
    """Test IndexDict comparison detects key and value differences."""
    result = _check_index_mapping_equivalent(dict1, dict2, None)
    assert not result.success
    assert expected_error in result.reason


@pytest.mark.skip("Too slow")
def test_fuzz_custom_sympy_simplifications_floor():
    x = sympy.Symbol("x", integer=True, nonnegative=True)
    a = sympy.Symbol("a")
    b = sympy.Symbol("b")
    c = sympy.Symbol("c")
    d = sympy.Symbol("d")

    import random

    orig_expr = sympy.floor(sympy.floor(x) * a / b + c / d)

    def check_specific(*vals):
        expr1 = orig_expr.subs({a: vals[0], b: vals[1], c: vals[2], d: vals[3]})
        expr1 = sympy.simplify(expr1)

        expr2 = _simplify_sympy_expr(expr1)
        assert expr1.subs({x: vals[4]}) == expr2.subs({x: vals[4]})

    check_specific(10, 11, 6, 10, 6)
    check_specific(8, 5, 1, 5, 8)

    total = 0
    outer_num_iters = 500
    for i in range(outer_num_iters):
        while True:
            a_val = 1  # random.randint(1, 10)
            b_val = random.randint(1, 10)
            if b_val == a_val:
                b_val += 1

            c_val = random.randint(1, 10)
            d_val = random.randint(1, 10)
            if d_val == c_val:
                d_val += 1

            vals = [a_val, b_val, c_val, d_val]
            expr = orig_expr.subs({a: vals[0], b: vals[1], c: vals[2], d: vals[3]})
            expr = sympy.simplify(expr)

            expr2 = _simplify_sympy_expr(expr)
            if expr != expr2:
                break

        if i % 50 == 0 and i > 0:
            print(f"{100*i/outer_num_iters}%")

        vals2 = vals + [-1, 0, 1]
        for j in range(100):
            val = vals2[j] if j < len(vals2) else random.randint(0, c_val * 2)
            if expr.subs({x: val}) != expr2.subs({x: val}):
                print(f"Failed: {vals}, {val}")

            assert expr.subs({x: val}) == expr2.subs({x: val})
            total += 1

    print(f"Sucess: {total} checks")
