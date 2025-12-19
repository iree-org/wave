# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from wave_lang.kernel._support.indexing import IndexSequence, index_symbol
from wave_lang.kernel.wave.analysis.index_sequence_analysis import (
    _check_index_difference_is_zero,
)


def test_equal_index_sequences_returns_true():
    """Test that equal index sequences return True."""
    dim1 = index_symbol("dim1")
    dim2 = index_symbol("dim2")

    index1 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
        dim2: IndexSequence(start=5, size=20, stride=2),
    }

    index2 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
        dim2: IndexSequence(start=5, size=20, stride=2),
    }

    result = _check_index_difference_is_zero(index1, index2)
    assert result is True


def test_equal_symbolic_index_sequences_returns_true():
    """Test that symbolically equal index sequences return True."""
    dim1 = index_symbol("dim1")
    M = index_symbol("M")
    N = index_symbol("N")

    index1 = {
        dim1: IndexSequence(start=M, size=N, stride=M + 1),
    }

    index2 = {
        dim1: IndexSequence(start=M, size=N, stride=M + 1),
    }

    result = _check_index_difference_is_zero(index1, index2)
    assert result is True


def test_different_start_raises_value_error():
    """Test that different start values raise ValueError."""
    dim1 = index_symbol("dim1")

    index1 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
    }

    index2 = {
        dim1: IndexSequence(start=5, size=10, stride=1),
    }

    with pytest.raises(ValueError, match="Start difference"):
        _check_index_difference_is_zero(index1, index2)


def test_different_size_raises_value_error():
    """Test that different size values raise ValueError."""
    dim1 = index_symbol("dim1")

    index1 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
    }

    index2 = {
        dim1: IndexSequence(start=0, size=20, stride=1),
    }

    with pytest.raises(ValueError, match="Size difference"):
        _check_index_difference_is_zero(index1, index2)


def test_different_stride_raises_value_error():
    """Test that different stride values raise ValueError."""
    dim1 = index_symbol("dim1")

    index1 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
    }

    index2 = {
        dim1: IndexSequence(start=0, size=10, stride=2),
    }

    with pytest.raises(ValueError, match="Stride difference"):
        _check_index_difference_is_zero(index1, index2)


def test_different_keys_returns_false():
    """Test that different dictionary keys return False."""
    dim1 = index_symbol("dim1")
    dim2 = index_symbol("dim2")

    index1 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
    }

    index2 = {
        dim2: IndexSequence(start=0, size=10, stride=1),
    }

    result = _check_index_difference_is_zero(index1, index2)
    assert result is False


def test_empty_dictionaries_returns_true():
    """Test that empty dictionaries return True."""
    index1 = {}
    index2 = {}

    result = _check_index_difference_is_zero(index1, index2)
    assert result is True


def test_one_empty_one_not_returns_false():
    """Test that one empty and one non-empty dictionary returns False."""
    dim1 = index_symbol("dim1")

    index1 = {}
    index2 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
    }

    result = _check_index_difference_is_zero(index1, index2)
    assert result is False


def test_symbolic_expressions_simplify_to_zero():
    """Test that symbolic expressions that simplify to zero are considered equal."""
    dim1 = index_symbol("dim1")
    M = index_symbol("M")
    N = index_symbol("N")

    index1 = {
        dim1: IndexSequence(start=M + N, size=M * 2, stride=N - 1),
    }

    index2 = {
        dim1: IndexSequence(start=N + M, size=2 * M, stride=N - 1),
    }

    result = _check_index_difference_is_zero(index1, index2)
    assert result is True


def test_symbolic_expressions_do_not_simplify_to_zero():
    """Test that symbolic expressions that don't simplify to zero raise ValueError."""
    dim1 = index_symbol("dim1")
    M = index_symbol("M")
    N = index_symbol("N")

    index1 = {
        dim1: IndexSequence(start=M, size=N, stride=1),
    }

    index2 = {
        dim1: IndexSequence(start=M + 1, size=N, stride=1),
    }

    with pytest.raises(ValueError, match="Start difference"):
        _check_index_difference_is_zero(index1, index2)


def test_multiple_dimensions_all_equal():
    """Test multiple dimensions where all are equal."""
    dim1 = index_symbol("dim1")
    dim2 = index_symbol("dim2")
    dim3 = index_symbol("dim3")

    index1 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
        dim2: IndexSequence(start=5, size=20, stride=2),
        dim3: IndexSequence(start=10, size=30, stride=3),
    }

    index2 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
        dim2: IndexSequence(start=5, size=20, stride=2),
        dim3: IndexSequence(start=10, size=30, stride=3),
    }

    result = _check_index_difference_is_zero(index1, index2)
    assert result is True


def test_multiple_dimensions_one_different():
    """Test multiple dimensions where one differs."""
    dim1 = index_symbol("dim1")
    dim2 = index_symbol("dim2")
    dim3 = index_symbol("dim3")

    index1 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
        dim2: IndexSequence(start=5, size=20, stride=2),
        dim3: IndexSequence(start=10, size=30, stride=3),
    }

    index2 = {
        dim1: IndexSequence(start=0, size=10, stride=1),
        dim2: IndexSequence(start=5, size=25, stride=2),  # Different size
        dim3: IndexSequence(start=10, size=30, stride=3),
    }

    with pytest.raises(ValueError, match="Size difference"):
        _check_index_difference_is_zero(index1, index2)
