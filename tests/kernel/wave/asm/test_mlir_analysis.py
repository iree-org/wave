# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Unit tests for mlir_analysis.py kernel selection policy.

These tests verify:
1. should_skip_function correctly identifies functions to skip
2. The skip policy covers known wrapper patterns
3. Actual kernels are NOT skipped
"""

import pytest
from unittest.mock import MagicMock, PropertyMock


class TestShouldSkipFunction:
    """
    Tests for the kernel selection policy in mlir_analysis.should_skip_function.

    This policy is critical: if it incorrectly skips the actual kernel,
    compilation produces empty assembly and runtime fails with
    "named symbol not found" or "no kernel image" errors.
    """

    def _make_mock_func(self, name: str):
        """Create a mock FuncOp with the given name."""
        mock_fn = MagicMock()
        mock_fn.sym_name = MagicMock()
        mock_fn.sym_name.value = name
        return mock_fn

    def test_skip_isolated_benchmark_prefix(self):
        """Functions starting with 'isolated_benchmark' should be skipped."""
        from wave_lang.kernel.wave.asm.mlir_analysis import should_skip_function

        fn = self._make_mock_func("isolated_benchmark")
        assert should_skip_function(fn) is True

        fn = self._make_mock_func("isolated_benchmark_wrapper")
        assert should_skip_function(fn) is True

        fn = self._make_mock_func("isolated_benchmark$async")
        assert should_skip_function(fn) is True

    def test_skip_async_suffix(self):
        """Functions ending with '$async' should be skipped."""
        from wave_lang.kernel.wave.asm.mlir_analysis import should_skip_function

        fn = self._make_mock_func("my_kernel$async")
        assert should_skip_function(fn) is True

        fn = self._make_mock_func("gemm_kernel$async")
        assert should_skip_function(fn) is True

    def test_do_not_skip_regular_kernels(self):
        """Regular kernel functions should NOT be skipped."""
        from wave_lang.kernel.wave.asm.mlir_analysis import should_skip_function

        # Typical kernel names
        fn = self._make_mock_func("my_kernel")
        assert should_skip_function(fn) is False

        fn = self._make_mock_func("gemm_kernel")
        assert should_skip_function(fn) is False

        fn = self._make_mock_func("copy_kernel")
        assert should_skip_function(fn) is False

        # Names that might look similar but shouldn't be skipped
        fn = self._make_mock_func("benchmark_kernel")  # "benchmark" not at start
        assert should_skip_function(fn) is False

        fn = self._make_mock_func("async_operation")  # "async" not "$async" suffix
        assert should_skip_function(fn) is False

    def test_wave_asm_backend_pattern(self):
        """
        Test the specific pattern used by wave asm backend tests.

        The IREE pipeline generates:
        - isolated_benchmark$async (wrapper, skip)
        - isolated_benchmark (wrapper, skip)
        - the_actual_kernel (compile this!)
        """
        from wave_lang.kernel.wave.asm.mlir_analysis import should_skip_function

        # Wrapper functions to skip
        assert (
            should_skip_function(self._make_mock_func("isolated_benchmark$async"))
            is True
        )
        assert should_skip_function(self._make_mock_func("isolated_benchmark")) is True

        # Actual kernel (should NOT be skipped)
        # Real kernel names seen in tests
        assert should_skip_function(self._make_mock_func("mma_kernel")) is False
        assert should_skip_function(self._make_mock_func("copy_kernel")) is False
        assert should_skip_function(self._make_mock_func("gemm_kernel")) is False

    def test_edge_cases(self):
        """Test edge cases for function name matching."""
        from wave_lang.kernel.wave.asm.mlir_analysis import should_skip_function

        # Empty-ish names
        fn = self._make_mock_func("")
        assert should_skip_function(fn) is False

        # Just the patterns
        fn = self._make_mock_func("$async")
        assert should_skip_function(fn) is True  # ends with $async

        # Case sensitivity (patterns should be case-sensitive)
        fn = self._make_mock_func("Isolated_benchmark")  # Capital I
        assert should_skip_function(fn) is False  # Should NOT match

        fn = self._make_mock_func("ISOLATED_BENCHMARK")
        assert should_skip_function(fn) is False


class TestExtractTranslationInfo:
    """
    Tests for translation_info extraction from MLIR functions.
    """

    def _make_mock_func_with_attrs(self, name: str, attrs: dict):
        """Create a mock FuncOp with given name and attributes."""
        mock_fn = MagicMock()
        mock_fn.sym_name = MagicMock()
        mock_fn.sym_name.value = name
        mock_fn.name = MagicMock()
        mock_fn.name.value = name
        mock_fn.attributes = attrs
        return mock_fn

    def test_missing_translation_info_raises(self):
        """
        extract_translation_info should raise ValueError when translation_info
        is missing and require_translation_info=True (the default).
        """
        from wave_lang.kernel.wave.asm.mlir_analysis import extract_translation_info

        fn = self._make_mock_func_with_attrs("test_kernel", {})

        with pytest.raises(ValueError) as exc_info:
            extract_translation_info(fn, require_translation_info=True)

        assert "translation_info" in str(exc_info.value)
        assert "test_kernel" in str(exc_info.value)

    def test_missing_translation_info_with_fallback(self):
        """
        extract_translation_info should return defaults when
        require_translation_info=False and attribute is missing.
        """
        from wave_lang.kernel.wave.asm.mlir_analysis import extract_translation_info

        fn = self._make_mock_func_with_attrs("test_kernel", {})

        result = extract_translation_info(fn, require_translation_info=False)

        assert result.wg_size == (64, 1, 1)
        assert result.subgroup_size == 64
