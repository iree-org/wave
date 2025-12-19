# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import subprocess
from unittest.mock import patch
from wave_lang.kernel.wave.water import (
    apply_water_middle_end_passes,
)
from wave_lang.support.detect_water import (
    find_binary,
    get_water_opt,
    is_water_available,
    is_water_binary_available,
)


@pytest.mark.skipif(
    not is_water_binary_available(), reason="water-opt binary not available."
)
def test_find_binary():
    subprocess.check_call([find_binary("water-opt"), "--version"])


class TestWaterLowering:
    """Test suite for Water lowering functionality."""

    def test_is_water_available(self):
        """Test is_water_available function."""
        result = is_water_available()
        assert isinstance(result, bool)

    def test_apply_water_middle_end_passes_unavailable(self):
        """Test apply_water_middle_end_passes when water-opt is not available."""
        # Mock find_binary to return None, simulating water-opt not being found
        get_water_opt.cache_clear()  # get_water_opt caches find_binary result
        try:
            with patch("wave_lang.support.detect_water.find_binary", return_value=None):
                with pytest.raises(RuntimeError, match="water-opt binary not found"):
                    apply_water_middle_end_passes("module {}")
        finally:
            get_water_opt.cache_clear()

    def test_apply_water_middle_end_passes_success(self):
        """Test apply_water_middle_end_passes with mocked subprocess."""
        mlir_input = (
            "module attributes {wave.normal_form = #wave.normal_form<full_types>} {}"
        )
        expected_output = "module {}"

        with patch("wave_lang.kernel.wave.water.get_water_opt") as mock_get_water_opt:
            mock_get_water_opt.return_value = "/mock/water-opt"

            with patch("subprocess.check_output") as mock_subprocess:
                mock_subprocess.return_value = expected_output

                result = apply_water_middle_end_passes(mlir_input)

                # Verify basic operation
                assert result is not None
                assert "wave.normal_form" not in result

                # Verify subprocess was called correctly
                mock_subprocess.assert_called_once()
                args, kwargs = mock_subprocess.call_args
                assert args[0][0] == "/mock/water-opt"
                assert "--allow-unregistered-dialect" in args[0]
                assert "--pass-pipeline=builtin.module(" in args[0][2]
                assert kwargs["input"] == mlir_input
                assert kwargs["text"] is True

    def test_apply_water_middle_end_passes_subprocess_error(self):
        """Test apply_water_middle_end_passes when subprocess fails."""
        with patch("wave_lang.kernel.wave.water.get_water_opt") as mock_get_water_opt:
            mock_get_water_opt.return_value = "/mock/water-opt"

            with patch("subprocess.check_output") as mock_subprocess:
                mock_subprocess.side_effect = subprocess.CalledProcessError(
                    1, "water-opt", stderr="Parse error"
                )

                with pytest.raises(
                    RuntimeError, match="water-opt subprocess failed with return code 1"
                ):
                    apply_water_middle_end_passes("module {}")


@pytest.mark.skipif(
    not is_water_binary_available(), reason="water-opt binary not available."
)
class TestWaterLoweringIntegration:
    def test_lowering_passes(self):
        # Test with simple Wave dialect operations - just register and add
        wave_mlir = """
        module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
          func.func @test_kernel() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
            %cst = arith.constant 0.0 : f32
            %lhs = wave.register %cst : vector<4xf32>
            %cst1 = arith.constant 1.0 : f32
            %rhs = wave.register %cst1 : vector<4xf32>
            %result = wave.add %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
            return
          }
        }
        """

        result = apply_water_middle_end_passes(wave_mlir)
        assert result is not None
        assert "module" in result
        assert "func.func @test_kernel" in result

        # Verify Wave operations are lowered
        assert "wave.register" not in result
        assert "wave.add" not in result

        # Wave attributes should be cleaned
        assert "wave.normal_form" not in result

        # Only `return` should remain
        assert "return" in result
