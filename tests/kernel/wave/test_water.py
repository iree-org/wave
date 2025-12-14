# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from wave_lang.kernel.wave.water import (
    apply_water_passes_with_passmanager,
    find_binary,
    is_water_available,
    _find_water_python_path,
)


@pytest.mark.skipif(
    not is_water_available(), reason="Water MLIR package not installed."
)
def test_find_binary():
    subprocess.check_call([find_binary("water-opt"), "--version"])


class TestWaterPassManager:
    """Test suite for Water PassManager functionality."""

    def test_find_water_python_path(self):
        """Test _find_water_python_path function."""
        # Test with mock get_water_opt
        with patch("wave_lang.kernel.wave.water.get_water_opt") as mock_get_water_opt:
            mock_get_water_opt.return_value = "/test/water/build/bin/water-opt"

            with patch.object(Path, "exists", return_value=True):
                result = _find_water_python_path()
                assert result == "/test/water/build/python_packages"

        # Test when no path found
        with patch(
            "wave_lang.kernel.wave.water.get_water_opt",
            side_effect=Exception("Not found"),
        ):
            with patch.object(Path, "exists", return_value=False):
                result = _find_water_python_path()
                assert result is None

    def test_apply_water_passes_with_passmanager_no_path(self):
        """Test apply_water_passes_with_passmanager when no Water path found."""
        with patch(
            "wave_lang.kernel.wave.water._find_water_python_path", return_value=None
        ):
            with pytest.raises(
                RuntimeError, match="Could not find Water Python bindings directory"
            ):
                apply_water_passes_with_passmanager("module {}")

    def test_apply_water_passes_with_passmanager_import_error(self):
        """Test apply_water_passes_with_passmanager when imports fail."""
        with patch(
            "wave_lang.kernel.wave.water._find_water_python_path",
            return_value="/mock/path",
        ):
            with pytest.raises(
                RuntimeError, match="Water Python bindings not available"
            ):
                apply_water_passes_with_passmanager("module {}")

    @patch("wave_lang.kernel.wave.water._find_water_python_path")
    def test_apply_water_passes_with_passmanager_success(self, mock_find_path):
        """Test apply_water_passes_with_passmanager with mocked Water components."""
        mock_find_path.return_value = "/mock/water/python_packages"

        mlir_input = (
            "module attributes {wave.normal_form = #wave.normal_form<full_types>} {}"
        )

        # Mock Water components
        mock_context = MagicMock()
        mock_module = MagicMock()
        mock_pass_manager = MagicMock()
        mock_wave_dialect = MagicMock()

        # Setup context manager
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=False)

        # Mock module output (after passes and cleaning)
        mock_module.__str__ = MagicMock(return_value="module {}")

        with patch.dict(
            "sys.modules",
            {
                "water_mlir": MagicMock(),
                "water_mlir.ir": MagicMock(
                    Context=MagicMock(return_value=mock_context),
                    Module=MagicMock(parse=MagicMock(return_value=mock_module)),
                ),
                "water_mlir.passmanager": MagicMock(
                    PassManager=MagicMock(
                        parse=MagicMock(return_value=mock_pass_manager)
                    )
                ),
                "water_mlir.dialects": MagicMock(),
                "water_mlir.dialects.wave": mock_wave_dialect,
            },
        ):
            result = apply_water_passes_with_passmanager(mlir_input)

            # Verify basic operation
            assert result is not None
            assert "wave.normal_form" not in result


@pytest.mark.skipif(
    not is_water_available(), reason="Water MLIR package not installed."
)
class TestWaterPassManagerIntegration:
    """Integration tests that require actual Water infrastructure."""

    def test_passmanager(self):
        # Test with minimal MLIR
        result = apply_water_passes_with_passmanager("module {}")
        assert result is not None
        assert "module" in result
        # Wave attributes should be cleaned
        assert "wave.normal_form" not in result
