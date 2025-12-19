# Copyright 2025, The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import os
from functools import lru_cache


@lru_cache
def get_water_mlir_pkg_path() -> Path:
    """Returns the path to the water_mlir package."""
    # Assumes we are located at wave_lang/support/detect_water.py
    assert Path(__file__).parent.name == "support"
    assert Path(__file__).parent.parent.name == "wave_lang"
    wave_lang_path = Path(__file__).parent.parent
    return wave_lang_path / "kernel" / "wave" / "water_mlir"


def find_binary(name: str) -> str | None:
    """Returns the path to the water binary with the given name."""
    tool_path = get_water_mlir_pkg_path() / "bin" / name
    if not tool_path.is_file() or not os.access(tool_path, os.X_OK):
        return None

    return str(tool_path)


@lru_cache
def is_water_available() -> bool:
    """Returns True if the water_mlir package is available."""
    return (get_water_mlir_pkg_path() / "water_mlir").exists()


@lru_cache
def is_water_binary_available() -> bool:
    """Returns True if the water-opt binary is available and executable."""
    return find_binary("water-opt") is not None


@lru_cache
def get_water_opt() -> str:
    """Returns the path to the water-opt binary."""
    path = find_binary("water-opt")
    if path is None:
        raise RuntimeError("water-opt binary not found")

    return path
