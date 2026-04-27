# Copyright 2025, The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import os
from functools import lru_cache


@lru_cache
def get_waveasm_pkg_path() -> Path:
    """Returns the path to the waveasm package."""
    # Assumes we are located at wave_lang/support/detect_waveasm.py.
    assert Path(__file__).parent.name == "support"
    assert Path(__file__).parent.parent.name == "wave_lang"
    wave_lang_path = Path(__file__).parent.parent
    return wave_lang_path / "kernel" / "wave" / "waveasm"


def find_binary(name: str) -> str | None:
    """Returns the path to the waveasm binary with the given name."""
    waveasm_dir = os.getenv("WAVE_WAVEASM_DIR")
    if waveasm_dir:
        tool_path = Path(waveasm_dir) / "bin" / name
        if tool_path.is_file() and os.access(tool_path, os.X_OK):
            return str(tool_path)

    tool_path = get_waveasm_pkg_path() / "bin" / name
    if tool_path.is_file() and os.access(tool_path, os.X_OK):
        return str(tool_path)

    repo_tool_path = (
        Path(__file__).parent.parent.parent / "waveasm" / "build" / "bin" / name
    )
    if repo_tool_path.is_file() and os.access(repo_tool_path, os.X_OK):
        return str(repo_tool_path)

    return None


@lru_cache
def is_waveasm_available() -> bool:
    """Returns True if the waveasm-translate binary is available."""
    return find_binary("waveasm-translate") is not None


@lru_cache
def get_waveasm_translate() -> str:
    """Returns the path to the waveasm-translate binary."""
    path = find_binary("waveasm-translate")
    if path is None:
        raise RuntimeError("waveasm-translate binary not found")

    return path


@lru_cache
def get_clang() -> str:
    """Returns the path to clang++ for assembling AMDGCN."""
    path = find_binary("clang++")
    if path is None:
        raise RuntimeError(
            "clang++ binary not found in waveasm package. "
            "Build with WAVE_BUILD_WAVEASM=1 or ensure clang is in the LLVM build."
        )

    return path
