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
    tool_path = get_waveasm_pkg_path() / "bin" / name
    if not tool_path.is_file() or not os.access(tool_path, os.X_OK):
        return None

    return str(tool_path)


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
