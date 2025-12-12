# Copyright 2024-2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pytest

from wave_lang.kernel.wave.compile import wave_compile
from ._test_util import get_test_shapes
from .test_copy import get_copy_template
from ..common.utils import use_water_backend_bool, glob_asm_files
from pathlib import Path


@pytest.mark.parametrize("shape", get_test_shapes("test_copy")[:1])
def test_dump_vmfb(shape: tuple[int, int], tmp_path: Path):
    vmfb_file = tmp_path / "test.vmfb"
    assert not os.path.exists(vmfb_file)
    options, test = get_copy_template(shape)
    options.create_vmfb_file = vmfb_file
    wave_compile(options, test)
    assert os.path.exists(vmfb_file)


@pytest.mark.parametrize("shape", get_test_shapes("test_copy")[:1])
@use_water_backend_bool("use_water_backend")
def test_dump_intermediates(
    shape: tuple[int, int],
    use_water_backend: bool,
    tmp_path: Path,
) -> None:
    assert len(list(tmp_path.glob("*"))) == 0, "Directory is not empty"
    options, test = get_copy_template(shape)
    options.dump_intermediates = tmp_path
    options.use_water_backend = use_water_backend
    wave_compile(options, test)
    asm_files = glob_asm_files(tmp_path)
    assert len(asm_files) == 1, "Expected 1 ASM file"
    text = asm_files[0].read_text()
    assert "global_load" in text, "Expected global_load instruction"
    assert "global_store" in text, "Expected global_store instruction"
