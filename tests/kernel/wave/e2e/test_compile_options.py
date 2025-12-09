# Copyright 2024-2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pytest
import sympy

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

from ..common.utils import param_bool, require_water_and_ee, glob_asm_files
from ._test_util import get_test_shapes
from pathlib import Path


@pytest.mark.parametrize("shape", get_test_shapes("test_copy")[:1])
def test_dump_vmfb(shape: tuple[int, int], tmp_path: Path):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    vmfb_file = tmp_path / "test.vmfb"
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        create_vmfb_file=vmfb_file,
    )
    options = set_default_run_config(options)
    options.target = "gfx942"

    assert not os.path.exists(vmfb_file)
    wave_compile(options, test)
    assert os.path.exists(vmfb_file)


_water_enable = [False, pytest.param(True, marks=require_water_and_ee)]


@pytest.mark.parametrize("shape", get_test_shapes("test_copy")[:1])
@param_bool("use_water_pipeline", "water", values=_water_enable)
def test_dump_intermediates(
    shape: tuple[int, int],
    use_water_pipeline: bool,
    tmp_path: Path,
) -> None:
    assert len(list(tmp_path.glob("*"))) == 0, "Directory is not empty"
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        use_buffer_ops=False,
        use_water_pipeline=use_water_pipeline,
        dump_intermediates=tmp_path,
    )
    options = set_default_run_config(options)
    options.target = "gfx942"
    wave_compile(options, test)
    asm_files = glob_asm_files(tmp_path)
    assert len(asm_files) == 1, "Expected 1 ASM file"
    text = asm_files[0].read_text()
    assert "global_load" in text, "Expected global_load instruction"
    assert "global_store" in text, "Expected global_store instruction"
