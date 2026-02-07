# Copyright 2024-2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sympy
import torch
from torch.testing import assert_close

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randint, device_zeros
from ..common.shapes import get_test_shapes as get_common_test_shape
from ..common.utils import require_e2e, require_cdna_2_or_3_or_4, require_rdna4

from ..common.utils import require_e2e


@require_e2e
@pytest.mark.parametrize(
    "shape",
    [
        (27,),
        (51,),
        (64,),
        (65,),
        (128,),
        (256,),
        (500,),
        (512,),
    ],
)
def test_1d_scanop_cumsum(shape, run_bench):
    N = tkl.sym.N
    wave_size = 64
    num_warps = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={N: BLOCK_N // num_warps},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[N, GLOBAL_ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        lhs = tkw.read(a)
        res = tkw.cumsum(lhs, dim=N)
        tkw.write(res, c)

    torch.manual_seed(1)
    input = device_randint(low=1, high=5, size=shape, dtype=torch.int32)
    output = device_zeros(shape, dtype=torch.int32)
    torch_ref = torch.cumsum((input), dim=-1, dtype=torch.int32)
    options = WaveCompileOptions(
        subs={
            N: shape[0],
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(input, output)
    assert_close(torch_ref, output, atol=1e-03, rtol=1e-05)


@require_e2e
@pytest.mark.parametrize(
    "shape",
    [
        (1, 27),
        (1, 64),
        (51, 64),
        (128, 64),
        (1, 256),
        (1, 512),
        (64, 500),
    ],
)
def test_2d_scanop_cumsum(shape, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    num_warps = 1
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N // num_warps},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        lhs = tkw.read(a)
        res = tkw.cumsum(lhs, dim=N)
        tkw.write(res, c)

    torch.manual_seed(1)
    input = device_randint(low=1, high=5, size=shape, dtype=torch.int32)
    output = device_zeros(shape, dtype=torch.int32)
    torch_ref = torch.cumsum((input), dim=-1, dtype=torch.int32)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(input, output)
    assert_close(torch_ref, output, atol=1e-03, rtol=1e-05)


@require_e2e
@pytest.mark.parametrize("shape", get_common_test_shape("test_block_scan"))
@pytest.mark.parametrize(
    "threads_per_wave",
    [
        # Enabling wave64 mode on RDNA generates %llvm.amdgcn.permlane32.swap,
        # this intrinsic gives access to v_permlane32_swap_b32 which is not a valid instruction in a RDNA4 device.
        pytest.param(64, marks=require_cdna_2_or_3_or_4),
        pytest.param(32, marks=require_rdna4),
    ],
)
def test_block_scanop_cumsum(shape, run_bench, threads_per_wave):
    round_to_divisible = lambda src, denom: sympy.ceiling(src / denom) * denom
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = threads_per_wave
    num_waves = 4
    BLOCK_M = 1

    # Distribute N dim across num_waves, and pad to disivible by wave_size.
    ELEMS_PER_WAVE = round_to_divisible(sympy.ceiling(N / num_waves), wave_size)
    # Minimum number of elems per wave should be size of wave.
    ELEMS_PER_WAVE = sympy.Max(ELEMS_PER_WAVE, wave_size)
    BLOCK_N = ELEMS_PER_WAVE * num_waves
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: 1, N: ELEMS_PER_WAVE},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, ELEMS_PER_WAVE)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        lhs = tkw.read(a)
        res = tkw.cumsum(lhs, dim=N, block=True)
        tkw.write(res, c)

    input = device_randint(low=1, high=99, size=shape, dtype=torch.int32)
    output = device_zeros(shape, dtype=torch.int32)
    torch_ref = torch.cumsum((input), dim=-1, dtype=torch.int32)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(input, output)
    assert_close(torch_ref, output, atol=1e-05, rtol=1e-05)
