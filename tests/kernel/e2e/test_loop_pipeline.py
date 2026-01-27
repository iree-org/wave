# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
End-to-end tests for loop pipelining with dynamic shapes.

These tests verify that the pipelined and remainder loops correctly handle
various K dimensions, including edge cases where K is:
- Less than num_stages (all remainder, no pipelining)
- Equal to num_stages (exactly one pipelined iteration)
- Not evenly divisible by num_stages (has remainder)
- Not a multiple of BLOCK_K (odd sizes)
"""

import pytest
import torch
from torch.testing import assert_close

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

from ..common.utils import (
    require_e2e,
    require_cdna_3_or_4,
    require_rdna4,
)


# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K

# Address space
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@require_e2e
@pytest.mark.parametrize(
    "mma_type,threads_per_wave",
    [
        pytest.param(tkw.MMAType.F32_16x16x16_F16, 64, marks=require_cdna_3_or_4),
        pytest.param(
            tkw.MMAType.RDNA4_WAVE32_F32_16x16x16_F16, 32, marks=require_rdna4
        ),
    ],
)
@pytest.mark.parametrize(
    "K_value",
    [
        pytest.param(32, id="K=32_one_tile_less_than_num_stages"),
        pytest.param(64, id="K=64_two_tiles_equals_num_stages"),
        pytest.param(96, id="K=96_three_tiles_has_remainder"),
        pytest.param(133, id="K=133_odd_size_not_multiple_of_BLOCK_K"),
    ],
)
def test_gemm_pipelined_dynamic_K(K_value, mma_type, threads_per_wave, run_bench):
    # Fixed matrix dimensions for M and N, variable K
    m_size = 64
    n_size = 64

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            mma_type=mma_type,
        ),
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE_0, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE_0, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    # Generate test inputs
    torch.manual_seed(42)
    a = device_randn((m_size, K_value), dtype=torch.float16)
    b = device_randn((n_size, K_value), dtype=torch.float16)
    c = device_zeros((m_size, n_size), dtype=torch.float32)

    # Compute reference result using PyTorch
    ref_result = torch.matmul(a.to(torch.float32), b.T.to(torch.float32))

    subs = {
        M: m_size,
        N: n_size,
        # K is NOT substituted - left symbolic to trigger dynamic pipelining
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 32,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_0: SHARED_ADDRESS_SPACE,
    }
    subs.update(get_default_scheduling_params())

    compile_options = WaveCompileOptions(
        subs=subs,
        dynamic_symbols=[K],  # K is dynamic to trigger dynamic pipelining
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,  # Enable pipelining
        run_bench=run_bench,
    )
    compile_options = set_default_run_config(compile_options)

    compiled_gemm = wave_compile(compile_options, gemm)

    compiled_gemm(a, b, c)

    assert_close(c, ref_result, rtol=1e-3, atol=1e-3)
