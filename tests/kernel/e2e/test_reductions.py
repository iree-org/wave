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
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

from ..common.utils import require_e2e, require_cdna_2_or_3_or_4, require_rdna4
from ..common.shapes import get_test_shapes as get_common_test_shape
from ._test_util import get_test_shapes


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_reduce_sum"))
@pytest.mark.parametrize(
    "threads_per_wave",
    [
        pytest.param(64, marks=require_cdna_2_or_3_or_4),
        pytest.param(32, marks=require_rdna4),
    ],
)
def test_reduce_sum(shape, run_bench, threads_per_wave):
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            vector_shapes={M: 1, N: BLOCK_N},
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
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a)
        rhs = tkw.read(b)
        res = lhs * rhs
        res = tkw.sum(res, dim=N)
        tkw.write(res, c)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float16)
    b = device_randn(shape, dtype=torch.float16)
    c = device_zeros((shape[0],), dtype=torch.float16)
    ref = torch.sum((a * b), dim=-1)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b, c)
    assert_close(ref, c, atol=0.1, rtol=1e-05)


@require_e2e
@pytest.mark.parametrize("shape", get_common_test_shape("test_block_reduce"))
@pytest.mark.parametrize(
    "threads_per_wave",
    [
        # Enabling wave64 mode on RDNA generates %llvm.amdgcn.permlane32.swap,
        # this intrinsic gives access to v_permlane32_swap_b32 which is not a valid instruction in a RDNA4 device.
        pytest.param(64, marks=require_cdna_2_or_3_or_4),
        pytest.param(32, marks=require_rdna4),
    ],
)
def test_block_reduce_sum(shape, run_bench, threads_per_wave):
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
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        lhs = tkw.read(a)
        rhs = tkw.read(b)
        res = lhs * rhs
        res = tkw.sum(res, dim=N, block=True)
        tkw.write(res, c)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float32)
    b = device_randn(shape, dtype=torch.float32)
    c = device_zeros((shape[0],), dtype=torch.float32)
    ref = torch.sum((a * b), dim=-1)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b, c)
    assert_close(ref, c, atol=2e-5, rtol=1e-05)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_tiled_reduce_max"))
@pytest.mark.parametrize(
    "threads_per_wave",
    [
        # Enabling wave64 mode on RDNA generates intrinsic %llvm.amdgcn.permlane32.swap,
        # this intrinsic gives access to v_permlane32_swap_b32 which is not a valid instruction in a RDNA4 device.
        pytest.param(64, marks=require_cdna_2_or_3_or_4),
        pytest.param(32, marks=require_rdna4),
    ],
)
def test_toy_online_softmax(shape, threads_per_wave):
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = 1
    BLOCK_N = tkl.sym.BLOCK_N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        init_max = tkl.Register[M, tkl.f32](-1e6)
        init_sum = tkl.Register[M, tkl.f32](0)

        @tkw.iterate(N, init_args=[init_max, init_sum])
        def repeat(
            partial_max: tkl.Register[M, tkl.f32],
            partial_sum: tkl.Register[M, tkl.f32],
        ) -> tkl.Register[M, tkl.f32]:
            lhs = tkw.read(a)
            rhs = tkw.read(b)
            res = lhs * rhs
            partial_max = tkw.max(res, partial_max, dim=N)
            partial_sum = tkw.sum(res, partial_sum, dim=N)
            return partial_max, partial_sum

        res_max, res_sum = repeat
        result = res_max / res_sum
        tkw.write(result, c)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float32)
    b = device_randn(shape, dtype=torch.float32)
    c = device_zeros((shape[0],), dtype=torch.float32)
    ref_max = torch.max((a * b), dim=-1).values
    ref_sum = torch.sum((a * b), dim=-1)
    ref = ref_max / ref_sum
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_N: max(min(128, shape[1]), threads_per_wave),
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=False,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b, c)
    assert_close(ref, c, atol=0.1, rtol=1e-4)
