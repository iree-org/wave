# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest
import torch
from torch.nn import functional as F
import wave_lang.kernel as tk
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from .common.utils import (
    require_e2e,
)
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions


@pytest.mark.parametrize(
    "shape",
    [
        (1, 5120),
    ],
)
@require_e2e
def test_rmsnorm(shape):
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    EMB_SIZE = tkl.sym.EMB_SIZE
    TOKENS_PER_WK = tkl.sym.TOKENS_PER_WK

    num_waves = 4
    wave_size = 64
    BLOCK_N = N
    BLOCK_M = TOKENS_PER_WK

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: ELEMS_PER_THREAD * wave_size},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / num_waves)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        gamma: tkl.Memory[N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        length_embedding = tkl.Register[M, tkl.f16](EMB_SIZE)
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        lhs_pow = lhs * lhs
        red = tkw.sum(lhs_pow, dim=N, block=True)
        result = red / length_embedding
        rms = tkw.sqrt(result)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = lhs / rms_broad
        gamma_reg = tkw.read(gamma, elements_per_thread=ELEMS_PER_THREAD)
        gamma_broad = tkw.broadcast(gamma_reg, [M, N])
        output = a_scaled * gamma_broad
        tkw.write(output, c, elements_per_thread=ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            TOKENS_PER_WK: 1,
            EMB_SIZE: shape[1],
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float16)
    gamma = device_randn(shape[1], dtype=torch.float16)
    c = device_zeros(shape, dtype=torch.float16)
    test(a, gamma, c)

    rms = torch.sqrt(torch.mean(a * a, dim=-1, keepdim=True))
    ref = (a / rms) * gamma
    torch.testing.assert_close(ref, c, atol=1e-02, rtol=1e-05)
