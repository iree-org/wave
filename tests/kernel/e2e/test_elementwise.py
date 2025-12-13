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
from wave_lang.kernel.wave.templates.test_kernels import get_broadcast_scaled_add
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

from ..common.utils import (
    param_bool,
    require_e2e,
    require_cdna_2_or_3_or_4,
    require_rdna4,
)
from ._test_util import get_test_shapes


@require_e2e
@pytest.mark.parametrize("shape", [(256, 64)])
def test_cast(shape, run_bench):
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
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        res = tkw.cast(res, tkl.f16)
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float32)
    b = device_zeros(shape, dtype=torch.float16)
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

    test(a, b)
    assert_close(a.to(dtype=torch.float16), b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_vector_add")[:2])
@param_bool("use_buffer_ops", "buf_ops")
def test_vector_add(shape, use_buffer_ops, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
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
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a)
        rhs = tkw.read(b)
        res = lhs + rhs
        tkw.write(res, c)

    a = device_randn(shape, dtype=torch.float16)
    b = device_randn(shape, dtype=torch.float16)
    c = device_zeros(shape, dtype=torch.float16)
    ref = a + b

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)

    test = wave_compile(options, test)

    test(a, b, c)
    assert_close(ref, c)


@require_e2e
@pytest.mark.parametrize("shape", [(256, 256)])
def test_broadcast_scaled_add(shape, run_bench):
    broadcast_scaled_add, hyperparams = get_broadcast_scaled_add(shape)

    scaled_shape = list(shape)
    scaled_shape[-1] = int(scaled_shape[-1] / 2)
    scaled_shape = tuple(scaled_shape)
    a = device_randn(scaled_shape, dtype=torch.float16)
    b = device_randn((scaled_shape[0]), dtype=torch.float16)
    c = device_zeros(scaled_shape, dtype=torch.float16)
    ref = a + b.view(-1, 1)

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=True,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, broadcast_scaled_add)

    test(a, b, c)
    assert_close(ref, c)
