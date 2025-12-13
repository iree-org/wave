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

from ..common.utils import param_bool, require_e2e, require_cdna4
from ._test_util import get_test_shapes


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_transpose_read"))
@param_bool("use_buffer_ops", "buf_ops")
def test_transpose_read(shape, use_buffer_ops, run_bench):
    shape = shape[::-1]
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_N = 1
    BLOCK_M = sympy.Max(sympy.Min(M, 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={N: BLOCK_N, M: BLOCK_M},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, M: j}, outputs={N: i, M: j}
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, mapping=mapping)
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape[::-1], dtype=torch.float16)
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

    test(a, b)
    assert_close(a.T, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_transpose_write"))
@param_bool("use_buffer_ops", "buf_ops")
def test_transpose_write(shape, use_buffer_ops, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.Max(sympy.Min(N, 256), wave_size)

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

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={M: i, N: j}, outputs={M: i, N: j}
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b, mapping=mapping)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape[::-1], dtype=torch.float16)
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

    test(a, b)
    assert_close(a.T, b)


@require_e2e
@require_cdna4
@pytest.mark.parametrize(
    "shape",
    [
        # Square matrices
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        # Rectangular (power of 2)
        (512, 256),
        (256, 512),
        (1024, 512),
        (512, 1024),
        (2048, 1024),
        # Aligned but not power of 2 (meet Tm%4==0, Tk%4==0 for f16)
        (384, 256),  # 3*128 x 256
        (768, 512),  # 3*256 x 512
        (1280, 512),  # 5*256 x 512
        (512, 768),  # 512 x 3*256
        # Extreme aspect ratios
        (64, 1024),  # Very wide (1:16 ratio)
        (1024, 64),  # Very tall (16:1 ratio)
        (128, 2048),  # Ultra-wide (1:16 ratio)
        (2048, 128),  # Ultra-tall (16:1 ratio)
    ],
)
@param_bool("use_buffer_ops", "buf_ops")
def test_arbitrary_transpose(shape, use_buffer_ops, run_bench):
    """
    Test transposing arbitrary shaped 2D tensors using transpose read.

    This test validates the unified transpose formula works correctly for:
    - Aligned and unaligned matrix dimensions
    - Various aspect ratios (square, wide, tall)
    - Edge cases and boundary conditions
    """
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_N = 1
    BLOCK_M = sympy.Max(sympy.Min(M, 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={N: BLOCK_N, M: BLOCK_M},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, M: j}, outputs={N: i, M: j}
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, mapping=mapping)
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape[::-1], dtype=torch.float16)
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

    test(a, b)
    assert_close(a.T, b)
