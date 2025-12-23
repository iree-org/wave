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
from wave_lang.kernel.lang.global_symbols import THREAD_0, WORKGROUP_0
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

from ..common.utils import param_bool, require_e2e, require_cdna3
from ._test_util import get_test_shapes


@require_e2e
@require_cdna3
@pytest.mark.parametrize(
    "shape", get_test_shapes("test_copy")[:2]
)  # testing on just two shapes.
@pytest.mark.parametrize(
    "tkl_dtype, torch_dtype, arg_vals",
    [  # arg_vals are c, d, e, res
        (tkl.i32, torch.int32, (1, 2, 1, 3)),
        (tkl.f32, torch.float32, (1.0, 2.0, 1.0, 3.0)),
    ],
    ids=["i32", "f32"],
)
@param_bool("use_wave_runtime", "wr", [False, True])
def test_scalar_codegen(
    shape, tkl_dtype, torch_dtype, arg_vals, use_wave_runtime, run_bench
):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

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
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl_dtype],
        c: tkl_dtype,  # type: ignore
        d: tkl_dtype,  # type: ignore
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl_dtype],
    ):
        res = tkw.read(a)
        c = tkw.broadcast(c, target_shape=[M, N])
        d = tkw.broadcast(d, target_shape=[M, N])
        e = tkl.Register[M, N, tkl_dtype](arg_vals[2]) * c
        res = res + e + d
        tkw.write(res, b)

    a = device_zeros(shape, dtype=torch_dtype)
    b = device_zeros(shape, dtype=torch_dtype)
    scalar_c = arg_vals[0]
    scalar_d = arg_vals[1]

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        wave_runtime=use_wave_runtime,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)
    test(a, scalar_c, scalar_d, b)

    expected_val = torch.full_like(b, arg_vals[3])
    if tkl.f32 == tkl_dtype and not use_wave_runtime:
        # TODO: iree runtime doesn't work with f32.
        with pytest.raises(Exception):
            assert_close(b, expected_val)
    else:
        assert_close(b, expected_val)


#  This kernel copies of data from a into b if tid.x < threshold.
#  This test is important to ensure:
#  1. tkw.Scalar can handle index expressions correctly.
#  2. Scalars in Wave can be used for comparison/binaryOps
#     as well as on select ops.
@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_scalar_cond_copy(shape, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

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
    # multiple of 4 to prevent to not require iota mask.
    # e.g if each thread has 4 values, and thresh is 10.
    # then t0 = [0, 1, 2, 3], t1 = [4, 5, 6, 7], t2 = [8, 9, 10, 11],
    # since t2_tidx_expr = 2 * 4 = 8, which is less than thresh, then
    # [10, 11] will also not be masked. To fix we'd need the iota mask
    # but not the main point of this test.
    thresh_value = 12
    tidx_expr = THREAD_0 * (BLOCK_N // wave_size) + (WORKGROUP_0 * BLOCK_N)

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        zero = tkw.scalar(0.0, tkl.f16)
        one = tkw.scalar(1.0, tkl.f16)

        tid = tkw.scalar(tidx_expr, tkl.i32)
        thresh = tkw.scalar(thresh_value, tkl.i32)

        mask = tkw.select(tid < thresh, one, zero)
        mask_broadcast = tkw.broadcast(mask, target_shape=[M, N])

        a_reg = tkw.read(a)
        res = a_reg * mask_broadcast
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
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
    # Check for data from tid.x < threshold
    assert_close(a[:, :thresh_value], b[:, :thresh_value])

    # Check for data from tid.x >= threshold
    ref_zeros = device_zeros([shape[0], shape[1] - thresh_value])
    assert_close(ref_zeros, b[:, thresh_value:], check_dtype=False)
