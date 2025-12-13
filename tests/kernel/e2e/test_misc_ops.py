# Copyright 2024-2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import sympy
from torch.testing import assert_close

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import (
    device_arange,
    device_ones,
    device_randn,
    device_randint,
    device_zeros,
)

from ..common.utils import require_e2e, require_cdna_2_or_3_or_4, require_rdna4
from ._test_util import get_test_shapes


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_set_symbol(shape, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    S = tkl.sym.S
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.

    # TODO: Only ELEMS_PER_THREAD == 1
    # BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    BLOCK_N = wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N, S: BLOCK_M},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: S, N: j},
        outputs={S: i, N: j},
    )

    dynamic_symbols = []

    dynamic_symbols.append(S)

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[S, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
        )
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    off = device_randint(shape[0], shape, dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        dynamic_symbols=dynamic_symbols,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.take_along_dim(a, off.to(torch.long), dim=0)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_apply_expr(shape, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    S = tkl.sym.S
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.

    # TODO: Only ELEMS_PER_THREAD == 1
    # BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    BLOCK_N = wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N, S: BLOCK_M},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: S, N: j},
        outputs={S: i, N: j},
    )

    dynamic_symbols = []

    dynamic_symbols.append(S)

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[S, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        offset = tkw.apply_expr(offset, lambda a: M - a - 1)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
        )
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    off = device_randint(shape[0], shape, dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        dynamic_symbols=dynamic_symbols,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.take_along_dim(a, (shape[0] - off - 1).to(torch.long), dim=0)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", [(48, 4, 128)])
@pytest.mark.parametrize(
    "threads_per_wave",
    [
        # Enabling wave64 mode on RDNA generates %llvm.amdgcn.permlane32.swap,
        # this intrinsic gives access to v_permlane32_swap_b32 which is not a valid instruction in a RDNA4 device.
        pytest.param(64, marks=require_cdna_2_or_3_or_4),
        pytest.param(32, marks=require_rdna4),
    ],
)
def test_self_index(shape, run_bench, threads_per_wave):
    M = tkl.sym.M
    K = tkl.sym.K
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = threads_per_wave
    BLOCK_M = shape[0]
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size

    constraints = [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 0, K: 0, N: BLOCK_N},
        )
    ]

    # This kernel contains reduction + self_index.
    # It is loosely based on the speculative decode kernel.
    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, K, N, ADDRESS_SPACE, tkl.i32],
        result_self_index: tkl.Memory[M, K, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        input = tkw.read(a)
        input_sum = tkw.sum(input, dim=N)
        threshold = tkw.broadcast(input_sum, target_shape=[M, K, N])
        cdf = tkw.cumsum(input, dim=N)
        greather_than_cond = threshold >= cdf
        self_idx = tkw.self_index(N, dtype=tkl.i32)
        self_idx = tkw.broadcast(self_idx, target_shape=[M, K, N])
        select_min_id = tkw.select(greather_than_cond, self_idx, threshold)
        tkw.write(select_min_id, result_self_index)

    ref = device_arange(128, dtype=torch.int32)
    a = device_ones(shape, dtype=torch.int32)
    result_self_index = device_zeros(shape, dtype=torch.int32)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            K: shape[1],
            N: shape[2],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, result_self_index)

    assert_close(ref, result_self_index[0, 0, :])


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_conditional(shape, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # TODO: Only ELEMS_PER_THREAD == 1
    # BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    BLOCK_N = wave_size

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
        mask: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        cond = tkw.read(mask)

        cond = tkw.apply_expr(cond, lambda a: a > 0)

        @tkw.conditional(cond)
        def then():
            tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    mask = device_randint(2, shape, dtype=torch.int32)
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

    test(a, mask, b)
    assert_close(a * mask, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy")[:1])
@pytest.mark.parametrize("threshold", [2, 50])
def test_conditional_return(shape, threshold, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

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
        threshold: tkl.i32,
    ):
        zero_reg = tkl.Register[M, N, tkl.f32](0.0)

        threshold_scalar = tkw.scalar(10, tkl.i32)

        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)

        @tkw.conditional(threshold > threshold_scalar, else_return=[a_reg])
        def process_data(
            a: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            return a * tkw.scalar(2.5, tkl.f16)

        tkw.write(process_data, b, elements_per_thread=STORE_ELEMS_PER_THREAD)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b, threshold)

    if threshold > 10:
        expected = a * 2.5
    else:
        expected = a

    assert_close(expected, b)
