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
from wave_lang.kernel.wave.utils.torch_utils import (
    device_arange,
    device_randint,
    device_randn,
    device_randperm,
    device_zeros,
)

from ..common.utils import (
    param_bool,
    require_e2e,
    require_cdna_2_or_3_or_4,
    require_rdna4,
)


@require_e2e
@pytest.mark.parametrize(
    "shape, elems_per_thread",
    [
        ((3840, 1), 1),
        ((64, 64), 1),
        ((64, 64), 2),
        ((64, 64), 4),
    ],
)
def test_scatter_add(shape, elems_per_thread, request):
    run_bench = request.config.getoption("--runperf")

    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    m_size, n_size = shape

    constraints = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 64, N: elems_per_thread},
        ),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i, N: j},
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        index: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
        lds: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        index_reg = tkw.read(index, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        tkw.scatter_add(
            a_reg,
            index_reg,
            dim=0,
            memory=lds,
            mapping=mapping,
            elements_per_thread=STORE_ELEMS_PER_THREAD,
        )
        lds_reg = tkw.read(
            lds, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=mapping
        )
        tkw.write(
            lds_reg, b, elements_per_thread=STORE_ELEMS_PER_THREAD, mapping=mapping
        )

    options = WaveCompileOptions(
        subs={
            M: m_size,
            N: n_size,
            BLOCK_M: m_size,
            BLOCK_N: n_size,
            LOAD_ELEMS_PER_THREAD: elems_per_thread,
            STORE_ELEMS_PER_THREAD: elems_per_thread,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test_fn = wave_compile(options, test)

    input = (
        device_arange(m_size * n_size, dtype=torch.float32)
        .reshape(m_size, n_size)
        .contiguous()
    )
    index = device_randint(0, m_size, (m_size, n_size), dtype=torch.int32).contiguous()
    lds = device_zeros((m_size, n_size), dtype=torch.float32).contiguous()
    output = device_zeros((m_size, n_size), dtype=torch.float32).contiguous()

    test_fn(input, index, lds, output)

    def scatter_add_baseline(input, index):
        index = index.to(dtype=torch.int64)
        if index.shape != input.shape:
            while index.dim() < input.dim():
                index = index.unsqueeze(-1)
            index = index.expand_as(input)
        baseline_output = device_zeros(input.shape, dtype=torch.float32)
        return baseline_output.scatter_add(dim=0, index=index, src=input)

    torch_output = scatter_add_baseline(input, index)
    assert_close(output, torch_output)


@require_e2e
@pytest.mark.parametrize("shape,k", [((32, 64), 2), ((64, 128), 4), ((128, 256), 8)])
@param_bool("allow_duplicates", "duplicates")
@pytest.mark.parametrize(
    "threads_per_wave",
    [
        # Enabling wave64 mode on RDNA generates %llvm.amdgcn.permlane32.swap,
        # this intrinsic gives access to v_permlane32_swap_b32 which is not a valid instruction in a RDNA4 device.
        pytest.param(64, marks=require_cdna_2_or_3_or_4),
        pytest.param(32, marks=require_rdna4),
    ],
)
def test_topk(shape, k, allow_duplicates, threads_per_wave, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / threads_per_wave) * threads_per_wave
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            vector_shapes={M: 1, N: BLOCK_N, K: K},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        values: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        indices: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i32],
    ):
        src = tkw.read(a)
        topk_values, topk_indices = tkw.topk(src, K, N)
        tkw.write(topk_values, values, elements_per_thread=K)
        tkw.write(topk_indices, indices, elements_per_thread=K)

    torch.manual_seed(1)

    if allow_duplicates:
        a = device_randn(shape, dtype=torch.float16)
    else:
        # Generate input with no duplicates per row.
        # Each row contains unique values by using a shuffled range.
        a = device_zeros(shape, dtype=torch.float16)
        for i in range(shape[0]):
            perm = device_randperm(shape[1], dtype=torch.int32)
            unique_values = (perm.float() + device_randn(shape[1]) * 0.1).to(
                torch.float16
            )
            a[i, :] = unique_values

    values_out = device_zeros((shape[0], k), dtype=torch.float16)
    indices_out = device_zeros((shape[0], k), dtype=torch.int32)

    ref_values, ref_indices = torch.topk(a, k, dim=-1)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            K: k,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, values_out, indices_out)

    assert values_out.shape == ref_values.shape
    assert indices_out.shape == ref_indices.shape
    assert_close(ref_values, values_out, atol=0.1, rtol=1e-05)
    # When there are duplicate values, the indices may be different due to
    # difference in tie breaking during sort.
    if not allow_duplicates:
        assert torch.equal(ref_indices, indices_out)


@require_e2e
@pytest.mark.parametrize("shape", [(2, 128), (256, 1024)])
@param_bool("use_buffer_ops", "buf_ops")
@pytest.mark.parametrize(
    "threads_per_wave",
    [
        # Enabling wave64 mode on RDNA generates intrinsic %llvm.amdgcn.permlane32.swap,
        # this intrinsic gives access to v_permlane32_swap_b32 which is not a valid instruction in a RDNA4 device.
        pytest.param(64, marks=require_cdna_2_or_3_or_4),
        pytest.param(32, marks=require_rdna4),
    ],
)
def test_fused_softmax(shape, use_buffer_ops, threads_per_wave):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    BLOCK_M = 1

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            vector_shapes={M: BLOCK_M, N: N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        val = tkw.read(a)
        row_max = tkw.max(val, dim=N)
        row_max_bcast = tkw.broadcast(row_max, [M, N])
        val -= row_max_bcast
        val = tkw.exp(val)
        denominator = tkw.sum(val, dim=N)
        denom_broadcast = tkw.broadcast(denominator, [M, N])
        val = val / denom_broadcast
        tkw.write(val, b)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float32)
    b = device_zeros(shape, dtype=torch.float32)

    ref = torch.softmax(a, dim=1)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=False,
        use_buffer_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)
    test(a, b)
    assert_close(ref, b)
