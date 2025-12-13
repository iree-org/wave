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
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import torch_dtype_to_wave
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randint, device_zeros

from ..common.utils import param_bool, require_e2e, require_cdna_2_or_3_or_4


@require_e2e
@pytest.mark.parametrize("shape", [(2, 64)])
@param_bool("use_buffer_ops", "buf_ops")
def test_atomic_min(shape, use_buffer_ops, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 2
    BLOCK_N = 64
    num_waves = 2

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={
                M: int(BLOCK_M / num_waves),
                N: BLOCK_N,
            },
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, sympy.Integer(1))]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: sympy.Integer(0), N: j},
        outputs={M: i, N: j},
    )
    read_mapping = tkw.IndexMapping(
        num_iterators=2, inputs={M: sympy.Integer(0), N: j}, outputs={M: i, N: j}
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
    ):
        res = tkw.read(a)
        # We allocate a buffer of (1,BLOCK_N) shape to perform reduction across
        # waves. Inputs are distributed with (1,BLOCK_N) shape across each wave
        # and performs atomic min operation on this shared memory space. Mapping
        # attribute to atomic_min op is utilized to access the same shared memory
        # from different waves.
        shmem = tkw.allocate(
            shape=(M, N),
            distributed_shape=(1, BLOCK_N),
            dtype=tkl.i32,
            address_space=SHARED_ADDRESS_SPACE,
        )
        inf_reg = tkl.Register[M, N, tkl.i32](1e6)
        tkw.write(inf_reg, shmem)
        res = tkw.atomic_min(res, shmem, mapping=mapping)
        res = tkw.read(shmem, mapping=read_mapping)
        tkw.write(res, c)

    a = device_randint(low=0, high=10, size=shape, dtype=torch.int32)
    b = torch.min(a, dim=0)[0].detach()
    c = device_zeros(size=shape, dtype=torch.int32)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
            SHARED_ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
        minimize_shared_allocs=False,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)
    test(a, c)
    assert_close(c[0, :], b)
    assert_close(c[1, :], b)


@require_e2e
@require_cdna_2_or_3_or_4
def test_no_map_atomic_add():
    M = tkl.sym.M
    B = tkl.sym.B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_B = tkl.sym.BLOCK_B
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
    LIMIT_VAL = tkl.sym.LIMIT_VAL

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={B: 0, M: 64, LIMIT_VAL: 0},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    # B is iterated over and so we define a tiling constraint on it.
    # However, there is no notion of tile size for the iteration as
    # it is an unstructured loop.
    constraints += [tkw.TilingConstraint(B)]

    @tkw.wave(constraints)
    def atomic_add_one(
        a: tkl.Memory[M, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, ADDRESS_SPACE_0, tkl.i32],
    ):

        one_reg = tkw.Register[M, tkl.i32](1)
        res = tkw.atomic_add(one_reg, a)
        tkw.write(res, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            B: 10,
            BLOCK_M: 64,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    )
    options = set_default_run_config(options)
    atomic_add_one = wave_compile(options, atomic_add_one)

    # generate random input tensors between -1 and 1
    a = torch.randint(0, 10, (64,), dtype=torch.int32).cuda()
    c = torch.zeros((64,), dtype=torch.int32).cuda()
    a_expected = a.clone() + 1
    a_original = a.clone()
    atomic_add_one(a, c)

    assert torch.equal(c, a_original)
    assert torch.equal(a, a_expected)


@require_e2e
@require_cdna_2_or_3_or_4
@pytest.mark.parametrize("shape", [(64, 4)])
def test_dyn_atomic_add(shape):
    # Input sizes
    NUMEL = tkl.sym.NUMEL
    NUM_EXPERTS = tkl.sym.NUM_EXPERTS

    constraints: list[tkw.Constraint] = []

    # one workgroup to handle the worload
    constraints += [tkw.WorkgroupConstraint(NUMEL, NUMEL, 0)]
    constraints += [tkw.WorkgroupConstraint(NUM_EXPERTS, NUM_EXPERTS, 1)]
    # one wave to handle the workload
    constraints += [tkw.WaveConstraint(NUMEL, NUMEL)]
    constraints += [tkw.WaveConstraint(NUM_EXPERTS, NUM_EXPERTS)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={NUMEL: NUMEL, NUM_EXPERTS: NUM_EXPERTS},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    histogram_read = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: d0},
        outputs={NUM_EXPERTS: i},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    tdtype = torch.int32
    dtype = torch_dtype_to_wave(tdtype)

    @tkw.wave(constraints)
    def create_histogram(
        indices: tkl.Memory[NUMEL, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype],
        experts: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
    ):

        # create a vector of zeros and ones
        zero_vec = tkl.Register[NUM_EXPERTS, dtype](0)
        one_vec = tkw.Register[NUM_EXPERTS, dtype](1)

        # read the index in the range [0, NUM_EXPERTS)
        idx = tkw.read(indices, elements_per_thread=1)

        # allocate shared memory for the histogram
        shmem = tkw.allocate(
            shape=(NUM_EXPERTS,),
            distributed_shape=(NUM_EXPERTS,),
            dtype=dtype,
        )

        # initialize the histogram to zero
        tkw.write(zero_vec, shmem)

        # atomic add 1 to the index read
        tkw.atomic_add(
            one_vec,
            shmem,
            mapping=histogram_read,
            mapping_dynamic_vals=(idx,),
            elements_per_thread=1,
        )

        # write back the result
        counts = tkw.read(shmem)
        tkw.write(
            counts,
            experts,
        )

    num_indices = shape[0]
    num_experts = shape[1]
    indices = device_randint(0, num_experts, (num_indices,), dtype=torch.int32)
    experts = device_zeros((num_experts,), dtype=torch.int32)

    hyperparams = {
        NUMEL: num_indices,
        NUM_EXPERTS: num_experts,
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        minimize_shared_allocs=False,
    )
    options = set_default_run_config(options)
    compiled_fn = wave_compile(options, create_histogram)
    compiled_fn(indices, experts)

    # verify the result
    expected_experts = torch.bincount(indices, minlength=num_experts).to(dtype=tdtype)
    assert_close(experts, expected_experts)
    assert experts.sum() == num_indices
