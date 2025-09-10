# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel._support.dtype import DataType
import sympy
import torch

from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)


def get_moe_align_block_size_kernel(
    num_tokens: int,
    num_experts: int,
    top_k_value: int = 2,
    dtype: torch.dtype = torch.int32,
):
    """
    Wave kernel for MoE token alignment and block size padding.

    This kernel sorts tokens by their assigned expert IDs and pads each expert's
    tokens to align with the specified block size for efficient processing.
    """
    dtype = torch_dtype_to_wave(dtype)

    # Input sizes
    NUM_TOKENS = tkl.sym.NUM_TOKENS
    NUM_EXPERTS = tkl.sym.NUM_EXPERTS
    NUMEL = tkl.sym.NUMEL
    TOPK = tkl.sym.TOPK
    BLOCK_SIZE = tkl.sym.BLOCK_SIZE

    # Workgroup tile sizes
    BLOCK_TOKENS = tkl.sym.BLOCK_TOKENS
    BLOCK_EXPERTS = tkl.sym.BLOCK_EXPERTS

    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []

    # one workgroup to handle the worload
    constraints += [tkw.WorkgroupConstraint(NUMEL, NUMEL, 0)]
    constraints += [tkw.WorkgroupConstraint(NUM_EXPERTS, NUM_EXPERTS, 1)]
    # one wave to handle the workload
    constraints += [tkw.WaveConstraint(NUMEL, NUMEL)]
    constraints += [tkw.WaveConstraint(NUM_EXPERTS, NUM_EXPERTS)]
    # constraints += [tkw.TilingConstraint(NUMEL, NUMEL)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={NUMEL: NUMEL, NUM_EXPERTS: NUM_EXPERTS},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    expert_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: d0},
        outputs={NUM_EXPERTS: i},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    expert_write_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: i},
        outputs={NUM_EXPERTS: d0},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    shifted_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: d0},
        outputs={NUM_EXPERTS: i},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    shifted_write_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: i},
        outputs={NUM_EXPERTS: d0 + 1},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    simple_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: i},
        outputs={NUM_EXPERTS: i},
    )

    printer_args = None

    def printer(*args):
        nonlocal printer_args
        printer_args = args

    @tkw.wave(constraints)
    def moe_align_block_size(
        topk_ids: tkl.Memory[NUMEL, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype],
        expert_counts: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
        padded_counts: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
        cumsum_buffer: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
        num_blocks_buffer: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
    ):

        tid = tkw.scalar(THREAD_0, tkl.i32)
        num_experts = tkw.scalar(NUM_EXPERTS - 1, tkl.i32)
        zero_counts = tkl.Register[NUM_EXPERTS, dtype](0)
        one_reg = tkw.Register[NUM_EXPERTS, dtype](1)
        shifted_cumsum = tkw.Register[NUM_EXPERTS, dtype](0)

        shmem = tkw.allocate(
            shape=(NUM_EXPERTS,),
            distributed_shape=(NUM_EXPERTS,),
            dtype=dtype,
        )
        cumsum_exclusive = tkw.allocate(
            shape=(NUM_EXPERTS,),
            distributed_shape=(NUM_EXPERTS,),
            dtype=dtype,
        )
        s_total_tokens_post_pad = tkw.allocate(
            (1,), distributed_shape=(1,), dtype=dtype
        )
        tkw.write(zero_counts, shmem)

        expert_id = tkw.read(topk_ids, elements_per_thread=1)
        tkw.atomic_add(
            one_reg,
            shmem,
            mapping=expert_read_map,
            mapping_dynamic_vals=(expert_id,),
            elements_per_thread=1,
        )

        counts = tkw.read(
            shmem,
            mapping=expert_read_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        # write the histogram counts to global memory
        tkw.write(
            counts,
            expert_counts,
            mapping=expert_write_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        # Implement the padding logic
        block_size_reg = tkl.Register[NUM_EXPERTS, dtype](BLOCK_SIZE)

        # (count + block_size - 1) // block_size * block_size
        temp1 = counts + block_size_reg - one_reg
        temp2 = temp1 / block_size_reg
        padded_counts_reg = temp2 * block_size_reg

        tkw.write(
            padded_counts_reg,
            padded_counts,
            mapping=expert_write_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )
        padded_counts_reg = tkw.read(
            padded_counts,
            mapping=expert_read_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        prefix_sums = tkw.cumsum(padded_counts_reg, dim=NUM_EXPERTS)

        # write the inclusive scan results to global memory
        tkw.write(
            prefix_sums,
            cumsum_buffer,
            mapping=expert_write_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        # write the exclusive scan results to the shared memory
        tkw.write(
            prefix_sums,
            cumsum_buffer,
            mapping=shifted_write_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        # read the last element from the cumsum buffer to get total tokens after padding
        total_tokens_post_pad = tkw.read(
            cumsum_buffer,
            mapping=expert_read_map,
            mapping_dynamic_vals=(num_experts,),
            elements_per_thread=1,
        )

        num_blocks = total_tokens_post_pad / block_size_reg
        tkw.write(
            num_blocks,
            num_blocks_buffer,
            mapping=expert_write_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

    hyperparams = {
        NUM_TOKENS: num_tokens,
        NUM_EXPERTS: num_experts,
        NUMEL: num_tokens * top_k_value,
        BLOCK_TOKENS: min(64, num_tokens) if num_tokens > 0 else 1,
        BLOCK_EXPERTS: min(8, num_experts) if num_experts > 0 else 1,
        ELEMS_PER_THREAD: 4,
        BLOCK_SIZE: 16,
        TOPK: top_k_value,
    }
    hyperparams.update(get_default_scheduling_params())
    dynamic_symbols = []

    return moe_align_block_size, hyperparams, dynamic_symbols


# Writing our own version of GEMM kernel to support more datatypes
# We'll be improving this kernel with more operations too.
def get_gemm_kernel(
    m: int,
    k: int,
    n: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    assert datatype in [tkl.f16, tkl.bf16], f"Unsupported datatype: {datatype}"

    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.MFMA_INPUT_ELS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.MFMA_OUTPUT_ELS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, datatype],
        b: tkl.Memory[N, K, ADDRESS_SPACE, datatype],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)

            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
    }

    return gemm, hyperparams


def get_silu_and_mul_kernel(
    m: int,
    n: int,
    datatype: DataType,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(n, 256), wave_size)
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def silu_and_mul(
        x1: tkl.Memory[M, N, ADDRESS_SPACE, datatype],
        x2: tkl.Memory[M, N, ADDRESS_SPACE, datatype],
        out: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, datatype],
    ):
        x1_reg = tkw.read(x1)
        cst_m1 = tkl.Register[M, N, datatype](-1.0)
        cst_1 = tkl.Register[M, N, datatype](-1.0)
        exp_out = tkw.exp2(x1_reg * cst_m1)
        sigmoid = cst_1 / (cst_1 + exp_out)
        silu = sigmoid * x1_reg

        x2_reg = tkw.read(x2)
        res = silu * x2_reg

        tkw.write(res, out)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        M: m,
        N: n,
    }

    return silu_and_mul, hyperparams
