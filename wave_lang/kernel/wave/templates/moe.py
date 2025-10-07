# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel._support.dtype import f16, f32, i32
from wave_lang.kernel._support.indexing import sym
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel._support.dtype import DataType
import sympy
import torch

from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)


def get_fused_moe_gemm(
    m: int,
    n: int,
    k: int,
    e: int,
    block_shape: int,
    total_elems: int,
    num_experts: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    E = sym.E
    TOTAL_ELEMS = sym.TOTAL_ELEMS
    NUM_BLOCKS = sym.NUM_BLOCKS
    BLOCK_SHAPE = sym.BLOCK_SHAPE
    PAD_VALUE = sym.PAD_VALUE

    # Define workgroup tile sizes
    BLOCK_M = sym.BLOCK_M
    BLOCK_N = sym.BLOCK_N
    BLOCK_K = sym.BLOCK_K

    # Define the address space for our memory buffers
    ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
    ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
    ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C

    IDX = sym.IDX
    SCATTER_IDX = sym.SCATTER_IDX
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(TOTAL_ELEMS, BLOCK_SHAPE, 2),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.WaveConstraint(TOTAL_ELEMS, BLOCK_SHAPE),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={
                E: E,
                TOTAL_ELEMS: TOTAL_ELEMS,
                BLOCK_SHAPE: BLOCK_SHAPE,
                M: 16,
                N: 16,
                K: 16,
                NUM_BLOCKS: NUM_BLOCKS,
            },
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    e = tkw.IndexMapping.iterator(2)
    d0 = tkw.IndexMapping.dynamic_val(0)

    b_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: IDX, N: i, K: j},
        outputs={N: i, K: j},
    )

    a_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, K: j},
        outputs={M: i, K: j},
        dynamic_val_mappings={M: i},
    )

    a_back_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={NUM_BLOCKS: WORKGROUP_2, M: d0, K: j},
        dynamic_val_mappings={M: i},
    )

    a_back_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={NUM_BLOCKS: WORKGROUP_2, M: i, K: j},
        outputs={M: i, K: j},
    )

    c_back_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={NUM_BLOCKS: WORKGROUP_2, M: i, N: j},
    )

    c_back_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={NUM_BLOCKS: WORKGROUP_2, M: d0, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i},
    )

    c_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: d0, N: j},
        dynamic_val_mappings={M: i},
    )

    dyn_reorder_a_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={TOTAL_ELEMS: d0},
        outputs={TOTAL_ELEMS: i},
        dynamic_val_mappings={TOTAL_ELEMS: i},
    )

    expert_id_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_BLOCKS: d0},
        outputs={NUM_BLOCKS: i},
        dynamic_val_mappings={NUM_BLOCKS: i},
    )

    @tkw.wave(constraints)
    def fused_moe_gemm(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        reorder_a: Memory[TOTAL_ELEMS, ADDRESS_SPACE_A, i32],  # Input matrix A
        expert_ids: Memory[NUM_BLOCKS, ADDRESS_SPACE_A, i32],  # Input matrix A
        a_back: Memory[NUM_BLOCKS, M, K, ADDRESS_SPACE_A, f16],  # Output matrix A
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
        c_back: Memory[NUM_BLOCKS, M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
    ):
        # Initialize the accumulator register with zeros
        zero_reg = Register[M, K, f16](0.0)
        zero_reg_mn = Register[M, N, f32](0.0)
        tkw.write(zero_reg, a_back)
        tkw.write(zero_reg_mn, c_back)
        mock_reg = tkw.read(reorder_a)

        wid = tkw.scalar(WORKGROUP_2, i32)
        expert_id = tkw.read(
            expert_ids, mapping=expert_id_read_map, mapping_dynamic_vals=(wid,)
        )
        tkw.set_symbol(IDX, expert_id)
        condition = THREAD_0 < BLOCK_SHAPE

        @tkw.conditional(condition)
        def scatter_op():
            tid = tkw.Register[TOTAL_ELEMS, i32](THREAD_0)
            wid = tkw.Register[TOTAL_ELEMS, i32](WORKGROUP_2)
            tid_offset = tkw.Register[TOTAL_ELEMS, i32](BLOCK_SHAPE) * wid + tid
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid_offset,),
            )

            tkw.set_symbol(SCATTER_IDX, reordered_idx)
            is_not_padding = SCATTER_IDX < PAD_VALUE

            @tkw.conditional(is_not_padding)
            def then():
                @tkw.iterate(K, init_args=[])
                def copy_row():
                    a_row_data = tkw.read(
                        a,
                        mapping=a_read_map,
                        mapping_dynamic_vals=(reordered_idx,),
                        elements_per_thread=16,
                    )

                    tkw.write(
                        a_row_data,
                        a_back,
                        mapping=a_back_write_map,
                        mapping_dynamic_vals=(tid,),
                        elements_per_thread=16,
                    )

        tkw.workgroup_barrier()
        c_reg = Register[M, N, f32](0.0)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def gemm_compute(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a_back, mapping=a_back_read_map)
            b_reg = tkw.read(b, mapping=b_read_map)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(gemm_compute, c_back, mapping=c_back_write_map)

        @tkw.conditional(condition)
        def scatter_op():
            tid = tkw.Register[TOTAL_ELEMS, i32](THREAD_0)
            wid = tkw.Register[TOTAL_ELEMS, i32](WORKGROUP_2)
            tid_offset = tkw.Register[TOTAL_ELEMS, i32](BLOCK_SHAPE) * wid + tid
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid_offset,),
            )

            tkw.set_symbol(SCATTER_IDX, reordered_idx)
            is_not_padding = SCATTER_IDX < PAD_VALUE

            @tkw.conditional(is_not_padding)
            def then():
                c_row_data = tkw.read(
                    c_back,
                    mapping=c_back_read_map,
                    mapping_dynamic_vals=(tid,),
                    elements_per_thread=16,
                )
                tkw.write(
                    c_row_data,
                    c,
                    mapping=c_write_map,
                    mapping_dynamic_vals=(reordered_idx,),
                    elements_per_thread=16,
                )

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: num_experts,
        BLOCK_SHAPE: block_shape,
        TOTAL_ELEMS: total_elems,
        NUM_BLOCKS: (total_elems + block_shape - 1) // block_shape,
        PAD_VALUE: m,
    }

    return fused_moe_gemm, hyperparams


def get_moe_align_block_size_kernel(
    num_tokens: int,
    num_experts: int,
    block_size: int,
    numel: int,
    max_num_blocks: int,
    max_num_tokens_padded: int,
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
    MAX_NUM_BLOCKS = tkl.sym.MAX_NUM_BLOCKS
    MAX_NUM_TOKENS_PADDED = tkl.sym.MAX_NUM_TOKENS_PADDED

    I = sympy.Symbol("I")
    I_MAX = sympy.Symbol("I_MAX")

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
    constraints += [
        tkw.WorkgroupConstraint(MAX_NUM_TOKENS_PADDED, MAX_NUM_TOKENS_PADDED, 2)
    ]
    # one wave to handle the workload
    constraints += [tkw.WaveConstraint(NUMEL, NUMEL)]
    constraints += [tkw.WaveConstraint(NUM_EXPERTS, NUM_EXPERTS)]
    constraints += [tkw.WaveConstraint(MAX_NUM_TOKENS_PADDED, MAX_NUM_TOKENS_PADDED)]

    constraints += [tkw.TilingConstraint(I)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={
                NUMEL: NUMEL,
                NUM_EXPERTS: NUM_EXPERTS,
                MAX_NUM_BLOCKS: MAX_NUM_BLOCKS,
                MAX_NUM_TOKENS_PADDED: MAX_NUM_TOKENS_PADDED,
                I: 0,
                I_MAX: 0,
            },
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

    expert_id_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={MAX_NUM_BLOCKS: d0},
        outputs={MAX_NUM_BLOCKS: i},
        dynamic_val_mappings={MAX_NUM_BLOCKS: i},
    )

    expert_id_write_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={MAX_NUM_BLOCKS: i},
        outputs={MAX_NUM_BLOCKS: d0},
        dynamic_val_mappings={MAX_NUM_BLOCKS: i},
    )

    shifted_write_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: i},
        outputs={NUM_EXPERTS: d0 + 1},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    topk_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUMEL: d0},
        outputs={NUMEL: i},
        dynamic_val_mappings={NUMEL: i},
    )

    sorted_token_ids_write_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={MAX_NUM_TOKENS_PADDED: i},
        outputs={MAX_NUM_TOKENS_PADDED: d0},
        dynamic_val_mappings={MAX_NUM_TOKENS_PADDED: i},
    )

    @tkw.wave(constraints)
    def moe_align_block_size(
        topk_ids: tkl.Memory[NUMEL, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype],
        expert_ids: tkl.Memory[
            MAX_NUM_BLOCKS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
        expert_counts: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
        padded_counts: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
        cumsum_buffer: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
        cumsum_exclusive: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
        num_blocks_buffer: tkl.Memory[
            NUM_EXPERTS, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
        ],
        sorted_token_ids: tkl.Memory[
            MAX_NUM_TOKENS_PADDED, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, dtype
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
        # cumsum_exclusive = tkw.allocate(
        #     shape=(NUM_EXPERTS,),
        #     distributed_shape=(NUM_EXPERTS,),
        #     dtype=dtype,
        # )
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
            cumsum_exclusive,
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

        """
        if (threadIdx.x < num_experts) {
            for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
                expert_ids[i / block_size] = threadIdx.x - 1;
            }
        }
        """

        expert_start_pos = tkw.read(
            cumsum_exclusive,
            mapping=expert_read_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        # Read the inclusive cumsum (end position for each expert)
        expert_end_pos = tkw.read(
            cumsum_buffer,
            mapping=expert_read_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )
        tkw.set_symbol(I_MAX, expert_end_pos)

        # Calculate expert ID to write (threadIdx.x - 1)
        condition = (I < I_MAX) & (THREAD_0 < NUM_EXPERTS)

        @tkw.iterate(I, start=expert_start_pos, condition=condition, init_args=[])
        def loop():
            thread_id_x = tkw.Register[MAX_NUM_BLOCKS, tkl.i32](tkw.THREAD_0)
            i_idx = tkw.self_index(I, tkl.i32)
            expert_id_idx = i_idx / tkw.Register[I, tkl.i32](BLOCK_SIZE)
            tkw.write(
                thread_id_x,
                expert_ids,
                mapping=expert_id_write_map,
                mapping_dynamic_vals=(expert_id_idx,),
                elements_per_thread=1,
            )
            next_idx = i_idx + tkw.Register[I, tkl.i32](BLOCK_SIZE)
            tkw.set_symbol(I, next_idx)

        # now write the sorted token ids to global memory
        """
        Reference implementation:
        for (size_t i = tid; i < numel; i += stride) {
            int32_t expert_id = topk_ids[i] + 1;
            int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
            sorted_token_ids[rank_post_pad] = i;  // Store original token index
        }
        """

        numel_value = tkw.Register[MAX_NUM_TOKENS_PADDED, tkl.i32](NUMEL)
        tkw.write(numel_value, sorted_token_ids)

        tid_reg = tkw.Register[MAX_NUM_TOKENS_PADDED, tkl.i32](tkw.THREAD_0)
        expert_id = tkw.read(
            topk_ids,
            mapping=topk_read_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )
        rank_post_pad = tkw.atomic_add(
            one_reg,
            cumsum_exclusive,
            mapping=expert_read_map,
            mapping_dynamic_vals=(expert_id,),
            elements_per_thread=1,
        )
        tkw.write(
            tid_reg,
            sorted_token_ids,
            mapping=sorted_token_ids_write_map,
            mapping_dynamic_vals=(rank_post_pad,),
            elements_per_thread=1,
        )

    hyperparams = {
        NUM_TOKENS: num_tokens,
        NUM_EXPERTS: num_experts,
        NUMEL: numel,
        MAX_NUM_BLOCKS: max_num_blocks,
        MAX_NUM_TOKENS_PADDED: max_num_tokens_padded,
        BLOCK_TOKENS: min(64, num_tokens) if num_tokens > 0 else 1,
        BLOCK_EXPERTS: min(8, num_experts) if num_experts > 0 else 1,
        ELEMS_PER_THREAD: 4,
        BLOCK_SIZE: block_size,
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
        cst_1 = tkl.Register[M, N, datatype](1.0)
        exp_out = tkw.exp(x1_reg * cst_m1)
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


def get_moe_reduce_sum_kernel(
    m: int,
    n: int,
    datatype: DataType,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def moe_reduce_sum(
        a: tkl.Memory[M, N, ADDRESS_SPACE, datatype],
        c: tkl.Memory[M, ADDRESS_SPACE, datatype],
    ):
        res = tkw.read(a)
        res = tkw.sum(res, dim=N)
        tkw.write(res, c)

    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        M: m,
        N: n,
    }

    return moe_reduce_sum, hyperparams
