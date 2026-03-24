# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MoE (Mixture of Experts) pipeline — Wave GPU kernel implementation.

All pipeline stages (alignment, gather, GEMM, scatter, activation, reduce)
are implemented as Wave kernels.
"""

import torch
import sympy
from typing import Any, NamedTuple

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel._support.dtype import DataType, f16, f32, i32
from wave_lang.kernel._support.indexing import sym
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.support.indexing import IndexSymbol

# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false


# ---------------------------------------------------------------------------
# TopK kernel
# ---------------------------------------------------------------------------
def get_topk_kernel(
    m: int,
    n: int,
    k: int,
    datatype: DataType,
    threads_per_wave: int = 32,
):
    """
    Wave kernel for computing top-k values and indices.

    Args:
        m: Number of rows (tokens)
        n: Number of columns (experts)
        k: Number of top elements to select
        datatype: Data type for input values
        threads_per_wave: Number of threads per wave (default 32 - RDNA4)
    """
    # Input sizes
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
    def topk_kernel(
        a: tkl.Memory[M, N, ADDRESS_SPACE, datatype],
        values: tkl.Memory[M, K, ADDRESS_SPACE, datatype],
        indices: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i32],
    ):
        src = tkw.read(a)
        topk_values, topk_indices = tkw.topk(src, K, N)
        tkw.write(topk_values, values, elements_per_thread=K)
        tkw.write(topk_indices, indices, elements_per_thread=K)

    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        M: m,
        N: n,
        K: k,
    }

    return topk_kernel, hyperparams


# ---------------------------------------------------------------------------
# SiLU & Mul activation kernel
# ---------------------------------------------------------------------------
def get_silu_and_mul_kernel(
    m: int,
    n: int,
    datatype: DataType,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    TWO_N = tkl.sym.TWO_N

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 32, each thread is operating on up to 8
    # elements.
    wave_size = 32
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = 32
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
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

    # Create index mappings to read gate (first half) and up (second half)
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    x1_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, TWO_N: k},
        outputs={M: i, N: j},
        dynamic_val_mappings={TWO_N: j},
    )

    x2_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, TWO_N: k + N},
        outputs={M: i, N: j},
        dynamic_val_mappings={TWO_N: j},
    )

    @tkw.wave(constraints)
    def silu_and_mul(
        gemm1_out: tkl.Memory[M, TWO_N, ADDRESS_SPACE, datatype],
        out: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, datatype],
    ):
        # Read x1 (first half: columns 0 to N-1)
        # Compute global thread ID accounting for workgroup offset
        tid = tkw.scalar(THREAD_0 + WORKGROUP_0 * wave_size, tkl.i32)
        x1_reg = tkw.read(gemm1_out, mapping=x1_read_map, mapping_dynamic_vals=(tid,))
        cst_m1 = tkw.Register[M, N, datatype](-1.0)
        cst_1 = tkw.Register[M, N, datatype](1.0)
        exp_out = tkw.exp(x1_reg * cst_m1)
        sigmoid = cst_1 / (cst_1 + exp_out)
        silu = sigmoid * x1_reg
        x2_reg = tkw.read(gemm1_out, mapping=x2_read_map, mapping_dynamic_vals=(tid,))
        res = silu * x2_reg
        tkw.write(res, out)

    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        M: m,
        N: n,
        TWO_N: 2 * n,
    }

    return silu_and_mul, hyperparams


# ---------------------------------------------------------------------------
# Reduce sum kernel
# ---------------------------------------------------------------------------
def get_moe_reduce_sum_kernel(
    b_size: int,
    k_size: int,
    d_size: int,
    datatype: DataType,
):
    # Input sizes
    B = tkl.sym.B
    K = tkl.sym.K
    D = tkl.sym.D
    # Iterator symbols for source/target transpose
    b = tkl.sym.b
    k = tkl.sym.k
    d = tkl.sym.d
    wave_size = 32
    BLOCK_B = 1
    BLOCK_K = sympy.ceiling(K / wave_size) * wave_size
    BLOCK_D = 1
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Register layout is [B, D, K] so K is the fast dimension for reduction
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=32,
            vector_shapes={B: BLOCK_B, D: BLOCK_D, K: BLOCK_K},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(D, BLOCK_D, 1)]
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 0)]
    constraints += [tkw.WaveConstraint(B, BLOCK_B)]
    constraints += [tkw.WaveConstraint(D, BLOCK_D)]
    constraints += [tkw.WaveConstraint(K, BLOCK_K)]
    # Iterator bindings for source/target transpose
    constraints += [tkw.IteratorBindings({b: B, k: K, d: D})]

    @tkw.wave(constraints)
    def moe_reduce_sum(
        a: tkl.Memory[B, K, D, ADDRESS_SPACE, datatype],
        weights: tkl.Memory[B, K, ADDRESS_SPACE, datatype],
        c: tkl.Memory[B, D, ADDRESS_SPACE, datatype],
    ):
        gemm2_out = tkw.read(a, source=(b, k, d), target=(b, d, k))
        topk_weights = tkw.read(weights, source=(b, k), target=(b, k))
        topk_weights_broadcasted = tkw.broadcast(topk_weights, [B, D, K])
        res = gemm2_out * topk_weights_broadcasted
        res = tkw.sum(res, dim=K)
        tkw.write(res, c, source=(b, d), target=(b, d))

    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        B: b_size,
        K: k_size,
        D: d_size,
    }

    return moe_reduce_sum, hyperparams


# ---------------------------------------------------------------------------
# MoE GEMM-only kernel (no gather/scatter — avoids cross-WG1 race cond.)
# ---------------------------------------------------------------------------
def get_moe_gemm_only_kernel(
    m: int,
    n: int,
    k: int,
    e: int,
    num_blocks: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    """
    Wave GEMM kernel for MoE: operates on pre-gathered a_back buffer.

    Gather/scatter must be done externally (e.g., via moe_gather/moe_scatter).

    Args:
        m: Total token count (num_tokens * topk).
        n: Output dimension per expert.
        k: Input/reduction dimension.
        e: Number of experts.
        num_blocks: Total number of expert blocks.
        mfma_variant: MMA instruction type.
        datatype: Input data type (f16 or bf16).
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    E = sym.E
    NUM_BLOCKS = sym.NUM_BLOCKS
    IDX = sym.IDX

    BLOCK_M = sym.BLOCK_M
    BLOCK_N = sym.BLOCK_N
    BLOCK_K = sym.BLOCK_K

    ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
    ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
    ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(NUM_BLOCKS, 1, 2),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.WaveConstraint(NUM_BLOCKS, 1),
        tkw.HardwareConstraint(
            threads_per_wave=32,
            mma_type=mfma_variant,
            vector_shapes={
                E: E,
                M: 16,
                N: 16,
                K: 16,
                NUM_BLOCKS: 1,
            },
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    d0 = tkw.IndexMapping.dynamic_val(0)

    b_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: IDX, N: i, K: j},
        outputs={N: i, K: j},
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

    expert_id_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_BLOCKS: d0},
        outputs={NUM_BLOCKS: i},
        dynamic_val_mappings={NUM_BLOCKS: i},
    )

    @tkw.wave(constraints)
    def moe_gemm_only(
        a_back: Memory[NUM_BLOCKS, M, K, ADDRESS_SPACE_A, f16],
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],
        expert_ids: Memory[NUM_BLOCKS, ADDRESS_SPACE_A, i32],
        c_back: Memory[NUM_BLOCKS, M, N, ADDRESS_SPACE_C, f32],
    ):
        wid = tkw.scalar(WORKGROUP_2, i32)
        expert_id = tkw.read(
            expert_ids, mapping=expert_id_read_map, mapping_dynamic_vals=(wid,)
        )
        tkw.set_symbol(IDX, expert_id)

        c_reg = Register[M, N, f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def gemm_compute(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            a_reg = tkw.read(a_back, mapping=a_back_read_map)
            b_reg = tkw.read(b, mapping=b_read_map)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(gemm_compute, c_back, mapping=c_back_write_map)

    block_m = min(m, 64)
    # BLOCK_N=128 gives ~1.7x speedup over 32 on RDNA4 by doubling N-tile
    # reuse per wave. Cap at n to avoid overflowing the N dimension.
    block_n = min(n, 128)
    hyperparams: dict[str | IndexSymbol, Any] = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: e,
        NUM_BLOCKS: num_blocks,
    }

    return moe_gemm_only, hyperparams


# ---------------------------------------------------------------------------
# Step 1: Histogram — Wave kernel
# ---------------------------------------------------------------------------
def get_moe_histogram_kernel(
    numel: int,
    num_experts: int,
    threads_per_wave: int = 32,
):
    """
    Wave kernel: count tokens per expert (histogram of topk_ids).

    Each thread reads one topk_id and atomically increments the corresponding
    expert count in global memory. Multiple workgroups handle large numel.

    Note: requires a dummy output buffer because the compiler needs a
    tkw.write leaf operation — atomic_add alone is not recognized as a leaf.
    """
    NUMEL = tkl.sym.NUMEL
    NUM_EXPERTS = tkl.sym.NUM_EXPERTS
    HIST_BLOCK = sym.HIST_BLOCK

    constraints = [
        tkw.WorkgroupConstraint(NUMEL, HIST_BLOCK, 0),
        tkw.WaveConstraint(NUMEL, HIST_BLOCK),
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            waves_per_block=(1, 1, 1),
            vector_shapes={NUMEL: HIST_BLOCK, NUM_EXPERTS: NUM_EXPERTS},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    scatter_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: d0},
        outputs={NUM_EXPERTS: i},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    @tkw.wave(constraints)
    def histogram(
        topk_ids: Memory[NUMEL, GLOBAL_ADDRESS_SPACE, tkl.i32],
        expert_counts: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, tkl.i32],
        dummy: Memory[NUMEL, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        expert_id = tkw.read(topk_ids, elements_per_thread=1)
        one = Register[NUM_EXPERTS, tkl.i32](1)
        tkw.atomic_add(
            one,
            expert_counts,
            mapping=scatter_map,
            mapping_dynamic_vals=(expert_id,),
            elements_per_thread=1,
        )
        # Leaf write required by compiler (atomic_add is not a recognized leaf)
        tkw.write(expert_id, dummy, elements_per_thread=1)

    hyperparams = {
        NUMEL: numel,
        NUM_EXPERTS: num_experts,
        HIST_BLOCK: threads_per_wave,
    }

    return histogram, hyperparams


# ---------------------------------------------------------------------------
# Steps 2+3: Pad counts + Prefix sum — Wave kernel
# ---------------------------------------------------------------------------
def get_moe_pad_cumsum_kernel(num_experts: int, block_size: int):
    """
    Wave kernel for Steps 2+3 of MoE alignment:
      Step 2: Pad each expert's token count up to a multiple of block_size.
      Step 3: Compute inclusive and exclusive prefix sums of padded counts.

    Dispatches a single wave of 32 threads. Each of the first num_experts
    threads handles one expert; remaining threads read/write OOB (harmlessly
    ignored on GPU). The cumsum uses a butterfly-shuffle scan across the wave.

    Args:
        num_experts: Number of experts (4-8 in tests).
        block_size: Block alignment size.
    """
    NUM_EXPERTS = tkl.sym.NUM_EXPERTS
    BLOCK_SZ = sym.BLOCK_SZ

    constraints = [
        tkw.WorkgroupConstraint(NUM_EXPERTS, NUM_EXPERTS, 0),
        tkw.WaveConstraint(NUM_EXPERTS, NUM_EXPERTS),
        tkw.HardwareConstraint(
            threads_per_wave=32,
            waves_per_block=(1, 1, 1),
            vector_shapes={NUM_EXPERTS: NUM_EXPERTS},
        ),
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

    @tkw.wave(constraints)
    def pad_cumsum_kernel(
        expert_counts: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, i32],
        padded_counts_out: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, i32],
        cumsum_out: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, i32],
        cumsum_exclusive_out: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, i32],
        dummy: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, i32],
    ):
        tid = tkw.scalar(THREAD_0, i32)

        # Read this expert's raw count
        count = tkw.read(
            expert_counts,
            mapping=expert_read_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        # Step 2: Pad — ceil_div(count, block_size) * block_size
        bs = tkw.Register[NUM_EXPERTS, i32](BLOCK_SZ)
        one = tkw.Register[NUM_EXPERTS, i32](1)
        padded = ((count + bs - one) / bs) * bs

        tkw.write(
            padded,
            padded_counts_out,
            mapping=expert_write_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        # Step 3: Inclusive prefix sum (butterfly scan across wave)
        prefix_sum = tkw.cumsum(padded, dim=NUM_EXPERTS)

        tkw.write(
            prefix_sum,
            cumsum_out,
            mapping=expert_write_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        # Exclusive prefix sum = inclusive - current element
        exclusive = prefix_sum - padded
        tkw.write(
            exclusive,
            cumsum_exclusive_out,
            mapping=expert_write_map,
            mapping_dynamic_vals=(tid,),
            elements_per_thread=1,
        )

        # Leaf write
        tkw.write(count, dummy, elements_per_thread=1)

    hyperparams = {
        NUM_EXPERTS: num_experts,
        BLOCK_SZ: block_size,
    }

    return pad_cumsum_kernel, hyperparams


# ---------------------------------------------------------------------------
# Step 4: Expert IDs fill — Wave kernel (one workgroup per expert)
# ---------------------------------------------------------------------------
def get_moe_expert_ids_kernel(
    num_experts: int,
    max_num_blocks: int,
    block_size: int,
):
    """
    Wave kernel for Step 4 of MoE alignment: fill expert_ids so that each
    block knows which expert owns it.

    Dispatches one workgroup per expert. Each workgroup reads its cumsum range
    and iterates over its block range, writing its expert index to expert_ids.

    Args:
        num_experts: Number of experts (4-8).
        max_num_blocks: Maximum number of blocks (ceil(total_padded_tokens / block_size)).
        block_size: Alignment block size.
    """
    NUM_EXPERTS = tkl.sym.NUM_EXPERTS
    MAX_NUM_BLOCKS = tkl.sym.MAX_NUM_BLOCKS
    BLOCK_SZ = sym.BLOCK_SZ

    I = sym.I
    I_MAX = sym.I_MAX

    constraints = [
        tkw.WorkgroupConstraint(NUM_EXPERTS, 1, 0),
        tkw.WaveConstraint(NUM_EXPERTS, 1),
        tkw.TilingConstraint(I),
        tkw.HardwareConstraint(
            threads_per_wave=32,
            waves_per_block=(1, 1, 1),
            vector_shapes={
                NUM_EXPERTS: 1,
                MAX_NUM_BLOCKS: MAX_NUM_BLOCKS,
                I: 0,
                I_MAX: 0,
            },
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    expert_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: d0},
        outputs={NUM_EXPERTS: i},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    expert_id_write_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={MAX_NUM_BLOCKS: i},
        outputs={MAX_NUM_BLOCKS: d0},
        dynamic_val_mappings={MAX_NUM_BLOCKS: i},
    )

    @tkw.wave(constraints)
    def expert_ids_kernel(
        cumsum_exclusive: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, i32],
        cumsum: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, i32],
        expert_ids: Memory[MAX_NUM_BLOCKS, GLOBAL_ADDRESS_SPACE, i32],
        dummy: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, i32],
    ):
        # This workgroup's expert index
        wid = tkw.scalar(WORKGROUP_0, i32)

        # Read this expert's start and end positions (token-space)
        start_pos = tkw.read(
            cumsum_exclusive,
            mapping=expert_read_map,
            mapping_dynamic_vals=(wid,),
            elements_per_thread=1,
        )
        end_pos = tkw.read(
            cumsum,
            mapping=expert_read_map,
            mapping_dynamic_vals=(wid,),
            elements_per_thread=1,
        )

        # Set I_MAX for the iterate condition — wave-uniform is correct here
        # because each workgroup handles exactly one expert.
        tkw.set_symbol(I_MAX, end_pos)

        condition = I < I_MAX

        # Iterate from start_pos to end_pos in steps determined by TilingConstraint(I)
        # Each step: write this expert's index to expert_ids[block_idx]
        @tkw.iterate(I, start=start_pos, condition=condition, init_args=[])
        def fill_loop():
            i_idx = tkw.self_index(I, i32)
            block_idx = i_idx / tkw.Register[I, i32](BLOCK_SZ)
            expert_id_val = tkw.Register[MAX_NUM_BLOCKS, i32](WORKGROUP_0)
            tkw.write(
                expert_id_val,
                expert_ids,
                mapping=expert_id_write_map,
                mapping_dynamic_vals=(block_idx,),
                elements_per_thread=1,
            )
            # Advance induction variable by block_size
            next_idx = i_idx + tkw.Register[I, i32](BLOCK_SZ)
            tkw.set_symbol(I, next_idx)

        # Leaf write
        dummy_val = tkw.read(cumsum_exclusive, elements_per_thread=1)
        tkw.write(dummy_val, dummy, elements_per_thread=1)

    hyperparams = {
        NUM_EXPERTS: num_experts,
        MAX_NUM_BLOCKS: max_num_blocks,
        BLOCK_SZ: block_size,
    }

    return expert_ids_kernel, hyperparams


# ---------------------------------------------------------------------------
# Step 5: Sorted IDs scatter — Wave kernel
# ---------------------------------------------------------------------------
def get_moe_sorted_ids_kernel(
    numel: int,
    num_experts: int,
    max_num_tokens_padded: int,
    threads_per_wave: int = 32,
):
    """
    Wave kernel for Step 5 of MoE alignment: scatter token indices into
    sorted order by expert.

    Each thread handles one token: reads its expert assignment from topk_ids,
    atomically claims a write position in write_pos[expert], and writes its
    token index to sorted_ids[pos].

    The caller must pre-initialize:
      - sorted_ids[:] = numel  (padding sentinel)
      - write_pos[:] = cumsum_exclusive[:]  (starting write positions per expert)

    Args:
        numel: Number of tokens (num_tokens * topk).
        num_experts: Number of experts.
        max_num_tokens_padded: Size of sorted_ids output buffer.
        threads_per_wave: Threads per wave (32 for RDNA4).
    """
    NUMEL = tkl.sym.NUMEL
    NUM_EXPERTS = tkl.sym.NUM_EXPERTS
    MAX_TOKENS_PADDED = sym.MAX_TOKENS_PADDED
    SCATTER_BLOCK = sym.SCATTER_BLOCK

    constraints = [
        tkw.WorkgroupConstraint(NUMEL, SCATTER_BLOCK, 0),
        tkw.WaveConstraint(NUMEL, SCATTER_BLOCK),
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            waves_per_block=(1, 1, 1),
            vector_shapes={
                NUMEL: SCATTER_BLOCK,
                NUM_EXPERTS: NUM_EXPERTS,
                MAX_TOKENS_PADDED: MAX_TOKENS_PADDED,
            },
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    d0 = tkw.IndexMapping.dynamic_val(0)

    # Map for atomic_add on write_pos[expert_id]
    expert_scatter_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_EXPERTS: d0},
        outputs={NUM_EXPERTS: i},
        dynamic_val_mappings={NUM_EXPERTS: i},
    )

    # Map for writing token_idx to sorted_ids[pos]
    sorted_write_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={MAX_TOKENS_PADDED: i},
        outputs={MAX_TOKENS_PADDED: d0},
        dynamic_val_mappings={MAX_TOKENS_PADDED: i},
    )

    @tkw.wave(constraints)
    def sorted_ids_kernel(
        topk_ids: Memory[NUMEL, GLOBAL_ADDRESS_SPACE, i32],
        write_pos: Memory[NUM_EXPERTS, GLOBAL_ADDRESS_SPACE, i32],
        sorted_ids: Memory[MAX_TOKENS_PADDED, GLOBAL_ADDRESS_SPACE, i32],
        dummy: Memory[NUMEL, GLOBAL_ADDRESS_SPACE, i32],
    ):
        # Read this token's expert assignment
        expert_id = tkw.read(topk_ids, elements_per_thread=1)

        # Atomically claim a write position in write_pos[expert_id]
        one = tkw.Register[NUM_EXPERTS, i32](1)
        pos = tkw.atomic_add(
            one,
            write_pos,
            mapping=expert_scatter_map,
            mapping_dynamic_vals=(expert_id,),
            elements_per_thread=1,
        )

        # Compute this thread's global token index
        tid = tkw.Register[MAX_TOKENS_PADDED, i32](THREAD_0)
        wid = tkw.Register[MAX_TOKENS_PADDED, i32](WORKGROUP_0)
        token_idx = wid * tkw.Register[MAX_TOKENS_PADDED, i32](SCATTER_BLOCK) + tid

        # Write token_idx to sorted_ids[pos]
        tkw.write(
            token_idx,
            sorted_ids,
            mapping=sorted_write_map,
            mapping_dynamic_vals=(pos,),
            elements_per_thread=1,
        )

        # Leaf write (atomic_add alone is not a recognized leaf)
        tkw.write(expert_id, dummy, elements_per_thread=1)

    hyperparams = {
        NUMEL: numel,
        NUM_EXPERTS: num_experts,
        MAX_TOKENS_PADDED: max_num_tokens_padded,
        SCATTER_BLOCK: threads_per_wave,
    }

    return sorted_ids_kernel, hyperparams


# ---------------------------------------------------------------------------
# MoE Gather kernel
# ---------------------------------------------------------------------------
def get_moe_gather_kernel(
    m: int,
    k: int,
    num_blocks: int,
    block_size: int,
    pad_value: int,
):
    """
    Wave kernel for MoE gather: copies input token rows into per-block scratch buffer.

    For each block, copies block_size rows from a[] at positions given by sorted_ids[]
    into a_back[block, :block_size, :]. Padding slots (sorted_ids >= pad_value) are
    skipped — the caller must zero-initialize a_back before calling this kernel.

    Args:
        m: Total token count (num_tokens * topk).
        k: Hidden dimension.
        num_blocks: Number of expert blocks.
        block_size: Slots per block.
        pad_value: Sentinel value in sorted_ids (= m) indicating padding.
    """
    M = tkl.sym.M
    K = tkl.sym.K
    NUM_BLOCKS = sym.NUM_BLOCKS
    BLOCK_SHAPE = sym.BLOCK_SHAPE
    PAD_VALUE = sym.PAD_VALUE
    TOTAL_ELEMS = sym.TOTAL_ELEMS
    SCATTER_IDX = sym.SCATTER_IDX
    BLOCK_K = sym.BLOCK_K

    total_elems = num_blocks * block_size

    constraints = [
        tkw.WorkgroupConstraint(TOTAL_ELEMS, BLOCK_SHAPE, 0),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(TOTAL_ELEMS, BLOCK_SHAPE),
        tkw.HardwareConstraint(
            threads_per_wave=32,
            waves_per_block=(1, 1, 1),
            vector_shapes={
                TOTAL_ELEMS: TOTAL_ELEMS,
                BLOCK_SHAPE: BLOCK_SHAPE,
                NUM_BLOCKS: NUM_BLOCKS,
                M: M,
                K: 16,
            },
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    d0 = tkw.IndexMapping.dynamic_val(0)

    sorted_ids_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={TOTAL_ELEMS: d0},
        outputs={TOTAL_ELEMS: i},
        dynamic_val_mappings={TOTAL_ELEMS: i},
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
        outputs={NUM_BLOCKS: WORKGROUP_0, M: d0, K: j},
        dynamic_val_mappings={M: i},
    )

    @tkw.wave(constraints)
    def moe_gather_kernel(
        sorted_ids: Memory[TOTAL_ELEMS, GLOBAL_ADDRESS_SPACE, i32],
        a: Memory[M, K, GLOBAL_ADDRESS_SPACE, f16],
        a_back: Memory[NUM_BLOCKS, M, K, GLOBAL_ADDRESS_SPACE, f16],
        dummy: Memory[TOTAL_ELEMS, GLOBAL_ADDRESS_SPACE, i32],
    ):
        condition = THREAD_0 < BLOCK_SHAPE

        @tkw.conditional(condition)
        def gather_op():
            tid = tkw.Register[TOTAL_ELEMS, i32](THREAD_0)
            wid = tkw.Register[TOTAL_ELEMS, i32](WORKGROUP_0)
            tid_offset = tkw.Register[TOTAL_ELEMS, i32](BLOCK_SHAPE) * wid + tid

            token_idx = tkw.read(
                sorted_ids,
                mapping=sorted_ids_read_map,
                mapping_dynamic_vals=(tid_offset,),
            )

            tkw.set_symbol(SCATTER_IDX, token_idx)
            is_not_padding = SCATTER_IDX < PAD_VALUE

            @tkw.conditional(is_not_padding)
            def copy_valid():
                @tkw.iterate(K, init_args=[])
                def copy_row():
                    a_row = tkw.read(
                        a,
                        mapping=a_read_map,
                        mapping_dynamic_vals=(token_idx,),
                        elements_per_thread=16,
                    )
                    tkw.write(
                        a_row,
                        a_back,
                        mapping=a_back_write_map,
                        mapping_dynamic_vals=(tid,),
                        elements_per_thread=16,
                    )

        # Leaf write required by compiler
        dummy_val = tkw.read(sorted_ids, elements_per_thread=1)
        tkw.write(dummy_val, dummy, elements_per_thread=1)

    hyperparams = {
        TOTAL_ELEMS: total_elems,
        BLOCK_SHAPE: block_size,
        NUM_BLOCKS: num_blocks,
        M: m,
        K: k,
        PAD_VALUE: pad_value,
        BLOCK_K: 32,
    }

    return moe_gather_kernel, hyperparams


# ---------------------------------------------------------------------------
# MoE Scatter kernel
# ---------------------------------------------------------------------------
def get_moe_scatter_kernel(
    m: int,
    k: int,
    num_blocks: int,
    block_size: int,
    pad_value: int,
):
    """
    Wave kernel for MoE scatter: copies per-block GEMM results back to token positions.

    For each block, copies block_size rows from c_back[block, :block_size, :] to
    output[token_idx, :] where token_idx comes from sorted_ids. Padding slots
    (sorted_ids >= pad_value) are skipped.

    Args:
        m: Total token count (num_tokens * topk).
        k: Output hidden dimension.
        num_blocks: Number of expert blocks.
        block_size: Slots per block.
        pad_value: Sentinel value in sorted_ids (= m) indicating padding.
    """
    M = tkl.sym.M
    K = tkl.sym.K
    NUM_BLOCKS = sym.NUM_BLOCKS
    BLOCK_SHAPE = sym.BLOCK_SHAPE
    PAD_VALUE = sym.PAD_VALUE
    TOTAL_ELEMS = sym.TOTAL_ELEMS
    SCATTER_IDX = sym.SCATTER_IDX
    BLOCK_K = sym.BLOCK_K

    total_elems = num_blocks * block_size

    constraints = [
        tkw.WorkgroupConstraint(TOTAL_ELEMS, BLOCK_SHAPE, 0),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(TOTAL_ELEMS, BLOCK_SHAPE),
        tkw.HardwareConstraint(
            threads_per_wave=32,
            waves_per_block=(1, 1, 1),
            vector_shapes={
                TOTAL_ELEMS: TOTAL_ELEMS,
                BLOCK_SHAPE: BLOCK_SHAPE,
                NUM_BLOCKS: NUM_BLOCKS,
                M: M,
                K: 16,
            },
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    d0 = tkw.IndexMapping.dynamic_val(0)

    sorted_ids_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={TOTAL_ELEMS: d0},
        outputs={TOTAL_ELEMS: i},
        dynamic_val_mappings={TOTAL_ELEMS: i},
    )

    c_back_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={NUM_BLOCKS: WORKGROUP_0, M: d0, K: j},
        outputs={M: i, K: j},
        dynamic_val_mappings={M: i},
    )

    output_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={M: d0, K: j},
        dynamic_val_mappings={M: i},
    )

    @tkw.wave(constraints)
    def moe_scatter_kernel(
        sorted_ids: Memory[TOTAL_ELEMS, GLOBAL_ADDRESS_SPACE, i32],
        c_back: Memory[NUM_BLOCKS, M, K, GLOBAL_ADDRESS_SPACE, f32],
        output: Memory[M, K, GLOBAL_ADDRESS_SPACE, f32],
        dummy: Memory[TOTAL_ELEMS, GLOBAL_ADDRESS_SPACE, i32],
    ):
        condition = THREAD_0 < BLOCK_SHAPE

        @tkw.conditional(condition)
        def scatter_op():
            tid = tkw.Register[TOTAL_ELEMS, i32](THREAD_0)
            wid = tkw.Register[TOTAL_ELEMS, i32](WORKGROUP_0)
            tid_offset = tkw.Register[TOTAL_ELEMS, i32](BLOCK_SHAPE) * wid + tid

            token_idx = tkw.read(
                sorted_ids,
                mapping=sorted_ids_read_map,
                mapping_dynamic_vals=(tid_offset,),
            )

            tkw.set_symbol(SCATTER_IDX, token_idx)
            is_not_padding = SCATTER_IDX < PAD_VALUE

            @tkw.conditional(is_not_padding)
            def copy_valid():
                @tkw.iterate(K, init_args=[])
                def copy_row():
                    c_row = tkw.read(
                        c_back,
                        mapping=c_back_read_map,
                        mapping_dynamic_vals=(tid,),
                        elements_per_thread=16,
                    )
                    tkw.write(
                        c_row,
                        output,
                        mapping=output_write_map,
                        mapping_dynamic_vals=(token_idx,),
                        elements_per_thread=16,
                    )

        # Leaf write required by compiler
        dummy_val = tkw.read(sorted_ids, elements_per_thread=1)
        tkw.write(dummy_val, dummy, elements_per_thread=1)

    hyperparams = {
        TOTAL_ELEMS: total_elems,
        BLOCK_SHAPE: block_size,
        NUM_BLOCKS: num_blocks,
        M: m,
        K: k,
        PAD_VALUE: pad_value,
        BLOCK_K: 32,
    }

    return moe_scatter_kernel, hyperparams


# ---------------------------------------------------------------------------
# Fused Scatter1 + SiLU + Gather2 kernel (in block/slot space)
# ---------------------------------------------------------------------------
def get_fused_silu_block_space_kernel(
    num_blocks: int,
    m: int,
    n: int,
    datatype: DataType,
):
    """
    Fused kernel replacing Scatter1 + SiLU&Mul + Gather2.

    Operates entirely in block/slot space, eliminating the round-trip through
    token space (scatter → token space → gather).  Because scatter and gather
    use the same sorted_ids mapping, applying the computation directly in
    block/slot space gives identical results with fewer kernel launches and
    no large intermediate buffers (gemm1_out and silu_and_mul_out).

    Given:
      c_back  : [NUM_BLOCKS, M, TWO_N]  -- GEMM1 output (float32, gate+up)
      a_out   : [NUM_BLOCKS, M, N]      -- output buffer for GEMM2 (float16)

    For each (block, slot):
      gate = c_back[block, slot, :n]
      up   = c_back[block, slot, n:2n]
      silu = sigmoid(gate) * gate * up
      a_out[block, slot, :n] = silu.to(f16)

    Args:
        num_blocks: Total number of expert blocks.
        m: Per-block slot count (typically m = num_tokens * topk; only
           the first block_size slots per block are valid but the full
           m-sized buffer is used by the GEMM kernel).
        n: Output dimension per expert (half of w1.shape[1]).
        datatype: Input data type (f32 — GEMM1 writes f32 to c_back).
    """
    M = tkl.sym.M
    N = tkl.sym.N
    TWO_N = tkl.sym.TWO_N
    NUM_BLOCKS = sym.NUM_BLOCKS

    wave_size = 32
    BLOCK_M = 1
    BLOCK_N = 32
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N, NUM_BLOCKS: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(NUM_BLOCKS, 1, 2)]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(NUM_BLOCKS, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)

    # Read gate portion: c_back[block, row, col] where col = tid (0..n-1)
    x1_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={NUM_BLOCKS: WORKGROUP_2, M: i, TWO_N: k},
        outputs={M: i, N: j},
        dynamic_val_mappings={TWO_N: j},
    )

    # Read up portion: c_back[block, row, col+n] where col = tid (n..2n-1)
    x2_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={NUM_BLOCKS: WORKGROUP_2, M: i, TWO_N: k + N},
        outputs={M: i, N: j},
        dynamic_val_mappings={TWO_N: j},
    )

    # Write: a_out[block, row, col]
    a_out_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={NUM_BLOCKS: WORKGROUP_2, M: i, N: j},
    )

    @tkw.wave(constraints)
    def fused_silu_block_space(
        c_back: tkl.Memory[NUM_BLOCKS, M, TWO_N, ADDRESS_SPACE, datatype],
        a_out: tkl.Memory[NUM_BLOCKS, M, N, GLOBAL_ADDRESS_SPACE, f16],
    ):
        # Global column index for this thread
        tid = tkw.scalar(THREAD_0 + WORKGROUP_0 * wave_size, tkl.i32)

        # Read gate (first half of 2n)
        x1_reg = tkw.read(c_back, mapping=x1_read_map, mapping_dynamic_vals=(tid,))

        # SiLU: sigmoid(x) * x
        cst_m1 = tkw.Register[M, N, datatype](-1.0)
        cst_1 = tkw.Register[M, N, datatype](1.0)
        exp_out = tkw.exp(x1_reg * cst_m1)
        sigmoid = cst_1 / (cst_1 + exp_out)
        silu = sigmoid * x1_reg

        # Read up (second half of 2n) and multiply
        x2_reg = tkw.read(c_back, mapping=x2_read_map, mapping_dynamic_vals=(tid,))
        res = silu * x2_reg

        # Write result cast to f16 into a_out
        tkw.write(res, a_out, mapping=a_out_write_map)

    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        NUM_BLOCKS: num_blocks,
        M: m,
        N: n,
        TWO_N: 2 * n,
    }

    return fused_silu_block_space, hyperparams


# ---------------------------------------------------------------------------
# Helper: compile a Wave kernel with default settings
# ---------------------------------------------------------------------------
def _compile_wave_kernel(kernel_fn, hyperparams, **compile_kwargs):
    """Compile a Wave kernel with default scheduling params and run config."""
    hp = dict(hyperparams)
    hp.update(get_default_scheduling_params())
    options = WaveCompileOptions(subs=hp, **compile_kwargs)
    options = set_default_run_config(options)
    return wave_compile(options, kernel_fn)


class MoEAlignKernels(NamedTuple):
    """Pre-compiled kernels for moe_align_block_size."""

    histogram: Any
    pad_cumsum: Any
    expert_ids: Any
    sorted_ids: Any


def compile_moe_align_kernels(
    numel: int,
    num_experts: int,
    block_size: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
) -> MoEAlignKernels:
    """Compile all 4 alignment kernels once. Pass the result to moe_align_block_size()."""
    hist_fn, hist_params = get_moe_histogram_kernel(numel, num_experts)
    pad_fn, pad_params = get_moe_pad_cumsum_kernel(num_experts, block_size)
    eid_fn, eid_params = get_moe_expert_ids_kernel(
        num_experts, max_num_m_blocks, block_size
    )
    scatter_fn, scatter_params = get_moe_sorted_ids_kernel(
        numel, num_experts, max_num_tokens_padded
    )
    return MoEAlignKernels(
        histogram=_compile_wave_kernel(hist_fn, hist_params),
        pad_cumsum=_compile_wave_kernel(pad_fn, pad_params),
        expert_ids=_compile_wave_kernel(eid_fn, eid_params),
        sorted_ids=_compile_wave_kernel(scatter_fn, scatter_params),
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    compiled_kernels: MoEAlignKernels | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    MoE token alignment and block size padding.

    Sorts tokens by their assigned expert IDs and pads each expert's token list
    to align with the specified block size for efficient blocked GEMM processing.

    Pipeline (all Wave kernels):
      Step 1: Histogram — expert token counts via atomic_add.
      Steps 2+3: Pad counts + prefix sums via cumsum.
      Step 4: Expert IDs fill — one workgroup per expert with iterate.
      Step 5: Sorted IDs scatter — parallel scatter via atomic_add.

    Args:
        topk_ids: Flat tensor of expert assignments, shape (num_tokens * topk,), int32.
        num_experts: Number of experts.
        block_size: Block size for alignment padding.
        max_num_tokens_padded: Maximum padded token count (for sorted_ids buffer size).
        max_num_m_blocks: Maximum number of blocks (for expert_ids buffer size).
        compiled_kernels: Pre-compiled alignment kernels from compile_moe_align_kernels().
            If None, kernels are compiled on each call (slow — avoid in benchmarks/inference).

    Returns:
        expert_ids: Shape (max_num_m_blocks,). Maps each block to its owning expert.
        sorted_ids: Shape (max_num_tokens_padded,). Token indices sorted by expert,
                    padded entries set to numel (== PAD_VALUE in the GEMM kernel).
        expert_counts: Shape (num_experts,). Raw count of tokens per expert.
        padded_counts: Shape (num_experts,). Counts padded up to block_size alignment.
        cumsum: Shape (num_experts,). Inclusive prefix sum of padded_counts.
        cumsum_exclusive: Shape (num_experts,). Exclusive prefix sum of padded_counts.
    """
    device = topk_ids.device
    topk_ids_flat = topk_ids.view(-1)
    numel = topk_ids_flat.numel()

    # Use pre-compiled kernels if provided, otherwise compile on the fly
    if compiled_kernels is None:
        compiled_kernels = compile_moe_align_kernels(
            numel,
            num_experts,
            block_size,
            max_num_tokens_padded,
            max_num_m_blocks,
        )

    # --- Step 1: Histogram — count tokens assigned to each expert ---
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    dummy = torch.empty(numel, dtype=torch.int32, device=device)
    compiled_kernels.histogram(topk_ids_flat.contiguous(), expert_counts, dummy)
    torch.cuda.synchronize()

    # --- Steps 2+3: Pad counts + prefix sums ---
    padded_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    cumsum = torch.zeros(num_experts, dtype=torch.int32, device=device)
    cumsum_exclusive = torch.zeros(num_experts, dtype=torch.int32, device=device)
    pad_dummy = torch.empty(num_experts, dtype=torch.int32, device=device)
    compiled_kernels.pad_cumsum(
        expert_counts, padded_counts, cumsum, cumsum_exclusive, pad_dummy
    )
    torch.cuda.synchronize()

    # --- Step 4: Expert IDs fill (one workgroup per expert) ---
    expert_ids = torch.zeros(max_num_m_blocks, dtype=torch.int32, device=device)
    expert_ids_dummy = torch.empty(num_experts, dtype=torch.int32, device=device)
    compiled_kernels.expert_ids(cumsum_exclusive, cumsum, expert_ids, expert_ids_dummy)
    torch.cuda.synchronize()

    # --- Step 5: Sorted IDs scatter ---
    sorted_ids = torch.full(
        (max_num_tokens_padded,), numel, dtype=torch.int32, device=device
    )
    write_pos = cumsum_exclusive.clone()
    scatter_dummy = torch.empty(numel, dtype=torch.int32, device=device)
    compiled_kernels.sorted_ids(
        topk_ids_flat.contiguous(), write_pos, sorted_ids, scatter_dummy
    )
    torch.cuda.synchronize()

    return (
        expert_ids,
        sorted_ids,
        expert_counts,
        padded_counts,
        cumsum,
        cumsum_exclusive,
    )
