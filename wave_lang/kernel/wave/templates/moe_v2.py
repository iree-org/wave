# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MoE implementation with progressive Wave migration.

Working Wave kernels are re-exported from moe.py.
Steps migrated to Wave are marked with WAVE.
Steps still in PyTorch are marked with TODO.
"""

import torch
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

# Re-export working Wave kernels from moe.py
from wave_lang.kernel.wave.templates.moe import (  # noqa: F401
    get_fused_moe_gemm,
    get_silu_and_mul_kernel,
    get_moe_reduce_sum_kernel,
    get_topk_kernel,
    get_gemm_kernel,
)


# ---------------------------------------------------------------------------
# MoE GEMM-only kernel (no gather/scatter — avoids cross-WG1 race)
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

    Unlike get_fused_moe_gemm, this kernel does NOT gather/scatter tokens.
    Gather/scatter must be done externally (e.g., via moe_gather/moe_scatter).

    This avoids the cross-WORKGROUP_1 race condition in the fused kernel where
    multiple N-tile workgroups concurrently zero-initialize and gather to the
    same shared a_back buffer without cross-workgroup synchronization.

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
    hyperparams: dict[str | IndexSymbol, Any] = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: 32,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: e,
        NUM_BLOCKS: num_blocks,
    }

    return moe_gemm_only, hyperparams


# ---------------------------------------------------------------------------
# PyTorch gather/scatter for MoE (replaces the fused kernel's inline version)
# ---------------------------------------------------------------------------
def moe_gather(a, sorted_ids, a_back, block_size, pad_value):
    """Gather input rows into per-block scratch buffer.

    a_back[block, :block_size, :] = a[sorted_ids[block*bs + t]] for valid entries.
    Invalid entries (sorted_ids >= pad_value) are left as zero.
    """
    num_blocks = a_back.shape[0]
    total_slots = min(num_blocks * block_size, sorted_ids.shape[0])
    idx = sorted_ids[:total_slots].long()
    valid = idx < pad_value
    safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
    gathered = a[safe_idx]
    gathered[~valid] = 0
    gathered = gathered.reshape(num_blocks, block_size, -1)
    a_back[:, :block_size, :] = gathered.to(a_back.dtype)


def moe_scatter(c_back, sorted_ids, c, block_size, pad_value):
    """Scatter per-block GEMM results back to output positions.

    c[sorted_ids[block*bs + t], :] = c_back[block, t, :] for valid entries.
    """
    num_blocks = c_back.shape[0]
    total_slots = min(num_blocks * block_size, sorted_ids.shape[0])
    idx = sorted_ids[:total_slots].long()
    valid = idx < pad_value
    values = c_back[:, :block_size, :].reshape(total_slots, -1)
    valid_idx = idx[valid]
    valid_values = values[valid]
    c[valid_idx] = valid_values


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

    # Maps dynamic index d0 (expert_id value) to position in expert_counts
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
# Helper: compile a Wave kernel with default settings
# ---------------------------------------------------------------------------
def _compile_wave_kernel(kernel_fn, hyperparams, **compile_kwargs):
    """Compile a Wave kernel with default scheduling params and run config."""
    hp = dict(hyperparams)
    hp.update(get_default_scheduling_params())
    options = WaveCompileOptions(subs=hp, **compile_kwargs)
    options = set_default_run_config(options)
    return wave_compile(options, kernel_fn)


def moe_align_block_size_pytorch(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    PyTorch host implementation of MoE token alignment and block size padding.

    Sorts tokens by their assigned expert IDs and pads each expert's token list
    to align with the specified block size for efficient blocked GEMM processing.

    TODO: Replace with Wave kernel implementation. The current Wave kernel
    (get_moe_align_block_size_kernel in moe.py) has these issues to fix:
      1. Only processes wave_size (32) topk_ids elements per workgroup — misses
         the rest when numel > 32.
      2. expert_ids loop uses wave-uniform symbols (I, I_MAX) causing all threads
         to write to the same positions — last thread's expert value wins.
      3. cumsum_exclusive is corrupted by atomic_adds in the sorted_token_ids
         phase that reuse the same buffer.

    Args:
        topk_ids: Flat tensor of expert assignments, shape (num_tokens * topk,), int32.
        num_experts: Number of experts.
        block_size: Block size for alignment padding.
        max_num_tokens_padded: Maximum padded token count (for sorted_ids buffer size).
        max_num_m_blocks: Maximum number of blocks (for expert_ids buffer size).

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

    # --- Step 1: Histogram — count tokens assigned to each expert ---
    # WAVE: atomic_add histogram kernel (see get_moe_histogram_kernel above)
    hist_fn, hist_params = get_moe_histogram_kernel(numel, num_experts)
    compiled_hist = _compile_wave_kernel(hist_fn, hist_params)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    dummy = torch.empty(numel, dtype=torch.int32, device=device)
    compiled_hist(topk_ids_flat.contiguous(), expert_counts, dummy)
    # Sync to ensure Wave kernel output is visible to subsequent PyTorch ops
    # (Wave/IREE kernels may run on a separate CUDA stream)
    torch.cuda.synchronize()

    # --- Step 2: Pad counts to block_size alignment ---
    # TODO: migrate to Wave (elementwise kernel)
    padded_counts = ((expert_counts + block_size - 1) // block_size * block_size).to(
        torch.int32
    )

    # --- Step 3: Prefix sums (inclusive and exclusive) ---
    # TODO: migrate to Wave (prefix sum / cumsum kernel)
    cumsum = torch.cumsum(padded_counts, dim=0).to(torch.int32)
    cumsum_exclusive = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        cumsum_exclusive[1:] = cumsum[:-1]

    # --- Step 4: Build expert_ids — for each block, which expert owns it ---
    # TODO: migrate to Wave (parallel expert_ids fill kernel)
    #   The Wave version needs per-thread iteration state, not wave-uniform symbols.
    #   Consider inverting: each block does a binary search over cumsum to find its expert.
    expert_ids = torch.zeros(max_num_m_blocks, dtype=torch.int32, device=device)
    for expert in range(num_experts):
        start_pos = cumsum_exclusive[expert].item()
        end_pos = cumsum[expert].item()
        for pos in range(start_pos, end_pos, block_size):
            block_idx = pos // block_size
            if block_idx < max_num_m_blocks:
                expert_ids[block_idx] = expert

    # --- Step 5: Build sorted_token_ids — place each token at its expert's offset ---
    # Initialize with padding value (numel = num_tokens * topk = PAD_VALUE in GEMM)
    # TODO: migrate to Wave (parallel scatter kernel)
    sorted_ids = torch.full(
        (max_num_tokens_padded,), numel, dtype=torch.int32, device=device
    )
    write_pos = cumsum_exclusive.clone()
    for i in range(numel):
        expert = topk_ids_flat[i].item()
        pos = write_pos[expert].item()
        if pos < max_num_tokens_padded:
            sorted_ids[pos] = i
        write_pos[expert] += 1

    return (
        expert_ids,
        sorted_ids,
        expert_counts,
        padded_counts,
        cumsum,
        cumsum_exclusive,
    )
