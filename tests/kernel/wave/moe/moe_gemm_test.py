# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from .torch_kernels import moe_align_block_size_pytorch
import torch.nn.functional as F
from wave_lang.kernel.lang import DataType

torch.manual_seed(0)


num_tokens_values = [1, 33, 256]
topk_values = [2]
block_size_values = [16, 32, 64]
num_experts_values = [8, 64]
k_values = [128, 511]
n_values = [128, 512, 1024]
dtype_values = [torch.float16, torch.bfloat16]


def moe_gemm_pytorch(
    a,  # Input tokens: [M, K]
    b,  # Expert weights: [E, N, K]
    c,  # Output: [M, topk, N]
    sorted_token_ids,  # Sorted token-expert pair indices: [EM] (padded)
    expert_ids,  # Expert ID for each block: [num_blocks]
    num_tokens_post_padded,  # Total padded length: [1]
    top_k,  # Number of experts per token
    block_size_m=64,
    block_size_n=64,
    block_size_k=64,
):
    """
    PyTorch equivalent of the Triton fused MoE kernel.

    Args:
        a: Input token embeddings [M, K]
        b: Expert weight matrices [E, N, K]
        sorted_token_ids: Token-expert pair indices sorted by expert [EM] (padded)
        expert_ids: Expert ID for each block [num_blocks]
        num_tokens_post_padded: Total padded length [1]
        top_k: Number of experts each token is routed to

    Returns:
        c: Output tensor [M, topk, N]
    """
    M, K = a.shape
    E, N, _ = b.shape
    EM = sorted_token_ids.shape[0]
    num_valid_tokens = M * top_k

    # Process tokens in blocks
    num_blocks = (EM + block_size_m - 1) // block_size_m

    for i, idx in enumerate(sorted_token_ids.tolist()):
        if i % block_size_m == 0:
            block_token_ids = sorted_token_ids[i : i + block_size_m]
            orig_token_ids = torch.clamp(block_token_ids // top_k, 0, M - 1)
            valid_mask = block_token_ids < num_valid_tokens
    for block_idx in range(num_blocks):
        # Determine token range for this block
        start_idx = block_idx * block_size_m
        end_idx = min(start_idx + block_size_m, EM)

        # Skip if we're past valid tokens
        if start_idx >= num_tokens_post_padded.item():
            continue

        # Get token-expert pair indices for this block
        block_token_ids = sorted_token_ids[start_idx:end_idx]

        # Create mask for valid token-expert pairs
        valid_mask = block_token_ids < num_valid_tokens

        if not valid_mask.any():
            continue

        # Get the expert ID for this block
        expert_id = expert_ids[block_idx].item()

        # Initialize accumulator for this block
        accumulator = torch.zeros(block_size_m, N, dtype=a.dtype, device=a.device)

        # Process K dimension in chunks (simulating the K-loop in Triton)
        for k_start in range(0, K, block_size_k):
            k_end = min(k_start + block_size_k, K)
            actual_k_size = k_end - k_start

            # Load block from A
            # Map token-expert pair indices to original token indices
            orig_token_ids = torch.clamp(block_token_ids // top_k, 0, M - 1)
            block_a = a[orig_token_ids, k_start:k_end]  # [block_size_m, actual_k_size]

            # Apply valid mask to A
            block_a = block_a * valid_mask.to(a.dtype).unsqueeze(1)

            # Load block from B (expert weights)
            # Process N dimension in chunks
            for n_start in range(0, N, block_size_n):
                n_end = min(n_start + block_size_n, N)
                actual_n_size = n_end - n_start

                # Get expert weights: B[expert_id, n_start:n_end, k_start:k_end]
                # Need to transpose to [k, n] for matrix multiplication
                block_b = b[
                    expert_id, n_start:n_end, k_start:k_end
                ].t()  # [actual_k_size, actual_n_size]

                # Compute matrix multiplication: [block_size_m, k] @ [k, n] = [block_size_m, n]
                partial_result = torch.matmul(
                    block_a, block_b
                )  # [block_size_m, actual_n_size]

                # Accumulate in the correct position
                accumulator[:, n_start:n_end] += partial_result

        # Write back to output tensor using the stride-based mapping
        # This is the key: sorted_token_ids encodes the mapping to 3D positions
        for i, token_id in enumerate(block_token_ids):
            if token_id >= num_valid_tokens:
                continue

            # Decode the 3D position from the flat index
            # Which original token (0 to M-1)
            orig_token = token_id // top_k
            # Which expert slot for that token (0 to top_k-1)
            expert_slot = token_id % top_k

            c[orig_token, expert_slot] = accumulator[i]


@pytest.mark.parametrize("num_tokens", num_tokens_values)
@pytest.mark.parametrize("topk", topk_values)
@pytest.mark.parametrize("block_size", block_size_values)
@pytest.mark.parametrize("num_experts", num_experts_values)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("dtype", dtype_values)
def test_moe_gemm(
    num_tokens: int,
    topk: int,
    block_size: int,
    num_experts: int,
    k: int,
    n: int,
    dtype: DataType,
):
    """
    Tests the fused_moe_kernel function using Pytest parameterization.
    """
    scores = torch.rand(num_tokens, num_experts)

    # Get topk expert indices for each token
    _, topk_ids = torch.topk(scores, k=topk, dim=1)

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    fuse_sorted_ids_padding = sorted_ids.shape[0] <= 4096
    if not fuse_sorted_ids_padding:
        sorted_ids.fill_(topk_ids.numel())

    moe_align_block_size_pytorch(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )

    a = torch.rand((num_tokens, k), dtype=dtype)
    b = torch.rand((num_experts, n, k), dtype=dtype)
    c = torch.zeros(num_tokens, topk, n, dtype=dtype)

    moe_gemm_pytorch(
        a, b, c, sorted_ids, expert_ids, num_tokens_post_pad, topk, block_size
    )

    ref_c = torch.zeros(num_tokens, topk, n, dtype=dtype)
    for expert_idx in range(num_experts):
        mask = topk_ids == expert_idx
        token_indices, topk_indices = torch.where(mask)

        if len(token_indices) > 0:
            tokens = a[token_indices]
            expert_out = torch.matmul(tokens, b[expert_idx].T)

            ref_c[token_indices, topk_indices] = expert_out

    rtol, atol = 1e-1, 1e-2

    torch.testing.assert_close(
        c,
        ref_c,
        rtol=rtol,
        atol=atol,
    )
