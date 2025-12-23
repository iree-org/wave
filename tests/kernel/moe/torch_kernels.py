# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch


def moe_align_block_size_pytorch(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
):
    """
    PyTorch implementation matching moe_align_block_size behavior.
    All output tensors are pre-allocated by the caller.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_experts: Total number of experts
        block_size: Block size for expert processing
        sorted_ids: Pre-allocated output tensor for sorted token indices
        expert_ids: Pre-allocated output tensor for expert block assignments
        num_tokens_post_pad: Pre-allocated output tensor for total padded tokens
    """
    device = topk_ids.device
    num_tokens = topk_ids.numel()
    padding_value = num_tokens  # Value for padding tokens

    # Initialize output buffers
    sorted_ids.fill_(padding_value)
    num_tokens_post_pad.zero_()

    # Flatten the input and get expert counts
    flat_topk = topk_ids.view(-1).to(torch.int32)
    expert_counts = torch.bincount(flat_topk, minlength=num_experts)

    # Calculate padding needed per expert
    blocks_per_expert = (expert_counts + block_size - 1) // block_size
    padded_counts = blocks_per_expert * block_size
    total_size_with_padding = padded_counts.sum().item()
    num_tokens_post_pad.fill_(total_size_with_padding)

    # Calculate exclusive cumsum for expert offsets
    cumsum = torch.cumsum(padded_counts, dim=0) - padded_counts

    # Assign expert IDs to blocks
    expert_starts = torch.cumsum(padded_counts, dim=0) - padded_counts
    num_blocks = total_size_with_padding // block_size
    expert_ids[:num_blocks] = torch.repeat_interleave(
        torch.arange(num_experts, device=device), blocks_per_expert
    )

    if num_tokens == 0:
        return

    # Sort tokens by expert and fill valid positions

    # Get sorted order of tokens by expert: tokens are first sorted by the id
    # of their assigned expert, and if two tokens have the same expert id,
    # they'll be sorted by their original position in the flatten tensor.
    # I.e., first comes all indices of tokens assigned to expert 0, then all
    # indices of tokens assigned to expert 1, and so on.
    sorted_indices = torch.argsort(
        flat_topk * (num_tokens + 1) + torch.arange(num_tokens, device=device)
    )
    sorted_values = flat_topk[sorted_indices]

    # Calculate destination positions for each token
    token_positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    current_offsets = cumsum.to(torch.int64)

    # 2. Get per-expert offsets
    offsets = torch.cat(
        [torch.zeros(1, device=device), expert_counts.cumsum(0)[:-1]]
    ).long()

    # 3. Calculate local positions (0,1,2,... within each expert)
    local_positions = torch.arange(
        num_tokens, device=device
    ) - offsets.repeat_interleave(expert_counts)

    # 4. Calculate final positions
    token_positions = expert_starts[sorted_values] + local_positions

    # Scatter the original token indices
    original_indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
    sorted_ids[token_positions.long()] = original_indices[sorted_indices]
