# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from .torch_kernels import moe_align_block_size_pytorch
import torch.nn.functional as F

torch.manual_seed(0)


num_tokens_values = [1, 33, 256]
topk_values = [2]
block_size_values = [16, 32, 64]
num_experts_values = [4, 8, 64]


def verify_moe_align_block_size_results(
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size: int,
    num_experts: int,
):
    device = topk_ids.device
    flat_topk = topk_ids.view(-1)
    num_tokens = flat_topk.numel()
    padding_value = num_tokens

    # 1. Verify padding count makes sense
    total_padded = num_tokens_post_pad.item()
    assert total_padded >= num_tokens, "Cannot have fewer positions than tokens"
    assert total_padded % block_size == 0, "Padded tokens should be block-aligned"

    # 2. Verify all original tokens appear exactly once
    valid_entries = sorted_ids[sorted_ids != padding_value]
    assert valid_entries.numel() == num_tokens, "Missing or extra tokens"
    assert torch.all(
        valid_entries.sort().values == torch.arange(num_tokens, device=device)
    ), "Token IDs corrupted"

    # 3. Verify expert assignments preserve original routing
    for expert in range(num_experts):
        # Get all tokens originally assigned to this expert
        original_token_mask = flat_topk == expert
        original_token_indices = original_token_mask.nonzero().squeeze(-1)

        # Get their positions in sorted_ids
        sorted_positions = (
            (sorted_ids[..., None] == original_token_indices)
            .any(-1)
            .nonzero()
            .squeeze(-1)
        )

        # Check they're in contiguous blocks
        if original_token_indices.numel() > 0:
            expert_start = sorted_positions.min()
            expert_end = sorted_positions.max() + 1
            assert torch.all(
                sorted_ids[expert_start:expert_end] != padding_value
            ), "Padding in middle of expert block"

            # Verify the expert_ids assignments match
            block_start = expert_start // block_size
            block_end = (expert_end + block_size - 1) // block_size
            assert torch.all(
                expert_ids[block_start:block_end] == expert
            ), "Expert ID mismatch"

    # 4. Verify padding only appears at end of each expert block
    for expert in range(num_experts):
        expert_mask = flat_topk == expert
        if expert_mask.any():
            # Find the expert's region in sorted_ids
            expert_token_mask = (
                sorted_ids[..., None] == expert_mask.nonzero().squeeze(-1)
            ).any(-1)
            first_pad = (sorted_ids == padding_value).nonzero()
            if first_pad.numel() > 0:
                first_pad = first_pad.min()
                assert not torch.any(
                    (sorted_ids == padding_value)
                    & (torch.arange(sorted_ids.numel(), device=device) < first_pad)
                ), "Padding appears before end"


@pytest.mark.parametrize("num_tokens", num_tokens_values)
@pytest.mark.parametrize("topk", topk_values)
@pytest.mark.parametrize("block_size", block_size_values)
@pytest.mark.parametrize("num_experts", num_experts_values)
def test_moe_align_block_size(
    num_tokens: int,
    topk: int,
    block_size: int,
    num_experts: int,
):
    """
    Tests the moe_align_block_size function using Pytest parameterization.
    """
    device = "cuda"

    scores = torch.rand(num_tokens, num_experts)

    # Get topk expert indices for each token
    _, topk_ids = torch.topk(scores, k=topk, dim=1)

    # Conservative upper bound that accounts for both the number of tokens and
    # the maximum possible padding needed per expert
    # The first component `topk_ids.numel()` handles the extreme case where all
    # tokens are assigned to the same expertm while the second component
    # `num_experts * (block_size - 1)` handles the extreme case where each
    # expert only has 1 token assigned.
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )

    # Ceiling div to get the total number of blocks
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)
    # Mapping block number -> expert id
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

    verify_moe_align_block_size_results(
        topk_ids, sorted_ids, expert_ids, num_tokens_post_pad, block_size, num_experts
    )
