# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import wave_lang.kernel as tk
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
    enable_scheduling_barriers,
    dump_generated_mlir,
    check_individual_kernels,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.templates.moe import (
    get_gemm_kernel,
    get_silu_and_mul_kernel,
)
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang import DataType
import torch.nn.functional as F

torch.manual_seed(0)


num_tokens_values = [1, 33, 256]
topk_values = [2]
block_size_values = [16, 32, 64]
num_experts_values = [4, 8, 64]


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

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    cumsum_buffer = torch.empty(
        (num_experts + 1,), dtype=torch.int32, device=topk_ids.device
    )
    token_cnts_buffer = torch.empty(
        (num_experts + 1) * num_experts,
        dtype=torch.int32,
        device=topk_ids.device,
    )

    fuse_sorted_ids_padding = sorted_ids.shape[0] <= 4096
    if not fuse_sorted_ids_padding:
        sorted_ids.fill_(topk_ids.numel())

    moe_align_block_size_pytorch(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )

    verify_moe_align_block_size_results(
        topk_ids, sorted_ids, expert_ids, num_tokens_post_pad, block_size, num_experts
    )
