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
    get_moe_align_block_size_kernel,
)
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang import DataType
import torch.nn.functional as F

from wave_lang.kernel.wave.utils.torch_utils import (
    device_arange,
    device_full,
    device_ones,
    device_randint,
    device_randn,
    device_randperm,
    device_zeros,
    to_default_device,
)

from wave_lang.kernel.wave.templates.moe import (
    get_fused_moe_gemm,
    get_silu_and_mul_kernel,
)

from tests.kernel.wave.moe.moe_align_block_size_test import (
    moe_align_block_size_pytorch,
)

import math

torch.manual_seed(0)


def fused_moe_pytorch_reference(
    # Input matrices
    a,
    b,
    bias,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    # Matrix dimensions
    M,
    N,
    K,
    EM,
    num_valid_tokens,
    # Configuration flags
    BLOCK_SIZE_M=64,
    top_k=2,
):
    """
    PyTorch reference implementation for the fused MOE kernel.

    This implements the core computation: each token is multiplied by its assigned
    expert's weight matrix, with optional bias, quantization, and routing weights.
    """
    device = a.device
    dtype = a.dtype

    # Initialize output tensor
    c = torch.zeros(M, top_k, N, dtype=dtype, device=device)

    # Process tokens in blocks
    num_blocks = (EM + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for block_idx in range(num_blocks):
        # Get block boundaries
        start_m = block_idx * BLOCK_SIZE_M
        end_m = min(start_m + BLOCK_SIZE_M, EM)

        if start_m >= num_tokens_post_padded:
            continue

        # Get expert for this block
        if block_idx >= len(expert_ids):
            continue

        expert_id = expert_ids[block_idx].item()

        # Skip invalid experts (-1 indicates no expert assigned or invalid expert id)
        if expert_id == -1 or expert_id >= len(b) or expert_id < 0:
            c[start_m:end_m] = 0
            continue

        # Get token indices for this block
        token_indices = sorted_token_ids[start_m:end_m]

        # Filter valid tokens (not padding)
        valid_mask = token_indices < num_valid_tokens
        if not valid_mask.any():
            continue

        valid_token_indices = token_indices[valid_mask]

        # Convert token indices accounting for top_k expansion
        # Each original token appears top_k times in the sorted list
        original_token_indices = valid_token_indices // top_k

        # Ensure indices are within bounds
        assert torch.all(original_token_indices < len(a))

        # Get input tokens for this block
        block_a = a[original_token_indices, :]  # [valid_tokens_in_block, K]

        # Get expert weights and bias
        expert_weights = b[expert_id]  # [K, N]
        expert_bias = bias[expert_id] if bias is not None else None  # [N]

        # Perform matrix multiplication: block_a @ expert_weights
        block_output = torch.matmul(
            block_a, expert_weights
        )  # [valid_tokens_in_block, N]

        # Add bias if present
        if expert_bias is not None:
            block_output = block_output + expert_bias

        # Ensure output matches the target dtype
        block_output = block_output.to(dtype)

        # Store results in output tensor
        valid_token_count = 0
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                token_id = token_indices[i].item()
                orig_token = token_id // top_k
                expert_slot = token_id % top_k
                c[orig_token, expert_slot] = block_output[valid_token_count]
                valid_token_count += 1

    return c


def create_test_data(
    num_tokens, num_experts, K, N, top_k, block_size, dtype=torch.float16, device="cuda"
):
    """Create test data for fused MOE kernel testing"""

    # Create input token matrix
    a = torch.randn(num_tokens, K, dtype=dtype, device=device)

    # Create expert weight matrices
    b = torch.randn(num_experts, K, N, dtype=dtype, device=device)

    # Create expert biases
    bias = torch.randn(num_experts, N, dtype=dtype, device=device)

    # Create routing scores and get top-k
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    scores = torch.softmax(scores, dim=-1)
    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=1)

    # Convert topk_weights to match input dtype
    topk_weights = topk_weights.to(dtype)

    # Flatten for processing
    topk_weights = topk_weights.view(-1)  # [num_tokens * top_k]
    topk_ids = topk_ids.view(-1)  # [num_tokens * top_k]

    # Use the block alignment logic to get sorted indices and expert assignments
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), topk_ids.numel(), dtype=torch.int32, device=device
    )
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.full((max_num_blocks,), -1, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)

    # Use the existing block alignment function
    from tests.kernel.wave.moe.moe_align_block_size_test import (
        moe_align_block_size_pytorch,
    )

    moe_align_block_size_pytorch(
        topk_ids.to(torch.int32),
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return {
        "a": a,
        "b": b,
        "bias": bias,
        "topk_weights": topk_weights,
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_padded": num_tokens_post_pad.item(),
        "M": num_tokens,
        "N": N,
        "K": K,
        "EM": num_tokens_post_pad.item(),
        "num_valid_tokens": topk_ids.numel(),
        "topk_ids": topk_ids,
        "topk_weights_original": topk_weights,
    }


num_tokens_values = [32, 64]
num_experts_values = [4, 8]
K_values = [128, 256]
N_values = [128, 256]
top_k_values = [2]
block_size_values = [16, 32]
dtypes = [torch.float16]


@pytest.mark.parametrize("num_tokens", num_tokens_values)
@pytest.mark.parametrize("num_experts", num_experts_values)
@pytest.mark.parametrize("K", K_values)
@pytest.mark.parametrize("N", N_values)
@pytest.mark.parametrize("top_k", top_k_values)
@pytest.mark.parametrize("block_size", block_size_values)
@pytest.mark.parametrize("dtype", dtypes)
def test_fused_moe_kernel_reference(
    num_tokens: int,
    num_experts: int,
    K: int,
    N: int,
    top_k: int,
    block_size: int,
    dtype: torch.dtype,
):
    """
    Test the PyTorch reference implementation of the fused MOE kernel
    """
    device = "cuda"

    # Create test data
    test_data = create_test_data(
        num_tokens=num_tokens,
        num_experts=num_experts,
        K=K,
        N=N,
        top_k=top_k,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )

    # Run the reference implementation
    output = fused_moe_pytorch_reference(
        a=test_data["a"],
        b=test_data["b"],
        bias=test_data["bias"],
        topk_weights=test_data["topk_weights"],
        sorted_token_ids=test_data["sorted_token_ids"],
        expert_ids=test_data["expert_ids"],
        num_tokens_post_padded=test_data["num_tokens_post_padded"],
        M=test_data["M"],
        N=test_data["N"],
        K=test_data["K"],
        EM=test_data["EM"],
        num_valid_tokens=test_data["num_valid_tokens"],
        top_k=top_k,
        BLOCK_SIZE_M=block_size,
    )

    # Verify output shape
    assert output.shape == (test_data["EM"], top_k, N)

    # Verify that output dtype matches input
    assert output.dtype == dtype

    # Basic sanity checks
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.isfinite(output).all(), "Output contains infinite values"

    print(
        f"Test passed for num_tokens={num_tokens}, num_experts={num_experts}, "
        f"K={K}, N={N}, top_k={top_k}, block_size={block_size}, dtype={dtype}"
    )


def SiluAndMul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def torch_ref_moe(
    a,
    w1,
    w2,
    topk_ids,
    topk,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    topk_ids = topk_ids.view(-1)

    if w1.dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
        w1_compute = w1.to(a.dtype)
        w2_compute = w2.to(a.dtype)

        if w1_scale is not None:
            w1_compute = (w1_compute * w1_scale.view(-1, 1, 1)).to(a.dtype)
        if w2_scale is not None:
            w2_compute = (w2_compute * w2_scale.view(-1, 1, 1)).to(a.dtype)
        if a1_scale is not None:
            a = (a * a1_scale).to(a.dtype)
        if a2_scale is not None:
            a = (a * a2_scale).to(a.dtype)
    else:
        w1_compute = w1
        w2_compute = w2

    for i in range(w1_compute.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul_ref(
                a[mask] @ w1_compute[i].transpose(0, 1)
            ) @ w2_compute[i].transpose(0, 1)
    return out


def get_wave_moe_fused_gemm_kernel(
    m: int,
    n: int,
    k: int,
    e,
    block_shape: int,
    total_elems: int,
    num_experts: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    gemm, symbols = get_fused_moe_gemm(
        m,
        n,
        k,
        e,
        block_shape,
        total_elems,
        num_experts,
        mfma_variant,
        datatype,
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
    )
    optons = set_default_run_config(options)
    gemm = wave_compile(options, gemm)
    print("--------------------------------")
    print(gemm.asm)
    print("--------------------------------")
    return gemm


def nit_tkw(
    a, w1, w2, topk, sorted_ids, expert_ids, num_experts, block_size, num_blocks
):
    m, k = a.shape
    a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)
    gemm1_out = torch.zeros(m * topk, w1.shape[1], dtype=torch.float32, device=a.device)
    silu_and_mul_out = torch.zeros(
        m * topk, w1.shape[1] // 2, dtype=torch.float32, device=a.device
    )
    # Final output tensor - matches w2.shape[1] (final output dimension)
    final_out = torch.zeros(m * topk, w2.shape[1], dtype=torch.float32, device=a.device)

    # Scratch tensors for GEMM1
    a_scratch = torch.zeros(
        num_blocks, a.shape[0], k, dtype=torch.float16, device=a.device
    )
    c_scratch = torch.zeros(
        num_blocks, a.shape[0], w1.shape[1], dtype=torch.float32, device=a.device
    )

    # GEMM1: a @ w1 -> gemm1_out [tokens, 2*n]
    gemm1 = get_wave_moe_fused_gemm_kernel(
        m * topk,
        w1.shape[1],
        k,
        w1.shape[0],
        block_size,
        sorted_ids.shape[0],
        num_experts,
        MMAType.F32_16x16x16_F16,
        torch.float16,
    )

    gemm1(a, w1, sorted_ids, expert_ids, a_scratch, gemm1_out, c_scratch)

    # SiluAndMul: split gemm1_out and apply activation
    d = gemm1_out.shape[-1] // 2
    gate = gemm1_out[..., :d].contiguous()
    up = gemm1_out[..., d:].contiguous()

    silu_and_mul, symbols = get_silu_and_mul_kernel(
        gate.shape[0],
        gate.shape[1],
        tkl.f32,
    )

    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=symbols,
    )
    options = set_default_run_config(options)
    silu_and_mul = wave_compile(options, silu_and_mul)
    silu_and_mul(gate, up, silu_and_mul_out)

    # GEMM2: silu_and_mul_out @ w2 -> final_out [tokens, final_dim]
    # We need scratch tensors for GEMM2
    a2_scratch = torch.zeros(
        num_blocks,
        silu_and_mul_out.shape[0],
        silu_and_mul_out.shape[1],
        dtype=torch.float16,
        device=a.device,
    )
    c2_scratch = torch.zeros(
        num_blocks,
        silu_and_mul_out.shape[0],
        w2.shape[1],
        dtype=torch.float32,
        device=a.device,
    )

    gemm2 = get_wave_moe_fused_gemm_kernel(
        m * topk,  # M: number of tokens
        w2.shape[1],  # N: final output dimension
        silu_and_mul_out.shape[1],  # K: intermediate dimension (w1.shape[1] // 2)
        w2.shape[0],  # E: number of experts
        block_size,
        sorted_ids.shape[0],  # total elements
        num_experts,
        MMAType.F32_16x16x16_F16,
        torch.float16,
    )

    # Convert silu_and_mul_out to f16 for GEMM2 input
    silu_and_mul_out_f16 = silu_and_mul_out.to(torch.float16)

    gemm2(
        silu_and_mul_out_f16,
        w2,
        sorted_ids,
        expert_ids,
        a2_scratch,
        final_out,
        c2_scratch,
    )

    return final_out


num_tokens_values = [32]
n_values = [64]
k_values = [128]
num_experts = [4]
top_ks = [2]
dtypes = [torch.float16]
rtol, atol = 1e-1, 1e-2
block_size_values = [4]


@pytest.mark.parametrize("num_tokens", num_tokens_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("num_experts", num_experts)
@pytest.mark.parametrize("topk", top_ks)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("block_size", block_size_values)
def testnittestReferenceMoe(
    num_tokens: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    dtype: DataType,
    block_size: int,
):
    device = "cuda"

    if dtype == torch.float16 and k == 1024:
        pytest.skip("This combination generates NaNs and INFs")

    # TODO: investigate why using torch.randn would have precision issue in silu computation
    a = torch.rand((num_tokens, k), dtype=dtype, device=device)
    w1 = torch.rand((num_experts, 2 * n, k), dtype=dtype, device=device)
    w2 = torch.rand((num_experts, k, n), dtype=dtype, device=device)

    score = torch.rand((num_tokens, num_experts), dtype=dtype, device=device)
    topk_ids = torch.topk(score, topk, dim=1)[1]

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )

    max_num_m_blocks = -(max_num_tokens_padded // -block_size)
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.full((1,), num_tokens, dtype=torch.int32, device=device)

    moe_align_block_size_pytorch(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )

    num_blocks = expert_ids.shape[0]
    ref_output = torch_ref_moe(a, w1, w2, topk_ids, topk)
    nit_tkw_output = nit_tkw(
        a, w1, w2, topk, sorted_ids, expert_ids, num_experts, block_size, num_blocks
    )

    print(nit_tkw_output)
    print(ref_output)
    torch.testing.assert_close(
        nit_tkw_output.to(torch.float16), ref_output, rtol=rtol, atol=atol
    )

    # # TODO: remove manual splitting
    # # We need to manually split w1 into 2 halves, since this is
    # # required by `silu_and_mul` kernel, and currently we can't
    # # do this in Wave.
    # w1_gate = w1[:, :n, :]  # First half for gate
    # w1_up = w1[:, n:, :]  # Second half for up projection

    # # Make sure the algorithm with w1 splitting works in PyTorch.
    # ref_split_output = torch_ref_moe_split_w1(a, w1_gate, w1_up, w2, score, topk)
    # torch.testing.assert_close(ref_split_output, ref_output, rtol=rtol, atol=atol)

    # # The implementation in Wave should also work.
    # tkw_output = tkw_moe_split_w1(a, w1_gate, w1_up, w2, score, topk)
    # torch.testing.assert_close(tkw_output, ref_output, rtol=rtol, atol=atol)
