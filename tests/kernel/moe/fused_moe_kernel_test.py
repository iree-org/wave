# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang import DataType
from wave_lang.kernel.wave.templates.moe import (
    get_fused_moe_gemm,
    get_moe_align_block_size_kernel,
    get_moe_reduce_sum_kernel,
    get_silu_and_mul_kernel,
    get_topk_kernel,
)
import torch.nn.functional as F

torch.manual_seed(0)


def SiluAndMul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def torch_ref_moe(
    a,
    w1,
    w2,
    score,
    topk,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
):
    """
    Reference implementation of MoE kernel based on sglang reference implementation
    https://github.com/harsh-nod/sglang/blob/wave_moe/test/srt/test_wave_fused_moe.py

    """
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=torch.float32, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score, topk)
    topk_weights = topk_weights.view(-1)
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

    gemm1_result = torch.zeros(
        B * topk, w1.shape[1], dtype=torch.float32, device=a.device
    )
    silu_mul_result = torch.zeros(
        B * topk, w1.shape[1] // 2, dtype=torch.float32, device=a.device
    )
    silu_mul_result_f16 = torch.zeros(
        B * topk, w1.shape[1] // 2, dtype=torch.float16, device=a.device
    )

    for i in range(w1_compute.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            gemm1_result[mask] = a[mask].float() @ w1_compute[i].transpose(0, 1).float()
            silu_mul_result[mask] = SiluAndMul_ref(gemm1_result[mask])
            silu_mul_result_f16[mask] = silu_mul_result[mask].to(torch.float16)
            out[mask] = (
                silu_mul_result_f16[mask].float()
                @ w2_compute[i].transpose(0, 1).float()
            )

    return (
        out.view(B, -1, w2.shape[1]) * topk_weights.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


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
    options = set_default_run_config(options)
    return wave_compile(options, gemm)


def get_wave_moe_align_block_size_kernel(
    num_tokens: int,
    num_experts: int,
    block_size: int,
    num_topk_ids: int,
    max_num_m_blocks: int,
    max_num_tokens_padded: int,
    topk: int,
):
    kernel, hyperparams, dynamic_symbols = get_moe_align_block_size_kernel(
        num_tokens,
        num_experts,
        block_size,
        num_topk_ids,
        max_num_m_blocks,
        max_num_tokens_padded,
        topk,
    )
    options = WaveCompileOptions(
        subs=hyperparams,
        minimize_shared_allocs=False,
    )
    return wave_compile(options, kernel)


def get_wave_silu_and_mul_kernel(m: int, n: int, dtype: DataType):
    kernel, symbols = get_silu_and_mul_kernel(m, n, dtype)
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=symbols,
    )
    options = set_default_run_config(options)
    return wave_compile(options, kernel)


def get_wave_reduce_sum_kernel(b: int, k: int, d: int, dtype: DataType):
    kernel, symbols = get_moe_reduce_sum_kernel(b, k, d, dtype)
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=symbols,
    )
    options = set_default_run_config(options)
    return wave_compile(options, kernel)


def get_wave_topk_kernel(m: int, n: int, k: int, dtype: DataType):
    kernel, symbols = get_topk_kernel(m, n, k, dtype)
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=symbols,
    )
    options = set_default_run_config(options)
    return wave_compile(options, kernel)


def tkw_moe(a, w1, w2, score, topk, num_experts, block_size, num_tokens):
    # Calculate buffer sizes for block-aligned computation
    max_num_tokens_padded = score.numel() + num_experts * (block_size - 1)
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)

    # Router: Select top-k experts for each token using Wave topk kernel
    score = torch.softmax(score, dim=-1, dtype=torch.float32)

    # Compile and run topk kernel
    topk_kernel = get_wave_topk_kernel(
        num_tokens,
        num_experts,
        topk,
        tkl.f32,
    )

    # Allocate output buffers for topk
    topk_weights = torch.zeros((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")

    # Run topk kernel
    topk_kernel(score, topk_weights, topk_ids)
    topk_weights = topk_weights.view(-1)
    topk_ids = topk_ids.view(-1)

    # Compile and run block alignment kernel to sort tokens by expert
    moe_align_block_size = get_wave_moe_align_block_size_kernel(
        num_tokens,
        num_experts,
        block_size,
        topk_ids.numel(),
        max_num_m_blocks,
        max_num_tokens_padded,
        topk,
    )

    # Output buffers for moe_align_block_size kernel
    expert_counts_buffer = torch.empty(num_experts, dtype=torch.int32, device="cuda")
    padded_counts_buffer = torch.empty(num_experts, dtype=torch.int32, device="cuda")
    cumsum_buffer = torch.empty(num_experts, dtype=torch.int32, device="cuda")
    cumsum_exclusive = torch.zeros(num_experts, dtype=torch.int32, device="cuda")
    num_blocks_buffer = torch.empty(num_experts, dtype=torch.int32, device="cuda")

    expert_ids = torch.zeros(
        max_num_m_blocks, dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids = torch.empty(
        max_num_tokens_padded, dtype=torch.int32, device=topk_ids.device
    )

    moe_align_block_size(
        topk_ids.to(torch.int32),
        expert_ids,
        expert_counts_buffer,
        padded_counts_buffer,
        cumsum_buffer,
        cumsum_exclusive,
        num_blocks_buffer,
        sorted_ids,
    )

    num_blocks = expert_ids.shape[0]

    # Replicate input activations for each selected expert
    m, k = a.shape
    a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)

    # Allocate output tensors
    gemm1_out = torch.zeros(m * topk, w1.shape[1], dtype=torch.float32, device=a.device)
    silu_and_mul_out = torch.zeros(
        m * topk, w1.shape[1] // 2, dtype=torch.float32, device=a.device
    )
    gemm2_out = torch.zeros(m * topk, w2.shape[1], dtype=torch.float32, device=a.device)

    # GEMM1: Compute gate and up projections (a @ w1.T)
    a_scratch = torch.zeros(
        num_blocks, a.shape[0], k, dtype=torch.float16, device=a.device
    )
    c_scratch = torch.zeros(
        num_blocks, a.shape[0], w1.shape[1], dtype=torch.float32, device=a.device
    )

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

    # Apply SiLU activation: SiLU(gate) * up
    # d = gemm1_out.shape[-1] // 2
    # gate = gemm1_out[..., :d].contiguous()
    # up = gemm1_out[..., d:].contiguous()

    silu_and_mul = get_wave_silu_and_mul_kernel(
        gemm1_out.shape[0],
        gemm1_out.shape[1] // 2,
        tkl.f32,
    )
    silu_and_mul(gemm1_out, silu_and_mul_out)

    # GEMM2: Down projection (silu_and_mul_out @ w2.T)
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
        gemm2_out,
        c2_scratch,
    )

    # Reduce: Sum across output dimension

    reshape_out = gemm2_out.view(m, -1, w2.shape[1])
    topk_weights_broadcasted = topk_weights.view(m, -1)

    final_out = torch.zeros(m, w2.shape[1], dtype=torch.float32, device=a.device)

    reduce_sum = get_wave_reduce_sum_kernel(
        reshape_out.shape[0],
        reshape_out.shape[1],
        reshape_out.shape[2],
        tkl.f32,
    )
    reduce_sum(reshape_out, topk_weights_broadcasted, final_out)

    return final_out


num_tokens_values = [32]
n_values = [64]
k_values = [128]
num_experts = [4]
top_ks = [2]
dtypes = [torch.float16]
rtol, atol = 1e-3, 1e-3
block_size_values = [4]


@pytest.mark.parametrize("num_tokens", num_tokens_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("num_experts", num_experts)
@pytest.mark.parametrize("topk", top_ks)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("block_size", block_size_values)
def test_fused_moe(
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
    a = torch.randn(num_tokens, k, dtype=dtype, device=device)
    w1 = torch.randn(num_experts, 2 * n, k, dtype=dtype, device=device)
    w2 = torch.randn(num_experts, k, n, dtype=dtype, device=device)

    score = torch.rand((num_tokens, num_experts), dtype=dtype, device=device)
    ref_output = torch_ref_moe(a, w1, w2, score.clone(), topk)
    tkw_output = tkw_moe(
        a, w1, w2, score.clone(), topk, num_experts, block_size, num_tokens
    )

    # torch.testing.assert_close(
    #     tkw_gemm1_out, ref_gemm1_out, rtol=rtol, atol=atol, msg="GEMM1 output mismatch"
    # )
    # torch.testing.assert_close(
    #     tkw_silu_and_mul_out,
    #     ref_silu_and_mul_out,
    #     rtol=rtol,
    #     atol=atol,
    #     msg="SiLU and Mul output mismatch",
    # )
    # torch.testing.assert_close(
    #     tkw_gemm2_out, ref_gemm2_out, rtol=rtol, atol=atol, msg="GEMM2 output mismatch"
    # )
    torch.testing.assert_close(
        tkw_output, ref_output, rtol=rtol, atol=atol, msg="Final output mismatch"
    )
