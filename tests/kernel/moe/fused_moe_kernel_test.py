# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
    get_arch_family,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang import DataType
from wave_lang.kernel._support.dtype import f16
from wave_lang.kernel.wave.templates.moe import (
    compile_moe_align_kernels,
    get_fused_silu_block_space_kernel,
    get_moe_gather_kernel,
    get_moe_gemm_only_kernel,
    get_moe_histogram_kernel,
    get_moe_pad_cumsum_kernel,
    get_moe_reduce_sum_kernel,
    get_moe_scatter_kernel,
    get_silu_and_mul_kernel,
    get_topk_kernel,
    moe_align_block_size,
)
import torch.nn.functional as F

# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false

torch.manual_seed(0)


def silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
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
    Reference MoE implementation.
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
            # Use f16 inputs to match MMA intrinsic (f16 in, f32 accumulate)
            gemm1_result[mask] = (
                a[mask].half() @ w1_compute[i].half().transpose(0, 1)
            ).float()
            silu_mul_result[mask] = silu_and_mul_ref(gemm1_result[mask])
            silu_mul_result_f16[mask] = silu_mul_result[mask].to(torch.float16)
            out[mask] = (
                silu_mul_result_f16[mask] @ w2_compute[i].half().transpose(0, 1)
            ).float()

    final = (
        out.view(B, -1, w2.shape[1]) * topk_weights.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)
    return final, gemm1_result, silu_mul_result


def _compile_kernel(kernel, symbols):
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(subs=symbols)
    options = set_default_run_config(options)
    return wave_compile(options, kernel)


def get_wave_silu_and_mul_kernel(m: int, n: int, dtype: DataType):
    return _compile_kernel(*get_silu_and_mul_kernel(m, n, dtype))


def get_wave_reduce_sum_kernel(b: int, k: int, d: int, dtype: DataType):
    return _compile_kernel(*get_moe_reduce_sum_kernel(b, k, d, dtype))


def get_wave_topk_kernel(m: int, n: int, k: int, dtype: DataType):
    return _compile_kernel(*get_topk_kernel(m, n, k, dtype, threads_per_wave=32))


def get_wave_moe_gemm_only(
    m: int,
    n: int,
    k: int,
    e: int,
    num_blocks: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    return _compile_kernel(
        *get_moe_gemm_only_kernel(m, n, k, e, num_blocks, mfma_variant, datatype)
    )


def get_wave_moe_gather(m, k, num_blocks, block_size, pad_value):
    return _compile_kernel(
        *get_moe_gather_kernel(m, k, num_blocks, block_size, pad_value)
    )


def get_wave_moe_scatter(m, k, num_blocks, block_size, pad_value):
    return _compile_kernel(
        *get_moe_scatter_kernel(m, k, num_blocks, block_size, pad_value)
    )


def get_wave_fused_silu_block_space(num_blocks, m, n, dtype):
    return _compile_kernel(*get_fused_silu_block_space_kernel(num_blocks, m, n, dtype))


def tkw_moe(a, w1, w2, score, topk, num_experts, block_size, num_tokens):
    max_num_tokens_padded = score.numel() + num_experts * (block_size - 1)
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)

    # Top-k routing
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights = torch.zeros((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_kernel = get_wave_topk_kernel(num_tokens, num_experts, topk, tkl.f32)
    topk_kernel(score, topk_weights, topk_ids)
    topk_weights = topk_weights.view(-1)
    topk_ids = topk_ids.view(-1)

    # Block alignment: histogram → pad+cumsum → expert_ids → sorted_ids
    numel = num_tokens * topk
    align_kernels = compile_moe_align_kernels(
        numel,
        num_experts,
        block_size,
        max_num_tokens_padded,
        max_num_m_blocks,
    )
    expert_ids, sorted_ids, *_ = moe_align_block_size(
        topk_ids.to(torch.int32),
        num_experts,
        block_size,
        max_num_tokens_padded,
        max_num_m_blocks,
        compiled_kernels=align_kernels,
    )
    num_blocks = expert_ids.shape[0]

    # Replicate input tokens for each selected expert
    m, k = a.shape
    a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)
    pad_value = m * topk  # sorted_ids padding sentinel

    # Pad sorted_ids to num_blocks * block_size (required by gather kernel)
    total_elems = num_blocks * block_size
    if sorted_ids.shape[0] < total_elems:
        sorted_ids_wave = torch.full(
            (total_elems,), pad_value, dtype=torch.int32, device=a.device
        )
        sorted_ids_wave[: sorted_ids.shape[0]] = sorted_ids
    else:
        sorted_ids_wave = sorted_ids[:total_elems].contiguous()

    gemm1_out = torch.zeros(m * topk, w1.shape[1], dtype=torch.float32, device=a.device)
    silu_and_mul_out = torch.zeros(
        m * topk, w1.shape[1] // 2, dtype=torch.float32, device=a.device
    )
    gemm2_out = torch.zeros(m * topk, w2.shape[1], dtype=torch.float32, device=a.device)
    gather_dummy = torch.empty(total_elems, dtype=torch.int32, device=a.device)

    # GEMM1: gather → batched GEMM (a @ w1.T) → scatter
    a_scratch = torch.zeros(
        num_blocks, m * topk, k, dtype=torch.float16, device=a.device
    )
    gather1 = get_wave_moe_gather(m * topk, k, num_blocks, block_size, pad_value)
    gather1(sorted_ids_wave, a, a_scratch, gather_dummy)
    torch.cuda.synchronize()

    c_scratch = torch.zeros(
        num_blocks, m * topk, w1.shape[1], dtype=torch.float32, device=a.device
    )
    gemm1 = get_wave_moe_gemm_only(
        m * topk,
        w1.shape[1],
        k,
        w1.shape[0],
        num_blocks,
        MMAType.RDNA4_WAVE32_F32_16x16x16_F16,
        f16,
    )
    gemm1(a_scratch, w1, expert_ids, c_scratch)
    torch.cuda.synchronize()

    scatter1 = get_wave_moe_scatter(
        m * topk, w1.shape[1], num_blocks, block_size, pad_value
    )
    scatter1(sorted_ids_wave, c_scratch, gemm1_out, gather_dummy)
    torch.cuda.synchronize()

    # SiLU & Mul activation
    silu_and_mul = get_wave_silu_and_mul_kernel(
        gemm1_out.shape[0], gemm1_out.shape[1] // 2, tkl.f32
    )
    silu_and_mul(gemm1_out, silu_and_mul_out)
    torch.cuda.synchronize()

    # GEMM2: gather → batched GEMM (silu_out @ w2.T) → scatter
    silu_and_mul_out_f16 = silu_and_mul_out.to(torch.float16)
    a2_scratch = torch.zeros(
        num_blocks,
        m * topk,
        silu_and_mul_out.shape[1],
        dtype=torch.float16,
        device=a.device,
    )
    gather2 = get_wave_moe_gather(
        m * topk, silu_and_mul_out.shape[1], num_blocks, block_size, pad_value
    )
    gather2(sorted_ids_wave, silu_and_mul_out_f16, a2_scratch, gather_dummy)
    torch.cuda.synchronize()

    c2_scratch = torch.zeros(
        num_blocks,
        m * topk,
        w2.shape[1],
        dtype=torch.float32,
        device=a.device,
    )
    gemm2 = get_wave_moe_gemm_only(
        m * topk,
        w2.shape[1],
        silu_and_mul_out.shape[1],
        w2.shape[0],
        num_blocks,
        MMAType.RDNA4_WAVE32_F32_16x16x16_F16,
        f16,
    )
    gemm2(a2_scratch, w2, expert_ids, c2_scratch)
    torch.cuda.synchronize()

    scatter2 = get_wave_moe_scatter(
        m * topk, w2.shape[1], num_blocks, block_size, pad_value
    )
    scatter2(sorted_ids_wave, c2_scratch, gemm2_out, gather_dummy)
    torch.cuda.synchronize()

    # Weighted reduce sum across experts
    reshape_out = gemm2_out.view(m, -1, w2.shape[1])
    final_out = torch.zeros(m, w2.shape[1], dtype=torch.float32, device=a.device)
    reduce_sum = get_wave_reduce_sum_kernel(
        reshape_out.shape[0], reshape_out.shape[1], reshape_out.shape[2], tkl.f32
    )
    reduce_sum(reshape_out, topk_weights.view(m, -1), final_out)
    torch.cuda.synchronize()

    return final_out, gemm1_out, silu_and_mul_out


def tkw_moe_fused_silu(a, w1, w2, score, topk, num_experts, block_size, num_tokens):
    """
    MoE pipeline with Scatter1 + SiLU + Gather2 fused into a single kernel.

    Replaces the three-kernel sequence:
        scatter1(c_scratch → gemm1_out)
        silu_and_mul(gemm1_out → silu_out)
        gather2(silu_out → a2_scratch)
    with a single fused kernel that applies SiLU directly in block/slot space:
        fused_silu_block_space(c_scratch → a2_scratch)

    No sorted_ids lookup is needed in the fused kernel because scatter and
    gather use the same sorted_ids mapping — they cancel each other out.
    """
    max_num_tokens_padded = score.numel() + num_experts * (block_size - 1)
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)

    # Top-k routing
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights = torch.zeros((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_kernel = get_wave_topk_kernel(num_tokens, num_experts, topk, tkl.f32)
    topk_kernel(score, topk_weights, topk_ids)
    topk_weights = topk_weights.view(-1)
    topk_ids = topk_ids.view(-1)

    # Block alignment: histogram → pad+cumsum → expert_ids → sorted_ids
    numel = num_tokens * topk
    align_kernels = compile_moe_align_kernels(
        numel,
        num_experts,
        block_size,
        max_num_tokens_padded,
        max_num_m_blocks,
    )
    expert_ids, sorted_ids, *_ = moe_align_block_size(
        topk_ids.to(torch.int32),
        num_experts,
        block_size,
        max_num_tokens_padded,
        max_num_m_blocks,
        compiled_kernels=align_kernels,
    )
    num_blocks = expert_ids.shape[0]

    # Replicate input tokens for each selected expert
    m, k = a.shape
    a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)
    pad_value = m * topk

    # Pad sorted_ids to num_blocks * block_size
    total_elems = num_blocks * block_size
    if sorted_ids.shape[0] < total_elems:
        sorted_ids_wave = torch.full(
            (total_elems,), pad_value, dtype=torch.int32, device=a.device
        )
        sorted_ids_wave[: sorted_ids.shape[0]] = sorted_ids
    else:
        sorted_ids_wave = sorted_ids[:total_elems].contiguous()

    gather_dummy = torch.empty(total_elems, dtype=torch.int32, device=a.device)

    # GEMM1: gather → batched GEMM (a @ w1.T)
    a_scratch = torch.zeros(
        num_blocks, m * topk, k, dtype=torch.float16, device=a.device
    )
    gather1 = get_wave_moe_gather(m * topk, k, num_blocks, block_size, pad_value)
    gather1(sorted_ids_wave, a, a_scratch, gather_dummy)
    torch.cuda.synchronize()

    c_scratch = torch.zeros(
        num_blocks, m * topk, w1.shape[1], dtype=torch.float32, device=a.device
    )
    gemm1 = get_wave_moe_gemm_only(
        m * topk,
        w1.shape[1],
        k,
        w1.shape[0],
        num_blocks,
        MMAType.RDNA4_WAVE32_F32_16x16x16_F16,
        f16,
    )
    gemm1(a_scratch, w1, expert_ids, c_scratch)
    torch.cuda.synchronize()

    # Fused: SiLU applied in block/slot space (replaces scatter1 + silu + gather2)
    # c_scratch: [num_blocks, m*topk, 2n] f32 → a2_scratch: [num_blocks, m*topk, n] f16
    a2_scratch = torch.zeros(
        num_blocks, m * topk, w1.shape[1] // 2, dtype=torch.float16, device=a.device
    )
    fused_silu = get_wave_fused_silu_block_space(
        num_blocks, m * topk, w1.shape[1] // 2, tkl.f32
    )
    fused_silu(c_scratch, a2_scratch)
    torch.cuda.synchronize()

    # GEMM2: batched GEMM (silu_out @ w2.T) → scatter
    gemm2_out = torch.zeros(m * topk, w2.shape[1], dtype=torch.float32, device=a.device)
    c2_scratch = torch.zeros(
        num_blocks, m * topk, w2.shape[1], dtype=torch.float32, device=a.device
    )
    gemm2 = get_wave_moe_gemm_only(
        m * topk,
        w2.shape[1],
        w1.shape[1] // 2,
        w2.shape[0],
        num_blocks,
        MMAType.RDNA4_WAVE32_F32_16x16x16_F16,
        f16,
    )
    gemm2(a2_scratch, w2, expert_ids, c2_scratch)
    torch.cuda.synchronize()

    scatter2 = get_wave_moe_scatter(
        m * topk, w2.shape[1], num_blocks, block_size, pad_value
    )
    scatter2(sorted_ids_wave, c2_scratch, gemm2_out, gather_dummy)
    torch.cuda.synchronize()

    # Weighted reduce sum across experts
    reshape_out = gemm2_out.view(m, -1, w2.shape[1])
    final_out = torch.zeros(m, w2.shape[1], dtype=torch.float32, device=a.device)
    reduce_sum = get_wave_reduce_sum_kernel(
        reshape_out.shape[0], reshape_out.shape[1], reshape_out.shape[2], tkl.f32
    )
    reduce_sum(reshape_out, topk_weights.view(m, -1), final_out)
    torch.cuda.synchronize()

    return final_out


_SKIP_RDNA = pytest.mark.skipif(
    get_arch_family() != "RDNA", reason="Requires RDNA4 GPU (gfx120x)"
)

# ---------------------------------------------------------------------------
# Unit tests — smallest kernels first
# ---------------------------------------------------------------------------


@_SKIP_RDNA
def test_moe_histogram():
    # numel must be >= HIST_BLOCK (32) to avoid partial-workgroup OOB reads.
    # 32 elements: each of 4 experts appears exactly 8 times.
    topk_ids = torch.tensor([0, 1, 2, 3] * 8, dtype=torch.int32, device="cuda")
    expert_counts = torch.zeros(4, dtype=torch.int32, device="cuda")
    dummy = torch.empty(32, dtype=torch.int32, device="cuda")
    _compile_kernel(*get_moe_histogram_kernel(32, 4))(topk_ids, expert_counts, dummy)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        expert_counts, torch.tensor([8, 8, 8, 8], dtype=torch.int32, device="cuda")
    )


@_SKIP_RDNA
def test_moe_pad_cumsum():
    expert_counts = torch.tensor([3, 2, 5, 1], dtype=torch.int32, device="cuda")
    padded = torch.zeros(4, dtype=torch.int32, device="cuda")
    cumsum = torch.zeros(4, dtype=torch.int32, device="cuda")
    cum_excl = torch.zeros(4, dtype=torch.int32, device="cuda")
    dummy = torch.empty(4, dtype=torch.int32, device="cuda")
    _compile_kernel(*get_moe_pad_cumsum_kernel(4, 4))(
        expert_counts, padded, cumsum, cum_excl, dummy
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        padded, torch.tensor([4, 4, 8, 4], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        cumsum, torch.tensor([4, 8, 16, 20], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        cum_excl, torch.tensor([0, 4, 8, 16], dtype=torch.int32, device="cuda")
    )


@_SKIP_RDNA
def test_moe_silu_and_mul():
    m, n = 8, 32
    x = torch.randn(m, 2 * n, dtype=torch.float32, device="cuda")
    out = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    get_wave_silu_and_mul_kernel(m, n, tkl.f32)(x, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, F.silu(x[:, :n]) * x[:, n:], rtol=1e-3, atol=1e-3)


@_SKIP_RDNA
def test_fused_silu_block_space():
    num_blocks, m, n = 4, 8, 32
    c_back = torch.randn(num_blocks, m, 2 * n, dtype=torch.float32, device="cuda")
    a_out = torch.zeros(num_blocks, m, n, dtype=torch.float16, device="cuda")
    get_wave_fused_silu_block_space(num_blocks, m, n, tkl.f32)(c_back, a_out)
    torch.cuda.synchronize()
    ref = (F.silu(c_back[..., :n]) * c_back[..., n:]).half()
    torch.testing.assert_close(a_out, ref, rtol=1e-2, atol=1e-2)


@_SKIP_RDNA
def test_moe_gather():
    m, k, num_blocks, block_size = 8, 64, 2, 4
    pad_value = m
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    sorted_ids = torch.tensor(
        [0, 2, 4, 1, 3, 5, m, m], dtype=torch.int32, device="cuda"
    )
    a_back = torch.zeros(num_blocks, m, k, dtype=torch.float16, device="cuda")
    dummy = torch.empty(num_blocks * block_size, dtype=torch.int32, device="cuda")
    get_wave_moe_gather(m, k, num_blocks, block_size, pad_value)(
        sorted_ids, a, a_back, dummy
    )
    torch.cuda.synchronize()
    ids = sorted_ids.tolist()
    for blk in range(num_blocks):
        for slot in range(block_size):
            idx = ids[blk * block_size + slot]
            if idx < pad_value:
                torch.testing.assert_close(a_back[blk, slot], a[idx])


@_SKIP_RDNA
def test_moe_scatter():
    m, k, num_blocks, block_size = 8, 64, 2, 4
    pad_value = m
    c_back = torch.randn(num_blocks, m, k, dtype=torch.float32, device="cuda")
    sorted_ids = torch.tensor(
        [0, 2, 4, 1, 3, 5, m, m], dtype=torch.int32, device="cuda"
    )
    output = torch.zeros(m, k, dtype=torch.float32, device="cuda")
    dummy = torch.empty(num_blocks * block_size, dtype=torch.int32, device="cuda")
    get_wave_moe_scatter(m, k, num_blocks, block_size, pad_value)(
        sorted_ids, c_back, output, dummy
    )
    torch.cuda.synchronize()
    ids = sorted_ids.tolist()
    for blk in range(num_blocks):
        for slot in range(block_size):
            idx = ids[blk * block_size + slot]
            if idx < pad_value:
                torch.testing.assert_close(output[idx], c_back[blk, slot])


@_SKIP_RDNA
def test_moe_gemm_only():
    m, n, k, e, num_blocks = 32, 64, 64, 2, 2
    a_back = torch.randn(num_blocks, m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(e, n, k, dtype=torch.float16, device="cuda")
    expert_ids = torch.zeros(num_blocks, dtype=torch.int32, device="cuda")
    c_back = torch.zeros(num_blocks, m, n, dtype=torch.float32, device="cuda")
    get_wave_moe_gemm_only(
        m, n, k, e, num_blocks, MMAType.RDNA4_WAVE32_F32_16x16x16_F16, f16
    )(a_back, b, expert_ids, c_back)
    torch.cuda.synchronize()
    for blk in range(num_blocks):
        ref = a_back[blk].float() @ b[expert_ids[blk].item()].float().T
        torch.testing.assert_close(c_back[blk, :m], ref, rtol=2e-2, atol=2e-2)


@_SKIP_RDNA
def test_moe_reduce_sum():
    b, topk, d = 4, 2, 32
    a = torch.randn(b, topk, d, dtype=torch.float32, device="cuda")
    weights = torch.rand(b, topk, dtype=torch.float32, device="cuda")
    c = torch.zeros(b, d, dtype=torch.float32, device="cuda")
    get_wave_reduce_sum_kernel(b, topk, d, tkl.f32)(a, weights, c)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        c, (a * weights.unsqueeze(-1)).sum(dim=1), rtol=1e-3, atol=1e-3
    )


@_SKIP_RDNA
def test_moe_align_block_size():
    num_experts, block_size = 4, 4
    # 16 tokens × topk=2 = 32 elements (>= HIST_BLOCK=32): each expert gets 8.
    topk_ids = torch.tensor([0, 1, 2, 3] * 8, dtype=torch.int32, device="cuda")
    numel = topk_ids.numel()
    max_np = numel + num_experts * (block_size - 1)
    max_nb = -(max_np // -block_size)
    aks = compile_moe_align_kernels(numel, num_experts, block_size, max_np, max_nb)
    expert_ids_out, sorted_ids_out, expert_counts, *_ = moe_align_block_size(
        topk_ids, num_experts, block_size, max_np, max_nb, compiled_kernels=aks
    )
    torch.testing.assert_close(
        expert_counts,
        torch.full((num_experts,), 8, dtype=torch.int32, device="cuda"),
    )
    valid = sorted_ids_out[sorted_ids_out < numel]
    assert valid.numel() == numel, "sorted_ids should contain all token indices"


# ---------------------------------------------------------------------------
# Integration test parameters
# ---------------------------------------------------------------------------

num_tokens_values = [32, 64]
n_values = [64, 128]
k_values = [128, 256]
num_experts_values = [4, 8]
top_ks = [2]
dtypes = [torch.float16]
block_size_values = [4]


@_SKIP_RDNA
@pytest.mark.parametrize("num_tokens", num_tokens_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("num_experts", num_experts_values)
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
    torch.manual_seed(0)  # per-test seed for determinism regardless of order
    device = "cuda"

    # Scale tolerance with k: FP16 accumulation error grows with reduction length.
    # Older ROCm drivers in CI produce ~2x more rounding error than current hardware.
    rtol = atol = 2e-2 * (k / 128)  # 2e-2 at k=128, 4e-2 at k=256

    # TODO: investigate why torch.randn has precision issues in silu computation
    a = torch.randn(num_tokens, k, dtype=dtype, device=device)
    w1 = torch.randn(num_experts, 2 * n, k, dtype=dtype, device=device)
    w2 = torch.randn(num_experts, k, n, dtype=dtype, device=device)
    score = torch.rand((num_tokens, num_experts), dtype=dtype, device=device)

    ref_output, ref_gemm1_out, ref_silu_out = torch_ref_moe(
        a, w1, w2, score.clone(), topk
    )
    tkw_output, gemm1_out, silu_out = tkw_moe(
        a, w1, w2, score.clone(), topk, num_experts, block_size, num_tokens
    )

    torch.testing.assert_close(
        gemm1_out.float(),
        ref_gemm1_out.float(),
        rtol=rtol,
        atol=atol,
        msg="GEMM1 output mismatch",
    )
    torch.testing.assert_close(
        silu_out.float(),
        ref_silu_out.float(),
        rtol=rtol,
        atol=atol,
        msg="SiLU output mismatch",
    )
    # Final tolerance is wider: errors accumulate through GEMM1 → SiLU → f16 cast → GEMM2 → reduce.
    torch.testing.assert_close(
        tkw_output,
        ref_output,
        rtol=5e-2,
        atol=2.0 * (k / 128),
        msg="Final output mismatch",
    )


@_SKIP_RDNA
@pytest.mark.parametrize("num_tokens", num_tokens_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("num_experts", num_experts_values)
@pytest.mark.parametrize("topk", top_ks)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("block_size", block_size_values)
def test_fused_moe_silu_fusion(
    num_tokens: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    dtype: DataType,
    block_size: int,
):
    """
    Tests the fused Scatter1+SiLU+Gather2 kernel against the reference.

    The fused kernel applies SiLU directly in block/slot space, eliminating
    three kernel launches and two large intermediate buffers compared to the
    unfused pipeline.  Only the final output is compared (intermediate
    gemm1_out and silu_out tensors are not produced by this path).
    """
    torch.manual_seed(0)
    device = "cuda"

    a = torch.randn(num_tokens, k, dtype=dtype, device=device)
    w1 = torch.randn(num_experts, 2 * n, k, dtype=dtype, device=device)
    w2 = torch.randn(num_experts, k, n, dtype=dtype, device=device)
    score = torch.rand((num_tokens, num_experts), dtype=dtype, device=device)

    ref_output, _, _ = torch_ref_moe(a, w1, w2, score.clone(), topk)
    tkw_output = tkw_moe_fused_silu(
        a, w1, w2, score.clone(), topk, num_experts, block_size, num_tokens
    )

    # Same tolerance as the original test
    torch.testing.assert_close(
        tkw_output,
        ref_output,
        rtol=5e-2,
        atol=2.0 * (k / 128),
        msg="Fused SiLU final output mismatch",
    )
