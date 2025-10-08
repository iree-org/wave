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
    get_moe_reduce_sum_kernel,
)

from tests.kernel.wave.moe.moe_align_block_size_test import (
    moe_align_block_size_pytorch,
)

import math

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
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
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

    return gemm1_result, silu_mul_result, out, out.sum(dim=1)


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


def tkw_moe(a, w1, w2, score, topk, num_experts, block_size, num_tokens):
    # based on the score, create sorted_ids and expert_ids for each aligned block
    max_num_tokens_padded = score.numel() + num_experts * (block_size - 1)
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)

    # TODO: replace with topk kernel implemented in Wave
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    moe_align_block_size, hyperparams, dynamic_symbols = (
        get_moe_align_block_size_kernel(
            num_tokens,
            num_experts,
            block_size,
            topk_ids.numel(),
            max_num_m_blocks,
            max_num_tokens_padded,
            topk,
        )
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        minimize_shared_allocs=False,
    )

    moe_align_block_size = wave_compile(
        options,
        moe_align_block_size,
    )

    expert_counts_buffer = torch.randint(
        size=(num_experts,), dtype=torch.int32, device="cuda", low=0, high=1
    )
    padded_counts_buffer = torch.randint(
        size=(num_experts,), dtype=torch.int32, device="cuda", low=0, high=1
    )
    cumsum_buffer = torch.randint(
        size=(num_experts,), dtype=torch.int32, device="cuda", low=0, high=1
    )
    cumsum_exclusive = torch.randint(
        size=(num_experts,), dtype=torch.int32, device="cuda", low=0, high=1
    )
    num_blocks_buffer = torch.randint(
        size=(num_experts,), dtype=torch.int32, device="cuda", low=0, high=1
    )

    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
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
    num_tokens_post_pad = cumsum_buffer[-1]

    # now do the gemm
    m, k = a.shape
    a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)
    gemm1_out = torch.zeros(m * topk, w1.shape[1], dtype=torch.float32, device=a.device)
    silu_and_mul_out = torch.zeros(
        m * topk, w1.shape[1] // 2, dtype=torch.float32, device=a.device
    )
    # Final output tensor - matches w2.shape[1] (final output dimension)
    gemm2_out = torch.zeros(m * topk, w2.shape[1], dtype=torch.float32, device=a.device)

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
        gemm2_out,
        c2_scratch,
    )

    final_out = torch.zeros(m * topk, dtype=torch.float32, device=a.device)

    reduce_sum, symbols = get_moe_reduce_sum_kernel(
        m * topk,
        w2.shape[1],
        tkl.f32,
    )
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=symbols,
    )
    options = set_default_run_config(options)

    reduce_sum = wave_compile(options, reduce_sum)

    reduce_sum(gemm2_out, final_out)

    return gemm1_out, silu_and_mul_out, gemm2_out, final_out


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
    a = torch.randn(num_tokens, k, dtype=dtype, device=device)
    w1 = torch.randn(num_experts, 2 * n, k, dtype=dtype, device=device)
    w2 = torch.randn(num_experts, k, n, dtype=dtype, device=device)

    score = torch.rand((num_tokens, num_experts), dtype=dtype, device=device)
    [ref_gemm1_out, ref_silu_and_mul_out, ref_gemm2_out, ref_output] = torch_ref_moe(
        a, w1, w2, score.clone(), topk
    )
    [tkw_gemm1_out, tkw_silu_and_mul_out, tkw_gemm2_out, tkw_output] = tkw_moe(
        a, w1, w2, score.clone(), topk, num_experts, block_size, num_tokens
    )

    print(tkw_output)
    print(ref_output)

    torch.testing.assert_close(
        tkw_gemm1_out, ref_gemm1_out, rtol=rtol, atol=atol, msg="GEMM1 output mismatch"
    )
    torch.testing.assert_close(
        tkw_silu_and_mul_out,
        ref_silu_and_mul_out,
        rtol=rtol,
        atol=atol,
        msg="SiLU and Mul output mismatch",
    )
    torch.testing.assert_close(
        tkw_gemm2_out, ref_gemm2_out, rtol=rtol, atol=atol, msg="GEMM2 output mismatch"
    )
    torch.testing.assert_close(
        tkw_output, ref_output, rtol=rtol, atol=atol, msg="Final output mismatch"
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
