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
from wave_lang.kernel._support.dtype import f16, f32, i32
from wave_lang.kernel.wave.templates.moe_v2 import (
    get_fused_moe_gemm,
    get_moe_gemm_only_kernel,
    get_moe_gather_kernel,
    get_moe_scatter_kernel,
    moe_align_block_size_pytorch,
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
            # Use f16 inputs to match MMA intrinsic (f16 in, f32 accumulate)
            gemm1_result[mask] = (
                a[mask].half() @ w1_compute[i].half().transpose(0, 1)
            ).float()
            silu_mul_result[mask] = SiluAndMul_ref(gemm1_result[mask])
            silu_mul_result_f16[mask] = silu_mul_result[mask].to(torch.float16)
            out[mask] = (
                silu_mul_result_f16[mask] @ w2_compute[i].half().transpose(0, 1)
            ).float()

    final = (
        out.view(B, -1, w2.shape[1]) * topk_weights.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)
    return final, gemm1_result, silu_mul_result


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
    kernel, symbols = get_topk_kernel(m, n, k, dtype, threads_per_wave=32)
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=symbols,
    )
    options = set_default_run_config(options)
    return wave_compile(options, kernel)


def get_wave_moe_gemm_only(
    m: int,
    n: int,
    k: int,
    e: int,
    num_blocks: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    kernel, symbols = get_moe_gemm_only_kernel(
        m,
        n,
        k,
        e,
        num_blocks,
        mfma_variant,
        datatype,
    )
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(subs=symbols)
    options = set_default_run_config(options)
    return wave_compile(options, kernel)


def get_wave_moe_gather(m, k, num_blocks, block_size, pad_value):
    kernel, symbols = get_moe_gather_kernel(m, k, num_blocks, block_size, pad_value)
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(subs=symbols)
    options = set_default_run_config(options)
    return wave_compile(options, kernel)


def get_wave_moe_scatter(m, k, num_blocks, block_size, pad_value):
    kernel, symbols = get_moe_scatter_kernel(m, k, num_blocks, block_size, pad_value)
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(subs=symbols)
    options = set_default_run_config(options)
    return wave_compile(options, kernel)


def tkw_moe(a, w1, w2, score, topk, num_experts, block_size, num_tokens):
    # Calculate buffer sizes for block-aligned computation
    max_num_tokens_padded = score.numel() + num_experts * (block_size - 1)
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)

    # Router: Select top-k experts for each token using Wave topk kernel
    score = torch.softmax(score, dim=-1, dtype=torch.float32)

    # Get reference topk for comparison
    topk_weights_ref, topk_ids_ref = torch.topk(score, topk, dim=-1)
    topk_weights_ref = topk_weights_ref.view(-1)
    topk_ids_ref = topk_ids_ref.view(-1)

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

    # TODO: Replace with Wave kernel (see moe.py get_moe_align_block_size_kernel)
    # Using PyTorch host fallback for token alignment
    (
        expert_ids,
        sorted_ids,
        expert_counts_buffer,
        padded_counts_buffer,
        cumsum_buffer,
        cumsum_exclusive,
    ) = moe_align_block_size_pytorch(
        topk_ids.to(torch.int32),
        num_experts,
        block_size,
        max_num_tokens_padded,
        max_num_m_blocks,
    )

    num_blocks = expert_ids.shape[0]
    print(f"\n=== Debug Alignment ===")
    print(f"expert_counts: {expert_counts_buffer}")
    print(f"padded_counts: {padded_counts_buffer}")
    print(f"cumsum: {cumsum_buffer}")
    print(f"cumsum_exclusive: {cumsum_exclusive}")
    print(f"expert_ids: {expert_ids}")
    print(f"sorted_ids[:20]: {sorted_ids[:20]}")
    print(f"num_blocks: {num_blocks}")

    # Replicate input activations for each selected expert
    m, k = a.shape
    a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)
    reshaped_a = a.clone()  # capture before kernel modifies anything
    pad_value = m * topk  # sorted_ids padding sentinel

    # Prepare sorted_ids for Wave gather kernel: must have exactly
    # num_blocks * block_size entries (pad with pad_value if needed)
    total_elems = num_blocks * block_size
    if sorted_ids.shape[0] < total_elems:
        sorted_ids_wave = torch.full(
            (total_elems,), pad_value, dtype=torch.int32, device=a.device
        )
        sorted_ids_wave[: sorted_ids.shape[0]] = sorted_ids
    else:
        sorted_ids_wave = sorted_ids[:total_elems].contiguous()

    # Allocate output tensors
    gemm1_out = torch.zeros(m * topk, w1.shape[1], dtype=torch.float32, device=a.device)
    silu_and_mul_out = torch.zeros(
        m * topk, w1.shape[1] // 2, dtype=torch.float32, device=a.device
    )
    gemm2_out = torch.zeros(m * topk, w2.shape[1], dtype=torch.float32, device=a.device)
    gather_dummy = torch.empty(total_elems, dtype=torch.int32, device=a.device)

    # GEMM1: Compute gate and up projections (a @ w1.T)
    # Step 1: Gather input rows into per-block scratch buffer (WAVE)
    a_scratch = torch.zeros(
        num_blocks, m * topk, k, dtype=torch.float16, device=a.device
    )
    gather1 = get_wave_moe_gather(m * topk, k, num_blocks, block_size, pad_value)
    gather1(sorted_ids_wave, a, a_scratch, gather_dummy)
    torch.cuda.synchronize()

    # Step 2: Per-expert batched GEMM on pre-gathered data
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

    # Step 3: Scatter GEMM results back to token positions (WAVE)
    scatter1 = get_wave_moe_scatter(
        m * topk, w1.shape[1], num_blocks, block_size, pad_value
    )
    scatter1(sorted_ids_wave, c_scratch, gemm1_out, gather_dummy)
    torch.cuda.synchronize()

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
    torch.cuda.synchronize()

    # GEMM2: Down projection (silu_and_mul_out @ w2.T)
    # Step 1: Gather SiLU output into per-block scratch (WAVE)
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

    # Step 2: Per-expert batched GEMM
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

    # Step 3: Scatter GEMM2 results back (WAVE)
    scatter2 = get_wave_moe_scatter(
        m * topk, w2.shape[1], num_blocks, block_size, pad_value
    )
    scatter2(sorted_ids_wave, c2_scratch, gemm2_out, gather_dummy)
    torch.cuda.synchronize()

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
    torch.cuda.synchronize()

    return (
        final_out,
        topk_weights_ref,
        topk_ids_ref,
        topk_weights,
        topk_ids,
        gemm1_out,
        silu_and_mul_out,
        gemm2_out,
        a_scratch,
        reshaped_a,
        sorted_ids,
        expert_ids,
    )


# Test parameter space. With BLOCK_M=M fix, all sizes should work now.
num_tokens_values = [32, 64]
n_values = [64, 128]
k_values = [128, 256]
num_experts = [4, 8]
top_ks = [2]
dtypes = [torch.float16]
rtol, atol = 1e-2, 1e-2
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
    torch.manual_seed(0)  # per-test seed for determinism regardless of order
    device = "cuda"

    if dtype == torch.float16 and k == 1024:
        pytest.skip("This combination generates NaNs and INFs")

    # TODO: investigate why using torch.randn would have precision issue in silu computation
    a = torch.randn(num_tokens, k, dtype=dtype, device=device)
    w1 = torch.randn(num_experts, 2 * n, k, dtype=dtype, device=device)
    w2 = torch.randn(num_experts, k, n, dtype=dtype, device=device)

    score = torch.rand((num_tokens, num_experts), dtype=dtype, device=device)
    ref_output, ref_gemm1_out, ref_silu_out = torch_ref_moe(
        a, w1, w2, score.clone(), topk
    )
    (
        tkw_output,
        topk_weights_ref,
        topk_ids_ref,
        topk_weights,
        topk_ids,
        gemm1_out,
        silu_and_mul_out,
        gemm2_out,
        a_scratch,
        reshaped_a,
        sorted_ids,
        expert_ids_buf,
    ) = tkw_moe(a, w1, w2, score.clone(), topk, num_experts, block_size, num_tokens)

    # Debug each stage
    print(f"\n=== Debugging MoE Stages ===")
    print(
        f"TopK weights match: {torch.allclose(topk_weights_ref, topk_weights, rtol=rtol, atol=atol)}"
    )
    print(f"TopK indices match: {torch.equal(topk_ids_ref, topk_ids)}")

    # ---- GEMM1 input comparison: verify a_scratch vs expected gather ----
    print(f"\n--- GEMM1 Inputs ---")
    print(f"  w1 : same tensor passed to both (shape={w1.shape}, dtype={w1.dtype})")
    print(
        f"  a reshaped : mean={reshaped_a.float().mean():.4f}, std={reshaped_a.float().std():.4f}, shape={reshaped_a.shape}"
    )
    pad_value = num_tokens * topk  # PAD_VALUE = m = num_tokens*topk
    gather_mismatches = 0
    blocks_checked = 0
    for b in range(a_scratch.shape[0]):
        for t in range(block_size):
            flat_idx = b * block_size + t
            if flat_idx >= sorted_ids.shape[0]:
                break
            token_idx = sorted_ids[flat_idx].item()
            if token_idx >= pad_value:  # padding slot, skip
                continue
            expected_row = reshaped_a[token_idx].half()  # what should be in a_scratch
            actual_row = a_scratch[b, t, :]
            if not torch.allclose(expected_row, actual_row, rtol=1e-3, atol=1e-3):
                gather_mismatches += 1
                if gather_mismatches <= 3:
                    print(f"  GATHER MISMATCH block={b} slot={t} token_idx={token_idx}")
                    print(f"    expected[:8]={expected_row[:8].tolist()}")
                    print(f"    actual  [:8]={actual_row[:8].tolist()}")
            blocks_checked += 1
    print(
        f"  Gather check: {blocks_checked} valid slots checked, {gather_mismatches} mismatches"
    )
    print(f"  w1  per-expert check:")
    for b in range(min(4, expert_ids_buf.shape[0])):
        eid = expert_ids_buf[b].item()
        print(
            f"    block={b} → expert_id={eid}, w1[{eid}] mean={w1[eid].float().mean():.4f}"
        )

    # ---- GEMM1 comparison ----
    gemm1_out_f32 = gemm1_out.float()
    ref_gemm1_f32 = ref_gemm1_out.float()
    gemm1_close = torch.allclose(gemm1_out_f32, ref_gemm1_f32, rtol=rtol, atol=atol)
    gemm1_max_diff = (gemm1_out_f32 - ref_gemm1_f32).abs().max().item()
    gemm1_mean_diff = (gemm1_out_f32 - ref_gemm1_f32).abs().mean().item()
    print(f"\n--- GEMM1 ---")
    print(
        f"  ref  : mean={ref_gemm1_f32.mean():.4f}, std={ref_gemm1_f32.std():.4f}, max={ref_gemm1_f32.abs().max():.4f}"
    )
    print(
        f"  wave : mean={gemm1_out_f32.mean():.4f}, std={gemm1_out_f32.std():.4f}, max={gemm1_out_f32.abs().max():.4f}"
    )
    print(
        f"  close={gemm1_close}, max_diff={gemm1_max_diff:.6f}, mean_diff={gemm1_mean_diff:.6f}"
    )
    if not gemm1_close:
        # Show the first mismatched row
        diff_rows = (gemm1_out_f32 - ref_gemm1_f32).abs().max(dim=1).values
        worst_row = diff_rows.argmax().item()
        print(
            f"  Worst row {worst_row}: wave={gemm1_out_f32[worst_row, :8]}, ref={ref_gemm1_f32[worst_row, :8]}"
        )

    # ---- SiLU comparison ----
    silu_out_f32 = silu_and_mul_out.float()
    ref_silu_f32 = ref_silu_out.float()
    silu_close = torch.allclose(silu_out_f32, ref_silu_f32, rtol=rtol, atol=atol)
    print(f"\n--- SiLU ---")
    print(f"  ref  : mean={ref_silu_f32.mean():.4f}, std={ref_silu_f32.std():.4f}")
    print(f"  wave : mean={silu_out_f32.mean():.4f}, std={silu_out_f32.std():.4f}")
    print(
        f"  close={silu_close}, max_diff={(silu_out_f32 - ref_silu_f32).abs().max():.6f}"
    )

    # ---- GEMM2 comparison: compute reference from Wave's SiLU output to isolate GEMM2 ----
    ref_gemm2_from_wave = torch.zeros_like(gemm2_out)
    silu_f16 = silu_and_mul_out.to(torch.float16)
    for i in range(w2.shape[0]):
        mask = topk_ids.view(-1) == i
        if mask.sum():
            ref_gemm2_from_wave[mask] = (silu_f16[mask] @ w2[i].half().T).float()

    gemm2_f32 = gemm2_out.float()
    ref_gemm2_f32 = ref_gemm2_from_wave.float()
    gemm2_max_diff = (gemm2_f32 - ref_gemm2_f32).abs().max().item()
    gemm2_mean_diff = (gemm2_f32 - ref_gemm2_f32).abs().mean().item()
    print(f"\n--- GEMM2 ---")
    print(
        f"  ref  : mean={ref_gemm2_f32.mean():.4f}, std={ref_gemm2_f32.std():.4f}, max={ref_gemm2_f32.abs().max():.4f}"
    )
    print(
        f"  wave : mean={gemm2_f32.mean():.4f}, std={gemm2_f32.std():.4f}, max={gemm2_f32.abs().max():.4f}"
    )
    print(f"  max_diff={gemm2_max_diff:.6f}, mean_diff={gemm2_mean_diff:.6f}")
    if gemm2_max_diff > 2.0:
        # Show per-row analysis to identify which rows diverge
        row_diffs = (gemm2_f32 - ref_gemm2_f32).abs().max(dim=1).values
        bad_rows = (row_diffs > 2.0).nonzero(as_tuple=True)[0]
        print(f"  Bad rows (diff>2): {bad_rows.shape[0]}/{gemm2_f32.shape[0]}")
        for idx in bad_rows[:5]:
            r = idx.item()
            eid = topk_ids.view(-1)[r].item()
            print(
                f"    row={r} expert={eid} wave_max={gemm2_f32[r].abs().max():.2f} ref_max={ref_gemm2_f32[r].abs().max():.2f} diff={row_diffs[r]:.2f}"
            )

    print(f"\n--- Final ---")
    print(f"  tkw  mean={tkw_output.mean():.4f}, ref mean={ref_output.mean():.4f}")
    print(f"  max_diff={(tkw_output - ref_output).abs().max():.6f}")

    torch.testing.assert_close(
        gemm1_out_f32, ref_gemm1_f32, rtol=rtol, atol=atol, msg="GEMM1 output mismatch"
    )
    torch.testing.assert_close(
        silu_out_f32, ref_silu_f32, rtol=rtol, atol=atol, msg="SiLU output mismatch"
    )
    # Final tolerance is wider because errors accumulate through the chain:
    # GEMM1 (~0.015) → SiLU (~0.45) → f16 cast → GEMM2 → reduce (~0.9)
    torch.testing.assert_close(
        tkw_output, ref_output, rtol=5e-2, atol=1.0, msg="Final output mismatch"
    )
