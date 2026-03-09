#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Benchmark for the Wave MoE multi-kernel pipeline.

Measures per-kernel and end-to-end latency for realistic MoE configurations
modeled after production architectures (Mixtral, DBRX, DeepSeek-V2-Lite).

Usage:
    python -m tests.kernel.moe.moe_benchmark
    python -m tests.kernel.moe.moe_benchmark --config mixtral
    python -m tests.kernel.moe.moe_benchmark --config all --warmup 5 --iters 20
"""

import argparse
import time
from dataclasses import dataclass

import torch
import wave_lang.kernel.lang as tkl
from wave_lang.kernel._support.dtype import f16, f32
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.templates.moe import (
    get_moe_gemm_only_kernel,
    get_moe_gather_kernel,
    get_moe_scatter_kernel,
    get_moe_reduce_sum_kernel,
    get_silu_and_mul_kernel,
    get_topk_kernel,
    compile_moe_align_kernels,
    moe_align_block_size,
)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
@dataclass
class MoEConfig:
    """Hyperparameters for a single MoE benchmark configuration."""

    name: str
    num_tokens: int  # Batch size (sequence positions)
    num_experts: int
    topk: int
    hidden_dim: int  # k — input/output hidden dimension
    intermediate_dim: int  # n — FFN intermediate dimension
    block_size: int  # Alignment block size


# Real-world MoE configurations, scaled down where needed to fit RDNA4
# constraints (BLOCK_M <= 64, i.e. m = num_tokens * topk <= 64).
CONFIGS = {
    # Mixtral 8x7B-like: 8 experts, top-2, hidden=4096, intermediate=14336
    # Scaled to fit RDNA4: num_tokens=32 (m=64), dims /16 for memory
    "mixtral": MoEConfig(
        name="Mixtral-8x7B (scaled)",
        num_tokens=32,
        num_experts=8,
        topk=2,
        hidden_dim=256,
        intermediate_dim=896,
        block_size=4,
    ),
    # DBRX-like: 16 experts, top-4
    # num_tokens=16 so m=16*4=64
    "dbrx": MoEConfig(
        name="DBRX (scaled)",
        num_tokens=16,
        num_experts=16,
        topk=4,
        hidden_dim=256,
        intermediate_dim=512,
        block_size=4,
    ),
    # DeepSeek-V2-Lite-like: 64 experts, top-6
    # num_tokens=8 so m=8*6=48, smaller dims for 64 experts
    "deepseek": MoEConfig(
        name="DeepSeek-V2-Lite (scaled)",
        num_tokens=8,
        num_experts=64,
        topk=6,
        hidden_dim=128,
        intermediate_dim=256,
        block_size=4,
    ),
    # Small config for quick smoke testing
    "small": MoEConfig(
        name="Small (test)",
        num_tokens=32,
        num_experts=4,
        topk=2,
        hidden_dim=128,
        intermediate_dim=64,
        block_size=4,
    ),
    # Larger batch — pushes more tokens through the pipeline
    "large_batch": MoEConfig(
        name="Large batch",
        num_tokens=64,
        num_experts=8,
        topk=2,
        hidden_dim=256,
        intermediate_dim=512,
        block_size=4,
    ),
}


# ---------------------------------------------------------------------------
# Compile helpers
# ---------------------------------------------------------------------------
def _compile(kernel_fn, symbols):
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(subs=symbols)
    options = set_default_run_config(options)
    return wave_compile(options, kernel_fn)


def compile_topk(num_tokens, num_experts, topk):
    kernel, symbols = get_topk_kernel(
        num_tokens, num_experts, topk, tkl.f32, threads_per_wave=32
    )
    return _compile(kernel, symbols)


def compile_gather(m, k, num_blocks, block_size, pad_value):
    kernel, symbols = get_moe_gather_kernel(m, k, num_blocks, block_size, pad_value)
    return _compile(kernel, symbols)


def compile_scatter(m, k, num_blocks, block_size, pad_value):
    kernel, symbols = get_moe_scatter_kernel(m, k, num_blocks, block_size, pad_value)
    return _compile(kernel, symbols)


def compile_gemm(m, n, k, e, num_blocks):
    kernel, symbols = get_moe_gemm_only_kernel(
        m,
        n,
        k,
        e,
        num_blocks,
        MMAType.RDNA4_WAVE32_F32_16x16x16_F16,
        f16,
    )
    return _compile(kernel, symbols)


def compile_silu(m, n):
    kernel, symbols = get_silu_and_mul_kernel(m, n, tkl.f32)
    return _compile(kernel, symbols)


def compile_reduce(b, topk, d):
    kernel, symbols = get_moe_reduce_sum_kernel(b, topk, d, tkl.f32)
    return _compile(kernel, symbols)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def benchmark_moe(cfg: MoEConfig, warmup: int = 3, iters: int = 10):
    """Run full MoE pipeline benchmark for a given configuration."""

    device = "cuda"
    torch.manual_seed(0)

    num_tokens = cfg.num_tokens
    num_experts = cfg.num_experts
    topk = cfg.topk
    k = cfg.hidden_dim
    n = cfg.intermediate_dim
    block_size = cfg.block_size
    m = num_tokens * topk  # total token slots after topk expansion

    print(f"\n{'='*70}")
    print(f"  {cfg.name}")
    print(
        f"  tokens={num_tokens}  experts={num_experts}  topk={topk}  "
        f"hidden={k}  intermediate={n}  block_size={block_size}"
    )
    print(f"  m={m}  w1=({num_experts}, {2*n}, {k})  w2=({num_experts}, {k}, {n})")
    print(f"{'='*70}")

    # --- Allocate inputs ---
    a = torch.randn(num_tokens, k, dtype=torch.float16, device=device)
    w1 = torch.randn(num_experts, 2 * n, k, dtype=torch.float16, device=device)
    w2 = torch.randn(num_experts, k, n, dtype=torch.float16, device=device)
    score = torch.rand(num_tokens, num_experts, dtype=torch.float16, device=device)
    score_f32 = torch.softmax(score.float(), dim=-1)

    max_num_tokens_padded = score.numel() + num_experts * (block_size - 1)
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)

    # --- Phase 1: Compile all kernels (not timed) ---
    print("\nCompiling kernels...", end=" ", flush=True)
    t0 = time.time()

    topk_fn = compile_topk(num_tokens, num_experts, topk)

    # Run alignment once to get num_blocks for GEMM compilation
    topk_weights = torch.zeros(num_tokens, topk, dtype=torch.float32, device=device)
    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
    topk_fn(score_f32, topk_weights, topk_ids)
    torch.cuda.synchronize()
    topk_weights = topk_weights.view(-1)
    topk_ids = topk_ids.view(-1)

    numel = num_tokens * topk
    align_kernels = compile_moe_align_kernels(
        numel,
        num_experts,
        block_size,
        max_num_tokens_padded,
        max_num_m_blocks,
    )
    (expert_ids, sorted_ids, _, _, _, _) = moe_align_block_size(
        topk_ids.to(torch.int32),
        num_experts,
        block_size,
        max_num_tokens_padded,
        max_num_m_blocks,
        compiled_kernels=align_kernels,
    )
    num_blocks = expert_ids.shape[0]
    pad_value = m
    total_elems = num_blocks * block_size

    gather1_fn = compile_gather(m, k, num_blocks, block_size, pad_value)
    gemm1_fn = compile_gemm(m, w1.shape[1], k, w1.shape[0], num_blocks)
    scatter1_fn = compile_scatter(m, w1.shape[1], num_blocks, block_size, pad_value)

    silu_fn = compile_silu(m, w1.shape[1] // 2)

    gather2_fn = compile_gather(m, w1.shape[1] // 2, num_blocks, block_size, pad_value)
    gemm2_fn = compile_gemm(m, w2.shape[1], w1.shape[1] // 2, w2.shape[0], num_blocks)
    scatter2_fn = compile_scatter(m, w2.shape[1], num_blocks, block_size, pad_value)

    reduce_fn = compile_reduce(num_tokens, topk, w2.shape[1])

    compile_time = time.time() - t0
    print(f"done ({compile_time:.1f}s)")

    # --- Phase 2: Benchmark ---
    # Pre-allocate all buffers
    topk_weights_buf = torch.zeros(num_tokens, topk, dtype=torch.float32, device=device)
    topk_ids_buf = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)

    a_expanded = a.view(num_tokens, -1, k).repeat(1, topk, 1).reshape(-1, k)

    sorted_ids_wave = torch.empty(total_elems, dtype=torch.int32, device=device)
    gather_dummy = torch.empty(total_elems, dtype=torch.int32, device=device)

    a_scratch = torch.zeros(num_blocks, m, k, dtype=torch.float16, device=device)
    c1_scratch = torch.zeros(
        num_blocks, m, w1.shape[1], dtype=torch.float32, device=device
    )
    gemm1_out = torch.zeros(m, w1.shape[1], dtype=torch.float32, device=device)
    silu_out = torch.zeros(m, w1.shape[1] // 2, dtype=torch.float32, device=device)
    silu_out_f16 = torch.zeros(m, w1.shape[1] // 2, dtype=torch.float16, device=device)
    a2_scratch = torch.zeros(
        num_blocks, m, w1.shape[1] // 2, dtype=torch.float16, device=device
    )
    c2_scratch = torch.zeros(
        num_blocks, m, w2.shape[1], dtype=torch.float32, device=device
    )
    gemm2_out = torch.zeros(m, w2.shape[1], dtype=torch.float32, device=device)
    final_out = torch.zeros(num_tokens, w2.shape[1], dtype=torch.float32, device=device)

    # Kernel names for reporting
    kernel_names = [
        "TopK",
        "Alignment",
        "Gather1",
        "GEMM1",
        "Scatter1",
        "SiLU",
        "Gather2",
        "GEMM2",
        "Scatter2",
        "Reduce",
    ]
    kernel_times = {name: [] for name in kernel_names}

    def timed_sync():
        """Return current GPU time via cuda event."""
        torch.cuda.synchronize()
        return time.perf_counter()

    def run_pipeline():
        """Execute the full MoE pipeline, recording per-kernel times."""
        times = {}

        # TopK
        t = timed_sync()
        topk_fn(score_f32, topk_weights_buf, topk_ids_buf)
        torch.cuda.synchronize()
        times["TopK"] = time.perf_counter() - t

        tw = topk_weights_buf.view(-1)
        ti = topk_ids_buf.view(-1)

        # Alignment (Steps 1-5)
        t = timed_sync()
        (eid, sid, _, _, _, _) = moe_align_block_size(
            ti.to(torch.int32),
            num_experts,
            block_size,
            max_num_tokens_padded,
            max_num_m_blocks,
            compiled_kernels=align_kernels,
        )
        torch.cuda.synchronize()
        times["Alignment"] = time.perf_counter() - t

        # Prepare sorted_ids
        if sid.shape[0] < total_elems:
            sorted_ids_wave.fill_(pad_value)
            sorted_ids_wave[: sid.shape[0]] = sid
        else:
            sorted_ids_wave.copy_(sid[:total_elems])

        # Gather1
        a_scratch.zero_()
        t = timed_sync()
        gather1_fn(sorted_ids_wave, a_expanded, a_scratch, gather_dummy)
        torch.cuda.synchronize()
        times["Gather1"] = time.perf_counter() - t

        # GEMM1
        c1_scratch.zero_()
        t = timed_sync()
        gemm1_fn(a_scratch, w1, eid, c1_scratch)
        torch.cuda.synchronize()
        times["GEMM1"] = time.perf_counter() - t

        # Scatter1
        gemm1_out.zero_()
        t = timed_sync()
        scatter1_fn(sorted_ids_wave, c1_scratch, gemm1_out, gather_dummy)
        torch.cuda.synchronize()
        times["Scatter1"] = time.perf_counter() - t

        # SiLU
        silu_out.zero_()
        t = timed_sync()
        silu_fn(gemm1_out, silu_out)
        torch.cuda.synchronize()
        times["SiLU"] = time.perf_counter() - t

        silu_out_f16.copy_(silu_out.half())

        # Gather2
        a2_scratch.zero_()
        t = timed_sync()
        gather2_fn(sorted_ids_wave, silu_out_f16, a2_scratch, gather_dummy)
        torch.cuda.synchronize()
        times["Gather2"] = time.perf_counter() - t

        # GEMM2
        c2_scratch.zero_()
        t = timed_sync()
        gemm2_fn(a2_scratch, w2, eid, c2_scratch)
        torch.cuda.synchronize()
        times["GEMM2"] = time.perf_counter() - t

        # Scatter2
        gemm2_out.zero_()
        t = timed_sync()
        scatter2_fn(sorted_ids_wave, c2_scratch, gemm2_out, gather_dummy)
        torch.cuda.synchronize()
        times["Scatter2"] = time.perf_counter() - t

        # Reduce
        final_out.zero_()
        reshape_out = gemm2_out.view(num_tokens, -1, w2.shape[1])
        tw_bc = tw.view(num_tokens, -1)
        t = timed_sync()
        reduce_fn(reshape_out, tw_bc, final_out)
        torch.cuda.synchronize()
        times["Reduce"] = time.perf_counter() - t

        return times

    # Warmup
    print(f"\nWarmup ({warmup} iterations)...", flush=True)
    for _ in range(warmup):
        run_pipeline()

    # Timed runs
    print(f"Benchmarking ({iters} iterations)...", flush=True)
    total_times = []
    for _ in range(iters):
        t_start = timed_sync()
        times = run_pipeline()
        t_total = time.perf_counter() - t_start
        total_times.append(t_total)
        for name in kernel_names:
            kernel_times[name].append(times[name])

    # --- Phase 3: Report ---
    print(
        f"\n{'Kernel':<14} {'Mean (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'% Total':>8}"
    )
    print("-" * 56)

    mean_total = sum(total_times) / len(total_times) * 1000
    for name in kernel_names:
        ts = kernel_times[name]
        mean_ms = sum(ts) / len(ts) * 1000
        min_ms = min(ts) * 1000
        max_ms = max(ts) * 1000
        pct = mean_ms / mean_total * 100
        print(
            f"{name:<14} {mean_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f} {pct:>7.1f}%"
        )

    min_total = min(total_times) * 1000
    max_total = max(total_times) * 1000
    print("-" * 56)
    print(
        f"{'TOTAL':<14} {mean_total:>10.3f} {min_total:>10.3f} {max_total:>10.3f} {'100.0':>7}%"
    )
    print(f"\nThroughput: {num_tokens / (mean_total / 1000):.0f} tokens/sec")

    return mean_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark Wave MoE pipeline")
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()) + ["all"],
        default="small",
        help="Which model configuration to benchmark (default: small)",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    args = parser.parse_args()

    configs = list(CONFIGS.values()) if args.config == "all" else [CONFIGS[args.config]]

    print(f"Wave MoE Pipeline Benchmark")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Warmup: {args.warmup}  Iterations: {args.iters}")

    results = {}
    for cfg in configs:
        mean_ms = benchmark_moe(cfg, warmup=args.warmup, iters=args.iters)
        results[cfg.name] = mean_ms

    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  Summary")
        print(f"{'='*70}")
        for name, ms in results.items():
            print(f"  {name:<35} {ms:.3f} ms")


if __name__ == "__main__":
    main()
