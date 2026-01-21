#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Benchmark script for Wave's attention implementation.

This script benchmarks the Wave attention kernel with specific configurations
for MI355 hardware, varying batch size and sequence length while keeping
other parameters fixed.
"""

import csv
import json
import sys
from datetime import datetime
from typing import List, Dict, Any

import torch

import wave_lang.kernel.wave.nn as wave_nn
from wave_lang.kernel.wave.templates.attention_common import AttentionShape

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================
# These parameters can be modified as needed for different benchmark runs.

# Default parameters (consistent across all runs unless overridden)
DEFAULT_PARAMS = {
    "dtype": torch.float16,  # Data type for tensors
    "device": "cuda:0",  # Device to run on
    "num_warmup": 10,  # Number of warmup iterations
    "num_iterations": 100,  # Number of benchmark iterations
}

# ============================================================================
# FIXED PARAMETERS
# ============================================================================
# These parameters are fixed for all benchmark runs as per requirements.

FIXED_PARAMS = {
    "head_dim": 128,  # D=128 (head dimension)
    "num_heads": 64,  # H=64 (number of attention heads)
    "dtype_name": "fp16",  # Data type name
}

# ============================================================================
# VARIABLE PARAMETERS
# ============================================================================
# These parameters vary across benchmark runs.

# Attention variants to test
# Options: "fp16" (vanilla FP16), "fp8" (quantized FP8)
ATTENTION_VARIANTS = ["fp16", "fp8"]

# FP8 quantization scaling factors (used when variant="fp8")
FP8_SCALE_PARAMS = {
    "q_scale": 1.0,
    "k_scale": 1.0,
    "v_scale": 1.0,
}

# Causal mask configurations to test
CAUSAL_VALUES = [0, 1]  # 0 = non-causal, 1 = causal

# (batch_size, sequence_length) pairs to test
# Note: N_CTX_Q and N_CTX_K are the same for these tests
BATCH_SEQLEN_PAIRS = [
    (32, 512),
    (16, 1024),
    (8, 2048),
    (4, 4096),
    (2, 8192),
    (1, 16384),
]

# ============================================================================
# BENCHMARK IMPLEMENTATION
# ============================================================================


def create_attention_inputs(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    """Create input tensors for attention benchmark."""
    query = torch.randn(
        [batch_size, num_heads, seq_len_q, head_dim],
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        [batch_size, num_heads, seq_len_k, head_dim],
        device=device,
        dtype=dtype,
    )
    value = torch.randn(
        [batch_size, num_heads, seq_len_k, head_dim],
        device=device,
        dtype=dtype,
    )
    return query, key, value


def benchmark_attention(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    is_causal: bool,
    variant: str,
    num_warmup: int,
    num_iterations: int,
    dtype: torch.dtype,
    device: str,
) -> Dict[str, Any]:
    """
    Benchmark Wave attention with given parameters.

    Args:
        variant: "fp16" for vanilla FP16 attention, "fp8" for quantized FP8 attention

    Returns:
        Dictionary containing benchmark results including timing information.
    """
    # Create input tensors
    query, key, value = create_attention_inputs(
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim, dtype, device
    )

    # Select the appropriate attention function based on variant
    if variant == "fp16":
        attention_func = lambda q, k, v: wave_nn.functional.wave_sdpa(
            q, k, v, is_causal=is_causal
        )
    elif variant == "fp8":
        attention_func = lambda q, k, v: wave_nn.functional.wave_sdpa_fp8(
            q,
            k,
            v,
            q_scale=FP8_SCALE_PARAMS["q_scale"],
            k_scale=FP8_SCALE_PARAMS["k_scale"],
            v_scale=FP8_SCALE_PARAMS["v_scale"],
            is_causal=is_causal,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Warmup phase
    for _ in range(num_warmup):
        _ = attention_func(query, key, value)
    torch.cuda.synchronize()

    # Benchmark phase
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    for i in range(num_iterations):
        start_events[i].record()
        _ = attention_func(query, key, value)
        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate timing statistics
    times_ms = [
        start_events[i].elapsed_time(end_events[i]) for i in range(num_iterations)
    ]
    avg_time_ms = sum(times_ms) / len(times_ms)
    min_time_ms = min(times_ms)
    max_time_ms = max(times_ms)

    # Calculate throughput (FLOPs)
    # For attention: 4 * B * H * Q * K * D operations
    flops = 4 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
    throughput_tflops = flops / (avg_time_ms / 1000) / 1e12

    return {
        "variant": variant,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len_q": seq_len_q,
        "seq_len_k": seq_len_k,
        "head_dim": head_dim,
        "is_causal": is_causal,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "throughput_tflops": throughput_tflops,
        "num_warmup": num_warmup,
        "num_iterations": num_iterations,
    }


def run_benchmarks() -> List[Dict[str, Any]]:
    """Run all benchmark configurations and return results."""
    results = []

    print("=" * 80)
    print("Wave Attention Benchmark")
    print("=" * 80)
    print(f"\nFixed Parameters:")
    print(f"  Head Dimension (D): {FIXED_PARAMS['head_dim']}")
    print(f"  Number of Heads (H): {FIXED_PARAMS['num_heads']}")
    print(f"  Data Type: {FIXED_PARAMS['dtype_name']}")
    print(f"\nDefault Settings:")
    print(f"  Warmup Iterations: {DEFAULT_PARAMS['num_warmup']}")
    print(f"  Benchmark Iterations: {DEFAULT_PARAMS['num_iterations']}")
    print(f"  Device: {DEFAULT_PARAMS['device']}")
    print(f"\nVariants to Test: {', '.join(ATTENTION_VARIANTS)}")
    if "fp8" in ATTENTION_VARIANTS:
        print(f"  FP8 Scales: q={FP8_SCALE_PARAMS['q_scale']}, "
              f"k={FP8_SCALE_PARAMS['k_scale']}, v={FP8_SCALE_PARAMS['v_scale']}")
    print("\n" + "=" * 80)

    total_configs = len(ATTENTION_VARIANTS) * len(CAUSAL_VALUES) * len(BATCH_SEQLEN_PAIRS)
    current_config = 0

    for variant in ATTENTION_VARIANTS:
        for is_causal in CAUSAL_VALUES:
            for batch_size, seq_len in BATCH_SEQLEN_PAIRS:
                current_config += 1
                print(
                    f"\n[{current_config}/{total_configs}] Running: "
                    f"variant={variant}, B={batch_size}, N_CTX={seq_len}, causal={is_causal}"
                )

                try:
                    result = benchmark_attention(
                        batch_size=batch_size,
                        num_heads=FIXED_PARAMS["num_heads"],
                        seq_len_q=seq_len,
                        seq_len_k=seq_len,
                        head_dim=FIXED_PARAMS["head_dim"],
                        is_causal=bool(is_causal),
                        variant=variant,
                        num_warmup=DEFAULT_PARAMS["num_warmup"],
                        num_iterations=DEFAULT_PARAMS["num_iterations"],
                        dtype=DEFAULT_PARAMS["dtype"],
                        device=DEFAULT_PARAMS["device"],
                    )
                    results.append(result)
                    print(
                        f"  ✓ Avg Time: {result['avg_time_ms']:.3f} ms, "
                        f"Throughput: {result['throughput_tflops']:.2f} TFLOPs"
                    )
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    # Continue with other configurations even if one fails

    return results


def save_results_json(results: List[Dict[str, Any]], filename: str):
    """Save benchmark results to a JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "fixed_params": FIXED_PARAMS,
        "default_params": {
            k: str(v) if isinstance(v, torch.dtype) else v
            for k, v in DEFAULT_PARAMS.items()
        },
        "results": results,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to JSON: {filename}")


def save_results_csv(results: List[Dict[str, Any]], filename: str):
    """Save benchmark results to a CSV file."""
    if not results:
        print("No results to save to CSV")
        return

    fieldnames = [
        "variant",
        "batch_size",
        "num_heads",
        "seq_len_q",
        "seq_len_k",
        "head_dim",
        "is_causal",
        "avg_time_ms",
        "min_time_ms",
        "max_time_ms",
        "throughput_tflops",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result[k] for k in fieldnames})

    print(f"✓ Results saved to CSV: {filename}")


def print_results_table(results: List[Dict[str, Any]]):
    """Print benchmark results as a formatted table."""
    if not results:
        print("\nNo results to display")
        return

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)

    # Header
    header = (
        f"{'Variant':>8} {'Batch':>6} {'SeqLen':>7} {'Causal':>7} "
        f"{'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'TFLOPs':>10}"
    )
    print(header)
    print("-" * 90)

    # Results
    for result in results:
        row = (
            f"{result['variant']:>8} "
            f"{result['batch_size']:>6} "
            f"{result['seq_len_q']:>7} "
            f"{result['is_causal']:>7} "
            f"{result['avg_time_ms']:>10.3f} "
            f"{result['min_time_ms']:>10.3f} "
            f"{result['max_time_ms']:>10.3f} "
            f"{result['throughput_tflops']:>10.2f}"
        )
        print(row)

    print("=" * 90)


def main():
    """Main entry point for the benchmark script."""
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"wave_attention_benchmark_{timestamp}.json"
    csv_filename = f"wave_attention_benchmark_{timestamp}.csv"

    # Run benchmarks
    results = run_benchmarks()

    # Save and display results
    if results:
        save_results_json(results, json_filename)
        save_results_csv(results, csv_filename)
        print_results_table(results)
    else:
        print("\n✗ No benchmark results to save")
        sys.exit(1)

    print("\n✓ Benchmark completed successfully")


if __name__ == "__main__":
    main()
