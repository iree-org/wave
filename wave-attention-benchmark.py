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

The script now supports Wave-specific compiler tuning parameters:
- Scheduling strategies (NONE, MODULO, PREFETCH_ATTENTION)
- Waves per EU (1-4)
- Scheduling barriers (True/False)

These parameters allow exploring different performance characteristics while
keeping the same attention problem size (batch, sequence length, heads, dimensions).
"""

import csv
import json
import sys
from datetime import datetime
from typing import List, Dict, Any

import functools
import math

import torch

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave.nn as wave_nn
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from wave_lang.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)

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
ATTENTION_VARIANTS = [
    "fp16",
    #"fp8",
]

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
# WAVE-SPECIFIC TUNING PARAMETERS
# ============================================================================
# These parameters control Wave compiler optimizations and can significantly
# impact performance while keeping the same attention problem size.
#
# IMPORTANT: Testing all combinations creates a large number of configurations:
#   - 2 variants × 2 causal × 6 batch/seqlen × 3 schedules × 4 waves × 2 barriers
#   - = 576 total configurations
#
# RECOMMENDED APPROACHES:
#   1. Test one parameter at a time (keep others at default)
#   2. Test specific combinations known to be interesting
#   3. Use a smaller subset of BATCH_SEQLEN_PAIRS for initial exploration
#
# To test only default Wave parameters, set:
#   SCHEDULING_STRATEGIES = [SchedulingType.NONE]
#   WAVES_PER_EU_VALUES = [2]
#   USE_SCHEDULING_BARRIERS_VALUES = [False]

# Scheduling strategies to test
# Options:
#   - SchedulingType.NONE: No scheduling optimization (baseline)
#   - SchedulingType.MODULO: Modulo scheduling (software pipelining)
#   - SchedulingType.PREFETCH: Prefetch scheduling
#   - SchedulingType.PREFETCH_ATTENTION: Attention-specific prefetch scheduling
SCHEDULING_STRATEGIES = [
    SchedulingType.NONE,
    SchedulingType.MODULO,
    #SchedulingType.PREFETCH,
    SchedulingType.PREFETCH_ATTENTION,
]

# Waves per EU (Execution Unit) to test
# Controls GPU occupancy. Higher values increase latency hiding but reduce
# per-wave resources. Typical values: 1-4
WAVES_PER_EU_VALUES = [
    2,
    # 1, 3, 4,
]

# Scheduling barriers
# Enable/disable scheduling barriers for instruction scheduling
USE_SCHEDULING_BARRIERS_VALUES = [
    False,
    True,
]

# MMA (Matrix Multiply-Accumulate) instruction variants to test
# Different MMA types can have different performance characteristics
# For FP16:
#   - F32_16x16x16_F16: Standard 16x16x16 MMA (default)
#   - F32_32x32x8_F16: Larger 32x32x8 MMA
#   - F32_16x16x32_K8_F16: 16x16x32 with K=8 (CDNA3+)
#   - F32_32x32x16_K8_F16: 32x32x16 with K=8 (CDNA3+)
# For FP8:
#   - F32_16x16x32_F8: Standard 16x16x32 FP8 MMA (default)
#   - F32_32x32x16_F8: Larger 32x32x16 FP8 MMA
MMA_VARIANTS_FP16 = [
    (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
    # (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    # (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
    # (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
]

MMA_VARIANTS_FP8 = [
    (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
    # (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
]

# Tile size hyperparameters
# These control the workgroup tile sizes for the attention kernel
# BLOCK_M: Query sequence tile size (default: 128)
# BLOCK_N: Head dimension tile size (default: 64)
# BLOCK_K2: KV sequence tile size (default: 64)
# Larger tiles may improve performance but increase shared memory usage
# Note: Must be multiples of the MMA instruction size
BLOCK_M_VALUES = [
    128,
    # 64, 256,
]

BLOCK_N_VALUES = [
    64,
    # 32, 128,
]

BLOCK_K2_VALUES = [
    64,
    # 32, 128,
]

# Compiler optimization flags
# These control various compiler optimizations that can affect performance
USE_BUFFER_OPS_VALUES = [
    False,  # Default in functional API
    # True,   # Used in kernel-benchmark
]

CANONICALIZE_VALUES = [
    True,  # Default (always enabled)
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


@functools.lru_cache(maxsize=256)
def get_custom_wave_kernel(
    shape: AttentionShape,
    variant: str,
    is_causal: bool,
    mfma_variant: tuple,
    block_m: int,
    block_n: int,
    block_k2: int,
    schedule: SchedulingType,
    waves_per_eu: int,
    use_scheduling_barriers: bool,
    use_buffer_ops: bool,
    canonicalize: bool,
):
    """
    Compile a Wave attention kernel with custom tuning parameters.
    
    Args:
        shape: Attention shape configuration
        variant: "fp16" or "fp8"
        is_causal: Whether to use causal masking
        mfma_variant: MMA instruction type tuple
        block_m: Query sequence tile size
        block_n: Head dimension tile size
        block_k2: KV sequence tile size
        schedule: Scheduling strategy
        waves_per_eu: Waves per execution unit
        use_scheduling_barriers: Whether to use scheduling barriers
        use_buffer_ops: Whether to use buffer operations
        canonicalize: Whether to canonicalize IR
    
    Returns:
        Compiled Wave kernel
    """
    if variant == "fp16":
        (
            attention_kernel,
            hyperparams,
            dynamic_symbols,
        ) = get_vanilla_attention_kernel(
            shape,
            mfma_variant,
            dynamic_dims=False,
            is_causal=is_causal,
        )
        hyperparams.update(get_default_scheduling_params())
        
        # Override tile sizes if different from defaults
        if block_m != 128:
            hyperparams[tkl.sym.BLOCK_M] = block_m
        if block_n != 64:
            hyperparams[tkl.sym.BLOCK_N] = block_n
        if block_k2 != 64:
            hyperparams[tkl.sym.BLOCK_K2] = block_k2
        
        del hyperparams[tkl.sym.B]
        del hyperparams[tkl.sym.M]
        del hyperparams[tkl.sym.N]
        del hyperparams[tkl.sym.K2]
        dynamic_symbols = [tkl.sym.B, tkl.sym.M, tkl.sym.N, tkl.sym.K2]
        
    elif variant == "fp8":
        (
            attention_kernel,
            hyperparams,
            _,
        ) = get_brevitas_pertensor_fp8_attention_kernel(
            shape,
            mfma_variant,
            q_scale=FP8_SCALE_PARAMS["q_scale"],
            k_scale=FP8_SCALE_PARAMS["k_scale"],
            v_scale=FP8_SCALE_PARAMS["v_scale"],
            is_causal=is_causal,
            logit_dtype=torch.float16,
            f8_dtype=torch.float8_e4m3fnuz,
        )
        hyperparams.update(get_default_scheduling_params())
        
        # Override tile sizes if different from defaults
        if block_m != 128:
            hyperparams[tkl.sym.BLOCK_M] = block_m
        if block_n != 64:
            hyperparams[tkl.sym.BLOCK_N] = block_n
        if block_k2 != 64:
            hyperparams[tkl.sym.BLOCK_K2] = block_k2
        
        del hyperparams[tkl.sym.B]
        del hyperparams[tkl.sym.N_Q]
        del hyperparams[tkl.sym.N_KV]
        dynamic_symbols = [tkl.sym.B, tkl.sym.N_Q, tkl.sym.N_KV]
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # Create compile options with custom tuning parameters
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=schedule,
        use_scheduling_barriers=use_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        waves_per_eu=waves_per_eu,
        denorm_fp_math_f32="preserve-sign",
        use_buffer_ops=use_buffer_ops,
        canonicalize=canonicalize,
    )
    options = set_default_run_config(options)
    attention_kernel = wave_compile(options, attention_kernel)
    return attention_kernel


def benchmark_attention(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    is_causal: bool,
    variant: str,
    mfma_variant: tuple,
    block_m: int,
    block_n: int,
    block_k2: int,
    schedule: SchedulingType,
    waves_per_eu: int,
    use_scheduling_barriers: bool,
    use_buffer_ops: bool,
    canonicalize: bool,
    num_warmup: int,
    num_iterations: int,
    dtype: torch.dtype,
    device: str,
) -> Dict[str, Any]:
    """
    Benchmark Wave attention with given parameters.

    Args:
        variant: "fp16" for vanilla FP16 attention, "fp8" for quantized FP8 attention
        mfma_variant: MMA instruction type tuple
        block_m: Query sequence tile size
        block_n: Head dimension tile size
        block_k2: KV sequence tile size
        schedule: Scheduling strategy
        waves_per_eu: Waves per execution unit
        use_scheduling_barriers: Whether to use scheduling barriers
        use_buffer_ops: Whether to use buffer operations
        canonicalize: Whether to canonicalize IR

    Returns:
        Dictionary containing benchmark results including timing information.
    """
    # Create input tensors
    query, key, value = create_attention_inputs(
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim, dtype, device
    )

    # Create attention shape
    batch = query.shape[:-2]
    flattened_batch_size = math.prod(batch)
    flat_q_shape = [flattened_batch_size, query.shape[-2], query.shape[-1]]
    flat_kv_shape = [flattened_batch_size, key.shape[-2], key.shape[-1]]
    flat_o_shape = [flattened_batch_size, query.shape[-2], key.shape[-1]]
    
    shape = AttentionShape(
        num_query_heads=flattened_batch_size,
        num_kv_heads=flattened_batch_size,
        query_seq_len=query.shape[-2],
        head_size_kv=key.shape[-1],
        head_size=query.shape[-1],
        kv_seq_len=key.shape[-2],
    )
    
    # Get compiled kernel with custom tuning parameters
    attention_kernel = get_custom_wave_kernel(
        shape=shape,
        variant=variant,
        is_causal=is_causal,
        mfma_variant=mfma_variant,
        block_m=block_m,
        block_n=block_n,
        block_k2=block_k2,
        schedule=schedule,
        waves_per_eu=waves_per_eu,
        use_scheduling_barriers=use_scheduling_barriers,
        use_buffer_ops=use_buffer_ops,
        canonicalize=canonicalize,
    )
    
    # Create output tensor
    output = torch.empty(flat_o_shape, dtype=torch.float32, device=device)
    
    # Create attention function
    def attention_func(q, k, v):
        attention_kernel(
            q.view(flat_q_shape),
            k.view(flat_kv_shape),
            v.view(flat_kv_shape),
            output,
        )
        return output.view(*batch, shape.query_seq_len, shape.head_size_kv)

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

    # Format MMA variant for output
    mma_str = f"{mfma_variant[0].name}+{mfma_variant[1].name}"
    
    return {
        "variant": variant,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len_q": seq_len_q,
        "seq_len_k": seq_len_k,
        "head_dim": head_dim,
        "is_causal": is_causal,
        "mfma_variant": mma_str,
        "block_m": block_m,
        "block_n": block_n,
        "block_k2": block_k2,
        "schedule": schedule.name,
        "waves_per_eu": waves_per_eu,
        "use_scheduling_barriers": use_scheduling_barriers,
        "use_buffer_ops": use_buffer_ops,
        "canonicalize": canonicalize,
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
    print(f"\nWave-Specific Tuning Parameters:")
    print(f"  Scheduling Strategies: {[s.name for s in SCHEDULING_STRATEGIES]}")
    print(f"  Waves Per EU: {WAVES_PER_EU_VALUES}")
    print(f"  Scheduling Barriers: {USE_SCHEDULING_BARRIERS_VALUES}")
    print(f"  MMA Variants (FP16): {len(MMA_VARIANTS_FP16)}")
    print(f"  MMA Variants (FP8): {len(MMA_VARIANTS_FP8)}")
    print(f"  BLOCK_M values: {BLOCK_M_VALUES}")
    print(f"  BLOCK_N values: {BLOCK_N_VALUES}")
    print(f"  BLOCK_K2 values: {BLOCK_K2_VALUES}")
    print(f"  use_buffer_ops: {USE_BUFFER_OPS_VALUES}")
    print("\n" + "=" * 80)

    # Calculate total configs dynamically based on variant
    total_configs = 0
    for variant in ATTENTION_VARIANTS:
        mma_variants = MMA_VARIANTS_FP16 if variant == "fp16" else MMA_VARIANTS_FP8
        total_configs += (
            len(CAUSAL_VALUES) 
            * len(BATCH_SEQLEN_PAIRS)
            * len(mma_variants)
            * len(BLOCK_M_VALUES)
            * len(BLOCK_N_VALUES)
            * len(BLOCK_K2_VALUES)
            * len(SCHEDULING_STRATEGIES)
            * len(WAVES_PER_EU_VALUES)
            * len(USE_SCHEDULING_BARRIERS_VALUES)
            * len(USE_BUFFER_OPS_VALUES)
            * len(CANONICALIZE_VALUES)
        )
    
    current_config = 0

    for variant in ATTENTION_VARIANTS:
        # Select appropriate MMA variants for this attention variant
        mma_variants = MMA_VARIANTS_FP16 if variant == "fp16" else MMA_VARIANTS_FP8
        
        for is_causal in CAUSAL_VALUES:
            for batch_size, seq_len in BATCH_SEQLEN_PAIRS:
                for mfma_variant in mma_variants:
                    for block_m in BLOCK_M_VALUES:
                        for block_n in BLOCK_N_VALUES:
                            for block_k2 in BLOCK_K2_VALUES:
                                for schedule in SCHEDULING_STRATEGIES:
                                    for waves_per_eu in WAVES_PER_EU_VALUES:
                                        for use_barriers in USE_SCHEDULING_BARRIERS_VALUES:
                                            for use_buffer_ops in USE_BUFFER_OPS_VALUES:
                                                for canonicalize in CANONICALIZE_VALUES:
                                                    current_config += 1
                                                    mma_short = f"{mfma_variant[0].name.split('_')[1]}"
                                                    print(
                                                        f"\n[{current_config}/{total_configs}] Running: "
                                                        f"variant={variant}, B={batch_size}, N_CTX={seq_len}, "
                                                        f"causal={is_causal}, mma={mma_short}, "
                                                        f"tiles={block_m}x{block_n}x{block_k2}, "
                                                        f"sched={schedule.name}, waves={waves_per_eu}"
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
                                                            mfma_variant=mfma_variant,
                                                            block_m=block_m,
                                                            block_n=block_n,
                                                            block_k2=block_k2,
                                                            schedule=schedule,
                                                            waves_per_eu=waves_per_eu,
                                                            use_scheduling_barriers=use_barriers,
                                                            use_buffer_ops=use_buffer_ops,
                                                            canonicalize=canonicalize,
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
        "mfma_variant",
        "block_m",
        "block_n",
        "block_k2",
        "schedule",
        "waves_per_eu",
        "use_scheduling_barriers",
        "use_buffer_ops",
        "canonicalize",
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
    """Print benchmark results as a formatted table (summary only - see CSV for full details)."""
    if not results:
        print("\nNo results to display")
        return

    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS SUMMARY (see CSV for full details)")
    print("=" * 120)

    # Header - simplified to show key metrics
    header = (
        f"{'Var':>4} {'B':>4} {'N_CTX':>6} {'Caus':>4} {'Tiles':>11} "
        f"{'Sched':>12} {'W':>2} {'Avg(ms)':>9} {'TFLOPs':>8}"
    )
    print(header)
    print("-" * 120)

    # Results - simplified
    for result in results:
        tiles = f"{result['block_m']}x{result['block_n']}x{result['block_k2']}"
        row = (
            f"{result['variant']:>4} "
            f"{result['batch_size']:>4} "
            f"{result['seq_len_q']:>6} "
            f"{result['is_causal']:>4} "
            f"{tiles:>11} "
            f"{result['schedule']:>12} "
            f"{result['waves_per_eu']:>2} "
            f"{result['avg_time_ms']:>9.3f} "
            f"{result['throughput_tflops']:>8.2f}"
        )
        print(row)

    print("=" * 120)
    print("\nNote: Full results including MMA variants, barriers, buffer_ops, etc. are in the CSV file.")


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
