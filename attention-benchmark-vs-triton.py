#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Benchmark script comparing Wave vs Triton attention implementations.

This script benchmarks both Wave and Triton attention kernels with identical
configurations to provide an apples-to-apples comparison. It uses the same
FIXED_PARAMS and variable parameters as wave-attention-benchmark.py.

Requirements:
    - wave_lang: pip install wave-lang (or install from source)
    - triton: pip install triton (or install from the ../triton branch)
    - torch: pip install torch with CUDA/ROCm support

The benchmark runs both implementations on the same input tensors and reports
timing and throughput metrics side-by-side.
"""

import csv
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import functools

import torch

# ============================================================================
# WAVE IMPORTS
# ============================================================================
try:
    import wave_lang.kernel.lang as tkl
    from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
    from wave_lang.kernel.wave.constraints import MMAType
    from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
    from wave_lang.kernel.wave.templates.attention_common import AttentionShape
    from wave_lang.kernel.wave.templates.tagged_attention import (
        get_tagged_bshd_attention_kernel,
    )
    from wave_lang.kernel.wave.schedules.attention_prefetch import (
        get_attention_prefetch_schedule,
    )
    from wave_lang.kernel.wave.utils.general_utils import (
        get_default_scheduling_params,
    )
    from wave_lang.kernel.wave.utils.run_utils import (
        set_default_run_config,
    )

    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    print("Warning: wave_lang not available. Wave benchmarks will be skipped.")

# ============================================================================
# TRITON IMPORTS
# ============================================================================
# Import Triton Flash Attention v3 from the triton repository.
# This uses the same implementation as triton's bench.sh script.
#
# The FA v3 module is NOT part of the standard triton pip package - it lives
# in the fa/ directory of the triton repo. Configure its location via:
#   - TRITON_FA_PATH environment variable (path to the fa/ directory)
#   - Or place triton repo as sibling: ../triton/fa/flash-attention.py

import os

# Set FAv3 environment variables (same as bench.sh)
os.environ.setdefault("DISABLE_LLVM_OPT", "disable-vector-combine")
os.environ.setdefault("TRITON_HIP_USE_PADDED_SHARED_LAYOUT", "1")
os.environ.setdefault("TRITON_HIP_USE_ASYNC_COPY", "1")
os.environ.setdefault("AMDGCN_SCALARIZE_PACKED_FOPS", "1")


def _find_triton_fa_path():
    """Find the Triton FA v3 module path."""
    # 1. Check environment variable
    if "TRITON_FA_PATH" in os.environ:
        return os.environ["TRITON_FA_PATH"]

    # 2. Check relative to this script (sibling repo layout)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir, "..", "triton", "fa")
    if os.path.exists(os.path.join(relative_path, "flash-attention.py")):
        return relative_path

    # 3. Check if triton is installed and has a fa/ directory
    try:
        import triton

        triton_root = os.path.dirname(os.path.dirname(triton.__file__))
        fa_path = os.path.join(triton_root, "fa")
        if os.path.exists(os.path.join(fa_path, "flash-attention.py")):
            return fa_path
    except ImportError:
        pass

    return None


TRITON_FA_PATH = _find_triton_fa_path()

try:
    if TRITON_FA_PATH and os.path.exists(
        os.path.join(TRITON_FA_PATH, "flash-attention.py")
    ):
        sys.path.insert(0, TRITON_FA_PATH)
        # Import the attention function and helpers from triton's flash-attention
        from importlib.machinery import SourceFileLoader

        flash_attn_module = SourceFileLoader(
            "flash_attention", os.path.join(TRITON_FA_PATH, "flash-attention.py")
        ).load_module()
        triton_attention_fn = flash_attn_module.attention
        TritonMetaData = flash_attn_module.MetaData
        triton_input_helper = flash_attn_module.input_helper
        TRITON_AVAILABLE = True
        print(f"Triton FA v3 loaded from: {TRITON_FA_PATH}")
    else:
        print(f"Warning: Triton FA v3 module not found.")
        print(f"  Set TRITON_FA_PATH environment variable to the triton/fa directory,")
        print(f"  or ensure triton repo is at ../triton relative to this script.")
        TRITON_AVAILABLE = False
except ImportError as e:
    TRITON_AVAILABLE = False
    print(f"Warning: triton not available ({e}). Triton benchmarks will be skipped.")

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
# These parameters control Wave compiler optimizations.

if WAVE_AVAILABLE:
    # MMA Types for FP16
    MMA_TYPES_FP16 = [
        MMAType.F32_16x16x16_F16,
        # MMAType.F32_16x16x32_F16,
    ]
else:
    MMA_TYPES_FP16 = []

# Waves per EU (Execution Unit) to test
WAVES_PER_EU_VALUES = [2]

# Scheduling barriers
USE_SCHEDULING_BARRIERS_VALUES = [False, True]

# Tile size hyperparameters
BLOCK_N_VALUES = [64]
BLOCK_K2_VALUES = [64]

# Compiler optimization flags
USE_BUFFER_OPS_VALUES = [False]
CANONICALIZE_VALUES = [True]

# ============================================================================
# TRITON FLASH ATTENTION v3 WRAPPER
# ============================================================================
# Uses the Flash Attention v3 implementation from ../triton/fa/flash-attention.py
# This is the same kernel used by triton's bench.sh script.

# Layout used for Triton FA (matching bench.sh default)
TRITON_LAYOUT = "bhsd"  # batch, heads, seq_len, head_dim


class TritonAttention:
    """Wrapper class for Triton Flash Attention v3 from the triton repo."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        is_causal: bool,
        device: str,
    ):
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton Flash Attention is not available")

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.device = device
        self.sm_scale = head_dim**-0.5

        # Pre-create metadata for this configuration
        self.metadata = TritonMetaData(sm_scale=self.sm_scale)
        self.metadata.max_seqlens_q = seq_len
        self.metadata.max_seqlens_k = seq_len
        self.metadata.layout = TRITON_LAYOUT
        if is_causal:
            self.metadata.need_causal()

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Run Triton Flash Attention v3 forward pass.

        Args:
            q, k, v: Input tensors in bhsd layout [batch, heads, seq_len, head_dim]

        Returns:
            Output tensor in same layout as input
        """
        # Create output tensor
        o = torch.empty_like(q)

        # Call the triton attention function from the imported module
        # The attention function signature is: attention(q, k, v, o, metadata)
        output, _, _ = triton_attention_fn(q, k, v, o, self.metadata)

        return output


# ============================================================================
# WAVE ATTENTION KERNEL
# ============================================================================

if WAVE_AVAILABLE:

    @functools.lru_cache(maxsize=256)
    def get_wave_kernel(
        shape: AttentionShape,
        is_causal: bool,
        qk_t_mma: MMAType,
        att_v_mma: MMAType,
        block_n: int,
        block_k2: int,
        waves_per_eu: int,
        use_scheduling_barriers: bool,
        use_buffer_ops: bool,
        canonicalize: bool,
    ):
        """Compile a Wave attention kernel with the manual prefetch schedule."""
        mfma_variant = (qk_t_mma, att_v_mma)

        # IMPORTANT: num_waves must be 8 for the ping-pong schedule
        if waves_per_eu != 2:
            print(
                f"Warning: Manual schedule designed for waves_per_eu=2 (8 waves), got {waves_per_eu}"
            )

        (
            attention_kernel,
            hyperparams,
            dynamic_symbols,
        ) = get_tagged_bshd_attention_kernel(
            shape,
            mfma_variant,
            dynamic_dims=False,
            is_causal=is_causal,
            num_waves=8,  # Required for ping-pong schedule
        )
        hyperparams.update(get_default_scheduling_params())

        # Tagged attention uses different tile size parameters
        # BLOCK_N_Q is automatically set based on num_waves in the kernel
        # We'll respect block_k2 by setting BLOCK_N_KV
        if block_k2 != 64:
            hyperparams[tkl.sym.BLOCK_N_KV] = block_k2
        # BLOCK_D_KV corresponds to block_n
        if block_n != 64:
            hyperparams[tkl.sym.BLOCK_D_KV] = block_n

        # Get the manual schedule function
        attention_schedule = get_attention_prefetch_schedule()

        # Create compile options - must use MANUAL schedule and enable use_global_to_shared
        options = WaveCompileOptions(
            subs=hyperparams,
            schedule=SchedulingType.MANUAL,
            use_scheduling_barriers=use_scheduling_barriers,
            waves_per_eu=waves_per_eu,
            denorm_fp_math_f32="preserve-sign",
            use_buffer_ops=use_buffer_ops,
            canonicalize=canonicalize,
            use_global_to_shared=True,  # Required for GatherToLDS operations
        )
        options = set_default_run_config(options)

        # Compile with the manual schedule
        attention_kernel = wave_compile(options, attention_kernel, attention_schedule)

        return attention_kernel


class WaveAttention:
    """Wrapper class for Wave attention with manual prefetch schedule."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        is_causal: bool,
        device: str,
        mma_type: "MMAType" = None,
        block_n: int = 64,
        block_k2: int = 64,
        waves_per_eu: int = 2,
        use_scheduling_barriers: bool = False,
        use_buffer_ops: bool = False,
        canonicalize: bool = True,
    ):
        if not WAVE_AVAILABLE:
            raise RuntimeError("wave_lang is not available")

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.device = device

        # Create attention shape for tagged attention (BSHD layout)
        self.shape = AttentionShape(
            num_query_heads=num_heads,
            num_kv_heads=num_heads,
            query_seq_len=seq_len,
            head_size_kv=head_dim,
            head_size=head_dim,
            kv_seq_len=seq_len,
        )

        self.output_shape = [batch_size, seq_len, num_heads, head_dim]

        # Get compiled kernel
        self.kernel = get_wave_kernel(
            shape=self.shape,
            is_causal=is_causal,
            qk_t_mma=mma_type,
            att_v_mma=mma_type,
            block_n=block_n,
            block_k2=block_k2,
            waves_per_eu=waves_per_eu,
            use_scheduling_barriers=use_scheduling_barriers,
            use_buffer_ops=use_buffer_ops,
            canonicalize=canonicalize,
        )

        # Pre-allocate output tensor
        self.output = torch.empty(self.output_shape, dtype=torch.float32, device=device)

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Run Wave attention forward pass."""
        # BSHD layout - convert from BHSD to BSHD
        # Input is BHSD [batch, heads, seq, dim]
        # Need BSHD [batch, seq, heads, dim]
        q_bshd = q.transpose(1, 2)  # [B, S, H, D]
        k_bshd = k.transpose(1, 2)
        v_bshd = v.transpose(1, 2)

        self.kernel(q_bshd, k_bshd, v_bshd, self.output)

        # Convert output back from BSHD to BHSD
        return self.output.transpose(1, 2)  # [B, H, S, D]


# ============================================================================
# BENCHMARK IMPLEMENTATION
# ============================================================================


def create_attention_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create input tensors for attention benchmark."""
    query = torch.randn(
        [batch_size, num_heads, seq_len, head_dim],
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        [batch_size, num_heads, seq_len, head_dim],
        device=device,
        dtype=dtype,
    )
    value = torch.randn(
        [batch_size, num_heads, seq_len, head_dim],
        device=device,
        dtype=dtype,
    )
    return query, key, value


def calculate_flops(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    is_causal: bool,
) -> float:
    """Calculate FLOPs for attention operation."""
    if is_causal:
        # For causal masking, only count valid (non-masked) elements
        if seq_len_q > seq_len_k:
            valid_out_elements = (seq_len_k * seq_len_k + seq_len_k) / 2
        else:
            valid_out_elements = (
                seq_len_q * seq_len_k - (seq_len_q * seq_len_q - seq_len_q) / 2
            )
        flops = valid_out_elements * batch_size * num_heads * head_dim * 2
    else:
        # Non-causal: full computation
        flops = 2.0 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
    return flops


def benchmark_kernel(
    kernel_fn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
) -> Tuple[float, float, float]:
    """Benchmark a kernel and return timing statistics."""
    # Warmup phase
    for _ in range(num_warmup):
        _ = kernel_fn(query, key, value)
    torch.cuda.synchronize()

    # Benchmark phase
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    for i in range(num_iterations):
        start_events[i].record()
        _ = kernel_fn(query, key, value)
        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate timing statistics
    times_ms = [
        start_events[i].elapsed_time(end_events[i]) for i in range(num_iterations)
    ]
    avg_time_ms = sum(times_ms) / len(times_ms)
    min_time_ms = min(times_ms)
    max_time_ms = max(times_ms)

    return avg_time_ms, min_time_ms, max_time_ms


def benchmark_attention_comparison(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool,
    mma_type: Optional["MMAType"],
    block_n: int,
    block_k2: int,
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
    Benchmark both Wave and Triton attention with given parameters.

    Returns:
        Dictionary containing benchmark results for both implementations.
    """
    # Create input tensors (same for both implementations)
    query, key, value = create_attention_inputs(
        batch_size, num_heads, seq_len, head_dim, dtype, device
    )

    # Calculate FLOPs (same for both implementations)
    flops = calculate_flops(
        batch_size, num_heads, seq_len, seq_len, head_dim, is_causal
    )

    result = {
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "is_causal": is_causal,
        "block_n": block_n,
        "block_k2": block_k2,
        "waves_per_eu": waves_per_eu,
        "use_scheduling_barriers": use_scheduling_barriers,
    }

    # Benchmark Triton
    if TRITON_AVAILABLE:
        try:
            triton_attn = TritonAttention(
                batch_size, num_heads, seq_len, head_dim, is_causal, device
            )
            triton_avg, triton_min, triton_max = benchmark_kernel(
                triton_attn, query, key, value, num_warmup, num_iterations
            )
            triton_tflops = flops / (triton_avg / 1000) / 1e12

            result["triton_avg_time_ms"] = triton_avg
            result["triton_min_time_ms"] = triton_min
            result["triton_max_time_ms"] = triton_max
            result["triton_throughput_tflops"] = triton_tflops
        except Exception as e:
            print(f"  Triton error: {e}")
            result["triton_avg_time_ms"] = None
            result["triton_min_time_ms"] = None
            result["triton_max_time_ms"] = None
            result["triton_throughput_tflops"] = None
    else:
        result["triton_avg_time_ms"] = None
        result["triton_min_time_ms"] = None
        result["triton_max_time_ms"] = None
        result["triton_throughput_tflops"] = None

    # Benchmark Wave
    if WAVE_AVAILABLE and mma_type is not None:
        try:
            wave_attn = WaveAttention(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                is_causal,
                device,
                mma_type=mma_type,
                block_n=block_n,
                block_k2=block_k2,
                waves_per_eu=waves_per_eu,
                use_scheduling_barriers=use_scheduling_barriers,
                use_buffer_ops=use_buffer_ops,
                canonicalize=canonicalize,
            )
            wave_avg, wave_min, wave_max = benchmark_kernel(
                wave_attn, query, key, value, num_warmup, num_iterations
            )
            wave_tflops = flops / (wave_avg / 1000) / 1e12

            result["mma_type"] = mma_type.name
            result["wave_avg_time_ms"] = wave_avg
            result["wave_min_time_ms"] = wave_min
            result["wave_max_time_ms"] = wave_max
            result["wave_throughput_tflops"] = wave_tflops
        except Exception as e:
            print(f"  Wave error: {e}")
            result["mma_type"] = mma_type.name if mma_type else None
            result["wave_avg_time_ms"] = None
            result["wave_min_time_ms"] = None
            result["wave_max_time_ms"] = None
            result["wave_throughput_tflops"] = None
    else:
        result["mma_type"] = None
        result["wave_avg_time_ms"] = None
        result["wave_min_time_ms"] = None
        result["wave_max_time_ms"] = None
        result["wave_throughput_tflops"] = None

    # Calculate speedup if both are available
    if result["wave_avg_time_ms"] and result["triton_avg_time_ms"]:
        result["wave_vs_triton_speedup"] = (
            result["triton_avg_time_ms"] / result["wave_avg_time_ms"]
        )
    else:
        result["wave_vs_triton_speedup"] = None

    return result


def run_benchmarks() -> List[Dict[str, Any]]:
    """Run all benchmark configurations and return results."""
    results = []

    print("=" * 80)
    print("Wave vs Triton Attention Benchmark")
    print("=" * 80)
    print(f"\nFixed Parameters:")
    print(f"  Head Dimension (D): {FIXED_PARAMS['head_dim']}")
    print(f"  Number of Heads (H): {FIXED_PARAMS['num_heads']}")
    print(f"  Data Type: {FIXED_PARAMS['dtype_name']}")
    print(f"\nDefault Settings:")
    print(f"  Warmup Iterations: {DEFAULT_PARAMS['num_warmup']}")
    print(f"  Benchmark Iterations: {DEFAULT_PARAMS['num_iterations']}")
    print(f"  Device: {DEFAULT_PARAMS['device']}")
    print(f"\nAvailable Implementations:")
    print(f"  Wave: {'Yes' if WAVE_AVAILABLE else 'No'}")
    print(f"  Triton: {'Yes' if TRITON_AVAILABLE else 'No'}")
    if WAVE_AVAILABLE:
        print(f"\nWave-Specific Tuning Parameters:")
        print(f"  Schedule: Manual prefetch schedule (tagged kernel)")
        print(f"  Waves Per EU: {WAVES_PER_EU_VALUES}")
        print(f"  Scheduling Barriers: {USE_SCHEDULING_BARRIERS_VALUES}")
        print(f"  MMA Types: {len(MMA_TYPES_FP16)}")
        print(f"  BLOCK_N values: {BLOCK_N_VALUES}")
        print(f"  BLOCK_K2 values: {BLOCK_K2_VALUES}")
    print("\n" + "=" * 80)

    if not WAVE_AVAILABLE and not TRITON_AVAILABLE:
        print("Error: Neither Wave nor Triton is available. Cannot run benchmarks.")
        return results

    # Calculate total configs
    if WAVE_AVAILABLE:
        total_configs = (
            len(CAUSAL_VALUES)
            * len(BATCH_SEQLEN_PAIRS)
            * len(MMA_TYPES_FP16)
            * len(BLOCK_N_VALUES)
            * len(BLOCK_K2_VALUES)
            * len(WAVES_PER_EU_VALUES)
            * len(USE_SCHEDULING_BARRIERS_VALUES)
        )
    else:
        # Run with just Triton configs
        total_configs = len(CAUSAL_VALUES) * len(BATCH_SEQLEN_PAIRS)

    current_config = 0

    # If Wave is available, iterate over Wave-specific params
    if WAVE_AVAILABLE:
        for is_causal in CAUSAL_VALUES:
            for batch_size, seq_len in BATCH_SEQLEN_PAIRS:
                if batch_size > 1:
                    print(
                        f"skipping, batch_size {batch_size} > 1 not yet supported by manual schedule"
                    )
                    continue

                for mma_type in MMA_TYPES_FP16:
                    for block_n in BLOCK_N_VALUES:
                        for block_k2 in BLOCK_K2_VALUES:
                            for waves_per_eu in WAVES_PER_EU_VALUES:
                                for use_barriers in USE_SCHEDULING_BARRIERS_VALUES:
                                    current_config += 1
                                    mma_short = f"{mma_type.name.split('_')[1]}"
                                    print(
                                        f"\n[{current_config}/{total_configs}] Running: "
                                        f"B={batch_size}, N_CTX={seq_len}, "
                                        f"causal={is_causal}, mma={mma_short}, "
                                        f"tiles={block_n}x{block_k2}, "
                                        f"barriers={use_barriers}"
                                    )

                                    try:
                                        result = benchmark_attention_comparison(
                                            batch_size=batch_size,
                                            num_heads=FIXED_PARAMS["num_heads"],
                                            seq_len=seq_len,
                                            head_dim=FIXED_PARAMS["head_dim"],
                                            is_causal=bool(is_causal),
                                            mma_type=mma_type,
                                            block_n=block_n,
                                            block_k2=block_k2,
                                            waves_per_eu=waves_per_eu,
                                            use_scheduling_barriers=use_barriers,
                                            use_buffer_ops=USE_BUFFER_OPS_VALUES[0],
                                            canonicalize=CANONICALIZE_VALUES[0],
                                            num_warmup=DEFAULT_PARAMS["num_warmup"],
                                            num_iterations=DEFAULT_PARAMS[
                                                "num_iterations"
                                            ],
                                            dtype=DEFAULT_PARAMS["dtype"],
                                            device=DEFAULT_PARAMS["device"],
                                        )
                                        results.append(result)

                                        # Print summary
                                        triton_str = (
                                            f"Triton: {result['triton_avg_time_ms']:.3f} ms "
                                            f"({result['triton_throughput_tflops']:.2f} TFLOPs)"
                                            if result["triton_avg_time_ms"]
                                            else "Triton: N/A"
                                        )
                                        wave_str = (
                                            f"Wave: {result['wave_avg_time_ms']:.3f} ms "
                                            f"({result['wave_throughput_tflops']:.2f} TFLOPs)"
                                            if result["wave_avg_time_ms"]
                                            else "Wave: N/A"
                                        )
                                        speedup_str = (
                                            f"Speedup: {result['wave_vs_triton_speedup']:.2f}x"
                                            if result["wave_vs_triton_speedup"]
                                            else ""
                                        )
                                        print(f"  {triton_str}")
                                        print(f"  {wave_str}")
                                        if speedup_str:
                                            print(f"  {speedup_str}")
                                    except Exception as e:
                                        print(f"  Error: {e}")
    else:
        # Run Triton-only benchmarks
        for is_causal in CAUSAL_VALUES:
            for batch_size, seq_len in BATCH_SEQLEN_PAIRS:
                current_config += 1
                print(
                    f"\n[{current_config}/{total_configs}] Running: "
                    f"B={batch_size}, N_CTX={seq_len}, causal={is_causal}"
                )

                try:
                    result = benchmark_attention_comparison(
                        batch_size=batch_size,
                        num_heads=FIXED_PARAMS["num_heads"],
                        seq_len=seq_len,
                        head_dim=FIXED_PARAMS["head_dim"],
                        is_causal=bool(is_causal),
                        mma_type=None,
                        block_n=BLOCK_N_VALUES[0],
                        block_k2=BLOCK_K2_VALUES[0],
                        waves_per_eu=WAVES_PER_EU_VALUES[0],
                        use_scheduling_barriers=False,
                        use_buffer_ops=False,
                        canonicalize=True,
                        num_warmup=DEFAULT_PARAMS["num_warmup"],
                        num_iterations=DEFAULT_PARAMS["num_iterations"],
                        dtype=DEFAULT_PARAMS["dtype"],
                        device=DEFAULT_PARAMS["device"],
                    )
                    results.append(result)

                    if result["triton_avg_time_ms"]:
                        print(
                            f"  Triton: {result['triton_avg_time_ms']:.3f} ms "
                            f"({result['triton_throughput_tflops']:.2f} TFLOPs)"
                        )
                except Exception as e:
                    print(f"  Error: {e}")

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
        "wave_available": WAVE_AVAILABLE,
        "triton_available": TRITON_AVAILABLE,
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

    # Base fieldnames (always included)
    fieldnames = [
        "batch_size",
        "num_heads",
        "seq_len",
        "head_dim",
        "is_causal",
    ]

    # Add Wave-specific fields if Wave is available
    if WAVE_AVAILABLE:
        fieldnames.extend(
            [
                "mma_type",
                "block_n",
                "block_k2",
                "waves_per_eu",
                "use_scheduling_barriers",
                "wave_avg_time_ms",
                "wave_throughput_tflops",
            ]
        )

    # Add Triton-specific fields if Triton is available
    if TRITON_AVAILABLE:
        fieldnames.extend(
            [
                "triton_avg_time_ms",
                "triton_throughput_tflops",
            ]
        )

    # Add comparison field only if both are available
    if WAVE_AVAILABLE and TRITON_AVAILABLE:
        fieldnames.append("wave_vs_triton_speedup")

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"✓ Results saved to CSV: {filename}")


def print_results_table(results: List[Dict[str, Any]]):
    """Print benchmark results as a formatted table."""
    if not results:
        print("\nNo results to display")
        return

    # Define base columns (always displayed)
    columns = [
        ("batch_size", "B"),
        ("seq_len", "N_CTX"),
        ("is_causal", "Caus"),
    ]

    # Add Wave-specific columns if Wave is available
    if WAVE_AVAILABLE:
        columns.extend(
            [
                ("mma_type", "MMA"),
                ("use_scheduling_barriers", "Barriers"),
                ("wave_avg_time_ms", "Wave(ms)"),
                ("wave_throughput_tflops", "Wave TF"),
            ]
        )

    # Add Triton-specific columns if Triton is available
    if TRITON_AVAILABLE:
        columns.extend(
            [
                ("triton_avg_time_ms", "Triton(ms)"),
                ("triton_throughput_tflops", "Triton TF"),
            ]
        )

    # Add comparison column only if both are available
    if WAVE_AVAILABLE and TRITON_AVAILABLE:
        columns.append(("wave_vs_triton_speedup", "Speedup"))

    # Calculate column widths
    col_widths = {}
    for key, header in columns:
        max_width = len(header)
        for result in results:
            value = result.get(key, "N/A")
            if value is None:
                value_str = "N/A"
            elif isinstance(value, float):
                value_str = f"{value:.2f}"
            elif isinstance(value, bool):
                value_str = str(value)
            else:
                value_str = str(value)
            max_width = max(max_width, len(value_str))
        col_widths[key] = max_width

    total_width = sum(col_widths.values()) + len(columns) - 1

    # Choose appropriate title based on availability
    if WAVE_AVAILABLE and TRITON_AVAILABLE:
        title = "BENCHMARK RESULTS - WAVE vs TRITON"
    elif WAVE_AVAILABLE:
        title = "BENCHMARK RESULTS - WAVE"
    else:
        title = "BENCHMARK RESULTS - TRITON"

    print("\n" + "=" * total_width)
    print(title)
    print("=" * total_width)

    # Print header
    header_parts = []
    for key, header in columns:
        width = col_widths[key]
        header_parts.append(f"{header:>{width}}")
    print(" ".join(header_parts))
    print("-" * total_width)

    # Print results
    for result in results:
        row_parts = []
        for key, _ in columns:
            value = result.get(key, "N/A")
            width = col_widths[key]

            if value is None:
                value_str = f"{'N/A':>{width}}"
            elif isinstance(value, float):
                value_str = f"{value:>{width}.2f}"
            elif isinstance(value, bool):
                value_str = f"{str(value):>{width}}"
            else:
                value_str = f"{str(value):>{width}}"

            row_parts.append(value_str)
        print(" ".join(row_parts))

    print("=" * total_width)


def main():
    """Main entry point for the benchmark script."""
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"attention_benchmark_wave_vs_triton_{timestamp}.json"
    csv_filename = f"attention_benchmark_wave_vs_triton_{timestamp}.csv"

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
