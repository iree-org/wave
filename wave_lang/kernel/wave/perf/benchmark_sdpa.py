# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json

import torch

import wave_lang.kernel.wave.nn as wave_nn
from wave_lang.kernel.wave.perf.utils import BaseBenchmark, benchmark_kernel
from wave_lang.support.logging import get_logger

try:
    from flash_attn import flash_attn_func

    _flash_attn_enabled = True
except:
    _flash_attn_enabled = False

logger = get_logger("wave.perf.benchmarking_sdpa")


class SPDABenchmark(BaseBenchmark):
    def __init__(self):
        usage = (
            "Example:\n"
            "  WAVE_LOG_LEVEL=INFO \\\n"
            "  python -u wave_lang/kernel/wave/perf/benchmark_sdpa.py \\\n"
            "    --config wave_lang/kernel/wave/perf/benchmark_configs.json \\\n"
            "    --num_warmup 10 \\\n"
            "    --num_iterations 100 \\\n"
            "    --output bm_trace.rpd \\\n"
            "    [--fa]\n\n"
            "Notes:\n"
            "  --fa : Use FlashAttention instead of Wave kernel (if available).\n"
        )
        super().__init__(
            description="Benchmark Wave attention (sdpa) kernel",
            epilog=usage,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.parser.add_argument(
            "--fa",
            action="store_true",
            help="Benchmark FlashAttention attention kernel",
        )


def benchmark_sdpa():
    """Benchmark wave_sdpa with given BHSD shapes.

    Args:
        batch_size: Batch dimension
        num_heads: Number of attention heads
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        head_dim: Head dimension
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
    """
    args = SPDABenchmark().parse()

    if args.fa and not _flash_attn_enabled:
        raise ValueError("FA enabled but not installed on system!")

    use_flash_attn = args.fa and _flash_attn_enabled

    # Load configuration from JSON file
    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda:0")

    # Run benchmark for each configuration
    for shape_config in config["attention_bhsd_shapes"]:
        batch_size = shape_config["batch_size"]
        num_heads = shape_config["num_heads"]
        seq_len_q = shape_config["seq_len_q"]
        seq_len_k = shape_config["seq_len_k"]
        head_dim = shape_config["head_dim"]

        # Create input tensors
        query = torch.randn(
            [batch_size, num_heads, seq_len_q, head_dim],
            device=device,
            dtype=torch.float16,
        )
        key = torch.randn(
            [batch_size, num_heads, seq_len_k, head_dim],
            device=device,
            dtype=torch.float16,
        )
        value = torch.randn(
            [batch_size, num_heads, seq_len_k, head_dim],
            device=device,
            dtype=torch.float16,
        )

        sdpa_inputs = lambda: (query, key, value)

        # Read profiling output and compute average time for speculative_sampling
        avg_time_us, _, _ = benchmark_kernel(
            inputs=sdpa_inputs,
            kernel_func=(
                flash_attn_func if use_flash_attn else wave_nn.functional.wave_sdpa
            ),
            warmup_iters=args.num_warmup,
            benchmark_iters=args.num_iterations,
            output_filename=args.output,
            kernel_name="base_attention",
        )

        throughput = (4 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim) / (
            avg_time_us / 1e6
        )

        logger.info("\nBenchmark Results for BHSD shapes:")
        logger.info("Batch size: %d", batch_size)
        logger.info("Number of heads: %d", num_heads)
        logger.info("Query sequence length: %d", seq_len_q)
        logger.info("Key sequence length: %d", seq_len_k)
        logger.info("Head dimension: %d", head_dim)
        logger.info("Average time per iteration: %.2f ms", avg_time_us / 1000)
        logger.info("Throughput: %.2f TFLOPs", throughput / 1e12)


if __name__ == "__main__":
    benchmark_sdpa()
