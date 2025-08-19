# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tests.kernel.wave.speculative_decode_test import testReferenceSpeculativeDecoding
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.perf.utils import BaseBenchmark, benchmark_kernel
from wave_lang.support.logging import get_logger

logger = get_logger("wave.perf.benchmark_speculative_sampling")


class SpecSamplingBenchmark(BaseBenchmark):
    def __init__(self):
        usage = (
            "Example:\n"
            "  WAVE_LOG_LEVEL=INFO \\\n"
            "  python -u wave_lang/kernel/wave/perf/benchmark_speculative_sampling.py \\\n"
            "    --config wave_lang/kernel/wave/perf/benchmark_configs.json \\\n"
            "    --num_warmup 10 \\\n"
            "    --num_iterations 100 \\\n"
            "    --output bm_trace.rpd"
        )
        super().__init__(
            description="Benchmark Wave speculative sampling kernel",
            epilog=usage,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.parser.add_argument(
            "--sampling_backend",
            type=str,
            default="wave",
            help="Sampling backend to use",
        )


def benchmark_speculative_sampling() -> None:
    args = SpecSamplingBenchmark().parse()

    if args.sampling_backend != "wave":
        raise NotImplementedError(
            "Only 'wave' sampling backend is currently supported."
        )

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    if not config_path.name.endswith(".json"):
        raise ValueError("Config file must be a .json file.")

    with open(args.config, "r") as f:
        config: Dict[str, List[Dict[str, Any]]] = json.load(f)

    # Load configuration
    for shape_config in config["spec_sampling_shapes"]:
        threshold_single = shape_config["threshold_single"]
        threshold_acc = shape_config["threshold_acc"]
        expected_predicts = shape_config["expected_predicts"]
        expected_accept_index = shape_config["expected_accept_index"]
        expected_accept_token_num = shape_config["expected_accept_token_num"]

        spec_sampling_inputs = lambda: (
            threshold_single,
            threshold_acc,
            expected_predicts,
            expected_accept_index,
            expected_accept_token_num,
        )

        # Profiles the speculative sampling kernel with warmup and benchmark iterations.
        avg_time_us, _, _ = benchmark_kernel(
            inputs=spec_sampling_inputs,
            kernel_func=testReferenceSpeculativeDecoding,
            warmup_iters=args.num_warmup,
            benchmark_iters=args.num_iterations,
            output_filename=args.output,
            kernel_name="speculative_sampling",
        )
        logger.info("Average time for 'speculative_sampling': %.3f us", avg_time_us)

    logger.info("Benchmark completed successfully.")


if __name__ == "__main__":
    benchmark_speculative_sampling()
