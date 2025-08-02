# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import argparse
import importlib
import json

from pathlib import Path
import sys
from typing import Any, Dict, List

import torch

from wave_lang.kernel.lang.global_symbols import *
from tests.kernel.wave.speculative_decode_test import testReferenceSpeculativeDecoding

from wave_lang.kernel.wave.perf.utils import analyze_rpd_trace

rpd_tracer_module = importlib.util.find_spec("rpdTracerControl")
if rpd_tracer_module is None:
    logger.error("rpdTracerControl not found. Profiling is disabled.")
    sys.exit(1)
else:
    rpdTracerControl = importlib.import_module("rpdTracerControl").rpdTracerControl

DEFAULT_OUTPUT_FILENAME = "trace.rpd"


def benchmark_speculative_sampling(
    threshold_single: float,
    threshold_acc: float,
    expected_predicts: torch.Tensor,
    expected_accept_index: torch.Tensor,
    expected_accept_token_num: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> None:
    """
    Profiles the speculative sampling kernel with warmup and benchmark iterations.
    """
    logger.info("Starting warmup iterations")
    for _ in range(num_warmup):
        _ = testReferenceSpeculativeDecoding(
            threshold_single,
            threshold_acc,
            expected_predicts,
            expected_accept_index,
            expected_accept_token_num,
        )

    torch.cuda.synchronize()

    rpdTracerControl.setFilename(name=output_filename, append=False)
    tracer = rpdTracerControl()
    tracer.start()

    logger.info("Starting benchmarking iterations")
    for _ in range(num_iterations):
        _ = testReferenceSpeculativeDecoding(
            threshold_single,
            threshold_acc,
            expected_predicts,
            expected_accept_index,
            expected_accept_token_num,
        )

    torch.cuda.synchronize()
    tracer.stop()
    tracer.flush()

    # Read profiling output and compute average time for speculative_sampling
    avg_time_us, _, _ = analyze_rpd_trace(
        output_filename, kernel_name="speculative_sampling"
    )
    logger.info("Average time for 'speculative_sampling': %.3f us", avg_time_us)


def main() -> None:
    usage = (
        "Example:\n"
        "  python -u wave_lang/kernel/wave/perf/benchmark_speculative_sampling.py \\\n"
        "    --config wave_lang/kernel/wave/perf/benchmark_configs.json \\\n"
        "    --num_warmup 10 \\\n"
        "    --num_iterations 100 \\\n"
        "    --output bm_trace.rpd"
    )
    parser = argparse.ArgumentParser(
        description="Benchmark Wave speculative sampling kernel",
        epilog=usage,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help="Path to output trace file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.json",
        required=True,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--num_warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--sampling_backend", type=str, default="wave", help="Sampling backend to use"
    )
    args = parser.parse_args()

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

        benchmark_speculative_sampling(
            threshold_single,
            threshold_acc,
            expected_predicts,
            expected_accept_index,
            expected_accept_token_num,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            output_filename=args.output,
        )

    logger.info("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
