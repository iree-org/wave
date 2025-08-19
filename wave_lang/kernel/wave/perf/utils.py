# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import argparse
from typing import Callable, Optional, Tuple

import torch

from wave_lang.support.logging import get_logger

logger = get_logger("wave.perf.utils")

# Import rpdTracerControl, required for profiling.
# If the library is unavailable, profiling will be skipped.
try:
    import sqlite3

    import pandas as pd
    from rpdTracerControl import rpdTracerControl
except ImportError:
    print("rpdTraceControl not found, skipping profiling")
    exit(1)

DEFAULT_OUTPUT_FILENAME = "trace.rpd"


class BaseBenchmark:
    """Base parser class for benchmarking kernels."""

    def __init__(
        self,
        description: str = "Benchmarking CLI for Wave kernels",
        epilog: str = "",
        formatter_class=argparse.RawTextHelpFormatter,
    ):
        self.parser = argparse.ArgumentParser(
            description=description,
            epilog=epilog,
            formatter_class=formatter_class,
        )
        self._add_common_args()

    def _add_common_args(self) -> None:
        self.parser.add_argument(
            "--output",
            type=str,
            default=DEFAULT_OUTPUT_FILENAME,
            help="Path to output trace file",
        )
        self.parser.add_argument(
            "--config", type=str, required=True, help="Path to JSON config file"
        )
        self.parser.add_argument(
            "--num_warmup", type=int, default=10, help="Number of warmup iterations"
        )
        self.parser.add_argument(
            "--num_iterations",
            type=int,
            default=100,
            help="Number of benchmark iterations",
        )

    def parse(self):
        return self.parser.parse_args()


def benchmark_kernel(
    inputs: Callable[[], Tuple],
    kernel_func: Callable,
    warmup_iters: int,
    benchmark_iters: int,
    output_filename: str,
    kernel_name: Optional[str] = None,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Run benchmark on `kernel_func` using inputs from `prepare_inputs`.

    Args:
        inputs: Function returning args to pass to kernel_func
        kernel_func: Kernel function to profile
        warmup_iters: Number of warmup runs (not profiled)
        benchmark_iters: Number of profiled runs
        output_filename: File to write RPD trace to
        kernel_name: Kernel name to be profiled

    Returns:
        Tuple of (avg_time_us, kernel_info_df, top_df)
    """
    # Warmup
    for _ in range(warmup_iters):
        kernel_func(*inputs())

    # Synchronize GPU
    torch.cuda.synchronize()

    # Initialize RPD tracer
    rpdTracerControl.setFilename(name=output_filename, append=False)
    tracer = rpdTracerControl()
    tracer.start()

    # Benchmark with profiling
    for _ in range(benchmark_iters):
        kernel_func(*inputs())

    torch.cuda.synchronize()

    # Stop profiling and get results
    tracer.stop()
    tracer.flush()

    with sqlite3.connect(output_filename) as conn:
        # Analyze RPD trace and return (avg time in µs, kernel_info_df, top_df).
        df_top = pd.read_sql_query("SELECT * FROM top", conn)
        if df_top.empty:
            logger.error("Empty 'top' dataframe from profiling output.")
            raise ValueError("'df_top' DataFrame is empty.")

        df_kernel_info = pd.read_sql_query(
            """
            SELECT s1.string AS api_name, k.stream, k.gridX, k.gridY, k.gridZ,
                   k.workgroupX, k.workgroupY, k.workgroupZ, s2.string AS kernel_name
            FROM rocpd_kernelapi k
            LEFT JOIN rocpd_string s1 ON k.api_ptr_id = s1.id
            LEFT JOIN rocpd_string s2 ON k.kernelName_id = s2.id;
            """,
            conn,
        ).drop_duplicates()

    logger.debug("Kernel Info:\n%s", df_kernel_info)
    logger.info("Top Kernels:\n%s", df_top.head(100))

    row = df_top[df_top["Name"] == kernel_name]
    if not row.empty:
        avg_time_us = row["Ave_us"].iloc[0]
    else:
        logger.warning("Kernel '%s' not found in profiling trace.", kernel_name)
        avg_time_us = 0.0

    return (avg_time_us, df_kernel_info, df_top)
