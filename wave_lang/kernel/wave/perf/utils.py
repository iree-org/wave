# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch

from wave_lang.support.logging import get_logger

logger = get_logger("wave.perf.utils")

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
) -> Tuple[float, Any, Any]:
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
    try:
        import sqlite3

        import pandas as pd
        from rpdTracerControl import rpdTracerControl
    except ImportError:
        print("rpdTracerControl not found, skipping profiling")
        raise SystemExit(1)

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


# ---------------------------------------------------------------------------
# rocprofv3 tracing and parsing (shared by benchmark_mxfp4 and benchmark_asm_backend)
# ---------------------------------------------------------------------------


def _csv_read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _pick_column(sample_row: dict[str, str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in sample_row:
            return c
    lowered = {k.lower(): k for k in sample_row.keys()}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def _parse_float(s: Any) -> float:
    if s is None:
        raise ValueError("Missing numeric field")
    txt = str(s).strip()
    if txt == "":
        raise ValueError("Empty numeric field")
    return float(txt)


def _parse_int(s: Any) -> int:
    if s is None:
        raise ValueError("Missing integer field")
    txt = str(s).strip()
    if txt == "":
        raise ValueError("Empty integer field")
    return int(float(txt))


def find_rocprof_outputs(
    output_dir: Path, prefix: Optional[str] = None
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (stats_path, trace_path) for rocprofv3 CSV outputs.

    If prefix is set, look for {prefix}_kernel_stats.csv and {prefix}_kernel_trace.csv.
    If prefix is None, glob for *kernel*_stats*.csv and *kernel*_trace*.csv under output_dir.
    """
    if prefix is not None:
        stats = output_dir / f"{prefix}_kernel_stats.csv"
        trace = output_dir / f"{prefix}_kernel_trace.csv"
        if stats.exists() or trace.exists():
            return (
                stats if stats.exists() else None,
                trace if trace.exists() else None,
            )
    stats_matches = sorted(output_dir.glob("**/*kernel*_stats*.csv"))
    trace_matches = sorted(output_dir.glob("**/*kernel*_trace*.csv"))
    return (
        stats_matches[0] if stats_matches else None,
        trace_matches[0] if trace_matches else None,
    )


def parse_rocprof_kernel_stats(path: Path, kernel_regex: str = "gemm") -> dict:
    """Parse rocprofv3 kernel_stats CSV; return dict with kernel_name, mean_duration_us, total_calls.

    path may be a file or a directory; if dir, first stats file is used (via find_rocprof_outputs).
    Returns empty dict if no matching row or on parse failure; raises RuntimeError on I/O/parse error.
    """
    try:
        if path.is_dir():
            stats_path, _ = find_rocprof_outputs(path, prefix=None)
            csv_path = stats_path
            if csv_path is None:
                return {}
        else:
            csv_path = path
        if csv_path is None or not csv_path.exists():
            return {}
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Name" in row and (not kernel_regex or kernel_regex in row["Name"]):
                    if "AverageNs" not in row:
                        return {}
                    average_ns = float(row["AverageNs"])
                    return {
                        "kernel_name": row["Name"],
                        "mean_duration_us": average_ns / 1000.0,
                        "total_calls": int(row.get("Calls", 1)),
                    }
        return {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse rocprof output from {path}: {e}") from e


def rocprof_avg_ms_from_kernel_trace_last_n(
    trace_csv: Path, *, num_iterations: int
) -> float:
    """Average time (ms) for the last N dispatches of the most frequent kernel in the trace.

    Handles column name variants (Kind, Kernel_Name, Start_Timestamp, etc.).
    """
    rows = _csv_read(trace_csv)
    if not rows:
        raise ValueError(f"Empty rocprof trace: {trace_csv}")

    kind_col = _pick_column(rows[0], ["Kind", "kind"])
    name_col = _pick_column(
        rows[0], ["Kernel_Name", "KernelName", "Kernel Name", "Name", "name"]
    )
    start_col = _pick_column(
        rows[0], ["Start_Timestamp", "StartNs", "Start (ns)", "start_ns"]
    )
    end_col = _pick_column(rows[0], ["End_Timestamp", "EndNs", "End (ns)", "end_ns"])

    if name_col is None or start_col is None or end_col is None:
        raise ValueError(
            f"Unexpected kernel trace schema in {trace_csv}. "
            f"cols={list(rows[0].keys())}"
        )

    def is_kernel_row(r: dict[str, str]) -> bool:
        if kind_col is None:
            return True
        return str(r.get(kind_col, "")).strip().upper() == "KERNEL_DISPATCH"

    per_kernel: dict[str, list[float]] = {}
    for r in rows:
        if not is_kernel_row(r):
            continue
        kname = str(r.get(name_col, "")).strip()
        if not kname:
            continue
        start = _parse_float(r.get(start_col))
        end = _parse_float(r.get(end_col))
        dur_ns = end - start
        if dur_ns < 0:
            continue
        per_kernel.setdefault(kname, []).append(dur_ns)

    if not per_kernel:
        raise ValueError(f"No kernel dispatch rows found in {trace_csv}")

    target_kernel = max(per_kernel.items(), key=lambda kv: len(kv[1]))[0]
    durations = per_kernel[target_kernel]

    if len(durations) < num_iterations:
        raise ValueError(
            f"Kernel trace has only {len(durations)} occurrences of selected kernel "
            f"'{target_kernel}', expected at least {num_iterations}. "
            f"Trace file: {trace_csv}"
        )

    tail = durations[-num_iterations:]
    avg_ms = (sum(tail) / len(tail)) / 1e6
    return avg_ms


def get_rocprofv3_cmd(
    output_path: Path,
    output_file_prefix: Optional[str] = None,
    kernel_regex: str = "gemm",
    att_library_path: Optional[str] = None,
) -> list[str]:
    """Build rocprofv3 argv up to and including '--'; callers append worker cmd."""
    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "--stats",
        "TRUE",
        "--output-format",
        "csv",
    ]
    if kernel_regex:
        cmd.extend(["--kernel-include-regex", kernel_regex])
    if output_file_prefix is None:
        cmd.extend(["-d", str(output_path)])
    else:
        cmd.extend(
            [
                "--output-directory",
                str(output_path),
                "--output-file",
                output_file_prefix,
            ]
        )
    if att_library_path:
        cmd = cmd[:4] + ["--att", "--att-library-path", att_library_path] + cmd[4:]
    cmd.append("--")
    return cmd


def ensure_rocprofv3() -> str:
    """Return path to rocprofv3 executable or raise RuntimeError with install hint."""
    rocprof = shutil.which("rocprofv3")
    if not rocprof:
        raise RuntimeError("rocprofv3 not found in PATH. Install ROCm/rocprofiler-sdk.")
    return rocprof
