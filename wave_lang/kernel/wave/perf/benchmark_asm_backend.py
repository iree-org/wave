# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Benchmark script to compare ASM backend vs LLVM backend performance for GEMM kernels.

This script:
1. Reads GEMM configuration from benchmark_configs.json
2. Runs the GEMM kernel with both ASM and LLVM backends
3. Verifies correctness of both backends against PyTorch reference
4. Reports performance metrics for comparison

Example usage:
    python -u wave_lang/kernel/wave/perf/benchmark_asm_backend.py \
        --config wave_lang/kernel/wave/perf/benchmark_configs.json \
        --num_warmup 10 \
        --num_iterations 100

Note: Each wave handles a WAVE_M x WAVE_N tile, which can be larger than the MMA
intrinsic size (16x16 for F32_16x16x16_F16). The wave internally performs multiple
MMA operations to cover its tile. The number of waves per workgroup is determined
by BLOCK_M/WAVE_M along M and BLOCK_N/WAVE_N along N.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from torch.testing import assert_close

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
    get_default_arch,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from wave_lang.support.logging import get_logger
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)

# Disable wave cache for fair comparison
os.environ["WAVE_CACHE_ON"] = "0"

logger = get_logger("wave.perf.benchmark_asm_backend")

_DEFAULT_CHECK_RTOL = float(os.environ.get("WAVE_BENCH_CHECK_RTOL", "1e-4"))
_DEFAULT_CHECK_ATOL = float(os.environ.get("WAVE_BENCH_CHECK_ATOL", "2e-5"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark ASM vs LLVM backend for GEMM kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -u wave_lang/kernel/wave/perf/benchmark_asm_backend.py \\
    --config wave_lang/kernel/wave/perf/benchmark_configs.json \\
    --num_warmup 10 \\
    --num_iterations 100 \\
    --check_correctness

Notes:
  - Both backends are tested for correctness against PyTorch reference
  - Performance is measured via rocprofv3 kernel tracing (no CUDA events)
  - Each wave handles a WAVE_M x WAVE_N tile (can be larger than 16x16 MMA intrinsic)
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file with GEMM shapes",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--check_correctness",
        action="store_true",
        default=False,
        help="Check correctness against PyTorch reference (default: False)",
    )
    parser.add_argument(
        "--skip_llvm",
        action="store_true",
        help="Skip LLVM backend benchmark (only run ASM)",
    )
    parser.add_argument(
        "--skip_asm",
        action="store_true",
        help="Skip ASM backend benchmark (only run LLVM)",
    )
    # Internal worker-mode args (used when launched under rocprofv3).
    parser.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_backend",
        type=str,
        choices=["asm", "llvm"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_shape_name",
        type=str,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def create_gemm_kernel(block_m: int, block_n: int, wave_m: int, wave_n: int):
    """Create a GEMM kernel with the specified block and wave configuration.

    Args:
        block_m: Block size along M dimension
        block_n: Block size along N dimension
        wave_m: Wave tile size along M dimension (each wave handles this many rows)
        wave_n: Wave tile size along N dimension (each wave handles this many cols)

    Returns:
        Tuple of (kernel function, symbol tuple)

    Note:
        The wave tile can be larger than the MMA intrinsic size (16x16).
        Each wave will perform multiple MMA operations to cover its tile.
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    wave_size = 64

    # Validate block sizes are divisible by wave sizes
    assert (
        block_m % wave_m == 0
    ), f"BLOCK_M ({block_m}) must be divisible by WAVE_M ({wave_m})"
    assert (
        block_n % wave_n == 0
    ), f"BLOCK_N ({block_n}) must be divisible by WAVE_N ({wave_n})"

    # Check wave count limit (max 16 waves = 1024 threads per workgroup)
    waves_along_m = block_m // wave_m
    waves_along_n = block_n // wave_n
    waves_per_wg = waves_along_m * waves_along_n
    assert waves_per_wg <= 16, (
        f"Too many waves per workgroup: {waves_per_wg} (max 16). "
        f"Reduce BLOCK_M or BLOCK_N, or increase WAVE_M/WAVE_N."
    )

    # Use BLOCK_M / waves_along_m for WaveConstraint (like the test files do)
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / waves_along_m),
        tkw.WaveConstraint(N, BLOCK_N / waves_along_n),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def gemm_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        """GEMM kernel: C = A @ B^T with K-loop."""
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    symbols = (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, ADDRESS_SPACE, ADDRESS_SPACE_0)
    return gemm_kernel, symbols


def compile_and_run_kernel(
    kernel,
    symbols,
    shape_config: dict,
    backend: str,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
):
    """Compile and run the GEMM kernel with the specified backend.

    Args:
        kernel: The Wave kernel function
        symbols: Symbol tuple from create_gemm_kernel
        shape_config: Configuration dict with M, N, K, BLOCK_M, etc.
        backend: Either "asm" or "llvm"
        a: Input tensor A (M x K)
        b: Input tensor B (N x K)
        c: Output tensor C (M x N)

    Returns:
        The compiled kernel function
    """
    M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, ADDRESS_SPACE, ADDRESS_SPACE_0 = symbols

    options = WaveCompileOptions(
        subs={
            M: shape_config["M"],
            N: shape_config["N"],
            K: shape_config["K"],
            BLOCK_M: shape_config["BLOCK_M"],
            BLOCK_N: shape_config["BLOCK_N"],
            BLOCK_K: shape_config["BLOCK_K"],
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        backend=backend,
        wave_runtime=True,
        compile_to_mlir=False,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, kernel)
    compiled_kernel(a, b, c)

    return compiled_kernel


def benchmark_kernel(
    compiled_kernel,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
) -> float:
    raise RuntimeError(
        "benchmark_kernel() is deprecated in this script. Timing must be collected "
        "via rocprofv3 (no CUDA events)."
    )


def compute_tflops(m: int, n: int, k: int, time_ms: float) -> float:
    """Compute throughput in TFLOPs for GEMM.

    Args:
        m: M dimension
        n: N dimension
        k: K dimension
        time_ms: Time in milliseconds

    Returns:
        Throughput in TFLOPs
    """
    # GEMM FLOPs = 2 * M * N * K (multiply-add)
    flops = 2 * m * n * k
    time_s = time_ms / 1000.0
    tflops = flops / time_s / 1e12
    return tflops


def _csv_read(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _pick_column(sample_row: dict[str, str], candidates: list[str]) -> Optional[str]:
    # Exact match first.
    for c in candidates:
        if c in sample_row:
            return c
    # Case-insensitive match next.
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


def _find_rocprof_outputs(
    output_dir: Path, prefix: str
) -> tuple[Optional[Path], Optional[Path]]:
    # Preferred files when --output-file is set:
    #   <prefix>_kernel_stats.csv and <prefix>_kernel_trace.csv
    stats = output_dir / f"{prefix}_kernel_stats.csv"
    trace = output_dir / f"{prefix}_kernel_trace.csv"
    if stats.exists() or trace.exists():
        return (stats if stats.exists() else None, trace if trace.exists() else None)

    # Fallback: best-effort search across rocprofv3 versions.
    stats_matches = sorted(output_dir.glob("*kernel*_stats*.csv"))
    trace_matches = sorted(output_dir.glob("*kernel*_trace*.csv"))
    return (
        stats_matches[0] if stats_matches else None,
        trace_matches[0] if trace_matches else None,
    )


def _rocprof_avg_ms_from_kernel_trace_last_n(
    trace_csv: Path, *, num_iterations: int
) -> float:
    """Compute average time (ms) for the last N dispatches of the most frequent kernel.

    This avoids requiring ROCTx/marker ranges (which may not be available for the
    Python version in use) by selecting the dominant repeatedly-called kernel and
    averaging its last num_iterations dispatch durations.
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

    # Filter to kernel dispatch rows if "Kind" exists; otherwise assume all rows are kernels.
    def is_kernel_row(r: dict[str, str]) -> bool:
        if kind_col is None:
            return True
        return str(r.get(kind_col, "")).strip().upper() == "KERNEL_DISPATCH"

    # Gather per-kernel durations in order.
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

    # Choose the most frequent kernel as the benchmark target.
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


def _rocprofv3_benchmark_avg_ms(
    *,
    config_path: str,
    shape_name: str,
    backend: str,
    num_warmup: int,
    num_iterations: int,
) -> float:
    rocprof = shutil.which("rocprofv3")
    if not rocprof:
        raise RuntimeError("rocprofv3 not found in PATH. Install ROCm/rocprofiler-sdk.")

    script_path = os.path.abspath(__file__)

    with tempfile.TemporaryDirectory(prefix="wave_rocprofv3_") as tmpdir:
        out_dir = Path(tmpdir)
        prefix = f"wave_bench_{backend}_{shape_name}".replace("/", "_").replace(
            " ", "_"
        )

        cmd = [
            rocprof,
            "--kernel-trace",
            "--stats",
            "--output-format",
            "csv",
            "--output-directory",
            str(out_dir),
            "--output-file",
            prefix,
            "--",
            sys.executable,
            script_path,
            "--config",
            config_path,
            "--num_warmup",
            str(num_warmup),
            "--num_iterations",
            str(num_iterations),
            # Worker args:
            "--_worker",
            "--_backend",
            backend,
            "--_shape_name",
            shape_name,
        ]

        env = os.environ.copy()
        # Keep cache disabled for fair comparison.
        env["WAVE_CACHE_ON"] = "0"

        # Keep rocprofv3 output noise down in normal runs.
        env.setdefault("ROCPROFILER_LOG_LEVEL", "error")

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                "rocprofv3 run failed.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

        _stats_csv, trace_csv = _find_rocprof_outputs(out_dir, prefix)
        if trace_csv is None:
            present = sorted([p.name for p in out_dir.glob("*.csv")])
            raise RuntimeError(
                "Failed to find rocprofv3 kernel trace CSV output.\n"
                f"Output dir: {out_dir}\n"
                f"CSV files: {present}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

        # Compute average from the last N dispatches of the dominant kernel.
        return _rocprof_avg_ms_from_kernel_trace_last_n(
            trace_csv,
            num_iterations=num_iterations,
        )


def _worker_main(args) -> None:
    # Minimal logs in worker mode: rocprofv3 does not need our verbose logs.
    # (Keep exceptions visible via non-zero exit).
    with open(args.config, "r") as f:
        config = json.load(f)
    shapes = config.get("gemm_shapes", [])
    match = None
    for sc in shapes:
        if sc.get("name") == args._shape_name:
            match = sc
            break
    if match is None:
        raise ValueError(f"Shape '{args._shape_name}' not found in {args.config}")

    m = match["M"]
    n = match["N"]
    k = match["K"]
    block_m = match["BLOCK_M"]
    block_n = match["BLOCK_N"]
    wave_m = match["WAVE_M"]
    wave_n = match["WAVE_N"]

    a = device_randn((m, k), dtype=torch.float16)
    b = device_randn((n, k), dtype=torch.float16)
    c = device_zeros((m, n), dtype=torch.float32)

    kernel, symbols = create_gemm_kernel(block_m, block_n, wave_m, wave_n)
    compiled = compile_and_run_kernel(kernel, symbols, match, args._backend, a, b, c)

    # Warmup (outside measured region).
    for _ in range(args.num_warmup):
        compiled(a, b, c)
    torch.cuda.synchronize()

    # Benchmark region (no ROCTx dependency): rocprofv3 parsing selects the dominant
    # repeatedly-called kernel and averages the last N dispatches.
    for _ in range(args.num_iterations):
        compiled(a, b, c)
    torch.cuda.synchronize()


def _check_correctness(
    *,
    backend: str,
    shape_config: dict,
    a: torch.Tensor,
    b: torch.Tensor,
    expected: torch.Tensor,
) -> bool:
    """Compile + run once in-process and compare against reference."""
    m = shape_config["M"]
    n = shape_config["N"]
    block_m = shape_config["BLOCK_M"]
    block_n = shape_config["BLOCK_N"]
    wave_m = shape_config["WAVE_M"]
    wave_n = shape_config["WAVE_N"]

    c = device_zeros((m, n), dtype=torch.float32)
    kernel, symbols = create_gemm_kernel(block_m, block_n, wave_m, wave_n)
    compile_and_run_kernel(kernel, symbols, shape_config, backend, a, b, c)
    assert_close(c, expected, rtol=_DEFAULT_CHECK_RTOL, atol=_DEFAULT_CHECK_ATOL)
    return True


def _run_backend(
    *,
    backend: str,
    args,
    shape_config: dict,
    a: torch.Tensor,
    b: torch.Tensor,
    expected: torch.Tensor,
) -> Optional[dict[str, float]]:
    """Runs optional correctness, then benchmarks via rocprofv3."""
    m = shape_config["M"]
    n = shape_config["N"]
    k = shape_config["K"]

    if args.check_correctness:
        _check_correctness(
            backend=backend,
            shape_config=shape_config,
            a=a,
            b=b,
            expected=expected,
        )
        logger.info(f"{backend.upper()} correctness: PASSED")

    avg_time_ms = _rocprofv3_benchmark_avg_ms(
        config_path=args.config,
        shape_name=shape_config["name"],
        backend=backend,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
    )
    tflops = compute_tflops(m, n, k, avg_time_ms)
    logger.info(f"{backend.upper()} time: {avg_time_ms:.4f} ms")
    logger.info(f"{backend.upper()} throughput: {tflops:.4f} TFLOPs")
    return {"time_ms": avg_time_ms, "tflops": tflops}


def main():
    args = parse_args()

    if args._worker:
        if not args._backend or not args._shape_name:
            raise ValueError("Worker mode requires --_backend and --_shape_name")
        _worker_main(args)
        return

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    if "gemm_shapes" not in config:
        logger.error("No 'gemm_shapes' section found in config file")
        return

    arch = get_default_arch()
    logger.info(f"Running on architecture: {arch}")

    # Process each GEMM configuration
    for shape_config in config["gemm_shapes"]:
        logger.info("=" * 80)
        logger.info(f"Benchmark: {shape_config['name']}")
        logger.info(f"Description: {shape_config['description']}")
        logger.info("=" * 80)

        m = shape_config["M"]
        n = shape_config["N"]
        k = shape_config["K"]
        block_m = shape_config["BLOCK_M"]
        block_n = shape_config["BLOCK_N"]
        block_k = shape_config["BLOCK_K"]
        wave_m = shape_config["WAVE_M"]
        wave_n = shape_config["WAVE_N"]

        waves_along_m = block_m // wave_m
        waves_along_n = block_n // wave_n
        total_waves = waves_along_m * waves_along_n

        logger.info(f"Shape: M={m}, N={n}, K={k}")
        logger.info(f"Block: BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}")
        logger.info(f"Wave: WAVE_M={wave_m}, WAVE_N={wave_n}")
        logger.info(
            f"Waves per workgroup: {waves_along_m}x{waves_along_n} = {total_waves}"
        )

        # Create test tensors
        # A is M x K, B is N x K (for B^T in MMA), C is M x N
        a = device_randn((m, k), dtype=torch.float16)
        b = device_randn((n, k), dtype=torch.float16)

        # Compute PyTorch reference: C = A @ B^T
        expected = torch.matmul(a.float(), b.float().T)

        results = {}

        # Benchmark LLVM backend
        if not args.skip_llvm:
            logger.info("\n--- LLVM Backend ---")
            try:
                results["llvm"] = _run_backend(
                    backend="llvm",
                    args=args,
                    shape_config=shape_config,
                    a=a,
                    b=b,
                    expected=expected,
                )
            except Exception as e:
                logger.error(f"LLVM backend failed: {e}")
                results["llvm"] = None

        # Benchmark ASM backend
        if not args.skip_asm:
            logger.info("\n--- ASM Backend ---")
            try:
                results["asm"] = _run_backend(
                    backend="asm",
                    args=args,
                    shape_config=shape_config,
                    a=a,
                    b=b,
                    expected=expected,
                )
            except Exception as e:
                logger.error(f"ASM backend failed: {e}")
                results["asm"] = None

        # Summary comparison
        logger.info("\n" + "=" * 80)
        logger.info("Summary")
        logger.info("=" * 80)

        if results.get("llvm") and results.get("asm"):
            speedup = results["llvm"]["time_ms"] / results["asm"]["time_ms"]
            logger.info(
                f"LLVM: {results['llvm']['time_ms']:.4f} ms "
                f"({results['llvm']['tflops']:.4f} TFLOPs)"
            )
            logger.info(
                f"ASM:  {results['asm']['time_ms']:.4f} ms "
                f"({results['asm']['tflops']:.4f} TFLOPs)"
            )
            logger.info(f"ASM speedup over LLVM: {speedup:.2f}x")
        elif results.get("llvm"):
            logger.info(
                f"LLVM: {results['llvm']['time_ms']:.4f} ms "
                f"({results['llvm']['tflops']:.4f} TFLOPs)"
            )
        elif results.get("asm"):
            logger.info(
                f"ASM: {results['asm']['time_ms']:.4f} ms "
                f"({results['asm']['tflops']:.4f} TFLOPs)"
            )
        else:
            logger.error("Both backends failed!")

        logger.info("")


if __name__ == "__main__":
    main()
