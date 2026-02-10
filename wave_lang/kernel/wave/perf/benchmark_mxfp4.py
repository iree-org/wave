# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Standalone reproducible benchmark for MXFP4 Wave GEMM.

Uses torch benchmarking (compile with wave_runtime, warmup + benchmark loop with
torch.cuda.synchronize). When run without --_worker, the script invokes itself
with --_worker wrapped in rocprofv3 so profiler output is still collected.

Requires: torch, wave_lang, aiter (cf29be372d2ecd20102cc22b74a64d75f0c99512)

Single shape:
  python benchmark_mxfp4.py --shape <M> <N> <K> [--tiles <block_m> <block_n> <block_k>]

Multiple shapes (CSV with M,N,K):
  python benchmark_mxfp4.py --shapes <path/to/csv> -o <output_csv> [--tiles ...]

Optional env: ATT_LIBRARY_PATH=/path/to/rocprof-trace-decoder/rocm/lib. If set, passed to rocprofv3 (--att --att-library-path).
Default dump dir: /tmp/bench_mxfp4_dump
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import aiter
import torch
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.wave import ScaledMMAType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

# ---------------------------------------------------------------------------
# Wave kernel template and compile options (edit here to change kernel/options)
# ---------------------------------------------------------------------------


def get_mxfp4_gemm_wave(
    shape: tuple[int, int, int],
    block_m: int,
    block_n: int,
    block_k: int,
    use_wave_runtime: bool = False,
    vmfb_path: Optional[Path] = None,
):
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4
    c_wave_dtype = tkl.bf16
    # Input sizes and tile symbols (same as wave_gemm.get_mxfp4_gemm)
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm_afp4_wfp4_wave(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, c_wave_dtype],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        casted = tkw.cast(repeat, c_wave_dtype)
        tkw.write(casted, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        canonicalize=True,
        schedule=SchedulingType.NONE,
        use_buffer_ops=False,
        waves_per_eu=1,
        use_global_to_shared=False,
        minimize_shared_allocs=False,
        subs=hyperparams,
        dynamic_symbols=[],
        device="hip",
        target=_get_hip_target_from_device(),
        iree_launch_async=False,
        run_bench=False,
        wave_runtime=use_wave_runtime,
    )
    if use_wave_runtime:
        options = set_default_run_config(options)
    if vmfb_path is not None:
        options.create_vmfb_file = str(vmfb_path)

    return wave_compile(options, gemm_afp4_wfp4_wave)


# ---------------------------------------------------------------------------
# Helpers (mxfp4 conversion, inputs, IREE/rocprof, validate, benchmark)
# ---------------------------------------------------------------------------


def _get_hip_target_from_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm not available.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No GPU detected.")
    arch_name = torch.cuda.get_device_properties(0).gcnArchName
    return arch_name.split(":")[0]


def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
    return mxfp4_in_f32[x.long()]


def _e8m0_to_f32(scale_e8m0_biased: torch.Tensor) -> torch.Tensor:
    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 0xFF
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    scale_f32[zero_case] = 0x00400000
    scale_f32[nan_case] = 0x7F800001
    return scale_f32.view(torch.float32)


def get_torch_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    m, k = x.shape
    n, k = w.shape
    x_f32 = _mxfp4_to_f32(x)
    w_f32 = _mxfp4_to_f32(w)
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(32, dim=1)
    x_scales_f32 = _e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(32, dim=1)
    w_scales_f32 = _e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


def get_mxfp4_inputs(M: int, N: int, K: int, slice_scales: bool = True):
    c_dtype = torch.bfloat16
    x = torch.randn((M, K), device="cuda", dtype=c_dtype)
    w = torch.randn((N, K), device="cuda", dtype=c_dtype)
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    _, x_scale = quant_func(x, shuffle=False)
    _, w_scale = quant_func(w, shuffle=False)
    x, _ = quant_func(x, shuffle=True)
    w, _ = quant_func(w, shuffle=True)
    # w = shuffle_weight(w, layout=(16, 16))
    out = torch.empty(M, N, device=x.device, dtype=c_dtype)
    if slice_scales:
        x_scale = x_scale[:M, : K // 32]
        w_scale = w_scale[:N, : K // 32]
    return x, w, x_scale, w_scale, out


def get_flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def get_byte_count_mxfp4(M: int, N: int, K: int) -> int:
    # Packed mxfp4: K/2 bytes per row for A (M rows), B (N rows)
    # Scales: K/32 per row for A and B. Output: M*N*2 (bf16).
    return M * (K // 2) + M * (K // 32) + N * (K // 2) + N * (K // 32) + M * N * 2


def runtime_us_to_tflops(M: int, N: int, K: int, runtime_us: float) -> float:
    if runtime_us <= 0:
        return 0.0
    flops = get_flops(M, N, K)
    return (flops / 1e12) / (runtime_us / 1e6)


# --- rocprof / worker helpers -------------------------------------------------


def _clear_dir(dir_path: os.PathLike) -> None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def _get_rocprofv3_cmd(
    dump_path: os.PathLike,
    att_library_path: Optional[str],
    kernel_regex: str = "gemm",
) -> list:
    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "--kernel-include-regex",
        kernel_regex,
        "--stats",
        "TRUE",
        "-d",
        f"{dump_path}",
        "--output-format",
        "csv",
        "--",
    ]
    if att_library_path:
        cmd = cmd[:4] + ["--att", "--att-library-path", att_library_path] + cmd[4:]
    return cmd


def _parse_rocprof_us(path: Path, kernel_regex: str = "gemm") -> dict:
    try:
        if path.is_dir():
            kernel_stats_files = list(path.glob("**/*kernel_stats.csv"))
            if not kernel_stats_files:
                return {}
            csv_path = kernel_stats_files[0]
        else:
            csv_path = path
        if not csv_path.exists():
            return {}
        with open(csv_path, "r") as f:
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
    except Exception:
        return {}


def _parse_worker_stdout_for_mean_us(stdout: str) -> tuple[float, bool]:
    """Parse worker stdout for a line 'MEAN_US: <float>'; return (mean_us, ok)."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("MEAN_US:"):
            try:
                value = float(line.split(":", 1)[1].strip())
                return value, True
            except (ValueError, IndexError):
                pass
    return 0.0, False


# Torch benchmark constants (worker mode)
WARMUP_ITERS = 10
BENCHMARK_ITERS = 100


def _run_torch_benchmark(
    kernel_func,
    inputs: tuple,
    warmup_iters: int = WARMUP_ITERS,
    benchmark_iters: int = BENCHMARK_ITERS,
) -> float:
    """Run warmup and benchmark loop with torch.cuda.synchronize; return mean runtime in microseconds."""
    for _ in range(warmup_iters):
        kernel_func(*inputs)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(benchmark_iters):
        kernel_func(*inputs)
    end_ev.record()
    torch.cuda.synchronize()
    mean_ms = start_ev.elapsed_time(end_ev) / benchmark_iters
    return mean_ms * 1000.0  # us


def run_worker(shape: tuple[int, int, int], tiles: tuple[int, int, int]) -> None:
    """Worker entry: compile GEMM with wave_runtime, run torch benchmark, print MEAN_US to stdout."""
    m, n, k = shape
    block_m, block_n, block_k = tiles
    gemm_rt = get_mxfp4_gemm_wave(
        (m, n, k), block_m, block_n, block_k, use_wave_runtime=True
    )
    x, w, x_scale, w_scale, wave_out = get_mxfp4_inputs(m, n, k)
    x_scale_u8 = x_scale.view(torch.uint8)
    w_scale_u8 = w_scale.view(torch.uint8)
    inputs = (x, x_scale_u8, w, w_scale_u8, wave_out)
    mean_us = _run_torch_benchmark(gemm_rt, inputs)
    print(f"MEAN_US: {mean_us}")


def validate_mxfp4_gemm(m: int, n: int, k: int, compiled_gemm) -> bool:
    """Run compiled Wave GEMM (with wave_runtime=True) and compare to torch reference."""
    try:
        x, w, x_scale, w_scale, wave_out = get_mxfp4_inputs(m, n, k)
        x_scale_u8 = x_scale.view(torch.uint8)
        w_scale_u8 = w_scale.view(torch.uint8)
        compiled_gemm(x, x_scale_u8, w, w_scale_u8, wave_out)
        torch_ref = get_torch_reference(x, w, x_scale_u8, w_scale_u8, torch.bfloat16)
        torch.testing.assert_close(wave_out, torch_ref, check_device=False)
        return True
    except Exception:
        return False


def benchmark_mxfp4_gemm_rocprof(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    profiler_dump_path: Path,
    att_library_path: Optional[str],
    kernel_regex: str = "gemm",
    timeout: Optional[float] = None,
) -> tuple[float, bool]:
    """Run self as worker under rocprofv3 (torch benchmark); return mean runtime in microseconds."""
    _clear_dir(profiler_dump_path)
    profile_prefix = _get_rocprofv3_cmd(
        profiler_dump_path, att_library_path, kernel_regex
    )
    worker_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--shape",
        str(m),
        str(n),
        str(k),
        "--tiles",
        str(block_m),
        str(block_n),
        str(block_k),
    ]
    full_cmd = profile_prefix + worker_cmd
    try:
        proc = subprocess.run(
            full_cmd,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return 0.0, False
    if proc.returncode != 0:
        return 0.0, False
    runtime_us, ok = _parse_worker_stdout_for_mean_us(proc.stdout)
    if ok:
        try:
            rocprof_stats = _parse_rocprof_us(profiler_dump_path, kernel_regex)
            if rocprof_stats and "mean_duration_us" in rocprof_stats:
                runtime_us = rocprof_stats["mean_duration_us"]
        except Exception:
            pass
    return runtime_us, ok


def run_validate_and_benchmark(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    dump_dir: Path,
    att_library_path: Optional[str],
    kernel_regex: str = "gemm",
) -> tuple[Optional[float], Optional[float], str]:
    """
    Compile with wave_runtime and validate; then benchmark via torch worker under rocprofv3.
    Returns (runtime_us, tflops, status) where status is "ok", "compile_validation", "validation_failed",
    "compile_benchmark", or "benchmark_failed".
    """
    shape = (m, n, k)

    # Compile for validation (wave_runtime=True)
    try:
        gemm_rt = get_mxfp4_gemm_wave(
            shape, block_m, block_n, block_k, use_wave_runtime=True
        )
    except Exception as e:
        print(
            f"Compilation (wave_runtime) failed for ({m},{n},{k}): {e}",
            file=sys.stderr,
        )
        return None, None, "compile_validation"

    # Save MLIR to dump directory
    mlir_dir = dump_dir / "mlir"
    mlir_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = mlir_dir / f"gemm_{m}_{n}_{k}.mlir"
    mlir_path.write_text(gemm_rt.asm)

    # Validate numerics
    if not validate_mxfp4_gemm(m, n, k, gemm_rt):
        print(f"Validation failed for ({m},{n},{k})", file=sys.stderr)
        return None, None, "validation_failed"

    # Benchmark via torch worker under rocprofv3
    profiler_dump_path = dump_dir / "rocprof" / f"gemm_{m}_{n}_{k}"
    runtime_us, ok = benchmark_mxfp4_gemm_rocprof(
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        profiler_dump_path,
        att_library_path,
        kernel_regex=kernel_regex,
    )
    if not ok:
        print(f"Benchmark failed for ({m},{n},{k})", file=sys.stderr)
        return None, None, "benchmark_failed"

    tflops = runtime_us_to_tflops(m, n, k, runtime_us)
    return runtime_us, tflops, "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone MXFP4 Wave GEMM benchmark (no kernel_bench)."
    )
    p.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="Single shape: M N K",
    )
    g.add_argument(
        "--shapes",
        type=Path,
        metavar="CSV",
        help="CSV path with header M,N,K (one shape per row)",
    )
    p.add_argument(
        "--tiles",
        type=int,
        nargs=3,
        default=[256, 256, 256],
        metavar=("block_m", "block_n", "block_k"),
        help="Tile sizes (default: 256 256 256)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV for --shapes mode (required when using --shapes)",
    )
    p.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("/tmp/bench_mxfp4_dump"),
        help="Directory for rocprof output (default: /tmp/bench_mxfp4_dump)",
    )
    p.add_argument(
        "--kernel-regex",
        type=str,
        default="gemm",
        help="Regex for rocprof kernel filter (default: gemm)",
    )
    return p.parse_args()


def load_shapes_csv(path: Path) -> list[tuple[int, int, int]]:
    shapes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "M" not in reader.fieldnames:
            # Allow headerless M,N,K
            f.seek(0)
            for line in csv.reader(f):
                if len(line) >= 3:
                    try:
                        shapes.append((int(line[0]), int(line[1]), int(line[2])))
                    except ValueError:
                        if line[0].upper() != "M":
                            raise
            return shapes
        for row in reader:
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            shapes.append((M, N, K))
    return shapes


def main():
    att_library_path = os.environ.get("ATT_LIBRARY_PATH") or None

    args = parse_args()

    if args._worker:
        if args.shape is None:
            print("--_worker requires --shape M N K.", file=sys.stderr)
            sys.exit(1)
        m, n, k = args.shape
        block_m, block_n, block_k = args.tiles
        run_worker((m, n, k), (block_m, block_n, block_k))
        return

    if args.shape is None and args.shapes is None:
        print("One of --shape or --shapes is required.", file=sys.stderr)
        sys.exit(1)

    block_m, block_n, block_k = args.tiles
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    kernel_regex = args.kernel_regex

    if args.shape is not None:
        M, N, K = args.shape
        if M < 4 or N < 4 or K < 4 or K % 2 != 0:
            print(
                f"Invalid shape ({M},{N},{K}): M,N,K>=4 and K even required.",
                file=sys.stderr,
            )
            sys.exit(1)
        runtime_us, tflops, status = run_validate_and_benchmark(
            M,
            N,
            K,
            block_m,
            block_n,
            block_k,
            dump_dir,
            att_library_path,
            kernel_regex=kernel_regex,
        )
        if status != "ok":
            sys.exit(1)
        print(f"Shape: {M} x {N} x {K}")
        print(f"Average runtime: {runtime_us:.2f} us")
        print(f"TFLOPs: {tflops:.4f}")
        return

    # --shapes mode
    if not args.shapes.exists():
        print(f"Shapes file not found: {args.shapes}", file=sys.stderr)
        sys.exit(1)
    if args.output is None:
        print("--shapes requires -o/--output for the result CSV.", file=sys.stderr)
        sys.exit(1)

    shapes = load_shapes_csv(args.shapes)
    if not shapes:
        print("No shapes found in CSV.", file=sys.stderr)
        sys.exit(1)

    results = []
    failed_validation = []
    failed_benchmark = []
    for M, N, K in shapes:
        runtime_us, tflops, status = run_validate_and_benchmark(
            M,
            N,
            K,
            block_m,
            block_n,
            block_k,
            dump_dir,
            att_library_path,
            kernel_regex=kernel_regex,
        )
        ok = status == "ok"
        if status in ("compile_validation", "validation_failed"):
            failed_validation.append((M, N, K))
        elif status in ("compile_benchmark", "benchmark_failed"):
            failed_benchmark.append((M, N, K))
        mean_us = runtime_us if runtime_us is not None else 0.0
        tflops_val = tflops if tflops is not None else 0.0
        results.append(
            {
                "M": M,
                "N": N,
                "K": K,
                "runtime_us": mean_us,
                "tflops": tflops_val,
                "ok": ok,
            }
        )
        status_str = "ok" if ok else status
        print(
            f"  ({M}, {N}, {K}): {mean_us:.2f} us, {tflops_val:.4f} TFLOPs [{status_str}]"
        )

    if failed_validation:
        print(f"Kernels that failed validation: {failed_validation}", file=sys.stderr)
    if failed_benchmark:
        print(f"Kernels that failed benchmarking: {failed_benchmark}", file=sys.stderr)

    valid = [r for r in results if r["ok"] and r["runtime_us"] > 0]
    if not valid:
        print("No successful runs.", file=sys.stderr)
        sys.exit(1)

    avg_us = sum(r["runtime_us"] for r in valid) / len(valid)
    avg_tflops = sum(r["tflops"] for r in valid) / len(valid)
    best = max(valid, key=lambda r: r["tflops"])

    print()
    print(
        f"Average across {len(valid)} kernels: {avg_us:.2f} us, {avg_tflops:.4f} TFLOPs"
    )
    print(
        f"Best kernel: M={best['M']}, N={best['N']}, K={best['K']} -> "
        f"{best['runtime_us']:.2f} us, {best['tflops']:.4f} TFLOPs"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "N", "K", "runtime_us", "tflops"])
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "M": r["M"],
                    "N": r["N"],
                    "K": r["K"],
                    "runtime_us": r["runtime_us"],
                    "tflops": r["tflops"],
                }
            )
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
