# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Compile a Wave MXFP4 preshuffle-B kernel, validate it, then integrate and
benchmark through hipblaslt-bench (rocRoller path) with baseline comparison.

Usage:
  python -m wave_lang.kernel.wave.perf.benchmark_mxfp4_rocroller \
      --rocm-libraries /workspace/rocm-libraries \
      --block 128 256 256 \
      --shapes shapes.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

import torch
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_zeros
from wave_lang.kernel.wave.utils.general_utils import wave_dtype_to_torch
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
    b_preshuffle,
    e8m0_shuffle,
)

WAVE_KERNEL_SUBDIR = (
    "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/"
    "rocroller/custom_kernels/wave"
)
KERNEL_YAML_SUBDIR = (
    "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/"
    "rocroller/custom_kernels/kernels.yaml"
)
HIPBLASLT_BUILD_DIR = "projects/hipblaslt/build"

VALIDATION_SHAPES = [
    (1024, 1024, 8192),
    (4096, 4096, 4096),
    (2048, 8192, 4096),
]


def _read_shapes_csv(path: str) -> list[tuple[int, int, int]]:
    """Read (M, N, K) triples from a CSV file with columns M, N, K."""
    shapes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shapes.append((int(row["M"]), int(row["N"]), int(row["K"])))
    if not shapes:
        raise ValueError(f"No shapes found in {path}")
    return shapes


def _compile_wave_kernel(
    block_shape: tuple[int, int, int],
    dump_dir: str,
) -> tuple:
    """Compile a dynamic Wave preshuffle-B MXFP4 GEMM kernel.

    Returns (compiled_gemm, options) on success, raises on failure.
    """
    import wave_lang.kernel.wave.cache as _cache_mod

    _cache_mod.WAVE_CACHE_ON = 0

    shape = VALIDATION_SHAPES[0]
    output_dtype = tkl.bf16

    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block_shape,
        wave_shape=(1, 4),
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
        output_dtype=output_dtype,
    )

    for sym in [tkl.sym.M, tkl.sym.N, tkl.sym.K]:
        del options.subs[sym]
    options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    options.wave_runtime = True
    options.use_water_backend = False
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.dump_intermediates = dump_dir

    schedule = get_mxfp4_asymmetric_schedule()
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm, schedule)

    return compiled_gemm, options


def _validate_wave_kernel(compiled_gemm, shapes: list[tuple[int, int, int]]):
    """Run the compiled kernel against torch reference for each shape."""
    output_dtype = tkl.bf16
    for shape in shapes:
        M, N, K = shape
        print(f"  Validating shape ({M}, {N}, {K})...", end=" ", flush=True)
        x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
        torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

        w_t = w.T.contiguous()
        w_t_ps = b_preshuffle(w_t)
        x_scales_ps = e8m0_shuffle(x_scales)
        w_scales_ps = e8m0_shuffle(w_scales)

        out = device_zeros(
            x.shape[0], w_t_ps.shape[0], dtype=wave_dtype_to_torch(output_dtype)
        )
        compiled_gemm(x, x_scales_ps, w_t_ps, w_scales_ps, out)
        torch.testing.assert_close(torch_out, out, check_dtype=False)
        print("PASS")


def _find_rocmasm(dump_dir: str) -> str:
    """Find the .rocmasm file produced by dump_intermediates."""
    pattern = os.path.join(dump_dir, "**", "*.rocmasm")
    matches = glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No .rocmasm file found under {dump_dir}")
    return matches[0]


def _patch_and_install_asm(
    rocmasm_path: str,
    block_shape: tuple[int, int, int],
    wave_kernel_dir: str,
) -> str:
    """Replace 'gemm' entry name with a tile-specific name, write to wave dir."""
    BLOCK_M, BLOCK_N, BLOCK_K = block_shape
    new_name = f"wave_mxfp4_dynamic_gemm_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}"

    with open(rocmasm_path) as f:
        asm = f.read()

    asm = re.sub(r"\bgemm\b", new_name, asm)

    out_path = os.path.join(wave_kernel_dir, f"{new_name}.s")
    os.makedirs(wave_kernel_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(asm)
    return new_name


def _write_kernel_yaml(
    yaml_path: str,
    entry_function: str,
    block_shape: tuple[int, int, int],
    block_size: tuple[int, int, int],
):
    """Write kernels.yaml with a single wave kernel entry."""
    BLOCK_M, BLOCK_N, BLOCK_K = block_shape
    bx, by, bz = block_size
    yaml_content = f"""\
# Auto-generated by benchmark_mxfp4_rocroller.py
kernels:
  - name: {entry_function}
    workgroup_size: [{BLOCK_M}, {BLOCK_N}, {BLOCK_K}]
    block_size: [{bx}, {by}, {bz}]
    entry_function: {entry_function}
    shuffle: true
    kernel_type: mxfp4
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)


def _write_empty_kernel_yaml(yaml_path: str):
    """Write kernels.yaml with no kernel entries (rocRoller baseline)."""
    with open(yaml_path, "w") as f:
        f.write("kernels:\n")


def _build_hipblaslt(rocm_libraries: str):
    """Reconfigure and build hipblaslt-bench inside the build directory."""
    build_dir = os.path.join(rocm_libraries, HIPBLASLT_BUILD_DIR)
    cmake_file = os.path.join(
        rocm_libraries, WAVE_KERNEL_SUBDIR, "..", "CMakeLists.txt"
    )
    if os.path.isfile(cmake_file):
        Path(cmake_file).touch()
    result = subprocess.run(
        ["ninja", "hipblaslt-bench"],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ninja hipblaslt-bench failed:\n{result.stdout}\n{result.stderr}"
        )


def _run_hipblaslt_bench(
    rocm_libraries: str,
    M: int,
    N: int,
    K: int,
    cold_iters: int = 1,
    bench_iters: int = 1,
    swizzle_a: bool = True,
) -> dict:
    """Run hipblaslt-bench and parse the CSV output."""
    build_dir = os.path.join(rocm_libraries, HIPBLASLT_BUILD_DIR)
    bench_bin = os.path.join(build_dir, "clients", "hipblaslt-bench")

    lib_dir = os.path.join(build_dir, "library")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        bench_bin,
        "--api_method",
        "c",
        "-m",
        str(M),
        "-n",
        str(N),
        "-k",
        str(K),
        "--alpha",
        "1",
        "--beta",
        "0",
        "--transA",
        "T",
        "--transB",
        "N",
        "--batch_count",
        "1",
        "--scaleA",
        "1001",
        "--scaleB",
        "1001",
        "--a_type",
        "f4_r",
        "--b_type",
        "f4_r",
        "--c_type",
        "bf16_r",
        "--d_type",
        "bf16_r",
        "--compute_type",
        "f32_r",
        "--rotating",
        "0",
        "--cold_iters",
        str(cold_iters),
        "--iters",
        str(bench_iters),
        "--verify",
    ]
    if swizzle_a:
        cmd.append("--swizzleA")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=build_dir, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"hipblaslt-bench failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )

    header_line = None
    data_line = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("[0]:"):
            header_line = stripped[len("[0]:") :]
        elif header_line and stripped and not stripped.startswith("["):
            data_line = stripped
            break

    if not header_line or not data_line:
        raise RuntimeError(f"Could not parse hipblaslt-bench output:\n{result.stdout}")

    headers = [h.strip() for h in header_line.split(",")]
    values = [v.strip() for v in data_line.split(",")]
    return dict(zip(headers, values))


def _parse_bench_result(bench_result: dict) -> tuple[bool, float, float]:
    """Extract (correct, tflops, runtime_us) from a hipblaslt-bench result dict."""
    atol = bench_result.get("atol", "")
    rtol = bench_result.get("rtol", "")
    correct = atol != "failed" and rtol != "failed"
    runtime_us = float(bench_result.get("us", "0"))
    tflops = float(bench_result.get("hipblaslt-Gflops", "0")) / 1000.0
    return correct, tflops, runtime_us


def _bench_all_shapes(
    label: str,
    rocm_libraries: str,
    shapes: list[tuple[int, int, int]],
    cold_iters: int,
    bench_iters: int,
    swizzle_a: bool = True,
) -> list[dict]:
    """Benchmark all shapes, returning a list of {correct, tflops, runtime_us} dicts."""
    results = []
    for i, (M, N, K) in enumerate(shapes, 1):
        tag = f"[{i}/{len(shapes)}]"
        print(f"  {tag} {M}x{N}x{K}...", end=" ", flush=True)
        entry: dict = {"correct": False, "tflops": 0.0, "runtime_us": 0.0}
        try:
            bench_result = _run_hipblaslt_bench(
                rocm_libraries,
                M,
                N,
                K,
                cold_iters=cold_iters,
                bench_iters=bench_iters,
                swizzle_a=swizzle_a,
            )
            correct, tflops, runtime_us = _parse_bench_result(bench_result)
            entry["correct"] = correct
            entry["tflops"] = tflops
            entry["runtime_us"] = runtime_us
            status = "CORRECT" if correct else "VERIFY FAILED"
            print(f"{tflops:.2f} TFLOPS, {runtime_us:.1f} us, {status}")
        except Exception as e:
            print(f"ERROR: {e}")
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compile Wave MXFP4 kernel and benchmark via hipblaslt-bench"
    )
    parser.add_argument(
        "--rocm-libraries",
        required=True,
        help="Path to rocm-libraries root directory",
    )
    parser.add_argument(
        "--block",
        type=int,
        nargs=3,
        required=True,
        metavar=("BLOCK_M", "BLOCK_N", "BLOCK_K"),
        help="Macrotile dimensions",
    )
    parser.add_argument(
        "--shapes",
        required=True,
        help="Path to CSV file with M,N,K columns",
    )
    parser.add_argument(
        "--cold-iters",
        type=int,
        default=1,
        help="Cold iterations for hipblaslt-bench (default: 1)",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=1,
        help="Benchmark iterations for hipblaslt-bench (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: wave_mxfp4_dynamic_gemm_{BM}x{BN}x{BK}_rocroller_results.csv)",
    )
    args = parser.parse_args()

    rocm_libraries = os.path.abspath(args.rocm_libraries)
    block_shape = tuple(args.block)
    BLOCK_M, BLOCK_N, BLOCK_K = block_shape

    output_csv = args.output or (
        f"wave_mxfp4_dynamic_gemm_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_rocroller_results.csv"
    )

    wave_kernel_dir = os.path.join(rocm_libraries, WAVE_KERNEL_SUBDIR)
    kernel_yaml_path = os.path.join(rocm_libraries, KERNEL_YAML_SUBDIR)

    if not os.path.isdir(wave_kernel_dir):
        print(f"ERROR: Wave kernel dir not found: {wave_kernel_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(kernel_yaml_path):
        print(f"ERROR: kernels.yaml not found: {kernel_yaml_path}", file=sys.stderr)
        sys.exit(1)

    bench_shapes = _read_shapes_csv(args.shapes)
    print(f"Loaded {len(bench_shapes)} shapes from {args.shapes}")

    dump_dir = tempfile.mkdtemp(prefix="wave_rocroller_bench_")
    print(f"Intermediate dump directory: {dump_dir}")

    # --- Step 1: Compile wave kernel ---
    print(f"\n[1/7] Compiling Wave kernel (block={BLOCK_M}x{BLOCK_N}x{BLOCK_K})...")
    try:
        compiled_gemm, options = _compile_wave_kernel(block_shape, dump_dir)
    except Exception as e:
        print(f"ERROR: Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("  Compilation succeeded.")

    # --- Step 2: Validate wave kernel ---
    print(f"\n[2/7] Validating against torch reference...")
    try:
        _validate_wave_kernel(compiled_gemm, VALIDATION_SHAPES)
    except Exception as e:
        print(f"ERROR: Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("  All validations passed.")

    # --- Step 3: Patch assembly and stage for later ---
    print(f"\n[3/7] Patching assembly...")
    try:
        rocmasm_path = _find_rocmasm(dump_dir)
        entry_function = _patch_and_install_asm(
            rocmasm_path, block_shape, wave_kernel_dir
        )
        block_size = options.kernel_launch_info.blocks
        print(f"  Prepared: {entry_function}.s")
    except Exception as e:
        print(f"ERROR: Assembly patching failed: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 4: Build rocRoller baseline (empty kernels.yaml) ---
    print(f"\n[4/7] Building rocRoller baseline (no custom kernels)...")
    try:
        _write_empty_kernel_yaml(kernel_yaml_path)
        _build_hipblaslt(rocm_libraries)
    except Exception as e:
        print(f"ERROR: Baseline build failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("  Build succeeded.")

    # --- Step 5: Benchmark rocRoller baseline (without --swizzleA) ---
    print(f"\n[5/7] Benchmarking rocRoller baseline ({len(bench_shapes)} shapes)...")
    rr_results = _bench_all_shapes(
        "ROCROLLER",
        rocm_libraries,
        bench_shapes,
        cold_iters=args.cold_iters,
        bench_iters=args.bench_iters,
        swizzle_a=False,
    )

    # --- Step 6: Build with wave kernel ---
    print(f"\n[6/7] Building with Wave kernel ({entry_function})...")
    try:
        _write_kernel_yaml(kernel_yaml_path, entry_function, block_shape, block_size)
        _build_hipblaslt(rocm_libraries)
    except Exception as e:
        print(f"ERROR: Wave build failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("  Build succeeded.")

    # --- Step 7: Benchmark wave kernel ---
    print(f"\n[7/7] Benchmarking Wave kernel ({len(bench_shapes)} shapes)...")
    wave_results = _bench_all_shapes(
        "WAVE",
        rocm_libraries,
        bench_shapes,
        cold_iters=args.cold_iters,
        bench_iters=args.bench_iters,
    )

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  RESULTS: block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")
    print(f"{'='*70}")

    for i, (M, N, K) in enumerate(bench_shapes):
        rr = rr_results[i]
        wv = wave_results[i]

        if rr["runtime_us"] > 0 and wv["runtime_us"] > 0:
            speedup = (rr["runtime_us"] - wv["runtime_us"]) / rr["runtime_us"] * 100
            delta_str = f"{speedup:+.0f}%"
        else:
            delta_str = "N/A"

        rr_status = "CORRECT" if rr["correct"] else "WRONG"
        wv_status = "CORRECT" if wv["correct"] else "WRONG"

        print(f"[{i+1}/{len(bench_shapes)}] {M}x{N}x{K}     {delta_str}")
        print(
            f"         [ROCROLLER] {rr['tflops']:.2f} TFLOPs, {rr['runtime_us']:.1f} us, {rr_status}"
        )
        print(
            f"         [WAVE]      {wv['tflops']:.2f} TFLOPs, {wv['runtime_us']:.1f} us, {wv_status}"
        )

    print(f"{'='*70}")

    # --- Write output CSV ---
    fieldnames = [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "M",
        "N",
        "K",
        "wave_tflops",
        "wave_us",
        "wave_correct",
        "rocroller_tflops",
        "rocroller_us",
        "rocroller_correct",
    ]
    rows = []
    for i, (M, N, K) in enumerate(bench_shapes):
        rr = rr_results[i]
        wv = wave_results[i]
        rows.append(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K,
                "M": M,
                "N": N,
                "K": K,
                "wave_tflops": wv["tflops"],
                "wave_us": wv["runtime_us"],
                "wave_correct": wv["correct"],
                "rocroller_tflops": rr["tflops"],
                "rocroller_us": rr["runtime_us"],
                "rocroller_correct": rr["correct"],
            }
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    main()
