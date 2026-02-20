"""
MXFP4 GEMM Quick Benchmark: Vanilla vs Preshuffle vs Split-K

Compares three MXFP4 GEMM kernels on one or more problem sizes:
  1. Vanilla (7.2): standard layout, f32 output
  2. Preshuffle (7.2): pre-shuffled scales, f32 output
  3. Split-K (ASM backend): bf16 atomic accumulation

Throughput is reported in TFLOPS using 2*M*N*K as the FLOP count.

Usage:
    # Single shape
    python mxfp4-quick-bench.py --m 1024 --n 1024 --k 8192

    # Sweep a CSV file of shapes (streams results as each shape completes)
    python mxfp4-quick-bench.py --shapes shapes-mxfp4-medium.csv

    # Combine: CSV file with overridden block/split settings
    python mxfp4-quick-bench.py --shapes shapes-mxfp4-medium.csv --splits 4

CSV format: header row "M,N,K" followed by one shape per line.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EXAMPLES_DIR = Path(__file__).parent
_WAVE_ROOT = _EXAMPLES_DIR.parent.parent
_E2E_DIR = (
    _WAVE_ROOT
    / "wave_lang"
    / "kernel"
    / "wave"
    / "asm"
    / "wave_asm"
    / "test"
    / "e2e"
)
for p in [str(_EXAMPLES_DIR), str(_WAVE_ROOT), str(_E2E_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import SHARED_ADDRESS_SPACE
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.templates.gemm import get_splitk_mxfp4_gemm_kernel
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

from waveasm_e2e import WaveASMCompiler, capture_wave_kernel_info
from test_asm_backend_e2e import get_target_arch

# Local 7.2 helpers (loaded once at import time)
from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location(
    "mxfp4_gemm_72", str(_EXAMPLES_DIR / "7.2_mxfp4_gemm_preshuffle_scale.py")
)
_mod72 = module_from_spec(_spec)
_spec.loader.exec_module(_mod72)

get_vanilla_kernel = _mod72.get_vanilla_kernel
get_preshuffle_kernel = _mod72.get_preshuffle_kernel
generate_mxfp4_inputs = _mod72.generate_mxfp4_inputs
e8m0_shuffle = _mod72.e8m0_shuffle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KERNELS = ["vanilla", "preshuffle", "splitk"]
_COL_W = {"shape": 22, "kernel": 12, "ms": 10, "tflops": 10, "ratio": 10}


def _compile_vanilla_or_preshuffle(kernel_fn, m, n, k, block_m, block_n, block_k):
    """Compile a vanilla or preshuffle kernel from 7.2."""
    k_scale_shuffled = (((k // 32) + 7) // 8) * 8
    hyperparams = {
        tkl.sym.ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        tkl.sym.BLOCK_M: block_m,
        tkl.sym.BLOCK_N: block_n,
        tkl.sym.BLOCK_K: block_k,
        tkl.sym.M: m,
        tkl.sym.N: n,
        tkl.sym.K: k,
        tkl.sym.K_SCALE_SHUFFLED: k_scale_shuffled,
    }
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        use_global_to_shared=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, kernel_fn)


def _bench(fn, warmup, iters):
    """Warm up then time fn over iters iterations; return average ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _tflops(m, n, k, avg_ms):
    return 2 * m * n * k / (avg_ms * 1e-3) / 1e12


def _load_shapes(path: str) -> list[tuple[int, int, int]]:
    shapes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shapes.append((int(row["M"]), int(row["N"]), int(row["K"])))
    return shapes


def _print_header():
    print(
        f"{'Shape (MxNxK)':<{_COL_W['shape']}}"
        f"{'Kernel':<{_COL_W['kernel']}}"
        f"{'ms':>{_COL_W['ms']}}"
        f"{'TFLOPS':>{_COL_W['tflops']}}"
        f"{'vs vanilla':>{_COL_W['ratio']}}"
    )
    print("-" * sum(_COL_W.values()))


def _print_row(shape_str, kernel, avg_ms, tfl, ratio_str):
    print(
        f"{shape_str:<{_COL_W['shape']}}"
        f"{kernel:<{_COL_W['kernel']}}"
        f"{avg_ms:>{_COL_W['ms']}.3f}"
        f"{tfl:>{_COL_W['tflops']}.2f}"
        f"{ratio_str:>{_COL_W['ratio']}}"
    )
    sys.stdout.flush()


def _print_failed_row(shape_str, kernel, reason="FAILED"):
    print(
        f"{shape_str:<{_COL_W['shape']}}"
        f"{kernel:<{_COL_W['kernel']}}"
        f"{'':>{_COL_W['ms']}}"
        f"{'':>{_COL_W['tflops']}}"
        f"{reason:>{_COL_W['ratio']}}"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Per-shape benchmark
# ---------------------------------------------------------------------------


def _check_splitk_correctness(
    m, n, k, block_m, block_n, block_k, num_splits, compiler,
    vanilla_compiled,
    x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, w_for_splitk,
):
    """Run one forward pass of vanilla and split-K and compare outputs.

    Returns (ok: bool, info: str) where info is a short human-readable summary.
    """
    # Vanilla reference (f32)
    c_ref = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    vanilla_compiled(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c_ref)
    torch.cuda.synchronize()

    splitk_fn, hyperparams = get_splitk_mxfp4_gemm_kernel(
        (m, n, k),
        num_splits=num_splits,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
        block_shape=(block_m, block_n, block_k),
    )
    # The C++ backend doesn't support vector.maskedload, which is emitted when
    # ADDRESS_SPACE is GLOBAL. Override to SHARED so global-to-shared loads are
    # used instead — matching what the e2e test does.
    hyperparams[tkl.sym.ADDRESS_SPACE] = SHARED_ADDRESS_SPACE
    options_sk = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=True,
    )
    options_sk = set_default_run_config(options_sk)
    kernel_info = capture_wave_kernel_info(options_sk, splitk_fn)

    cpp_result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
    if not cpp_result.success:
        return False, None, f"compile failed: {cpp_result.error_message}"

    import wave_runtime
    wave_runtime.load_hip_functions()
    _gpu_binary, gpu_func = wave_runtime.load_binary(
        str(cpp_result.binary_path),
        cpp_result.get_kernel_name() or kernel_info.kernel_name,
    )

    c_sk = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    kli = wave_runtime.KernelLaunchInfo(
        stream, gpu_func, kernel_info.lds_size,
        kernel_info.grid_size[0], kernel_info.grid_size[1], kernel_info.grid_size[2],
        kernel_info.workgroup_size[0], kernel_info.workgroup_size[1], kernel_info.workgroup_size[2],
        1, 1, 1,
    )
    kern_args = wave_runtime.Int64Vector(
        [t.data_ptr() for t in [x_gpu, x_scales_gpu, w_for_splitk, w_scales_gpu, c_sk]]
    )
    wave_runtime.launch(kli, kern_args, [], [])
    torch.cuda.synchronize()

    c_sk_f32 = c_sk.to(torch.float32)
    abs_diff = (c_sk_f32 - c_ref).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    # Both vanilla (f32) and split-K (bf16 atomic) are approximate vs truth.
    # The bf16 format has ULP of 0.25 at magnitude 4 and 0.5 at magnitude 8,
    # so small-magnitude outputs can have absolute errors up to ~2 ULP from
    # the truncation on each atomic add.  rtol covers large-magnitude elements;
    # atol=2.0 covers the small-magnitude tail.
    rtol, atol = 0.02, 2.0
    ok = bool(torch.allclose(c_sk_f32, c_ref, rtol=rtol, atol=atol))

    info = f"max_diff={max_diff:.4f} mean_diff={mean_diff:.4f}"
    return ok, (gpu_func, cpp_result, kernel_info), info


def bench_shape(m, n, k, block_m, block_n, block_k, num_splits, warmup, iters, compiler):
    """Benchmark all three kernels for one (M, N, K) shape.

    Streams one result row per kernel as each finishes.
    Returns a dict mapping kernel name -> avg_ms (or None on failure).
    """
    shape_str = f"{m}x{n}x{k}"
    results = {}

    # Shared inputs (generated once per shape)
    x, w, x_scales, w_scales = generate_mxfp4_inputs(
        (m, n, k), device=torch.device("cpu")
    )
    x_scales_sh = e8m0_shuffle(x_scales)
    w_scales_sh = e8m0_shuffle(w_scales)

    x_gpu = x.cuda()
    w_gpu = w.cuda()
    x_scales_gpu = x_scales.cuda()
    w_scales_gpu = w_scales.cuda()
    x_scales_sh_gpu = x_scales_sh.cuda()
    w_scales_sh_gpu = w_scales_sh.cuda()
    # generate_mxfp4_inputs returns w already in [N, K//2] layout, which is
    # what the splitk kernel expects for its b argument — no transpose needed.
    w_for_splitk = w_gpu

    # ------------------------------------------------------------------
    # 1. Vanilla
    # ------------------------------------------------------------------
    vanilla_compiled = None
    try:
        vanilla_compiled = _compile_vanilla_or_preshuffle(
            get_vanilla_kernel(), m, n, k, block_m, block_n, block_k
        )
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

        def run():
            vanilla_compiled(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c)

        avg_ms = _bench(run, warmup, iters)
        results["vanilla"] = avg_ms
        _print_row(shape_str, "vanilla", avg_ms, _tflops(m, n, k, avg_ms), "-")
    except Exception as e:
        results["vanilla"] = None
        _print_failed_row(shape_str, "vanilla", reason=str(e)[:_COL_W["ratio"]])

    # ------------------------------------------------------------------
    # 2. Preshuffle
    # ------------------------------------------------------------------
    try:
        compiled = _compile_vanilla_or_preshuffle(
            get_preshuffle_kernel(), m, n, k, block_m, block_n, block_k
        )
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

        def run():
            compiled(x_gpu, x_scales_sh_gpu, w_gpu, w_scales_sh_gpu, c)

        avg_ms = _bench(run, warmup, iters)
        results["preshuffle"] = avg_ms
        baseline = results.get("vanilla")
        ratio = f"{baseline/avg_ms:.2f}x" if baseline else "-"
        _print_row(shape_str, "preshuffle", avg_ms, _tflops(m, n, k, avg_ms), ratio)
    except Exception as e:
        results["preshuffle"] = None
        _print_failed_row(shape_str, "preshuffle", reason=str(e)[:_COL_W["ratio"]])

    # ------------------------------------------------------------------
    # 3. Split-K — correctness check first, then benchmark
    # ------------------------------------------------------------------
    try:
        if vanilla_compiled is None:
            raise RuntimeError("skipping: vanilla failed so no correctness reference")

        ok, compiled_sk, corr_info = _check_splitk_correctness(
            m, n, k, block_m, block_n, block_k, num_splits, compiler,
            vanilla_compiled,
            x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, w_for_splitk,
        )
        if not ok:
            raise RuntimeError(f"correctness FAIL ({corr_info})")

        # Unpack the already-compiled binary so we don't recompile for bench
        _gpu_func_ref, cpp_result, kernel_info = compiled_sk
        binary_path = cpp_result.binary_path
        kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name
        block = kernel_info.workgroup_size
        lds_size = kernel_info.lds_size
        grid = kernel_info.grid_size

        import wave_runtime
        wave_runtime.load_hip_functions()
        _gpu_binary, gpu_func = wave_runtime.load_binary(str(binary_path), kernel_name)

        c_sk = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")

        def run():
            c_sk.zero_()
            stream = torch.cuda.current_stream().cuda_stream
            kli = wave_runtime.KernelLaunchInfo(
                stream, gpu_func, lds_size,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                1, 1, 1,
            )
            kern_args = wave_runtime.Int64Vector(
                [t.data_ptr() for t in [x_gpu, x_scales_gpu, w_for_splitk, w_scales_gpu, c_sk]]
            )
            wave_runtime.launch(kli, kern_args, [], [])

        avg_ms = _bench(run, warmup, iters)
        results["splitk"] = avg_ms
        baseline = results.get("vanilla")
        ratio = f"{baseline/avg_ms:.2f}x" if baseline else "-"
        _print_row(shape_str, "splitk", avg_ms, _tflops(m, n, k, avg_ms), ratio)
        print(f"    ^ correctness ok  {corr_info}")
        sys.stdout.flush()
    except Exception as e:
        results["splitk"] = None
        _print_failed_row(shape_str, "splitk", reason=str(e))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MXFP4 GEMM quick benchmark")

    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument(
        "--shapes",
        metavar="CSV",
        help="CSV file with M,N,K columns; benchmarks every shape in the file",
    )
    shape_group.add_argument("--m", type=int, default=None, help="M dimension (single shape)")

    parser.add_argument("--n", type=int, default=None, help="N dimension (single shape)")
    parser.add_argument("--k", type=int, default=None, help="K dimension (single shape)")
    parser.add_argument("--block_m", type=int, default=128)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--block_k", type=int, default=128)
    parser.add_argument("--splits", type=int, default=2, help="num_splits for split-K")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.shapes:
        shapes = _load_shapes(args.shapes)
    else:
        m = args.m if args.m is not None else 1024
        n = args.n if args.n is not None else 1024
        k = args.k if args.k is not None else 8192
        shapes = [(m, n, k)]

    print(
        f"block=({args.block_m},{args.block_n},{args.block_k})  "
        f"splits={args.splits}  warmup={args.warmup}  iters={args.iters}"
    )
    print()
    _print_header()

    target = get_target_arch()
    print(f"target arch: {target}")
    compiler = WaveASMCompiler(target=target)

    all_results = []
    for i, (m, n, k) in enumerate(shapes):
        if i > 0:
            # Blank line between shapes for readability
            print()
        row = bench_shape(
            m, n, k,
            args.block_m, args.block_n, args.block_k,
            args.splits, args.warmup, args.iters,
            compiler,
        )
        all_results.append((m, n, k, row))

    # Final aggregate summary (if more than one shape)
    if len(shapes) > 1:
        print()
        print("=" * sum(_COL_W.values()))
        print(f"Summary: {len(shapes)} shapes")
        for kernel in KERNELS:
            vals = [r[kernel] for _, _, _, r in all_results if r.get(kernel) is not None]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {kernel:<12}  mean {avg:.3f} ms  ({len(vals)}/{len(shapes)} succeeded)")
        print("=" * sum(_COL_W.values()))


if __name__ == "__main__":
    main()
