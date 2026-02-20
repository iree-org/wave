"""
MXFP4 GEMM Quick Benchmark

Compares MXFP4 GEMM kernels on one or more problem sizes:
  1. vanilla              (7.2): unshuffled scales, no schedule
  2. preshuffle           (7.2): shuffled scales, IndexMapping, no schedule
  3. splitk              (7.4): split-K via wave_asm, unshuffled scales
  4. scheduled           (7.4): double-buffer schedule, unshuffled scales
  5. preshuffle-sched    (7.4): double-buffer schedule + shuffled scales
  6. preshuffle-B-sched  (7.4): asymmetric schedule + preshuffled B, unshuffled scales
  7. preshuffle-all-sched (7.4): asymmetric schedule + preshuffled B + shuffled scales

Throughput is reported in TFLOPS using 2*M*N*K as the FLOP count.

Usage:
    python mxfp4-quick-bench.py --m 1024 --n 1024 --k 8192
    python mxfp4-quick-bench.py --shapes shapes-mxfp4-medium.csv
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
    _WAVE_ROOT / "wave_lang" / "kernel" / "wave" / "asm" / "wave_asm" / "test" / "e2e"
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

# ---------------------------------------------------------------------------
# Load helpers from 7.2 and 7.4
# ---------------------------------------------------------------------------
from importlib.util import spec_from_file_location, module_from_spec


def _load_module(name, path):
    spec = spec_from_file_location(name, str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod72 = _load_module("mxfp4_gemm_72", _EXAMPLES_DIR / "7.2_mxfp4_gemm_preshuffle_scale.py")
_mod74 = _load_module("mxfp4_gemm_74", _EXAMPLES_DIR / "7.4_mxfp4_gemm_preshuffle_scale_splitk.py")

# From 7.2
get_vanilla_kernel = _mod72.get_vanilla_kernel
get_preshuffle_kernel = _mod72.get_preshuffle_kernel
generate_mxfp4_inputs = _mod72.generate_mxfp4_inputs
e8m0_shuffle = _mod72.e8m0_shuffle

# From 7.4
get_kernel = _mod74.get_kernel
get_splitk_kernel = _mod74.get_splitk_kernel
run_splitk_kernel = _mod74.run_splitk_kernel
preshuffle_b_aiter = _mod74.preshuffle_b_aiter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KERNELS = [
    "vanilla",
    "preshuffle",
    "splitk",
    "scheduled",
    "preshuffle-sched",
    "preshuffle-B-sched",
    "preshuffle-all-sched",
]
_COL_W = {"shape": 22, "kernel": 20, "ms": 10, "tflops": 10, "ratio": 10}


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
        f"{reason[:_COL_W['ratio']]:>{_COL_W['ratio']}}"
    )
    sys.stdout.flush()


def _ratio(baseline_ms, avg_ms):
    return f"{baseline_ms/avg_ms:.2f}x" if baseline_ms else "-"


# ---------------------------------------------------------------------------
# Per-shape benchmark
# ---------------------------------------------------------------------------

def bench_shape(m, n, k, block_m, block_n, block_k, num_splits, warmup, iters, compiler):
    shape_str = f"{m}x{n}x{k}"
    results = {}

    # Prepare all input variants once
    x, w, x_scales, w_scales = generate_mxfp4_inputs(
        (m, n, k), device=torch.device("cpu")
    )
    # w from generate_mxfp4_inputs is [N, K/2]
    x_scales_sh = e8m0_shuffle(x_scales)
    w_scales_sh = e8m0_shuffle(w_scales)
    w_ps = preshuffle_b_aiter(w)  # preshuffled B, still [N, K/2]

    x_gpu = x.cuda()
    w_gpu = w.cuda()
    w_ps_gpu = w_ps.cuda()
    x_scales_gpu = x_scales.cuda()
    w_scales_gpu = w_scales.cuda()
    x_scales_sh_gpu = x_scales_sh.cuda()
    w_scales_sh_gpu = w_scales_sh.cuda()

    vanilla_ms = None  # used as ratio baseline

    # ------------------------------------------------------------------
    # 1. Vanilla (7.2) — unshuffled scales, no schedule
    # ------------------------------------------------------------------
    vanilla_compiled = None
    try:
        vanilla_compiled = _compile_vanilla_or_preshuffle(
            get_vanilla_kernel(), m, n, k, block_m, block_n, block_k
        )
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
        avg_ms = _bench(lambda: vanilla_compiled(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c), warmup, iters)
        results["vanilla"] = avg_ms
        vanilla_ms = avg_ms
        _print_row(shape_str, "vanilla", avg_ms, _tflops(m, n, k, avg_ms), "-")
    except Exception as e:
        results["vanilla"] = None
        _print_failed_row(shape_str, "vanilla", reason=str(e))

    # ------------------------------------------------------------------
    # 2. Preshuffle (7.2) — shuffled scales, IndexMapping, no schedule
    # ------------------------------------------------------------------
    try:
        compiled_ps = _compile_vanilla_or_preshuffle(
            get_preshuffle_kernel(), m, n, k, block_m, block_n, block_k
        )
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
        avg_ms = _bench(lambda: compiled_ps(x_gpu, x_scales_sh_gpu, w_gpu, w_scales_sh_gpu, c), warmup, iters)
        results["preshuffle"] = avg_ms
        _print_row(shape_str, "preshuffle", avg_ms, _tflops(m, n, k, avg_ms), _ratio(vanilla_ms, avg_ms))
    except Exception as e:
        results["preshuffle"] = None
        _print_failed_row(shape_str, "preshuffle", reason=str(e))

    # ------------------------------------------------------------------
    # 3. Split-K (7.4) — wave_asm backend, unshuffled scales, bf16 out
    # ------------------------------------------------------------------
    try:
        if vanilla_compiled is None:
            raise RuntimeError("skipping: no vanilla reference for correctness check")

        handle = get_splitk_kernel(
            (m, n, k), block=(block_m, block_n, block_k),
            num_splits=num_splits, compiler=compiler,
        )

        # Correctness check vs vanilla
        c_ref = torch.zeros(m, n, dtype=torch.float32, device="cuda")
        vanilla_compiled(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c_ref)
        torch.cuda.synchronize()

        c_sk = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
        run_splitk_kernel(handle, x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c_sk)
        torch.cuda.synchronize()

        bf16_eps = 2**-7
        atol = num_splits * bf16_eps * max(c_ref.abs().max().item(), 1.0)
        if not torch.allclose(c_sk.float(), c_ref, rtol=0.0, atol=atol):
            diff = (c_sk.float() - c_ref).abs()
            raise RuntimeError(f"correctness FAIL max_diff={diff.max():.4f}")

        c_sk_bench = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
        def run_sk():
            c_sk_bench.zero_()
            run_splitk_kernel(handle, x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c_sk_bench)

        avg_ms = _bench(run_sk, warmup, iters)
        results["splitk"] = avg_ms
        _print_row(shape_str, "splitk", avg_ms, _tflops(m, n, k, avg_ms), _ratio(vanilla_ms, avg_ms))
    except Exception as e:
        results["splitk"] = None
        _print_failed_row(shape_str, "splitk", reason=str(e))

    # ------------------------------------------------------------------
    # 4. Scheduled (7.4) — double-buffer schedule, unshuffled scales
    # ------------------------------------------------------------------
    try:
        fn = get_kernel((m, n, k), (block_m, block_n, block_k),
                        preshuffle_scales=False, preshuffle_B=False)
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
        avg_ms = _bench(lambda: fn(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c), warmup, iters)
        results["scheduled"] = avg_ms
        _print_row(shape_str, "scheduled", avg_ms, _tflops(m, n, k, avg_ms), _ratio(vanilla_ms, avg_ms))
    except Exception as e:
        results["scheduled"] = None
        _print_failed_row(shape_str, "scheduled", reason=str(e))

    # ------------------------------------------------------------------
    # 5. Preshuffle-sched (7.4) — schedule + shuffled scales
    # ------------------------------------------------------------------
    try:
        fn = get_kernel((m, n, k), (block_m, block_n, block_k),
                        preshuffle_scales=True, preshuffle_B=False)
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
        avg_ms = _bench(lambda: fn(x_gpu, x_scales_sh_gpu, w_gpu, w_scales_sh_gpu, c), warmup, iters)
        results["preshuffle-sched"] = avg_ms
        _print_row(shape_str, "preshuffle-sched", avg_ms, _tflops(m, n, k, avg_ms), _ratio(vanilla_ms, avg_ms))
    except Exception as e:
        results["preshuffle-sched"] = None
        _print_failed_row(shape_str, "preshuffle-sched", reason=str(e))

    # ------------------------------------------------------------------
    # 6. Preshuffle-B-sched (7.4) — asymmetric schedule + preshuffled B
    # ------------------------------------------------------------------
    try:
        fn = get_kernel((m, n, k), (block_m, block_n, block_k),
                        preshuffle_scales=False, preshuffle_B=True)
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
        avg_ms = _bench(lambda: fn(x_gpu, x_scales_gpu, w_ps_gpu, w_scales_gpu, c), warmup, iters)
        results["preshuffle-B-sched"] = avg_ms
        _print_row(shape_str, "preshuffle-B-sched", avg_ms, _tflops(m, n, k, avg_ms), _ratio(vanilla_ms, avg_ms))
    except Exception as e:
        results["preshuffle-B-sched"] = None
        _print_failed_row(shape_str, "preshuffle-B-sched", reason=str(e))

    # ------------------------------------------------------------------
    # 7. Preshuffle-all-sched (7.4) — schedule + shuffled scales + preshuffled B
    # ------------------------------------------------------------------
    try:
        fn = get_kernel((m, n, k), (block_m, block_n, block_k),
                        preshuffle_scales=True, preshuffle_B=True)
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
        avg_ms = _bench(lambda: fn(x_gpu, x_scales_sh_gpu, w_ps_gpu, w_scales_sh_gpu, c), warmup, iters)
        results["preshuffle-all-sched"] = avg_ms
        _print_row(shape_str, "preshuffle-all-sched", avg_ms, _tflops(m, n, k, avg_ms), _ratio(vanilla_ms, avg_ms))
    except Exception as e:
        results["preshuffle-all-sched"] = None
        _print_failed_row(shape_str, "preshuffle-all-sched", reason=str(e))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MXFP4 GEMM quick benchmark")

    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument(
        "--shapes", metavar="CSV",
        help="CSV file with M,N,K columns",
    )
    shape_group.add_argument("--m", type=int, default=None)

    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--block_m", type=int, default=128)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--block_k", type=int, default=128)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.shapes:
        shapes = _load_shapes(args.shapes)
    else:
        shapes = [(
            args.m if args.m is not None else 1024,
            args.n if args.n is not None else 1024,
            args.k if args.k is not None else 8192,
        )]

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
            print()
        row = bench_shape(
            m, n, k,
            args.block_m, args.block_n, args.block_k,
            args.splits, args.warmup, args.iters,
            compiler,
        )
        all_results.append((m, n, k, row))

    if len(shapes) > 1:
        print()
        print("=" * sum(_COL_W.values()))
        print(f"Summary: {len(shapes)} shapes")
        for kernel in KERNELS:
            vals = [r[kernel] for _, _, _, r in all_results if r.get(kernel) is not None]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {kernel:<20}  mean {avg:.3f} ms  ({len(vals)}/{len(shapes)} succeeded)")
        print("=" * sum(_COL_W.values()))


if __name__ == "__main__":
    main()
