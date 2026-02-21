"""
MXFP4 GEMM Quick Benchmark

Compares MXFP4 GEMM kernels on one or more problem sizes.
All kernels are unscheduled (no double-buffer pipelining) unless noted.

  1. vanilla                  : unshuffled scales, all through LDS, no schedule
  2. preshuffle-scales        : shuffled scales read from GLOBAL, B through LDS, no schedule
  3. preshuffle-B             : preshuffled B read from GLOBAL, unshuffled scales through LDS, no schedule
  4. preshuffle-all           : preshuffled scales + preshuffled B, "all opts except scheduling"
  5. splitk                   : split-K via wave_asm, unshuffled scales, bf16 out
  6. splitk-preshuffle-scales : split-K via wave_asm + preshuffled scales, bf16 out
  7. splitk-preshuffle-B      : split-K via wave_asm + preshuffled B, unshuffled scales, bf16 out

Throughput is reported in TFLOPS using 2*M*N*K as the FLOP count.

Usage:
    python mxfp4-quick-bench.py --m 1024 --n 1024 --k 8192
    python mxfp4-quick-bench.py --shapes shapes-mxfp4-medium.csv
    python mxfp4-quick-bench.py --shapes shapes-mxfp4-medium.csv --splits 4
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths — add e2e dir first so waveasm_e2e / test_asm_backend_e2e are found,
# then insert examples/python at position 0 so our utils.py wins over
# wave_lang/kernel/wave/asm/utils.py (which test_asm_backend_e2e puts on path).
# ---------------------------------------------------------------------------
_EXAMPLES_DIR = Path(__file__).resolve().parent
_WAVE_ROOT = _EXAMPLES_DIR.parent.parent
_E2E_DIR = (
    _WAVE_ROOT / "wave_lang" / "kernel" / "wave" / "asm" / "wave_asm" / "test" / "e2e"
)

for _p in [str(_WAVE_ROOT), str(_E2E_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import e2e helpers before pinning _EXAMPLES_DIR at front.
# test_asm_backend_e2e inserts wave_lang/kernel/wave/asm/ into sys.path[0],
# which would shadow examples/python/utils.py; we fix that below.
from waveasm_e2e import WaveASMCompiler, capture_wave_kernel_info
from test_asm_backend_e2e import get_target_arch

# Always put examples/python at the very front — even if it's already in
# sys.path — so that "from utils import …" in 7.x files finds our utils.py
# instead of wave_lang/kernel/wave/asm/utils.py.
if str(_EXAMPLES_DIR) in sys.path:
    sys.path.remove(str(_EXAMPLES_DIR))
sys.path.insert(0, str(_EXAMPLES_DIR))

# ---------------------------------------------------------------------------
# Wave imports
# ---------------------------------------------------------------------------
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.templates.gemm import get_splitk_mxfp4_gemm_kernel
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

# ---------------------------------------------------------------------------
# Load helpers from 7.2
# ---------------------------------------------------------------------------
from importlib.util import spec_from_file_location, module_from_spec


def _load_module(name, path):
    spec = spec_from_file_location(name, str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod72 = _load_module("mxfp4_gemm_72", _EXAMPLES_DIR / "7.2_mxfp4_gemm_preshuffle_scale.py")

generate_mxfp4_inputs = _mod72.generate_mxfp4_inputs
e8m0_shuffle = _mod72.e8m0_shuffle
reference_mxfp4_gemm = _mod72.reference_mxfp4_gemm
get_vanilla_kernel = _mod72.get_vanilla_kernel
get_preshuffle_kernel = _mod72.get_preshuffle_kernel

# preshuffle_b_aiter from 7.3 (load after _EXAMPLES_DIR is on sys.path)
_mod73 = _load_module("mxfp4_gemm_73", _EXAMPLES_DIR / "7.3_mxfp4_gemm_preshuffle_B.py")
preshuffle_b_aiter = _mod73.preshuffle_b_aiter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Comment out entries to skip them in the benchmark.
ENABLED_KERNELS = [
    "vanilla",
    "preshuffle-scales",
    # "preshuffle-B",
    # "preshuffle-all",
    "splitk",
    "splitk-preshuffle-scales",
    # "splitk-preshuffle-B",
]

ALL_KERNELS = [
    "vanilla",
    "preshuffle-scales",
    "preshuffle-B",
    "preshuffle-all",
    "splitk",
    "splitk-preshuffle-scales",
    "splitk-preshuffle-B",
]
_COL_W = {"shape": 22, "kernel": 22, "ms": 10, "tflops": 10, "ratio": 10, "check": 8}


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def _compile_vanilla_or_preshuffle(kernel_fn, m, n, k, block_m, block_n, block_k):
    """Compile a vanilla or preshuffle-scales kernel from 7.2 (no schedule)."""
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


def _build_noschedule_kernel(m, n, k, block_m, block_n, block_k,
                              preshuffle_scales: bool, preshuffle_B: bool):
    """
    Build an unscheduled MXFP4 GEMM kernel with any combination of:
      - preshuffle_scales: a_scale and b_scale read from GLOBAL with IndexMapping
      - preshuffle_B: b read from GLOBAL with aiter preshuffled IndexMapping

    No double-buffer schedule is applied; this is a plain no-schedule kernel.
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    K_PACKED = tkl.sym.K_PACKED
    K_SCALE_SHUFFLED = tkl.sym.K_SCALE_SHUFFLED

    k_packed_val = k // 2
    k_scale_shuffled_val = (((k // 32) + 7) // 8) * 8
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    # Wave config: 8-wave (4×2) when B comes from global, 4-wave (2×2) otherwise.
    if preshuffle_B:
        wave_m, wave_n = 4, 2
        m_iter = tkl.sym.m_iter
        n_iter = tkl.sym.n_iter
        extra_constraints = [tkw.IteratorBindings({m_iter: M, n_iter: N})]
    else:
        wave_m, wave_n = 2, 2
        extra_constraints = []

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / wave_m),
        tkw.WaveConstraint(N, BLOCK_N / wave_n),
        *extra_constraints,
        tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant),
    ]

    # Build IndexMappings as needed.
    a_scale_mapping = None
    b_scale_mapping = None
    b_preshuffle_mapping = None

    if preshuffle_scales:
        i = tkw.IndexMapping.iterator(0)
        j = tkw.IndexMapping.iterator(1)
        # Substitute k_scale_shuffled_val directly to avoid leaking iterator
        # symbols into the grid computation when combined with preshuffle_B.
        _flat_a = (
            (j // 32) * ((k_scale_shuffled_val // 8) * 256)
            + (i // 8) * 256
            + ((i % 8) % 4) * 64
            + ((j % 32) % 16) * 4
            + (((i % 8) // 4) * 2)
            + ((j % 32) // 16)
        )
        a_scale_mapping = tkw.IndexMapping(
            num_iterators=2,
            inputs={M: _flat_a // k_scale_shuffled_val, K: _flat_a % k_scale_shuffled_val},
            outputs={K: i, M: j},
        )
        kk = tkw.IndexMapping.iterator(0)
        n_s = tkw.IndexMapping.iterator(1)
        _flat_b = (
            (n_s // 32) * ((k_scale_shuffled_val // 8) * 256)
            + (kk // 8) * 256
            + ((kk % 8) % 4) * 64
            + ((n_s % 32) % 16) * 4
            + (((kk % 8) // 4) * 2)
            + ((n_s % 32) // 16)
        )
        b_scale_mapping = tkw.IndexMapping(
            num_iterators=2,
            inputs={N: _flat_b // k_scale_shuffled_val, K: _flat_b % k_scale_shuffled_val},
            outputs={K: kk, N: n_s},
        )

    if preshuffle_B:
        n_it = tkw.IndexMapping.iterator(0)
        k_it = tkw.IndexMapping.iterator(1)
        within_nblk = (
            (k_it // 32) * 512 + ((k_it // 16) % 2) * 256 + (n_it % 16) * 16 + k_it % 16
        )
        # Substitute the concrete value of K_PACKED directly to avoid leaking
        # the symbolic K_PACKED / K_SCALE_SHUFFLED iterator into grid computation.
        b_preshuffle_mapping = tkw.IndexMapping(
            num_iterators=2,
            inputs={
                N: (n_it // 16) * 16 + within_nblk // k_packed_val,
                K: within_nblk % k_packed_val,
            },
            outputs={N: n_it, K: k_it},
        )

    # Preshuffled tensors must be in GLOBAL; others use the shared address space.
    a_scale_space = GLOBAL_ADDRESS_SPACE if preshuffle_scales else ADDRESS_SPACE
    b_space = GLOBAL_ADDRESS_SPACE if preshuffle_B else ADDRESS_SPACE
    b_scale_space = GLOBAL_ADDRESS_SPACE if preshuffle_scales else ADDRESS_SPACE

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, a_scale_space, tkl.i8],
        b: tkl.Memory[N, K / 2, b_space, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, b_scale_space, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale, mapping=a_scale_mapping)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b, mapping=b_preshuffle_mapping)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale, mapping=b_scale_mapping)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        K_PACKED: k_packed_val,
        K_SCALE_SHUFFLED: k_scale_shuffled_val,
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        use_global_to_shared=True,
        use_buffer_ops=preshuffle_B,
    )
    options = set_default_run_config(options)
    return wave_compile(options, gemm)


def _get_splitk_handle(m, n, k, block_m, block_n, block_k, num_splits, compiler,
                        preshuffle_B: bool = False, preshuffle_scales: bool = False):
    """Compile a split-K MXFP4 GEMM kernel via the wave_asm backend."""
    import wave_runtime

    splitk_fn, hyperparams = get_splitk_mxfp4_gemm_kernel(
        (m, n, k),
        num_splits=num_splits,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
        block_shape=(block_m, block_n, block_k),
        preshuffle_B=preshuffle_B,
        preshuffle_scales=preshuffle_scales,
    )

    # The template defaults to GLOBAL_ADDRESS_SPACE to work around a shared-
    # memory index bug, but the C++ ASM backend requires A/scales in SHARED +
    # GatherToLDS.  For preshuffle_B, B must stay in GLOBAL so the mapping can
    # read directly from the preshuffled layout without going through LDS.
    # For preshuffle_scales, a_scale/b_scale are already GLOBAL in the template.
    hyperparams[tkl.sym.ADDRESS_SPACE] = SHARED_ADDRESS_SPACE
    if preshuffle_B:
        hyperparams[tkl.sym.B_ADDRESS_SPACE] = GLOBAL_ADDRESS_SPACE
    else:
        hyperparams[tkl.sym.B_ADDRESS_SPACE] = SHARED_ADDRESS_SPACE

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=True,
    )
    options = set_default_run_config(options)

    kernel_info = capture_wave_kernel_info(options, splitk_fn)
    cpp_result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
    if not cpp_result.success:
        raise RuntimeError(f"wave_asm compilation failed: {cpp_result.error_message}")

    wave_runtime.load_hip_functions()
    _binary, gpu_func = wave_runtime.load_binary(
        str(cpp_result.binary_path),
        cpp_result.get_kernel_name() or kernel_info.kernel_name,
    )

    return {
        "gpu_func": gpu_func,
        "grid": kernel_info.grid_size,
        "block": kernel_info.workgroup_size,
        "lds_size": kernel_info.lds_size,
        "num_splits": num_splits,
    }


def _run_splitk(handle, x, x_scales, w, w_scales, c_out):
    """Launch a compiled split-K kernel (c_out must be zero-initialised bf16)."""
    import wave_runtime

    stream = torch.cuda.current_stream().cuda_stream
    kli = wave_runtime.KernelLaunchInfo(
        stream, handle["gpu_func"], handle["lds_size"],
        handle["grid"][0], handle["grid"][1], handle["grid"][2],
        handle["block"][0], handle["block"][1], handle["block"][2],
        1, 1, 1,
    )
    kern_args = wave_runtime.Int64Vector(
        [t.data_ptr() for t in [x, x_scales, w, w_scales, c_out]]
    )
    wave_runtime.launch(kli, kern_args, [], [])


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

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
        f"{'check':>{_COL_W['check']}}"
    )
    print("-" * sum(_COL_W.values()))


def _print_row(shape_str, kernel, avg_ms, tfl, ratio_str):
    print(
        f"{shape_str:<{_COL_W['shape']}}"
        f"{kernel:<{_COL_W['kernel']}}"
        f"{avg_ms:>{_COL_W['ms']}.3f}"
        f"{tfl:>{_COL_W['tflops']}.2f}"
        f"{ratio_str:>{_COL_W['ratio']}}"
        f"{'OK':>{_COL_W['check']}}"
    )
    sys.stdout.flush()


def _print_failed_row(shape_str, kernel, reason="FAILED"):
    print(
        f"{shape_str:<{_COL_W['shape']}}"
        f"{kernel:<{_COL_W['kernel']}}"
        f"{'':>{_COL_W['ms']}}"
        f"{'':>{_COL_W['tflops']}}"
        f"{'':>{_COL_W['ratio']}}"
        f"{'FAIL':>{_COL_W['check']}}"
    )
    print(f"  [error] {reason}", file=sys.stderr)
    sys.stdout.flush()


def _ratio(baseline_ms, avg_ms):
    return f"{baseline_ms/avg_ms:.2f}x" if baseline_ms else "-"


def _correctness_check(ref_f32, got, name, num_splits=1):
    """Check got (f32 or bf16) against ref_f32.  Raises on failure."""
    got_f32 = got.float().cpu()
    ref_cpu = ref_f32.cpu()
    diff = (got_f32 - ref_cpu).abs()
    max_diff = diff.max().item()
    ref_max = ref_cpu.abs().max().item()
    got_max = got_f32.abs().max().item()

    if got_max == 0.0 and ref_max > 0.01:
        raise RuntimeError(
            f"{name} correctness FAIL: output is all zeros "
            f"(ref_max={ref_max:.2f})"
        )

    if got.dtype == torch.bfloat16:
        # bf16 split-K truncation + FP4 MMA imprecision vs torch reference.
        # Use 1% of output magnitude as the absolute floor, plus a per-split
        # component for the bf16 atomic add truncation.
        atol = max(1.0, num_splits * 1.0) + ref_max * 1e-2
        rtol = 5e-2
    else:
        atol = ref_max * 1e-3
        rtol = 1e-2

    if not torch.allclose(got_f32, ref_cpu, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"{name} correctness FAIL  max_diff={max_diff:.4f}  "
            f"atol={atol:.4f}  ref_max={ref_max:.2f}  got_max={got_max:.2f}"
        )


# ---------------------------------------------------------------------------
# Per-shape benchmark
# ---------------------------------------------------------------------------

def bench_shape(m, n, k, block_m, block_n, block_k, num_splits, warmup, iters, compiler):
    shape_str = f"{m}x{n}x{k}"
    results = {}

    # Prepare all input variants once (on CPU, then move to GPU as needed).
    x, w, x_scales, w_scales = generate_mxfp4_inputs(
        (m, n, k), device=torch.device("cpu")
    )
    # Always compute an independent torch reference for correctness checks.
    ref_f32 = reference_mxfp4_gemm(x, w, x_scales, w_scales)

    need_shuffled_scales = any(
        k in ENABLED_KERNELS
        for k in ("preshuffle-scales", "preshuffle-all",
                   "splitk-preshuffle-scales")
    )
    need_preshuffle_B = any(
        k in ENABLED_KERNELS
        for k in ("preshuffle-B", "preshuffle-all", "splitk-preshuffle-B")
    )

    x_gpu = x.cuda()
    w_gpu = w.cuda()
    x_scales_gpu = x_scales.cuda()
    w_scales_gpu = w_scales.cuda()

    if need_shuffled_scales:
        x_scales_sh = e8m0_shuffle(x_scales)
        w_scales_sh = e8m0_shuffle(w_scales)
        x_scales_sh_gpu = x_scales_sh.cuda()
        w_scales_sh_gpu = w_scales_sh.cuda()

    if need_preshuffle_B:
        w_ps = preshuffle_b_aiter(w)
        w_ps_gpu = w_ps.cuda()

    vanilla_ms = None

    # ------------------------------------------------------------------
    # 1. Vanilla — unshuffled scales, all through LDS, no schedule
    # ------------------------------------------------------------------
    if "vanilla" in ENABLED_KERNELS:
        try:
            vanilla_fn = _compile_vanilla_or_preshuffle(
                get_vanilla_kernel(), m, n, k, block_m, block_n, block_k
            )
            c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
            vanilla_fn(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c)
            torch.cuda.synchronize()
            _correctness_check(ref_f32, c, "vanilla")

            avg_ms = _bench(
                lambda: vanilla_fn(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c),
                warmup, iters,
            )
            results["vanilla"] = avg_ms
            vanilla_ms = avg_ms
            _print_row(shape_str, "vanilla", avg_ms, _tflops(m, n, k, avg_ms), "-")
        except Exception as e:
            results["vanilla"] = None
            _print_failed_row(shape_str, "vanilla", reason=str(e))

    # ------------------------------------------------------------------
    # 2. Preshuffle-scales — shuffled scales from GLOBAL, B through LDS, no schedule
    # ------------------------------------------------------------------
    if "preshuffle-scales" in ENABLED_KERNELS:
        try:
            fn = _compile_vanilla_or_preshuffle(
                get_preshuffle_kernel(), m, n, k, block_m, block_n, block_k
            )
            c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
            fn(x_gpu, x_scales_sh_gpu, w_gpu, w_scales_sh_gpu, c)
            torch.cuda.synchronize()
            _correctness_check(ref_f32, c, "preshuffle-scales")

            avg_ms = _bench(
                lambda: fn(x_gpu, x_scales_sh_gpu, w_gpu, w_scales_sh_gpu, c),
                warmup, iters,
            )
            results["preshuffle-scales"] = avg_ms
            _print_row(shape_str, "preshuffle-scales", avg_ms, _tflops(m, n, k, avg_ms),
                       _ratio(vanilla_ms, avg_ms))
        except Exception as e:
            results["preshuffle-scales"] = None
            _print_failed_row(shape_str, "preshuffle-scales", reason=str(e))

    # ------------------------------------------------------------------
    # 3. Preshuffle-B — preshuffled B from GLOBAL, unshuffled scales, no schedule
    # ------------------------------------------------------------------
    if "preshuffle-B" in ENABLED_KERNELS:
        try:
            fn = _build_noschedule_kernel(
                m, n, k, block_m, block_n, block_k,
                preshuffle_scales=False, preshuffle_B=True,
            )
            c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
            fn(x_gpu, x_scales_gpu, w_ps_gpu, w_scales_gpu, c)
            torch.cuda.synchronize()
            _correctness_check(ref_f32, c, "preshuffle-B")

            avg_ms = _bench(
                lambda: fn(x_gpu, x_scales_gpu, w_ps_gpu, w_scales_gpu, c),
                warmup, iters,
            )
            results["preshuffle-B"] = avg_ms
            _print_row(shape_str, "preshuffle-B", avg_ms, _tflops(m, n, k, avg_ms),
                       _ratio(vanilla_ms, avg_ms))
        except Exception as e:
            results["preshuffle-B"] = None
            _print_failed_row(shape_str, "preshuffle-B", reason=str(e))

    # ------------------------------------------------------------------
    # 4. Preshuffle-all — preshuffled scales + preshuffled B, no schedule
    # ------------------------------------------------------------------
    if "preshuffle-all" in ENABLED_KERNELS:
        try:
            fn = _build_noschedule_kernel(
                m, n, k, block_m, block_n, block_k,
                preshuffle_scales=True, preshuffle_B=True,
            )
            c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
            fn(x_gpu, x_scales_sh_gpu, w_ps_gpu, w_scales_sh_gpu, c)
            torch.cuda.synchronize()
            _correctness_check(ref_f32, c, "preshuffle-all")

            avg_ms = _bench(
                lambda: fn(x_gpu, x_scales_sh_gpu, w_ps_gpu, w_scales_sh_gpu, c),
                warmup, iters,
            )
            results["preshuffle-all"] = avg_ms
            _print_row(shape_str, "preshuffle-all", avg_ms, _tflops(m, n, k, avg_ms),
                       _ratio(vanilla_ms, avg_ms))
        except Exception as e:
            results["preshuffle-all"] = None
            _print_failed_row(shape_str, "preshuffle-all", reason=str(e))

    # ------------------------------------------------------------------
    # 5. Splitk — wave_asm backend, unshuffled scales, bf16 out
    # ------------------------------------------------------------------
    if "splitk" in ENABLED_KERNELS:
        try:
            handle = _get_splitk_handle(
                m, n, k, block_m, block_n, block_k, num_splits, compiler,
                preshuffle_B=False,
            )
            c_bf16 = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
            _run_splitk(handle, x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c_bf16)
            torch.cuda.synchronize()
            _correctness_check(ref_f32, c_bf16, "splitk", num_splits=num_splits)

            def run_sk():
                c_bf16.zero_()
                _run_splitk(handle, x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c_bf16)

            avg_ms = _bench(run_sk, warmup, iters)
            results["splitk"] = avg_ms
            _print_row(shape_str, "splitk", avg_ms, _tflops(m, n, k, avg_ms),
                       _ratio(vanilla_ms, avg_ms))
        except Exception as e:
            results["splitk"] = None
            _print_failed_row(shape_str, "splitk", reason=str(e))

    # ------------------------------------------------------------------
    # 6. Splitk-preshuffle-scales — wave_asm splitk + preshuffled scales, bf16 out
    # ------------------------------------------------------------------
    if "splitk-preshuffle-scales" in ENABLED_KERNELS:
        try:
            handle_ps_scales = _get_splitk_handle(
                m, n, k, block_m, block_n, block_k, num_splits, compiler,
                preshuffle_B=False, preshuffle_scales=True,
            )
            c_bf16 = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
            _run_splitk(handle_ps_scales, x_gpu, x_scales_sh_gpu, w_gpu, w_scales_sh_gpu, c_bf16)
            torch.cuda.synchronize()
            _correctness_check(ref_f32, c_bf16, "splitk-preshuffle-scales", num_splits=num_splits)

            def run_sk_ps_scales():
                c_bf16.zero_()
                _run_splitk(handle_ps_scales, x_gpu, x_scales_sh_gpu, w_gpu, w_scales_sh_gpu, c_bf16)

            avg_ms = _bench(run_sk_ps_scales, warmup, iters)
            results["splitk-preshuffle-scales"] = avg_ms
            _print_row(shape_str, "splitk-preshuffle-scales", avg_ms, _tflops(m, n, k, avg_ms),
                       _ratio(vanilla_ms, avg_ms))
        except Exception as e:
            results["splitk-preshuffle-scales"] = None
            _print_failed_row(shape_str, "splitk-preshuffle-scales", reason=str(e))

    # ------------------------------------------------------------------
    # 7. Splitk-preshuffle-B — wave_asm splitk + preshuffled B, bf16 out
    # ------------------------------------------------------------------
    if "splitk-preshuffle-B" in ENABLED_KERNELS:
        try:
            handle_ps = _get_splitk_handle(
                m, n, k, block_m, block_n, block_k, num_splits, compiler,
                preshuffle_B=True,
            )
            c_bf16 = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
            _run_splitk(handle_ps, x_gpu, x_scales_gpu, w_ps_gpu, w_scales_gpu, c_bf16)
            torch.cuda.synchronize()
            _correctness_check(ref_f32, c_bf16, "splitk-preshuffle-B", num_splits=num_splits)

            def run_sk_ps():
                c_bf16.zero_()
                _run_splitk(handle_ps, x_gpu, x_scales_gpu, w_ps_gpu, w_scales_gpu, c_bf16)

            avg_ms = _bench(run_sk_ps, warmup, iters)
            results["splitk-preshuffle-B"] = avg_ms
            _print_row(shape_str, "splitk-preshuffle-B", avg_ms, _tflops(m, n, k, avg_ms),
                       _ratio(vanilla_ms, avg_ms))
        except Exception as e:
            results["splitk-preshuffle-B"] = None
            _print_failed_row(shape_str, "splitk-preshuffle-B", reason=str(e))

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

    target = get_target_arch()
    print(f"target arch: {target}")
    compiler = WaveASMCompiler(target=target)

    print()
    _print_header()

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
        for kernel in ENABLED_KERNELS:
            vals = [r[kernel] for _, _, _, r in all_results if r.get(kernel) is not None]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {kernel:<22}  mean {avg:.3f} ms  ({len(vals)}/{len(shapes)} succeeded)")
        print("=" * sum(_COL_W.values()))


if __name__ == "__main__":
    main()
