"""
MXFP4 Scaled GEMM Scheduling for GFX950 (MI350)

Double-buffered MXFP4 GEMM with 4-wave and 8-wave configurations, plus split-K.
Uses get_tagged_mxfp4_gemm (templates) + get_mxfp4_dbuf_schedule (schedules).
Split-K kernels use the wave_asm backend with atomic bf16 output.

Usage:
    python 7.1_schedule.py --test test_dbuf_4wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm --debug
    python 7.1_schedule.py --test test_splitk_gemm
    python 7.1_schedule.py --test test_splitk_preshuffle_scales_gemm
    python 7.1_schedule.py --list_tests
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.templates import (
    get_tagged_mxfp4_gemm,
    get_tagged_mxfp4_gemm_preshuffle_b,
)
from wave_lang.kernel.wave.templates.gemm import get_splitk_mxfp4_gemm_kernel
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_dbuf_schedule,
    get_mxfp4_dbuf_pingpong_schedule,
    get_mxfp4_dbuf_mixed_pingpong_schedule,
    get_mxfp4_asymmetric_schedule,
)
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
    b_preshuffle,
    e8m0_shuffle,
)
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from utils import parse_args, list_tests, run_test

_EXAMPLES_DIR = Path(__file__).parent
_WAVE_ROOT = _EXAMPLES_DIR.parent.parent
_E2E_DIR = (
    _WAVE_ROOT / "wave_lang" / "kernel" / "wave" / "asm" / "wave_asm" / "test" / "e2e"
)
for _p in [str(_EXAMPLES_DIR), str(_WAVE_ROOT), str(_E2E_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _run_mxfp_gemm(gemm, shape):
    """Run compiled GEMM kernel and verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    gemm(x, x_scales, w.T.contiguous(), w_scales, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def _run_mxfp_gemm_preshuffle_b(gemm, shape):
    """Run compiled GEMM kernel with preshuffled B and B_scale, verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    w_t = w.T.contiguous()
    w_t_ps = b_preshuffle(w_t)
    x_scales_ps = e8m0_shuffle(x_scales)
    w_scales_ps = e8m0_shuffle(w_scales)

    x, w_t_ps = x.cuda(), w_t_ps.cuda()
    x_scales_ps, w_scales_ps = x_scales_ps.cuda(), w_scales_ps.cuda()
    out = torch.zeros(x.shape[0], w_t_ps.shape[0], dtype=torch.float32).cuda()

    gemm(x, x_scales_ps, w_t_ps, w_scales_ps, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def test_dbuf_4wave_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 4 waves, no stagger."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(2, 2))
    schedule = get_mxfp4_dbuf_schedule(use_stagger=False)

    options.print_ir_after = "all" if is_debug else []
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave.mlir"
    options.print_mlir = True
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 4-wave test passed!")


def test_dbuf_8wave_pingpong_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 8 waves, with stagger."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(4, 2))
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_pingpong_schedule(use_stagger=True, shape=shape)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)
    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 8-wave ping pong test passed!")


def test_dbuf_8wave_mixed_pingpong_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 8 waves, with stagger.

    A variant of the ping-pong schedule that hides the latency of the extra
    WorkgroupBarrier required for large shapes. With staggering, the two
    clusters of waves write to LDS at different times, so a second barrier is
    needed to ensure all writes are visible before any wave reads. This
    schedule overlaps that barrier with useful work by splitting LDS loads:

      - "Safe" loads: rows this wave wrote itself — readable immediately after
        memory_counter_wait, before the global WorkgroupBarrier.
      - "Dependent" loads: rows written by other waves — deferred until after
        the global WorkgroupBarrier.

    This lets the MFMAs on the safe operands start firing as soon as the
    barrier releases, effectively hiding the second barrier's latency behind
    the early loads and compute.
    """
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(4, 2))
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_mixed_pingpong_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 8-wave mixed ping pong test passed!")


def test_dbuf_4wave_mxfp_asymmetric_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Asymmetric-prefetch MXFP4 GEMM: A through LDS (2x prefetch), B direct from global."""
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric.mlir"
    options.print_mlir = True
    options.dump_binaries = "build/binaries"
    options.dump_intermediates = "build/intermediates"
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.use_water_backend = True
    schedule = get_mxfp4_asymmetric_schedule()

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM asymmetric-prefetch 4-wave test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Asymmetric MXFP4 GEMM with preshuffled B data and B scales."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.dump_intermediates = "build/intermediates"
    schedule = get_mxfp4_asymmetric_schedule()

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle_b(gemm, shape)
    print("MXFP GEMM preshuffle-B 4-wave test passed!")


@dataclass
class SplitKKernelHandle:
    """Opaque handle for a compiled split-K MXFP4 kernel."""

    gpu_func: object
    binary_path: Path
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    lds_size: int
    num_splits: int


def get_splitk_kernel(
    shape: tuple[int, int, int],
    block: tuple[int, int, int] = (128, 128, 128),
    num_splits: int = 2,
    waves_per_block: tuple[int, int] = (2, 2),
    preshuffle_scales: bool = False,
    compiler=None,
) -> SplitKKernelHandle:
    """Compile a split-K MXFP4 GEMM kernel through the wave_asm backend.

    Output tensor must be bf16 and zero-initialised before each call.
    w must be in [N, K/2] layout.

    Args:
        preshuffle_scales: If True, a_scale and b_scale are read from GLOBAL
            memory using the e8m0_shuffle IndexMapping.  The caller must pass
            scales pre-shuffled with e8m0_shuffle().
    """
    from waveasm_e2e import WaveASMCompiler, capture_wave_kernel_info
    from test_asm_backend_e2e import get_target_arch

    if compiler is None:
        compiler = WaveASMCompiler(target=get_target_arch())

    splitk_fn, hyperparams = get_splitk_mxfp4_gemm_kernel(
        shape,
        num_splits=num_splits,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
        block_shape=block,
        waves_per_block=waves_per_block,
        preshuffle_scales=preshuffle_scales,
    )
    hyperparams[tkl.sym.ADDRESS_SPACE] = SHARED_ADDRESS_SPACE
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
    cpp_result = compiler.compile_full(
        kernel_info.mlir_text, kernel_info.workgroup_size
    )
    if not cpp_result.success:
        raise RuntimeError(f"wave_asm compilation failed: {cpp_result.error_message}")

    import wave_runtime

    wave_runtime.load_hip_functions()
    _binary, gpu_func = wave_runtime.load_binary(
        str(cpp_result.binary_path),
        cpp_result.get_kernel_name() or kernel_info.kernel_name,
    )

    return SplitKKernelHandle(
        gpu_func=gpu_func,
        binary_path=cpp_result.binary_path,
        kernel_name=cpp_result.get_kernel_name() or kernel_info.kernel_name,
        grid=kernel_info.grid_size,
        block=kernel_info.workgroup_size,
        lds_size=kernel_info.lds_size,
        num_splits=num_splits,
    )


def run_splitk_kernel(
    handle: SplitKKernelHandle,
    x: torch.Tensor,
    x_scales: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    c_out: torch.Tensor,
) -> None:
    """Launch a compiled split-K kernel.

    c_out must be zero-initialised (dtype=torch.bfloat16).
    w must be in [N, K/2] layout.
    """
    import wave_runtime

    stream = torch.cuda.current_stream().cuda_stream
    kli = wave_runtime.KernelLaunchInfo(
        stream,
        handle.gpu_func,
        handle.lds_size,
        handle.grid[0],
        handle.grid[1],
        handle.grid[2],
        handle.block[0],
        handle.block[1],
        handle.block[2],
        1,
        1,
        1,
    )
    kern_args = wave_runtime.Int64Vector(
        [t.data_ptr() for t in [x, x_scales, w, w_scales, c_out]]
    )
    wave_runtime.launch(kli, kern_args, [], [])


def test_splitk_gemm(
    is_debug: bool = False,
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block: tuple[int, int, int] = (128, 128, 256),
):
    """Split-K MXFP4 GEMM (wave_asm backend, bf16 output, unshuffled scales)."""
    m, n, k = shape
    num_splits = 2
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    # w from generate_gemm_afp4wfp4_inputs is [K/2, N]; split-K kernel wants [N, K/2]
    w_nk = w.T.contiguous()
    torch_ref = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    handle = get_splitk_kernel(shape, block=block, num_splits=num_splits)

    c_out = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
    run_splitk_kernel(
        handle, x.cuda(), x_scales.cuda(), w_nk.cuda(), w_scales.cuda(), c_out
    )
    torch.cuda.synchronize()

    bf16_eps = 2**-7
    atol = num_splits * bf16_eps * max(torch_ref.abs().max().item(), 1.0)
    torch.testing.assert_close(
        torch_ref,
        c_out.cpu().to(torch.float32),
        check_dtype=False,
        check_device=False,
        atol=atol,
        rtol=0.0,
    )
    print("Split-K MXFP4 GEMM test passed!")


def test_splitk_preshuffle_scales_gemm(
    is_debug: bool = False,
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block: tuple[int, int, int] = (128, 128, 256),
):
    """Split-K MXFP4 GEMM (wave_asm backend, bf16 output, preshuffled scales)."""
    m, n, k = shape
    num_splits = 2
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    w_nk = w.T.contiguous()
    torch_ref = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x_scales_sh = e8m0_shuffle(x_scales)
    w_scales_sh = e8m0_shuffle(w_scales)

    handle = get_splitk_kernel(
        shape, block=block, num_splits=num_splits, preshuffle_scales=True
    )

    c_out = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
    run_splitk_kernel(
        handle, x.cuda(), x_scales_sh.cuda(), w_nk.cuda(), w_scales_sh.cuda(), c_out
    )
    torch.cuda.synchronize()

    bf16_eps = 2**-7
    atol = num_splits * bf16_eps * max(torch_ref.abs().max().item(), 1.0)
    torch.testing.assert_close(
        torch_ref,
        c_out.cpu().to(torch.float32),
        check_dtype=False,
        check_device=False,
        atol=atol,
        rtol=0.0,
    )
    print("Split-K MXFP4 GEMM (preshuffled scales) test passed!")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(
        args.test, globals(), args.debug, args.repeat, args.shape, args.block
    )
    exit(0 if success else 1)
