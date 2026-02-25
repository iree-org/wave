"""
MXFP4 Scaled GEMM Scheduling for GFX950 (MI350)

Double-buffered MXFP4 GEMM with 4-wave and 8-wave configurations.
Uses get_tagged_mxfp4_gemm (templates) + get_mxfp4_dbuf_schedule (schedules).

Usage:
    python 7.1_schedule.py --test test_dbuf_4wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm --debug
    python 7.1_schedule.py --list_tests
"""

import torch

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.templates import (
    get_tagged_mxfp4_gemm,
    get_tagged_mxfp4_gemm_preshuffle_b,
)
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
from wave_lang.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE
from utils import parse_args, list_tests, run_test


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


def test_dbuf_4wave_mxfp_hipblaslt_gemm_cpp(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 4 waves (C++ WaveASM backend)."""
    import sys, os

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(__file__),
            "../../wave_lang/kernel/wave/asm/wave_asm/test/e2e",
        ),
    )
    from waveasm_e2e import (
        capture_wave_kernel_info,
        WaveASMCompiler,
        run_with_wave_runtime,
    )

    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    schedule = get_mxfp4_asymmetric_schedule()
    options.backend = "asm"
    options.wave_runtime = True
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)

    kernel_info = capture_wave_kernel_info(options, gemm, schedule=schedule)

    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_hipblaslt_cpp.mlir"
    with open(options.print_mlir_file, "w") as f:
        f.write(kernel_info.mlir_text)

    compiler = WaveASMCompiler(target="gfx950", keep_temp_files=True)
    result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
    if not result.success:
        raise RuntimeError(f"C++ ASM compilation failed: {result.error_message}")

    if result.asm_text:
        asm_path = "build/intermediates/hipblaslt_cpp.s"
        os.makedirs(os.path.dirname(asm_path), exist_ok=True)
        with open(asm_path, "w") as f:
            f.write(result.asm_text)
        print(
            f"C++ ASM written to {asm_path} ({len(result.asm_text.splitlines())} lines)"
        )

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)
    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(shape[0], shape[1], dtype=torch.float32).cuda()

    kernel_name = result.get_kernel_name() or kernel_info.kernel_name
    run_with_wave_runtime(
        binary_path=result.binary_path,
        inputs=[x, x_scales, w.T.contiguous(), w_scales],
        outputs=[out],
        grid=kernel_info.grid_size,
        block=kernel_info.workgroup_size,
        shared_memory_bytes=kernel_info.lds_size,
        func_name=kernel_name,
    )

    out_cpu = out.cpu()
    ref = torch_out if not torch_out.is_cuda else torch_out.cpu()
    mismatch = ~torch.isclose(ref, out_cpu, atol=1e-2, rtol=1e-2)
    num_mismatch = mismatch.sum().item()

    if num_mismatch > 0:
        M, N = ref.shape
        block_m, block_n = block[0], block[1]
        print(f"\n{'='*70}")
        print(f"MISMATCH MAP  ({num_mismatch}/{M*N} = {100*num_mismatch/(M*N):.1f}%)")
        print(f"Shape: M={M}, N={N}  |  Block: {block_m}x{block_n}")
        print(f"{'='*70}")

        wg_rows = (M + block_m - 1) // block_m
        wg_cols = (N + block_n - 1) // block_n

        # Intra-tile M-half breakdown: for each workgroup, split into
        # first half (0..block_m/2-1) and second half (block_m/2..block_m-1)
        half_m = block_m // 2
        print(f"\nIntra-tile M-half breakdown (per workgroup):")
        print(
            f"  Each {block_m}-row tile split into M_lo[0:{half_m}] and M_hi[{half_m}:{block_m}]"
        )
        print(f"{'':>8s}", end="")
        for wg_n in range(wg_cols):
            print(
                f"  {'N['+str(wg_n*block_n)+':'+str((wg_n+1)*block_n)+']':>20s}", end=""
            )
        print()
        for wg_m in range(wg_rows):
            m_base = wg_m * block_m
            for half_label, m_off_lo, m_off_hi in [
                ("lo", 0, half_m),
                ("hi", half_m, block_m),
            ]:
                m_lo = m_base + m_off_lo
                m_hi = min(m_base + m_off_hi, M)
                if m_lo >= M:
                    continue
                print(f"M[{m_lo:4d}:{m_hi:<4d}]", end="")
                for wg_n in range(wg_cols):
                    n_lo = wg_n * block_n
                    n_hi = min((wg_n + 1) * block_n, N)
                    cnt = mismatch[m_lo:m_hi, n_lo:n_hi].sum().item()
                    total = (m_hi - m_lo) * (n_hi - n_lo)
                    if cnt == 0:
                        print(f"  {'OK':>20s}", end="")
                    else:
                        print(
                            f"  {cnt:>6d}/{total:<6d} ({100*cnt/total:4.1f}%)", end=""
                        )
                print()

        # Per-wave N breakdown within first workgroup that has errors
        # wave_shape=(1,4) → 4 waves in N, each handles block_n/4 cols
        wave_n_count = 4
        wave_n_size = block_n // wave_n_count
        print(
            f"\nPer-wave N breakdown (wave_shape=(1,4), {wave_n_count} waves x {wave_n_size} N-cols each):"
        )
        print(f"  Showing WG row M[0:{block_m}]:")
        print(f"{'':>14s}", end="")
        for wg_n in range(wg_cols):
            for w in range(wave_n_count):
                n0 = wg_n * block_n + w * wave_n_size
                print(f"  w{w}[{n0}:{n0+wave_n_size}]", end="")
        print()
        for half_label, m_off_lo, m_off_hi in [
            ("lo", 0, half_m),
            ("hi", half_m, block_m),
        ]:
            m_lo, m_hi = m_off_lo, min(m_off_hi, M)
            print(f"  M[{m_lo:4d}:{m_hi:<4d}]", end="")
            for wg_n in range(wg_cols):
                for w in range(wave_n_count):
                    n_lo = wg_n * block_n + w * wave_n_size
                    n_hi = n_lo + wave_n_size
                    cnt = mismatch[m_lo:m_hi, n_lo : min(n_hi, N)].sum().item()
                    total = (m_hi - m_lo) * (min(n_hi, N) - n_lo)
                    if cnt == 0:
                        print(f"  {'OK':>14s}", end="")
                    else:
                        print(f"  {cnt:>5d}/{total:<5d}", end="")
            print()

        # 16x16 MFMA tile heatmap for first workgroup
        mfma_m, mfma_n = 16, 16
        tiles_m = block_m // mfma_m
        tiles_n = block_n // mfma_n
        print(f"\nMFMA tile heatmap ({mfma_m}x{mfma_n} tiles, WG at M[0],N[0]):")
        print(f"{'':>10s}", end="")
        for tn in range(tiles_n):
            print(f" {tn:>4d}", end="")
        print("  ← N tile index")
        for tm in range(tiles_m):
            m_lo, m_hi = tm * mfma_m, (tm + 1) * mfma_m
            print(f"  M_t{tm:>2d}  ", end="")
            for tn in range(tiles_n):
                n_lo, n_hi = tn * mfma_n, (tn + 1) * mfma_n
                cnt = mismatch[m_lo:m_hi, n_lo:n_hi].sum().item()
                if cnt == 0:
                    print(f"  .  ", end="")
                else:
                    print(f" {cnt:>3d} ", end="")
            print()

        # Show a few sample mismatches
        mis_coords = mismatch.nonzero(as_tuple=False)
        print(f"\nSample mismatches (first 10):")
        print(
            f"  {'M':>6s} {'N':>6s} {'M%blk':>6s} {'N%blk':>6s} {'ref':>12s} {'got':>12s} {'abs_diff':>12s}"
        )
        for i in range(min(10, len(mis_coords))):
            m_i, n_i = mis_coords[i]
            r, g = ref[m_i, n_i].item(), out_cpu[m_i, n_i].item()
            print(
                f"  {m_i.item():>6d} {n_i.item():>6d} {m_i.item()%block_m:>6d} {n_i.item()%block_n:>6d} {r:>12.4f} {g:>12.4f} {abs(r-g):>12.4f}"
            )

        # Row/col summary
        row_mismatch = mismatch.any(dim=1)
        col_mismatch = mismatch.any(dim=0)
        bad_rows = row_mismatch.nonzero(as_tuple=False).squeeze(-1).tolist()
        bad_cols = col_mismatch.nonzero(as_tuple=False).squeeze(-1).tolist()
        print(
            f"\nAffected M rows: {len(bad_rows)}/{M}  range=[{min(bad_rows)}, {max(bad_rows)}]"
        )
        print(
            f"Affected N cols: {len(bad_cols)}/{N}  range=[{min(bad_cols)}, {max(bad_cols)}]"
        )

        # Intra-block row histogram
        row_in_block = torch.tensor(bad_rows) % block_m
        print(f"\nIntra-block M offsets of affected rows (within {block_m}-row tile):")
        print(
            f"  min={row_in_block.min().item()}, max={row_in_block.max().item()}, "
            f"count={len(row_in_block)}"
        )
        # Intra-block col histogram
        col_in_block = torch.tensor(bad_cols) % block_n
        print(f"Intra-block N offsets of affected cols (within {block_n}-col tile):")
        print(
            f"  min={col_in_block.min().item()}, max={col_in_block.max().item()}, "
            f"count={len(col_in_block)}"
        )
        print(f"{'='*70}")

        raise AssertionError(f"Mismatch: {num_mismatch}/{M*N} elements differ")

    print("MXFP GEMM double-buffer 4-wave HIPBLASLT (C++ backend) test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm_cpp(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 4 waves (C++ WaveASM backend)."""
    import sys, os

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(__file__),
            "../../wave_lang/kernel/wave/asm/wave_asm/test/e2e",
        ),
    )
    from waveasm_e2e import (
        capture_wave_kernel_info,
        WaveASMCompiler,
        run_with_wave_runtime,
    )

    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    schedule = get_mxfp4_asymmetric_schedule()
    options.backend = "asm"
    options.wave_runtime = True
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)

    kernel_info = capture_wave_kernel_info(options, gemm, schedule=schedule)

    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_preshuffle_b_cpp.mlir"
    with open(options.print_mlir_file, "w") as f:
        f.write(kernel_info.mlir_text)

    compiler = WaveASMCompiler(target="gfx950", keep_temp_files=True)
    result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
    if not result.success:
        raise RuntimeError(f"C++ ASM compilation failed: {result.error_message}")

    if result.asm_text:
        asm_path = "build/intermediates/preshuffle_b_cpp.s"
        os.makedirs(os.path.dirname(asm_path), exist_ok=True)
        with open(asm_path, "w") as f:
            f.write(result.asm_text)
        print(
            f"C++ ASM written to {asm_path} ({len(result.asm_text.splitlines())} lines)"
        )

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    w_t = w.T.contiguous()
    w_t_ps = b_preshuffle(w_t)
    x_scales_ps = e8m0_shuffle(x_scales)
    w_scales_ps = e8m0_shuffle(w_scales)

    x, w_t_ps = x.cuda(), w_t_ps.cuda()
    x_scales_ps, w_scales_ps = x_scales_ps.cuda(), w_scales_ps.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(shape[0], shape[1], dtype=torch.float32).cuda()

    kernel_name = result.get_kernel_name() or kernel_info.kernel_name
    run_with_wave_runtime(
        binary_path=result.binary_path,
        inputs=[x, x_scales_ps, w_t_ps, w_scales_ps],
        outputs=[out],
        grid=kernel_info.grid_size,
        block=kernel_info.workgroup_size,
        shared_memory_bytes=kernel_info.lds_size,
        func_name=kernel_name,
    )

    out_cpu = out.cpu()
    ref = torch_out if not torch_out.is_cuda else torch_out.cpu()
    mismatch = ~torch.isclose(ref, out_cpu, atol=1e-2, rtol=1e-2)
    num_mismatch = mismatch.sum().item()

    if num_mismatch > 0:
        M, N = ref.shape
        block_m, block_n = block[0], block[1]
        print(f"\n{'='*70}")
        print(f"MISMATCH MAP  ({num_mismatch}/{M*N} = {100*num_mismatch/(M*N):.1f}%)")
        print(f"Shape: M={M}, N={N}  |  Block: {block_m}x{block_n}")
        print(f"{'='*70}")

        wg_rows = (M + block_m - 1) // block_m
        wg_cols = (N + block_n - 1) // block_n

        # Intra-tile M-half breakdown: for each workgroup, split into
        # first half (0..block_m/2-1) and second half (block_m/2..block_m-1)
        half_m = block_m // 2
        print(f"\nIntra-tile M-half breakdown (per workgroup):")
        print(
            f"  Each {block_m}-row tile split into M_lo[0:{half_m}] and M_hi[{half_m}:{block_m}]"
        )
        print(f"{'':>8s}", end="")
        for wg_n in range(wg_cols):
            print(
                f"  {'N['+str(wg_n*block_n)+':'+str((wg_n+1)*block_n)+']':>20s}", end=""
            )
        print()
        for wg_m in range(wg_rows):
            m_base = wg_m * block_m
            for half_label, m_off_lo, m_off_hi in [
                ("lo", 0, half_m),
                ("hi", half_m, block_m),
            ]:
                m_lo = m_base + m_off_lo
                m_hi = min(m_base + m_off_hi, M)
                if m_lo >= M:
                    continue
                print(f"M[{m_lo:4d}:{m_hi:<4d}]", end="")
                for wg_n in range(wg_cols):
                    n_lo = wg_n * block_n
                    n_hi = min((wg_n + 1) * block_n, N)
                    cnt = mismatch[m_lo:m_hi, n_lo:n_hi].sum().item()
                    total = (m_hi - m_lo) * (n_hi - n_lo)
                    if cnt == 0:
                        print(f"  {'OK':>20s}", end="")
                    else:
                        print(
                            f"  {cnt:>6d}/{total:<6d} ({100*cnt/total:4.1f}%)", end=""
                        )
                print()

        # Per-wave N breakdown within first workgroup that has errors
        # wave_shape=(1,4) → 4 waves in N, each handles block_n/4 cols
        wave_n_count = 4
        wave_n_size = block_n // wave_n_count
        print(
            f"\nPer-wave N breakdown (wave_shape=(1,4), {wave_n_count} waves x {wave_n_size} N-cols each):"
        )
        print(f"  Showing WG row M[0:{block_m}]:")
        print(f"{'':>14s}", end="")
        for wg_n in range(wg_cols):
            for w in range(wave_n_count):
                n0 = wg_n * block_n + w * wave_n_size
                print(f"  w{w}[{n0}:{n0+wave_n_size}]", end="")
        print()
        for half_label, m_off_lo, m_off_hi in [
            ("lo", 0, half_m),
            ("hi", half_m, block_m),
        ]:
            m_lo, m_hi = m_off_lo, min(m_off_hi, M)
            print(f"  M[{m_lo:4d}:{m_hi:<4d}]", end="")
            for wg_n in range(wg_cols):
                for w in range(wave_n_count):
                    n_lo = wg_n * block_n + w * wave_n_size
                    n_hi = n_lo + wave_n_size
                    cnt = mismatch[m_lo:m_hi, n_lo : min(n_hi, N)].sum().item()
                    total = (m_hi - m_lo) * (min(n_hi, N) - n_lo)
                    if cnt == 0:
                        print(f"  {'OK':>14s}", end="")
                    else:
                        print(f"  {cnt:>5d}/{total:<5d}", end="")
            print()

        # 16x16 MFMA tile heatmap for first workgroup
        mfma_m, mfma_n = 16, 16
        tiles_m = block_m // mfma_m
        tiles_n = block_n // mfma_n
        print(f"\nMFMA tile heatmap ({mfma_m}x{mfma_n} tiles, WG at M[0],N[0]):")
        print(f"{'':>10s}", end="")
        for tn in range(tiles_n):
            print(f" {tn:>4d}", end="")
        print("  ← N tile index")
        for tm in range(tiles_m):
            m_lo, m_hi = tm * mfma_m, (tm + 1) * mfma_m
            print(f"  M_t{tm:>2d}  ", end="")
            for tn in range(tiles_n):
                n_lo, n_hi = tn * mfma_n, (tn + 1) * mfma_n
                cnt = mismatch[m_lo:m_hi, n_lo:n_hi].sum().item()
                if cnt == 0:
                    print(f"  .  ", end="")
                else:
                    print(f" {cnt:>3d} ", end="")
            print()

        # Show a few sample mismatches
        mis_coords = mismatch.nonzero(as_tuple=False)
        print(f"\nSample mismatches (first 10):")
        print(
            f"  {'M':>6s} {'N':>6s} {'M%blk':>6s} {'N%blk':>6s} {'ref':>12s} {'got':>12s} {'abs_diff':>12s}"
        )
        for i in range(min(10, len(mis_coords))):
            m_i, n_i = mis_coords[i]
            r, g = ref[m_i, n_i].item(), out_cpu[m_i, n_i].item()
            print(
                f"  {m_i.item():>6d} {n_i.item():>6d} {m_i.item()%block_m:>6d} {n_i.item()%block_n:>6d} {r:>12.4f} {g:>12.4f} {abs(r-g):>12.4f}"
            )

        # Row/col summary
        row_mismatch = mismatch.any(dim=1)
        col_mismatch = mismatch.any(dim=0)
        bad_rows = row_mismatch.nonzero(as_tuple=False).squeeze(-1).tolist()
        bad_cols = col_mismatch.nonzero(as_tuple=False).squeeze(-1).tolist()
        print(
            f"\nAffected M rows: {len(bad_rows)}/{M}  range=[{min(bad_rows)}, {max(bad_rows)}]"
        )
        print(
            f"Affected N cols: {len(bad_cols)}/{N}  range=[{min(bad_cols)}, {max(bad_cols)}]"
        )

        # Intra-block row histogram
        row_in_block = torch.tensor(bad_rows) % block_m
        print(f"\nIntra-block M offsets of affected rows (within {block_m}-row tile):")
        print(
            f"  min={row_in_block.min().item()}, max={row_in_block.max().item()}, "
            f"count={len(row_in_block)}"
        )
        # Intra-block col histogram
        col_in_block = torch.tensor(bad_cols) % block_n
        print(f"Intra-block N offsets of affected cols (within {block_n}-col tile):")
        print(
            f"  min={col_in_block.min().item()}, max={col_in_block.max().item()}, "
            f"count={len(col_in_block)}"
        )
        print(f"{'='*70}")

        raise AssertionError(f"Mismatch: {num_mismatch}/{M*N} elements differ")

    print("MXFP GEMM double-buffer 4-wave preshuffle-B (C++ backend) test passed!")


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
