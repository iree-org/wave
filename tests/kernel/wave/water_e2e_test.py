"""End-to-end tests for Water middle-end pipeline.

This test has LIT counterparts in $WAVE_ROOT/water/test/Integration/matmul*.mlir
and $WAVE_ROOT/water/test/Integration/attention*.mlir.
If it fails, check the LIT counterpart for error messages. If those pass, the
error is likely in the Python/C++ interfacing.
"""

import time
from pathlib import Path

import torch
from torch.testing import assert_close

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType, ScaledMMAType
from wave_lang.kernel.wave.mlir_converter.diagnostics import error_diagnostics
from wave_lang.kernel.wave.mlir_converter.mlir_converter import PersistentEmitter
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.templates import AttentionShape
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
)
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
from wave_lang.kernel.wave.water import apply_water_middle_end_passes
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)

from tests.kernel.common.utils import require_cdna4, require_e2e, require_water_and_ee


# def _run_matmul_water_e2e(minimize_shared_allocs: bool):
#     """Test Water PassManager with matmul kernel and e2e execution."""
#     from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel

#     m = 1024
#     n = 5120
#     k = 640

#     gemm, hyperparams, _ = get_gemm_kernel(
#         shape=(m, n, k),
#         dynamic_dims=False,
#         mfma_variant=MMAType.F32_32x32x8_F16,
#         block_shape=(64, 64, 32),
#         waves_per_block=(2, 2),
#     )

#     options_mlir = WaveCompileOptions(
#         subs=hyperparams,
#         compile_to_mlir=True,
#         linearize_reads=False,
#         location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
#         enforce_locations=False,
#         minimize_shared_allocs=minimize_shared_allocs,
#     )
#     options_mlir = set_default_run_config(options_mlir)

#     compiled_kernel = wave_compile(options_mlir, gemm)
#     trace = compiled_kernel.compiled_graph
#     constraints = gemm.constraints

#     with PersistentEmitter() as emitter:
#         wave_dialect_mlir, diagnostics, _ = emitter.emit_wave_dialect(
#             trace, constraints, options_mlir
#         )

#     lowered_mlir = apply_water_middle_end_passes(wave_dialect_mlir)

#     a_tensor = device_randn(m, k, dtype=torch.float16)
#     b_tensor = device_randn(n, k, dtype=torch.float16)
#     c_tensor = device_zeros(m, n, dtype=torch.float32)

#     expected = torch.matmul(a_tensor, b_tensor.T).float()

#     options_e2e = WaveCompileOptions(
#         subs=hyperparams,
#         canonicalize=True,
#         location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
#         enforce_locations=False,
#         override_mlir=lowered_mlir,
#         minimize_shared_allocs=minimize_shared_allocs,
#     )
#     options_e2e = set_default_run_config(options_e2e)

#     compiled_e2e = wave_compile(options_e2e, gemm)
#     compiled_e2e(a_tensor, b_tensor, c_tensor)

#     assert_close(c_tensor, expected, rtol=1e-3, atol=1e-3)


# @require_e2e
# @require_water_and_ee
# def test_matmul_water_e2e():
#     """Test matmul with separate shared memory allocations."""
#     _run_matmul_water_e2e(minimize_shared_allocs=False)


# @require_e2e
# @require_water_and_ee
# def test_matmul_water_e2e_minimize_shared_allocs():
#     """Test matmul with minimized shared memory allocations (parent allocations)."""
#     _run_matmul_water_e2e(minimize_shared_allocs=True)


# @require_e2e
# @require_water_and_ee
# def test_attention_water():
#     torch.manual_seed(0)

#     attention, hyperparams, _ = get_vanilla_attention_kernel(
#         AttentionShape(
#             num_query_heads=8,
#             num_kv_heads=2,
#             query_seq_len=256,
#             head_size_kv=64,
#             head_size=64,
#             kv_seq_len=256,
#         ),
#         (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
#         dynamic_dims=False,
#     )

#     options_mlir = WaveCompileOptions(
#         subs=hyperparams,
#         compile_to_mlir=True,
#         linearize_reads=False,
#         # TODO(#982): this pass creates IR that appears malformed, though pywave
#         # manages to execute it.
#         enable_mark_hardware_transpose_candidates=False,
#     )
#     options_mlir = set_default_run_config(options_mlir)
#     compiled_kernel = wave_compile(options_mlir, attention)
#     trace = compiled_kernel.compiled_graph
#     constraints = attention.constraints

#     with PersistentEmitter() as emitter:
#         wave_dialect_mlir, diagnostics, _ = emitter.emit_wave_dialect(
#             trace, constraints, options_mlir
#         )
#     assert len(diagnostics) == 0, f"Should have no diagnostics, got: {diagnostics}"

#     lowered_mlir = apply_water_middle_end_passes(wave_dialect_mlir)

#     options_e2e = WaveCompileOptions(
#         subs=hyperparams,
#         canonicalize=True,
#         location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
#         enforce_locations=False,
#         override_mlir=lowered_mlir,
#         enable_mark_hardware_transpose_candidates=False,
#     )
#     options_e2e = set_default_run_config(options_e2e)
#     compiled_e2e = wave_compile(options_e2e, attention)
#     query = device_randn(8, 256, 64, dtype=torch.float16)
#     key = device_randn(8, 256, 64, dtype=torch.float16)
#     value = device_randn(8, 256, 64, dtype=torch.float16)
#     output = device_zeros(8, 256, 64, dtype=torch.float32)
#     compiled_e2e(query, key, value, output)

#     expected = torch.nn.functional.scaled_dot_product_attention(
#         query, key, value, attn_mask=None
#     )

#     assert_close(output, expected, rtol=1e-3, atol=1e-3, check_dtype=False)


@require_e2e
@require_water_and_ee
@require_cdna4
def test_scaled_mma_mxfp4_water_e2e():
    """Test scaled MMA (MXFP4) through the Water middle-end pipeline on MI350x."""
    from wave_lang.kernel.wave.templates.gemm import get_scaled_gemm_kernel

    M_val, N_val, K_val = 10240, 10240, 10240

    scaled_gemm, hyperparams = get_scaled_gemm_kernel(
        shape=(M_val, N_val, K_val),
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )

    scaled_gemm2, hyperparams2 = get_scaled_gemm_kernel(
        shape=(M_val, N_val, K_val),
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )

    # Step 1: Compile to Wave-dialect MLIR
    options_mlir = WaveCompileOptions(
        linearize_reads=False,
        subs=hyperparams,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options_mlir = set_default_run_config(options_mlir)

    compiled_kernel = wave_compile(options_mlir, scaled_gemm)
    trace = compiled_kernel.compiled_graph
    kernel_constraints = scaled_gemm.constraints

    with PersistentEmitter() as emitter:
        wave_dialect_mlir, diagnostics, _ = emitter.emit_wave_dialect(
            trace, kernel_constraints, options_mlir
        )
    assert (
        len(error_diagnostics(diagnostics)) == 0
    ), f"Should have no error diagnostics, got: {diagnostics}"

    # Step 2: Lower through Water middle-end
    lowered_mlir = apply_water_middle_end_passes(wave_dialect_mlir)

    wave_mlir_file = Path.cwd() / "scaled_mma_mxfp4_water_e2e_wave_lowered.mlir"
    water_mlir_file = Path.cwd() / "scaled_mma_mxfp4_water_e2e_water_lowered.mlir"
    wave_mlir_file.write_text(compiled_kernel.asm)
    water_mlir_file.write_text(lowered_mlir)
    print(f"Wrote wave lowered MLIR to {wave_mlir_file}")
    print(f"Wrote water lowered MLIR to {water_mlir_file}")

    # Step 3: Execute and verify
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs((M_val, N_val, K_val))
    out = device_zeros(M_val, N_val, dtype=torch.float32)
    out_wave = device_zeros(M_val, N_val, dtype=torch.float32)

    dump_wave = Path.cwd() / "iree_dump_wave_kernel"
    dump_water_e2e = Path.cwd() / "iree_dump_water_e2e_kernel"
    dump_wave.mkdir(parents=True, exist_ok=True)
    dump_water_e2e.mkdir(parents=True, exist_ok=True)

    options_wave = WaveCompileOptions(
        subs=hyperparams2,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        dump_binaries=str(dump_wave),
    )
    options_wave = set_default_run_config(options_wave)
    wave_kernel = wave_compile(options_wave, scaled_gemm2)

    options_e2e = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        override_mlir=lowered_mlir,
        dump_binaries=str(dump_water_e2e),
    )
    options_e2e = set_default_run_config(options_e2e)
    compiled_e2e = wave_compile(options_e2e, scaled_gemm)

    print(f"IREE dumped wave-kernel binaries to {dump_wave}")
    print(f"IREE dumped water-e2e-kernel binaries to {dump_water_e2e}")

    wt = w.T.contiguous()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    expected = torchScaledGemmMXFP4(x, w, x_scales, w_scales)
    torch.cuda.synchronize()
    print(
        f"test_scaled_mma_mxfp4_water_e2e: torch wall time "
        f"(synced): {(time.perf_counter() - t0) * 1000:.3f} ms"
    )

    compiled_e2e(x, x_scales, wt, w_scales, out)
    compiled_e2e(x, x_scales, wt, w_scales, out)
    compiled_e2e(x, x_scales, wt, w_scales, out)
    time.sleep(3)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled_e2e(x, x_scales, wt, w_scales, out)
    torch.cuda.synchronize()
    print(
        f"test_scaled_mma_mxfp4_water_e2e: compiled_e2e wall time "
        f"(synced): {(time.perf_counter() - t0) * 1000:.3f} ms"
    )

    wave_kernel(x, x_scales, wt, w_scales, out_wave)
    wave_kernel(x, x_scales, wt, w_scales, out_wave)
    wave_kernel(x, x_scales, wt, w_scales, out_wave)
    time.sleep(3)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wave_kernel(x, x_scales, wt, w_scales, out_wave)
    torch.cuda.synchronize()
    print(
        f"test_scaled_mma_mxfp4_water_e2e: wave_kernel wall time "
        f"(synced): {(time.perf_counter() - t0) * 1000:.3f} ms"
    )

    assert_close(out, expected, rtol=1e-3, atol=1e-3, check_dtype=False)
    assert_close(out_wave, expected, rtol=1e-3, atol=1e-3, check_dtype=False)
