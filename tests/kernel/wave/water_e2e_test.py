"""End-to-end tests for Water middle-end pipeline.

This test has LIT counterparts in $WAVE_ROOT/water/test/Integration/matmul*.mlir
and $WAVE_ROOT/water/test/Integration/attention*.mlir.
If it fails, check the LIT counterpart for error messages. If those pass, the
error is likely in the Python/C++ interfacing.
"""

from pathlib import Path

import torch
from torch.testing import assert_close
import triton

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
        use_global_to_shared=True,
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

    expected = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    compiled_e2e(x, x_scales, wt, w_scales, out)
    wave_kernel(x, x_scales, wt, w_scales, out_wave)

    assert_close(out, expected, rtol=1e-3, atol=1e-3, check_dtype=False)
    assert_close(out_wave, expected, rtol=1e-3, atol=1e-3, check_dtype=False)

    # Step 4: Benchmark with triton.testing
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[(M_val, N_val, K_val)],
            line_arg="impl",
            line_vals=["torch", "water_e2e", "wave_kernel"],
            line_names=["torch (torchScaledGemmMXFP4)", "water_e2e", "wave_kernel"],
            plot_name="scaled_gemm_mxfp4",
            args={},
            xlabel="(M, N, K)",
            ylabel="ms",
        )
    )
    def bench_scaled_gemm(M, N, K, impl):
        if impl == "torch":
            fn = lambda: torchScaledGemmMXFP4(x, w, x_scales, w_scales)
        elif impl == "water_e2e":
            fn = lambda: compiled_e2e(x, x_scales, wt, w_scales, out)
        else:
            fn = lambda: wave_kernel(x, x_scales, wt, w_scales, out_wave)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
        return ms, min_ms, max_ms

    bench_scaled_gemm.run(print_data=True)
