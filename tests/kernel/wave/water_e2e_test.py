"""End-to-end tests for Water middle-end pipeline.

This test has LIT counterparts in $WAVE_ROOT/water/test/Integration/matmul*.mlir
and $WAVE_ROOT/water/test/Integration/attention*.mlir.
If it fails, check the LIT counterpart for error messages. If those pass, the
error is likely in the Python/C++ interfacing.
"""

import torch
from torch.testing import assert_close

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType, ScaledMMAType
from wave_lang.kernel.wave.mlir_converter.mlir_converter import PersistentEmitter
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


def _run_matmul_water_e2e(minimize_shared_allocs: bool):
    """Test Water PassManager with matmul kernel and e2e execution."""
    from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel

    m = 1024
    n = 5120
    k = 640

    gemm, hyperparams, _ = get_gemm_kernel(
        shape=(m, n, k),
        dynamic_dims=False,
        mfma_variant=MMAType.F32_32x32x8_F16,
        block_shape=(64, 64, 32),
        waves_per_block=(2, 2),
    )

    options_mlir = WaveCompileOptions(
        subs=hyperparams,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        minimize_shared_allocs=minimize_shared_allocs,
    )
    options_mlir = set_default_run_config(options_mlir)

    compiled_kernel = wave_compile(options_mlir, gemm)
    trace = compiled_kernel.compiled_graph
    constraints = gemm.constraints

    with PersistentEmitter() as emitter:
        wave_dialect_mlir, diagnostics, _ = emitter.emit_wave_dialect(
            trace, constraints, options_mlir
        )

    lowered_mlir = apply_water_middle_end_passes(wave_dialect_mlir)

    a_tensor = device_randn(m, k, dtype=torch.float16)
    b_tensor = device_randn(n, k, dtype=torch.float16)
    c_tensor = device_zeros(m, n, dtype=torch.float32)

    expected = torch.matmul(a_tensor, b_tensor.T).float()

    options_e2e = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        override_mlir=lowered_mlir,
        minimize_shared_allocs=minimize_shared_allocs,
    )
    options_e2e = set_default_run_config(options_e2e)

    compiled_e2e = wave_compile(options_e2e, gemm)
    compiled_e2e(a_tensor, b_tensor, c_tensor)

    assert_close(c_tensor, expected, rtol=1e-3, atol=1e-3)


@require_e2e
@require_water_and_ee
def test_matmul_water_e2e():
    """Test matmul with separate shared memory allocations."""
    _run_matmul_water_e2e(minimize_shared_allocs=False)


@require_e2e
@require_water_and_ee
def test_matmul_water_e2e_minimize_shared_allocs():
    """Test matmul with minimized shared memory allocations (parent allocations)."""
    _run_matmul_water_e2e(minimize_shared_allocs=True)


@require_e2e
@require_water_and_ee
def test_attention_water():
    torch.manual_seed(0)

    attention, hyperparams, _ = get_vanilla_attention_kernel(
        AttentionShape(
            num_query_heads=8,
            num_kv_heads=2,
            query_seq_len=256,
            head_size_kv=64,
            head_size=64,
            kv_seq_len=256,
        ),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        dynamic_dims=False,
    )

    options_mlir = WaveCompileOptions(
        subs=hyperparams,
        compile_to_mlir=True,
        # TODO(#982): this pass creates IR that appears malformed, though pywave
        # manages to execute it.
        enable_mark_hardware_transpose_candidates=False,
    )
    options_mlir = set_default_run_config(options_mlir)
    compiled_kernel = wave_compile(options_mlir, attention)
    trace = compiled_kernel.compiled_graph
    constraints = attention.constraints

    with PersistentEmitter() as emitter:
        wave_dialect_mlir, diagnostics, _ = emitter.emit_wave_dialect(
            trace, constraints, options_mlir
        )
    assert len(diagnostics) == 0, f"Should have no diagnostics, got: {diagnostics}"

    lowered_mlir = apply_water_middle_end_passes(wave_dialect_mlir)

    options_e2e = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        override_mlir=lowered_mlir,
        enable_mark_hardware_transpose_candidates=False,
    )
    options_e2e = set_default_run_config(options_e2e)
    compiled_e2e = wave_compile(options_e2e, attention)
    query = device_randn(8, 256, 64, dtype=torch.float16)
    key = device_randn(8, 256, 64, dtype=torch.float16)
    value = device_randn(8, 256, 64, dtype=torch.float16)
    output = device_zeros(8, 256, 64, dtype=torch.float32)
    compiled_e2e(query, key, value, output)

    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None
    )

    assert_close(output, expected, rtol=1e-3, atol=1e-3, check_dtype=False)


@require_e2e
@require_water_and_ee
@require_cdna4
def test_scaled_mma_mxfp4_water_e2e():
    """Test scaled MMA (MXFP4) through the Water middle-end pipeline on MI350x."""
    M_val, N_val, K_val = 1024, 1024, 1024
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def scaled_gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
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

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 256,
        M: M_val,
        N: N_val,
        K: K_val,
    }

    # Step 1: Compile to Wave-dialect MLIR
    options_mlir = WaveCompileOptions(
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
    assert len(diagnostics) == 0, f"Should have no error diagnostics, got: {diagnostics}"

    # Step 2: Lower through Water middle-end
    lowered_mlir = apply_water_middle_end_passes(wave_dialect_mlir)

    # Step 3: Execute and verify
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs((M_val, N_val, K_val))
    out = device_zeros(M_val, N_val, dtype=torch.float32)

    options_e2e = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        override_mlir=lowered_mlir,
    )
    options_e2e = set_default_run_config(options_e2e)

    compiled_e2e = wave_compile(options_e2e, scaled_gemm)
    compiled_e2e(x, x_scales, w.T.contiguous(), w_scales, out)

    expected = torchScaledGemmMXFP4(x, w, x_scales, w_scales)
    assert_close(out, expected, rtol=1e-3, atol=1e-3, check_dtype=False)
