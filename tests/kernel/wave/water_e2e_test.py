"""End-to-end tests for Water middle-end pipeline."""

import torch
from torch.testing import assert_close

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.mlir_converter.mlir_converter import PersistentEmitter
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
from wave_lang.kernel.wave.water import apply_water_middle_end_passes
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)

from tests.kernel.common.utils import require_e2e, require_water_and_ee


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
