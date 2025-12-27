# REQUIRES: water
# RUN: python %s | FileCheck %s

import torch
from torch.testing import assert_close
from typing import Any
import sympy

from wave_lang.kernel._support.indexing import IndexSymbol
import wave_lang.kernel.wave as wave
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.mlir_converter.mlir_converter import emit_wave_dialect
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
from wave_lang.kernel.wave.water import apply_water_middle_end_passes
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)


@run_test
def test_matrix_add_water_e2e():
    """Test Water PassManager with Wave MLIR dialect generation and e2e execution."""
    torch.manual_seed(0)

    # Simple matrix addition kernel
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ADDRESS_SPACE_A = tkl.sym.ADDRESS_SPACE_A
    ADDRESS_SPACE_B = tkl.sym.ADDRESS_SPACE_B
    ADDRESS_SPACE_C = tkl.sym.ADDRESS_SPACE_C

    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        tkw.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    @wave.wave(constraints)
    def matrix_add(
        a: Memory[M, N, ADDRESS_SPACE_A, tkl.f16],
        b: Memory[M, N, ADDRESS_SPACE_B, tkl.f16],
        c: Memory[M, N, ADDRESS_SPACE_C, tkl.f16],
    ):
        # Load values from memory into registers
        a_reg = wave.read(a)
        b_reg = wave.read(b)

        # Compute the sum
        c_reg = a_reg + b_reg

        # Write results back to memory
        wave.write(c_reg, c)

    # Set parameters for compilation
    subs: dict[str | IndexSymbol, Any] = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        M: 128,
        N: 128,
    }

    options_mlir = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options_mlir = set_default_run_config(options_mlir)

    compiled_kernel = wave_compile(options_mlir, matrix_add)
    trace = compiled_kernel.compiled_graph
    constraints = matrix_add.constraints

    # Emit Wave dialect MLIR
    wave_dialect_mlir, diagnostics = emit_wave_dialect(trace, constraints, options_mlir)

    # Apply Water PassManager lowering
    lowered_mlir = apply_water_middle_end_passes(wave_dialect_mlir)

    print(lowered_mlir)

    # Create test tensors
    shape = (128, 128)
    a_tensor = device_randn(*shape, dtype=torch.float16)
    b_tensor = device_randn(*shape, dtype=torch.float16)
    c_tensor = device_zeros(*shape, dtype=torch.float16)

    # Expected result (CPU computation)
    expected = a_tensor + b_tensor

    # Test execution with lowered MLIR
    options_e2e = WaveCompileOptions(
        subs=subs,
        canonicalize=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        override_mlir=lowered_mlir,
    )
    options_e2e = set_default_run_config(options_e2e)

    compiled_e2e = wave_compile(options_e2e, matrix_add)

    compiled_e2e(a_tensor, b_tensor, c_tensor)

    assert_close(c_tensor, expected, rtol=1e-4, atol=1e-4)


# CHECK-LABEL:  test_matrix_add_water_e2e
# CHECK:        module
# CHECK-NOT:    wave.normal_form
# CHECK:        func.func @kernel(
# CHECK-NOT:    wave.read
# CHECK:        vector.maskedload
# CHECK:        vector.maskedload
# CHECK-NOT:    wave.add
# CHECK:        arith.addf
# CHECK-NOT:    wave.write
# CHECK:        vector.maskedstore
