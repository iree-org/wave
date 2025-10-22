# REQUIRES: water
# RUN: python %s | FileCheck %s


import sympy
from typing import Any

import torch

from wave_lang.kernel._support.indexing import IndexSymbol
import wave_lang.kernel.wave as wave
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.mlir_converter.mlir_converter import emit_wave_dialect
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.general_utils import run_test

M = tkl.sym.M
N = tkl.sym.N
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
ADDRESS_SPACE_A = tkl.sym.ADDRESS_SPACE_A
ADDRESS_SPACE_B = tkl.sym.ADDRESS_SPACE_B
ADDRESS_SPACE_C = tkl.sym.ADDRESS_SPACE_C

# Define constraints for the kernel
constraints = [
    # specifies how computation is tiled
    tkw.WorkgroupConstraint(M, BLOCK_M, 0),
    tkw.WorkgroupConstraint(N, BLOCK_N, 1),
    tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
    tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
    tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}),
]


@wave.wave(constraints)
def matrix_add(
    # defines matrix in memory of req dimension with specific data types
    a: Memory[M, N, ADDRESS_SPACE_A, tkl.f16],
    b: Memory[M, N, ADDRESS_SPACE_B, tkl.f16],
    c: Memory[M, N, ADDRESS_SPACE_C, tkl.f16],
):
    # Initialize the accumulator register with zeroes
    c_reg = Register[M, N, tkl.f16](0.0)

    # loads values from memory into registers
    a_reg = wave.read(a)
    b_reg = wave.read(b)

    # compute the sum
    c_reg = a_reg + b_reg

    # writing results back to memory
    wave.write(c_reg, c)


@run_test
def mlir_converter_matrix_add():
    """Test MLIR converter with matrix addition kernel."""
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

    # Compile the kernel to get the trace
    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,  # Avoid IREE compilation
    )
    options = set_default_run_config(options)

    # Compile the kernel to get the trace
    compiled_kernel = wave_compile(options, matrix_add)

    # Get the compiled graph from the compiled kernel
    trace = compiled_kernel.get_compiled_graph()

    # Use the mlir_converter to emit wave MLIR dialect
    mlir_output = emit_wave_dialect(trace, options)

    # Print to stdout for FileCheck
    print(mlir_output)

    # CHECK-LABEL: mlir_converter_matrix_add
    # CHECK: module
    # CHECK: func.func @kernel(%[[ARG0:.*]]: !wave.tensor<[@M, @N] of f16>, %[[ARG1:.*]]: !wave.tensor<[@M, @N] of f16>, %[[ARG2:.*]]: !wave.tensor<[@M, @N] of f16>
    # CHECK-SAME: BLOCK_M = 64 : i64
    # CHECK-SAME: BLOCK_N = 64 : i64
    # CHECK-SAME: M = 128 : i64
    # CHECK-SAME: N = 128 : i64

    # CHECK: %[[READ_A:.*]] = wave.read %[[ARG0]]
    # CHECK-SAME: index
    # CHECK-SAME: M : [{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, 1, 64)
    # CHECK-SAME: N : [{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, BLOCK_N ceildiv 2, 1)
    # CHECK-SAME: bounds
    # CHECK-SAME: #wave.read_write_bounds
    # CHECK-SAME: M = #wave.expr_list
    # CHECK-SAME: N = #wave.expr_list
    # CHECK-SAME: wave.elements_per_thread = 32
    # CHECK-SAME: (!wave.tensor<[@M, @N] of f16>) -> !wave.tensor<[@M, @N] of f16, <register>>

    # CHECK: %[[READ_B:.*]] = wave.read %[[ARG1]]
    # CHECK-SAME: index
    # CHECK-SAME: M : [{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, 1, 64)
    # CHECK-SAME: N : [{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, BLOCK_N ceildiv 2, 1)
    # CHECK-SAME: bounds
    # CHECK-SAME: #wave.read_write_bounds
    # CHECK-SAME: M = #wave.expr_list
    # CHECK-SAME: N = #wave.expr_list
    # CHECK-SAME: wave.elements_per_thread = 32
    # CHECK-SAME: (!wave.tensor<[@M, @N] of f16>) -> !wave.tensor<[@M, @N] of f16, <register>>

    # CHECK: %[[ADD:.*]] = wave.add %[[READ_A]], %[[READ_B]]
    # CHECK-SAME: index
    # CHECK-SAME: M : [{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, 1, 64)
    # CHECK-SAME: N : [{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, BLOCK_N ceildiv 2, 1)
    # CHECK-SAME: (!wave.tensor<[@M, @N] of f16, <register>>, !wave.tensor<[@M, @N] of f16, <register>>) -> !wave.tensor<[@M, @N] of f16, <register>>

    # CHECK: wave.write %[[ADD]], %[[ARG2]]
    # CHECK-SAME: index
    # CHECK-SAME: M : [{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, 1, 64)
    # CHECK-SAME: N : [{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, BLOCK_N ceildiv 2, 1)
    # CHECK-SAME: bounds
    # CHECK-SAME: #wave.read_write_bounds
    # CHECK-SAME: M = #wave.expr_list
    # CHECK-SAME: N = #wave.expr_list
    # CHECK-SAME: wave.elements_per_thread = 32
    # CHECK-SAME: !wave.tensor<[@M, @N] of f16, <register>>, !wave.tensor<[@M, @N] of f16>

    # CHECK: return


@run_test
def mlir_converter_gemm():
    """Test MLIR converter with matrix addition kernel."""
    gemm, subs, dynamic_symbols = get_gemm_kernel(
        shape=(1024, 5120, 640),
        dynamic_dims=False,
        mfma_variant=MMAType.F32_32x32x8_F16,
        dtype=torch.float16,
        threads_per_wave=64,
    )

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,  # Avoid IREE compilation
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, gemm)

    # Get the trace from the compiled kernel
    trace = compiled_kernel.trace

    # Use the mlir_converter to emit wave MLIR dialect
    mlir_output = emit_wave_dialect(trace, options)

    # Print to stdout for FileCheck
    print(mlir_output)

    # TODO: Check lines
