# REQUIRES: water
# RUN: python %s | FileCheck %s


import sympy
from typing import Any


from wave_lang.kernel._support.indexing import IndexSymbol
import wave_lang.kernel.wave as wave
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.mlir_converter.mlir_converter import emit_wave_dialect
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)

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
def mlir_converter_location():
    """Test MLIR converter debug location emission with a matrix addition kernel."""
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
        location_capture_config=LocationCaptureConfig(
            level=LocationCaptureLevel.FILE_LINE_COL
        ),
    )
    options = set_default_run_config(options)

    # Compile the kernel to get the trace
    compiled_kernel = wave_compile(options, matrix_add)

    # Get the compiled graph from the compiled kernel
    trace = compiled_kernel.get_compiled_graph()

    constraints = matrix_add.constraints

    # Use the mlir_converter to emit wave MLIR dialect
    mlir_output, diagnostics = emit_wave_dialect(trace, constraints, options, False)

    if diagnostics:
        print(diagnostics)
    assert (
        len(diagnostics) == 0
    ), "dialect emission should create valid IR, therefore diagnostics should be empty"

    # Print to stdout for FileCheck
    print(mlir_output)

    # CHECK-LABEL: mlir_converter_location
    # CHECK: #loc = loc("{{.*}}mlir_converter_debug_locations.py":44
    # CHECK: module
    # CHECK: func.func @kernel(%arg0: !wave.tensor<[@M, @N] of f16> loc("{{.*}}mlir_converter_debug_locations.py":44{{.*}}), %arg1: !wave.tensor<[@M, @N] of f16> loc("{{.*}}mlir_converter_debug_locations.py":44{{.*}}), %arg2: !wave.tensor<[@M, @N] of f16> loc("{{.*}}mlir_converter_debug_locations.py":44

    # CHECK: wave.read
    # CHECK-SAME: loc(#loc1)

    # CHECK: wave.read
    # CHECK-SAME: loc(#loc2)

    # CHECK: wave.add
    # CHECK-SAME: loc(#loc3)

    # CHECK: wave.write
    # CHECK-SAME: loc(#loc4)

    # CHECK: return loc(#loc)

    # CHECK: loc(#loc)
    # CHECK: loc(#loc)

    # CHECK: #loc1 = loc("{{.*}}mlir_converter_debug_locations.py":55
    # CHECK: #loc2 = loc("{{.*}}mlir_converter_debug_locations.py":56
    # CHECK: #loc3 = loc("{{.*}}mlir_converter_debug_locations.py":59
    # CHECK: #loc4 = loc("{{.*}}mlir_converter_debug_locations.py":62


@run_test
def mlir_converter_location_iterate():
    """Test MLIR converter debug location emission with iterate."""

    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    dtype = tkl.f16
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2))]
    constraints += [tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2))]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, mma_type=MMAType.F32_32x32x8_F16)
    ]

    @tkw.wave(constraints)
    def matmul(
        a: tkl.Memory[M, K, ADDRESS_SPACE, dtype],
        b: tkl.Memory[N, K, ADDRESS_SPACE, dtype],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, dtype]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, dtype]
            b_reg = tkw.read(b)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c)

    subs = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 8,
        M: 1024,
        N: 5120,
        K: 640,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,  # Avoid IREE compilation
        location_capture_config=LocationCaptureConfig(
            level=LocationCaptureLevel.FILE_LINE_COL
        ),
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, matmul)

    # Get the trace from the compiled kernel
    trace = compiled_kernel.compiled_graph

    constraints = matmul.constraints

    # Use the mlir_converter to emit wave MLIR dialect
    mlir_output, diagnostics = emit_wave_dialect(trace, constraints, options, False)

    if diagnostics:
        print(diagnostics)
    assert (
        len(diagnostics) == 0
    ), "dialect emission should create valid IR, therefore diagnostics should be empty"

    # Print to stdout for FileCheck
    print(mlir_output)

    # CHECK-LABEL: mlir_converter_location_iterate
    # CHECK: #loc = loc("{{.*}}mlir_converter_debug_locations.py":163
    # CHECK: #loc5 = loc("{{.*}}mlir_converter_debug_locations.py":180
    # CHECK: module
    # CHECK: func.func @kernel

    # CHECK: wave.iterate
    # CHECK: %arg3: !wave.tensor<[@M, @N] of f32, <register>> loc("{{.*}}mlir_converter_debug_locations.py":180

    # CHECK:        wave.read
    # CHECK-SAME:   loc(#loc3)

    # CHECK:        amdgpu.lds_barrier
    # CHECK-SAME:   loc(#loc3)

    # CHECK:        wave.write
    # CHECK-SAME:   loc(#loc3)

    # CHECK:        wave.read
    # CHECK-SAME:   loc(#loc1)

    # CHECK:        wave.write
    # CHECK-SAME:   loc(#loc1)

    # CHECK:        amdgpu.lds_barrier
    # CHECK-SAME:   loc(#loc1)

    # CHECK:        wave.read
    # CHECK-SAME:   loc(#loc1)

    # CHECK:        wave.read
    # CHECK-SAME:   loc(#loc3)

    # CHECK:        wave.mma
    # CHECK-SAME:   loc(#loc5)

    # CHECK:        wave.yield
    # CHECK-SAME:   loc(#loc4)

    # CHECK: (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>> loc(#loc4)

    # CHECK: #loc1 = loc("{{.*}}mlir_converter_debug_locations.py":178
    # CHECK: #loc2 = loc("{{.*}}mlir_converter_debug_locations.py":169
    # CHECK: #loc3 = loc("{{.*}}mlir_converter_debug_locations.py":176
    # CHECK: #loc4 = loc("{{.*}}mlir_converter_debug_locations.py":173
    # CHECK: #loc6 = loc("{{.*}}mlir_converter_debug_locations.py":184
