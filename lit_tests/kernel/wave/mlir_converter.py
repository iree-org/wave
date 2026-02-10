# REQUIRES: water
# RUN: python %s | FileCheck %s


import sys
import sympy
from typing import Any


from wave_lang.kernel._support.indexing import IndexSymbol
from wave_lang.kernel._support.tracing import CapturedTrace
import wave_lang.kernel.wave as wave
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import Constraint, MMAType
from wave_lang.kernel.wave.mlir_converter.mlir_converter import emit_wave_dialect
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.water import apply_water_middle_end_passes
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)
from wave_lang.support.ir_imports import (
    Context,
    Location,
    Module,
    UnitAttr,
)
from iree.compiler.dialects.transform.extras import insert_transform_script, OpHandle

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE_A = tkl.sym.ADDRESS_SPACE_A
ADDRESS_SPACE_B = tkl.sym.ADDRESS_SPACE_B
ADDRESS_SPACE_C = tkl.sym.ADDRESS_SPACE_C
ADDRESS_SPACE_TEST = tkl.sym.ADDRESS_SPACE_TEST

# Define constraints for the kernel
constraints = [
    # specifies how computation is tiled
    tkw.WorkgroupConstraint(M, BLOCK_M, 0),
    tkw.WorkgroupConstraint(N, BLOCK_N, 1),
    tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
    tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
    tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}),
]


def _get_dummy_trace_options_and_constraints() -> (
    tuple[CapturedTrace, WaveCompileOptions, list[Constraint]]
):
    options = WaveCompileOptions(
        subs={M: 128, N: 128, BLOCK_M: 64, BLOCK_N: 64},
        compile_to_mlir=True,
    )

    global constraints

    @wave.wave(constraints)
    def dummy_kernel(a: Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f32]):
        r = wave.read(a)
        wave.write(r, a)

    compiled_kernel = wave_compile(options, dummy_kernel)
    trace = compiled_kernel.get_compiled_graph()
    constraints = dummy_kernel.constraints

    return trace, options, constraints


# CHECK-LABEL: failure_to_parse_override_mlir
@run_test
def failure_to_parse_override_mlir():
    trace, options, constraints = _get_dummy_trace_options_and_constraints()

    # Override the MLIR module after `wave_compile` so it doesn't attempt to parse it.
    options.override_mlir = "module {"
    _, diagnostics, _ = emit_wave_dialect(trace, constraints, options)

    assert len(diagnostics) == 1
    # CHECK: Unable to parse module assembly
    print(diagnostics[0])


# CHECK-LABEL: failure_to_parse_pipeline
@run_test
def failure_to_parse_pipeline():
    trace, options, constraints = _get_dummy_trace_options_and_constraints()
    _, diagnostics, _ = emit_wave_dialect(
        trace, constraints, options, pipeline="module {"
    )

    assert len(diagnostics) == 1
    # CHECK: Failed to apply transform script: Unable to parse module assembly
    print(diagnostics[0])


# CHECK-LABEL: pipeline_is_empty
@run_test
def pipeline_is_empty():
    trace, options, constraints = _get_dummy_trace_options_and_constraints()
    _, diagnostics, _ = emit_wave_dialect(
        trace, constraints, options, pipeline="module {}"
    )

    assert len(diagnostics) == 1
    # CHECK: Failed to apply transform script: Transform module is empty
    print(diagnostics[0])


# CHECK-LABEL: pipeline_is_not_a_named_sequence
@run_test
def pipeline_is_not_a_named_sequence():
    trace, options, constraints = _get_dummy_trace_options_and_constraints()
    _, diagnostics, _ = emit_wave_dialect(
        trace, constraints, options, pipeline="module { module {}}"
    )

    assert len(diagnostics) == 1
    # CHECK: Failed to apply transform script: Expected first op to be "transform.named_sequence", got "builtin.module"
    print(diagnostics[0])


# This script is guaranteed to fail unless we somehow have a root op called
# "foo.bar", which is extremely unlikely.
GUARANTEED_FAIL_TRANSFORM_SCRIPT = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.cast %arg0 : !transform.any_op to !transform.op<"foo.bar">
    transform.yield
  }
}
"""


# CHECK-LABEL: failure_in_pipeline
@run_test
def failure_in_pipeline():
    trace, options, constraints = _get_dummy_trace_options_and_constraints()
    options.override_mlir = "module {}"
    _, diagnostics, _ = emit_wave_dialect(
        trace, constraints, options, pipeline=GUARANTEED_FAIL_TRANSFORM_SCRIPT
    )
    assert len(diagnostics) == 1
    # CHECK: Failed to apply transform script:
    # CHECK: incompatible payload operation name expected foo.bar vs builtin.module
    print(diagnostics[0])


# CHECK-LABEL: override_mlir
@run_test
def override_mlir():
    trace, options, constraints = _get_dummy_trace_options_and_constraints()
    options.override_mlir = """
module {
  func.func private @overridden_mlir()
}"""
    emitted, diagnostics, _ = emit_wave_dialect(trace, constraints, options)
    assert len(diagnostics) == 0, "Did not expect errors in overridden IR."

    # CHECK: func.func private @overridden_mlir()
    print(emitted)


@wave.wave(constraints)
def matrix_add(
    # defines matrix in memory of req dimension with specific data types
    a: Memory[M, N, ADDRESS_SPACE_A, tkl.f16],
    b: Memory[M, N, ADDRESS_SPACE_B, tkl.f16],
    c: Memory[M, N, ADDRESS_SPACE_C, tkl.f32],
):
    # Initialize the accumulator register with zeroes
    c_reg = Register[M, N, tkl.f16](0.0)

    # loads values from memory into registers
    a_reg = wave.read(a)
    b_reg = wave.read(b)

    # compute the sum
    c_reg = a_reg + b_reg

    c_casted = wave.cast(c_reg, tkl.f32)

    # writing results back to memory
    wave.write(c_casted, c)


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
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    # Compile the kernel to get the trace
    compiled_kernel = wave_compile(options, matrix_add)

    # Get the compiled graph from the compiled kernel
    trace = compiled_kernel.get_compiled_graph()

    constraints = matrix_add.constraints

    # Use the mlir_converter to emit wave MLIR dialect
    mlir_output, diagnostics, _ = emit_wave_dialect(trace, constraints, options)

    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
    assert (
        len(diagnostics) == 0
    ), "dialect emission should create valid IR, therefore diagnostics should be empty"

    # Print to stdout for FileCheck.
    print(mlir_output)

    # Apply Water middle-end pipeline.
    lowered_mlir = apply_water_middle_end_passes(mlir_output)
    print(lowered_mlir)

    # CHECK-LABEL: mlir_converter_matrix_add
    # CHECK: module
    # CHECK-NEXT: func.func @kernel(%[[ARG0:.*]]: !wave.tensor<[@M, @N] of f16, <global>>, %[[ARG1:.*]]: !wave.tensor<[@M, @N] of f16, <global>>, %[[ARG2:.*]]: !wave.tensor<[@M, @N] of f32, <global>>)
    # CHECK-SAME: wave.constraints =
    # CHECK-SAME: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
    # CHECK-SAME: #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
    # CHECK-SAME: #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 2)>>, #wave.wave_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N floordiv 2)>>
    # CHECK-SAME: #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 2, 1], mma_type = <f32_16x16x16_f16>, vector_shapes = {M = 64 : i64, N = 64 : i64}>
    # CHECK-SAME: BLOCK_M = 64 : i64
    # CHECK-SAME: BLOCK_N = 64 : i64
    # CHECK-SAME: M = 128 : i64
    # CHECK-SAME: N = 128 : i64

    # CHECK: %[[READ_A:.*]] = wave.read %[[ARG0]]
    # CHECK-SAME: index
    # CHECK-SAME: M : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, 1, 64)>
    # CHECK-SAME: N : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, BLOCK_N ceildiv 2, 1)>
    # CHECK-SAME: bounds
    # CHECK-SAME: #wave.read_write_bounds
    # CHECK-SAME: M = #wave.expr_list
    # CHECK-SAME: N = #wave.expr_list
    # CHECK-SAME: elements_per_thread = 32 : i64
    # CHECK-SAME: (!wave.tensor<[@M, @N] of f16, <global>>) -> !wave.tensor<[@M, @N] of f16, <register>>

    # CHECK: %[[READ_B:.*]] = wave.read %[[ARG1]]
    # CHECK-SAME: index
    # CHECK-SAME: M : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, 1, 64)>
    # CHECK-SAME: N : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, BLOCK_N ceildiv 2, 1)>
    # CHECK-SAME: bounds
    # CHECK-SAME: #wave.read_write_bounds
    # CHECK-SAME: M = #wave.expr_list
    # CHECK-SAME: N = #wave.expr_list
    # CHECK-SAME: elements_per_thread = 32 : i64
    # CHECK-SAME: (!wave.tensor<[@M, @N] of f16, <global>>) -> !wave.tensor<[@M, @N] of f16, <register>>

    # CHECK: %[[ADD:.*]] = wave.add %[[READ_A]], %[[READ_B]]
    # CHECK-SAME: index
    # CHECK-SAME: M : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, 1, 64)>
    # CHECK-SAME: N : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, BLOCK_N ceildiv 2, 1)>
    # CHECK-SAME: (!wave.tensor<[@M, @N] of f16, <register>>, !wave.tensor<[@M, @N] of f16, <register>>) -> !wave.tensor<[@M, @N] of f16, <register>>

    # CHECK: %[[CAST:.*]] = wave.cast %[[ADD]]
    # CHECK-SAME: index
    # CHECK-SAME: M : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, 1, 64)>
    # CHECK-SAME: N : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, BLOCK_N ceildiv 2, 1)>
    # CHECK-SAME: : !wave.tensor<[@M, @N] of f16, <register>> to !wave.tensor<[@M, @N] of f32, <register>>

    # CHECK: wave.write %[[CAST]], %[[ARG2]]
    # CHECK-SAME: index
    # CHECK-SAME: M : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, 1, 64)>
    # CHECK-SAME: N : <[{{.*}}, {{.*}}, {{.*}}] -> ({{.*}}, BLOCK_N ceildiv 2, 1)>
    # CHECK-SAME: bounds
    # CHECK-SAME: #wave.read_write_bounds
    # CHECK-SAME: M = #wave.expr_list
    # CHECK-SAME: N = #wave.expr_list
    # CHECK-SAME: elements_per_thread = 32 : i64
    # CHECK-SAME: !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>

    # CHECK: return

    # Water lowered output.
    # CHECK: module {
    # CHECK: func.func @kernel(
    # CHECK-NOT: wave.read
    # CHECK: vector.maskedload
    # CHECK: vector.maskedload
    # CHECK-NOT: wave.add
    # CHECK: arith.addf
    # CHECK-NOT: wave.write
    # CHECK: vector.maskedstore


@run_test
def multi_result_handling():
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        ),
    ]

    @tkw.wave(constraints)
    def multi_result_iteration(
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        d: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        init_a = tkl.Register[M, N, tkl.f32](0.0)
        init_b = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[init_a, init_b])
        def repeat(
            acc_a: tkl.Register[M, N, tkl.f32],
            acc_b: tkl.Register[M, N, tkl.f32],
        ):
            return acc_a, acc_b

        r1, r2 = repeat
        tkw.write(r2, d)
        tkw.write(r1, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            K: 32,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
        },
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, multi_result_iteration)
    trace = compiled_kernel.compiled_graph
    constraints = multi_result_iteration.constraints

    with Context(), Location.unknown():
        asm, diagnostics, _ = emit_wave_dialect(trace, constraints, options)

    assert len(diagnostics) == 0, "No diagnostics should be produced."
    print(asm)

    # CHECK-LABEL: multi_result_handling
    # CHECK: %[[ITER:.+]]:2 = wave.iterate
    # CHECK: wave.write %[[ITER]]#1
    # CHECK: wave.write %[[ITER]]#0


@run_test
def mlir_converter_sum():
    """Test MLIR converter with sum kernel."""

    permute_constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        tkw.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    @tkw.wave(permute_constraints)
    def sum(
        a: tkl.Memory[M, N, ADDRESS_SPACE_A, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE_C, tkl.f16],
    ):
        res = tkw.read(a)
        init = tkw.read(c)
        res = tkw.sum(res, init, dim=N)
        tkw.write(res, c)

    subs = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        M: 128,
        N: 128,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,  # Avoid IREE compilation
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, sum)

    # Get the trace from the compiled kernel
    trace = compiled_kernel.compiled_graph

    constraints = sum.constraints

    # Use the mlir_converter to emit wave MLIR dialect
    mlir_output, diagnostics, _ = emit_wave_dialect(trace, constraints, options)

    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
    assert (
        len(diagnostics) == 0
    ), "dialect emission should create valid IR, therefore diagnostics should be empty"

    # Print to stdout for FileCheck.
    print(mlir_output)

    # CHECK-LABEL: mlir_converter_sum
    # CHECK: wave.read
    # CHECK: wave.read
    # CHECK: wave.extract
    # CHECK: wave.shuffle
    # CHECK: wave.add
    # CHECK: wave.shuffle
    # CHECK: wave.add
    # CHECK: wave.shuffle
    # CHECK: wave.add
    # CHECK: wave.shuffle
    # CHECK: wave.add
    # CHECK: wave.shuffle
    # CHECK: wave.add
    # CHECK: wave.shuffle
    # CHECK: wave.add
    # CHECK: wave.add
    # CHECK: wave.write


@run_test
def mlir_converter_emit_wave_dialect_loop_implicit_capture():
    """Test emit_wave_dialect with a kernel that has a loop using registers defined in the
    kernel as implicit captures.
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints_loop: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=MMAType.F32_32x32x8_F16,
            vector_shapes={M: BLOCK_M, N: BLOCK_N, K: BLOCK_K},
        ),
    ]

    @tkw.wave(constraints_loop)
    def kernel_loop_implicit_capture(
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        # Registers defined in the kernel body (not function args)
        acc = tkl.Register[M, N, tkl.f32](0.0)
        bias = tkl.Register[M, N, tkl.f32](0.0)  # kernel-defined, captured by loop

        # Loop body implicitly captures bias (kernel-defined register); acc is iter arg
        @tkw.iterate(K, init_args=[acc])
        def k_loop(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            acc = acc + bias
            return acc

        tkw.write(k_loop, c)

    subs_loop = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 16,
        M: 64,
        N: 64,
        K: 32,
    }
    options_loop = WaveCompileOptions(
        subs=subs_loop,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options_loop = set_default_run_config(options_loop)

    compiled = wave_compile(options_loop, kernel_loop_implicit_capture)
    trace = compiled.compiled_graph

    mlir_output, diagnostics, _ = emit_wave_dialect(
        trace, kernel_loop_implicit_capture.constraints, options_loop
    )

    if diagnostics:
        for d in diagnostics:
            print(d, file=sys.stderr)
    assert (
        len(diagnostics) == 0
    ), "emit_wave_dialect should succeed for loop with implicit captures"

    print(mlir_output)
    # CHECK-LABEL: mlir_converter_emit_wave_dialect_loop_implicit_capture
    # CHECK: func.func
    # CHECK: %[[ACC:.+]] = wave.register
    # CHECK: %[[BIAS:.+]] = wave.register
    # CHECK: wave.iterate @K iter_args(%[[ACC]])
    # CHECK: ^{{.*}}(%[[INNER:.+]]: !wave.tensor
    # CHECK:   wave.add %[[INNER]], %[[BIAS]]


@run_test
def mlir_converter_matmul():
    """Test MLIR converter with matmul kernel."""

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
        BLOCK_K: 32,
        M: 1024,
        N: 5120,
        K: 640,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,  # Avoid IREE compilation
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        minimize_shared_allocs=True,  # Test allocate with parent/offset.
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, matmul)

    # Get the trace from the compiled kernel
    trace = compiled_kernel.compiled_graph

    constraints = matmul.constraints

    # Check that providing a custom transform script to water does not
    # throw errors
    with Context(), Location.unknown():
        transform_module = Module.create()
        transform_module_op = transform_module.operation
        transform_module_op.attributes["transform.with_named_sequence"] = UnitAttr.get()

        def pipeline(root: OpHandle):
            pass

        insert_transform_script(transform_module.body, pipeline)
        pipeline_asm = str(transform_module)

    # Use the mlir_converter to emit wave MLIR dialect and apply the empty
    # pipeline.
    mlir_output, diagnostics, _ = emit_wave_dialect(
        trace, constraints, options, pipeline=pipeline_asm
    )

    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
    assert (
        len(diagnostics) == 0
    ), "dialect emission should create valid IR, therefore diagnostics should be empty"

    # Print to stdout for FileCheck.
    # CHECK-LABEL: mlir_converter_matmul
    print(pipeline_asm)
    # CHECK: module
    # CHECK-NEXT: transform.named_sequence @__transform_main
    # CHECK-NEXT:   transform.yield

    print(mlir_output)
    # CHECK: module
    # CHECK-NEXT: func.func @kernel(
    # CHECK-SAME: %[[ARG0:.*]]: !wave.tensor<[@M, @K] of f16, <global>>
    # CHECK-SAME: %[[ARG1:.*]]: !wave.tensor<[@N, @K] of f16, <global>>
    # CHECK-SAME: %[[ARG2:.*]]: !wave.tensor<[@M, @N] of f32, <global>>
    # CHECK-SAME: wave.constraints =
    # CHECK-SAME: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
    # CHECK-SAME: #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
    # CHECK-SAME: #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    # CHECK-SAME: #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 2)>>
    # CHECK-SAME: #wave.wave_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N floordiv 2)>>
    # CHECK-SAME: #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 2, 1], mma_type = <f32_32x32x8_f16>>
    # CHECK-SAME: #wave.hyperparameters<{BLOCK_K = 32 : i64, BLOCK_M = 64 : i64, BLOCK_N = 64 : i64, K = 640 : i64, M = 1024 : i64, N = 5120 : i64}>
    #
    # With minimize_shared_allocs enabled, the first allocate is the parent (combined buffer)
    # and subsequent allocates reference it with "in %parent" and an offset attribute.
    #
    # CHECK-NEXT: %[[ALLOCATE:.*]] = wave.allocate
    # CHECK-SAME: distributed_shape
    # CHECK-SAME: : !wave.tensor<{{.*}} of i8, <shared>>
    # CHECK-NEXT: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
    # CHECK-NEXT: %[[REG:.*]] = wave.register %[[CST_0]]
    #
    # Child allocation with parent reference and offset.
    # Symbols related to induction variables must be dropped from the mapping. This is a bug in the
    # Python propagation algorithm that is immediately caught by the verifier on construction.
    #
    # CHECK-NEXT: %[[ALLOCATE_1:.*]] = wave.allocate in %[[ALLOCATE]] : !wave.tensor<{{.*}} of i8, <shared>>
    # CHECK-SAME: distributed_shape
    # CHECK-SAME: index =
    # CHECK-SAME: K = #wave.index_mapping<
    # CHECK-NOT:  ARGK
    # CHECK-SAME: offset =
    #
    # Another child allocation with parent reference and offset.
    #
    # CHECK-NEXT: %[[ALLOCATE_2:.*]] = wave.allocate in %[[ALLOCATE]] : !wave.tensor<{{.*}} of i8, <shared>>
    # CHECK-SAME: distributed_shape
    # CHECK-SAME: index =
    # CHECK-SAME: offset =
    # CHECK-NEXT: %[[ITERATE:.*]] = wave.iterate @K iter_args(%[[REG]]) {
    # CHECK-NEXT:   ^{{.*}}(%[[ARG3:.*]]: !wave.tensor<[@M, @N] of f32, <register>>):
    # CHECK-NEXT:     %[[READ_A:.*]] = wave.read %[[ARG0]]
    # CHECK-NEXT:     amdgpu.lds_barrier
    # CHECK-NEXT:     wave.write %[[READ_A]], %[[ALLOCATE_2]]
    # CHECK-NEXT:     %[[READ_B:.*]] = wave.read %[[ARG1]]
    # CHECK-NEXT:     wave.write %[[READ_B]], %[[ALLOCATE_1]]
    # CHECK-NEXT:     amdgpu.lds_barrier
    # CHECK-NEXT:     %[[READ_SHARED_A_0:.*]] = wave.read %[[ALLOCATE_1]]
    # CHECK-NEXT:     %[[READ_SHARED_A_1:.*]] = wave.read %[[ALLOCATE_1]]
    # CHECK-NEXT:     %[[READ_SHARED_A_2:.*]] = wave.read %[[ALLOCATE_1]]
    # CHECK-NEXT:     %[[READ_SHARED_A_3:.*]] = wave.read %[[ALLOCATE_1]]
    # CHECK-NEXT:     %[[READ_SHARED_B_0:.*]] = wave.read %[[ALLOCATE_2]]
    # CHECK-NEXT:     %[[READ_SHARED_B_1:.*]] = wave.read %[[ALLOCATE_2]]
    # CHECK-NEXT:     %[[READ_SHARED_B_2:.*]] = wave.read %[[ALLOCATE_2]]
    # CHECK-NEXT:     %[[READ_SHARED_B_3:.*]] = wave.read %[[ALLOCATE_2]]
    # CHECK-NEXT:     %[[MMA_0:.*]] = wave.mma %[[READ_SHARED_B_0]], %[[READ_SHARED_A_0]], %[[ARG3]]
    # CHECK-COUNT-2:  {K : <[
    # CHECK-SAME:     {M : <[
    # CHECK-SAME:     #wave.mma_kind<f32_32x32x8_f16>
    # CHECK-NEXT:     %[[MMA_1:.*]] = wave.mma %[[READ_SHARED_B_1]], %[[READ_SHARED_A_1]], %[[MMA_0]]
    # CHECK-COUNT-2:  {K : <[
    # CHECK-SAME:     {M : <[
    # CHECK-SAME:     #wave.mma_kind<f32_32x32x8_f16>
    # CHECK-NEXT:     %[[MMA_2:.*]] = wave.mma %[[READ_SHARED_B_2]], %[[READ_SHARED_A_2]], %[[MMA_1]]
    # CHECK-COUNT-2:  {K : <[
    # CHECK-SAME:     {M : <[
    # CHECK-SAME:     #wave.mma_kind<f32_32x32x8_f16>
    # CHECK-NEXT:     %[[MMA_3:.*]] = wave.mma %[[READ_SHARED_B_3]], %[[READ_SHARED_A_3]], %[[MMA_2]]
    # CHECK-COUNT-2:  {K : <[
    # CHECK-SAME:     {M : <[
    # CHECK-SAME:     #wave.mma_kind<f32_32x32x8_f16>
    # CHECK-NEXT:     wave.yield %[[MMA_3]] : !wave.tensor<[@M, @N] of f32, <register>>
    # CHECK-NEXT: }
    # CHECK-NEXT: %[[SLICE_0:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (0)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_0]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_1:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (1)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_1]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_2:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (2)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_2]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_3:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (3)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_3]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_4:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (4)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_4]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_5:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (5)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_5]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_6:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (6)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_6]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_7:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (7)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_7]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_8:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (8)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_8]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_9:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (9)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_9]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_10:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (10)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_10]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_11:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (11)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_11]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_12:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (12)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_12]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_13:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (13)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_13]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_14:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (14)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_14]], %[[ARG2]]
    # CHECK-NEXT: %[[SLICE_15:.*]] = wave.extract_slice %[[ITERATE]]{{.*}} offset = #wave.expr_list<[] -> (15)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>
    # CHECK-NEXT: wave.write %[[SLICE_15]], %[[ARG2]]
    # CHECK-NEXT: return

    # Apply Water middle-end pipeline.
    lowered_mlir = apply_water_middle_end_passes(mlir_output)
    print(lowered_mlir)

    # Water lowered output.
    # CHECK: module {
    # CHECK: func.func @kernel(
    # CHECK: memref.alloc() : memref<9216xi8, #gpu.address_space<workgroup>>
    # CHECK: memref.view
    # CHECK-NOT: wave.iterate
    # CHECK: scf.for
    # CHECK-NOT: wave.mma
    # CHECK: amdgpu.mfma 32x32x8


@run_test
def mlir_converter_mixed_memory_spaces():
    global constraints

    @tkw.wave(constraints)
    def mixed_memory_kernel(
        global_direct: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        symbolic_a: tkl.Memory[M, N, ADDRESS_SPACE_A, tkl.f16],
        symbolic_b: tkl.Memory[M, N, ADDRESS_SPACE_B, tkl.f16],
    ):
        read_1 = wave.read(symbolic_a)
        read_2 = wave.read(symbolic_b)

        result = tkl.Register[M, N, tkl.f16](42.0)
        result = read_1 + read_2
        wave.write(result, global_direct)

    subs = {
        # Mix of symbolic address spaces resolving to different spaces
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,  # Symbolic -> Global
        ADDRESS_SPACE_B: SHARED_ADDRESS_SPACE,  # Symbolic -> Global (after `promote_placeholders`)
        M: 128,
        N: 128,
        BLOCK_M: 64,
        BLOCK_N: 64,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, mixed_memory_kernel)
    trace = compiled_kernel.compiled_graph
    constraints = mixed_memory_kernel.constraints

    with Context(), Location.unknown():
        mlir_output, diagnostics, _ = emit_wave_dialect(trace, constraints, options)

    assert len(diagnostics) == 0, f"Should have no diagnostics, got: {diagnostics}"

    # CHECK-LABEL: mlir_converter_mixed_memory_spaces
    print(mlir_output)
    # All function arguments should be <global>
    # CHECK: func.func @kernel(
    # Verify that ADDRESS_SPACE_* are NOT in hyperparameters but others are
    # CHECK-SAME: #wave.hyperparameters<{BLOCK_M = 64 : i64, BLOCK_N = 64 : i64, M = 128 : i64, N = 128 : i64}>
    # CHECK-NOT: ADDRESS_SPACE_A
    # CHECK-NOT: ADDRESS_SPACE_B


@run_test
def mlir_converter_invalid_non_int_hyperparameter():
    global constraints

    @tkw.wave(constraints)
    def invalid_hyperparameter_kernel(
        out: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        arg: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        read = wave.read(arg)

        result = tkl.Register[M, N, tkl.f16](42.0)
        result = read + result
        wave.write(result, out)

    # Create an invalid non-int hyperparameter (not an address space)
    INVALID_SYMBOL = tkl.sym.INVALID_SYMBOL
    subs = {
        INVALID_SYMBOL: "invalid_string_value",  # This should trigger validation error
        M: 128,
        N: 128,
        BLOCK_M: 64,
        BLOCK_N: 64,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, invalid_hyperparameter_kernel)
    trace = compiled_kernel.compiled_graph
    constraints = invalid_hyperparameter_kernel.constraints

    # This should raise a RuntimeError due to invalid non-int hyperparameter
    try:
        with Context(), Location.unknown():
            mlir_output, diagnostics, _ = emit_wave_dialect(trace, constraints, options)
        assert False, "Expected RuntimeError for invalid non-int hyperparameter"
    except RuntimeError as e:
        # Verify the error message is what we expect
        assert "Unexpected non-int mapping in hyperparameters" in str(e)
        assert "INVALID_SYMBOL -> invalid_string_value" in str(e)
        assert "Expected all non-int values to be address spaces" in str(e)


@run_test
def mlir_converter_permute():
    """Test MLIR converter with permute operation."""

    # Define constraints for the kernel
    permute_constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        tkw.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    @wave.wave(permute_constraints)
    def permute_kernel(
        a: Memory[M, N, ADDRESS_SPACE_A, tkl.f16],
        c: Memory[N, M, ADDRESS_SPACE_C, tkl.f16],
    ):
        # Load values from memory into registers
        a_reg = wave.read(a, elements_per_thread=1)

        # Permute dimensions from [M, N] to [N, M]
        permuted = wave.permute(a_reg, target_shape=[N, M])

        wave.write(permuted, c, elements_per_thread=1)

    # Set parameters for compilation
    subs = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        M: 128,
        N: 128,
    }

    # Compile the kernel to get the trace
    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, permute_kernel)
    trace = compiled_kernel.get_compiled_graph()
    kernel_constraints = permute_kernel.constraints

    # Use the mlir_converter to emit wave MLIR dialect
    mlir_output, diagnostics, _ = emit_wave_dialect(trace, kernel_constraints, options)

    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
    assert (
        len(diagnostics) == 0
    ), "dialect emission should create valid IR, therefore diagnostics should be empty"

    # Print to stdout for FileCheck
    print(mlir_output)

    # CHECK-LABEL: mlir_converter_permute
    # CHECK: func.func @kernel(%[[ARG0:.*]]: !wave.tensor<[@M, @N] of f16, <global>>, %[[ARG1:.*]]: !wave.tensor<[@N, @M] of f16, <global>>)

    # CHECK: %[[READ:.*]] = wave.read %[[ARG0]]
    # CHECK-SAME: (!wave.tensor<[@M, @N] of f16, <global>>) -> !wave.tensor<[@M, @N] of f16, <register>>

    # CHECK: %[[PERMUTE:.*]] = wave.permute %[[READ]]
    # CHECK-SAME: !wave.tensor<[@M, @N] of f16, <register>> to !wave.tensor<[@N, @M] of f16, <register>>

    # CHECK: wave.write %[[PERMUTE]], %[[ARG1]]
    # CHECK-SAME: !wave.tensor<[@N, @M] of f16, <register>>, !wave.tensor<[@N, @M] of f16, <global>>
