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
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.mlir_converter.mlir_converter import (
    emit_wave_dialect,
    format_diagnostics,
)
from wave_lang.kernel.wave.mlir_converter.diagnostics import (
    FileLocation,
    NameLocation,
    MLIRDiagnostic,
    WaterDiagTestingMode,
    WaterError,
)
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


# Common substitutions for all tests
SUBS: dict[str | IndexSymbol, Any] = {
    ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
    ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
    ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
    BLOCK_M: 64,
    BLOCK_N: 64,
    M: 128,
    N: 128,
}


def compile_and_emit_diagnostics(
    location_level: LocationCaptureLevel, test_diagnostic_emission: WaterDiagTestingMode
) -> list[MLIRDiagnostic | WaterError]:
    """Helper to compile kernel and emit diagnostics with given location level.

    Args:
        location_level: The LocationCaptureLevel to use for capturing locations.
        test_diagnostic_emission: Kind of diagnostics to emit to test the
            mechanics of diagnostic emission.

    Returns:
        List of MLIRDiagnostic or WaterError objects.
    """
    # When location capture is disabled, we must also disable location enforcement
    enforce_locations = location_level != LocationCaptureLevel.NONE

    options = WaveCompileOptions(
        subs=SUBS,
        compile_to_mlir=True,
        drop_debug_info_before_mlir=False,
        location_capture_config=LocationCaptureConfig(level=location_level),
        enforce_locations=enforce_locations,
        use_local_scope=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, matrix_add)
    trace = compiled_kernel.get_compiled_graph()

    _, diagnostics, _ = emit_wave_dialect(
        trace,
        matrix_add.constraints,
        options,
        test_diagnostic_emission=test_diagnostic_emission,
    )
    return diagnostics


@run_test
def test_location_capture_none():
    """Test with LocationCaptureLevel.NONE - compilation without location capture.

    With NONE, locations are not captured so the test diagnostic should have an
    unknown location.
    """
    diagnostics = compile_and_emit_diagnostics(
        LocationCaptureLevel.NONE, test_diagnostic_emission=WaterDiagTestingMode.DIRECT
    )

    assert len(diagnostics) > 0, "Expected at least one diagnostic"
    diag = diagnostics[0]

    print(format_diagnostics(diagnostics, use_color=False))

    # Verify structured data — location should be unknown with NONE
    location = diag.location
    print(f"location frame count: {len(location)}")

    print(f"diagnostics count: {len(diagnostics)}")
    print("location capture none: compilation succeeded")

    # Location cannot be known when it isn't captured.
    # CHECK-LABEL: test_location_capture_none
    # CHECK: ERROR: test error
    # CHECK: Traceback (Wave DSL source):
    # CHECK:   <unknown location>
    # CHECK: location frame count: 1
    # CHECK: diagnostics count: 1
    # CHECK: location capture none: compilation succeeded


@run_test
def test_location_capture_file_line_col():
    """Test diagnostics with LocationCaptureLevel.FILE_LINE_COL - single location."""
    diagnostics = compile_and_emit_diagnostics(
        LocationCaptureLevel.FILE_LINE_COL,
        test_diagnostic_emission=WaterDiagTestingMode.DIRECT,
    )

    assert len(diagnostics) > 0, "Expected at least one diagnostic"
    diag = diagnostics[0]

    print(format_diagnostics(diagnostics, use_color=False))

    # Verify structured data
    location = diag.location
    print(f"location frame count: {len(location)}")
    if location:
        frame = location[0]
        if isinstance(frame, FileLocation):
            frame_type = "file"
        elif isinstance(frame, NameLocation):
            frame_type = "name"
        else:
            frame_type = "unknown"
        print(f"first frame type: {frame_type}")
        print(f"has filename: {hasattr(frame, 'filename')}")
        print(f"has start_line: {hasattr(frame, 'start_line')}")

    # CHECK-LABEL: test_location_capture_file_line_col
    # CHECK: ERROR: test error
    # CHECK: Traceback (Wave DSL source):
    # CHECK:   File "{{.*}}mlir_converter_diagnostics.py", line 62
    # CHECK:     a_reg = wave.read(a)
    # CHECK: location frame count: 1
    # CHECK: first frame type: file
    # CHECK: has filename: True
    # CHECK: has start_line: True


@run_test
def test_location_capture_stack_trace():
    """Test diagnostics with LocationCaptureLevel.STACK_TRACE - multiple frames."""
    diagnostics = compile_and_emit_diagnostics(
        LocationCaptureLevel.STACK_TRACE,
        test_diagnostic_emission=WaterDiagTestingMode.DIRECT,
    )

    assert len(diagnostics) > 0, "Expected at least one diagnostic"

    print(format_diagnostics(diagnostics, use_color=False))

    # Verify we have location frames
    diag = diagnostics[0]
    location = diag.location
    assert len(location) > 1, "Expected a stack trace."

    # CHECK-LABEL: test_location_capture_stack_trace
    # CHECK: ERROR: test error
    # CHECK: Traceback (Wave DSL source):
    # CHECK:   File "{{.*}}mlir_converter_diagnostics.py"
    # CHECK:     @run_test
    # CHECK:   File "{{.*}}mlir_converter_diagnostics.py"
    # CHECK:     diagnostics = compile_and_emit_diagnostics
    # CHECK:   File "{{.*}}mlir_converter_diagnostics.py"
    # CHECK:     compiled_kernel = wave_compile(options, matrix_add)
    # CHECK:   File "{{.*}}mlir_converter_diagnostics.py", line 62
    # CHECK:     a_reg = wave.read


@run_test
def test_location_capture_stack_trace_with_system():
    """Test diagnostics with LocationCaptureLevel.STACK_TRACE_WITH_SYSTEM."""
    diagnostics = compile_and_emit_diagnostics(
        LocationCaptureLevel.STACK_TRACE_WITH_SYSTEM,
        test_diagnostic_emission=WaterDiagTestingMode.DIRECT,
    )

    assert len(diagnostics) > 0, "Expected at least one diagnostic"

    print(format_diagnostics(diagnostics, use_color=False))

    # Verify we have location frames
    diag = diagnostics[0]
    location = diag.location
    assert len(location) > 0, "Expected a stack trace."

    # When system frames are preserved, we should see them in the stack trace,
    # in particular check for the `capture_location` function that should
    # normally be there when locations are available.

    # CHECK-LABEL: test_location_capture_stack_trace_with_system
    # CHECK: ERROR: test error
    # CHECK: Traceback (Wave DSL source):
    # CHECK:   File "{{.*}}mlir_converter_diagnostics.py", line 62
    # CHECK:   File "{{.*}}wave_lang/kernel/ops/wave_ops.py"
    # CHECK:     capture_location


@run_test
def test_verification_failure():
    diagnostics = compile_and_emit_diagnostics(
        LocationCaptureLevel.NONE,
        test_diagnostic_emission=WaterDiagTestingMode.VERIFIER,
    )
    assert len(diagnostics) > 0, "Expected at least one diagnostic"
    # CHECK-LABEL: test_verification_failure
    # CHECK: ERROR: redefinition of symbol named 'repeated_name'
    # CHECK: Traceback (Wave DSL source):
    # CHECK:   <unknown location>
    print(format_diagnostics(diagnostics, use_color=False))
