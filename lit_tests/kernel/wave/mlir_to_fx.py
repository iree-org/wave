# REQUIRES: water
# RUN: python %s | FileCheck %s

import sympy

from wave_lang.kernel._support.tracing import CapturedTrace
import wave_lang.kernel.wave as wave
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import (
    Constraint,
    MMAType,
    HardwareConstraint,
)
from wave_lang.kernel.wave.mlir_converter.diagnostics import MLIRDiagnostic
from wave_lang.kernel.wave.mlir_converter.mlir_converter import (
    emit_wave_dialect,
    mlir_to_fx,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.graph_utils import (
    assert_traces_equivalent,
    assert_constraints_equivalent,
)
from wave_lang.kernel.ops.wave_ops import get_custom, Placeholder


def _error_diagnostics(diags: list[MLIRDiagnostic]) -> list[MLIRDiagnostic]:
    """Filter structured diagnostics to errors only."""
    return [d for d in diags if "error" in d.severity.lower()]


M = tkl.sym.M
N = tkl.sym.N
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N


def _check_hyperparameters_roundtrip(
    source_subs: dict,
    roundtripped_subs: dict,
    core_params: set | None = None,
) -> None:
    """
    Check that hyperparameters match after MLIR roundtrip.

    Args:
        source_subs: Original subs dictionary from WaveCompileOptions
        roundtripped_subs: Subs dictionary after MLIR roundtrip
        core_params: Optional set of core parameters to check. If None, checks all.
    """
    params_to_check = core_params if core_params is not None else source_subs.keys()
    for param in params_to_check:
        assert source_subs.get(param) == roundtripped_subs.get(
            param
        ), f"Hyperparameter {param} mismatch: {source_subs.get(param)} vs {roundtripped_subs.get(param)}"


def _compare_hardware_constraints_for_mlir_roundtrip(
    source: HardwareConstraint, roundtripped: HardwareConstraint
) -> bool:
    """
    Compare HardwareConstraints for MLIR roundtrip testing.

    The MLIR representation intentionally excludes certain Python-specific configuration
    fields (workgroups_per_cluster, n_service_waves) that represent scheduling decisions
    and runtime configuration rather than fundamental hardware constraints. This comparator
    checks only the fields that are serialized to MLIR.

    Args:
        source: Source constraint (from Python, before MLIR roundtrip)
        roundtripped: Constraint after MLIR roundtrip

    Returns:
        True if constraints are equivalent for MLIR roundtrip purposes
    """
    # Compare fields that are serialized to MLIR
    if source.threads_per_wave != roundtripped.threads_per_wave:
        return False
    if source.waves_per_block != roundtripped.waves_per_block:
        return False
    if source.mma_type != roundtripped.mma_type:
        return False
    if source.max_bits_per_load != roundtripped.max_bits_per_load:
        return False

    # vector_shapes may not be present in the source trace if the set_node_indices pass
    # (which populates vector_shapes on nodes from hardware constraints) hasn't run yet.
    # On the MLIR side, vector_shapes are always inferred from the HardwareConstraint
    # during conversion to fx, so roundtripped traces will always have them populated.
    if (
        source.vector_shapes is not None
        and source.vector_shapes != roundtripped.vector_shapes
    ):
        return False

    # workgroups_per_cluster and n_service_waves are intentionally NOT compared
    # as they are not part of the MLIR representation

    return True


def _get_read_write_trace() -> (
    tuple[CapturedTrace, WaveCompileOptions, list[Constraint]]
):
    options = WaveCompileOptions(
        subs={M: 128, N: 128, BLOCK_M: 64, BLOCK_N: 64},
        compile_to_mlir=True,
    )

    # Define constraints for the kernel
    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    @wave.wave(constraints)
    def dummy_kernel(a: Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f32]):
        r = wave.read(a)
        wave.write(r, a)

    compiled_kernel = wave_compile(options, dummy_kernel)
    trace = compiled_kernel.get_compiled_graph()
    constraints = dummy_kernel.constraints

    return trace, options, constraints


# CHECK-LABEL: mlir_to_fx_minimal_roundtrip
@run_test
def mlir_to_fx_minimal_roundtrip():
    """Test MLIR roundtrip for fully-compiled trace."""
    # Get the fully compiled trace
    trace, options, test_constraints = _get_read_write_trace()

    # Emit MLIR from the traced kernel.
    mlir_text, diagnostics, _ = emit_wave_dialect(trace, test_constraints, options)
    errors = _error_diagnostics(diagnostics)
    assert errors == [], f"unexpected errors from wave to mlir conversion: {errors}"

    # Convert back to FX trace
    fx_trace, fx_constraints, fx_options, fx_diags = mlir_to_fx(mlir_text)
    errors = _error_diagnostics(fx_diags)
    assert errors == [], f"unexpected errors from mlir to fx conversion: {errors}"

    _check_hyperparameters_roundtrip(options.subs, fx_options.subs)
    assert_constraints_equivalent(
        test_constraints,
        fx_constraints,
        custom_comparators={
            HardwareConstraint: _compare_hardware_constraints_for_mlir_roundtrip
        },
    )
    assert_traces_equivalent(trace, fx_trace, subs=options.subs)

    # CHECK: OK: minimal roundtrip
    print("OK: minimal roundtrip")


# CHECK-LABEL: mlir_to_fx_simple_matmul_roundtrip
@run_test
def mlir_to_fx_simple_matmul_roundtrip():
    """Test MLIR roundtrip for fully-compiled matmul trace."""
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K

    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.TilingConstraint(K, BLOCK_K),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(threads_per_wave=64, mma_type=MMAType.F32_16x16x16_F16),
    ]

    @wave.wave(constraints)
    def matmul_simple(
        a: tkl.Memory[M, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @wave.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = wave.read(a, bounds={M: M, K: K})
            b_reg = wave.read(b, bounds={N: N, K: K})
            acc = wave.mma(a_reg, b_reg, acc)
            return acc

        wave.write(repeat, c)

    subs = {
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        M: 128,
        N: 128,
        K: 16,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
    )

    compiled_kernel = wave_compile(options, matmul_simple)
    trace = compiled_kernel.get_compiled_graph()
    # Get constraints from the compiled kernel
    source_constraints = matmul_simple.constraints

    # Emit MLIR from the traced kernel.
    mlir_text, diagnostics, _ = emit_wave_dialect(trace, source_constraints, options)
    errors = _error_diagnostics(diagnostics)
    assert errors == [], f"unexpected errors from wave to mlir conversion: {errors}"

    # Convert back to FX trace
    fx_trace, fx_constraints, fx_options, fx_diags = mlir_to_fx(mlir_text)
    errors = _error_diagnostics(fx_diags)
    assert errors == [], f"unexpected errors from mlir to fx conversion: {errors}"

    _check_hyperparameters_roundtrip(options.subs, fx_options.subs)
    assert_constraints_equivalent(
        source_constraints,
        fx_constraints,
        custom_comparators={
            HardwareConstraint: _compare_hardware_constraints_for_mlir_roundtrip
        },
    )
    assert_traces_equivalent(trace, fx_trace, subs=options.subs)

    # CHECK: OK: matmul roundtrip
    print("OK: matmul roundtrip")


# CHECK-LABEL: mlir_to_fx_pipelined_gemm_roundtrip
@run_test
def mlir_to_fx_pipelined_gemm_roundtrip():
    """Test MLIR→FX roundtrip for pipelined GEMM with software pipelining."""
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.TilingConstraint(K, BLOCK_K),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(
            threads_per_wave=64,
            mma_type=MMAType.F32_16x16x16_F16,
        ),
    ]

    @wave.wave(constraints)
    def gemm_pipelined(
        a: tkl.Memory[M, K, ADDRESS_SPACE_0, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE_0, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @wave.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = wave.read(a)
            b_reg = wave.read(b)
            acc = wave.mma(a_reg, b_reg, acc)
            return acc

        wave.write(repeat, c)

    subs = {
        M: 64,
        N: 64,
        K: 128,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 32,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_0: SHARED_ADDRESS_SPACE,
    }
    subs.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
        schedule=SchedulingType.PREFETCH,  # Enable software pipelining
    )

    compiled_kernel = wave_compile(options, gemm_pipelined)
    trace = compiled_kernel.get_compiled_graph()
    source_constraints = gemm_pipelined.constraints

    # Emit MLIR from the traced kernel.
    mlir_text, diagnostics, _ = emit_wave_dialect(trace, source_constraints, options)
    errors = _error_diagnostics(diagnostics)
    assert errors == [], f"unexpected errors from wave to mlir conversion: {errors}"

    # Convert back to FX trace
    fx_trace, fx_constraints, fx_options, fx_diags = mlir_to_fx(mlir_text)
    errors = _error_diagnostics(fx_diags)
    assert errors == [], f"unexpected errors from mlir to fx conversion: {errors}"

    # Check roundtrip worked
    # Note: options.subs includes scheduling parameters and address space symbols
    # that are not serialized to MLIR hyperparameters. Only check core numeric parameters.
    core_params = {M, N, K, BLOCK_M, BLOCK_N, BLOCK_K}
    _check_hyperparameters_roundtrip(options.subs, fx_options.subs, core_params)
    assert_constraints_equivalent(
        source_constraints,
        fx_constraints,
        custom_comparators={
            HardwareConstraint: _compare_hardware_constraints_for_mlir_roundtrip
        },
    )
    assert_traces_equivalent(trace, fx_trace, subs=options.subs)

    # CHECK: OK: pipelined gemm roundtrip
    print("OK: pipelined gemm roundtrip")


# CHECK-LABEL: mlir_to_fx_unspecified_address_space
@run_test
def mlir_to_fx_unspecified_address_space():
    """Test that Unspecified address spaces are converted to unique Memory symbols.

    Wave infers concrete address spaces during compilation, so MLIR inputs
    earlier compilation stages may still carry `#wave.address_space<unspecified>`.
    The converter must handle these gracefully, assigning a fresh unique symbol to
    each occurrence so that distinct unresolved spaces are never accidentally conflated.
    """
    # Start from the matmul kernel (which has multiple Memory arguments)
    # and emit its MLIR.
    K = tkl.sym.K
    BLOCK_K = tkl.sym.BLOCK_K

    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.TilingConstraint(K, BLOCK_K),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(threads_per_wave=64, mma_type=MMAType.F32_16x16x16_F16),
    ]

    @wave.wave(constraints)
    def matmul_for_addr_test(
        a: tkl.Memory[M, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @wave.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = wave.read(a, bounds={M: M, K: K})
            b_reg = wave.read(b, bounds={N: N, K: K})
            acc = wave.mma(a_reg, b_reg, acc)
            return acc

        wave.write(repeat, c)

    subs = {M: 128, N: 128, K: 16, BLOCK_M: 16, BLOCK_N: 16, BLOCK_K: 16}
    options = WaveCompileOptions(subs=subs, compile_to_mlir=True)

    compiled_kernel = wave_compile(options, matmul_for_addr_test)
    trace = compiled_kernel.get_compiled_graph()
    mlir_text, diagnostics, _ = emit_wave_dialect(
        trace, matmul_for_addr_test.constraints, options
    )
    errors = _error_diagnostics(diagnostics)
    assert errors == [], f"unexpected errors from wave to mlir conversion: {errors}"

    # Replace all concrete address spaces with unspecified to simulate
    # an MLIR module where address spaces have not been resolved yet.
    # It would be better to do this using proper MLIR manipulation APIs,
    # but we never have the module here, only the text.
    mlir_unspecified = mlir_text.replace("<global>", "<unspecified>").replace(
        "<shared>", "<unspecified>"
    )

    fx_trace, _, _, fx_diags = mlir_to_fx(mlir_unspecified)
    errors = _error_diagnostics(fx_diags)
    assert errors == [], f"unexpected errors: {errors}"

    # Collect address space symbols from Memory-typed placeholders (each
    # corresponds to a distinct function argument).
    placeholder_addrs = [
        node.type.address_space
        for node in fx_trace.walk(lambda n: n)
        if isinstance(get_custom(node), Placeholder)
        and node.type is not None
        and issubclass(node.type, Memory)
    ]

    assert (
        len(placeholder_addrs) >= 2
    ), f"expected at least two Memory placeholders, got {len(placeholder_addrs)}"
    for addr in placeholder_addrs:
        assert str(addr).startswith(
            "$UNSPECIFIED_ADDRESS_SPACE_"
        ), f"Expected $UNSPECIFIED_ADDRESS_SPACE_* symbol, got {addr}"

    # Each function argument must receive its own unique symbol so that two
    # independently unresolved address spaces are never conflated.
    assert len(placeholder_addrs) == len(set(placeholder_addrs)), (
        f"Unspecified address space symbols must be unique across arguments, "
        f"got: {placeholder_addrs}"
    )

    # CHECK: OK: unspecified address space
    print("OK: unspecified address space")
