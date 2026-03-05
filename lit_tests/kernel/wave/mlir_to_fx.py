# REQUIRES: water
# RUN: python %s | FileCheck %s

import atexit
import sympy

import wave_lang.kernel.wave as wave
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import (
    MMAType,
    HardwareConstraint,
)
from wave_lang.kernel.wave.mlir_converter.diagnostics import error_diagnostics
from wave_lang.kernel.wave.mlir_converter.mlir_converter import PersistentEmitter
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.graph_utils import (
    assert_traces_equivalent,
    assert_constraints_equivalent,
    compare_hardware_constraints_for_mlir_roundtrip,
)
from wave_lang.kernel.ops.wave_ops import get_custom, Placeholder

# Keep emitter subprocesses alive for the entire test file instead of
# spawning fresh ones per call (~2s import overhead each).
# Lit runs each test file as its own `python %s` process (see the RUN line),
# so the emitter lives only for this file. atexit closes the subprocesses
# cleanly when the interpreter exits at the end of the run.
emitter = PersistentEmitter()
atexit.register(emitter.close)

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


def _assert_roundtrip(kernel, subs: dict, label: str) -> None:
    """Compile a kernel, roundtrip through MLIR, assert equivalence."""
    options = WaveCompileOptions(subs=subs, compile_to_mlir=True)
    compiled_kernel = wave_compile(options, kernel)
    trace = compiled_kernel.get_compiled_graph()
    source_constraints = kernel.constraints

    mlir_text, diagnostics, _ = emitter.emit_wave_dialect(
        trace, source_constraints, options
    )
    errors = error_diagnostics(diagnostics)
    assert errors == [], f"[{label}] unexpected emit errors: {errors}"

    fx_trace, fx_constraints, fx_options, fx_diags = emitter.mlir_to_fx(mlir_text)
    errors = error_diagnostics(fx_diags)
    assert errors == [], f"[{label}] unexpected import errors: {errors}"

    _check_hyperparameters_roundtrip(options.subs, fx_options.subs)
    assert_constraints_equivalent(
        source_constraints,
        fx_constraints,
        custom_comparators={
            HardwareConstraint: compare_hardware_constraints_for_mlir_roundtrip
        },
    )
    assert_traces_equivalent(trace, fx_trace, subs=options.subs)
    print(f"  {label}: OK")


# CHECK-LABEL: mlir_to_fx_minimal_roundtrip
@run_test
def mlir_to_fx_minimal_roundtrip():
    """Test MLIR roundtrip for a minimal read/write kernel."""
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

    subs = {M: 128, N: 128, BLOCK_M: 64, BLOCK_N: 64}
    _assert_roundtrip(dummy_kernel, subs, "minimal roundtrip")

    # CHECK: minimal roundtrip: OK


# CHECK-LABEL: mlir_to_fx_simple_matmul_roundtrip
@run_test
def mlir_to_fx_simple_matmul_roundtrip():
    """Test MLIR roundtrip for fully-compiled matmul trace."""
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

    subs = {BLOCK_M: 16, BLOCK_N: 16, BLOCK_K: 16, M: 128, N: 128, K: 16}
    _assert_roundtrip(matmul_simple, subs, "matmul roundtrip")

    # CHECK: matmul roundtrip: OK


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
    mlir_text, diagnostics, _ = emitter.emit_wave_dialect(
        trace, source_constraints, options
    )
    errors = error_diagnostics(diagnostics)
    assert errors == [], f"unexpected errors from wave to mlir conversion: {errors}"

    # Convert back to FX trace
    fx_trace, fx_constraints, fx_options, fx_diags = emitter.mlir_to_fx(mlir_text)
    errors = error_diagnostics(fx_diags)
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
            HardwareConstraint: compare_hardware_constraints_for_mlir_roundtrip
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
    mlir_text, diagnostics, _ = emitter.emit_wave_dialect(
        trace, matmul_for_addr_test.constraints, options
    )
    errors = error_diagnostics(diagnostics)
    assert errors == [], f"unexpected errors from wave to mlir conversion: {errors}"

    # Replace all concrete address spaces with unspecified to simulate
    # an MLIR module where address spaces have not been resolved yet.
    # It would be better to do this using proper MLIR manipulation APIs,
    # but we never have the module here, only the text.
    mlir_unspecified = mlir_text.replace("<global>", "<unspecified>").replace(
        "<shared>", "<unspecified>"
    )

    fx_trace, _, _, fx_diags = emitter.mlir_to_fx(mlir_unspecified)
    errors = error_diagnostics(fx_diags)
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


# CHECK-LABEL: mlir_to_fx_mapping_roundtrip
@run_test
def mlir_to_fx_mapping_roundtrip():
    """Test that various IndexMapping permutations survive the MLIR roundtrip."""
    K = tkl.sym.K
    i = wave.IndexMapping.iterator(0)
    j = wave.IndexMapping.iterator(1)
    k = wave.IndexMapping.iterator(2)

    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: 64, N: 64, K: 32}
        ),
    ]
    subs = {BLOCK_M: 64, BLOCK_N: 64, M: 128, N: 128, K: 128}

    # Cyclic permutation: (M,N,K) -> (N,K,M).
    @wave.wave(constraints)
    def cyclic_read(
        a: Memory[M, N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: Memory[N, K, M, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        mapping = wave.IndexMapping(
            num_iterators=3, inputs={M: k, N: i, K: j}, outputs={N: i, K: j, M: k}
        )
        wave.write(wave.read(a, mapping=mapping), b)

    _assert_roundtrip(cyclic_read, subs, "cyclic read")

    # Swap of first two dims, third fixed: (M,N,K) -> (N,M,K).
    @wave.wave(constraints)
    def swap_read(
        a: Memory[M, N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: Memory[N, M, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        mapping = wave.IndexMapping(
            num_iterators=3, inputs={M: j, N: i, K: k}, outputs={N: i, M: j, K: k}
        )
        wave.write(wave.read(a, mapping=mapping), b)

    _assert_roundtrip(swap_read, subs, "swap read")

    # Identity mapping (degenerate, mapping present but no reorder).
    @wave.wave(constraints)
    def identity_read(
        a: Memory[M, N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: Memory[M, N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        mapping = wave.IndexMapping(
            3, inputs={M: i, N: j, K: k}, outputs={M: i, N: j, K: k}
        )
        wave.write(wave.read(a, mapping=mapping), b)

    _assert_roundtrip(identity_read, subs, "identity read")

    # Write mapping: cyclic permutation on the write side.
    @wave.wave(constraints)
    def cyclic_write(
        a: Memory[M, N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: Memory[N, K, M, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        mapping = wave.IndexMapping(
            3, inputs={M: i, N: j, K: k}, outputs={N: j, K: k, M: i}
        )
        a_reg = wave.read(a)
        wave.write(a_reg, b, mapping=mapping)

    _assert_roundtrip(cyclic_write, subs, "cyclic write")

    # Note: 2D-only permutations (e.g. [M,N]->[N,M] without a third dim) are not
    # tested here because the compiler produces wave.extract_slice ops for 2D
    # mapped reads/writes, and wave.extract_slice is not yet supported in the
    # MLIR-to-FX converter. The swap_read test above covers 2D swap semantics
    # (M<->N with K fixed) via a 3D mapping.
    #
    # Note: dynamic_val_mappings (IndexMapping with dynamic values) are not yet
    # represented in the MLIR attribute and are not roundtripped.

    # CHECK: OK: mapping roundtrip
    print("OK: mapping roundtrip")
