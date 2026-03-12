# REQUIRES: water
# RUN: python %s | FileCheck %s

import atexit
import sympy
from collections.abc import Callable

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
from wave_lang.kernel.ops.wave_ops import (
    get_custom,
    Placeholder,
    ShuffleOp as Shuffle,
)
from wave_lang.kernel.wave.compile import build_graph_passes
from wave_lang.kernel.wave.decompose_reduce_ops import decompose_reduce_ops
from wave_lang.kernel.wave.expansion.expansion import expand_graph
from wave_lang.kernel._support.indexing import IndexingContext
from wave_lang.kernel._support.tracing import CapturedTrace

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


def _assert_roundtrip(
    kernel,
    subs: dict,
    label: str,
    stop_before: Callable[[], None] | None = None,
    stop_after: Callable[[], None] | None = None,
) -> CapturedTrace:
    """Compile a kernel, roundtrip through MLIR, and assert equivalence.

    When neither `stop_before` nor `stop_after` is set, runs all
    graph passes. `stop_before` stops just before the named pass;
    `stop_after` runs it and then stops. Returns the original trace
    for additional caller-side assertions.
    """
    assert (
        stop_before is None or stop_after is None
    ), "stop_before and stop_after are mutually exclusive"
    options = WaveCompileOptions(subs=subs, compile_to_mlir=True)
    with IndexingContext() as idxc:
        # The indexing context is usually initialized by wave_compile.
        # To enable running graph passes manually, we initialize it here.
        idxc.set_subs(options.subs)
        kernel.initialize_wave_constraints()
        kernel.initialize_symbolic_constraints()
        kernel.initialize_workgroup_constraints()
        trace = kernel._trace(location_capture_config=options.location_capture_config)
        graph_passes = build_graph_passes(kernel, trace, options)
        stop_pass = stop_before or stop_after
        if stop_pass is not None:
            assert any(
                p.__name__ == stop_pass.__name__ for p in graph_passes
            ), f"stop pass {stop_pass.__name__!r} not found in graph passes"
        for p in graph_passes:
            if stop_before is not None and p.__name__ == stop_before.__name__:
                break
            p()
            if stop_after is not None and p.__name__ == stop_after.__name__:
                break
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
    return trace


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


# CHECK-LABEL: mlir_to_fx_multi_result_iterate_roundtrip
@run_test
def mlir_to_fx_multi_result_iterate_roundtrip():
    """Test that multi-result iterate preserves GetResult ordering.

    A single-result iterate doesn't exercise GetResult ordering because
    there is only one result. With multiple init_args the FX importer
    must emit GetResult nodes in the same order as the original trace.
    """
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
    def multi_result_iterate(
        a: tkl.Memory[M, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        acc_reg = tkl.Register[M, N, tkl.f32](0.0)
        extra_reg = tkl.Register[M, N, tkl.f32](0.0)

        @wave.iterate(K, init_args=[acc_reg, extra_reg])
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
            extra: tkl.Register[M, N, tkl.f32],
        ) -> tuple[tkl.Register[M, N, tkl.f32], tkl.Register[M, N, tkl.f32]]:
            a_reg = wave.read(a, bounds={M: M, K: K})
            b_reg = wave.read(b, bounds={N: N, K: K})
            acc = wave.mma(a_reg, b_reg, acc)
            return acc, extra

        result = repeat[0] + repeat[1]
        wave.write(result, c)

    subs = {BLOCK_M: 16, BLOCK_N: 16, BLOCK_K: 16, M: 128, N: 128, K: 16}
    _assert_roundtrip(multi_result_iterate, subs, "multi-result iterate roundtrip")

    # CHECK: multi-result iterate roundtrip: OK


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


# CHECK-LABEL: mlir_to_fx_reduction_roundtrip
@run_test
def mlir_to_fx_reduction_roundtrip():
    """Test MLIR roundtrip for sum and max reductions.

    Tests both reduction input representations with the same kernel:
    - Single-input (stop_before=expand_graph): reduction arg is a
      single node; verifies the importer unwraps single-element
      variadic inputs back to a scalar.
    - Variadic-input (stop_before=decompose_reduce_ops): reduction arg
      is a list of tiled slices; verifies multi-input reductions
      roundtrip as lists.

    A single wave covers N so that the halved vector shape produces
    dim_scaling = tile / (waves * vshape) = 128 / (1 * 64) = 2,
    giving two reduction tiles after expansion.
    """
    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, BLOCK_N),
        wave.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: BLOCK_M, N: sympy.floor(BLOCK_N / 2)},
        ),
    ]

    subs = {BLOCK_M: 64, BLOCK_N: 128, M: 128, N: 128}

    @wave.wave(constraints)
    def sum_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        res = wave.read(a)
        init = wave.read(c)
        res = wave.sum(res, init, dim=N)
        wave.write(res, c)

    @wave.wave(constraints)
    def max_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        res = wave.read(a)
        init = wave.read(c)
        res = wave.max(res, init, dim=N)
        wave.write(res, c)

    _assert_roundtrip(sum_kernel, subs, "sum single-input", stop_before=expand_graph)
    _assert_roundtrip(max_kernel, subs, "max single-input", stop_before=expand_graph)
    _assert_roundtrip(
        sum_kernel, subs, "sum variadic-input", stop_before=decompose_reduce_ops
    )
    _assert_roundtrip(
        max_kernel, subs, "max variadic-input", stop_before=decompose_reduce_ops
    )

    # CHECK: sum single-input: OK
    # CHECK: max single-input: OK
    # CHECK: sum variadic-input: OK
    # CHECK: max variadic-input: OK


# CHECK-LABEL: mlir_to_fx_binary_op_roundtrip
@run_test
def mlir_to_fx_binary_op_roundtrip():
    """Test MLIR roundtrip for binary arithmetic operations."""
    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    subs = {BLOCK_M: 64, BLOCK_N: 64, M: 128, N: 128}

    @wave.wave(constraints)
    def add_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = wave.read(a)
        b_reg = wave.read(b)
        res = a_reg + b_reg
        wave.write(res, c)

    _assert_roundtrip(add_kernel, subs, "add", stop_before=decompose_reduce_ops)

    @wave.wave(constraints)
    def sub_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = wave.read(a)
        b_reg = wave.read(b)
        res = a_reg - b_reg
        wave.write(res, c)

    _assert_roundtrip(sub_kernel, subs, "sub", stop_before=decompose_reduce_ops)

    @wave.wave(constraints)
    def mul_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = wave.read(a)
        b_reg = wave.read(b)
        res = a_reg * b_reg
        wave.write(res, c)

    _assert_roundtrip(mul_kernel, subs, "mul", stop_before=decompose_reduce_ops)

    # NOTE: Division (`a / b`) is not tested here because the Python-side FX
    # op name is "truediv" but the Water emitter maps it as "div". This is a
    # pre-existing emitter gap; the FX->MLIR direction needs fixing first.

    @wave.wave(constraints)
    def max_min_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = wave.read(a)
        b_reg = wave.read(b)
        mx = wave.maximum(a_reg, b_reg)
        mn = wave.minimum(a_reg, b_reg)
        wave.write(mx + mn, c)

    _assert_roundtrip(max_min_kernel, subs, "max_min", stop_before=decompose_reduce_ops)

    # CHECK: add: OK
    # CHECK: sub: OK
    # CHECK: mul: OK
    # CHECK: max_min: OK


# CHECK-LABEL: mlir_to_fx_unary_op_roundtrip
@run_test
def mlir_to_fx_unary_op_roundtrip():
    """Test MLIR roundtrip for unary operations (exp2, reciprocal)."""
    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    subs = {BLOCK_M: 64, BLOCK_N: 64, M: 128, N: 128}

    @wave.wave(constraints)
    def exp2_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = wave.read(a)
        res = wave.exp2(a_reg)
        wave.write(res, c)

    _assert_roundtrip(exp2_kernel, subs, "exp2", stop_before=decompose_reduce_ops)

    @wave.wave(constraints)
    def reciprocal_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = wave.read(a)
        res = wave.reciprocal(a_reg)
        wave.write(res, c)

    _assert_roundtrip(
        reciprocal_kernel, subs, "reciprocal", stop_before=decompose_reduce_ops
    )

    # CHECK: exp2: OK
    # CHECK: reciprocal: OK


# CHECK-LABEL: mlir_to_fx_cast_roundtrip
@run_test
def mlir_to_fx_cast_roundtrip():
    """Test MLIR roundtrip for cast operation."""
    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    subs = {BLOCK_M: 64, BLOCK_N: 64, M: 128, N: 128}

    @wave.wave(constraints)
    def cast_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = wave.read(a)
        res = wave.cast(a_reg, tkl.f32)
        wave.write(res, c)

    _assert_roundtrip(
        cast_kernel, subs, "cast f16->f32", stop_before=decompose_reduce_ops
    )

    # CHECK: cast f16->f32: OK


# CHECK-LABEL: mlir_to_fx_permute_roundtrip
@run_test
def mlir_to_fx_permute_roundtrip():
    """Test MLIR roundtrip for permute operation."""
    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    subs = {BLOCK_M: 64, BLOCK_N: 64, M: 128, N: 128}

    @wave.wave(constraints)
    def permute_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[N, M, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = wave.read(a)
        res = wave.permute(a_reg, target_shape=[N, M])
        wave.write(res, c)

    _assert_roundtrip(permute_kernel, subs, "permute", stop_before=decompose_reduce_ops)

    # CHECK: permute: OK


# CHECK-LABEL: mlir_to_fx_mask_pattern_roundtrip
@run_test
def mlir_to_fx_mask_pattern_roundtrip():
    """Test MLIR roundtrip for the attention mask pattern.

    Exercises self_index, apply_expr, cast, broadcast, and select in
    combination: create an index, compare it against a dimension bound,
    cast to i1, broadcast, and select.
    """
    K2 = tkl.sym.K2
    BLOCK_K2 = tkl.sym.BLOCK_K2

    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(K2, BLOCK_K2, 1),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(K2, sympy.floor(BLOCK_K2 / 2)),
        wave.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, K2: BLOCK_K2}
        ),
    ]

    subs = {BLOCK_M: 64, BLOCK_K2: 64, M: 128, K2: 128}

    @wave.wave(constraints)
    def self_index_kernel(
        a: tkl.Memory[M, K2, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, K2, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        k2_idx = wave.self_index(K2, tkl.i64)
        mask = wave.apply_expr(k2_idx, lambda x: x < K2)
        mask = wave.cast(mask, wave.i1)
        mask = wave.broadcast(mask, target_shape=[M, K2])
        a_reg = wave.read(a)
        res = wave.select(mask, a_reg, a_reg)
        wave.write(res, c)

    _assert_roundtrip(
        self_index_kernel,
        subs,
        "self_index + apply_expr + select",
        stop_before=decompose_reduce_ops,
    )

    # CHECK: self_index + apply_expr + select: OK


# CHECK-LABEL: mlir_to_fx_shuffle_roundtrip
@run_test
def mlir_to_fx_shuffle_roundtrip():
    """Test MLIR roundtrip for shuffle ops introduced by decompose_reduce_ops.

    Runs through decompose_reduce_ops to generate butterfly shuffles,
    emits MLIR, imports back, and checks full trace equivalence.
    """
    constraints = [
        wave.WorkgroupConstraint(M, BLOCK_M, 0),
        wave.WorkgroupConstraint(N, BLOCK_N, 1),
        wave.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        wave.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        wave.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    subs = {BLOCK_M: 64, BLOCK_N: 64, M: 128, N: 128}

    def _assert_has_shuffles(trace: CapturedTrace, label: str) -> None:
        assert any(
            isinstance(get_custom(n), Shuffle)
            for n in trace.walk(lambda n: isinstance(get_custom(n), Shuffle))
        ), f"[{label}] no shuffles after decomposition"

    @wave.wave(constraints)
    def sum_shuffle_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        res = wave.read(a)
        init = wave.read(c)
        res = wave.sum(res, init, dim=N)
        wave.write(res, c)

    trace = _assert_roundtrip(
        sum_shuffle_kernel, subs, "sum shuffle", stop_after=decompose_reduce_ops
    )
    _assert_has_shuffles(trace, "sum shuffle")

    @wave.wave(constraints)
    def max_shuffle_kernel(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        res = wave.read(a)
        init = wave.read(c)
        res = wave.max(res, init, dim=N)
        wave.write(res, c)

    trace = _assert_roundtrip(
        max_shuffle_kernel, subs, "max shuffle", stop_after=decompose_reduce_ops
    )
    _assert_has_shuffles(trace, "max shuffle")

    # CHECK: sum shuffle: OK
    # CHECK: max shuffle: OK
