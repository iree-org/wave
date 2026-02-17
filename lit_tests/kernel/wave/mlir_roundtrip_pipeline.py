# REQUIRES: water
# RUN: python %s | FileCheck %s

"""Progressive MLIR roundtrip tests across the compilation pipeline.

Tests that FX <-> Water MLIR roundtrip holds at each stage of the
PyWave compilation pipeline.

Each pass is classified against a per-kernel `expected_failures` set,
using the standard LLVM lit result codes:

    OK    -- roundtrip succeeded, pass is NOT in the xfail set  (working)
    XFAIL -- roundtrip failed,    pass IS in the xfail set      (expected failure)
    XPASS -- roundtrip succeeded, pass IS in the xfail set      (unexpected pass)
    FAIL  -- roundtrip failed,    pass is NOT in the xfail set  (regression)

The test asserts zero FAIL (regressions) and zero XPASS (stale xfails).
When a pass is fixed, the XPASS forces the developer to remove it from
the `expected_failures` set, keeping it in sync with reality.
"""

from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel._support.indexing import IndexingContext
from wave_lang.kernel.wave.wave import LaunchableWave
from wave_lang.kernel.wave.compile import WaveCompileOptions, build_graph_passes
from wave_lang.kernel.wave.constraints import (
    Constraint,
    HardwareConstraint,
    MMAType,
)
from wave_lang.kernel.wave.mlir_converter.diagnostics import error_diagnostics
from wave_lang.kernel.wave.mlir_converter.mlir_converter import (
    emit_wave_dialect,
    format_diagnostics,
    mlir_to_fx,
)
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.utils.classes import Failure, Result, Success
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.utils.graph_utils import (
    assert_traces_equivalent,
    assert_constraints_equivalent,
    compare_hardware_constraints_for_mlir_roundtrip,
)


def _try_roundtrip(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
) -> Result[None]:
    """Attempt an MLIR roundtrip on the current trace state."""
    try:
        # Emit FX -> Water MLIR.
        mlir_text, diagnostics, _ = emit_wave_dialect(trace, constraints, options)
        errors = error_diagnostics(diagnostics)
        if errors:
            return Failure(f"emit:\n{format_diagnostics(errors, use_color=False)}")

        # Import Water MLIR -> FX.
        fx_trace, fx_constraints, fx_options, fx_diags = mlir_to_fx(mlir_text)
        errors = error_diagnostics(fx_diags)
        if errors:
            return Failure(f"import:\n{format_diagnostics(errors, use_color=False)}")

        # Check structural equivalence.
        assert_traces_equivalent(trace, fx_trace, subs=options.subs)

        # Check constraints equivalence.
        assert_constraints_equivalent(
            constraints,
            fx_constraints,
            custom_comparators={
                HardwareConstraint: compare_hardware_constraints_for_mlir_roundtrip,
            },
        )

        return Success()
    except Exception as e:
        return Failure(str(e))


def _run_progressive_roundtrip(
    launchable: LaunchableWave,
    options: WaveCompileOptions,
    expected_failures: frozenset[str],
) -> None:
    """Run all compilation passes, roundtripping after each one.

    Prints per-pass results (OK / XFAIL / XPASS / FAIL) and a summary
    line. Raises `AssertionError` on unexpected failures or unexpected
    passes (stale xfail entries).
    """
    # Replicate the setup that wave_compile performs before running passes.
    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)
        launchable.initialize_wave_constraints()
        launchable.initialize_symbolic_constraints()
        launchable.initialize_workgroup_constraints()

        trace = launchable._trace(
            location_capture_config=options.location_capture_config
        )

        graph_passes = build_graph_passes(launchable, trace, options)

        # Validate that every name in the xfail set corresponds to an actual
        # pass.  Catches stale entries after pass renames or removals.
        actual_names = {p.__name__ for p in graph_passes}
        stale = expected_failures - actual_names
        assert not stale, (
            f"expected_failures contains pass names that no longer "
            f"exist in the pipeline -- remove them: {', '.join(sorted(stale))}"
        )

        total = len(graph_passes)
        ok_count = 0
        xfail_count = 0
        xpass_count = 0
        fail_count = 0
        xpass_names: list[str] = []
        fail_names: list[str] = []

        for i, p in enumerate(graph_passes, 1):
            name = p.__name__
            expected_fail = name in expected_failures
            # Each pass mutates `trace` in place (captured by reference in
            # the partial). _try_roundtrip is read-only on the trace: it
            # serializes to MLIR text and compares it against a fresh import.
            p()

            result = _try_roundtrip(trace, launchable.constraints, options)

            if result and not expected_fail:
                ok_count += 1
                print(f"[{i}/{total}] {name}: [OK]")
            elif result and expected_fail:
                xpass_count += 1
                xpass_names.append(name)
                print(f"[{i}/{total}] {name}: [XPASS] remove from expected_failures")
            elif not result and expected_fail:
                xfail_count += 1
                err = result.error
                short_err = (err[:120] + "...") if len(err) > 120 else err
                print(f"[{i}/{total}] {name}: [XFAIL] {short_err}")
            else:
                fail_count += 1
                fail_names.append(name)
                err = result.error
                short_err = (err[:120] + "...") if len(err) > 120 else err
                print(f"[{i}/{total}] {name}: [FAIL] {short_err}")

        print(
            f"summary: {ok_count} OK, {xfail_count} XFAIL, "
            f"{xpass_count} XPASS, {fail_count} FAIL"
        )

        if fail_count:
            raise AssertionError(
                f"{fail_count} unexpected roundtrip failure(s): "
                + ", ".join(fail_names)
            )
        if xpass_count:
            raise AssertionError(
                f"{xpass_count} pass(es) now roundtrip successfully -- "
                f"remove from expected_failures: " + ", ".join(xpass_names)
            )


# CHECK-LABEL: gemm_progressive_roundtrip
@run_test
def gemm_progressive_roundtrip():
    """Test MLIR roundtrip at each stage of the GEMM compilation pipeline."""
    gemm, hyperparams, _ = get_gemm_kernel(
        shape=(128, 128, 16),
        dynamic_dims=False,
        mfma_variant=MMAType.F32_16x16x16_F16,
        block_shape=(16, 16, 16),
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        compile_to_mlir=True,
    )

    # Passes whose MLIR roundtrip is known to fail for this kernel.
    # As the emitters improve, passes should be REMOVED from this set so
    # the test locks in the progress.
    expected_failures = frozenset(
        {
            "debug_log_hoist",
            "initialize_iter_args",
            "create_induction_vars",
            "initialize_reductions",
            "finalize_indices",
            "substitute_vector_shapes",
            "add_get_results",
            "infer_types",
            "construct_index_mapping",
            "debug_log_write_replace",
            "promote_placeholders",
            "set_node_indices",
            "reorder_workgroups",
            "expand_graph",
            "set_post_expansion_indices",
            "remove_chained_getresult",
            "decompose_vmma_ops",
            "decompose_dot_mma",
            "generate_bounds_exprs",
            "location_check_pass",
        }
    )

    # CHECK: {{[0-9]+}} OK, {{[0-9]+}} XFAIL, 0 XPASS, 0 FAIL
    _run_progressive_roundtrip(gemm, options, expected_failures)
