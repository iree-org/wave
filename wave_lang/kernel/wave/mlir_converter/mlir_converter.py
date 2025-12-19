# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MLIR Converter for Wave Dialect

This provides functionality to convert Wave traces into MLIR code
using the Wave dialect. It serializes the trace data and spawns a separate water emitter
process that uses Water Python bindings to generate the MLIR output.

The converter handles:
- Serialization of Wave kernel traces using the dill library
- Spawning the water emitter as a subprocess
- Triggering operation type inference and some simple wave type mapping
"""

import subprocess
import sys
from pathlib import Path
from typing import Any
import dill
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.constraints import Constraint


def emit_wave_dialect(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
    test_diagnostic_emission: bool = False,
    pipeline: str = "",
) -> tuple[str, list[str], dict[str, dict[str, Any]]]:
    """Emit Wave MLIR by sending the pickled trace and options to the emitter.

    The `subs` field of options is the only option used during emission. If
    `pipeline` is provided, it must be a parsable MLIR transform module
    containing a transform.named_sequence to be applied to the emitted module
    via the Transform dialect interpreter.

    Returns the string representation of the MLIR module if all stages succeeded
    and a list of diagnostic messages, including parsing and pass pipeline
    errors produced by MLIR.
    """

    child = Path(__file__).with_name("water_emitter.py")
    if not child.exists():
        raise RuntimeError(f"water emitter helper not found: {child}")

    # Ensure additional node fields (like .type) are not lost during pickling
    trace.snapshot_node_state()

    args = [sys.executable, str(child)]

    if test_diagnostic_emission:
        args.append("--test-diagnostic-emission")

    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert (
        not options.check_water_analysis or not pipeline
    ), "Cannot check water analysis and use a pipeline"
    if options.check_water_analysis:
        pipeline = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.apply_registered_pass "water-wave-detect-normal-forms" to %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.apply_registered_pass "water-wave-propagate-defaults-from-constraints" to %0 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "water-wave-infer-index-exprs" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}"""

    output, err = proc.communicate(
        dill.dumps(
            {
                "trace": trace,
                "constraints": constraints,
                "options": options,
                "pipeline": pipeline,
            }
        )
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"water_emitter failed (code {proc.returncode}):\n"
            f"{err.decode('utf-8', errors='replace')}\n"
            f"{output.decode('utf-8', errors='replace')}"
        )

    try:
        unpickled = dill.loads(output)
    except Exception as e:
        raise RuntimeError(
            f"Failed to unpickle output from water_emitter (code {proc.returncode}):\n"
            f"Output: {output!r}\n"
            f"Exception: {e}"
        ) from e
    diagnostics = unpickled.get("diagnostics") if isinstance(unpickled, dict) else None
    module = unpickled.get("module") if isinstance(unpickled, dict) else None
    inferred_attributes = (
        unpickled.get("inferred_attributes") if isinstance(unpickled, dict) else None
    )

    # Preserve stderr messages.
    if err:
        print(err.decode("utf-8", errors="replace"), file=sys.stderr)

    return (
        module.decode("utf-8"),
        [d.decode("utf-8") for d in diagnostics],
        inferred_attributes,
    )
