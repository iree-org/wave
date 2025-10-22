# Copyright 2025 The IREE Authors
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
import dill
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel.wave.compile_options import WaveCompileOptions


def emit_wave_dialect(
    trace: CapturedTrace, options: WaveCompileOptions, test_diagnostic_emission: bool
) -> tuple[str, list[str]]:
    """Emit Wave MLIR by sending the pickled trace and options to the emitter.

    The `subs` field of options is the only option used during emission."""

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

    output, err = proc.communicate(
        dill.dumps({"trace": trace, "options": options, "pipeline": ""})
    )

    if proc.returncode != 0:
        raise RuntimeError(f"water_emitter failed (code {proc.returncode}):\n{err}")

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

    return module, diagnostics
