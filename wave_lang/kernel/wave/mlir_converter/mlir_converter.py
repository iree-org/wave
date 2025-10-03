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


def emit_wave_dialect(trace: CapturedTrace, options: WaveCompileOptions) -> str:
    """Emit Wave MLIR by sending the pickled trace and options to the emitter.

    The `subs` field of options is the only option used during emission."""

    child = Path(__file__).with_name("water_emitter.py")
    if not child.exists():
        raise RuntimeError(f"water emitter helper not found: {child}")

    # Ensure additional node fields (like .type) are not lost during pickling
    trace.snapshot_node_state()

    proc = subprocess.run(
        [sys.executable, str(child)],
        input=dill.dumps({"trace": trace, "options": options}),
        text=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"water_emitter failed (code {proc.returncode}):\n{proc.stderr}"
        )
    return proc.stdout.decode("utf-8")
