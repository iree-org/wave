# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Bidirectional converter between Wave FX traces and Wave MLIR.

Both directions run in a subprocess to isolate the Water MLIR Python bindings
from the host process.

`emit_wave_dialect` serializes a CapturedTrace via dill, spawns
`water_emitter.py`, and returns the MLIR module text.

`mlir_to_fx` sends MLIR text to `fx_emitter.py` and returns a
reconstructed CapturedTrace with constraints and compile options.
"""

import linecache
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import dill
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.constraints import Constraint
from wave_lang.kernel.wave.mlir_converter.diagnostics import (
    FileLocation,
    MLIRDiagnostic,
    NameLocation,
    WaterError,
)


# ANSI color codes for terminal output
_COLORS = {
    "red": "\033[91m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}

_NO_COLORS = {k: "" for k in _COLORS}


def _get_severity_color(severity: str) -> str:
    """Get ANSI color code based on diagnostic severity."""
    severity_lower = severity.lower()
    if "error" in severity_lower:
        return _COLORS["red"]
    if "warning" in severity_lower:
        return ""
    return _COLORS["cyan"]


def _format_frame(
    frame: FileLocation | NameLocation, lines: list[str], name: str | None = None
) -> None:
    """Recursively format a single location frame into *lines*.

    Args:
        frame: The location frame to format.
        lines: Accumulator list that formatted strings are appended to.
        name: Optional name context inherited from a parent `NameLocation`.
              When present it is shown as `in <name>` after the file/line.
    """
    if isinstance(frame, FileLocation):
        suffix = f", in {name}" if name else ""
        lines.append(f'  File "{frame.filename}", line {frame.start_line}{suffix}')

        source_line = linecache.getline(frame.filename, frame.start_line).rstrip()
        if source_line:
            lines.append(f"    {source_line}")
            if frame.start_col > 0:
                lines.append(f"    {' ' * (frame.start_col - 1)}^")

    elif isinstance(frame, NameLocation):
        if frame.child_location is not None:
            _format_frame(frame.child_location, lines, name=frame.name)
        else:
            lines.append(f'  In "{frame.name}"')

    else:
        # UnknownLocation or any other unrecognised frame type.
        suffix = f", in {name}" if name else ""
        lines.append(f"  <unknown location>{suffix}")


def format_diagnostic(diag: MLIRDiagnostic, use_color: bool = True) -> str:
    """Format a single MLIR diagnostic as a Python-style stack trace.

    Args:
        diag: An MLIRDiagnostic instance.
        use_color: Whether to use ANSI color codes in the output.

    Returns:
        A formatted string resembling a Python traceback.
    """
    lines = []
    colors = _COLORS if use_color else _NO_COLORS

    message = diag.message
    severity = diag.severity
    location = diag.location

    # Header with severity
    severity_color = _get_severity_color(severity) if use_color else ""
    if severity:
        lines.append(
            f"{severity_color}{colors['bold']}{severity}{colors['reset']}: {message}"
        )
    else:
        lines.append(f"{colors['bold']}MLIRDiagnostic{colors['reset']}: {message}")

    # Stack trace
    lines.append("Traceback (Wave DSL source):")
    if not location:
        lines.append("  <missing location>")
    else:
        for frame in location:
            _format_frame(frame, lines)

    return "\n".join(lines)


def format_error(diag: WaterError, use_color: bool = True) -> str:
    lines = []
    colors = _COLORS if use_color else _NO_COLORS
    message = diag.message
    severity = diag.severity

    # Header with severity
    severity_color = _get_severity_color(severity) if use_color else ""
    if severity:
        lines.append(
            f"{severity_color}{colors['bold']}{severity}{colors['reset']}: {message}"
        )
    else:
        lines.append(f"{colors['bold']}WaterError{colors['reset']}: {message}")

    error_diags = diag.error_diagnostics
    if error_diags:
        lines.append("MLIR errors:")
        for err in error_diags:
            lines.append(f"  {err}")

    return "\n".join(lines)


def format_diagnostics(
    diagnostics: list[MLIRDiagnostic | WaterError], use_color: bool = True
) -> str:
    """Format a list of diagnostics as stack traces.

    Args:
        diagnostics: List of MLIRDiagnostic or WaterError instances.
        use_color: Whether to use ANSI color codes in the output.

    Returns:
        A formatted string with all diagnostics separated by blank lines.
    """
    if not diagnostics:
        return ""

    lines = []

    for d in diagnostics:
        if isinstance(d, MLIRDiagnostic):
            lines.append(format_diagnostic(d, use_color))
        elif isinstance(d, WaterError):
            lines.append(format_error(d, use_color))

    return "\n\n".join(lines)


def print_diagnostics(
    diagnostics: list[MLIRDiagnostic | WaterError],
    file=None,
    use_color: bool | None = None,
) -> None:
    """Print diagnostics to a file (default: stderr) as stack traces.

    Args:
        diagnostics: List of MLIRDiagnostic or WaterError instances.
        file: File to print to (default: sys.stderr).
        use_color: Whether to use ANSI colors. If None, auto-detect based on terminal.
    """
    if file is None:
        file = sys.stderr
    if use_color is None:
        # Auto-detect: use color if output is a terminal
        use_color = hasattr(file, "isatty") and file.isatty()

    formatted = format_diagnostics(diagnostics, use_color)
    if formatted:
        print(formatted, file=file)


@dataclass
class FxEmitterResponse:
    """Structured response from the fx_emitter subprocess."""

    trace: CapturedTrace | None = None
    constraints: list[Constraint] = field(default_factory=list)
    options: WaveCompileOptions = field(default_factory=WaveCompileOptions)
    diagnostics: list[MLIRDiagnostic] = field(default_factory=list)


def emit_wave_dialect(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
    test_diagnostic_emission: bool = False,
    pipeline: str = "",
) -> tuple[str, list[MLIRDiagnostic | WaterError], dict[str, dict[str, Any]]]:
    """Emit Wave MLIR by sending the pickled trace and options to the emitter.

    The `subs` field of options is the only option used during emission. If
    `pipeline` is provided, it must be a parsable MLIR transform module
    containing a transform.named_sequence to be applied to the emitted module
    via the Transform dialect interpreter.

    Returns:
        A tuple of:
        - The string representation of the MLIR module if all stages succeeded.
        - A list of MLIRDiagnostic or WaterError instances.
        - A dict of inferred attributes per water ID.
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
    %1 = transform.structured.match ops{["normalform.module"]} in %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.apply_registered_pass "water-wave-propagate-defaults-from-constraints" to %1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "water-wave-infer-index-exprs" to %2 : (!transform.any_op) -> !transform.any_op
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
        diagnostics,
        inferred_attributes,
    )


def mlir_to_fx(
    mlir_text: str,
) -> tuple[CapturedTrace, list[Constraint], WaveCompileOptions, list[MLIRDiagnostic]]:
    """Convert Wave MLIR text back into a Wave FX trace via subprocess.

    Spawns `fx_emitter.py`, sends the MLIR text over stdin, and returns
    the reconstructed FX graph together with its associated metadata.

    Args:
        mlir_text: Textual representation of a Wave MLIR module containing a
            single function with Wave dialect operations.

    Returns:
        A 4-tuple of:
        - trace: The reconstructed `CapturedTrace` (FX graph with subgraphs).
        - constraints: Wave constraints extracted from the function attributes
          (workgroup, wave, tiling, device, and hardware constraints).
        - options: `WaveCompileOptions` with hyperparameters recovered from
          the `wave.hyperparameters` function attribute.
        - diagnostics: List of `MLIRDiagnostic` instances (errors, warnings,
          remarks) collected during parsing, verification, and conversion.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero code or the
            response cannot be unpickled / validated.
    """
    if not isinstance(mlir_text, str):
        raise ValueError(f"Expected MLIR text as str, got: {type(mlir_text)}")
    child = Path(__file__).with_name("fx_emitter.py")
    if not child.exists():
        raise RuntimeError(f"fx_emitter helper not found: {child}")
    proc = subprocess.Popen(
        [sys.executable, str(child)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, err = proc.communicate(dill.dumps({"mlir": mlir_text}))
    if proc.returncode != 0:
        diagnostics: list[MLIRDiagnostic] = []
        try:
            response = dill.loads(output)
            if isinstance(response, FxEmitterResponse):
                diagnostics = response.diagnostics
        except Exception:
            pass
        diag_text = (
            f"\n{format_diagnostics(diagnostics, use_color=False)}"
            if diagnostics
            else ""
        )
        raise RuntimeError(
            f"fx_emitter failed (code {proc.returncode}):\n"
            f"{err.decode('utf-8', errors='replace')}{diag_text}"
        )
    try:
        response = dill.loads(output)
    except Exception as e:
        raise RuntimeError(
            f"Failed to unpickle output from fx_emitter (code {proc.returncode}):\n"
            f"Output: {output!r}\n"
            f"Exception: {e}"
        ) from e
    if not isinstance(response, FxEmitterResponse):
        raise RuntimeError(f"fx_emitter output has unexpected type: {type(response)}")
    if not isinstance(response.trace, CapturedTrace):
        raise RuntimeError(
            f"fx_emitter trace has unexpected type: {type(response.trace)}"
        )
    response.trace.restore_node_state()
    return response.trace, response.constraints, response.options, response.diagnostics
