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

import linecache
import subprocess
import sys
from pathlib import Path
from typing import Any
import dill
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.constraints import Constraint


# ANSI color codes for terminal output
_COLORS = {
    "red": "\033[91m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}


def _get_severity_color(severity: str) -> str:
    """Get ANSI color code based on diagnostic severity."""
    severity_lower = severity.lower()
    if "error" in severity_lower:
        return _COLORS["red"]
    if "warning" in severity_lower:
        return _COLORS["yellow"]
    return _COLORS["cyan"]


def format_diagnostic(diag: dict, use_color: bool = True) -> str:
    """Format a single diagnostic as a Python-style stack trace.

    Args:
        diag: A diagnostic dict with "type", "message", and optionally "location" and "severity".
        use_color: Whether to use ANSI color codes in the output.

    Returns:
        A formatted string resembling a Python traceback.
    """
    lines = []
    colors = _COLORS if use_color else {k: "" for k in _COLORS}

    diag_type = diag.get("type", "Unknown")
    message = diag.get("message", "")
    severity = diag.get("severity", "")
    location = diag.get("location", [])

    # Header with severity
    severity_color = _get_severity_color(severity) if use_color else ""
    if severity:
        severity_short = severity.replace("DiagnosticSeverity.", "")
        lines.append(
            f"{severity_color}{colors['bold']}{severity_short}{colors['reset']}: {message}"
        )
    else:
        lines.append(f"{colors['bold']}{diag_type}{colors['reset']}: {message}")

    # Stack trace (if location frames present)
    if location:
        lines.append("Traceback (Wave DSL source):")
        for frame in location:
            if frame.get("type") == "file":
                filename = frame.get("filename", "<unknown>")
                line_num = frame.get("line", 0)
                col = frame.get("column", 0)

                # Format file location
                lines.append(f'  File "{filename}", line {line_num}')

                # Try to read the actual source line
                source_line = linecache.getline(filename, line_num).rstrip()
                if source_line:
                    lines.append(f"    {source_line}")
                    # Add caret pointing to column if available
                    if col > 0:
                        lines.append(f"    {' ' * (col - 1)}^")
            else:
                # Unknown location type
                lines.append(f"  {frame.get('repr', '<unknown location>')}")

    # Include error_diagnostics if present (for MLIRError type)
    error_diags = diag.get("error_diagnostics", [])
    if error_diags:
        lines.append("Related errors:")
        for err in error_diags:
            lines.append(f"  {err}")

    return "\n".join(lines)


def format_diagnostics(diagnostics: list[dict], use_color: bool = True) -> str:
    """Format a list of diagnostics as stack traces.

    Args:
        diagnostics: List of diagnostic dicts.
        use_color: Whether to use ANSI color codes in the output.

    Returns:
        A formatted string with all diagnostics separated by blank lines.
    """
    if not diagnostics:
        return ""
    return "\n\n".join(format_diagnostic(d, use_color) for d in diagnostics)


def print_diagnostics(
    diagnostics: list[dict],
    file=None,
    use_color: bool | None = None,
) -> None:
    """Print diagnostics to a file (default: stderr) as stack traces.

    Args:
        diagnostics: List of diagnostic dicts.
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


def emit_wave_dialect(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
    test_diagnostic_emission: bool = False,
    pipeline: str = "",
) -> tuple[str, list[dict], dict[str, dict[str, Any]]]:
    """Emit Wave MLIR by sending the pickled trace and options to the emitter.

    The `subs` field of options is the only option used during emission. If
    `pipeline` is provided, it must be a parsable MLIR transform module
    containing a transform.named_sequence to be applied to the emitted module
    via the Transform dialect interpreter.

    Returns:
        A tuple of:
        - The string representation of the MLIR module if all stages succeeded.
        - A list of diagnostic dicts. Each diagnostic has a "type" key and may include:
          - "message": The diagnostic message text.
          - "location": A list of location frames (stack trace), each with:
            - "type": "file" or "unknown"
            - "filename", "line", "column" (for "file" type)
            - "repr" (for "unknown" type)
          - "severity": The severity level (for MLIRDiagnostic type).
          - "error_diagnostics": List of error strings (for MLIRError type).
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
