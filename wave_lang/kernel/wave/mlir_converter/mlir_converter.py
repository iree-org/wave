# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Bidirectional converter between Wave FX traces and Wave MLIR.

Both directions run in a subprocess to isolate the Water MLIR Python bindings
from the host process.  Use `PersistentEmitter` as a context manager to keep
the subprocesses alive and amortize the ~2 s import overhead across calls.

Usage:
    with PersistentEmitter() as emitter:
        mlir, diags, attrs = emitter.emit_wave_dialect(trace, constraints, options)
        trace2, cons2, opts2, diags2 = emitter.mlir_to_fx(mlir)
"""

import linecache
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.constraints import Constraint
from wave_lang.kernel.wave.mlir_converter.diagnostics import (
    FileLocation,
    MLIRDiagnostic,
    NameLocation,
    WaterDiagTestingMode,
    WaterError,
)
from wave_lang.kernel.wave.mlir_converter import dill_util


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
                lines.append(
                    f"    {' ' * (frame.start_col)}^{'~' * (frame.end_col - frame.start_col - 1)}"
                )

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
    if location:
        lines.append("Traceback (Wave DSL source):")
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
    diagnostics: list[MLIRDiagnostic | WaterError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared helpers -- used by both standalone functions and PersistentEmitter
# ---------------------------------------------------------------------------

_WATER_ANALYSIS_PIPELINE = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.apply_registered_pass "water-wave-detect-normal-forms" to %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["normalform.module"]} in %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.apply_registered_pass "water-wave-propagate-defaults-from-constraints" to %1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "water-wave-infer-index-exprs" to %2 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}"""


def _start_emitter(script_name: str) -> subprocess.Popen:
    """Locate and spawn an emitter subprocess."""
    child = Path(__file__).with_name(script_name)
    if not child.exists():
        raise RuntimeError(f"Emitter helper not found: {child}")
    return subprocess.Popen(
        [sys.executable, str(child)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )


def _prepare_water_request(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
    test_diagnostic_emission: WaterDiagTestingMode = WaterDiagTestingMode.NO,
    pipeline: str = "",
) -> bytes:
    """Build and serialize a water_emitter request.

    Snapshots the trace's node state, expands the water-analysis pipeline if
    requested, and returns the dill-serialized request bytes.
    """
    # Ensure additional node fields (like .type) are not lost during pickling.
    trace.snapshot_node_state()

    assert not (
        options.check_water_analysis and pipeline
    ), "Cannot check water analysis and use a pipeline"
    if options.check_water_analysis:
        pipeline = _WATER_ANALYSIS_PIPELINE

    return dill_util.dumps(
        {
            "trace": trace,
            "constraints": constraints,
            "options": options,
            "pipeline": pipeline,
            "test_diagnostic_emission": test_diagnostic_emission,
        }
    )


def _unpack_water_response(
    raw: bytes,
) -> tuple[str, list[MLIRDiagnostic | WaterError], dict[str, dict[str, Any]]]:
    """Deserialize and unpack a water_emitter response."""
    unpickled = dill_util.loads(raw)
    if not isinstance(unpickled, dict):
        raise RuntimeError(
            f"water_emitter response has unexpected type: {type(unpickled)}"
        )
    module = unpickled.get("module")
    diagnostics = unpickled.get("diagnostics")
    inferred_attributes = unpickled.get("inferred_attributes")
    return (
        module.decode("utf-8") if module else "",
        diagnostics or [],
        inferred_attributes or {},
    )


def _unpack_fx_response(
    raw: bytes,
) -> tuple[
    CapturedTrace,
    list[Constraint],
    WaveCompileOptions,
    list[MLIRDiagnostic | WaterError],
]:
    """Deserialize and validate an fx_emitter response."""
    response = dill_util.loads(raw)
    if not isinstance(response, FxEmitterResponse):
        raise RuntimeError(f"fx_emitter output has unexpected type: {type(response)}")
    if response.trace is None:
        raise RuntimeError(
            f"fx_emitter returned no trace. Diagnostics: "
            f"{format_diagnostics(response.diagnostics, use_color=False)}"
        )
    if not isinstance(response.trace, CapturedTrace):
        raise RuntimeError(
            f"fx_emitter trace has unexpected type: {type(response.trace)}"
        )
    response.trace.restore_node_state()
    return response.trace, response.constraints, response.options, response.diagnostics


from wave_lang.kernel.wave.mlir_converter.protocol import recv_message, send_message


# ---------------------------------------------------------------------------
# PersistentEmitter -- keeps emitter subprocesses alive across calls
# ---------------------------------------------------------------------------


class PersistentEmitter:
    """Keeps water_emitter and fx_emitter subprocesses alive for repeated use.

    Subprocesses are started lazily on first use, so callers that only need
    one direction (e.g. FX -> MLIR) don't pay for the other subprocess.

    Can be used as a context manager for automatic cleanup, or standalone
    with an explicit `close()` call (e.g. via `atexit`):

        # Context manager - subprocesses closed on block exit.
        with PersistentEmitter() as emitter:
            mlir, diags, attrs = emitter.emit_wave_dialect(trace, constraints, options)

        # Standalone - caller is responsible for closing.
        emitter = PersistentEmitter()
        atexit.register(emitter.close)
        mlir, diags, attrs = emitter.emit_wave_dialect(trace, constraints, options)
    """

    def __init__(self) -> None:
        self._water_proc: subprocess.Popen | None = None
        self._fx_proc: subprocess.Popen | None = None

    # -- lifecycle ------------------------------------------------------------

    def __enter__(self) -> "PersistentEmitter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for proc in (self._water_proc, self._fx_proc):
            if proc is not None and proc.poll() is None:
                proc.stdin.close()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        self._water_proc = None
        self._fx_proc = None

    def _get_water_proc(self) -> subprocess.Popen:
        if self._water_proc is None or self._water_proc.poll() is not None:
            self._water_proc = _start_emitter("water_emitter.py")
        return self._water_proc

    def _get_fx_proc(self) -> subprocess.Popen:
        if self._fx_proc is None or self._fx_proc.poll() is not None:
            self._fx_proc = _start_emitter("fx_emitter.py")
        return self._fx_proc

    # -- public API -----------------------------------------------------------

    def emit_wave_dialect(
        self,
        trace: CapturedTrace,
        constraints: list[Constraint],
        options: WaveCompileOptions,
        test_diagnostic_emission: WaterDiagTestingMode = WaterDiagTestingMode.NO,
        pipeline: str = "",
    ) -> tuple[str, list[MLIRDiagnostic | WaterError], dict[str, dict[str, Any]]]:
        """Emit Wave MLIR from a traced FX graph."""
        proc = self._get_water_proc()
        request = _prepare_water_request(
            trace, constraints, options, test_diagnostic_emission, pipeline
        )
        send_message(proc.stdin, request)
        return _unpack_water_response(recv_message(proc.stdout))

    def mlir_to_fx(
        self,
        mlir_text: str,
    ) -> tuple[
        CapturedTrace,
        list[Constraint],
        WaveCompileOptions,
        list[MLIRDiagnostic | WaterError],
    ]:
        """Convert Wave MLIR text back into a Wave FX trace."""
        if not isinstance(mlir_text, str):
            raise ValueError(f"Expected MLIR text as str, got: {type(mlir_text)}")
        proc = self._get_fx_proc()
        send_message(proc.stdin, dill_util.dumps({"mlir": mlir_text}))
        return _unpack_fx_response(recv_message(proc.stdout))
