"""Diagnostic dataclasses for the MLIR converter.

These are used to communicate diagnostics between the water emitter subprocess
and the parent MLIR converter process via pickling.

The location dataclasses mirror the ir.Location types exposed by the MLIR
Python bindings (nanobind):

  ir.Location type-check methods & properties
  ─────────────────────────────────────────────
  FileLineColLoc / FileLineColRange  (is_a_file)
      filename, start_line, start_col, end_line, end_col

  CallSiteLoc  (is_a_callsite)
      callee, caller

  FusedLoc  (is_a_fused)
      locations  →  list[ir.Location]

  NameLoc  (is_a_name)
      name_str, child_loc
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class FileLocation:
    """A file-based location with filename, line range, and column range.

    Field names match the ir.Location Python binding properties for
    `FileLineColLoc` / `FileLineColRange`.
    """

    filename: str
    start_line: int
    start_col: int = 0
    end_line: int = 0
    end_col: int = 0


@dataclass
class NameLocation:
    """A named location wrapping a child location.

    Maps to MLIR's `NameLoc`.  Contains the name identifier and the
    recursively-serialized child location (if any).
    """

    name: str
    child_location: LocationFrame | None = None


# Convenience alias used throughout the diagnostic pipeline.
LocationFrame = Union[FileLocation, NameLocation, None]


@dataclass
class MLIRDiagnostic:
    """A wrapper for mlir diagnostics."""

    message: str
    severity: str
    location: list[LocationFrame] = field(default_factory=list)


def error_diagnostics(diags: list[MLIRDiagnostic]) -> list[MLIRDiagnostic]:
    """Filter a list of diagnostics to errors only."""
    return [d for d in diags if "error" in d.severity.lower()]


@dataclass
class WaterError:
    """An error originating from the Water/Wave compilation pipeline."""

    message: str
    severity: str = "ERROR"
    error_diagnostics: list[str] = field(default_factory=list)
