# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Instruction Latency and Throughput Database.

Stores measured or documented latency/throughput values per architecture.

Database Schema:
{
  "architecture": "gfx942",
  "version": "1.0",
  "timestamp": "2025-11-01T12:00:00Z",
  "instructions": {
    "buffer_load_dword": {
      "latency_cycles": 80,
      "throughput_ops_per_cycle": 0.25,
      "measurement_source": "measured",
      "notes": "Average latency from microbench"
    },
    ...
  },
  "instruction_classes": {
    "vmem_load": {
      "latency_cycles": 80,
      "throughput_ops_per_cycle": 0.25
    },
    ...
  }
}
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from pathlib import Path
from enum import Enum


class MeasurementSource(Enum):
    """Source of latency/throughput data."""

    MEASURED = "measured"  # From microbenchmarks
    ISA_MANUAL = "isa_manual"  # From AMD ISA documentation
    ESTIMATED = "estimated"  # Conservative estimate
    PROFILED = "profiled"  # From rocprof/rocblas data
    LLVM_CODEGEN = "llvm_codegen"  # From LLVM compiler codegen patterns


@dataclass
class InstructionTiming:
    """Timing characteristics for an instruction."""

    latency_cycles: float
    throughput_ops_per_cycle: float
    measurement_source: MeasurementSource
    notes: str
    stddev_latency: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["measurement_source"] = self.measurement_source.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "InstructionTiming":
        """Create from dictionary."""
        return cls(
            latency_cycles=d["latency_cycles"],
            throughput_ops_per_cycle=d["throughput_ops_per_cycle"],
            measurement_source=MeasurementSource(d["measurement_source"]),
            notes=d.get("notes", ""),
            stddev_latency=d.get("stddev_latency", 0.0),
        )


@dataclass
class ArchitectureLatencyDB:
    """Latency database for a specific architecture."""

    architecture: str
    version: str
    timestamp: str
    instructions: Dict[str, InstructionTiming]
    instruction_classes: Dict[str, InstructionTiming]

    def get_timing(self, instruction: str) -> Optional[InstructionTiming]:
        """Get timing for an instruction."""
        return self.instructions.get(instruction)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "architecture": self.architecture,
            "version": self.version,
            "timestamp": self.timestamp,
            "instructions": {k: v.to_dict() for k, v in self.instructions.items()},
            "instruction_classes": {
                k: v.to_dict() for k, v in self.instruction_classes.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArchitectureLatencyDB":
        """Create from dictionary."""
        instructions = {
            name: InstructionTiming.from_dict(timing_dict)
            for name, timing_dict in d["instructions"].items()
        }
        instruction_classes = {
            name: InstructionTiming.from_dict(timing_dict)
            for name, timing_dict in d.get("instruction_classes", {}).items()
        }
        return cls(
            architecture=d["architecture"],
            version=d["version"],
            timestamp=d["timestamp"],
            instructions=instructions,
            instruction_classes=instruction_classes,
        )

    def save(self, path: str):
        """Save database to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ArchitectureLatencyDB":
        """Load database from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class LatencyDatabaseManager:
    """Manages latency databases for multiple architectures."""

    def __init__(self, db_root: str):
        """Initialize database manager."""
        self.db_root = Path(db_root)
        self.databases: Dict[str, ArchitectureLatencyDB] = {}

    def load_architecture(self, arch: str) -> ArchitectureLatencyDB:
        """Load database for a specific architecture."""
        if arch in self.databases:
            return self.databases[arch]

        db_path = self.db_root / f"{arch}.json"
        db = ArchitectureLatencyDB.load(str(db_path))
        self.databases[arch] = db
        return db

    def save_architecture(self, arch: str, db: ArchitectureLatencyDB):
        """Save database for a specific architecture."""
        self.db_root.mkdir(parents=True, exist_ok=True)
        db_path = self.db_root / f"{arch}.json"
        db.save(str(db_path))
        self.databases[arch] = db
