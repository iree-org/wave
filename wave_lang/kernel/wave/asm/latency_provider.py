# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Latency Provider Interface.

Provides instruction latency and throughput information to the ASM emitter
for hazard detection and scheduling decisions.

"""

from typing import Optional
from pathlib import Path

from .latency_database import (
    LatencyDatabaseManager,
    InstructionTiming,
)


class LatencyProvider:
    """Provides instruction latency and throughput information."""

    def __init__(self, arch: str):
        """Initialize latency provider."""
        self.arch = arch
        db_root = str(Path(__file__).parent / "latency_db")
        self.db_manager = LatencyDatabaseManager(db_root=db_root)
        self.db = self.db_manager.load_architecture(arch)

    def get_timing(self, instruction: str) -> Optional[InstructionTiming]:
        """Get timing for an instruction."""
        return self.db.get_timing(instruction)

    def get_latency(self, instruction: str) -> Optional[float]:
        """Get latency in cycles for an instruction."""
        timing = self.get_timing(instruction)
        return timing.latency_cycles if timing else None

    def get_throughput(self, instruction: str) -> Optional[float]:
        """Get throughput in operations per cycle."""
        timing = self.get_timing(instruction)
        return timing.throughput_ops_per_cycle if timing else None
