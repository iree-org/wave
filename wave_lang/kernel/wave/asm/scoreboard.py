# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Scoreboard-based Hazard Detection and Instruction Scheduling.

Tracks outstanding operations and their readiness to detect:
- RAW (Read-After-Write) hazards
- WAW (Write-After-Write) hazards
- Structural hazards (resource conflicts)

Provides scheduling hints to:
- Insert minimal waits (s_waitcnt, s_nop)
- Reorder independent instructions to fill bubbles
- Track VMEM, LGKM, MFMA, and VALU readiness
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .latency_provider import LatencyProvider
from .instruction_categories import InstructionCategory, categorize_instruction


@dataclass
class PendingInstruction:
    """Represents an instruction that has been issued but not completed."""

    cycle: int  # Cycle when issued
    instruction: str  # Instruction name (e.g., "buffer_load_dwordx4")
    category: InstructionCategory
    latency: float  # Expected latency in cycles
    writes: Set[str] = field(
        default_factory=set
    )  # Resources written (e.g., "v0", "s4")
    reads: Set[str] = field(default_factory=set)  # Resources read
    ticket: Optional[int] = None  # VMEM/LGKM ticket for wait tracking

    def ready_at_cycle(self) -> int:
        """Cycle when this instruction's result will be ready."""
        return self.cycle + int(self.latency)


class Scoreboard:
    """
    Tracks outstanding instructions and detects hazards.

    Maintains separate queues for different instruction categories
    and provides scheduling decisions.
    """

    def __init__(self, latency_provider: LatencyProvider):
        """Initialize scoreboard."""
        self.latency_provider = latency_provider

    def __init__(self, latency_provider: LatencyProvider):
        """Initialize scoreboard."""
        self.latency_provider = latency_provider
        self.current_cycle = 0
        self.pending: Dict[InstructionCategory, List[PendingInstruction]] = {
            cat: [] for cat in InstructionCategory
        }
        self.last_write: Dict[str, int] = {}
        self.last_read: Dict[str, int] = {}

    def _categorize_instruction(self, instruction: str) -> InstructionCategory:
        """
        Categorize an instruction.

        Args:
            instruction: Instruction name

        Returns:
            InstructionCategory
        """
        return categorize_instruction(instruction)

    def issue(
        self,
        instruction: str,
        writes: Optional[Set[str]] = None,
        reads: Optional[Set[str]] = None,
        ticket: Optional[int] = None,
    ) -> PendingInstruction:
        """
        Issue an instruction and add to scoreboard.

        Args:
            instruction: Instruction name
            writes: Set of resources written (e.g., {"v0", "v1"})
            reads: Set of resources read
            ticket: Optional VMEM/LGKM ticket

        Returns:
            PendingInstruction object
        """
        category = self._categorize_instruction(instruction)
        latency = self.latency_provider.get_latency(instruction)

        pending_inst = PendingInstruction(
            cycle=self.current_cycle,
            instruction=instruction,
            category=category,
            latency=latency,
            writes=writes or set(),
            reads=reads or set(),
            ticket=ticket,
        )

        self.pending[category].append(pending_inst)

        # Update register state
        for reg in pending_inst.writes:
            self.last_write[reg] = self.current_cycle
        for reg in pending_inst.reads:
            self.last_read[reg] = self.current_cycle

        return pending_inst

    def advance(self, cycles: int = 1):
        """
        Advance the scoreboard by a number of cycles.

        Retires completed instructions.

        Args:
            cycles: Number of cycles to advance
        """
        self.current_cycle += cycles

        # Retire completed instructions
        for category, pending_list in self.pending.items():
            # Keep only instructions that are not ready yet
            self.pending[category] = [
                inst
                for inst in pending_list
                if inst.ready_at_cycle() > self.current_cycle
            ]

    def check_hazard(
        self, reads: Optional[Set[str]] = None, writes: Optional[Set[str]] = None
    ) -> Optional[Tuple[str, int]]:
        """
        Check for hazards with pending instructions.

        Args:
            reads: Resources to be read
            writes: Resources to be written

        Returns:
            Tuple of (hazard_type, cycles_needed) or None if no hazard
        """
        reads = reads or set()
        writes = writes or set()

        max_cycles_needed = 0
        hazard_type = None

        # Check all pending instructions
        for category, pending_list in self.pending.items():
            for inst in pending_list:
                # RAW hazard: We read what they write
                if reads & inst.writes:
                    cycles_needed = inst.ready_at_cycle() - self.current_cycle
                    if cycles_needed > max_cycles_needed:
                        max_cycles_needed = cycles_needed
                        hazard_type = "RAW"

                # WAW hazard: We write what they write
                if writes & inst.writes:
                    cycles_needed = inst.ready_at_cycle() - self.current_cycle
                    if cycles_needed > max_cycles_needed:
                        max_cycles_needed = cycles_needed
                        hazard_type = "WAW"

                # WAR hazard (less common in out-of-order execution)
                # We write what they read - usually not a problem

        if max_cycles_needed > 0:
            return (hazard_type, max_cycles_needed)
        return None

    def get_vmem_wait_count(self, ticket: int) -> Optional[int]:
        """
        Get vmcnt value needed to wait for a specific VMEM ticket.

        Args:
            ticket: VMEM ticket to wait for

        Returns:
            vmcnt value or None if ticket already complete
        """
        # Find all pending VMEM instructions with tickets > ticket
        pending_vmem = self.pending[InstructionCategory.VMEM]
        newer_vmem = [
            inst
            for inst in pending_vmem
            if inst.ticket is not None and inst.ticket > ticket
        ]

        if not newer_vmem:
            # If ticket is in pending list, need to wait for all
            has_ticket = any(inst.ticket == ticket for inst in pending_vmem)
            if has_ticket:
                return 0  # Wait for all pending VMEM
            else:
                return None  # Ticket already complete
        else:
            return len(newer_vmem)  # Wait allows newer ops to complete

    def get_lgkm_wait_count(self, ticket: int) -> Optional[int]:
        """
        Get lgkmcnt value needed to wait for a specific LGKM ticket.

        Similar to VMEM but for LDS/scalar memory.

        Args:
            ticket: LGKM ticket to wait for

        Returns:
            lgkmcnt value or None if ticket already complete
        """
        pending_lgkm = self.pending[InstructionCategory.LGKM]
        newer_lgkm = [
            inst
            for inst in pending_lgkm
            if inst.ticket is not None and inst.ticket > ticket
        ]

        if not newer_lgkm:
            has_ticket = any(inst.ticket == ticket for inst in pending_lgkm)
            if has_ticket:
                return 0
            else:
                return None
        else:
            return len(newer_lgkm)
