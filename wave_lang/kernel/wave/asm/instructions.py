# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
AMDGCN assembly instruction definitions.

This module provides base instruction classes and the instruction registry.
Actual instruction definitions are in instruction_defs/*.yaml files.

Use the UnifiedEmitter for instruction emission - it provides a consistent
API backed by the YAML-based instruction registry.
"""

from typing import List, Optional
from abc import ABC

from .instruction_registry import get_registry, InstructionDef, InstructionCategory


# ==============================================================================
# Base Classes
# ==============================================================================

class Instruction(ABC):
    """
    Base class for AMDGCN assembly instructions.
    
    This class is kept for backwards compatibility with test code that creates
    mock instructions. For actual code emission, use the UnifiedEmitter.
    """

    def __init__(
        self, 
        mnemonic: str, 
        operands: List[str] = None, 
        comment: str = None,
        instr_def: InstructionDef = None,
    ):
        self.mnemonic = mnemonic
        self.operands = operands or []
        self.comment = comment
        self._instr_def = instr_def or get_registry().get_by_mnemonic(mnemonic)

    def __str__(self) -> str:
        """Generate the assembly instruction string."""
        if not self.mnemonic:
            # This is a comment-only instruction
            return f"    # {self.comment}" if self.comment else ""

        parts = [self.mnemonic]

        if self.operands:
            parts.append(", ".join(self.operands))

        result = "    " + " ".join(parts)

        return result
    
    @property
    def latency(self) -> int:
        """Get instruction latency from registry."""
        return self._instr_def.latency if self._instr_def else 1
    
    @property
    def category(self) -> Optional[InstructionCategory]:
        """Get instruction category from registry."""
        return self._instr_def.category if self._instr_def else None


# ==============================================================================
# Export
# ==============================================================================

__all__ = [
    # Base class (for backwards compatibility)
    "Instruction",
]
