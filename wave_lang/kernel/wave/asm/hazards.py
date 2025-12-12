# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Hardware Hazard Detection and Mitigation for AMDGCN.

This module provides hazard detection and mitigation for AMD GPU architectures,
particularly gfx950 (CDNA 3) which has specific VALU hazards that require
software workarounds.

gfx950 VALU Hazards (from LLVM's GCNHazardRecognizer.cpp):
----------------------------------------------------------
The gfx950 architecture (hasVDecCoExecHazard) lacks hardware interlocks for
certain VALU instruction sequences:

1. VALU writes VGPR, then v_readfirstlane reads it -> 1 wait state needed
   (VALUWriteVGPRReadlaneRead = 1)

2. VALU writes SGPR, then VALU reads it -> 2 wait states needed
   (VALUWriteSGPRVALUReadSGPR = 2)

Mitigation: Insert s_nop 0 after v_add instructions to provide the required
wait state before v_readfirstlane can safely read the result.
"""

from typing import List, Optional


class HazardDetector:
    """
    Detects and mitigates hardware hazards for AMDGCN code generation.

    This class tracks instruction sequences and inserts s_nop instructions
    to handle gfx950 VALU hazards.
    """

    # gfx950 hazard constants from LLVM's GCNHazardRecognizer.cpp
    VALU_WRITE_VGPR_READLANE_READ = 1  # Wait states for VALU->readfirstlane
    VALU_WRITE_SGPR_VALU_READ_SGPR = 2  # Wait states for VALU->SGPR->VALU

    # Instructions that trigger VALU hazards when followed by v_readfirstlane
    # We apply s_nop mitigation to v_add instructions used in address
    # computations before v_readfirstlane for M0.
    HAZARDOUS_VALU_OPS = frozenset(
        [
            "v_add_u32",
            "v_add_co_u32",
            "v_add_nc_u32",
        ]
    )

    def __init__(self):
        """Initialize hazard detector."""
        pass

    def check_valu_hazard(self, instruction: str) -> bool:
        """
        Check if an instruction may cause a VALU hazard.

        On gfx950, VALU instructions that write VGPRs can cause hazards
        if followed by v_readfirstlane reading those VGPRs.

        Args:
            instruction: The instruction string (e.g., "v_add_u32 v5, v3, v4")

        Returns:
            True if this instruction may need hazard mitigation
        """
        instr_lower = instruction.strip().lower()
        # Extract the mnemonic (first word after any leading whitespace)
        parts = instr_lower.split()
        if not parts:
            return False
        mnemonic = parts[0]
        return mnemonic in self.HAZARDOUS_VALU_OPS

    def get_mitigation(self, instruction: str) -> Optional[str]:
        """
        Get the hazard mitigation instruction for a given instruction.

        Args:
            instruction: The instruction that was just emitted

        Returns:
            Mitigation instruction string, or None if no mitigation needed
        """
        if not self.check_valu_hazard(instruction):
            return None

        # s_nop 0 provides 1 wait cycle for VALU->readfirstlane hazard
        return "    s_nop 0"

    def get_mitigations(self, instruction: str) -> List[str]:
        """
        Get list of mitigation instructions (for compatibility).

        Args:
            instruction: The instruction that was just emitted

        Returns:
            List of mitigation instruction strings (may be empty)
        """
        mitigation = self.get_mitigation(instruction)
        return [mitigation] if mitigation else []


# Convenience function for simple usage
def needs_valu_hazard_mitigation(instruction: str) -> bool:
    """
    Check if an instruction needs VALU hazard mitigation on gfx950.

    Args:
        instruction: The instruction string

    Returns:
        True if the instruction may cause VALU hazards
    """
    instr_lower = instruction.strip().lower()
    parts = instr_lower.split()
    if not parts:
        return False
    mnemonic = parts[0]
    return mnemonic in HazardDetector.HAZARDOUS_VALU_OPS
