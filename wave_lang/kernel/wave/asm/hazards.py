# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Hardware Hazard Detection and Mitigation for AMDGCN.

This module provides hazard detection and mitigation for AMD GPU architectures,
particularly gfx940+ (CDNA 3) which has specific VALU hazards that require
software workarounds.

gfx940+ VALU Hazards (from LLVM's GCNHazardRecognizer.cpp):
-----------------------------------------------------------
The gfx940+ architectures (gfx940, gfx941, gfx942/MI300, gfx950) have
hasVDecCoExecHazard=true and lack hardware interlocks for certain VALU
instruction sequences:

1. VALU writes VGPR, then v_readfirstlane reads it -> 1 wait state needed
   (VALUWriteVGPRReadlaneRead = 1)

2. VALU writes SGPR, then VALU reads it -> 2 wait states needed
   (VALUWriteSGPRVALUReadSGPR = 2)

Mitigation: Insert s_nop 0 after v_add instructions to provide the required
wait state before v_readfirstlane can safely read the result.
"""

from typing import List, Optional

from .instructions import Instruction, SNop


class HazardDetector:
    """
    Detects and mitigates hardware hazards for AMDGCN code generation.

    This class tracks instruction sequences and inserts s_nop instructions
    to handle gfx940+ VALU hazards (applies to gfx940, gfx941, gfx942, gfx950).
    """

    # gfx940+ hazard constants from LLVM's GCNHazardRecognizer.cpp
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

    def check_valu_hazard(self, mnemonic: str) -> bool:
        """
        Check if a mnemonic may cause a VALU hazard.

        On gfx940+ (including gfx942/MI300 and gfx950), VALU instructions
        that write VGPRs can cause hazards if followed by v_readfirstlane
        reading those VGPRs.

        Args:
            mnemonic: The instruction mnemonic (e.g., "v_add_u32")

        Returns:
            True if this instruction may need hazard mitigation
        """
        return mnemonic.lower() in self.HAZARDOUS_VALU_OPS

    def get_mitigation(self, mnemonic: str) -> Optional[Instruction]:
        """
        Get the hazard mitigation instruction for a given mnemonic.

        Args:
            mnemonic: The mnemonic of the instruction that was just emitted

        Returns:
            Mitigation instruction object, or None if no mitigation needed
        """
        if not mnemonic or not self.check_valu_hazard(mnemonic):
            return None

        # s_nop 0 provides 1 wait cycle for VALU->readfirstlane hazard
        return SNop(0)

    def get_mitigations(self, mnemonic: str) -> List[Instruction]:
        """
        Get list of mitigation instructions (for compatibility).

        Args:
            mnemonic: The mnemonic of the instruction that was just emitted

        Returns:
            List of mitigation instruction objects (may be empty)
        """
        mitigation = self.get_mitigation(mnemonic)
        return [mitigation] if mitigation else []
