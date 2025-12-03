# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Instruction categorization for AMDGCN assembly.

Shared utility for categorizing instructions by their functional class
(VMEM, LGKM, VALU, SALU, MFMA, etc.), used by both the scoreboard and
the assembler emitter for consistent ticketing and hazard detection.
"""

from enum import Enum


class InstructionCategory(Enum):
    """Categories of instructions for scoreboarding and ticketing."""

    VMEM = "vmem"  # Vector memory (buffer_load/store)
    LGKM = "lgkm"  # LDS/GDS/scalar memory/messages
    VALU = "valu"  # Vector ALU
    SALU = "salu"  # Scalar ALU
    MFMA = "mfma"  # Matrix FMA
    OTHER = "other"  # Unknown/other


def categorize_instruction(instruction: str) -> InstructionCategory:
    """
    Categorize an instruction by its mnemonic.

    Uses fast prefix-based rules to classify instructions into functional
    categories. This categorization is used for both ticketing (VMEM/LGKM
    operations need tickets) and hazard detection (scoreboard tracking).

    Args:
        instruction: Instruction mnemonic (e.g., "buffer_load_dwordx4", "ds_read_b64")

    Returns:
        InstructionCategory for the instruction

    Examples:
        >>> categorize_instruction("buffer_load_dwordx4")
        InstructionCategory.VMEM
        >>> categorize_instruction("ds_write_b32")
        InstructionCategory.LGKM
        >>> categorize_instruction("v_add_u32")
        InstructionCategory.VALU
        >>> categorize_instruction("s_mov_b32")
        InstructionCategory.SALU
        >>> categorize_instruction("v_mfma_f32_16x16x16_f16")
        InstructionCategory.MFMA
    """
    # Vector memory operations (buffer loads/stores)
    if instruction.startswith("buffer_"):
        return InstructionCategory.VMEM

    # LDS/GDS/scalar memory operations
    # Includes: ds_read, ds_write, s_load, messages
    elif instruction.startswith("ds_") or instruction.startswith("s_load"):
        return InstructionCategory.LGKM

    # Matrix FMA operations (must check before generic v_ prefix)
    elif instruction.startswith("v_mfma"):
        return InstructionCategory.MFMA

    # Vector ALU operations
    elif instruction.startswith("v_"):
        return InstructionCategory.VALU

    # Scalar ALU operations
    elif instruction.startswith("s_"):
        return InstructionCategory.SALU

    # Everything else (NOPs, waits, barriers, etc.)
    else:
        return InstructionCategory.OTHER
