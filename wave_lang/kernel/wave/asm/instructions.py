# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
AMDGCN assembly instruction definitions and builders.
"""

from typing import List, Tuple, Union
from abc import ABC


class Instruction(ABC):
    """Base class for AMDGCN assembly instructions."""

    def __init__(self, mnemonic: str, operands: List[str] = None, comment: str = None):
        self.mnemonic = mnemonic
        self.operands = operands or []
        self.comment = comment

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


class MemoryInstruction(Instruction):
    """Base class for memory-related instructions."""

    pass


class ArithmeticInstruction(Instruction):
    """Base class for arithmetic instructions."""

    pass


class ControlFlowInstruction(Instruction):
    """Base class for control flow instructions."""

    pass


class WaitInstruction(Instruction):
    """Base class for wait/synchronization instructions."""

    pass


# Memory Instructions
class LoadInstruction(MemoryInstruction):
    """Base class for load instructions."""

    pass


class StoreInstruction(MemoryInstruction):
    """Base class for store instructions."""

    pass


class SLoadDwordx2(LoadInstruction):
    """Load 2 dwords from scalar memory."""

    def __init__(
        self,
        dst_regs: Tuple[int, int],
        src_regs: Tuple[int, int],
        offset: int,
        comment: str = None,
    ):
        super().__init__(
            "s_load_dwordx2",
            [
                f"s[{dst_regs[0]}:{dst_regs[1]}]",
                f"s[{src_regs[0]}:{src_regs[1]}]",
                f"0x{offset:x}",
            ],
            comment,
        )


class BufferLoadDwordx4(LoadInstruction):
    """Load 4 dwords from buffer memory."""

    def __init__(
        self,
        dst_regs: Tuple[int, int, int, int],
        vindex_reg: str,
        srd_regs: Tuple[int, int, int, int],
        offset: int,
        comment: str = None,
    ):
        super().__init__(
            "buffer_load_dwordx4",
            [
                f"v[{dst_regs[0]}:{dst_regs[3]}]",
                vindex_reg,
                f"s[{srd_regs[0]}:{srd_regs[3]}]",
                "0",
                "offen",
                f"offset:{offset}",
            ],
            comment,
        )

    def __str__(self) -> str:
        """Generate the assembly instruction string with special formatting for buffer instructions."""
        if not self.mnemonic:
            return f"    # {self.comment}" if self.comment else ""

        # Special formatting for buffer instructions: comma after first 3 operands, then space
        if len(self.operands) >= 6:
            formatted_operands = (
                ", ".join(self.operands[:3]) + ", " + " ".join(self.operands[3:])
            )
        else:
            formatted_operands = ", ".join(self.operands)

        return f"    {self.mnemonic} {formatted_operands}"


class BufferStoreDwordx4(StoreInstruction):
    """Store 4 dwords to buffer memory."""

    def __init__(
        self,
        src_regs: Tuple[int, int, int, int],
        vindex_reg: str,
        srd_regs: Tuple[int, int, int, int],
        offset: int,
        comment: str = None,
    ):
        super().__init__(
            "buffer_store_dwordx4",
            [
                f"v[{src_regs[0]}:{src_regs[3]}]",
                vindex_reg,
                f"s[{srd_regs[0]}:{srd_regs[3]}]",
                "0",
                "offen",
                f"offset:{offset}",
            ],
            comment,
        )

    def __str__(self) -> str:
        """Generate the assembly instruction string with special formatting for buffer instructions."""
        if not self.mnemonic:
            return f"    # {self.comment}" if self.comment else ""

        # Special formatting for buffer instructions: comma after first 3 operands, then space
        if len(self.operands) >= 6:
            formatted_operands = (
                ", ".join(self.operands[:3]) + ", " + " ".join(self.operands[3:])
            )
        else:
            formatted_operands = ", ".join(self.operands)

        return f"    {self.mnemonic} {formatted_operands}"


# Arithmetic Instructions
class SMovB32(ArithmeticInstruction):
    """Move 32-bit scalar value."""

    def __init__(
        self,
        destination_register: int,
        source_value: Union[int, str],
        comment: str = None,
    ):
        super().__init__(
            "s_mov_b32", [f"s{destination_register}", str(source_value)], comment
        )


class VMbcntLoU32B32(ArithmeticInstruction):
    """Count active lanes in lower 32 bits."""

    def __init__(
        self, destination_register: int, source_register: int, comment: str = None
    ):
        super().__init__(
            "v_mbcnt_lo_u32_b32",
            [f"v{destination_register}", str(source_register), "0"],
            comment,
        )


class VMbcntHiU32B32(ArithmeticInstruction):
    """Count active lanes in upper 32 bits."""

    def __init__(
        self,
        destination_register: int,
        source_register: int,
        source2_register: int,
        comment: str = None,
    ):
        super().__init__(
            "v_mbcnt_hi_u32_b32",
            [f"v{destination_register}", str(source_register), f"v{source2_register}"],
            comment,
        )


class VLshlRevB32(ArithmeticInstruction):
    """Left shift with reversed operands."""

    def __init__(
        self,
        destination_register: int,
        shift_amount: int,
        source_register: int,
        comment: str = None,
    ):
        super().__init__(
            "v_lshlrev_b32",
            [f"v{destination_register}", str(shift_amount), f"v{source_register}"],
            comment,
        )


class VMovB32(ArithmeticInstruction):
    """Move 32-bit vector value."""

    def __init__(
        self,
        destination_register: int,
        source_value: Union[int, str],
        comment: str = None,
    ):
        super().__init__(
            "v_mov_b32", [f"v{destination_register}", str(source_value)], comment
        )


class VMulLoU32(ArithmeticInstruction):
    """Multiply unsigned 32-bit values, low result."""

    def __init__(
        self,
        destination_register: int,
        source1_register: int,
        source2_register: int,
        comment: str = None,
    ):
        super().__init__(
            "v_mul_lo_u32",
            [
                f"v{destination_register}",
                f"v{source1_register}",
                f"v{source2_register}",
            ],
            comment,
        )


class VAddU32(ArithmeticInstruction):
    """Add 32-bit values."""

    def __init__(
        self,
        destination_register: int,
        source1_register: int,
        source2_register: int,
        comment: str = None,
    ):
        super().__init__(
            "v_add_u32",
            [
                f"v{destination_register}",
                f"v{source1_register}",
                f"v{source2_register}",
            ],
            comment,
        )


# Wait Instructions
class SWaitcnt(WaitInstruction):
    """Wait for specific counters."""

    def __init__(self, counter: str, comment: str = None):
        super().__init__("s_waitcnt", [counter], comment)


# Control Flow Instructions
class SEndpgm(ControlFlowInstruction):
    """End program execution."""

    def __init__(self, comment: str = None):
        super().__init__("s_endpgm", [], comment)


# Instruction Builder Classes
class InstructionBuilder:
    """Builder class for creating AMDGCN instructions."""

    @staticmethod
    def load_kernarg(
        destination_low: int, destination_high: int, offset: int
    ) -> SLoadDwordx2:
        """Load kernel argument from memory."""
        return SLoadDwordx2(
            (destination_low, destination_high),
            (0, 1),  # s[0:1] contains kernarg pointer
            offset,
            f"Load kernarg at offset {offset}",
        )

    @staticmethod
    def setup_srd(
        srd_registers: Tuple[int, int, int, int],
        base_low: int,
        base_high: int,
        limit_bytes: int,
        srd_upper: str,
    ) -> List[SMovB32]:
        """Setup Shader Resource Descriptor (SRD)."""
        srd_register_0, srd_register_1, srd_register_2, srd_register_3 = srd_registers
        return [
            SMovB32(srd_register_0, f"s{base_low}", "SRD base address low"),
            SMovB32(srd_register_1, f"s{base_high}", "SRD base address high"),
            SMovB32(srd_register_2, limit_bytes, "SRD limit bytes"),
            SMovB32(srd_register_3, srd_upper, "SRD upper word"),
        ]

    @staticmethod
    def compute_lane_id(subgroup_size: int) -> List[Instruction]:
        """Compute lane ID for current thread."""
        return [
            Instruction("", [], f"lane id (0..{subgroup_size-1})"),
            VMbcntLoU32B32(0, -1, "count active lanes in lower 32 bits"),
            VMbcntHiU32B32(0, -1, 0, "count active lanes in upper 32 bits"),
        ]

    @staticmethod
    def compute_vector_offset(lane_id_register: int, shift_amount: int) -> VLshlRevB32:
        """Compute vector offset by shifting lane ID."""
        return VLshlRevB32(2, shift_amount, lane_id_register, "compute vector offset")

    @staticmethod
    def buffer_load(
        destination_registers: Tuple[int, int, int, int],
        vector_index_register: str,
        srd_registers: Tuple[int, int, int, int],
        offset: int,
        vector_bytes: int,
    ) -> List[Instruction]:
        """Load vector data from buffer."""
        instructions = []
        instructions.append(Instruction("", [], f"load {vector_bytes}B"))
        instructions.append(
            BufferLoadDwordx4(
                destination_registers, vector_index_register, srd_registers, offset
            )
        )
        instructions.append(SWaitcnt("vmcnt(0)", "wait for load completion"))
        return instructions

    @staticmethod
    def buffer_store(
        source_registers: Tuple[int, int, int, int],
        vector_index_register: str,
        srd_registers: Tuple[int, int, int, int],
        offset: int,
        vector_bytes: int,
    ) -> Instruction:
        """Store vector data to buffer."""
        return BufferStoreDwordx4(
            source_registers,
            vector_index_register,
            srd_registers,
            offset,
            f"store {vector_bytes}B",
        )

    @staticmethod
    def end_program() -> SEndpgm:
        """End the program."""
        return SEndpgm()


# Convenience functions for common instruction patterns
def emit_kernargs(num_args: int) -> List[Instruction]:
    """Emit instructions to load kernel arguments."""
    instructions = []
    for i in range(num_args):
        instructions.append(
            InstructionBuilder.load_kernarg(2 + i * 2, 3 + i * 2, i * 8)
        )
    instructions.append(SWaitcnt("lgkmcnt(0)", "wait for all kernarg loads"))
    return instructions


def emit_srd_setup(
    memref_ssa: str,
    argument_index: int,
    limit_bytes: int,
    srd_registers: Tuple[int, int, int, int],
    base_registers: Tuple[int, int],
) -> List[Instruction]:
    """Emit instructions to setup SRD for a memory reference."""
    instructions = []
    instructions.append(
        Instruction("", [], f"SRD for {memref_ssa} (arg{argument_index})")
    )
    srd_setup = InstructionBuilder.setup_srd(
        srd_registers, base_registers[0], base_registers[1], limit_bytes, "Srd127_96"
    )
    instructions.extend(srd_setup)
    return instructions


def emit_vector_load_store(
    operation: str,
    registers: Tuple[int, int, int, int],
    vector_index_register: str,
    srd_registers: Tuple[int, int, int, int],
    offset: int,
    vector_bytes: int,
) -> List[Instruction]:
    """Emit vector load or store instructions."""
    if operation == "load":
        return InstructionBuilder.buffer_load(
            registers, vector_index_register, srd_registers, offset, vector_bytes
        )
    elif operation == "store":
        return [
            InstructionBuilder.buffer_store(
                registers, vector_index_register, srd_registers, offset, vector_bytes
            )
        ]
    else:
        raise ValueError(f"Unknown operation: {operation}")


# Export commonly used classes and functions
__all__ = [
    "Instruction",
    "MemoryInstruction",
    "ArithmeticInstruction",
    "ControlFlowInstruction",
    "WaitInstruction",
    "LoadInstruction",
    "StoreInstruction",
    "SLoadDwordx2",
    "BufferLoadDwordx4",
    "BufferStoreDwordx4",
    "SMovB32",
    "VMbcntLoU32B32",
    "VMbcntHiU32B32",
    "VLshlRevB32",
    "VMovB32",
    "VMulLoU32",
    "VAddU32",
    "SWaitcnt",
    "SEndpgm",
    "InstructionBuilder",
    "emit_kernargs",
    "emit_srd_setup",
    "emit_vector_load_store",
]
