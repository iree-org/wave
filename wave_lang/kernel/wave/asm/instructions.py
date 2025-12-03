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
        vindex_reg: Union[str, int],
        srd_regs: Tuple[int, int, int, int],
        offset: int,
        comment: str = None,
    ):
        # Format vindex_reg as string if it's an integer
        if isinstance(vindex_reg, int):
            vindex_reg = f"v{vindex_reg}"
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
        vindex_reg: Union[str, int],
        srd_regs: Tuple[int, int, int, int],
        offset: int,
        comment: str = None,
    ):
        # Format vindex_reg as string if it's an integer
        if isinstance(vindex_reg, int):
            vindex_reg = f"v{vindex_reg}"
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


# Additional Buffer Instructions
class BufferLoadDwordx2(LoadInstruction):
    """Load 2 dwords (8 bytes) from buffer memory."""

    def __init__(
        self,
        dst_regs: Tuple[int, int],
        vindex_reg: Union[str, int],
        srd_regs: Tuple[int, int, int, int],
        offset: int,
        comment: str = None,
    ):
        # Format vindex_reg as string if it's an integer
        if isinstance(vindex_reg, int):
            vindex_reg = f"v{vindex_reg}"
        super().__init__(
            "buffer_load_dwordx2",
            [
                f"v[{dst_regs[0]}:{dst_regs[1]}]",
                vindex_reg,
                f"s[{srd_regs[0]}:{srd_regs[3]}]",
                "0",
                "offen",
                f"offset:{offset}",
            ],
            comment,
        )

    def __str__(self) -> str:
        if not self.mnemonic:
            return f"    # {self.comment}" if self.comment else ""
        if len(self.operands) >= 6:
            formatted_operands = (
                ", ".join(self.operands[:3]) + ", " + " ".join(self.operands[3:])
            )
        else:
            formatted_operands = ", ".join(self.operands)
        return f"    {self.mnemonic} {formatted_operands}"


class BufferStoreDword(StoreInstruction):
    """Store 1 dword (4 bytes) to buffer memory."""

    def __init__(
        self,
        src_reg: int,
        vindex_reg: Union[str, int],
        srd_regs: Tuple[int, int, int, int],
        offset: int,
        comment: str = None,
    ):
        # Format vindex_reg as string if it's an integer
        if isinstance(vindex_reg, int):
            vindex_reg = f"v{vindex_reg}"
        super().__init__(
            "buffer_store_dword",
            [
                f"v{src_reg}",
                vindex_reg,
                f"s[{srd_regs[0]}:{srd_regs[3]}]",
                "0",
                "offen",
                f"offset:{offset}",
            ],
            comment,
        )

    def __str__(self) -> str:
        if not self.mnemonic:
            return f"    # {self.comment}" if self.comment else ""
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


class SMovkI32(ArithmeticInstruction):
    """Move 16-bit literal into scalar register (sign-extended)."""

    def __init__(self, destination_register: int, literal: int, comment: str = None):
        super().__init__(
            "s_movk_i32", [f"s{destination_register}", str(literal)], comment
        )


class VMbcntLoU32B32(ArithmeticInstruction):
    """Count active lanes in lower 32 bits."""

    def __init__(self, destination_register: int, mask: int, comment: str = None):
        super().__init__(
            "v_mbcnt_lo_u32_b32",
            [f"v{destination_register}", str(mask), "0"],
            comment,
        )


class VMbcntHiU32B32(ArithmeticInstruction):
    """Count active lanes in upper 32 bits."""

    def __init__(
        self,
        destination_register: int,
        mask: int,
        source_register: int,
        comment: str = None,
    ):
        super().__init__(
            "v_mbcnt_hi_u32_b32",
            [f"v{destination_register}", str(mask), f"v{source_register}"],
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


class VLshrrevB32(ArithmeticInstruction):
    """Logical right shift with reversed operands."""

    def __init__(
        self,
        destination_register: int,
        shift_amount: int,
        source_register: int,
        comment: str = None,
    ):
        super().__init__(
            "v_lshrrev_b32",
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


class VAddU32Any(ArithmeticInstruction):
    """Add 32-bit values allowing scalar or vector for source2."""

    def __init__(
        self,
        destination_register: int,
        source1_register: int,
        source2_any: str,
        comment: str = None,
    ):
        super().__init__(
            "v_add_u32",
            [f"v{destination_register}", f"v{source1_register}", source2_any],
            comment,
        )


# Bitwise AND
class VAndB32(ArithmeticInstruction):
    """Bitwise AND 32-bit."""

    def __init__(
        self,
        destination_register: int,
        source1: Union[int, str],
        source2_register: int,
        comment: str = None,
    ):
        src1 = str(source1) if isinstance(source1, int) else source1
        super().__init__(
            "v_and_b32",
            [f"v{destination_register}", src1, f"v{source2_register}"],
            comment,
        )


# DS (LDS) Instructions
class DSWriteB32(MemoryInstruction):
    """Write 32 bits to LDS."""

    def __init__(self, addr_vreg: int, src_vreg: int, comment: str = None):
        super().__init__("ds_write_b32", [f"v{addr_vreg}", f"v{src_vreg}"], comment)


class DSWriteB64(MemoryInstruction):
    """Write 64 bits to LDS."""

    def __init__(self, addr_vreg: int, src_vregs: Tuple[int, int], comment: str = None):
        super().__init__(
            "ds_write_b64",
            [f"v{addr_vreg}", f"v[{src_vregs[0]}:{src_vregs[1]}]"],
            comment,
        )


class DSWriteB128(MemoryInstruction):
    """Write 128 bits to LDS."""

    def __init__(
        self, addr_vreg: int, src_vregs: Tuple[int, int, int, int], comment: str = None
    ):
        super().__init__(
            "ds_write_b128",
            [f"v{addr_vreg}", f"v[{src_vregs[0]}:{src_vregs[3]}]"],
            comment,
        )


class DSReadB64(MemoryInstruction):
    """Read 64 bits from LDS."""

    def __init__(self, dst_vregs: Tuple[int, int], addr_vreg: int, comment: str = None):
        super().__init__(
            "ds_read_b64",
            [f"v[{dst_vregs[0]}:{dst_vregs[1]}]", f"v{addr_vreg}"],
            comment,
        )


# MFMA + AGPR spill
class VMfmaF32_16x16x16F16(ArithmeticInstruction):
    """Matrix FMA f32_16x16x16_f16 - supports both VGPR and AGPR destinations."""

    def __init__(
        self,
        dest_base: int,
        a_src_pair: Tuple[int, int],
        b_src_pair: Tuple[int, int],
        c_sel: int = 0,
        comment: str = None,
        use_vgpr: bool = True,
    ):
        # Determine destination register type (VGPR or AGPR)
        if use_vgpr:
            dest_str = f"v[{dest_base}:{dest_base+3}]"
            # For VGPR variant with accumulation, c_sel is the accumulator register base
            # If c_sel == dest_base, use dest_str (standard in-place accumulation)
            # If c_sel == 0, use literal 0 (zero initialization)
            if c_sel == dest_base:
                acc_str = dest_str  # In-place accumulation
            elif c_sel == 0:
                acc_str = "0"  # Zero initialization
            else:
                acc_str = f"v[{c_sel}:{c_sel+3}]"  # Different accumulator
        else:
            dest_str = f"a[{dest_base}:{dest_base+3}]"
            acc_str = str(c_sel)  # AGPR uses selector

        super().__init__(
            "v_mfma_f32_16x16x16_f16",
            [
                dest_str,
                f"v[{a_src_pair[0]}:{a_src_pair[1]}]",
                f"v[{b_src_pair[0]}:{b_src_pair[1]}]",
                acc_str,
            ],
            comment,
        )


class VAccvgprReadB32(ArithmeticInstruction):
    """Read 32-bit from AGPR to VGPR."""

    def __init__(self, dst_vreg: int, acc_idx: int, comment: str = None):
        super().__init__("v_accvgpr_read_b32", [f"v{dst_vreg}", f"a{acc_idx}"], comment)


# Scalar control
class SBarrier(ControlFlowInstruction):
    """Barrier across wave/wg."""

    def __init__(self, comment: str = None):
        super().__init__("s_barrier", [], comment)


class SNop(ControlFlowInstruction):
    """Scalar no-op."""

    def __init__(self, count: int = 0, comment: str = None):
        super().__init__("s_nop", [str(count)], comment)


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


# Loop Control Flow Instructions
class SCmpLtU32(ControlFlowInstruction):
    """Compare less-than (unsigned 32-bit) - for scf.for [lower, upper) semantics."""

    def __init__(self, src0, src1, comment: str = None):
        super().__init__("s_cmp_lt_u32", [str(src0), str(src1)], comment)


class SCmpLeU32(ControlFlowInstruction):
    """Compare less-than-or-equal (unsigned 32-bit)."""

    def __init__(self, src0, src1, comment: str = None):
        super().__init__("s_cmp_le_u32", [str(src0), str(src1)], comment)


class SCBranchSCC1(ControlFlowInstruction):
    """Conditional branch if SCC==1."""

    def __init__(self, label: str, comment: str = None):
        super().__init__("s_cbranch_scc1", [label], comment)


class SBranch(ControlFlowInstruction):
    """Unconditional branch."""

    def __init__(self, label: str, comment: str = None):
        super().__init__("s_branch", [label], comment)


class SAddU32(ArithmeticInstruction):
    """Scalar 32-bit unsigned addition."""

    def __init__(self, dst, src0, src1, comment: str = None):
        super().__init__("s_add_u32", [str(dst), str(src0), str(src1)], comment)


# Instruction Builder Classes
class InstructionBuilder:
    """Builder class for creating AMDGCN instructions."""

    @staticmethod
    def load_kernarg(
        destination_low: int,
        destination_high: int,
        offset: int,
        kernarg_ptr_sgprs: tuple = (0, 1),
    ) -> SLoadDwordx2:
        """Load kernel argument from memory.

        Args:
            destination_low: Low SGPR for destination
            destination_high: High SGPR for destination
            offset: Offset in bytes
            kernarg_ptr_sgprs: Tuple of (low, high) SGPRs containing kernarg pointer
        """
        return SLoadDwordx2(
            (destination_low, destination_high),
            kernarg_ptr_sgprs,
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
def emit_kernargs(
    num_args: int, kernarg_ptr_sgprs: tuple = (0, 1)
) -> List[Instruction]:
    """Emit instructions to load kernel arguments.

    Args:
        num_args: Number of kernel arguments to load
        kernarg_ptr_sgprs: Tuple of (low, high) SGPRs containing kernarg pointer
    """
    instructions = []
    for i in range(num_args):
        instructions.append(
            InstructionBuilder.load_kernarg(
                2 + i * 2, 3 + i * 2, i * 8, kernarg_ptr_sgprs
            )
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
