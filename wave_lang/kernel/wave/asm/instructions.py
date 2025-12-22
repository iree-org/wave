# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
AMDGCN assembly instruction definitions and builders.

This module provides instruction classes that use the unified instruction
registry for consistent instruction definitions. Each class wraps the
registry lookup and provides backwards-compatible initialization.

The actual instruction definitions are in instruction_defs/common.yaml
and architecture-specific YAML files.
"""

from typing import List, Tuple, Union, Optional
from abc import ABC

from .instruction_registry import get_registry, InstructionDef, InstructionCategory


# ==============================================================================
# Base Classes
# ==============================================================================

class Instruction(ABC):
    """
    Base class for AMDGCN assembly instructions.
    
    Uses the unified instruction registry for definition lookup.
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


class LoadInstruction(MemoryInstruction):
    """Base class for load instructions."""
    pass


class StoreInstruction(MemoryInstruction):
    """Base class for store instructions."""
    pass


# ==============================================================================
# Helper Functions
# ==============================================================================

def _format_vreg(value: Union[int, str]) -> str:
    """Format a value as a VGPR reference."""
    if isinstance(value, int):
        return f"v{value}"
    return str(value)


def _format_sreg(value: Union[int, str]) -> str:
    """Format a value as an SGPR reference."""
    if isinstance(value, int):
        return f"s{value}"
    return str(value)


def _format_vreg_range(regs: Tuple[int, ...]) -> str:
    """Format a tuple of register indices as a VGPR range."""
    if len(regs) == 2:
        return f"v[{regs[0]}:{regs[1]}]"
    elif len(regs) == 4:
        return f"v[{regs[0]}:{regs[3]}]"
    return f"v[{regs[0]}:{regs[-1]}]"


def _format_sreg_range(regs: Tuple[int, ...]) -> str:
    """Format a tuple of register indices as an SGPR range."""
    if len(regs) == 2:
        return f"s[{regs[0]}:{regs[1]}]"
    elif len(regs) == 4:
        return f"s[{regs[0]}:{regs[3]}]"
    return f"s[{regs[0]}:{regs[-1]}]"


def _format_imm(value: int) -> str:
    """Format an immediate value."""
    if abs(value) > 0xFFFF:
        return f"0x{value & 0xFFFFFFFF:x}"
    return str(value)


# ==============================================================================
# Scalar Memory Instructions
# ==============================================================================

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
                _format_sreg_range(dst_regs),
                _format_sreg_range(src_regs),
                f"0x{offset:x}",
            ],
            comment,
        )


# ==============================================================================
# Buffer Memory Instructions
# ==============================================================================

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
        super().__init__(
            "buffer_load_dwordx4",
            [
                _format_vreg_range(dst_regs),
                _format_vreg(vindex_reg),
                _format_sreg_range(srd_regs),
                "0",
                "offen",
                f"offset:{offset}",
            ],
            comment,
        )

    def __str__(self) -> str:
        if not self.mnemonic:
            return f"    # {self.comment}" if self.comment else ""
        # Special formatting: comma after first 3, then space-separated
        if len(self.operands) >= 6:
            formatted = ", ".join(self.operands[:3]) + ", " + " ".join(self.operands[3:])
        else:
            formatted = ", ".join(self.operands)
        return f"    {self.mnemonic} {formatted}"


class BufferLoadDwordx2(LoadInstruction):
    """Load 2 dwords from buffer memory."""

    def __init__(
        self,
        dst_regs: Tuple[int, int],
        vindex_reg: Union[str, int],
        srd_regs: Tuple[int, int, int, int],
        offset: int,
        comment: str = None,
    ):
        super().__init__(
            "buffer_load_dwordx2",
            [
                _format_vreg_range(dst_regs),
                _format_vreg(vindex_reg),
                _format_sreg_range(srd_regs),
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
            formatted = ", ".join(self.operands[:3]) + ", " + " ".join(self.operands[3:])
        else:
            formatted = ", ".join(self.operands)
        return f"    {self.mnemonic} {formatted}"


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
        super().__init__(
            "buffer_store_dwordx4",
            [
                _format_vreg_range(src_regs),
                _format_vreg(vindex_reg),
                _format_sreg_range(srd_regs),
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
            formatted = ", ".join(self.operands[:3]) + ", " + " ".join(self.operands[3:])
        else:
            formatted = ", ".join(self.operands)
        return f"    {self.mnemonic} {formatted}"


class BufferStoreDword(StoreInstruction):
    """Store 1 dword to buffer memory."""

    def __init__(
        self,
        src_reg: int,
        vindex_reg: Union[str, int],
        srd_regs: Tuple[int, int, int, int],
        offset: int,
        comment: str = None,
    ):
        super().__init__(
            "buffer_store_dword",
            [
                _format_vreg(src_reg),
                _format_vreg(vindex_reg),
                _format_sreg_range(srd_regs),
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
            formatted = ", ".join(self.operands[:3]) + ", " + " ".join(self.operands[3:])
        else:
            formatted = ", ".join(self.operands)
        return f"    {self.mnemonic} {formatted}"


# ==============================================================================
# Buffer Load to LDS Instructions
# ==============================================================================

class BufferLoadDwordLds(LoadInstruction):
    """Load 1 dword from buffer memory directly to LDS.

    M0 must be set to the LDS destination offset before this instruction.
    """

    def __init__(
        self,
        vaddr_reg: Union[str, int],
        srd_regs: Tuple[int, int, int, int],
        soffset: Union[str, int] = 0,
        offset: int = 0,
        comment: str = None,
    ):
        soffset_str = str(soffset) if isinstance(soffset, int) else soffset
        operands = [
            _format_vreg(vaddr_reg),
            _format_sreg_range(srd_regs),
            soffset_str,
            "offen",
        ]
        if offset != 0:
            operands.append(f"offset:{offset}")
        operands.append("lds")
        super().__init__("buffer_load_dword", operands, comment)

    def __str__(self) -> str:
        if not self.mnemonic:
            return f"    # {self.comment}" if self.comment else ""
        if len(self.operands) >= 4:
            formatted = ", ".join(self.operands[:3]) + " " + " ".join(self.operands[3:])
        else:
            formatted = ", ".join(self.operands)
        return f"    {self.mnemonic} {formatted}"


class BufferLoadDwordx4Lds(LoadInstruction):
    """Load 4 dwords from buffer memory directly to LDS."""

    def __init__(
        self,
        vaddr_reg: Union[str, int],
        srd_regs: Tuple[int, int, int, int],
        soffset: Union[str, int] = 0,
        offset: int = 0,
        comment: str = None,
    ):
        soffset_str = str(soffset) if isinstance(soffset, int) else soffset
        operands = [
            _format_vreg(vaddr_reg),
            _format_sreg_range(srd_regs),
            soffset_str,
            "offen",
        ]
        if offset != 0:
            operands.append(f"offset:{offset}")
        operands.append("lds")
        super().__init__("buffer_load_dwordx4", operands, comment)

    def __str__(self) -> str:
        if not self.mnemonic:
            return f"    # {self.comment}" if self.comment else ""
        if len(self.operands) >= 4:
            formatted = ", ".join(self.operands[:3]) + " " + " ".join(self.operands[3:])
        else:
            formatted = ", ".join(self.operands)
        return f"    {self.mnemonic} {formatted}"


# ==============================================================================
# Scalar Move Instructions
# ==============================================================================

class SMovB32(ArithmeticInstruction):
    """Move 32-bit scalar value."""

    def __init__(
        self,
        destination_register: int,
        source_value: Union[int, str],
        comment: str = None,
    ):
        super().__init__(
            "s_mov_b32",
            [_format_sreg(destination_register), str(source_value)],
            comment,
        )


class SMovB32M0(ArithmeticInstruction):
    """Move 32-bit value to M0 register."""

    def __init__(
        self,
        source_value: Union[int, str],
        comment: str = None,
    ):
        super().__init__(
            "s_mov_b32",
            ["m0", str(source_value)],
            comment,
        )


class SMovkI32(ArithmeticInstruction):
    """Move 16-bit literal into scalar register (sign-extended)."""

    def __init__(self, destination_register: int, literal: int, comment: str = None):
        super().__init__(
            "s_movk_i32",
            [_format_sreg(destination_register), str(literal)],
            comment,
        )


# ==============================================================================
# Vector Move/Shift Instructions
# ==============================================================================

class VMovB32(ArithmeticInstruction):
    """Move 32-bit vector value."""

    def __init__(
        self,
        destination_register: int,
        source_value: Union[int, str],
        comment: str = None,
    ):
        super().__init__(
            "v_mov_b32",
            [_format_vreg(destination_register), str(source_value)],
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
            [_format_vreg(destination_register), str(shift_amount), _format_vreg(source_register)],
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
            [_format_vreg(destination_register), str(shift_amount), _format_vreg(source_register)],
            comment,
        )


class VLshlAddU32(ArithmeticInstruction):
    """Fused left-shift-and-add: dst = (src0 << shift) + src2."""

    def __init__(
        self,
        destination_register: int,
        shift_source: int,
        shift_amount: int,
        add_source: int,
        comment: str = None,
    ):
        super().__init__(
            "v_lshl_add_u32",
            [
                _format_vreg(destination_register),
                _format_vreg(shift_source),
                str(shift_amount),
                _format_vreg(add_source),
            ],
            comment,
        )


# ==============================================================================
# Vector Arithmetic Instructions
# ==============================================================================

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
                _format_vreg(destination_register),
                _format_vreg(source1_register),
                _format_vreg(source2_register),
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
                _format_vreg(destination_register),
                _format_vreg(source1_register),
                _format_vreg(source2_register),
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
            [_format_vreg(destination_register), _format_vreg(source1_register), source2_any],
            comment,
        )


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
            [_format_vreg(destination_register), src1, _format_vreg(source2_register)],
            comment,
        )


# ==============================================================================
# Lane Operations
# ==============================================================================

class VMbcntLoU32B32(ArithmeticInstruction):
    """Count active lanes in lower 32 bits."""

    def __init__(self, destination_register: int, mask: int, comment: str = None):
        super().__init__(
            "v_mbcnt_lo_u32_b32",
            [_format_vreg(destination_register), str(mask), "0"],
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
            [_format_vreg(destination_register), str(mask), _format_vreg(source_register)],
            comment,
        )


class VReadfirstlaneB32(ArithmeticInstruction):
    """Read value from first active lane of VGPR to SGPR."""

    def __init__(self, dst_sgpr: int, src_vgpr: int, comment: str = None):
        super().__init__(
            "v_readfirstlane_b32",
            [_format_sreg(dst_sgpr), _format_vreg(src_vgpr)],
            comment,
        )


# ==============================================================================
# DS (LDS) Instructions
# ==============================================================================

class DSWriteB32(MemoryInstruction):
    """Write 32 bits to LDS."""

    def __init__(self, addr_vreg: int, src_vreg: int, offset: int = 0, comment: str = None):
        self._offset = offset
        if offset != 0 and (offset < 0 or offset > 65535):
            raise ValueError(f"ds_write_b32 offset must be 0-65535, got {offset}")
        super().__init__(
            "ds_write_b32",
            [_format_vreg(addr_vreg), _format_vreg(src_vreg)],
            comment,
        )

    def __str__(self) -> str:
        base = super().__str__()
        if self._offset != 0:
            base += f" offset:{self._offset}"
        return base


class DSWriteB64(MemoryInstruction):
    """Write 64 bits to LDS."""

    def __init__(
        self, 
        addr_vreg: int, 
        src_vregs: Tuple[int, int],
        offset: int = 0, 
        comment: str = None,
    ):
        self._offset = offset
        if offset != 0 and (offset < 0 or offset > 65535):
            raise ValueError(f"ds_write_b64 offset must be 0-65535, got {offset}")
        super().__init__(
            "ds_write_b64",
            [_format_vreg(addr_vreg), _format_vreg_range(src_vregs)],
            comment,
        )

    def __str__(self) -> str:
        base = super().__str__()
        if self._offset != 0:
            base += f" offset:{self._offset}"
        return base


class DSWriteB128(MemoryInstruction):
    """Write 128 bits to LDS."""

    def __init__(
        self, 
        addr_vreg: int, 
        src_vregs: Tuple[int, int, int, int],
        offset: int = 0, 
        comment: str = None,
    ):
        self._offset = offset
        if offset != 0 and (offset < 0 or offset > 65535):
            raise ValueError(f"ds_write_b128 offset must be 0-65535, got {offset}")
        super().__init__(
            "ds_write_b128",
            [_format_vreg(addr_vreg), _format_vreg_range(src_vregs)],
            comment,
        )

    def __str__(self) -> str:
        base = super().__str__()
        if self._offset != 0:
            base += f" offset:{self._offset}"
        return base


class DSReadB64(MemoryInstruction):
    """Read 64 bits from LDS."""

    def __init__(
        self, 
        dst_vregs: Tuple[int, int], 
        addr_vreg: int,
        offset: int = 0, 
        comment: str = None,
    ):
        self._offset = offset
        if offset != 0 and (offset < 0 or offset > 65535):
            raise ValueError(f"ds_read_b64 offset must be 0-65535, got {offset}")
        super().__init__(
            "ds_read_b64",
            [_format_vreg_range(dst_vregs), _format_vreg(addr_vreg)],
            comment,
        )

    def __str__(self) -> str:
        base = super().__str__()
        if self._offset != 0:
            base += f" offset:{self._offset}"
        return base


# ==============================================================================
# MFMA Instructions
# ==============================================================================

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
        if use_vgpr:
            dest_str = f"v[{dest_base}:{dest_base+3}]"
            if c_sel == dest_base:
                acc_str = dest_str
            elif c_sel == 0:
                acc_str = "0"
            else:
                acc_str = f"v[{c_sel}:{c_sel+3}]"
        else:
            dest_str = f"a[{dest_base}:{dest_base+3}]"
            acc_str = str(c_sel)

        super().__init__(
            "v_mfma_f32_16x16x16_f16",
            [
                dest_str,
                _format_vreg_range(a_src_pair),
                _format_vreg_range(b_src_pair),
                acc_str,
            ],
            comment,
        )


class VAccvgprReadB32(ArithmeticInstruction):
    """Read 32-bit from AGPR to VGPR."""

    def __init__(self, dst_vreg: int, acc_idx: int, comment: str = None):
        super().__init__(
            "v_accvgpr_read_b32",
            [_format_vreg(dst_vreg), f"a{acc_idx}"],
            comment,
        )


# ==============================================================================
# Scalar Arithmetic Instructions
# ==============================================================================

class SAddU32(ArithmeticInstruction):
    """Scalar 32-bit unsigned addition."""

    def __init__(self, dst, src0, src1, comment: str = None):
        super().__init__(
            "s_add_u32",
            [str(dst), str(src0), str(src1)],
            comment,
        )


class SAndB32(ArithmeticInstruction):
    """Scalar 32-bit bitwise AND."""

    def __init__(
        self,
        dst: int,
        src0: Union[int, str],
        src1: Union[int, str],
        comment: str = None,
    ):
        dst_str = _format_sreg(dst) if isinstance(dst, int) else str(dst)
        src0_str = _format_sreg(src0) if isinstance(src0, int) else str(src0)
        src1_str = hex(src1) if isinstance(src1, int) else str(src1)
        super().__init__("s_and_b32", [dst_str, src0_str, src1_str], comment)


class SOrB32(ArithmeticInstruction):
    """Scalar 32-bit bitwise OR."""

    def __init__(
        self,
        dst: int,
        src0: Union[int, str],
        src1: Union[int, str],
        comment: str = None,
    ):
        dst_str = _format_sreg(dst) if isinstance(dst, int) else str(dst)
        src0_str = _format_sreg(src0) if isinstance(src0, int) else str(src0)
        src1_str = hex(src1) if isinstance(src1, int) else str(src1)
        super().__init__("s_or_b32", [dst_str, src0_str, src1_str], comment)


# ==============================================================================
# Control Flow Instructions
# ==============================================================================

class SBarrier(ControlFlowInstruction):
    """Barrier across wave/wg."""

    def __init__(self, comment: str = None):
        super().__init__("s_barrier", [], comment)


class SNop(ControlFlowInstruction):
    """Scalar no-op."""

    def __init__(self, count: int = 0, comment: str = None):
        super().__init__("s_nop", [str(count)], comment)


class SWaitcnt(WaitInstruction):
    """Wait for specific counters."""

    def __init__(self, counter: str, comment: str = None):
        super().__init__("s_waitcnt", [counter], comment)


class SEndpgm(ControlFlowInstruction):
    """End program execution."""

    def __init__(self, comment: str = None):
        super().__init__("s_endpgm", [], comment)


class SCmpLtU32(ControlFlowInstruction):
    """Compare less-than (unsigned 32-bit)."""

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


# ==============================================================================
# Instruction Builder Classes
# ==============================================================================

class InstructionBuilder:
    """Builder class for creating AMDGCN instructions."""

    @staticmethod
    def load_kernarg(
        destination_low: int,
        destination_high: int,
        offset: int,
        kernarg_ptr_sgprs: tuple = (0, 1),
    ) -> SLoadDwordx2:
        """Load kernel argument from memory."""
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
        return [
            Instruction("", [], f"load {vector_bytes}B"),
            BufferLoadDwordx4(
                destination_registers, vector_index_register, srd_registers, offset
            ),
            SWaitcnt("vmcnt(0)", "wait for load completion"),
        ]

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


# ==============================================================================
# Convenience Functions
# ==============================================================================

def emit_kernargs(
    num_args: int, kernarg_ptr_sgprs: tuple = (0, 1)
) -> List[Instruction]:
    """Emit instructions to load kernel arguments."""
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
    instructions = [
        Instruction("", [], f"SRD for {memref_ssa} (arg{argument_index})"),
    ]
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


# ==============================================================================
# Export
# ==============================================================================

__all__ = [
    # Base classes
    "Instruction",
    "MemoryInstruction",
    "ArithmeticInstruction",
    "ControlFlowInstruction",
    "WaitInstruction",
    "LoadInstruction",
    "StoreInstruction",
    # Memory instructions
    "SLoadDwordx2",
    "BufferLoadDwordx4",
    "BufferLoadDwordx2",
    "BufferStoreDwordx4",
    "BufferStoreDword",
    "BufferLoadDwordLds",
    "BufferLoadDwordx4Lds",
    "DSWriteB32",
    "DSWriteB64",
    "DSWriteB128",
    "DSReadB64",
    # Scalar instructions
    "SMovB32",
    "SMovB32M0",
    "SMovkI32",
    "SAddU32",
    "SAndB32",
    "SOrB32",
    # Vector instructions
    "VMovB32",
    "VLshlRevB32",
    "VLshrrevB32",
    "VLshlAddU32",
    "VMulLoU32",
    "VAddU32",
    "VAddU32Any",
    "VAndB32",
    "VMbcntLoU32B32",
    "VMbcntHiU32B32",
    "VReadfirstlaneB32",
    # MFMA instructions
    "VMfmaF32_16x16x16F16",
    "VAccvgprReadB32",
    # Control flow
    "SBarrier",
    "SNop",
    "SWaitcnt",
    "SEndpgm",
    "SCmpLtU32",
    "SCmpLeU32",
    "SCBranchSCC1",
    "SBranch",
    # Builders
    "InstructionBuilder",
    "emit_kernargs",
    "emit_srd_setup",
    "emit_vector_load_store",
]
