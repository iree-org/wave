# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Kernel-level IR for whole-program register allocation.

This module defines a structured intermediate representation for entire AMDGCN
kernels, enabling a single linear-scan register allocator to manage all VGPRs
and SGPRs. The key insight is that MLIR provides SSA values, so we can build
a complete picture of register lifetimes before allocation.

Architecture:
    1. Operation handlers emit KInstr to KernelProgram using virtual regs
    2. Liveness analysis computes live ranges for all virtual regs
    3. Linear scan allocator assigns physical registers
    4. Renderer emits final assembly with physical register numbers

Virtual register types:
    - KVReg: Virtual VGPR (placeholder for physical vN)
    - KSReg: Virtual SGPR (placeholder for physical sN)
    - KPhysVReg/KPhysSReg: Precolored physical regs (ABI-mandated)

Operand types:
    - KImm: Immediate constant value
    - KRegRange: Contiguous register range (for pairs, quads, MFMA blocks)

Instruction representation:
    - KInstr: Single instruction with opcode, defs, uses, and constraints
    - KernelProgram: Ordered sequence of instructions + metadata
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, FrozenSet
from enum import Enum, auto


# =============================================================================
# Register Classes
# =============================================================================

class RegClass(Enum):
    """Register class for allocation."""
    VGPR = auto()  # Vector general-purpose register
    SGPR = auto()  # Scalar general-purpose register
    AGPR = auto()  # Accumulator register (MFMA)


# =============================================================================
# Virtual Registers
# =============================================================================

@dataclass(frozen=True)
class KVReg:
    """Virtual VGPR - placeholder for physical vN."""
    id: int
    
    def __repr__(self) -> str:
        return f"kv{self.id}"


@dataclass(frozen=True)
class KSReg:
    """Virtual SGPR - placeholder for physical sN."""
    id: int
    
    def __repr__(self) -> str:
        return f"ks{self.id}"


@dataclass(frozen=True)
class KPhysVReg:
    """Precolored physical VGPR - for ABI-mandated registers."""
    index: int
    
    def __repr__(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class KPhysSReg:
    """Precolored physical SGPR - for ABI-mandated registers."""
    index: int
    
    def __repr__(self) -> str:
        return f"s{self.index}"


# Type aliases
KVirtualReg = Union[KVReg, KSReg]
KPhysicalReg = Union[KPhysVReg, KPhysSReg]
KReg = Union[KVirtualReg, KPhysicalReg]


def is_virtual(reg: KReg) -> bool:
    """Check if register is virtual (needs allocation)."""
    return isinstance(reg, (KVReg, KSReg))


def is_vgpr(reg: KReg) -> bool:
    """Check if register is a VGPR (virtual or physical)."""
    return isinstance(reg, (KVReg, KPhysVReg))


def is_sgpr(reg: KReg) -> bool:
    """Check if register is an SGPR (virtual or physical)."""
    return isinstance(reg, (KSReg, KPhysSReg))


def get_reg_class(reg: KReg) -> RegClass:
    """Get the register class for a register."""
    if isinstance(reg, (KVReg, KPhysVReg)):
        return RegClass.VGPR
    elif isinstance(reg, (KSReg, KPhysSReg)):
        return RegClass.SGPR
    raise ValueError(f"Unknown register type: {type(reg)}")


# =============================================================================
# Register Ranges (for pairs, quads, MFMA accumulators)
# =============================================================================

@dataclass(frozen=True)
class KRegRange:
    """
    A contiguous range of registers of the same type.
    
    Used for instructions that require register pairs (ds_read_b64),
    quads (buffer_load_dwordx4), or larger blocks (MFMA accumulators).
    
    The base_reg determines the register class. The range spans
    [base, base+count) in physical register numbering after allocation.
    """
    base_reg: KReg  # Starting register (virtual or precolored)
    count: int      # Number of consecutive registers
    alignment: int = 1  # Required alignment (e.g., 2 for pairs, 4 for quads)
    
    def __post_init__(self):
        if self.count < 1:
            raise ValueError(f"Register range count must be >= 1, got {self.count}")
        if self.alignment < 1:
            raise ValueError(f"Alignment must be >= 1, got {self.alignment}")
    
    @property
    def reg_class(self) -> RegClass:
        return get_reg_class(self.base_reg)
    
    def __repr__(self) -> str:
        base = repr(self.base_reg)
        if self.count == 1:
            return base
        return f"{base}..+{self.count}"


# =============================================================================
# Immediate Values
# =============================================================================

@dataclass(frozen=True)
class KImm:
    """Immediate constant value."""
    value: int
    
    def __repr__(self) -> str:
        if -16 <= self.value <= 64:
            return str(self.value)
        return f"0x{self.value:x}"


@dataclass(frozen=True)
class KMemOffset:
    """Memory offset for load/store instructions."""
    bytes: int
    
    def __repr__(self) -> str:
        return f"offset:{self.bytes}"


# Type alias for all operand types
KOperand = Union[KReg, KRegRange, KImm, KMemOffset]


# =============================================================================
# Instruction Opcodes
# =============================================================================

class KOpcode(Enum):
    """Kernel IR opcodes covering all instructions needed by the ASM backend."""
    
    # Register moves
    V_MOV_B32 = auto()      # v_mov_b32 dst, src
    S_MOV_B32 = auto()      # s_mov_b32 dst, src
    S_MOV_B64 = auto()      # s_mov_b64 dst, src
    
    # Vector arithmetic
    V_ADD_U32 = auto()      # v_add_u32 dst, src1, src2
    V_SUB_U32 = auto()      # v_sub_u32 dst, src1, src2
    V_MUL_LO_U32 = auto()   # v_mul_lo_u32 dst, src1, src2
    V_MUL_HI_U32 = auto()   # v_mul_hi_u32 dst, src1, src2
    
    # Vector bitwise
    V_AND_B32 = auto()      # v_and_b32 dst, src1, src2
    V_OR_B32 = auto()       # v_or_b32 dst, src1, src2
    V_XOR_B32 = auto()      # v_xor_b32 dst, src1, src2
    
    # Vector shifts
    V_LSHLREV_B32 = auto()  # v_lshlrev_b32 dst, shift, src
    V_LSHRREV_B32 = auto()  # v_lshrrev_b32 dst, shift, src
    V_ASHRREV_I32 = auto()  # v_ashrrev_i32 dst, shift, src
    
    # Fused vector ops
    V_LSHL_ADD_U32 = auto() # v_lshl_add_u32 dst, src, shift, addend
    V_LSHL_OR_B32 = auto()  # v_lshl_or_b32 dst, src, shift, orend
    V_ADD_LSHL_U32 = auto() # v_add_lshl_u32 dst, src1, src2, shift
    V_OR3_B32 = auto()      # v_or3_b32 dst, src1, src2, src3
    
    # Bit field extraction
    V_BFE_U32 = auto()      # v_bfe_u32 dst, src, offset, width
    
    # Scalar arithmetic
    S_ADD_U32 = auto()      # s_add_u32 dst, src1, src2
    S_MUL_I32 = auto()      # s_mul_i32 dst, src1, src2
    S_LSHL_B32 = auto()     # s_lshl_b32 dst, src, shift
    S_LSHR_B32 = auto()     # s_lshr_b32 dst, src, shift
    S_AND_B32 = auto()      # s_and_b32 dst, src1, src2
    S_OR_B32 = auto()       # s_or_b32 dst, src1, src2
    
    # Scalar 64-bit
    S_ADD_U64 = auto()      # s_add_u64 dst, src1, src2
    S_LSHL_B64 = auto()     # s_lshl_b64 dst, src, shift
    
    # Memory loads
    S_LOAD_DWORD = auto()       # s_load_dword dst, base, offset
    S_LOAD_DWORDX2 = auto()     # s_load_dwordx2 dst, base, offset
    S_LOAD_DWORDX4 = auto()     # s_load_dwordx4 dst, base, offset
    BUFFER_LOAD_DWORD = auto()  # buffer_load_dword dst, vaddr, srd, soffset
    BUFFER_LOAD_DWORDX2 = auto()
    BUFFER_LOAD_DWORDX4 = auto()
    GLOBAL_LOAD_DWORD = auto()  # global_load_dword dst, vaddr, soffset
    GLOBAL_LOAD_DWORDX2 = auto()
    GLOBAL_LOAD_DWORDX4 = auto()
    
    # Memory stores
    BUFFER_STORE_DWORD = auto()
    BUFFER_STORE_DWORDX2 = auto()
    BUFFER_STORE_DWORDX4 = auto()
    GLOBAL_STORE_DWORD = auto()
    GLOBAL_STORE_DWORDX2 = auto()
    GLOBAL_STORE_DWORDX4 = auto()
    
    # LDS operations
    DS_READ_B32 = auto()    # ds_read_b32 dst, addr [offset]
    DS_READ_B64 = auto()    # ds_read_b64 dst, addr [offset]
    DS_READ_B128 = auto()   # ds_read_b128 dst, addr [offset]
    DS_WRITE_B32 = auto()   # ds_write_b32 addr, src [offset]
    DS_WRITE_B64 = auto()   # ds_write_b64 addr, src [offset]
    DS_WRITE_B128 = auto()  # ds_write_b128 addr, src [offset]
    
    # MFMA instructions
    V_MFMA_F32_16X16X16_F16 = auto()
    V_MFMA_F32_32X32X8_F16 = auto()
    V_MFMA_F16_16X16X16_F16 = auto()
    V_MFMA_F16_32X32X8_F16 = auto()
    
    # Pack/conversion
    V_CVT_F32_F16 = auto()
    V_CVT_F16_F32 = auto()
    V_PACK_B32_F16 = auto()
    V_CVT_PK_FP8_F32 = auto()
    
    # Control flow / sync
    S_WAITCNT = auto()
    S_BARRIER = auto()
    S_NOP = auto()
    S_ENDPGM = auto()
    
    # Readfirstlane / broadcast
    V_READFIRSTLANE_B32 = auto()
    
    # Pseudo-ops (internal use)
    COMMENT = auto()        # Emit a comment
    LABEL = auto()          # Define a label


# =============================================================================
# Instruction Constraints
# =============================================================================

@dataclass
class KInstrConstraints:
    """
    Constraints on instruction operands for register allocation.
    
    These constraints inform the allocator about:
    - Register class requirements (VGPR vs SGPR)
    - Alignment requirements (pairs, quads)
    - Precoloring (must use specific physical reg)
    """
    # Destination constraints
    dst_class: Optional[RegClass] = None     # Required class for destination
    dst_alignment: int = 1                   # Required alignment (1, 2, 4, ...)
    dst_size: int = 1                        # Number of consecutive regs needed
    
    # Source constraints (per-operand)
    src_classes: Tuple[Optional[RegClass], ...] = ()  # Required class per source
    
    # Special constraints
    same_as_src: Optional[int] = None        # dst must be same as src[N]
    tied_operands: Tuple[Tuple[int, int], ...] = ()  # (dst_idx, src_idx) pairs that must match
    
    # Memory constraints
    has_offset_field: bool = False           # Can encode immediate offset


def get_default_constraints(opcode: KOpcode) -> KInstrConstraints:
    """Get default constraints for an opcode."""
    # Vector ALU - dst is VGPR, sources can be VGPR or immediate
    valu_ops = {
        KOpcode.V_MOV_B32, KOpcode.V_ADD_U32, KOpcode.V_SUB_U32,
        KOpcode.V_MUL_LO_U32, KOpcode.V_MUL_HI_U32,
        KOpcode.V_AND_B32, KOpcode.V_OR_B32, KOpcode.V_XOR_B32,
        KOpcode.V_LSHLREV_B32, KOpcode.V_LSHRREV_B32, KOpcode.V_ASHRREV_I32,
        KOpcode.V_LSHL_ADD_U32, KOpcode.V_LSHL_OR_B32, KOpcode.V_ADD_LSHL_U32,
        KOpcode.V_OR3_B32, KOpcode.V_BFE_U32,
        KOpcode.V_CVT_F32_F16, KOpcode.V_CVT_F16_F32, KOpcode.V_PACK_B32_F16,
        KOpcode.V_CVT_PK_FP8_F32,
    }
    if opcode in valu_ops:
        return KInstrConstraints(dst_class=RegClass.VGPR)
    
    # Scalar ALU - dst is SGPR
    salu_ops = {
        KOpcode.S_MOV_B32, KOpcode.S_MOV_B64,
        KOpcode.S_ADD_U32, KOpcode.S_MUL_I32,
        KOpcode.S_LSHL_B32, KOpcode.S_LSHR_B32,
        KOpcode.S_AND_B32, KOpcode.S_OR_B32,
        KOpcode.S_ADD_U64, KOpcode.S_LSHL_B64,
    }
    if opcode in salu_ops:
        return KInstrConstraints(dst_class=RegClass.SGPR)
    
    # Scalar loads - dst is SGPR range
    if opcode == KOpcode.S_LOAD_DWORD:
        return KInstrConstraints(dst_class=RegClass.SGPR, dst_size=1)
    if opcode == KOpcode.S_LOAD_DWORDX2:
        return KInstrConstraints(dst_class=RegClass.SGPR, dst_size=2, dst_alignment=2)
    if opcode == KOpcode.S_LOAD_DWORDX4:
        return KInstrConstraints(dst_class=RegClass.SGPR, dst_size=4, dst_alignment=4)
    
    # Buffer/global loads - dst is VGPR range
    if opcode in {KOpcode.BUFFER_LOAD_DWORD, KOpcode.GLOBAL_LOAD_DWORD}:
        return KInstrConstraints(dst_class=RegClass.VGPR, dst_size=1)
    if opcode in {KOpcode.BUFFER_LOAD_DWORDX2, KOpcode.GLOBAL_LOAD_DWORDX2}:
        return KInstrConstraints(dst_class=RegClass.VGPR, dst_size=2, dst_alignment=2)
    if opcode in {KOpcode.BUFFER_LOAD_DWORDX4, KOpcode.GLOBAL_LOAD_DWORDX4}:
        return KInstrConstraints(dst_class=RegClass.VGPR, dst_size=4, dst_alignment=4)
    
    # LDS loads
    if opcode == KOpcode.DS_READ_B32:
        return KInstrConstraints(dst_class=RegClass.VGPR, dst_size=1, has_offset_field=True)
    if opcode == KOpcode.DS_READ_B64:
        return KInstrConstraints(dst_class=RegClass.VGPR, dst_size=2, dst_alignment=2, has_offset_field=True)
    if opcode == KOpcode.DS_READ_B128:
        return KInstrConstraints(dst_class=RegClass.VGPR, dst_size=4, dst_alignment=4, has_offset_field=True)
    
    # LDS writes have no destination (only use)
    if opcode in {KOpcode.DS_WRITE_B32, KOpcode.DS_WRITE_B64, KOpcode.DS_WRITE_B128}:
        return KInstrConstraints(has_offset_field=True)
    
    # MFMA - large accumulator ranges
    if opcode in {KOpcode.V_MFMA_F32_16X16X16_F16, KOpcode.V_MFMA_F16_16X16X16_F16}:
        return KInstrConstraints(dst_class=RegClass.VGPR, dst_size=4, dst_alignment=4)
    if opcode in {KOpcode.V_MFMA_F32_32X32X8_F16, KOpcode.V_MFMA_F16_32X32X8_F16}:
        return KInstrConstraints(dst_class=RegClass.VGPR, dst_size=16, dst_alignment=4)
    
    # Control flow / sync - no register constraints
    return KInstrConstraints()


# =============================================================================
# Instructions
# =============================================================================

@dataclass
class KInstr:
    """
    A single kernel IR instruction.
    
    Each instruction has:
    - opcode: What operation to perform
    - defs: Registers/ranges defined (written) by this instruction
    - uses: Operands read by this instruction
    - constraints: Allocation constraints
    - comment: Optional comment for debugging
    """
    opcode: KOpcode
    defs: Tuple[Union[KReg, KRegRange], ...]  # Defined registers
    uses: Tuple[KOperand, ...]                 # Used operands
    constraints: KInstrConstraints = field(default_factory=KInstrConstraints)
    comment: Optional[str] = None
    
    def __post_init__(self):
        # Auto-set constraints if not provided
        if self.constraints == KInstrConstraints():
            object.__setattr__(self, 'constraints', get_default_constraints(self.opcode))
    
    def get_virtual_defs(self) -> List[KVirtualReg]:
        """Get all virtual registers defined by this instruction."""
        result = []
        for d in self.defs:
            if isinstance(d, KRegRange):
                if is_virtual(d.base_reg):
                    result.append(d.base_reg)
            elif is_virtual(d):
                result.append(d)
        return result
    
    def get_virtual_uses(self) -> List[KVirtualReg]:
        """Get all virtual registers used by this instruction."""
        result = []
        for u in self.uses:
            if isinstance(u, KRegRange):
                if is_virtual(u.base_reg):
                    result.append(u.base_reg)
            elif isinstance(u, (KVReg, KSReg)):
                result.append(u)
        return result
    
    def __repr__(self) -> str:
        defs_str = ", ".join(repr(d) for d in self.defs) if self.defs else ""
        uses_str = ", ".join(repr(u) for u in self.uses)
        if defs_str:
            return f"{self.opcode.name} {defs_str} = {uses_str}"
        return f"{self.opcode.name} {uses_str}"


# =============================================================================
# Kernel Program
# =============================================================================

@dataclass
class KernelABI:
    """
    ABI bindings for precolored registers.
    
    These registers are fixed by the GPU ABI and cannot be allocated
    to other values.
    """
    # Flat thread ID in v0 (contains packed tid_x, tid_y, tid_z)
    flat_tid_vreg: KPhysVReg = field(default_factory=lambda: KPhysVReg(0))
    
    # Kernarg pointer in s[0:1]
    kernarg_ptr_sreg_lo: KPhysSReg = field(default_factory=lambda: KPhysSReg(0))
    kernarg_ptr_sreg_hi: KPhysSReg = field(default_factory=lambda: KPhysSReg(1))
    
    # Workgroup ID in s[2], s[3], s[4] (optional, depends on kernel)
    workgroup_id_x_sreg: Optional[KPhysSReg] = None
    workgroup_id_y_sreg: Optional[KPhysSReg] = None
    workgroup_id_z_sreg: Optional[KPhysSReg] = None
    
    def get_reserved_vgprs(self) -> Set[int]:
        """Get set of reserved VGPR indices."""
        return {self.flat_tid_vreg.index}
    
    def get_reserved_sgprs(self) -> Set[int]:
        """Get set of reserved SGPR indices."""
        reserved = {self.kernarg_ptr_sreg_lo.index, self.kernarg_ptr_sreg_hi.index}
        if self.workgroup_id_x_sreg:
            reserved.add(self.workgroup_id_x_sreg.index)
        if self.workgroup_id_y_sreg:
            reserved.add(self.workgroup_id_y_sreg.index)
        if self.workgroup_id_z_sreg:
            reserved.add(self.workgroup_id_z_sreg.index)
        return reserved


@dataclass
class KernelProgram:
    """
    A complete kernel program in IR form.
    
    Contains:
    - instructions: Ordered list of kernel instructions
    - abi: ABI bindings for precolored registers
    - Additional metadata for allocation limits
    """
    instructions: List[KInstr] = field(default_factory=list)
    abi: KernelABI = field(default_factory=KernelABI)
    
    # Allocation limits
    max_vgprs: int = 256
    max_sgprs: int = 104
    
    # Virtual register counters
    _next_vreg_id: int = field(default=0, repr=False)
    _next_sreg_id: int = field(default=0, repr=False)
    
    def alloc_vreg(self) -> KVReg:
        """Allocate a new virtual VGPR."""
        vreg = KVReg(self._next_vreg_id)
        self._next_vreg_id += 1
        return vreg
    
    def alloc_vreg_range(self, count: int, alignment: int = 1) -> KRegRange:
        """Allocate a range of virtual VGPRs."""
        base = self.alloc_vreg()
        # Reserve additional IDs for the range
        for _ in range(count - 1):
            self.alloc_vreg()
        return KRegRange(base, count, alignment)
    
    def alloc_sreg(self) -> KSReg:
        """Allocate a new virtual SGPR."""
        sreg = KSReg(self._next_sreg_id)
        self._next_sreg_id += 1
        return sreg
    
    def alloc_sreg_range(self, count: int, alignment: int = 1) -> KRegRange:
        """Allocate a range of virtual SGPRs."""
        base = self.alloc_sreg()
        # Reserve additional IDs for the range
        for _ in range(count - 1):
            self.alloc_sreg()
        return KRegRange(base, count, alignment)
    
    def emit(self, instr: KInstr):
        """Add an instruction to the program."""
        self.instructions.append(instr)
    
    def emit_comment(self, text: str):
        """Emit a comment."""
        self.emit(KInstr(KOpcode.COMMENT, (), (KImm(0),), comment=text))
    
    def __len__(self) -> int:
        return len(self.instructions)
    
    def __iter__(self):
        return iter(self.instructions)


# =============================================================================
# Builder Helpers
# =============================================================================

class KernelBuilder:
    """
    Helper class for building KernelProgram.
    
    Provides convenient methods for emitting common instruction patterns
    and managing virtual registers.
    """
    
    def __init__(self, program: Optional[KernelProgram] = None):
        self.program = program if program is not None else KernelProgram()
    
    # Virtual register allocation
    def vreg(self) -> KVReg:
        return self.program.alloc_vreg()
    
    def vreg_pair(self) -> KRegRange:
        return self.program.alloc_vreg_range(2, alignment=2)
    
    def vreg_quad(self) -> KRegRange:
        return self.program.alloc_vreg_range(4, alignment=4)
    
    def sreg(self) -> KSReg:
        return self.program.alloc_sreg()
    
    def sreg_pair(self) -> KRegRange:
        return self.program.alloc_sreg_range(2, alignment=2)
    
    # Instruction emission helpers
    def v_mov_b32(self, src: KOperand, comment: str = None) -> KVReg:
        """Emit v_mov_b32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_MOV_B32, (dst,), (src,), comment=comment))
        return dst
    
    def v_add_u32(self, src1: KOperand, src2: KOperand, comment: str = None) -> KVReg:
        """Emit v_add_u32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_ADD_U32, (dst,), (src1, src2), comment=comment))
        return dst
    
    def v_mul_lo_u32(self, src1: KOperand, src2: KOperand, comment: str = None) -> KVReg:
        """Emit v_mul_lo_u32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_MUL_LO_U32, (dst,), (src1, src2), comment=comment))
        return dst
    
    def v_and_b32(self, src1: KOperand, src2: KOperand, comment: str = None) -> KVReg:
        """Emit v_and_b32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_AND_B32, (dst,), (src1, src2), comment=comment))
        return dst
    
    def v_or_b32(self, src1: KOperand, src2: KOperand, comment: str = None) -> KVReg:
        """Emit v_or_b32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_OR_B32, (dst,), (src1, src2), comment=comment))
        return dst
    
    def v_lshlrev_b32(self, shift: KOperand, src: KOperand, comment: str = None) -> KVReg:
        """Emit v_lshlrev_b32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_LSHLREV_B32, (dst,), (shift, src), comment=comment))
        return dst
    
    def v_lshrrev_b32(self, shift: KOperand, src: KOperand, comment: str = None) -> KVReg:
        """Emit v_lshrrev_b32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_LSHRREV_B32, (dst,), (shift, src), comment=comment))
        return dst
    
    def v_bfe_u32(self, src: KOperand, offset: KOperand, width: KOperand, comment: str = None) -> KVReg:
        """Emit v_bfe_u32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_BFE_U32, (dst,), (src, offset, width), comment=comment))
        return dst
    
    def v_lshl_add_u32(self, src: KOperand, shift: KOperand, addend: KOperand, comment: str = None) -> KVReg:
        """Emit v_lshl_add_u32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_LSHL_ADD_U32, (dst,), (src, shift, addend), comment=comment))
        return dst
    
    def s_mov_b32(self, src: KOperand, comment: str = None) -> KSReg:
        """Emit s_mov_b32 and return destination sreg."""
        dst = self.sreg()
        self.program.emit(KInstr(KOpcode.S_MOV_B32, (dst,), (src,), comment=comment))
        return dst
    
    def s_load_dwordx2(self, base: KRegRange, offset: KImm, comment: str = None) -> KRegRange:
        """Emit s_load_dwordx2 and return destination sreg range."""
        dst = self.sreg_pair()
        self.program.emit(KInstr(KOpcode.S_LOAD_DWORDX2, (dst,), (base, offset), comment=comment))
        return dst
    
    def ds_read_b64(self, addr: KVReg, offset: int = 0, comment: str = None) -> KRegRange:
        """Emit ds_read_b64 and return destination vreg range."""
        dst = self.vreg_pair()
        uses = (addr,) if offset == 0 else (addr, KMemOffset(offset))
        self.program.emit(KInstr(KOpcode.DS_READ_B64, (dst,), uses, comment=comment))
        return dst
    
    def ds_write_b64(self, addr: KVReg, src: KRegRange, offset: int = 0, comment: str = None):
        """Emit ds_write_b64."""
        uses = (addr, src) if offset == 0 else (addr, src, KMemOffset(offset))
        self.program.emit(KInstr(KOpcode.DS_WRITE_B64, (), uses, comment=comment))
    
    def s_waitcnt(self, vmcnt: int = 0, lgkmcnt: int = 0, comment: str = None):
        """Emit s_waitcnt."""
        # Encode wait counts as immediate
        waitcnt = (vmcnt & 0x3f) | ((lgkmcnt & 0xf) << 8)
        self.program.emit(KInstr(KOpcode.S_WAITCNT, (), (KImm(waitcnt),), comment=comment))
    
    def s_barrier(self, comment: str = None):
        """Emit s_barrier."""
        self.program.emit(KInstr(KOpcode.S_BARRIER, (), (), comment=comment))
    
    def s_endpgm(self, comment: str = None):
        """Emit s_endpgm."""
        self.program.emit(KInstr(KOpcode.S_ENDPGM, (), (), comment=comment))
    
    def comment(self, text: str):
        """Emit a comment."""
        self.program.emit_comment(text)

