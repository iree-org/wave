# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Renderer for kernel IR to AMDGCN assembly.

This module converts a KernelProgram (with virtual registers allocated to
physical registers) into final assembly text or instruction objects.

The renderer:
1. Takes a KernelProgram and a mapping from virtual to physical registers
2. Substitutes physical register numbers for virtual registers
3. Formats instructions according to AMDGCN assembly syntax
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .kernel_ir import (
    KernelProgram, KInstr, KOpcode,
    KVReg, KSReg, KPhysVReg, KPhysSReg,
    KReg, KRegRange, KImm, KMemOffset, KOperand,
    is_virtual, is_vgpr, is_sgpr,
)


@dataclass
class PhysicalMapping:
    """
    Mapping from virtual registers to physical register indices.
    """
    vreg_map: Dict[int, int]  # KVReg.id -> physical VGPR index
    sreg_map: Dict[int, int]  # KSReg.id -> physical SGPR index
    
    def get_phys_vreg(self, vreg: KVReg) -> int:
        """Get physical VGPR index for a virtual VGPR."""
        if vreg.id not in self.vreg_map:
            raise ValueError(f"Virtual VGPR {vreg} not allocated")
        return self.vreg_map[vreg.id]
    
    def get_phys_sreg(self, sreg: KSReg) -> int:
        """Get physical SGPR index for a virtual SGPR."""
        if sreg.id not in self.sreg_map:
            raise ValueError(f"Virtual SGPR {sreg} not allocated")
        return self.sreg_map[sreg.id]


class KernelRenderer:
    """
    Renders KernelProgram to assembly text.
    """
    
    def __init__(self, program: KernelProgram, mapping: PhysicalMapping):
        self.program = program
        self.mapping = mapping
    
    def render(self) -> List[str]:
        """Render the entire program to assembly lines."""
        lines = []
        for instr in self.program:
            line = self._render_instr(instr)
            if line:
                lines.append(line)
        return lines
    
    def render_to_string(self) -> str:
        """Render the entire program to a single string."""
        return "\n".join(self.render())
    
    def _resolve_reg(self, reg: KReg) -> str:
        """Resolve a register to its physical name."""
        if isinstance(reg, KPhysVReg):
            return f"v{reg.index}"
        elif isinstance(reg, KPhysSReg):
            return f"s{reg.index}"
        elif isinstance(reg, KVReg):
            phys_idx = self.mapping.get_phys_vreg(reg)
            return f"v{phys_idx}"
        elif isinstance(reg, KSReg):
            phys_idx = self.mapping.get_phys_sreg(reg)
            return f"s{phys_idx}"
        raise ValueError(f"Unknown register type: {type(reg)}")
    
    def _resolve_reg_range(self, rng: KRegRange) -> str:
        """Resolve a register range to its physical representation."""
        base_reg = rng.base_reg
        count = rng.count
        
        if isinstance(base_reg, KPhysVReg):
            start = base_reg.index
            return f"v[{start}:{start + count - 1}]"
        elif isinstance(base_reg, KPhysSReg):
            start = base_reg.index
            return f"s[{start}:{start + count - 1}]"
        elif isinstance(base_reg, KVReg):
            start = self.mapping.get_phys_vreg(base_reg)
            return f"v[{start}:{start + count - 1}]"
        elif isinstance(base_reg, KSReg):
            start = self.mapping.get_phys_sreg(base_reg)
            return f"s[{start}:{start + count - 1}]"
        raise ValueError(f"Unknown base register type: {type(base_reg)}")
    
    def _resolve_operand(self, op: KOperand) -> str:
        """Resolve an operand to its string representation."""
        if isinstance(op, KRegRange):
            return self._resolve_reg_range(op)
        elif isinstance(op, (KVReg, KSReg, KPhysVReg, KPhysSReg)):
            return self._resolve_reg(op)
        elif isinstance(op, KImm):
            # Format immediate appropriately
            if -16 <= op.value <= 64:
                return str(op.value)
            return f"0x{op.value & 0xffffffff:x}"
        elif isinstance(op, KMemOffset):
            return f"offset:{op.bytes}"
        raise ValueError(f"Unknown operand type: {type(op)}")
    
    def _render_instr(self, instr: KInstr) -> Optional[str]:
        """Render a single instruction to assembly."""
        opcode = instr.opcode
        
        # Handle pseudo-ops
        if opcode == KOpcode.COMMENT:
            if instr.comment:
                return f"    // {instr.comment}"
            return None
        
        if opcode == KOpcode.LABEL:
            if instr.comment:
                return f"{instr.comment}:"
            return None
        
        # Map opcode to mnemonic
        mnemonic = self._get_mnemonic(opcode)
        
        # Build operand list
        operands = []
        
        # Add destinations first
        for d in instr.defs:
            if isinstance(d, KRegRange):
                operands.append(self._resolve_reg_range(d))
            else:
                operands.append(self._resolve_reg(d))
        
        # Add uses
        for u in instr.uses:
            operands.append(self._resolve_operand(u))
        
        # Format instruction
        if not operands:
            line = f"    {mnemonic}"
        else:
            line = f"    {mnemonic} {', '.join(operands)}"
        
        # Add comment if present
        if instr.comment:
            line += f"  // {instr.comment}"
        
        # Special formatting for certain instructions
        line = self._apply_special_formatting(opcode, line, operands)
        
        return line
    
    def _get_mnemonic(self, opcode: KOpcode) -> str:
        """Map opcode to assembly mnemonic."""
        mnemonic_map = {
            # Moves
            KOpcode.V_MOV_B32: "v_mov_b32",
            KOpcode.S_MOV_B32: "s_mov_b32",
            KOpcode.S_MOV_B64: "s_mov_b64",
            
            # Vector arithmetic
            KOpcode.V_ADD_U32: "v_add_u32",
            KOpcode.V_SUB_U32: "v_sub_u32",
            KOpcode.V_MUL_LO_U32: "v_mul_lo_u32",
            KOpcode.V_MUL_HI_U32: "v_mul_hi_u32",
            
            # Vector bitwise
            KOpcode.V_AND_B32: "v_and_b32",
            KOpcode.V_OR_B32: "v_or_b32",
            KOpcode.V_XOR_B32: "v_xor_b32",
            
            # Vector shifts
            KOpcode.V_LSHLREV_B32: "v_lshlrev_b32",
            KOpcode.V_LSHRREV_B32: "v_lshrrev_b32",
            KOpcode.V_ASHRREV_I32: "v_ashrrev_i32",
            
            # Fused ops
            KOpcode.V_LSHL_ADD_U32: "v_lshl_add_u32",
            KOpcode.V_LSHL_OR_B32: "v_lshl_or_b32",
            KOpcode.V_ADD_LSHL_U32: "v_add_lshl_u32",
            KOpcode.V_OR3_B32: "v_or3_b32",
            
            # Bit field
            KOpcode.V_BFE_U32: "v_bfe_u32",
            
            # Scalar arithmetic
            KOpcode.S_ADD_U32: "s_add_u32",
            KOpcode.S_MUL_I32: "s_mul_i32",
            KOpcode.S_LSHL_B32: "s_lshl_b32",
            KOpcode.S_LSHR_B32: "s_lshr_b32",
            KOpcode.S_AND_B32: "s_and_b32",
            KOpcode.S_OR_B32: "s_or_b32",
            KOpcode.S_ADD_U64: "s_add_u64",
            KOpcode.S_LSHL_B64: "s_lshl_b64",
            
            # Scalar loads
            KOpcode.S_LOAD_DWORD: "s_load_dword",
            KOpcode.S_LOAD_DWORDX2: "s_load_dwordx2",
            KOpcode.S_LOAD_DWORDX4: "s_load_dwordx4",
            
            # Buffer loads
            KOpcode.BUFFER_LOAD_DWORD: "buffer_load_dword",
            KOpcode.BUFFER_LOAD_DWORDX2: "buffer_load_dwordx2",
            KOpcode.BUFFER_LOAD_DWORDX4: "buffer_load_dwordx4",
            
            # Global loads
            KOpcode.GLOBAL_LOAD_DWORD: "global_load_dword",
            KOpcode.GLOBAL_LOAD_DWORDX2: "global_load_dwordx2",
            KOpcode.GLOBAL_LOAD_DWORDX4: "global_load_dwordx4",
            
            # Buffer stores
            KOpcode.BUFFER_STORE_DWORD: "buffer_store_dword",
            KOpcode.BUFFER_STORE_DWORDX2: "buffer_store_dwordx2",
            KOpcode.BUFFER_STORE_DWORDX4: "buffer_store_dwordx4",
            
            # Global stores
            KOpcode.GLOBAL_STORE_DWORD: "global_store_dword",
            KOpcode.GLOBAL_STORE_DWORDX2: "global_store_dwordx2",
            KOpcode.GLOBAL_STORE_DWORDX4: "global_store_dwordx4",
            
            # LDS ops
            KOpcode.DS_READ_B32: "ds_read_b32",
            KOpcode.DS_READ_B64: "ds_read_b64",
            KOpcode.DS_READ_B128: "ds_read_b128",
            KOpcode.DS_WRITE_B32: "ds_write_b32",
            KOpcode.DS_WRITE_B64: "ds_write_b64",
            KOpcode.DS_WRITE_B128: "ds_write_b128",
            
            # MFMA
            KOpcode.V_MFMA_F32_16X16X16_F16: "v_mfma_f32_16x16x16_f16",
            KOpcode.V_MFMA_F32_32X32X8_F16: "v_mfma_f32_32x32x8_f16",
            KOpcode.V_MFMA_F16_16X16X16_F16: "v_mfma_f16_16x16x16_f16",
            KOpcode.V_MFMA_F16_32X32X8_F16: "v_mfma_f16_32x32x8_f16",
            
            # Conversion
            KOpcode.V_CVT_F32_F16: "v_cvt_f32_f16",
            KOpcode.V_CVT_F16_F32: "v_cvt_f16_f32",
            KOpcode.V_PACK_B32_F16: "v_pack_b32_f16",
            KOpcode.V_CVT_PK_FP8_F32: "v_cvt_pk_fp8_f32",
            
            # Control flow
            KOpcode.S_WAITCNT: "s_waitcnt",
            KOpcode.S_BARRIER: "s_barrier",
            KOpcode.S_NOP: "s_nop",
            KOpcode.S_ENDPGM: "s_endpgm",
            
            # Lane ops
            KOpcode.V_READFIRSTLANE_B32: "v_readfirstlane_b32",
        }
        
        if opcode not in mnemonic_map:
            raise ValueError(f"Unknown opcode: {opcode}")
        return mnemonic_map[opcode]
    
    def _apply_special_formatting(self, opcode: KOpcode, line: str, operands: List[str]) -> str:
        """Apply special formatting rules for certain instructions."""
        # ds_read/ds_write: offset is space-separated, not comma-separated
        if opcode in {KOpcode.DS_READ_B32, KOpcode.DS_READ_B64, KOpcode.DS_READ_B128,
                      KOpcode.DS_WRITE_B32, KOpcode.DS_WRITE_B64, KOpcode.DS_WRITE_B128}:
            if "offset:" in line:
                # Move offset to space-separated position
                parts = line.split(", offset:")
                if len(parts) == 2:
                    line = parts[0] + " offset:" + parts[1]
        
        # s_waitcnt: decode and format wait counts
        if opcode == KOpcode.S_WAITCNT:
            # The operand is the encoded waitcnt value
            # Extract and format nicely
            pass  # Keep simple format for now
        
        return line


def render_program(program: KernelProgram, mapping: PhysicalMapping) -> str:
    """Convenience function to render a program to assembly string."""
    renderer = KernelRenderer(program, mapping)
    return renderer.render_to_string()

