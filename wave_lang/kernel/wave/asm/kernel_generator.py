# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Generator for kernel IR to AMDGCN assembly.

This module converts a KernelProgram (with virtual registers allocated to
physical registers) into final assembly text.

The generator:
1. Takes a KernelProgram and a mapping from virtual to physical registers
2. Substitutes physical register numbers for virtual registers
3. Uses the instruction registry to look up mnemonics and formatting
4. Formats instructions according to AMDGCN assembly syntax

Note: Instruction mnemonics are looked up from the YAML registry, making
it the single source of truth for instruction definitions.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .kernel_ir import (
    KernelProgram, KInstr,
    KVReg, KSReg, KPhysVReg, KPhysSReg, KSpecialReg,
    KReg, KRegRange, KImm, KMemOffset, KOperand,
    is_virtual, is_vgpr, is_sgpr, is_special,
)
from .instruction_registry import get_registry, InstructionRegistry


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


class KernelGenerator:
    """
    Generates AMDGCN assembly from KernelProgram.
    
    Uses the instruction registry to look up mnemonics, ensuring the
    YAML files are the single source of truth for instruction definitions.
    """
    
    def __init__(self, program: KernelProgram, mapping: PhysicalMapping, architecture: str = "common"):
        self.program = program
        self.mapping = mapping
        self._registry = get_registry(architecture)
    
    def generate(self) -> List[str]:
        """Generate the entire program as assembly lines."""
        lines = []
        for instr in self.program.instructions:
            line = self._generate_instr(instr)
            if line is not None:
                lines.append(line)
        return lines
    
    def generate_to_string(self) -> str:
        """Generate the entire program as a single string."""
        return "\n".join(self.generate())
    
    def _resolve_reg(self, reg: KReg) -> str:
        """Resolve a register to its physical string representation."""
        if isinstance(reg, KVReg):
            phys = self.mapping.get_phys_vreg(reg)
            return f"v{phys}"
        elif isinstance(reg, KSReg):
            phys = self.mapping.get_phys_sreg(reg)
            return f"s{phys}"
        elif isinstance(reg, KPhysVReg):
            return f"v{reg.index}"
        elif isinstance(reg, KPhysSReg):
            return f"s{reg.index}"
        elif isinstance(reg, KSpecialReg):
            return reg.name
        raise ValueError(f"Unknown register type: {type(reg)}")
    
    def _resolve_reg_range(self, range_: KRegRange) -> str:
        """Resolve a register range to its physical string representation."""
        base_reg = range_.base_reg
        count = range_.count
        
        if isinstance(base_reg, KVReg):
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
        elif isinstance(op, (KVReg, KSReg, KPhysVReg, KPhysSReg, KSpecialReg)):
            return self._resolve_reg(op)
        elif isinstance(op, KImm):
            # Format immediate appropriately
            if -16 <= op.value <= 64:
                return str(op.value)
            return f"0x{op.value & 0xffffffff:x}"
        elif isinstance(op, int):
            # Handle raw integer immediates (convenience)
            if -16 <= op <= 64:
                return str(op)
            return f"0x{op & 0xffffffff:x}"
        elif isinstance(op, KMemOffset):
            return f"offset:{op.bytes}"
        raise ValueError(f"Unknown operand type: {type(op)}")
    
    def _generate_instr(self, instr: KInstr) -> Optional[str]:
        """Generate a single instruction to assembly."""
        name = instr.name
        
        # Handle pseudo-ops
        if instr.is_comment:
            if instr.comment:
                return f"    // {instr.comment}"
            return None
        
        if instr.is_label:
            if instr.comment:
                return f"{instr.comment}:"
            return None
        
        if instr.is_raw_asm:
            # RAW_ASM uses comment field for the raw assembly line
            return instr.comment if instr.comment else None
        
        # Look up instruction in registry to get mnemonic
        instr_def = self._registry.get(name)
        if instr_def:
            mnemonic = instr_def.mnemonic
        else:
            # Fall back to using name as mnemonic (for instructions not in registry)
            mnemonic = name
        
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
        line = self._apply_special_formatting(name, instr_def, line, operands)
        
        return line
    
    def _apply_special_formatting(self, name: str, instr_def, line: str, operands: List[str]) -> str:
        """Apply instruction-specific formatting rules."""
        # DS instructions: offset is space-separated
        if instr_def and instr_def.offset_format == "space_separated":
            if ", offset:" in line:
                parts = line.split(", offset:")
                if len(parts) == 2:
                    line = parts[0] + " offset:" + parts[1]
        
        # s_waitcnt: format the wait count value
        if name == "s_waitcnt":
            # Parse the waitcnt immediate and format nicely
            # The immediate is encoded as: vmcnt[5:0] | lgkmcnt[11:8]
            for i, op in enumerate(operands):
                try:
                    val = int(op.replace("0x", ""), 16 if "0x" in op else 10)
                    vmcnt = val & 0x3f
                    lgkmcnt = (val >> 8) & 0xf
                    if lgkmcnt == 0:
                        line = f"    s_waitcnt vmcnt({vmcnt})"
                    else:
                        line = f"    s_waitcnt vmcnt({vmcnt}) lgkmcnt({lgkmcnt})"
                except:
                    pass  # Use original if parsing fails
        
        # Buffer instructions: add offen modifier
        if name.startswith("buffer_"):
            # Add offen for buffer operations using VGPR offset
            if " " in line:
                prefix, rest = line.split(" ", 1)
                parts = rest.split(", ")
                if len(parts) >= 4:
                    base = ", ".join(parts[:4])
                    modifiers = ["offen"]
                    for part in parts[4:]:
                        if part.startswith("offset:"):
                            modifiers.append(part)
                    line = f"{prefix} {base} {' '.join(modifiers)}"
        
        # Buffer load with LDS: add lds modifier
        if name.endswith("_lds"):
            if "  //" in line:
                parts = line.split("  //")
                line = parts[0] + " lds  //" + parts[1]
            else:
                line = line + " lds"
        
        return line


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_program(program: KernelProgram, mapping: PhysicalMapping) -> str:
    """Generate assembly string from a kernel program."""
    return KernelGenerator(program, mapping).generate_to_string()


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# Keep old names for backwards compatibility
KernelRenderer = KernelGenerator
render_program = lambda program, mapping: KernelGenerator(program, mapping).generate_to_string()
