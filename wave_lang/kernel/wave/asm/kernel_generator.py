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
from dataclasses import dataclass, field

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
    
    Supports both single registers and range components.
    """
    vreg_map: Dict[int, int]  # KVReg.id -> physical VGPR index
    sreg_map: Dict[int, int]  # KSReg.id -> physical SGPR index
    # Range membership: component_id -> (base_id, size) for resolving range components
    vreg_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    sreg_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    
    def get_phys_vreg(self, vreg: KVReg) -> int:
        """Get physical VGPR index for a virtual VGPR."""
        # Direct lookup
        if vreg.id in self.vreg_map:
            return self.vreg_map[vreg.id]
        
        # Check if this is a component of a range
        if vreg.id in self.vreg_ranges:
            base_id, size = self.vreg_ranges[vreg.id]
            offset = vreg.id - base_id
            if base_id in self.vreg_map:
                return self.vreg_map[base_id] + offset
        
        raise ValueError(f"Virtual VGPR {vreg} not allocated")
    
    def get_phys_sreg(self, sreg: KSReg) -> int:
        """Get physical SGPR index for a virtual SGPR."""
        # Direct lookup
        if sreg.id in self.sreg_map:
            return self.sreg_map[sreg.id]
        
        # Check if this is a component of a range
        if sreg.id in self.sreg_ranges:
            base_id, size = self.sreg_ranges[sreg.id]
            offset = sreg.id - base_id
            if base_id in self.sreg_map:
                return self.sreg_map[base_id] + offset
        
        raise ValueError(f"Virtual SGPR {sreg} not allocated")


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
        elif isinstance(base_reg, KPhysVReg):
            # Already physical
            start = base_reg.index
            return f"v[{start}:{start + count - 1}]"
        elif isinstance(base_reg, KPhysSReg):
            # Already physical
            start = base_reg.index
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
        elif isinstance(op, str):
            # Handle string operands (e.g., waitcnt values like "vmcnt(0)")
            return op
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
        
        # Handle pseudo-instructions for SRD setup
        if name == "_srd_define":
            # This is a pseudo-instruction that defines a range for allocation
            # It doesn't emit any assembly - just used for liveness tracking
            return None  # Skip emission
        
        if name == "_srd_copy_define":
            # Pseudo-instruction for SRD copy range definition
            # No assembly emitted - just for liveness analysis
            return None  # Skip emission
        
        if name == "_g2s_srd_copy":
            # G2S SRD copy: new_range = copy of orig_range with LDS format
            # defs: (new_srd_range,)
            # uses: (orig_srd_range, cache_swizzle_bits)
            new_range = instr.defs[0]
            orig_range = instr.uses[0]
            cache_swizzle = instr.uses[1].value if isinstance(instr.uses[1], KImm) else 0
            
            new_base = self._resolve_reg_range(new_range)
            orig_base = self._resolve_reg_range(orig_range)
            
            # Parse s[N:M] format to get start index
            new_start = int(new_base.split('[')[1].split(':')[0])
            orig_start = int(orig_base.split('[')[1].split(':')[0])
            
            lines = []
            # Word0: copy base address low
            lines.append(f"    s_mov_b32 s{new_start}, s{orig_start}  // SRD word0")
            
            # Word1: base address high (with optional cache swizzle)
            # For G2S, we typically don't need cache swizzle in the SRD itself
            # The swizzle is handled by the LLVM cache swizzle mode
            lines.append(f"    s_and_b32 s{new_start + 1}, s{orig_start + 1}, 0xffff")
            if cache_swizzle != 0:
                lines.append(f"    s_or_b32 s{new_start + 1}, s{new_start + 1}, {hex(cache_swizzle)}  // cache swizzle")
            
            # Word2: max buffer size
            lines.append(f"    s_mov_b32 s{new_start + 2}, 0x7ffffffd  // SRD word2")
            
            # Word3: LDS format (data_format=4, num_format=7, add_tid_enable=1)
            lines.append(f"    s_mov_b32 s{new_start + 3}, 0x27000  // SRD word3")
            
            return "\n".join(lines)
        
        if name == "_srd_load_base":
            # Load base address into SRD[0:1]
            # uses: (srd_range, kernarg_pair, offset_imm)
            srd_range = instr.uses[0]
            kernarg_pair = instr.uses[1]
            offset_imm = instr.uses[2]
            srd_phys = self._resolve_reg_range(srd_range)
            kernarg_phys = self._resolve_reg_range(kernarg_pair) if isinstance(kernarg_pair, KRegRange) else self._resolve_operand(kernarg_pair)
            offset = offset_imm.value if isinstance(offset_imm, KImm) else offset_imm
            # Extract base from "s[N:M]" format
            base = int(srd_phys.split('[')[1].split(':')[0])
            line = f"    s_load_dwordx2 s[{base}:{base+1}], {kernarg_phys}, {offset}"
            if instr.comment:
                line += f"  // {instr.comment}"
            return line
        
        if name == "_srd_fill_size":
            # Fill SRD[2] with size value
            # uses: (srd_range, size_imm)
            srd_range = instr.uses[0]
            size_val = instr.uses[1].value if isinstance(instr.uses[1], KImm) else instr.uses[1]
            srd_phys = self._resolve_reg_range(srd_range)
            base = int(srd_phys.split('[')[1].split(':')[0])
            line = f"    s_mov_b32 s{base + 2}, {hex(size_val)}"
            if instr.comment:
                line += f"  // {instr.comment}"
            return line
        
        if name == "_srd_fill_stride":
            # Fill SRD[3] with stride descriptor
            srd_range = instr.uses[0]
            stride_val = instr.uses[1].value if isinstance(instr.uses[1], KImm) else instr.uses[1]
            srd_phys = self._resolve_reg_range(srd_range)
            base = int(srd_phys.split('[')[1].split(':')[0])
            line = f"    s_mov_b32 s{base + 3}, {hex(stride_val)}"
            if instr.comment:
                line += f"  // {instr.comment}"
            return line
        
        # Handle label pseudo-instruction
        if name == "_label":
            label = instr.comment if instr.comment else "label"
            return f"{label}:"
        
        # Handle branch instructions (label is in comment)
        if name == "s_cbranch_scc1":
            label = instr.comment if instr.comment else "target"
            return f"    s_cbranch_scc1 {label}"
        
        if name == "s_branch":
            label = instr.comment if instr.comment else "target"
            return f"    s_branch {label}"
        
        # Handle comparison instruction
        if name == "s_cmp_lt_u32":
            if len(instr.uses) >= 2:
                op0 = self._resolve_operand(instr.uses[0])
                op1 = self._resolve_operand(instr.uses[1])
                line = f"    s_cmp_lt_u32 {op0}, {op1}"
                if instr.comment:
                    line += f"  // {instr.comment}"
                return line
        
        # Handle loop increment (physical register, no def)
        if name == "_loop_inc":
            if len(instr.uses) >= 2:
                counter = self._resolve_operand(instr.uses[0])
                step = self._resolve_operand(instr.uses[1])
                line = f"    s_add_u32 {counter}, {counter}, {step}"
                if instr.comment:
                    line += f"  // {instr.comment}"
                return line
        
        # Handle MFMA with accumulator (in-place update, no new defs)
        if name == "_mfma_acc":
            if len(instr.uses) >= 3:
                acc = self._resolve_operand(instr.uses[0])  # accumulator/destination
                a = self._resolve_operand(instr.uses[1])    # A operand
                b = self._resolve_operand(instr.uses[2])    # B operand
                line = f"    v_mfma_f32_16x16x16_f16 {acc}, {a}, {b}, {acc}"
                if instr.comment:
                    line += f"  // {instr.comment}"
                return line
        
        # Handle buffer_load_dword_lds (gather to LDS)
        if name == "buffer_load_dword_lds":
            # uses: (vaddr, srd_range, soffset)
            if len(instr.uses) >= 3:
                vaddr = self._resolve_operand(instr.uses[0])
                srd = self._resolve_reg_range(instr.uses[1]) if hasattr(instr.uses[1], 'start') else self._resolve_operand(instr.uses[1])
                soffset = self._resolve_operand(instr.uses[2])
                line = f"    buffer_load_dword {vaddr}, {srd}, {soffset} offen lds"
                if instr.comment:
                    line += f"  // {instr.comment}"
                return line
        
        # Handle buffer_load_dwordx4_lds (gather 16B to LDS)
        if name == "buffer_load_dwordx4_lds":
            # uses: (vaddr, srd_range, soffset)
            if len(instr.uses) >= 3:
                vaddr = self._resolve_operand(instr.uses[0])
                srd = self._resolve_reg_range(instr.uses[1]) if hasattr(instr.uses[1], 'start') else self._resolve_operand(instr.uses[1])
                soffset = self._resolve_operand(instr.uses[2])
                line = f"    buffer_load_dwordx4 {vaddr}, {srd}, {soffset} offen lds"
                if instr.comment:
                    line += f"  // {instr.comment}"
                return line
        
        # Handle accumulator quad initialization
        if name == "_init_acc_quad":
            # Emit 4 v_mov_b32 instructions to initialize the quad
            if len(instr.defs) >= 1:
                dst_range = instr.defs[0]
                base = self._resolve_operand(dst_range)
                # Parse the base register index from the range string
                # e.g., "v[4:7]" -> base is 4
                if '[' in base:
                    base_idx = int(base.split('[')[1].split(':')[0])
                else:
                    base_idx = int(base[1:])
                
                lines = []
                for i in range(4):
                    lines.append(f"    v_mov_b32 v{base_idx + i}, 0")
                if instr.comment:
                    lines[-1] += f"  // {instr.comment}"
                return '\n'.join(lines)
        
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
            # Handle string operands like "vmcnt(0)" or "lgkmcnt(0)"
            for i, op in enumerate(operands):
                if isinstance(op, str) and ("vmcnt" in op or "lgkmcnt" in op):
                    # Already formatted as waitcnt string
                    line = f"    s_waitcnt {op}"
                    break
                try:
                    # Parse the waitcnt immediate and format nicely
                    # The immediate is encoded as: vmcnt[5:0] | lgkmcnt[11:8]
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
            # First, separate any comment
            comment_part = ""
            main_line = line
            if "  //" in line:
                main_line, comment_part = line.split("  //", 1)
                comment_part = "  //" + comment_part
            
            if " " in main_line:
                prefix, rest = main_line.split(" ", 1)
                parts = rest.split(", ")
                if len(parts) >= 4:
                    base = ", ".join(parts[:4])
                    modifiers = ["offen"]
                    for part in parts[4:]:
                        if part.startswith("offset:"):
                            # Skip offset:0 (it's the default)
                            offset_val = part.split(":")[1].strip()
                            if offset_val != "0":
                                modifiers.append(part.strip())
                        else:
                            modifiers.append(part.strip())
                    line = f"{prefix} {base} {' '.join(modifiers)}{comment_part}"
        
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
