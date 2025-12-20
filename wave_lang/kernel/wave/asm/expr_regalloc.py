# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Register allocator for Expression IR.

This module provides a linear-scan register allocator that maps virtual registers
(VReg, SReg) in an ExprProgram to physical registers (PhysVReg, PhysSReg).

Key features:
- Respects ABI-reserved registers (v0 for flat tid, s[0:1] for kernarg)
- Configurable allocation policies for testing allocation-order independence
- Supports register reuse through lifetime analysis
- Optional randomization for stress testing

Usage:
    from expr_ir import ExprProgram, VReg
    from expr_regalloc import ExprRegAlloc, AllocPolicy
    
    prog = ExprProgram()
    # ... emit instructions ...
    
    alloc = ExprRegAlloc(policy=AllocPolicy.SEQUENTIAL)
    alloc.reserve_vgpr(0)  # v0 for flat tid
    mapping = alloc.allocate(prog)
    
    # Apply mapping to get physical instructions
    phys_prog = prog.apply_mapping(mapping)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import random

from .expr_ir import (
    ExprProgram, ExprInstr, VReg, SReg, PhysVReg, PhysSReg, Imm, OpCode,
    VirtualReg, PhysicalReg, Operand, is_inline_constant, get_immediate_info
)


class AllocPolicy(Enum):
    """
    Allocation policies for testing allocation-order independence.
    
    - SEQUENTIAL: Allocate registers in order (v1, v2, v3, ...)
    - LIFO_REUSE: Reuse freed registers in LIFO order (like current allocator)
    - FIFO_REUSE: Reuse freed registers in FIFO order
    - RANDOM: Randomize allocation order (for stress testing)
    - REVERSE: Allocate in reverse order (v255, v254, ...)
    """
    SEQUENTIAL = auto()
    LIFO_REUSE = auto()
    FIFO_REUSE = auto()
    RANDOM = auto()
    REVERSE = auto()


@dataclass
class LiveRange:
    """
    Live range for a virtual register.
    
    Represents the instruction range where a virtual register is live
    (from definition to last use).
    """
    reg: VirtualReg
    start: int  # Instruction index of definition
    end: int    # Instruction index of last use
    
    def overlaps(self, other: "LiveRange") -> bool:
        """Check if this live range overlaps with another."""
        return not (self.end < other.start or other.end < self.start)


@dataclass
class ExprRegAlloc:
    """
    Register allocator for Expression IR.
    
    Maps virtual registers to physical registers using linear scan algorithm,
    with support for different allocation policies.
    """
    
    # Allocation policy
    policy: AllocPolicy = AllocPolicy.LIFO_REUSE
    
    # Random seed for RANDOM policy
    random_seed: Optional[int] = None
    
    # Reserved physical registers
    reserved_vgprs: Set[int] = field(default_factory=set)
    reserved_sgprs: Set[int] = field(default_factory=set)
    
    # Starting register index (first allocatable)
    vgpr_base: int = 0
    sgpr_base: int = 2  # s[0:1] typically reserved for kernarg
    
    # Maximum register indices
    vgpr_max: int = 255
    sgpr_max: int = 103
    
    # Tracking
    next_vgpr: int = field(init=False)
    next_sgpr: int = field(init=False)
    free_vgprs: List[int] = field(default_factory=list)
    free_sgprs: List[int] = field(default_factory=list)
    
    # Statistics
    peak_vgpr: int = field(default=0, init=False)
    peak_sgpr: int = field(default=0, init=False)
    
    def __post_init__(self):
        self.next_vgpr = self.vgpr_base
        self.next_sgpr = self.sgpr_base
        
        # Initialize random generator if needed
        if self.policy == AllocPolicy.RANDOM:
            self._rng = random.Random(self.random_seed)
        else:
            self._rng = None
    
    def reserve_vgpr(self, reg: int) -> None:
        """Reserve a VGPR (e.g., v0 for flat tid)."""
        self.reserved_vgprs.add(reg)
        # Adjust base if needed
        if reg >= self.vgpr_base:
            self.vgpr_base = reg + 1
            self.next_vgpr = max(self.next_vgpr, self.vgpr_base)
    
    def reserve_sgpr(self, reg: int) -> None:
        """Reserve an SGPR (e.g., s2 for wgid_x)."""
        self.reserved_sgprs.add(reg)
        # Adjust base if needed
        if reg >= self.sgpr_base:
            self.sgpr_base = reg + 1
            self.next_sgpr = max(self.next_sgpr, self.sgpr_base)
    
    def _alloc_vgpr(self) -> int:
        """Allocate a physical VGPR based on policy."""
        if self.policy == AllocPolicy.SEQUENTIAL or not self.free_vgprs:
            # Allocate new register
            while self.next_vgpr in self.reserved_vgprs:
                self.next_vgpr += 1
            reg = self.next_vgpr
            self.next_vgpr += 1
        elif self.policy == AllocPolicy.LIFO_REUSE:
            reg = self.free_vgprs.pop()  # LIFO
        elif self.policy == AllocPolicy.FIFO_REUSE:
            reg = self.free_vgprs.pop(0)  # FIFO
        elif self.policy == AllocPolicy.RANDOM:
            idx = self._rng.randrange(len(self.free_vgprs))
            reg = self.free_vgprs.pop(idx)
        elif self.policy == AllocPolicy.REVERSE:
            if not self.free_vgprs:
                reg = self.vgpr_max
                while reg in self.reserved_vgprs or reg < self.next_vgpr:
                    reg -= 1
                self.next_vgpr = reg + 1  # Track for next allocation
            else:
                reg = self.free_vgprs.pop()
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
        
        self.peak_vgpr = max(self.peak_vgpr, reg)
        return reg
    
    def _alloc_sgpr(self) -> int:
        """Allocate a physical SGPR based on policy."""
        if self.policy == AllocPolicy.SEQUENTIAL or not self.free_sgprs:
            while self.next_sgpr in self.reserved_sgprs:
                self.next_sgpr += 1
            reg = self.next_sgpr
            self.next_sgpr += 1
        elif self.policy == AllocPolicy.LIFO_REUSE:
            reg = self.free_sgprs.pop()
        elif self.policy == AllocPolicy.FIFO_REUSE:
            reg = self.free_sgprs.pop(0)
        elif self.policy == AllocPolicy.RANDOM:
            idx = self._rng.randrange(len(self.free_sgprs))
            reg = self.free_sgprs.pop(idx)
        elif self.policy == AllocPolicy.REVERSE:
            if not self.free_sgprs:
                reg = self.sgpr_max
                while reg in self.reserved_sgprs or reg < self.next_sgpr:
                    reg -= 1
                self.next_sgpr = reg + 1
            else:
                reg = self.free_sgprs.pop()
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
        
        self.peak_sgpr = max(self.peak_sgpr, reg)
        return reg
    
    def _free_vgpr(self, reg: int) -> None:
        """Return a VGPR to the free list."""
        if reg not in self.reserved_vgprs:
            self.free_vgprs.append(reg)
    
    def _free_sgpr(self, reg: int) -> None:
        """Return an SGPR to the free list."""
        if reg not in self.reserved_sgprs:
            self.free_sgprs.append(reg)
    
    def compute_live_ranges(self, prog: ExprProgram) -> Dict[VirtualReg, LiveRange]:
        """
        Compute live ranges for all virtual registers in the program.
        
        Returns:
            Dict mapping virtual registers to their live ranges
        """
        first_def: Dict[VirtualReg, int] = {}
        last_use: Dict[VirtualReg, int] = {}
        
        for i, instr in enumerate(prog.instructions):
            # Record definition
            dst = instr.get_def()
            if isinstance(dst, (VReg, SReg)):
                if dst not in first_def:
                    first_def[dst] = i
            
            # Record uses
            for use in instr.get_uses():
                if isinstance(use, (VReg, SReg)):
                    last_use[use] = i
        
        # Build live ranges
        ranges: Dict[VirtualReg, LiveRange] = {}
        all_regs = set(first_def.keys()) | set(last_use.keys())
        for reg in all_regs:
            start = first_def.get(reg, 0)
            end = last_use.get(reg, start)
            ranges[reg] = LiveRange(reg, start, end)
        
        return ranges
    
    def allocate(self, prog: ExprProgram) -> Dict[VirtualReg, PhysicalReg]:
        """
        Allocate physical registers for all virtual registers in the program.
        
        Uses linear-scan allocation with the configured policy.
        
        Args:
            prog: The ExprProgram to allocate registers for
            
        Returns:
            Mapping from virtual registers to physical registers
        """
        # Compute live ranges
        live_ranges = self.compute_live_ranges(prog)
        
        # Sort intervals by start position
        sorted_ranges = sorted(live_ranges.values(), key=lambda r: (r.start, r.end))
        
        # Allocation mapping
        mapping: Dict[VirtualReg, PhysicalReg] = {}
        
        # Active intervals (sorted by end position)
        active: List[LiveRange] = []
        
        for lr in sorted_ranges:
            # Expire old intervals
            expired = []
            for active_lr in active:
                if active_lr.end < lr.start:
                    expired.append(active_lr)
                    # Free the physical register
                    phys_reg = mapping[active_lr.reg]
                    if isinstance(phys_reg, PhysVReg):
                        self._free_vgpr(phys_reg.index)
                    elif isinstance(phys_reg, PhysSReg):
                        self._free_sgpr(phys_reg.index)
            
            for exp in expired:
                active.remove(exp)
            
            # Allocate register for current interval
            if isinstance(lr.reg, VReg):
                phys_idx = self._alloc_vgpr()
                mapping[lr.reg] = PhysVReg(phys_idx)
            elif isinstance(lr.reg, SReg):
                phys_idx = self._alloc_sgpr()
                mapping[lr.reg] = PhysSReg(phys_idx)
            
            # Add to active
            active.append(lr)
            active.sort(key=lambda r: r.end)
        
        return mapping
    
    def apply_mapping(
        self,
        prog: ExprProgram,
        mapping: Dict[VirtualReg, PhysicalReg]
    ) -> List[ExprInstr]:
        """
        Apply the register mapping to produce physical instructions.
        
        Args:
            prog: The original ExprProgram
            mapping: Virtual to physical register mapping
            
        Returns:
            List of instructions with physical registers
        """
        return [instr.replace_operands(mapping) for instr in prog.instructions]
    
    def get_stats(self) -> Dict[str, int]:
        """Get allocation statistics."""
        return {
            "peak_vgpr": self.peak_vgpr,
            "peak_sgpr": self.peak_sgpr,
            "reserved_vgprs": len(self.reserved_vgprs),
            "reserved_sgprs": len(self.reserved_sgprs),
        }


def allocate_program(
    prog: ExprProgram,
    reserved_vgprs: Set[int] = None,
    reserved_sgprs: Set[int] = None,
    policy: AllocPolicy = AllocPolicy.LIFO_REUSE,
    random_seed: Optional[int] = None,
) -> Tuple[List[ExprInstr], Dict[VirtualReg, PhysicalReg]]:
    """
    Convenience function to allocate registers for an ExprProgram.
    
    Args:
        prog: The program to allocate
        reserved_vgprs: Set of reserved VGPR indices (default: {0} for v0)
        reserved_sgprs: Set of reserved SGPR indices (default: {0, 1} for kernarg)
        policy: Allocation policy
        random_seed: Random seed for RANDOM policy
        
    Returns:
        Tuple of (physical instructions, mapping)
    """
    if reserved_vgprs is None:
        reserved_vgprs = {0}  # v0 for flat workitem ID
    if reserved_sgprs is None:
        reserved_sgprs = {0, 1}  # s[0:1] for kernarg ptr
    
    alloc = ExprRegAlloc(
        policy=policy,
        random_seed=random_seed,
        reserved_vgprs=reserved_vgprs,
        reserved_sgprs=reserved_sgprs,
    )
    
    # Set base to skip reserved registers
    if reserved_vgprs:
        alloc.vgpr_base = max(reserved_vgprs) + 1
        alloc.next_vgpr = alloc.vgpr_base
    if reserved_sgprs:
        alloc.sgpr_base = max(reserved_sgprs) + 1
        alloc.next_sgpr = alloc.sgpr_base
    
    mapping = alloc.allocate(prog)
    phys_instrs = alloc.apply_mapping(prog, mapping)
    
    return phys_instrs, mapping


# ============================================================================
# Immediate Legalization
# ============================================================================

class ConstantMaterializer:
    """
    Manages materialized constants for reuse across instructions.
    
    When an instruction needs a constant in a register (e.g., MUL_IMM),
    this class tracks which constants have been materialized so we can
    reuse them instead of materializing again.
    """
    
    def __init__(self, alloc: ExprRegAlloc):
        self.alloc = alloc
        # Maps constant value -> physical VGPR index
        self._materialized: Dict[int, int] = {}
        # Track which VGPRs we've allocated (for freeing later)
        self._allocated_vgprs: List[int] = []
    
    def get_or_materialize(
        self, 
        value: int, 
        instructions: List[ExprInstr]
    ) -> PhysVReg:
        """
        Get a VGPR containing the constant value.
        
        If already materialized, returns the existing VGPR.
        Otherwise, materializes the constant and returns the new VGPR.
        
        Args:
            value: The constant value to materialize
            instructions: List to append MOV_IMM instructions to
            
        Returns:
            PhysVReg containing the constant value
        """
        if value in self._materialized:
            return PhysVReg(self._materialized[value])
        
        # Allocate new VGPR for this constant
        phys_idx = self.alloc._alloc_vgpr()
        self._materialized[value] = phys_idx
        self._allocated_vgprs.append(phys_idx)
        
        # Emit MOV_IMM to materialize the constant
        mov_instr = ExprInstr(
            opcode=OpCode.MOV_IMM,
            dst=PhysVReg(phys_idx),
            operands=(Imm(value),),
            comment=f"materialize const {value}"
        )
        instructions.append(mov_instr)
        
        return PhysVReg(phys_idx)
    
    def get_materialized_count(self) -> int:
        """Return count of unique constants materialized."""
        return len(self._materialized)
    
    def free_all(self) -> None:
        """Free all allocated VGPRs (call when constants are no longer needed)."""
        for vgpr in self._allocated_vgprs:
            self.alloc._free_vgpr(vgpr)
        self._materialized.clear()
        self._allocated_vgprs.clear()


def legalize_immediates(
    phys_instrs: List[ExprInstr],
    alloc: ExprRegAlloc
) -> List[ExprInstr]:
    """
    Legalize instructions by handling immediate operands.
    
    For instructions that can use immediates directly, leaves them as-is.
    For instructions that need constants in registers (like MUL), materializes
    the constants with CSE (reusing previously materialized constants).
    
    Args:
        phys_instrs: List of instructions with physical registers
        alloc: The allocator to use for constant materialization
        
    Returns:
        List of legalized instructions with constants materialized where needed
    """
    materializer = ConstantMaterializer(alloc)
    result: List[ExprInstr] = []
    
    for instr in phys_instrs:
        imm_info = get_immediate_info(instr.opcode)
        
        # Check if instruction has Imm operands that need materialization
        needs_materialization = False
        new_operands = list(instr.operands)
        
        for i, op in enumerate(instr.operands):
            if not isinstance(op, Imm):
                continue
                
            # Check if this immediate can stay as-is
            if is_inline_constant(op.value):
                # Inline constants are always okay
                continue
            
            if imm_info['can_use_literal']:
                # Check if this is the right operand for the literal
                if imm_info['literal_operand_idx'] == i or imm_info['literal_operand_idx'] is None:
                    # This instruction can use this immediate as a literal
                    # But we can only have one literal per instruction
                    # For now, allow the first one and materialize the rest
                    continue
            
            # This immediate needs to be materialized
            needs_materialization = True
            phys_reg = materializer.get_or_materialize(op.value, result)
            new_operands[i] = phys_reg
        
        # Build the potentially modified instruction
        if needs_materialization:
            new_instr = ExprInstr(
                opcode=instr.opcode,
                dst=instr.dst,
                operands=tuple(new_operands),
                comment=instr.comment,
                source_expr=instr.source_expr
            )
            result.append(new_instr)
        else:
            result.append(instr)
    
    return result

