# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Expression IR optimization passes: copy propagation, dead code elimination, and register coalescing.

This module provides optimization passes for ExprProgram that reduce register
pressure and eliminate redundant MOV instructions before register allocation.

Key passes:
1. copy_propagation: Forward-propagate copies to eliminate intermediate moves
2. dead_code_elimination: Remove unused instructions (especially MOV chains)
3. coalesce_virtual_regs: Union-Find based coalescing to merge equivalent vregs

Usage:
    from expr_opt import optimize_program
    
    prog = ExprProgram()
    # ... emit instructions ...
    
    prog, stats = optimize_program(prog)
    # prog now has fewer MOV instructions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from .expr_ir import (
    ExprProgram, ExprInstr, VReg, SReg, PhysVReg, PhysSReg, Imm, OpCode,
    VirtualReg, PhysicalReg, Operand, AnyReg,
)


@dataclass
class OptimizationStats:
    """Statistics from optimization passes."""
    copies_propagated: int = 0
    movs_eliminated: int = 0
    vregs_coalesced: int = 0
    instructions_before: int = 0
    instructions_after: int = 0


class UnionFind:
    """
    Union-Find data structure for register coalescing.
    
    Groups virtual registers into equivalence classes so that registers
    connected by MOV instructions can be assigned the same physical register.
    """
    
    def __init__(self):
        self._parent: Dict[VirtualReg, VirtualReg] = {}
        self._rank: Dict[VirtualReg, int] = {}
    
    def find(self, x: VirtualReg) -> VirtualReg:
        """Find the representative of x's equivalence class with path compression."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            return x
        
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]
    
    def union(self, x: VirtualReg, y: VirtualReg) -> bool:
        """
        Union the equivalence classes of x and y.
        
        Returns True if they were in different classes (i.e., a merge happened).
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        
        # Union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        
        return True
    
    def get_all_representatives(self) -> Set[VirtualReg]:
        """Get all unique representatives (for counting equivalence classes)."""
        return {self.find(x) for x in self._parent}


def is_virtual_reg(op: Operand) -> bool:
    """Check if operand is a virtual register (VReg or SReg)."""
    return isinstance(op, (VReg, SReg))


def is_physical_reg(op: Operand) -> bool:
    """Check if operand is a physical register."""
    return isinstance(op, (PhysVReg, PhysSReg))


def same_reg_type(a: Operand, b: Operand) -> bool:
    """
    Check if two operands have compatible register types.
    
    VReg/PhysVReg are compatible; SReg/PhysSReg are compatible.
    Mixing VGPR and SGPR types is not safe for propagation.
    """
    if isinstance(a, (VReg, PhysVReg)) and isinstance(b, (VReg, PhysVReg)):
        return True
    if isinstance(a, (SReg, PhysSReg)) and isinstance(b, (SReg, PhysSReg)):
        return True
    return False


def copy_propagation(prog: ExprProgram) -> Tuple[ExprProgram, int]:
    """
    Forward copy propagation pass.
    
    For MOV dst, src instructions where both are virtual registers,
    replace all uses of dst with src throughout the program.
    
    This enables dead code elimination to remove the MOV entirely.
    
    Args:
        prog: The ExprProgram to optimize
        
    Returns:
        Tuple of (optimized program, number of copies propagated)
    """
    # Build copy map: dst -> src for MOV instructions
    # Only track virtual-to-virtual or physical-to-virtual copies
    copy_map: Dict[VirtualReg, Operand] = {}
    
    for instr in prog.instructions:
        if instr.opcode == OpCode.MOV:
            dst = instr.dst
            src = instr.operands[0]
            
            # Only track if dst is virtual
            if is_virtual_reg(dst):
                # Source can be virtual or physical
                if is_virtual_reg(src) or is_physical_reg(src):
                    # Check type compatibility
                    if same_reg_type(dst, src):
                        copy_map[dst] = src
    
    if not copy_map:
        return prog, 0
    
    # Compute transitive closure of copy_map
    def get_ultimate_source(reg: Operand) -> Operand:
        """Follow copy chain to find the ultimate source."""
        visited = set()
        current = reg
        while is_virtual_reg(current) and current in copy_map:
            if current in visited:
                # Cycle detected - stop
                break
            visited.add(current)
            current = copy_map[current]
        return current
    
    # Resolve all copy chains
    resolved_map: Dict[VirtualReg, Operand] = {}
    for dst in copy_map:
        ultimate = get_ultimate_source(dst)
        if ultimate != dst:
            resolved_map[dst] = ultimate
    
    if not resolved_map:
        return prog, 0
    
    # Rewrite all instructions to use ultimate sources
    copies_propagated = 0
    new_instructions: List[ExprInstr] = []
    
    for instr in prog.instructions:
        # Rewrite operands
        new_operands = []
        changed = False
        
        for op in instr.operands:
            if is_virtual_reg(op) and op in resolved_map:
                new_op = resolved_map[op]
                # Verify type compatibility for this specific use
                if same_reg_type(op, new_op):
                    new_operands.append(new_op)
                    changed = True
                    copies_propagated += 1
                else:
                    new_operands.append(op)
            else:
                new_operands.append(op)
        
        if changed:
            new_instr = ExprInstr(
                opcode=instr.opcode,
                dst=instr.dst,
                operands=tuple(new_operands),
                comment=instr.comment,
                source_expr=instr.source_expr,
            )
            new_instructions.append(new_instr)
        else:
            new_instructions.append(instr)
    
    # Build new program
    new_prog = ExprProgram()
    new_prog._next_vreg = prog._next_vreg
    new_prog._next_sreg = prog._next_sreg
    new_prog.instructions = new_instructions
    
    return new_prog, copies_propagated


def dead_code_elimination(prog: ExprProgram) -> Tuple[ExprProgram, int]:
    """
    Dead code elimination pass.
    
    Removes instructions whose results are never used:
    - MOV dst, src where dst has no uses
    - MOV dst, dst (identity moves after copy propagation)
    - Any instruction with unused result and no side effects
    
    Args:
        prog: The ExprProgram to optimize
        
    Returns:
        Tuple of (optimized program, number of instructions eliminated)
    """
    # Build use count for each virtual register
    use_count: Dict[VirtualReg, int] = {}
    
    for instr in prog.instructions:
        for use in instr.get_uses():
            if is_virtual_reg(use):
                use_count[use] = use_count.get(use, 0) + 1
    
    # Also count the "implicit" use of the last instruction's result
    # (it's the output of the expression)
    if prog.instructions:
        last_dst = prog.instructions[-1].get_def()
        if is_virtual_reg(last_dst):
            use_count[last_dst] = use_count.get(last_dst, 0) + 1
    
    # Identify dead instructions
    # An instruction is dead if:
    # 1. It's a MOV and dst has no uses
    # 2. It's a MOV dst, dst (identity)
    # 3. It's any pure instruction with unused result
    
    dead_indices: Set[int] = set()
    
    for i, instr in enumerate(prog.instructions):
        dst = instr.get_def()
        
        # Skip if not virtual (physical regs may have external uses)
        if not is_virtual_reg(dst):
            continue
        
        # Check for identity MOV (dst == src after propagation)
        if instr.opcode == OpCode.MOV:
            src = instr.operands[0]
            if dst == src:
                dead_indices.add(i)
                continue
        
        # Check for unused result (but not the last instruction)
        if i < len(prog.instructions) - 1:
            if use_count.get(dst, 0) == 0:
                # Only eliminate side-effect-free instructions
                if instr.opcode in {OpCode.MOV, OpCode.MOV_IMM, OpCode.ADD, OpCode.ADD_IMM,
                                   OpCode.MUL, OpCode.MUL_IMM, OpCode.LSHL, OpCode.LSHR,
                                   OpCode.AND, OpCode.AND_IMM, OpCode.OR, OpCode.BFE}:
                    dead_indices.add(i)
    
    if not dead_indices:
        return prog, 0
    
    # Build new program without dead instructions
    new_instructions = [instr for i, instr in enumerate(prog.instructions) if i not in dead_indices]
    
    new_prog = ExprProgram()
    new_prog._next_vreg = prog._next_vreg
    new_prog._next_sreg = prog._next_sreg
    new_prog.instructions = new_instructions
    
    return new_prog, len(dead_indices)


def coalesce_virtual_regs(prog: ExprProgram) -> Tuple[ExprProgram, int]:
    """
    Virtual register coalescing pass using Union-Find.
    
    For remaining MOV vA, vB instructions (after copy-prop), union the
    registers so that regalloc can assign them the same physical register.
    
    This rewrites all uses to use the equivalence class representative.
    
    Args:
        prog: The ExprProgram to optimize
        
    Returns:
        Tuple of (optimized program, number of register pairs coalesced)
    """
    uf = UnionFind()
    coalesced = 0
    
    # First pass: build equivalence classes from remaining MOV instructions
    for instr in prog.instructions:
        if instr.opcode == OpCode.MOV:
            dst = instr.dst
            src = instr.operands[0]
            
            # Only coalesce virtual-to-virtual with same type
            if is_virtual_reg(dst) and is_virtual_reg(src) and same_reg_type(dst, src):
                if uf.union(dst, src):
                    coalesced += 1
    
    if coalesced == 0:
        return prog, 0
    
    # Second pass: rewrite all virtual registers to their representative
    def rewrite_reg(reg: Operand) -> Operand:
        if is_virtual_reg(reg):
            rep = uf.find(reg)
            return rep
        return reg
    
    new_instructions: List[ExprInstr] = []
    
    for instr in prog.instructions:
        # Rewrite dst
        new_dst = rewrite_reg(instr.dst)
        
        # Rewrite operands
        new_operands = tuple(rewrite_reg(op) for op in instr.operands)
        
        # Check if this became an identity MOV after coalescing
        if instr.opcode == OpCode.MOV and new_dst == new_operands[0]:
            # Skip this instruction - it's now a no-op
            continue
        
        new_instr = ExprInstr(
            opcode=instr.opcode,
            dst=new_dst,
            operands=new_operands,
            comment=instr.comment,
            source_expr=instr.source_expr,
        )
        new_instructions.append(new_instr)
    
    new_prog = ExprProgram()
    new_prog._next_vreg = prog._next_vreg
    new_prog._next_sreg = prog._next_sreg
    new_prog.instructions = new_instructions
    
    return new_prog, coalesced


def optimize_program(prog: ExprProgram) -> Tuple[ExprProgram, OptimizationStats]:
    """
    Run all optimization passes on an ExprProgram.
    
    Passes are run in order:
    1. Copy propagation (forward propagate copies)
    2. Dead code elimination (remove unused MOVs)
    3. Register coalescing (merge equivalent vregs)
    4. Final DCE pass (cleanup after coalescing)
    
    Args:
        prog: The ExprProgram to optimize
        
    Returns:
        Tuple of (optimized program, statistics)
    """
    stats = OptimizationStats()
    stats.instructions_before = len(prog.instructions)
    
    # Pass 1: Copy propagation
    prog, copies = copy_propagation(prog)
    stats.copies_propagated = copies
    
    # Pass 2: DCE after copy-prop
    prog, eliminated1 = dead_code_elimination(prog)
    
    # Pass 3: Register coalescing
    prog, coalesced = coalesce_virtual_regs(prog)
    stats.vregs_coalesced = coalesced
    
    # Pass 4: Final DCE (coalescing may create more dead code)
    prog, eliminated2 = dead_code_elimination(prog)
    
    stats.movs_eliminated = eliminated1 + eliminated2
    stats.instructions_after = len(prog.instructions)
    
    return prog, stats

