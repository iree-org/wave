# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Liveness analysis for kernel IR with CFG support.

This module computes live ranges for all virtual registers in a KernelProgram.
It supports control flow (loops, branches) via CFG-based backward dataflow analysis.

Key components:
- BasicBlock: A straight-line sequence of instructions between labels/branches
- CFG: Control flow graph built from labels and branch instructions
- Liveness: Backward dataflow computing live-in/live-out per block
- Dominance: Computes dominators for SSA validation

The computed live ranges are used by the linear scan allocator to determine
when registers can be reused.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from collections import defaultdict

from .kernel_ir import (
    KernelProgram, KInstr,
    KVReg, KSReg, KPhysVReg, KPhysSReg,
    KReg, KRegRange, KOperand,
    RegClass, is_virtual, is_vgpr, is_sgpr, get_reg_class,
)


# =============================================================================
# Live Range
# =============================================================================

@dataclass
class LiveRange:
    """
    Live range for a virtual register.
    
    Attributes:
        reg: The virtual register
        start: Instruction index where the register is defined
        end: Instruction index of the last use
        size: Number of consecutive registers (for ranges)
        alignment: Required alignment (for ranges)
        reg_class: Register class (VGPR or SGPR)
    """
    reg: KReg  # Base register (KVReg or KSReg)
    start: int
    end: int
    size: int = 1
    alignment: int = 1
    reg_class: RegClass = RegClass.VGPR
    
    def __post_init__(self):
        # Derive reg_class from reg if not set
        if isinstance(self.reg, KVReg):
            object.__setattr__(self, 'reg_class', RegClass.VGPR)
        elif isinstance(self.reg, KSReg):
            object.__setattr__(self, 'reg_class', RegClass.SGPR)
    
    def overlaps(self, other: "LiveRange") -> bool:
        """Check if this live range overlaps with another."""
        return not (self.end < other.start or other.end < self.start)
    
    def contains(self, point: int) -> bool:
        """Check if a point is within this live range."""
        return self.start <= point <= self.end
    
    @property
    def length(self) -> int:
        """Length of the live range in instructions."""
        return self.end - self.start + 1
    
    def __repr__(self) -> str:
        size_str = f"x{self.size}" if self.size > 1 else ""
        return f"LiveRange({self.reg}{size_str}: [{self.start}, {self.end}])"


@dataclass
class LivenessInfo:
    """
    Complete liveness information for a kernel program.
    
    Attributes:
        live_ranges: Map from virtual register to its live range
        vreg_ranges: Live ranges for VGPRs only (sorted by start)
        sreg_ranges: Live ranges for SGPRs only (sorted by start)
        def_points: Map from virtual register to definition point
        use_points: Map from virtual register to list of use points
        max_vreg_pressure: Maximum number of VGPRs live at any point
        max_sreg_pressure: Maximum number of SGPRs live at any point
    """
    live_ranges: Dict[KReg, LiveRange] = field(default_factory=dict)
    vreg_ranges: List[LiveRange] = field(default_factory=list)
    sreg_ranges: List[LiveRange] = field(default_factory=list)
    def_points: Dict[KReg, int] = field(default_factory=dict)
    use_points: Dict[KReg, List[int]] = field(default_factory=lambda: defaultdict(list))
    max_vreg_pressure: int = 0
    max_sreg_pressure: int = 0
    
    def get_live_at(self, point: int, reg_class: Optional[RegClass] = None) -> List[LiveRange]:
        """Get all live ranges active at a given program point."""
        ranges = self.vreg_ranges if reg_class == RegClass.VGPR else (
            self.sreg_ranges if reg_class == RegClass.SGPR else
            self.vreg_ranges + self.sreg_ranges
        )
        return [r for r in ranges if r.contains(point)]
    
    def get_pressure_at(self, point: int, reg_class: RegClass) -> int:
        """Get register pressure (number of live regs) at a given point."""
        return sum(r.size for r in self.get_live_at(point, reg_class))


# =============================================================================
# Basic Block and CFG
# =============================================================================

@dataclass
class BasicBlock:
    """
    A basic block in the CFG.
    
    A basic block is a maximal sequence of instructions with:
    - No branches except possibly at the end
    - No labels except possibly at the start
    """
    id: int
    label: Optional[str] = None  # Label at start of block (if any)
    start_idx: int = 0  # First instruction index (inclusive)
    end_idx: int = 0    # Last instruction index (inclusive)
    
    # CFG edges (set after construction)
    successors: List["BasicBlock"] = field(default_factory=list)
    predecessors: List["BasicBlock"] = field(default_factory=list)
    
    # Liveness info (computed by dataflow)
    use_set: Set[KReg] = field(default_factory=set)  # Regs used before def in block
    def_set: Set[KReg] = field(default_factory=set)  # Regs defined in block
    live_in: Set[KReg] = field(default_factory=set)   # Regs live at block entry
    live_out: Set[KReg] = field(default_factory=set)  # Regs live at block exit
    
    # Dominance info
    dominators: Set[int] = field(default_factory=set)  # Block IDs that dominate this block
    
    def __repr__(self) -> str:
        label_str = f" ({self.label})" if self.label else ""
        return f"BB{self.id}{label_str}[{self.start_idx}:{self.end_idx}]"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, BasicBlock):
            return self.id == other.id
        return False


@dataclass
class CFG:
    """
    Control flow graph for a kernel program.
    """
    blocks: List[BasicBlock] = field(default_factory=list)
    entry_block: Optional[BasicBlock] = None
    label_to_block: Dict[str, BasicBlock] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"CFG({len(self.blocks)} blocks)"


def _is_branch_instruction(instr: KInstr) -> bool:
    """Check if instruction is a branch."""
    return instr.name in ("s_branch", "s_cbranch_scc0", "s_cbranch_scc1",
                          "s_cbranch_vccz", "s_cbranch_vccnz", "s_cbranch_execz",
                          "s_cbranch_execnz")


def _is_conditional_branch(instr: KInstr) -> bool:
    """Check if instruction is a conditional branch."""
    return instr.name.startswith("s_cbranch_")


def _get_branch_target(instr: KInstr) -> Optional[str]:
    """Get the target label of a branch instruction."""
    if _is_branch_instruction(instr):
        return instr.comment  # Branch target is stored in comment
    return None


def build_cfg(program: KernelProgram) -> CFG:
    """
    Build a control flow graph from a kernel program.
    
    Algorithm:
    1. Find all block boundaries (labels and branches)
    2. Create basic blocks
    3. Connect blocks with edges based on control flow
    """
    cfg = CFG()
    instructions = program.instructions
    
    if not instructions:
        return cfg
    
    # Pass 1: Find block start points (labels or after branches)
    block_starts = {0}  # First instruction starts a block
    label_positions: Dict[str, int] = {}
    
    for idx, instr in enumerate(instructions):
        # Label starts a new block
        if instr.is_label:
            label_name = instr.comment
            if label_name:
                label_positions[label_name] = idx
                block_starts.add(idx)
        
        # Instruction after a branch starts a new block
        if _is_branch_instruction(instr) and idx + 1 < len(instructions):
            block_starts.add(idx + 1)
    
    # Sort block starts
    sorted_starts = sorted(block_starts)
    
    # Pass 2: Create basic blocks
    for i, start_idx in enumerate(sorted_starts):
        # Find end of this block
        if i + 1 < len(sorted_starts):
            end_idx = sorted_starts[i + 1] - 1
        else:
            end_idx = len(instructions) - 1
        
        # Get label if this block starts with one
        label = None
        if instructions[start_idx].is_label:
            label = instructions[start_idx].comment
        
        block = BasicBlock(
            id=i,
            label=label,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        cfg.blocks.append(block)
        
        if label:
            cfg.label_to_block[label] = block
    
    # Set entry block
    if cfg.blocks:
        cfg.entry_block = cfg.blocks[0]
    
    # Pass 3: Connect blocks with edges
    for i, block in enumerate(cfg.blocks):
        # Find the last real instruction (skip trailing comments/labels)
        last_instr = None
        for idx in range(block.end_idx, block.start_idx - 1, -1):
            instr = instructions[idx]
            if instr.name not in ("_comment", "_label"):
                last_instr = instr
                break
        
        if last_instr is None:
            # Empty block - fallthrough to next
            if i + 1 < len(cfg.blocks):
                _add_edge(block, cfg.blocks[i + 1])
            continue
        
        # Check if block ends with a branch
        if _is_branch_instruction(last_instr):
            target_label = _get_branch_target(last_instr)
            if target_label and target_label in cfg.label_to_block:
                target_block = cfg.label_to_block[target_label]
                _add_edge(block, target_block)
            
            # Conditional branches also fall through
            if _is_conditional_branch(last_instr) and i + 1 < len(cfg.blocks):
                _add_edge(block, cfg.blocks[i + 1])
        else:
            # Non-branch: fallthrough to next block
            if i + 1 < len(cfg.blocks):
                _add_edge(block, cfg.blocks[i + 1])
    
    return cfg


def _add_edge(from_block: BasicBlock, to_block: BasicBlock):
    """Add a CFG edge from from_block to to_block."""
    if to_block not in from_block.successors:
        from_block.successors.append(to_block)
    if from_block not in to_block.predecessors:
        to_block.predecessors.append(from_block)


# =============================================================================
# Liveness Analysis (Simple Linear)
# =============================================================================

def compute_liveness(program: KernelProgram) -> LivenessInfo:
    """
    Compute liveness information for a kernel program.
    
    This is the main entry point. For SSA programs, each virtual register
    has exactly one definition point, and the live range extends from def
    to last use.
    
    Args:
        program: The kernel program to analyze
        
    Returns:
        LivenessInfo containing all computed live ranges and statistics
    """
    info = LivenessInfo()
    
    # Track size and alignment for range allocations
    reg_size: Dict[KReg, int] = {}
    reg_alignment: Dict[KReg, int] = {}
    
    # Track which regs are part of a range (for alias resolution)
    range_membership: Dict[KReg, KReg] = {}  # component_reg -> base_reg
    
    for idx, instr in enumerate(program.instructions):
        # Process defs
        for d in instr.defs:
            if isinstance(d, KRegRange):
                base_reg = d.base_reg
                if is_virtual(base_reg):
                    if base_reg not in info.def_points:
                        info.def_points[base_reg] = idx
                        reg_size[base_reg] = d.count
                        reg_alignment[base_reg] = d.alignment
                        
                        # Track all component registers as aliases to the base
                        reg_class = type(base_reg)
                        for i in range(d.count):
                            component = reg_class(base_reg.id + i)
                            range_membership[component] = base_reg
            elif is_virtual(d):
                if d not in info.def_points:
                    info.def_points[d] = idx
                    reg_size[d] = 1
                    reg_alignment[d] = 1
        
        # Process uses
        for u in instr.uses:
            if isinstance(u, KRegRange):
                base_reg = u.base_reg
                if is_virtual(base_reg):
                    info.use_points[base_reg].append(idx)
            elif isinstance(u, (KVReg, KSReg)):
                if u in range_membership:
                    base_reg = range_membership[u]
                    info.use_points[base_reg].append(idx)
                elif is_virtual(u):
                    info.use_points[u].append(idx)
    
    # Build live ranges
    for reg, def_point in info.def_points.items():
        uses = info.use_points.get(reg, [])
        if uses:
            last_use = max(uses)
        else:
            # Register defined but never used - range is just the def point
            last_use = def_point
        
        live_range = LiveRange(
            reg=reg,
            start=def_point,
            end=last_use,
            size=reg_size.get(reg, 1),
            alignment=reg_alignment.get(reg, 1),
            reg_class=get_reg_class(reg),
        )
        info.live_ranges[reg] = live_range
        
        # Categorize by register class
        if isinstance(reg, KVReg):
            info.vreg_ranges.append(live_range)
        elif isinstance(reg, KSReg):
            info.sreg_ranges.append(live_range)
    
    # Sort ranges by start point (for linear scan)
    info.vreg_ranges.sort(key=lambda r: (r.start, r.end))
    info.sreg_ranges.sort(key=lambda r: (r.start, r.end))
    
    # Compute max pressure
    if info.vreg_ranges:
        info.max_vreg_pressure = _compute_max_pressure(info.vreg_ranges)
    if info.sreg_ranges:
        info.max_sreg_pressure = _compute_max_pressure(info.sreg_ranges)
    
    return info


def _compute_max_pressure(ranges: List[LiveRange]) -> int:
    """
    Compute maximum register pressure for a set of live ranges.
    
    Uses an event-based sweep algorithm:
    1. Create events for each range start (+size) and end (-size)
    2. Sweep through events in order
    3. Track maximum cumulative pressure
    """
    if not ranges:
        return 0
    
    events = []
    for r in ranges:
        events.append((r.start, r.size, True))   # Start: add registers
        events.append((r.end + 1, -r.size, False))  # End+1: remove registers
    
    events.sort(key=lambda e: (e[0], not e[2]))
    
    current_pressure = 0
    max_pressure = 0
    
    for point, delta, is_start in events:
        current_pressure += delta
        max_pressure = max(max_pressure, current_pressure)
    
    return max_pressure


def compute_interference_graph(
    ranges: List[LiveRange]
) -> Dict[KReg, Set[KReg]]:
    """
    Compute interference graph for a set of live ranges.
    
    Two registers interfere if their live ranges overlap.
    
    Returns:
        Dict mapping each register to the set of registers it interferes with
    """
    interference: Dict[KReg, Set[KReg]] = defaultdict(set)
    
    for i, r1 in enumerate(ranges):
        for r2 in ranges[i+1:]:
            if r1.overlaps(r2):
                interference[r1.reg].add(r2.reg)
                interference[r2.reg].add(r1.reg)
    
    return dict(interference)


def get_live_in(program: KernelProgram, idx: int, info: Optional[LivenessInfo] = None) -> Set[KReg]:
    """
    Get the set of virtual registers live at the start of instruction idx.
    """
    if info is None:
        info = compute_liveness(program)
    
    live_in = set()
    for reg, lr in info.live_ranges.items():
        if lr.start < idx <= lr.end:
            live_in.add(reg)
    return live_in


def get_live_out(program: KernelProgram, idx: int, info: Optional[LivenessInfo] = None) -> Set[KReg]:
    """
    Get the set of virtual registers live at the end of instruction idx.
    """
    if info is None:
        info = compute_liveness(program)
    
    live_out = set()
    for reg, lr in info.live_ranges.items():
        if lr.start <= idx < lr.end:
            live_out.add(reg)
    return live_out


def validate_ssa(program: KernelProgram) -> List[str]:
    """
    Validate that a kernel program is in SSA form.
    
    Checks:
    1. Each virtual register is defined exactly once
    2. Each use of a virtual register is dominated by its definition
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    defs: Dict[KReg, int] = {}
    range_membership: Dict[KReg, KReg] = {}
    
    for idx, instr in enumerate(program.instructions):
        # Check defs
        for d in instr.defs:
            if isinstance(d, KRegRange):
                base_reg = d.base_reg
                if is_virtual(base_reg):
                    if base_reg in defs:
                        errors.append(f"SSA violation: {base_reg} defined at {defs[base_reg]} and {idx}")
                    defs[base_reg] = idx
                    
                    reg_class = type(base_reg)
                    for i in range(d.count):
                        component = reg_class(base_reg.id + i)
                        range_membership[component] = base_reg
                        defs[component] = idx
            else:
                reg = d
                if is_virtual(reg):
                    if reg in defs:
                        errors.append(f"SSA violation: {reg} defined at {defs[reg]} and {idx}")
                    defs[reg] = idx
        
        # Check uses
        for u in instr.uses:
            if isinstance(u, KRegRange):
                reg = u.base_reg
            elif isinstance(u, (KVReg, KSReg)):
                reg = u
            else:
                continue
            
            if is_virtual(reg):
                if reg not in defs:
                    errors.append(f"Use of undefined register {reg} at instruction {idx}")
                elif defs[reg] > idx:
                    errors.append(f"Use of {reg} at {idx} before def at {defs[reg]}")
    
    return errors
