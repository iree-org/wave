# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Liveness analysis for kernel IR.

This module computes live ranges for all virtual registers in a KernelProgram.
Since the MLIR input is SSA (each value has exactly one definition), liveness
analysis is straightforward:
- Each virtual register has exactly one def point
- The live range extends from def to last use

The computed live ranges are used by the linear scan allocator to determine
when registers can be reused.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .kernel_ir import (
    KernelProgram, KInstr,
    KVReg, KSReg, KPhysVReg, KPhysSReg,
    KReg, KRegRange, KOperand,
    RegClass, is_virtual, is_vgpr, is_sgpr, get_reg_class,
)


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


def compute_liveness(program: KernelProgram) -> LivenessInfo:
    """
    Compute liveness information for a kernel program.
    
    This function:
    1. Scans instructions to find def points for each virtual register
    2. Scans instructions to find use points for each virtual register
    3. Creates live ranges from (def_point, last_use_point)
    4. Computes register pressure statistics
    
    For SSA programs (which kernel IR is), each virtual register has exactly
    one definition point, so liveness is simply [def, last_use].
    
    Args:
        program: The kernel program to analyze
        
    Returns:
        LivenessInfo containing all computed live ranges and statistics
    """
    info = LivenessInfo()
    
    # Pass 1: Find all defs and uses
    reg_size: Dict[KReg, int] = {}  # Track size for range allocations
    reg_alignment: Dict[KReg, int] = {}  # Track alignment requirements
    
    for idx, instr in enumerate(program.instructions):
        # Process defs
        for d in instr.defs:
            if isinstance(d, KRegRange):
                base_reg = d.base_reg
                if is_virtual(base_reg):
                    if base_reg in info.def_points:
                        raise ValueError(f"SSA violation: {base_reg} defined twice "
                                       f"(at {info.def_points[base_reg]} and {idx})")
                    info.def_points[base_reg] = idx
                    reg_size[base_reg] = d.count
                    reg_alignment[base_reg] = d.alignment
            elif is_virtual(d):
                if d in info.def_points:
                    raise ValueError(f"SSA violation: {d} defined twice "
                                   f"(at {info.def_points[d]} and {idx})")
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
                info.use_points[u].append(idx)
    
    # Pass 2: Build live ranges
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
    
    # Pass 3: Compute max pressure
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
    
    # Create events: (point, delta, is_start)
    # is_start=True events come before is_start=False at same point
    events = []
    for r in ranges:
        events.append((r.start, r.size, True))   # Start: add registers
        events.append((r.end + 1, -r.size, False))  # End+1: remove registers
    
    # Sort by point, then by is_start (starts before ends at same point)
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
    
    Two registers interfere if their live ranges overlap. This is useful
    for graph coloring allocators (not used by linear scan but provided
    for analysis purposes).
    
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
    
    A register is live-in at idx if:
    - It is defined before idx, AND
    - It is used at or after idx
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
    
    A register is live-out at idx if:
    - It is defined at or before idx, AND
    - It is used after idx
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
       (i.e., def comes before use in the instruction stream)
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    defs: Dict[KReg, int] = {}
    
    for idx, instr in enumerate(program.instructions):
        # Check defs
        for d in instr.defs:
            if isinstance(d, KRegRange):
                reg = d.base_reg
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

