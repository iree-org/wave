# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Bridge between expression emitter and kernel-level IR.

This module provides a compatibility layer that allows the new kernel-level
IR and register allocation to work alongside the existing expression emitter.

The bridge supports two modes:
1. Expression-level emission (current): Each expression gets its own micro-program
2. Kernel-level emission (new): All expressions emit to a shared KernelProgram

This allows incremental migration without breaking existing functionality.

When WAVE_KERNEL_LSRA=1 is set, the bridge:
- Collects all expressions emitted during kernel compilation
- At finalization, performs whole-program liveness analysis and allocation
- Generates the final assembly with optimally allocated registers
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
import sympy

from .kernel_ir import (
    KernelProgram, KernelBuilder, KInstr, KOpcode,
    KVReg, KSReg, KPhysVReg, KPhysSReg,
    KReg, KRegRange, KImm, KMemOffset,
    RegClass,
)
from .kernel_liveness import compute_liveness
from .kernel_regalloc import allocate_kernel, AllocationError
from .kernel_render import PhysicalMapping, render_program


# Feature flag
def use_kernel_lsra() -> bool:
    """Check if kernel-level LSRA is enabled."""
    return os.environ.get("WAVE_KERNEL_LSRA", "0") == "1"


@dataclass
class ExpressionBridge:
    """
    Bridge that adapts expression emission to kernel-level IR.
    
    This class wraps expression emission to optionally use the new
    kernel-level IR while maintaining compatibility with the existing
    expression emitter interface.
    
    In bridge mode, it:
    1. Intercepts expression emission calls
    2. Builds equivalent KernelProgram instructions
    3. Defers physical register allocation until finalization
    
    This allows testing the new infrastructure without modifying
    the core expression emitter.
    """
    
    # The underlying kernel program (when in kernel LSRA mode)
    program: Optional[KernelProgram] = None
    builder: Optional[KernelBuilder] = None
    
    # Mapping from physical regs (returned by expr_emitter) to virtual regs
    _phys_to_virtual: Dict[str, KVReg] = field(default_factory=dict, init=False)
    
    # Reverse mapping (virtual to physical, after allocation)
    _virtual_to_phys: Dict[int, str] = field(default_factory=dict, init=False)
    
    # Expression CSE cache
    _expr_cache: Dict[tuple, KVReg] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        if use_kernel_lsra():
            self.program = KernelProgram()
            self.builder = KernelBuilder(self.program)
    
    @property
    def is_active(self) -> bool:
        """Check if kernel LSRA mode is active."""
        return self.program is not None
    
    def record_expr_result(self, phys_reg: str, vreg: KVReg) -> None:
        """Record mapping from physical register string to virtual register."""
        self._phys_to_virtual[phys_reg] = vreg
    
    def wrap_expr_emission(
        self,
        emit_fn: Callable[[], str],  # Returns "vN" physical reg string
        expr_key: Optional[tuple] = None,  # For CSE
    ) -> str:
        """
        Wrap expression emission to record virtual registers.
        
        In kernel LSRA mode:
        - Allocates a virtual register
        - Records the mapping from physical result to virtual
        - Returns the physical register string (for compatibility)
        
        Args:
            emit_fn: Function that emits the expression and returns physical reg
            expr_key: Optional CSE key for deduplication
            
        Returns:
            Physical register string (e.g., "v5")
        """
        if not self.is_active:
            # Not in kernel LSRA mode - just call the function
            return emit_fn()
        
        # Check CSE cache
        if expr_key is not None and expr_key in self._expr_cache:
            # Found in cache - return the previously allocated physical reg
            cached_vreg = self._expr_cache[expr_key]
            # Find the physical reg that maps to this virtual reg
            for phys, vreg in self._phys_to_virtual.items():
                if vreg == cached_vreg:
                    return phys
        
        # Emit the expression (uses existing infrastructure)
        phys_reg = emit_fn()
        
        # Allocate a virtual register and record mapping
        vreg = self.builder.vreg()
        self.record_expr_result(phys_reg, vreg)
        
        # Cache for CSE
        if expr_key is not None:
            self._expr_cache[expr_key] = vreg
        
        return phys_reg
    
    def get_virtual_for_phys(self, phys_reg: str) -> Optional[KVReg]:
        """Get the virtual register corresponding to a physical register."""
        return self._phys_to_virtual.get(phys_reg)
    
    def finalize(self) -> Optional[PhysicalMapping]:
        """
        Finalize kernel-level allocation.
        
        This performs:
        1. Liveness analysis on the recorded program
        2. Linear scan allocation
        3. Updates physical register mappings
        
        Returns:
            PhysicalMapping if in kernel LSRA mode, None otherwise
        """
        if not self.is_active:
            return None
        
        if len(self.program) == 0:
            # Empty program - nothing to allocate
            return PhysicalMapping({}, {})
        
        try:
            mapping, stats = allocate_kernel(self.program)
            
            # Update mappings
            for phys_str, vreg in self._phys_to_virtual.items():
                if isinstance(vreg, KVReg):
                    new_phys = mapping.vreg_map.get(vreg.id)
                    if new_phys is not None:
                        self._virtual_to_phys[vreg.id] = f"v{new_phys}"
            
            return mapping
            
        except AllocationError as e:
            # Log the error but don't fail - fall back to existing allocation
            print(f"[KERNEL_LSRA] Allocation failed: {e}")
            return None


def create_expression_bridge() -> ExpressionBridge:
    """Create an expression bridge for the current compilation."""
    return ExpressionBridge()

