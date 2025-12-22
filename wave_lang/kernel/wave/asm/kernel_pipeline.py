# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Kernel-level compilation pipeline with whole-program register allocation.

This module provides the integration point for the new kernel-level linear scan
register allocator. It can be enabled via the WAVE_KERNEL_LSRA environment variable.

When enabled:
1. Expression emission goes to KernelProgram with virtual registers
2. After all instructions are emitted, liveness is computed
3. Linear scan allocator assigns physical registers
4. Renderer generates final assembly

When disabled (default):
- The original per-expression allocation path is used

Usage:
    from kernel_pipeline import use_kernel_lsra, KernelCompilationContext
    
    if use_kernel_lsra():
        ctx = KernelCompilationContext(kernel_info)
        # Emit instructions to ctx.program
        ctx.v_add_u32(...)
        # Finalize and get assembly
        asm = ctx.finalize()
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .kernel_ir import (
    KernelProgram, KernelBuilder, KInstr, KOpcode,
    KVReg, KSReg, KPhysVReg, KPhysSReg,
    KReg, KRegRange, KImm, KMemOffset,
    KernelABI, RegClass,
)
from .kernel_liveness import compute_liveness, LivenessInfo
from .kernel_regalloc import KernelRegAlloc, allocate_kernel, AllocationStats, AllocationError
from .kernel_generator import KernelGenerator, PhysicalMapping, generate_program


# Environment variable to enable kernel-level LSRA
# Default is "0" (disabled) during development
WAVE_KERNEL_LSRA_ENV = "WAVE_KERNEL_LSRA"


def use_kernel_lsra() -> bool:
    """Check if kernel-level LSRA is enabled."""
    return os.environ.get(WAVE_KERNEL_LSRA_ENV, "0") == "1"


@dataclass
class KernelCompilationContext:
    """
    Context for kernel compilation with whole-program register allocation.
    
    This context manages:
    - The KernelProgram being built
    - Symbol bindings (MLIR SSA values to virtual registers)
    - ABI register configuration
    - CSE cache for expression deduplication
    
    Usage:
        ctx = KernelCompilationContext(max_vgprs=256, max_sgprs=104)
        
        # Emit instructions
        result = ctx.v_add_u32(src1, src2)
        ctx.ds_read_b64(addr)
        
        # Finalize and get assembly
        asm_lines = ctx.finalize()
    """
    
    # Configuration
    max_vgprs: int = 256
    max_sgprs: int = 104
    
    # ABI configuration
    use_flat_tid: bool = True
    use_workgroup_ids: Tuple[bool, bool, bool] = (True, True, True)  # x, y, z
    
    # Internal state
    program: KernelProgram = field(init=False)
    builder: KernelBuilder = field(init=False)
    
    # Symbol bindings: MLIR SSA value string -> virtual register
    _symbol_bindings: Dict[str, KReg] = field(default_factory=dict, init=False)
    
    # CSE cache: expression key -> virtual register
    _cse_cache: Dict[tuple, KVReg] = field(default_factory=dict, init=False)
    
    # Statistics
    _cse_hits: int = field(default=0, init=False)
    
    def __post_init__(self):
        # Initialize ABI
        abi = KernelABI()
        if self.use_flat_tid:
            abi.flat_tid_vreg = KPhysVReg(0)
        if self.use_workgroup_ids[0]:
            abi.workgroup_id_x_sreg = KPhysSReg(2)
        if self.use_workgroup_ids[1]:
            abi.workgroup_id_y_sreg = KPhysSReg(3)
        if self.use_workgroup_ids[2]:
            abi.workgroup_id_z_sreg = KPhysSReg(4)
        
        # Create program
        self.program = KernelProgram(abi=abi, max_vgprs=self.max_vgprs, max_sgprs=self.max_sgprs)
        self.builder = KernelBuilder(self.program)
    
    # =========================================================================
    # Symbol binding (for MLIR SSA value tracking)
    # =========================================================================
    
    def bind_symbol(self, symbol: str, reg: KReg) -> None:
        """Bind an MLIR SSA value name to a virtual register."""
        self._symbol_bindings[symbol] = reg
    
    def get_binding(self, symbol: str) -> Optional[KReg]:
        """Get the virtual register bound to an MLIR SSA value."""
        return self._symbol_bindings.get(symbol)
    
    def require_binding(self, symbol: str) -> KReg:
        """Get the virtual register bound to an MLIR SSA value, or raise."""
        if symbol not in self._symbol_bindings:
            raise ValueError(f"Symbol '{symbol}' not bound to any register")
        return self._symbol_bindings[symbol]
    
    # =========================================================================
    # CSE support
    # =========================================================================
    
    def cse_lookup(self, key: tuple) -> Optional[KVReg]:
        """Look up a value in the CSE cache."""
        return self._cse_cache.get(key)
    
    def cse_insert(self, key: tuple, reg: KVReg) -> None:
        """Insert a value into the CSE cache."""
        self._cse_cache[key] = reg
    
    def cse_get_or_emit(self, key: tuple, emit_fn) -> KVReg:
        """Get from CSE cache or emit using the provided function."""
        if key in self._cse_cache:
            self._cse_hits += 1
            return self._cse_cache[key]
        result = emit_fn()
        self._cse_cache[key] = result
        return result
    
    # =========================================================================
    # Instruction emission (delegates to builder)
    # =========================================================================
    
    def vreg(self) -> KVReg:
        """Allocate a new virtual VGPR."""
        return self.builder.vreg()
    
    def sreg(self) -> KSReg:
        """Allocate a new virtual SGPR."""
        return self.builder.sreg()
    
    def v_mov_b32(self, src, comment: str = None) -> KVReg:
        return self.builder.v_mov_b32(src, comment)
    
    def v_add_u32(self, src1, src2, comment: str = None) -> KVReg:
        return self.builder.v_add_u32(src1, src2, comment)
    
    def v_mul_lo_u32(self, src1, src2, comment: str = None) -> KVReg:
        return self.builder.v_mul_lo_u32(src1, src2, comment)
    
    def v_and_b32(self, src1, src2, comment: str = None) -> KVReg:
        return self.builder.v_and_b32(src1, src2, comment)
    
    def v_or_b32(self, src1, src2, comment: str = None) -> KVReg:
        return self.builder.v_or_b32(src1, src2, comment)
    
    def v_lshlrev_b32(self, shift, src, comment: str = None) -> KVReg:
        return self.builder.v_lshlrev_b32(shift, src, comment)
    
    def v_lshrrev_b32(self, shift, src, comment: str = None) -> KVReg:
        return self.builder.v_lshrrev_b32(shift, src, comment)
    
    def v_bfe_u32(self, src, offset, width, comment: str = None) -> KVReg:
        return self.builder.v_bfe_u32(src, offset, width, comment)
    
    def v_lshl_add_u32(self, src, shift, addend, comment: str = None) -> KVReg:
        return self.builder.v_lshl_add_u32(src, shift, addend, comment)
    
    def s_mov_b32(self, src, comment: str = None) -> KSReg:
        return self.builder.s_mov_b32(src, comment)
    
    def ds_read_b64(self, addr: KVReg, offset: int = 0, comment: str = None) -> KRegRange:
        return self.builder.ds_read_b64(addr, offset, comment)
    
    def ds_write_b64(self, addr: KVReg, src: KRegRange, offset: int = 0, comment: str = None):
        self.builder.ds_write_b64(addr, src, offset, comment)
    
    def s_waitcnt(self, vmcnt: int = 0, lgkmcnt: int = 0, comment: str = None):
        self.builder.s_waitcnt(vmcnt, lgkmcnt, comment)
    
    def s_barrier(self, comment: str = None):
        self.builder.s_barrier(comment)
    
    def s_endpgm(self, comment: str = None):
        self.builder.s_endpgm(comment)
    
    def comment(self, text: str):
        self.builder.comment(text)
    
    def emit(self, instr: KInstr):
        """Emit a raw instruction."""
        self.program.emit(instr)
    
    # =========================================================================
    # Finalization
    # =========================================================================
    
    def finalize(self) -> Tuple[List[str], AllocationStats]:
        """
        Finalize the kernel program and generate assembly.
        
        This:
        1. Computes liveness for all virtual registers
        2. Runs linear scan allocation
        3. Renders to assembly
        
        Returns:
            Tuple of (assembly lines, allocation statistics)
        """
        # Get reserved registers from ABI
        reserved_vgprs = self.program.abi.get_reserved_vgprs()
        reserved_sgprs = self.program.abi.get_reserved_sgprs()
        
        # Allocate
        mapping, stats = allocate_kernel(
            self.program,
            reserved_vgprs=reserved_vgprs,
            reserved_sgprs=reserved_sgprs,
        )
        
        # Render
        generator = KernelGenerator(self.program, mapping)
        asm_lines = generator.generate()
        
        return asm_lines, stats
    
    def finalize_to_string(self) -> str:
        """Finalize and return assembly as a single string."""
        lines, _ = self.finalize()
        return "\n".join(lines)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    @property
    def num_instructions(self) -> int:
        return len(self.program)
    
    @property
    def num_virtual_vregs(self) -> int:
        return self.program._next_vreg_id
    
    @property
    def num_virtual_sregs(self) -> int:
        return self.program._next_sreg_id
    
    @property
    def cse_hit_count(self) -> int:
        return self._cse_hits


# =============================================================================
# Integration helpers
# =============================================================================

def create_kernel_context(
    kernel_info: "KernelInfo",
    max_vgprs: int = 256,
    max_sgprs: int = 104,
) -> KernelCompilationContext:
    """
    Create a kernel compilation context configured for the given kernel.
    
    Args:
        kernel_info: KernelInfo describing the kernel
        max_vgprs: Maximum VGPRs available
        max_sgprs: Maximum SGPRs available
        
    Returns:
        Configured KernelCompilationContext
    """
    # Determine ABI configuration from kernel_info
    wg_size = getattr(kernel_info, 'workgroup_size', (256, 1, 1))
    num_waves = max(1, wg_size[0] * wg_size[1] * wg_size[2] // 64)
    
    ctx = KernelCompilationContext(
        max_vgprs=max_vgprs,
        max_sgprs=max_sgprs,
        use_flat_tid=(num_waves > 1),
        use_workgroup_ids=(True, True, True),
    )
    
    return ctx

