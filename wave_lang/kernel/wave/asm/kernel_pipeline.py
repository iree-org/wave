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
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .kernel_model import KernelInfo

from .kernel_ir import (
    KernelProgram, KernelBuilder, KInstr, KOpcode,
    KVReg, KSReg, KPhysVReg, KPhysSReg, KSpecialReg,
    KReg, KRegRange, KImm, KMemOffset,
    KernelABI, RegClass, M0,
)
from .kernel_liveness import compute_liveness, LivenessInfo
from .kernel_regalloc import KernelRegAlloc, allocate_kernel, AllocationStats, AllocationError
from .kernel_generator import KernelGenerator, PhysicalMapping, generate_program
from .unified_emitter import UnifiedEmitter, EmissionMode


# Environment variable to enable kernel-level LSRA
# Default is "0" (disabled) during development
WAVE_KERNEL_LSRA_ENV = "WAVE_KERNEL_LSRA"

# Environment variable to use legacy streaming emission (bypass kernel IR)
# Default is "1" (use legacy) during development; will flip to "0" when stable
WAVE_USE_LEGACY_STREAMING_ENV = "WAVE_USE_LEGACY_STREAMING"


def use_kernel_ir_path() -> bool:
    """Check if kernel IR compilation path should be used."""
    # Legacy streaming is the default for now
    return os.environ.get(WAVE_USE_LEGACY_STREAMING_ENV, "1") == "0"


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
    _unified: UnifiedEmitter = field(init=False)
    
    # Symbol bindings: MLIR SSA value string -> virtual register
    _symbol_bindings: Dict[str, KReg] = field(default_factory=dict, init=False)
    
    # SSA to register mapping for MLIR SSA values (like walker.ssa_to_vgpr)
    # Maps SSA value string to tuple of virtual registers (single or range)
    ssa_to_reg: Dict[str, Tuple[KVReg, ...]] = field(default_factory=dict, init=False)
    
    # CSE cache: expression key -> virtual register
    _cse_cache: Dict[tuple, KVReg] = field(default_factory=dict, init=False)
    
    # SRD tracking (like emitter.srds)
    srds: Dict[str, Tuple[int, ...]] = field(default_factory=dict, init=False)
    
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
        
        # Create unified emitter in KERNEL_IR mode
        # This allows callers to use kernel_ctx.unified.v_add_u32(...) syntax
        self._unified = UnifiedEmitter(
            architecture="common",
            mode=EmissionMode.KERNEL_IR,
            context=self,
        )
    
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
    
    def emit_raw(self, asm_line: str):
        """Emit a raw assembly line (escape hatch)."""
        self.program.emit(KInstr(KOpcode.RAW_ASM, (), (), comment=asm_line))
    
    def emit_label(self, label: str):
        """Emit a label."""
        self.program.emit(KInstr(KOpcode.LABEL, (), (), comment=label))
    
    @property
    def unified(self) -> UnifiedEmitter:
        """
        Get the unified emitter for this context.
        
        This provides a consistent API with AsmEmitter.unified, allowing
        callers to use kernel_ctx.unified.v_add_u32(...) syntax.
        
        When using the unified emitter:
        - Methods that exist on KernelCompilationContext are called directly
        - Methods that don't exist fall back to emit_raw()
        - Virtual registers are returned for instructions with destinations
        
        Example:
            result = kernel_ctx.unified.v_add_u32(src0, src1, comment="add")
        """
        return self._unified

    # =========================================================================
    # Additional instruction emission
    # =========================================================================
    
    def vreg_pair(self) -> KRegRange:
        """Allocate a pair of virtual VGPRs."""
        return self.program.alloc_vreg_range(2, alignment=2)
    
    def vreg_quad(self) -> KRegRange:
        """Allocate a quad of virtual VGPRs."""
        return self.program.alloc_vreg_range(4, alignment=4)
    
    def sreg_pair(self) -> KRegRange:
        """Allocate a pair of virtual SGPRs."""
        return self.program.alloc_sreg_range(2, alignment=2)
    
    def sreg_quad(self) -> KRegRange:
        """Allocate a quad of virtual SGPRs."""
        return self.program.alloc_sreg_range(4, alignment=4)
    
    def ds_write_b32(self, addr: KVReg, src: KVReg, offset: int = 0, comment: str = None):
        """Emit ds_write_b32."""
        uses = (addr, src) if offset == 0 else (addr, src, KMemOffset(offset))
        self.program.emit(KInstr(KOpcode.DS_WRITE_B32, (), uses, comment=comment))
    
    def ds_write_b128(self, addr: KVReg, src: KRegRange, offset: int = 0, comment: str = None):
        """Emit ds_write_b128."""
        uses = (addr, src) if offset == 0 else (addr, src, KMemOffset(offset))
        self.program.emit(KInstr(KOpcode.DS_WRITE_B128, (), uses, comment=comment))
    
    def ds_read_b128(self, addr: KVReg, offset: int = 0, comment: str = None) -> KRegRange:
        """Emit ds_read_b128 and return destination vreg quad."""
        dst = self.vreg_quad()
        uses = (addr,) if offset == 0 else (addr, KMemOffset(offset))
        self.program.emit(KInstr(KOpcode.DS_READ_B128, (dst,), uses, comment=comment))
        return dst
    
    def v_readfirstlane_b32(self, src: KVReg, comment: str = None) -> KSReg:
        """Emit v_readfirstlane_b32 and return destination sreg."""
        dst = self.sreg()
        self.program.emit(KInstr(KOpcode.V_READFIRSTLANE_B32, (dst,), (src,), comment=comment))
        return dst
    
    def s_mov_b32_to_m0(self, src, comment: str = None):
        """Emit s_mov_b32 m0, src."""
        self.program.emit(KInstr(KOpcode.S_MOV_B32, (M0,), (src,), comment=comment))
    
    def v_mfma_f32_16x16x16_f16(
        self, 
        a_src: KRegRange, 
        b_src: KRegRange, 
        c_acc: KRegRange, 
        comment: str = None
    ) -> KRegRange:
        """Emit v_mfma_f32_16x16x16_f16 and return destination quad."""
        dst = self.vreg_quad()
        self.program.emit(KInstr(
            KOpcode.V_MFMA_F32_16X16X16_F16, 
            (dst,), 
            (a_src, b_src, c_acc), 
            comment=comment
        ))
        return dst
    
    def buffer_load_dwordx4(
        self,
        vaddr: KVReg,
        srd: KRegRange,
        soffset,
        offset: int = 0,
        comment: str = None
    ) -> KRegRange:
        """Emit buffer_load_dwordx4."""
        dst = self.vreg_quad()
        uses = (vaddr, srd, soffset, KMemOffset(offset))
        self.program.emit(KInstr(KOpcode.BUFFER_LOAD_DWORDX4, (dst,), uses, comment=comment))
        return dst
    
    def buffer_store_dwordx4(
        self,
        src: KRegRange,
        vaddr: KVReg,
        srd: KRegRange,
        soffset,
        offset: int = 0,
        comment: str = None
    ):
        """Emit buffer_store_dwordx4."""
        uses = (src, vaddr, srd, soffset, KMemOffset(offset))
        self.program.emit(KInstr(KOpcode.BUFFER_STORE_DWORDX4, (), uses, comment=comment))
    
    def s_load_dwordx2(
        self,
        base: KRegRange,
        offset: int,
        comment: str = None
    ) -> KRegRange:
        """Emit s_load_dwordx2."""
        dst = self.sreg_pair()
        self.program.emit(KInstr(KOpcode.S_LOAD_DWORDX2, (dst,), (base, KImm(offset)), comment=comment))
        return dst
    
    def s_cmp_lt_u32(self, src0, src1, comment: str = None):
        """Emit s_cmp_lt_u32."""
        self.program.emit(KInstr(KOpcode.S_CMP_LT_U32, (), (src0, src1), comment=comment))
    
    def s_cbranch_scc1(self, label: str, comment: str = None):
        """Emit s_cbranch_scc1."""
        # Label is stored as comment for the instruction
        self.program.emit(KInstr(KOpcode.S_CBRANCH_SCC1, (), (), comment=f"{label}"))
    
    def s_branch(self, label: str, comment: str = None):
        """Emit s_branch."""
        self.program.emit(KInstr(KOpcode.S_BRANCH, (), (), comment=f"{label}"))
    
    def s_add_u32(self, src1, src2, comment: str = None) -> KSReg:
        """Emit s_add_u32 and return destination sreg."""
        dst = self.sreg()
        self.program.emit(KInstr(KOpcode.S_ADD_U32, (dst,), (src1, src2), comment=comment))
        return dst
    
    def v_mbcnt_lo_u32_b32(self, src0, src1, comment: str = None) -> KVReg:
        """Emit v_mbcnt_lo_u32_b32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_MBCNT_LO_U32_B32, (dst,), (src0, src1), comment=comment))
        return dst
    
    def v_mbcnt_hi_u32_b32(self, src0, src1, comment: str = None) -> KVReg:
        """Emit v_mbcnt_hi_u32_b32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_MBCNT_HI_U32_B32, (dst,), (src0, src1), comment=comment))
        return dst
    
    def buffer_load_dword_lds(
        self,
        vaddr: KVReg,
        srd: KRegRange,
        soffset,
        offset: int = 0,
        comment: str = None
    ):
        """Emit buffer_load_dword with LDS modifier (no destination, writes to LDS via m0)."""
        uses = (vaddr, srd, soffset, KMemOffset(offset))
        # Note: This instruction has no destination (writes to LDS)
        self.program.emit(KInstr(KOpcode.BUFFER_LOAD_DWORD_LDS, (), uses, comment=comment))
    
    def buffer_load_dwordx4_lds(
        self,
        vaddr: KVReg,
        srd: KRegRange,
        soffset,
        offset: int = 0,
        comment: str = None
    ):
        """Emit buffer_load_dwordx4 with LDS modifier (no destination, writes to LDS via m0)."""
        uses = (vaddr, srd, soffset, KMemOffset(offset))
        # Note: This instruction has no destination (writes to LDS)
        self.program.emit(KInstr(KOpcode.BUFFER_LOAD_DWORDX4_LDS, (), uses, comment=comment))
    
    def ds_read_b32(self, addr: KVReg, offset: int = 0, comment: str = None) -> KVReg:
        """Emit ds_read_b32 and return destination vreg."""
        dst = self.vreg()
        uses = (addr,) if offset == 0 else (addr, KMemOffset(offset))
        self.program.emit(KInstr(KOpcode.DS_READ_B32, (dst,), uses, comment=comment))
        return dst
    
    def v_sub_u32(self, src1, src2, comment: str = None) -> KVReg:
        """Emit v_sub_u32 and return destination vreg."""
        dst = self.vreg()
        self.program.emit(KInstr(KOpcode.V_SUB_U32, (dst,), (src1, src2), comment=comment))
        return dst
    
    def s_and_b32(self, src1, src2, comment: str = None) -> KSReg:
        """Emit s_and_b32 and return destination sreg."""
        dst = self.sreg()
        self.program.emit(KInstr(KOpcode.S_AND_B32, (dst,), (src1, src2), comment=comment))
        return dst
    
    def s_or_b32(self, src1, src2, comment: str = None) -> KSReg:
        """Emit s_or_b32 and return destination sreg."""
        dst = self.sreg()
        self.program.emit(KInstr(KOpcode.S_OR_B32, (dst,), (src1, src2), comment=comment))
        return dst
    
    def s_movk_i32(self, imm16: int, comment: str = None) -> KSReg:
        """Emit s_movk_i32 and return destination sreg."""
        dst = self.sreg()
        self.program.emit(KInstr(KOpcode.S_MOVK_I32, (dst,), (KImm(imm16),), comment=comment))
        return dst
    
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

