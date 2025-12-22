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
        # Emit instructions via dynamic dispatch
        v1 = ctx.v_mov_b32(42)
        v2 = ctx.v_add_u32(v1, 100)
        # Finalize and get assembly
        asm = ctx.finalize()
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .kernel_model import KernelInfo

from .kernel_ir import (
    KernelProgram, KernelBuilder, KInstr,
    KVReg, KSReg, KPhysVReg, KPhysSReg, KSpecialReg,
    KReg, KRegRange, KImm, KMemOffset,
    KernelABI, RegClass, M0,
)
from .kernel_liveness import compute_liveness, LivenessInfo
from .kernel_regalloc import KernelRegAlloc, allocate_kernel, AllocationStats, AllocationError
from .kernel_generator import KernelGenerator, PhysicalMapping, generate_program
from .unified_emitter import UnifiedEmitter, EmissionMode
from .instruction_registry import (
    InstructionRegistry,
    InstructionDef,
    OperandType,
    InstructionCategory,
    get_registry,
)


# Environment variable to enable kernel-level LSRA (advanced allocation)
# Default is "0" (disabled) during development
WAVE_KERNEL_LSRA_ENV = "WAVE_KERNEL_LSRA"

# Environment variable to use kernel IR compilation path
# Default is "0" (disabled) - requires full handler migration to work correctly
WAVE_USE_KERNEL_IR_ENV = "WAVE_USE_KERNEL_IR"


def use_kernel_ir_path() -> bool:
    """Check if kernel IR compilation path should be used.
    
    Kernel IR mode is currently disabled by default because handlers still
    allocate physical registers from AsmEmitter while emitting virtual 
    instructions to KernelCompilationContext. These need to be coordinated.
    
    Set WAVE_USE_KERNEL_IR=1 to enable for testing.
    """
    return os.environ.get(WAVE_USE_KERNEL_IR_ENV, "0") == "1"


def use_kernel_lsra() -> bool:
    """Check if kernel-level LSRA is enabled."""
    return os.environ.get(WAVE_KERNEL_LSRA_ENV, "0") == "1"


# =============================================================================
# Operand type to register allocation info
# =============================================================================

def _get_def_info(operand_types: Tuple[OperandType, ...]) -> Tuple[str, int, int]:
    """
    Get destination register info from operand types.
    
    Returns: (class, count, alignment) where:
        - class: 'v' for VGPR, 's' for SGPR, None for no destination
        - count: number of registers (1, 2, 4, 16)
        - alignment: alignment requirement
    """
    for ot in operand_types:
        if ot == OperandType.VGPR:
            return ('v', 1, 1)
        elif ot == OperandType.VGPR_PAIR:
            return ('v', 2, 2)
        elif ot == OperandType.VGPR_QUAD:
            return ('v', 4, 4)
        elif ot == OperandType.VGPR_16:
            return ('v', 16, 4)
        elif ot == OperandType.SGPR:
            return ('s', 1, 1)
        elif ot == OperandType.SGPR_PAIR:
            return ('s', 2, 2)
        elif ot == OperandType.SGPR_QUAD:
            return ('s', 4, 4)
    return (None, 0, 1)


@dataclass
class KernelCompilationContext:
    """
    Context for kernel compilation with whole-program register allocation.
    
    This context manages:
    - The KernelProgram being built
    - Symbol bindings (MLIR SSA values to virtual registers)
    - ABI register configuration
    - CSE cache for expression deduplication
    
    Instructions are emitted via dynamic dispatch:
        ctx.v_add_u32(src0, src1)  # Calls _emit_instruction("v_add_u32", ...)
    
    Usage:
        ctx = KernelCompilationContext(max_vgprs=256, max_sgprs=104)
        
        # Emit instructions - methods generated dynamically
        v1 = ctx.v_mov_b32(42)
        v2 = ctx.v_add_u32(v1, 100)
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
    _registry: InstructionRegistry = field(init=False)
    
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
        
        # Load instruction registry
        self._registry = get_registry("common")
        
        # Create unified emitter in KERNEL_IR mode
        # This allows callers to use kernel_ctx.unified.v_add_u32(...) syntax
        self._unified = UnifiedEmitter(
            architecture="common",
            mode=EmissionMode.KERNEL_IR,
            context=self,
        )
    
    # =========================================================================
    # Dynamic instruction dispatch
    # =========================================================================
    
    def __getattr__(self, name: str) -> Any:
        """
        Dynamic dispatch for instruction methods.
        
        When ctx.v_add_u32(...) is called and v_add_u32 isn't explicitly defined,
        this method handles it by:
        1. Looking up the instruction in the registry
        2. Allocating destination registers based on operand types
        3. Emitting a KInstr with the instruction name
        """
        # Avoid recursion on internal attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Check if it's an instruction in the registry
        instr_def = self._registry.get(name)
        if instr_def is not None:
            # Create and return an emission method
            def emit_method(*args, comment: str = None, **kwargs):
                return self._emit_instruction(name, instr_def, args, kwargs, comment)
            return emit_method
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def _emit_instruction(
        self,
        name: str,
        instr_def: InstructionDef,
        args: tuple,
        kwargs: dict,
        comment: str,
    ) -> Optional[Any]:
        """
        Emit an instruction with automatic register allocation.
        
        This method:
        1. Allocates destination registers based on operand types
        2. Emits a KInstr to the program with the instruction name
        3. Returns the destination register(s) if any
        """
        # Determine destination type and allocate
        dst = None
        defs = ()
        
        if instr_def.defs:
            # Get the first def's type info
            def_op = instr_def.defs[0]
            reg_class, count, alignment = _get_def_info(def_op.types)
            
            if reg_class == 'v':
                if count == 1:
                    dst = self.vreg()
                else:
                    dst = self.program.alloc_vreg_range(count, alignment=alignment)
                defs = (dst,)
            elif reg_class == 's':
                if count == 1:
                    dst = self.sreg()
                else:
                    dst = self.program.alloc_sreg_range(count, alignment=alignment)
                defs = (dst,)
        
        # Build uses from args
        uses = []
        for arg in args:
            if isinstance(arg, int):
                # Let kernel_generator handle raw ints
                uses.append(arg)
            else:
                uses.append(arg)
        
        # Add kwargs (offset handling)
        if 'offset' in kwargs and kwargs['offset']:
            uses.append(KMemOffset(kwargs['offset']))
        
        # Emit the instruction using name string
        self.program.emit(KInstr(name, defs, tuple(uses), comment=comment))
        
        return dst
    
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
    # Register allocation helpers
    # =========================================================================
    
    def vreg(self) -> KVReg:
        """Allocate a new virtual VGPR."""
        return self.builder.vreg()
    
    def sreg(self) -> KSReg:
        """Allocate a new virtual SGPR."""
        return self.builder.sreg()
    
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
    
    # =========================================================================
    # Special emission methods (not auto-generated)
    # =========================================================================
    
    def emit(self, instr: KInstr):
        """Emit a raw instruction."""
        self.program.emit(instr)
    
    def emit_raw(self, asm_line: str):
        """Emit a raw assembly line (escape hatch)."""
        self.program.emit(KInstr("_raw_asm", (), (), comment=asm_line))
    
    def emit_label(self, label: str):
        """Emit a label."""
        self.program.emit(KInstr("_label", (), (), comment=label))
    
    def comment(self, text: str):
        """Emit a comment."""
        self.builder.comment(text)
    
    def s_mov_b32_to_m0(self, src, comment: str = None):
        """Emit s_mov_b32 m0, src (special: destination is M0)."""
        self.program.emit(KInstr("s_mov_b32", (M0,), (src,), comment=comment))
    
    def s_cbranch_scc1(self, label: str, comment: str = None):
        """Emit s_cbranch_scc1 (label stored in comment)."""
        self.program.emit(KInstr("s_cbranch_scc1", (), (), comment=label))
    
    def s_branch(self, label: str, comment: str = None):
        """Emit s_branch (label stored in comment)."""
        self.program.emit(KInstr("s_branch", (), (), comment=label))
    
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
