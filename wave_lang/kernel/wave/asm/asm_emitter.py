# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Assembly emitter for generating AMDGCN assembly instructions.
"""

from typing import Dict, List, Tuple, Union, Optional, Set
from dataclasses import dataclass, field

from wave_lang.support.ir_imports import (
    func_d,
    Context,
    Module,
)

from .kernel_model import KernelInfo, MemRefInfo
from .utils import normalize_wg_size
from .register_allocator import RegFile, SGPRAllocator, VGPRAllocator, AGPRAllocator
from .mlir_walker import IRWalker
# Instructions removed - using unified emitter directly
from .instruction_categories import InstructionCategory, categorize_instruction

# Import latency-aware scheduling infrastructure
from .latency_provider import LatencyProvider
from .scoreboard import Scoreboard
from .ticketing import Ticketing
from .hazards import HazardDetector
from .unified_emitter import UnifiedEmitter, EmissionMode


def get_register_granularity(target: str) -> Tuple[int, int]:
    """
    Get VGPR and SGPR allocation granularities for a target architecture.

    Args:
        target: Target GPU architecture (e.g., "gfx942", "gfx90a", "gfx1100")

    Returns:
        Tuple of (vgpr_granularity, sgpr_granularity)

    Architecture-specific granularities:
        - CDNA2/CDNA3 (gfx90a, gfx940, gfx941, gfx942): VGPRs in blocks of 4, SGPRs in blocks of 8
    """
    return (4, 8)


class TicketingEmitterWrapper:
    """
    Wrapper around UnifiedEmitter that handles ticketing for memory operations.
    
    Intercepts all method calls and issues VMEM/LGKM tickets for memory operations
    before delegating to the underlying emitter.
    """
    
    def __init__(self, base_emitter: UnifiedEmitter, asm_emitter: "AsmEmitter"):
        self._base = base_emitter
        self._asm_emitter = asm_emitter
        self._registry = base_emitter._registry
    
    def __getattr__(self, name: str):
        """Intercept method calls to add ticketing."""
        attr = getattr(self._base, name)
        
        if callable(attr):
            # Look up instruction to check category
            instr_def = self._registry.get(name)
            if instr_def:
                mnemonic = instr_def.mnemonic
                
                def wrapper(*args, **kwargs):
                    # Issue tickets for memory operations
                    category = categorize_instruction(mnemonic)
                    if category == InstructionCategory.VMEM:
                        self._asm_emitter.ticketing.next_vmem_ticket()
                    elif category == InstructionCategory.LGKM:
                        self._asm_emitter.ticketing.next_lgkm_ticket()
                    
                    # Call the actual method
                    result = attr(*args, **kwargs)
                    
                    # Hazard mitigation
                    mitigation = self._asm_emitter.hazard_detector.get_mitigation(mnemonic)
                    if mitigation:
                        self._asm_emitter.lines.append(str(mitigation))
                    
                    return result
                
                return wrapper
        
        return attr


@dataclass
class SpecialRegs:
    """
    ABI/system register contract.
    
    Centralizes all ABI-mandated and system register assignments.
    These are the ONLY registers that may be referenced by hardcoded indices.
    All other register usage must go through the allocators.
    
    ABI-mandated SGPR layout (when enabled):
        - s[0:1]: kernarg_segment_ptr (always present)
        - s2, s3, s4: workgroup_id_{x,y,z} (in order, only if requested)
    
    ABI-mandated VGPR layout (system_vgpr_workitem_id):
        - When == 1: v0 = flat workitem ID (packed tid_x, tid_y, tid_z)
        - When == 0: No system VGPRs
    
    Flat workitem ID encoding (AMD hardware fixed):
        - Bits 0-9: tid_x (0-1023)
        - Bits 10-19: tid_y (0-1023)
        - Bits 20-29: tid_z (0-1023)
    """
    # SGPR assignments (None if not requested/available)
    kernarg_ptr_lo: int = 0  # s0 - always present
    kernarg_ptr_hi: int = 1  # s1 - always present
    workgroup_id_x: Optional[int] = None  # s2 if requested
    workgroup_id_y: Optional[int] = None  # s3 if requested (and x requested)
    workgroup_id_z: Optional[int] = None  # s4 if requested (and x,y requested)
    
    # VGPR assignments (None if not requested/available)
    flat_tid_vgpr: Optional[int] = None  # v0 when system_vgpr_workitem_id >= 1
    
    # Configuration
    system_vgpr_workitem_id: int = 0  # 0, 1, 2, or 3
    
    def get_flat_tid_vgpr(self) -> int:
        """Get the VGPR containing flat thread ID, or raise if not available."""
        if self.flat_tid_vgpr is None:
            raise ValueError(
                "Flat thread ID VGPR not available. "
                "This kernel may be single-wave or system_vgpr_workitem_id=0."
            )
        return self.flat_tid_vgpr
    
    def has_flat_tid(self) -> bool:
        """Check if flat thread ID VGPR is available."""
        return self.flat_tid_vgpr is not None
    
    def get_workgroup_id_sgpr(self, dim: str) -> int:
        """Get SGPR containing workgroup ID for given dimension."""
        if dim == "x":
            if self.workgroup_id_x is None:
                raise ValueError("workgroup_id_x not available")
            return self.workgroup_id_x
        elif dim == "y":
            if self.workgroup_id_y is None:
                raise ValueError("workgroup_id_y not available")
            return self.workgroup_id_y
        elif dim == "z":
            if self.workgroup_id_z is None:
                raise ValueError("workgroup_id_z not available")
            return self.workgroup_id_z
        else:
            raise ValueError(f"Invalid dimension: {dim}")
    
    def first_user_sgpr(self) -> int:
        """Get the first SGPR index available for user allocation."""
        # SGPRs 0,1 are kernarg ptr, then workgroup IDs
        next_sgpr = 2
        if self.workgroup_id_x is not None:
            next_sgpr = max(next_sgpr, self.workgroup_id_x + 1)
        if self.workgroup_id_y is not None:
            next_sgpr = max(next_sgpr, self.workgroup_id_y + 1)
        if self.workgroup_id_z is not None:
            next_sgpr = max(next_sgpr, self.workgroup_id_z + 1)
        return next_sgpr
    
    def first_user_vgpr(self) -> int:
        """Get the first VGPR index available for user allocation."""
        # If we have flat_tid, it's in v0, so user starts at v1
        # But with system_vgpr_workitem_id=1, only v0 is reserved
        return self.system_vgpr_workitem_id  # 0 or 1


class AsmEmitter:
    """
    AMDGCN Assembly Emitter for Wave-based kernels.

    Generates optimized assembly code for AMD CDNA architectures (gfx90a, gfx940, gfx942)
    with support for:
    - Single-wave and multi-wave workgroups
    - Single-workgroup and multi-workgroup dispatches
    - MFMA (Matrix Fused Multiply-Add) instructions
    - LDS (Local Data Share) operations
    - Latency-aware instruction scheduling with waitcnt optimization

    ## Multi-Wave and Multi-Workgroup Support

    The emitter supports flexible workgroup configurations:

    **Thread ID Handling**:
    - Single-wave (wg_size = [64, 1, 1]): No system VGPRs requested (tid_x uses lane_id)
    - Multi-wave (wg_size = [N, M, 1]): Requests `.amdhsa_system_vgpr_workitem_id 1`
      - v0 contains flat thread ID with encoding: bits[0:9]=tid_x, bits[10:19]=tid_y
      - tid_x extracted via `v_and_b32 v_temp, 0x3ff, v0`
      - tid_y extracted via `v_bfe_u32 v_temp, v0, 10, 10`

    **Workgroup ID Handling**:
    - Dynamically detects `gpu.block_id` operations in MLIR
    - Requests only needed system SGPRs: `.amdhsa_system_sgpr_workgroup_id_{x,y,z}`
    - SGPRs allocated at s2, s3, s4 (after kernarg pointer at s0:s1)
    - Workgroup IDs used in affine address expressions for global memory indexing

    **MFMA Instructions**:
    - Uses VGPR-variant MFMA (writes directly to VGPRs, not accumulators)
    - Example: `v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]`

    ## Thread-to-Dimension Mapping

    Row-major layout (AMD standard):
    - tid_x = fastest-varying local thread ID (within wave, lane 0-63)
    - tid_y = slower-varying (across waves in Y dimension)
    - tid_z = slowest-varying (rarely used, reserved for future 3D workgroups)

    ## Address Offset Limits

    Buffer instructions (`buffer_load_dword`, `buffer_store_dword`) have a 16-bit
    unsigned immediate offset limit (0-65535 bytes).

    **Current Limitations**:
    - Large memory footprints can exceed this offset range
    - Affects very large workgroup counts or memory-intensive kernels
    - Example: 32x32 workgroup grid with large tile sizes may exceed 65535 bytes

    **Future Work**: Implement offset splitting by adjusting SRD base pointer
    dynamically using `s_add_u32`/`s_addc_u32` for offsets >= 65536.

    ## Latency-Aware Scheduling

    The emitter uses a ticketing system for optimal `s_waitcnt` placement:
    - VMEM (vector memory) instructions tracked for `vmcnt` waits
    - LGKM (LDS/GDS/constant/message) instructions tracked for `lgkmcnt` waits
    - Minimizes stalls by placing waits as late as possible before data dependencies
    """

    SRD127_96 = "0x20000"  # data_format=4 for gfx9xx

    def __init__(
        self,
        targetid: str,
        codeobj: str,
    ):
        self.targetid = targetid
        self.codeobj = codeobj
        self.lines: List[str] = []
        self.register_file = RegFile()
        self.sgpr_allocator = SGPRAllocator(self.register_file)
        self.vgpr_allocator = VGPRAllocator(self.register_file)
        self.agpr_allocator = AGPRAllocator(self.register_file)
        self.ptr_pairs: Dict[int, Tuple[int, int]] = {}  # arg_index -> (s_lo, s_hi)
        self.srds: Dict[str, Tuple[int, int, int, int]] = {}  # memref_ssa -> srd quad
        self.lane_id_emitted = False
        self.lane_id_v = None  # Store which VGPR holds lane ID
        self.current_vgpr_quad = None  # Track current VGPR quad for MFMA results
        
        # Unified emitter bridge - provides unified API while emitting to self.lines
        self._unified_emitter: Optional[UnifiedEmitter] = None
        
        # Track pinned VGPRs for future lifetime management (API surface)
        self._pinned_vgprs = set()

        # Centralized ABI/system register contract
        # This is the single source of truth for all ABI-mandated registers
        self.special_regs = SpecialRegs()
        
        # Workgroup ID tracking (for multi-workgroup support)
        # These are assigned sequentially after kernarg ptr based on what we request
        # NOTE: These are aliased via special_regs for backwards compatibility
        self.sgpr_workgroup_id_x = None
        self.sgpr_workgroup_id_y = None
        self.sgpr_workgroup_id_z = None

        # Thread/workitem ID tracking (for multi-wave support)
        # System VGPRs are allocated by hardware at v0, v1, v2 for tid_x, tid_y, tid_z
        # NOTE: These are aliased via special_regs for backwards compatibility
        self.vgpr_tid_x = None  # Will be v0 when system_vgpr_workitem_id >= 1
        self.vgpr_tid_y = None  # Will be v1 when system_vgpr_workitem_id >= 2
        self.vgpr_tid_z = None  # Will be v2 when system_vgpr_workitem_id == 3

        # Ticket-based VMEM/LGKM tracking for optimal wait placement
        self.ticketing = Ticketing()

        # MFMA tracking (matrix operations have fixed latency)
        self._mfma_last_cycle = None  # cycle when last MFMA was issued

        # Latency-aware scheduling
        self.latency_provider = LatencyProvider(arch=targetid)
        self.scoreboard = Scoreboard(latency_provider=self.latency_provider)

        # Track which workgroup IDs are needed (detected from MLIR)
        self.needs_wgid_x = False
        self.needs_wgid_y = False
        self.needs_wgid_z = False

        # Hazard detector for gfx950 VALU hazards
        # See hazards.py for details on the hazard types and mitigations
        self.hazard_detector = HazardDetector()

    @staticmethod
    def _detect_needed_workgroup_ids(fn) -> tuple[bool, bool, bool]:
        """
        Scan MLIR function to detect which workgroup IDs are needed.

        Returns:
            (needs_wgid_x, needs_wgid_y, needs_wgid_z) tuple
        """
        from wave_lang.support.ir_imports import gpu_d

        needs_x, needs_y, needs_z = False, False, False

        # Recursively walk all operations
        def walk_ops(op):
            nonlocal needs_x, needs_y, needs_z

            # Check if this is a gpu.block_id operation
            if isinstance(op, gpu_d.BlockIdOp):
                # Extract dimension from the operation
                # dimension is an Attribute, convert to string for comparison
                dim_str = str(op.dimension)
                if "dim x" in dim_str:
                    needs_x = True
                elif "dim y" in dim_str:
                    needs_y = True
                elif "dim z" in dim_str:
                    needs_z = True

            # Recurse into regions
            if hasattr(op, "regions"):
                for region in op.regions:
                    for block in region.blocks:
                        for inner in block.operations:
                            walk_ops(inner)

        walk_ops(fn)
        return needs_x, needs_y, needs_z

    @classmethod
    def from_mlir_string(
        cls, mlir_string: str, targetid: str = "gfx942", codeobj: str = "5"
    ) -> str:
        """
        Process MLIR string and return AMDGCN assembly.

        Args:
            mlir_string: MLIR code as string
            targetid: Target GPU (e.g., "gfx942")
            codeobj: Code object version ("4" or "5")

        Returns:
            AMDGCN assembly code as string
        """
        emitter = cls(targetid=targetid, codeobj=codeobj)

        with Context() as ctx:
            module = Module.parse(mlir_string, ctx)

            for fn in emitter._walk_ops_recursively(module.operation):
                if isinstance(fn, func_d.FuncOp):
                    # Skip async functions and other non-kernel functions
                    if fn.name.value.startswith(
                        "isolated_benchmark"
                    ) or fn.name.value.endswith("$async"):
                        continue

                    # Extract basic info directly from MLIR function
                    kernel_name = fn.sym_name.value
                    num_args = len(fn.entry_block.arguments)

                    # Extract workgroup size from function attributes
                    from .utils import parse_wg_and_subgroup
                    from wave_lang.support.ir_imports import OpAttributeMap

                    wg_size = None
                    function_attributes = (
                        dict(fn.attributes)
                        if isinstance(fn.attributes, OpAttributeMap)
                        else {}
                    )
                    translation_info = function_attributes.get("translation_info")
                    if translation_info is not None:
                        workgroup_size_tuple, _ = parse_wg_and_subgroup(
                            translation_info
                        )
                        if workgroup_size_tuple:
                            wg_size = workgroup_size_tuple

                    # Workgroup size is required for code generation
                    assert (
                        wg_size is not None
                    ), "translation_info with workgroup_size must be present in MLIR function attributes"

                    # Detect which workgroup IDs are needed
                    needs_wgid_x, needs_wgid_y, needs_wgid_z = (
                        cls._detect_needed_workgroup_ids(fn)
                    )
                    emitter.needs_wgid_x = needs_wgid_x
                    emitter.needs_wgid_y = needs_wgid_y
                    emitter.needs_wgid_z = needs_wgid_z

                    # Emit kernel preamble with workgroup size
                    emitter.emit_prologue(kernel_name, wg_size)
                    
                    # Walk MLIR and emit instructions via kernel IR
                    from .kernel_pipeline import KernelCompilationContext
                    
                    # Check if multi-wave (need flat_tid from system VGPR)
                    wg_x, wg_y, wg_z = normalize_wg_size(wg_size)
                    is_multi_wave = wg_y > 1 or wg_z > 1
                    kernel_ctx = KernelCompilationContext(
                        use_flat_tid=is_multi_wave,
                        use_workgroup_ids=(needs_wgid_x, needs_wgid_y, needs_wgid_z),
                    )
                    walker = IRWalker(emitter, kernel_ctx=kernel_ctx)
                    kernel_info = walker.interpret_func(fn)
                    body_lines, allocation_stats = kernel_ctx.finalize()
                    emitter.lines.extend(body_lines)
                    
                    # Update register file with kernel IR allocation info
                    # This ensures emit_epilogue uses correct register counts
                    if allocation_stats.peak_vgprs > 0:
                        emitter.register_file.v_used.add(allocation_stats.peak_vgprs - 1)
                    if allocation_stats.peak_sgprs > 0:
                        emitter.register_file.s_max = max(
                            emitter.register_file.s_max,
                            allocation_stats.peak_sgprs - 1
                        )

                    emitter.emit_epilogue(
                        kernel_info.name,
                        kernel_info.wg_size,
                        kernel_info.subgroup_size,
                        len(kernel_info.arg_ssa_order),
                        kernel_info.lds_size_bytes,
                    )

        return "\n".join(emitter.lines)

    def _walk_ops_recursively(self, op):
        """Helper method to walk operations recursively."""
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    yield inner
                    yield from self._walk_ops_recursively(inner)

    # ---- Unified Emitter Integration ----
    
    @property
    def unified(self) -> "TicketingEmitterWrapper":
        """
        Get the unified instruction emitter with ticketing support.
        
        Provides a consistent API for instruction emission that works with
        both the legacy line-based emission and kernel IR paths.
        
        Usage:
            emitter.unified.v_mov_b32("v0", 42, comment="load constant")
            emitter.unified.s_barrier()
        """
        if self._unified_emitter is None:
            # Create unified emitter in direct mode, sharing our line buffer
            base_emitter = UnifiedEmitter(
                architecture=self._get_architecture(),
                mode=EmissionMode.DIRECT,
            )
            # Share the lines buffer
            base_emitter._lines = self.lines
            # Wrap with ticketing support
            self._unified_emitter = TicketingEmitterWrapper(base_emitter, self)
        return self._unified_emitter
    
    def _get_architecture(self) -> str:
        """Map targetid to architecture for instruction registry."""
        if "gfx942" in self.targetid or "gfx940" in self.targetid:
            return "gfx942"
        elif "gfx950" in self.targetid:
            return "gfx950"
        return "common"

    # ---- low-level ----
    def emit(self, s: str):
        """Emit a line of assembly."""
        self.lines.append(s)

    def emit_instruction(self, instr):
        """
        Emit an instruction object directly.

        Automatically issues VMEM/LGKM tickets for memory operations based on
        instruction categorization. This ensures uniform, centralized ticketing
        for all emitted instructions.
        """
        from .instruction_categories import InstructionCategory, categorize_instruction

        mnemonic = getattr(instr, "mnemonic", None)

        # Issue tickets for memory operations based on instruction category
        if mnemonic:
            category = categorize_instruction(mnemonic)

            if category == InstructionCategory.VMEM:
                self.ticketing.next_vmem_ticket()
            elif category == InstructionCategory.LGKM:
                self.ticketing.next_lgkm_ticket()

        self.lines.append(str(instr))

        # Hazard mitigation (pass mnemonic directly, not full instruction string)
        if mnemonic:
            mitigation = self.hazard_detector.get_mitigation(mnemonic)
            if mitigation:
                self.lines.append(str(mitigation))

        # Debug mode: add barriers after every instruction for debugging race conditions
        # Set WAVE_DEBUG_BARRIERS=1 to enable
        import os

        if os.environ.get("WAVE_DEBUG_BARRIERS", "0") == "1":
            skip_mnemonics = {"s_barrier", "s_waitcnt", "s_endpgm", "s_sleep", "s_nop"}
            if mnemonic and mnemonic.lower() not in skip_mnemonics:
                self.lines.append("    s_waitcnt vmcnt(0)")
                self.lines.append("    s_waitcnt lgkmcnt(0)")
                self.lines.append("    s_barrier")

    def _setup_workgroup_id_sgprs(self):
        """
        Set up workgroup ID SGPRs using SYSTEM SGPR mechanism.

        AMD ABI SGPR Layout:
        - User SGPRs (allocated first):
          * s[0:1] = kernarg_segment_ptr (when .amdhsa_user_sgpr_kernarg_segment_ptr 1)
        - System SGPRs (come immediately after user SGPRs):
          * s2 = workgroup_id_x (when .amdhsa_system_sgpr_workgroup_id_x 1)
          * s3 = workgroup_id_y (when .amdhsa_system_sgpr_workgroup_id_y 1)
        - Free SGPRs for user allocation: s4+

        Returns:
            Number of workgroup ID system SGPRs requested (0-3)

        NOTE: Dynamically requests only the workgroup IDs that are actually used
        in the MLIR (detected by scanning for gpu.block_id operations).
        """
        # SGPR layout: kernarg ptr (s0:s1), then workgroup IDs as system SGPRs
        kernarg_ptr_sgprs = 2  # s[0:1]
        next_system_sgpr = kernarg_ptr_sgprs

        # Allocate system SGPRs based on what's actually needed
        if self.needs_wgid_x:
            self.sgpr_workgroup_id_x = next_system_sgpr
            self.special_regs.workgroup_id_x = next_system_sgpr
            next_system_sgpr += 1
        else:
            self.sgpr_workgroup_id_x = None
            self.special_regs.workgroup_id_x = None

        if self.needs_wgid_y:
            self.sgpr_workgroup_id_y = next_system_sgpr
            self.special_regs.workgroup_id_y = next_system_sgpr
            next_system_sgpr += 1
        else:
            self.sgpr_workgroup_id_y = None
            self.special_regs.workgroup_id_y = None

        if self.needs_wgid_z:
            self.sgpr_workgroup_id_z = next_system_sgpr
            self.special_regs.workgroup_id_z = next_system_sgpr
            next_system_sgpr += 1
        else:
            self.sgpr_workgroup_id_z = None
            self.special_regs.workgroup_id_z = None

        # Reserve all allocated workgroup ID SGPRs
        if next_system_sgpr > kernarg_ptr_sgprs:
            self.register_file.s_max = max(
                self.register_file.s_max, next_system_sgpr - 1
            )

        # Update SGPR allocator to start after system workgroup ID SGPRs
        # Round up to s4 for even alignment (required for dwordx2 loads)
        self.sgpr_allocator.next_sgpr = max(4, next_system_sgpr)

        # Return count of requested workgroup IDs
        return sum([self.needs_wgid_x, self.needs_wgid_y, self.needs_wgid_z])

    def _setup_workitem_id_vgprs(self, wg_size: tuple) -> int:
        """
        Set up workitem ID VGPRs using SYSTEM VGPR mechanism.

        AMD ABI VGPR Layout for system_vgpr_workitem_id:
        - When system_vgpr_workitem_id == 0: No system VGPRs allocated
        - When system_vgpr_workitem_id == 1: v0 = flat workitem_id

        For multi-wave kernels (wg_size_y > 1 or wg_size_z > 1):
        - v0 contains flat thread ID with fixed encoding:
          * Bits 0-9: thread_id_x
          * Bits 10-19: thread_id_y
          * Bits 20-29: thread_id_z
        - We'll extract tid_x/tid_y/tid_z from v0 on-demand in expression_emitter.py

        For single-wave kernels (wg_size_y == 1 and wg_size_z == 1):
        - Don't request system VGPRs (matches LLVM behavior)
        - MLIR doesn't have gpu.thread_id ops, uses lane-based indexing

        Returns:
            system_vgpr_workitem_id value (0 for single-wave, 1 for multi-wave)
        """
        # Store workgroup size for multi-wave tid extraction
        self.wg_size = wg_size

        # Check if multi-wave (need thread IDs from multiple dimensions)
        wg_size_x, wg_size_y, wg_size_z = normalize_wg_size(wg_size)
        is_multi_wave = wg_size_y > 1 or wg_size_z > 1

        if is_multi_wave:
            # Multi-wave: request system VGPR for flat thread ID
            requested_dims = 1
            self.vgpr_tid_x = 0  # v0 contains flat thread ID
            self.vgpr_tid_y = None  # Will be extracted on-demand
            self.vgpr_tid_z = None  # Will be extracted on-demand
            # Update special_regs contract
            self.special_regs.flat_tid_vgpr = 0
            self.special_regs.system_vgpr_workitem_id = requested_dims
            self._reserve_system_vgprs(requested_dims)
        else:
            # Single-wave: no system VGPRs (matches LLVM)
            requested_dims = 0
            self.vgpr_tid_x = None  # No system VGPR, use lane_id fallback
            self.vgpr_tid_y = None
            self.vgpr_tid_z = None
            # Update special_regs contract
            self.special_regs.flat_tid_vgpr = None
            self.special_regs.system_vgpr_workitem_id = 0
            # Don't reserve v0 - it's a regular VGPR

        return requested_dims

    def _reserve_system_vgprs(self, n: int):
        """
        Reserve v0..v(n-1) as system VGPRs and set allocator base.

        Args:
            n: Number of system VGPRs to reserve (0-3)
        """
        for i in range(n):
            self.vgpr_allocator.reserve(i)
        # User-allocated VGPRs start after system VGPRs
        self.vgpr_allocator.base = n
    
    def get_flat_tid_vgpr(self) -> int:
        """
        Get the VGPR containing flat thread ID.
        
        This is the single source of truth for accessing the system VGPR
        that contains the packed thread IDs.
        
        Returns:
            VGPR index (typically 0) containing flat thread ID
            
        Raises:
            ValueError: If not in multi-wave mode (no flat tid available)
        """
        return self.special_regs.get_flat_tid_vgpr()
    
    def has_flat_tid(self) -> bool:
        """Check if flat thread ID VGPR is available (multi-wave mode)."""
        return self.special_regs.has_flat_tid()

    # ---- high-level sections ----
    def emit_prologue(self, kernel_name: str, wg_size: tuple):
        """
        Emit kernel prologue with metadata directives.

        Args:
            kernel_name: Name of the kernel function
            wg_size: Workgroup size tuple (x, y, z) from MLIR attributes
        """
        self.emit(f'.amdgcn_target "amdgcn-amd-amdhsa--{self.targetid}"')
        self.emit(".text")
        self.emit(f".protected {kernel_name}")
        self.emit(f".globl {kernel_name}")
        self.emit(".p2align 8")
        self.emit(f".type {kernel_name},@function\n")
        self.emit(".section .rodata,#alloc")
        self.emit(".p2align 6")
        self.emit(f".amdhsa_kernel {kernel_name}")
        self.emit("  .amdhsa_user_sgpr_kernarg_segment_ptr 1")

        # Set up workgroup ID SGPRs - use system SGPRs (the assembler recognizes these)
        self._setup_workgroup_id_sgprs()

        # Emit user SGPR count - this tells the hardware where to place system SGPRs
        # With count=2 (just kernarg ptr), system SGPRs will be at s2, s3, s4...
        self.emit("  .amdhsa_user_sgpr_count 2")

        self.emit("  .amdhsa_accum_offset 0")  # patched later
        self.emit("  .amdhsa_next_free_vgpr 0")  # patched later
        self.emit("  .amdhsa_next_free_sgpr 0")  # patched later
        self.emit("  .amdhsa_group_segment_fixed_size 0")
        self.emit("  .amdhsa_private_segment_fixed_size 0")
        # Request workgroup IDs as system SGPRs (based on MLIR analysis)
        self.emit(
            f"  .amdhsa_system_sgpr_workgroup_id_x {1 if self.needs_wgid_x else 0}"
        )
        self.emit(
            f"  .amdhsa_system_sgpr_workgroup_id_y {1 if self.needs_wgid_y else 0}"
        )
        self.emit(
            f"  .amdhsa_system_sgpr_workgroup_id_z {1 if self.needs_wgid_z else 0}"
        )

        # Set up workitem ID VGPRs and emit directive based on workgroup size
        system_vgpr_workitem_id = self._setup_workitem_id_vgprs(wg_size)
        self.emit(f"  .amdhsa_system_vgpr_workitem_id {system_vgpr_workitem_id}")

        self.emit("  .amdhsa_float_denorm_mode_32 3")
        self.emit("  .amdhsa_float_denorm_mode_16_64 3")
        self.emit(".end_amdhsa_kernel")
        self.emit(".text\n")
        self.emit("# SRD upper word (gfx9xx): data_format=4 => 0x20000")
        self.emit(f".set Srd127_96, {self.SRD127_96}\n")
        self.emit(f"{kernel_name}:")

    def emit_epilogue(
        self,
        kernel_name: str,
        wg_size: tuple,
        subgroup_size: int,
        num_args: int,
        lds_size_bytes: int = 0,
    ):
        self.unified.s_endpgm()
        self.emit("")

        # Extract workgroup dimensions
        wg_size_x, wg_size_y, wg_size_z = normalize_wg_size(wg_size)

        # Compute actual register usage
        vgprs_used = (
            max(self.register_file.v_used) + 1 if self.register_file.v_used else 0
        )
        sgprs_used = self.register_file.s_max + 1

        # Get architecture-specific granularities
        vgpr_granularity, sgpr_granularity = get_register_granularity(self.targetid)

        # Round up to allocation granularity
        vgprs_used = (
            (vgprs_used + vgpr_granularity - 1) // vgpr_granularity
        ) * vgpr_granularity
        sgprs_used = (
            (sgprs_used + sgpr_granularity - 1) // sgpr_granularity
        ) * sgpr_granularity

        # Compute accum_offset (must be in range [4..256] in increments of 4)
        # For MFMA kernels with AGPRs, accum_offset indicates where AGPRs are mapped in VGPR space
        # For non-MFMA kernels, accum_offset still needs to be valid but AGPRs aren't used
        accum_offset = max(4, vgprs_used)

        # For MFMA kernels, reserve space for AGPRs beyond VGPRs
        # Compute AGPRs used dynamically from actual allocations
        if self.register_file.a_used:
            # AGPRs are allocated, compute how many we need
            agprs_used = max(self.register_file.a_used) + 1
            # Round up to AGPR granularity (same as VGPR granularity)
            agprs_used = (
                (agprs_used + vgpr_granularity - 1) // vgpr_granularity
            ) * vgpr_granularity
            # Total arch VGPRs must accommodate both VGPRs and AGPRs
            total_arch_vgprs = accum_offset + agprs_used
            vgprs_used = max(vgprs_used, total_arch_vgprs)

        txt = "\n".join(self.lines)
        txt = txt.replace(
            "  .amdhsa_next_free_vgpr 0", f"  .amdhsa_next_free_vgpr {vgprs_used}"
        )
        txt = txt.replace(
            "  .amdhsa_accum_offset 0", f"  .amdhsa_accum_offset {accum_offset}"
        )
        txt = txt.replace(
            "  .amdhsa_next_free_sgpr 0", f"  .amdhsa_next_free_sgpr {sgprs_used}"
        )
        txt = txt.replace(
            "  .amdhsa_group_segment_fixed_size 0",
            f"  .amdhsa_group_segment_fixed_size {lds_size_bytes}",
        )
        self.lines = txt.splitlines()

        amdhsa_minor = "2" if self.codeobj == "5" else "1"
        # Build YAML args with generic names (arg0_ptr, arg1_ptr, ...)
        args_yaml = []
        for i in range(num_args):
            args_yaml.append(
                f"""      - .name: arg{i}_ptr
        .size: 8
        .offset: {i*8}
        .value_kind: global_buffer
        .value_type: i8*"""
            )
        args_yaml_string = "\n".join(args_yaml)

        metadata = f"""
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - {amdhsa_minor}
amdhsa.kernels:
  - .name: {kernel_name}
    .symbol: '{kernel_name}.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
{args_yaml_string}
    .group_segment_fixed_size:   {lds_size_bytes}
    .kernarg_segment_align:      8
    .kernarg_segment_size:       {num_args*8}
    .max_flat_workgroup_size:    {wg_size_x * wg_size_y * wg_size_z}
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - {wg_size_x}
      - {wg_size_y}
      - {wg_size_z}
    .sgpr_count:                 {sgprs_used}
    .sgpr_spill_count:           0
    .uniform_work_group_size:    1
    .vgpr_count:                 {vgprs_used}
    .vgpr_spill_count:           0
    .wavefront_size:             {subgroup_size}
...
.end_amdgpu_metadata
""".rstrip(
            "\n"
        )
        self.emit(metadata)

    # ---- helpers used during per-op emission ----
    def ensure_lane_id(self, subgroup_size: int) -> int:
        """
        Ensure lane ID is computed and available in an allocated VGPR.

        Returns:
            VGPR index holding the lane ID
        """
        if self.lane_id_emitted:
            return self.lane_id_v

        # Allocate a VGPR for lane ID
        self.lane_id_v = self.vgpr_allocator.alloc_v()

        # Reserve it so it's never freed or reused
        self.vgpr_allocator.reserve(self.lane_id_v)

        self.emit(f"    # lane id (0..{subgroup_size-1})")
        # Compute lane ID directly into allocated VGPR
        self.unified.v_mbcnt_lo_u32_b32(f"v{self.lane_id_v}", -1, 0)
        self.unified.v_mbcnt_hi_u32_b32(f"v{self.lane_id_v}", -1, f"v{self.lane_id_v}")
        self.lane_id_emitted = True
        return self.lane_id_v

    # ========= Temp allocation/pinning helpers (API surface) =========
    def pin_vgpr(self, vreg: int) -> None:
        """Mark a VGPR as pinned to avoid reuse by external policies."""
        self._pinned_vgprs.add(vreg)

    def unpin_vgpr(self, vreg: int) -> None:
        """Unmark a VGPR as pinned."""
        self._pinned_vgprs.discard(vreg)

    # ========= MFMA tracking for matrix operations =========
    def track_mfma(self, mfma_instruction: str) -> None:
        """
        Track MFMA instruction issue for latency-aware scheduling.

        Args:
            mfma_instruction: MFMA instruction name (e.g., "v_mfma_f32_16x16x16_f16")
        """
        if self.scoreboard is not None:
            self._mfma_last_cycle = self.scoreboard.current_cycle
            latency = self.latency_provider.get_latency(mfma_instruction)
            if latency:
                self.emit(f"    # MFMA issued, latency ~{latency:.0f} cycles")

    def wait_for_mfma_ready(self) -> None:
        """
        Ensure MFMA result is ready before consuming it.

        Inserts s_nop if needed based on MFMA→AGPR read latency from database.
        """
        if self.scoreboard is None or self._mfma_last_cycle is None:
            return

        # Check if enough cycles have elapsed
        elapsed = self.scoreboard.current_cycle - self._mfma_last_cycle

        # Get MFMA→AGPR read latency from database
        mfma_to_agpr_read_latency = self.latency_provider.get_latency(
            "mfma_to_agpr_read"
        )
        if mfma_to_agpr_read_latency is None:
            return

        if elapsed < mfma_to_agpr_read_latency:
            cycles_needed = mfma_to_agpr_read_latency - elapsed
            nops = min(int(cycles_needed), 15)
            if nops > 0:
                self.unified.s_nop(nops)
                self.scoreboard.advance(nops)

    # ========= Scoreboard-based hazard detection (optional) =========
    def track_instruction(
        self,
        instruction: str,
        writes: Optional[Set[str]] = None,
        reads: Optional[Set[str]] = None,
        ticket: Optional[int] = None,
    ) -> None:
        """
        Track an instruction in the scoreboard for hazard detection.

        Args:
            instruction: Instruction name (e.g., "buffer_load_dwordx4")
            writes: Set of resources written (e.g., {"v0", "v1"})
            reads: Set of resources read
            ticket: Optional VMEM/LGKM ticket
        """
        if self.scoreboard is not None:
            self.scoreboard.issue(
                instruction, writes=writes, reads=reads, ticket=ticket
            )

    def check_and_insert_wait(
        self,
        reads: Optional[Set[str]] = None,
        writes: Optional[Set[str]] = None,
        instruction: Optional[str] = None,
    ) -> None:
        """
        Check for hazards and insert wait if needed.

        Args:
            reads: Resources to be read by upcoming instruction
            writes: Resources to be written by upcoming instruction
            instruction: Optional instruction name for better wait selection
        """
        if self.scoreboard is None:
            return

        hazard = self.scoreboard.check_hazard(reads=reads, writes=writes)
        if hazard:
            hazard_type, cycles_needed = hazard
            # For now, emit conservative wait
            # Future: could insert nops or try to reorder independent instructions
            if cycles_needed > 0:
                self.emit(
                    f"    # Hazard detected: {hazard_type}, {cycles_needed:.0f} cycles"
                )
                # Insert s_nop if < 10 cycles, otherwise s_waitcnt
                if cycles_needed < 10:
                    nops = min(int(cycles_needed), 15)
                    self.unified.s_nop(nops)
                else:
                    # Insert wait for pending operations
                    self.unified.s_waitcnt("vmcnt(0) lgkmcnt(0)")

    # ---- synchronization and LDS helpers ----
    def emit_barrier(self):
        """
        Emit a shared memory barrier with optimal LGKM wait coalescing.

        This drains all outstanding LGKM operations (lgkmcnt(0)) only if needed,
        then emits the workgroup barrier (s_barrier).

        Coalescing: If no LGKM operations are outstanding or we've already
        drained to lgkmcnt(0) since the last LGKM producer, we skip the wait.
        """
        # Check if there are outstanding LGKM operations that need draining
        # _lgkm_last_ticket >= 0 means at least one LGKM op has been issued
        # _lgkm_last_wait_threshold != 0 means we haven't already drained to 0
        has_outstanding_lgkm = (
            self.ticketing._lgkm_last_ticket >= 0
            and self.ticketing._lgkm_last_wait_threshold != 0
        )

        # Emit lgkmcnt(0) only if there are outstanding LGKM operations
        if has_outstanding_lgkm:
            self.unified.s_waitcnt("lgkmcnt(0)")

        # Always record that we've observed an lgkmcnt(0) at this barrier
        # This prevents redundant waits after the barrier until new LGKM ops occur
        self.ticketing.observe_lgkm_wait(0)

        # Emit the workgroup synchronization barrier
        self.unified.s_barrier()

    def emit_lds_write_b32(self, addr_vreg: int, src_vreg: int):
        self.unified.ds_write_b32(f"v{addr_vreg}", f"v{src_vreg}")

    def emit_lds_write_b64(self, addr_vreg: int, src_pair: Tuple[int, int]):
        self.unified.ds_write_b64(f"v{addr_vreg}", f"v[{src_pair[0]}:{src_pair[1]}]")

    def emit_lds_write_b128(self, addr_vreg: int, src_quad: Tuple[int, int, int, int]):
        self.unified.ds_write_b128(f"v{addr_vreg}", f"v[{src_quad[0]}:{src_quad[3]}]")

    def emit_lds_read_b64(self, dst_pair: Tuple[int, int], addr_vreg: int, offset: int = 0):
        """Emit ds_read_b64 instruction.
        
        Args:
            dst_pair: Tuple of (start, end) VGPRs for destination
            addr_vreg: VGPR containing base address
            offset: Optional immediate offset in bytes (0-65535)
        """
        self.unified.ds_read_b64(f"v[{dst_pair[0]}:{dst_pair[1]}]", f"v{addr_vreg}", offset=offset)

    def emit_mfma_16x16x16_f16(
        self, a_pair: Tuple[int, int], b_pair: Tuple[int, int], acc_quad=None
    ):
        """
        Emit MFMA instruction with VGPR result (not AGPR).

        Uses VGPR-variant of MFMA instruction to write results directly to VGPRs,
        avoiding accumulator complexity. This is required for multi-wave support
        and matches LLVM backend behavior.

        Args:
            a_pair: VGPR pair for A operand
            b_pair: VGPR pair for B operand
            acc_quad: Optional VGPR quad to use as accumulator (for chained MFMAs)
        """
        # Wait for LDS reads to complete before MFMA
        self.unified.s_waitcnt("lgkmcnt(0)")

        # Determine result/accumulator quad
        use_loop_accumulator = False
        result_quad = None

        # If explicit accumulator quad is passed (for chained MFMAs), use it
        if acc_quad is not None:
            result_quad = acc_quad
            use_loop_accumulator = True  # Use acc_quad as both input and output
        # Otherwise check if we're in a loop with pre-allocated accumulator
        elif hasattr(self, "loop_stack") and self.loop_stack:
            loop_ctx = self.loop_stack[-1]
            iter_arg_vgprs = loop_ctx.get("iter_arg_vgprs", [])
            if iter_arg_vgprs:
                # Use first accumulator (K-loop pattern)
                result_quad = iter_arg_vgprs[0]
                use_loop_accumulator = True

        # Fall back to allocating new quad if no loop accumulator
        if result_quad is None:
            result_quad = self.vgpr_allocator.alloc_v_quad()

        vgpr_base = result_quad[0]

        # Emit MFMA into VGPR quad
        # If using loop accumulator or explicit accumulator, pass it as the accumulator operand
        # Otherwise use 0 (zero initialization)
        acc_operand = f"v[{vgpr_base}:{vgpr_base+3}]" if use_loop_accumulator else "0"

        self.unified.v_mfma_f32_16x16x16_f16(
            f"v[{vgpr_base}:{vgpr_base+3}]",
            f"v[{a_pair[0]}:{a_pair[1]}]",
            f"v[{b_pair[0]}:{b_pair[1]}]",
            acc_operand
        )
        # Track MFMA for latency-aware scheduling
        self.track_mfma("v_mfma_f32_16x16x16_f16")

        # Store the result quad for later use
        self.current_vgpr_quad = result_quad

    def compute_lane_scaled_offset(self, scale_bytes: int) -> int:
        """
        Compute lane_id * scale_bytes into an allocated VGPR.

        Args:
            scale_bytes: Scale factor to multiply lane ID by

        Returns:
            VGPR index containing the result. Caller is responsible for freeing it.
        """
        lane_id_v = self.ensure_lane_id(64)
        result_v = self.vgpr_allocator.alloc_v()

        if (scale_bytes & (scale_bytes - 1)) == 0:
            # Power of 2: use shift
            shift_amount = scale_bytes.bit_length() - 1
            self.unified.v_lshlrev_b32(f"v{result_v}", shift_amount, f"v{lane_id_v}")
        else:
            # Non-power of 2: use multiply
            temp_v = self.vgpr_allocator.alloc_v()
            self.unified.v_mov_b32(f"v{temp_v}", scale_bytes)
            self.unified.v_mul_lo_u32(f"v{result_v}", f"v{lane_id_v}", f"v{temp_v}")
            self.vgpr_allocator.free_v(temp_v)

        return result_v

    def materialize_scalar_literal(self, sreg: int, literal: int):
        # Use s_movk_i32 for small literals (16-bit signed); fallback to s_mov_b32 for larger values
        if -32768 <= literal <= 32767:
            self.unified.s_movk_i32(f"s{sreg}", int(literal))
        else:
            self.unified.s_mov_b32(f"s{sreg}", int(literal))
        self.register_file.s_max = max(self.register_file.s_max, sreg)
