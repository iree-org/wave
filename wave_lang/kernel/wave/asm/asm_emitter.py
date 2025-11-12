# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Assembly emitter for generating AMDGCN assembly instructions.
"""

from typing import Dict, List, Tuple, Union, Optional, Set

from wave_lang.support.ir_imports import (
    func_d,
    Context,
    Module,
)

from .kernel_model import KernelInfo, MemRefInfo
from .utils import emit_expression_asm, normalize_wg_size
from .register_allocator import RegFile, SGPRAllocator, VGPRAllocator, AGPRAllocator
from .mlir_walker import IRWalker
from .instructions import (
    SLoadDwordx2,
    BufferLoadDwordx4,
    BufferLoadDwordx2,
    BufferStoreDwordx4,
    BufferStoreDword,
    SMovB32,
    SMovkI32,
    VMbcntLoU32B32,
    VMbcntHiU32B32,
    VLshlRevB32,
    VMovB32,
    VMulLoU32,
    SWaitcnt,
    SEndpgm,
    SBarrier,
    emit_kernargs,
)

# Import latency-aware scheduling infrastructure
from .latency_provider import LatencyProvider
from .scoreboard import Scoreboard
from .ticketing import Ticketing


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
        # Track pinned VGPRs for future lifetime management (API surface)
        self._pinned_vgprs = set()

        # Workgroup ID tracking (for multi-workgroup support)
        # These are assigned sequentially after kernarg ptr based on what we request
        self.sgpr_workgroup_id_x = None
        self.sgpr_workgroup_id_y = None
        self.sgpr_workgroup_id_z = None

        # Thread/workitem ID tracking (for multi-wave support)
        # System VGPRs are allocated by hardware at v0, v1, v2 for tid_x, tid_y, tid_z
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
                    emitter.emit_kernargs(num_args)

                    # Walk MLIR and emit instructions
                    walker = IRWalker(emitter)
                    kernel_info = walker.interpret_func(fn)

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

    # ---- low-level ----
    def emit(self, s: str):
        self.lines.append(s)

    def emit_instruction(self, instr):
        """
        Emit an instruction object directly.

        Automatically issues VMEM/LGKM tickets for memory operations based on
        instruction categorization. This ensures uniform, centralized ticketing
        for all emitted instructions.
        """
        # Import here to avoid circular dependency
        from .instruction_categories import InstructionCategory, categorize_instruction

        # Issue tickets for memory operations based on instruction category
        if hasattr(instr, "mnemonic") and instr.mnemonic:
            category = categorize_instruction(instr.mnemonic)

            if category == InstructionCategory.VMEM:
                self.ticketing.next_vmem_ticket()
            elif category == InstructionCategory.LGKM:
                self.ticketing.next_lgkm_ticket()

        self.lines.append(str(instr))

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
            next_system_sgpr += 1
        else:
            self.sgpr_workgroup_id_x = None

        if self.needs_wgid_y:
            self.sgpr_workgroup_id_y = next_system_sgpr
            next_system_sgpr += 1
        else:
            self.sgpr_workgroup_id_y = None

        if self.needs_wgid_z:
            self.sgpr_workgroup_id_z = next_system_sgpr
            next_system_sgpr += 1
        else:
            self.sgpr_workgroup_id_z = None

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
            self._reserve_system_vgprs(requested_dims)
        else:
            # Single-wave: no system VGPRs (matches LLVM)
            requested_dims = 0
            self.vgpr_tid_x = None  # No system VGPR, use lane_id fallback
            self.vgpr_tid_y = None
            self.vgpr_tid_z = None
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
        self.emit_instruction(SEndpgm())
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
        # Patch system_vgpr_workitem_id if lane-id sequence was emitted
        if self.lane_id_emitted:
            txt = txt.replace(
                "  .amdhsa_system_vgpr_workitem_id 0",
                "  .amdhsa_system_vgpr_workitem_id 1",
            )
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
    .sgpr_count:                 {sgprs_used}
    .sgpr_spill_count:           0
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
        self.emit_instruction(VMbcntLoU32B32(self.lane_id_v, -1))
        self.emit_instruction(VMbcntHiU32B32(self.lane_id_v, -1, self.lane_id_v))
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
                from .instructions import SNop

                self.emit_instruction(SNop(nops))
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
                    from .instructions import SNop

                    self.emit_instruction(SNop(nops))
                else:
                    # Insert wait for pending operations
                    self.emit_instruction(SWaitcnt("vmcnt(0) lgkmcnt(0)"))

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
            self.emit_instruction(SWaitcnt("lgkmcnt(0)"))

        # Always record that we've observed an lgkmcnt(0) at this barrier
        # This prevents redundant waits after the barrier until new LGKM ops occur
        self.ticketing.observe_lgkm_wait(0)

        # Emit the workgroup synchronization barrier
        self.emit_instruction(SBarrier())

    def emit_lds_write_b32(self, addr_vreg: int, src_vreg: int):
        from .instructions import DSWriteB32

        self.emit_instruction(DSWriteB32(addr_vreg, src_vreg))

    def emit_lds_write_b64(self, addr_vreg: int, src_pair: Tuple[int, int]):
        from .instructions import DSWriteB64

        self.emit_instruction(DSWriteB64(addr_vreg, src_pair))

    def emit_lds_write_b128(self, addr_vreg: int, src_quad: Tuple[int, int, int, int]):
        from .instructions import DSWriteB128

        self.emit_instruction(DSWriteB128(addr_vreg, src_quad))

    def emit_lds_read_b64(self, dst_pair: Tuple[int, int], addr_vreg: int):
        from .instructions import DSReadB64

        self.emit_instruction(DSReadB64(dst_pair, addr_vreg))

    def emit_mfma_16x16x16_f16(self, a_pair: Tuple[int, int], b_pair: Tuple[int, int]):
        """
        Emit MFMA instruction with VGPR result (not AGPR).

        Uses VGPR-variant of MFMA instruction to write results directly to VGPRs,
        avoiding accumulator complexity. This is required for multi-wave support
        and matches LLVM backend behavior.
        """
        from .instructions import VMfmaF32_16x16x16F16, SWaitcnt

        # Wait for LDS reads to complete before MFMA
        self.emit_instruction(SWaitcnt("lgkmcnt(0)"))

        # Allocate VGPR quad for MFMA result (not AGPR)
        result_quad = self.vgpr_allocator.alloc_v_quad()
        vgpr_base = result_quad[0]

        # Emit MFMA into VGPR quad (VGPR-variant instruction)
        # The 0 at the end means no accumulator input (zero initialization)
        # use_vgpr=True makes it write to VGPRs instead of AGPRs
        self.emit_instruction(
            VMfmaF32_16x16x16F16(vgpr_base, a_pair, b_pair, 0, use_vgpr=True)
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
            self.emit_instruction(VLshlRevB32(result_v, shift_amount, lane_id_v))
        else:
            # Non-power of 2: use multiply
            temp_v = self.vgpr_allocator.alloc_v()
            self.emit_instruction(VMovB32(temp_v, scale_bytes))
            self.emit_instruction(VMulLoU32(result_v, lane_id_v, temp_v))
            self.vgpr_allocator.free_v(temp_v)

        return result_v

    def materialize_scalar_literal(self, sreg: int, literal: int):
        # Use s_movk_i32 for small literals (16-bit signed); fallback to s_mov_b32 for larger values
        if -32768 <= literal <= 32767:
            self.emit_instruction(SMovkI32(sreg, int(literal)))
        else:
            self.emit_instruction(SMovB32(sreg, int(literal)))
        self.register_file.s_max = max(self.register_file.s_max, sreg)

    def emit_kernargs(self, num_args: int):
        # Kernarg pointer is ALWAYS at s[0:1] (user SGPR, comes before system SGPRs)
        kernarg_ptr_low = 0
        kernarg_ptr_high = 1

        kernarg_instructions = emit_kernargs(
            num_args, (kernarg_ptr_low, kernarg_ptr_high)
        )
        for i, instruction in enumerate(kernarg_instructions):
            if isinstance(instruction, SLoadDwordx2):
                # Extract register numbers from the instruction and store them
                low_register, high_register = self.sgpr_allocator.pair()
                self.ptr_pairs[i] = (low_register, high_register)
                # Create a new instruction with the allocated registers
                new_instruction = SLoadDwordx2(
                    (low_register, high_register),
                    (kernarg_ptr_low, kernarg_ptr_high),
                    i * 8,
                )
                self.emit_instruction(new_instruction)
            else:
                self.emit_instruction(instruction)
        self.emit("")  # Add empty line

    def ensure_srd_for_subspan(self, memref_ssa: str, arg_index: int, limit_bytes: int):
        if memref_ssa in self.srds:
            return
        base_low_register, base_high_register = self.ptr_pairs[arg_index]
        srd_register_0, srd_register_1, srd_register_2, srd_register_3 = (
            self.sgpr_allocator.quad()
        )
        self.srds[memref_ssa] = (
            srd_register_0,
            srd_register_1,
            srd_register_2,
            srd_register_3,
        )

        # Use instruction classes for SRD setup
        self.emit(f"    # SRD for {memref_ssa} (arg{arg_index})")
        self.emit_instruction(SMovB32(srd_register_0, f"s{base_low_register}"))
        self.emit_instruction(SMovB32(srd_register_1, f"s{base_high_register}"))
        self.emit_instruction(SMovB32(srd_register_2, limit_bytes))
        self.emit_instruction(SMovB32(srd_register_3, "Srd127_96"))

    def compute_voffset_and_instoffset_expr(
        self,
        kernel_info: KernelInfo,
        byte_offset_expr,
    ) -> Tuple[int, int]:
        """
        Compute voffset and instoffset from a prebuilt SymPy byte-offset expression.

        Args:
            kernel_info: Kernel information (for subgroup_size, etc.)
            byte_offset_expr: SymPy expression representing the byte offset

        Returns:
            Tuple of (voffset_vgpr, instoffset)
            - voffset_vgpr: VGPR index allocated for dynamic offset (caller owns lifetime)
            - instoffset: Integer constant offset for instruction immediate
        """
        from .utils import split_const_dynamic
        from .instructions import VMovB32

        # Split into constant and dynamic parts
        const_offset, dynamic_expr = split_const_dynamic(byte_offset_expr)

        # If entirely constant, use zero VGPR
        if dynamic_expr == 0 or (
            hasattr(dynamic_expr, "is_zero") and dynamic_expr.is_zero
        ):
            voffset_v = self.vgpr_allocator.alloc_v()
            self.emit_instruction(VMovB32(voffset_v, 0))
            return voffset_v, const_offset

        # Materialize dynamic expression into a VGPR
        voffset_v = self.vgpr_allocator.alloc_v()
        emit_expression_asm(dynamic_expr, self, kernel_info, dst_reg=f"v{voffset_v}")

        return voffset_v, const_offset

    def materialize_byte_offset_expr(
        self,
        kernel_info: KernelInfo,
        byte_offset_expr,
    ) -> int:
        """
        Materialize a byte-offset expression into a VGPR for LDS addressing.

        Unlike compute_voffset_and_instoffset_expr, this does not split into
        constant/dynamic parts - it materializes the full expression into a VGPR.

        Args:
            kernel_info: Kernel information (for subgroup_size, etc.)
            byte_offset_expr: SymPy expression representing the byte offset

        Returns:
            VGPR index containing the computed address (caller owns lifetime)
        """
        from .instructions import VMovB32

        # If expression is a constant, just materialize it
        if isinstance(byte_offset_expr, int) or (
            hasattr(byte_offset_expr, "is_number") and byte_offset_expr.is_number
        ):
            addr_v = self.vgpr_allocator.alloc_v()
            const_val = int(byte_offset_expr)
            self.emit_instruction(VMovB32(addr_v, const_val))
            return addr_v

        # Materialize the full expression into a VGPR
        addr_v = self.vgpr_allocator.alloc_v()
        emit_expression_asm(byte_offset_expr, self, kernel_info, dst_reg=f"v{addr_v}")

        return addr_v

    def compute_voffset_and_instoffset_list(
        self,
        kernel_info: KernelInfo,
        byte_exprs: List,
    ) -> List[Tuple[int, int]]:
        """
        Compute voffset and instoffset for multiple byte-offset expressions.

        Args:
            kernel_info: Kernel information (for subgroup_size, etc.)
            byte_exprs: List of SymPy expressions representing byte offsets

        Returns:
            List of (voffset_vgpr, instoffset) tuples, one per expression.
            Caller owns lifetime of all allocated VGPRs.
        """
        results = []
        for expr in byte_exprs:
            voffset_v, instoffset = self.compute_voffset_and_instoffset_expr(
                kernel_info, expr
            )
            results.append((voffset_v, instoffset))
        return results

    def compute_voffset_and_instoffset(
        self,
        kernel_info: KernelInfo,
        memref_info: MemRefInfo,
        indices: List[str],
    ) -> Tuple[int, int]:
        """
        Compute:
          - an allocated VGPR index to use as voffset (caller owns lifetime),
          - an instoffset immediate (folded constant byte part).

        This is a convenience wrapper that builds the byte offset expression
        from indices and forwards to compute_voffset_and_instoffset_expr.
        """
        from .utils import build_memref_byte_offset_expr

        # Build byte offset expression from indices
        byte_offset_expr = build_memref_byte_offset_expr(
            indices, kernel_info, memref_info
        )

        # Use the expression-based method
        return self.compute_voffset_and_instoffset_expr(kernel_info, byte_offset_expr)

    def chunk_offsets(self, vector_bytes: int) -> List[int]:
        """
        Split vector bytes into 16-byte chunks for AMDGCN buffer instructions.

        The 16-byte chunk size is hardcoded because:
        - buffer_load_dwordx4 loads 4 dwords (4 × 4 bytes = 16 bytes)
        - buffer_store_dwordx4 stores 4 dwords (4 × 4 bytes = 16 bytes)
        - Each AMDGCN buffer instruction can handle exactly 16 bytes at a time

        This is a hardware limitation of the AMDGCN instruction set, not a configurable parameter.
        """
        if vector_bytes % 16 != 0:
            raise ValueError(f"Tail not supported (remaining {vector_bytes} bytes).")
        return list(map(lambda x: x * 16, range(vector_bytes // 16)))

    def emit_load(
        self,
        memref_ssa: str,
        vector_bytes: int,
        vector_index_register: Union[str, int],
        instruction_offset_base: int,
    ):
        srd_register_0, srd_register_1, srd_register_2, srd_register_3 = self.srds[
            memref_ssa
        ]
        register_list = []
        self.emit(f"    # load {vector_bytes}B from {memref_ssa}")
        if vector_bytes == 8:
            # Allocate a pair of VGPRs using the allocator
            vpair = self.vgpr_allocator.alloc_v_pair()
            register_list.append(vpair)
            load_instruction = BufferLoadDwordx2(
                (vpair[0], vpair[1]),
                vector_index_register,
                (srd_register_0, srd_register_1, srd_register_2, srd_register_3),
                instruction_offset_base,
            )
            self.emit_instruction(load_instruction)
            # Don't wait immediately - let caller decide when to wait
        else:
            for offset in self.chunk_offsets(vector_bytes):
                vector_quad = self.vgpr_allocator.quad()
                register_list.append(vector_quad)
                total_offset = instruction_offset_base + offset
                load_instruction = BufferLoadDwordx4(
                    vector_quad,
                    vector_index_register,
                    (srd_register_0, srd_register_1, srd_register_2, srd_register_3),
                    total_offset,
                )
                self.emit_instruction(load_instruction)
                # Don't wait immediately - let caller decide when to wait
        # Return the current VMEM ticket (the last one issued by the loads above)
        ticket = self.ticketing._vmem_last_ticket
        return register_list, ticket

    def emit_store_with_regs(
        self,
        memref_ssa: str,
        register_list: List[Tuple[int, int, int, int]],
        vector_bytes: int,
        vector_index_register: Union[str, int],
        instruction_offset_base: int,
    ):
        srd_register_0, srd_register_1, srd_register_2, srd_register_3 = self.srds[
            memref_ssa
        ]
        self.emit(f"    # store {vector_bytes}B to {memref_ssa}")
        if vector_bytes == 4:
            # Treat register_list[0] as a scalar VGPR
            src_reg = (
                register_list[0]
                if isinstance(register_list[0], int)
                else register_list[0][0]
            )
            store_instruction = BufferStoreDword(
                src_reg,
                vector_index_register,
                (srd_register_0, srd_register_1, srd_register_2, srd_register_3),
                instruction_offset_base,
            )
            self.emit_instruction(store_instruction)
        else:
            for vector_quad, offset in zip(
                register_list, self.chunk_offsets(vector_bytes)
            ):
                total_offset = instruction_offset_base + offset
                store_instruction = BufferStoreDwordx4(
                    vector_quad,
                    vector_index_register,
                    (srd_register_0, srd_register_1, srd_register_2, srd_register_3),
                    total_offset,
                )
                self.emit_instruction(store_instruction)

    def emit_store_scalar(
        self,
        memref_ssa: str,
        src_vreg: int,
        vector_index_register: str,
        instruction_offset: int,
    ):
        srd_register_0, srd_register_1, srd_register_2, srd_register_3 = self.srds[
            memref_ssa
        ]
        self.emit(f"    # store 4B to {memref_ssa}")
        store_instruction = BufferStoreDword(
            src_vreg,
            vector_index_register,
            (srd_register_0, srd_register_1, srd_register_2, srd_register_3),
            instruction_offset,
        )
        self.emit_instruction(store_instruction)

    def emit_store_scalar_with_vindex(
        self,
        memref_ssa: str,
        src_vreg: int,
        vindex_vreg: int,
        instoffset: int = 0,
    ):
        srd_register_0, srd_register_1, srd_register_2, srd_register_3 = self.srds[
            memref_ssa
        ]
        self.emit(f"    # store 4B to {memref_ssa}")
        store_instruction = BufferStoreDword(
            src_vreg,
            f"v{vindex_vreg}",
            (srd_register_0, srd_register_1, srd_register_2, srd_register_3),
            instoffset,
        )
        self.emit_instruction(store_instruction)
