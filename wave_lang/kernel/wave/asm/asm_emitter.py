# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Assembly emitter for generating AMDGCN assembly instructions.
"""

from typing import Dict, List, Tuple, Union

from wave_lang.support.ir_imports import (
    func_d,
    Context,
    Module,
)

from .kernel_model import KernelInfo, MemRefInfo
from .utils import emit_expression_asm
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
    SRD127_96 = "0x20000"  # data_format=4 for gfx9xx

    def __init__(self, targetid: str, codeobj: str):
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
        self.current_agpr_quad = None  # Track current AGPR quad allocated for MFMA
        # Track pinned VGPRs for future lifetime management (API surface)
        self._pinned_vgprs = set()
        # VMEM ticket tracking for optimal vmcnt placement
        self._vmem_last_ticket = -1  # last issued VMEM ticket
        self._vmem_last_wait_threshold = (
            None  # last vmcnt(N) emitted, to coalesce waits
        )

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

                    # Emit kernel preamble and kernargs
                    emitter.emit_prologue(kernel_name)
                    emitter.emit_kernargs(num_args)

                    # Do full traversal with emitter to emit instructions
                    walker = IRWalker(emitter)
                    kernel_info = walker.interpret_func(fn)

                    emitter.emit_epilogue(
                        kernel_info.name,
                        kernel_info.wg_size[0],
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
        """Emit an instruction object directly."""
        self.lines.append(str(instr))

    # ---- high-level sections ----
    def emit_prologue(self, kernel_name: str):
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
        self.emit("  .amdhsa_accum_offset 0")  # patched later
        self.emit("  .amdhsa_next_free_vgpr 0")  # patched later
        self.emit("  .amdhsa_next_free_sgpr 0")  # patched later
        self.emit("  .amdhsa_group_segment_fixed_size 0")
        self.emit("  .amdhsa_private_segment_fixed_size 0")
        self.emit("  .amdhsa_system_sgpr_workgroup_id_x 0")
        self.emit("  .amdhsa_system_sgpr_workgroup_id_y 0")
        self.emit("  .amdhsa_system_sgpr_workgroup_id_z 0")
        self.emit("  .amdhsa_system_vgpr_workitem_id 0")
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
        wg_size_x: int,
        subgroup_size: int,
        num_args: int,
        lds_size_bytes: int = 0,
    ):
        self.emit_instruction(SEndpgm())
        self.emit("")

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
    .max_flat_workgroup_size:    {wg_size_x}
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

    # ========= VMEM ticket tracking for optimal vmcnt placement =========
    def _next_vmem_ticket(self) -> int:
        """Issue next VMEM ticket for a buffer load operation."""
        self._vmem_last_ticket += 1
        # Reset wait threshold since a new load invalidates previous waits
        self._vmem_last_wait_threshold = None
        return self._vmem_last_ticket

    def wait_for_vmem_ticket(self, min_required_ticket: int) -> None:
        """
        Emit minimal s_waitcnt vmcnt(N) to ensure ticket is ready.

        Computes threshold to allow newer loads to remain in-flight while
        guaranteeing min_required_ticket is complete.

        Args:
            min_required_ticket: Ticket that must be ready before proceeding
        """
        # Compute threshold: allow (last_ticket - min_required_ticket) newer loads
        threshold = max(0, self._vmem_last_ticket - min_required_ticket)

        # Coalesce: only emit if stricter than last wait (or first wait)
        if (
            self._vmem_last_wait_threshold is None
            or threshold < self._vmem_last_wait_threshold
        ):
            self.emit_instruction(SWaitcnt(f"vmcnt({threshold})"))
            self._vmem_last_wait_threshold = threshold

    # ---- synchronization and LDS helpers ----
    def emit_barrier(self):
        self.emit_instruction(SWaitcnt("lgkmcnt(0)"))
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
        Emit MFMA instruction with dynamically allocated AGPR result.

        Allocates an AGPR quad for the MFMA result and records it for later spilling.
        """
        from .instructions import VMfmaF32_16x16x16F16, SNop, SWaitcnt

        # Wait for LDS reads to complete before MFMA
        self.emit_instruction(SWaitcnt("lgkmcnt(0)"))

        # Allocate AGPR quad for MFMA result
        self.current_agpr_quad = self.agpr_allocator.alloc_a_quad()
        agpr_base = self.current_agpr_quad[0]

        # Emit MFMA into allocated AGPR quad
        self.emit_instruction(VMfmaF32_16x16x16F16(agpr_base, a_pair, b_pair, 0))
        self.emit_instruction(SNop(6))

    def spill_agprs_to_vgprs(
        self, dst_v_quad: Tuple[int, int, int, int] = None
    ) -> Tuple[int, int, int, int]:
        """
        Spill MFMA AGPRs to VGPRs for use in stores.

        Args:
            dst_v_quad: Optional VGPR quad to spill into. If None, allocates a new quad.

        Returns:
            VGPR quad (v0, v1, v2, v3) containing the spilled MFMA results
        """
        from .instructions import VAccvgprReadB32

        if dst_v_quad is None:
            dst_v_quad = self.vgpr_allocator.alloc_v_quad()

        # Read from currently allocated AGPR quad into VGPRs
        if self.current_agpr_quad is None:
            raise RuntimeError(
                "No AGPR quad allocated; call emit_mfma_16x16x16_f16 first"
            )

        for i in range(4):
            self.emit_instruction(
                VAccvgprReadB32(dst_v_quad[i], self.current_agpr_quad[i])
            )

        return dst_v_quad

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
        kernarg_instructions = emit_kernargs(num_args)
        for i, instruction in enumerate(kernarg_instructions):
            if isinstance(instruction, SLoadDwordx2):
                # Extract register numbers from the instruction and store them
                low_register, high_register = self.sgpr_allocator.pair()
                self.ptr_pairs[i] = (low_register, high_register)
                # Create a new instruction with the allocated registers
                new_instruction = SLoadDwordx2(
                    (low_register, high_register), (0, 1), i * 8
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
        ticket = self._next_vmem_ticket()  # Issue ticket for this load
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
