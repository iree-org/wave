# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Assembly emitter for generating AMDGCN assembly instructions.
"""

from typing import Dict, List, Tuple

from wave_lang.support.ir_imports import (
    func_d,
    Context,
    Module,
)

from .kernel_model import KernelInfo, MemRefInfo
from .register_allocator import RegFile, SGPRAllocator, VGPRAllocator
from .mlir_walker import IRWalker
from .instructions import (
    SLoadDwordx2,
    BufferLoadDwordx4,
    BufferStoreDwordx4,
    SMovB32,
    VMbcntLoU32B32,
    VMbcntHiU32B32,
    VLshlRevB32,
    VMovB32,
    VMulLoU32,
    SWaitcnt,
    SEndpgm,
    emit_kernargs,
)


class AsmEmitter:
    SRD127_96 = "0x20000"  # data_format=4 for gfx9xx

    def __init__(self, targetid: str, codeobj: str):
        self.targetid = targetid
        self.codeobj = codeobj
        self.lines: List[str] = []
        self.register_file = RegFile()
        self.sgpr_allocator = SGPRAllocator(self.register_file)
        self.vgpr_allocator = VGPRAllocator(self.register_file)
        self.ptr_pairs: Dict[int, Tuple[int, int]] = {}  # arg_index -> (s_lo, s_hi)
        self.srds: Dict[str, Tuple[int, int, int, int]] = {}  # memref_ssa -> srd quad
        self.lane_id_emitted = False

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
                    ki = walker.interpret_func(fn)

                    emitter.emit_epilogue(
                        ki.name, ki.wg_size[0], ki.subgroup_size, len(ki.arg_ssa_order)
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
        self.emit("  .amdhsa_system_sgpr_workgroup_id_x 1")
        self.emit("  .amdhsa_system_sgpr_workgroup_id_y 1")
        self.emit("  .amdhsa_system_sgpr_workgroup_id_z 1")
        self.emit("  .amdhsa_system_vgpr_workitem_id 0")
        self.emit("  .amdhsa_float_denorm_mode_32 3")
        self.emit("  .amdhsa_float_denorm_mode_16_64 3")
        self.emit(".end_amdhsa_kernel")
        self.emit(".text\n")
        self.emit("# SRD upper word (gfx9xx): data_format=4 => 0x20000")
        self.emit(f".set Srd127_96, {self.SRD127_96}\n")
        self.emit(f"{kernel_name}:")

    def emit_epilogue(
        self, kernel_name: str, wg_size_x: int, subgroup_size: int, num_args: int
    ):
        self.emit_instruction(SEndpgm())
        self.emit("")

        vgprs_used = max(self.register_file.v_used) + 1
        sgprs_used = self.register_file.s_max + 1
        txt = "\n".join(self.lines)
        txt = txt.replace(
            "  .amdhsa_next_free_vgpr 0", f"  .amdhsa_next_free_vgpr {vgprs_used}"
        )
        txt = txt.replace(
            "  .amdhsa_accum_offset 0", f"  .amdhsa_accum_offset {vgprs_used}"
        )
        txt = txt.replace(
            "  .amdhsa_next_free_sgpr 0", f"  .amdhsa_next_free_sgpr {sgprs_used}"
        )
        self.lines = txt.splitlines()

        amdhsa_minor = "2" if self.codeobj == "5" else "1"
        # Build YAML args from num_args
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
    .group_segment_fixed_size:   0
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
    def ensure_lane_id(self, subgroup_size: int):
        if self.lane_id_emitted:
            return
        self.emit(f"    # lane id (0..{subgroup_size-1})")
        self.emit_instruction(VMbcntLoU32B32(0, -1))
        self.emit_instruction(VMbcntHiU32B32(0, -1, 0))
        self.lane_id_emitted = True
        self.register_file.v_used.add(0)

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

    def compute_voffset_and_instoffset(
        self,
        kernel_info: KernelInfo,
        memref_info: MemRefInfo,
        indices: List[str],
    ) -> Tuple[str, int]:
        """
        Compute:
          - a VGPR name to use as voffset (we use v2 consistently),
          - an instoffset immediate (folded constant byte part).
        Supports patterns:
          i = tid.x (possibly via affine.apply simplification),
          j = constant small int,
        over memref strided<[s0, s1]> with elem_bytes.
        """
        strides = memref_info.strides_elems
        element_bytes = memref_info.elem_bytes

        # Default instoffset
        instruction_offset_bytes = 0

        # Row index (dim0)
        row_index_ssa = indices[0]
        row_stride_bytes = strides[0] * element_bytes

        # Column index (dim1) — fold into instoffset if constant
        if len(indices) > 1:
            column_index_ssa = indices[1]
            column_value = kernel_info.index_env.get(column_index_ssa, None)
            if isinstance(column_value, int):
                instruction_offset_bytes += column_value * strides[1] * element_bytes

        # Ensure lane id if row index depends on tid.x
        row_value = kernel_info.index_env.get(row_index_ssa, None)
        if row_value == "tid.x":
            self.ensure_lane_id(kernel_info.subgroup_size)
            # Compute v2 = lane * row_stride_bytes
            if (row_stride_bytes & (row_stride_bytes - 1)) == 0:
                shift_amount = row_stride_bytes.bit_length() - 1
                self.emit_instruction(VLshlRevB32(2, shift_amount, 0))
            else:
                self.emit_instruction(VMovB32(1, row_stride_bytes))
                self.register_file.v_used.add(1)
                self.emit_instruction(VMulLoU32(2, 0, 1))
            self.register_file.v_used.add(2)
            return "v2", instruction_offset_bytes

        # If row index is an integer constant, fold entirely into instoffset (no offen)
        if isinstance(row_value, int):
            instruction_offset_bytes += row_value * row_stride_bytes
            # We'll still return v2 as 0 for uniformity; just set it once
            self.emit_instruction(VMovB32(2, 0))
            self.register_file.v_used.add(2)
            return "v2", instruction_offset_bytes

        # If we can't determine the row index value, raise an error
        raise ValueError(
            f"Cannot determine row index value for {row_index_ssa}. Expected 'tid.x' or integer constant, got: {row_value}"
        )

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
        vector_index_register: str,
        instruction_offset_base: int,
    ):
        srd_register_0, srd_register_1, srd_register_2, srd_register_3 = self.srds[
            memref_ssa
        ]
        register_list = []
        self.emit(f"    # load {vector_bytes}B from {memref_ssa}")
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
            self.emit_instruction(SWaitcnt("vmcnt(0)"))
        return register_list

    def emit_store_with_regs(
        self,
        memref_ssa: str,
        register_list: List[Tuple[int, int, int, int]],
        vector_bytes: int,
        vector_index_register: str,
        instruction_offset_base: int,
    ):
        srd_register_0, srd_register_1, srd_register_2, srd_register_3 = self.srds[
            memref_ssa
        ]
        self.emit(f"    # store {vector_bytes}B to {memref_ssa}")
        for vector_quad, offset in zip(register_list, self.chunk_offsets(vector_bytes)):
            total_offset = instruction_offset_base + offset
            store_instruction = BufferStoreDwordx4(
                vector_quad,
                vector_index_register,
                (srd_register_0, srd_register_1, srd_register_2, srd_register_3),
                total_offset,
            )
            self.emit_instruction(store_instruction)
