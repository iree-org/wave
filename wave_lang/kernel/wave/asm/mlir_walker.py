# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
MLIR IR Walker for ASM code generation.

Implements LLVM-style vmcnt optimization by pre-computing M0 base values
BEFORE barriers, then using scalar operations for M0 assignment.
"""

from wave_lang.support.ir_imports import (
    affine_d,
    amdgpu_d,
    arith_d,
    func_d,
    gpu_d,
    memref_d,
    rocdl_d,
    scf_d,
    stream_d,
    vector_d,
    OpAttributeMap,
)

from .utils import (
    parse_wg_and_subgroup,
)

from .kernel_model import KernelInfo
from .handlers import OperationHandlers
from .gather_to_shared import analyze_g2s_region, precompute_m0_values


class IRWalker:
    def __init__(self, emitter=None):
        """Initialize IRWalker with optional emitter."""
        self.emitter = emitter

        # Unified SSA-to-VGPR mapping for all vector operations
        # Maps SSA value strings to register allocations (tuples of register indices)
        self.ssa_to_vgpr: dict[str, tuple[int, ...]] = {}

        # Supporting fields
        self.last_vmem_ticket = None  # Used for wait count computation
        self._lds_view_base_bytes: dict[str, int] = {}  # LDS view offsets
        # Initialize operation handlers
        self.handlers = OperationHandlers(self)

    def interpret_func(self, fn: func_d.FuncOp) -> KernelInfo:
        kernel_info = KernelInfo(name=fn.sym_name.value)

        entry_block = fn.entry_block
        kernel_info.arg_ssa_order = [str(arg) for arg in entry_block.arguments]

        # Extract translation_info from function attributes
        assert isinstance(fn.attributes, OpAttributeMap)
        function_attributes = dict(fn.attributes)
        translation_info = function_attributes.get("translation_info")
        if translation_info is not None:
            workgroup_size, subgroup_size = parse_wg_and_subgroup(translation_info)
            if workgroup_size:
                kernel_info.wg_size = workgroup_size
            if subgroup_size:
                kernel_info.subgroup_size = subgroup_size

        # Walk operations and fill environment + accesses
        self._walk_block(fn.entry_block, kernel_info)

        return kernel_info

    def _walk_block(self, block, kernel_info: KernelInfo):
        """Walk operations in a block and dispatch to handlers.

        For g2s regions: dispatches ops before first g2s, pre-computes M0,
        then dispatches remaining ops.
        """
        ops = list(block.operations)

        # Analyze for g2s scheduling
        schedule = analyze_g2s_region(ops)

        if schedule is None:
            # No g2s ops - dispatch all ops sequentially
            for op in ops:
                self._dispatch_operation(op, kernel_info)
        else:
            # Dispatch ops before first g2s (populates index_env)
            for i in range(schedule.first_g2s_idx):
                self._dispatch_operation(ops[i], kernel_info)

            # Pre-compute M0 values before any buffer_load...lds
            precompute_m0_values(schedule, kernel_info, self.handlers)

            # Dispatch remaining ops
            for i in range(schedule.first_g2s_idx, len(ops)):
                self._dispatch_operation(ops[i], kernel_info)

    def _dispatch_operation(self, operation, kernel_info: KernelInfo):
        """Dispatch a single operation to its appropriate handler."""
        if isinstance(operation, arith_d.ConstantOp):
            self.handlers.handle_arith_constant_op(operation, kernel_info)
        elif isinstance(operation, arith_d.AddIOp):
            self.handlers.handle_arith_addi_op(operation, kernel_info)
        elif isinstance(operation, arith_d.MulIOp):
            self.handlers.handle_arith_muli_op(operation, kernel_info)
        elif isinstance(operation, arith_d.IndexCastOp):
            self.handlers.handle_arith_index_cast_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.ThreadIdOp):
            self.handlers.handle_gpu_thread_id_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.BlockIdOp):
            self.handlers.handle_gpu_block_id_op(operation, kernel_info)
        elif isinstance(operation, affine_d.AffineApplyOp):
            self.handlers.handle_affine_apply_op(operation, kernel_info)
        elif isinstance(operation, vector_d.LoadOp):
            self.handlers.handle_vector_load_op(operation, kernel_info)
        elif isinstance(operation, vector_d.StoreOp):
            self.handlers.handle_vector_store_op(operation, kernel_info)
        elif isinstance(operation, amdgpu_d.MFMAOp):
            self.handlers.handle_mfma_op(operation, kernel_info)
        elif isinstance(operation, amdgpu_d.LDSBarrierOp):
            self.handlers.handle_lds_barrier_op(operation, kernel_info)
        elif isinstance(operation, memref_d.ViewOp):
            self.handlers.handle_view_op(operation, kernel_info)
        elif isinstance(operation, memref_d.AllocOp):
            self.handlers.handle_alloc_op(operation, kernel_info)
        elif isinstance(operation, memref_d.CastOp):
            self.handlers.handle_memref_cast_op(operation, kernel_info)
        elif isinstance(operation, memref_d.ReinterpretCastOp):
            self.handlers.handle_memref_reinterpret_cast_op(operation, kernel_info)
        elif isinstance(operation, stream_d.BindingSubspanOp):
            self.handlers.handle_stream_binding_subspan_op(operation, kernel_info)
        elif isinstance(operation, scf_d.ForOp):
            self.handlers.handle_scf_for_op(operation, kernel_info)
        elif isinstance(operation, vector_d.ExtractStridedSliceOp):
            self.handlers.handle_vector_extract_strided_slice_op(operation, kernel_info)
        # Critical operations for gather_to_lds support
        elif isinstance(operation, amdgpu_d.GatherToLDSOp):
            self.handlers.g2s.handle_gather_to_lds_op(operation, kernel_info)
        elif isinstance(operation, amdgpu_d.FatRawBufferCastOp):
            self.handlers.handle_fat_raw_buffer_cast_op(operation, kernel_info)
        elif isinstance(operation, rocdl_d.ReadfirstlaneOp):
            self.handlers.handle_readfirstlane_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.SubgroupBroadcastOp):
            self.handlers.handle_subgroup_broadcast_op(operation, kernel_info)
        elif isinstance(operation, rocdl_d.SWaitcntOp):
            self.handlers.handle_s_waitcnt_op(operation, kernel_info)
        else:
            pass
