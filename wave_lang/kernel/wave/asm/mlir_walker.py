# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from wave_lang.support.ir_imports import (
    affine_d,
    amdgpu_d,
    arith_d,
    func_d,
    gpu_d,
    memref_d,
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
        """Walk operations in a block and dispatch to appropriate handlers."""
        for operation in block.operations:
            if isinstance(operation, arith_d.ConstantOp):
                self.handlers.handle_arith_constant_op(operation, kernel_info)
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
            elif isinstance(operation, stream_d.BindingSubspanOp):
                self.handlers.handle_stream_binding_subspan_op(operation, kernel_info)
            elif isinstance(operation, scf_d.ForOp):
                self.handlers.handle_scf_for_op(operation, kernel_info)
            elif isinstance(operation, vector_d.ExtractStridedSliceOp):
                self.handlers.handle_vector_extract_strided_slice_op(
                    operation, kernel_info
                )
            else:
                pass
