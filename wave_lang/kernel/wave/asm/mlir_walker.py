# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from wave_lang.support.ir_imports import (
    func_d,
    affine_d,
    gpu_d,
    arith_d,
    vector_d,
    stream_d,
)

from .utils import (
    parse_vector_type_from_obj,
    parse_memref_type_from_obj,
    attrs_to_dict,
    parse_wg_and_subgroup,
    tid_upper_bound_from_thread_id,
    simplify_expression,
)

from .kernel_model import KernelInfo, MemRefInfo, BindingUse, VecAccess


class IRWalker:
    def __init__(self, emitter=None):
        """Initialize IRWalker with optional emitter for single-pass traversal."""
        self.emitter = emitter
        self.last_regs = None
        self.last_vec_bytes = 0

    def interpret_func(self, fn: func_d.FuncOp) -> KernelInfo:
        kernel_info = KernelInfo(name=fn.sym_name.value)

        entry_block = fn.entry_block
        kernel_info.arg_ssa_order = [str(arg) for arg in entry_block.arguments]

        # Extract translation_info from function attributes
        function_attributes = attrs_to_dict(fn.attributes)
        translation_info = function_attributes.get("translation_info")
        if translation_info is not None:
            workgroup_size, subgroup_size = parse_wg_and_subgroup(translation_info)
            if workgroup_size:
                kernel_info.wg_size = workgroup_size
            if subgroup_size:
                kernel_info.subgroup_size = subgroup_size

        # Walk operations and fill environment + accesses
        for block in fn.body.blocks:
            for operation in block.operations:
                if isinstance(operation, arith_d.ConstantOp):
                    self._handle_arith_constant_op(operation, kernel_info)
                elif isinstance(operation, gpu_d.ThreadIdOp):
                    self._handle_gpu_thread_id_op(operation, kernel_info)
                elif isinstance(operation, affine_d.AffineApplyOp):
                    self._handle_affine_apply_op(operation, kernel_info)
                elif isinstance(operation, vector_d.LoadOp):
                    self._handle_vector_load_op(operation, kernel_info)
                elif isinstance(operation, vector_d.StoreOp):
                    self._handle_vector_store_op(operation, kernel_info)
                elif isinstance(operation, stream_d.BindingSubspanOp):
                    self._handle_stream_binding_subspan_op(operation, kernel_info)

        return kernel_info

    def _handle_arith_constant_op(
        self, operation: arith_d.ConstantOp, kernel_info: KernelInfo
    ):
        """Handle arith.constant operations - extract constant values for index environment."""
        # Use MLIR Python bindings to get the constant value
        try:
            # Get the value attribute from the operation
            value_attribute = operation.value
            if hasattr(value_attribute, "value"):
                constant_value = int(value_attribute.value)
                kernel_info.index_env[str(operation.result)] = constant_value
        except (AttributeError, ValueError, TypeError):
            # If we can't extract the value, skip this constant
            pass

    def _handle_gpu_thread_id_op(
        self, operation: gpu_d.ThreadIdOp, kernel_info: KernelInfo
    ):
        """Handle gpu.thread_id operations - extract thread ID information."""
        upper_bound = tid_upper_bound_from_thread_id(operation)
        # Get the actual dimension from the operation
        dimension = operation.dimension
        # Extract dimension from MLIR attribute string like "#gpu<dim x>"
        dimension_string = str(dimension)
        if "dim x" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "tid.x"
            if upper_bound is not None:
                kernel_info.tid_ub_x = upper_bound
        elif "dim y" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "tid.y"
            if upper_bound is not None:
                kernel_info.tid_ub_y = upper_bound
        elif "dim z" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "tid.z"
            if upper_bound is not None:
                kernel_info.tid_ub_z = upper_bound

    def _extract_dimension_values(
        self,
        operation: affine_d.AffineApplyOp,
        kernel_info: KernelInfo,
        num_dimensions: int,
    ) -> list:
        """Extract dimension values from the first num_dimensions operands."""
        dimension_values = []

        for i in range(num_dimensions):
            if i < len(operation.operands):
                operand_ssa = str(operation.operands[i])
                operand_value = kernel_info.index_env.get(operand_ssa)

                if isinstance(operand_value, int):
                    dimension_values.append(operand_value)
                elif operand_value in ["tid.x", "tid.y", "tid.z"]:
                    # Thread IDs can be represented as symbols in the expression
                    dimension_values.append(operand_value)
                else:
                    # If we can't resolve the dimension value, we can't simplify
                    return None
            else:
                # Not enough operands for the expected number of dimensions
                return None

        return dimension_values

    def _extract_symbol_values(
        self,
        operation: affine_d.AffineApplyOp,
        kernel_info: KernelInfo,
        num_dimensions: int,
        num_symbols: int,
    ) -> list:
        """Extract symbol values from the next num_symbols operands."""
        symbol_values = []

        for i in range(num_symbols):
            operand_index = num_dimensions + i
            if operand_index < len(operation.operands):
                operand_ssa = str(operation.operands[operand_index])
                operand_value = kernel_info.index_env.get(operand_ssa)

                if isinstance(operand_value, int):
                    symbol_values.append(operand_value)
                elif operand_value in ["tid.x", "tid.y", "tid.z"]:
                    # Thread IDs can be used as symbol values
                    symbol_values.append(operand_value)
                else:
                    # If we can't resolve the symbol value, we can't simplify
                    return None
            else:
                # Not enough operands for the expected number of symbols
                return None

        return symbol_values

    def _handle_affine_apply_op(
        self, operation: affine_d.AffineApplyOp, kernel_info: KernelInfo
    ):
        """Handle affine.apply operations - simplify affine expressions."""
        # The first operands correspond to dimensions, the rest to symbols
        affine_map_attribute = operation.map
        affine_map = affine_map_attribute.value
        num_dimensions = affine_map.n_dims
        num_symbols = affine_map.n_symbols

        # Extract dimension and symbol values from operands
        dimension_values = self._extract_dimension_values(
            operation, kernel_info, num_dimensions
        )
        symbol_values = self._extract_symbol_values(
            operation, kernel_info, num_dimensions, num_symbols
        )

        # Try to simplify the expression with actual values
        simplified_expression = simplify_expression(
            operation.map, kernel_info.tid_ub_x, dimension_values, symbol_values
        )

        destination_ssa = str(operation.result)
        if simplified_expression is not None:
            # Check if the simplified expression is a constant
            if len(simplified_expression.free_symbols) == 0:
                try:
                    constant_value = int(simplified_expression)
                    kernel_info.index_env[destination_ssa] = constant_value
                except:
                    pass
            # Check if the simplified expression is a thread ID symbol
            else:
                import sympy as sp

                # Check for all thread ID types
                thread_id_symbols = {
                    "tid_x": sp.Symbol("tid_x", nonnegative=True),
                    "tid_y": sp.Symbol("tid_y", nonnegative=True),
                    "tid_z": sp.Symbol("tid_z", nonnegative=True),
                }

                for thread_id_name, thread_id_symbol in thread_id_symbols.items():
                    if simplified_expression == thread_id_symbol:
                        # Map back to the original thread ID format
                        original_thread_id = thread_id_name.replace("_", ".")
                        source_ssa = (
                            str(operation.operands[0])
                            if len(operation.operands) > 0
                            else None
                        )
                        if (
                            source_ssa
                            and kernel_info.index_env.get(source_ssa)
                            == original_thread_id
                        ):
                            kernel_info.index_env[destination_ssa] = original_thread_id
                        break

    def _handle_vector_load_op(
        self, operation: vector_d.LoadOp, kernel_info: KernelInfo
    ):
        """Handle vector.load operations - track memory accesses and emit load instructions."""
        memref_ssa = str(operation.operands[0])  # memref is first operand
        num_elements, element_bytes, _ = parse_vector_type_from_obj(
            operation.results[0].type
        )
        indices = [
            str(operation.operands[i]) for i in range(1, len(operation.operands))
        ]

        # The memref should already be in kernel_info.subspans from a stream.binding.subspan operation
        if memref_ssa not in kernel_info.subspans:
            raise ValueError(
                f"Memref {memref_ssa} used in vector.load but not found in subspans. Missing stream.binding.subspan operation."
            )

        # Update memref info if not already set
        binding_use = kernel_info.subspans[memref_ssa]
        if not binding_use.memref_info:
            try:
                memref_type_object = operation.operands[0].type
                shape, strides, element_bytes = parse_memref_type_from_obj(
                    memref_type_object
                )
                binding_use.memref_info = MemRefInfo(shape, strides, element_bytes)
            except Exception as e:
                raise ValueError(f"Cannot parse memref type for load operation: {e}")

        kernel_info.accesses.append(
            VecAccess("load", memref_ssa, num_elements, element_bytes, indices)
        )

        # Emit load instruction
        self._emit_load_instruction(operation, kernel_info, memref_ssa, indices)

    def _handle_vector_store_op(
        self, operation: vector_d.StoreOp, kernel_info: KernelInfo
    ):
        """Handle vector.store operations - track memory accesses and emit store instructions."""
        memref_ssa = str(
            operation.operands[1]
        )  # memref is second operand (after value)
        num_elements, element_bytes, _ = parse_vector_type_from_obj(
            operation.operands[0].type
        )  # value is first operand
        indices = [
            str(operation.operands[i]) for i in range(2, len(operation.operands))
        ]

        # The memref should already be in kernel_info.subspans from a stream.binding.subspan operation
        if memref_ssa not in kernel_info.subspans:
            raise ValueError(
                f"Memref {memref_ssa} used in vector.store but not found in subspans. Missing stream.binding.subspan operation."
            )

        # Update memref info if not already set
        binding_use = kernel_info.subspans[memref_ssa]
        if not binding_use.memref_info:
            try:
                memref_type_object = operation.operands[1].type
                shape, strides, element_bytes = parse_memref_type_from_obj(
                    memref_type_object
                )
                binding_use.memref_info = MemRefInfo(shape, strides, element_bytes)
            except Exception as e:
                raise ValueError(f"Cannot parse memref type for store operation: {e}")

        kernel_info.accesses.append(
            VecAccess("store", memref_ssa, num_elements, element_bytes, indices)
        )

        # Emit store instruction
        self._emit_store_instruction(operation, kernel_info, memref_ssa, indices)

    def _handle_stream_binding_subspan_op(
        self, operation: stream_d.BindingSubspanOp, kernel_info: KernelInfo
    ):
        """Handle stream.binding.subspan operations - map memrefs to function arguments."""
        # map memref SSA -> which function arg index it came from
        source_ssa = str(operation.operands[0])  # function arg SSA
        result_ssa = str(operation.results[0])  # memref SSA
        argument_index = kernel_info.arg_ssa_order.index(source_ssa)
        binding_use = kernel_info.subspans.setdefault(
            result_ssa, BindingUse(result_ssa, argument_index)
        )

        # Try to extract memref information from the result type
        try:
            memref_type_object = operation.results[0].type
            shape, strides, element_bytes = parse_memref_type_from_obj(
                memref_type_object
            )
            binding_use.memref_info = MemRefInfo(shape, strides, element_bytes)
        except Exception:
            # If we can't parse the memref type, we'll need to get it from load/store operations
            pass

        # Emit SRD setup
        self._emit_srd_setup(operation, kernel_info, result_ssa, argument_index)

    def _emit_srd_setup(self, operation, kernel_info, memref_ssa, argument_index):
        """Emit SRD setup for a binding subspan operation."""
        # Compute limit = wg_size_x * (stride0 * elem_bytes)
        binding_use = kernel_info.subspans.get(memref_ssa)
        if not binding_use or not binding_use.memref_info:
            raise ValueError(
                f"Cannot determine memref information for {memref_ssa}. SRD setup requires memref stride and element size."
            )
        row_stride_bytes = (
            binding_use.memref_info.strides_elems[0]
            * binding_use.memref_info.elem_bytes
        )
        limit_bytes = kernel_info.wg_size[0] * row_stride_bytes
        self.emitter.ensure_srd_for_subspan(memref_ssa, argument_index, limit_bytes)

    def _emit_load_instruction(self, operation, kernel_info, memref_ssa, indices):
        """Emit load instruction for a vector.load operation."""
        # Ensure SRD exists
        if memref_ssa not in kernel_info.subspans:
            raise ValueError(
                f"Memref {memref_ssa} not found in subspans. This indicates a missing stream.binding.subspan operation."
            )
        binding_use = kernel_info.subspans[memref_ssa]
        if memref_ssa not in self.emitter.srds:
            if not binding_use.memref_info:
                raise ValueError(
                    f"Cannot determine memref information for {memref_ssa}. SRD setup requires memref stride and element size."
                )
            row_stride_bytes = (
                binding_use.memref_info.strides_elems[0]
                * binding_use.memref_info.elem_bytes
            )
            limit_bytes = kernel_info.wg_size[0] * row_stride_bytes
            self.emitter.ensure_srd_for_subspan(
                memref_ssa,
                binding_use.arg_index if binding_use.arg_index >= 0 else 0,
                limit_bytes,
            )

        # Get memref info
        memref_type_object = operation.operands[0].type
        try:
            shape, strides, element_bytes = parse_memref_type_from_obj(
                memref_type_object
            )
            memref_info = MemRefInfo(shape, strides, element_bytes)
        except Exception as e:
            raise ValueError(f"Cannot parse memref type for load operation: {e}")

        # Compute voffset and instoffset
        vector_register, instruction_offset = (
            self.emitter.compute_voffset_and_instoffset(
                kernel_info, memref_info, indices
            )
        )

        # Get vector bytes
        vector_bytes = self.last_vec_bytes if self.last_vec_bytes else 32

        # Emit load instruction
        self.last_regs = self.emitter.emit_load(
            memref_ssa, vector_bytes, vector_register, instruction_offset
        )
        self.last_vec_bytes = vector_bytes

    def _emit_store_instruction(self, operation, kernel_info, memref_ssa, indices):
        """Emit store instruction for a vector.store operation."""
        # Ensure SRD exists
        if memref_ssa not in kernel_info.subspans:
            raise ValueError(
                f"Memref {memref_ssa} not found in subspans. This indicates a missing stream.binding.subspan operation."
            )
        binding_use = kernel_info.subspans[memref_ssa]
        if memref_ssa not in self.emitter.srds:
            if not binding_use.memref_info:
                raise ValueError(
                    f"Cannot determine memref information for {memref_ssa}. SRD setup requires memref stride and element size."
                )
            row_stride_bytes = (
                binding_use.memref_info.strides_elems[0]
                * binding_use.memref_info.elem_bytes
            )
            limit_bytes = kernel_info.wg_size[0] * row_stride_bytes
            self.emitter.ensure_srd_for_subspan(
                memref_ssa,
                binding_use.arg_index if binding_use.arg_index >= 0 else 1,
                limit_bytes,
            )

        # Get memref info
        memref_type_object = operation.operands[1].type
        try:
            shape, strides, element_bytes = parse_memref_type_from_obj(
                memref_type_object
            )
            memref_info = MemRefInfo(shape, strides, element_bytes)
        except Exception as e:
            raise ValueError(f"Cannot parse memref type for store operation: {e}")

        # Compute voffset and instoffset
        vector_register, instruction_offset = (
            self.emitter.compute_voffset_and_instoffset(
                kernel_info, memref_info, indices
            )
        )

        # Get vector bytes
        vector_bytes = self.last_vec_bytes if self.last_vec_bytes else 32

        # Emit store instruction
        if not self.last_regs:
            raise SystemExit(
                "Store encountered without a preceding load to provide data regs."
            )
        self.emitter.emit_store_with_regs(
            memref_ssa,
            self.last_regs,
            vector_bytes,
            vector_register,
            instruction_offset,
        )
