# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Operation handlers for MLIR operations in the ASM backend.

This module contains handlers for various MLIR operations that are encountered
during the IR traversal for assembly code generation.
"""

from wave_lang.support.ir_imports import (
    affine_d,
    amdgpu_d,
    arith_d,
    gpu_d,
    memref_d,
    stream_d,
    vector_d,
)

from .utils import (
    parse_vector_type_from_obj,
    parse_memref_type_from_obj,
    tid_upper_bound_from_thread_id,
    simplify_expression,
    split_const_dynamic,
)
from .expression_emitter import ExprEmitter

from .kernel_model import KernelInfo, MemRefInfo, BindingUse, VecAccess


class OperationHandlers:
    """
    Handles MLIR operations for the ASM backend.

    This class contains all the operation-specific handlers that process
    MLIR operations and emit corresponding assembly code.
    """

    def __init__(self, walker):
        """
        Initialize the handlers with a reference to the walker.

        Args:
            walker: The IRWalker instance that owns this handler
        """
        self.walker = walker
        # Per-kernel expression emitters with CSE (one per kernel)
        self._expr_emitters_by_kernel = {}

    def _get_expr_emitter(self, kernel_info: KernelInfo) -> ExprEmitter:
        """Get or create expression emitter for this kernel (with CSE)."""
        key = id(kernel_info)
        if key not in self._expr_emitters_by_kernel:
            self._expr_emitters_by_kernel[key] = ExprEmitter(
                self.walker.emitter, kernel_info
            )
        return self._expr_emitters_by_kernel[key]

    def handle_arith_constant_op(
        self, operation: arith_d.ConstantOp, kernel_info: KernelInfo
    ):
        """Handle arith.constant operations - extract constant values for index environment."""
        # Extract integer constants for the index environment
        # Skip non-integer constants (floats, vectors, etc.) without error
        value_attribute = operation.value
        if not hasattr(value_attribute, "value"):
            return

        value = value_attribute.value
        # Only store if it's an integer or can be safely converted to one
        if isinstance(value, int):
            kernel_info.index_env[str(operation.result)] = value
        elif (
            hasattr(value, "is_integer")
            and callable(value.is_integer)
            and value.is_integer()
        ):
            # Handle float-like types that represent exact integers
            kernel_info.index_env[str(operation.result)] = int(value)

    def handle_gpu_thread_id_op(
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

    def handle_affine_apply_op(
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
                # Expression has no free symbols, so it's a constant - convert to int
                assert hasattr(simplified_expression, "__int__") or hasattr(
                    simplified_expression, "is_integer"
                ), f"Simplified expression without free symbols should be convertible to int: {simplified_expression}"
                constant_value = int(simplified_expression)
                kernel_info.index_env[destination_ssa] = constant_value
            # Check if the simplified expression is a thread ID symbol
            else:
                import sympy

                # Check for all thread ID types
                thread_id_symbols = {
                    "tid_x": sympy.Symbol("tid_x", nonnegative=True),
                    "tid_y": sympy.Symbol("tid_y", nonnegative=True),
                    "tid_z": sympy.Symbol("tid_z", nonnegative=True),
                }

                matched_tid = False
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
                            matched_tid = True
                        break

                if not matched_tid:
                    # Store the simplified SymPy expression for later ASM emission
                    kernel_info.index_env[destination_ssa] = simplified_expression

    def handle_vector_load_op(
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

        # If memref is not in subspans, it may be LDS (workgroup) memory; handle later in emit

        # Update memref info if not already set
        if memref_ssa in kernel_info.subspans:
            binding_use = kernel_info.subspans[memref_ssa]
            if not binding_use.memref_info:
                try:
                    memref_type_object = operation.operands[0].type
                    shape, strides, element_bytes = parse_memref_type_from_obj(
                        memref_type_object
                    )
                    binding_use.memref_info = MemRefInfo(shape, strides, element_bytes)
                except Exception as e:
                    raise ValueError(
                        f"Cannot parse memref type for load operation: {e}"
                    )

        kernel_info.accesses.append(
            VecAccess("load", memref_ssa, num_elements, element_bytes, indices)
        )

        # Emit load instruction
        self._emit_load_instruction(operation, kernel_info, memref_ssa, indices)

    def handle_vector_store_op(
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

        # If memref is not in subspans, it may be LDS (workgroup) memory; handle later in emit

        # Update memref info if not already set
        if memref_ssa in kernel_info.subspans:
            binding_use = kernel_info.subspans[memref_ssa]
            if not binding_use.memref_info:
                try:
                    memref_type_object = operation.operands[1].type
                    shape, strides, element_bytes = parse_memref_type_from_obj(
                        memref_type_object
                    )
                    binding_use.memref_info = MemRefInfo(shape, strides, element_bytes)
                except Exception as e:
                    raise ValueError(
                        f"Cannot parse memref type for store operation: {e}"
                    )

        kernel_info.accesses.append(
            VecAccess("store", memref_ssa, num_elements, element_bytes, indices)
        )

        # Emit store instruction
        self._emit_store_instruction(operation, kernel_info, memref_ssa, indices)

    def handle_stream_binding_subspan_op(
        self, operation: stream_d.BindingSubspanOp, kernel_info: KernelInfo
    ):
        """Handle stream.binding.subspan operations - map memrefs to function arguments."""

        # Subspan is immediately consumed by a reinterpret cast
        users = list(operation.result.uses)
        assert (
            len(users) == 1
        ), f"Expected 1 user for stream.binding.subspan operation, got {users}"
        reinterpret = users[0].owner.operation.opview
        assert isinstance(
            reinterpret, memref_d.ReinterpretCastOp
        ), f"Expected memref.reinterpret_cast operation, got {reinterpret}"

        # map memref SSA -> which function arg index it came from
        source_ssa = str(operation.operands[0])  # function arg SSA
        result_ssa = str(reinterpret.results[0])  # memref SSA
        argument_index = kernel_info.arg_ssa_order.index(source_ssa)
        binding_use = kernel_info.subspans.setdefault(
            result_ssa, BindingUse(result_ssa, argument_index)
        )

        # Extract memref information from the result type
        # This must succeed for SRD setup to work
        memref_type_object = reinterpret.results[0].type
        shape, strides, element_bytes = parse_memref_type_from_obj(memref_type_object)
        binding_use.memref_info = MemRefInfo(shape, strides, element_bytes)

        # Emit SRD setup
        self._emit_srd_setup(operation, kernel_info, result_ssa, argument_index)

    def _compute_buffer_size(self, memref_info):
        """Compute buffer size in bytes from memref shape and element size."""
        if not memref_info.shape:
            # Scalar or unranked: use single element
            return memref_info.elem_bytes
        else:
            # Compute total buffer size: product of all dimensions * element size
            total_elements = 1
            for dim in memref_info.shape:
                total_elements *= dim
            return total_elements * memref_info.elem_bytes

    def _emit_srd_setup(self, operation, kernel_info, memref_ssa, argument_index):
        """Emit SRD setup for a binding subspan operation."""
        binding_use = kernel_info.subspans.get(memref_ssa)
        if not binding_use or not binding_use.memref_info:
            raise ValueError(
                f"Cannot determine memref information for {memref_ssa}. "
                f"SRD setup requires memref shape and element size."
            )

        limit_bytes = self._compute_buffer_size(binding_use.memref_info)
        self.walker.emitter.ensure_srd_for_subspan(
            memref_ssa, argument_index, limit_bytes
        )

    def handle_mfma_op(self, operation: amdgpu_d.MFMAOp, kernel_info: KernelInfo):
        """Handle amdgpu.mfma operations - emit MFMA instruction with proper input sourcing."""
        # Skip if MFMA has already been emitted
        if self.walker._mfma_emitted:
            return

        # Get the operand SSA values from the MFMA operation
        # MFMA format: %result = amdgpu.mfma %lhs * %rhs + %acc
        if len(operation.operands) >= 2:
            lhs_ssa = str(operation.operands[0])  # First operand (LHS of multiply)
            rhs_ssa = str(operation.operands[1])  # Second operand (RHS of multiply)

            # Map the SSA values to their corresponding LDS-loaded register pairs
            lhs_pair = self.walker._lds_last_pair.get(lhs_ssa)
            rhs_pair = self.walker._lds_last_pair.get(rhs_ssa)

            if lhs_pair and rhs_pair:
                self.walker.emitter.emit_mfma_16x16x16_f16(lhs_pair, rhs_pair)
                spilled_quad = self.walker.emitter.spill_agprs_to_vgprs()

                # Free input pairs
                self.walker.emitter.vgpr_allocator.free_v_pair(lhs_pair)
                self.walker.emitter.vgpr_allocator.free_v_pair(rhs_pair)

                self.walker.last_regs = list(spilled_quad)
                self.walker.last_vmem_ticket = None  # Data now from MFMA, not VMEM
                self.walker._mfma_emitted = True
                return

        # If we reach here, MFMA inputs weren't properly set up
        raise RuntimeError(
            f"MFMA operation encountered but inputs not available. "
            f"Operands: {[str(op) for op in operation.operands]}, "
            f"LDS pairs: {list(self.walker._lds_last_pair.keys())}"
        )

    def handle_barrier_op(self, operation: gpu_d.BarrierOp, kernel_info: KernelInfo):
        """Handle gpu.barrier operations - emit synchronization barrier."""
        self.walker.emitter.emit_barrier()

    def handle_lds_barrier_op(
        self, operation: amdgpu_d.LDSBarrierOp, kernel_info: KernelInfo
    ):
        """Handle amdgpu.lds_barrier operations - emit LDS synchronization barrier."""
        self.walker.emitter.emit_barrier()

    def handle_view_op(self, operation: memref_d.ViewOp, kernel_info: KernelInfo):
        """Handle memref.view operations - capture view base byte offset for LDS-backed memrefs."""
        result_ssa = str(operation.results[0])
        # The offset operand is already in bytes (index into xi8 buffer)
        # Only capture if the offset is a known integer constant in index_env
        base_bytes = None
        for operand in operation.operands:
            key = str(operand)
            if key in kernel_info.index_env and isinstance(
                kernel_info.index_env[key], int
            ):
                base_bytes = kernel_info.index_env[key]
                break
        if base_bytes is not None:
            self.walker._lds_view_base_bytes[result_ssa] = int(base_bytes)

    def handle_alloc_op(self, operation: memref_d.AllocOp, kernel_info: KernelInfo):
        """Handle memref.alloc operations - capture LDS allocation size."""
        # Parse the memref type to get shape and element size
        shape, strides, elem_bytes = parse_memref_type_from_obj(
            operation.results[0].type
        )

        # Compute total LDS allocation size
        if shape:
            total_elements = 1
            for dim in shape:
                total_elements *= dim
            alloc_size_bytes = total_elements * elem_bytes

            # Track the maximum LDS size (in case of multiple allocations)
            kernel_info.lds_size_bytes = max(
                kernel_info.lds_size_bytes, alloc_size_bytes
            )

    def _parse_load_memref_info(self, operation):
        """Parse memref information from a vector.load operation."""
        memref_type_object = operation.operands[0].type
        try:
            shape, strides, element_bytes = parse_memref_type_from_obj(
                memref_type_object
            )
            return MemRefInfo(shape, strides, element_bytes)
        except Exception as e:
            raise ValueError(f"Cannot parse memref type for load operation: {e}")

    def _emit_lds_load(self, operation, kernel_info, memref_ssa, byte_offset_expr):
        """Emit an LDS load operation using MLIR's 2D memref indices.

        Uses the byte_offset_expr computed from MLIR's actual indices rather than
        forcing lane-linear addressing.
        """

        # If LDS address wasn't established by a prior store, compute it from the passed-in expression
        if memref_ssa not in self.walker._lds_addr_vreg:
            # Add view base offset if present
            vbase_val = self.walker._lds_view_base_bytes.get(memref_ssa, 0)
            if vbase_val:
                byte_offset_expr = byte_offset_expr + sympy.Integer(vbase_val)

            # Materialize the byte offset expression into a VGPR for LDS addressing (with CSE)
            addr_reg = self._get_expr_emitter(kernel_info).get_or_emit(byte_offset_expr)
            addr_v = int(addr_reg[1:])
            self.walker._lds_addr_vreg[memref_ssa] = addr_v

        # Emit LDS read into allocated VGPR pairs
        addr_v = self.walker._lds_addr_vreg[memref_ssa]
        pair = self.walker.emitter.vgpr_allocator.alloc_v_pair()
        self.walker.emitter.emit_lds_read_b64(pair, addr_v)

        # Track the loaded data by result SSA value (for MFMA operand matching)
        result_ssa = str(operation.results[0])
        self.walker._lds_last_pair[result_ssa] = pair
        self.walker._lds_read_order.append(result_ssa)
        self.walker.last_regs = [pair]

    def _ensure_global_load_srd(self, kernel_info, memref_ssa):
        """Ensure SRD is set up for a global load."""
        if memref_ssa in self.walker.emitter.srds:
            return

        binding_use = kernel_info.subspans[memref_ssa]
        if not binding_use.memref_info:
            raise ValueError(
                f"Cannot determine memref information for {memref_ssa}. "
                f"SRD setup requires memref shape and element size."
            )

        limit_bytes = self._compute_buffer_size(binding_use.memref_info)
        arg_idx = binding_use.arg_index if binding_use.arg_index >= 0 else 0
        self.walker.emitter.ensure_srd_for_subspan(memref_ssa, arg_idx, limit_bytes)

    def _parse_vector_load_type(self, operation):
        """Parse vector type from load operation result."""
        try:
            num_elements, element_bytes, _ = parse_vector_type_from_obj(
                operation.results[0].type
            )
            return num_elements * element_bytes
        except Exception as e:
            raise ValueError(f"Cannot parse vector type for global load: {e}")

    def _emit_buffer_load_and_track(
        self, kernel_info, memref_ssa, vector_bytes, voffset_v, instoffset
    ):
        """Emit buffer load instruction and track loaded registers and ticket."""
        self.walker.last_regs, self.walker.last_vmem_ticket = (
            self.walker.emitter.emit_load(
                memref_ssa, vector_bytes, voffset_v, instoffset
            )
        )
        self.walker.last_vec_bytes = vector_bytes

    def _emit_global_load(self, operation, kernel_info, memref_ssa, byte_offset_expr):
        """Emit a global buffer load operation."""
        self._ensure_global_load_srd(kernel_info, memref_ssa)

        # Split constant/dynamic and materialize dynamic part via cached emitter (CSE)
        const_offset, dynamic_expr = split_const_dynamic(byte_offset_expr)
        if dynamic_expr == 0 or (
            hasattr(dynamic_expr, "is_zero") and dynamic_expr.is_zero
        ):
            # No dynamic part: use a zero VGPR
            voffset_v, instoffset = (
                self.walker.emitter.vgpr_allocator.alloc_v(),
                const_offset,
            )
            from .instructions import VMovB32

            self.walker.emitter.emit_instruction(VMovB32(voffset_v, 0))
            voffset_is_temp = True
        else:
            voffset_reg = self._get_expr_emitter(kernel_info).get_or_emit(dynamic_expr)
            voffset_v, instoffset = int(voffset_reg[1:]), const_offset
            voffset_is_temp = False

        vector_bytes = self._parse_vector_load_type(operation)

        self._emit_buffer_load_and_track(
            kernel_info, memref_ssa, vector_bytes, voffset_v, instoffset
        )

        # Free voffset only if it was a temporary (not from cache)
        if voffset_is_temp:
            self.walker.emitter.vgpr_allocator.free_v(voffset_v)

    def _is_lds_memref(self, operation):
        """Check if the memref has LDS (workgroup) address space."""
        memref_type = operation.operands[0].type
        # Check if the memref has #gpu.address_space<workgroup> attribute
        if (
            hasattr(memref_type, "memory_space")
            and memref_type.memory_space is not None
        ):
            # Convert memory_space attribute to string and check for "workgroup"
            memory_space_str = str(memref_type.memory_space)
            return "workgroup" in memory_space_str.lower()
        return False

    def _emit_load_instruction(self, operation, kernel_info, memref_ssa, indices):
        """Emit load instruction for a vector.load operation derived purely from indices."""
        from .utils import build_memref_byte_offset_expr

        # Parse memref info and build byte offset expression
        memref_info = self._parse_load_memref_info(operation)
        byte_offset_expr = build_memref_byte_offset_expr(
            indices, kernel_info, memref_info
        )

        # Check address space to determine LDS vs global
        if self._is_lds_memref(operation):
            # LDS load path (workgroup address space)
            self._emit_lds_load(operation, kernel_info, memref_ssa, byte_offset_expr)
            return

        # Global buffer load path
        self._emit_global_load(operation, kernel_info, memref_ssa, byte_offset_expr)

    def _parse_store_type_info(self, operation):
        """Parse memref and vector type information from a vector.store operation."""
        # Get memref info
        memref_type_object = operation.operands[1].type
        try:
            shape, strides, element_bytes = parse_memref_type_from_obj(
                memref_type_object
            )
            memref_info = MemRefInfo(shape, strides, element_bytes)
        except Exception as e:
            raise ValueError(f"Cannot parse memref type for store operation: {e}")

        # Get vector type info
        value_vector_type = operation.operands[0].type
        try:
            num_elements, elem_bytes, _ = parse_vector_type_from_obj(value_vector_type)
            vector_bytes = num_elements * elem_bytes
        except Exception as e:
            raise ValueError(f"Cannot parse vector type for store value: {e}")

        return memref_info, value_vector_type, num_elements, vector_bytes

    def _compute_lds_address(
        self, kernel_info, memref_ssa, value_vector_type, indices, memref_info
    ):
        """Compute LDS address using MLIR's 2D memref indices and strides.

        Uses the actual indices from vector.store and the memref's stride information
        to compute the byte offset, rather than forcing lane-linear packing.
        """
        import sympy
        from .utils import build_memref_byte_offset_expr

        # Use MLIR's actual indices with memref strides
        byte_offset_expr = build_memref_byte_offset_expr(
            indices, kernel_info, memref_info
        )

        # Add view base offset (use MLIR offsets directly)
        vbase_val = self.walker._lds_view_base_bytes.get(memref_ssa, 0)
        if vbase_val:
            byte_offset_expr = byte_offset_expr + sympy.Integer(vbase_val)

        addr_reg = self._get_expr_emitter(kernel_info).get_or_emit(byte_offset_expr)
        return int(addr_reg[1:])

    def _extract_source_registers(self, vector_bytes):
        """Extract source registers from last_regs based on vector size."""
        first = (
            self.walker.last_regs[0]
            if isinstance(self.walker.last_regs, list)
            else self.walker.last_regs
        )

        if vector_bytes == 4:
            # Single register (scalar)
            return first[0] if isinstance(first, tuple) else first
        elif vector_bytes == 8:
            # Register pair
            return first if isinstance(first, tuple) else (first[0], first[1])
        elif vector_bytes == 16:
            # Register quad
            return (
                first
                if isinstance(first, tuple)
                else (first[0], first[1], first[2], first[3])
            )
        else:
            raise NotImplementedError(
                f"LDS stores of {vector_bytes} bytes not yet supported. "
                f"Supported sizes: 4 (ds_write_b32), 8 (ds_write_b64), 16 (ds_write_b128)"
            )

    def _emit_ds_write_and_track(self, memref_ssa, addr_v, src_regs, vector_bytes):
        """Emit ds_write instruction and track state for later reads."""
        if vector_bytes == 4:
            self.walker.emitter.emit_lds_write_b32(addr_v, src_regs)
            self.walker._lds_addr_vreg[memref_ssa] = addr_v
            self.walker._lds_last_pair[memref_ssa] = (
                src_regs,
                src_regs,
            )  # Store as pair for consistency
        elif vector_bytes == 8:
            self.walker.emitter.emit_lds_write_b64(addr_v, src_regs)
            self.walker._lds_addr_vreg[memref_ssa] = addr_v
            self.walker._lds_last_pair[memref_ssa] = src_regs
        elif vector_bytes == 16:
            self.walker.emitter.emit_lds_write_b128(addr_v, src_regs)
            self.walker._lds_addr_vreg[memref_ssa] = addr_v
            self.walker._lds_last_pair[memref_ssa] = (
                src_regs[0],
                src_regs[1],
            )  # Store first pair

    def _emit_lds_store(
        self,
        kernel_info,
        memref_ssa,
        value_vector_type,
        indices,
        memref_info,
        vector_bytes,
    ):
        """Emit an LDS store operation."""
        addr_v = self._compute_lds_address(
            kernel_info, memref_ssa, value_vector_type, indices, memref_info
        )
        # Wait for VMEM load to complete before extracting/using loaded data
        if self.walker.last_vmem_ticket is not None:
            self.walker.emitter.wait_for_vmem_ticket(self.walker.last_vmem_ticket)
        src_regs = self._extract_source_registers(vector_bytes)
        self._emit_ds_write_and_track(memref_ssa, addr_v, src_regs, vector_bytes)

    def _ensure_global_store_srd(self, kernel_info, memref_ssa):
        """Ensure SRD is set up for a global store."""
        binding_use = kernel_info.subspans[memref_ssa]
        if memref_ssa not in self.walker.emitter.srds:
            if not binding_use.memref_info:
                raise ValueError(
                    f"Cannot determine memref information for {memref_ssa}. "
                    f"SRD setup requires memref shape and element size."
                )

            limit_bytes = self._compute_buffer_size(binding_use.memref_info)
            arg_idx = binding_use.arg_index if binding_use.arg_index >= 0 else 0
            self.walker.emitter.ensure_srd_for_subspan(memref_ssa, arg_idx, limit_bytes)

    def _pop_next_scalar_register(self):
        """Pop and return the next scalar register from the queue, updating state."""
        # Normalize to list
        src_regs = (
            self.walker.last_regs
            if isinstance(self.walker.last_regs, list)
            else [self.walker.last_regs]
        )
        is_multi_element = len(src_regs) > 1

        # Pop first element
        src_v = src_regs[0]
        if not isinstance(src_v, int):
            src_v = src_v[0] if isinstance(src_v, tuple) else src_v

        # Update queue
        remaining = src_regs[1:]
        if remaining:
            self.walker.last_regs = remaining
        else:
            self.walker.last_regs = None
            if is_multi_element:
                self.walker._mfma_stores_emitted = True

        return src_v

    def _compute_store_address(
        self, value_vector_type, indices, kernel_info, memref_info
    ):
        """Compute voffset and instoffset for a store operation.

        Returns (voffset_v, instoffset, voffset_is_temp)
        """
        from .utils import build_element_byte_offset_exprs

        byte_exprs = build_element_byte_offset_exprs(
            value_vector_type, indices, kernel_info, memref_info
        )
        const_offset, dynamic_expr = split_const_dynamic(byte_exprs[0])
        if dynamic_expr == 0 or (
            hasattr(dynamic_expr, "is_zero") and dynamic_expr.is_zero
        ):
            voffset_v = self.walker.emitter.vgpr_allocator.alloc_v()
            from .instructions import VMovB32

            self.walker.emitter.emit_instruction(VMovB32(voffset_v, 0))
            return voffset_v, const_offset, True
        else:
            voffset_reg = self._get_expr_emitter(kernel_info).get_or_emit(dynamic_expr)
            voffset_v = int(voffset_reg[1:])
            return voffset_v, const_offset, False

    def _emit_global_store(
        self,
        kernel_info,
        memref_ssa,
        value_vector_type,
        indices,
        memref_info,
        num_elements,
        vector_bytes,
    ):
        """Emit a global buffer store operation."""
        voffset_v, instoffset, voffset_is_temp = self._compute_store_address(
            value_vector_type, indices, kernel_info, memref_info
        )

        # Wait for VMEM load to complete before using loaded data
        # (stores may come from prior loads, e.g., copy kernels)
        if self.walker.last_vmem_ticket is not None:
            self.walker.emitter.wait_for_vmem_ticket(self.walker.last_vmem_ticket)

        if num_elements == 1:
            # Scalar store: pop register from queue, emit, free
            src_v = self._pop_next_scalar_register()
            self.walker.emitter.emit_store_scalar_with_vindex(
                memref_ssa, src_v, voffset_v, instoffset
            )
            if voffset_is_temp:
                self.walker.emitter.vgpr_allocator.free_v(voffset_v)
            if isinstance(src_v, int):
                self.walker.emitter.vgpr_allocator.free_v(src_v)
        else:
            # Vectorized store: emit all elements at once
            self.walker.emitter.emit_store_with_regs(
                memref_ssa, self.walker.last_regs, vector_bytes, voffset_v, instoffset
            )
            if voffset_is_temp:
                self.walker.emitter.vgpr_allocator.free_v(voffset_v)

    def _is_lds_store_memref(self, operation):
        """Check if the store destination memref has LDS (workgroup) address space."""
        memref_type = operation.operands[1].type  # For stores, memref is operand[1]
        # Check if the memref has #gpu.address_space<workgroup> attribute
        if (
            hasattr(memref_type, "memory_space")
            and memref_type.memory_space is not None
        ):
            # Convert memory_space attribute to string and check for "workgroup"
            memory_space_str = str(memref_type.memory_space)
            return "workgroup" in memory_space_str.lower()
        return False

    def _emit_store_instruction(self, operation, kernel_info, memref_ssa, indices):
        """Emit store instruction for a vector.store operation derived purely from indices."""
        # Parse type information
        memref_info, value_vector_type, num_elements, vector_bytes = (
            self._parse_store_type_info(operation)
        )

        # Check if we have data to store
        if not self.walker.last_regs:
            raise RuntimeError(
                "Store encountered without a preceding load to provide data regs."
            )

        # Check address space to determine LDS vs global
        if self._is_lds_store_memref(operation):
            # LDS store path (workgroup address space)
            self._emit_lds_store(
                kernel_info,
                memref_ssa,
                value_vector_type,
                indices,
                memref_info,
                vector_bytes,
            )
            return

        # Global buffer store path
        self._ensure_global_store_srd(kernel_info, memref_ssa)
        self._emit_global_store(
            kernel_info,
            memref_ssa,
            value_vector_type,
            indices,
            memref_info,
            num_elements,
            vector_bytes,
        )
