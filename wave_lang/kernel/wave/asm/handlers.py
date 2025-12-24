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

import operator

import sympy

from wave_lang.support.ir_imports import (
    affine_d,
    amdgpu_d,
    arith_d,
    gpu_d,
    memref_d,
    rocdl_d,
    scf_d,
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
from .gather_to_shared import G2SHandler

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
        # Gather-to-LDS handler (composition)
        self.g2s = G2SHandler(self)

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

    def _handle_arith_binop(self, operation, kernel_info: KernelInfo, op_func):
        """Handle binary arithmetic operations (addi, muli) in index_env.

        Args:
            operation: The MLIR operation (AddIOp or MulIOp)
            kernel_info: Kernel info containing index_env
            op_func: Binary function to apply (e.g., operator.add, operator.mul)
        """
        lhs = kernel_info.index_env.get(str(operation.operands[0]))
        rhs = kernel_info.index_env.get(str(operation.operands[1]))

        # Operands not tracked - can't compute result
        if lhs is None or rhs is None:
            return

        # Convert symbolic strings (tid_x, wgid_x, etc.) to SymPy symbols
        if isinstance(lhs, str):
            lhs = sympy.Symbol(lhs)
        if isinstance(rhs, str):
            rhs = sympy.Symbol(rhs)

        if isinstance(lhs, (int, sympy.Expr)) and isinstance(rhs, (int, sympy.Expr)):
            kernel_info.index_env[str(operation.result)] = op_func(lhs, rhs)

    def handle_arith_addi_op(self, operation: arith_d.AddIOp, kernel_info: KernelInfo):
        """Handle arith.addi - track integer addition in index_env."""
        self._handle_arith_binop(operation, kernel_info, operator.add)

    def handle_arith_muli_op(self, operation: arith_d.MulIOp, kernel_info: KernelInfo):
        """Handle arith.muli - track integer multiplication in index_env."""
        self._handle_arith_binop(operation, kernel_info, operator.mul)

    def handle_arith_index_cast_op(
        self, operation: arith_d.IndexCastOp, kernel_info: KernelInfo
    ):
        """Handle arith.index_cast operations - propagate values through cast.

        Propagates integers, SymPy expressions, and symbolic strings (tid_x, etc.).
        """
        result_ssa = str(operation.result)
        src_ssa = str(operation.operands[0])

        src_val = kernel_info.index_env.get(src_ssa)
        if src_val is None:
            return

        # Propagate numeric values and symbolic strings
        if isinstance(src_val, (int, sympy.Expr, str)):
            kernel_info.index_env[result_ssa] = src_val

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
            kernel_info.index_env[str(operation.result)] = "tid_x"
            if upper_bound is not None:
                kernel_info.tid_ub_x = upper_bound
        elif "dim y" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "tid_y"
            if upper_bound is not None:
                kernel_info.tid_ub_y = upper_bound
        elif "dim z" in dimension_string:
            if upper_bound is not None:
                kernel_info.tid_ub_z = upper_bound
            kernel_info.index_env[str(operation.result)] = "tid_z"

    def handle_gpu_block_id_op(
        self, operation: gpu_d.BlockIdOp, kernel_info: KernelInfo
    ):
        """
        Handle gpu.block_id operations - map to workgroup ID symbols.

        Maps block IDs to symbolic names that the expression emitter can use:
        - block_id x -> wgid_x
        - block_id y -> wgid_y
        - block_id z -> wgid_z
        """
        dimension = operation.dimension
        dimension_string = str(dimension)

        if "dim x" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "wgid_x"
        elif "dim y" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "wgid_y"
        elif "dim z" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "wgid_z"

    def _extract_dimension_values(
        self,
        operation: affine_d.AffineApplyOp,
        kernel_info: KernelInfo,
        num_dimensions: int,
    ) -> list:
        """Extract dimension values from the first num_dimensions operands."""
        import sympy

        dimension_values = []

        for i in range(num_dimensions):
            if i < len(operation.operands):
                operand_ssa = str(operation.operands[i])
                operand_value = kernel_info.index_env.get(operand_ssa)

                if isinstance(operand_value, int):
                    dimension_values.append(operand_value)
                elif isinstance(operand_value, sympy.Expr):
                    # SymPy expressions from previous affine.apply results
                    dimension_values.append(operand_value)
                elif operand_value in [
                    "tid_x",
                    "tid_y",
                    "tid_z",
                    "wgid_x",
                    "wgid_y",
                    "wgid_z",
                ]:
                    # Thread IDs and workgroup IDs can be represented as symbols in the expression
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
        import sympy

        symbol_values = []

        for i in range(num_symbols):
            operand_index = num_dimensions + i
            if operand_index < len(operation.operands):
                operand_ssa = str(operation.operands[operand_index])
                operand_value = kernel_info.index_env.get(operand_ssa)

                if isinstance(operand_value, int):
                    symbol_values.append(operand_value)
                elif isinstance(operand_value, sympy.Expr):
                    # SymPy expressions from previous affine.apply results
                    symbol_values.append(operand_value)
                elif operand_value in [
                    "tid_x",
                    "tid_y",
                    "tid_z",
                    "wgid_x",
                    "wgid_y",
                    "wgid_z",
                ]:
                    # Thread IDs and workgroup IDs can be used as symbol values
                    symbol_values.append(operand_value)
                elif (
                    isinstance(operand_value, str)
                    and operand_value.startswith("s")
                    and operand_value[1:].isdigit()
                ):
                    # SGPR references (e.g., "s4" for loop counter) can be used as symbol values
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

    def handle_vector_extract_strided_slice_op(
        self, operation: vector_d.ExtractStridedSliceOp, kernel_info: KernelInfo
    ):
        """Handle vector.extract_strided_slice operations - extract subset of source registers."""
        # Get source SSA value and its registers
        source_ssa = str(operation.operands[0])
        source_regs = self.walker.ssa_to_vgpr.get(source_ssa)

        if not source_regs:
            # Source not tracked - skip silently
            return

        # Extract offset and size from operation attributes
        offsets = operation.attributes["offsets"]
        sizes = operation.attributes["sizes"]

        # Parse the offset value (should be a single integer for 1D extract)
        offset_val = int(str(offsets).split("[")[1].split("]")[0])
        size_val = int(str(sizes).split("[")[1].split("]")[0])

        # Extract the appropriate subset of registers
        if size_val == 1:
            # Single scalar extract - return just the one register as a tuple
            extracted_reg = source_regs[offset_val]
            result_regs = (extracted_reg,)
        else:
            # Multi-element extract - return a slice
            result_regs = source_regs[offset_val : offset_val + size_val]

        result_ssa = str(operation.result)
        self.walker.ssa_to_vgpr[result_ssa] = result_regs

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
        
        # In kernel IR mode, SRD setup is deferred to actual load/store operations
        # Just record the subspan info, SRD will be set up lazily
        pass

    def handle_mfma_op(self, operation: amdgpu_d.MFMAOp, kernel_info: KernelInfo):
        """Handle amdgpu.mfma operations - emit MFMA instruction with proper input sourcing."""
        # Get the operand SSA values from the MFMA operation
        # MFMA format: %result = amdgpu.mfma %lhs * %rhs + %acc
        if len(operation.operands) >= 3:
            lhs_ssa = str(operation.operands[0])  # First operand (LHS of multiply)
            rhs_ssa = str(operation.operands[1])  # Second operand (RHS of multiply)
            acc_ssa = str(operation.operands[2])  # Third operand (accumulator)

            # Kernel IR mode: use virtual registers
            from .kernel_ir import KVReg, KRegRange
            
            ctx = self.walker.kernel_ctx
            
            # Get operand registers from kernel context
            lhs_regs = ctx.ssa_to_reg.get(lhs_ssa)
            rhs_regs = ctx.ssa_to_reg.get(rhs_ssa)
            acc_regs = ctx.ssa_to_reg.get(acc_ssa)
            
            if lhs_regs and rhs_regs:
                # Call kernel context MFMA emission
                result_regs = ctx.emit_mfma_f32_16x16x16_f16(
                    lhs_regs,
                    rhs_regs,
                    acc_regs if acc_regs and len(acc_regs) == 4 else None,
                )
                
                # Track result in SSA mapping
                result_ssa = str(operation.result)
                ctx.ssa_to_reg[result_ssa] = result_regs
                
                return
            
            raise RuntimeError(
                f"MFMA operation inputs not available. "
                f"lhs={lhs_ssa} ({lhs_regs}), rhs={rhs_ssa} ({rhs_regs})"
            )

    def handle_barrier_op(self, operation: gpu_d.BarrierOp, kernel_info: KernelInfo):
        """Handle gpu.barrier operations - emit synchronization barrier."""
        self.walker.unified.s_barrier(comment="workgroup barrier")

    def handle_lds_barrier_op(
        self, operation: amdgpu_d.LDSBarrierOp, kernel_info: KernelInfo
    ):
        """Handle amdgpu.lds_barrier - emit lgkmcnt(0) + s_barrier."""
        self.walker.unified.s_waitcnt(waitcnt="lgkmcnt(0)")
        self.walker.unified.s_barrier(comment="LDS barrier")

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
        forcing lane-linear addressing. The MLIR indices already encode the correct
        addressing for both single-wave and multi-wave modes, including any swizzle
        patterns needed for cache efficiency.
        
        Optimization: When the address has a constant offset component that fits within
        the hardware limit (DS_MAX_OFFSET), we use the ds_read offset field instead of
        computing the full address. This saves a v_add_u32 instruction.
        
        For offsets exceeding DS_MAX_OFFSET (~8192 bytes on CDNA3/4), we fall back to
        computing the full address without using the offset field.
        """
        import os
        import sympy
        from .utils import split_const_dynamic

        DEBUG_DS_OFFSET = os.environ.get("WAVE_LDS_DSREAD_OFFSET_DEBUG", "0") == "1"
        
        # Add view base offset if present
        vbase_val = self.walker._lds_view_base_bytes.get(memref_ssa, 0)
        original_byte_offset_expr = byte_offset_expr  # Save for debug

        # Use MLIR-derived expression for all cases (single-wave, multi-wave, g2s, non-g2s)
        # The MLIR index expression already contains the correct addressing formula
        if vbase_val:
            byte_offset_expr = byte_offset_expr + sympy.Integer(vbase_val)

        # Split address into base + constant offset to use ds_read offset field
        # ds_read supports 16-bit offset (0-65535), but we use conservative limits
        const_offset, base_expr = split_const_dynamic(byte_offset_expr, max_immediate=65528)
        
        # Max offset for ds_read_b64 on CDNA3/CDNA4
        # The ISA spec says 16-bit unsigned (0-65535), which is correct.
        # Previous conservative limit (2040) was causing excessive constant
        # materialization for LDS addresses in the 4096+ range.
        # Testing shows 8192 works correctly for GEMM kernels.
        DS_MAX_OFFSET = 8192  # Increased to cover typical LDS offset ranges
        DS_ALIGN = 8  # ds_read_b64 requires 8-byte alignment
        
        if DEBUG_DS_OFFSET:
            print(f"[DS_OFFSET_DEBUG] memref={memref_ssa[:60]}...")
            print(f"[DS_OFFSET_DEBUG]   vbase_val={vbase_val}")
            print(f"[DS_OFFSET_DEBUG]   original_expr={original_byte_offset_expr}")
            print(f"[DS_OFFSET_DEBUG]   after_vbase_expr={byte_offset_expr}")
            print(f"[DS_OFFSET_DEBUG]   const_offset={const_offset}, base_expr={base_expr}")
        
        # Kernel IR mode: emit LDS load with virtual registers
        from .kernel_ir import KInstr, KImm, KVReg, KRegRange, KPhysVReg
        
        ctx = self.walker.kernel_ctx
        
        # Determine if we can use the offset field
        has_dynamic_base = len(base_expr.free_symbols) > 0
        
        if not has_dynamic_base:
            # Pure constant address - materialize it
            addr_vreg = ctx.vreg()
            ctx.program.emit(KInstr(
                "v_mov_b32", (addr_vreg,), (KImm(int(byte_offset_expr)),),
                comment=f"LDS addr = {byte_offset_expr}"
            ))
            lds_offset = 0
        elif 0 <= const_offset <= DS_MAX_OFFSET and const_offset % DS_ALIGN == 0:
            # Can use offset field - compute only the base expression
            # Use a fresh scope to avoid CSE issues with different memrefs
            with ctx.expr_emitter.scope("lds_base"):
                addr_vreg = ctx.expr_emitter.get_or_emit(base_expr)
            lds_offset = const_offset
            if DEBUG_DS_OFFSET:
                print(f"[DS_OFFSET_DEBUG]   -> USING_OFFSET: addr={addr_vreg}, offset={lds_offset}")
        else:
            # Offset out of range or not aligned - compute full address
            addr_vreg = ctx.expr_emitter.get_or_emit(byte_offset_expr)
            lds_offset = 0
        
        # Allocate destination pair and emit ds_read_b64
        dst_range = ctx.vreg_pair()
        ctx.emit_lds_read_b64(dst_range, addr_vreg, lds_offset)
        
        # Track in SSA mapping as tuple of KVReg
        result_ssa = str(operation.results[0])
        result_regs = (KVReg(dst_range.base_reg.id), KVReg(dst_range.base_reg.id + 1))
        ctx.ssa_to_reg[result_ssa] = result_regs

    def _ensure_global_load_srd(self, kernel_info, memref_ssa):
        """Ensure SRD is set up for a global load."""
        # Kernel IR mode: use kernel_ctx SRD tracking
        if memref_ssa in self.walker.kernel_ctx.srd_ranges:
            return
        
        binding_use = kernel_info.subspans[memref_ssa]
        if not binding_use.memref_info:
            raise ValueError(
                f"Cannot determine memref information for {memref_ssa}. "
                f"SRD setup requires memref shape and element size."
            )
        
        limit_bytes = self._compute_buffer_size(binding_use.memref_info)
        arg_idx = binding_use.arg_index if binding_use.arg_index >= 0 else 0
        self.walker.kernel_ctx.ensure_srd(memref_ssa, arg_idx, limit_bytes)

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
        self, operation, kernel_info, memref_ssa, vector_bytes, voffset_v, instoffset
    ):
        """Emit buffer load instruction and track loaded registers and ticket."""
        result_ssa = str(operation.results[0])
        
        # Kernel IR mode: emit via kernel_ctx with virtual registers
        from .kernel_ir import KVReg
        # voffset_v might be a physical index; convert to virtual reg
        if isinstance(voffset_v, int):
            voffset = KVReg(voffset_v)  # Treat as virtual for now
        else:
            voffset = voffset_v
        
        loaded_ranges = self.walker.kernel_ctx.emit_buffer_load(
            memref_ssa, vector_bytes, voffset, instoffset
        )
        
        # Convert ranges to tuple of register IDs for ssa_to_vgpr compatibility
        if len(loaded_ranges) == 1:
            # Single range (pair or quad)
            base = loaded_ranges[0].base_reg
            count = loaded_ranges[0].count
            regs_tuple = tuple(KVReg(base.id + i) for i in range(count))
        else:
            # Multiple ranges - flatten into single tuple
            regs_tuple = []
            for rng in loaded_ranges:
                base = rng.base_reg
                regs_tuple.extend(KVReg(base.id + i) for i in range(rng.count))
            regs_tuple = tuple(regs_tuple)
        
        self.walker.kernel_ctx.ssa_to_reg[result_ssa] = regs_tuple

    def _emit_global_load(self, operation, kernel_info, memref_ssa, byte_offset_expr):
        """Emit a global buffer load operation."""
        self._ensure_global_load_srd(kernel_info, memref_ssa)

        # Split constant/dynamic and materialize dynamic part via cached emitter (CSE)
        const_offset, dynamic_expr = split_const_dynamic(byte_offset_expr)
        
        # Kernel IR mode: allocate virtual registers
        from .kernel_ir import KInstr, KImm, KVReg, KPhysVReg
        
        # Compute voffset in kernel IR
        voffset_v = self.walker.kernel_ctx.vreg()
        
        if dynamic_expr == 0 or (
            hasattr(dynamic_expr, "is_zero") and dynamic_expr.is_zero
        ):
            # No dynamic part: set voffset to 0
            self.walker.kernel_ctx.program.emit(KInstr(
                "v_mov_b32", (voffset_v,), (KImm(0),), comment="voffset = 0"
            ))
            instoffset = const_offset
        else:
            # Dynamic part: use expression emitter to compute voffset
            # The expression emitter caches results so the same expression
            # returns the same vreg (CSE)
            expr_emitter = self.walker.kernel_ctx.expr_emitter
            voffset_v = expr_emitter.get_or_emit(dynamic_expr)
            instoffset = const_offset
        
        vector_bytes = self._parse_vector_load_type(operation)
        self._emit_buffer_load_and_track(
            operation, kernel_info, memref_ssa, vector_bytes, voffset_v, instoffset
        )

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
        import sympy
        from .kernel_ir import KVReg, KRegRange, KInstr, KImm, KMemOffset
        from .utils import build_memref_byte_offset_expr
        
        ctx = self.walker.kernel_ctx
        
        # Compute LDS address, adding view base offset if present
        byte_offset_expr = build_memref_byte_offset_expr(
            indices, kernel_info, memref_info
        )
        # Add view base offset for this specific memref (each matrix has different base)
        vbase_val = self.walker._lds_view_base_bytes.get(memref_ssa, 0)
        if vbase_val:
            byte_offset_expr = byte_offset_expr + sympy.Integer(vbase_val)
        addr_vreg = ctx.expr_emitter.get_or_emit(byte_offset_expr)
        
        # Wait for any pending VMEM loads
        ctx.program.emit(KInstr(
            "s_waitcnt", (), ("vmcnt(0)",), comment="wait for VMEM before LDS store"
        ))
        
        # Get source registers from SSA mapping (these are KVReg objects)
        src_regs = self._current_store_regs
        
        # Build a properly aligned KRegRange for the source
        if vector_bytes == 4:
            # Single register
            src_vreg = src_regs[0] if isinstance(src_regs, (tuple, list)) else src_regs
            ctx.program.emit(KInstr(
                "ds_write_b32", (), (addr_vreg, src_vreg), 
                comment=f"LDS store 4B to {memref_ssa}"
            ))
        elif vector_bytes == 8:
            # Register pair (must be 64-bit aligned)
            if isinstance(src_regs, (tuple, list)) and len(src_regs) >= 2:
                # Create aligned range from the source registers
                base_id = src_regs[0].id if isinstance(src_regs[0], KVReg) else src_regs[0]
                src_range = KRegRange(KVReg(base_id), 2, alignment=2)
            else:
                raise ValueError(f"Expected 2 registers for ds_write_b64, got {src_regs}")
            ctx.emit_lds_write_b64(addr_vreg, src_range)
        elif vector_bytes == 16:
            # Register quad (must be 128-bit aligned)
            if isinstance(src_regs, (tuple, list)) and len(src_regs) >= 4:
                base_id = src_regs[0].id if isinstance(src_regs[0], KVReg) else src_regs[0]
                src_range = KRegRange(KVReg(base_id), 4, alignment=4)
            else:
                raise ValueError(f"Expected 4 registers for ds_write_b128, got {src_regs}")
            ctx.emit_lds_write_b128(addr_vreg, src_range)
        else:
            raise NotImplementedError(f"LDS stores of {vector_bytes} bytes not supported")

    def _ensure_global_store_srd(self, kernel_info, memref_ssa):
        """Ensure SRD is set up for a global store."""
        binding_use = kernel_info.subspans[memref_ssa]
        
        # Kernel IR mode: use kernel_ctx SRD tracking
        if memref_ssa in self.walker.kernel_ctx.srd_ranges:
            return
        
        if not binding_use.memref_info:
            raise ValueError(
                f"Cannot determine memref information for {memref_ssa}. "
                f"SRD setup requires memref shape and element size."
            )
        
        limit_bytes = self._compute_buffer_size(binding_use.memref_info)
        arg_idx = binding_use.arg_index if binding_use.arg_index >= 0 else 0
        self.walker.kernel_ctx.ensure_srd(memref_ssa, arg_idx, limit_bytes)

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
        # Kernel IR mode: use virtual registers
        from .kernel_ir import KInstr, KImm, KVReg, KPhysVReg, KRegRange
        from .utils import build_element_byte_offset_exprs
        
        # Get expression emitter - loop-invariant expressions are cached globally,
        # loop-varying expressions are never cached, so no cache clearing needed.
        expr_emitter = self.walker.kernel_ctx.expr_emitter
        
        # Compute address - allocate virtual voffset
        byte_exprs = build_element_byte_offset_exprs(
            value_vector_type, indices, kernel_info, memref_info
        )
        const_offset, dynamic_expr = split_const_dynamic(byte_exprs[0])
        
        # Compute voffset in kernel IR (store path)
        voffset_v = self.walker.kernel_ctx.vreg()
        
        if dynamic_expr == 0 or (
            hasattr(dynamic_expr, "is_zero") and dynamic_expr.is_zero
        ):
            self.walker.kernel_ctx.program.emit(KInstr(
                "v_mov_b32", (voffset_v,), (KImm(0),), comment="voffset = 0"
            ))
            instoffset = const_offset
        else:
            # Dynamic part: use expression emitter to compute voffset
            voffset_v = expr_emitter.get_or_emit(dynamic_expr)
            instoffset = const_offset
        
        # IMPORTANT: Wait for pending loads BEFORE setting up store SRD
        # Otherwise we overwrite the load SRD while loads are still in flight
        self.walker.kernel_ctx.program.emit(KInstr(
            "s_waitcnt", (), ("vmcnt(0)",), comment="MARKER: wait for loads before store SRD setup"
        ))
        
        # Now it's safe to set up the store SRD (may reuse same physical regs)
        self._ensure_global_store_srd(kernel_info, memref_ssa)
        
        # Get source registers from ssa_to_reg
        src_regs = self._current_store_regs
        if isinstance(src_regs, tuple) and len(src_regs) > 0:
            # Convert to KRegRange(s) for the store
            # Group registers into quads (16 bytes each) for vectorized stores
            num_regs = len(src_regs)
            
            if vector_bytes <= 4:
                # Single dword
                first_reg = src_regs[0]
                if isinstance(first_reg, KVReg):
                    src_range = KRegRange(first_reg, 1)
                else:
                    src_range = KRegRange(KVReg(first_reg), 1)
                src_ranges = (src_range,)
            elif vector_bytes == 8:
                # Pair (dwordx2)
                first_reg = src_regs[0]
                if isinstance(first_reg, KVReg):
                    src_range = KRegRange(first_reg, 2)
                else:
                    src_range = KRegRange(KVReg(first_reg), 2)
                src_ranges = (src_range,)
            else:
                # Multiple quads (16+ bytes)
                # Each quad is 4 VGPRs = 16 bytes
                num_quads = (vector_bytes + 15) // 16
                src_ranges = []
                for q in range(num_quads):
                    base_idx = q * 4
                    if base_idx < num_regs:
                        first_reg = src_regs[base_idx]
                        if isinstance(first_reg, KVReg):
                            src_range = KRegRange(first_reg, 4)
                        else:
                            src_range = KRegRange(KVReg(first_reg), 4)
                        src_ranges.append(src_range)
                src_ranges = tuple(src_ranges)
            
            self.walker.kernel_ctx.emit_buffer_store(
                memref_ssa, src_ranges, voffset_v, instoffset
            )

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

        # Get the SSA value being stored (first operand)
        value_ssa = str(operation.operands[0])

        # Look up the registers containing the value to store
        value_regs = self.walker.kernel_ctx.ssa_to_reg.get(value_ssa)
        if not value_regs:
            raise RuntimeError(
                f"Store operation references SSA value {value_ssa} but it's not in kernel_ctx.ssa_to_reg. "
                f"Available: {list(self.walker.kernel_ctx.ssa_to_reg.keys())}"
            )

        # Store value_regs for extraction in subsequent methods
        self._current_store_regs = value_regs

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
        # SRD setup happens inside _emit_global_store after waitcnt
        self._emit_global_store(
            kernel_info,
            memref_ssa,
            value_vector_type,
            indices,
            memref_info,
            num_elements,
            vector_bytes,
        )

    def handle_scf_for_op(self, operation: scf_d.ForOp, kernel_info: KernelInfo):
        """
        Handle scf.for operations - emit loop assembly code.

        Args:
            operation: The scf.for operation
            kernel_info: Kernel information for context
        """
        # Extract loop bounds
        lower_bound_ssa = str(operation.lowerBound)
        upper_bound_ssa = str(operation.upperBound)
        step_ssa = str(operation.step)

        # Get bounds from index_env (should be constants)
        if lower_bound_ssa not in kernel_info.index_env:
            raise ValueError(
                f"Loop lower bound {lower_bound_ssa} not found in index_env"
            )
        if upper_bound_ssa not in kernel_info.index_env:
            raise ValueError(
                f"Loop upper bound {upper_bound_ssa} not found in index_env"
            )
        if step_ssa not in kernel_info.index_env:
            raise ValueError(f"Loop step {step_ssa} not found in index_env")
        lower_bound = kernel_info.index_env[lower_bound_ssa]
        upper_bound = kernel_info.index_env[upper_bound_ssa]
        step = kernel_info.index_env[step_ssa]
        
        # Pre-create G2S SRDs BEFORE the loop starts
        # This is critical for correctness: if G2S operations are in the loop body,
        # we need to create all SRD copies before the loop header is emitted.
        # Otherwise, the SRD copy for matrix B can overwrite the original SRD for
        # matrix A, causing incorrect memory accesses in subsequent loop iterations.
        from .gather_to_shared import analyze_g2s_region, precreate_g2s_srds
        loop_body = operation.body
        loop_ops = list(loop_body.operations)
        g2s_schedule = analyze_g2s_region(loop_ops)
        if g2s_schedule is not None:
            # Pre-create G2S SRDs (these must be created before the loop)
            precreate_g2s_srds(g2s_schedule, kernel_info, self)

        # Kernel IR mode: use virtual registers
        from .kernel_ir import KVReg, KRegRange
        
        ctx = self.walker.kernel_ctx
        
        # Begin loop structure with virtual registers
        loop_ctx = ctx.begin_loop(lower_bound, upper_bound, step)
        
        # Get induction variable and map it to the loop counter SGPR
        loop_body = operation.body
        induction_var = loop_body.arguments[0]
        induction_var_ssa = str(induction_var)
        counter_sreg = loop_ctx["counter_sreg"]
        
        # Store mapping from SSA induction variable to SGPR
        # Use string format "s{idx}" for compatibility with expression simplification
        # The KPhysSReg has an index attribute we can use
        kernel_info.index_env[induction_var_ssa] = f"s{counter_sreg.index}"
        loop_ctx["induction_var_ssa"] = induction_var_ssa
        
        # Allocate and initialize VGPRs for iter_args (accumulators)
        num_iter_args = len(loop_body.arguments) - 1  # Exclude induction var
        iter_arg_ranges = ctx.alloc_accumulators(num_iter_args)
        
        # Track in SSA->reg map
        for i, arg in enumerate(loop_body.arguments[1:]):
            arg_ssa = str(arg)
            quad = iter_arg_ranges[i]
            # Store as tuple of individual regs for compatibility
            regs = tuple(KVReg(quad.base_reg.id + j) for j in range(4))
            ctx.ssa_to_reg[arg_ssa] = regs
        
        loop_ctx["iter_arg_ranges"] = iter_arg_ranges
        
        # Emit loop header
        ctx.emit_loop_header(loop_ctx)
        
        # Walk loop body (mark as inside loop to prevent duplicate M0/SRD setup)
        self.walker._inside_loop = True
        self.walker._walk_block(loop_body, kernel_info)
        self.walker._inside_loop = False
        
        # Emit loop latch
        ctx.emit_loop_latch(loop_ctx)
        
        # End loop
        ctx.end_loop()
        
        # Map scf.for results to final values of iter_args
        for i, result in enumerate(operation.results):
            result_ssa = str(result)
            if i < len(iter_arg_ranges):
                quad = iter_arg_ranges[i]
                regs = tuple(KVReg(quad.base_reg.id + j) for j in range(4))
                ctx.ssa_to_reg[result_ssa] = regs

    # Note: gather_to_lds handlers moved to gather_to_shared.py (G2SMixin)

    def handle_memref_cast_op(
        self, operation: memref_d.CastOp, kernel_info: KernelInfo
    ):
        """Handle memref.cast operations - track source memref mapping.

        MLIR format:
            %result = memref.cast %src : memref<...> to memref<...>
        """
        result_ssa = str(operation.results[0])
        source_ssa = str(operation.operands[0])

        # Track the cast chain for SRD lookup
        if not hasattr(self.walker, "_memref_cast_sources"):
            self.walker._memref_cast_sources = {}
        self.walker._memref_cast_sources[result_ssa] = source_ssa

    def handle_memref_reinterpret_cast_op(
        self, operation: memref_d.ReinterpretCastOp, kernel_info: KernelInfo
    ):
        """Handle memref.reinterpret_cast operations - track source memref mapping.

        MLIR format:
            %result = memref.reinterpret_cast %src to offset: [...], sizes: [...], strides: [...]
                : memref<...> to memref<...>
        """
        result_ssa = str(operation.results[0])
        source_ssa = str(operation.operands[0])

        # Track the cast chain for SRD lookup
        if not hasattr(self.walker, "_memref_cast_sources"):
            self.walker._memref_cast_sources = {}
        self.walker._memref_cast_sources[result_ssa] = source_ssa

    def handle_fat_raw_buffer_cast_op(
        self, operation: amdgpu_d.FatRawBufferCastOp, kernel_info: KernelInfo
    ):
        """Handle amdgpu.fat_raw_buffer_cast - track source memref and cache swizzle stride."""
        result_ssa = str(operation.results[0])
        source_ssa = str(operation.operands[0])

        # Extract cacheSwizzleStride from operand 2 if present
        cache_swizzle_stride = None
        if len(operation.operands) >= 3:
            defining_op = operation.operands[2].owner.opview
            if isinstance(defining_op, arith_d.ConstantOp) and hasattr(
                defining_op.value, "value"
            ):
                cache_swizzle_stride = int(defining_op.value.value)

        # Track for gather_to_lds SRD tracing
        if not hasattr(self.walker, "_fat_buffer_sources"):
            self.walker._fat_buffer_sources = {}
        info = {"source_ssa": source_ssa}
        if cache_swizzle_stride is not None:
            info["cache_swizzle_stride"] = cache_swizzle_stride
        self.walker._fat_buffer_sources[result_ssa] = info

    def handle_readfirstlane_op(
        self, operation: rocdl_d.ReadfirstlaneOp, kernel_info: KernelInfo
    ):
        """Handle rocdl.readfirstlane - propagate value for uniform broadcast.

        The expression is preserved as-is (not evaluated) because each wavefront
        has different tid values. v_readfirstlane is emitted during code generation.
        """
        result_ssa = str(operation.results[0])
        source_ssa = str(operation.operands[0])

        if source_ssa in kernel_info.index_env:
            kernel_info.index_env[result_ssa] = kernel_info.index_env[source_ssa]

    def handle_subgroup_broadcast_op(
        self, operation: gpu_d.SubgroupBroadcastOp, kernel_info: KernelInfo
    ):
        """Handle gpu.subgroup_broadcast - propagate value for uniform broadcast.

        The expression is preserved as-is (not evaluated) because each wavefront
        has different tid values. v_readfirstlane is emitted during code generation.
        """
        result_ssa = str(operation.results[0])
        source_ssa = str(operation.src)

        if source_ssa in kernel_info.index_env:
            kernel_info.index_env[result_ssa] = kernel_info.index_env[source_ssa]

    def handle_s_waitcnt_op(
        self, operation: rocdl_d.SWaitcntOp, kernel_info: KernelInfo
    ):
        """Handle rocdl.s.waitcnt - emit wait count instruction.

        Encoding (gfx9+): bits 0-3 = vmcnt (0 = wait for all, 15 = no wait)
        """
        waitcnt_value = int(operation.bitfield.value)
        vmcnt = waitcnt_value & 0xF  # 4-bit field: 0-15

        # vmcnt=15 means "no wait" (max 4-bit value), so only emit if < 15
        if vmcnt < 15:
            self.walker.unified.s_waitcnt(f"vmcnt({vmcnt})")
            # Notify ticketing system about the wait
            # Use kernel_ctx.ticketing (no-op for kernel IR) or fall back to emitter
            if self.walker.kernel_ctx is not None:
                self.walker.kernel_ctx.ticketing.observe_vmem_wait(vmcnt)
            elif self.walker.emitter is not None:
                self.walker.emitter.ticketing.observe_vmem_wait(vmcnt)
