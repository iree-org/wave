# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import Optional
import inspect

from wave_lang.support.ir_imports import (
    IndexType,
    IntegerAttr,
    MemRefType,
    RankedTensorType,
    Value,
    arith_d,
    flow_d,
    tensor_d,
)

from .kernel_codegen import BindingDesc, KernelSignature, BindingType
from ..wave.constraints import DeviceConstraint
from ..lang.kernel_buffer import KernelBufferUsage
from ..lang.grid import Grid

from iree.compiler._mlir_libs._mlir.ir import Attribute


def substitute_dimensions_in_shape(symbolic_shape, symbol_map):
    """
    recursively substitutes symbolic dimensions in a shape with their full dimensions
    based on the provided symbol_map.
    """
    if isinstance(symbolic_shape, (list, tuple)):
        return [
            substitute_dimensions_in_shape(dim, symbol_map) for dim in symbolic_shape
        ]
    elif symbolic_shape in symbol_map:
        return symbol_map[symbolic_shape]
    else:
        return symbolic_shape


def get_or_create_index_constant(value: int, constant_map: dict) -> Value:
    """
    This function reuses existing constants in the mlir output if they exist.
    """
    if value not in constant_map:
        constant_map[value] = arith_d.constant(
            IndexType.get(),
            IntegerAttr.get(IndexType.get(), value),
        )
    return constant_map[value]


class HostSignature:
    """
    With the introduction of multi-device support, the host signature may not be the same as the kernel signature.
    This class is used to convert the kernel signature to a host signature by substituting the device constraints
    with the full dimensions of the buffers.
    """

    def __init__(
        self,
        kernel_sig: KernelSignature,
        device_constraints: list[DeviceConstraint] = None,
    ):
        self.bindings: list[BindingDesc] = []
        if device_constraints is None:
            self.bindings = kernel_sig.bindings
        else:
            self.get_host_buffer(kernel_sig.bindings, device_constraints)

    @property
    def buffer_bindings(self) -> list[BindingDesc]:
        """Gets all buffer bindings."""
        return [b for b in self.bindings if b.binding_type == BindingType.KERNEL_BUFFER]

    @property
    def input_buffer_bindings(self) -> list[BindingDesc]:
        """Gets all buffer bindings with input usage."""
        return [
            b
            for b in self.bindings
            if b.binding_type == BindingType.KERNEL_BUFFER
            and b.kernel_buffer_type.usage == KernelBufferUsage.INPUT
        ]

    @property
    def output_buffer_bindings(self) -> list[BindingDesc]:
        """Gets all buffer bindings with input usage."""
        return [
            b
            for b in self.bindings
            if b.binding_type == BindingType.KERNEL_BUFFER
            and b.kernel_buffer_type.usage == KernelBufferUsage.OUTPUT
        ]

    @property
    def scalar_buffer_bindings(self) -> list[BindingDesc]:
        """Gets all scalar bindings."""
        return [b for b in self.bindings if b.binding_type == BindingType.SCALAR_VALUE]

    def get_host_buffer(self, kernel_buffer_bindings, device_constraints):
        """
        Convert per-device kernel buffer bindings to host buffer bindings.
        """

        # Create a mapping from tile dimensions to their full dimensions
        tile_to_full_dim = {}
        for constraint in device_constraints:
            # constraint.dim is the full dimension (M, N)
            # constraint.tile_size is the tile dimension (BLOCK_M, BLOCK_N)
            tile_to_full_dim[constraint.tile_size] = constraint.dim

        for binding in kernel_buffer_bindings:
            # Update the kernel buffer type's symbolic shape, if it's part of
            # of the device constraints
            original_type = binding.kernel_buffer_type
            new_kernel_buffer_type = original_type
            if hasattr(original_type, "symbolic_shape"):
                original_shape = original_type.symbolic_shape
                new_shape = substitute_dimensions_in_shape(
                    original_shape, tile_to_full_dim
                )

                # if the new shape is the same as the original shape, no change is needed
                if new_shape != original_shape:
                    sig = inspect.signature(original_type.new_subtype)
                    kwargs = {"symbolic_shape": new_shape}

                    for param in sig.parameters:
                        if param == "cls" or param == "symbolic_shape":
                            continue

                        # keep the rest of the parameters as they are
                        kwargs[param] = getattr(original_type, param, None)

                    # create a new kernel buffer type with the updated symbolic shape
                    # and other parameters
                    new_kernel_buffer_type = original_type.new_subtype(**kwargs)

            # create a host binding descriptor
            host_binding = BindingDesc(
                reference=binding.reference,
                binding_type=binding.binding_type,
                name=binding.name,
                kernel_buffer_type=new_kernel_buffer_type,
                symbol_type=binding.symbol_type,
                scalar_type=binding.scalar_type,
            )

            self.bindings.append(host_binding)
        return


def split_input_tensors(
    host_tensor: Value,
    host_buffer_binding: BindingDesc,
    device_layout: Grid,
    device_constraints: list[DeviceConstraint],
    dynamic_argument_map: dict = None,
    device_map: Optional[dict[int, list]] = None,
    constant_map: Optional[dict] = None,
) -> tuple[dict, dict]:
    """
    Create tensor slices for each device from the host tensor using static dimensions.
    Reuses slices when multiple devices need identical slices.
    """

    # Get the mapping from dimensions to device layout
    constraint_map = {c.dim: (c.tile_size, c.device_dim) for c in device_constraints}
    host_shape = host_buffer_binding.kernel_buffer_type.symbolic_shape
    device_map = device_map if device_map is not None else {}

    # Tracks constants to avoid repeated constants in the IR
    constant_map = constant_map if constant_map is not None else {}

    # Track unique slices to avoid duplicates
    slice_cache = {}

    # Get the MLIR type to extract static dimensions
    host_type = host_tensor.type
    if not isinstance(host_type, (RankedTensorType, MemRefType)):
        return [host_tensor], {
            0: {"coords": [0] * len(device_layout.dims), "slice": host_tensor}
        }

    # Extract static dimensions from MLIR type
    if isinstance(host_type, MemRefType):
        mlir_shape = [host_type.get_dim_size(i) for i in range(host_type.rank)]
        element_type = host_type.element_type
    elif isinstance(host_type, RankedTensorType):
        mlir_shape = list(host_type.shape)
        element_type = host_type.element_type
    else:
        return [host_tensor], {
            0: {"coords": [0] * len(device_layout.dims), "slice": host_tensor}
        }

    # Calculate total number of devices
    total_devices = 1
    for dim_size in device_layout.dims:
        total_devices *= dim_size

    # Create slices for each device
    for device_id in range(total_devices):
        # Calculate device coordinates in the grid (row-major order)
        # For a 2x2 grid [2, 2]: Device 0: [0, 0], Device 1: [1, 0],
        # Device 2: [0, 1], Device 3: [1, 1]
        device_coords = []
        temp_id = device_id
        for dim_size in device_layout.dims:
            device_coords.append(temp_id % dim_size)
            temp_id //= dim_size

        # Calculate slice parameters for this device
        slice_signature = []  # Will be used as cache key
        start_indices = []
        lengths = []
        result_shape = []
        slice_info = {}

        for i, dim in enumerate(host_shape):
            if dim in constraint_map:
                tile_size, device_dim = constraint_map[dim]
                device_coord = device_coords[device_dim]

                # Calculate tile size from the static dimension and device layout
                host_dim_size = mlir_shape[i]
                devices_in_this_dim = device_layout.dims[device_dim]
                tile_dim_size = host_dim_size // devices_in_this_dim

                # Start index = device_coord * tile_size
                start_offset = device_coord * tile_dim_size
                start_offset_int = (
                    int(start_offset)
                    if hasattr(start_offset, "__int__")
                    else start_offset
                )
                tile_dim_size_int = (
                    int(tile_dim_size)
                    if hasattr(tile_dim_size, "__int__")
                    else tile_dim_size
                )
            else:
                # This dimension is not split across devices
                full_dim_size = mlir_shape[i]
                full_dim_size = (
                    int(full_dim_size)
                    if hasattr(full_dim_size, "__int__")
                    else full_dim_size
                )

                start_offset_int = 0
                tile_dim_size_int = full_dim_size
                device_coord = None
                device_dim = None

            # Add to slice signature for caching
            slice_signature.append((start_offset_int, tile_dim_size_int))
            start_idx = get_or_create_index_constant(start_offset_int, constant_map)
            length = get_or_create_index_constant(tile_dim_size_int, constant_map)

            start_indices.append(start_idx)
            lengths.append(length)
            result_shape.append(tile_dim_size_int)

            # Store slice info for this dimension
            slice_info[f"dim_{i}"] = {
                "symbol": dim,
                "start_offset": start_offset_int,
                "length": tile_dim_size_int,
                "device_coord": device_coord,
                "device_dim": device_dim,
            }

        # Convert slice signature to a hashable key
        slice_key = tuple(slice_signature)

        # check if we've already created this slice
        if slice_key in slice_cache:
            device_slice_result = slice_cache[slice_key]
        else:
            # check if the host_tensor and the slice dimensions match
            if host_type.shape != result_shape:
                # Create a new slice
                result_type = RankedTensorType.get(result_shape, element_type)
                device_slice = flow_d.TensorSliceOp(
                    result_type,
                    host_tensor,
                    [],  # source_dims
                    start_indices,
                    lengths,
                    [],  # result_dims
                )
                device_slice_result = device_slice.result
            else:
                device_slice_result = host_tensor

            transferred_slice = flow_d.TensorTransferOp(
                device_slice_result,
                [],
                Attribute.parse(f"#hal.device.promise<@__device_{device_id}>"),
            )
            device_slice_result = transferred_slice.result

            # Cache the slice
            slice_cache[slice_key] = device_slice_result

        # store device mapping
        if device_id not in device_map:
            device_map[device_id] = []

        device_map[device_id].append(
            {
                "coords": device_coords,
                "slice": device_slice_result,
                "slice_info": slice_info,
                "result_shape": result_shape,
                "binding_name": host_buffer_binding.name,
                "slice_signature": slice_key,
            }
        )

    return device_map, constant_map


def merge_output_slices(
    arguments: list[Value],
    host_sig: HostSignature,
    output_list,
    constant_map: dict,
    device_tensor_map: list[dict],
):

    original_tensor = arguments[
        len(host_sig.buffer_bindings) - len(host_sig.output_buffer_bindings)
    ]
    output_idx = len(host_sig.buffer_bindings) - len(host_sig.output_buffer_bindings)
    result_tensor = original_tensor

    for i, dispatch_result in enumerate(output_list):
        # Get the device map for the output from the i-th device
        # TODO: Handle multiple outputs per device, currently assuming one output per device
        device_info = device_tensor_map[i][output_idx]
        slice_info = device_info["slice_info"]

        # for each dimension in the output tensor, get the start offset and data length
        start_offsets = []
        data_lengths = []
        strides = []

        for dim_key in sorted(slice_info.keys()):  # dim_0, dim_1
            start_offset = slice_info[dim_key]["start_offset"]
            length = slice_info[dim_key]["length"]
            start_offsets.append(start_offset)
            data_lengths.append(length)
            strides.append(1)  # stride of 1 always?

        # Update the result tensor with this device's output
        # tensor_d.InsertSliceOp signature: (source, dest, offsets, sizes, strides, static_offsets, static_sizes, static_strides)
        result_value = tensor_d.InsertSliceOp(
            dispatch_result.results[0],  # source
            result_tensor,  # dest
            [],  # offsets
            [],  # sizes
            [],  # strides
            start_offsets,  # [start_m, start_n]
            data_lengths,  # [tile_m, tile_n]
            strides,  # [1, 1]
        ).result
        result_tensor = result_value

    return [result_tensor]
