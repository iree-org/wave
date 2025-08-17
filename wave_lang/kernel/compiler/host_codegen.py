from typing import Optional

from wave_lang.support.ir_imports import (
    ArrayAttr,
    Block,
    F32Type,
    F64Type,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    MemRefType,
    RankedTensorType,
    SymbolRefAttr,
    Value,
    arith_d,
    flow_d,
    func_d,
    hal_d,
    tensor_d,
)

from .._support.indexing import IndexSymbol
from .._support.location import capture_location
from ...support.location_config import LocationCaptureConfig
from .builder import (
    ModuleBuilder,
)
from .dispatch_codegen import StreamExecutable
from .kernel_codegen import BindingDesc, KernelSignature, BindingType
from ..wave.constraints import DeviceConstraint

from ..lang import Grid
from ..lang.kernel_buffer import KernelBufferUsage
import inspect

from iree.compiler._mlir_libs._mlir.ir import Attribute


def memref_to_tensor(memrefs: list[IrType], use_views: bool = False):
    if use_views:
        view_type = IrType.parse("!hal.buffer_view")

    tensors = []
    for m in memrefs:
        # append scalars as-it-is to tensors list
        if isinstance(m, (F32Type, F64Type, IndexType)) or (
            isinstance(m, IntegerType) and m.is_signless
        ):
            tensors.append(m)
            continue
        assert isinstance(m, MemRefType)
        t = view_type if use_views else RankedTensorType.get(m.shape, m.element_type)
        tensors.append(t)
    return tensors


def get_dynamic_dims(bindings: list[BindingDesc], dynamic_symbols: list[IndexSymbol]):
    dynamic_dims: list[IndexSymbol] = []
    for b in bindings:
        node_type = b.reference[1].type
        if node_type.physical_layout:
            if all(node_type.physical_layout.shape):
                continue
        for dim in b.kernel_buffer_type.symbolic_shape:
            if dim in dynamic_symbols:
                dynamic_dims.append(dim)
    return dynamic_dims


def to_index(v: Value) -> Value:
    t = v.type
    if isinstance(t, IndexType):
        return v

    if isinstance(t, IntegerType):
        return arith_d.index_cast(IndexType.get(), v)

    assert False, f"Expected IndexType or IntegerType, got {t}"


def substitute_dimensions_in_shape(symbolic_shape, symbol_map):
    if isinstance(symbolic_shape, (list, tuple)):
        return [
            substitute_dimensions_in_shape(dim, symbol_map) for dim in symbolic_shape
        ]
    elif symbolic_shape in symbol_map:
        return symbol_map[symbolic_shape]
    else:
        return symbolic_shape


class HostSignature:
    """
    A signature for a host function that can be used to generate code.
    This is used to create a host function that can be called from the kernel.
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


def create_device_tensor_slices(
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

                # Add to slice signature for caching
                slice_signature.append((start_offset_int, tile_dim_size_int))

                if start_offset_int not in constant_map:
                    constant_map[start_offset_int] = arith_d.constant(
                        IndexType.get(),
                        IntegerAttr.get(IndexType.get(), start_offset_int),
                    )
                start_idx = constant_map[start_offset_int]
                start_indices.append(start_idx)

                if tile_dim_size_int not in constant_map:
                    constant_map[tile_dim_size_int] = arith_d.constant(
                        IndexType.get(),
                        IntegerAttr.get(IndexType.get(), tile_dim_size_int),
                    )
                length = constant_map[tile_dim_size_int]
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
            else:
                # This dimension is not split across devices
                full_dim_size = mlir_shape[i]
                full_dim_size = (
                    int(full_dim_size)
                    if hasattr(full_dim_size, "__int__")
                    else full_dim_size
                )

                slice_signature.append((0, full_dim_size))

                if 0 not in constant_map:
                    constant_map[0] = arith_d.constant(
                        IndexType.get(), IntegerAttr.get(IndexType.get(), 0)
                    )
                start_indices.append(constant_map[0])

                if full_dim_size not in constant_map:
                    constant_map[full_dim_size] = arith_d.constant(
                        IndexType.get(), IntegerAttr.get(IndexType.get(), full_dim_size)
                    )
                length = constant_map[full_dim_size]
                lengths.append(length)
                result_shape.append(full_dim_size)

                # Store slice info for this dimension
                slice_info[f"dim_{i}"] = {
                    "symbol": dim,
                    "start_offset": 0,
                    "length": full_dim_size,
                    "device_coord": None,
                    "device_dim": None,
                }

        # Convert slice signature to a hashable key
        slice_key = tuple(slice_signature)

        # check if we've already created this slice
        if 0:
            device_slice_result, cached_slice_info = slice_cache[slice_key]
            print(f"Reusing slice for device {device_id} with signature {slice_key}")
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
            slice_cache[slice_key] = (device_slice_result, slice_info)
            print(
                f"Created new slice for device {device_id} with signature {slice_key}"
            )

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


def isolated_test_call(
    mb: ModuleBuilder,
    exe: StreamExecutable,
    sig: KernelSignature,
    entrypoint: str,
    func_name: str = "isolated_benchmark",
    dynamic_symbols: list[IndexSymbol] = [],
    *,
    location_capture_config: Optional[LocationCaptureConfig] = None,
    async_dispatch: bool = False,
    device_layout: Optional[Grid] = None,
    device_constraints: Optional[list[DeviceConstraint]] = None,
):
    with InsertionPoint(mb.body_block), Location.unknown():

        # Given kernel signature, create a host signature.
        # This will be the same if no device constraints are provided.
        host_sig = HostSignature(sig, device_constraints)

        input_types = [b.as_mlir_type() for b in host_sig.buffer_bindings] + [
            b.as_mlir_type() for b in host_sig.scalar_buffer_bindings
        ]

        input_tensors = memref_to_tensor(input_types, use_views=async_dispatch)
        argument_dims = get_dynamic_dims(host_sig.buffer_bindings, dynamic_symbols)

        # Map dynamic symbols to buffer argument indices and dimensions.
        arg_dim_mapping: dict[IndexSymbol, tuple[int, int]] = {}
        for arg_idx, b in enumerate(host_sig.buffer_bindings):
            shape = b.kernel_buffer_type.symbolic_shape
            for dim_idx, dim_symbol in enumerate(shape):
                if dim_symbol in arg_dim_mapping:
                    continue

                arg_dim_mapping[dim_symbol] = (arg_idx, dim_idx)

        if async_dispatch:
            fence_type = IrType.parse("!hal.fence")
            input_tensors += [fence_type] * 2
            func_name = func_name + "$async"

        output_types = [b.as_mlir_type() for b in host_sig.output_buffer_bindings]
        output_tensors = memref_to_tensor(output_types, use_views=async_dispatch)
        result_dims = get_dynamic_dims(host_sig.output_buffer_bindings, dynamic_symbols)

        ftype = FunctionType.get(input_tensors, output_tensors)
        func_op = func_d.FuncOp(func_name, ftype)
        captured_loc = capture_location(location_capture_config)
        actual_loc = captured_loc.to_mlir() if captured_loc else Location.unknown()
        scalar_bindings = sig.scalar_bindings
        arg_locs = [
            (Location.name(b.name, actual_loc) if b.name is not None else actual_loc)
            for b in sig.kernel_buffer_bindings + scalar_bindings
        ]

        if async_dispatch:
            arg_locs += [Location.unknown()] * 2

        entry_block = func_op.add_entry_block(arg_locs)
        scalars_offset = len(host_sig.buffer_bindings)
        scalars_count = len(scalar_bindings)
        dynamic_offset = scalars_offset + scalars_count

        with InsertionPoint(entry_block):
            arguments = entry_block.arguments
            if async_dispatch:
                in_fence = arguments[-2]
                out_fence = arguments[-1]
                arguments = list(arguments[:-2])

                for i, b in enumerate(host_sig.buffer_bindings):
                    shape = b.kernel_buffer_type.symbolic_shape

                    arg = arguments[i]
                    arg_type = memref_to_tensor([b.as_mlir_type()])[0]
                    target_dims = [
                        hal_d.buffer_view_dim(arg, d)
                        for d in range(len(shape))
                        if arg_type.is_dynamic_dim(d)
                    ]
                    arguments[i] = hal_d.tensor_import(
                        arg_type,
                        arg,
                        wait_fence=in_fence,
                        target_encoding=arg_type,
                        target_dims=target_dims,
                    )

            scalars_args = [
                to_index(v)
                for v, b in zip(
                    arguments[scalars_offset:dynamic_offset], scalar_bindings
                )
                if b.symbol_type is not None
            ]

            # Get the dynamic symbols values from the buffer dimensions.
            dynamic_argument_map: dict[IndexSymbol, Value] = {}
            for symbol in dynamic_symbols:
                arg_idx, dim_idx = arg_dim_mapping[symbol]
                idx = arith_d.constant(IndexType.get(), dim_idx)
                dynamic_argument_map[symbol] = tensor_d.dim(arguments[arg_idx], idx)

            device_tensor_slices = []
            device_tensor_maps = {}  # Store all device maps
            constant_map = {}
            for i, (host_tensor, binding) in enumerate(
                zip(
                    arguments[: len(host_sig.buffer_bindings)], host_sig.buffer_bindings
                )
            ):
                if device_constraints and device_layout:
                    device_tensor_maps, constant_map = create_device_tensor_slices(
                        host_tensor,
                        binding,
                        device_layout,
                        device_constraints,
                        dynamic_argument_map,
                        device_tensor_maps,
                        constant_map,
                    )

            assert isinstance(entry_block, Block)
            # Create a flow.dispatch op to the kernel
            dispatch = SymbolRefAttr.get([exe.sym_name.value, entrypoint])
            entrypoints = ArrayAttr.get([dispatch])

            buffer_binding_count = len(host_sig.buffer_bindings)
            input_binding_count = len(host_sig.input_buffer_bindings)
            tied_operands = ArrayAttr.get(
                [
                    IntegerAttr.get(IndexType.get(), out_idx)
                    for out_idx in range(input_binding_count, buffer_binding_count)
                ]
            )

            out = None
            if device_constraints and device_layout:
                output_list = []
                for i in range(0, len(device_tensor_maps.keys())):
                    block_argument_list = []
                    output_slices = []
                    for arg in device_tensor_maps[i]:
                        block_argument_list.append(arg["slice"])
                        if arg["binding_name"] in [
                            b.name for b in host_sig.output_buffer_bindings
                        ]:
                            # Get the slice shape from the device mapping
                            slice_shape = arg["result_shape"]
                            element_type = arg["slice"].type.element_type
                            output_slices.append(
                                RankedTensorType.get(slice_shape, element_type)
                            )

                    out = flow_d.DispatchOp(
                        output_slices,
                        [dynamic_argument_map[dim] for dim in dynamic_symbols]
                        + scalars_args,
                        entrypoints,
                        block_argument_list,
                        [dynamic_argument_map[dim] for dim in argument_dims],
                        [dynamic_argument_map[dim] for dim in result_dims],
                        tied_operands=tied_operands,
                    )

                    # Create a device affinity attribute
                    # device_name = f"__device_{i}"
                    # affinity_attr_str = f"#hal.device.affinity<@{device_name}>"
                    # affinity_attr = Attribute.parse(affinity_attr_str)
                    # out.attributes["stream.affinity"] = affinity_attr
                    output_list.append(out)

                # Now collect all the results back into the original tensor shape
                if len(output_list) > 1:
                    # getting the orignial output tensor
                    result_tensor = arguments[
                        len(host_sig.buffer_bindings)
                        - len(host_sig.output_buffer_bindings)
                    ]
                    output_idx = len(host_sig.buffer_bindings) - len(
                        host_sig.output_buffer_bindings
                    )

                    for i, dispatch_result in enumerate(output_list):
                        # Get the device coordinates for this result
                        # TODO: Handle multiple outputs per device, currently assuming one output per device
                        device_info = device_tensor_maps[i][output_idx]
                        slice_shape = device_info["result_shape"]
                        slice_info = device_info["slice_info"]
                        offsets = []
                        for dim_key in sorted(slice_info.keys()):  # dim_0, dim_1, etc.
                            start_offset = slice_info[dim_key]["start_offset"]
                            if start_offset not in constant_map:
                                constant_map[start_offset] = arith_d.constant(
                                    IndexType.get(),
                                    IntegerAttr.get(IndexType.get(), start_offset),
                                )
                            offsets.append(constant_map[start_offset])

                        # Update the result tensor with this device's output
                        # flow_d.TensorUpdateOp signature: (target, target_dims, start_indices, update, update_dims)
                        result_value = flow_d.TensorUpdateOp(
                            result_tensor,  # target tensor
                            [],  # target_dims (empty list for dynamic dims)
                            offsets,  # start_indices (where to place it)
                            dispatch_result.results[0],  # update (the slice to insert)
                            [],  # update_dims (empty list for dynamic dims)
                        ).result
                        result_tensor = result_value

                    out = [result_tensor]
                else:
                    out = output_list[0]
            else:
                # If no device constraints, just dispatch to the kernel directly
                breakpoint()
                out = flow_d.DispatchOp(
                    memref_to_tensor(output_types),
                    [dynamic_argument_map[dim] for dim in dynamic_symbols]
                    + scalars_args,
                    entrypoints,
                    arguments,
                    [dynamic_argument_map[dim] for dim in argument_dims],
                    [dynamic_argument_map[dim] for dim in result_dims],
                    tied_operands=tied_operands,
                )

            if async_dispatch:
                out = list(out.results)
                out_types = memref_to_tensor(
                    [b.as_mlir_type() for b in sig.kernel_buffer_output_bindings]
                )
                barrier = hal_d.tensor_barrier(out_types, out, signal_fence=out_fence)
                if len(out_types) == 1:
                    barrier = [barrier]

                view_type = IrType.parse("!hal.buffer_view")
                for i, b in enumerate(sig.kernel_buffer_output_bindings):
                    shape = b.kernel_buffer_type.symbolic_shape

                    out_type = out_types[i]
                    source_dims = [
                        tensor_d.dim(out[i], arith_d.constant(IndexType.get(), d))
                        for d in range(len(shape))
                        if out_type.is_dynamic_dim(d)
                    ]
                    out[i] = hal_d.tensor_export(
                        view_type, barrier[i], out_type, source_dims=source_dims
                    )

            func_d.ReturnOp(out)
