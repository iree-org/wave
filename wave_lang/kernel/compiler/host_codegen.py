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

import sympy

from .._support.indexing import IndexSymbol
from ..wave.utils.general_utils import infer_dim
from ...support.location_config import LocationCaptureConfig
from .builder import (
    ModuleBuilder,
)
from .dispatch_codegen import StreamExecutable
from .kernel_codegen import (
    BindingDesc,
    KernelSignature,
    create_argument_locations,
)
from ..wave.constraints import DeviceConstraint
from .host_utils import HostSignature, split_input_tensors, merge_output_slices

from ..lang import Grid


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


def _contains_dynamic_symbol(
    expr: sympy.Basic, dynamic_symbols: list[IndexSymbol]
) -> bool:
    """Check if a shape expression contains any dynamic symbol."""
    if expr in dynamic_symbols:
        return True
    if hasattr(expr, "free_symbols"):
        return bool(expr.free_symbols & set(dynamic_symbols))
    return False


def get_dynamic_dims(bindings: list[BindingDesc], dynamic_symbols: list[IndexSymbol]):
    dynamic_dims: list[IndexSymbol] = []
    for b in bindings:
        node_type = b.reference[1].type
        if node_type.physical_layout:
            if all(node_type.physical_layout.shape):
                continue
        for dim in b.kernel_buffer_type.symbolic_shape:
            if _contains_dynamic_symbol(dim, dynamic_symbols):
                dynamic_dims.append(dim)
    return dynamic_dims


def to_index(v: Value) -> Value:
    t = v.type
    if isinstance(t, IndexType):
        return v

    if isinstance(t, IntegerType):
        return arith_d.index_cast(IndexType.get(), v)

    assert False, f"Expected IndexType or IntegerType, got {t}"


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
        # For derived shapes like K/2, also store the inverse expression
        # so we can recover K from the buffer dimension at runtime.
        arg_dim_mapping: dict[IndexSymbol, tuple[int, int]] = {}
        # Maps symbol -> sympy expression to recover it from the dim value.
        # For direct matches (M in shape[M, ...]) this is just a dummy d.
        # For derived (K/2 in shape[M, K/2]) this is e.g. 2*d.
        _dim_val = sympy.Symbol("_dim_val")
        arg_dim_inverse: dict[IndexSymbol, sympy.Expr] = {}
        for arg_idx, b in enumerate(host_sig.buffer_bindings):
            shape = b.kernel_buffer_type.symbolic_shape
            for dim_idx, dim_expr in enumerate(shape):
                base_sym = infer_dim(dim_expr)
                if base_sym in arg_dim_mapping:
                    continue
                arg_dim_mapping[base_sym] = (arg_idx, dim_idx)
                if dim_expr == base_sym:
                    arg_dim_inverse[base_sym] = _dim_val
                else:
                    # Solve shape_expr = d for the base symbol.
                    solutions = sympy.solve(dim_expr - _dim_val, base_sym)
                    assert len(solutions) == 1, (
                        f"Cannot solve {dim_expr} = _dim_val for {base_sym}"
                    )
                    arg_dim_inverse[base_sym] = solutions[0]

        if async_dispatch:
            fence_type = IrType.parse("!hal.fence")
            input_tensors += [fence_type] * 2
            func_name = func_name + "$async"

        output_types = [b.as_mlir_type() for b in host_sig.output_buffer_bindings]
        output_tensors = memref_to_tensor(output_types, use_views=async_dispatch)
        result_dims = get_dynamic_dims(host_sig.output_buffer_bindings, dynamic_symbols)

        ftype = FunctionType.get(input_tensors, output_tensors)
        func_op = func_d.FuncOp(func_name, ftype)
        scalar_bindings = sig.scalar_bindings
        arg_locs = create_argument_locations(
            sig.kernel_buffer_bindings + scalar_bindings
        )

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
            # For derived shapes (K/2), apply the inverse expression to
            # recover the original symbol value.
            dynamic_argument_map: dict[IndexSymbol, Value] = {}
            for symbol in dynamic_symbols:
                arg_idx, dim_idx = arg_dim_mapping[symbol]
                idx = arith_d.constant(IndexType.get(), dim_idx)
                dim_value = tensor_d.dim(arguments[arg_idx], idx)
                inverse_expr = arg_dim_inverse[symbol]
                if inverse_expr == _dim_val:
                    dynamic_argument_map[symbol] = dim_value
                else:
                    # Emit the inverse expression (e.g. _dim_val * 2 for K/2).
                    from .wave_codegen.emitter import gen_sympy_index
                    subs = {_dim_val: dim_value}
                    dynamic_argument_map[symbol] = gen_sympy_index(subs, inverse_expr)

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
                device_tensor_map = {}  # Store all device maps
                constant_map = {}
                # If device constraints are provided, we need to split the input tensors
                # into device-specific tensors.
                for i, binding in enumerate(host_sig.buffer_bindings):
                    host_tensor = arguments[i]
                    device_tensor_map, constant_map = split_input_tensors(
                        host_tensor,
                        binding,
                        device_layout,
                        device_constraints,
                        dynamic_argument_map,
                        device_tensor_map,
                        constant_map,
                    )

                # flow dispatch to each device and collect the results
                output_list = []
                for i in range(0, len(device_tensor_map.keys())):
                    block_argument_list = []
                    output_slices = []
                    # for each device where the kernel is dispatched
                    # collect the arguments from the device tensor map
                    # and append inputs to block_arugment_list
                    # and outputs to output_slices
                    for arg in device_tensor_map[i]:
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

                    output_list.append(out)

                # Now collect all the results back into the original tensor shape
                if len(output_list) > 1:
                    out = merge_output_slices(
                        arguments,
                        host_sig,
                        output_list,
                        constant_map,
                        device_tensor_map,
                    )
                else:
                    out = output_list[0]
            else:
                # If no device constraints, just dispatch the kernel directly
                # with the provided host signature arguments.
                from .wave_codegen.emitter import gen_sympy_index as _gen

                def _resolve_dim(expr):
                    """Resolve a shape expression to an IR value."""
                    if expr in dynamic_argument_map:
                        return dynamic_argument_map[expr]
                    return _gen(dynamic_argument_map, expr)

                out = flow_d.DispatchOp(
                    memref_to_tensor(output_types),
                    [dynamic_argument_map[dim] for dim in dynamic_symbols]
                    + scalars_args,
                    entrypoints,
                    list(arguments) + list(dynamic_argument_map.values()),
                    [_resolve_dim(dim) for dim in argument_dims],
                    [_resolve_dim(dim) for dim in result_dims],
                    tied_operands=tied_operands,
                )

                if async_dispatch:
                    out = list(out.results)

            if async_dispatch:
                out_types = memref_to_tensor(
                    [b.as_mlir_type() for b in host_sig.output_buffer_bindings]
                )
                barrier = hal_d.tensor_barrier(out_types, out, signal_fence=out_fence)
                if len(out_types) == 1:
                    barrier = [barrier]

                view_type = IrType.parse("!hal.buffer_view")

                exported_tensors = []
                for i, b in enumerate(host_sig.output_buffer_bindings):
                    shape = b.kernel_buffer_type.symbolic_shape
                    out_type = out_types[i]
                    source_dims = [
                        tensor_d.dim(out[i], arith_d.constant(IndexType.get(), d))
                        for d in range(len(shape))
                        if out_type.is_dynamic_dim(d)
                    ]
                    exported_tensor = hal_d.tensor_export(
                        view_type, barrier[i], out_type, source_dims=source_dims
                    )
                    exported_tensors.append(exported_tensor)

                out = exported_tensors

            func_d.ReturnOp(out)
