"""
Water Emitter for Wave Dialect

This generates operations of the Wave dialect from a serialized wave trace.
It runs as a standalone process with access to Water Python bindings and emits
mlir operations of wave and other dialects.
"""

from __future__ import annotations
import dill
import torch.fx as fx
import sys
from typing import TYPE_CHECKING, Callable, Sequence
import sympy
import argparse


if TYPE_CHECKING:
    from wave_lang.kernel._support.tracing import CapturedTrace
    from wave_lang.kernel.wave.compile_options import WaveCompileOptions
    from wave_lang.kernel.lang.wave_types import Memory, Register
    from wave_lang.kernel._support import dtype
    from wave_lang.kernel.ops.wave_ops import *

try:
    from water_mlir.water_mlir import ir
    from water_mlir.water_mlir.dialects.wave import (
        AddOp,
        AllocateOp,
        DivOp,
        ExtractSliceOp,
        Exp2Op,
        MmaOp,
        MulOp,
        ReadOp,
        RegisterOp,
        WriteOp,
        IterateOp,
        YieldOp,
        WaveExprListAttr,
    )
    from water_mlir.water_mlir.sympy_to_affine_converter import (
        convert_sympy_to_affine_map,
    )
    from water_mlir.water_mlir.dialects import arith
    from water_mlir.water_mlir.dialects import func
    from water_mlir.water_mlir.dialects import wave
    from water_mlir.water_mlir.dialects import amdgpu
except Exception as e:
    print(f"FATAL: failed to import water_mlir: {e}", file=sys.stderr)
    sys.exit(1)

# Mapping from tkw_op_name to actual op constructors
WAVE_OP_CONSTRUCTORS = {
    "add": AddOp,
    "allocate": AllocateOp,
    "extract_slice": ExtractSliceOp,
    "mma": MmaOp,
    "mul": MulOp,
    "div": DivOp,
    "exp2": Exp2Op,
    "read": ReadOp,
    "register": RegisterOp,
    "iterate": IterateOp,
    "output": YieldOp,
    "write": WriteOp,
    # TODO: Add more or find a good way of avoiding needing a mapping
}

# TODO: Rework wave dtype.Datatype so this does not need to rely on strings
# This maps to a lambda to create the respective MLIR type. An MLIR context
# is required to invoke the lambda.
DATATYPE_MAP: dict[str, Callable[[], ir.Type]] = {
    "f16": ir.F16Type.get,
    "f32": ir.F32Type.get,
    "f64": ir.F64Type.get,
    "i1": lambda: ir.IntegerType.get_signless(1),
    "i32": lambda: ir.IntegerType.get_signless(32),
    "i64": lambda: ir.IntegerType.get_signless(64),
}


def _dtype_to_mlir_scalar_type(t: dtype.Datatype) -> ir.Type:
    """Convert a Wave dtype (DataType) to a scalar MLIR ir.Type

    An active MLIR context is required for this."""

    # TODO: Move this into a decorator
    assert ir.Context.current is not None

    if str(t) in DATATYPE_MAP:
        return DATATYPE_MAP[str(t)]()

    raise RuntimeError(f"Unsupported scalar dtype: {t}")


def _map_address_space(addr: str) -> str:
    # TODO: Use WaveAddressSpaceAttr bindings when they exist
    if not addr:
        return ""
    a = str(addr).lower()
    if "global" in a:
        return "global"
    if "shared" in a:
        return "shared"
    if "register" in a:
        return "register"
    # If we can't determine the address space, use unspecified for now
    # TODO: Sometimes the address space is determined in the hyperparameters
    #       Check whether a mapping for `addr` is defined there.
    return "unspecified"


def _type_to_wave_mlir(
    ctx: ir.Context, type_: type[Register] | type[Memory]
) -> ir.Type:
    # TODO: Use WaveTensorType bindings when they exist
    # TODO: Properly importing this unconditionally at top of the file still
    #       clashes with IREE bindings.
    from wave_lang.kernel.lang.wave_types import Memory, Register

    # Map Python Wave types to MLIR types
    if issubclass(type_, Register):
        if not type_.symbolic_shape or type_.dtype is None:
            raise RuntimeError(
                "Register type must have concrete symbolic_shape and dtype"
            )
        shape_txt = ", ".join([f"@{s}" for s in type_.symbolic_shape])
        return ir.Type.parse(
            f"!wave.tensor<[{shape_txt}] of {type_.dtype}, <register>>", context=ctx
        )
    if issubclass(type_, Memory):
        if (
            not type_.symbolic_shape
            or type_.dtype is None
            or type_.address_space is None
        ):
            raise RuntimeError(
                "Memory type must have concrete symbolic_shape, dtype, and address_space"
            )
        addr_txt = _map_address_space(type_.address_space)
        shape_txt = ", ".join([f"@{s}" for s in type_.symbolic_shape])
        return ir.Type.parse(
            f"!wave.tensor<[{shape_txt}] of {type_.dtype}, <{addr_txt}>>", context=ctx
        )
    raise RuntimeError(f"Unsupported wave type for MLIR conversion: {type_}")


def _parse_input() -> tuple[CapturedTrace, WaveCompileOptions, str]:
    """Parses and returns the pickled trace, options, and pipeline from stdin.

    The input is expected to be a dill-serialized dict with keys:
    - "trace": CapturedTrace object
    - "options": WaveCompileOptions
    - "pipeline": A string containing the transform dialect pass pipeline

    Restores supplemental fx.Node fields (e.g., .type) from node.meta.
    """
    try:
        unpickled = dill.loads(sys.stdin.buffer.read())
    except Exception as e:
        raise SystemExit(f"FATAL: failed to unpickle: {e}")
    trace = unpickled.get("trace") if isinstance(unpickled, dict) else None
    options = unpickled.get("options") if isinstance(unpickled, dict) else None
    pipeline = unpickled.get("pipeline") if isinstance(unpickled, dict) else None

    # TODO: Properly importing this unconditionally at top of the file still
    #       clashes with IREE bindings.
    from wave_lang.kernel._support.tracing import CapturedTrace
    from wave_lang.kernel.wave.compile_options import WaveCompileOptions

    if not isinstance(trace, CapturedTrace):
        raise SystemExit(
            f"FATAL: unpickled object is not CapturedTrace (got {type(trace)})"
        )

    if not isinstance(options, WaveCompileOptions):
        raise SystemExit(
            f"FATAL: unpickled object is not WaveCompileOptions (got {type(options)})"
        )

    if not isinstance(pipeline, str):
        raise SystemExit(f"FATAL: unpickled object is not str (got {type(pipeline)})")

    # Restore supplemental node fields captured in the meta field
    trace.restore_node_state()
    return trace, options, pipeline


def _convert_sympy_expr_to_affine_map(
    expr: sympy.Expr | int, symbol_mapping: dict[sympy.Symbol, sympy.Symbol]
) -> ir.AffineMap:
    """Converts sympy expressions to affine maps, handling the case
    where the expression is already simplified to integer.
    """
    if isinstance(expr, int):
        return ir.AffineMap.get(
            0, len(symbol_mapping), [ir.AffineExpr.get_constant(expr)]
        )

    # Simplify the expression with the assumption that all symbols are positive. This allows for rewriting, for instance,
    # `Max(1, ceiling(x/2))` into `ceiling(x/2)`.
    expr = expr.subs(symbol_mapping)
    expr = sympy.simplify(expr)

    return convert_sympy_to_affine_map(
        expr, [sym.name for sym in symbol_mapping.values()]
    )


def _preprocess_symbols(
    symbols: Sequence[sympy.Symbol],
) -> dict[sympy.Symbol, sympy.Symbol]:
    """
    Preprocess symbols by:
    (1) adding assumptions about all symbols being positive to later enable more simplifications.
    (2) replacing `$` prefix of special symbols (e.g. `$WG0`) by `_` since MLIR affine expressions
    do not accept `$`.
    """
    return {
        sym: sympy.Symbol(sym.name.replace("$", "_"), positive=True) for sym in symbols
    }


def _attach_attributes(node: CustomOp, op: ir.Operation):
    from wave_lang.kernel.ops.wave_ops import MMA

    if isinstance(node, MMA):
        # TODO: Have special handling for MMA index as it uses Piecewise.
        return

    if getattr(node, "index", None) and isinstance(node.index, dict):
        index_mappings = {}
        for dim, exprs in node.index.items():
            all_symbols = list(
                set().union(
                    *[
                        expr.free_symbols
                        for expr in [exprs.start, exprs.size, exprs.stride]
                        if isinstance(expr, sympy.Expr)
                    ]
                )
            )
            symbol_mapping = _preprocess_symbols(all_symbols)
            start = _convert_sympy_expr_to_affine_map(exprs.start, symbol_mapping)
            size = _convert_sympy_expr_to_affine_map(exprs.size, symbol_mapping)
            stride = _convert_sympy_expr_to_affine_map(exprs.stride, symbol_mapping)
            index_mappings[dim.name] = wave.WaveIndexMappingAttr.get(
                [sym.name for sym in symbol_mapping.values()], start, size, stride
            )
        op.attributes["index"] = ir.DictAttr.get(index_mappings)

    if getattr(node, "elements_per_thread", None):
        op.attributes["wave.elements_per_thread"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(32), node.elements_per_thread
        )

    if getattr(node, "bounds", None):
        bounds = {}
        for dim, expr in node.bounds.items():
            symbol_mapping = _preprocess_symbols(
                list(expr.free_symbols) if isinstance(expr, sympy.Expr) else []
            )
            result = _convert_sympy_expr_to_affine_map(expr, symbol_mapping)
            bounds[dim.name] = wave.WaveExprListAttr.get(
                [sym.name for sym in symbol_mapping.values()], result
            )
        op.attributes["bounds"] = wave.WaveReadWriteBoundsAttr.get(bounds)


def _convert_to_wave_expr_list_tuple(
    exprs: Sequence[sympy.Expr], ctx: ir.Context
) -> WaveExprListAttr:
    """
    Returns a WaveExprListAttr from a sequence of wave IndexExpr.
    """
    symbols = list(
        set().union(
            *[
                expr.free_symbols if isinstance(expr, sympy.Expr) else []
                for expr in exprs
            ]
        )
    )
    return ir.Attribute.parse(
        f"#wave.expr_list<{symbols} -> ({', '.join(map(str, exprs))})>", context=ctx
    )


def _emit_ops_from_graph(
    graph: fx.Graph,
    trace: CapturedTrace,
    value_map: dict[fx.Node | fx.Proxy, ir.Value],
    ctx: ir.Context,
):
    # Import wave types locally to avoid clashing with iree bindings
    from wave_lang.kernel.ops.wave_ops import (
        get_custom,
        Allocate,
        ExtractSlice,
        GetResult,
        Output,
        Placeholder,
        Write,
        MMA,
        NewRegister,
        Iterate,
        SharedMemoryBarrier,
    )

    # Emit in original order to preserve dependencies
    for fx_node in graph.nodes:
        node = get_custom(fx_node)
        # Ensure types are inferred
        node.infer_type()

        # No MLIR ops are emitted for placeholder and output nodes
        if isinstance(node, Placeholder | Output):
            continue

        # Collect already emitted mlir values for the args of this node
        mlir_operands = [
            mlir_arg
            for arg in fx_node.args
            if (mlir_arg := value_map.get(arg)) is not None
        ]
        if isinstance(node, GetResult):
            # Map to correct result of the corresponding iterate node
            iterate_op = value_map[node.value].owner
            # Create mapping to correct result
            if node.res_idx >= len(iterate_op.results):
                raise RuntimeError(
                    f"GetResult index is higher than number of results of corresponding iterate node ({node.res_idx} vs {len(iterate_op.results)})"
                )
            value_map[fx_node] = iterate_op.results[node.res_idx]
            # additional handling for this op is not needed, skip rest
            continue
        if isinstance(node, SharedMemoryBarrier):
            # TODO: For now we simply emit this here. This might need more careful handling in the future
            amdgpu.lds_barrier()
            # additional handling for this op is not needed, skip rest
            continue

        result_type = _type_to_wave_mlir(ctx, node.type)

        mlir_op = None
        if node.tkw_op_name in WAVE_OP_CONSTRUCTORS:
            # The general case is to pass `result_type` followed by
            # the unpacked operands. MLIR constructors that do not
            # follow this structure need special casing.
            # (e.g. operations like Write, which do not have results
            # and thus don't take `result_type` as argument)
            op_builder = WAVE_OP_CONSTRUCTORS[node.tkw_op_name]
            # TODO: Add special handling for Iterate node
            if isinstance(node, Write):
                mlir_op = op_builder(value_map[node.register_], value_map[node.memory])
            elif isinstance(node, NewRegister):
                dtype = getattr(node, "dtype", None)
                if dtype is None:
                    raise RuntimeError("Register op missing dtype")
                element_type = _dtype_to_mlir_scalar_type(dtype)
                constant_op = arith.ConstantOp(result=element_type, value=node.value)
                mlir_op = op_builder(result_type, constant_op.results[0])
            elif isinstance(node, Iterate):
                axis = wave.WaveSymbolAttr.get(node.axis.name)
                carried_values = [value_map.get(arg) for arg in node.init_args]

                result_types = []
                outputs = node.outputs()
                for fx_output in outputs:
                    output = get_custom(fx_output)
                    output.infer_type()
                    result_types.append(_type_to_wave_mlir(ctx, output.type))

                mlir_op = op_builder(result_types, axis, carried_values)
                body = ir.Block.create_at_start(mlir_op.regions[0], result_types)

                for idx, iter_arg in enumerate(node.iter_args()):
                    iter_arg.iter_idx = idx

                # add mapping for iter args
                for wave_arg, mlir_arg in zip(node.iter_args(), body.arguments):
                    value_map[wave_arg] = mlir_arg

                # Emit subgraph of the iterate node
                with ir.InsertionPoint(body):
                    _emit_ops_from_graph(
                        trace.get_subgraph(node.subgraph_name), trace, value_map, ctx
                    )

                    # create YieldOp
                    YieldOp([value_map[output] for output in outputs])
            elif isinstance(node, MMA):
                if node.mma_type is None:
                    raise RuntimeError("MMA op missing mma_type")
                mma_kind = ir.Attribute.parse(
                    f"#wave.mma_kind<{node.mma_type.name.lower()}>", context=ctx
                )
                mlir_op = op_builder(result_type, *mlir_operands, mma_kind)
            elif isinstance(node, Allocate):
                mlir_op = op_builder(
                    result_type,
                    distributed_shape=_convert_to_wave_expr_list_tuple(
                        node.distributed_shape, ctx
                    ),
                )
            elif isinstance(node, ExtractSlice):
                size = _convert_to_wave_expr_list_tuple(node.size, ctx)
                stride = _convert_to_wave_expr_list_tuple(node.stride, ctx)
                offset = _convert_to_wave_expr_list_tuple(node.offset, ctx)
                mlir_op = op_builder(result_type, *mlir_operands, size, stride, offset)
            else:
                try:
                    mlir_op = op_builder(result_type, *mlir_operands)
                except Exception:
                    raise RuntimeError(
                        f"Could not map arguments correctly for MLIR constructor of '{node.tkw_op_name}' operation"
                    )

        if mlir_op is None:
            raise NotImplementedError(
                f"Missing support for '{node.tkw_op_name}' operation"
            )

        _attach_attributes(node, mlir_op.operation)

        # Add results to the value map in case they are used as
        # operands later
        if len(mlir_op.results) > 1:
            # TODO: rework value_map to always map to a sequence of results
            raise NotImplementedError(
                f"Missing support for operations with multiple results"
            )
        for result in mlir_op.results:
            value_map[fx_node] = result


def _emit_from_captured_trace(
    trace: "CapturedTrace",
    options: WaveCompileOptions,
    pipeline: str,
    test_diagnostics=False,
) -> int:
    from wave_lang.kernel.ops.wave_ops import get_custom, IterArg  # type: ignore

    # keep track of which emitted value stems from what node to wire
    # arguments correctly
    value_map: dict[fx.Node | fx.Proxy, ir.Value] = {}

    if pipeline:
        raise NotImplementedError(
            "Transform dialect pipelines are not implemented yet."
        )

    # TODO: Forward locations properly
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = False
        wave.register_dialect(ctx)
        module = ir.Module.create()

        diagnostics = []

        def diagnostics_handler(d):
            diagnostics.append(f"{d.location}: {d.message}")
            return True

        ctx.attach_diagnostic_handler(diagnostics_handler)

        if test_diagnostics:
            loc = ir.Location.unknown(ctx)
            loc.emit_error("test error")

        # Collect placeholders from graph
        placeholders = [
            n for n in trace.walk() if getattr(n, "op", "") == "placeholder"
        ]
        top_level_placeholders = [
            p
            for p in placeholders
            if getattr(p, "graph", None) is trace.get_root_graph()
        ]
        top_level_names = [p.name for p in top_level_placeholders]

        # Build function argument types from top-level placeholders
        arg_types = []
        for p in top_level_placeholders:
            c = get_custom(p)
            t = getattr(c, "_type", None) or getattr(c, "type", None)
            arg_types.append(_type_to_wave_mlir(ctx, t))

        # Return type of the function is always empty
        func_type = ir.FunctionType.get(arg_types, [])
        with ir.InsertionPoint(module.body):
            func_op = func.FuncOp("kernel", func_type)
            # TODO: WaveHyperparameterAttr only supports int currently, so we
            #       lose mappings like ADDRESS_SPACE_A: SHARED_ADDRESS_SPACE
            # Convert the symbols in subs to strings and attach as
            # WaveHyperparameterAttr to func_op
            func_op.operation.attributes["wave.hyperparameters"] = (
                wave.WaveHyperparameterAttr.get(
                    {str(k): v for k, v in options.subs.items() if isinstance(v, int)}
                )
            )
            entry_block = ir.Block.create_at_start(func_op.regions[0], arg_types)

            # Map placeholders to function arguments
            for i, fx_node in enumerate(top_level_placeholders):
                value_map[fx_node] = entry_block.arguments[i]

            # Subgraphs duplicate the placeholders of surrounding graphs so there
            # are multiple placeholders representing the same values.
            # Add mapping for these repeated placeholders as well
            for nested_placeholder in placeholders:
                if nested_placeholder in top_level_placeholders:
                    continue
                if isinstance(get_custom(nested_placeholder), IterArg):
                    continue
                # With top-level placeholders and iterargs filtered out the remaining
                # placeholders are duplicates. Find the original one by name
                if not nested_placeholder.name in top_level_names:
                    raise RuntimeError(
                        f"Incorrectly structured placeholders in trace: "
                        f"placeholder '{nested_placeholder.name}' not found in top-level names {top_level_names}."
                    )
                value_map[nested_placeholder] = value_map[
                    top_level_placeholders[
                        top_level_names.index(nested_placeholder.name)
                    ]
                ]

            with ir.InsertionPoint(entry_block):
                _emit_ops_from_graph(trace.get_root_graph(), trace, value_map, ctx)
                func.ReturnOp(operands_=[])

        # Verify the module before printing
        try:
            module.operation.verify()
        except ir.MLIRError as e:
            diagnostics.append(str(e))

        output = dill.dumps({"diagnostics": diagnostics, "module": str(module)})
        sys.stdout.buffer.write(output)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water dialect emitter")

    parser.add_argument(
        "--test-diagnostic-emission",
        action="store_true",
        help="Test diagnostic serialization and deserialization through stdin and stdout",
    )

    args = parser.parse_args()

    trace, options, pipeline = _parse_input()
    sys.exit(
        _emit_from_captured_trace(
            trace, options, pipeline, args.test_diagnostic_emission
        )
    )
