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
        DivOp,
        Exp2Op,
        MulOp,
        ReadOp,
        RegisterOp,
        WriteOp,
    )
    from water_mlir.water_mlir.sympy_to_affine_converter import (
        convert_sympy_to_affine_map,
    )
    from water_mlir.water_mlir.dialects import arith
    from water_mlir.water_mlir.dialects import func
    from water_mlir.water_mlir.dialects import wave
except Exception as e:
    print(f"FATAL: failed to import water_mlir: {e}", file=sys.stderr)
    sys.exit(1)

# Mapping from tkw_op_name to actual op constructors
WAVE_OP_CONSTRUCTORS = {
    "add": AddOp,
    "mul": MulOp,
    "div": DivOp,
    "exp2": Exp2Op,
    "read": ReadOp,
    "write": WriteOp,
    "register": RegisterOp,
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


def _read_trace() -> tuple[CapturedTrace, WaveCompileOptions]:
    """Reads and returns a pickled trace and options from stdin.

    The input is expected to be a dill-serialized dict with keys:
    - "trace": CapturedTrace object
    - "options": WaveCompileOptions

    Restores supplemental fx.Node fields (e.g., .type) from node.meta.
    """
    try:
        unpickled = dill.loads(sys.stdin.buffer.read())
    except Exception as e:
        raise SystemExit(f"FATAL: failed to unpickle: {e}")
    trace = unpickled.get("trace") if isinstance(unpickled, dict) else None
    options = unpickled.get("options") if isinstance(unpickled, dict) else None

    # TODO: Properly importing this unconditionally at top of the file still
    #       clashes with IREE bindings.
    from wave_lang.kernel._support.tracing import CapturedTrace
    from wave_lang.kernel.wave.compile_options import WaveCompileOptions

    if not isinstance(trace, CapturedTrace) or not isinstance(
        options, WaveCompileOptions
    ):
        raise SystemExit(
            f"FATAL: unpickled objects are not CapturedTrace and WaveCompileOptions (got {type(trace)} and {type(options)})"
        )

    # Restore supplemental node fields captured in the meta field
    trace.restore_node_state()
    return trace, options


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
    if getattr(node, "index", None):
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
            bounds[dim.name] = wave.WaveWaveExprListAttr.get(
                [sym.name for sym in symbol_mapping.values()], result
            )
        op.attributes["bounds"] = wave.WaveReadWriteBoundsAttr.get(bounds)


def _emit_from_captured_trace(
    trace: "CapturedTrace", options: WaveCompileOptions
) -> int:
    # Import wave types locally to avoid clashing with iree bindings
    from wave_lang.kernel.ops.wave_ops import get_custom, Placeholder, Output, Write, NewRegister  # type: ignore

    # keep track of which emitted value stems from what node to wire
    # arguments correctly
    value_map: dict[fx.Node | fx.Proxy, ir.Value] = {}

    # TODO: Forward locations properly
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = False
        wave.register_dialect(ctx)

        module = ir.Module.create()
        root_graph: fx.Graph = trace.get_root_graph()

        # Collect placeholders from root graph
        placeholders = [
            n
            for n in trace.walk()
            if getattr(n, "op", "") == "placeholder"
            and getattr(n, "graph", None) is root_graph
        ]

        # Build function argument types from placeholders
        # TODO: This might need special handling when iterate nodes are present
        arg_types = []
        for p in placeholders:
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
            for i, fx_node in enumerate(placeholders):
                value_map[fx_node] = entry_block.arguments[i]

            with ir.InsertionPoint(entry_block):
                # Emit in original order to preserve dependencies
                for fx_node in root_graph.nodes:
                    node = get_custom(fx_node)
                    # Ensure types are inferred
                    node.infer_type()

                    # No MLIR ops are emitted for placeholder and output nodes
                    if isinstance(node, Placeholder | Output):
                        continue

                    # Collect already emitted mlir values for the args of this node
                    operands = [
                        mlir_arg
                        for arg in fx_node.args
                        if (mlir_arg := value_map.get(arg)) is not None
                    ]
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
                            mlir_op = op_builder(
                                value_map[node.register_], value_map[node.memory]
                            )
                        elif isinstance(node, NewRegister):
                            dtype = getattr(node, "dtype", None)
                            if dtype is None:
                                raise RuntimeError("Register op missing dtype")
                            element_type = _dtype_to_mlir_scalar_type(dtype)
                            constant_op = arith.ConstantOp(
                                result=element_type, value=node.value
                            )
                            mlir_op = op_builder(result_type, constant_op.results[0])
                        else:
                            try:
                                mlir_op = op_builder(result_type, *operands)
                            except Exception:
                                raise RuntimeError(
                                    f"Could not map arguments correctly for MLIR constructor of '{node.tkw_op_name}' operation"
                                )

                    if mlir_op is None:
                        raise RuntimeError(
                            f"Missing support for '{node.tkw_op_name}' operation"
                        )

                    _attach_attributes(node, mlir_op.operation)

                    # Add results to the value map in case they are used as
                    # operands later
                    if len(mlir_op.results) > 1:
                        # TODO: rework value_map to always map to a sequence of results
                        raise RuntimeError(
                            f"Missing support for operations with multiple results"
                        )
                    for result in mlir_op.results:
                        value_map[fx_node] = result

                # Finally emit func.return operation
                func.ReturnOp(operands_=[])

        # Verify the module before printing
        # TODO: Report back diagnostics emitted by the verification
        try:
            module.operation.verify()
        except ir.MLIRError as e:
            raise RuntimeError(f"Emitted MLIR module does not verify") from e
        sys.stdout.write(str(module))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    data, options = _read_trace()
    sys.exit(_emit_from_captured_trace(data, options))
