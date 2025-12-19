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
from pathlib import Path

if __name__ == "__main__":
    # Add parent directory to sys.path to enable relative imports when running standalone
    # This allows importing water_mlir from ../water_mlir/ as if it were a relative import
    _current_file = Path(__file__).resolve()
    _parent_dir = str(_current_file.parent.parent)  # Go up to wave_lang/kernel/wave/
    if _parent_dir not in sys.path:
        sys.path.append(_parent_dir)
    # Add current directory to enable importing mlir_to_wave without full package path
    _current_dir = str(_current_file.parent)
    if _current_dir not in sys.path:
        sys.path.append(_current_dir)

from mlir_to_wave import (
    INDEX_SYMBOL_MAP,
    ITER_SYMBOL_NAME_WATER_PREFIX,
    convert_index_mapping_array_to_sympy,
    ITER_SYMBOL_NAME_WAVE_PREFIX,
)


if TYPE_CHECKING:
    from wave_lang.kernel._support.indexing import IndexSequence, IndexSymbol
    from wave_lang.kernel._support import dtype
    from wave_lang.kernel.ops.wave_ops import *

from wave_lang.support.location_config import LocationCaptureLevel
from wave_lang.kernel.lang.wave_types import Memory, Register
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel._support.indexing import safe_subs
from wave_lang.kernel.wave.utils.symbol_utils import get_induction_symbol

from wave_lang.kernel.ops.wave_ops import (
    Allocate,
    ExtractSlice,
    get_custom,
    GetResult,
    IterArg,
    Iterate,
    MMA,
    NewRegister,
    Output,
    Placeholder,
    SharedMemoryBarrier,
    Write,
)
from wave_lang.kernel.wave.constraints import (
    Constraint,
    DeviceConstraint,
    HardwareConstraint,
    TilingConstraint,
    WaveConstraint,
    WorkgroupConstraint,
)
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)

assert "iree" not in sys.modules, (
    "IREE was transitively imported into the water emitter, "
    + "which will lead to MLIR library clashes. "
    + "Please clean up the import list so as not to import IREE."
)

try:
    from water_mlir.water_mlir import ir
    from water_mlir.water_mlir.dialects.wave import (
        AddOp,
        AllocateOp,
        DivOp,
        Exp2Op,
        ExtractSliceOp,
        IterateOp,
        MmaOp,
        MulOp,
        ReadOp,
        RegisterOp,
        WriteOp,
        YieldOp,
        DeviceConstraintAttr,
        HardwareConstraintAttr,
        TilingConstraintAttr,
        WaveConstraintAttr,
        WaveExprListAttr,
        WaveMmaKind,
        WaveMmaKindAttr,
        WaveWorkgroupDimAttr,
        WorkgroupConstraintAttr,
    )
    from water_mlir.water_mlir.sympy_to_affine_converter import (
        convert_sympy_to_affine_map,
    )
    from water_mlir.water_mlir.dialects import arith
    from water_mlir.water_mlir.dialects import func
    from water_mlir.water_mlir.dialects import wave
    from water_mlir.water_mlir.dialects import amdgpu
    from water_mlir.water_mlir.dialects.transform import interpreter
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


def _parse_input() -> tuple[CapturedTrace, list[Constraint], WaveCompileOptions, str]:
    """Parses and returns the pickled trace, options, and pipeline from stdin.

    The input is expected to be a dill-serialized dict with keys:
    - "trace": CapturedTrace object
    - "constraints": list[Constraint]
    - "options": WaveCompileOptions
    - "pipeline": A string containing the transform dialect pass pipeline

    Restores supplemental fx.Node fields (e.g., .type) from node.meta.
    """
    try:
        unpickled = dill.loads(sys.stdin.buffer.read())
    except Exception as e:
        raise SystemExit(f"FATAL: failed to unpickle: {e}")
    trace = unpickled.get("trace") if isinstance(unpickled, dict) else None
    constraints = unpickled.get("constraints") if isinstance(unpickled, dict) else None
    options = unpickled.get("options") if isinstance(unpickled, dict) else None
    pipeline = unpickled.get("pipeline") if isinstance(unpickled, dict) else None

    if not isinstance(trace, CapturedTrace):
        raise SystemExit(
            f"FATAL: unpickled object is not CapturedTrace (got {type(trace)})"
        )

    if not isinstance(constraints, list) or not all(
        isinstance(c, Constraint) for c in constraints
    ):
        raise SystemExit(
            f"FATAL: unpickled object is not list of Constraints (got {type(constraints)})"
        )

    if not isinstance(options, WaveCompileOptions):
        raise SystemExit(
            f"FATAL: unpickled object is not WaveCompileOptions (got {type(options)})"
        )

    if not isinstance(pipeline, str):
        raise SystemExit(f"FATAL: unpickled object is not str (got {type(pipeline)})")

    # Restore supplemental node fields captured in the meta field
    trace.restore_node_state()
    return trace, constraints, options, pipeline


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

      1. adding assumptions about all symbols being positive to later enable
         more simplifications.
      2. replacing ITER_SYMBOL_NAME_WAVE_PREFIX (`$ARG`) prefix of argument
         symbols (e.g. `ARG0`) by ITER_SYMBOL_NAME_WATER_PREFIX (`_Iter_`) to
         match dialect expectations.
    """
    result = {}
    for sym in symbols:
        # Special case: rename $ARG* symbols to _Iter_*.
        if sym.name.startswith(ITER_SYMBOL_NAME_WAVE_PREFIX):
            new_name = sym.name.replace(
                ITER_SYMBOL_NAME_WAVE_PREFIX, ITER_SYMBOL_NAME_WATER_PREFIX
            )
            result[sym] = sympy.Symbol(new_name, positive=True)
        else:
            result[sym] = sympy.Symbol(sym.name, positive=True)
    return result


def _symbol_name_to_attribute(name: str) -> ir.Attribute:
    """
    Convert a symbol name to either a WaveSymbolAttr or WaveIndexSymbolAttr.

    Special symbols starting with $ are converted to WaveIndexSymbolAttr,
    while regular symbols are converted to WaveSymbolAttr.
    """

    if name in INDEX_SYMBOL_MAP:
        return wave.WaveIndexSymbolAttr.get(INDEX_SYMBOL_MAP[name])
    if name.startswith(ITER_SYMBOL_NAME_WATER_PREFIX):
        return wave.WaveIterSymbolAttr.get(
            name.replace(ITER_SYMBOL_NAME_WATER_PREFIX, "")
        )
    else:
        return wave.WaveSymbolAttr.get(name)


def _build_index_mapping_dict(
    index: dict[IndexSymbol, IndexSequence], allowed_induction_symbols: set[IndexSymbol]
) -> ir.DictAttr:
    """
    Convert a Wave index dictionary into a DictionaryAttr of
    WaveIndexMappingAttr.

    For MMA, multiple DictAttr objects are assembled into an ArrayAttr (one per
    operand). For all other nodes a single-element ArrayAttr is used.

    The `allowed_induction_symbols` argument lists induction variable-related
    symbols that are allowed to be present in the expressions. Other symbols
    will be removed and a warning will be generated if it is the case.
    """

    index_mappings: dict[str, ir.Attribute] = {}
    for dim, exprs in index.items():
        all_symbols_set = set().union(
            *[
                expr.free_symbols
                for expr in [exprs.start, exprs.size, exprs.stride]
                if isinstance(expr, sympy.Expr)
            ]
        )
        induction_symbols_to_remove = {
            symbol
            for symbol in all_symbols_set
            if symbol.name.startswith(ITER_SYMBOL_NAME_WAVE_PREFIX)
            and symbol not in allowed_induction_symbols
        }
        if induction_symbols_to_remove:
            induction_symbols_subs = {
                symbol: sympy.Integer(0) for symbol in induction_symbols_to_remove
            }
            # TODO: can we wrap this into a diagnostic?
            print(
                f"WARNING: Removing invalid induction symbols {induction_symbols_to_remove} from {index}",
                file=sys.stderr,
            )
            exprs.start = safe_subs(exprs.start, induction_symbols_subs)
            exprs.size = safe_subs(exprs.size, induction_symbols_subs)
            exprs.stride = safe_subs(exprs.stride, induction_symbols_subs)

        all_symbols = list(all_symbols_set - induction_symbols_to_remove)
        symbol_mapping = _preprocess_symbols(all_symbols)
        start = _convert_sympy_expr_to_affine_map(exprs.start, symbol_mapping)
        size = _convert_sympy_expr_to_affine_map(exprs.size, symbol_mapping)
        stride = _convert_sympy_expr_to_affine_map(exprs.stride, symbol_mapping)
        symbol_attrs = [
            _symbol_name_to_attribute(sym.name) for sym in symbol_mapping.values()
        ]
        index_mappings[dim.name] = wave.WaveIndexMappingAttr.get(
            symbol_attrs, start, size, stride
        )
    return ir.DictAttr.get(index_mappings)


def _attach_attributes(
    node: CustomOp, op: ir.Operation, known_ids: set[str] | None = None
):
    if getattr(node, "index", None) and isinstance(node.index, dict):
        dict_attrs: list[ir.DictAttr] = []

        # XXX: Collect induction-related symbols that make sense in the current
        # context; the frontend is buggy and may have these symbols outside of
        # the respective loops.
        parent_fx_node = node.fx_node
        allowed_induction_symbols: set[IndexSymbol] = set()
        while parent_fx_node := getattr(parent_fx_node.graph, "parent_op", None):
            parent_custom = get_custom(parent_fx_node)
            if isinstance(parent_custom, Iterate):
                induction_symbol = get_induction_symbol(parent_custom.axis)
                allowed_induction_symbols.add(induction_symbol)

        if isinstance(node, MMA):
            # Build one index mapping dict per operand for MMA nodes
            if lhs_index := getattr(node, "lhs_index", None):
                dict_attrs.append(
                    _build_index_mapping_dict(lhs_index, allowed_induction_symbols)
                )
            if rhs_index := getattr(node, "rhs_index", None):
                dict_attrs.append(
                    _build_index_mapping_dict(rhs_index, allowed_induction_symbols)
                )
            if acc_index := getattr(node, "acc_index", None):
                dict_attrs.append(
                    _build_index_mapping_dict(acc_index, allowed_induction_symbols)
                )
        else:
            dict_attrs.append(
                _build_index_mapping_dict(node.index, allowed_induction_symbols)
            )

        op.attributes["index"] = ir.ArrayAttr.get(dict_attrs)

    if getattr(node, "elements_per_thread", None):
        op.attributes["elements_per_thread"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), node.elements_per_thread
        )

    if getattr(node, "bounds", None):
        bounds = {}
        for dim, expr in node.bounds.items():
            symbol_mapping = _preprocess_symbols(
                list(expr.free_symbols) if isinstance(expr, sympy.Expr) else []
            )
            result = _convert_sympy_expr_to_affine_map(expr, symbol_mapping)
            symbol_attrs = [
                _symbol_name_to_attribute(sym.name) for sym in symbol_mapping.values()
            ]
            bounds[dim.name] = wave.WaveExprListAttr.get(symbol_attrs, result)
        op.attributes["bounds"] = wave.WaveReadWriteBoundsAttr.get(bounds)

    if water_id := getattr(node.fx_node, "_water_id", None):
        op.attributes[_INTERNAL_WATER_ID_ATTR_NAME] = ir.StringAttr.get(water_id)
        if known_ids is not None:
            known_ids.add(water_id)
    elif known_ids is not None:
        raise RuntimeError(f"Water id requested but not specified for node {node}.")


def _convert_to_wave_expr_list_tuple(
    exprs: Sequence[sympy.Expr | int],
) -> WaveExprListAttr:
    """
    Returns a WaveExprListAttr from a sequence of wave IndexExpr.
    Creates a multi-result affine map from the sequence of expressions.
    """
    # Collect all symbols from all expressions
    all_symbols = set()
    for expr in exprs:
        if isinstance(expr, sympy.Expr):
            all_symbols.update(expr.free_symbols)

    # Preprocess symbols and create mapping
    symbol_mapping = _preprocess_symbols(list(all_symbols))

    # Convert each expression to an affine expression
    affine_exprs = []
    for expr in exprs:
        affine_map = _convert_sympy_expr_to_affine_map(expr, symbol_mapping)
        # Extract the single result from the map (each expr creates a 1-result map)
        affine_exprs.append(affine_map.results[0])

    # Create a multi-result affine map
    multi_result_map = ir.AffineMap.get(0, len(symbol_mapping), affine_exprs)

    # Convert symbol names to attributes
    symbol_attrs = [
        _symbol_name_to_attribute(sym.name) for sym in symbol_mapping.values()
    ]

    return WaveExprListAttr.get(symbol_attrs, multi_result_map)


def _emit_ops_from_graph(
    graph: fx.Graph,
    trace: CapturedTrace,
    value_map: dict[fx.Node | fx.Proxy, ir.Value],
    ctx: ir.Context,
    known_ids: set[str] | None = None,
):
    # Emit in original order to preserve dependencies
    for fx_node in graph.nodes:
        node = get_custom(fx_node)
        # Ensure types are inferred
        node.infer_type()

        with node.location.to_water() if node.location else ir.Location.current:
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

                # Attach IDs of `get_result` to the loop instead so we can recover them
                # later because `get_result` doesn't exist in the dialect.
                if known_ids is not None:
                    water_id = getattr(fx_node, "_water_id", None)
                    if water_id is None:
                        raise RuntimeError(
                            f"Water id requested for 'get_result' but not specified: {node}"
                        )
                    known_ids.add(water_id)
                    current_attribute = (
                        iterate_op.attributes[_INTERNAL_RESULT_WATER_IRS_ATTR_NAME]
                        if _INTERNAL_RESULT_WATER_IRS_ATTR_NAME in iterate_op.attributes
                        else ir.ArrayAttr.get(
                            [ir.UnitAttr.get()] * len(iterate_op.results)
                        )
                    )
                    attribute_list = list(current_attribute)
                    attribute_list[node.res_idx] = ir.StringAttr.get(water_id)
                    iterate_op.attributes[_INTERNAL_RESULT_WATER_IRS_ATTR_NAME] = (
                        ir.ArrayAttr.get(attribute_list)
                    )

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
                elif isinstance(node, Iterate):
                    axis = wave.WaveSymbolAttr.get(node.axis.name)
                    carried_values = [value_map.get(arg) for arg in node.init_args]

                    result_types = []
                    result_locs = []
                    outputs = node.outputs()
                    if not isinstance(outputs, Sequence):
                        outputs = [outputs]
                    for fx_output in outputs:
                        output = get_custom(fx_output)
                        output.infer_type()
                        result_types.append(_type_to_wave_mlir(ctx, output.type))
                        result_locs.append(
                            output.location.to_water()
                            if output.location
                            else ir.Location.current
                        )

                    mlir_op = op_builder(result_types, axis, carried_values, [])
                    body = ir.Block.create_at_start(
                        mlir_op.regions[0], result_types, result_locs
                    )

                    for idx, iter_arg in enumerate(node.iter_args()):
                        iter_arg.iter_idx = idx

                    # add mapping for iter args
                    for wave_arg, mlir_arg in zip(node.iter_args(), body.arguments):
                        value_map[wave_arg] = mlir_arg

                    # Emit subgraph of the iterate node
                    with ir.InsertionPoint(body):
                        _emit_ops_from_graph(
                            trace.get_subgraph(node.subgraph_name),
                            trace,
                            value_map,
                            ctx,
                            known_ids,
                        )

                        # create YieldOp
                        YieldOp([value_map[output] for output in outputs])
                elif isinstance(node, MMA):
                    mma_kind = (
                        ir.Attribute.parse(
                            f"#wave.mma_kind<{node.mma_type.name.lower()}>", context=ctx
                        )
                        if node.mma_type is not None
                        else None
                    )
                    mlir_op = op_builder(result_type, *mlir_operands, kind=mma_kind)
                elif isinstance(node, Allocate):
                    mlir_op = op_builder(
                        result_type,
                        distributed_shape=_convert_to_wave_expr_list_tuple(
                            node.distributed_shape
                        ),
                    )
                elif isinstance(node, ExtractSlice):
                    size = _convert_to_wave_expr_list_tuple(node.size)
                    stride = _convert_to_wave_expr_list_tuple(node.stride)
                    offset = _convert_to_wave_expr_list_tuple(node.offset)
                    mlir_op = op_builder(
                        result_type, *mlir_operands, offset, size, stride
                    )
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

            _attach_attributes(node, mlir_op.operation, known_ids)

            # Add results to the value map in case they are used as
            # operands later
            if len(mlir_op.results) > 1:
                # TODO: rework value_map to always map to a sequence of results
                raise NotImplementedError(
                    f"Missing support for operations with multiple results"
                )
            for result in mlir_op.results:
                value_map[fx_node] = result


def _emit_wave_constraints(constraint: Constraint) -> ir.Attribute:
    if isinstance(constraint, HardwareConstraint):
        mma_type_attr = None
        if constraint.mma_type:
            mma_type_attr = WaveMmaKindAttr.get(WaveMmaKind(constraint.mma_type.value))

        shape_dict = None
        if constraint.vector_shapes:
            i64 = ir.IntegerType.get_signless(64)
            dict = {
                k.name: ir.IntegerAttr.get(i64, v)
                for k, v in constraint.vector_shapes.items()
            }
            shape_dict = ir.DictAttr.get(dict)

        attr = HardwareConstraintAttr.get(
            threads_per_wave=constraint.threads_per_wave,
            waves_per_block=constraint.waves_per_block,
            mma_type=mma_type_attr,
            vector_shapes=shape_dict,
            max_bits_per_load=constraint.max_bits_per_load,
        )

        return attr

    if isinstance(constraint, WorkgroupConstraint):
        size = _convert_to_wave_expr_list_tuple([constraint.tile_size])
        wg_dim = WaveWorkgroupDimAttr.get(constraint.workgroup_dim)
        attr = WorkgroupConstraintAttr.get(
            dim=constraint.dim.name, tile_size=size, workgroup_dim=wg_dim
        )
        return attr

    if isinstance(constraint, WaveConstraint):
        size = _convert_to_wave_expr_list_tuple([constraint.tile_size])
        attr = WaveConstraintAttr.get(dim=constraint.dim.name, tile_size=size)
        return attr

    if isinstance(constraint, TilingConstraint):
        size = _convert_to_wave_expr_list_tuple([constraint.tile_size])
        return TilingConstraintAttr.get(dim=constraint.dim.name, tile_size=size)

    if isinstance(constraint, DeviceConstraint):
        size = _convert_to_wave_expr_list_tuple([constraint.tile_size])
        return DeviceConstraintAttr.get(
            dim=constraint.dim.name, tile_size=size, device_dim=constraint.device_dim
        )

    raise NotImplementedError(f"Unsupported constraint type: {type(constraint)}")


def _flush_output(
    module_str: str,
    diagnostics: list[str],
    inferred_attributes: dict[str, dict[str, Any]] | None = None,
) -> None:
    output = dill.dumps(
        {
            "diagnostics": [d.encode("utf-8") for d in diagnostics],
            "module": module_str.encode("utf-8"),
            "inferred_attributes": (
                inferred_attributes if inferred_attributes is not None else {}
            ),
        }
    )
    sys.stdout.buffer.write(output)
    sys.stdout.flush()


def _create_kernel_module(
    ctx: ir.Context,
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
    test_diagnostics: bool = False,
) -> tuple[ir.Module | None, list[str], set[str]]:
    """Creates an MLIR module containing the kernel function from the captured trace.

    Args:
        ctx: MLIR context set up with correct dialects.
        trace: Captured Wave trace to convert.
        constraints: List of Wave constraints to attach to the function.
        options: Compilation options including hyperparameters.
        test_diagnostics: Whether to emit a test diagnostic

    Returns:
        - The created MLIR module, or None if creation failed.
        - List of diagnostic messages.
        - Set of known water IDs if options require checking water analysis.
    """
    diagnostics: list[str] = []
    known_ids: set[str] | None = set() if options.check_water_analysis else None

    def diagnostics_handler(d):
        diagnostics.append(f"{d.location}: {d.message}")
        return True

    ctx.attach_diagnostic_handler(diagnostics_handler)

    if options.override_mlir:
        try:
            module = ir.Module.parse(options.override_mlir, context=ctx)
        except ir.MLIRError as e:
            diagnostics.append(str(e))
            return None, diagnostics, known_ids
        else:
            return module, diagnostics, known_ids

    # Keep track of which emitted value stems from what node to wire
    # arguments correctly.
    value_map: dict[fx.Node | fx.Proxy, ir.Value] = {}

    module = ir.Module.create()

    if test_diagnostics:
        loc = ir.Location.current
        loc.emit_error("test error")

    # Collect placeholders from graph
    placeholders = [n for n in trace.walk() if getattr(n, "op", "") == "placeholder"]
    top_level_placeholders = [
        p for p in placeholders if getattr(p, "graph", None) is trace.get_root_graph()
    ]
    top_level_names = [p.name for p in top_level_placeholders]

    # Build function argument types from top-level placeholders
    arg_types = []
    arg_locs = []
    for p in top_level_placeholders:
        c = get_custom(p)
        t = getattr(c, "_type", None) or getattr(c, "type", None)
        # At this pipeline stage, symbolic address spaces should
        # have been handled by Python `promote_placeholders` transformation, and all function arguments
        # should be global by now (shared memory allocation happens inside the kernel).
        # Thus, resolve symbolic address spaces from hyperparameters.

        if issubclass(t, Memory) and t.address_space in options.subs:
            # Create a new type with resolved address space
            resolved_address_space = options.subs[t.address_space]
            if resolved_address_space != GLOBAL_ADDRESS_SPACE:
                raise RuntimeError(
                    f"Unexpected address space in hyperparameters: {t.address_space} -> {resolved_address_space}"
                )
            t = Memory[t.symbolic_shape, resolved_address_space, t.dtype]
        arg_types.append(_type_to_wave_mlir(ctx, t))
        arg_locs.append(c.location.to_water() if c.location else ir.Location.current)

    # Return type of the function is always empty
    func_type = ir.FunctionType.get(arg_types, [])
    with ir.InsertionPoint(module.body):
        func_op = func.FuncOp("kernel", func_type)
        # Validate that all non-int mappings are address spaces (either SHARED_ADDRESS_SPACE or GLOBAL_ADDRESS_SPACE).
        # These mappings can be dropped safely because the information has been encoded in either `arg_types` (for GLOBAL_ADDRESS_SPACE) or LDS allocations inside the kernel (done by `promote_placeholders for SHARED_ADDRESS_SPACE).
        # print([(str(k), v) for k, v in options.subs.items()])
        for k, v in options.subs.items():
            if not isinstance(v, int):
                if v not in (SHARED_ADDRESS_SPACE, GLOBAL_ADDRESS_SPACE):
                    raise RuntimeError(
                        f"Unexpected non-int mapping in hyperparameters: {k} -> {v}. "
                        f"Expected all non-int values to be address spaces"
                    )
        # Convert the symbols in subs to strings and attach as
        # WaveHyperparameterAttr to func_op
        func_op.operation.attributes["wave.hyperparameters"] = (
            wave.WaveHyperparameterAttr.get(
                {str(k): v for k, v in options.subs.items() if isinstance(v, int)}
            )
        )

        wave_constraints = list(map(_emit_wave_constraints, constraints))
        array_attr = ir.ArrayAttr.get(wave_constraints)
        func_op.operation.attributes[wave.WAVE_CONSTRAINTS_ATTR_NAME] = array_attr

        entry_block = ir.Block.create_at_start(func_op.regions[0], arg_types, arg_locs)

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
                top_level_placeholders[top_level_names.index(nested_placeholder.name)]
            ]

        with ir.InsertionPoint(entry_block):
            _emit_ops_from_graph(
                trace.get_root_graph(), trace, value_map, ctx, known_ids
            )
            func.ReturnOp(operands_=[])

    return module, diagnostics, known_ids


_INTERNAL_WATER_ID_ATTR_NAME = "_water_internal.id"
_INTERNAL_RESULT_WATER_IRS_ATTR_NAME = "_water_internal.result_ids"


def _emit_from_captured_trace(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
    pipeline: str = "",
    test_diagnostics=False,
) -> int:

    diagnostics = []

    enable_debug_info = (
        options.location_capture_config.level is not LocationCaptureLevel.NONE
    )

    if enable_debug_info and not trace.location:
        diagnostics.append("Missing debug location for wave trace")

    with (
        ir.Context() as ctx,
        trace.location.to_water() if trace.location else ir.Location.unknown(),
    ):
        ctx.allow_unregistered_dialects = False
        wave.register_dialect(ctx)
        wave.register_passes()

        module, creation_diagnostics, known_ids = _create_kernel_module(
            ctx, trace, constraints, options, test_diagnostics
        )
        diagnostics.extend(creation_diagnostics)
        if module is None:
            _flush_output("", diagnostics, None)
            return 0

        # Verify the module before transforming or printing.
        try:
            module.operation.verify()
        except ir.MLIRError as e:
            diagnostics.append(str(e))
            # Print in generic form if verification fails, this form should be
            # robust to that.
            _flush_output(
                module.operation.get_asm(
                    enable_debug_info=enable_debug_info, print_generic_op_form=True
                ),
                diagnostics,
                None,
            )
            return 0

        if options.print_mlir_before_water:
            print(module.operation.get_asm(), file=sys.stderr)

        # If a transform script was provided, parse and apply it to the module.
        # This expects a transform module with a named sequence as first operation.
        if pipeline:
            try:
                transform_module = ir.Module.parse(pipeline)
                ops = list(transform_module.body.operations)
                if not ops:
                    raise RuntimeError("Transform module is empty")
                entry_op = ops[0]
                # Require the first op to be a named sequence.
                if entry_op.operation.name != "transform.named_sequence":
                    raise RuntimeError(
                        f'Expected first op to be "transform.named_sequence", got "{entry_op.operation.name}"'
                    )
                interpreter.apply_named_sequence(
                    module,
                    entry_op,
                    transform_module,
                )
            except Exception as e:
                diagnostics.append(f"Failed to apply transform script: {e}")

        module_str = module.operation.get_asm(enable_debug_info=enable_debug_info)
        if options.print_mlir_after_water:
            print(module_str, file=sys.stderr)

        # Collect attributes inferred by the pass and store them in the per-id dictionary.
        inferred_attributes: dict[str, dict[str, Any]] = (
            {id: {} for id in known_ids} if known_ids else {}
        )
        if options.check_water_analysis:

            def extractor(op: ir.Operation) -> ir.WalkResult:
                attribute: ir.Attribute | None = (
                    op.attributes[_INTERNAL_WATER_ID_ATTR_NAME]
                    if _INTERNAL_WATER_ID_ATTR_NAME in op.attributes
                    else None
                )
                result_attribute: ir.Attribute | None = (
                    op.attributes[_INTERNAL_RESULT_WATER_IRS_ATTR_NAME]
                    if _INTERNAL_RESULT_WATER_IRS_ATTR_NAME in op.attributes
                    else None
                )
                if attribute is None and result_attribute is None:
                    return ir.WalkResult.ADVANCE

                def record_index(
                    attribute: ir.Attribute,
                    inferred_attributes: dict[str, dict[str, Any]],
                ):
                    assert isinstance(
                        attribute, ir.StringAttr
                    ), f"Unexpected attribute type: {attribute}."
                    assert (
                        attribute.value in inferred_attributes
                    ), f"Unknown water id {attribute.value}."
                    assert (
                        "index" not in inferred_attributes[attribute.value]
                    ), f"Index already set for water id {attribute.value}."
                    assert "index" in op.attributes, f"Index not inferred for {op}."

                    inferred_attributes[attribute.value].update(
                        {
                            "index": convert_index_mapping_array_to_sympy(
                                op, op.attributes["index"]
                            )
                        }
                    )

                if attribute is not None:
                    record_index(attribute, inferred_attributes)
                if result_attribute is not None:
                    assert isinstance(
                        result_attribute, ir.ArrayAttr
                    ), f"Unexpected attribute type: {result_attribute}."
                    for attribute in result_attribute:
                        record_index(attribute, inferred_attributes)

                return ir.WalkResult.ADVANCE

            module.operation.walk(extractor)
            for water_id, inferred_attribute in inferred_attributes.items():
                if "index" not in inferred_attribute:
                    raise RuntimeError(f"Index not inferred for water id {water_id}.")

        module_str = module.operation.get_asm(enable_debug_info=enable_debug_info)
        _flush_output(module_str, diagnostics, inferred_attributes)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water dialect emitter")

    parser.add_argument(
        "--test-diagnostic-emission",
        action="store_true",
        help="Test diagnostic serialization and deserialization through stdin and stdout",
    )

    args = parser.parse_args()

    trace, constraints, options, pass_pipeline = _parse_input()
    sys.exit(
        _emit_from_captured_trace(
            trace, constraints, options, pass_pipeline, args.test_diagnostic_emission
        )
    )
