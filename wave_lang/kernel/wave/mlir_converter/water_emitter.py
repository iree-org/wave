"""
Water Emitter for Wave Dialect

This generates operations of the Wave dialect from a serialized wave trace.
It runs as a standalone process with access to Water Python bindings and emits
mlir operations of wave and other dialects.
"""

from __future__ import annotations
import argparse
import dill
import sys
import torch.fx as fx
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import sympy

if __name__ == "__main__":
    # Add parent directory to sys.path to enable relative imports when running standalone
    # This allows importing water_mlir from ../water_mlir/ as if it were a relative import
    _current_file = Path(__file__).resolve()
    _parent_dir = str(_current_file.parent.parent)  # Go up to wave_lang/kernel/wave/
    if _parent_dir not in sys.path:
        sys.path.append(_parent_dir)
    # Add current directory to enable importing attr_type_converter without full package path
    _current_dir = str(_current_file.parent)
    if _current_dir not in sys.path:
        sys.path.append(_current_dir)

from attr_type_converter import (
    convert_index_mapping_array_to_sympy,
    dtype_to_mlir_scalar_type,
    get_operand_symbol_placeholders,
    preprocess_symbols,
    symbol_name_to_attribute,
)

from wave_lang.kernel._support.indexing import IndexSymbol, safe_subs


if TYPE_CHECKING:
    from wave_lang.kernel._support.indexing import IndexSequence, IndexSymbol
    from wave_lang.kernel.ops.wave_ops import *

from wave_lang.kernel.wave.mlir_converter.diagnostics import (
    FileLocation,
    LocationFrame,
    MLIRDiagnostic,
    NameLocation,
    WaterError,
)
from wave_lang.support.location_config import LocationCaptureLevel
from wave_lang.kernel.lang.wave_types import Memory, Register
from wave_lang.kernel.lang.kernel_buffer import AddressSpace
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.utils.symbol_utils import (
    collect_allowed_induction_symbols,
    strip_out_of_scope_induction_symbols,
)

from wave_lang.kernel.ops.wave_ops import (
    Allocate,
    ApplyExpr,
    Extract,
    ExtractSlice,
    get_custom,
    GetResult,
    IterArg,
    Iterate,
    MMA,
    NestedRegionOp,
    NewRegister,
    Output,
    Placeholder,
    Placeholder,
    ReduceOp as Reduce,
    SelfIndex,
    SharedMemoryBarrier,
    ShuffleOp as Shuffle,
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
        ApplyExprOp,
        SubOp,
        AllocateOp,
        BroadcastOp,
        CastOp,
        DivOp,
        ReciprocalOp,
        Exp2Op,
        ExtractOp,
        ExtractSliceOp,
        IterateOp,
        MaxElementOp,
        MaxOp,
        MinOp,
        MmaOp,
        MulOp,
        ReadOp,
        RegisterOp,
        SelectOp,
        SelfIndexOp,
        ShuffleOp,
        SumOp,
        WriteOp,
        YieldOp,
        PermuteOp,
        DeviceConstraintAttr,
        HardwareConstraintAttr,
        TilingConstraintAttr,
        WaveConstraintAttr,
        WaveExprListAttr,
        WaveMmaKind,
        WaveMmaKindAttr,
        WaveWorkgroupDimAttr,
        WorkgroupConstraintAttr,
        WaveAddressSpace,
        WaveAddressSpaceAttr,
        WaveSymbolAttr,
        WaveTensorType,
    )
    from water_mlir.water_mlir.sympy_to_affine_converter import (
        convert_sympy_to_affine_map,
        convert_sympy_to_affine_map_and_combinator,
    )
    from water_mlir.water_mlir.dialects import arith, func, wave, amdgpu
    from water_mlir.water_mlir.dialects.transform import interpreter
except Exception as e:
    print(f"FATAL: failed to import water_mlir: {e}", file=sys.stderr)
    sys.exit(1)

# Mapping from tkw_op_name to actual op constructors
WAVE_OP_CONSTRUCTORS = {
    "add": AddOp,
    "apply_expr": ApplyExprOp,
    "broadcast": BroadcastOp,
    "sub": SubOp,
    "allocate": AllocateOp,
    "cast": CastOp,
    "extract": ExtractOp,
    "extract_slice": ExtractSliceOp,
    "maximum": MaxOp,
    "minimum": MinOp,
    "mma": MmaOp,
    "mul": MulOp,
    "div": DivOp,
    "reciprocal": ReciprocalOp,
    "exp2": Exp2Op,
    "read": ReadOp,
    "register": RegisterOp,
    "shuffle": ShuffleOp,
    "iterate": IterateOp,
    "output": YieldOp,
    "write": WriteOp,
    "permute": PermuteOp,
    "max": MaxElementOp,
    "sum": SumOp,
    "select": SelectOp,
    "self_index": SelfIndexOp,
    # TODO: Add more or find a good way of avoiding needing a mapping
}

# TODO: Rework wave dtype.Datatype so this does not need to rely on strings
# This maps to a lambda to create the respective MLIR type. An MLIR context
# is required to invoke the lambda.
DATATYPE_MAP: dict[str, Callable[[], ir.Type]] = {
    "bf16": ir.BF16Type.get,
    "f16": ir.F16Type.get,
    "f32": ir.F32Type.get,
    "f64": ir.F64Type.get,
    "i1": lambda: ir.IntegerType.get_signless(1),
    # 'bool' is an alias for 'i1'.
    "bool": lambda: ir.IntegerType.get_signless(1),
    "i8": lambda: ir.IntegerType.get_signless(8),
    "i16": lambda: ir.IntegerType.get_signless(16),
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


def _map_address_space(addr) -> WaveAddressSpaceAttr:
    """Map Python AddressSpace (int, enum, or IndexSymbol) to MLIR WaveAddressSpaceAttr."""
    # Handle IndexSymbol (sympy.Symbol) - convert to string and match
    # IndexSymbol is a sympy.Symbol with a name attribute
    if hasattr(addr, "name"):
        addr_str = str(addr).lower()
        if "global" in addr_str:
            return WaveAddressSpaceAttr.get(WaveAddressSpace.Global)
        elif "shared" in addr_str:
            return WaveAddressSpaceAttr.get(WaveAddressSpace.Shared)
        elif "register" in addr_str:
            return WaveAddressSpaceAttr.get(WaveAddressSpace.Register)
        else:
            return WaveAddressSpaceAttr.get(WaveAddressSpace.Unspecified)

    # Handle AddressSpace enum
    if isinstance(addr, AddressSpace):
        addr = addr.value

    # Handle int values
    if addr == AddressSpace.GLOBAL_MEMORY.value:
        return WaveAddressSpaceAttr.get(WaveAddressSpace.Global)
    elif addr == AddressSpace.SHARED_MEMORY.value:
        return WaveAddressSpaceAttr.get(WaveAddressSpace.Shared)
    elif addr == AddressSpace.REGISTER.value:
        return WaveAddressSpaceAttr.get(WaveAddressSpace.Register)
    else:
        # If we can't determine the address space, use unspecified for now
        # TODO: Sometimes the address space is determined in the hyperparameters
        #       Check whether a mapping for `addr` is defined there.
        return WaveAddressSpaceAttr.get(WaveAddressSpace.Unspecified)


def _create_wave_tensor_type(
    ctx: ir.Context,
    symbolic_shape: tuple,
    dtype,
    address_space_attr: WaveAddressSpaceAttr,
) -> ir.Type:
    """Helper function to create a WaveTensorType from shape, dtype, and address space."""
    shape_attrs = [WaveSymbolAttr.get(str(s)) for s in symbolic_shape]
    element_type = _dtype_to_mlir_scalar_type(dtype)
    return WaveTensorType.get(shape_attrs, True, element_type, address_space_attr)


def _type_to_wave_mlir(
    ctx: ir.Context, type_: type[Register] | type[Memory]
) -> ir.Type:
    # Map Python Wave types to MLIR types using WaveTensorType bindings
    if issubclass(type_, Register):
        if not type_.symbolic_shape or type_.dtype is None:
            raise RuntimeError(
                "Register type must have concrete symbolic_shape and dtype"
            )
        address_space_attr = WaveAddressSpaceAttr.get(WaveAddressSpace.Register)
        return _create_wave_tensor_type(
            ctx, type_.symbolic_shape, type_.dtype, address_space_attr
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
        address_space_attr = _map_address_space(type_.address_space)
        return _create_wave_tensor_type(
            ctx, type_.symbolic_shape, type_.dtype, address_space_attr
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


def _build_index_mapping_dict(
    index: dict[IndexSymbol, IndexSequence], allowed_induction_symbols: set[IndexSymbol]
) -> ir.DictAttr:
    """
    Convert a Wave index dictionary into a DictionaryAttr of
    WaveIndexMappingAttr.

    For MMA, multiple DictAttr objects are assembled into an ArrayAttr (one per
    operand). For all other nodes a single-element ArrayAttr is used.

    Out-of-scope induction symbols are stripped before conversion.
    """
    index = strip_out_of_scope_induction_symbols(index, allowed_induction_symbols)

    index_mappings: dict[str, ir.Attribute] = {}
    for dim, exprs in index.items():
        all_symbols = set()
        for component in (exprs.start, exprs.size, exprs.stride):
            if isinstance(component, sympy.Expr):
                all_symbols |= component.free_symbols
        symbol_mapping = preprocess_symbols(list(all_symbols))
        start = _convert_sympy_expr_to_affine_map(exprs.start, symbol_mapping)
        size = _convert_sympy_expr_to_affine_map(exprs.size, symbol_mapping)
        stride = _convert_sympy_expr_to_affine_map(exprs.stride, symbol_mapping)
        symbol_attrs = [
            symbol_name_to_attribute(sym.name) for sym in symbol_mapping.values()
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

        allowed_induction_symbols = collect_allowed_induction_symbols(node.fx_node)

        if isinstance(node, MMA):
            # MMA needs exactly 4 index entries (lhs, rhs, acc, result) to
            # match MmaOp::setIndexFromLattices which serialises
            # operandExprs + resultExprs.  The Python-side index sequence
            # analysis only tracks 3 (lhs, rhs, acc), so we emit acc_index
            # twice: once for the accumulator operand and once for the
            # result (MMA result type == acc type).
            if lhs_index := getattr(node, "lhs_index", None):
                dict_attrs.append(
                    _build_index_mapping_dict(lhs_index, allowed_induction_symbols)
                )
            if rhs_index := getattr(node, "rhs_index", None):
                dict_attrs.append(
                    _build_index_mapping_dict(rhs_index, allowed_induction_symbols)
                )
            if acc_index := getattr(node, "acc_index", None):
                acc_attr = _build_index_mapping_dict(
                    acc_index, allowed_induction_symbols
                )
                # Append acc_index for both the accumulator operand and the
                # result, since MMA result type == acc type.
                dict_attrs.append(acc_attr)
                dict_attrs.append(acc_attr)
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
            symbol_mapping = preprocess_symbols(
                list(expr.free_symbols) if isinstance(expr, sympy.Expr) else []
            )
            result = _convert_sympy_expr_to_affine_map(expr, symbol_mapping)
            symbol_attrs = [
                symbol_name_to_attribute(sym.name) for sym in symbol_mapping.values()
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
    symbol_mapping = preprocess_symbols(list(all_symbols))

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
        symbol_name_to_attribute(sym.name) for sym in symbol_mapping.values()
    ]

    return WaveExprListAttr.get(symbol_attrs, multi_result_map)


def _emit_ops_from_graph(
    graph: fx.Graph,
    trace: CapturedTrace,
    value_map: dict[fx.Node | fx.Proxy, tuple[ir.Value, ...]],
    implicit_captures: dict[fx.Node, fx.Node],
    ctx: ir.Context,
    known_ids: set[str] | None = None,
):
    # Emit in original order to preserve dependencies
    for fx_node in graph.nodes:
        node = get_custom(fx_node)
        # Ensure types are inferred
        node.infer_type()

        with node.location.to_water() if node.location else ir.Location.current:
            # Remap implicit captures to the captured values, MLIR modeling supports
            # implicit captures without additional constructs.
            if isinstance(node, Placeholder):
                if fx_node not in implicit_captures:
                    continue

                assert (
                    implicit_captures[fx_node] in value_map
                ), f"{node} implicitly captures a value that was not translated."

                value_map[fx_node] = value_map[implicit_captures[fx_node]]
                continue

            # No MLIR ops are emitted for output nodes.
            if isinstance(node, Output):
                continue

            def get_single_mapped_value(
                node: fx.Node | fx.Proxy, *, allow_missing: bool = False
            ) -> ir.Value | None:
                """Get the single mapped value for the given node.

                Raises a RuntimeError if the value is not found and allow_missing is False.
                """
                if (mlir_args := value_map.get(node)) is not None:
                    assert len(mlir_args) == 1, "A single-result node is expected."
                    return mlir_args[0]
                if allow_missing:
                    return None
                raise RuntimeError(f"No mapped value found for node {node}")

            def create_mlir_operands():
                """Create a list of MLIR operands from the arguments of the current node.

                Do so lazily as most explicitly handled nodes don't use these.
                """
                mlir_operands = []
                for arg in fx_node.args:
                    if (
                        mlir_arg := get_single_mapped_value(arg, allow_missing=True)
                    ) is not None:
                        mlir_operands.append(mlir_arg)
                return mlir_operands

            if isinstance(node, GetResult):
                if node.res_idx >= len(value_map[node.value]):
                    raise RuntimeError(
                        f"GetResult index ({node.res_idx}) is higher than the "
                        f"number of mapped values ({len(value_map[node.value])})."
                    )
                value_map[fx_node] = (value_map[node.value][node.res_idx],)

                # Attach IDs of `get_result` to the loop instead so we can recover them
                # later because `get_result` doesn't exist in the dialect. Only do so when
                # there are some results.
                if known_ids is not None and len(value_map[node.value]) > 0:
                    iterate_op = get_single_mapped_value(fx_node).owner
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

            result_type: ir.Type | list[ir.Type] = (
                _type_to_wave_mlir(ctx, node.type)
                if not isinstance(node.type, Sequence)
                else [_type_to_wave_mlir(ctx, t) for t in node.type]
            )
            # TODO: relax this as needed, e.g., topK.
            assert isinstance(result_type, ir.Type) or isinstance(
                node, NestedRegionOp
            ), "Only nested region ops can have multiple result types."

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
                        get_single_mapped_value(node.register_),
                        get_single_mapped_value(node.memory),
                    )
                elif isinstance(node, NewRegister):
                    dtype = getattr(node, "dtype", None)
                    if dtype is None:
                        raise RuntimeError("Register op missing dtype")
                    element_type = dtype_to_mlir_scalar_type(dtype)
                    constant_op = arith.ConstantOp(
                        result=element_type, value=node.value
                    )
                    mlir_op = op_builder(result_type, constant_op.results[0])
                elif isinstance(node, Iterate):
                    axis = wave.WaveSymbolAttr.get(node.axis.name)
                    carried_values = [
                        get_single_mapped_value(arg) for arg in node.init_args
                    ]

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
                        value_map[wave_arg] = (mlir_arg,)

                    # Emit subgraph of the iterate node
                    with ir.InsertionPoint(body):
                        _emit_ops_from_graph(
                            trace.get_subgraph(node.subgraph_name),
                            trace,
                            value_map,
                            implicit_captures,
                            ctx,
                            known_ids,
                        )

                        # create YieldOp
                        YieldOp([get_single_mapped_value(output) for output in outputs])
                elif isinstance(node, MMA):
                    mma_kind = (
                        ir.Attribute.parse(
                            f"#wave.mma_kind<{node.mma_type.name.lower()}>", context=ctx
                        )
                        if node.mma_type is not None
                        else None
                    )
                    mlir_op = op_builder(
                        result_type, *create_mlir_operands(), kind=mma_kind
                    )
                elif isinstance(node, Reduce):
                    if isinstance(node.arg, Sequence):
                        raise NotImplementedError(
                            "Only single-operand reductions are currently supported."
                        )
                    mlir_op = op_builder(
                        result_type,
                        *create_mlir_operands(),
                        scope=wave.WaveReductionScopeAttr.get(
                            wave.WaveReductionScope.Block
                            if node.block
                            else wave.WaveReductionScope.Warp
                        ),
                    )
                elif isinstance(node, Allocate):
                    # Get parent value from value_map if it exists.
                    parent_value = None
                    offset_attr = None
                    if node.parent is not None:
                        parent_value = get_single_mapped_value(node.parent)
                        # Offset must be present when parent is present.
                        if node.offset is None:
                            raise RuntimeError(
                                "Allocate op has parent but missing offset"
                            )
                        offset_attr = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(64), int(node.offset)
                        )
                    i64 = ir.IntegerType.get_signless(64)
                    padding_attr = (
                        ir.IntegerAttr.get(i64, node.padding)
                        if node.padding != 0
                        else None
                    )
                    tail_padding_attr = (
                        ir.IntegerAttr.get(i64, node.tail_padding)
                        if node.tail_padding != 0
                        else None
                    )
                    # Use unpadded_shape when distributed_shape matches
                    # shape rank (standard allocation). For flattened
                    # allocations (e.g. after minimize_shared_allocs),
                    # use distributed_shape directly since the padding
                    # is already baked into the flat size.
                    dist_shape = (
                        node.unpadded_shape
                        if len(node.distributed_shape) == len(node.shape)
                        else node.distributed_shape
                    )
                    mlir_op = op_builder(
                        result_type,
                        parent=parent_value,
                        distributed_shape=_convert_to_wave_expr_list_tuple(dist_shape),
                        offset=offset_attr,
                        padding=padding_attr,
                        tail_padding=tail_padding_attr,
                    )
                elif isinstance(node, ExtractSlice):
                    size = _convert_to_wave_expr_list_tuple(node.size)
                    stride = _convert_to_wave_expr_list_tuple(node.stride)
                    offset = _convert_to_wave_expr_list_tuple(node.offset)
                    mlir_op = op_builder(
                        result_type, *create_mlir_operands(), offset, size, stride
                    )
                elif isinstance(node, Extract):
                    assert len(node.offset) == 1
                    position = _convert_to_wave_expr_list_tuple(node.offset)
                    mlir_op = op_builder(result_type, *create_mlir_operands(), position)
                elif isinstance(node, SelfIndex):
                    if not isinstance(node.dim, IndexSymbol):
                        raise RuntimeError(
                            f"SelfIndex op has non-index symbol dimension: {node.dim}"
                        )
                    mlir_op = op_builder(
                        result_type,
                        *create_mlir_operands(),
                        dim=symbol_name_to_attribute(node.dim.name),
                        elements_per_thread=node.elements_per_thread,
                    )
                elif isinstance(node, Shuffle):
                    offset = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(32), node.offset
                    )
                    width = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(32), node.width
                    )
                    mode = wave.WaveShuffleModeAttr.get(node.mode.value)
                    mlir_op = op_builder(
                        result_type, *create_mlir_operands(), offset, width, mode
                    )
                elif isinstance(node, ApplyExpr):
                    reg = node.register_
                    if not isinstance(reg, Sequence):
                        reg = [reg]

                    placeholders = get_operand_symbol_placeholders(len(reg))
                    sympy_expr = node.expr(*list(placeholders.keys()))

                    # Sort all free_symbols by name for deterministic symbol order in the expr_list.
                    ordered_symbols = sorted(
                        sympy_expr.free_symbols, key=lambda s: s.name
                    )
                    symbol_mapping = preprocess_symbols(ordered_symbols)
                    affine_map, combinator = convert_sympy_to_affine_map_and_combinator(
                        sympy_expr, [symbol.name for symbol in symbol_mapping.keys()]
                    )
                    symbol_attrs = [
                        (
                            placeholders[orig_sym]
                            if orig_sym in placeholders
                            else symbol_name_to_attribute(new_sym.name)
                        )
                        for orig_sym, new_sym in symbol_mapping.items()
                    ]
                    expr_attr = WaveExprListAttr.get(symbol_attrs, affine_map)
                    operands = [get_single_mapped_value(arg) for arg in reg]
                    # TODO: need to add automatic constructor of WaveApplyExprCombinatorAttr
                    # from enum.
                    mlir_op = op_builder(
                        result_type,
                        operands,
                        expr_attr,
                        combinator=(
                            wave.WaveApplyExprCombinatorAttr.get(combinator)
                            if combinator is not None
                            else None
                        ),
                    )
                else:
                    try:
                        mlir_op = op_builder(result_type, *create_mlir_operands())
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
            value_map[fx_node] = tuple(mlir_op.results)


def _resolve_vector_shapes_for_attr(
    vector_shapes: dict[IndexSymbol, int | IndexExpr],
    subs: dict[IndexSymbol, Any],
) -> dict[str, int]:
    """Resolve vector_shapes values using subs (int-only).

    Raises ValueError if any vector_shapes value cannot be resolved to an integer.
    """
    int_subs = {k: v for k, v in subs.items() if isinstance(v, int)}
    resolved = {}
    for dim, v in vector_shapes.items():
        val = safe_subs(v, int_subs)
        if isinstance(val, int):
            resolved[dim.name] = val
        elif isinstance(val, sympy.Basic) and val.is_number:
            resolved[dim.name] = int(val)
        else:
            missing = getattr(val, "free_symbols", None) or {v}
            raise ValueError(
                f"Vector shape {v} in hardware constraints could not be resolved to an integer.\n"
                f"Note: symbols {missing} do not have substitutions."
            )
    return resolved


def _emit_wave_constraints(
    constraint: Constraint,
    subs: dict[IndexSymbol, Any] = {},
) -> ir.Attribute:
    if isinstance(constraint, HardwareConstraint):
        mma_type_attr = None
        if constraint.mma_type:
            mma_type_attr = WaveMmaKindAttr.get(WaveMmaKind(constraint.mma_type.value))

        shape_dict = None
        if constraint.vector_shapes:
            resolved = _resolve_vector_shapes_for_attr(constraint.vector_shapes, subs)
            i64 = ir.IntegerType.get_signless(64)
            dict = {k: ir.IntegerAttr.get(i64, v) for k, v in resolved.items()}
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


def serialize_location(loc: ir.Location) -> list[LocationFrame]:
    """Convert ir.Location to a serializable list of location frames (stack trace).

    Uses the MLIR Python binding properties to extract structured location
    information.

    Handles:
    - File locations (FileLineColLoc / FileLineColRange) via `is_a_file`
      → `filename`, `start_line`, `start_col`, `end_line`, `end_col`
    - Callsite locations via `is_a_callsite` → `caller`, `callee`
    - Fused locations via `is_a_fused` → `locations`
    - Name locations via `is_a_name` → `name_str`, `child_loc`

    Returns frames from outermost (caller) to innermost (callee).
    """

    def _collect_frames(
        loc: ir.Location,
    ) -> list[LocationFrame]:
        # Note: is_a_file / is_a_callsite / is_a_name / is_a_fused are
        # methods (not properties) in the nanobind bindings and must be called.
        if loc.is_a_file():
            return [
                FileLocation(
                    filename=str(loc.filename),
                    start_line=loc.start_line,
                    start_col=loc.start_col,
                    end_line=loc.end_line,
                    end_col=loc.end_col,
                )
            ]
        if loc.is_a_callsite():
            # Callsite: caller (outermost) first, then callee (innermost)
            caller_frames = _collect_frames(loc.caller)
            callee_frames = _collect_frames(loc.callee)
            return caller_frames + callee_frames
        if loc.is_a_name():
            child_frames = _collect_frames(loc.child_loc)
            first_child = child_frames[0] if child_frames else None
            remaining = child_frames[1:] if len(child_frames) > 1 else []
            return [
                NameLocation(
                    name=str(loc.name_str),
                    child_location=first_child,
                )
            ] + remaining
        if loc.is_a_fused():
            frames: list[LocationFrame] = []
            for sub_loc in loc.locations:
                frames.extend(_collect_frames(sub_loc))
            return frames

        return [None]

    return _collect_frames(loc)


def _flush_output(
    module_str: str,
    diagnostics: list[MLIRDiagnostic | WaterError],
    inferred_attributes: dict[str, dict[str, Any]] | None = None,
) -> None:
    output = dill.dumps(
        {
            "diagnostics": diagnostics,
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
) -> tuple[ir.Module | None, list[MLIRDiagnostic | WaterError], set[str] | None]:
    """Creates an MLIR module containing the kernel function from the captured trace.

    Args:
        ctx: MLIR context set up with correct dialects.
        trace: Captured Wave trace to convert.
        constraints: List of Wave constraints to attach to the function.
        options: Compilation options including hyperparameters.
        test_diagnostics: Whether to emit a test diagnostic

    Returns:
        A tuple containing:
        - The created MLIR module, or None if creation failed.
        - List of diagnostic messages (MLIRDiagnostic or WaterError instances).
        - Set of known water IDs if options.check_water_analysis is True, otherwise None.
    """
    diagnostics: list[MLIRDiagnostic | WaterError] = []
    known_ids: set[str] | None = set() if options.check_water_analysis else None

    if options.override_mlir:
        try:
            module = ir.Module.parse(options.override_mlir, context=ctx)
        except ir.MLIRError as e:
            diagnostics.append(WaterError(message=e.message))
            return None, diagnostics, known_ids
        else:
            return module, diagnostics, known_ids

    # Keep track of which emitted value stems from what node to wire
    # arguments correctly.
    value_map: dict[fx.Node | fx.Proxy, tuple[ir.Value, ...]] = {}

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
        # Validate that all non-int mappings are address spaces (either
        # SHARED_ADDRESS_SPACE or GLOBAL_ADDRESS_SPACE). These mappings can be
        # dropped safely because the information has been encoded in either
        # `arg_types` (for GLOBAL_ADDRESS_SPACE) or LDS allocations inside the
        # kernel (done by `promote_placeholders for SHARED_ADDRESS_SPACE).
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

        try:
            wave_constraints = [
                _emit_wave_constraints(c, options.subs) for c in constraints
            ]
        except ValueError as e:
            diagnostics.append(WaterError(message=str(e)))
            return None, diagnostics, known_ids
        array_attr = ir.ArrayAttr.get(wave_constraints)
        func_op.operation.attributes[wave.WAVE_CONSTRAINTS_ATTR_NAME] = array_attr

        entry_block = ir.Block.create_at_start(func_op.regions[0], arg_types, arg_locs)

        # Map placeholders to function arguments
        for i, fx_node in enumerate(top_level_placeholders):
            value_map[fx_node] = (entry_block.arguments[i],)

        # Subgraphs duplicate the placeholders of surrounding graphs so there
        # are multiple placeholders representing the same values.
        # Add mapping for these repeated placeholders as well. In addition,
        # subgraphs have implicit capture placeholders that refer to nodes in
        # the outer graph, store those separately.
        implicit_captures: dict[fx.Node, fx.Node] = {}
        for nested_placeholder in placeholders:
            if nested_placeholder in top_level_placeholders:
                continue
            if isinstance(get_custom(nested_placeholder), IterArg):
                continue
            # See if this placeholder is an implicit capture.
            captured_node: fx.Node | None = get_custom(
                nested_placeholder
            ).get_captured_fx_node()
            if not nested_placeholder.name in top_level_names:
                if captured_node is None:
                    raise RuntimeError(
                        f"Incorrectly structured placeholders in trace: "
                        f"placeholder '{nested_placeholder.name}' not found in top-level "
                        f"names {top_level_names} and does not capture another node."
                    )
                implicit_captures[nested_placeholder] = captured_node
                continue
            value_map[nested_placeholder] = value_map[
                top_level_placeholders[top_level_names.index(nested_placeholder.name)]
            ]

        with ir.InsertionPoint(entry_block):
            _emit_ops_from_graph(
                trace.get_root_graph(),
                trace,
                value_map,
                implicit_captures,
                ctx,
                known_ids,
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

    diagnostics: list[MLIRDiagnostic | WaterError] = []

    def diagnostics_handler(d):
        diagnostics.append(
            MLIRDiagnostic(
                message=d.message,
                location=serialize_location(d.location),
                severity=d.severity.name,
            )
        )
        return True

    enable_debug_info = (
        options.location_capture_config.level is not LocationCaptureLevel.NONE
    )

    if enable_debug_info and not trace.location:
        diagnostics.append(WaterError(message="Missing debug location for wave trace"))

    with (
        ir.Context() as ctx,
        trace.location.to_water() if trace.location else ir.Location.unknown(),
    ):
        ctx.attach_diagnostic_handler(diagnostics_handler)
        ctx.allow_unregistered_dialects = False
        wave.register_dialect(ctx)
        wave.register_passes()

        module, creation_diagnostics, known_ids = _create_kernel_module(
            ctx, trace, constraints, options, test_diagnostics
        )
        diagnostics += creation_diagnostics
        if module is None:
            _flush_output("", diagnostics, None)
            return 0

        # Verify the module before transforming or printing.
        try:
            module.operation.verify()
        except ir.MLIRError as e:
            diagnostics.append(WaterError(message=e.message))
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
                diagnostics.append(
                    WaterError(message=f"Failed to apply transform script: {e}")
                )

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
