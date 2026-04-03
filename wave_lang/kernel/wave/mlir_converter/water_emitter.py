"""
Water Emitter for Wave Dialect

This generates operations of the Wave dialect from a serialized wave trace.
It runs as a standalone process with access to Water Python bindings and emits
mlir operations of wave and other dialects.
"""

from __future__ import annotations
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


if TYPE_CHECKING:
    from wave_lang.kernel._support.indexing import IndexSequence, IndexSymbol
    from wave_lang.kernel.ops.wave_ops import *

from wave_lang.kernel.wave.mlir_converter.diagnostics import (
    FileLocation,
    LocationFrame,
    MLIRDiagnostic,
    NameLocation,
    WaterDiagTestingMode,
    WaterError,
)
from wave_lang.kernel.wave.mlir_converter import dill_util
from wave_lang.support.location_config import LocationCaptureLevel
from wave_lang.kernel.lang.wave_types import Memory, Register, IndexMapping
from wave_lang.kernel.lang.kernel_buffer import AddressSpace
from wave_lang.kernel._support.indexing import IndexSymbol, IndexSequence, safe_subs
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.utils.general_utils import infer_dim
from wave_lang.kernel.wave.utils.symbol_utils import (
    collect_allowed_induction_symbols,
    strip_out_of_scope_induction_symbols,
)

from wave_lang.kernel.ops.wave_ops import (
    Allocate,
    ApplyExpr,
    BinaryOpBase,
    BitcastOp as Bitcast,
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
    Read,
    ReduceOp as Reduce,
    Reshape,
    ScaledMMA,
    SelfIndex,
    SelectOp as Select,
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
        BitcastOp,
        SubOp,
        AllocateOp,
        BroadcastOp,
        CastOp,
        DivOp,
        ReciprocalOp,
        ReshapeOp,
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
        ScaledMmaOp,
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
    "bitcast": BitcastOp,
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
    "reshape": ReshapeOp,
    "exp2": Exp2Op,
    "read": ReadOp,
    "register": RegisterOp,
    "scaled_mma": ScaledMmaOp,
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
    "f8E5M2": ir.Float8E5M2Type.get,
    "f8E5M2FNUZ": ir.Float8E5M2FNUZType.get,
    "f8E4M3FN": ir.Float8E4M3FNType.get,
    "f8E4M3FNUZ": ir.Float8E4M3FNUZType.get,
    "f8E8M0FNU": ir.Float8E8M0FNUType.get,
    "f6E2M3FN": ir.Float6E2M3FNType.get,
    "f4E2M1FN": ir.Float4E2M1FNType.get,
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


def _derived_dim_clean_name(expr: sympy.Expr) -> str:
    """Return a clean MLIR-friendly name for a derived SymPy expression, or
    the original name if the expression is already a plain symbol"""
    if expr.is_Symbol or expr.is_Number:
        return expr.name
    if len(expr.free_symbols) != 1:
        raise NotImplementedError(
            f"expressions with more than one free symbol {expr} are not yet supported"
        )
    base = list(expr.free_symbols)[0]
    _, denom = expr.as_numer_denom()
    if isinstance(denom, sympy.Integer) and denom > 1:
        return f"{base.name}{int(denom)}"
    if isinstance(expr, sympy.Mul):
        args = expr.args
        ints = [a for a in args if a.is_Integer]
        if len(ints) == 1:
            return f"{base.name}{int(ints[0])}"
    raise NotImplementedError(f"shortening expression {expr} is not yet supported")


def _create_wave_tensor_type(
    ctx: ir.Context,
    symbolic_shape: tuple,
    dtype,
    address_space_attr: WaveAddressSpaceAttr,
) -> ir.Type:
    """Helper function to create a WaveTensorType from shape, dtype, and address space."""
    shape_attrs = []
    for s in symbolic_shape:
        clean = _derived_dim_clean_name(s)
        shape_attrs.append(WaveSymbolAttr.get(clean if clean else str(s)))
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

    # Bare symbolic fractions like BLOCK_K/32 cannot be represented as affine
    # expressions directly. Wrap them in ceiling() so they lower to ceildiv.
    # All Wave symbols are positive integers, so ceiling(x/n) == x/n when x is
    # divisible by n.
    if isinstance(expr, sympy.Mul):
        _, denom = expr.as_numer_denom()
        if denom != 1 and not isinstance(denom, sympy.Integer):
            pass
        elif isinstance(denom, sympy.Integer) and denom > 1:
            expr = sympy.ceiling(expr)

    return convert_sympy_to_affine_map(
        expr, [sym.name for sym in symbol_mapping.values()]
    )


def _remap_scaled_index_keys(
    index: dict[IndexSymbol, IndexSequence], shape: list[sympy.Expr]
) -> dict[sympy.Expr, IndexSequence]:
    """Remap base symbol keys in an index dict to scaled expressions from shape.

    The Water verifier requires index dict key names to match the dimension
    names in the op's result type.  For scaled dims (e.g. K/2 → "K2", K/32 →
    "K32") the Python index analysis stores the base symbol key (K), so we
    remap it to the scaled expression using the provided shape.
    """
    base_to_scaled = {
        s: expr
        for expr in shape
        if not expr.is_Symbol
        for s in expr.free_symbols
        if s != expr
    }
    if not base_to_scaled:
        return index
    return {base_to_scaled.get(dim, dim): seq for dim, seq in index.items()}


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
        index_mappings[_derived_dim_clean_name(dim)] = wave.WaveIndexMappingAttr.get(
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
            # match MmaOp::getIndexExprValuesAndDescriptions which serializes
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
        elif isinstance(node, ScaledMMA):
            # ScaledMMA needs exactly 6 index entries (lhs, lhs_scale, rhs, rhs_scale, acc, result) to
            # match ScaledMmaOp::getIndexExprValuesAndDescriptions which serializes
            # operandExprs + resultExprs.  The Python-side index sequence
            # analysis only tracks 5 (lhs, lhs_scale, rhs, rhs_scale, acc), so we emit acc_index
            # twice: once for the accumulator operand and once for the
            # result (ScaledMMA result type == acc type).
            if lhs_index := getattr(node, "lhs_index", None):
                dict_attrs.append(
                    _build_index_mapping_dict(lhs_index, allowed_induction_symbols)
                )
            if lhs_scale_index := getattr(node, "lhs_scale_index", None):
                # index_expr from lhs_scale_index is the un-scaled index, so we need to remap it to the original index.
                lhs_scale_shape = node.lhs_scale_type.symbolic_shape
                res = {}
                for dim_expr in lhs_scale_shape:
                    dim = infer_dim(dim_expr)
                    res[dim_expr] = IndexSequence(
                        start=dim_expr.subs({dim: lhs_scale_index[dim].start}),
                        size=dim_expr.subs({dim: lhs_scale_index[dim].size}),
                        stride=1,
                    )
                dict_attrs.append(
                    _build_index_mapping_dict(res, allowed_induction_symbols)
                )
            if rhs_index := getattr(node, "rhs_index", None):
                dict_attrs.append(
                    _build_index_mapping_dict(rhs_index, allowed_induction_symbols)
                )
            if rhs_scale_index := getattr(node, "rhs_scale_index", None):
                # index_expr from rhs_scale_index is the un-scaled index, so we need to remap it to the original index.
                rhs_scale_shape = node.rhs_scale_type.symbolic_shape
                res = {}
                for dim_expr in rhs_scale_shape:
                    dim = infer_dim(dim_expr)
                    res[dim_expr] = IndexSequence(
                        start=dim_expr.subs({dim: rhs_scale_index[dim].start}),
                        size=int(dim_expr.subs({dim: rhs_scale_index[dim].size})),
                        stride=1,
                    )
                dict_attrs.append(
                    _build_index_mapping_dict(res, allowed_induction_symbols)
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
            result_shape = getattr(getattr(node, "type", None), "symbolic_shape", [])
            index = _remap_scaled_index_keys(node.index, result_shape)
            dict_attrs.append(
                _build_index_mapping_dict(index, allowed_induction_symbols)
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
        op.attributes["bounds"] = wave.WaveSymbolMappingAttr.get(bounds)

    if getattr(node, "vector_shapes", None):
        if isinstance(node, ScaledMMA):
            # ScaledMMA needs exactly 6 vector_shape entries: lhs, lhs_scale, rhs, rhs_scale, acc, result.
            vector_shape_entries = []
            lhs_vector_shapes = getattr(node.lhs, "vector_shapes", None)
            lhs_scale_vector_shapes = getattr(node.lhs_scale, "vector_shapes", None)
            rhs_vector_shapes = getattr(node.rhs, "vector_shapes", None)
            rhs_scale_vector_shapes = getattr(node.rhs_scale, "vector_shapes", None)
            acc_vector_shapes = getattr(node.acc, "vector_shapes", None)
            result_vector_shapes = getattr(node, "vector_shapes", None)
            some_vector_shapes = any(
                [
                    lhs_vector_shapes,
                    lhs_scale_vector_shapes,
                    rhs_vector_shapes,
                    rhs_scale_vector_shapes,
                    acc_vector_shapes,
                    result_vector_shapes,
                ]
            )
            if some_vector_shapes:
                assert (
                    lhs_vector_shapes is not None
                ), "ScaledMMA missing LHS vector shapes."
                assert (
                    lhs_scale_vector_shapes is not None
                ), "ScaledMMA missing LHS scale vector shapes."
                assert (
                    rhs_vector_shapes is not None
                ), "ScaledMMA missing RHS vector shapes."
                assert (
                    rhs_scale_vector_shapes is not None
                ), "ScaledMMA missing RHS scale vector shapes."
                assert (
                    acc_vector_shapes is not None
                ), "ScaledMMA missing ACC vector shapes."
                assert (
                    result_vector_shapes is not None
                ), "ScaledMMA missing result vector shapes."
                vector_shape_entries.extend(
                    _convert_vector_shapes(x)
                    for x in [
                        lhs_vector_shapes,
                        lhs_scale_vector_shapes,
                        rhs_vector_shapes,
                        rhs_scale_vector_shapes,
                        acc_vector_shapes,
                        result_vector_shapes,
                    ]
                )
            op.attributes["vector_shape"] = ir.ArrayAttr.get(vector_shape_entries)
        elif isinstance(node, MMA):
            # MMA needs exactly 4 vector_shape entries: lhs, rhs, acc, result.
            vector_shape_entries = []
            lhs_vector_shapes = getattr(node.lhs, "vector_shapes", None)
            rhs_vector_shapes = getattr(node.rhs, "vector_shapes", None)
            acc_vector_shapes = getattr(node.acc, "vector_shapes", None)
            result_vector_shapes = getattr(node, "vector_shapes", None)
            some_vector_shapes = any(
                [
                    lhs_vector_shapes,
                    rhs_vector_shapes,
                    acc_vector_shapes,
                    result_vector_shapes,
                ]
            )
            if some_vector_shapes:
                assert lhs_vector_shapes is not None, "MMA missing LHS vector shapes."
                assert rhs_vector_shapes is not None, "MMA missing RHS vector shapes."
                assert acc_vector_shapes is not None, "MMA missing ACC vector shapes."
                assert (
                    result_vector_shapes is not None
                ), "MMA missing result vector shapes."
                vector_shape_entries.extend(
                    _convert_vector_shapes(x)
                    for x in [
                        lhs_vector_shapes,
                        rhs_vector_shapes,
                        acc_vector_shapes,
                        result_vector_shapes,
                    ]
                )
            op.attributes["vector_shape"] = ir.ArrayAttr.get(vector_shape_entries)
        else:
            op.attributes["vector_shape"] = ir.ArrayAttr.get(
                [_convert_vector_shapes(node.vector_shapes)]
            )

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


def _parse_mma_kind(mma_type, ctx: ir.Context) -> ir.Attribute | None:
    """Parse an MMA type into an MLIR WaveMmaKindAttr, or None if mma_type is None."""
    if mma_type is None:
        return None
    return ir.Attribute.parse(f"#wave.mma_kind<{mma_type.name.lower()}>", context=ctx)


def _convert_vector_shapes(
    shapes: dict[IndexSymbol, int | IndexExpr], subs: dict[IndexExpr, Any] = {}
) -> "wave.WaveSymbolMappingAttr":
    """Converts a dictionary of index symbols mapped to integers to a WaveSymbolMappingAttr."""
    resolved = _resolve_vector_shapes_for_attr(shapes, subs)
    i64 = ir.IntegerType.get_signless(64)
    mapping = {k: ir.IntegerAttr.get(i64, v) for k, v in resolved.items()}
    return wave.WaveSymbolMappingAttr.get(mapping)


def _symbol_mapping_to_dict_attr(
    attr: "wave.WaveSymbolMappingAttr",
) -> ir.DictAttr:
    """Converts a WaveSymbolMappingAttr to a DictionaryAttr.

    For use with attributes that have not yet been migrated to WaveSymbolMappingAttr.
    """
    return ir.DictAttr.get({key.name: value for key, value in attr})


def _convert_index_mapping_to_water(
    mapping: IndexMapping, memory_shape: Sequence[IndexExpr] | None, *, is_read: bool
) -> ir.Attribute:
    """Converts an IndexMapping to a water attribute assuming it is used for the specified shape.

    Currently only supports permutation mappings. The `is_read` flag indicates
    whether to use the input mapping (True) or output mapping (False). This is
    due to excessively expressive design choice in pywave: the unused mapping
    isn't actually needed beyond IndexMapping construction.
    """
    if memory_shape is None:
        raise RuntimeError(
            "Memory type shape is required for index mapping conversion."
        )

    filtered_shape: list[IndexSymbol] = []
    for dim in memory_shape:
        if not isinstance(dim, IndexSymbol):
            raise NotImplementedError(
                f"Only single-symbol shapes are currently supported, got {dim}"
            )
        filtered_shape.append(dim)
    iterators: tuple[IndexExpr, ...] = (
        # TODO: the function expects a `tuple` but doesn't actually need that, any iterable would do.
        mapping.map_input_indices(tuple(filtered_shape))
        if is_read
        else mapping.map_output_indices(tuple(filtered_shape))
    )

    # Consecutive iterators are assigned to the value (non-memory)
    # shape, and their order may be different in the memory shape.
    # The mapping is therefore (d0, d1, d2, ...) -> (iterator positions)
    # inverted because, in water, the mapping goes the from memory shape
    # to the value shape.
    #
    # Construct a list of iterators in order and look up for
    # position in that list to find numeric permutation indices
    # without parsing symbol names to extract the iterator position.
    # If we had a proper data structure instead of blindly relying
    # on sympy symbols everywhere, this would have been an easy to
    # access property...
    ordered_iterators = [IndexMapping.iterator(i) for i in range(len(iterators))]
    permutation = [ordered_iterators.index(iter) for iter in iterators]
    inverse_permutation = [permutation.index(i) for i in range(len(permutation))]
    return wave.WaveExprListAttr.get(
        [], ir.AffineMap.get_permutation(inverse_permutation)
    )


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

                captured_fx_node = implicit_captures[fx_node]
                assert (
                    captured_fx_node in value_map
                ), f"{node} implicitly captures a value that was not translated."

                # Body nodes downstream of implicit captures need typed inputs
                # to infer their own types. Before infer_types runs, the type
                # of the placeholders is unset - propagate from the
                # already-typed captured node.
                if node.type is None:
                    captured_type = get_custom(captured_fx_node).type
                    if captured_type is not None:
                        node.type = captured_type

                value_map[fx_node] = value_map[captured_fx_node]
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
                    if len(mlir_args) == 0:
                        if allow_missing:
                            return None
                        raise RuntimeError(f"Empty mapping for node {node}")
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
                    args = (
                        arg
                        if isinstance(arg, Sequence) and not isinstance(arg, str)
                        else [arg]
                    )
                    mlir_operands.extend(
                        mlir_arg
                        for subarg in args
                        if (
                            mlir_arg := get_single_mapped_value(
                                subarg, allow_missing=True
                            )
                        )
                        is not None
                    )
                return mlir_operands

            def attach_get_result_attr(
                iterate_op: IterateOp,
                position: int,
                name: str,
                attr: ir.Attribute,
                *,
                prepend: bool,
            ):
                current_attribute = (
                    iterate_op.attributes[name]
                    if name in iterate_op.attributes
                    else None
                )
                # When prepend=True, slot 0 is reserved for the iterate's
                # own entry (already set by _attach_attributes).  Result
                # entries occupy slots 1..N.
                effective_position = (position + 1) if prepend else position
                total_slots = len(iterate_op.results) + (1 if prepend else 0)
                if current_attribute is None:
                    attribute_list = [ir.UnitAttr.get()] * total_slots
                elif (
                    isinstance(current_attribute, ir.ArrayAttr)
                    and len(current_attribute) < total_slots
                ):
                    attribute_list = list(current_attribute) + [ir.UnitAttr.get()] * (
                        total_slots - len(current_attribute)
                    )
                else:
                    attribute_list = list(current_attribute)
                attribute_list[effective_position] = attr
                iterate_op.attributes[name] = ir.ArrayAttr.get(attribute_list)

            if isinstance(node, GetResult):
                if node.res_idx >= len(value_map[node.value]):
                    raise RuntimeError(
                        f"GetResult index ({node.res_idx}) is higher than the "
                        f"number of mapped values ({len(value_map[node.value])})."
                    )
                value_map[fx_node] = (value_map[node.value][node.res_idx],)

                # Attach IDs and vector_shapes of `get_result` to the loop
                # instead so we can recover them later because `get_result`
                # doesn't exist in the dialect. Only do so when there are some
                # results.
                if len(value_map[node.value]) > 0:
                    iterate_op = get_single_mapped_value(fx_node).owner
                    if vector_shapes := getattr(fx_node, "vector_shapes", None):
                        attach_get_result_attr(
                            iterate_op,
                            node.res_idx,
                            "vector_shape",
                            _convert_vector_shapes(vector_shapes),
                            prepend=True,
                        )
                    if known_ids is not None:
                        water_id = getattr(fx_node, "_water_id", None)
                        if water_id is None:
                            raise RuntimeError(
                                f"Water id requested for 'get_result' but not specified: {node}"
                            )
                        known_ids.add(water_id)
                        attach_get_result_attr(
                            iterate_op,
                            node.res_idx,
                            _INTERNAL_RESULT_WATER_IRS_ATTR_NAME,
                            ir.StringAttr.get(water_id),
                            prepend=False,
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
                if isinstance(node, Write):
                    mlir_op = op_builder(
                        get_single_mapped_value(node.register_),
                        get_single_mapped_value(node.memory),
                        mapping=(
                            _convert_index_mapping_to_water(
                                node.mapping,
                                node.memory_type.symbolic_shape,
                                is_read=False,
                            )
                            if node.mapping
                            else None
                        ),
                    )
                elif isinstance(node, Read):
                    mlir_op = op_builder(
                        result_type,
                        get_single_mapped_value(node.memory),
                        mapping=(
                            _convert_index_mapping_to_water(
                                node.mapping,
                                node.memory_type.symbolic_shape,
                                is_read=True,
                            )
                            if node.mapping
                            else None
                        ),
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

                    # Derive result types from init_args rather than from
                    # the body outputs.  init_args reside in the enclosing
                    # graph whose nodes have already been emitted, so their
                    # types are always available.  Body outputs may reference
                    # nodes whose types are not yet set (e.g. reductions in
                    # attention's online-softmax loop).
                    result_types = []
                    result_locs = []
                    for init_arg in node.init_args:
                        init_custom = get_custom(init_arg)
                        result_types.append(_type_to_wave_mlir(ctx, init_custom.type))
                        result_locs.append(
                            init_custom.location.to_water()
                            if init_custom.location
                            else ir.Location.current
                        )
                    outputs = node.outputs()

                    mlir_op = op_builder(result_types, axis, carried_values, [])
                    body = ir.Block.create_at_start(
                        mlir_op.regions[0], result_types, result_locs
                    )

                    # Collect IterArgs in graph order and assign
                    # iter_idx before any call that sorts by it.
                    # node.iter_args() sorts by iter_idx which may
                    # still be None before initialize_iter_args runs.
                    subgraph = trace.get_subgraph(node.subgraph_name)
                    iter_arg_nodes = [
                        n for n in subgraph.nodes if isinstance(get_custom(n), IterArg)
                    ]
                    for idx, ia_node in enumerate(iter_arg_nodes):
                        get_custom(ia_node).iter_idx = idx

                    for wave_arg, mlir_arg in zip(iter_arg_nodes, body.arguments):
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
                    mma_kind = _parse_mma_kind(node.mma_type, ctx)
                    mlir_op = op_builder(
                        result_type, *create_mlir_operands(), kind=mma_kind
                    )
                elif isinstance(node, ScaledMMA):
                    mma_kind = _parse_mma_kind(node.mma_type, ctx)
                    mlir_op = op_builder(
                        result_type, *create_mlir_operands(), kind=mma_kind
                    )
                elif isinstance(node, Bitcast):
                    mlir_op = op_builder(result_type, get_single_mapped_value(node.arg))
                elif isinstance(node, Reduce):
                    args = node.arg if isinstance(node.arg, Sequence) else [node.arg]
                    inputs = [get_single_mapped_value(a) for a in args]
                    init = get_single_mapped_value(node.init)
                    # The axis attribute is only emitted when the input type
                    # is not fully specified (uses 'any' shapes). For
                    # fully-specified types, the reduction dimension is
                    # inferred from the shape difference between input and
                    # result, and emitting axis would be rejected by the
                    # verifier.
                    input_type = inputs[0].type
                    fully_specified = (
                        isinstance(input_type, WaveTensorType)
                        and input_type.fully_specified
                    )
                    axis = (
                        symbol_name_to_attribute(node.dim.name)
                        if node.dim is not None and not fully_specified
                        else None
                    )
                    mlir_op = op_builder(
                        result_type,
                        inputs,
                        init,
                        scope=wave.WaveReductionScopeAttr.get(
                            wave.WaveReductionScope.Block
                            if node.block
                            else wave.WaveReductionScope.Warp
                        ),
                        axis=axis,
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
                elif isinstance(node, Reshape):
                    mlir_op = op_builder(
                        result_type,
                        source=create_mlir_operands(),
                        # ReshapeOp.target_vector_shape is still DictionaryAttr.
                        target_vector_shape=_symbol_mapping_to_dict_attr(
                            _convert_vector_shapes(node.target_vector_shape)
                        ),
                        logical_slice=node.logical_slice,
                        num_slices=node.num_slices,
                    )
                elif isinstance(node, (BinaryOpBase, Select)):
                    # BinaryOpBase and SelectOp allow implicit broadcasting
                    # on the Python side (e.g. a rank-2 reduction result
                    # used with a rank-3 tensor), but the MLIR verifier
                    # requires matching operand ranks. Insert explicit
                    # broadcasts for any operand whose shape differs from
                    # the result. The broadcast target keeps each operand's own
                    # element type so that e.g. a SelectOp condition (i1) is
                    # broadcast to the result shape without being cast to the
                    # result element type (e.g. f32).
                    #
                    # We intentionally limit this to ops with known
                    # broadcasting semantics rather than applying it
                    # generically, because many ops (read, mma, ...)
                    # have operands that are intentionally shaped
                    # differently from the result.
                    operands = create_mlir_operands()
                    result_shape = result_type.shape
                    operands = [
                        (
                            BroadcastOp(
                                WaveTensorType.get(
                                    result_shape,
                                    result_type.fully_specified,
                                    v.type.element_type,
                                    result_type.address_space,
                                ),
                                v,
                            ).results[0]
                            if v.type.shape != result_shape
                            else v
                        )
                        for v in operands
                    ]
                    mlir_op = op_builder(result_type, *operands)
                else:
                    try:
                        mlir_op = op_builder(result_type, *create_mlir_operands())
                    except Exception as e:
                        raise RuntimeError(
                            f"Could not map arguments correctly for MLIR constructor of '{node.tkw_op_name}' operation"
                        ) from e

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
            shape_dict = _convert_vector_shapes(constraint.vector_shapes, subs)

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


def diagnostic_from_mlir_error(e: ir.MLIRError) -> list[MLIRDiagnostic]:
    """Convert an MLIR exception into a list of Wave-compatible diagnostics.

    Each note is added as a separate diagnostic. If it has the same location as
    the previous diagnostic, the location is omitted.
    """
    diagnostics = []
    for d in e.error_diagnostics:
        diagnostics.append(
            MLIRDiagnostic(
                message=d.message,
                severity=d.severity.name,
                location=serialize_location(d.location),
            )
        )
        previous_location = d.location
        for n in d.notes:
            diagnostics.append(
                MLIRDiagnostic(
                    message=n.message,
                    severity=n.severity.name,
                    location=(
                        serialize_location(n.location)
                        if n.location != previous_location
                        else []
                    ),
                )
            )
            previous_location = n.location
    return diagnostics


def _create_kernel_module(
    ctx: ir.Context,
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
    *,
    test_diagnostics: WaterDiagTestingMode = WaterDiagTestingMode.NO,
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
            diagnostics.extend(diagnostic_from_mlir_error(e))
            return None, diagnostics, known_ids
        else:
            return module, diagnostics, known_ids

    # Keep track of which emitted value stems from what node to wire
    # arguments correctly.
    value_map: dict[fx.Node | fx.Proxy, tuple[ir.Value, ...]] = {}

    module = ir.Module.create()

    if test_diagnostics == WaterDiagTestingMode.DIRECT:
        # A suitable location is the first non-placeholder op (placeholders use
        # "name" locations that are near-useless).
        candidates = trace.walk(
            lambda n: not isinstance(get_custom(n), Placeholder)
            and get_custom(n).location
        )
        loc = candidates[0].location.to_water() if candidates else ir.Location.current
        loc.emit_error("test error")
        return module, None, None

    # Having a duplicate function name in the module is very likely to remain a
    # verifier error for the observable future. If that is not the case,
    # programmatically construct other IR that causes verification failure so we
    # can exercise the diagnostic reporting mechanism.
    if test_diagnostics == WaterDiagTestingMode.VERIFIER:
        with ir.InsertionPoint(module.body):
            func.FuncOp(
                "repeated_name", ir.FunctionType.get([], []), visibility="private"
            )
            func.FuncOp(
                "repeated_name", ir.FunctionType.get([], []), visibility="private"
            )
        return module, None, None

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
            if resolved_address_space not in (
                GLOBAL_ADDRESS_SPACE,
                SHARED_ADDRESS_SPACE,
            ):
                raise RuntimeError(
                    f"Unexpected address space in hyperparameters: "
                    f"{t.address_space} -> {resolved_address_space}"
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
        # Build the hyperparameters dict: base symbols map to integers,
        # derived dims (e.g. K/2) map to WaveExprListAttr expressions.
        hyper_params: dict[str, int | WaveExprListAttr] = {
            str(k): v for k, v in options.subs.items() if isinstance(v, int)
        }
        # Collect derived symbolic dimensions from tensor shapes across the
        # graph.  Instead of pre-evaluating them to integers, encode the
        # affine relationship so the C++ backend can resolve them symbolically.
        for node in trace.walk():
            c = get_custom(node)
            t = getattr(c, "_type", None) or getattr(c, "type", None)
            shapes = []
            if t is not None:
                shapes = getattr(t, "symbolic_shape", None) or []
            for dim_expr in shapes:
                clean = _derived_dim_clean_name(dim_expr)
                if clean is None or clean in hyper_params:
                    continue
                free = list(dim_expr.free_symbols)
                if len(free) != 1:
                    continue
                base_sym = free[0]
                if str(base_sym) not in hyper_params:
                    continue
                sym_mapping = preprocess_symbols(free)
                affine_map = _convert_sympy_expr_to_affine_map(dim_expr, sym_mapping)
                sym_attrs = [
                    symbol_name_to_attribute(sym.name) for sym in sym_mapping.values()
                ]
                hyper_params[clean] = WaveExprListAttr.get(sym_attrs, affine_map)
        func_op.operation.attributes["wave.hyperparameters"] = (
            wave.WaveHyperparameterAttr.get(hyper_params)
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


def _build_response(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
    pipeline: str = "",
    test_diagnostics: WaterDiagTestingMode = WaterDiagTestingMode.NO,
) -> bytes:
    """Core conversion logic that returns the dill-serialized response bytes.

    Shared between single-shot mode and server mode.
    """
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
            ctx, trace, constraints, options, test_diagnostics=test_diagnostics
        )
        if creation_diagnostics:
            diagnostics.extend(creation_diagnostics)
        if module is None:
            return _serialize_response("", diagnostics, None)

        # Verify the module before transforming or printing. Note that the call
        # to explicit `verify` registers its own diagnostic handler and raises
        # an exception instead of going through the handler we registered.
        # Therefore we need to manually unpack the exception into `diagnostics`.
        try:
            module.operation.verify()
        except ir.MLIRError as e:
            diagnostics.extend(diagnostic_from_mlir_error(e))
            # Print in generic form if verification fails, this form should be
            # robust to that.
            return _serialize_response(
                module.operation.get_asm(
                    enable_debug_info=enable_debug_info, print_generic_op_form=True
                ),
                diagnostics,
                None,
            )

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
                        f'Expected first op to be "transform.named_sequence", '
                        f'got "{entry_op.operation.name}"'
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
                    element: int = 0,
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
                                op, op.attributes["index"], element
                            )
                        }
                    )

                if attribute is not None:
                    record_index(attribute, inferred_attributes)
                if result_attribute is not None:
                    assert isinstance(
                        result_attribute, ir.ArrayAttr
                    ), f"Unexpected attribute type: {result_attribute}."
                    for i, attribute in enumerate(result_attribute):
                        record_index(attribute, inferred_attributes, i)

                return ir.WalkResult.ADVANCE

            module.operation.walk(extractor)
            for water_id, inferred_attribute in inferred_attributes.items():
                if "index" not in inferred_attribute:
                    raise RuntimeError(f"Index not inferred for water id {water_id}.")

        module_str = module.operation.get_asm(enable_debug_info=enable_debug_info)
        return _serialize_response(module_str, diagnostics, inferred_attributes)


def _serialize_response(
    module_str: str,
    diagnostics: list[MLIRDiagnostic | WaterError],
    inferred_attributes: dict[str, dict[str, Any]] | None = None,
) -> bytes:
    return dill_util.dumps(
        {
            "diagnostics": diagnostics,
            "module": module_str.encode("utf-8"),
            "inferred_attributes": (
                inferred_attributes if inferred_attributes is not None else {}
            ),
        }
    )


def _server_mode() -> None:
    """Run as a persistent server, reading length-prefixed requests from stdin.

    Instead of spawning a fresh process per request (which costs ~2 s in
    module imports each time), the parent keeps this process alive and
    sends multiple requests over the same stdin/stdout pipes.

    Each request payload is a dill-serialized dict with the same keys as
    in single-shot mode (trace, constraints, options, pipeline, ...).
    Each response payload is a dill-serialized dict with keys
    `diagnostics`, `module`, and `inferred_attributes`.

    The loop exits when stdin is closed (EOF on the length header), i.e.
    when the parent closes its end of the pipe.
    """
    from wave_lang.kernel.wave.mlir_converter.protocol import (
        recv_message,
        send_message,
    )

    while True:
        try:
            raw_request = recv_message(sys.stdin.buffer)
        except EOFError:
            break  # Parent closed stdin, clean shutdown.
        except ConnectionError as e:
            # The parent likely crashed mid-write. The response pipe is
            # broken so we can't send a diagnostic back, stderr is
            # best-effort.
            print(f"WARNING: {e}", file=sys.stderr)
            break

        try:
            request = dill_util.loads(raw_request)
        except Exception as e:
            response_data = _serialize_response(
                "", [WaterError(message=f"Failed to unpickle request: {e}")], None
            )
            send_message(sys.stdout.buffer, response_data)
            continue

        trace = request.get("trace")
        constraints = request.get("constraints", [])
        options = request.get("options", WaveCompileOptions())
        pipeline = request.get("pipeline", "")
        test_diags = request.get("test_diagnostic_emission", WaterDiagTestingMode.NO)

        if not isinstance(trace, CapturedTrace):
            response_data = _serialize_response(
                "",
                [WaterError(message=f"Expected CapturedTrace, got {type(trace)}")],
                None,
            )
            send_message(sys.stdout.buffer, response_data)
            continue

        trace.restore_node_state()

        try:
            response_data = _build_response(
                trace, constraints, options, pipeline, test_diags
            )
        except Exception as e:
            response_data = _serialize_response("", [WaterError(message=str(e))], None)

        send_message(sys.stdout.buffer, response_data)


if __name__ == "__main__":
    _server_mode()
