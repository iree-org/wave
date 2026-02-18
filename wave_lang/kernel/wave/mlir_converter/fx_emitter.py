# Copyright 2026 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

from __future__ import annotations

from enum import Enum
import itertools
import sys
from typing import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import dill

if __name__ == "__main__":
    # Add parent directory to sys.path to enable relative imports when running standalone.
    # This allows importing water_mlir from ../water_mlir/ as if it were a relative import.
    _current_file = Path(__file__).resolve()
    _parent_dir = str(_current_file.parent.parent)  # Go up to wave_lang/kernel/wave/
    if _parent_dir not in sys.path:
        sys.path.append(_parent_dir)
    # Add current directory to enable importing attr_type_converter without full package path.
    _current_dir = str(_current_file.parent)
    if _current_dir not in sys.path:
        sys.path.append(_current_dir)

try:
    from water_mlir.water_mlir import ir
    from water_mlir.water_mlir.dialects import wave, arith, amdgpu, func
    from water_mlir.water_mlir.dialects.wave import (
        AllocateOp,
        ReadOp,
        WriteOp,
        MmaOp,
        RegisterOp,
        ExtractSliceOp,
        IterateOp,
        YieldOp,
        WaveAddressSpaceAttr,
        WaveMmaKindAttr,
        WaveSymbolMappingAttr,
        WaveWorkgroupDimAttr,
        WaveTensorType,
        iterate_make_isolated,
    )

except Exception as e:
    # Keep import-time errors explicit so callers can act on missing deps.
    raise ImportError(f"Failed to import water_mlir bindings: {e}") from e

import torch.fx as fx

from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel._support.regions import RegionGraph
from wave_lang.kernel._support.tracing import KernelRegionGraph
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.constraints import (
    Constraint,
    MMAType,
    ScaledMMAType,
    WorkgroupConstraint,
    WaveConstraint,
    TilingConstraint,
    DeviceConstraint,
    HardwareConstraint,
)
from wave_lang.kernel.wave.mlir_converter.diagnostics import MLIRDiagnostic
from wave_lang.kernel.wave.mlir_converter.mlir_converter import FxEmitterResponse
from wave_lang.kernel.wave.mlir_converter.water_emitter import serialize_location
from wave_lang.kernel.wave.utils.symbol_utils import get_induction_symbol
from wave_lang.support.indexing import index_symbol, IndexSequence
from wave_lang.kernel._support.indexing import IndexExpr, IndexSymbol, safe_subs
from wave_lang.kernel.lang.wave_types import Memory, Register
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from wave_lang.kernel.ops.wave_ops import (
    Allocate,
    Read,
    Write,
    MMA,
    MMABase,
    NewRegister,
    ExtractSlice,
    Iterate,
    IterArg,
    Placeholder,
    Output,
    GetResult,
    SharedMemoryBarrier,
    get_custom,
)
from attr_type_converter import (
    convert_index_mapping_array_to_sympy,
    expr_list_attr_to_exprs,
    mlir_element_type_to_dtype,
    symbol_attr_to_name,
)


# Converted attribute value: The union of all types that may be produced
# for a single MLIR attribute.
AttrValue = (
    dict[IndexSymbol, IndexSequence]
    | dict[IndexSymbol, IndexExpr]
    | int
    | MMAType
    | ScaledMMAType
    | str
)


class AttrNames(Enum):
    """Maps MLIR attribute names to FX node property names.

    Each member is a (mlir_attr_name, fx_property_name) tuple. Use .mlir_name
    to look up attributes on MLIR operations and .fx_property to set/get the
    corresponding property on FX nodes.
    """

    INDEX = ("index", "index")
    BOUNDS = ("bounds", "bounds")
    ELEMENTS_PER_THREAD = ("elements_per_thread", "elements_per_thread")
    KIND = ("kind", "mma_type")
    WATER_INTERNAL_ID = ("_water_internal.id", "_water_id")
    OFFSET = ("offset", "offset")
    SIZE = ("size", "size")
    STRIDE = ("stride", "stride")
    ORDERED_SYMS = ("ordered_syms", "ordered_syms")
    VALUE = ("value", "value")

    @property
    def mlir_name(self) -> str:
        """MLIR attribute name."""
        return self.value[0]

    @property
    def fx_property(self) -> str:
        """FX node property name."""
        return self.value[1]


def _address_space_to_symbol(
    addr_space_attr: WaveAddressSpaceAttr,
    parse_ctx: _OpParseContext,
) -> IndexSymbol:
    """Map a WaveAddressSpaceAttr to the IndexSymbol used in Memory type constructors.

    Global and Shared map to the well-known symbols GLOBAL_ADDRESS_SPACE and
    SHARED_ADDRESS_SPACE.  Unspecified produces a fresh unique symbol each time
    so that distinct unresolved address spaces are never accidentally conflated.
    """
    match addr_space_attr.value:
        case wave.WaveAddressSpace.Global:
            return GLOBAL_ADDRESS_SPACE
        case wave.WaveAddressSpace.Shared:
            return SHARED_ADDRESS_SPACE
        case wave.WaveAddressSpace.Unspecified:
            idx = next(parse_ctx.unspecified_addr_counter)
            return index_symbol(f"$UNSPECIFIED_ADDRESS_SPACE_{idx}")
        case _:
            raise ValueError(f"Unsupported address space: {addr_space_attr.value}")


def _convert_wave_tensor_type(
    type_: WaveTensorType,
    parse_ctx: _OpParseContext,
) -> type[Memory] | type[Register]:
    """Converts a WaveTensorType to Memory or Register, dispatching on address space."""
    dims: list[IndexExpr | int] = [
        index_symbol(symbol_attr_to_name(attr)) for attr in type_.shape
    ]
    dtype = mlir_element_type_to_dtype(type_.element_type)
    if type_.address_space.value == wave.WaveAddressSpace.Register:
        return Register[(*dims, dtype)]
    address_space = _address_space_to_symbol(type_.address_space, parse_ctx)
    return Memory[(*dims, address_space, dtype)]


_MMA_KIND_BY_VALUE: dict[int, MMAType | ScaledMMAType] = {
    m.value: m for m in (*MMAType, *ScaledMMAType)
}


def _convert_mma_kind(attr: WaveMmaKindAttr) -> MMAType | ScaledMMAType:
    """Converts a WaveMmaKindAttr into a Python MMAType or ScaledMMAType enum.

    MMA types are partitioned into two enums with non-overlapping integer ranges.
    """
    enum_value = attr.value.value
    if (mma_type := _MMA_KIND_BY_VALUE.get(enum_value)) is not None:
        return mma_type
    raise ValueError(
        f"Unknown MMA type value: 0x{enum_value:04x}. "
        f"Expected MMAType or ScaledMMAType"
    )


def _convert_read_write_bounds(
    attr: WaveSymbolMappingAttr,
) -> dict[IndexSymbol, IndexExpr]:
    """
    Converts Wave read/write bounds attribute into a dictionary mapping dimensions to expressions.

    Bounds specify the iteration space for memory operations (read/write) along each dimension.
    """
    bounds: dict[IndexSymbol, IndexExpr] = {}
    for key, value in attr:
        exprs = expr_list_attr_to_exprs(value)
        bounds[index_symbol(key.name)] = exprs[0]
    return bounds


def _convert_supported_attrs(
    op: ir.OpView, ignore_attrs: set[str] | None = None
) -> dict[str, AttrValue]:
    """Converts supported MLIR attributes of an operation into Python objects.

    Args:
        op: MLIR operation (OpView subclass)
        ignore_attrs: Set of attribute names to skip

    Returns:
        Dictionary mapping attribute names to converted Python values:
        - INDEX: dict[IndexSymbol, IndexSequence]
        - BOUNDS: dict[IndexSymbol, IndexExpr]
        - ELEMENTS_PER_THREAD: int
        - KIND: MMAType | ScaledMMAType
        - WATER_INTERNAL_ID: str
    """
    attrs = op.attributes
    ignore_attrs = ignore_attrs or set()
    # Only allow attributes we can faithfully convert.
    converters = {
        AttrNames.INDEX.mlir_name: lambda a: convert_index_mapping_array_to_sympy(
            op, a
        ),
        AttrNames.BOUNDS.mlir_name: _convert_read_write_bounds,
        AttrNames.ELEMENTS_PER_THREAD.mlir_name: lambda a: int(a.value),
        AttrNames.KIND.mlir_name: _convert_mma_kind,
        AttrNames.WATER_INTERNAL_ID.mlir_name: lambda a: a.value,
    }
    converted: dict[str, AttrValue] = {}
    for name in attrs:
        if name in ignore_attrs:
            continue
        if name not in converters:
            raise ValueError(f"Unsupported attribute '{name}' on op '{op.name}'")
        attr = attrs[name]
        converted[name] = converters[name](attr)
    return converted


def _apply_mlir_attrs_to_fx_node(
    fx_node: fx.Node,
    converted_attrs: dict[
        str,
        dict[IndexSymbol, IndexSequence]
        | dict[IndexSymbol, IndexExpr]
        | int
        | MMAType
        | ScaledMMAType
        | str,
    ],
) -> None:
    """Attaches converted MLIR attributes to an FX node using the AttrNames mapping."""
    for attr in AttrNames:
        if attr.mlir_name in converted_attrs:
            setattr(fx_node, attr.fx_property, converted_attrs[attr.mlir_name])


def _convert_constraints(op: ir.Operation) -> list[Constraint]:
    """Converts Wave constraints from an operation's wave.constraints attribute.

    Reconstructs WorkgroupConstraint, WaveConstraint, TilingConstraint,
    DeviceConstraint, and HardwareConstraint objects. Also restores the
    wave_id back-reference on WaveConstraints from the HardwareConstraint
    and the matching WorkgroupConstraint on the same dimension.

    Raises ValueError if the attribute is missing or contains unsupported
    constraint types.
    """
    constraints: list[Constraint] = []
    attrs = op.attributes
    if wave.WAVE_CONSTRAINTS_ATTR_NAME not in attrs:
        raise ValueError(f"{op.name} missing wave.constraints attribute")
    array_attr = attrs[wave.WAVE_CONSTRAINTS_ATTR_NAME]

    def _wg_dim_to_int(attr: WaveWorkgroupDimAttr) -> int:
        match attr.value:
            case wave.WaveWorkgroupDim.X:
                return 0
            case wave.WaveWorkgroupDim.Y:
                return 1
            case wave.WaveWorkgroupDim.Z:
                return 2
            case _:
                raise ValueError(f"Unsupported workgroup dim: {attr.value}")

    def _get_tile_size(attr) -> IndexExpr:
        """Extract the single tile-size expression from a constraint attribute."""
        exprs = expr_list_attr_to_exprs(attr)
        assert (
            len(exprs) == 1
        ), f"Expected exactly one tile-size expression, got {len(exprs)}"
        return exprs[0]

    hw: HardwareConstraint | None = None
    wg_by_dim: dict[IndexSymbol, WorkgroupConstraint] = {}

    for entry in array_attr:
        c = entry
        match c:
            case wave.WorkgroupConstraintAttr():
                dim = index_symbol(symbol_attr_to_name(c.dim))
                tile = _get_tile_size(c.tile_size)
                wg_dim = _wg_dim_to_int(c.workgroup_dim)
                wg = WorkgroupConstraint(dim, tile, wg_dim)
                wg_by_dim[dim] = wg
                constraints.append(wg)
            case wave.WaveConstraintAttr():
                dim = index_symbol(symbol_attr_to_name(c.dim))
                tile = _get_tile_size(c.tile_size)
                constraints.append(WaveConstraint(dim, tile))
            case wave.TilingConstraintAttr():
                dim = index_symbol(symbol_attr_to_name(c.dim))
                tile = _get_tile_size(c.tile_size)
                constraints.append(TilingConstraint(dim, tile))
            case wave.DeviceConstraintAttr():
                dim = index_symbol(symbol_attr_to_name(c.dim))
                tile = _get_tile_size(c.tile_size)
                device_dim = int(c.device_dim.value)
                constraints.append(DeviceConstraint(dim, tile, device_dim))
            case wave.HardwareConstraintAttr():
                threads_per_wave = int(c.threads_per_wave)
                waves_per_block_tuple = (
                    tuple(int(x) for x in c.waves_per_block)
                    if c.waves_per_block is not None
                    else None
                )

                vector_shapes = {
                    index_symbol(named.name): int(named.attr.value)
                    for named in (c.vector_shapes or [])
                }

                max_bits_per_load = int(c.max_bits_per_load)
                mma_type = (
                    _convert_mma_kind(c.mma_type) if c.mma_type is not None else None
                )

                hw = HardwareConstraint(
                    threads_per_wave=threads_per_wave,
                    waves_per_block=waves_per_block_tuple,
                    mma_type=mma_type,
                    vector_shapes=vector_shapes if vector_shapes else None,
                    max_bits_per_load=max_bits_per_load,
                )
                constraints.append(hw)
            case _:
                raise ValueError(f"Unsupported constraint attribute: {c}")

    # MLIR only stores dim and tile_size for WaveConstraints. Reconstruct the
    # wave_id back-reference from the HardwareConstraint and the matching
    # WorkgroupConstraint on the same dimension.
    if hw is not None:
        for con in constraints:
            if isinstance(con, WaveConstraint):
                wg = wg_by_dim.get(con.dim)
                if wg is not None:
                    con.set_wave_id_from_hardware_and_workgroup_constraint(hw, wg)

    return constraints


def _derive_vector_shapes_from_constraints(
    constraints: Sequence[Constraint], subs: dict[IndexSymbol, int] | None
) -> dict[IndexSymbol, int] | None:
    """Derive vector_shapes from constraint tile sizes for non-MMA kernels.

    This is the fallback used when `_derive_vector_shapes_from_graph` found no
    MMA nodes.  It resolves concrete tile sizes from WorkgroupConstraint and
    TilingConstraint.
    """
    vector_shapes: dict[IndexSymbol, int] = {}
    for con in constraints:
        if isinstance(con, (WorkgroupConstraint, TilingConstraint)):
            resolved = safe_subs(con.tile_size, subs or {})
            if isinstance(resolved, IndexExpr) and resolved.free_symbols:
                raise ValueError(f"Unresolved symbols in tile size: {resolved}")
            vector_shapes[con.dim] = int(resolved)
    return vector_shapes or None


def _derive_vector_shapes_from_graph(
    trace: CapturedTrace, hw: HardwareConstraint
) -> dict[IndexSymbol, int] | None:
    """Derive vector_shapes by inspecting MMA/ScaledMMA nodes in the built FX graph.

    Mirrors the core logic of `get_mma_dimensional_mapping` (mma_utils.py):
    for the first MMA-like node found, determines M, N, K from the operand
    tensor shapes and maps them to the intrinsic sizes via `mma_matrix_shapes`.

    Returns None when the graph contains no MMA nodes (the caller should
    fall back to `_derive_vector_shapes_from_constraints`).
    """
    for node in trace.walk(lambda n: n):
        custom = get_custom(node)
        if not isinstance(custom, MMABase):
            continue
        mma_type = getattr(custom, "mma_type", None) or hw.mma_type
        if mma_type is None:
            continue
        m, n = custom.acc_type.symbolic_shape[-2:]
        k = getattr(custom.fx_node, "reduction_dim", None)
        if k is None:
            # Derive reduction_dim the same way _handle_mma_op does.
            lhs_shape = custom.lhs_type.symbolic_shape
            rhs_shape = custom.rhs_type.symbolic_shape
            acc_shape = custom.acc_type.symbolic_shape
            candidates = (set(lhs_shape) & set(rhs_shape)) - set(acc_shape)
            if len(candidates) != 1:
                continue
            k = candidates.pop()
        m_size, n_size, k_size = hw.mma_matrix_shapes(mma_type)
        return {m: m_size, n: n_size, k: k_size}
    return None


def _initialize_vector_shapes(
    trace: CapturedTrace, hw: HardwareConstraint | None
) -> None:
    """
    Initializes vector_shapes on FX nodes from HardwareConstraint.

    MLIR does not carry per-node vector_shapes; they only exist on the
    HardwareConstraintAttr. In the normal pipeline (index_sequence_analysis),
    vector_shapes propagation is tightly coupled with index propagation: both
    are spread together via BFS from privileged source nodes (MMA, Read, Write).
    Reusing that machinery here is not possible without also recomputing indices,
    which would overwrite the indices from MLIR.

    Since the propagation ultimately assigns the full hw.vector_shapes to every
    reachable node anyway, we replicate the end result directly by assigning
    hw.vector_shapes to all applicable nodes.

    Skips Placeholder (except IterArg), and Output, matching the
    skip logic in verify_nodes / propagate_indices in index_sequence_analysis.
    """
    if hw is None or not hw.vector_shapes:
        return
    # IterArg is a Placeholder subclass but is NOT skipped in the wave pipeline.
    # Allocate is included here (unlike verify_nodes which skips it) so that the
    # roundtrip faithfully reproduces vector_shapes on all nodes that carry them.
    skip = (Placeholder, Output)
    for node in trace.walk(lambda n: n):
        custom = get_custom(node)
        if isinstance(custom, skip) and not isinstance(custom, IterArg):
            continue
        if not getattr(custom, "vector_shapes", None):
            custom.vector_shapes = dict(hw.vector_shapes)


def _initialize_tiling_constraints(
    trace: CapturedTrace, constraints: Sequence[Constraint]
) -> None:
    """
    Initialize induction_var on TilingConstraints from Iterate operations.

    For each Iterate node with axis A, sets induction_var = $ARGA on all
    TilingConstraints with dim == A. Mirrors LaunchableWave.initialize_tiling_constraints.
    """
    tiling_by_dim = {c.dim: c for c in constraints if isinstance(c, TilingConstraint)}
    if not tiling_by_dim:
        return
    for node in trace.walk(lambda n: isinstance(get_custom(n), Iterate)):
        axis = get_custom(node).axis
        if axis in tiling_by_dim:
            tiling_by_dim[axis].induction_var = get_induction_symbol(axis)


@dataclass
class _OpParseContext:
    """
    Context for parsing MLIR operations into an FX graph.

    Wave.iterate regions are made isolated from above before parsing (via makeIsolated).
    This means all values from outer scopes are explicitly passed as captures through
    operands and become block arguments. Each scope is self-contained - operations only
    reference block arguments (including init args and captures) or locally-defined operations.

    Attributes:
        graph: The FX graph being constructed for the current scope.
        region_graph: Container for managing multiple subgraphs.
        default_mma_type: Default MMA type from hardware constraints.
        value_map: Maps MLIR ir.Value to FX nodes/scalars in the current scope.
        unspecified_addr_counter: Shared counter for generating unique symbols
            for Unspecified address spaces (so distinct unresolved spaces are
            never conflated).
    """

    graph: fx.Graph
    region_graph: RegionGraph
    default_mma_type: MMAType | ScaledMMAType | None = None
    value_map: dict[ir.Value, fx.Node | int | float] = field(default_factory=dict)
    unspecified_addr_counter: itertools.count = field(default_factory=itertools.count)

    def resolve_operand(self, value: ir.Value) -> fx.Node | int | float:
        """Resolve MLIR value to its corresponding FX node or scalar."""
        if value not in self.value_map:
            raise ValueError(f"Missing operand mapping for value {value}")
        return self.value_map[value]

    def add_mapping(self, mlir_value: ir.Value, fx_node: fx.Node | int | float) -> None:
        """Add MLIR value → FX node mapping."""
        assert (
            mlir_value not in self.value_map
        ), f"Duplicate mapping for MLIR value: {mlir_value}"
        self.value_map[mlir_value] = fx_node


def _handle_constant_op(op: arith.ConstantOp, parse_ctx: _OpParseContext) -> None:
    """Handle arith.constant operation."""
    value = op.attributes[AttrNames.VALUE.mlir_name].value
    parse_ctx.add_mapping(op.results[0], value)


def _handle_register_op(op: RegisterOp, parse_ctx: _OpParseContext) -> None:
    """Handle wave.register operation."""
    init_value = parse_ctx.resolve_operand(op.init)
    result_type = op.result.type
    dims = tuple(index_symbol(symbol_attr_to_name(attr)) for attr in result_type.shape)
    dtype = mlir_element_type_to_dtype(result_type.element_type)
    converted_attrs = _convert_supported_attrs(op)

    register_op = NewRegister.create(
        parse_ctx.graph,
        shape=dims,
        dtype=dtype,
        value=init_value,
        type=Register[(*dims, dtype)],
    )
    _apply_mlir_attrs_to_fx_node(register_op.fx_node, converted_attrs)
    parse_ctx.add_mapping(op.result, register_op.fx_node)


def _handle_lds_barrier_op(op: amdgpu.LDSBarrierOp, parse_ctx: _OpParseContext) -> None:
    """Handle amdgpu.lds_barrier operation."""
    # AMD GPU lds_barrier maps to SharedMemoryBarrier in FX
    # wait_async_ops and tensor_wait default to False
    barrier_op = SharedMemoryBarrier.create(
        parse_ctx.graph,
        wait_async_ops=False,
        tensor_wait=False,
    )
    # Barriers don't produce values, so no value_map entry needed


def _handle_allocate_op(op: AllocateOp, parse_ctx: _OpParseContext) -> None:
    """Handle wave.allocate operation."""
    result_type = op.result.type
    shape = tuple(index_symbol(symbol_attr_to_name(attr)) for attr in result_type.shape)
    dtype = mlir_element_type_to_dtype(result_type.element_type)

    if result_type.address_space.value == wave.WaveAddressSpace.Register:
        raise ValueError("Allocate cannot target Register address space")
    address_space = _address_space_to_symbol(result_type.address_space, parse_ctx)

    # MLIR stores the *unpadded* distributed shape, padding is a separate
    # attribute. Reconstruct the padded distributed_shape that the Python
    # Allocate operation expects, i.e. padding added to the last dimension.
    unpadded_dist = list(expr_list_attr_to_exprs(op.distributed_shape))
    padding = int(op.padding) if op.padding is not None else 0
    tail_padding = int(op.tail_padding) if op.tail_padding is not None else 0
    if padding and unpadded_dist:
        unpadded_dist[-1] = unpadded_dist[-1] + padding
    distributed_shape = tuple(unpadded_dist)

    parent_node = (
        parse_ctx.resolve_operand(op.parent) if op.parent is not None else None
    )
    offset = int(op.offset) if op.offset is not None else None

    # distributed_shape, offset, padding, and tail_padding are handled above
    # as explicit constructor arguments - ignore them so
    # _convert_supported_attrs picks up only the remaining attrs.
    converted_attrs = _convert_supported_attrs(
        op,
        ignore_attrs={"distributed_shape", "offset", "padding", "tail_padding"},
    )

    allocate_op = Allocate.create(
        parse_ctx.graph,
        shape=shape,
        distributed_shape=distributed_shape,
        dtype=dtype,
        address_space=address_space,
        padding=padding,
        parent=parent_node,
        offset=offset,
        tail_padding=tail_padding,
    )
    _apply_mlir_attrs_to_fx_node(allocate_op.fx_node, converted_attrs)
    parse_ctx.add_mapping(op.result, allocate_op.fx_node)


def _handle_read_op(op: ReadOp, parse_ctx: _OpParseContext) -> None:
    """Handle wave.read operation."""
    memory_node = parse_ctx.resolve_operand(op.memory)
    converted_attrs = _convert_supported_attrs(
        op,
        ignore_attrs={AttrNames.ORDERED_SYMS.mlir_name},
    )

    read_op = Read.create(
        parse_ctx.graph,
        memory=memory_node,
        elements_per_thread=converted_attrs.pop(
            AttrNames.ELEMENTS_PER_THREAD.mlir_name, None
        ),
        bounds=converted_attrs.pop(AttrNames.BOUNDS.mlir_name, None),
        type=_convert_wave_tensor_type(op.result.type, parse_ctx),
    )
    _apply_mlir_attrs_to_fx_node(read_op.fx_node, converted_attrs)
    parse_ctx.add_mapping(op.result, read_op.fx_node)


def _handle_write_op(op: WriteOp, parse_ctx: _OpParseContext) -> None:
    """Handle wave.write operation. Produces no results."""
    reg_node = parse_ctx.resolve_operand(op.value_to_store)
    mem_node = parse_ctx.resolve_operand(op.memory)
    converted_attrs = _convert_supported_attrs(
        op,
        ignore_attrs={AttrNames.ORDERED_SYMS.mlir_name},
    )

    write_op = Write.create(
        parse_ctx.graph,
        register_=reg_node,
        memory=mem_node,
        elements_per_thread=converted_attrs.pop(
            AttrNames.ELEMENTS_PER_THREAD.mlir_name, None
        ),
        bounds=converted_attrs.pop(AttrNames.BOUNDS.mlir_name, None),
        type=get_custom(mem_node).type,
    )
    _apply_mlir_attrs_to_fx_node(write_op.fx_node, converted_attrs)
    # No mapping added because write produces no SSA results.


def _handle_mma_op(op: MmaOp, parse_ctx: _OpParseContext) -> None:
    """Handle wave.mma operation."""
    lhs_node = parse_ctx.resolve_operand(op.lhs)
    rhs_node = parse_ctx.resolve_operand(op.rhs)
    acc_node = parse_ctx.resolve_operand(op.accumulator)
    converted_attrs = _convert_supported_attrs(op)
    mma_type = (
        converted_attrs.pop(AttrNames.KIND.mlir_name, None)
        or parse_ctx.default_mma_type
    )

    mma_op = MMA.create(
        parse_ctx.graph,
        lhs=lhs_node,
        rhs=rhs_node,
        acc=acc_node,
        mma_type=mma_type,
        type=_convert_wave_tensor_type(op.result.type, parse_ctx),
    )

    _apply_mlir_attrs_to_fx_node(mma_op.fx_node, converted_attrs)

    # Derive reduction_dim: the dimension shared by lhs and rhs but absent from acc.
    lhs_shape = get_custom(lhs_node).type.symbolic_shape
    rhs_shape = get_custom(rhs_node).type.symbolic_shape
    acc_shape = get_custom(acc_node).type.symbolic_shape
    reduction_dim = (set(lhs_shape) & set(rhs_shape)) - set(acc_shape)
    assert (
        len(reduction_dim) == 1
    ), f"Expected exactly one reduction dimension for MMA, got {reduction_dim}"
    mma_op.fx_node.reduction_dim = reduction_dim.pop()

    parse_ctx.add_mapping(op.result, mma_op.fx_node)


def _handle_extract_slice_op(op: ExtractSliceOp, parse_ctx: _OpParseContext) -> None:
    """Handle wave.extract_slice operation."""
    src_node = parse_ctx.resolve_operand(op.memory)
    offset = tuple(expr_list_attr_to_exprs(op.offset))
    size = tuple(expr_list_attr_to_exprs(op.size))
    stride = tuple(expr_list_attr_to_exprs(op.stride))
    converted_attrs = _convert_supported_attrs(
        op,
        ignore_attrs={
            AttrNames.OFFSET.mlir_name,
            AttrNames.SIZE.mlir_name,
            AttrNames.STRIDE.mlir_name,
        },
    )

    slice_op = ExtractSlice.create(
        parse_ctx.graph,
        register_=src_node,
        offset=offset,
        size=size,
        stride=stride,
        type=_convert_wave_tensor_type(op.result.type, parse_ctx),
    )
    _apply_mlir_attrs_to_fx_node(slice_op.fx_node, converted_attrs)
    parse_ctx.add_mapping(op.result, slice_op.fx_node)


def _handle_yield_op(op: YieldOp, parse_ctx: _OpParseContext) -> None:
    """Handle wave.yield operation. Creates an FX output node for the subgraph."""
    operands = list(op.operands)
    if not operands:
        parse_ctx.graph.output(None)
        return

    yield_values = [parse_ctx.resolve_operand(v) for v in operands]
    parse_ctx.graph.output(yield_values)


def _create_get_result_nodes(
    parse_ctx: _OpParseContext,
    iterate_node: fx.Node,
    results: Sequence[ir.Value],
    result_types: Sequence[type],
    iter_index: (
        dict[IndexSymbol, IndexSequence]
        | Sequence[dict[IndexSymbol, IndexSequence]]
        | None
    ),
    output_values: Sequence[fx.Node],
    axis: IndexSymbol,
) -> None:
    """Creates GetResult nodes for each result of a wave.iterate operation.

    Each iterate result gets a GetResult node that extracts its value from the
    iterate's output tuple.  The index is used as-is: `Iterate.index`
    delegates to `MMA.acc_index` which already specialises MMA_ACC and
    drops the reduction dimension before the emitters serialise it.

    # TODO: revisit when converting partially-compiled traces where
    # index analysis has not yet run.
    """
    with parse_ctx.graph.inserting_after(iterate_node):
        for idx, result in enumerate(results):
            get_result_op = GetResult.create(
                parse_ctx.graph,
                value=iterate_node,
                res_idx=idx,
                type=result_types[idx],
            )

            # Prefer the explicit iterate index from MLIR (present after index
            # propagation). Fall back to the output value's index for early
            # pipeline stages where index_sequence_analysis hasn't run yet.
            result_index = None
            if iter_index is not None:
                result_index = (
                    iter_index[idx] if isinstance(iter_index, Sequence) else iter_index
                )
            elif isinstance(output_values[idx], fx.Node):
                result_index = getattr(
                    output_values[idx], AttrNames.INDEX.fx_property, None
                )

            if isinstance(result_index, dict):
                get_result_op.fx_node.index = result_index

            parse_ctx.add_mapping(result, get_result_op.fx_node)


def _handle_iterate_op(op: IterateOp, parse_ctx: _OpParseContext) -> None:
    """
    Handle wave.iterate operation.

    First calls makeIsolated() to convert any implicit captures to explicit captures
    (operands and block arguments). Then creates a nested subgraph for the iterate body:
    - Iterator axis from the iterator attribute
    - Init args become IterArg placeholders in the subgraph
    - Captures (explicit after makeIsolated) are mapped directly to outer values
    - GetResult nodes for each iterate result
    """
    axis = index_symbol(symbol_attr_to_name(op.iterator))
    init_args = [parse_ctx.resolve_operand(v) for v in op.iter_args]
    captures = [parse_ctx.resolve_operand(v) for v in op.captures]
    results = list(op.results_)

    subgraph = fx.Graph()
    subgraph_name = parse_ctx.region_graph.add_subgraph(
        "region", subgraph, inner_freevars=[]
    )
    subgraph.parent_op = None

    # Always call makeIsolated() to standardize the MLIR format. This converts any
    # implicit captures to explicit operands and block arguments, making parsing
    # predictable and eliminating the need to track outer values.
    iterate_make_isolated(op.operation)

    # After makeIsolated, refresh the body and capture list.
    body = op.body.blocks[0]
    block_args = list(body.arguments)
    captures = [parse_ctx.resolve_operand(v) for v in op.captures]
    iter_count = len(init_args)

    # Create a local scope for the iterate body.
    # - IterArg block arguments → new IterArg placeholder nodes in subgraph
    # - Capture block arguments → mapped directly to outer values (no placeholders)
    local_map: dict[ir.Value, fx.Node | int | float] = {}

    # Map iter args to new placeholder nodes in the subgraph
    for idx, (block_arg, init_node) in enumerate(
        zip(block_args[:iter_count], init_args)
    ):
        arg_node = subgraph.placeholder(f"iter_arg{idx}")
        arg_node.type = init_node.type
        arg_node.tkw_op = IterArg
        arg_node.tkw_op_name = "iter_arg"
        arg_node.iter_idx = idx
        # Copy index and vector_shapes from init node if present
        if hasattr(init_node, AttrNames.INDEX.fx_property):
            arg_node.index = init_node.index
        if hasattr(init_node, "vector_shapes") and init_node.vector_shapes:
            arg_node.vector_shapes = dict(init_node.vector_shapes)
        local_map[block_arg] = arg_node

    # Map capture block arguments directly to their outer values rather than
    # creating lifted placeholders (the graph comparison handles both forms).
    for block_arg, capture_node in zip(block_args[iter_count:], captures):
        local_map[block_arg] = capture_node

    # Parse the body operations. All values now resolve within local_map.
    _convert_ops(
        list(body.operations),
        _OpParseContext(
            graph=subgraph,
            region_graph=parse_ctx.region_graph,
            default_mma_type=parse_ctx.default_mma_type,
            value_map=local_map,
            unspecified_addr_counter=parse_ctx.unspecified_addr_counter,
        ),
    )

    output_node = next(node for node in subgraph.nodes if node.op == "output")
    output_custom = get_custom(output_node)
    output_values = output_custom.return_vals
    if not isinstance(output_values, Sequence):
        output_values = [output_values]

    iterate_op = Iterate.create(
        parse_ctx.graph,
        axis=axis,
        init_args=init_args,
        subgraph_name=subgraph_name,
        implicit_captures=captures,
        step=1,
    )
    subgraph.parent_op = iterate_op.fx_node

    # Infer index for the iterate node
    iter_index = None
    if AttrNames.INDEX.mlir_name in op.attributes:
        iter_index = convert_index_mapping_array_to_sympy(
            op, op.attributes[AttrNames.INDEX.mlir_name]
        )
    else:
        # Try to infer from output values
        output_indices = [
            getattr(val, AttrNames.INDEX.fx_property, None)
            for val in output_values
            if isinstance(val, fx.Node)
        ]
        if len(output_indices) == 1 and output_indices[0] is not None:
            iter_index = output_indices[0]
        elif all(idx is not None for idx in output_indices) and output_indices:
            iter_index = output_indices

    result_types = [
        _convert_wave_tensor_type(result.type, parse_ctx) for result in results
    ]
    if len(result_types) == 1:
        iterate_op.fx_node.type = result_types[0]
    else:
        iterate_op.fx_node.type = result_types

    # Create GetResult nodes for each iterate result
    _create_get_result_nodes(
        parse_ctx,
        iterate_op.fx_node,
        results,
        result_types,
        iter_index,
        output_values,
        axis,
    )


def _convert_ops(ops: Sequence[ir.Operation], parse_ctx: _OpParseContext) -> None:
    """Dispatches MLIR operations to type-specific handlers, building FX graph nodes.

    Each handler updates the parse context's value_map so that later operations can
    resolve their operands. Called once for the top-level function body and recursively
    for nested regions (e.g. iterate bodies) with a fresh parse_ctx sharing the same
    counters.
    """

    for op in ops:
        match op:
            case arith.ConstantOp():
                _handle_constant_op(op, parse_ctx)
            case RegisterOp():
                _handle_register_op(op, parse_ctx)
            case AllocateOp():
                _handle_allocate_op(op, parse_ctx)
            case ReadOp():
                _handle_read_op(op, parse_ctx)
            case WriteOp():
                _handle_write_op(op, parse_ctx)
            case MmaOp():
                _handle_mma_op(op, parse_ctx)
            case ExtractSliceOp():
                _handle_extract_slice_op(op, parse_ctx)
            case IterateOp():
                _handle_iterate_op(op, parse_ctx)
            case YieldOp():
                _handle_yield_op(op, parse_ctx)
            case func.ReturnOp():
                pass  # No FX node needed - produces no results.
            case amdgpu.LDSBarrierOp():
                _handle_lds_barrier_op(op, parse_ctx)
            case _:
                raise ValueError(f"Unsupported op in MLIR-to-FX conversion: {op.name}")


def convert_mlir_to_trace(
    mlir_module_text: str,
) -> tuple[
    CapturedTrace | None, list[Constraint], WaveCompileOptions, list[MLIRDiagnostic]
]:
    """Convert MLIR (Wave dialect) text into a CapturedTrace (FX graph).

    This is the main conversion entry point, called from the __main__ subprocess
    harness below. Parses and verifies the module, extracts constraints and
    hyperparameters from func.func attributes, then converts each operation into
    the corresponding FX node.
    """
    diagnostics: list[MLIRDiagnostic] = []
    with ir.Context() as ctx:
        wave.register_dialect(ctx)

        def _diagnostics_handler(d: ir.Diagnostic) -> bool:
            diagnostics.append(
                MLIRDiagnostic(
                    message=d.message,
                    severity=d.severity.name,
                    location=serialize_location(d.location),
                )
            )
            return True

        ctx.attach_diagnostic_handler(_diagnostics_handler)

        try:
            module = ir.Module.parse(mlir_module_text, context=ctx)
        except ir.MLIRError:
            return None, [], WaveCompileOptions(), diagnostics

        # Verify the MLIR module.
        # We rely on verified MLIR throughout the converter.
        if not module.operation.verify():
            return None, [], WaveCompileOptions(), diagnostics

        # Find the first func.func operation in the module
        func_op = None
        for op in module.body.operations:
            if op.operation.name == "func.func":
                func_op = op.operation
                break
        if func_op is None:
            raise ValueError("No func.func found in module")

        # Convert hyperparameters into options.
        options = WaveCompileOptions()
        if (hyper := func_op.attributes.get("wave.hyperparameters")) is not None:
            subs: dict = {}
            for named in hyper.mapping:
                subs[index_symbol(named.name)] = named.attr.value
            options.subs = subs
        constraints = _convert_constraints(func_op)

        # Get the hardware constraint
        hw = next((c for c in constraints if isinstance(c, HardwareConstraint)), None)
        default_mma_type = hw.mma_type if hw and hw.mma_type is not None else None

        entry_block = func_op.regions[0].blocks[0]
        region_graph: RegionGraph = KernelRegionGraph()
        root_graph = fx.Graph()

        parse_ctx = _OpParseContext(
            graph=root_graph,
            region_graph=region_graph,
            default_mma_type=default_mma_type,
        )

        # Create placeholders for function arguments
        for idx, arg in enumerate(entry_block.arguments):
            node = root_graph.placeholder(f"arg{idx}")
            node.type = _convert_wave_tensor_type(arg.type, parse_ctx)
            parse_ctx.value_map[arg] = node

        _convert_ops(entry_block.operations, parse_ctx)

        # Add an output node to the root graph. This is required for the FX graph
        root_graph.output(None)
        root_name = region_graph.add_subgraph("region", root_graph, inner_freevars=[])

        trace = CapturedTrace(
            region_graph=region_graph, root_graph=root_name, location=None
        )

        # Derive hw.vector_shapes from the built graph. When the graph
        # contains MMA nodes we read M/N/K directly from their tensor shapes,
        # mirroring get_mma_dimensional_mapping (mma_utils.py). For non-MMA
        # kernels we fall back to constraint-based derivation.
        if hw is not None and not hw.vector_shapes:
            hw.vector_shapes = _derive_vector_shapes_from_graph(trace, hw)
            if not hw.vector_shapes:
                derived = _derive_vector_shapes_from_constraints(
                    constraints, options.subs
                )
                if derived:
                    hw.vector_shapes = derived

        _initialize_vector_shapes(trace, hw)
        _initialize_tiling_constraints(trace, constraints)
        return trace, constraints, options, diagnostics


if __name__ == "__main__":
    try:
        request = dill.loads(sys.stdin.buffer.read())
    except Exception as e:
        sys.stderr.write(f"FATAL: failed to unpickle input: {e}\n")
        sys.exit(1)

    mlir_text = request.get("mlir") if isinstance(request, dict) else None
    if not isinstance(mlir_text, str):
        sys.stderr.write(
            f"FATAL: expected 'mlir' string in request, got: {type(mlir_text)}\n"
        )
        sys.exit(1)

    response = FxEmitterResponse()
    try:
        trace, constraints, options, diagnostics = convert_mlir_to_trace(mlir_text)
        if trace is None:
            raise ValueError("MLIR conversion/verification failed; see diagnostics")
        trace.snapshot_node_state()
        response = FxEmitterResponse(
            trace=trace,
            constraints=constraints,
            options=options,
            diagnostics=diagnostics,
        )
        sys.stdout.buffer.write(dill.dumps(response))
        sys.stdout.flush()
        sys.exit(0)
    except Exception as e:
        response.diagnostics.append(MLIRDiagnostic(message=str(e), severity="ERROR"))
        try:
            sys.stdout.buffer.write(dill.dumps(response))
            sys.stdout.flush()
        finally:
            sys.stderr.write(f"FATAL: fx_emitter failed: {e}\n")
            sys.exit(1)
