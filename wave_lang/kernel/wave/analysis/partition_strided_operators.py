# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from itertools import groupby
from operator import itemgetter
from typing import Callable, NamedTuple

import numpy as np
import sympy
import torch.fx as fx

from wave_lang.support.logging import get_logger

from ..._support.indexing import IndexSequence, IndexSymbol
from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import *
from ...lang.wave_types import IndexMapping
from ..index_mapping_simplify import simplify_index_mapping, get_tile_sizes_from_index
from ...ops.wave_ops import (
    CustomOp,
    ExtractSlice,
    Read,
    Reshape,
    SelfIndex,
    Write,
    get_custom,
)
from ..region_canonicalization import RegionFormat, requires_region_format
from ..assumptions import get_divisibility_subs
from ..constraints import Constraint
from ..utils.mapping_utils import transform_index_on_mapping
from ..utils.tag_utils import propagate_tag
from ..utils.general_utils import (
    all_equal,
    get_fastest_index,
    get_hardware_constraint,
    get_largest_index_and_size,
    infer_dim,
)
from ..utils.mma_utils import (
    simplify_index,
)
from ..utils.symbol_utils import (
    eval_expr,
    expr_is_zero_under_probes,
    get_start_expr,
    safe_subs,
    simplify as sym_simplify,
    subs_idxc,
)

logger = get_logger("wave.partition_strided_operators")


class ReadInfo(NamedTuple):
    flat_offset: sympy.Expr
    phys_start: dict
    custom: Read
    node: fx.Node


class MultiwayGroupEntry(NamedTuple):
    idx: int
    node: fx.Node
    custom: Read
    byte_pos: int
    phys_start: dict


class ReadTransformInfo(NamedTuple):
    transformed_index: dict | None
    can_use_transformed_mask: bool


def get_vector_shape(
    vector_shapes: dict[IndexSymbol, int],
    symbolic_shape: list[IndexSymbol],
) -> list[int]:
    # Normalize scaled dimensions (like K/2) to base dimensions (like K) for lookup
    result = [max(vector_shapes.get(infer_dim(dim), 1), 1) for dim in symbolic_shape]
    return result


def _get_symbolic_shape_and_vector_shapes(
    custom: CustomOp,
):
    register_shape = custom.register_type.symbolic_shape
    vector_shapes = custom.vector_shapes
    memory_shape = custom.memory_type.symbolic_shape
    # Check to see if the memory shape does not match with the vector shapes.
    if not set(memory_shape).issubset(set(vector_shapes.keys())):
        return register_shape, vector_shapes
    # Pick the shape with the most dimensions.
    if len(memory_shape) > len(register_shape):
        return memory_shape, vector_shapes
    return register_shape, vector_shapes


@requires_region_format(RegionFormat.DIRECT_OUTER_REF)
def partition_strided_operators(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function analyzes the index sequence of operators in the graph
    that are writes on 2d tensors. If the operator has an access pattern where
    the strides are greater than one on a single dimension, this function splits the
    operands into individual elements and constructs a write for
    each individual element.
    """

    def has_strided_access(node: fx.Node) -> bool:
        """
        Checks for writes on 2d tensors with strided access on a single dimension that
        read more than a single element.
        """
        custom = get_custom(node)
        if isinstance(custom, Write):
            if custom.elements_per_thread == 1:
                return False

            # `custom.register_index` calls are expensive, try to minimize the number
            # of calls.
            strides_and_sizes = [
                (
                    val.stride,
                    val.size,
                )
                for val in map(simplify_index, custom.register_index.values())
            ]
            strides = [x for x, y in strides_and_sizes if y > 1]
            num_strided_accesses = sum(1 for stride in strides if stride > 1)
            if num_strided_accesses > 1:
                raise NotImplementedError(
                    "Support for strided accesses on more than one dimension not implemented yet!"
                )
            return num_strided_accesses == 1
        return False

    strided_operators = trace.walk(has_strided_access)
    for operator in strided_operators:
        custom = get_custom(operator)
        register_index = custom.register_index
        simplified_index = {
            dim: simplify_index(register_index.get(dim, idx))
            for dim, idx in custom.index.items()
        }

        symbolic_shape, vector_shapes = _get_symbolic_shape_and_vector_shapes(custom)

        shape = get_vector_shape(vector_shapes, symbolic_shape)
        elements_per_thread = subs_idxc(custom.elements_per_thread)
        max_stride_dim, max_stride = max(
            [(dim, seq.stride) for dim, seq in simplified_index.items()],
            key=lambda item: item[1],
        )

        # Compute offsets we will aplly to each index element for each partitioned
        # write.
        offsets = np.array(
            [
                np.unravel_index(int(i * max_stride), shape)
                for i in range(elements_per_thread)
            ]
        )

        def check_contiguous_index():
            """
            Check if resulted partitioned write is equivalent to contiguous write.

            Write is contiguous if offsets has form `[0,1,2...N]` for
            the fastest mem dim and 0s for all others dims.
            """
            fastest_mem_dim = custom.memory.type.symbolic_shape[-1]

            # Find fastest mem dim index in the register symbolic shape.
            # If we have write with mapping, order of dims between register and
            # mem may differ or mem dims may not even present in reg index.
            if fastest_mem_dim not in symbolic_shape:
                return False

            fastest_mem_dim_idx = symbolic_shape.index(fastest_mem_dim)

            # Construct expected offsets in the form:
            # [[0 0 0]
            #  [0 1 0]
            #  [0 2 0]
            #  [0 3 0]]
            expected_offsets = np.array(
                [
                    [
                        (j if i == fastest_mem_dim_idx else 0)
                        for j in range(elements_per_thread)
                    ]
                    for i in range(len(shape))
                ]
            ).T
            if not np.array_equal(offsets, expected_offsets):
                return False

            mapping = custom.mapping
            # For writes without mapping this is enough.
            if mapping is None:
                return True

            # If we have mapping, check that fastest dim mapping is trivial, i.e.:
            # IndexMapping(inputs={X: i, ...}, outputs={X: i, ...})
            return (
                mapping.input_mapping.get(fastest_mem_dim, None)
                == mapping.output_mapping[fastest_mem_dim]
            )

        if check_contiguous_index():
            continue

        ops_to_combine = []
        with custom.graph.inserting_before(operator):
            for i in range(elements_per_thread):
                # Non-contiguous access patterns can have varying offsets. We
                # handle that here.
                extract = ExtractSlice(custom.register_, [i], [1], [1]).add_to_graph(
                    custom.graph, loc=custom.location
                )

                offset = offsets[i]
                write = Write(
                    extract,
                    custom.memory,
                    mapping=custom.mapping,
                    elements_per_thread=1,
                    flags=custom.flags,
                ).add_to_graph(custom.graph, loc=custom.location)
                write.index = {
                    dim: IndexSequence(
                        simplified_index[dim].start.subs({GPR_NUM: 0}) + offset[j], 1, 1
                    )
                    for j, dim in enumerate(symbolic_shape)
                }
                extract.index = write.index
                write.vector_shapes = vector_shapes
                ops_to_combine.append(write)

        # Useful to handle write/read dependency
        custom.replace_all_uses_with(ops_to_combine)
        custom.graph.erase_node(operator)


@requires_region_format(RegionFormat.DIRECT_OUTER_REF)
def partition_ops_with_gpr_offsets(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function analyzes the index sequence of reads and writes in a graph.
    If the reads or writes have incontiguous offsets based on GPR_NUM, we'd
    need to split these reads/writes appropriately.

    e.g a vector<16xf16> may be owned by lane 0, and lane 16 in this layout:
    [0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 0, 16, 16, 16, 16].

    With our current glossary, this means we have 2 VGPR "chunks".
    [0:4) and [8:12) for lane0, and [4:8) and [12:16) for lane16.
    To the lane it should just look like vector<8xf16>.
    Hence for this example, we'd need two reads of vector<4xf16> and a couple
    insert_slices to combine them together into a single vector<8xf16>.

    """

    def has_gpr_offsets(node: fx.Node) -> bool:
        """
        Checks for writes on 2d tensors with strided access on a single dimension that
        read more than a single element.
        """
        custom = get_custom(node)
        if not isinstance(custom, (Read, Write, SelfIndex)):
            return False
        num_dims_with_gpr = sum(
            1 for v in custom.index.values() if sympy.sympify(v.start).has(GPR_NUM)
        )
        if num_dims_with_gpr == 1:
            return True
        elif num_dims_with_gpr == 0:
            return False
        raise NotImplementedError("Currently only handles 1 dim with GPR offset.")

    strided_operators = trace.walk(has_gpr_offsets)
    for operator in strided_operators:
        custom = get_custom(operator)
        index = custom.index
        simplified_index = {
            dim: simplify_index(index.get(dim, idx)) for dim, idx in index.items()
        }
        if isinstance(custom, SelfIndex):
            # If specified use element_per_thread instead of IndexExpr size.
            elements_per_thread = subs_idxc(
                custom.elements_per_thread or index[custom.dim].size
            )
        else:
            elements_per_thread = subs_idxc(custom.elements_per_thread)
        dim_with_gpr_offsets = [
            (k, v.start) for k, v in simplified_index.items() if v.start.has(GPR_NUM)
        ]
        assert len(dim_with_gpr_offsets) == 1, "Expected only 1-Dim has gpr offsets"
        gpr_offset_dim, gpr_offset_expr = dim_with_gpr_offsets[0]
        gpr_offsets = [
            gpr_offset_expr.subs({GPR_NUM: i}) for i in range(elements_per_thread)
        ]

        # Cluster contiguous indices of reads/writes
        # e.g given indices  [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
        # Will cluster to:  [[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19], [24, 25, 26, 27]]
        gpr_relative_offsets = [x - gpr_offsets[0] for x in gpr_offsets]
        gpr_chunks = [
            list(map(itemgetter(1), g))
            for _, g in groupby(enumerate(gpr_relative_offsets), lambda x: x[0] - x[1])
        ]

        # Compute number of GPR chunks.
        num_gpr_chunks = len(gpr_chunks)

        # Compute size of each chunk and ensure they are the same size.
        gpr_sizes = [len(gpr_chunk) for gpr_chunk in gpr_chunks]
        assert all_equal(
            gpr_sizes
        ), "Only support strided GPR offset with uniform sizes."
        gpr_size = gpr_sizes[0]

        # Break apart Reads/Writes that has non-contiguous GPR Read/Writes.
        with custom.graph.inserting_before(operator):
            ops_to_combine = []
            for chunk_id in range(num_gpr_chunks):
                cur_gpr_start_id = chunk_id * gpr_size
                # Get updated index with VGPR offset.
                if hasattr(custom, "mapping") and custom.mapping is not None:
                    output_mapping = list(custom.mapping.output_mapping.keys())
                else:
                    output_mapping = list(index)
                # Modify stride to 1 S.T we can have vectorized read/write
                # iff gpr_offset_dim is or will be (after mapping) fastest dim.
                updated_index_with_gpr_offset = deepcopy(simplified_index)
                updated_dim_with_gpr_offset = IndexSequence(
                    updated_index_with_gpr_offset[gpr_offset_dim].start.subs(
                        {GPR_NUM: cur_gpr_start_id}
                    ),
                    gpr_size,
                    (
                        1
                        if output_mapping[-1] == gpr_offset_dim
                        else simplified_index[gpr_offset_dim].stride
                    ),
                )
                updated_index_with_gpr_offset[gpr_offset_dim] = (
                    updated_dim_with_gpr_offset
                )

                if hasattr(custom, "mapping_dynamic_vals"):
                    # If we are partitioning read/write ops, dynamic_vals can be
                    # potentially partitioned as well. Partitioned dyn vals are
                    # are merged into single value using Reshape op which still
                    # holds the original index containing `GPR_NUM`.
                    # Extract corresponding partitioned chunk from such ops.
                    new_dynamic_vals = []
                    for dyn_val in custom.mapping_dynamic_vals:
                        if any(
                            sympy.sympify(v.start).has(GPR_NUM)
                            for v in get_custom(dyn_val).index.values()
                        ):
                            extract = ExtractSlice(
                                dyn_val, [cur_gpr_start_id], [gpr_size], [1]
                            ).add_to_graph(custom.graph, loc=custom.location)
                            extract.index = updated_index_with_gpr_offset
                            new_dynamic_vals.append(extract)
                        else:
                            new_dynamic_vals.append(dyn_val)

                # Generate new Read/Write that has contiguous VGPR elements.
                if isinstance(custom, Write):
                    extract = ExtractSlice(
                        custom.register_, [cur_gpr_start_id], [gpr_size], [1]
                    ).add_to_graph(custom.graph, loc=custom.location)
                    extract.index = updated_index_with_gpr_offset
                    new_node = Write(
                        extract,
                        custom.memory,
                        mapping=custom.mapping,
                        mapping_dynamic_vals=new_dynamic_vals,
                        elements_per_thread=gpr_size,
                        flags=custom.flags,
                    ).add_to_graph(custom.graph, loc=custom.location)
                elif isinstance(custom, Read):
                    # TODO: Add support on how to handle strided reads.
                    new_node = Read(
                        custom.memory,
                        elements_per_thread=gpr_size,
                        mapping=custom.mapping,
                        mapping_dynamic_vals=new_dynamic_vals,
                        _write_dependency=custom._write_dependency,
                        flags=custom.flags,
                    ).add_to_graph(custom.graph, loc=custom.location)
                elif isinstance(custom, SelfIndex):
                    # iff elements_per_thread is specified, we update
                    # elements_per_thread to chunk size, else return None.
                    self_index_size = gpr_size if custom.elements_per_thread else None
                    new_node = SelfIndex(
                        custom.dim, custom.dtype, self_index_size
                    ).add_to_graph(custom.graph, loc=custom.location)

                # Update new_node information
                new_node.index = updated_index_with_gpr_offset
                new_node.vector_shapes = custom.vector_shapes
                ops_to_combine.append(new_node)

            # Update users of original op.
            if isinstance(custom, Write):
                # Useful to handle write/read dependency
                custom.replace_all_uses_with(ops_to_combine)
            elif isinstance(custom, (Read, SelfIndex)):
                reshape = Reshape(ops_to_combine, custom.vector_shapes).add_to_graph(
                    custom.graph, loc=custom.location
                )
                reshape.expanded_dims = custom.expanded_dims
                reshape.vector_shapes = custom.vector_shapes
                propagate_tag(custom.fx_node, reshape)
                # Also propagate tag to the underlying Read nodes
                for op in ops_to_combine:
                    propagate_tag(custom.fx_node, op)

                # Save the original index on the reshape op so later we can
                # detect if op was part of `gpr_offset` partition.
                reshape.index = index
                custom.replace_all_uses_with(reshape)

            custom.graph.erase_node(custom.fx_node)


@requires_region_format(RegionFormat.DIRECT_OUTER_REF)
def merge_contiguous_reads(
    trace: CapturedTrace, constraints: list[Constraint], target: str
):
    """
    Merge reads that access contiguous physical memory into wider vector loads.

        Each merge must satisfy three proof obligations:
        1. Address equivalence: every original lane r,k reads the same
           physical address from the wide read -- A_r(k) = A_W(o_r + k).
        2. Mask equivalence: every original validity predicate is preserved
           lane-by-lane in the wide mask -- V_r(k) = V_W(o_r + k).
        3. Lowering support: the backend codegen path actually enforces V_W.

        Two phases:

        1. **Single-pass wide merge** (``_wide_merge_fastpath_pass``): groups
           reads into maximal power-of-2 dense windows along the fastest
           physical dimension and emits one wide read per group.  Handles
           both identity/unmapped reads and mapped reads with uniform
           transformed bounds.  This does the bulk of the work.

        2. **Iterative pairwise fallback** (``_merge_contiguous_reads_once``):
           catches residual reads the fast path cannot handle, such as
           shared-memory reads from gather expansion, sparse byte windows,
           or groups with mixed mapping types.  Runs to a fixed point but
           typically completes in one pass with near-zero work after the
           fast path.
    """
    hw_constraint = get_hardware_constraint(constraints)
    fwd, _ = get_divisibility_subs(constraints)
    _wide_merge_fastpath_pass(trace, hw_constraint, divisibility_fwd=fwd)
    while _merge_contiguous_reads_once(trace, hw_constraint, divisibility_fwd=fwd):
        pass


def _get_physical_start(
    custom: Read,
    symbolic_shape: tuple,
    symbolic_dims: list,
) -> dict:
    """Get the physical start coordinates for a read.

    Establishes the address proof: computes per-lane base addresses
    for each dimension so that contiguity (A_r(k) = B + o_r + k) can
    be verified.

    For reads with a flattened LINEAR_INDEX, returns the flat offset directly.
    For reads with a non-identity mapping, applies the mapping to get physical
    coordinates. For identity-mapped reads (mapping=None), reads the start
    offsets directly from the index.
    """
    if LINEAR_INDEX in custom.index:
        return {LINEAR_INDEX: custom.index[LINEAR_INDEX].start}
    if custom.mapping is not None and not custom.has_identity_mapping():
        physical = _get_read_transform_info(custom, symbolic_shape).transformed_index
        if not all(dim in physical for dim in symbolic_dims):
            return None
        return {dim: physical[dim] for dim in symbolic_dims}
    if not all(dim in custom.index for dim in symbolic_dims):
        return None
    return {dim: custom.index[dim].start for dim in symbolic_dims}


def _should_use_transformed_index(
    bounds: dict,
    mapping,
    transformed: dict,
    symbolic_shape: tuple,
) -> bool:
    """Decide whether to use the transformed (physical) index for masking.

    This predicate mirrors the codegen decision in ``_build_mask_with_mapping``
    (``read_write.py``). The codegen version uses ``emitter.dynamic_dims``
    to check for static memory dims; this version uses ``subs_idxc`` which
    is semantically equivalent in the analysis pass context.

    All three conditions must hold:
    - all bounded dims appear in the transformed index
    - the mapping has no dynamic_val_indices
    - all memory shape dims are statically known
    """
    static_memory_dims = all(
        isinstance(subs_idxc(dim), (int, sympy.Integer)) for dim in symbolic_shape
    )
    return (
        all(dim in transformed for dim in bounds)
        and not mapping.dynamic_val_indices
        and static_memory_dims
    )


def _get_read_transform_info(
    custom: Read,
    symbolic_shape: tuple,
) -> ReadTransformInfo:
    """Return cached transformed-index data for masking and physical starts."""
    node = custom.fx_node
    cache = getattr(node, "_read_transform_info_cache", None)
    if cache is not None:
        cached_shape, cached_info = cache
        if cached_shape == symbolic_shape:
            return cached_info

    transformed_index = None
    can_use_transformed_mask = False
    if custom.mapping is not None and not custom.has_identity_mapping():
        transformed_index = transform_index_on_mapping(
            custom.mapping, symbolic_shape, custom.index, is_read=True
        )
        if custom.bounds:
            can_use_transformed_mask = _should_use_transformed_index(
                custom.bounds,
                custom.mapping,
                transformed_index,
                symbolic_shape,
            )

    info = ReadTransformInfo(
        transformed_index=transformed_index,
        can_use_transformed_mask=can_use_transformed_mask,
    )
    node._read_transform_info_cache = (symbolic_shape, info)
    return info


def _flatten_bounds_to_mask_expr(
    custom: Read,
    symbolic_shape: tuple,
):
    """Pre-compute a read's bounds check as a flat sympy boolean.

    Replicates the _build_mask_with_mapping decision logic: when the
    mapping's transformed index contains all bounded dims (and has no
    dynamic_val_indices), the transformed index is used; otherwise the
    original logical index is used.  Returns a sympy boolean expression
    (eg. And(idx < bound, ...)) or None if the read has no bounds.

    """
    if not custom.bounds:
        return None

    index = custom.index

    if custom.mapping is not None and not custom.has_identity_mapping():
        transform_info = _get_read_transform_info(custom, symbolic_shape)
        if transform_info.can_use_transformed_mask:
            index = transform_info.transformed_index

    conditions = []
    for dim, bound in custom.bounds.items():
        if dim not in index:
            continue
        start = (
            index[dim].start if isinstance(index[dim], IndexSequence) else index[dim]
        )
        if isinstance(start, int):
            start = sympy.Integer(start)
        if isinstance(bound, int):
            bound = sympy.Integer(bound)
        conditions.append(sympy.StrictLessThan(start, bound))

    if not conditions:
        return None

    return functools.reduce(sympy.And, conditions)


def _dense_window_has_uniform_mask(masks, wide_ept) -> bool:
    """Check whether sub-read masks form a dense window with one shared mask."""
    if not masks or any(mask is None for _, _, mask in masks):
        return False

    sorted_masks = sorted(masks, key=lambda x: x[0])
    cursor = 0
    first_mask = sorted_masks[0][2]
    for offset, size, mask in sorted_masks:
        if offset != cursor:
            return False
        if mask != first_mask:
            return False
        cursor += size
    return cursor == wide_ept


def _build_wide_mask_expr(sub_reads, symbolic_shape, wide_ept):
    """Build a concatenated sympy mask for a wide read from its sub-reads.

    Each entry in *sub_reads* is ``(offset, size, orig_custom)`` where
    *offset* is the lane offset within the wide vector and *size* is the
    number of lanes that sub-read occupies.  *wide_ept* is the total
    number of elements in the wide read (may exceed ``sum(sizes)`` when
    there are gaps, e.g. multiway coalesce with non-contiguous offsets).

    Builds ``Or(And(lane_in_range_0, mask_0), And(lane_in_range_1, mask_1), ...)``
    using pure boolean ops so that ``gen_sympy_index`` can lower it without
    the nested-Piecewise ``select_stack`` ordering issue.

    Returns a sympy boolean expression or ``None`` when no sub-read has
    bounds.
    """
    from ..._support.indexing import IndexingContext, index_symbol

    masks = []
    for offset, size, custom in sub_reads:
        existing = getattr(custom.fx_node, "precomputed_mask_expr", None)
        if existing is not None:
            masks.append((offset, size, existing))
        else:
            masks.append(
                (offset, size, _flatten_bounds_to_mask_expr(custom, symbolic_shape))
            )

    if not any(m is not None for _, _, m in masks):
        return None

    # When the merged lanes form a dense window and every sub-read carries
    # the same scalar mask, keep that scalar mask instead of materialising
    # a lane-by-lane Piecewise-style predicate. This preserves recursive
    # pairwise merges of bounded mapped reads.
    if _dense_window_has_uniform_mask(masks, wide_ept):
        return masks[0][2]

    idxc = IndexingContext.current()
    iota = idxc.iota(wide_ept)

    terms = []
    for offset, size, mask in masks:
        upper = offset + size
        lane_cond = sympy.And(
            sympy.GreaterThan(iota, offset) if offset > 0 else sympy.true,
            sympy.StrictLessThan(iota, upper),
        )
        if mask is not None:
            # Remap any iota from a previous merge level to the new wide iota.
            old_iota_sym = index_symbol(f"$IOTA{size}")
            if mask.has(old_iota_sym):
                mask = mask.subs(old_iota_sym, iota - offset)
            bound_cond = mask
        else:
            bound_cond = sympy.true
        terms.append(sympy.And(lane_cond, bound_cond))

    return functools.reduce(sympy.Or, terms)


_MASK_REJECTED = sympy.false


def _prove_mask_equivalent(
    reads: list,
    offsets: list[int],
    sizes: list[int],
    merge_dim,
    symbolic_shape: tuple,
    wide_ept: int,
    divisibility_fwd=None,
):
    """Prove that original per-read validity predicates can be re-expressed
    as a single wide-lane predicate V_W such that V_r(k) = V_W(o_r + k).

    Tries strategies cheapest-first:

    1. No bounds on any read -- trivially safe, return None.
    2. All reads have preservable masks (identity/unmapped, precomputed,
       or transformed-index masking available) -- compose per-lane via
       ``_build_wide_mask_expr``.
    3. All reads are mapped with uniform transformed bounds on non-merge
       dimensions -- return a single scalar guard.
    4. Otherwise -- return ``_MASK_REJECTED``.

    Establishes the mask proof: V_r(k) = V_W(o_r + k) for every
    original read r and lane k.
    """
    customs = [r.custom if hasattr(r, "custom") else r for r in reads]

    if not any(c.bounds for c in customs):
        return None

    transform_infos = [_get_read_transform_info(c, symbolic_shape) for c in customs]
    all_preservable = all(
        _can_preserve_bounds_mask(c, symbolic_shape, ti)
        for c, ti in zip(customs, transform_infos)
    )
    if all_preservable:
        return _build_wide_mask_expr(
            [(off, sz, c) for off, sz, c in zip(offsets, sizes, customs)],
            symbolic_shape,
            wide_ept,
        )

    all_have_mapping = all(
        c.mapping is not None and not c.has_identity_mapping() for c in customs
    )
    if not all_have_mapping:
        return _MASK_REJECTED

    anchor = customs[0]
    ref_bounds = anchor.bounds
    if not ref_bounds or any(c.bounds != ref_bounds for c in customs[1:]):
        return _MASK_REJECTED

    anchor_transformed = transform_infos[0].transformed_index
    if anchor_transformed is None:
        return _MASK_REJECTED

    conditions = []
    for dim, bound in ref_bounds.items():
        if dim == merge_dim:
            return _MASK_REJECTED
        if dim not in anchor_transformed:
            return _MASK_REJECTED
        anchor_start = get_start_expr(anchor_transformed[dim])

        for ti in transform_infos[1:]:
            if ti.transformed_index is None or dim not in ti.transformed_index:
                return _MASK_REJECTED
            other_start = get_start_expr(ti.transformed_index[dim])
            raw_diff = anchor_start - other_start
            if not expr_is_zero_under_probes(raw_diff, _MERGE_PROBES, divisibility_fwd):
                return _MASK_REJECTED

        if isinstance(bound, int):
            bound = sympy.Integer(bound)
        conditions.append(sympy.StrictLessThan(anchor_start, bound))

    if not conditions:
        return None

    scalar_guard = functools.reduce(sympy.And, conditions)
    total_covered = sum(sizes)
    if total_covered < wide_ept:
        from ..._support.indexing import IndexingContext

        idxc = IndexingContext.current()
        iota = idxc.iota(wide_ept)
        scalar_guard = sympy.And(
            scalar_guard, sympy.StrictLessThan(iota, total_covered)
        )
    return scalar_guard


def _check_lowering_ok(
    wide_ept: int,
    element_type,
    hw_constraint,
) -> bool:
    """Check that the backend can realize the merged read.

    Verifies hardware resource limits (max vector width).

    Establishes the lowering proof: the selected backend codegen path
    can enforce the merged validity predicate V_W exactly.
    """
    if wide_ept > hw_constraint.max_elems_per_load(element_type):
        return False
    return True


def _emit_wide_read(anchor_custom, wide_index, wide_ept, tag_source, mask_expr=None):
    """Create a merged Read node covering ``wide_ept`` elements."""
    wide_read = Read(
        anchor_custom.memory,
        elements_per_thread=wide_ept,
        mapping=None,
        _write_dependency=anchor_custom._write_dependency,
        flags=anchor_custom.flags,
    ).add_to_graph(anchor_custom.graph, loc=anchor_custom.location)
    wide_custom = get_custom(wide_read)
    wide_custom.index = wide_index
    if hasattr(tag_source, "vector_shapes"):
        wide_read.vector_shapes = deepcopy(tag_source.vector_shapes)
    if hasattr(tag_source, "expanded_dims") and tag_source.expanded_dims is not None:
        wide_read.expanded_dims = deepcopy(tag_source.expanded_dims)
    if mask_expr is not None:
        wide_read.precomputed_mask_expr = mask_expr
    propagate_tag(tag_source, wide_read)
    return wide_read


def _emit_wide_read_linear(
    anchor_custom, min_offset, wide_ept, anchor_stride, tag_source, mask_expr=None
):
    """Create a merged Read for LINEAR_INDEX reads."""
    wide_index = {LINEAR_INDEX: IndexSequence(min_offset, wide_ept, anchor_stride)}
    return _emit_wide_read(anchor_custom, wide_index, wide_ept, tag_source, mask_expr)


def _emit_extract_slice(
    wide_read, offset, size, orig_custom, orig_node, symbolic_shape
):
    """Create an ExtractSlice from a wide read and propagate metadata."""
    extract = ExtractSlice(wide_read, [offset], [size], [1]).add_to_graph(
        orig_custom.graph, loc=orig_custom.location
    )
    extract_custom = get_custom(extract)
    extract_custom.index = deepcopy(orig_custom.index)
    if hasattr(orig_node, "vector_shapes"):
        extract_custom.vector_shapes = deepcopy(orig_node.vector_shapes)
    if hasattr(orig_node, "expanded_dims") and orig_node.expanded_dims is not None:
        extract.expanded_dims = deepcopy(orig_node.expanded_dims)
    propagate_tag(orig_node, extract)
    return extract


def _group_reads_by_memory(
    trace: CapturedTrace,
) -> dict[tuple, list[fx.Node]]:
    """Group reads by (memory, ept, region).

    A new region starts at each subgraph boundary and whenever a
    side-effecting op (write, barrier, ...) is encountered, so we never
    merge reads across such ops.  Reads with dynamic mapping values are
    skipped to keep the merge logic simple.
    """
    groups: dict[tuple, list[fx.Node]] = defaultdict(list)
    region_id = 0
    for subgraph in trace.region_graph.subgraphs.values():
        region_id += 1
        for node in subgraph.nodes:
            custom = get_custom(node)
            if not isinstance(custom, CustomOp):
                continue
            if custom.has_side_effects:
                region_id += 1
                continue
            if not isinstance(custom, Read):
                continue
            if custom.mapping_dynamic_vals:
                continue
            key = (custom.memory, custom.elements_per_thread, region_id)
            groups[key].append(node)
    return groups


def _can_preserve_bounds_mask(
    custom: Read,
    symbolic_shape: tuple,
    transform_info: ReadTransformInfo | None = None,
) -> bool:
    """Check whether a read's bounds mask survives mapping=None lowering.

    Part of the mask proof (V_r(k) = V_W(o_r + k)): checks whether
    the original per-lane validity predicate survives the loss of
    mapping on the merged read.

    The merged wide read is emitted with ``mapping=None``.  In codegen
    ``handle_read``, the mask is generated by one of three paths:

    1. ``precomputed_mask_expr`` -- used when set and buffer_ops is off.
    2. ``_build_mask_with_mapping`` -- used when mapping is present.
    3. ``_build_mask`` -- fallback when mapping is None.

    Path 2 is lost on the merged read.  Path 1 is the replacement, but
    it must faithfully capture the original mask.  For mapped reads the
    pre-flattener ``_flatten_bounds_to_mask_expr`` can only produce a
    correct mask when it uses the *transformed* (physical) index -- the
    same index the codegen would use to build the mask in path 2.
    When the pre-flattener falls back to the *logical* index, the
    resulting mask is evaluated in a different coordinate system than
    the merged read's physical addresses, producing wrong results.

    The ``use_transformed`` predicate mirrors the codegen decision in
    ``_build_mask_with_mapping``:

    - all bounded dims must appear in the transformed index
    - the mapping must have no dynamic_val_indices
    - all memory shape dims must be statically known

    Returns True when it is safe to merge.
    """
    if not custom.bounds:
        return True
    if getattr(custom.fx_node, "precomputed_mask_expr", None) is not None:
        return True
    if custom.mapping is None or custom.has_identity_mapping():
        return True
    if transform_info is None:
        transform_info = _get_read_transform_info(custom, symbolic_shape)
    return transform_info.can_use_transformed_mask


def _do_merge(
    lo_i,
    hi_i,
    merge_dim,
    read_infos,
    ept,
    symbolic_dims,
    symbolic_shape,
    hw_constraint,
    divisibility_fwd=None,
):
    """Emit a wide read merging reads at lo_i and hi_i. Returns True on success.

    Proof obligations:
    1. Address equivalence (A_r(k) = A_W(o_r + k)) -- established by
       caller via ProbeEvaluator.
    2. Mask equivalence (V_r(k) = V_W(o_r + k)) -- proved here via
       _prove_mask_equivalent.
    3. Lowering support (backend realizes V_W) -- proved here via
       _check_lowering_ok.
    """
    _, lo_phys, lo_custom, lo_node = read_infos[lo_i]
    _, _, hi_custom, hi_node = read_infos[hi_i]
    new_ept = 2 * ept
    wide_mask = _prove_mask_equivalent(
        [lo_custom, hi_custom],
        [0, ept],
        [ept, ept],
        merge_dim,
        symbolic_shape,
        new_ept,
        divisibility_fwd,
    )
    if wide_mask is _MASK_REJECTED:
        return False
    if not _check_lowering_ok(new_ept, lo_custom.type.dtype, hw_constraint):
        return False
    with lo_custom.graph.inserting_before(lo_node):
        if LINEAR_INDEX in symbolic_dims:
            anchor_seq = lo_custom.index[LINEAR_INDEX]
            new_index = {
                LINEAR_INDEX: IndexSequence(
                    lo_phys[LINEAR_INDEX], new_ept, anchor_seq.stride
                )
            }
        else:
            new_index = {
                dim: IndexSequence(
                    lo_phys[dim],
                    new_ept if dim == merge_dim else 1,
                    1,
                )
                for dim in symbolic_dims
            }
        merged_read = _emit_wide_read(
            lo_custom, new_index, new_ept, lo_node, mask_expr=wide_mask
        )
        lo_extract = _emit_extract_slice(
            merged_read, 0, ept, lo_custom, lo_node, symbolic_shape
        )
        hi_extract = _emit_extract_slice(
            merged_read, ept, ept, hi_custom, hi_node, symbolic_shape
        )
    lo_custom.replace_all_uses_with(lo_extract)
    hi_custom.replace_all_uses_with(hi_extract)
    lo_custom.graph.erase_node(lo_node)
    hi_custom.graph.erase_node(hi_node)
    return True


def _find_merge_dim_from_diffs(dim_diffs, ept, symbolic_dims):
    """Return the single dimension whose diff equals ept, or None."""
    merge_dim = None
    for dim in symbolic_dims:
        d = dim_diffs[dim]
        if d == ept:
            if merge_dim is not None:
                return None
            merge_dim = dim
        elif d != 0:
            return None
    return merge_dim


def _get_multiway_unmerged_infos(read_infos, merged):
    """Return unmerged byte reads that can be emitted with mapping=None."""
    return [
        info
        for k, info in enumerate(read_infos)
        if k not in merged
        and (info.custom.mapping is None or info.custom.has_identity_mapping())
    ]


# Probe value sets for numeric offset evaluation.  Diverse primes avoid
# floor/Mod aliasing; all positive (symbols are nonneg).
# The generators use three different parity patterns so that at every
# symbol index, at least one probe produces an even value and at least
# one produces an odd value.  This catches wrapping bugs in expressions
# like Mod(coeff*sym + offset, 2^k) where all-even or all-odd probe
# values would land on the same side of a power-of-two boundary.
_MERGE_PROBES: tuple[Callable[[int], int], ...] = (
    lambda i: 137 + i * 31,
    lambda i: 251 + i * 47,
    lambda i: 503 + i * 17,
    lambda i: 48 + i * 37,
    lambda i: 97 + i * 10,
    lambda i: 73 + i * 23,
)


class ProbeEvaluator:
    """Shared numeric probing infrastructure for read coalescing.

    Supports the address proof by providing candidate discovery via
    numeric evaluation and cross-probe consistency verification for the
    contiguity condition A_r(k) = B + o_r + k.

    Resolves symbolic flat offsets and per-dim physical starts into
    concrete values using multiple probe sets, enabling O(1) candidate
    lookup and cross-probe consistency verification.
    """

    def __init__(self, read_infos, symbolic_dims, divisibility_fwd=None):
        n = len(read_infos)
        self.n = n
        self.symbolic_dims = symbolic_dims

        resolved_flat = [subs_idxc(info.flat_offset) for info in read_infos]
        resolved_phys = [
            {dim: subs_idxc(info.phys_start[dim]) for dim in symbolic_dims}
            for info in read_infos
        ]

        if divisibility_fwd:
            resolved_flat = [safe_subs(e, divisibility_fwd) for e in resolved_flat]
            resolved_phys = [
                {dim: safe_subs(e, divisibility_fwd) for dim, e in phys.items()}
                for phys in resolved_phys
            ]

        self._resolved_flat = resolved_flat
        self._resolved_phys = resolved_phys

        all_free = set()
        for expr in resolved_flat:
            if hasattr(expr, "free_symbols"):
                all_free.update(expr.free_symbols)
        for phys in resolved_phys:
            for expr in phys.values():
                if hasattr(expr, "free_symbols"):
                    all_free.update(expr.free_symbols)
        free_list = sorted(all_free, key=str)

        self._probe_maps = [
            {s: gen(i) for i, s in enumerate(free_list)} for gen in _MERGE_PROBES
        ]

        n_probes = len(self._probe_maps)
        self._all_flat = [[None] * n for _ in range(n_probes)]
        self._all_phys = [[None] * n for _ in range(n_probes)]

        for p_idx, probe in enumerate(self._probe_maps):
            for i in range(n):
                try:
                    flat_val = eval_expr(resolved_flat[i], probe)
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                self._all_flat[p_idx][i] = flat_val
                try:
                    self._all_phys[p_idx][i] = {
                        dim: eval_expr(resolved_phys[i][dim], probe)
                        for dim in symbolic_dims
                    }
                except (TypeError, ValueError, ZeroDivisionError):
                    self._all_flat[p_idx][i] = None

        self.numeric_flat = self._all_flat[0]
        self.numeric_phys = self._all_phys[0]

        self.offset_map: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            if self.numeric_flat[i] is not None:
                self.offset_map[self.numeric_flat[i]].append(i)

    def verify_per_dim_deltas(
        self, idx_a, idx_b, symbolic_dims, expected_fastest_delta
    ):
        """Prove per-dim deltas match the emitted wide-read geometry.

        The wide read assumes non-fastest dims have delta 0 and the
        fastest dim has delta ``expected_fastest_delta``.  Both per-dim
        and flat-offset deltas must be symbolically provable.  Each
        delta is verified with ``sym_simplify``; if any residual is
        non-zero / non-constant the candidate is conservatively rejected.

        Symbolic verification is essential because numeric probing with
        correlated trial values cannot detect non-constant diffs in
        Mod/floor/xor expressions that depend on cross-symbol
        interactions.
        """
        flat_diff = sym_simplify(
            self._resolved_flat[idx_b] - self._resolved_flat[idx_a]
        )
        if flat_diff != expected_fastest_delta:
            return False
        fastest_dim = symbolic_dims[-1]
        for dim in symbolic_dims:
            raw_diff = self._resolved_phys[idx_b][dim] - self._resolved_phys[idx_a][dim]
            simplified = sym_simplify(raw_diff)
            expected = expected_fastest_delta if dim == fastest_dim else 0
            if simplified != expected:
                return False
        return True

    def verify_numeric_deltas(
        self, idx_a, idx_b, symbolic_dims, expected_fastest_delta
    ):
        """Verify per-dim deltas using all precomputed probe values."""
        fastest_dim = symbolic_dims[-1]
        for p_idx in range(len(self._probe_maps)):
            flat_a = self._all_flat[p_idx][idx_a]
            flat_b = self._all_flat[p_idx][idx_b]
            if flat_a is None or flat_b is None:
                return False
            if flat_b - flat_a != expected_fastest_delta:
                return False
            phys_a = self._all_phys[p_idx][idx_a]
            phys_b = self._all_phys[p_idx][idx_b]
            if phys_a is None or phys_b is None:
                return False
            for dim in symbolic_dims:
                expected = expected_fastest_delta if dim == fastest_dim else 0
                if phys_b[dim] - phys_a[dim] != expected:
                    return False
        return True

    def verify_pairwise_diffs(
        self, lo_idx, hi_idx, expected_flat_diff, expected_dim_diffs
    ):
        """Confirm both flat and per-dim diffs are consistent across extra probes."""
        for p_idx in range(1, len(self._probe_maps)):
            flat_lo = self._all_flat[p_idx][lo_idx]
            flat_hi = self._all_flat[p_idx][hi_idx]
            if flat_lo is None or flat_hi is None:
                return False
            if flat_hi - flat_lo != expected_flat_diff:
                return False
            phys_lo = self._all_phys[p_idx][lo_idx]
            phys_hi = self._all_phys[p_idx][hi_idx]
            if phys_lo is None or phys_hi is None:
                return False
            for dim in self.symbolic_dims:
                if phys_hi[dim] - phys_lo[dim] != expected_dim_diffs[dim]:
                    return False
        return True


def _build_multiway_group(
    anchor_idx: int,
    unmerged_infos,
    prober: ProbeEvaluator,
    symbolic_dims,
    max_load_width: int,
    coalesced_set: set[int],
) -> list[MultiwayGroupEntry]:
    """Collect reads that share an anchor's aligned byte window."""
    _, anchor_phys, anchor_custom, anchor_node = unmerged_infos[anchor_idx]
    anchor_flat = prober.numeric_flat[anchor_idx]
    group = [
        MultiwayGroupEntry(
            idx=anchor_idx,
            node=anchor_node,
            custom=anchor_custom,
            byte_pos=0,
            phys_start=anchor_phys,
        )
    ]
    for candidate_offset in range(1, max_load_width):
        target_flat = anchor_flat + candidate_offset
        for candidate_idx in prober.offset_map.get(target_flat, []):
            if candidate_idx == anchor_idx or candidate_idx in coalesced_set:
                continue
            if not prober.verify_per_dim_deltas(
                anchor_idx, candidate_idx, symbolic_dims, candidate_offset
            ):
                continue
            _, candidate_phys, candidate_custom, candidate_node = unmerged_infos[
                candidate_idx
            ]
            group.append(
                MultiwayGroupEntry(
                    idx=candidate_idx,
                    node=candidate_node,
                    custom=candidate_custom,
                    byte_pos=candidate_offset,
                    phys_start=candidate_phys,
                )
            )
    return group


def _get_multiway_wide_ept(
    group: list[MultiwayGroupEntry], max_load_width: int
) -> int | None:
    """Return the smallest power-of-2 window covering the group."""
    max_off = max(entry.byte_pos for entry in group)
    wide_ept = 1
    while wide_ept <= max_off:
        wide_ept *= 2
    return wide_ept if wide_ept <= max_load_width else None


def _get_earliest_group_node(group: list[MultiwayGroupEntry]) -> fx.Node:
    """Return the earliest graph node among the group entries."""
    graph = group[0].custom.graph
    node_position = {node: idx for idx, node in enumerate(graph.nodes)}
    return min((entry.node for entry in group), key=lambda node: node_position[node])


def _build_multiway_wide_index(base_phys, symbolic_dims, wide_ept: int):
    """Build the index for a wide read spanning the group's fastest dimension."""
    wide_index = {}
    for dim_idx, dim in enumerate(symbolic_dims):
        size = wide_ept if dim_idx == len(symbolic_dims) - 1 else 1
        wide_index[dim] = IndexSequence(base_phys[dim], size, 1)
    return wide_index


def _coalesce_multiway_group(
    group: list[MultiwayGroupEntry], symbolic_dims, symbolic_shape, wide_ept: int
) -> None:
    """Replace a byte-read group with one wide read and per-byte extracts."""
    anchor_custom = group[0].custom
    base_phys = group[0].phys_start
    earliest_node = _get_earliest_group_node(group)
    wide_mask = _build_wide_mask_expr(
        [(entry.byte_pos, 1, entry.custom) for entry in group],
        symbolic_shape,
        wide_ept,
    )
    with get_custom(earliest_node).graph.inserting_before(earliest_node):
        wide_read = _emit_wide_read(
            anchor_custom,
            _build_multiway_wide_index(base_phys, symbolic_dims, wide_ept),
            wide_ept,
            earliest_node,
            mask_expr=wide_mask,
        )

    extracts = []
    for entry in group:
        with entry.custom.graph.inserting_before(entry.node):
            extract = _emit_extract_slice(
                wide_read,
                entry.byte_pos,
                1,
                entry.custom,
                entry.node,
                symbolic_shape,
            )
            extracts.append((extract, entry.custom, entry.node))

    for extract, custom, node in extracts:
        custom.replace_all_uses_with(extract)
        custom.graph.erase_node(node)


def _is_fastpath_eligible(custom: Read) -> bool:
    """Check if a read qualifies for the single-pass wide-merge fast path.

    The fast path assumes reads that are consecutive in the fastest
    physical dimension also have consecutive flat offsets.  This holds
    for mapped reads (where transform_index_on_mapping already produced
    physical coordinates) and for LINEAR_INDEX reads (pre-linearized),
    but not necessarily for identity/unmapped reads in multi-dimensional
    tensors with stride > 1.

    Those reads are left to the iterative pairwise path which handles
    arbitrary stride patterns via per-dim delta verification.
    """
    if LINEAR_INDEX in custom.index:
        return True
    if custom.mapping is None or custom.has_identity_mapping():
        return False
    return True


class _WideGroupEntry(NamedTuple):
    idx: int
    node: fx.Node
    custom: Read
    offset: int
    size: int
    phys_start: dict


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _resolve_group_mask(
    entries: list,
    merge_dim,
    symbolic_shape: tuple,
    wide_ept: int,
    divisibility_fwd=None,
):
    """Determine the mask for a wide-merge group.

    Delegates to ``_prove_mask_equivalent`` which tries strategies
    cheapest-first.  Returns the mask expression or ``_MASK_REJECTED``
    if the group cannot be safely merged.

    Establishes the mask proof: V_r(k) = V_W(o_r + k).
    """
    return _prove_mask_equivalent(
        [e.custom for e in entries],
        [e.offset for e in entries],
        [e.size for e in entries],
        merge_dim,
        symbolic_shape,
        wide_ept,
        divisibility_fwd,
    )


def _wide_group_merge_fastpath(
    read_infos,
    ept,
    symbolic_dims,
    symbolic_shape,
    hw_constraint,
    divisibility_fwd=None,
) -> set[int]:
    """Single-pass wide merge for dense read windows.

    Partitions eligible reads into maximal power-of-2 dense groups along
    the fastest physical dimension and emits one wide read per group,
    avoiding the iterative pairwise doubling loop.  Handles both
    identity/unmapped reads and mapped reads with uniform transformed
    bounds.

    Returns the set of ``read_infos`` indices that were consumed.
    """
    if len(read_infos) < 2:
        return set()

    if not isinstance(ept, (int, sympy.Integer)):
        return set()
    ept = int(ept)
    element_type = read_infos[0].custom.type.dtype
    max_load_width = hw_constraint.max_elems_per_load(element_type)
    if max_load_width <= ept:
        return set()

    eligible_indices = [
        i for i, info in enumerate(read_infos) if _is_fastpath_eligible(info.custom)
    ]
    if len(eligible_indices) < 2:
        return set()

    eligible_infos = [read_infos[i] for i in eligible_indices]
    prober = ProbeEvaluator(eligible_infos, symbolic_dims, divisibility_fwd)

    consumed: set[int] = set()
    consumed_eligible: set[int] = set()

    sorted_eligible = sorted(
        (i for i in range(len(eligible_infos)) if prober.numeric_flat[i] is not None),
        key=lambda i: prober.numeric_flat[i],
    )

    graph = eligible_infos[0].custom.graph if eligible_infos else None
    node_position = {n: idx for idx, n in enumerate(graph.nodes)} if graph else {}

    i = 0
    while i < len(sorted_eligible):
        anchor = sorted_eligible[i]
        if anchor in consumed_eligible:
            i += 1
            continue

        anchor_flat = prober.numeric_flat[anchor]
        group = [anchor]

        j = i + 1
        next_expected = anchor_flat + ept
        while j < len(sorted_eligible):
            cand = sorted_eligible[j]
            if cand in consumed_eligible:
                j += 1
                continue
            cand_flat = prober.numeric_flat[cand]
            if cand_flat != next_expected:
                break
            if not prober.verify_numeric_deltas(
                anchor, cand, symbolic_dims, cand_flat - anchor_flat
            ):
                break
            group.append(cand)
            next_expected += ept
            j += 1

        if len(group) < 2:
            i += 1
            continue

        wide_ept = _next_power_of_two(len(group) * ept)
        if wide_ept > max_load_width:
            wide_ept = max_load_width
        n_reads_in_window = wide_ept // ept
        group = group[:n_reads_in_window]

        if len(group) < 2:
            i += 1
            continue

        wide_ept = _next_power_of_two(len(group) * ept)
        if wide_ept > max_load_width:
            i += 1
            continue

        anchor_info = eligible_infos[group[0]]
        base_phys = anchor_info.phys_start

        entries = []
        for rank, eidx in enumerate(group):
            info = eligible_infos[eidx]
            entries.append(
                _WideGroupEntry(
                    idx=eligible_indices[eidx],
                    node=info.node,
                    custom=info.custom,
                    offset=rank * ept,
                    size=ept,
                    phys_start=info.phys_start,
                )
            )

        merge_dim = symbolic_dims[-1]
        wide_mask = _resolve_group_mask(
            entries, merge_dim, symbolic_shape, wide_ept, divisibility_fwd
        )
        if wide_mask is _MASK_REJECTED:
            i += 1
            continue

        earliest_node = min((e.node for e in entries), key=lambda n: node_position[n])

        with get_custom(earliest_node).graph.inserting_before(earliest_node):
            if LINEAR_INDEX in symbolic_dims:
                anchor_seq = anchor_info.custom.index[LINEAR_INDEX]
                wide_index = {
                    LINEAR_INDEX: IndexSequence(
                        base_phys[LINEAR_INDEX], wide_ept, anchor_seq.stride
                    )
                }
            else:
                wide_index = {}
                for dim_idx, dim in enumerate(symbolic_dims):
                    size = wide_ept if dim_idx == len(symbolic_dims) - 1 else 1
                    wide_index[dim] = IndexSequence(base_phys[dim], size, 1)

            wide_read = _emit_wide_read(
                anchor_info.custom,
                wide_index,
                wide_ept,
                earliest_node,
                mask_expr=wide_mask,
            )

        for entry in entries:
            with entry.custom.graph.inserting_before(entry.node):
                extract = _emit_extract_slice(
                    wide_read,
                    entry.offset,
                    entry.size,
                    entry.custom,
                    entry.node,
                    symbolic_shape,
                )
            entry.custom.replace_all_uses_with(extract)
            entry.custom.graph.erase_node(entry.node)

        for eidx in group:
            consumed_eligible.add(eidx)
            consumed.add(eligible_indices[eidx])

        resume_pos = i + 1
        while resume_pos < len(sorted_eligible):
            if sorted_eligible[resume_pos] not in consumed_eligible:
                break
            resume_pos += 1
        i = resume_pos

    return consumed


def _wide_merge_fastpath_pass(
    trace: CapturedTrace, hw_constraint, divisibility_fwd=None
) -> None:
    """Run the single-pass wide merge fast path across all read groups.

    For each (memory, ept, region) group, attempts to coalesce eligible
    identity/unmapped reads directly into the widest legal reads without
    iterative pairwise doubling.  Reads that are not eligible or not
    consumed are left for the subsequent iterative pairwise/multiway pass.
    """
    from ...compiler.utils import strides_from_symbolic_shape
    from ..._support.indexing import IndexingContext

    groups = _group_reads_by_memory(trace)
    idxc = IndexingContext.current()

    for (memory_node, ept, _region), reads in groups.items():
        if len(reads) < 2:
            continue

        customs = [(get_custom(n), n) for n in reads]
        memory = get_custom(memory_node)
        symbolic_shape = memory.type.symbolic_shape
        strides = strides_from_symbolic_shape(
            idxc, symbolic_shape, allow_mixed_shapes=True
        )
        symbolic_dims = [infer_dim(d) for d in symbolic_shape]

        read_infos = []
        for custom, node in customs:
            phys_start = _get_physical_start(custom, symbolic_shape, symbolic_dims)
            if phys_start is None:
                continue
            if LINEAR_INDEX in phys_start:
                flat_offset = phys_start[LINEAR_INDEX]
                read_infos.append(ReadInfo(flat_offset, phys_start, custom, node))
            else:
                flat_offset = sum(
                    phys_start[dim] * stride
                    for dim, stride in zip(symbolic_dims, strides)
                )
                read_infos.append(ReadInfo(flat_offset, phys_start, custom, node))

        if len(read_infos) < 2:
            continue

        # Split into LINEAR_INDEX reads and N-D reads, handle separately.
        linear_infos = [ri for ri in read_infos if LINEAR_INDEX in ri.phys_start]
        nd_infos = [ri for ri in read_infos if LINEAR_INDEX not in ri.phys_start]

        if len(linear_infos) >= 2:
            linear_dims = [LINEAR_INDEX]
            _wide_group_merge_fastpath(
                linear_infos,
                ept,
                linear_dims,
                symbolic_shape,
                hw_constraint,
                divisibility_fwd=divisibility_fwd,
            )

        if len(nd_infos) >= 2:
            _wide_group_merge_fastpath(
                nd_infos,
                ept,
                symbolic_dims,
                symbolic_shape,
                hw_constraint,
                divisibility_fwd=divisibility_fwd,
            )


def _pairwise_merge(
    read_infos, ept, symbolic_dims, symbolic_shape, hw_constraint, divisibility_fwd=None
):
    """Merge pairs of reads whose flat offsets differ by exactly ``ept``.

    Returns ``(merged_indices, did_merge)`` where *merged_indices* is
    the set of read_infos indices that were consumed.

    Evaluates each offset independently with concrete probe values and
    uses dict lookup for O(n) candidate matching, avoiding O(n²)
    symbolic diff resolution.
    """
    if len(read_infos) < 2:
        return set(), False

    prober = ProbeEvaluator(read_infos, symbolic_dims, divisibility_fwd)
    merged = set()
    did_merge = False

    for i in range(prober.n):
        if i in merged or prober.numeric_flat[i] is None:
            continue
        flat_i = prober.numeric_flat[i]
        found = False
        for target_flat, i_is_lo in ((flat_i + ept, True), (flat_i - ept, False)):
            for j in prober.offset_map.get(target_flat, []):
                if j in merged or j == i:
                    continue
                if prober.numeric_phys[j] is None:
                    continue
                lo_idx, hi_idx = (i, j) if i_is_lo else (j, i)
                dim_diffs = {
                    dim: prober.numeric_phys[hi_idx][dim]
                    - prober.numeric_phys[lo_idx][dim]
                    for dim in symbolic_dims
                }
                merge_dim = _find_merge_dim_from_diffs(dim_diffs, ept, symbolic_dims)
                if merge_dim is None:
                    continue
                flat_diff = prober.numeric_flat[hi_idx] - prober.numeric_flat[lo_idx]
                if not prober.verify_pairwise_diffs(
                    lo_idx, hi_idx, flat_diff, dim_diffs
                ):
                    continue
                if _do_merge(
                    lo_idx,
                    hi_idx,
                    merge_dim,
                    read_infos,
                    ept,
                    symbolic_dims,
                    symbolic_shape,
                    hw_constraint,
                    divisibility_fwd,
                ):
                    merged.update({i, j})
                    did_merge = True
                    found = True
                    break
            if found:
                break

    return merged, did_merge


def _multiway_coalesce(
    read_infos,
    merged,
    reads,
    symbolic_dims,
    symbolic_shape,
    hw_constraint,
    divisibility_fwd=None,
):
    """Coalesce unmerged ept==1 reads whose flat offsets fall in an aligned window.

    Groups reads whose numerically-probed flat offsets fall within a
    power-of-2 aligned window (up to ``max_elems_per_load``), then emits
    a single wide read with per-byte ExtractSlice ops.

    Only identity-mapped (or unmapped) reads are eligible.  The wide
    read is emitted with ``mapping=None``, so reads whose physical
    coordinates are produced by a non-identity mapping must be excluded
    to avoid address mismatches in codegen.

    Each candidate is admitted only when ``sym_simplify`` can prove that
    non-fastest dimension deltas are exactly 0 and the fastest dimension
    delta equals the byte offset.
    """
    element_type = get_custom(reads[0]).type.dtype
    max_load_width = hw_constraint.max_elems_per_load(element_type)

    unmerged_infos = _get_multiway_unmerged_infos(read_infos, merged)
    if len(unmerged_infos) < 2:
        return False

    prober = ProbeEvaluator(unmerged_infos, symbolic_dims, divisibility_fwd)

    coalesced_any = False
    coalesced_set: set[int] = set()
    for anchor_idx in range(len(unmerged_infos)):
        if anchor_idx in coalesced_set:
            continue
        if prober.numeric_flat[anchor_idx] is None:
            continue
        group = _build_multiway_group(
            anchor_idx,
            unmerged_infos,
            prober,
            symbolic_dims,
            max_load_width,
            coalesced_set,
        )
        if len(group) < 2:
            continue

        group.sort(key=lambda entry: entry.byte_pos)
        wide_ept = _get_multiway_wide_ept(group, max_load_width)
        if wide_ept is None:
            continue

        _coalesce_multiway_group(group, symbolic_dims, symbolic_shape, wide_ept)

        coalesced_set.update(entry.idx for entry in group)
        coalesced_any = True

    return coalesced_any


def _merge_contiguous_reads_once(
    trace: CapturedTrace, hw_constraint, divisibility_fwd=None
) -> bool:
    """Single merge pass: merge reads that access nearby physical memory.

    Two strategies are applied per (memory, ept) group:

    1. **Pairwise contiguous merge** (``_pairwise_merge``): pairs whose
       physical flat offset starts differ by exactly ``ept`` are merged
       into a ``2*ept`` read with two ExtractSlice outputs.

    2. **Multi-way coalescing** (``_multiway_coalesce``, ``ept==1`` only):
       unmerged byte reads whose flat offsets fall within a power-of-2
       aligned window are replaced by a single wide read with per-byte
       ExtractSlice outputs.

    Returns True if any merges or coalescing happened.
    """
    from ...compiler.utils import strides_from_symbolic_shape
    from ..._support.indexing import IndexingContext

    groups = _group_reads_by_memory(trace)
    idxc = IndexingContext.current()
    merged_any = False

    for (memory_node, ept, _region), reads in groups.items():
        if len(reads) < 2:
            continue

        customs = [(get_custom(n), n) for n in reads]
        memory = get_custom(memory_node)
        symbolic_shape = memory.type.symbolic_shape
        strides = strides_from_symbolic_shape(
            idxc, symbolic_shape, allow_mixed_shapes=True
        )
        symbolic_dims = [infer_dim(d) for d in symbolic_shape]

        read_infos = []
        for custom, node in customs:
            phys_start = _get_physical_start(custom, symbolic_shape, symbolic_dims)
            if phys_start is None:
                continue
            if LINEAR_INDEX in phys_start:
                flat_offset = phys_start[LINEAR_INDEX]
            else:
                flat_offset = sum(
                    phys_start[dim] * stride
                    for dim, stride in zip(symbolic_dims, strides)
                )
            read_infos.append(ReadInfo(flat_offset, phys_start, custom, node))

        # Split into LINEAR_INDEX reads and N-D reads.
        linear_infos = [ri for ri in read_infos if LINEAR_INDEX in ri.phys_start]
        nd_infos = [ri for ri in read_infos if LINEAR_INDEX not in ri.phys_start]

        if len(linear_infos) >= 2:
            linear_dims = [LINEAR_INDEX]
            merged, did_merge = _pairwise_merge(
                linear_infos,
                ept,
                linear_dims,
                symbolic_shape,
                hw_constraint,
                divisibility_fwd=divisibility_fwd,
            )
            merged_any |= did_merge

        if len(nd_infos) >= 2:
            merged, did_merge = _pairwise_merge(
                nd_infos,
                ept,
                symbolic_dims,
                symbolic_shape,
                hw_constraint,
                divisibility_fwd=divisibility_fwd,
            )
            merged_any |= did_merge

        if ept == 1 and len(nd_infos) >= 2:
            merged_any |= _multiway_coalesce(
                nd_infos,
                merged,
                reads,
                symbolic_dims,
                symbolic_shape,
                hw_constraint,
                divisibility_fwd=divisibility_fwd,
            )

    return merged_any


@requires_region_format(RegionFormat.DIRECT_OUTER_REF)
def partition_gather_like_ops(
    trace: CapturedTrace, constraints: list[Constraint], target: str
):
    """
    This pass partitions gather-like operations (reads/writes with non-contiguous access patterns) into
    multiple contiguous operations.

    For example, if we have a write operation that writes elements with stride 2:
    write([0,2,4,6]) -> write([0]), write([2]), write([4]), write([6])

    The pass:
    1. Identifies reads/writes with non-contiguous access patterns
    2. For each such operation:
       - Creates multiple single-element reads/writes
       - Updates indices to access the correct elements
       - Handles dynamic values in mappings
       - Combines results back together if needed (for reads)
    """

    def has_gather_mapping(node: fx.Node) -> bool:
        """
        Checks for writes on 2d tensors with strided access on a single dimension that
        read more than a single element.
        """
        custom = get_custom(node)
        if isinstance(custom, (Read, Write)):
            return not custom.is_contiguous_vec(constraints, target)

        return False

    strided_operators = trace.walk(has_gather_mapping)
    for operator in strided_operators:
        custom = get_custom(operator)
        index = custom.index
        fastest_index = get_fastest_index(index)
        elements_per_thread = subs_idxc(custom.elements_per_thread)

        # Break apart Reads/Writes that has non-contiguous access patterns.
        with custom.graph.inserting_before(operator):
            ops_to_combine = []
            for i in range(elements_per_thread):
                new_index = deepcopy(index)
                for j, v in enumerate(new_index.values()):
                    if j == fastest_index:
                        v.start = v.start + i
                    v.size = 1
                    v.stride = 1

                new_dynamic_vals = []
                for dynamic_val in custom.mapping_dynamic_vals:
                    _, size = get_largest_index_and_size(dynamic_val.index)
                    if size == 1:
                        # If size is 1, it means we are broadcasting same dynamic value to all
                        # vector elements.
                        new_dynamic_vals.append(dynamic_val)
                        continue

                    assert (
                        size == elements_per_thread
                    ), f"Expected size to be equal to {elements_per_thread}, got {size}"

                    # Otherwise, we need to extract the dynamic value for the current vector element.
                    extract = ExtractSlice(dynamic_val, [i], [1], [1]).add_to_graph(
                        custom.graph, loc=custom.location
                    )
                    extract.index = new_index
                    new_dynamic_vals.append(extract)

                if isinstance(custom, Write):
                    extract = ExtractSlice(
                        custom.register_, [i], [1], [1]
                    ).add_to_graph(custom.graph, loc=custom.location)
                    extract.index = new_index
                    new_node = Write(
                        extract,
                        custom.memory,
                        mapping=custom.mapping,
                        mapping_dynamic_vals=new_dynamic_vals,
                        elements_per_thread=1,
                        flags=custom.flags,
                    ).add_to_graph(custom.graph, loc=custom.location)
                elif isinstance(custom, Read):
                    new_node = Read(
                        custom.memory,
                        elements_per_thread=1,
                        mapping=custom.mapping,
                        mapping_dynamic_vals=new_dynamic_vals,
                        _write_dependency=custom._write_dependency,
                        flags=custom.flags,
                    ).add_to_graph(custom.graph, loc=custom.location)
                else:
                    raise NotImplementedError(f"Unsupported op type: {custom}")

                # Update new_node information
                new_node.index = new_index
                new_node.vector_shapes = custom.vector_shapes
                ops_to_combine.append(new_node)

            # Update users of original op.
            if isinstance(custom, Write):
                # Useful to handle write/read dependency
                custom.replace_all_uses_with(ops_to_combine)
            elif isinstance(custom, Read):
                reshape = Reshape(ops_to_combine, custom.vector_shapes).add_to_graph(
                    custom.graph
                )
                reshape.expanded_dims = custom.expanded_dims
                reshape.vector_shapes = custom.vector_shapes
                propagate_tag(custom.fx_node, reshape)
                # Also propagate tag to the underlying Read nodes
                for op in ops_to_combine:
                    propagate_tag(custom.fx_node, op)

                reshape.index = index
                custom.replace_all_uses_with(reshape)
            else:
                raise NotImplementedError(f"Unsupported op type: {custom}")

            custom.erase()


def _simplify_expr(
    expr: sympy.Expr,
    fwd: list[tuple[sympy.Symbol, sympy.Expr]],
    bwd: list[tuple[sympy.Symbol, sympy.Expr]],
) -> sympy.Expr:
    """Run subs_idxc + simplify with divisibility rewriting."""
    expr = subs_idxc(expr)
    if fwd:
        expr = safe_subs(expr, fwd)
    expr = sym_simplify(expr)
    if bwd:
        expr = safe_subs(expr, bwd)
    return expr


def _simplify_symbols_map(
    mapping: dict,
    fwd: list,
    bwd: list,
) -> tuple[dict, bool]:
    """Simplify values in a symbol map (``{dim: expr}``).

    Returns ``(new_map, changed)``."""
    new_map = {}
    changed = False
    for key, val in mapping.items():
        new_val = _simplify_expr(val, fwd, bwd)
        new_map[key] = new_val
        if new_val != val:
            changed = True
    return new_map, changed


def _simplify_mapping(
    mapping: IndexMapping,
    fwd: list,
    bwd: list,
) -> tuple[IndexMapping, bool]:
    """Simplify expressions inside an IndexMapping.

    Returns ``(new_mapping, True)`` if any expression changed,
    ``(original_mapping, False)`` otherwise.
    """
    new_inputs, inp_changed = _simplify_symbols_map(mapping.input_mapping, fwd, bwd)
    new_outputs, out_changed = _simplify_symbols_map(mapping.output_mapping, fwd, bwd)
    new_dyn_mappings = []
    dyn_changed = False
    for dvm in mapping.dynamic_val_mappings or ():
        new_dvm, c = _simplify_symbols_map(dvm, fwd, bwd)
        new_dyn_mappings.append(new_dvm)
        dyn_changed |= c
    if not (inp_changed or out_changed or dyn_changed):
        return mapping, False
    return (
        IndexMapping(
            mapping.num_iterators,
            new_inputs,
            new_outputs,
            dynamic_val_mappings=tuple(new_dyn_mappings),
        ),
        True,
    )


@requires_region_format(RegionFormat.DIRECT_OUTER_REF)
def simplify_indices(trace: CapturedTrace, constraints: Sequence[Constraint] = ()):
    """Pre-simplify index expressions on all ops.

    Runs ``simplify(subs_idxc(component))`` on every ``start``, ``size``,
    and ``stride`` of every ``IndexSequence`` in every op's index dict,
    and on every expression in Read/Write index mappings.

    Divisibility assumptions (``Assumption(Eq(Mod(S, d), 0))``) are
    extracted from *constraints* and used to resolve floor/Mod sub-expressions.

    This normalises indices once so downstream passes (contiguity checks,
    merge, partition) don't each pay the simplification cost independently.
    """
    fwd, bwd = get_divisibility_subs(constraints)
    for subgraph in trace.region_graph.subgraphs.values():
        for node in subgraph.nodes:
            custom = get_custom(node)
            if not isinstance(custom, CustomOp):
                continue
            # Simplify index mappings on Read/Write ops.
            if isinstance(custom, (Read, Write)) and custom.mapping is not None:
                new_mapping, mapping_changed = _simplify_mapping(
                    custom.mapping, fwd, bwd
                )
                if mapping_changed:
                    custom.mapping = new_mapping
                # Try to eliminate flat//D and flat%D patterns using
                # tile-level iterator bounds from the node's index.
                # Re-read custom from node because setting custom.mapping
                # above writes to the IR node, invalidating the wrapper.
                custom = get_custom(node)
                if custom.mapping is not None:
                    try:
                        node_index = custom.index
                    except (ValueError, AttributeError):
                        node_index = None
                    if isinstance(node_index, dict):
                        tile_sizes = get_tile_sizes_from_index(
                            custom.mapping, node_index
                        )
                        if tile_sizes:
                            new_mapping2, mapping_changed2 = simplify_index_mapping(
                                custom.mapping, constraints, tile_sizes
                            )
                            if mapping_changed2:
                                custom.update_arg("mapping", new_mapping2)
            # Simplify index sequences.
            try:
                index = custom.index
            except (ValueError, AttributeError):
                continue
            if isinstance(index, dict):
                new_index = {}
                changed = False
                for dim, seq in index.items():
                    if not isinstance(seq, IndexSequence):
                        new_index[dim] = seq
                        continue
                    new_start = _simplify_expr(seq.start, fwd, bwd)
                    new_size = _simplify_expr(seq.size, fwd, bwd)
                    new_stride = _simplify_expr(seq.stride, fwd, bwd)
                    if (
                        new_start != seq.start
                        or new_size != seq.size
                        or new_stride != seq.stride
                    ):
                        new_index[dim] = IndexSequence(new_start, new_size, new_stride)
                        changed = True
                    else:
                        new_index[dim] = seq
                if changed:
                    custom.index = new_index
