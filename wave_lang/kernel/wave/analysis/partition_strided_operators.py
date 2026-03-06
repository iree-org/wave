# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from collections.abc import Sequence
from copy import deepcopy
from itertools import groupby
from operator import itemgetter

import numpy as np
import sympy
import torch.fx as fx

from wave_lang.support.logging import get_logger

from ..._support.indexing import IndexSequence, IndexSymbol
from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import *
from ...lang.wave_types import IndexMapping
from ...ops.wave_ops import (
    CustomOp,
    ExtractSlice,
    Read,
    Reshape,
    SelfIndex,
    Write,
    get_custom,
)
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
    _numeric_eval_constant,
    safe_subs,
    simplify as sym_simplify,
    subs_idxc,
)

logger = get_logger("wave.partition_strided_operators")


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


def merge_contiguous_reads(
    trace: CapturedTrace, constraints: list[Constraint], target: str
):
    """
    Merge reads that access contiguous physical memory into wider vector loads.

    Runs to a fixed point, doubling the vector width each iteration:
    ept=1 pairs → ept=2, ept=2 pairs → ept=4, etc. Works regardless of how
    the reads were created (expansion, manual, etc.).

    Reads are grouped by (memory operand, ept). Within each group, pairs whose
    physical flat offset starts differ by exactly ept are merged.
    """
    hw_constraint = get_hardware_constraint(constraints)
    while _merge_contiguous_reads_once(trace, hw_constraint):
        pass


def _get_physical_start(
    custom: Read,
    symbolic_shape: tuple,
    symbolic_dims: list,
) -> dict:
    """Get the physical start coordinates for a read.

    For reads with a non-identity mapping, applies the mapping to get physical
    coordinates. For identity-mapped reads (mapping=None), reads the start
    offsets directly from the index.
    """
    if custom.mapping is not None and not custom.has_identity_mapping():
        physical = transform_index_on_mapping(
            custom.mapping, symbolic_shape, custom.index, is_read=True
        )
        if not all(dim in physical for dim in symbolic_dims):
            return None
        return {dim: physical[dim] for dim in symbolic_dims}
    if not all(dim in custom.index for dim in symbolic_dims):
        return None
    return {dim: custom.index[dim].start for dim in symbolic_dims}


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
        transformed = transform_index_on_mapping(
            custom.mapping, symbolic_shape, index, is_read=True
        )
        use_transformed = (
            all(dim in transformed for dim in custom.bounds)
            and not custom.mapping.dynamic_val_indices
        )
        if use_transformed:
            index = transformed

    conditions = []
    for dim, bound in custom.bounds.items():
        if dim not in index:
            continue
        start = (
            index[dim].start if isinstance(index[dim], IndexSequence) else index[dim]
        )
        start = sympy.sympify(start)
        bound = sympy.sympify(bound)
        conditions.append(sympy.StrictLessThan(start, bound))

    if not conditions:
        return None

    return functools.reduce(sympy.And, conditions)


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
    from ..._support.indexing import IndexingContext

    masks = [
        (offset, size, _flatten_bounds_to_mask_expr(custom, symbolic_shape))
        for offset, size, custom in sub_reads
    ]

    if not any(m is not None for _, _, m in masks):
        return None

    idxc = IndexingContext.current()
    iota = idxc.iota(wide_ept)

    terms = []
    for offset, size, mask in masks:
        upper = offset + size
        lane_cond = sympy.And(
            sympy.GreaterThan(iota, offset) if offset > 0 else sympy.true,
            sympy.StrictLessThan(iota, upper),
        )
        bound_cond = mask if mask is not None else sympy.true
        terms.append(sympy.And(lane_cond, bound_cond))

    return functools.reduce(sympy.Or, terms)


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
    if mask_expr is not None:
        wide_read.precomputed_mask_expr = mask_expr
    propagate_tag(tag_source, wide_read)
    return wide_read


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
    from collections import defaultdict

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
            if getattr(node, "precomputed_mask_expr", None) is not None:
                continue
            key = (custom.memory, custom.elements_per_thread, region_id)
            groups[key].append(node)
    return groups


def _resolve_symbolic_diff(raw_diff, has_complex_mapping, expected_vals=None):
    """Resolve a raw sympy offset difference to a value, or None.

    Strategy:
      1. If already a plain int / sympy.Integer, return it directly.
      2. If the mapping is complex (non-identity), use numeric probing.
      3. Otherwise try sym_simplify.  If ``expected_vals`` is given and
         the result isn't among them, fall back to numeric probing;
         return None when neither approach succeeds.
    """
    if isinstance(raw_diff, (int, sympy.Integer)):
        return int(raw_diff)
    if has_complex_mapping:
        return _numeric_eval_constant(raw_diff)
    simplified = sym_simplify(raw_diff)
    if expected_vals is None or simplified in expected_vals:
        return simplified
    nv = _numeric_eval_constant(raw_diff)
    return nv


def _pairwise_merge(read_infos, ept, symbolic_dims, symbolic_shape, hw_constraint):
    """Merge pairs of reads whose flat offsets differ by exactly ``ept``.

    Returns ``(merged_indices, did_merge)`` where *merged_indices* is
    the set of read_infos indices that were consumed.
    """
    merged = set()
    did_merge = False

    for i in range(len(read_infos)):
        if i in merged:
            continue
        for j in range(i + 1, len(read_infos)):
            if j in merged:
                continue
            off1, phys1, custom1, node1 = read_infos[i]
            off2, phys2, custom2, node2 = read_infos[j]

            raw_diff = subs_idxc(off2 - off1)
            has_complex_mapping = (
                custom1.mapping is not None and not custom1.has_identity_mapping()
            )
            diff = _resolve_symbolic_diff(
                raw_diff, has_complex_mapping, expected_vals={ept, -ept}
            )
            if diff is None:
                continue

            if diff == ept:
                lo_phys, hi_phys = phys1, phys2
                lo_custom, hi_custom = custom1, custom2
                lo_node, hi_node = node1, node2
            elif diff == -ept:
                lo_phys, hi_phys = phys2, phys1
                lo_custom, hi_custom = custom2, custom1
                lo_node, hi_node = node2, node1
            else:
                continue

            merge_dim = None
            for dim in symbolic_dims:
                raw_d = subs_idxc(hi_phys[dim] - lo_phys[dim])
                d = _resolve_symbolic_diff(
                    raw_d, has_complex_mapping, expected_vals={ept, 0}
                )
                if d is None:
                    merge_dim = None
                    break
                if d == ept:
                    merge_dim = dim
                elif d != 0:
                    merge_dim = None
                    break
            if merge_dim is None:
                continue

            new_ept = 2 * ept
            element_type = lo_custom.type.dtype
            if new_ept > hw_constraint.max_elems_per_load(element_type):
                continue
            wide_mask = _build_wide_mask_expr(
                [(0, ept, lo_custom), (ept, ept, hi_custom)],
                symbolic_shape,
                new_ept,
            )
            with lo_custom.graph.inserting_before(lo_node):
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

            merged.update({i, j})
            did_merge = True
            break

    return merged, did_merge


def _multiway_coalesce(
    read_infos, merged, reads, symbolic_dims, symbolic_shape, hw_constraint
):
    """Coalesce unmerged ept==1 reads whose flat offsets fall in an aligned window.

    Groups reads whose numerically-probed flat offsets fall within a
    power-of-2 aligned window (up to ``max_elems_per_load``), then emits
    a single wide read with per-byte ExtractSlice ops.
    """
    element_type = get_custom(reads[0]).type.dtype
    max_load_width = hw_constraint.max_elems_per_load(element_type)

    unmerged_infos = [read_infos[k] for k in range(len(read_infos)) if k not in merged]
    if len(unmerged_infos) < 2:
        return False

    coalesced_any = False
    coalesced_set: set[int] = set()
    for anchor_idx in range(len(unmerged_infos)):
        if anchor_idx in coalesced_set:
            continue
        off_a, phys_a, custom_a, node_a = unmerged_infos[anchor_idx]

        group = [(anchor_idx, node_a, custom_a, 0, phys_a)]
        for probe_idx in range(len(unmerged_infos)):
            if probe_idx == anchor_idx or probe_idx in coalesced_set:
                continue
            off_p, _, custom_p, node_p = unmerged_infos[probe_idx]
            raw_diff = subs_idxc(off_p - off_a)
            diff_val = _numeric_eval_constant(raw_diff)
            if diff_val is None:
                continue
            if 0 < diff_val < max_load_width:
                _, _, _, phys_p = unmerged_infos[probe_idx]
                group.append((probe_idx, node_p, custom_p, diff_val, phys_p))

        if len(group) < 2:
            continue

        group.sort(key=lambda x: x[3])
        max_off = group[-1][3]
        wide_ept = 1
        while wide_ept <= max_off:
            wide_ept *= 2
        if wide_ept > max_load_width:
            continue

        base_phys = group[0][4]

        earliest_node = group[0][1]
        for g in group[1:]:
            candidate = g[1]
            for n in custom_a.graph.nodes:
                if n is candidate:
                    earliest_node = candidate
                    break
                if n is earliest_node:
                    break

        wide_mask = _build_wide_mask_expr(
            [(byte_pos, 1, g_custom) for _, _, g_custom, byte_pos, _ in group],
            symbolic_shape,
            wide_ept,
        )
        with get_custom(earliest_node).graph.inserting_before(earliest_node):
            wide_index = {}
            for dim_idx, dim in enumerate(symbolic_dims):
                if dim_idx == len(symbolic_dims) - 1:
                    wide_index[dim] = IndexSequence(base_phys[dim], wide_ept, 1)
                else:
                    wide_index[dim] = IndexSequence(base_phys[dim], 1, 1)

            wide_read = _emit_wide_read(
                custom_a, wide_index, wide_ept, earliest_node, mask_expr=wide_mask
            )

        extracts = []
        for g_idx, g_node, g_custom, byte_pos, _ in group:
            with g_custom.graph.inserting_before(g_node):
                ext = _emit_extract_slice(
                    wide_read, byte_pos, 1, g_custom, g_node, symbolic_shape
                )
                extracts.append((ext, g_custom, g_node))

        for ext, g_custom, g_node in extracts:
            g_custom.replace_all_uses_with(ext)
            g_custom.graph.erase_node(g_node)

        coalesced_set.update(g[0] for g in group)
        coalesced_any = True

    return coalesced_any


def _merge_contiguous_reads_once(trace: CapturedTrace, hw_constraint) -> bool:
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
            flat_offset = sum(
                phys_start[dim] * stride for dim, stride in zip(symbolic_dims, strides)
            )
            read_infos.append((flat_offset, phys_start, custom, node))

        merged, did_merge = _pairwise_merge(
            read_infos, ept, symbolic_dims, symbolic_shape, hw_constraint
        )
        merged_any |= did_merge

        # Only ept==1 (byte) reads need multi-way coalescing; wider reads
        # are already handled by the pairwise merge above.
        if ept == 1 and len(read_infos) >= 2:
            merged_any |= _multiway_coalesce(
                read_infos, merged, reads, symbolic_dims, symbolic_shape, hw_constraint
            )

    return merged_any


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
