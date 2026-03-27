# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Flatten Read indices from N-D to 1-D physical offsets.

This pass runs after ``generate_bounds_exprs`` and before
``merge_contiguous_reads``.  For every eligible Read it:

1. Resolves the index mapping (if any) into physical coordinates.
2. Linearizes the N-D physical starts into a single flat offset.
3. Converts bounds to expression-keyed form via ``delinearize_index``.
4. Replaces the index with ``{LINEAR_INDEX: IndexSequence(flat, ept, 1)}``.

Reads with ``mapping_dynamic_vals`` are skipped -- the existing codegen
path handles them and ``merge_contiguous_reads`` already skips them.
"""

from ..._support.indexing import IndexSequence, IndexingContext
from ..._support.tracing import CapturedTrace
from ...compiler.utils import strides_from_symbolic_shape
from ...lang.global_symbols import LINEAR_INDEX, SHARED_ADDRESS_SPACE
from ...ops.wave_ops import Read, get_custom
from ..constraints import Constraint
from ..utils.general_utils import delinearize_index, infer_dim
from ..utils.mapping_utils import transform_index_on_mapping
from ..utils.symbol_utils import subs_idxc


def _get_physical_starts(
    custom: Read,
    symbolic_shape: tuple,
    symbolic_dims: list,
) -> dict | None:
    """Return per-dim physical start expressions for a Read.

    For mapped reads, applies the mapping.  For unmapped / identity-mapped
    reads, reads the start directly from the index.  Returns ``None`` when
    required dimensions are missing.
    """
    if custom.mapping is not None and not custom.has_identity_mapping():
        transformed = transform_index_on_mapping(
            custom.mapping, symbolic_shape, custom.index, is_read=True
        )
        if not all(dim in transformed for dim in symbolic_dims):
            return None
        return {dim: transformed[dim] for dim in symbolic_dims}
    if not all(dim in custom.index for dim in symbolic_dims):
        return None
    return {
        dim: (
            custom.index[dim].start
            if isinstance(custom.index[dim], IndexSequence)
            else custom.index[dim]
        )
        for dim in symbolic_dims
    }


def _convert_bounds(
    bounds: dict,
    flat_start,
    ept,
    symbolic_shape: tuple,
    symbolic_dims: list,
) -> dict | None:
    """Convert per-dim bounds to expression-keyed form.

    Delinearizes ``flat_start + iota(ept)`` back to per-dim coordinates
    and maps each bounded dim to its delinearized expression.
    Returns ``None`` if there are no applicable bounds.
    """
    idxc = IndexingContext.current()
    flat_with_iota = flat_start + idxc.iota(ept)
    shape_sizes = [subs_idxc(d) for d in symbolic_shape]
    coords = delinearize_index(flat_with_iota, shape_sizes)

    new_bounds = {}
    for dim, bound in bounds.items():
        if dim not in symbolic_dims:
            continue
        dim_idx = symbolic_dims.index(dim)
        new_bounds[coords[dim_idx]] = bound
    return new_bounds or None


def flatten_read_indices(trace: CapturedTrace, constraints: list[Constraint]):
    """Flatten every eligible Read's N-D index to a 1-D physical offset."""
    idxc = IndexingContext.current()

    for subgraph in trace.region_graph.subgraphs.values():
        for node in subgraph.nodes:
            custom = get_custom(node)
            if not isinstance(custom, Read):
                continue
            # Skip reads with dynamic mapping values.
            if custom.mapping_dynamic_vals:
                continue

            memory = get_custom(custom.memory)
            symbolic_shape = memory.type.symbolic_shape
            symbolic_dims = [infer_dim(d) for d in symbolic_shape]

            # Step 1: get physical starts.
            phys_starts = _get_physical_starts(custom, symbolic_shape, symbolic_dims)
            if phys_starts is None:
                continue

            # Step 2: pick the right shape for stride computation.
            if memory.type.address_space == SHARED_ADDRESS_SPACE and hasattr(
                memory, "distributed_shape"
            ):
                stride_shape = memory.distributed_shape
            else:
                stride_shape = symbolic_shape

            strides = strides_from_symbolic_shape(
                idxc, stride_shape, allow_mixed_shapes=True
            )
            flat_start = sum(
                phys_starts[dim] * stride for dim, stride in zip(symbolic_dims, strides)
            )

            # Step 3: convert bounds.
            ept = custom.elements_per_thread
            new_bounds = None
            if custom.bounds:
                new_bounds = _convert_bounds(
                    custom.bounds,
                    flat_start,
                    ept,
                    symbolic_shape,
                    symbolic_dims,
                )

            # Step 4: rewrite the op.
            custom.index = {LINEAR_INDEX: IndexSequence(flat_start, ept, 1)}
            if custom.mapping is not None:
                custom.update_arg("mapping", None)
            custom.update_arg("bounds", new_bounds)
