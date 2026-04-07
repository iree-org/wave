# Copyright 2026 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Flatten N-D read indices to 1-D physical offsets (LINEAR_INDEX).

For every eligible Read or GatherToLDS (unmapped and mapped), this pass:

1. Resolves the index mapping (if any) into physical coordinates.
2. Linearizes them into a single flat offset using memory strides.
3. Converts bounds to expression-keyed form via ``delinearize_index``.
4. Replaces the index with ``{LINEAR_INDEX: IndexSequence(flat, ept, 1)}``.

Shared-memory targets are skipped.  Operations with
``mapping_dynamic_vals`` are flattened normally; the dynamic val symbols
(``$dynamic_val0``, etc.) remain as free symbols in the flat expression.

For mapped reads the flat expression is intentionally left unsimplified
so that ``gen_sympy_index`` lowers each floor/Mod term independently,
producing correct integer MLIR ops.  Algebraic simplification (e.g.
``mem_simplify``) is only applied to unmapped reads where the
expressions are simple enough not to cause floor/Mod mismatches.

IV stride extraction is NOT done here -- that happens in the separate
``annotate_iv_strides`` post-merge pass.
"""

from collections.abc import Sequence

import sympy

from ..._support.indexing import IndexingContext, IndexSequence
from ..._support.tracing import CapturedTrace
from ...compiler.utils import (
    strides_from_symbolic_shape,
    symbolic_strides_match_physical_memory,
)
from ...lang.global_symbols import LINEAR_INDEX, SHARED_ADDRESS_SPACE
from ...ops.wave_ops import (
    ExtractSlice,
    GatherToLDS,
    MemoryAccessFlags,
    Read,
    get_custom,
)
from ..assumptions import get_divisibility_subs
from ..compile_options import WaveCompileOptions
from ..constraints import Constraint
from ..utils.general_utils import (
    infer_dim,
    is_flattened_index,
)
from ..utils.mapping_utils import (
    _infer_floor_to_exact,
    linearize_dims,
    mem_simplify,
    transform_index_on_mapping,
)
from ..utils.symbol_utils import subs_idxc


def _should_use_original_index(phys, index_entry):
    """Return True when the physical start should NOT be used for masking.

    The physical start (post-mapping) may contain symbols introduced by
    the mapping that shift the coordinate into a different space than the
    bound applies to.  This happens with:
    - ``$dynamic_val`` symbols from ``dynamic_val_mappings``
    - runtime offsets added via ``set_symbol`` (e.g. OFFSET, EXT_IDX)

    We detect this by checking whether the physical start has free symbols
    that are absent from the original (pre-mapping) index start.  Any
    extra symbols indicate a coordinate-space shift that makes the bound
    comparison invalid.
    """
    orig_start = (
        index_entry.start if isinstance(index_entry, IndexSequence) else index_entry
    )
    phys_syms = sympy.sympify(phys).free_symbols
    orig_syms = sympy.sympify(orig_start).free_symbols
    return bool(phys_syms - orig_syms)


def _convert_bounds(bounds, index, phys_starts, ept, symbolic_shape, symbolic_dims):
    """Convert per-dim bounds to expression-keyed form.

    Uses the *physical* per-dimension start expressions (post-mapping)
    as the keys for the converted bounds so the mask correctly checks
    the physical coordinate against the bound.  For the innermost
    (fastest) dimension, ``iota(ept)`` is added so the mask is
    per-element; for all other dimensions the start is scalar and
    gets broadcast.

    Falls back to the original (pre-mapping) index for dimensions whose
    physical start contains extra symbols not present in the original
    index (e.g. ``$dynamic_val`` symbols or ``set_symbol``'d offsets),
    because the bound value applies to the logical coordinate space,
    not the shifted physical space.

    Returns ``None`` if there are no applicable bounds.
    """
    idxc = IndexingContext.current()
    fastest_dim = symbolic_dims[-1]

    new_bounds = {}
    for dim, bound in bounds.items():
        if dim not in symbolic_dims:
            continue
        phys = phys_starts[dim]
        if dim in index and _should_use_original_index(phys, index[dim]):
            dim_start = (
                index[dim].start
                if isinstance(index[dim], IndexSequence)
                else index[dim]
            )
        else:
            dim_start = phys
        if dim == fastest_dim:
            dim_start = dim_start + idxc.iota(ept)
        new_bounds[dim_start] = bound
    return new_bounds or None


def _get_physical_starts(mapping, index, symbolic_shape, symbolic_dims):
    """Return per-dim physical start expressions for a read operation.

    For mapped reads, applies the mapping via ``transform_index_on_mapping``.
    For unmapped / identity-mapped reads, reads starts directly from the index.
    Returns ``None`` when required dimensions are missing.
    """
    if mapping is not None and not mapping.is_identity():
        transformed = transform_index_on_mapping(
            mapping, symbolic_shape, index, is_read=True
        )
        if not all(dim in transformed for dim in symbolic_dims):
            return None
        return {dim: transformed[dim] for dim in symbolic_dims}
    if not all(dim in index for dim in symbolic_dims):
        return None
    return {
        dim: (index[dim].start if isinstance(index[dim], IndexSequence) else index[dim])
        for dim in symbolic_dims
    }


def _is_shared_memory(mem_node):
    """Return True if *mem_node* targets shared address space."""
    memory = get_custom(mem_node)
    return (
        hasattr(memory, "type")
        and hasattr(memory.type, "address_space")
        and subs_idxc(memory.type.address_space) == SHARED_ADDRESS_SPACE
    )


def _linearize_to_flat(
    mapping,
    index,
    symbolic_shape,
    symbolic_dims,
    stride_shape,
    div_fwd,
    div_bwd,
    idxc,
):
    """Compute the flat LINEAR_INDEX start for *index* given *mapping*.

    *stride_shape* is the shape used for computing memory strides.  It
    equals the physical_layout shape when a MemoryLayout is present,
    otherwise it is the same as *symbolic_shape*.

    Returns ``(flat_start, new_bounds_or_None)`` or ``None`` when the
    index cannot be flattened.
    """
    has_mapping = mapping is not None and not mapping.is_identity()

    phys_starts = _get_physical_starts(mapping, index, symbolic_shape, symbolic_dims)
    if phys_starts is None:
        return None

    mem_strides = list(
        strides_from_symbolic_shape(idxc, stride_shape, allow_mixed_shapes=True)
    )

    if has_mapping:
        dim_exprs = [sympy.sympify(phys_starts[dim]) for dim in symbolic_dims]
        flat_start = sum(expr * stride for expr, stride in zip(dim_exprs, mem_strides))
    else:
        dim_exprs = [sympy.sympify(phys_starts[dim]) for dim in symbolic_dims]
        dim_exprs = [subs_idxc(e) for e in dim_exprs]

        if div_fwd:
            fwd_dict = dict(div_fwd)
            dim_exprs = [sympy.sympify(e).subs(fwd_dict) for e in dim_exprs]
            applied_strides = [sympy.sympify(s).subs(fwd_dict) for s in mem_strides]
        else:
            floor_subs = _infer_floor_to_exact(mem_strides)
            if floor_subs:
                dim_exprs = [sympy.sympify(e).subs(floor_subs) for e in dim_exprs]
            applied_strides = mem_strides

        flat_start = linearize_dims(dim_exprs, applied_strides)

        if div_bwd:
            bwd_dict = dict(div_bwd)
            flat_start = mem_simplify(sympy.sympify(flat_start).subs(bwd_dict))

    return flat_start


def flatten_read_indices(
    trace: CapturedTrace,
    constraints: Sequence[Constraint] = (),
    options: WaveCompileOptions | None = None,
):
    """Flatten N-D read indices to 1-D LINEAR_INDEX for eligible Reads and GatherToLDS ops.

    *options* is required when invoked from ``compile.py``; a default of ``None``
    keeps ad-hoc unit tests from constructing a full options object.
    """
    idxc = IndexingContext.current()
    div_fwd, div_bwd = get_divisibility_subs(constraints)

    for node in trace.walk(lambda n: isinstance(get_custom(n), (Read, GatherToLDS))):
        custom = get_custom(node)
        is_g2l = isinstance(custom, GatherToLDS)

        index = custom.src_index if is_g2l else custom.index
        mem_node = custom.src if is_g2l else custom.memory
        bounds = custom.src_bounds if is_g2l else custom.bounds
        mapping = custom.src_mapping if is_g2l else custom.mapping

        if is_flattened_index(index):
            continue

        if not is_g2l and _is_shared_memory(mem_node):
            continue

        if not is_g2l and custom.flags != MemoryAccessFlags.NONE:
            continue

        if any(isinstance(get_custom(u), ExtractSlice) for u in node.users):
            continue

        memory = get_custom(mem_node)
        symbolic_shape = memory.type.symbolic_shape
        layout = getattr(memory.type, "physical_layout", None)

        # Dynamic-strides-specific guards: under the LLVM + wave runtime
        # ABI the only correct dense-stride assumption is when physical
        # layout matches or is absent AND non-contiguous buffers are not
        # expected.  The symbolic_strides_match_physical_memory check
        # handles layout skew (e.g. attention's transposed layouts); the
        # allow_noncontiguous_runtime_buffers opt-out handles slice views.
        if options is not None and options.dynamic_strides:
            if not symbolic_strides_match_physical_memory(memory, symbolic_shape):
                continue
            if options.allow_noncontiguous_runtime_buffers and layout is None:
                continue
        symbolic_dims = [infer_dim(d) for d in symbolic_shape]

        stride_shape = layout.shape if layout is not None else symbolic_shape

        phys_starts = _get_physical_starts(
            mapping, index, symbolic_shape, symbolic_dims
        )
        if phys_starts is None:
            continue

        flat_start = _linearize_to_flat(
            mapping,
            index,
            symbolic_shape,
            symbolic_dims,
            stride_shape,
            div_fwd,
            div_bwd,
            idxc,
        )
        if flat_start is None:
            continue

        ept = custom.elements_per_thread
        ept_val = subs_idxc(ept) if not isinstance(ept, int) else ept

        new_bounds = None
        if bounds:
            new_bounds = _convert_bounds(
                bounds,
                index,
                phys_starts,
                ept_val,
                symbolic_shape,
                symbolic_dims,
            )

        new_index = {LINEAR_INDEX: IndexSequence(flat_start, ept_val, 1)}

        if is_g2l:
            custom.update_arg("src_index", new_index)
            custom.update_arg("src_mapping", None)
            custom.update_arg("src_bounds", new_bounds)
        else:
            custom.index = new_index
            custom.update_arg("mapping", None)
            custom.update_arg("bounds", new_bounds)
