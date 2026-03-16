# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Annotate Read and GatherToLDS nodes with pre-computed IV strides.

Walks all Read and GatherToLDS nodes that have a non-identity mapping and
computes the linearized IV stride through the mapping via numerical probing.
The result is stored in ``node.meta["iv_stride"]`` so that codegen can use
the PRECOMPUTED fast path in ``_try_iv_split_offset`` without performing
any symbolic analysis at MLIR emission time.

This pass should run after ``generate_bounds_exprs`` (all indices are final)
and before ``merge_contiguous_reads`` (which drops mappings on merged reads
but propagates ``meta["iv_stride"]`` from the anchor).
"""

from collections.abc import Sequence

from ..._support.indexing import IndexingContext
from ..._support.tracing import CapturedTrace
from ...compiler.utils import strides_from_symbolic_shape
from ...ops.wave_ops import Read, GatherToLDS, get_custom
from ..constraints import Constraint
from ..utils.mapping_utils import compute_iv_stride_through_mapping


def annotate_iv_strides(
    trace: CapturedTrace,
    constraints: Sequence[Constraint] = (),
):
    """Annotate every mapped Read/GatherToLDS with ``meta["iv_stride"]``."""
    idxc = IndexingContext.current()

    for node in trace.walk(lambda n: isinstance(get_custom(n), (Read, GatherToLDS))):
        if node.meta.get("iv_stride") is not None:
            continue

        custom = get_custom(node)

        if isinstance(custom, GatherToLDS):
            mapping = custom.src_mapping
            index = custom.src_index
            mem_node = custom.src
        else:
            mapping = custom.mapping
            index = custom.index
            mem_node = custom.memory

        if mapping is None:
            continue
        if hasattr(custom, "has_identity_mapping") and custom.has_identity_mapping():
            continue

        symbolic_shape = custom.type.symbolic_shape
        mem_sym_shape = get_custom(mem_node).type.symbolic_shape
        phys_strides = strides_from_symbolic_shape(
            idxc, mem_sym_shape, allow_mixed_shapes=True
        )

        iv_stride = compute_iv_stride_through_mapping(
            mapping,
            symbolic_shape,
            index,
            is_read=True,
            mem_strides=list(phys_strides),
            constraints=constraints,
        )
        if iv_stride is not None:
            node.meta["iv_stride"] = iv_stride
