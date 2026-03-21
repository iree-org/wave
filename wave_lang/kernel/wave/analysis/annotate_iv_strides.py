# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Annotate IV strides on flattened Read ops.

For each Read with a ``LINEAR_INDEX`` inside a loop body, this pass
computes ``stride = simplify(flat(IV + step) - flat(IV))``.  If the
stride is loop-invariant (does not contain the IV), the flat offset is
rewritten to ``base + IV * stride`` form.

This makes strength reduction trivial for downstream backends: the
address increment per iteration is a known constant or loop-invariant
expression.
"""

import sympy

from ..._support.indexing import IndexSequence
from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import LINEAR_INDEX
from ...ops.wave_ops import Iterate, Read, get_custom
from ..constraints import Constraint
from ..utils.general_utils import is_flattened_index, get_flat_offset
from ..utils.symbol_utils import get_induction_symbol, safe_subs
from ..utils.symbol_utils import simplify as sym_simplify


def annotate_iv_strides(trace: CapturedTrace, constraints: list[Constraint]):
    """Rewrite flattened Read indices to ``base + IV * stride`` form.

    Walks loop subgraphs, identifies the induction variable from the
    parent ``Iterate`` op, and for each flattened Read proves that the
    flat offset is affine in the IV.  When successful, rewrites the
    offset and stores the stride on the node for downstream consumers.
    """
    for subgraph in trace.region_graph.subgraphs.values():
        parent_node = getattr(subgraph, "parent_op", None)
        if parent_node is None:
            continue
        parent = get_custom(parent_node)
        if not isinstance(parent, Iterate):
            continue

        iv_sym = get_induction_symbol(parent.axis)
        step = parent.step if parent.step is not None else 1

        for node in subgraph.nodes:
            custom = get_custom(node)
            if not isinstance(custom, Read):
                continue
            if not is_flattened_index(custom.index):
                continue

            flat = get_flat_offset(custom.index)
            if iv_sym not in flat.free_symbols:
                continue

            # stride = f(IV + step) - f(IV).
            shifted = safe_subs(flat, {iv_sym: iv_sym + step})
            stride = sym_simplify(shifted - flat)

            # Stride must be loop-invariant (IV-free).
            if iv_sym in stride.free_symbols:
                continue

            # Rewrite: base = f(IV=0), flat = base + IV * stride.
            base = sym_simplify(safe_subs(flat, {iv_sym: sympy.Integer(0)}))
            ept = custom.index[LINEAR_INDEX].size

            custom.index = {LINEAR_INDEX: IndexSequence(base + iv_sym * stride, ept, 1)}
