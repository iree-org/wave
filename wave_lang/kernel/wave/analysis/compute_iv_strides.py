# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Pre-codegen pass that computes the linearized K-stride for global reads
inside tiled loops.  Tags eligible read nodes with ``iv_k_stride`` (int)
and ``iv_linear = True`` so that the codegen emitter can produce
VALU-free addressing (voffset + soffset) without any numerical estimation
at emit time.

The analysis operates on the *pre-mapping* index expressions (single-variable
sympy objects), which are tractable for linearity checks.  The post-mapping
composition is intractable but not needed here.
"""

from __future__ import annotations

from typing import Optional

import sympy
import torch.fx as fx

from ..._support.indexing import (
    IndexExpr,
    IndexingContext,
    IndexSequence,
    IndexSymbol,
    subs_idxc,
)
from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import SHARED_ADDRESS_SPACE
from ...lang.wave_types import IndexMapping
from ...ops.wave_ops import Read, get_custom
from ..constraints import Constraint, TilingConstraint
from ..utils.general_utils import infer_dim
from ..utils.symbol_utils import safe_subs
from ...compiler.utils import strides_from_symbolic_shape

import logging

logger = logging.getLogger(__name__)


def _get_tiling_constraints(
    constraints: list[Constraint],
) -> list[TilingConstraint]:
    return [c for c in constraints if isinstance(c, TilingConstraint)]


def _linearity_check_3pt(
    expr: IndexExpr, sym: IndexSymbol, step: int
) -> Optional[IndexExpr]:
    """
    3-point linearity check on *expr* w.r.t. *sym*.
    Returns the constant stride per unit of *sym*, or None if non-linear.
    All other free symbols are zeroed for the check.
    """
    others = {s: 0 for s in expr.free_symbols if s != sym}
    try:
        v0 = safe_subs(expr, {**others, sym: 0})
        v1 = safe_subs(expr, {**others, sym: step})
        v2 = safe_subs(expr, {**others, sym: 2 * step})
    except (TypeError, ValueError, sympy.SympifyError):
        return None
    d1 = sympy.simplify(v1 - v0)
    d2 = sympy.simplify(v2 - v1)
    if sympy.simplify(d2 - d1) != 0:
        return None
    if step == 0:
        return None
    return sympy.simplify(d1 / step)


def _compute_stride_for_read(
    node: fx.Node,
    tiling_constraints: list[TilingConstraint],
    idxc: IndexingContext,
) -> Optional[IndexExpr]:
    """
    For a single Read node, compute the linearized K-stride per IV step.
    Returns the integer stride, or None if the read is ineligible.
    """
    custom = get_custom(node)
    if not isinstance(custom, Read):
        return None

    if not hasattr(node, "index") or node.index is None:
        return None

    index = node.index  # dict[IndexSymbol, IndexSequence]
    mapping: Optional[IndexMapping] = custom.mapping

    # Identify which tiling constraint's IV appears in the index expressions.
    iv_constraint = None
    iv_sym = None
    for tc in tiling_constraints:
        if tc.induction_var is None:
            continue
        for dim_sym, seq in index.items():
            start = seq.start if isinstance(seq, IndexSequence) else seq
            if isinstance(start, (int, float)):
                continue
            if tc.induction_var in start.free_symbols:
                if iv_constraint is not None and iv_constraint is not tc:
                    # Multiple IVs — bail
                    return None
                iv_constraint = tc
                iv_sym = tc.induction_var
                break

    if iv_constraint is None or iv_sym is None:
        return None

    tiled_dim = iv_constraint.dim
    # Step 1: Find the tiled dimension in the index dict and verify linearity.
    #
    # The index dict maps logical dimensions to IndexSequence objects.
    # The tiled dimension's start should be linear in IV.
    tiled_dim_key = None
    for dim_sym in index:
        base = infer_dim(dim_sym)
        if base == infer_dim(tiled_dim):
            tiled_dim_key = dim_sym
            break

    if tiled_dim_key is None:
        return None

    tiled_seq = index[tiled_dim_key]
    tiled_start = tiled_seq.start if isinstance(tiled_seq, IndexSequence) else tiled_seq

    stride_into_iter = _linearity_check_3pt(tiled_start, iv_sym, 1)
    if stride_into_iter is None:
        return None

    # Get the memory node's symbolic shape
    memory_node = custom.memory
    mem_custom = get_custom(memory_node)
    mem_shape = mem_custom.type.symbolic_shape

    # Get physical strides from the memory shape
    phys_strides = strides_from_symbolic_shape(idxc, mem_shape, allow_mixed_shapes=True)
    if phys_strides is None:
        return None
    # Substitute concrete values
    phys_strides_expr = [subs_idxc(s) for s in phys_strides]

    # Step 2: If mapping present, verify mapping linearity and compute per-dim deltas.
    if mapping is not None:
        # The output_mapping maps logical dims to iterators.
        # Find which iterator carries the tiled axis.
        k_iter = None
        for dim_sym, iter_expr in mapping.output_mapping.items():
            if infer_dim(dim_sym) == infer_dim(tiled_dim):
                k_iter = iter_expr
                break

        if k_iter is None:
            return None

        # The IV advances the iterator by stride_into_iter per step.
        iv_stride_into_iter = stride_into_iter

        # 3-point check on each physical dimension's input_mapping expression.
        all_iters = list(mapping.iters.keys())
        other_iters = [it for it in all_iters if it != k_iter]

        per_dim_delta = {}
        for dim_sym, expr in mapping.input_mapping.items():
            zero_others = {it: 0 for it in other_iters}
            d = _linearity_check_3pt(
                safe_subs(expr, zero_others), k_iter, iv_stride_into_iter
            )
            if d is None:
                return None
            per_dim_delta[dim_sym] = d

        # Compute linearized stride
        k_stride = 0
        for dim_sym, ps in zip(mem_shape, phys_strides_expr):
            base_dim = infer_dim(dim_sym)
            delta = 0
            for pd_dim, pd_val in per_dim_delta.items():
                if infer_dim(pd_dim) == base_dim:
                    delta = pd_val
                    break
            k_stride += delta * ps

    else:
        # No mapping: logical dimensions ARE physical dimensions.
        # The tiled dimension advances by stride_into_iter, others by 0.
        k_stride = 0
        for dim_sym, ps in zip(mem_shape, phys_strides_expr):
            if infer_dim(dim_sym) == infer_dim(tiled_dim):
                k_stride += stride_into_iter * ps

    k_stride = sympy.simplify(k_stride)
    return k_stride if k_stride != 0 else None


def compute_iv_strides(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    Walk the graph and tag each eligible global-memory Read with:
      - node.iv_k_stride  (int)   — linearized bytes-offset per IV step
      - node.iv_linear    (bool)  — True
    """
    tiling_constraints = _get_tiling_constraints(constraints)
    if not tiling_constraints:
        return

    idxc = IndexingContext.current()

    def tag_node(node: fx.Node) -> bool:
        custom = get_custom(node)
        if not isinstance(custom, Read):
            return False
        # Only global reads
        if hasattr(custom, "memory_type") and hasattr(
            custom.memory_type, "address_space"
        ):
            if custom.memory_type.address_space == SHARED_ADDRESS_SPACE:
                return False

        stride = _compute_stride_for_read(node, tiling_constraints, idxc)
        if stride is not None:
            node.iv_k_stride = stride
            node.iv_linear = True
            logger.debug(f"Tagged read {node.name} with iv_k_stride={stride}")
        return False

    trace.walk(tag_node)
