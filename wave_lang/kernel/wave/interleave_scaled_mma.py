# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    NewScalar,
    Reshape,
    ScaledMMA,
    get_custom,
)
from .constraints import (
    Constraint,
    ScaledMMAType,
)
from .utils.general_utils import get_hardware_constraint


def interleave_scaled_mma(trace: CapturedTrace, constraints: list[Constraint]):
    """
    Transforms ScaledMMA operations using F32_16x16x128_F8F6F4 into
    interleaved variants that pack scale values into a single VGPR.

    When two ScaledMMA ops exist in the same subgraph, their scales are
    packed into one register [a0, b0, a1, b1] and the first op uses
    SCALES_INTERLEAVED (bytes 0,1) while the second uses
    SCALES_INTERLEAVED_UPPER (bytes 2,3).

    Unpaired ops fall back to [a, b, 0, 0] with SCALES_INTERLEAVED.
    """
    hardware_constraint = get_hardware_constraint(constraints)

    def is_target_scaled_mma(node: fx.Node) -> bool:
        custom = get_custom(node)
        if not isinstance(custom, ScaledMMA):
            return False
        mma_type = custom.mma_type or hardware_constraint.mma_type
        return mma_type == ScaledMMAType.F32_16x16x128_F8F6F4

    nodes = trace.walk(is_target_scaled_mma)
    if not nodes:
        return

    # Group nodes by subgraph so we can pair within each one.
    graph_groups: dict[fx.Graph, list[fx.Node]] = {}
    for node in nodes:
        graph_groups.setdefault(node.graph, []).append(node)

    for group in graph_groups.values():
        i = 0
        while i < len(group):
            mma_a = get_custom(group[i])
            scale_dtype = get_custom(mma_a.lhs_scale).type.dtype

            if i + 1 < len(group):
                # Pair: pack all 4 scales into one register.
                mma_b = get_custom(group[i + 1])

                with mma_a.graph.inserting_before(mma_a.fx_node):
                    combined = Reshape(
                        [
                            mma_a.lhs_scale,
                            mma_a.rhs_scale,
                            mma_b.lhs_scale,
                            mma_b.rhs_scale,
                        ],
                        {},
                    ).add_to_graph(mma_a.graph, loc=mma_a.location)

                    new_a = ScaledMMA(
                        mma_a.lhs,
                        combined,
                        mma_a.rhs,
                        combined,
                        mma_a.acc,
                        ScaledMMAType.F32_16x16x128_F8F6F4_SCALES_INTERLEAVED,
                    ).add_to_graph(mma_a.graph, loc=mma_a.location)
                    new_a.index = mma_a.index
                    new_a.vector_shapes = mma_a.vector_shapes

                with mma_b.graph.inserting_before(mma_b.fx_node):
                    new_b = ScaledMMA(
                        mma_b.lhs,
                        combined,
                        mma_b.rhs,
                        combined,
                        mma_b.acc,
                        ScaledMMAType.F32_16x16x128_F8F6F4_SCALES_INTERLEAVED_UPPER,
                    ).add_to_graph(mma_b.graph, loc=mma_b.location)
                    new_b.index = mma_b.index
                    new_b.vector_shapes = mma_b.vector_shapes

                mma_a.replace_all_uses_with(new_a)
                mma_a.erase()
                mma_b.replace_all_uses_with(new_b)
                mma_b.erase()
                i += 2
            else:
                # Unpaired: pad upper half with zeros.
                with mma_a.graph.inserting_before(mma_a.fx_node):
                    zero = NewScalar(0.0, scale_dtype).add_to_graph(
                        mma_a.graph, loc=mma_a.location
                    )
                    combined = Reshape(
                        [mma_a.lhs_scale, mma_a.rhs_scale, zero, zero],
                        {},
                    ).add_to_graph(mma_a.graph, loc=mma_a.location)

                    new_a = ScaledMMA(
                        mma_a.lhs,
                        combined,
                        mma_a.rhs,
                        combined,
                        mma_a.acc,
                        ScaledMMAType.F32_16x16x128_F8F6F4_SCALES_INTERLEAVED,
                    ).add_to_graph(mma_a.graph, loc=mma_a.location)
                    new_a.index = mma_a.index
                    new_a.vector_shapes = mma_a.vector_shapes

                mma_a.replace_all_uses_with(new_a)
                mma_a.erase()
                i += 1
