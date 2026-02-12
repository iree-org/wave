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
    F32_16x16x128_F8F6F4_SCALES_INTERLEAVED by combining separate
    lhs_scale and rhs_scale into a single 4-element vector
    [a_scale, b_scale, 0, 0] passed as both scale inputs.

    This reduces register pressure by packing both scale values
    into a single VGPR, using byte index 0 for a_scale and
    byte index 1 for b_scale.
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

    for node in nodes:
        mma_op = get_custom(node)
        scale_dtype = get_custom(mma_op.lhs_scale).type.dtype

        with mma_op.graph.inserting_before(mma_op.fx_node):
            # Create zero padding scalar.
            zero = NewScalar(0.0, scale_dtype).add_to_graph(
                mma_op.graph, loc=mma_op.location
            )

            # Combine scales: [lhs_scale, rhs_scale, 0, 0].
            combined = Reshape(
                [mma_op.lhs_scale, mma_op.rhs_scale, zero, zero],
                {},
            ).add_to_graph(mma_op.graph, loc=mma_op.location)

            # Create new ScaledMMA with interleaved type.
            new_mma = ScaledMMA(
                mma_op.lhs,
                combined,
                mma_op.rhs,
                combined,
                mma_op.acc,
                ScaledMMAType.F32_16x16x128_F8F6F4_SCALES_INTERLEAVED,
            ).add_to_graph(mma_op.graph, loc=mma_op.location)
            new_mma.index = mma_op.index
            new_mma.vector_shapes = mma_op.vector_shapes

        mma_op.replace_all_uses_with(new_mma)
        mma_op.erase()
