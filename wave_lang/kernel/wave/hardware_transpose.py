# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import torch.fx as fx

from .minimize_global_loads import is_transposed_read, materialize_shape
from .utils.general_utils import get_hardware_constraint

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import Read, get_custom
from ..wave.constraints import Constraint, WorkgroupConstraint, TilingConstraint
from ..wave.utils.run_utils import get_default_arch


def meets_hw_transpose_requirements(read: Read, constraints: list[Constraint]):
    if not get_default_arch() == "gfx950":
        return False

    if read.mapping_dynamic_vals:
        return False

    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, (TilingConstraint, WorkgroupConstraint))
    }

    materialized_shape = materialize_shape(
        constraint_tile_size, read.type.symbolic_shape, read.vector_shapes
    )

    if any(s > 1 for s in materialized_shape[:-2]) or any(
        s <= 1 for s in materialized_shape[-2:]
    ):
        return False

    hardware_constraint = get_hardware_constraint(constraints)
    return hardware_constraint.threads_per_wave >= 16


def mark_hardware_transpose_candidates(
    trace: CapturedTrace, constraints: list[Constraint]
):
    def transpose_wrapper(node: fx.Node) -> bool:
        read = get_custom(node)
        if not isinstance(read, Read) or len(read.type.symbolic_shape) <= 1:
            return False
        return is_transposed_read(read)

    for read in trace.walk(transpose_wrapper):
        if meets_hw_transpose_requirements(get_custom(read), constraints):
            get_custom(read).update_arg("transpose", True)
