# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx

from ..ops.wave_ops import Read, Write
from .constraints import Constraint
from .utils.general_utils import (
    find_index_bounds,
    get_hardware_constraint,
    is_shared_mem_access,
    remove_global_indexing,
)
from .utils.graph_utils import get_custom
from .utils.symbol_utils import safe_subs, subs_idxc
from .wave import CapturedTrace


def generate_bounds_exprs(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This pass generates bounds expressions for read and write ops.

    Bounds are used during MLIR lowering to handle partial access.
    """
    hardware_constraint = get_hardware_constraint(constraints)

    def is_read_write(node: fx.Node):
        return isinstance(get_custom(node), (Read, Write))

    nodes = trace.walk(is_read_write)
    for node in nodes:
        node = get_custom(node)
        if node.bounds is not None:
            continue

        vector_shapes = node.vector_shapes or hardware_constraint.vector_shapes
        is_shared_mem = is_shared_mem_access(node)
        bounds = find_index_bounds(
            constraints, node.index, vector_shapes, node.type.symbolic_shape
        )
        if is_shared_mem and bounds:
            bounds = remove_global_indexing(bounds, constraints)
            # Masking against global bounds was already handled when reading from
            # global mem, but we still need to handle masking against vector
            # size during shared mem access.
            unpadded_dims = get_custom(node.memory).get_unpadded_dims
            bounds = {
                k: (safe_subs(v, {k: unpadded_dims[k]})) for k, v in bounds.items()
            }
            # Shared mem accesses always access the full vector_shape tile,
            # so we can remove bounds that are divisible by vector size.
            bounds = {
                k: v
                for k, v in bounds.items()
                if subs_idxc(v % (vector_shapes[k] or 1)) != 0
            }

        if not bounds:
            continue

        node.update_arg("bounds", bounds)
