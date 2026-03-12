# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sympy
import torch.fx as fx

from ..ops.wave_ops import Read, Write
from .assumptions import get_divisibility_subs
from .constraints import Constraint, DistributionConstraint, ReorderingConstraint
from .utils.general_utils import (
    find_index_bounds,
    get_hardware_constraint,
    is_shared_mem_access,
    remove_global_indexing,
)
from .utils.graph_utils import get_custom, propagate_loop_carried_vars
from .utils.symbol_utils import (
    IndexExpr,
    IndexSymbol,
    safe_subs,
    simplify,
    subs_idxc,
)
from .wave import CapturedTrace


def _get_max_tile_size(
    dim: IndexSymbol,
    constraints: list[Constraint],
    unpadded_dims: dict[IndexSymbol, IndexExpr],
) -> IndexExpr:
    ret = sympy.sympify(unpadded_dims[dim])
    for constraint in constraints:
        if isinstance(constraint, DistributionConstraint) and constraint.dim == dim:
            ret = sympy.Max(ret, constraint.tile_size)
    return ret


def is_divisible(
    dim: IndexSymbol,
    tile_size: IndexExpr,
    fwd: list[tuple[sympy.Symbol, sympy.Expr]],
) -> bool:
    """Check if dim is provably divisible by tile_size given divisibility subs.

    Substitutes divisibility assumptions (e.g. K -> 256*K_div_256) and checks
    whether ceiling(dim / tile_size) * tile_size simplifies to dim.
    """
    if not fwd:
        return False
    tile = subs_idxc(tile_size)
    work = sympy.ceiling(dim / tile) * tile
    diff = safe_subs(work - dim, fwd)
    return simplify(diff) == 0


def generate_bounds_exprs(
    trace: CapturedTrace,
    constraints: list[Constraint],
    reordering_constraints: list[ReorderingConstraint] = None,
):
    """
    This pass generates bounds expressions for read and write ops.

    Bounds are used during MLIR lowering to handle partial access.
    """
    hardware_constraint = get_hardware_constraint(constraints)
    fwd, _ = get_divisibility_subs(constraints)

    wg_subs = {}
    if reordering_constraints:
        for c in reordering_constraints:
            wg_subs[c.wg_dim] = c.reordered_equation

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
        # Remove bounds where divisibility assumptions prove alignment.
        if bounds and fwd:
            for c in constraints:
                if (
                    isinstance(c, DistributionConstraint)
                    and c.dim in bounds
                    and is_divisible(c.dim, c.tile_size, fwd)
                ):
                    del bounds[c.dim]
            bounds = bounds or None
        if is_shared_mem and bounds:
            bounds = remove_global_indexing(bounds, constraints)
            # Masking against global bounds was already handled when reading from
            # global mem, but we still need to handle masking against vector
            # size during shared mem access.
            memory = propagate_loop_carried_vars(node.memory)
            unpadded_dims = get_custom(memory).unpadded_dims
            bounds = {
                k: (
                    safe_subs(v, {k: _get_max_tile_size(k, constraints, unpadded_dims)})
                )
                for k, v in bounds.items()
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

        if wg_subs:
            bounds = {
                k: (
                    v.subs(wg_subs, simultaneous=True)
                    if isinstance(v, sympy.Expr)
                    else v
                )
                for k, v in bounds.items()
            }

        node.update_arg("bounds", bounds)
