# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sympy import Integer
from torch import fx

from wave_lang.kernel._support.tracing import CapturedTrace

from ..ops.wave_ops import (
    Iterate,
    Output,
    Placeholder,
    GatherToLDS,
    get_custom,
)
from .constraints import Constraint
from .utils.general_utils import get_tiling_constraint
from .utils.symbol_utils import subs_idxc, get_induction_symbol


def remap_iter_args(
    iter_args: list[fx.Node], output: Output, value_use_map: dict[fx.Node, fx.Node]
) -> None:
    """
    Add a mapping of iter_args to the output return values in order to keep track
    of how uses need to be updated for unrolled iterations.
    """
    for iter_arg, return_val in zip(iter_args, output.yielded_values):
        value_use_map[iter_arg] = return_val


def update_index_for_unroll(index: dict, induction_var, unroll_offset: int) -> dict:
    """
    Update an index dict by substituting the induction variable with
    induction_var + unroll_offset.

    This is used to update indexing for unrolled iterations.
    """
    if index is None:
        return None
    updated_index = {}
    for key, dim in index.items():
        updated_index[key] = dim.subs({induction_var: induction_var + unroll_offset})
    return updated_index


def unroll(
    iterate: Iterate,
    unroll_factor: int,
    trace: CapturedTrace,
    constraints: list[Constraint],
) -> None:
    """
    Unroll an iterate node in the graph `unroll_factor` times.

    This is done by creating `unroll_factor` - 1 copies of the iteration body
    and adjusting the step size and boundaries accordingly. The original output
    node is retained until the unrolling concludes and then deleted.
    """
    assert unroll_factor > 1, "Unroll factor must be greater than 1"

    if iterate.count is None:
        # This is required if the upper bound has not yet been statically
        # determined from the constraints, e.g. when unrolling is used before the scheduling pass.
        tiling_constraint = get_tiling_constraint(iterate, constraints)
        iterate.count = subs_idxc(tiling_constraint.count)
        assert isinstance(
            iterate.count, int | Integer
        ), "Iteration count must be a statically determinable integer"
    if iterate.count / unroll_factor < 1:
        raise ValueError("Unroll factor is too large for the iteration count.")
    if iterate.count % unroll_factor != 0:
        raise ValueError("Unroll factor must divide the iteration count evenly.")
    if iterate.condition is not None:
        raise ValueError("Unrolling is not supported for iterates with conditions.")

    # Only update the step, not the count. The scf.for loop uses:
    #   scf.for %i = 0 to count step step_val
    # With unrolling, we process `unroll_factor` iterations worth of work per
    # loop iteration. To maintain the correct number of total iterations:
    # - Original: count iterations with step=1
    # - After unroll: count iterations with step=unroll_factor
    #   This gives count/unroll_factor actual loop iterations, each doing
    #   unroll_factor iterations worth of work.
    iterate.update_arg("step", iterate.step * unroll_factor)

    graph = trace.get_subgraph(iterate.subgraph_name)

    # Keep track of the required remappings:
    # Cloned nodes that referred to the iter args need to be mapped to
    # the return values from iterate
    output = get_custom(list(graph.nodes)[-1])
    assert isinstance(output, Output)
    value_use_map: dict[fx.Node, fx.Node] = {}
    remap_iter_args(iterate.iter_args(graph), output, value_use_map)

    def value_mapper(old_arg: fx.Node) -> fx.Node:
        if old_arg in value_use_map:
            return value_use_map[old_arg]
        else:
            return old_arg

    # Iterate `unroll_factor - 1` times over the original body nodes and create
    # copies of them. We only need (unroll_factor-1) new copies because the
    # original body already counts as the first instance, giving us a total of
    # unroll_factor instances when combined.
    #
    # For each copy, we maintain a value_use_map that tracks the mapping from
    # original fx.Nodes to their corresponding copied nodes. This ensures that:
    # 1. When a node refers to a value produced by a previous node, it refers to
    #    the correct copy of that value in the unrolled sequence
    # 2. Each unrolled iteration correctly chains its inputs/outputs with the
    #    previous iteration (using value_mapper to translate references)
    # 3. The final output from the last unrolled copy becomes the new output
    #    of the entire unrolled loop body
    reduction_axis = iterate.axis
    induction_var = get_induction_symbol(reduction_axis)
    original_body_nodes = list(graph.nodes)

    # Mark original nodes as iteration 0 (excluding placeholders)
    for node in original_body_nodes:
        original = get_custom(node)
        if not isinstance(original, Placeholder):
            original.unroll_iteration = 0

    for unroll_idx in range(0, unroll_factor - 1):
        for node in original_body_nodes:
            original = get_custom(node)
            if isinstance(original, Placeholder):
                continue
            copy = original.copy(
                original.name + f"_unrolled{unroll_idx}",
                arg_transform=value_mapper,
                anchor=list(graph.nodes)[-2],
            )
            # Mark this copy with its unroll iteration index
            # unroll_idx=0 means this is the 2nd iteration (iteration 1), etc.
            copy.unroll_iteration = unroll_idx + 1

            # update nodes using the induction_var for indexing
            unroll_offset = unroll_idx + 1

            # Handle GatherToLDS which has separate src_index/dst_index
            # Use update_arg to properly update both the dataclass field AND the fx.Node args
            if isinstance(copy, GatherToLDS):
                if copy.src_index:
                    updated_src = update_index_for_unroll(
                        copy.src_index, induction_var, unroll_offset
                    )
                    copy.update_arg("src_index", updated_src)
                if copy.dst_index:
                    updated_dst = update_index_for_unroll(
                        copy.dst_index, induction_var, unroll_offset
                    )
                    copy.update_arg("dst_index", updated_dst)

            # Standard index update for other operations
            if copy.index:
                copy.index = update_index_for_unroll(
                    copy.index, induction_var, unroll_offset
                )
            value_use_map[original.fx_node] = copy.fx_node

            if isinstance(copy, Output):
                remap_iter_args(iterate.iter_args(graph), copy, value_use_map)
                # At this point we have two output nodes in the graph.
                # We erase the original output node when unrolling is complete.
                # Otherwise, we erase the copy.
                if unroll_idx != unroll_factor - 2:
                    get_custom(value_use_map[node]).erase()
                else:
                    original.erase()
                value_use_map[node] = copy.fx_node
