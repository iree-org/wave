# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Pass to fuse adjacent TensorLoadToLDS operations.

This pass identifies consecutive TensorLoadToLDS operations that can be combined
into a single optimized load operation to reduce memory access overhead and
improve performance.

Fusion candidates are identified based on:
- Spatial locality: loads accessing adjacent memory regions
- Temporal locality: loads occurring close together in the execution order
- Memory alignment: loads with compatible alignment constraints
- Shared memory layout: loads targeting adjacent LDS regions
"""

import logging
import math

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    TensorLoadToLDS,
    get_custom,
)
from ..wave.constraints import Constraint, HardwareConstraint
from ..wave.utils.graph_utils import DCE
from ..wave.utils.print_utils import print_trace

logger = logging.getLogger(__name__)


def get_wave_count(constraints: list[Constraint]) -> int:
    """
    Extract the total number of waves from the hardware constraints.

    Args:
        constraints: List of constraints for the kernel

    Returns:
        Total number of waves (product of waves_per_block dimensions)
    """
    for constraint in constraints:
        if isinstance(constraint, HardwareConstraint):
            if constraint.waves_per_block is not None:
                # Calculate total wave count as product of all dimensions
                return math.prod(constraint.waves_per_block)
    return 0


def is_fusable_tensor_load(node: fx.Node) -> bool:
    """
    Check if a node is a TensorLoadToLDS operation that can be fused.

    Args:
        node: The node to check

    Returns:
        True if the node is a fusable TensorLoadToLDS operation
    """
    custom = get_custom(node)
    if not isinstance(custom, TensorLoadToLDS):
        return False

    # TODO: Add additional checks for fusability
    # - Check memory alignment
    # - Check load size constraints
    # - Check hardware capabilities

    return True


def has_side_effecting_ops_between(
    subgraph: fx.Graph, load1_node: fx.Node, load2_node: fx.Node
) -> bool:
    """
    Check if there are any side-effecting operations between two nodes.

    Args:
        subgraph: The subgraph containing the nodes
        load1_node: The first load node (earlier in topological order)
        load2_node: The second load node (later in topological order)

    Returns:
        True if there are side-effecting ops between the two nodes
    """
    for node in subgraph.nodes:
        # Check if node is between load1_node and load2_node
        if load1_node < node < load2_node:
            custom = get_custom(node)
            if custom.has_side_effects:
                logger.debug(
                    f"Found side-effecting op {node.name} ({type(custom).__name__}) "
                    f"between {load1_node.name} and {load2_node.name}"
                )
                return True
    return False


def has_first_load_users_between(
    subgraph: fx.Graph, load1_node: fx.Node, load2_node: fx.Node
) -> bool:
    """
    Check if any users of the first load are between the two loads.

    Args:
        subgraph: The subgraph containing the nodes
        load1_node: The first load node (earlier in topological order)
        load2_node: The second load node (later in topological order)

    Returns:
        True if any users of load1_node are between the two nodes
    """
    for user in load1_node.users:
        if load1_node < user < load2_node:
            logger.debug(
                f"User {user.name} of {load1_node.name} is between "
                f"{load1_node.name} and {load2_node.name}"
            )
            return True
    return False


def find_adjacent_loads(
    trace: CapturedTrace,
) -> list[tuple[fx.Node, fx.Node]]:
    """
    Find pairs of adjacent TensorLoadToLDS operations that can be fused.

    Fusion only occurs within the same subgraph to maintain correct program semantics.
    Pairs are fusable if:
    - They are in the same subgraph
    - There are no side-effecting ops between them
    - Users of the first op are not between them

    Args:
        trace: The captured trace to analyze

    Returns:
        List of pairs (node1, node2) of adjacent loads that can be fused
    """
    fusable_pairs = []

    # Group tensor loads by subgraph to ensure fusion only happens within same subgraph
    loads_by_subgraph = {}
    for subgraph_name, subgraph in trace.region_graph.subgraphs.items():
        subgraph_loads = []
        for node in subgraph.nodes:
            if is_fusable_tensor_load(node):
                subgraph_loads.append(node)

        if subgraph_loads:
            loads_by_subgraph[subgraph_name] = (subgraph, subgraph_loads)
            logger.info(
                f"Found {len(subgraph_loads)} fusable tensor loads in subgraph '{subgraph_name}'"
            )

    # Find fusable pairs within each subgraph
    for subgraph_name, (subgraph, loads) in loads_by_subgraph.items():
        # Check each consecutive pair of loads
        for i in range(len(loads)):
            for j in range(i + 1, len(loads)):
                load1_node = loads[i]
                load2_node = loads[j]

                # Ensure load1 comes before load2
                if not (load1_node < load2_node):
                    continue

                # Check if there are side-effecting ops between them
                if has_side_effecting_ops_between(subgraph, load1_node, load2_node):
                    logger.debug(
                        f"Cannot fuse {load1_node.name} and {load2_node.name} in '{subgraph_name}': "
                        "side-effecting ops in between"
                    )
                    continue

                # Check if users of first load are between them
                if has_first_load_users_between(subgraph, load1_node, load2_node):
                    logger.debug(
                        f"Cannot fuse {load1_node.name} and {load2_node.name} in '{subgraph_name}': "
                        "users of first load are in between"
                    )
                    continue

                # This pair is fusable
                fusable_pairs.append((load1_node, load2_node))
                logger.debug(
                    f"Fusable pair found in '{subgraph_name}': {load1_node.name} and {load2_node.name}"
                )

    total_loads = sum(len(loads) for _, loads in loads_by_subgraph.values())
    logger.info(
        f"Found {total_loads} total fusable tensor loads across {len(loads_by_subgraph)} subgraphs, "
        f"identified {len(fusable_pairs)} fusable pairs"
    )

    return fusable_pairs


def can_fuse_loads(
    load1: TensorLoadToLDS,
    load2: TensorLoadToLDS,
) -> bool:
    """
    Determine if two TensorLoadToLDS operations can be fused.

    Args:
        load1: First load operation
        load2: Second load operation

    Returns:
        True if the loads can be fused
    """
    # TODO: Implement fusion feasibility checks
    # - Check memory regions are adjacent
    # - Check element types match
    # - Check bounds are compatible
    # - Check no dependencies prevent fusion

    return False


def fuse_load_pair(
    trace: CapturedTrace,
    load1_node: fx.Node,
    load2_node: fx.Node,
    load1: TensorLoadToLDS,
    load2: TensorLoadToLDS,
) -> fx.Node:
    """
    Fuse two TensorLoadToLDS operations into a single operation.

    Args:
        trace: The captured trace
        load1_node: First load node
        load2_node: Second load node
        load1: First load operation
        load2: Second load operation

    Returns:
        The newly created fused load node
    """
    # TODO: Implement fusion logic
    # - Calculate combined memory regions
    # - Create new fused TensorLoadToLDS with merged parameters
    # - Update distributed shapes
    # - Update bounds
    # - Replace original loads with fused load

    logger.info(f"Fusing tensor loads: {load1_node.name} and {load2_node.name}")

    # Placeholder for fused load creation
    # fused_load = TensorLoadToLDS(...)
    # with trace.graph.inserting_before(load1_node):
    #     fused_node = fused_load.add_to_graph(trace.graph)

    return None


def fuse_tensor_loads(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    Fuse adjacent TensorLoadToLDS operations to optimize memory access.

    This pass identifies and combines consecutive tensor loads that access
    adjacent memory regions, reducing the number of memory operations and
    improving overall performance.

    Fusion is only performed when we have an even number of waves, as this
    is a requirement for correct fusion behavior.

    Args:
        trace: The captured trace to transform
        constraints: List of constraints for the kernel
    """
    logger.info("Running fuse_tensor_loads pass")
    print_trace(trace)

    # Check if we have an even number of waves (required for fusion)
    wave_count = get_wave_count(constraints)
    if wave_count == 0:
        logger.info("No wave count found in constraints, skipping fusion")
        return

    if wave_count % 2 != 0:
        logger.info(
            f"Skipping tensor load fusion: odd number of waves ({wave_count}). "
            "Fusion requires an even number of waves."
        )
        return

    logger.info(f"Wave count is {wave_count} (even), proceeding with fusion")

    # Find pairs of adjacent loads that can be fused
    adjacent_pairs = find_adjacent_loads(trace)

    if not adjacent_pairs:
        logger.info("No adjacent tensor loads found to fuse")
        return

    # Attempt to fuse each pair
    fused_count = 0
    for load1_node, load2_node in adjacent_pairs:
        load1 = get_custom(load1_node)
        load2 = get_custom(load2_node)

        if not can_fuse_loads(load1, load2):
            logger.debug(f"Cannot fuse loads: {load1_node.name} and {load2_node.name}")
            continue

        fuse_load_pair(trace, load1_node, load2_node, load1, load2)
        fused_count += 1

    logger.info(f"Fused {fused_count} pairs of tensor loads")

    # Clean up dead code after fusion
    if fused_count > 0:
        DCE(trace)
