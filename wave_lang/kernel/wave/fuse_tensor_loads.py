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


def find_adjacent_loads(
    trace: CapturedTrace,
) -> list[tuple[fx.Node, fx.Node]]:
    """
    Find pairs of adjacent TensorLoadToLDS operations that can be fused.

    Fusion only occurs within the same subgraph to maintain correct program semantics.
    Pairs are fusable if:
    - They are in the same subgraph
    - They have the same element type
    - There are no side-effecting ops between them
    - Users of the first op are not between them
    - Each load appears in at most one pair

    Args:
        trace: The captured trace to analyze

    Returns:
        List of pairs (node1, node2) of adjacent loads that can be fused
    """
    fusable_pairs = []

    # Collect TensorLoadToLDS nodes by subgraph
    for subgraph_name, subgraph in trace.region_graph.subgraphs.items():
        loads = [
            node
            for node in subgraph.nodes
            if isinstance(get_custom(node), TensorLoadToLDS)
        ]

        if not loads:
            continue

        logger.info(f"Found {len(loads)} tensor loads in subgraph '{subgraph_name}'")

        # Track which loads have already been paired to avoid duplicates
        paired_loads = set()

        # Check each pair of loads for fusability
        for i in range(len(loads)):
            load1 = loads[i]

            # Skip if this load is already in a pair
            if load1 in paired_loads:
                continue

            for j in range(i + 1, len(loads)):
                load2 = loads[j]

                # Skip if this load is already in a pair
                if load2 in paired_loads:
                    continue

                if not (load1 < load2):
                    continue

                # Check if both loads have the same element type
                load1_custom = get_custom(load1)
                load2_custom = get_custom(load2)
                if load1_custom.element_type != load2_custom.element_type:
                    logger.debug(
                        f"Cannot fuse {load1.name} and {load2.name}: "
                        f"different element types ({load1_custom.element_type} vs {load2_custom.element_type})"
                    )
                    continue

                # Check for side-effecting ops between the loads
                has_side_effects_between = any(
                    get_custom(node).has_side_effects
                    for node in subgraph.nodes
                    if load1 < node < load2
                )
                if has_side_effects_between:
                    logger.debug(
                        f"Cannot fuse {load1.name} and {load2.name}: side-effecting ops in between"
                    )
                    continue

                # Check if users of first load are between the two loads
                has_users_between = any(load1 < user < load2 for user in load1.users)
                if has_users_between:
                    logger.debug(
                        f"Cannot fuse {load1.name} and {load2.name}: users of first load in between"
                    )
                    continue

                # This pair is fusable - add it and mark both loads as paired
                fusable_pairs.append((load1, load2))
                paired_loads.add(load1)
                paired_loads.add(load2)
                logger.debug(f"Fusable pair: {load1.name} and {load2.name}")
                # Break to avoid pairing load1 with multiple loads
                break

    logger.info(f"Identified {len(fusable_pairs)} fusable pairs")
    return fusable_pairs


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
    fusable_pairs = find_adjacent_loads(trace)

    if not fusable_pairs:
        logger.info("No fusable tensor load pairs found")
        return

    # TODO: Implement actual fusion logic
    # For each pair in fusable_pairs:
    # - Merge the two TensorLoadToLDS operations
    # - Update distributed shapes and bounds
    # - Replace uses and remove old nodes
    # - Run DCE to clean up

    logger.info(
        f"Found {len(fusable_pairs)} pairs ready for fusion (fusion not yet implemented)"
    )
