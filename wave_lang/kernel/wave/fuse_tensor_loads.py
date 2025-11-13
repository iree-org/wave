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

import sympy
import torch.fx as fx

from .._support.indexing import IndexSequence
from .._support.tracing import CapturedTrace
from ..lang.global_symbols import INPUT_SELECTOR, THREAD_0, THREAD_1, THREAD_2
from ..ops.wave_ops import (
    TensorLoadToLDS,
    get_custom,
)
from ..wave.constraints import Constraint, HardwareConstraint
from ..wave.utils.general_utils import get_hardware_constraint
from ..wave.utils.graph_utils import DCE
from ..wave.utils.print_utils import print_trace

logger = logging.getLogger(__name__)


def merge_with_piecewise(value1, value2, selector_symbol):
    """
    Merge two values using sympy.Piecewise to select based on selector_symbol.

    Args:
        value1: Value to use when selector is 0 (even waves)
        value2: Value to use when selector is 1 (odd waves)
        selector_symbol: Symbol to use for selection (INPUT_SELECTOR)

    Returns:
        Piecewise expression or original value if they're identical
    """
    # If values are identical, no need for Piecewise
    if value1 == value2:
        return value1

    # For IndexSequence, merge start, size, and stride separately
    if isinstance(value1, IndexSequence) and isinstance(value2, IndexSequence):
        start = sympy.Piecewise(
            (value1.start, sympy.Eq(selector_symbol, 0)), (value2.start, True)
        )
        size = sympy.Piecewise(
            (value1.size, sympy.Eq(selector_symbol, 0)), (value2.size, True)
        )
        stride = sympy.Piecewise(
            (value1.stride, sympy.Eq(selector_symbol, 0)), (value2.stride, True)
        )
        return IndexSequence(start, size, stride)

    # For scalar values, create Piecewise expression
    return sympy.Piecewise((value1, sympy.Eq(selector_symbol, 0)), (value2, True))


def merge_dicts_with_piecewise(dict1, dict2, selector_symbol):
    """
    Merge two dictionaries using Piecewise for differing values.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        selector_symbol: Symbol to use for selection (INPUT_SELECTOR)

    Returns:
        Merged dictionary with Piecewise expressions for differing values.
        Keys present in only one dict are included as-is without Piecewise.
    """
    result = {}
    # Sort keys for stable iteration order
    all_keys = sorted(set(dict1.keys()) | set(dict2.keys()), key=str)

    for key in all_keys:
        # Key only in dict1 - use value as-is
        if key not in dict2:
            result[key] = dict1[key]
        # Key only in dict2 - use value as-is
        elif key not in dict1:
            result[key] = dict2[key]
        # Key in both - merge with Piecewise
        else:
            result[key] = merge_with_piecewise(dict1[key], dict2[key], selector_symbol)

    return result


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

    # Get hardware constraints for wave calculation
    hardware_constraint = get_hardware_constraint(constraints)
    threads_per_wave = hardware_constraint.threads_per_wave
    waves_per_block = hardware_constraint.waves_per_block

    # Calculate wave_id in each dimension
    wave_id_0 = THREAD_0 // threads_per_wave
    wave_id_1 = THREAD_1 if waves_per_block[1] > 1 else 0
    wave_id_2 = THREAD_2 if waves_per_block[2] > 1 else 0

    # Linearize wave_id: linear_id = id_0 + id_1 * dim_0 + id_2 * dim_0 * dim_1
    wave_id = (
        wave_id_0
        + wave_id_1 * waves_per_block[0]
        + wave_id_2 * waves_per_block[0] * waves_per_block[1]
    )

    # input_selector will be wave_id % 2 (0 for even waves, 1 for odd waves)
    input_selector = wave_id % 2

    logger.info(f"Fusing {len(fusable_pairs)} tensor load pairs")

    # Fuse each pair
    for load1_node, load2_node in fusable_pairs:
        load1 = get_custom(load1_node)
        load2 = get_custom(load2_node)

        logger.info(f"Fusing {load1_node.name} and {load2_node.name}")

        # Merge sources and destinations into lists
        merged_src = load1.src + load2.src
        merged_dst = load1.dst + load2.dst

        # Element type must be the same (already checked in find_adjacent_loads)
        merged_element_type = load1.element_type

        # Identify wave-dependent dimensions for load1
        wave_dependent_dims_load1 = set()
        for dim, idx_seq in load1.global_tile_index.items():
            start_expr = idx_seq.start
            free_symbols = (
                start_expr.free_symbols
                if hasattr(start_expr, "free_symbols")
                else set()
            )
            if any(t in free_symbols for t in [THREAD_0, THREAD_1, THREAD_2]):
                wave_dependent_dims_load1.add(dim)

        # Identify wave-dependent dimensions for load2
        wave_dependent_dims_load2 = set()
        for dim, idx_seq in load2.global_tile_index.items():
            start_expr = idx_seq.start
            free_symbols = (
                start_expr.free_symbols
                if hasattr(start_expr, "free_symbols")
                else set()
            )
            if any(t in free_symbols for t in [THREAD_0, THREAD_1, THREAD_2]):
                wave_dependent_dims_load2.add(dim)

        logger.debug(
            f"Wave-dependent dimensions for {load1_node.name}: {wave_dependent_dims_load1}"
        )
        logger.debug(
            f"Wave-dependent dimensions for {load2_node.name}: {wave_dependent_dims_load2}"
        )

        # Scale distributed_shape by 2 for wave-dependent dimensions only
        # After fusion: even waves execute load1, odd waves execute load2
        # Each wave needs to do 2x the work in wave-dependent dims
        scaled_load1_shape = {
            dim: load1.distributed_shape[dim]
            * (2 if dim in wave_dependent_dims_load1 else 1)
            for dim in load1.distributed_shape.keys()
        }
        scaled_load2_shape = {
            dim: load2.distributed_shape[dim]
            * (2 if dim in wave_dependent_dims_load2 else 1)
            for dim in load2.distributed_shape.keys()
        }

        # Merge distributed_shape using Piecewise
        merged_distributed_shape = merge_dicts_with_piecewise(
            scaled_load1_shape, scaled_load2_shape, INPUT_SELECTOR
        )

        # Merge shared_tile_index using Piecewise
        merged_shared_tile_index = merge_with_piecewise(
            load1.shared_tile_index, load2.shared_tile_index, INPUT_SELECTOR
        )

        # Merge global_tile_index using Piecewise
        merged_global_tile_index = merge_dicts_with_piecewise(
            load1.global_tile_index, load2.global_tile_index, INPUT_SELECTOR
        )

        # Merge bounds using Piecewise
        merged_bounds = merge_dicts_with_piecewise(
            load1.bounds, load2.bounds, INPUT_SELECTOR
        )

        # Create the fused TensorLoadToLDS node
        # Insert it before the first load
        with load2_node.graph.inserting_before(load2_node):
            fused_load = TensorLoadToLDS(
                src=merged_src,
                dst=merged_dst,
                element_type=merged_element_type,
                distributed_shape=merged_distributed_shape,
                shared_tile_index=merged_shared_tile_index,
                global_tile_index=merged_global_tile_index,
                bounds=merged_bounds,
                input_selector=input_selector,
            ).add_to_graph(
                load2_node.graph,
                loc=load2.location,
            )

        # Copy pre_expansion_id from the first load
        if hasattr(load1_node, "pre_expansion_id"):
            fused_load.pre_expansion_id = load1_node.pre_expansion_id

        logger.debug(f"Created fused load: {fused_load.name}")

        # Replace all uses of load1 and load2 with the fused load
        # Note: TensorLoadToLDS operations typically don't have return values,
        # so this mainly updates the graph structure
        load1_node.replace_all_uses_with(fused_load)
        load2_node.replace_all_uses_with(fused_load)

        # Erase the old nodes
        logger.debug(f"Erasing {load1_node.name}")
        get_custom(load1_node).erase()

        logger.debug(f"Erasing {load2_node.name}")
        get_custom(load2_node).erase()

    # Run dead code elimination to clean up
    logger.info("Running DCE after fusion")
    DCE(trace)

    logger.info(f"Successfully fused {len(fusable_pairs)} tensor load pairs")
