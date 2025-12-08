# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
from typing import Any

import sympy
import torch.fx as fx

from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr
from .._support.tracing import CapturedTrace
from ..lang.global_symbols import INPUT_SELECTOR, THREAD_0, THREAD_1, THREAD_2
from ..ops.wave_ops import (
    TensorLoadToLDS,
    get_custom,
)
from ..wave.constraints import Constraint
from ..wave.compile_options import WaveCompileOptions
from ..wave.utils.general_utils import get_hardware_constraint
from ..wave.utils.graph_utils import DCE
from ..wave.utils.symbol_utils import is_literal, subs_idxc

logger = logging.getLogger(__name__)


def merge_with_piecewise(
    value1: Any, value2: Any, selector_symbol: sympy.Symbol
) -> Any:
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


def merge_dicts_with_piecewise(
    dict1: dict[Any, Any], dict2: dict[Any, Any], selector_symbol: sympy.Symbol
) -> dict[Any, Any]:
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
    all_keys = dict1 | dict2

    for key in all_keys:
        if key not in dict2:
            result[key] = dict1[key]
        elif key not in dict1:
            result[key] = dict2[key]
        else:
            result[key] = merge_with_piecewise(dict1[key], dict2[key], selector_symbol)

    return result


def _scale_distributed_shape(
    load: TensorLoadToLDS, mult: int
) -> dict[IndexSymbol, IndexExpr]:
    """
    Assuming tensor loads are distributed across one of the dimensions starting
    on the leftmost, return the updated distributed shape for the given load and multiplier.
    """
    assert len(load.dst) == 1, "Only one destination memory is supported"
    mem = get_custom(load.dst[0])
    symbolic_shape = mem.type.symbolic_shape
    distributed_shape = [load.distributed_shape[k] for k in symbolic_shape]
    mem_distributed_shape = mem.distributed_shape
    for i, (s1, s2) in enumerate(zip(distributed_shape, mem_distributed_shape)):
        if s1 == s2:
            continue

        if subs_idxc(s2 % s1) != 0:
            raise ValueError(
                f"Destination memory distributed shape must be a multiple of the source distributed shape, got {s2} % {s1} != 0, for {load.name} with memory shape {mem_distributed_shape}"
            )

        distributed_shape[i] = s1 * mult
        return {k: v for k, v in zip(symbolic_shape, distributed_shape)}

    raise ValueError(
        f"Invalid distributed shape: {distributed_shape} for {load.name} with memory shape {mem_distributed_shape}"
    )


def compute_fused_parameters(
    load1: TensorLoadToLDS,
    load2: TensorLoadToLDS,
    threads_per_wave: int,
    waves_per_block: tuple[int, int, int],
) -> tuple[dict[Any, Any], dict[Any, Any], dict[Any, Any], dict[Any, Any], Any]:
    """
    Compute fused parameters for two tensor loads.

    Args:
        load1: First TensorLoadToLDS operation
        load2: Second TensorLoadToLDS operation
        threads_per_wave: Number of threads per wave
        waves_per_block: Number of waves per block

    Returns:
        Tuple of (merged_distributed_shape, merged_shared_tile_index,
                  merged_global_tile_index, merged_bounds, multicast_mask)
    """
    # As we are cutting number of waves doing the particular load in half, we need
    # to scale the distributed shape by 2. The code relies on the fact that the
    # tensor_load_to_shared distributes shape starting from the leftmost dimension.
    scaled_load1_shape = _scale_distributed_shape(load1, 2)
    scaled_load2_shape = _scale_distributed_shape(load2, 2)

    merged_distributed_shape = merge_dicts_with_piecewise(
        scaled_load1_shape, scaled_load2_shape, INPUT_SELECTOR
    )

    # Adjust indices for load2 so odd waves act as even waves
    # After fusion: even waves (0,2,4,...) use load1, odd waves (1,3,5,...) use load2
    # We need odd waves to use even wave indices, so wave 1 acts like wave 0, wave 3 like wave 2, etc.
    if waves_per_block[0] > 1:
        wave_offset_subs = (THREAD_0, THREAD_0 - threads_per_wave)
    elif waves_per_block[1] > 1:
        wave_offset_subs = (THREAD_1, THREAD_1 - 1)
    elif waves_per_block[2] > 1:
        wave_offset_subs = (THREAD_2, THREAD_2 - 1)
    else:
        raise ValueError(f"Invalid waves_per_block: {waves_per_block}")

    adjusted_load2_global_tile_index = {}
    for dim, idx_seq in load2.global_tile_index.items():
        adjusted_start = idx_seq.start.subs(*wave_offset_subs, simultaneous=True)

        adjusted_load2_global_tile_index[dim] = IndexSequence(
            adjusted_start, idx_seq.size, idx_seq.stride
        )

    # Adjust shared_tile_index for load2 if it depends on thread IDs
    adjusted_load2_shared_tile_index = {}
    for dim, idx_seq in load2.shared_tile_index.items():
        adjusted_start = idx_seq.start.subs(*wave_offset_subs, simultaneous=True)

        adjusted_load2_shared_tile_index[dim] = IndexSequence(
            adjusted_start, idx_seq.size, idx_seq.stride
        )

    merged_shared_tile_index = merge_dicts_with_piecewise(
        load1.shared_tile_index, adjusted_load2_shared_tile_index, INPUT_SELECTOR
    )

    merged_global_tile_index = merge_dicts_with_piecewise(
        load1.global_tile_index, adjusted_load2_global_tile_index, INPUT_SELECTOR
    )

    merged_bounds = merge_dicts_with_piecewise(
        load1.bounds, load2.bounds, INPUT_SELECTOR
    )

    merged_multicast_mask = merge_with_piecewise(
        load1.multicast_mask, load2.multicast_mask, INPUT_SELECTOR
    )

    return (
        merged_distributed_shape,
        merged_shared_tile_index,
        merged_global_tile_index,
        merged_bounds,
        merged_multicast_mask,
    )


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
    options: WaveCompileOptions,
) -> None:
    """
    Fuse adjacent TensorLoadToLDS operations to reduce the number of tensor loads.

    Fusion is only performed when we have an even number of waves.

    Example:
        Before fusion (4 waves total):
            load1 = TensorLoadToLDS(...)  # All waves execute
            load2 = TensorLoadToLDS(...)  # All waves execute

        After fusion (4 waves total):
            fused = TensorLoadToLDS(...)  # Even waves (0,2) do load1's work
                                          # Odd waves (1,3) do load2's work
                                          # Uses INPUT_SELECTOR for conditional behavior

        This reduces the total number of load operations from 2 to 1.

    Args:
        trace: The captured trace to transform
        constraints: List of constraints for the kernel
    """
    logger.info("Running fuse_tensor_loads pass")

    # Get hardware constraints for wave calculation
    hardware_constraint = get_hardware_constraint(constraints)
    threads_per_wave = hardware_constraint.threads_per_wave
    waves_per_block = hardware_constraint.waves_per_block
    wave_count = subs_idxc(math.prod(waves_per_block))

    if options.specialize:
        logger.info(
            f"Skipping tensor load fusion: Specialization option is set. "
            "Specialization with fused-tensor loads are not supported yet."
        )
        return

    # Check if we have an even number of waves (required for fusion)
    if (
        not is_literal(wave_count)
        or wave_count < 2
        or any(v > 1 and v % 2 != 0 for v in waves_per_block)
    ):
        logger.info(
            f"Skipping tensor load fusion: odd number of waves ({waves_per_block}). "
            "Fusion requires an even number of waves."
        )
        return

    logger.info(f"Wave count is {wave_count} (even), proceeding with fusion")

    # Find pairs of adjacent loads that can be fused
    fusable_pairs = find_adjacent_loads(trace)

    if not fusable_pairs:
        logger.info("No fusable tensor load pairs found")
        return

    wave_id = (
        hardware_constraint.linearized_thread_id // hardware_constraint.threads_per_wave
    )

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

        (
            merged_distributed_shape,
            merged_shared_tile_index,
            merged_global_tile_index,
            merged_bounds,
            merged_multicast_mask,
        ) = compute_fused_parameters(load1, load2, threads_per_wave, waves_per_block)

        # Create the fused TensorLoadToLDS node
        # Insert it before the second load
        with load2_node.graph.inserting_before(load2_node):
            fused_load = TensorLoadToLDS(
                src=merged_src,
                dst=merged_dst,
                element_type=merged_element_type,
                distributed_shape=merged_distributed_shape,
                shared_tile_index=merged_shared_tile_index,
                global_tile_index=merged_global_tile_index,
                bounds=merged_bounds,
                multicast_mask=merged_multicast_mask,
                input_selector=input_selector,
            ).add_to_graph(
                load2_node.graph,
                loc=load2.location,
            )

        if hasattr(load1_node, "pre_expansion_id"):
            fused_load.pre_expansion_id = load1_node.pre_expansion_id

        logger.debug(f"Created fused load: {fused_load.name}")

        load1_node.replace_all_uses_with(fused_load)
        load2_node.replace_all_uses_with(fused_load)

        # Erase the old nodes
        get_custom(load1_node).erase()
        get_custom(load2_node).erase()

    DCE(trace)

    logger.info(f"Successfully fused {len(fusable_pairs)} tensor load pairs")
