# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import Optional

import sympy
import torch.fx as fx

from .._support.tracing import CapturedTrace
from .compile_options import WaveCompileOptions
from .constraints import Constraint, TilingConstraint
from ..ops.wave_ops import (
    get_custom,
    Iterate,
    TensorLoadToLDS,
    SharedMemoryBarrierSignal,
    SharedMemoryBarrierWait,
    Conditional,
)

logger = logging.getLogger(__name__)

# Cluster barrier ID (-3 = cluster barrier)
CLUSTER_BARRIER_ID = -3


def add_cluster_barriers_to_iterate(
    trace: CapturedTrace,
    node: fx.Node,
    multiplier: Optional[int],
    axis_to_induction_var: dict,
):
    """
    Add cluster barriers to an iterate node.

    When multiplier is None: Add barrier signal and wait before the loop.
    When multiplier is set: Add pipelined barriers:
      - barrier_signal and wait before the loop
      - barrier_signal at start of body if current_iteration % multiplier == 0
      - barrier_wait at end of body if current_iteration % multiplier == multiplier - 1

    Args:
        trace: The captured trace
        node: The iterate node
        multiplier: Barrier multiplier for pipelined synchronization
        axis_to_induction_var: Map from axis to induction variable
    """
    graph = node.graph
    custom = get_custom(node)
    location = custom.location

    # Always synchronize before loop
    with graph.inserting_before(node):
        SharedMemoryBarrierSignal(
            barId=CLUSTER_BARRIER_ID, tensor_wait=False
        ).add_to_graph(graph, loc=location)
        SharedMemoryBarrierWait(barId=CLUSTER_BARRIER_ID).add_to_graph(
            graph, loc=location
        )

    if multiplier is None:
        # No barrier inside the loop
        return

    # Pipelined case: Add conditional barriers inside the loop
    logger.debug(f"  Pipelined barriers (multiplier={multiplier}) for {node.name}")

    # Get the subgraph to add barriers inside
    subgraph = trace.get_subgraph(custom.subgraph_name)

    # Get the induction variable for this axis
    induction_var = axis_to_induction_var.get(custom.axis)
    if induction_var is None:
        raise ValueError(
            f"Could not find induction variable for axis {custom.axis} in TilingConstraints"
        )

    # Add conditional barrier_signal at start of body
    first_node = next(
        n for n in subgraph.nodes if n.op not in ["placeholder", "output"]
    )
    with subgraph.inserting_before(first_node):
        condition = sympy.Eq(induction_var % multiplier, 0)

        signal_subgraph_name = f"{custom.subgraph_name}_barrier_signal"
        signal_subgraph = fx.Graph()

        SharedMemoryBarrierSignal(
            barId=CLUSTER_BARRIER_ID, tensor_wait=False
        ).add_to_graph(signal_subgraph, loc=location)

        signal_subgraph.output(None)

        trace.add_subgraph(signal_subgraph_name, signal_subgraph)

        Conditional(condition, signal_subgraph_name, []).add_to_graph(
            subgraph, loc=location
        )

    # Add conditional barrier_wait at end of body
    output_node = next(n for n in subgraph.nodes if n.op == "output")
    with subgraph.inserting_before(output_node):
        condition = sympy.Eq(induction_var % multiplier, multiplier - 1)

        wait_subgraph_name = f"{custom.subgraph_name}_barrier_wait"
        wait_subgraph = fx.Graph()

        SharedMemoryBarrierWait(barId=CLUSTER_BARRIER_ID).add_to_graph(
            wait_subgraph, loc=location
        )
        wait_subgraph.output(None)

        trace.add_subgraph(wait_subgraph_name, wait_subgraph)

        Conditional(condition, wait_subgraph_name, []).add_to_graph(
            subgraph, loc=location
        )


def is_multicast_tensor_load(node: fx.Node) -> bool:
    custom = get_custom(node)
    if not isinstance(custom, TensorLoadToLDS):
        return False
    if not custom.multicast_mask:
        return False

    return True


def add_cluster_barriers(
    trace: CapturedTrace, constraints: list[Constraint], options: WaveCompileOptions
):
    """
    Adds cluster barriers to the graph for cross-workgroup synchronization.
    This pass handles barrier insertion for cluster-level synchronization using
    barId=-3 (cluster barrier).

    Similar to add_shared_memory_barriers but operates at cluster scope across
    multiple workgroups within a cluster.

    Args:
        trace: The captured trace containing the computation graph
        constraints: List of constraints including TilingConstraints
        options: Wave compilation options
    """

    logger.debug("adding cluster barriers")

    # Build map from axis to induction variable from TilingConstraints
    axis_to_induction_var = {}
    for constraint in constraints:
        if isinstance(constraint, TilingConstraint):
            axis_to_induction_var[constraint.dim] = constraint.induction_var

    # Look for iterate ops and check if they contain tensor load ops with multicast mask
    iterate_nodes = trace.walk(lambda node: isinstance(get_custom(node), Iterate))

    delay = options.cluster_barrier_delay
    if delay is not None and delay < 0:
        raise ValueError(f"Invalid cluster barrier delay: {delay}. Must be >= 0")

    multiplier = None if delay is None else delay + 1

    for node in iterate_nodes:
        custom = get_custom(node)
        subgraph = trace.get_subgraph(custom.subgraph_name)

        if not any(is_multicast_tensor_load(n) for n in subgraph.nodes):
            continue

        logger.debug(f"Adding cluster barriers to {node.name}")

        add_cluster_barriers_to_iterate(trace, node, multiplier, axis_to_induction_var)
