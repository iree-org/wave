# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import Optional

import torch.fx as fx

from .._support.tracing import CapturedTrace
from .compile_options import WaveCompileOptions
from ..ops.wave_ops import (
    get_custom,
    Iterate,
    TensorLoadToLDS,
    SharedMemoryBarrierSignal,
    SharedMemoryBarrierWait,
)

logger = logging.getLogger(__name__)

# Cluster barrier ID (-3 = cluster barrier)
CLUSTER_BARRIER_ID = -3


def add_cluster_barriers_to_iterate(
    trace: CapturedTrace, node: fx.Node, multiplier: Optional[int]
):
    """
    Add cluster barriers to an iterate node.

    When multiplier is None: Add barrier signal and wait before the loop.
    When multiplier is set: Add pipelined barriers:
      - barrier_signal before the loop
      - barrier_wait at start of body if current_iteration % multiplier == 0
      - barrier_signal at end of body if (current_iteration + 1) % multiplier == 0
      - barrier_wait after the loop

    Args:
        trace: The captured trace
        node: The iterate node
        multiplier: Barrier multiplier for pipelined synchronization
    """
    graph = node.graph
    custom = get_custom(node)
    location = custom.location

    if multiplier is None:
        # Simple case: Add barriers before the loop
        with graph.inserting_before(node):
            SharedMemoryBarrierSignal(
                barId=CLUSTER_BARRIER_ID, tensor_wait=False
            ).add_to_graph(graph, loc=location)
            logger.debug(f"  Added cluster barrier signal before {node.name}")

            SharedMemoryBarrierWait(barId=CLUSTER_BARRIER_ID).add_to_graph(
                graph, loc=location
            )
            logger.debug(f"  Added cluster barrier wait before {node.name}")
    else:
        # Pipelined case: Add conditional barriers inside the loop
        logger.debug(
            f"  Adding pipelined cluster barriers with multiplier={multiplier}"
        )

        # Add barrier_signal before the loop
        with graph.inserting_before(node):
            SharedMemoryBarrierSignal(
                barId=CLUSTER_BARRIER_ID, tensor_wait=False
            ).add_to_graph(graph, loc=location)
            logger.debug(f"  Added cluster barrier signal before {node.name}")

        # Get the subgraph to add barriers inside
        subgraph = trace.get_subgraph(custom.subgraph_name)

        # TODO: Add conditional barrier_wait at start of body
        # TODO: Add conditional barrier_signal at end of body

        # Add barrier_wait after the loop
        with graph.inserting_after(node):
            SharedMemoryBarrierWait(barId=CLUSTER_BARRIER_ID).add_to_graph(
                graph, loc=location
            )
            logger.debug(f"  Added cluster barrier wait after {node.name}")


def add_cluster_memory_barriers(trace: CapturedTrace, options: WaveCompileOptions):
    """
    Adds cluster memory barriers to the graph for cross-workgroup synchronization.
    This pass handles barrier insertion for cluster-level synchronization using
    barId=-3 (cluster barrier).

    Similar to add_shared_memory_barriers but operates at cluster scope across
    multiple workgroups within a cluster.

    Args:
        trace: The captured trace containing the computation graph
        options: Wave compilation options
    """

    logger.debug("Running add_cluster_memory_barriers pass")
    logger.debug(f"  cluster_barrier_multiplier={options.cluster_barrier_multiplier}")

    # Step 1: Look for iterate ops and check if they contain tensor load ops
    iterate_nodes = trace.walk(lambda node: isinstance(get_custom(node), Iterate))

    for node in iterate_nodes:
        custom = get_custom(node)
        logger.debug(f"Found iterate op: {node.name}")

        # Get the subgraph for this iterate op
        subgraph = trace.get_subgraph(custom.subgraph_name)

        # Check if subgraph contains TensorLoadToLDS ops
        tensor_load_nodes = [
            n for n in subgraph.nodes if isinstance(get_custom(n), TensorLoadToLDS)
        ]

        if tensor_load_nodes:
            logger.debug(
                f"  Iterate op {node.name} contains {len(tensor_load_nodes)} tensor load op(s)"
            )
            add_cluster_barriers_to_iterate(
                trace, node, options.cluster_barrier_multiplier
            )
        else:
            logger.debug(f"  Iterate op {node.name} does not contain tensor load ops")
