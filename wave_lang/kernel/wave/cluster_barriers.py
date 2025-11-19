# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

import torch.fx as fx

from .._support.tracing import CapturedTrace
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


def add_cluster_barriers_to_iterate(node: fx.Node):
    """
    Add cluster barrier signal and wait before an iterate node.

    Args:
        node: The iterate node before which to insert barriers
    """
    graph = node.graph
    custom = get_custom(node)
    location = custom.location

    with graph.inserting_before(node):
        # Add cluster barrier signal
        signal_node = SharedMemoryBarrierSignal(
            barId=CLUSTER_BARRIER_ID, tensor_wait=False
        ).add_to_graph(graph, loc=location)
        logger.debug(f"  Added cluster barrier signal before {node.name}")

        # Add cluster barrier wait
        wait_node = SharedMemoryBarrierWait(barId=CLUSTER_BARRIER_ID).add_to_graph(
            graph, loc=location
        )
        logger.debug(f"  Added cluster barrier wait before {node.name}")


def add_cluster_memory_barriers(trace: CapturedTrace):
    """
    Adds cluster memory barriers to the graph for cross-workgroup synchronization.
    This pass handles barrier insertion for cluster-level synchronization using
    barId=-3 (cluster barrier).

    Similar to add_shared_memory_barriers but operates at cluster scope across
    multiple workgroups within a cluster.

    Args:
        trace: The captured trace containing the computation graph
    """

    logger.debug("Running add_cluster_memory_barriers pass")

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
            add_cluster_barriers_to_iterate(node)
        else:
            logger.debug(f"  Iterate op {node.name} does not contain tensor load ops")
