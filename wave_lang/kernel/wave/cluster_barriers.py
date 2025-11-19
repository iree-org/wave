# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    get_custom,
    Iterate,
    TensorLoadToLDS,
)

logger = logging.getLogger(__name__)


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
        else:
            logger.debug(f"  Iterate op {node.name} does not contain tensor load ops")
