# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
import logging
import copy
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import BoundsCheck, Read, Write, get_custom


logger = logging.getLogger(__name__)


def generate_bound_checks(trace: CapturedTrace):
    """
    Generate bound checks for read and write ops.

    Bound checks are generated based on the tensor's memory shape.

    If a bounds check fails, an error message is reported (typically via printf)
    and execution is halted.
    """
    logger.info("Generating bound checks")

    def is_read_write(node: fx.Node):
        return isinstance(get_custom(node), (Read, Write))

    nodes = trace.walk(is_read_write)
    for node in nodes:
        logger.info(f"Node: {node}")

        index = node.index
        assert index is not None, f"Index not set for node {node}"
        node = get_custom(node)

        mapping = node.mapping
        if mapping and isinstance(node, Write):
            # Transform write mapping into read mapping, so we don't need to distinguish between read and write
            # during lowerings.
            mapping = copy.copy(mapping)
            mapping.input_mapping, mapping.output_mapping = (
                mapping.output_mapping,
                mapping.input_mapping,
            )

        # TODO: It may not work in all cases, as we still don't have a proper
        # separation between iteration shape and physical shape.
        mem_shape = node.memory_type.symbolic_shape
        bounds = {b: b for b in mem_shape}

        graph = node.graph
        with graph.inserting_before(node.fx_node):
            op = BoundsCheck(
                index,
                bounds,
                mapping,
                node.mapping_dynamic_vals,
                node.bounds,
            ).add_to_graph(graph)
            op.location = node.location
