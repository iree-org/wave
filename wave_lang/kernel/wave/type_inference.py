# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx

from wave_lang.support.logging import get_logger
from .constraints import Constraint
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import *

logger = get_logger("wave.type_inference")


def infer_types(
    trace: CapturedTrace,
    constraints: Optional[list[Constraint]] = None,
    subgraph: Optional[fx.Graph] = None,
):
    if subgraph:
        all_nodes = subgraph.nodes
    else:
        all_nodes = trace.get_root_graph().nodes
    # Infer and set the types for all nodes in the graph.
    for node in all_nodes:
        custom = get_custom(node)
        if isinstance(custom, NestedRegionOp):
            infer_types(
                trace, constraints, trace.region_graph.subgraphs[custom.subgraph_name]
            )
        custom.infer_type(constraints)
        # Captured placeholders represent outer values, so their type should be
        # inherited from that defining source rather than re-inferred locally.
        captured_source = NestedRegionOp.capture_source(custom.fx_node)
        if captured_source is not custom.fx_node:
            custom.type = captured_source.type
        logger.debug(f"Setting type for {custom.fx_node} = {custom.type}")
