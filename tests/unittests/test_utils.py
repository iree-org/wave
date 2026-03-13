# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx

from wave_lang.kernel._support.dtype import DataType
from wave_lang.kernel.ops.wave_ops import NewScalar


def add_test_node(graph: fx.Graph, name: str) -> fx.Node:
    """
    Add a test node to the graph.

    Args:
        graph: The fx.Graph to add the node to
        name: A name/identifier for the node (used as the value for NewScalar)

    Returns:
        The created fx.Node
    """
    # Create a NewScalar node with a unique float value based on name hash
    # This ensures each node has a distinct value while being deterministic
    value = float(hash(name) % (2**31))
    node = NewScalar(value=value, dtype=DataType("f32"))
    node.add_to_graph(graph)
    return node.fx_node
