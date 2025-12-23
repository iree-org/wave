# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
from typing import Callable


def filter_fx_graph(
    graph: fx.Graph, filter: Callable[[fx.Node], bool]
) -> list[fx.Node]:
    """Return a list of nodes in the graph that match the specified predicate."""
    filtered_nodes: list[fx.Node] = []
    for node in graph.nodes:
        if filter(node):
            filtered_nodes.append(node)
    return filtered_nodes
