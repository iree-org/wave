# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .utils.graph_utils import is_reduction_subgraph, is_barrier_between
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    AtomicOp,
    CustomOp,
    GatherToLDS,
    NestedRegionOp,
    Read,
    SharedMemoryBarrier,
    Write,
    get_custom,
)
from ..lang.global_symbols import SHARED_ADDRESS_SPACE
import torch.fx as fx
from typing import Optional
from enum import Enum, auto


class MemoryAccessType(Enum):
    """Enum to classify memory access operations."""

    NONE = auto()
    READ = auto()
    WRITE = auto()
    READ_WRITE = auto()


def is_shared_memory_op(node: CustomOp) -> bool:
    if isinstance(node, (Read, Write, AtomicOp)):
        return node.memory_type.address_space == SHARED_ADDRESS_SPACE
    elif isinstance(node, GatherToLDS):
        return True

    return False


def get_memory_access_type(node: CustomOp) -> MemoryAccessType:
    if isinstance(node, Read):
        return MemoryAccessType.READ
    elif isinstance(node, Write):
        return MemoryAccessType.WRITE
    elif isinstance(node, AtomicOp):
        return MemoryAccessType.READ_WRITE
    elif isinstance(node, GatherToLDS):
        return MemoryAccessType.WRITE
    else:
        return MemoryAccessType.NONE


def need_barrier(node1: CustomOp, node2: CustomOp) -> bool:
    access_type1 = get_memory_access_type(node1)
    if access_type1 == MemoryAccessType.NONE:
        return False
    access_type2 = get_memory_access_type(node2)
    if access_type2 == MemoryAccessType.NONE:
        return False

    if access_type1 != access_type2:
        return True

    if access_type1 == MemoryAccessType.READ_WRITE:
        return True

    return False


def add_shared_memory_barriers(
    trace: CapturedTrace,
    graph: Optional[fx.Graph] = None,
    last_node: Optional[fx.Node] = None,
    checking_next_iter: Optional[bool] = False,
) -> fx.Node:
    """
    Adds shared memory barriers to the graph. The barriers are inserted
    following a simple heuristic:
    - Read and write operations need a barrier between them.
    So we walk through the graph keeping track of the last read or write,
    and inserting a barrier before the next write or read.
    While sub-optimal, we use this as a baseline to compare more
    sophisticated barrier insertion strategies.
    """
    if not graph:
        graph = trace.get_root_graph()

    for node in graph.nodes:
        custom = get_custom(node)
        if is_shared_memory_op(custom):
            if last_node is None:
                last_node = custom
                continue
            if need_barrier(custom, last_node) and not is_barrier_between(
                last_node.fx_node, custom.fx_node
            ):
                is_async = isinstance(last_node, GatherToLDS)
                # Synchronize after the write to shared memory before we read from it.
                with graph.inserting_before(node):
                    SharedMemoryBarrier(wait_async_ops=is_async).add_to_graph(graph)
            last_node = custom
        if isinstance(custom, NestedRegionOp):
            last_node = add_shared_memory_barriers(
                trace, trace.get_subgraph(custom.subgraph_name), last_node
            )

    # Synchronize before the write to shared memory to avoid stepping over
    # shared reads in the previous iteration of a loop.
    if is_reduction_subgraph(graph) and last_node and not checking_next_iter:
        # Add barriers between ops from different iterations in the same loop.
        add_shared_memory_barriers(trace, graph, last_node, checking_next_iter=True)

    return last_node
