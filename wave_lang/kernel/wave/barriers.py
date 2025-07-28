# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum, auto
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import SHARED_ADDRESS_SPACE
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
from .utils.graph_utils import (
    is_barrier_between,
    is_reduction_subgraph,
    propagate_placeholders,
)


class MemoryAccessType(Enum):
    """Enum to classify memory access operations."""

    NONE = auto()
    READ = auto()
    WRITE = auto()
    READ_WRITE = auto()


def is_shared_memory_op(node: CustomOp) -> Optional[fx.Node]:
    if (
        isinstance(node, (Read, Write, AtomicOp))
        and node.memory_type.address_space == SHARED_ADDRESS_SPACE
    ):
        return propagate_placeholders(node.memory)
    elif isinstance(node, GatherToLDS):
        return propagate_placeholders(node.dst)

    return None


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


@dataclass
class SharedMemoryBarrierInfo:
    is_async: bool = False
    last_node: Optional[fx.Node] = None


def add_shared_memory_barriers(
    trace: CapturedTrace,
    graph: Optional[fx.Graph] = None,
    info: Optional[dict[fx.Node, SharedMemoryBarrierInfo]] = None,
    checking_next_iter: Optional[bool] = False,
):
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

    if info is None:
        info = defaultdict(SharedMemoryBarrierInfo)

    for node in graph.nodes:
        custom = get_custom(node)
        if mem := is_shared_memory_op(custom):
            state = info[mem]
            if state.last_node and need_barrier(custom, state.last_node):
                if barrier := is_barrier_between(
                    state.last_node.fx_node, custom.fx_node
                ):
                    barrier = get_custom(barrier)
                    # Promote the barrier to wait for async ops
                    if state.is_async and not barrier.wait_async_ops:
                        barrier.update_arg("wait_async_ops", True)
                else:
                    # Synchronize after the write to shared memory before we read from it.
                    with graph.inserting_before(node):
                        SharedMemoryBarrier(wait_async_ops=state.is_async).add_to_graph(
                            graph
                        )

                state.is_async = False

            state.last_node = custom
            if isinstance(custom, GatherToLDS):
                state.is_async = True

        if isinstance(custom, NestedRegionOp):
            add_shared_memory_barriers(
                trace, trace.get_subgraph(custom.subgraph_name), info
            )

    # Synchronize before the write to shared memory to avoid stepping over
    # shared reads in the previous iteration of a loop.
    if is_reduction_subgraph(graph) and info and not checking_next_iter:
        # Add barriers between ops from different iterations in the same loop.
        add_shared_memory_barriers(trace, graph, info, checking_next_iter=True)
