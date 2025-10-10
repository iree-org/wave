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
    SharedMemoryBarrierSignal,
    SharedMemoryBarrierWait,
    Write,
    get_custom,
)
from .utils.graph_utils import (
    is_barrier_between,
    is_reduction_subgraph,
    propagate_loop_carried_vars,
)


class MemoryAccessType(Enum):
    """Enum to classify memory access operations."""

    NONE = auto()
    READ = auto()
    WRITE = auto()
    READ_WRITE = auto()


def is_shared_memory_op(node: CustomOp, depth: int) -> Optional[fx.Node]:
    if (
        isinstance(node, (Read, Write, AtomicOp))
        and node.memory_type.address_space == SHARED_ADDRESS_SPACE
    ):
        return propagate_loop_carried_vars(node.memory, depth)
    elif isinstance(node, GatherToLDS):
        return propagate_loop_carried_vars(node.dst, depth)

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


def move_to_valid(node: fx.Node):
    while node and (
        isinstance(node.next, SharedMemoryBarrierSignal)
        or isinstance(node.next, SharedMemoryBarrierWait)
    ):
        node = node.next
    return node


@dataclass
class SharedMemoryBarrierInfo:
    is_async: bool = False
    last_node: Optional[fx.Node] = None
    last_node_type: MemoryAccessType = MemoryAccessType.NONE


def add_shared_memory_barriers(
    trace: CapturedTrace,
    graph: Optional[fx.Graph] = None,
    info: Optional[dict[fx.Node, SharedMemoryBarrierInfo]] = None,
    checking_next_iter: Optional[bool] = False,
    target: str = "",
    parent_graph_bars: dict = dict(),
    last_to_signal: dict = None,
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

    split_barrier = "gfx12" in target

    if not graph:
        graph = trace.get_root_graph()

    if info is None:
        info = defaultdict(SharedMemoryBarrierInfo)

    if last_to_signal is None:
        last_to_signal = defaultdict()

    # WAR: 1, RAW: 2 [insert when meet last node]
    named_bars = {MemoryAccessType.READ: 1, MemoryAccessType.WRITE: 2}
    # a dict with key: barId, value: fx.Node to keep track of last node to signal

    for node in graph.nodes:
        custom = get_custom(node)
        depth = 1 if checking_next_iter else 0
        if mem := is_shared_memory_op(custom, depth):
            state = info[mem]
            node_type = get_memory_access_type(custom)

            barId = -1  # named_bars.get(node_type, -1)
            if state.last_node and need_barrier(custom, state.last_node):

                if barrier := is_barrier_between(
                    state.last_node.fx_node, custom.fx_node, barId
                ):
                    barrier = get_custom(barrier)
                    # Promote the barrier to wait for async ops
                    if (
                        state.is_async
                        and hasattr(barrier, "wait_async_ops")
                        and not barrier.wait_async_ops
                    ):
                        barrier.update_arg("wait_async_ops", True)
                else:
                    # Synchronize after the write to shared memory before we read from it.
                    if split_barrier:
                        # certain scenarios to consider if split barriers are enabled
                        # 1. root graph, dependencies are straightforward,
                        #    the algorithm iterate to the first node to wait,
                        #    and we keep track of the last node to signal.
                        #    -> signal after last node tracked, wait before node iterated.
                        #
                        # 2. subgraph, dependencies can produced from root graph, reduction graph
                        #    should also be taken care of.
                        #
                        #    1) no dependency from root and not a reduction graph
                        #       -> works like 1.
                        #
                        #    2) dependency from root, not a reduction graph
                        #       -> insert signal node in the root graph
                        #       -> wait at node iterated.
                        #
                        #    3) is a reduction graph
                        #       -> insert signal and write at the end of state.last_node
                        barrier_wait_node = None
                        barrier_signal_node = None

                        # avoid waiting at the front of iteration (no signal leads to deadlock)
                        if checking_next_iter:
                            barrier_signal_node = move_to_valid(state.last_node.fx_node)
                            with barrier_signal_node.graph.inserting_after(
                                state.last_node.fx_node
                            ):
                                barrier_signal_node = SharedMemoryBarrierSignal(
                                    barId,
                                    wait_async_ops=state.is_async,
                                ).add_to_graph(graph, loc=custom.location)
                            with barrier_signal_node.graph.inserting_after(
                                barrier_signal_node
                            ):
                                barrier_wait_node = SharedMemoryBarrierWait(
                                    barId
                                ).add_to_graph(
                                    state.last_node.fx_node.graph,
                                    loc=barrier_signal_node.location,
                                )
                        else:
                            barrier_signal_node = move_to_valid(
                                last_to_signal.get(barId)
                            )  # state.last_node.fx_node
                            with graph.inserting_before(node):
                                _ = SharedMemoryBarrierWait(barId).add_to_graph(
                                    graph, loc=custom.location
                                )

                            with barrier_signal_node.graph.inserting_after(
                                barrier_signal_node
                            ):
                                _ = SharedMemoryBarrierSignal(
                                    barId,
                                    wait_async_ops=state.is_async,
                                ).add_to_graph(
                                    barrier_signal_node.graph,
                                    loc=get_custom(barrier_signal_node).location,
                                )
                    else:
                        with graph.inserting_before(node):
                            barrier_node = SharedMemoryBarrier(
                                wait_async_ops=state.is_async,
                            ).add_to_graph(graph, loc=custom.location)

                state.is_async = False

            state.last_node = custom
            state.last_node_type = node_type
            last_to_signal.update({barId: node})

            if isinstance(custom, GatherToLDS):
                state.is_async = True

        if isinstance(custom, NestedRegionOp):
            add_shared_memory_barriers(
                trace,
                trace.get_subgraph(custom.subgraph_name),
                info,
                target=target,
                last_to_signal=last_to_signal,
            )

    # Synchronize before the write to shared memory to avoid stepping over
    # shared reads in the previous iteration of a loop.
    if is_reduction_subgraph(graph) and info and not checking_next_iter:
        # Add barriers between ops from different iterations in the same loop.
        add_shared_memory_barriers(
            trace,
            graph,
            info,
            checking_next_iter=True,
            target=target,
            last_to_signal=last_to_signal,
        )
