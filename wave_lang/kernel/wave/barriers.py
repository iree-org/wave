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
    Iterate,
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
        isinstance(node, (Read, Write))
        and node.memory_type.address_space == SHARED_ADDRESS_SPACE
    ):
        return propagate_loop_carried_vars(node.memory, depth)
    if (
        isinstance(node, AtomicOp)
        and node.memory_type.address_space == SHARED_ADDRESS_SPACE
    ):
        return propagate_loop_carried_vars(node.rhs, depth)
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


def add_shared_memory_barriers(
    trace: CapturedTrace,
    graph: Optional[fx.Graph] = None,
    info: Optional[dict[fx.Node, SharedMemoryBarrierInfo]] = None,
    checking_next_iter: Optional[bool] = False,
    target: str = "",
    last_producer: dict = None,
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

    # a map with key: barId, value: fx.Node to keep track of last node to signal
    if last_producer is None:
        last_producer = defaultdict()

    for node in graph.nodes:
        custom = get_custom(node)
        depth = 1 if checking_next_iter else 0
        if mem := is_shared_memory_op(custom, depth):
            state = info[mem]

            barId = -1  # TODO named_bars.get(get_memory_access_type(node), -1)
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
                        consumer = node
                        producer = last_producer.get(barId)
                        assert (
                            consumer and producer
                        ), "Bug: Consumer node and producer node should never be None."

                        has_root_dependency = producer.graph != consumer.graph

                        # root dependency will be handled in separate pass: add_signal_prolog_wait_epilog_to_graph
                        if not has_root_dependency:
                            add_shared_memory_split_barriers(
                                producer, consumer, barId, state.is_async
                            )
                    else:
                        with graph.inserting_before(node):
                            barrier_node = SharedMemoryBarrier(
                                wait_async_ops=state.is_async,
                            ).add_to_graph(graph, loc=custom.location)

                state.is_async = False

            state.last_node = custom
            last_producer.update({barId: node})

            if isinstance(custom, GatherToLDS):
                state.is_async = True

        if isinstance(custom, NestedRegionOp):
            add_shared_memory_barriers(
                trace,
                trace.get_subgraph(custom.subgraph_name),
                info,
                target=target,
                last_producer=last_producer,
            )
            add_signal_prolog_wait_epilog_to_graph(trace, graph, custom)

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
            last_producer=last_producer,
        )


def add_shared_memory_split_barriers(
    producer: fx.Node, consumer: fx.Node, barId: int = -1, is_async: bool = False
):
    """
    certain scenarios to consider if split barriers are enabled
    1. root graph, dependencies are straightforward,
       the algorithm iterate to the first node to wait,
       and we keep track of the last node to signal.
       -> signal after last node tracked, wait before the current node.
    2. subgraph, dependencies can produced from root graph, reduction graph
       should also be taken care of.

        1) no dependency from root and not a reduction graph
          -> works like 1.
        2) dependency from root, not a reduction graph
          -> insert signal node in the root graph
          -> wait before current node.
        3) no root dependencies, is a reduction graph
          -> insert signal and write at the end of state.last_node
        4) dependency from root, is a reduction graph
          -> insert signal in the root graph
          -> wait at the root_depend_node
          -> insert signal at the next iterate last_producer node
          -> insert wait at the end of iterate
    These scenarios can be generalized to patterns like this:

    <wait>
    (nodes)
    <signal>

    for circular dependencies introduced by reduction graphs, it will be handled by add_signal_prolog_wait_epilog_to_graph pass.
    """

    if producer:
        signal_node = move_to_valid(producer)
        signal_graph = producer.graph
        with signal_graph.inserting_after(signal_node):
            _ = SharedMemoryBarrierSignal(barId, wait_async_ops=is_async).add_to_graph(
                signal_graph, loc=get_custom(signal_node).location
            )

    if consumer:
        wait_node = move_to_valid(consumer)
        wait_graph = consumer.graph
        with wait_graph.inserting_before(wait_node):
            _ = SharedMemoryBarrierWait(barId).add_to_graph(
                wait_graph, loc=get_custom(wait_node).location
            )

    return signal_graph != wait_graph


def add_signal_prolog_wait_epilog_to_graph(trace, graph, custom):
    """
    Pattern: custom iterate node and wait node is before signal

    This pass insert signal and wait barrier based on root_dependency.
    Specifically, if has_root_dependency is set, a signal should already be inserted by
    `add_shared_memory_barriers` pass, we only need to insert a wait at the epilog.
    If has_root_dependency is not set but it is an iterate subgraph, that means producers
    and consumers are both inside the subgraph, we insert signal at prolog and wait at epilog
    to deal with circular dependency.

    [root]
    ...
    <signal>
        [subgraph]
        ...
        [end subgraph]
    <wait>
    ...
    [end root]
    """

    if not isinstance(custom, Iterate):
        return

    subgraph = trace.get_subgraph(custom.subgraph_name)
    if all_signals_before_waits(subgraph):
        return

    producer = custom.fx_node.prev
    consumer = custom.fx_node.next

    same_graph = add_shared_memory_split_barriers(producer, consumer)
    assert (
        same_graph is False
    ), "prolog and epilog should be inserted in the same graph."


def all_signals_before_waits(graph):
    """
    For difference scheduling such as Prefetch / Modulo, LR and LW may appear at prolog or epilog of a subgraph.
    This function checks if there are waits before any signals.
    Granuarity of this function is a graph (subgraphs should already be handled.)
    """

    signals = defaultdict(bool)  # barId : signal exist
    lonely_waits = set()

    for node in graph.nodes:
        custom = get_custom(node)
        if isinstance(custom, SharedMemoryBarrierSignal):
            assert (
                signals[custom.barId] is False
            ), "Bug: signal the same barId twice before any watis."
            signals.update({custom.barId: True})
        if isinstance(custom, SharedMemoryBarrierWait):
            if not signals[custom.barId]:
                lonely_waits.add(custom)
            else:
                signals.update({custom.barId: False})

    assert (
        len(lonely_waits) <= 1
    ), "Wait barrier appear more than once before any signals, this is a serious bug."
    assert (
        sum(signals.values()) <= 1
    ), "Signals are not consumed by waits for more than twice"

    return len(lonely_waits) == 0
