# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Sequence

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    GatherToLDS,
    NestedRegionOp,
    Iterate,
    Conditional,
    SharedMemoryBarrier,
    SharedMemoryBarrierSignal,
    SharedMemoryBarrierWait,
    get_custom,
)
from .utils.graph_utils import (
    is_barrier_between,
    is_iterate_subgraph,
)

from .utils.barriers_utils import (
    need_barrier,
    TargetConfig,
    SyncRequirement,
    get_barriers_analysis,
)


class BarrierEmitter:
    def __init__(self, cfg: TargetConfig):
        self.cfg = cfg

    def dispatch(self):
        if not self.cfg.has_split_barriers:
            return LegacyBarrierEmitter(self.cfg)
        elif not self.cfg.has_named_barriers:
            return BasicSplitBarrierEmitter(self.cfg)
        else:
            assert True, "Should not reach here for now"

    def emit(self, sync_reqs: Sequence[SyncRequirement]) -> None:
        for req in sync_reqs:
            self.place_barrier(req)

    def place_barrier(self, req: SyncRequirement) -> None:
        raise NotImplementedError


class LegacyBarrierEmitter(BarrierEmitter):
    def place_barrier(self, req: SyncRequirement) -> None:
        is_async_op = lambda node: (
            True if isinstance(get_custom(node), GatherToLDS) else False
        )

        producer = next(iter(req.prod_region))
        consumer = next(iter(req.cons_region))
        barrier = is_barrier_between(producer, consumer)

        if barrier is None:
            with consumer.graph.inserting_before(consumer):
                SharedMemoryBarrier().add_to_graph(
                    consumer.graph, loc=get_custom(consumer).location
                )
        else:
            if is_async_op(producer) or is_async_op(consumer):
                barrier.update_arg("wait_async_ops", True)


class BasicSplitBarrierEmitter(BarrierEmitter):
    def place_barrier(self, req: SyncRequirement) -> None:
        barId = -1
        producer = next(iter(req.prod_region))
        consumer = next(iter(req.cons_region))
        barrier = is_barrier_between(producer, consumer, barId)

        if barrier is None:
            with producer.graph.inserting_after(producer):
                SharedMemoryBarrierSignal(barId).add_to_graph(
                    producer.graph, loc=get_custom(producer).location
                )
            with consumer.graph.inserting_before(consumer):
                SharedMemoryBarrierWait(barId).add_to_graph(
                    consumer.graph, loc=get_custom(consumer).location
                )


def add_shared_memory_barriers(
    trace: CapturedTrace,
    target: str = "",
):
    target_arch = TargetConfig(target)
    graph = trace.get_root_graph()

    sync_requests = get_barriers_analysis(trace, graph)

    handle = BarrierEmitter(target_arch)
    emitter = handle.dispatch()
    emitter.emit(sync_requests)


## ----- split ---- ##
@dataclass
class SharedMemoryBarrierInfo:
    is_async: bool = False
    last_node: Optional[fx.Node] = None


def add_shared_memory_barriers_bc(
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

            # TODO(megan.kuo) Add support for named barriers.
            barId = -1
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

                        # Adding signals and waits in the same graph.
                        # other variants of dependencies will be handled in separate pass: add_signal_wait_to_subgraph
                        if producer.graph == consumer.graph:
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
            # the node items set is later used to compare if producers are updated in the next `add_shared_memory_barriers` recursive call.
            producers = set(last_producer.items())
            add_shared_memory_barriers(
                trace,
                trace.get_subgraph(custom.subgraph_name),
                info,
                target=target,
                last_producer=last_producer,
            )
            producers_in_subgraph = producers != set(last_producer.items())
            if should_insert_split_barrier_for_nested_region_op(
                split_barrier, checking_next_iter, producers_in_subgraph
            ):
                add_signal_wait_to_subgraph(trace, graph, custom)

    # Synchronize before the write to shared memory to avoid stepping over
    # shared reads in the previous iteration of a loop.
    if is_iterate_subgraph(graph) and info and not checking_next_iter:
        # Add barriers between ops from different iterations in the same loop.
        add_shared_memory_barriers(
            trace,
            graph,
            info,
            checking_next_iter=True,
            target=target,
            last_producer=last_producer,
        )


def should_insert_split_barrier_for_nested_region_op(
    split_barrier: bool, checking_next_iter: bool, producers_in_subgraph: bool
):
    # only add signal and wait around a subgraph if
    # 1) it is a reduction graph and we are not checking for next_iterations, or
    # 2) it is a conditional subgraph and a producer is inside the graph.
    return split_barrier and producers_in_subgraph and not checking_next_iter


def add_shared_memory_split_barriers(
    producer: fx.Node, consumer: fx.Node, barId: int = -1, is_async: bool = False
):
    """
    This function adds a signal barrier after a producer and a wait before a consumer with barrier: barId
    for circular dependencies introduced by reduction graphs, it will be handled by add_signal_wait_to_subgraph pass.
    """

    if producer:
        with producer.graph.inserting_after(producer):
            _ = SharedMemoryBarrierSignal(barId, wait_async_ops=is_async).add_to_graph(
                producer.graph, loc=get_custom(producer).location
            )

    if consumer:
        with consumer.graph.inserting_before(consumer):
            _ = SharedMemoryBarrierWait(barId).add_to_graph(
                consumer.graph, loc=get_custom(consumer).location
            )

    return producer.graph != consumer.graph


def add_signal_wait_to_subgraph(trace, graph, custom):
    """
    Pattern: custom NestedRegion nodes + barrier wait appear before barrier signal

    This pass insert signal and wait barrier around entry point and exit point of a subgraph.

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

    if isinstance(custom, Iterate):
        subgraph = trace.get_subgraph(custom.subgraph_name)
        if all_signals_before_waits(subgraph):
            return
        producer = custom.fx_node.prev
        consumer = custom.fx_node.next
        add_shared_memory_split_barriers(producer, consumer)
    elif isinstance(custom, Conditional):
        producer = custom.fx_node.prev
        consumer = custom.fx_node
        add_shared_memory_split_barriers(producer, consumer)
        producer = custom.fx_node
        consumer = custom.fx_node.next
        add_shared_memory_split_barriers(producer, consumer)
    else:
        return


def all_signals_before_waits(graph):
    """
    For difference scheduling such as Prefetch / Modulo, LR and LW may appear at prolog or epilog of a subgraph.
    This function checks if there are waits before any signals.
    Granuarity of this function is a graph (subgraphs are expected to be handled by nested calls.)
    """

    signals = defaultdict(bool)  # barId : signal exist
    lonely_waits = set()

    for node in graph.nodes:
        custom = get_custom(node)
        if isinstance(custom, SharedMemoryBarrierSignal):
            assert (
                signals[custom.barId] is False
            ), "Bug: signal the same barId twice before any waits."
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
    ), "More than one signal exists without corresponding waits; this indicates a serious bug."

    return len(lonely_waits) == 0
