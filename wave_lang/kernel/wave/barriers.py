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



# --- unused func (for reference) ---
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
