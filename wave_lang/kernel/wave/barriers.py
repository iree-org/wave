# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import defaultdict
from typing import Sequence


from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    GatherToLDS,
    Iterate,
    Conditional,
    SharedMemoryBarrier,
    SharedMemoryBarrierSignal,
    SharedMemoryBarrierWait,
    get_custom,
)
from .utils.graph_utils import (
    is_barrier_between,
)

from .utils.barriers_utils import (
    TargetConfig,
    SyncRequirement,
    get_barriers_analysis,
    minimize_placement_strategy,
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
        sync_reqs = self.optimize(sync_reqs)
        for req in sync_reqs:
            self.place_barrier(req)

    def place_barrier(self, req: SyncRequirement) -> None:
        raise NotImplementedError

    def optimize(
        self, sync_reqs: Sequence[SyncRequirement]
    ) -> Sequence[SyncRequirement]:
        raise NotImplementedError


class LegacyBarrierEmitter(BarrierEmitter):
    def optimize(
        self, sync_reqs: Sequence[SyncRequirement]
    ) -> Sequence[SyncRequirement]:
        return minimize_placement_strategy(sync_reqs)

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
            barrierOp = get_custom(barrier)
            if is_async_op(producer) or is_async_op(consumer):
                barrierOp.update_arg("wait_async_ops", True)


class BasicSplitBarrierEmitter(BarrierEmitter):
    def optimize(
        self, sync_reqs: Sequence[SyncRequirement]
    ) -> Sequence[SyncRequirement]:
        return minimize_placement_strategy(sync_reqs)

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

