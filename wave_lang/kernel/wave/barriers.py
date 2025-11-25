# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence
from collections import defaultdict

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    GatherToLDS,
    TensorLoadToLDS,
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
    SyncRegion,
    get_barriers_analysis,
    minimize_placement_strategy,
    find_disjoint_interval_strategy,
)


class BarrierEmitter:
    """
    Base class of barrier emitters.
    Derived classes should implement the optimize and emit methods and register handlers for the emitter to enable proper dispatching.
    """

    def __init__(self, cfg: TargetConfig):
        self.cfg = cfg

    def __new__(cls, cfg: TargetConfig):
        """
        Return subclass instance
        """
        if cls is BarrierEmitter:
            if not cfg.has_split_barriers:
                return super().__new__(LegacyBarrierEmitter)
            return super().__new__(BasicSplitBarrierEmitter)
        return super().__new__(cls)

    def emit(self, sync_regions: Sequence[SyncRegion]) -> None:
        """
        Optimizes barrier placement using the derived class's strategy and places the resulting barriers.
        """
        sync_regions = self.optimize(sync_regions)
        for region in sync_regions:
            self.place_barrier(region)

    def place_barrier(self, region: SyncRegion) -> None:
        raise NotImplementedError

    def optimize(self, sync_regions: Sequence[SyncRegion]) -> Sequence[SyncRegion]:
        raise NotImplementedError

    def verify(self, trace: CapturedTrace) -> None:
        pass


class LegacyBarrierEmitter(BarrierEmitter):
    """
    This class emits amdgpu.lds_barrier using minimize_placement_strategy.
    """

    def optimize(self, sync_regions: Sequence[SyncRegion]) -> Sequence[SyncRegion]:
        return minimize_placement_strategy(sync_regions)

    def place_barrier(self, region: SyncRegion) -> None:
        """
        Places a single shared memory barrier between producer and consumer.
        """
        is_async_op = lambda node: isinstance(get_custom(node), GatherToLDS)

        producer = region.producer
        consumer = region.consumer

        wait_async = is_async_op(producer) or is_async_op(consumer)
        with consumer.graph.inserting_before(consumer):
            SharedMemoryBarrier(wait_async_ops=wait_async).add_to_graph(
                consumer.graph, loc=get_custom(consumer).location
            )


class BasicSplitBarrierEmitter(BarrierEmitter):
    """
    This class emits rocdl.s.barrier.signal and rocdl.s.barrier.wait using find_disjoint_interval_strategy,
    and provides a verify method to ensure proper placement before invoking the runtime.
    """

    def verify(self, trace) -> None:
        """
        For difference scheduling such as Prefetch / Modulo, LR and LW may appear at prolog or epilog of a subgraph.
        This function checks if there are waits before any signals.
        """
        signals = defaultdict(bool)  # barId : signal exists
        lonely_waits = set()
        nodes = trace.preorder_walk()

        for node in nodes:
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
            len(lonely_waits) == 0
        ), "Wait barrier appears before any signals, this is a serious bug."
        assert len(signals) <= 1, "Only -1 barrier ID is supported on gfx120x."
        assert not signals.get(
            -1
        ), "All signals and waits should be paired, there are some leftover signals, this is a serious bug."

    def optimize(self, sync_regions: Sequence[SyncRegion]) -> Sequence[SyncRegion]:
        # note. we can also change the approach to minimize_placement_strategy.
        return find_disjoint_interval_strategy(sync_regions)

    def place_barrier(self, region: SyncRegion) -> None:
        """
        Place split barriers (signal/wait) for synchronization.
        """
        is_tensor_op = lambda node: isinstance(get_custom(node), TensorLoadToLDS)

        barId = -1
        producer = region.producer
        consumer = region.consumer
        barrier = is_barrier_between(producer, consumer, barId)

        is_tdm = is_tensor_op(producer) or is_tensor_op(consumer)
        if barrier is None:
            with producer.graph.inserting_after(producer):
                SharedMemoryBarrierSignal(barId, tensor_wait=is_tdm).add_to_graph(
                    producer.graph, loc=get_custom(producer).location
                )
            with consumer.graph.inserting_before(consumer):
                SharedMemoryBarrierWait(barId).add_to_graph(
                    consumer.graph, loc=get_custom(consumer).location
                )


def add_shared_memory_barriers(
    trace: CapturedTrace,
    target: str = "",
    is_specialized: bool = False,
):
    if is_specialized:
        """
        TODO(megan.kuo)
        currently respect barriers inserted by specializations
        """
        return

    target_arch = TargetConfig(target)

    sync_regions = get_barriers_analysis(trace, target_arch)

    emitter = BarrierEmitter(target_arch)
    emitter.emit(sync_regions)
    emitter.verify(trace)
