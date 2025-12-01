# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from functools import Placeholder
from typing import Sequence
from collections import defaultdict

from wave_lang.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE
from wave_lang.kernel.wave.utils.general_utils import ceildiv

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    GatherToLDS,
    GetResult,
    IterArg,
    Iterate,
    NullAsyncDep,
    Output,
    Read,
    TensorLoadToLDS,
    SharedMemoryBarrier,
    SharedMemoryBarrierSignal,
    SharedMemoryBarrierWait,
    Write,
    get_custom,
)
from .utils.graph_utils import (
    find_all_paths,
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

        producers = region.producers if is_async_op else []
        consumer = region.consumer

        with consumer.graph.inserting_before(consumer):
            SharedMemoryBarrier(async_deps=producers).add_to_graph(
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

        is_tdm = is_tensor_op(producer) or is_tensor_op(consumer) or region.is_tdm

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
):
    target_arch = TargetConfig(target)

    sync_regions = get_barriers_analysis(trace, target_arch)

    emitter = BarrierEmitter(target_arch)
    emitter.emit(sync_regions)
    emitter.verify(trace)


@dataclass
class AsyncDepNode:
    node: fx.Node
    dep_node: fx.Node
    read_count: int = 0
    write_count: int = 0


_MAX_ASYNC_READ_WRITE_COUNT = 8


def calculate_counter_value(elements_per_thread: int, dtype: "DataType") -> int:
    """
    Counter is incremented by 1 for each DWORD read/write.
    """
    return ceildiv(elements_per_thread * dtype.bitwidth(), 32)


def find_async_read_write_counts(
    barrier: fx.Node, async_dep: fx.Node
) -> tuple[int, int]:
    def get_edges(node: AsyncDepNode) -> list[AsyncDepNode]:
        if (
            node.read_count > _MAX_ASYNC_READ_WRITE_COUNT
            or node.write_count > _MAX_ASYNC_READ_WRITE_COUNT
        ):
            return []

        custom = get_custom(node.node)
        if isinstance(custom, NullAsyncDep):
            return []

        read_count = node.read_count
        write_count = node.write_count
        if (
            isinstance(custom, Read)
            and custom.memory_type.address_space == GLOBAL_ADDRESS_SPACE
        ):
            read_count += calculate_counter_value(
                custom.elements_per_thread, custom.type.dtype
            )

        elif (
            isinstance(custom, Write)
            and custom.memory_type.address_space == GLOBAL_ADDRESS_SPACE
        ):
            write_count += calculate_counter_value(
                custom.elements_per_thread, custom.type.dtype
            )

        elif isinstance(custom, GatherToLDS) and node.node != node.dep_node:
            read_count += calculate_counter_value(
                custom.elements_per_thread, custom.dtype
            )

        if node.node == node.dep_node:
            if isinstance(custom, IterArg):
                idx = custom.iter_idx
                graph = node.node.graph
                iterate = get_custom(graph.parent_op)
                assert isinstance(
                    iterate, Iterate
                ), f"Expected Iterate, but got {iterate}"
                async_dep_iterate = iterate.init_args[idx]

                output = get_custom(list(graph.nodes)[-1])
                assert isinstance(output, Output), f"Expected Output, but got {output}"
                async_dep_output = output.return_vals[0][idx]
                return [
                    AsyncDepNode(
                        iterate.fx_node, async_dep_iterate, read_count, write_count
                    ),
                    AsyncDepNode(
                        output.fx_node, async_dep_output, read_count, write_count
                    ),
                ]

            elif isinstance(custom, GetResult):
                idx = custom.res_idx
                iterate = get_custom(custom.value)
                graph = iterate.get_root_graph().subgraphs[iterate.subgraph_name]
                assert isinstance(
                    iterate, Iterate
                ), f"Expected Iterate, but got {iterate}"
                async_dep_iterate = iterate.init_args[idx]

                output = get_custom(list(graph.nodes)[-1])
                assert isinstance(output, Output), f"Expected Output, but got {output}"
                async_dep_output = output.return_vals[0][idx]
                return [
                    AsyncDepNode(
                        iterate.fx_node, async_dep_iterate, read_count, write_count
                    ),
                    AsyncDepNode(
                        output.fx_node, async_dep_output, read_count, write_count
                    ),
                ]

            elif isinstance(custom, Placeholder):
                if parent_op := getattr(node.node.graph, "parent_op", None):
                    lifted = node.node.meta["lifted"]
                    return [AsyncDepNode(parent_op, lifted, read_count, write_count)]
                else:
                    raise ValueError(f"Reached function argument: {custom}")
            else:
                return []

        return [AsyncDepNode(node.node.prev, node.dep_node, read_count, write_count)]

    paths = find_all_paths(AsyncDepNode(barrier, async_dep), get_edges)

    read_count = _MAX_ASYNC_READ_WRITE_COUNT
    write_count = _MAX_ASYNC_READ_WRITE_COUNT

    for path in paths:
        last_node = path[-1]
        if not isinstance(get_custom(last_node.node), GatherToLDS):
            continue

        read_count = min(read_count, last_node.read_count)
        write_count = min(write_count, last_node.write_count)

    return read_count, write_count


def populate_bariers_counters(
    trace: CapturedTrace,
):
    """
    Populate the read and write counters for the barriers neded for async
    dependencies.
    """

    def is_barrier(node: fx.Node) -> bool:
        return isinstance(get_custom(node), SharedMemoryBarrier)

    for node in trace.walk(is_barrier):
        barrier = get_custom(node)
        async_deps = barrier.async_deps
        if not async_deps:
            continue

        read_count = _MAX_ASYNC_READ_WRITE_COUNT
        write_count = _MAX_ASYNC_READ_WRITE_COUNT
        for dep in async_deps:
            r, w = find_async_read_write_counts(node, dep)
            read_count = min(read_count, r)
            write_count = min(write_count, w)

        barrier.update_arg("read_counter", read_count)
        barrier.update_arg("write_counter", write_count)
