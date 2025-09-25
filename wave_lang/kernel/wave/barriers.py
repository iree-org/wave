# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Any, Iterable

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import SHARED_ADDRESS_SPACE, GLOBAL_ADDRESS_SPACE
from ..ops.wave_ops import (
    AtomicOp,
    CustomOp,
    GatherToLDS,
    GetResult,
    IterArg,
    Iterate,
    NestedRegionOp,
    NullAsyncDep,
    Output,
    Placeholder,
    Read,
    SharedMemoryBarrier,
    Write,
    get_custom,
)
from .utils.graph_utils import (
    find_all_paths,
    is_barrier_between,
    is_reduction_subgraph,
    propagate_loop_carried_vars,
)
from .utils.general_utils import ceildiv


class MemoryAccessType(Enum):
    """Enum to classify memory access operations."""

    NONE = auto()
    READ = auto()
    WRITE = auto()
    READ_WRITE = auto()


def get_all_sources(node: fx.Node) -> list[fx.Node]:
    source1 = propagate_loop_carried_vars(node, 0)
    source2 = propagate_loop_carried_vars(node, 1)
    if source1 != source2:
        return [source1, source2]
    return [source1]


def is_shared_memory_op(node: CustomOp) -> list[fx.Node]:
    if (
        isinstance(node, (Read, Write, AtomicOp))
        and node.memory_type.address_space == SHARED_ADDRESS_SPACE
    ):
        return get_all_sources(node.memory)
    elif isinstance(node, GatherToLDS):
        return get_all_sources(node.dst)

    return []


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


def get_first(seq: Iterable[Any]) -> Any:
    """
    Get the first element of the sequence or generator.
    """
    return next(iter(seq))


def get_last(seq: Iterable[Any]) -> Any:
    """
    Get the last element of the sequence or generator.
    """
    *_, last = iter(seq)
    return last


@dataclass
class SharedMemoryBarrierInfo:
    async_deps: list[fx.Node] = field(default_factory=list)
    last_node: Optional[CustomOp] = None


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
        mems = is_shared_memory_op(custom)
        for mem in mems:
            state = info[mem]
            if state.last_node and need_barrier(custom, state.last_node):
                if barrier := is_barrier_between(state.last_node.fx_node, node):
                    barrier = get_custom(barrier)
                    # Add async deps to the barrier.
                    if state.async_deps:
                        deps = list(
                            dict.fromkeys(barrier.async_deps)
                            | dict.fromkeys(state.async_deps)
                        )
                        barrier.update_arg("async_deps", deps)
                else:
                    # Synchronize after the write to shared memory before we read from it.
                    deps = list(dict.fromkeys(state.async_deps))
                    with graph.inserting_before(node):
                        SharedMemoryBarrier(async_deps=deps).add_to_graph(graph)

                state.async_deps = []

            state.last_node = custom
            if isinstance(custom, GatherToLDS):
                state.async_deps.append(node)

        if isinstance(custom, NestedRegionOp):
            subgraph = trace.get_subgraph(custom.subgraph_name)
            if not any(i.async_deps for i in info.values()):
                add_shared_memory_barriers(trace, subgraph, info)
                continue

            first = get_first(subgraph.nodes)

            # Convert dependencies to placeholders.
            with subgraph.inserting_before(first):
                for node, inf in info.items():
                    for i, async_dep in enumerate(inf.async_deps):
                        placeholder = Placeholder(str(async_dep)).add_to_graph(subgraph)
                        placeholder.meta["lifted"] = async_dep
                        inf.async_deps[i] = placeholder
                        custom.update_arg(
                            "implicit_captures", [async_dep] + custom.implicit_captures
                        )

            add_shared_memory_barriers(trace, subgraph, info)

    # Synchronize before the write to shared memory to avoid stepping over
    # shared reads in the previous iteration of a loop.
    if is_reduction_subgraph(graph) and info and not checking_next_iter:
        # Add barriers between ops from different iterations in the same loop.
        if not any(i.async_deps for i in info.values()):
            add_shared_memory_barriers(trace, graph, info, checking_next_iter=True)
            return

        parent_node = graph.parent_op
        parent_graph = parent_node.graph

        iterate = get_custom(parent_node)
        assert isinstance(iterate, Iterate), f"Expected Iterate, but got {iterate}"

        first = get_first(graph.nodes)

        output = get_custom(get_last(graph.nodes))
        assert isinstance(output, Output), f"Expected Output, but got {output}"
        return_vals = list(output.return_vals[0])

        # Convert dependencies to iter args.
        info_copy = defaultdict(SharedMemoryBarrierInfo)
        for node, inf in info.items():
            inf_copy = info_copy[node]
            inf_copy.last_node = inf.last_node
            for i, async_dep in enumerate(inf.async_deps):
                out_idx = len(return_vals)
                return_vals.append(async_dep)

                with parent_graph.inserting_before(parent_node):
                    init = NullAsyncDep().add_to_graph(parent_graph)

                iterate.update_arg("init_args", iterate.init_args + [init])

                with parent_graph.inserting_after(parent_node):
                    result = GetResult(parent_node, out_idx).add_to_graph(parent_graph)

                inf.async_deps[i] = result

                with graph.inserting_before(first):
                    iter_arg = IterArg(str(async_dep)).add_to_graph(graph)
                    iter_arg.iter_idx = out_idx

                inf_copy.async_deps.append(iter_arg)

        output.update_arg("return_vals", return_vals)

        add_shared_memory_barriers(trace, graph, info_copy, checking_next_iter=True)


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
