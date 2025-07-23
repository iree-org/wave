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
from ..lang.global_symbols import SHARED_ADDRESS_SPACE
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


def get_first(seq: Iterable[Any]) -> Any:
    return next(iter(seq))


def get_last(seq: Iterable[Any]) -> Any:
    *_, last = iter(seq)
    return last


@dataclass
class SharedMemoryBarrierInfo:
    async_deps: list[fx.Node] = field(default_factory=list)
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
            first = get_first(subgraph.nodes)

            # Convert dependencies to placeholders.
            with subgraph.inserting_before(first):
                for node, inf in info.items():
                    for i, async_dep in enumerate(inf.async_deps):
                        placeholder = Placeholder().add_to_graph(subgraph)
                        placeholder.meta["lifted"] = async_dep
                        inf.async_deps[i] = placeholder

            add_shared_memory_barriers(trace, subgraph, info)

    # Synchronize before the write to shared memory to avoid stepping over
    # shared reads in the previous iteration of a loop.
    if is_reduction_subgraph(graph) and info and not checking_next_iter:
        # Add barriers between ops from different iterations in the same loop.
        parent_node = graph.parent_op
        parent_graph = parent_node.graph

        iterate = get_custom(parent_node)
        assert isinstance(iterate, Iterate), f"Expected Iterate, but got {iterate}"

        first = get_first(graph.nodes)

        output = get_custom(get_last(graph.nodes))
        assert isinstance(output, Output), f"Expected Output, but got {output}"

        # Convert dependencies to iter args.
        info_copy = defaultdict(SharedMemoryBarrierInfo)
        for node, inf in info.items():
            inf_copy = info_copy[node]
            for i, async_dep in enumerate(inf.async_deps):
                out_idx = len(output.return_vals)
                output.return_vals.append(async_dep)

                with parent_graph.inserting_before(parent_node):
                    init = NullAsyncDep().add_to_graph(parent_graph)
                iterate.init_args.append(init)

                with parent_graph.inserting_after(parent_node):
                    inf.async_deps[i] = GetResult(parent_node, out_idx).add_to_graph(
                        parent_graph
                    )

                with graph.inserting_before(first):
                    inf_copy.async_deps.append(
                        IterArg(parent_node, out_idx).add_to_graph(graph)
                    )

        add_shared_memory_barriers(trace, graph, info_copy, checking_next_iter=True)
