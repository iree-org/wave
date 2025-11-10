# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum, IntFlag, auto
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union, Set

import torch.fx as fx

from .graph_utils import propagate_loop_carried_vars

from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import SHARED_ADDRESS_SPACE
from ...ops.wave_ops import (
    AtomicOp,
    CustomOp,
    GatherToLDS,
    Read,
    Write,
    get_custom,
    Iterate,
    Conditional,
    NestedRegionOp,
)


@dataclass
class TargetConfig:
    target: str
    has_split_barriers: bool = False
    has_named_barriers: bool = False
    max_named_barriers: int = 0

    def __post_init__(self):
        if "gfx12" in self.target:
            self.has_split_barriers = True

        if "gfx1250" in self.target:
            self.has_named_barriers = True
            self.max_named_barriers = 16


class BarrierType(IntFlag):
    """
    For RAW Write -> Read: guard fill
    For WAR Read -> Write: guard ready
    """

    NONE = auto()
    READY = auto()
    FILL = auto()


@dataclass
class SyncRequirement:
    """
    Synchronization requirements in between producers and consumers.
    """

    resource: Any = None
    producers: Sequence[Any] = None
    consumers: Sequence[Any] = None
    is_loop: bool = False
    prod_region: fx.Node = None
    cons_region: fx.Node = None
    prod_topo_location: int = -1
    cons_topo_location: int = -1
    graph_start: fx.Node = None
    graph_end: fx.Node = None
    barrier_type: BarrierType = BarrierType.NONE


class EpisodeState:
    def __init__(
        self, producers: List[fx.Node] = None, consumers: List[fx.Node] = None
    ):
        self.producers: List[fx.Node] = producers if producers is not None else []
        self.consumers: List[fx.Node] = consumers if consumers is not None else []

    def reset(self):
        self.producers = []
        self.consumers = []


class MemoryAccessType(Enum):
    """
    Enum to classify memory access operations.
    """

    NONE = auto()
    READ = auto()
    WRITE = auto()
    READ_WRITE = auto()


def assign_preorder_index(nodes: List[fx.Node]) -> None:
    """
    Given a list of nodes, assign `_topo_location` attribute to the node with enumeration order.
    """
    for idx, node in enumerate(nodes):
        setattr(node, "_topo_location", idx)


def get_memory_access_type(op: CustomOp) -> MemoryAccessType:
    """
    Get the memory access type of a custom node.
    """
    if isinstance(op, Read):
        return MemoryAccessType.READ
    elif isinstance(op, Write):
        return MemoryAccessType.WRITE
    elif isinstance(op, AtomicOp):
        return MemoryAccessType.READ_WRITE
    elif isinstance(op, GatherToLDS):
        return MemoryAccessType.WRITE
    else:
        return MemoryAccessType.NONE


def get_shared_memory_from_op(op: CustomOp, depth: int = 0) -> Optional[fx.Node]:
    """
    Given a customOp, returns whether it operates on a shared memory region.
    """
    if (
        isinstance(op, (Read, Write))
        and op.memory_type.address_space == SHARED_ADDRESS_SPACE
    ):
        return propagate_loop_carried_vars(op.memory, depth)
    if (
        isinstance(op, AtomicOp)
        and op.memory_type.address_space == SHARED_ADDRESS_SPACE
    ):
        return propagate_loop_carried_vars(op.rhs, depth)
    elif isinstance(op, GatherToLDS):
        return propagate_loop_carried_vars(op.dst, depth)

    return None


def need_barrier(
    node1: Union[fx.Node, CustomOp], node2: Union[fx.Node, CustomOp]
) -> bool:
    """
    Check if node1 and node2 have different memory access types.
    If so, we need a barrier in between.
    Else, we don't need a barrier.
    """
    node1 = get_custom(node1) if isinstance(node1, fx.Node) else node1
    node2 = get_custom(node2) if isinstance(node2, fx.Node) else node2

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


def add_sync_requirements(
    results: List[SyncRequirement],
    resource: fx.Node,
    state: EpisodeState,
    graph_info: List[fx.Node],
    barrier_type: BarrierType.NONE,
) -> None:
    """
    Add a SyncRequirement to the results list.
    """
    cross_iter = False
    last_prod = state.producers[-1]
    first_con = state.consumers[0]

    assert get_shared_memory_from_op(last_prod) == get_shared_memory_from_op(
        first_con
    ), f"BUG: {last_prod} and {first_con} reference different shared-memory regions. This indicates a bug in handle_hazard: producers and consumers should operate on the same region but are currently mismatched."

    if resource is not None and not need_barrier(last_prod, first_con):
        return

    last_prod_loc = last_prod._topo_location
    first_con_loc = first_con._topo_location
    cross_iter = last_prod_loc > first_con_loc

    req = SyncRequirement(
        resource=resource,
        producers=list(state.producers),
        consumers=list(state.consumers),
        is_loop=cross_iter,  # when producer appear after consumer, we identify a loop
        prod_region=last_prod,
        cons_region=first_con,
        prod_topo_location=last_prod_loc,
        cons_topo_location=first_con_loc,
        graph_start=graph_info[0],
        graph_end=graph_info[1],
        barrier_type=barrier_type,
    )

    if req in results:
        return

    results.append(req)


def handle_hazard(
    results: List[SyncRequirement],
    nodes: List[SyncRequirement],
    producer_kinds: Set[MemoryAccessType],
    consumer_kinds: Set[MemoryAccessType],
    barrier_type: BarrierType,
    is_nested: bool = False,
    iterate_region: int = 0,
) -> None:
    """
    Scans the graph and append SyncRequirements to results if any.
    The `states` dictionary tracks which producers and consumers are holding the resource.
    """
    if not nodes:
        return

    states: Dict[fx.Node, EpisodeState] = defaultdict(EpisodeState)
    n = len(nodes)
    graph_info = [None, None]

    # duplicate nodes to find cross-iter dependencies
    # e.g.,
    #
    # original loop has nodes
    # [w1:0, w2:1, r1:0, r2:1]
    #
    # after duplication
    # [w1:0, w2:1, r1:0, r2:1, w1:0, w2:1, r1:0, r2:1]
    #
    # after propagating depth
    # [w1:0, w2:1, r1:0, r2:1, w1:1, w2:0, r1:1, r2:0]
    #
    # where is hazard ?
    # - w1:0 -> r1:0
    # - w2:1 -> r2:1
    # - r2:1 -> w1:1 <- cross-iter dep
    # - r1:0 -> w2:0 <- cross-iter dep
    if is_nested:
        graph_info = [nodes[0], nodes[-1]]
        nodes = nodes * 2

    for idx, node in enumerate(nodes):
        op = get_custom(node)
        access_kind = get_memory_access_type(op)
        if access_kind == MemoryAccessType.NONE:
            continue

        depth = idx // n if iterate_region == 0 else int(idx < iterate_region)
        resource = get_shared_memory_from_op(op, depth)
        assert resource is not None, "op has no smem access"

        state = states[resource]

        if access_kind in producer_kinds:
            if state.producers and state.consumers:
                add_sync_requirements(
                    results,
                    resource,
                    state,
                    graph_info=graph_info,
                    barrier_type=barrier_type,
                )
                state.reset()
            state.producers.append(node)
        if access_kind in consumer_kinds:
            if state.producers:
                state.consumers.append(node)

    # final cleanup of each state.
    for resource, state in states.items():
        if state.producers and state.consumers:
            add_sync_requirements(
                results,
                resource,
                state,
                graph_info=graph_info,
                barrier_type=barrier_type,
            )


def get_subgraph_nodes(trace: CapturedTrace, node: fx.Node) -> List[fx.Node]:
    """
    Returns the list of nodes in the subgraph associated with the given node.
    Args:
        - trace: The trace object containing the graph and walk_graph method.
        - node (fx.Node): The node whose subgraph nodes are to be retrieved.
    Returns:
        List[fx.Node]: A list of nodes in the subgraph if the node is a NestedRegionOp, otherwise an empty list.
    """
    op = get_custom(node)
    if not isinstance(op, NestedRegionOp):
        return []

    subgraph_name = op.subgraph_name
    return trace.walk_graph(name=subgraph_name)


def get_barriers_analysis(
    trace: CapturedTrace, graph: fx.Graph, target_arch: TargetConfig
) -> List[SyncRequirement]:
    nodes = trace.preorder_walk()
    assign_preorder_index(nodes)

    is_shared_memory_node = (
        lambda node: get_shared_memory_from_op(get_custom(node)) is not None
    )
    is_iterate_node = lambda node: isinstance(get_custom(node), Iterate)
    is_condition_node = lambda node: isinstance(get_custom(node), Conditional)

    results: List[SyncRequirement] = []
    collections: List[SyncRequirement] = []
    nodes = trace.walk_graph(trace.root_graph)

    def dfs(
        nodes: List[fx.Node],
        collections: List[fx.Node],
        results: List[SyncRequirement],
        is_nested: bool = False,
        iterate_region: int = 0,
    ) -> None:
        for node in nodes:
            if is_shared_memory_node(node):
                collections.append(node)

            if is_iterate_node(node):
                subgraph_nodes = get_subgraph_nodes(trace, node)
                dfs(subgraph_nodes, collections, results)
                collections = []
                dfs(subgraph_nodes, collections, results, is_nested=True)
                iterate_region = sum(
                    [is_shared_memory_node(node) for node in subgraph_nodes]
                )

            if is_condition_node(node):
                subgraph_nodes = get_subgraph_nodes(trace, node)
                dfs(subgraph_nodes, collections, results)

        # RAW
        handle_hazard(
            results,
            collections,
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
            {MemoryAccessType.READ},
            barrier_type=BarrierType.FILL,
            is_nested=is_nested,
            iterate_region=iterate_region,
        )

        # WAR
        handle_hazard(
            results,
            collections,
            {MemoryAccessType.READ},
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
            barrier_type=BarrierType.READY,
            is_nested=is_nested,
            iterate_region=iterate_region,
        )

    dfs(nodes, collections, results)

    return results


def minimize_placement_strategy(
    sync_reqs: Sequence[SyncRequirement],
) -> Sequence[SyncRequirement]:
    if len(sync_reqs) == 0:
        return sync_reqs

    placements: List[Tuple[int, BarrierType]] = []
    results = []
    cross_iters = []

    # helper
    get_location = lambda req: (req.prod_topo_location, req.cons_topo_location)

    # 1) sort barrier placement location from pos low to high
    ascending_reqs = sorted(
        sync_reqs,
        key=lambda req: (req.cons_topo_location, req.prod_topo_location),
    )

    # 2) add to result if no barriers are placed in between topo_region
    for req in ascending_reqs:
        if req.is_loop:
            cross_iters.append(req)
            continue

        start, end = get_location(req)
        btype = req.barrier_type

        if any([pos in range(start + 1, end + 1) for pos, _ in placements]):
            continue

        results.append(req)
        placements.append((end, btype))

    # 3) handle cross iteration sync req separately (they have producer loc > consumer loc)
    for req in cross_iters:
        start, end = get_location(req)
        graph_start, graph_end = (
            req.graph_start._topo_location,
            req.graph_end._topo_location,
        )
        btype = req.barrier_type

        assert (
            start > end
        ), "Got producer location < consumer location but identified as cross-iter loop."
        assert graph_start < graph_end, "graph start < graph end."

        # 3.1) if graph start ~ sync request start already has barrier placements: skip
        if any([p in range(graph_start, end + 1) for p, _ in placements]):
            continue

        # 3.2) if graph start ~ sync request start already has barrier placements: skip
        if any([p in range(start, graph_end + 1) for p, _ in placements]):
            continue

        # 3.4) else valid placements
        results.append(req)
        placements.append((end, btype))

    return results


def find_overlapping_interval_strategy(
    sync_reqs: Sequence[SyncRequirement],
) -> Sequence[SyncRequirement]:
    """
    def position:
    - node <- smallest
    - node
    - node <- largest

    The algorithm keep tracks of the smallest wait position, and
    update the signal position if a request has
    1) a wait with larger position than track-recorded wait position, and
    2) a signal with larger position than track-recorded signal position

    We add synchronization requirment to the result when current signal position is larger than track-recorded wait position
    """
    get_location = lambda req: (req.prod_topo_location, req.cons_topo_location)

    if len(sync_reqs) == 0:
        return sync_reqs

    placements = []
    results = []
    cross_iters = []

    # 1) sort barrier placement location from pos small to large
    ascending_reqs = sorted(
        sync_reqs,
        key=lambda req: (req.cons_topo_location, req.prod_topo_location),
    )

    # 2) define algorithm
    def run_algorithm(reqs: List[SyncRequirement]) -> bool:
        idx = 0
        signal = None
        wait = None
        inter_graph = True

        # find the first non-cross-iter hazard
        while idx < len(reqs):
            req = reqs[idx]
            if req.is_loop:
                cross_iters.append(req)
                idx += 1
                continue
            else:
                signal = req.prod_region
                wait = req.cons_region
                break
            idx += 1

        if signal is None and wait is None:
            return

        # track-recorded positions
        rec_signal_pos = req.prod_topo_location
        rec_wait_pos = req.cons_topo_location

        for req in reqs[idx:]:
            if req.is_loop:
                cross_iters.append(req)
                continue

            # current barrier request region
            cur_signal_pos, cur_wait_pos = get_location(req)

            # current signal position > recorded wait position
            if cur_signal_pos > rec_wait_pos:
                new_req = SyncRequirement(
                    prod_region=signal,
                    cons_region=wait,
                )

                if new_req not in results:
                    inter_graph &= signal.graph == wait.graph
                    results.append(new_req)

                rec_signal_pos = cur_signal_pos
                rec_wait_pos = cur_wait_pos
                signal = req.prod_region
                wait = req.cons_region
            # update signal condition
            else:
                if cur_signal_pos > rec_signal_pos:
                    rec_signal_pos = cur_signal_pos
                    signal = req.prod_region
                if cur_wait_pos < rec_wait_pos:
                    rec_wait_pos = cur_wait_pos
                    wait = req.cons_region

        new_req = SyncRequirement(
            prod_region=signal,
            cons_region=wait,
        )
        if new_req not in results:
            results.append(new_req)
            inter_graph &= signal.graph == wait.graph

        return inter_graph

    run_algorithm(ascending_reqs)

    # 3) handle cross iter
    #    for cross iteration hazards, we will get dependencies like
    #    producer: loc=55
    #    consumer: loc=20
    #    producer: loc=45
    #    consumer: loc=25
    #    -> where to place barrier? producer: 55, consumer: 20
    #
    #    we can reduce this problem and make it runnable by the algorithm
    #    simply by adding offset to the consumer.
    #    where offset = len(graph.nodes), we use 40 in this example
    #
    #    producer: loc 55
    #    consumer: loc (20+40) 60
    #    producer: loc 45
    #    consumer: loc (25+40) 65
    #    -> where to place barrier? producer: 55, consumer: 60
    #
    #    note for cross-iter special cases
    #    0. ---------------
    #           consumer
    #           producer
    #    1. ---------------
    #       producer
    #           consumer
    #           producer
    #       consumer
    #    2. ---------------
    #       producer
    #           consumer
    #           producer
    #    3. ---------------
    #           consumer
    #           producer
    #       consumer
    #
    cross_iter_reqs = defaultdict(list)
    if len(cross_iters) != 0:
        # collected nested region sync points
        for req in cross_iters:
            prod_loc, cons_loc = get_location(req)
            assert req.prod_region.graph == req.cons_region.graph
            assert prod_loc > cons_loc

            req.is_loop = False
            graph = req.prod_region.graph
            graph_start, graph_end = (
                req.graph_start._topo_location,
                req.graph_end._topo_location,
            )

            # check for special cases
            if any(
                res.cons_region._topo_location in range(graph_start, cons_loc + 1)
                for res in results
            ) and any(
                res.prod_region._topo_location in range(prod_loc, graph_end + 1)
                for res in results
            ):
                continue
            elif any(
                res.cons_region._topo_location in range(graph_start, cons_loc + 1)
                for res in results
            ) and not any(
                res.prod_region._topo_location in range(prod_loc, graph_end + 1)
                for res in results
            ):
                req.cons_region = graph.parent_op.next
                req.cons_topo_location = req.cons_region._topo_location
                cross_iter_reqs[graph].append(req)
                continue
            elif not any(
                res.cons_region._topo_location in range(graph_start, cons_loc + 1)
                for res in results
            ) and any(
                res.prod_region._topo_location in range(prod_loc, graph_end + 1)
                for res in results
            ):
                continue

            n = len(graph.nodes)
            req.cons_topo_location += n
            cross_iter_reqs[graph].append(req)

    for graph, reqs in cross_iter_reqs.items():
        n_hazards = len(results)
        inter_graph = run_algorithm(reqs)
        if len(results) > n_hazards and inter_graph:
            # this mean we add a cross iter signal and wait
            iterateOp = graph.parent_op
            new_req = SyncRequirement(
                prod_region=iterateOp.prev,
                cons_region=iterateOp.next,
            )
            results.append(new_req)

    return results
