# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum, IntFlag, auto
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import torch.fx as fx

from .graph_utils import propagate_loop_carried_vars

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
    Syncrhonization requirements in between producers and consumers.
    """

    resource: Any = None
    producers: Sequence[Any] = None
    consumers: Sequence[Any] = None
    is_loop: bool = False
    prod_region: Set[Any] = field(default_factory=set)
    cons_region: Set[Any] = field(default_factory=set)
    prod_topo_location: int = -1
    cons_topo_location: int = -1
    graph_start: int = -1
    graph_end: int = -1
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


def assign_preorder_index(nodes: List[fx.Node]):
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
    graph_info: List[int],
    barrier_type: BarrierType.NONE,
) -> bool:
    """
    Add cut (Synchronization requirements) to the state (in between last producer nodes and first consumer nodes)
    Returns whether the sync requirement is cross-iter bounds.
    """
    cross_iter = False
    last_prod = state.producers[-1]
    first_con = state.consumers[0]

    assert get_shared_memory_from_op(last_prod) == get_shared_memory_from_op(
        first_con
    ), "Bug"

    if resource is not None and not need_barrier(last_prod, first_con):
        # resource is None when we force-add barriers surrounding iterate and conditional nodes
        return False

    last_prod_loc = last_prod._topo_location
    first_con_loc = first_con._topo_location
    cross_iter = last_prod_loc > first_con_loc

    req = SyncRequirement(
        resource=resource,
        producers=list(state.producers),
        consumers=list(state.consumers),
        is_loop=cross_iter,  # when producer appear after consumer, we identify a loop
        prod_region={last_prod},
        cons_region={first_con},
        prod_topo_location=last_prod_loc,
        cons_topo_location=first_con_loc,
        graph_start=graph_info[0],
        graph_end=graph_info[1],
        barrier_type=barrier_type,
    )

    if req in results:
        return False
    results.append(req)

    return cross_iter


def handle_hazard(
    results,
    nodes,
    producer_kinds,
    consumer_kinds,
    barrier_type,
    is_nested: bool = False,
) -> BarrierType:
    """
    Scans the graph and append SyncRequirements to results if any.
    Returns if barriers are required for NestedRegionOps
    """
    states: Dict[fx.Node, EpisodeState] = defaultdict(EpisodeState)

    # duplicate nodes to find cross-iter dependencies
    # e.g.,
    # loop[w1:0, w2:1, r1:2, r2:3] -> flat[w1:0, w2:1, r1:2, r2:3, w1:0, w2:1, r1:2, r2:3]
    # dependencies for smem region 1:
    # - w1:0 -> r1:2
    # - r1:2 -> w1:0 ** cross-iter dep
    if is_nested:
        nodes = nodes * 2

    if len(nodes) == 0:
        return BarrierType.NONE

    graph_info = [nodes[0]._topo_location, nodes[-1]._topo_location]

    cross_iter = False
    cross_graph = False

    for node in nodes:
        op = get_custom(node)
        access_kind = get_memory_access_type(op)
        if access_kind == MemoryAccessType.NONE:
            continue
        resource = get_shared_memory_from_op(op)
        if resource is None:
            continue
        state = states[resource]

        if access_kind in producer_kinds:
            if state.producers and state.consumers:
                cross_iter |= add_sync_requirements(
                    results,
                    resource,
                    state,
                    graph_info=graph_info,
                    barrier_type=barrier_type,
                )
                cross_graph |= state.producers[-1].graph != state.consumers[0].graph
                state.reset()
            state.producers.append(node)
        if access_kind in consumer_kinds:
            if state.producers:
                state.consumers.append(node)

    for resource, state in states.items():
        if state.producers and state.consumers:
            cross_iter |= add_sync_requirements(
                results,
                resource,
                state,
                graph_info=graph_info,
                barrier_type=barrier_type,
            )
            cross_graph |= state.producers[-1].graph != state.consumers[0].graph

    # barriers are needed if cross iteration dependencies exist and no producers from outside the subgraph.
    return barrier_type if cross_iter and not cross_graph else BarrierType.NONE


def get_barriers_analysis(trace, graph, target_arch):
    nodes = trace.preorder_walk()
    assign_preorder_index(nodes)

    is_shared_memory_node = lambda node: (
        True if get_shared_memory_from_op(get_custom(node)) is not None else False
    )
    is_iterate_node = lambda node: (
        True if isinstance(get_custom(node), Iterate) else False
    )
    is_condition_node = lambda node: (
        True if isinstance(get_custom(node), Conditional) else False
    )

    # root graph
    # handle linear dependencies
    smem_nodes = trace.preorder_walk(filter=is_shared_memory_node)
    iterate_nodes = trace.preorder_walk(filter=is_iterate_node)
    condition_nodes = trace.preorder_walk(filter=is_condition_node)

    results: List[SyncRequirement] = []

    # RAW
    handle_hazard(
        results,
        smem_nodes,
        {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
        {MemoryAccessType.READ},
        barrier_type=BarrierType.FILL,
    )

    # WAR
    handle_hazard(
        results,
        smem_nodes,
        {MemoryAccessType.READ},
        {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
        barrier_type=BarrierType.READY,
    )

    # handle nested iterate graph
    for regionNode in reversed(iterate_nodes):
        regionOp = get_custom(regionNode)
        smem_nodes = trace.preorder_walk(
            name=regionOp.subgraph_name, filter=is_shared_memory_node
        )
        need_iterate_barriers = BarrierType.NONE
        graph_info = [smem_nodes[0]._topo_location, smem_nodes[-1]._topo_location]

        # RAW
        need_iterate_barriers |= handle_hazard(
            results,
            smem_nodes,
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
            {MemoryAccessType.READ},
            is_nested=True,
            barrier_type=BarrierType.FILL,
        )

        # WAR
        need_iterate_barriers |= handle_hazard(
            results,
            smem_nodes,
            {MemoryAccessType.READ},
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
            is_nested=True,
            barrier_type=BarrierType.READY,
        )

        if target_arch.has_split_barriers and need_iterate_barriers != BarrierType.NONE:
            add_sync_requirements(
                results,
                None,
                EpisodeState([regionNode.prev], [regionNode.next]),
                graph_info=graph_info,
                barrier_type=need_iterate_barriers,
            )

    # handle conditional graph
    for regionNode in condition_nodes:
        regionOp = get_custom(regionNode)
        smem_nodes = trace.preorder_walk(
            name=regionOp.subgraph_name, filter=is_shared_memory_node
        )
        graph_info = [smem_nodes[0]._topo_location, smem_nodes[-1]._topo_location]

        # RAW
        handle_hazard(
            results,
            smem_nodes,
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
            {MemoryAccessType.READ},
            barrier_type=BarrierType.FILL,
        )

        # WAR
        handle_hazard(
            results,
            smem_nodes,
            {MemoryAccessType.READ},
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
            barrier_type=BarrierType.READY,
        )

        if target_arch.has_split_barriers:
            add_sync_requirements(
                results,
                None,
                EpisodeState([regionNode.prev], [regionNode]),
                graph_info=graph_info,
            )
            add_sync_requirements(
                results,
                None,
                EpisodeState([regionNode], [regionNode.next]),
                graph_info=graph_info,
            )

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

        if any(
            [
                pos in range(start, end + 1) and btype == bexist
                for pos, bexist in placements
            ]
        ):
            continue

        results.append(req)
        placements.append((end, btype))

    # 3) handle cross iteration sync req separately (they have producer loc > consumer loc)
    for req in cross_iters:
        start, end = get_location(req)
        graph_start, graph_end = req.graph_start, req.graph_end
        btype = req.barrier_type

        assert (
            start > end
        ), "Got producer location < consumer location but identified as cross-iter loop."
        assert graph_start < graph_end, "graph start < graph end."

        # 3.1) if graph start ~ sync request start has barrier placements: skip
        if any(
            [
                p in range(graph_start, end) and btype == bexist
                for p, bexist in placements
            ]
        ):
            continue

        # 3.2) if graph start ~ sync request start has barrier placements: skip
        if any(
            [
                p in range(start, graph_end) and btype == bexist
                for p, bexist in placements
            ]
        ):
            continue

        # 3.3) if cross-iter requiers both RAW and WAR guards
        if btype in (BarrierType.FILL | BarrierType.READY):
            if any(
                [
                    p in range(graph_start, end) and btype == bexist
                    for p, bexist in placements
                ]
            ) and any(
                [
                    p in range(start, graph_end) and btype == bexist
                    for p, bexist in placements
                ]
            ):
                continue

        # 3.4) else valid placements
        results.append(req)
        placements.append(end)

    return results


def find_smallest_interval_strategy(
    sync_reqs: Sequence[SyncRequirement],
) -> Sequence[SyncRequirement]:

    get_location = lambda req: (req.prod_topo_location, req.cons_topo_location)
    get_prod = lambda req: next(iter(req.prod_region))
    get_cons = lambda req: next(iter(req.cons_region))

    if len(sync_reqs) == 0:
        return sync_reqs

    placements = []
    results = []

    # 1) sort barrier placement location from pos low to high
    ascending_reqs = sorted(
        sync_reqs,
        key=lambda req: (req.prod_topo_location, req.cons_topo_location),
    )

    # 2) variables
    prev = ascending_reqs[0]
    run_lo, run_hi = get_location(prev)
    run_len = 1
    signal = get_prod(prev)
    wait = get_cons(prev)

    for cur in ascending_reqs[1:]:
        cur_lo, cur_hi = get_location(cur)

        lo, hi = max(run_lo, cur_lo), min(run_hi, cur_hi)
        signal = get_prod(cur) if run_lo < cur_lo else signal
        wait = get_cons(cur) if run_hi > cur_hi else wait

        if lo <= hi:
            run_lo, run_hi = lo, hi
            run_len += 1
            prev = cur
        else:
            req = SyncRequirement(
                prod_region={signal},
                cons_region={wait},
                prod_topo_location=run_lo,
                cons_topo_location=run_hi,
            )
            results.append(req if run_len > 1 else prev)

            prev_loc = get_location(prev)
            cur_loc = get_location(cur)

            signal = get_prod(cur) if prev_loc[0] < cur_loc[0] else get_prod(prev)
            wait = get_cons(cur) if prev_loc[1] > cur_loc[1] else get_cons(prev)

            lo_update, hi_update = max(prev_loc[0], cur_loc[0]), min(
                prev_loc[1], cur_loc[1]
            )
            if lo_update <= hi_update:
                run_lo, run_hi = lo_update, hi_update
                run_len = 2
            else:
                run_lo, run_hi = cur_loc
                run_len = 1

            prev = cur

    req = SyncRequirement(
        prod_region={signal},
        cons_region={wait},
        prod_topo_location=run_lo,
        cons_topo_location=run_hi,
    )
    results.append(req)

    return results
