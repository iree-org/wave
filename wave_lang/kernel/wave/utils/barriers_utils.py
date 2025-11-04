# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum, auto
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


@dataclass
class SyncRequirement:
    """
    Syncrhonization requirements in between producers and consumers.
    """

    resource: Any
    producers: Sequence[Any]
    consumers: Sequence[Any]
    is_loop: bool = False
    prod_region: Set[Any] = field(default_factory=set)
    cons_region: Set[Any] = field(default_factory=set)


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


def update_topo_location(last_prod, first_con, offset: int = 0):
    """
    prod has topo_location > consumer topo_location iff an iterate exist.
    arange the topo_location of a consumer as
    new_consumer_location =  producer_location + (producer_location - old_cons_loc)
    """
    prod_loc = last_prod._topo_location
    cons_loc = first_con._topo_location

    # valid
    if prod_loc < cons_loc:
        return

    setattr(first_con, "_topo_location", cons_loc + offset)
    return


def add_sync_requirements(
    results: List[SyncRequirement],
    resource: fx.Node,
    state: EpisodeState,
    offset: int = 0,
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

    if resource is not None:
        if not need_barrier(last_prod, first_con):
            return False
        last_prod_loc = last_prod._topo_location
        first_con_loc = first_con._topo_location
        cross_iter = last_prod_loc > first_con_loc

    update_topo_location(last_prod, first_con, offset)
    req = SyncRequirement(
        resource=resource,
        producers=list(state.producers),
        consumers=list(state.consumers),
        is_loop=cross_iter,  # when producer appear after consumer, we identify a loop
        prod_region={last_prod},
        cons_region={first_con},
    )

    if req in results:
        return False
    results.append(req)

    return cross_iter


def handle_hazard(
    results, nodes, producer_kinds, consumer_kinds, is_nested: bool = False
):
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

    cross_iter = False
    cross_graph = False
    offset = len(nodes)

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
                cross_iter |= add_sync_requirements(results, resource, state, offset)
                cross_graph |= state.producers[-1].graph != state.consumers[0].graph
                state.reset()
            state.producers.append(node)
        if access_kind in consumer_kinds:
            if state.producers:
                state.consumers.append(node)

    for resource, state in states.items():
        if state.producers and state.consumers:
            cross_iter |= add_sync_requirements(results, resource, state, offset)

    # barriers are needed if cross iteration dependencies exist and no producers from outside the subgraph.
    return cross_iter and not cross_graph


def get_barriers_analysis(trace, graph):
    nodes = trace.walk()
    assign_preorder_index(nodes)

    get_topo_location = lambda x: getattr(next(iter(x)), "_topo_location", 0)

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
    root_graph_name = trace.root_graph
    smem_nodes = trace.walk_graph(name=root_graph_name, filter=is_shared_memory_node)
    iterate_nodes = trace.walk(is_iterate_node)
    condition_nodes = trace.walk(is_condition_node)

    results: List[SyncRequirement] = []

    # RAW
    handle_hazard(
        results,
        smem_nodes,
        {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
        {MemoryAccessType.READ},
    )

    # WAR
    handle_hazard(
        results,
        smem_nodes,
        {MemoryAccessType.READ},
        {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
    )

    # handle nested iterate graph
    for regionNode in reversed(iterate_nodes):
        regionOp = get_custom(regionNode)
        smem_nodes = trace.walk_graph(
            name=regionOp.subgraph_name, filter=is_shared_memory_node
        )
        need_iterate_barriers = False

        # RAW
        need_iterate_barriers |= handle_hazard(
            results,
            smem_nodes,
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
            {MemoryAccessType.READ},
            is_nested=True,
        )

        # WAR
        need_iterate_barriers |= handle_hazard(
            results,
            smem_nodes,
            {MemoryAccessType.READ},
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
            is_nested=True,
        )

        if need_iterate_barriers:
            add_sync_requirements(
                results, None, EpisodeState([regionNode.prev], [regionNode.next])
            )

    # handle conditional graph
    for regionNode in condition_nodes:
        regionOp = get_custom(regionNode)
        smem_nodes = trace.walk_graph(
            name=regionOp.subgraph_name, filter=is_shared_memory_node
        )

        # RAW
        handle_hazard(
            results,
            smem_nodes,
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
            {MemoryAccessType.READ},
        )

        # WAR
        handle_hazard(
            results,
            smem_nodes,
            {MemoryAccessType.READ},
            {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
        )

        add_sync_requirements(
            results, None, EpisodeState([regionNode.prev], [regionNode])
        )
        add_sync_requirements(
            results, None, EpisodeState([regionNode], [regionNode.next])
        )

    return results


def minimize_placement_strategy(
    sync_reqs: Sequence[SyncRequirement],
) -> Sequence[SyncRequirement]:

    get_topo_location = lambda x: getattr(next(iter(x)), "_topo_location", 0)

    if len(sync_reqs) == 0:
        return sync_reqs

    placements = []
    results = []

    # 1) sort barrier placement location from pos low to high
    ascending_reqs = sorted(
        sync_reqs,
        key=lambda req: (
            get_topo_location(req.cons_region),
            get_topo_location(req.prod_region),
        ),
    )

    # 2) add to result if no barriers are placed in between topo_region
    for req in ascending_reqs:
        start = get_topo_location(req.prod_region)
        end = get_topo_location(req.cons_region)
        if any([p in range(start + 1, end) for p in placements]):
            continue

        results.append(req)
        placements.append(end)

    return results


def find_smallest_interval_strategy(
    sync_reqs: Sequence[SyncRequirement],
) -> Sequence[SyncRequirement]:

    get_topo_location = lambda x: getattr(next(iter(x)), "_topo_location", 0)

    if len(sync_reqs) == 0:
        return sync_reqs

    placements = []
    results = []

    # 1) sort barrier placement location from pos low to high
    ascending_reqs = sorted(
        sync_reqs,
        key=lambda req: (
            get_topo_location(req.cons_region),
            get_topo_location(req.prod_region),
        ),
    )

    # 2) add to result if no barriers are placed in between topo_region
    for req in ascending_reqs:
        start = get_topo_location(req.prod_region)
        end = get_topo_location(req.cons_region)
        if any([p in range(start + 1, end) for p in placements]):
            continue

        results.append(req)
        placements.append(end)

    return results
