# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set

import torch.fx as fx

from ..lang.global_symbols import SHARED_ADDRESS_SPACE
from ..ops.wave_ops import (
    AtomicOp,
    CustomOp,
    GatherToLDS,
    Read,
    Iterate,
    Conditional,
    Write,
    get_custom,
)


class BarrierMode(str, Enum):
    """
    LEGACY: Single shared memory barrier
    SPLIT: Simple Split barriers without named barriers
    SPLIT_NAMED: Split barriers with named barriers
    """

    LEGACY = "legacy"
    SPLIT = "split"
    SPLIT_NAMED = "split_named"


@dataclass
class SyncRequirement:
    """
    Syncrhonization requirements in between producers and consumers.
    """

    resource: Any
    producers: Sequence[Any]
    consumers: Sequence[Any]
    next_iter: bool = False
    prod_region: Set[Any] = field(default_factory=set)
    cons_region: Set[Any] = field(default_factory=set)


class EpisodeState:
    __slots__ = ("producers", "consumers")

    def __init__(self):
        self.producers = List[Any] = []
        self.consumers = List[Any] = []


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


def get_shared_memory_from_op(op: CustomOp, depth: int) -> Optional[fx.Node]:
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


def need_barrier(node1: CustomOp, node2: CustomOp) -> bool:
    """
    Check if node1 and node2 have different memory access types.
    If so, we need a barrier in between.
    Else, we don't need a barrier.
    """
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
    results: List[SyncRequirement], resource: fx.Node, state: EpisodeState
):
    """
    Add cut (Synchronization requirements) to the state (in between last producer nodes and first consumer nodes)
    """
    last_prod = state.producers[-1]
    first_con = state.consumers[0]

    req = SyncRequirement(
        resource=resource,
        producers=list(state.producers),
        consumers=list(state.consumers),
        loop=False,  # this will be update when handling iterate subgraph
        prod_region={last_prod},
        cons_region={first_con},
    )

    results.append(req)


def handle_hazard(results, states, nodes, producer_kinds, consumer_kinds):
    states = Dict[Hashable, EpisodeState] = defaultdict(EpisodeState)

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
                add_sync_requirements(results, resource, state)
                state.producers.clear()
                state.consumers.clear()
            state.producers.append(node)

        if access_kind in consumer_kinds:
            if state.producers:
                state.consumers.append(node)

    for resrouce, state in states.items():
        if state.producers and state.consumers:
            add_sync_requirements(results, resource, state)


def get_barriers_analysis(trace, graph, options):

    is_shared_memory_node = lambda node: (
        True if get_shared_memory_from_op(get_custom(node)) is not None else False
    )
    is_subgraph_node = lambda node: (
        True
        if isinstance(get_custom(node), Iterate)
        or isinstance(get_custom(node), Conditional)
        else False
    )

    smem_nodes = trace.walk(is_shared_memory_node)
    subgraph_nodes = trace.walk(is_subgraph_node)

    assign_preorder_index(smem_nodes)

    results = List[SyncRequirement] = []

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

    # handle subgraph requirements

    return results
