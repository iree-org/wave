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

    NONE = 0
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


class ResourceAccessWindow:
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
        return MemoryAccessType.READ_WRITE
    else:
        return MemoryAccessType.NONE


def get_shared_memory_from_op(op: CustomOp, depth: int = 0) -> Optional[fx.Node]:
    """
    Given a customOp, returns the shared memory node if it operates on a shared memory region, otherwise None.
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
    if isinstance(op, GatherToLDS):
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
    state: ResourceAccessWindow,
    graph_info: List[fx.Node],
    barrier_type: BarrierType = BarrierType.NONE,
) -> None:
    """
    Add a SyncRequirement to the results list.
    """
    cross_iter = False
    last_prod = state.producers[-1]
    first_con = state.consumers[0]

    if resource is not None and not need_barrier(last_prod, first_con):
        return

    last_prod_loc = last_prod._topo_location
    first_con_loc = first_con._topo_location
    cross_iter = last_prod_loc > first_con_loc

    req = SyncRequirement(
        resource=resource,
        producers=list(state.producers),
        consumers=list(state.consumers),
        is_loop=cross_iter,  # when producer appears after consumer, we identify a loop
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
    nodes: List[fx.Node],
    producer_kinds: Set[MemoryAccessType],
    consumer_kinds: Set[MemoryAccessType],
    barrier_type: BarrierType,
    is_nested: bool = False,
    iterate_region: int = 0,
) -> None:
    """
    Scans the provided list of graph nodes for memory access hazards between producers and consumers,
    and appends any required SyncRequirement objects to the `results` list.
    This function analyzes the sequence of nodes to detect situations where synchronization barriers
    are needed to ensure correct memory access ordering (e.g., between writes and reads to shared resources).
    It handles cross-iteration dependencies by duplicating the node list to
    simulate multiple loop iterations, allowing detection of hazards that span loop boundaries.

    Parameters:
        - results (List[SyncRequirement]): The list to which any detected SyncRequirement objects will be appended.
        - nodes (List[fx.Node]): The ordered list of graph nodes to scan for hazards.
        - producer_kinds (Set[Any]): The set of node kinds considered as producers (e.g., write operations).
        - consumer_kinds (Set[Any]): The set of node kinds considered as consumers (e.g., read operations).
        - barrier_type (BarrierType): The type of barrier to use if a hazard is detected.
        - is_nested (bool, optional): Whether the scan is occurring within a nested region. Defaults to False.
        - iterate_region (int, optional): A split index that marks where the "second" copy of an iterate body begins when we duplicate the body to detect cross-iter hazards.
            Used for detecting cross-iteration dependencies. Defaults to 0 (no duplication).

    Algorithm:
        - Check for the 2nd iterations if `is_nested` flag is set.
        - Iterates through the nodes, tracking producers and consumers for each resource.
        - When a hazard is detected, a SyncRequirement is created and appended to `results`.
        - Uses a window dictionary to track the current set of resource accesses.
        - Handles both intra- and inter-iteration hazards.

    Side Effects:
        Modifies the `results` list in place by appending new SyncRequirement objects as needed.
    """
    if not nodes:
        return

    n = len(nodes)
    iters = 2 if is_nested else 1
    graph_info = [nodes[0], nodes[-1]] if is_nested else [None, None]
    windows: Dict[fx.Node, ResourceAccessWindow] = defaultdict(ResourceAccessWindow)

    def add_hazard_if_window_valid(
        resource: fx.Node, window: ResourceAccessWindow
    ) -> None:
        if not (window.producers and window.consumers):
            return
        add_sync_requirements(
            results,
            resource,
            window,
            graph_info=graph_info,
            barrier_type=barrier_type,
        )
        window.reset()

    for i in range(n * iters):
        idx = i % n
        node = nodes[idx]

        op = get_custom(node)
        access_kind = get_memory_access_type(op)
        if access_kind == MemoryAccessType.NONE:
            continue

        # Depth for loop-carried analysis:
        # - If we don't know the split: iteration = i // n over the virtual 2x pass.
        # - If we do know the split: second copy starts at iterate_region within the body.
        depth = (i // n) if iterate_region == 0 else int(i < iterate_region)
        resource = get_shared_memory_from_op(op, depth)
        assert resource is not None, "op has no smem access"

        hazard_window = windows[resource]

        is_prod = access_kind in producer_kinds
        is_cons = access_kind in consumer_kinds

        if is_prod:
            add_hazard_if_window_valid(resource, hazard_window)
            hazard_window.producers.append(node)

        # Consumers only count after at least one producer for this resource.
        if is_cons and hazard_window.producers:
            hazard_window.consumers.append(node)

    # Final cleanup
    for resource, hazard_window in windows.items():
        add_hazard_if_window_valid(resource, hazard_window)


def get_subgraph_nodes(trace: CapturedTrace, node: fx.Node) -> List[fx.Node]:
    """
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


def get_all_hazards(
    nodes: List[fx.Node],
    is_nested: bool = False,
    iterate_region: int = 0,
) -> List[SyncRequirement]:
    """
    Get all possible hazards (RAW, WAR) from provided set of nodes.
    """
    results: List[SyncRequirement] = []

    # RAW
    handle_hazard(
        results,
        nodes,
        {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
        {MemoryAccessType.READ},
        barrier_type=BarrierType.FILL,
        is_nested=is_nested,
        iterate_region=iterate_region,
    )

    # WAR
    handle_hazard(
        results,
        nodes,
        {MemoryAccessType.READ},
        {MemoryAccessType.WRITE, MemoryAccessType.READ_WRITE},
        barrier_type=BarrierType.READY,
        is_nested=is_nested,
        iterate_region=iterate_region,
    )

    return results


def get_barriers_analysis(
    trace: CapturedTrace, target_arch: TargetConfig
) -> List[SyncRequirement]:
    """
    Analyzes the given computational graph to determine synchronization (barrier) requirements for shared memory accesses, based on the target architecture.
    Args:
        - trace: The traced representation of the computation, expected to provide graph traversal methods such as `preorder_walk` and `walk_graph`.
        - target_arch: The target architecture identifier (e.g., string) used to determine architecture-specific barrier handling.
    Returns: List[SyncRequirement]: A list of synchronization requirements (barriers) needed to ensure correct ordering of shared memory accesses in the graph.
    """
    nodes = trace.preorder_walk()
    assign_preorder_index(nodes)

    is_shared_memory_node = (
        lambda node: get_shared_memory_from_op(get_custom(node)) is not None
    )
    is_iterate_node = lambda node: isinstance(get_custom(node), Iterate)
    is_condition_node = lambda node: isinstance(get_custom(node), Conditional)

    def dfs(
        nodes: List[fx.Node],
        collections: List[fx.Node],
        is_nested: bool = False,
        iterate_region: int = 0,
    ) -> List[SyncRequirement]:
        results: List[SyncRequirement] = []

        for node in nodes:
            if is_shared_memory_node(node):
                collections.append(node)

            if is_iterate_node(node):
                subgraph_nodes = get_subgraph_nodes(trace, node)
                results.extend(dfs(subgraph_nodes, collections))
                collections.clear()
                results.extend(dfs(subgraph_nodes, collections, is_nested=True))
                iterate_region = sum(
                    [is_shared_memory_node(node) for node in subgraph_nodes]
                )

            if is_condition_node(node):
                subgraph_nodes = get_subgraph_nodes(trace, node)
                results.extend(dfs(subgraph_nodes, collections))

        results.extend(get_all_hazards(collections, is_nested, iterate_region))
        return results

    collections: List[fx.Node] = []
    nodes = trace.walk_graph(trace.root_graph)

    return dfs(nodes, collections)


def minimize_placement_strategy(
    sync_reqs: Sequence[SyncRequirement],
) -> Sequence[SyncRequirement]:
    """
    Efficient greedy barrier placement.
        - Forward hazards: O(n log n) sort + O(n) sweep with a single "last_pos".
        - Cross-iter hazards: two O(log m) range checks via binary search over an always-sorted list of chosen placement positions.
    """
    if not sync_reqs:
        return []

    placements_pos: List[int] = []
    results: List[SyncRequirement] = []

    # helpers
    get_location = lambda req: (req.prod_topo_location, req.cons_topo_location)

    def add_placement_at(end_pos: int, req: SyncRequirement) -> None:
        """Add a synchronization requirement to the result list"""
        placements_pos.append(end_pos)
        results.append(req)

    def in_range(ranges: List[int], lo: int, hi: int) -> bool:
        """Return True if there exist a value p in range with lo <= p <= hi."""
        if not ranges or lo > hi:
            return False
        return any([p in range(lo, hi + 1) for p in ranges])

    # 1) sort by (consumer, producer)
    reqs = sorted(
        sync_reqs,
        key=lambda req: (req.cons_topo_location, req.prod_topo_location),
    )

    # ---- Forward hazard (start < end) ----
    # 2) Greedy interval stabbing: pick end if interval not already stabbed by last_position.
    last_pos = -1  # topo location can never be < 0, choose -1 as anchor
    for req in reqs:
        if req.is_loop:
            continue
        start, end = get_location(req)

        # A hazard window is covered if placement is at (start, end]
        # We append to result if this window is not covered.
        if not (last_pos > start and last_pos <= end):
            add_placement_at(end, req)
            last_pos = end

    # ---- Cross-iteration hazards (start > end) ----
    # 3) Handle circular interval.
    # Need to check if any chosen point lies in:
    #   A) [graph_start, end]
    #   B) [start, graph_end]
    # If neither contains a point, place a new one at `end`.
    for req in reqs:
        if not req.is_loop:
            continue

        start, end = get_location(req)
        graph_start, graph_end = (
            req.graph_start._topo_location,
            req.graph_end._topo_location,
        )

        # sanity checks
        assert (
            start > end
        ), "Got producer location < consumer location but identified as cross-iter loop."
        assert (
            graph_start < graph_end
        ), f"Expected graph_start ({graph_start}) position to be less than graph_end ({graph_end}) position."

        is_covered = in_range(placements_pos, start, graph_end) or in_range(
            placements_pos, graph_start, end
        )

        # Already covered by an existing placement on at least one wrapped segment.
        if is_covered:
            continue

        # Otherwise, place at `end` (greedy choice consistent with forward logic)
        add_placement_at(end, req)

    return results


def find_intersecting_interval_strategy(
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

    We add synchronization requirement to the result when current signal position is larger than track-recorded wait position
    - Groups by graph.
    - Normalizes loop-carried hazards by shifting consumer by |body|.
    - Single sweep coalescing intervals.
    - Dedups on emit.
    """

    if not sync_reqs:
        return []

    graph_groups = defaultdict(list)
    results: List[SyncRequirement] = []
    seen = set()

    # --- Helpers ----
    get_location = lambda req: (req.prod_topo_location, req.cons_topo_location)

    def window_is_covered(start: int, end: int, is_cons: int = 0):
        return any(get_location(req)[is_cons] in range(start, end+1) for req in results)

    def add_placement_at(prod: fx.Node, cons: fx.Node) -> None:
        """Add a synchronization requirement to the result list"""
        key = (id(prod), id(cons))
        if key in seen:
            return
        seen.add(key)
        results.append(type(proto)(prod_region=prod, cons_region=cons))

    def sweep(reqs: List[SyncRequirement]) -> bool:
        '''Coalesce forward intervals. Returns True if all placements are intra-graph.'''
        inter_graph = True
        window = None # (current seen largest p,
                      #  current seen smallest c,
                      #  producer at p,
                      #  consumer at c)

        for req in sorted(reqs, key=lambda req: (req.cons_topo_location, req.prod_topo_location)):
            p, c = get_location(req)
            prod, cons = req.prod_region, req.cons_region

            if window is None:
                window = (p, c, prod, cons)
                continue

            max_p, min_c, max_p_prod, min_c_cons = window

            if p > min_c:
                add_placement_at(max_p_prod, min_c_cons)
                inter_graph &= (max_p_prod.graph == min_c_cons.graph)
                window = (p, c, prod, cons)
            else:
                if p > max_p: max_p, max_p_prod = p, prod
                if c < min_c: min_c, min_c_cons = c, cons
                window = (max_p, min_c, max_p_prod, min_c_cons)

        # Final cleanup
        if window is not None:
            _, _, prod, cons = window
            add_placement_at(prod, cons)
            inter_graph &= (prod.graph == cons.graph)

        return inter_graph


    # --- Procedure ----
    # 1) group by graphs
    for req in sync_reqs:
        graph = getattr(req.prod_region, "graph", None) or getattr(req.cons_region, "graph", None)
        graph_groups[graph].append(req)

    # 2) process per graph
    for graph, reqs in graph_groups.items():
        if not reqs: continue

        # prepare requests for 2 passes: forward and loop-carried (needs preprocess)
        forward, loops = [], []
        for req in reqs:
            if req.is_loop: loops.append(req)
            else: forward.append(req)

        # forward pass
        if forward:
            proto = forward[0]
            sweep(forward)

        # loop-carried pass, preprocess all reqs before sweeping
        if loops:
            proto = loops[0]
            norm: List[SyncRequirement]= []
            body_len = len(graph.nodes)

            for req in loops:
                p, c = get_location(req)
                assert p > c, "Cross-iter hazard should have producer location larger than consumer location."
                graph_start, graph_end = (
                    req.graph_start._topo_location,
                    req.graph_end._topo_location,
                )

                cons_is_covered = window_is_covered(graph_start, c, 1)
                prod_is_covered = window_is_covered(p, graph_end, 0)

                if prod_is_covered and cons_is_covered: continue

                # If only cons is covered, snap consumer to loop-exit and treat as forward
                if cons_is_covered and not prod_is_covered:
                    r = _snap_consumer_to_epilog(req, graph)
                    norm.append(r)
                    continue

                # If only producer is covered, skip it
                if prod_is_covered:
                    continue

                # Normalize cross-iter: shift consumer by body length, clear is_loop
                r = _shift_consumer(req, body_len)
                norm.append(r)

            if norm:
                added_before = len(results)
                inter_graph = sweep(norm)
                # Optional loop wrap if we added only intra-graph deps
                if len(results) > added_before and inter_graph and hasattr(graph, "parent_op"):
                    it = graph.parent_op
                    add_placement_at(it.prev, it.next)


    return results

def _shift_consumer(r: SyncRequirement, body_len: int) -> SyncRequirement:
    r = _clone_like(r)
    r.is_loop = False
    r.cons_topo_location = r.cons_topo_location + body_len
    return r

def _snap_consumer_to_loop_exit(r: SyncRequirement, graph: fx.Graph) -> SyncRequirement:
    r = _clone_like(r)
    r.is_loop = False
    r.cons_region = graph.parent_op.next
    r.cons_topo_location = r.cons_region._topo_location
    return r

def _clone_like(r: SyncRequirement) -> SyncRequirement:
    # Lightweight structural clone for fields we rely on.
    cls = type(r)
    return cls(
        resource=r.resource,
        producers=r.producers,
        consumers=r.consumers,
        is_loop=r.is_loop,
        prod_region=r.prod_region,
        cons_region=r.cons_region,
        prod_topo_location=r.prod_topo_location,
        cons_topo_location=r.cons_topo_location,
        graph_start=r.graph_start,
        graph_end=r.graph_end,
        barrier_type=r.barrier_type,

    )
