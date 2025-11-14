# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import auto, IntFlag
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union
from functools import partial

import torch.fx as fx

from .graph_utils import propagate_loop_carried_vars

from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import SHARED_ADDRESS_SPACE
from ...ops.wave_ops import (
    AtomicOp,
    CustomOp,
    GatherToLDS,
    TensorLoadToLDS,
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
    prod_location: int = -1
    cons_location: int = -1
    graph_start: fx.Node = None
    graph_end: fx.Node = None


class ResourceAccessWindow:
    def __init__(
        self,
        producers: List[fx.Node] = None,
        consumers: List[fx.Node] = None,
        graph_info: List[fx.Node] = None,
    ):
        self.producers: List[fx.Node] = producers if producers is not None else []
        self.consumers: List[fx.Node] = consumers if consumers is not None else []
        self.graph_info: List[fx.Node] = (
            graph_info if graph_info is not None else [None, None]
        )

    def reset(self):
        self.producers = []
        self.consumers = []
        self.graph_info = [None, None]


class MemoryAccessType(IntFlag):
    """
    Enum to classify memory access operations.
    """

    NONE = 0
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
    elif isinstance(op, TensorLoadToLDS):
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
    if isinstance(op, TensorLoadToLDS):
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
    window: ResourceAccessWindow,
) -> None:
    """
    Add a SyncRequirement to the results list.
    """
    cross_iter = False
    last_prod = window.producers[-1]
    first_con = window.consumers[0]

    if resource is not None and not need_barrier(last_prod, first_con):
        return

    last_prod_loc = last_prod._topo_location
    first_con_loc = first_con._topo_location
    cross_iter = last_prod_loc > first_con_loc

    req = SyncRequirement(
        resource=resource,
        producers=list(window.producers),
        consumers=list(window.consumers),
        is_loop=cross_iter,  # when producer location is greater than consumer location, we identify a loop
        prod_region=last_prod,
        cons_region=first_con,
        prod_location=last_prod_loc,
        cons_location=first_con_loc,
        graph_start=window.graph_info[0],
        graph_end=window.graph_info[1],
    )

    if req in results:
        return

    results.append(req)


def add_hazard_if_window_valid(
    results: List[SyncRequirement], resource: fx.Node, window: ResourceAccessWindow
) -> None:
    """Add a SyncRequirement if producers and consumers are present in current hazard window."""
    if not (window.producers and window.consumers):
        return
    add_sync_requirements(results, resource, window)
    window.reset()


def handle_hazard(
    results: List[SyncRequirement],
    windows: Dict[fx.Node, ResourceAccessWindow],
    node: fx.Node,
    producer_kinds: MemoryAccessType,
    consumer_kinds: MemoryAccessType,
    graph_info: List[fx.Node] = None,
    depth: int = 0,
) -> None:
    """
    Process a single shared-memory node and update hazard tracking state for barrier analysis.

    This function classifies the node's memory access (producer vs. consumer) using
    MemoryAccessType, updates the per-resource hazard window,
    and emits any pending SyncRequirement when a new producer begins a window.

    Behavior:
    - If the node has no memory access or is None, it is ignored.
    - The shared-memory resource is derived from the operation (op) and the iteration depth.
      When a producer is found:
        * Flush any existing window via `add_hazard_if_window_valid(...)` (emits a barrier if needed).
        * Start/extend the producer list for the current resource.
      Consumers are only recorded if there is at least one producer in the current window
      (i.e., they form a potential hazard with prior producers).

    Args:
        results (List[SyncRequirement]):
            Accumulator list where detected synchronization requirements (barriers)
            are appended.
        windows (Dict[fx.Node, ResourceAccessWindow]):
            Mapping from shared-memory resource to its current hazard tracking window.
            Each window typically contains:
              - producers: List[fx.Node]
              - consumers: List[fx.Node]
              - graph_info: Optional[List[fx.Node]] context (e.g., [entry, exit] of subgraph)
        node (fx.Node):
            The current graph node to analyze. If None, the function returns immediately.
        producer_kinds (MemoryAccessType):
            Bitmask of access types that count as producers (e.g., WRITE | READ_WRITE).
        consumer_kinds (MemoryAccessType):
            Bitmask of access types that count as consumers (e.g., READ).
        graph_info (List[fx.Node], optional):
            Optional graph context markers (e.g., subgraph entry/exit nodes). If provided,
            it is stored in the resource window for downstream handling/reporting.
        depth (int, optional):
            Depth or phase used when deriving the shared-memory resource (e.g., for
            nested regions/iteration phases). Defaults to 0.

    Returns:
        None
    """
    if not node:
        return

    op = get_custom(node)
    access_kind = get_memory_access_type(op)
    if access_kind == MemoryAccessType.NONE:
        return

    resource = get_shared_memory_from_op(op, depth)
    assert (
        resource is not None
    ), "Expected op to access shared memory, but no shared memory resource found."

    hazard_window = windows[resource]
    if graph_info is not None:
        hazard_window.graph_info = graph_info

    is_prod = access_kind & producer_kinds
    is_cons = access_kind & consumer_kinds

    if is_prod:
        add_hazard_if_window_valid(results, resource, hazard_window)
        hazard_window.producers.append(node)

    # Consumers only count after at least one producer for this resource.
    if is_cons and hazard_window.producers:
        hazard_window.consumers.append(node)


def get_hazard_handle(
    producer_kinds: MemoryAccessType,
    consumer_kinds: MemoryAccessType,
):
    return partial(
        handle_hazard, producer_kinds=producer_kinds, consumer_kinds=consumer_kinds
    )


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
    all_nodes = trace.preorder_walk()
    assign_preorder_index(all_nodes)

    is_shared_memory_node = (
        lambda node: get_shared_memory_from_op(get_custom(node)) is not None
    )
    is_iterate_node = lambda node: isinstance(get_custom(node), Iterate)
    is_condition_node = lambda node: isinstance(get_custom(node), Conditional)

    get_subgraph_nodes = lambda node: (
        trace.walk_graph(op.subgraph_name)
        if isinstance(op := get_custom(node), NestedRegionOp)
        else []
    )

    results: List[SyncRequirement] = []

    def walk_nodes(nodes, graph_info: Optional[List[fx.Node]] = None, depth: int = 0):
        """
        Traverse a sequence of graph nodes, updating hazard windows and collecting
        synchronization requirements for shared-memory accesses. This function
        handles nested subgraphs for Iterate and Conditional regions.

        Behavior:
        - For each node:
          - If it represents a shared-memory access, invoke `handle(...)` to update
            per-resource access windows and append any detected hazards to `results`.
          - If it is an Iterate region, walk its subgraph twice:
            * First with depth=0 and graph_info marking the subgraph's entry/exit nodes.
            * Then with depth=1 using the same markers (e.g., separate phases/iterations).
          - If it is a Conditional region, walk its subgraph once.
        - After traversing the provided `nodes`, perform a final cleanup pass:
          - For each resource window in `windows`, call `add_hazard_if_window_valid(...)`
            to flush any pending hazards.

        Args:
            nodes (Iterable[fx.Node]): Nodes to traverse in the current graph or subgraph.
            graph_info (List[fx.Node], optional): A pair [entry_node, exit_node] giving
                context for subgraph traversal. Defaults to None.
            depth (int): Phase indicator for Iterate subgraphs (iter i, iter i+1). Defaults to 0.

        Side effects:
            - Mutates `windows` (a mapping of resource -> ResourceAccessWindow).
            - Appends `SyncRequirement` entries to `results` via `handle(...)` and
              `add_hazard_if_window_valid(...)`.
        """
        for node in nodes:
            if is_shared_memory_node(node):
                handle(
                    results=results,
                    windows=windows,
                    node=node,
                    graph_info=graph_info,
                    depth=depth,
                )

            if is_iterate_node(node):
                subgraph_nodes = get_subgraph_nodes(node)
                walk_nodes(
                    subgraph_nodes,
                    graph_info=[subgraph_nodes[0], subgraph_nodes[-1]],
                    depth=0,
                )
                walk_nodes(
                    subgraph_nodes,
                    graph_info=[subgraph_nodes[0], subgraph_nodes[-1]],
                    depth=1,
                )

            if is_condition_node(node):
                subgraph_nodes = get_subgraph_nodes(node)
                walk_nodes(subgraph_nodes)

        for resource, window in windows.items():
            add_hazard_if_window_valid(results, resource, window)

    nodes = trace.get_root_graph().nodes
    # handle WAR
    windows: Dict[fx.Node, ResourceAccessWindow] = defaultdict(ResourceAccessWindow)
    handle = get_hazard_handle(
        MemoryAccessType.READ, MemoryAccessType.WRITE | MemoryAccessType.READ_WRITE
    )
    walk_nodes(nodes)

    # handle RAW
    windows: Dict[fx.Node, ResourceAccessWindow] = defaultdict(ResourceAccessWindow)
    handle = get_hazard_handle(
        MemoryAccessType.WRITE | MemoryAccessType.READ_WRITE, MemoryAccessType.READ
    )
    walk_nodes(nodes)

    return results


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
    get_location = lambda req: (req.prod_location, req.cons_location)

    def place_barrier_at(end_pos: int, req: SyncRequirement) -> None:
        """Add a synchronization requirement to the result list"""
        placements_pos.append(end_pos)
        results.append(req)

    def in_range(ranges: List[int], lo: int, hi: int) -> bool:
        """Return True if there exist a value p in range with lo <= p <= hi."""
        if not ranges or lo > hi:
            return False
        return any(lo <= p <= hi for p in ranges)

    # 1) sort by (consumer, producer)
    reqs = sorted(
        sync_reqs,
        key=lambda req: (req.cons_location, req.prod_location),
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
            place_barrier_at(end, req)
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
        place_barrier_at(end, req)

    return results


def find_intersecting_interval_strategy(
    sync_reqs: Sequence[SyncRequirement],
) -> Sequence[SyncRequirement]:
    """
    def position:
    - node <- smallest
    - node
    - node <- largest

    The algorithm keeps track of the smallest wait position, and
    updates the signal position if a request has
    1) a wait with larger position than tracked wait position, and
    2) a signal with larger position than tracked signal position

    We add synchronization requirement to the result when current signal position is larger than tracked wait position
    - Groups by graph.
    - Normalizes loop-carried hazards by shifting consumer by |body|.
    - Single sweep coalescing intervals.
    - Dedups on emit.
    """

    if not sync_reqs:
        return []

    # --- Helpers ----
    get_location = lambda req: (req.prod_location, req.cons_location)

    def make_request(
        proto: SyncRequirement, prod: fx.Node, cons: fx.Node
    ) -> SyncRequirement:
        """Make a new request based on provided producer and consumer. dont care values are default to None."""
        cls = type(proto)
        return cls(
            is_loop=False,
            prod_region=prod,
            cons_region=cons,
            prod_location=prod._topo_location,
            cons_location=cons._topo_location,
            graph_start=proto.graph_start,
            graph_end=proto.graph_end,
        )

    def shift_consumer(req: SyncRequirement, body_len):
        new_req = make_request(req, req.prod_region, req.cons_region)
        new_req.cons_location += body_len
        return new_req

    def snap_consumer_to_epilog(req: SyncRequirement, graph: fx.Graph):
        new_req = make_request(req, req.prod_region, req.cons_region)
        epilog_cons = graph.parent_op.next
        new_req.cons_region = epilog_cons
        new_req.cons_location = epilog_cons._topo_location
        return new_req

    def window_is_covered(start: int, end: int, index: int = 0):
        """
        `Covered` means that at least one sync requirement's position is inside the window.
        Index determine whether we are checking for producer presence (0) or consumer presence (1).
        """
        return any(
            (start <= get_location(req)[index] and get_location(req)[index] <= end)
            for req in results
        )

    results: List[SyncRequirement] = []
    seen = set()

    def place_barrier_at(proto: SyncRequirement, prod: fx.Node, cons: fx.Node) -> None:
        """Add a synchronization requirement to the result list"""
        key = (id(prod), id(cons))
        if key in seen:
            return
        seen.add(key)
        results.append(make_request(proto, prod, cons))

    # Core sweep (forward intervals only)
    def sweep(reqs: List[SyncRequirement]) -> bool:
        """Coalesce forward intervals. Returns True if all placements are intra-graph."""
        inter_graph = True
        proto = reqs[0]

        # (current seen largest p,
        #  current seen smallest c,
        #  producer at p,
        #  consumer at c)
        window = None

        for req in sorted(reqs, key=lambda req: (req.cons_location, req.prod_location)):
            p, c = get_location(req)
            prod, cons = req.prod_region, req.cons_region

            if window is None:
                window = (p, c, prod, cons)
                continue

            max_p, min_c, max_p_prod, min_c_cons = window

            if p > min_c:
                place_barrier_at(proto, max_p_prod, min_c_cons)
                inter_graph &= max_p_prod.graph == min_c_cons.graph
                window = (p, c, prod, cons)
            else:
                if p > max_p:
                    max_p, max_p_prod = p, prod
                if c < min_c:
                    min_c, min_c_cons = c, cons
                window = (max_p, min_c, max_p_prod, min_c_cons)

        # Final cleanup
        if window is not None:
            _, _, prod, cons = window
            place_barrier_at(proto, prod, cons)
            inter_graph &= prod.graph == cons.graph

        return inter_graph

    # --- Procedure ----
    graph_groups = defaultdict(list)

    # 1) group by graphs
    for req in sync_reqs:
        graph = getattr(req.prod_region, "graph", None) or getattr(
            req.cons_region, "graph", None
        )
        graph_groups[graph].append(req)

    # 2) process per graph
    for graph, reqs in graph_groups.items():
        if not reqs:
            continue

        # prepare requests for 2 passes: forward and loop-carried (needs preprocess)
        forward, loops = [], []
        for req in reqs:
            if req.is_loop:
                loops.append(req)
            else:
                forward.append(req)

        # forward pass
        if forward:
            sweep(forward)

        # loop-carried pass, preprocess all reqs before sweeping
        if loops:
            proto = loops[0]
            norm: List[SyncRequirement] = []
            body_len = len(graph.nodes)

            for req in loops:
                p, c = get_location(req)
                assert (
                    p > c
                ), "Cross-iter hazard should have producer location larger than consumer location."
                graph_start, graph_end = (
                    req.graph_start._topo_location,
                    req.graph_end._topo_location,
                )

                cons_is_covered = window_is_covered(graph_start, c, 1)
                prod_is_covered = window_is_covered(p, graph_end, 0)

                if prod_is_covered and cons_is_covered:
                    continue

                # If only cons is covered, snap consumer to loop-exit and treat as forward
                if cons_is_covered and not prod_is_covered:
                    r = snap_consumer_to_epilog(req, graph)
                    norm.append(r)
                    continue

                # If only producer is covered, skip it
                if prod_is_covered:
                    continue

                # Normalize cross-iter: shift consumer by body length, clear is_loop
                norm.append(shift_consumer(req, body_len))

            if norm:
                added_before = len(results)
                inter_graph = sweep(norm)
                # Optional loop wrap if we added only intra-graph deps
                if (
                    len(results) > added_before
                    and inter_graph
                    and hasattr(graph, "parent_op")
                ):
                    it = graph.parent_op
                    place_barrier_at(proto, it.prev, it.next)

    return results
