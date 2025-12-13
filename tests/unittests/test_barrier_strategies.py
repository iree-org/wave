# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# test_barrier_strategies.py

from dataclasses import dataclass
from typing import Optional, List

from wave_lang.kernel.wave.utils.barriers_utils import (
    minimize_placement_strategy,
    find_disjoint_interval_strategy,
)


# Stubs just for these tests
class DummyGraph:
    def __init__(self, name="g"):
        self.name = name
        self.nodes: List["DummyNode"] = []


class DummyNode:
    def __init__(self, topo: int, graph: DummyGraph):
        self._topo_location = topo
        self.graph = graph
        self.prev = None


# helpers
def make_dependency(p: int, c: int, g: DummyGraph):
    p1, c1 = DummyNode(p, g), DummyNode(c, g)
    p1.prev, c1.prev = DummyNode(p - 1, g), DummyNode(c - 1, g)
    req = DummySyncRequirement(
        cross_iter=p > c,
        producer=p1,
        producers=[p1],
        consumer=c1,
        consumers=[c1],
        prod_location=p,
        cons_location=c,
        graph_start=g.nodes[0],
        graph_end=g.nodes[1],
    )
    return req


@dataclass
class DummySyncRequirement:
    # Only fields actually read by the two strategies
    cross_iter: bool
    producer: DummyNode
    producers: list[DummyNode]
    consumer: DummyNode
    consumers: list[DummyNode]
    prod_location: int
    cons_location: int
    graph_start: Optional[DummyNode] = None
    graph_end: Optional[DummyNode] = None


# Tests for minimize_placement_strategy
def test_minimize_placement_empty():
    assert minimize_placement_strategy([]) == []


def test_minimize_placement_overlapping_forward_intervals():
    """
    Two overlapping forward hazards:
      [1,3] and [2,4]

    Greedy "interval stabbing" should place a single barrier
    that covers both, so we keep only the first request.
    """
    g = DummyGraph()
    p1, c1 = DummyNode(1, g), DummyNode(3, g)
    p2, c2 = DummyNode(2, g), DummyNode(4, g)

    req1 = DummySyncRequirement(
        cross_iter=False,
        producer=p1,
        producers=[p1],
        consumer=c1,
        consumers=[c1],
        prod_location=1,
        cons_location=3,
    )
    req2 = DummySyncRequirement(
        cross_iter=False,
        producer=p2,
        producers=[p2],
        consumer=c2,
        consumers=[c2],
        prod_location=2,
        cons_location=4,
    )

    out = minimize_placement_strategy([req1, req2])

    assert len(out) == 1
    # The chosen barrier should correspond to the first interval [1,3]
    assert out[0].prod_location == 1
    assert out[0].cons_location == 3


# Tests for find_disjoint_interval_strategy
def test_find_disjoint_interval_empty():
    assert find_disjoint_interval_strategy([]) == []


def test_find_disjoint_interval_coalesces_overlapping_simple():
    """
    Same two overlapping forward hazards:
      [1,3] and [2,4]

    This strategy should coalesce them into a single barrier
    at the disjoint interval [2,3].
    """
    g = DummyGraph()
    # Graph nodes list only needed for more complex loop logic, but keep it simple
    g.nodes = [None, None]

    req1 = make_dependency(1, 3, g)
    req2 = make_dependency(2, 4, g)

    out = find_disjoint_interval_strategy([req1, req2])

    assert len(out) == 1
    barrier = out[0]
    # Coalesced barrier should be at [2,3]
    assert barrier.prod_location == 2
    assert barrier.cons_location == 3


def test_find_disjoint_interval_coalesces_overlapping_simple_1():
    """
    These are the patterns observed in pipelined kernels
    [15, 20], [17, 18], [27, 36], [29, 34], [35, 57], [37, 39], [161, 170], [169, 184], [163, 168], [171, 190]

    This strategy should coalesce them into barriers at:
    [[17, 18], [29, 34], [37, 39], [163, 168], [171, 184]]
    """
    g = DummyGraph()
    # Graph nodes list only needed for more complex loop logic, but keep it simple
    g.nodes = [None, None]

    deps = [
        [15, 20],
        [17, 18],
        [27, 36],
        [29, 34],
        [35, 57],
        [37, 39],
        [161, 170],
        [169, 184],
        [163, 168],
        [171, 190],
    ]
    reqs = []
    for dep in deps:
        reqs.append(make_dependency(dep[0], dep[1], g))

    out = find_disjoint_interval_strategy(reqs)
    out_intervals = [[reg.prod_location, reg.cons_location] for reg in out]

    assert len(out) == 5
    assert [29, 34] in out_intervals
    assert [37, 39] in out_intervals
    assert [163, 168] in out_intervals
    assert [171, 184] in out_intervals


def test_find_disjoint_interval_coalesces_overlapping_with_loop():
    """
    These are the patterns observed in bshd attention kernels with modulo pipeline
    [[159, 530], [211, 587], [630, 530], [682, 587], [65, 88], [121, 132], [533, 557], [590, 601]]]
                             ^           ^
                             cross-iter  cross-iter

    This strategy should coalesce them into barriers at:
    [[65, 88], [121, 132], [529, 530], [533, 557], [590, 601]]
    """
    rg = DummyGraph()
    g = DummyGraph()
    sg = DummyGraph()
    # Graph nodes list only needed for more complex loop logic, but keep it simple
    sg.nodes = [DummyNode(462, rg), DummyNode(932, rg)]

    deps = [
        [159, 530],
        [211, 587],
        [630, 530],
        [682, 587],
        [65, 88],
        [121, 132],
        [533, 557],
        [590, 601],
    ]

    reqs = []
    for dep in deps:
        reqs.append(make_dependency(dep[0], dep[1], sg))
    for i in range(2):
        reqs[i].producer.graph = g

    out = find_disjoint_interval_strategy(reqs)
    out_intervals = [[reg.prod_location, reg.cons_location] for reg in out]

    assert len(out) == 5
    assert [65, 88] in out_intervals
    assert [121, 132] in out_intervals
    assert [529, 530] in out_intervals
    assert [533, 557] in out_intervals
    assert [590, 601] in out_intervals
