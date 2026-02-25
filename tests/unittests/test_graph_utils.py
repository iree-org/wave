# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch.fx as fx
from wave_lang.kernel.wave.utils.graph_utils import (
    is_barrier_between,
    is_barrier_between_same_graph,
    _check_nodes_equivalent,
)
from wave_lang.kernel.ops.wave_ops import (
    SharedMemoryBarrier,
    NewScalar,
    Iterate,
    Conditional,
    Output,
    get_custom,
)
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel._support.indexing import IndexSymbol
from wave_lang.kernel._support.dtype import DataType


def create_simple_graph():
    """Create a simple fx.Graph for testing."""
    graph = fx.Graph()
    return graph


def add_test_node(graph: fx.Graph, name: str) -> fx.Node:
    """
    Add a test node to the graph.

    Args:
        graph: The fx.Graph to add the node to
        name: A name/identifier for the node (used as the value for NewScalar)

    Returns:
        The created fx.Node
    """
    # Create a NewScalar node with a unique float value based on name hash
    # This ensures each node has a distinct value while being deterministic
    value = float(hash(name) % 1000)
    node = NewScalar(value=value, dtype=DataType("f32"))
    node.add_to_graph(graph)
    return node.fx_node


def add_barrier_node(graph: fx.Graph) -> fx.Node:
    """Add a SharedMemoryBarrier node to the graph."""
    barrier = SharedMemoryBarrier()
    barrier.add_to_graph(graph)
    return barrier.fx_node


def create_nested_graph_with_iterate() -> tuple[fx.Graph, fx.Graph, fx.Node]:
    """
    Create a nested graph structure with an Iterate node.
    Returns (main_graph, subgraph, iterate_node).
    """
    main_graph = fx.Graph()
    main_graph.subgraphs = {}

    # Create nodes in main graph
    node_before = add_test_node(main_graph, "before_iterate")

    # Create subgraph for iterate
    subgraph = fx.Graph()
    subgraph._name = "iterate_subgraph"

    # Create an Iterate node
    M = IndexSymbol("M")
    iterate = Iterate(
        axis=M,
        init_args=[node_before],
        subgraph_name="iterate_subgraph",
        implicit_captures=[],
    )
    iterate.add_to_graph(main_graph)
    iterate_node = iterate.fx_node

    # Link subgraph to iterate node
    subgraph.parent_op = iterate_node
    main_graph.subgraphs["iterate_subgraph"] = subgraph

    return main_graph, subgraph, iterate_node


def create_nested_graph_with_conditional() -> tuple[fx.Graph, fx.Graph, fx.Node]:
    """
    Create a nested graph structure with a Conditional node.
    Returns (main_graph, subgraph, conditional_node).
    """
    main_graph = fx.Graph()
    main_graph.subgraphs = {}

    # Create condition node in main graph
    condition = add_test_node(main_graph, "condition")

    # Create subgraph for conditional
    subgraph = fx.Graph()
    subgraph._name = "conditional_subgraph"

    # Create a Conditional node
    conditional = Conditional(
        condition=condition, subgraph_name="conditional_subgraph", implicit_captures=[]
    )
    conditional.add_to_graph(main_graph)
    conditional_node = conditional.fx_node

    # Link subgraph to conditional node
    subgraph.parent_op = conditional_node
    main_graph.subgraphs["conditional_subgraph"] = subgraph

    return main_graph, subgraph, conditional_node


class TestIsBarrierBetweenSameGraph:
    """Tests for is_barrier_between_same_graph helper function."""

    def test_basic_barrier_detection(self):
        """Test basic barrier detection in same graph."""
        graph = create_simple_graph()
        node1 = add_test_node(graph, "a")
        barrier = add_barrier_node(graph)
        node2 = add_test_node(graph, "b")

        result = is_barrier_between_same_graph(node1, node2)
        assert result is not None
        assert result == barrier

    def test_no_barrier_same_graph(self):
        """Test no barrier between adjacent nodes in same graph."""
        graph = create_simple_graph()
        node1 = add_test_node(graph, "a")
        node2 = add_test_node(graph, "b")

        result = is_barrier_between_same_graph(node1, node2)
        assert result is None


class TestIsBarrierBetweenNestedGraphs:
    """Tests for is_barrier_between with nodes in different nested graphs."""

    def test_parent_to_child_no_barrier(self):
        """Test nodes from parent graph to child graph without barrier."""
        main_graph, subgraph, iterate_node = create_nested_graph_with_iterate()

        # Create nodes
        parent_node = add_test_node(main_graph, "parent")
        child_node = add_test_node(subgraph, "child")

        # No barrier between parent and child
        result = is_barrier_between(parent_node, child_node)
        assert result is None

    def test_parent_to_child_with_barrier_in_child_beginning(self):
        """Test barrier at beginning of child graph."""
        main_graph, subgraph, iterate_node = create_nested_graph_with_iterate()

        # Parent is the node created before iterate
        parent_node = list(main_graph.nodes)[0]  # Get the "before_iterate" node

        # Add barrier and a placeholder before it to establish context
        first_child = add_test_node(subgraph, "first")
        barrier = add_barrier_node(subgraph)
        child_node = add_test_node(subgraph, "child")

        # Barrier between first_child and child_node in subgraph
        result = is_barrier_between(parent_node, child_node)
        assert result is not None
        assert result == barrier

    def test_child_to_parent_no_barrier(self):
        """Test nodes from child graph back to parent graph without barrier."""
        main_graph, subgraph, iterate_node = create_nested_graph_with_iterate()

        child_node = add_test_node(subgraph, "child")
        parent_node = add_test_node(main_graph, "parent_after")

        # No barrier between child and parent
        result = is_barrier_between(child_node, parent_node)
        assert result is None

    def test_child_to_parent_with_barrier_in_child(self):
        """Test barrier in child graph when going from child to parent."""
        main_graph, subgraph, iterate_node = create_nested_graph_with_iterate()

        child_node1 = add_test_node(subgraph, "child1")
        barrier = add_barrier_node(subgraph)
        child_node2 = add_test_node(subgraph, "child2")
        parent_node = add_test_node(main_graph, "parent_after")

        # Barrier in child should be detected when exiting to parent
        result = is_barrier_between(child_node1, parent_node)
        assert result is not None
        assert result == barrier

    def test_child_to_parent_with_barrier_in_parent(self):
        """Test barrier in parent graph after nested region."""
        main_graph, subgraph, iterate_node = create_nested_graph_with_iterate()

        child_node = add_test_node(subgraph, "child")
        # iterate_node already exists
        barrier = add_barrier_node(main_graph)
        parent_node = add_test_node(main_graph, "parent_after")

        # Barrier in parent after iterate should be detected
        result = is_barrier_between(child_node, parent_node)
        assert result is not None
        assert result == barrier

    def test_nested_iterate_two_levels(self):
        """Test deeply nested graphs with barriers at different levels."""
        main_graph = fx.Graph()
        main_graph.subgraphs = {}

        # Level 0: Main graph
        node_main = add_test_node(main_graph, "main")

        # Level 1: First iterate
        subgraph1 = fx.Graph()
        subgraph1._name = "iterate1"
        subgraph1.subgraphs = {}
        M = IndexSymbol("M")
        iterate1 = Iterate(
            axis=M,
            init_args=[node_main],
            subgraph_name="iterate1",
            implicit_captures=[],
        )
        iterate1.add_to_graph(main_graph)
        subgraph1.parent_op = iterate1.fx_node
        main_graph.subgraphs["iterate1"] = subgraph1

        node_level1 = add_test_node(subgraph1, "level1")
        barrier_level1 = add_barrier_node(subgraph1)

        # Level 2: Nested iterate inside first iterate
        subgraph2 = fx.Graph()
        subgraph2._name = "iterate2"
        N = IndexSymbol("N")
        iterate2 = Iterate(
            axis=N,
            init_args=[node_level1],
            subgraph_name="iterate2",
            implicit_captures=[],
        )
        iterate2.add_to_graph(subgraph1)
        subgraph2.parent_op = iterate2.fx_node
        subgraph1.subgraphs["iterate2"] = subgraph2

        node_level2 = add_test_node(subgraph2, "level2")

        # Test: barrier at level 1 should be detected from main to level 2
        result = is_barrier_between(node_main, node_level2)
        assert result is not None
        assert result == barrier_level1

    def test_sibling_subgraphs(self):
        """Test nodes in sibling subgraphs (both children of same parent)."""
        main_graph = fx.Graph()
        main_graph.subgraphs = {}

        # Create first iterate (sibling 1)
        node_main1 = add_test_node(main_graph, "main1")
        subgraph1 = fx.Graph()
        subgraph1._name = "iterate1"
        M = IndexSymbol("M")
        iterate1 = Iterate(
            axis=M,
            init_args=[node_main1],
            subgraph_name="iterate1",
            implicit_captures=[],
        )
        iterate1.add_to_graph(main_graph)
        subgraph1.parent_op = iterate1.fx_node
        main_graph.subgraphs["iterate1"] = subgraph1

        node_sibling1 = add_test_node(subgraph1, "sibling1")

        # Add barrier in parent BETWEEN the two iterates
        barrier_main = add_barrier_node(main_graph)

        # Create second iterate (sibling 2) AFTER the barrier
        node_main2 = add_test_node(main_graph, "main2")
        subgraph2 = fx.Graph()
        subgraph2._name = "iterate2"
        N = IndexSymbol("N")
        iterate2 = Iterate(
            axis=N,
            init_args=[node_main2],
            subgraph_name="iterate2",
            implicit_captures=[],
        )
        iterate2.add_to_graph(main_graph)
        subgraph2.parent_op = iterate2.fx_node
        main_graph.subgraphs["iterate2"] = subgraph2

        node_sibling2 = add_test_node(subgraph2, "sibling2")

        # Check if barrier is detected between siblings
        result = is_barrier_between(node_sibling1, node_sibling2)
        assert result is not None
        assert result == barrier_main


class TestOutputComparison:
    """Tests for Output node equivalence checking."""

    def test_different_return_value_count(self):
        """Outputs returning different numbers of values are not equivalent."""
        graph_a = fx.Graph()
        val_a1 = add_test_node(graph_a, "x")
        val_a2 = add_test_node(graph_a, "y")
        out_a = Output(return_vals=([val_a1, val_a2],))
        out_a.add_to_graph(graph_a)

        graph_b = fx.Graph()
        val_b = add_test_node(graph_b, "x")
        out_b = Output(return_vals=([val_b],))
        out_b.add_to_graph(graph_b)

        node_map = {val_a1: val_b}
        result = _check_nodes_equivalent(
            get_custom(out_a.fx_node),
            get_custom(out_b.fx_node),
            None,
            None,
            None,
            node_map,
        )
        assert not result

    def test_different_return_value(self):
        """Outputs returning different nodes are not equivalent."""
        graph_a = fx.Graph()
        val_a1 = add_test_node(graph_a, "x")
        val_a2 = add_test_node(graph_a, "y")
        out_a = Output(return_vals=([val_a1],))
        out_a.add_to_graph(graph_a)

        graph_b = fx.Graph()
        val_b1 = add_test_node(graph_b, "x")
        val_b2 = add_test_node(graph_b, "y")
        out_b = Output(return_vals=([val_b2],))
        out_b.add_to_graph(graph_b)

        node_map = {val_a1: val_b1, val_a2: val_b2}
        result = _check_nodes_equivalent(
            get_custom(out_a.fx_node),
            get_custom(out_b.fx_node),
            None,
            None,
            None,
            node_map,
        )
        assert not result

    def test_single_vs_list_wrapped_return(self):
        """Single-return and list-wrapped single-return are equivalent."""
        graph_a = fx.Graph()
        val_a = add_test_node(graph_a, "x")
        out_node_a = graph_a.create_node("output", target="output", args=([val_a],))

        graph_b = fx.Graph()
        val_b = add_test_node(graph_b, "x")
        out_node_b = graph_b.create_node("output", target="output", args=(val_b,))

        node_map = {val_a: val_b}
        result = _check_nodes_equivalent(
            get_custom(out_node_a),
            get_custom(out_node_b),
            None,
            None,
            None,
            node_map,
        )
        assert result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
