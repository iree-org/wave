# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch.fx as fx

from wave_lang.kernel._support.dtype import DataType
from wave_lang.kernel._support.indexing import IndexSymbol
from wave_lang.kernel._support.tracing import CapturedTrace, KernelRegionGraph
from wave_lang.kernel.ops.wave_ops import (
    Iterate,
    NestedRegionOp,
    Output,
    Placeholder,
    get_custom,
)
from tests.unittests.test_utils import add_test_node
from wave_lang.kernel.wave.region_canonicalization import (
    RegionFormat,
    canonicalize_region_captures,
    enable_direct_capture_refs,
    enable_legacy_capture_placeholders,
    prepare_region_captures,
    requires_region_format,
    verify_canonical_region_captures,
)


def _make_trace(main_graph: fx.Graph, subgraph: fx.Graph) -> CapturedTrace:
    region_graph = KernelRegionGraph()
    region_graph.subgraphs["root"] = main_graph
    region_graph.subgraphs[subgraph._name] = subgraph
    return CapturedTrace(region_graph, "root", None)


def _build_nested_region_with_lifted_capture_placeholder() -> (
    tuple[CapturedTrace, fx.Graph, fx.Node]
):
    main_graph = fx.Graph()
    main_graph.subgraphs = {}

    init_arg = add_test_node(main_graph, "init")
    outer_value = add_test_node(main_graph, "captured")

    subgraph = fx.Graph()
    subgraph._name = "iterate_subgraph"

    iterate = Iterate(
        axis=IndexSymbol("M"),
        init_args=[init_arg],
        subgraph_name="iterate_subgraph",
        implicit_captures=[outer_value],
    )
    iterate.add_to_graph(main_graph)
    subgraph.parent_op = iterate.fx_node
    main_graph.subgraphs["iterate_subgraph"] = subgraph

    capture_placeholder = NestedRegionOp.materialize_capture_placeholder(
        subgraph, outer_value
    )
    Output(return_vals=([capture_placeholder],)).add_to_graph(subgraph)
    return _make_trace(main_graph, subgraph), subgraph, outer_value


def _build_captured_trace_with_direct_capture_ref() -> (
    tuple[CapturedTrace, fx.Graph, fx.Node]
):
    main_graph = fx.Graph()
    main_graph._name = "root"
    main_graph.subgraphs = {}

    init_arg = add_test_node(main_graph, "init")
    outer_value = add_test_node(main_graph, "captured")

    subgraph = fx.Graph()
    subgraph._name = "iterate_subgraph"

    iterate = Iterate(
        axis=IndexSymbol("M"),
        init_args=[init_arg],
        subgraph_name="iterate_subgraph",
        implicit_captures=[outer_value],
    )
    iterate.add_to_graph(main_graph)
    subgraph.parent_op = iterate.fx_node

    NestedRegionOp.materialize_capture_placeholder(subgraph, outer_value)
    Output(return_vals=([outer_value],)).add_to_graph(subgraph)

    return _make_trace(main_graph, subgraph), subgraph, outer_value


def _build_nested_region_with_outer_placeholder_capture() -> (
    tuple[CapturedTrace, fx.Graph, fx.Node]
):
    main_graph = fx.Graph()
    main_graph.subgraphs = {}

    init_arg = add_test_node(main_graph, "init")
    outer_value = Placeholder("captured", DataType("f32")).add_to_graph(main_graph)

    subgraph = fx.Graph()
    subgraph._name = "iterate_subgraph"

    iterate = Iterate(
        axis=IndexSymbol("M"),
        init_args=[init_arg],
        subgraph_name="iterate_subgraph",
        implicit_captures=[outer_value],
    )
    iterate.add_to_graph(main_graph)
    subgraph.parent_op = iterate.fx_node
    main_graph.subgraphs["iterate_subgraph"] = subgraph

    capture_placeholder = NestedRegionOp.materialize_capture_placeholder(
        subgraph, outer_value
    )
    Output(return_vals=([capture_placeholder],)).add_to_graph(subgraph)
    return _make_trace(main_graph, subgraph), subgraph, outer_value


def test_enable_legacy_capture_placeholders_rebuilds_local_capture_placeholders():
    """The legacy adapter reintroduces local placeholders for exposed captures."""

    trace, subgraph, outer_value = (
        _build_nested_region_with_lifted_capture_placeholder()
    )

    canonicalize_region_captures(trace)
    verify_canonical_region_captures(trace)

    enable_direct_capture_refs(trace)
    assert subgraph.output_node().args[0][0][0] is outer_value
    assert _count_cross_graph_refs(subgraph) > 0
    with pytest.raises(ValueError, match="Direct outer capture references remain"):
        verify_canonical_region_captures(trace)

    # The legacy-placeholder view restores a local Placeholder for the
    # captured outer value instead of keeping the direct parent-graph reference.
    enable_legacy_capture_placeholders(trace)
    local_capture = subgraph.output_node().args[0][0][0]
    assert local_capture.graph is subgraph
    assert NestedRegionOp.capture_source(local_capture) is outer_value
    assert _count_cross_graph_refs(subgraph) == 0
    verify_canonical_region_captures(trace)


def test_enable_direct_capture_refs_exposes_outer_placeholder_captures():
    """The direct-ref adapter rewrites lifted placeholders back to outer values."""

    trace, subgraph, outer_value = _build_nested_region_with_outer_placeholder_capture()

    canonicalize_region_captures(trace)
    verify_canonical_region_captures(trace)

    # The temporary DIRECT_OUTER_REF view replaces the local lifted placeholder
    # with the actual parent-graph node, which breaks the canonical invariant.
    enable_direct_capture_refs(trace)
    assert subgraph.output_node().args[0][0][0] is outer_value
    assert _count_cross_graph_refs(subgraph) > 0
    with pytest.raises(ValueError, match="Direct outer capture references remain"):
        verify_canonical_region_captures(trace)

    canonicalize_region_captures(trace)
    verify_canonical_region_captures(trace)


def test_canonicalize_region_captures_propagates_vector_shapes():
    """Capture placeholders inherit vector_shapes from the outer source node."""

    trace, subgraph, outer_value = (
        _build_nested_region_with_lifted_capture_placeholder()
    )

    outer_vector_shapes = {IndexSymbol("M"): 16, IndexSymbol("N"): 16}
    outer_value.vector_shapes = outer_vector_shapes

    canonicalize_region_captures(trace)
    verify_canonical_region_captures(trace)

    capture_placeholder = next(
        node
        for node in subgraph.nodes
        if isinstance(get_custom(node), Placeholder) and "lifted" in node.meta
    )
    assert hasattr(
        capture_placeholder, "vector_shapes"
    ), "Capture placeholder should inherit vector_shapes from outer source"
    assert capture_placeholder.vector_shapes == outer_vector_shapes


def test_canonicalize_region_captures_rejects_missing_nested_subgraph():
    """Canonicalization fails fast when a nested region references no subgraph."""

    trace, _, _ = _build_nested_region_with_lifted_capture_placeholder()
    del trace.get_root_graph().subgraphs["iterate_subgraph"]

    with pytest.raises(
        ValueError, match="references missing subgraph iterate_subgraph"
    ):
        canonicalize_region_captures(trace)


def test_canonicalize_region_captures_rejects_unresolvable_legacy_placeholder():
    """Canonicalization rejects legacy placeholders that cannot be resolved."""

    main_graph = fx.Graph()
    main_graph.subgraphs = {}
    init_arg = add_test_node(main_graph, "init")

    subgraph = fx.Graph()
    subgraph._name = "iterate_subgraph"

    iterate = Iterate(
        axis=IndexSymbol("M"),
        init_args=[init_arg],
        subgraph_name="iterate_subgraph",
        implicit_captures=[],
    )
    iterate.add_to_graph(main_graph)
    subgraph.parent_op = iterate.fx_node
    main_graph.subgraphs["iterate_subgraph"] = subgraph

    bogus_capture = Placeholder("bogus", DataType("f32")).add_to_graph(subgraph)
    Output(return_vals=([bogus_capture],)).add_to_graph(subgraph)

    trace = _make_trace(main_graph, subgraph)
    with pytest.raises(
        ValueError, match="Could not resolve legacy capture placeholder bogus"
    ):
        canonicalize_region_captures(trace)


def test_verify_canonical_region_captures_rejects_non_lifted_placeholders():
    main_graph = fx.Graph()
    main_graph.subgraphs = {}
    init_arg = add_test_node(main_graph, "init")

    subgraph = fx.Graph()
    subgraph._name = "iterate_subgraph"

    iterate = Iterate(
        axis=IndexSymbol("M"),
        init_args=[init_arg],
        subgraph_name="iterate_subgraph",
        implicit_captures=[],
    )
    iterate.add_to_graph(main_graph)
    subgraph.parent_op = iterate.fx_node
    main_graph.subgraphs["iterate_subgraph"] = subgraph

    bogus_capture = Placeholder("bogus", DataType("f32")).add_to_graph(subgraph)
    Output(return_vals=([bogus_capture],)).add_to_graph(subgraph)
    trace = _make_trace(main_graph, subgraph)

    with pytest.raises(ValueError, match="Non-lifted region placeholders remain"):
        verify_canonical_region_captures(trace)


def test_verify_canonical_region_captures_rejects_misplaced_placeholders():
    trace, subgraph, _ = _build_nested_region_with_lifted_capture_placeholder()
    capture_placeholder = subgraph.output_node().args[0][0][0]
    local_node = add_test_node(subgraph, "local")
    local_node.append(capture_placeholder)

    with pytest.raises(
        ValueError, match="Canonical capture placeholders must be leading region inputs"
    ):
        verify_canonical_region_captures(trace)


def test_verify_canonical_region_captures_rejects_capture_count_mismatch():
    trace, subgraph, _ = _build_nested_region_with_lifted_capture_placeholder()
    region = get_custom(subgraph.parent_op)
    region.update_arg("implicit_captures", [])

    with pytest.raises(ValueError, match="Capture placeholder count mismatch"):
        verify_canonical_region_captures(trace)


def test_verify_canonical_region_captures_rejects_capture_source_mismatch():
    trace, subgraph, _ = _build_nested_region_with_lifted_capture_placeholder()
    other_outer = add_test_node(trace.get_root_graph(), "other")
    capture_placeholder = subgraph.output_node().args[0][0][0]
    capture_placeholder.meta["lifted"] = other_outer

    with pytest.raises(ValueError, match="Capture placeholder source mismatch"):
        verify_canonical_region_captures(trace)


def _count_cross_graph_refs(graph: fx.Graph) -> int:
    """Count operand references in `graph` that point to nodes from other graphs."""

    count = 0

    def visit(arg):
        nonlocal count
        if isinstance(arg, fx.Node) and arg.graph is not graph:
            count += 1
        return arg

    for node in graph.nodes:
        fx.map_arg(node.args, visit)
        fx.map_arg(node.kwargs, visit)
    return count


def test_requires_region_format_can_skip_post_pass_canonicalization():
    """`canonicalize_output=False` leaves the requested temporary region view in place."""

    @requires_region_format(RegionFormat.DIRECT_OUTER_REF)
    def no_op(trace: CapturedTrace) -> None:
        return None

    trace, subgraph, _ = _build_captured_trace_with_direct_capture_ref()

    # The default wrapper behavior canonicalizes on return, so the temporary
    # DIRECT_OUTER_REF view does not leak out of the decorated call.
    no_op(trace)
    assert _count_cross_graph_refs(subgraph) == 0
    verify_canonical_region_captures(trace)

    # `canonicalize_output=False` skips that final return to canonical form, so
    # the temporary direct outer references remain visible after the call.
    no_op(trace, canonicalize_output=False)
    assert _count_cross_graph_refs(subgraph) > 0
    with pytest.raises(ValueError, match="Direct outer capture references remain"):
        verify_canonical_region_captures(trace)


def test_requires_region_format_rejects_unbound_trace_at_call_time():
    @requires_region_format(RegionFormat.DIRECT_OUTER_REF)
    def no_op(trace: CapturedTrace) -> None:
        return None

    with pytest.raises(TypeError, match="without binding `trace`"):
        no_op()


def test_requires_region_format_rejects_non_trace_argument_at_call_time():
    @requires_region_format(RegionFormat.DIRECT_OUTER_REF)
    def no_op(trace: CapturedTrace) -> None:
        return None

    with pytest.raises(TypeError, match="expected `trace` to be a CapturedTrace"):
        no_op(object())


def test_requires_region_format_rejects_missing_trace_parameter():
    with pytest.raises(TypeError, match="must expose a `trace` parameter"):

        @requires_region_format(RegionFormat.DIRECT_OUTER_REF)
        def no_trace(foo: int) -> None:
            return None


def test_requires_region_format_rejects_multiple_trace_candidates():
    with pytest.raises(TypeError, match="multiple possible trace parameters"):

        @requires_region_format(RegionFormat.DIRECT_OUTER_REF)
        def ambiguous(lhs: CapturedTrace, rhs: CapturedTrace) -> None:
            return None


def test_prepare_region_captures_rejects_unsupported_region_format():
    trace, _, _ = _build_nested_region_with_lifted_capture_placeholder()

    with pytest.raises(ValueError, match="Unsupported region format: invalid"):
        prepare_region_captures(trace, "invalid")
