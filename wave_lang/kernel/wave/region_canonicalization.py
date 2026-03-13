# Copyright 2026 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import inspect
from enum import Enum
from typing import Callable

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    Conditional,
    IterArg,
    Iterate,
    NestedRegionOp,
    NewRegister,
    Placeholder,
    get_custom,
)


class RegionFormat(str, Enum):
    """Region representation required by a pass.

    `ISOLATED` is the canonical/default format of the compilation pipeline, so
    passes that operate on the canonical form do not need an annotation. See
    `docs/wave/canonical_ir_format.md` for the structural definition of each
    supported region form.
    """

    ISOLATED = "canonical"
    LEGACY_PLACEHOLDERS = "legacy_capture_placeholders"
    DIRECT_OUTER_REF = "direct_capture_refs"
    SCHEDULE_SIGNATURE_PLACEHOLDERS = "schedule_signature_placeholders"


_REQUIRED_REGION_FORMAT_ATTR = "_wave_required_region_format"
_RAW_GRAPH_PASS_ATTR = "_wave_raw_graph_pass"
_CANONICALIZE_OUTPUT_KWARG = "canonicalize_output"


def _iter_nested_regions(graph: fx.Graph):
    """Yield each nested region op together with its subgraph."""
    for node in graph.nodes:
        custom = get_custom(node)
        if not isinstance(custom, NestedRegionOp):
            continue
        root_subgraphs = custom.get_root_graph().subgraphs
        if custom.subgraph_name not in root_subgraphs:
            raise ValueError(
                f"Nested region {custom.fx_node.name} references missing subgraph "
                f"{custom.subgraph_name}"
            )
        subgraph = root_subgraphs[custom.subgraph_name]
        yield custom, subgraph
        yield from _iter_nested_regions(subgraph)


def _replace_direct_capture_aliases(
    graph: fx.Graph, source: fx.Node, replacement: fx.Node
) -> None:
    """Rewrite direct outer capture references to a local replacement node."""
    # Some region-local users may still point at the outer value directly.
    # Rewrite just those aliases to the new local placeholder.
    for user in graph.nodes:

        def rewrite(arg):
            if not isinstance(arg, fx.Node):
                return arg
            if arg.graph is graph:
                return arg
            if NestedRegionOp.capture_source(arg) is source:
                return replacement
            return arg

        user._update_args_kwargs(
            fx.map_arg(user.args, rewrite),
            fx.map_arg(user.kwargs, rewrite),
        )


def _collect_capture_sources(
    region: NestedRegionOp, subgraph: fx.Graph
) -> list[fx.Node]:
    """Collect outer capture sources in stable signature order."""
    captures: list[fx.Node] = []
    seen: set[fx.Node] = set()

    def remember(source: fx.Node):
        source = region.capture_source(source)
        if source in seen:
            return
        seen.add(source)
        captures.append(source)

    for source in region.implicit_captures:
        remember(source)
    ordered_placeholders = [
        node
        for node in subgraph.nodes
        if isinstance(get_custom(node), Placeholder)
        and not isinstance(get_custom(node), IterArg)
    ]
    for local_capture_idx, local_capture in enumerate(ordered_placeholders):
        source = NestedRegionOp.capture_source(local_capture)
        if source is local_capture or source.graph is subgraph:
            continue
        resolved = _try_resolve_legacy_capture_source(
            region, subgraph, local_capture, local_capture_idx
        )
        if resolved is not None:
            remember(resolved)
    for node in subgraph.nodes:
        custom = get_custom(node)
        if isinstance(custom, (IterArg, Placeholder)):
            continue
        source = NestedRegionOp.capture_source(node)
        if source is node or source.graph is subgraph:
            continue
        remember(source)
    for source in _collect_direct_capture_uses(subgraph):
        remember(source)
    return captures


def _collect_direct_capture_uses(graph: fx.Graph) -> list[fx.Node]:
    """Return outer values referenced directly in `graph`."""
    captures: list[fx.Node] = []
    seen: set[fx.Node] = set()

    def visit(arg):
        if not isinstance(arg, fx.Node):
            return arg
        if arg.graph is graph:
            return arg
        source = NestedRegionOp.capture_source(arg)
        if source.graph is graph or source in seen:
            return arg
        seen.add(source)
        captures.append(source)
        return arg

    for nested_node in graph.nodes:
        custom = get_custom(nested_node)
        if isinstance(custom, Iterate):
            fx.map_arg((custom.init_args, custom.start, custom.condition), visit)
            continue
        if isinstance(custom, Conditional):
            fx.map_arg((custom.condition, custom.else_return), visit)
            continue
        fx.map_arg((nested_node.args, nested_node.kwargs), visit)
    return captures


def _build_capture_lookup(
    region: NestedRegionOp, subgraph: fx.Graph
) -> tuple[dict[fx.Node, fx.Node], list[fx.Node]]:
    """Build direct-ref-aware capture lookup data for one region subgraph."""
    by_outer: dict[fx.Node, fx.Node] = {}
    for var in region.captured_vars(subgraph):
        # Keep the first local representative we encounter for each outer
        # source. Canonicalization collapses duplicates later, so lookup only
        # needs one stable local binding per outer capture.
        by_outer.setdefault(region.capture_source(var), var)
    direct_sources_in_order = [
        region.capture_source(source)
        for source in _collect_direct_capture_uses(subgraph)
    ]
    for source in direct_sources_in_order:
        by_outer.setdefault(source, source)
    return by_outer, direct_sources_in_order


def _try_resolve_legacy_capture_source(
    region: NestedRegionOp, subgraph: fx.Graph, local_capture: fx.Node, capture_idx: int
) -> fx.Node | None:
    """Try to map a legacy local capture node back to its outer source."""
    source = NestedRegionOp.capture_source(local_capture)
    if source.graph is not subgraph:
        return region.capture_source(source)

    for implicit_capture in region.implicit_captures:
        outer_source = region.capture_source(implicit_capture)
        if outer_source.name == local_capture.name:
            return outer_source

    # Legacy placeholder-only regions may lose explicit outer-source links.
    # In that representation the placeholder prefix still follows capture
    # signature order, so placeholder position is a safe final fallback. Keep
    # this path narrow to legacy placeholders rather than applying it to all
    # lifted local nodes.
    if isinstance(get_custom(local_capture), Placeholder) and capture_idx < len(
        region.implicit_captures
    ):
        return region.capture_source(region.implicit_captures[capture_idx])

    return None


def _ensure_capture_placeholder(
    region: NestedRegionOp, subgraph: fx.Graph, source: fx.Node
) -> fx.Node:
    """Ensure `source` is represented by a lifted placeholder in `subgraph`."""
    outer_source = region.capture_source(source)
    local_capture = region.get_captured_fx_node(
        subgraph, outer_source, lookup=_build_capture_lookup(region, subgraph)
    )
    if local_capture is None:
        return region.materialize_capture_placeholder(subgraph, outer_source)
    if local_capture is outer_source:
        placeholder = region.materialize_capture_placeholder(
            subgraph, outer_source, getattr(outer_source, "location", None)
        )
        _replace_direct_capture_aliases(subgraph, outer_source, placeholder)
        return placeholder
    local_capture.meta["lifted"] = outer_source
    return local_capture


def _canonicalize_nested_region(region: NestedRegionOp, subgraph: fx.Graph) -> None:
    """Rewrite one nested region into the canonical isolated form."""
    sources = _collect_capture_sources(region, subgraph)
    region.update_arg("implicit_captures", sources)

    # Gather every legacy local representative for each outer capture source so
    # they can be collapsed to a single canonical placeholder.
    legacy_capture_nodes: dict[fx.Node, list[fx.Node]] = {}
    ordered_placeholders = [
        node
        for node in subgraph.nodes
        if isinstance(get_custom(node), Placeholder)
        and not isinstance(get_custom(node), IterArg)
    ]
    for idx, local_capture in enumerate(ordered_placeholders):
        source = _try_resolve_legacy_capture_source(
            region, subgraph, local_capture, idx
        )
        if source is None:
            raise ValueError(
                f"Could not resolve legacy capture placeholder {local_capture.name} "
                f"in {region.subgraph_name} to an outer source"
            )
        legacy_capture_nodes.setdefault(source, []).append(local_capture)
    for local_capture in subgraph.nodes:
        custom = get_custom(local_capture)
        if isinstance(custom, IterArg):
            continue
        source = NestedRegionOp.capture_source(local_capture)
        if source is local_capture or source.graph is subgraph:
            continue
        if isinstance(custom, Placeholder):
            continue
        legacy_capture_nodes.setdefault(source, []).append(local_capture)

    anchor = subgraph._root
    for node in subgraph.nodes:
        if isinstance(get_custom(node), IterArg):
            anchor = node
            continue
        break
    for source in sources:
        legacy = legacy_capture_nodes.get(source, [])
        template = legacy[0] if legacy else None
        location = getattr(template, "location", None) if template else None
        canonical = next(
            (
                node
                for node in legacy
                if isinstance(get_custom(node), Placeholder)
                and not isinstance(get_custom(node), IterArg)
            ),
            None,
        )
        if canonical is None:
            canonical = region.materialize_capture_placeholder(
                subgraph, source, location
            )

        # Canonical capture placeholders must form the leading non-IterArg
        # region input prefix in signature order. In `torch.fx`, `append`
        # moves an existing node in place instead of duplicating it, so this
        # handles both newly created and reused placeholders.
        anchor.append(canonical)
        anchor = canonical
        canonical.meta["lifted"] = source

        _replace_direct_capture_aliases(subgraph, source, canonical)
        for legacy_capture in legacy:
            if legacy_capture is canonical:
                continue
            get_custom(legacy_capture).replace_uses_with(
                canonical, graph=subgraph, propagate_location=False
            )
            get_custom(legacy_capture).erase()

    region.refresh_captures(subgraph)


def canonicalize_region_captures(trace: CapturedTrace) -> None:
    """Canonicalize capture handling for every nested region in `trace`."""
    root_graph = trace.get_root_graph()
    for region, subgraph in _iter_nested_regions(root_graph):
        _canonicalize_nested_region(region, subgraph)


def verify_canonical_region_captures(trace: CapturedTrace, where: str = "") -> None:
    """Check that all nested regions satisfy the canonical capture invariant."""
    root_graph = trace.get_root_graph()
    context = f" ({where})" if where else ""
    for region, subgraph in _iter_nested_regions(root_graph):
        direct_uses = _collect_direct_capture_uses(subgraph)
        if direct_uses:
            raise ValueError(
                f"Direct outer capture references remain in {region.subgraph_name}{context}: "
                + ", ".join(node.name for node in direct_uses)
            )

        capture_placeholders = [
            node
            for node in subgraph.nodes
            if isinstance(get_custom(node), Placeholder)
            and not isinstance(get_custom(node), IterArg)
        ]
        non_lifted_placeholders = [
            node for node in capture_placeholders if "lifted" not in node.meta
        ]
        if non_lifted_placeholders:
            raise ValueError(
                f"Non-lifted region placeholders remain in {region.subgraph_name}{context}: "
                + ", ".join(node.name for node in non_lifted_placeholders)
            )

        expected_prefix = []
        for node in subgraph.nodes:
            custom = get_custom(node)
            if isinstance(custom, IterArg):
                continue
            if isinstance(custom, Placeholder):
                expected_prefix.append(node)
                continue
            break
        if len(expected_prefix) != len(capture_placeholders):
            misplaced = [
                node for node in capture_placeholders if node not in expected_prefix
            ]
            raise ValueError(
                f"Canonical capture placeholders must be leading region inputs in "
                f"{region.subgraph_name}{context}: "
                + ", ".join(node.name for node in misplaced)
            )

        if len(expected_prefix) != len(region.implicit_captures):
            raise ValueError(
                f"Capture placeholder count mismatch in {region.subgraph_name}{context}: "
                f"{len(expected_prefix)} local placeholders vs "
                f"{len(region.implicit_captures)} implicit captures"
            )

        for source, capture_placeholder in zip(
            region.implicit_captures, expected_prefix
        ):
            if NestedRegionOp.capture_source(
                capture_placeholder
            ) is not region.capture_source(source):
                raise ValueError(
                    f"Capture placeholder source mismatch in {region.subgraph_name}{context}: "
                    f"{capture_placeholder.name} does not match the parent capture signature"
                )


def enable_legacy_capture_placeholders(trace: CapturedTrace) -> None:
    """Convert canonical captures to the legacy placeholder-based view."""
    root_graph = trace.get_root_graph()
    for region, subgraph in _iter_nested_regions(root_graph):
        sources = _collect_capture_sources(region, subgraph)
        region.update_arg("implicit_captures", sources)
        for source in sources:
            _ensure_capture_placeholder(region, subgraph, source)


def enable_direct_capture_refs(trace: CapturedTrace) -> None:
    """Convert lifted placeholders back into direct outer references."""

    def erase_unused_capture_sources(
        sources: list[fx.Node], subgraph: fx.Graph
    ) -> None:
        """Erase dead outer capture sources that became redundant after rewriting."""

        for source in sources:
            outer_source = NestedRegionOp.capture_source(source)
            if outer_source.graph is subgraph or outer_source.users:
                continue
            if isinstance(get_custom(outer_source), Placeholder):
                continue
            if isinstance(get_custom(outer_source), NestedRegionOp):
                continue
            if isinstance(get_custom(outer_source), NewRegister):
                continue
            get_custom(outer_source).erase()

    root_graph = trace.get_root_graph()
    for region, subgraph in _iter_nested_regions(root_graph):
        sources = _collect_capture_sources(region, subgraph)
        replacements = {
            local_capture: region.capture_source(source)
            for source in sources
            if (local_capture := region.get_captured_fx_node(subgraph, source))
            is not None
            and local_capture is not region.capture_source(source)
        }
        for user in subgraph.nodes:
            changed = False

            def rewrite(arg):
                nonlocal changed
                if isinstance(arg, fx.Node) and arg in replacements:
                    changed = True
                    return replacements[arg]
                return arg

            new_args = fx.map_arg(user.args, rewrite)
            new_kwargs = fx.map_arg(user.kwargs, rewrite)
            if changed:
                user._update_args_kwargs(new_args, new_kwargs)
        for local_capture in replacements:
            subgraph.erase_node(local_capture)
        region.refresh_captures(
            subgraph, lookup=_build_capture_lookup(region, subgraph)
        )
        erase_unused_capture_sources(sources, subgraph)


def enable_schedule_signature_placeholders(trace: CapturedTrace) -> None:
    """Keep placeholders only for schedule-signature sources."""
    root_graph = trace.get_root_graph()
    for region, subgraph in _iter_nested_regions(root_graph):
        all_sources = _collect_capture_sources(region, subgraph)
        by_outer, direct_sources_in_order = _build_capture_lookup(region, subgraph)
        signature_sources = [
            source
            for source in all_sources
            if isinstance(get_custom(source), (Placeholder, NewRegister))
        ]
        signature_source_set = set(signature_sources)
        for source in all_sources:
            outer_source = region.capture_source(source)
            if outer_source in signature_source_set:
                by_outer[outer_source] = _ensure_capture_placeholder(
                    region, subgraph, outer_source
                )
                continue
            local_capture = by_outer.get(outer_source)
            if local_capture is None or local_capture is outer_source:
                continue
            get_custom(local_capture).replace_uses_with(
                outer_source, graph=subgraph, propagate_location=False
            )
            get_custom(local_capture).erase()
            by_outer[outer_source] = outer_source
        region.update_arg("implicit_captures", signature_sources)


def graph_pass_region_mode(graph_pass: Callable) -> RegionFormat:
    """Return the declared region format for a graph pass."""
    graph_pass = raw_graph_pass(graph_pass)
    graph_pass_fn = (
        graph_pass.func if isinstance(graph_pass, functools.partial) else graph_pass
    )
    return getattr(graph_pass_fn, _REQUIRED_REGION_FORMAT_ATTR, RegionFormat.ISOLATED)


def raw_graph_pass(graph_pass: Callable) -> Callable:
    """Unwrap a decorated, partial, or bound pass to its raw implementation."""
    if isinstance(graph_pass, functools.partial):
        raw_partial = functools.partial(
            raw_graph_pass(graph_pass.func),
            *graph_pass.args,
            **(graph_pass.keywords or {}),
        )
        setattr(
            raw_partial,
            "__name__",
            getattr(raw_partial.func, "__name__", type(raw_partial.func).__name__),
        )
        return raw_partial
    if inspect.ismethod(graph_pass):
        raw_func = getattr(
            graph_pass.__func__, _RAW_GRAPH_PASS_ATTR, graph_pass.__func__
        )
        return raw_func.__get__(graph_pass.__self__, type(graph_pass.__self__))
    return getattr(graph_pass, _RAW_GRAPH_PASS_ATTR, graph_pass)


def requires_region_format(
    region_format: RegionFormat,
) -> Callable[[Callable], Callable]:
    """Wrap a graph pass so it runs in the requested region format.

    The decorated pass must expose a uniquely identifiable `trace` parameter,
    either by name or by annotating exactly one parameter as `CapturedTrace`.
    The wrapped pass enforces the full region-format boundary contract:
    it canonicalizes the input trace, enables the requested temporary region
    format for the pass body, and then canonicalizes the trace again after the
    pass returns so the rest of the pipeline always sees canonical
    `RegionFormat.ISOLATED` regions.

    By default, the wrapper canonicalizes the pass output before returning. The
    wrapper also accepts a reserved keyword argument, `canonicalize_output=False`,
    which disables the return to canonical form for one specific invocation.
    This is useful for compatibility with legacy FileCheck tests.
    """

    def decorator(graph_pass: Callable) -> Callable:
        trace_parameter_name = _resolve_trace_parameter_name(graph_pass)
        graph_pass_signature = inspect.signature(graph_pass)
        setattr(graph_pass, _REQUIRED_REGION_FORMAT_ATTR, region_format)

        @functools.wraps(graph_pass)
        def wrapped(*args, **kwargs):
            canonicalize_output_enabled = kwargs.pop(_CANONICALIZE_OUTPUT_KWARG, True)
            if not isinstance(canonicalize_output_enabled, bool):
                raise TypeError(
                    f"{_CANONICALIZE_OUTPUT_KWARG} must be a bool, got "
                    f"{type(canonicalize_output_enabled).__name__}"
                )
            bound_args = graph_pass_signature.bind_partial(*args, **kwargs)
            if trace_parameter_name not in bound_args.arguments:
                raise TypeError(
                    f"Decorated pass {graph_pass.__name__} was called "
                    f"without binding `{trace_parameter_name}`"
                )
            trace = bound_args.arguments[trace_parameter_name]
            if not isinstance(trace, CapturedTrace):
                raise TypeError(
                    f"Decorated pass {graph_pass.__name__} expected "
                    f"`{trace_parameter_name}` to be a CapturedTrace, got "
                    f"{type(trace).__name__}"
                )
            prepare_region_captures_for_pass(trace, graph_pass, region_format)
            result = graph_pass(*args, **kwargs)
            if canonicalize_output_enabled:
                canonicalize_region_captures(trace)
            return result

        setattr(wrapped, _REQUIRED_REGION_FORMAT_ATTR, region_format)
        setattr(wrapped, _RAW_GRAPH_PASS_ATTR, graph_pass)
        return wrapped

    return decorator


def prepare_region_captures_for_pass(
    trace: CapturedTrace,
    graph_pass: Callable,
    mode: RegionFormat | None = None,
) -> RegionFormat:
    """Canonicalize `trace` and enable the region view required by a pass."""
    mode = graph_pass_region_mode(graph_pass) if mode is None else mode
    prepare_region_captures(trace, mode)
    return mode


def prepare_region_captures(trace: CapturedTrace, mode: RegionFormat) -> None:
    """Canonicalize `trace` and enable a specific temporary region view."""
    canonicalize_region_captures(trace)
    _rewrite_region(trace, mode)


def _rewrite_region(trace: CapturedTrace, mode: RegionFormat) -> None:
    """Rewrite an already-canonical trace into a temporary region format."""
    if mode == RegionFormat.ISOLATED:
        return
    if mode == RegionFormat.LEGACY_PLACEHOLDERS:
        enable_legacy_capture_placeholders(trace)
    elif mode == RegionFormat.DIRECT_OUTER_REF:
        enable_direct_capture_refs(trace)
    elif mode == RegionFormat.SCHEDULE_SIGNATURE_PLACEHOLDERS:
        enable_schedule_signature_placeholders(trace)
    else:
        raise ValueError(f"Unsupported region format: {mode}")


def wrap_graph_passes_with_region_adapters(
    trace: CapturedTrace, graph_passes: list[Callable]
) -> list[Callable]:
    """Wrap raw graph passes so each pass sees its declared region view."""
    canonicalize_region_captures(trace)
    wrapped_passes = []
    assume_canonical_input = True
    for graph_pass in graph_passes:
        mode = graph_pass_region_mode(graph_pass)

        def wrapped(
            graph_pass=graph_pass,
            mode=mode,
            assume_canonical_input=assume_canonical_input,
        ):
            if not assume_canonical_input:
                canonicalize_region_captures(trace)
            _rewrite_region(trace, mode)
            result = graph_pass()
            canonicalize_region_captures(trace)
            return result

        wrapped.__name__ = getattr(graph_pass, "__name__", type(graph_pass).__name__)
        setattr(wrapped, _RAW_GRAPH_PASS_ATTR, graph_pass)
        wrapped_passes.append(wrapped)
        assume_canonical_input = True
    return wrapped_passes


def _resolve_trace_parameter_name(graph_pass: Callable) -> str:
    """Find the parameter that should receive the `CapturedTrace`."""
    try:
        signature = inspect.signature(graph_pass)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Cannot inspect pass signature for {graph_pass.__name__}"
        ) from exc

    parameters = tuple(signature.parameters.values())
    candidates = [param.name for param in parameters if param.name == "trace"]
    if not candidates:
        candidates = [
            param.name
            for param in parameters
            if param.annotation is CapturedTrace
            or param.annotation == "CapturedTrace"
            or getattr(param.annotation, "__forward_arg__", None) == "CapturedTrace"
        ]

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise TypeError(
            f"Decorated pass {graph_pass.__name__} must expose a `trace` "
            "parameter or annotate exactly one parameter as CapturedTrace"
        )
    raise TypeError(
        f"Decorated pass {graph_pass.__name__} has multiple possible "
        f"trace parameters: {', '.join(candidates)}"
    )
