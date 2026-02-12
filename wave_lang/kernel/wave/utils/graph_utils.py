# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import fields as dataclass_fields, is_dataclass
from typing import (
    Callable,
    Optional,
    Sequence,
    TypeAlias,
    Any,
    NamedTuple,
)

import sympy
import torch.fx as fx

import wave_lang.kernel.lang as tkl

from ....support.indexing import IndexSequence
from ..._support.dtype import DataType
from ..._support.indexing import IndexSymbol
from ..._support.location import CapturedLocation
from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import GLOBAL_ADDRESS_SPACE, MMA_ACC, MMA_LHS, MMA_RHS
from ...lang.wave_types import Memory, Register
from ...ops.wave_ops import (
    MMA,
    Allocate,
    Conditional,
    CustomOp,
    ExtractSlice,
    GetResult,
    IterArg,
    Iterate,
    NestedRegionOp,
    Output,
    Placeholder,
    SharedMemoryBarrier,
    TopkOp,
    SharedMemoryBarrierSignal,
    SharedMemoryBarrierWait,
    MemoryCounterWaitBarrier,
    Write,
    get_custom,
)
from .symbol_utils import (
    collect_allowed_induction_symbols,
    strip_out_of_scope_induction_symbols,
    subs_idxc,
)


class CheckResult(NamedTuple):
    """Result of equivalence checking with success flag and optional reason."""

    success: bool
    reason: str | None


IndexDict: TypeAlias = dict[IndexSymbol, IndexSequence | sympy.Basic | int]
Payload: TypeAlias = (
    fx.Node
    | IndexSequence
    | sympy.Basic
    | int
    | DataType
    | dict[Any, "Payload"]
    | Sequence["Payload"]
    | None
)
ResultType: TypeAlias = DataType | Memory | Register | Sequence["ResultType"] | None

# Semantic attributes stored on fx.Node (via setattr) rather than declared as
# dataclass fields.  These complement the dataclass field introspection in
# _check_nodes_equivalent: dataclass fields are compared automatically via
# fields(), while these must be checked explicitly because they live on the
# fx.Node, not on the CustomOp dataclass.
#
# Excluded from comparison (non-semantic / scheduling artifacts):
#   location, expanded_dims, scheduling_parameters
#
# See docs/wave/ir_design_notes.rst for background on this distinction.
_ADDITIONAL_NODE_ATTRS = ("vector_shapes", "reduction_dim", "iter_idx")


def _is_effectively_none(val: Any) -> bool:
    """Return True for None or list/tuple containers of all-None values.

    During graph construction, an unset index can surface as either `None`
    or `[None]` depending on the op.  Treat both as "not yet populated".
    """
    if val is None:
        return True
    return isinstance(val, (list, tuple)) and all(v is None for v in val)


def DCE(trace: CapturedTrace) -> None:
    """
    Removes all operators that are not used in the graph,
    excluding output and global write nodes.
    Repeats this process till no more operators can be removed.
    """

    def is_global_write(node: fx.Node) -> bool:
        custom = get_custom(node)
        return isinstance(custom, Write) and (
            subs_idxc(custom.type.address_space)
            in [GLOBAL_ADDRESS_SPACE, tkl.AddressSpace.GLOBAL_MEMORY.value]
        )

    def has_nested_irremovable(node: fx.Node) -> bool:
        custom = get_custom(node)
        if not isinstance(custom, NestedRegionOp):
            return False

        subgraph = custom.get_root_graph().subgraphs[custom.subgraph_name]
        for node in subgraph.nodes:
            if not is_removable_operator(node):
                return True

        return False

    def is_removable_operator(node: fx.Node) -> bool:
        custom = get_custom(node)

        if custom.users or custom.has_side_effects or is_global_write(node):
            return False

        if isinstance(custom, Placeholder) and custom.graph == trace.get_root_graph():
            # Do not remove root placeholders as they correspond to kernel
            # arguments and removing them will change the kernel signature.
            return False

        if has_nested_irremovable(node):
            return False

        return True

    while removable_nodes := trace.walk(is_removable_operator):
        for node in removable_nodes:
            get_custom(node).erase()


def _expr_symbols(expr: IndexSequence | sympy.Basic | int) -> set[str]:
    """Collect free-symbol names from index expressions."""
    if isinstance(expr, IndexSequence):
        return (
            _expr_symbols(expr.start)
            | _expr_symbols(expr.size)
            | _expr_symbols(expr.stride)
        )
    if isinstance(expr, sympy.Basic):
        return {s.name for s in expr.free_symbols}
    if isinstance(expr, int):
        return set()
    raise ValueError(f"Unsupported expression type: {type(expr)}")


def _apply_subs(
    expr: IndexSequence | sympy.Basic | int, subs: Optional[dict[IndexSymbol, int]]
) -> IndexSequence | sympy.Basic | int:
    """Apply substitutions to expressions or index sequences."""
    if subs is None:
        return expr
    if isinstance(expr, int):
        return expr
    if isinstance(expr, (sympy.Basic, IndexSequence)):
        return expr.subs(subs)
    raise ValueError(f"Unsupported expression type: {type(expr)}")


def _check_expr_equivalent(
    lhs: IndexSequence | sympy.Basic | int,
    rhs: IndexSequence | sympy.Basic | int,
    subs: Optional[dict[IndexSymbol, int]],
) -> CheckResult:
    """Check symbolic equivalence after substitution and simplification."""
    if _expr_symbols(lhs) != _expr_symbols(rhs):
        return CheckResult(
            False, f"symbol mismatch: {_expr_symbols(lhs)} vs {_expr_symbols(rhs)}"
        )
    lhs = _apply_subs(lhs, subs)
    rhs = _apply_subs(rhs, subs)
    if isinstance(lhs, IndexSequence) and isinstance(rhs, IndexSequence):
        if not (
            check_result := _check_expr_equivalent(lhs.start, rhs.start, subs)
        ).success:
            return CheckResult(
                False, f"IndexSequence.start mismatch: {check_result.reason}"
            )
        if not (
            check_result := _check_expr_equivalent(lhs.size, rhs.size, subs)
        ).success:
            return CheckResult(
                False, f"IndexSequence.size mismatch: {check_result.reason}"
            )
        if not (
            check_result := _check_expr_equivalent(lhs.stride, rhs.stride, subs)
        ).success:
            return CheckResult(
                False, f"IndexSequence.stride mismatch: {check_result.reason}"
            )
        return CheckResult(True, None)
    if isinstance(lhs, int) and isinstance(rhs, int):
        return (
            CheckResult(True, None)
            if lhs == rhs
            else CheckResult(False, f"int mismatch: {lhs} vs {rhs}")
        )
    if isinstance(lhs, sympy.Basic) and isinstance(rhs, sympy.Basic):
        if sympy.simplify(lhs - rhs) == 0:
            return CheckResult(True, None)
        return CheckResult(False, f"symbolic expr mismatch: {lhs} vs {rhs}")
    if isinstance(lhs, (int, sympy.Basic)) and isinstance(rhs, (int, sympy.Basic)):
        if sympy.simplify(sympy.sympify(lhs) - sympy.sympify(rhs)) == 0:
            return CheckResult(True, None)
        return CheckResult(False, f"expr mismatch: {lhs} vs {rhs}")
    raise ValueError(f"Unsupported expression types: {type(lhs)} vs {type(rhs)}")


def _check_index_mapping_equivalent(
    lhs: IndexDict, rhs: IndexDict, subs: Optional[dict[IndexSymbol, int]]
) -> CheckResult:
    """Compare two index-mapping dictionaries for symbolic equivalence."""
    if lhs.keys() != rhs.keys():
        return CheckResult(False, f"index keys mismatch: {lhs.keys()} vs {rhs.keys()}")
    for key in lhs:
        if not (
            check_result := _check_expr_equivalent(lhs[key], rhs[key], subs)
        ).success:
            return CheckResult(
                False, f"index expr mismatch for '{key}': {check_result.reason}"
            )
    return CheckResult(True, None)


def _check_payloads_equivalent(
    lhs: Payload,
    rhs: Payload,
    subs: Optional[dict[IndexSymbol, int]],
    node_map: dict[fx.Node, fx.Node],
) -> CheckResult:
    """Compare nested payloads, resolving nodes via node_map."""
    # fx.Node comparison
    if isinstance(lhs, fx.Node) or isinstance(rhs, fx.Node):
        if not (isinstance(lhs, fx.Node) and isinstance(rhs, fx.Node)):
            return CheckResult(False, f"node vs non-node: {type(lhs)} vs {type(rhs)}")
        expected = node_map.get(lhs)
        if expected is None:
            return CheckResult(False, "node mapping not provided")
        return (
            CheckResult(True, None)
            if expected is rhs
            else CheckResult(False, f"node mapping mismatch: {lhs} vs {rhs}")
        )

    # Dict comparison
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        if all(isinstance(k, IndexSymbol) for k in lhs.keys()):
            return _check_index_mapping_equivalent(lhs, rhs, subs)
        if lhs.keys() != rhs.keys():
            return CheckResult(
                False, f"dict keys mismatch: {lhs.keys()} vs {rhs.keys()}"
            )
        for key in lhs:
            if not (
                check_result := _check_payloads_equivalent(
                    lhs[key], rhs[key], subs, node_map
                )
            ).success:
                return CheckResult(
                    False, f"dict value mismatch at '{key}': {check_result.reason}"
                )
        return CheckResult(True, None)

    # Sequence comparison (list, tuple, etc.) - handles node.args, return_vals, etc.
    if isinstance(lhs, Sequence) and not isinstance(lhs, (str, bytes, dict)):
        if not (isinstance(rhs, Sequence) and not isinstance(rhs, (str, bytes, dict))):
            return CheckResult(
                False, f"sequence vs non-sequence: {type(lhs)} vs {type(rhs)}"
            )
        if len(lhs) != len(rhs):
            return CheckResult(
                False, f"sequence length mismatch: {len(lhs)} vs {len(rhs)}"
            )
        for idx, (lval, rval) in enumerate(zip(lhs, rhs)):
            if not (
                check_result := _check_payloads_equivalent(lval, rval, subs, node_map)
            ).success:
                return CheckResult(
                    False, f"sequence mismatch at {idx}: {check_result.reason}"
                )
        return CheckResult(True, None)

    # IndexSequence and symbolic expression comparison
    if isinstance(lhs, IndexSequence) or isinstance(rhs, IndexSequence):
        if not (isinstance(lhs, IndexSequence) and isinstance(rhs, IndexSequence)):
            return CheckResult(
                False, f"index sequence type mismatch: {type(lhs)} vs {type(rhs)}"
            )
        if not (check_result := _check_expr_equivalent(lhs, rhs, subs)).success:
            return CheckResult(False, f"index sequence mismatch: {check_result.reason}")
        return CheckResult(True, None)
    if isinstance(lhs, (sympy.Basic, int)) or isinstance(rhs, (sympy.Basic, int)):
        if not (check_result := _check_expr_equivalent(lhs, rhs, subs)).success:
            return CheckResult(False, f"expr mismatch: {check_result.reason}")
        return CheckResult(True, None)

    # DataType comparison
    if isinstance(lhs, DataType) or isinstance(rhs, DataType):
        if not (isinstance(lhs, DataType) and isinstance(rhs, DataType)):
            return CheckResult(False, f"dtype vs non-dtype: {type(lhs)} vs {type(rhs)}")
        return (
            CheckResult(True, None)
            if lhs == rhs
            else CheckResult(False, f"dtype mismatch: {lhs} vs {rhs}")
        )

    # Fallback to default equality, e.g. lhs == rhs == None
    return (
        CheckResult(True, None)
        if lhs == rhs
        else CheckResult(False, f"value mismatch: {lhs} vs {rhs}")
    )


def _check_result_types_equivalent(lhs: ResultType, rhs: ResultType) -> CheckResult:
    """Compare result types by shape, dtype, and address space."""
    if lhs == rhs:
        return CheckResult(True, None)
    # Directional: reference absent is OK (type may not yet be set),
    # but reference present and actual absent means type was lost.
    if lhs is None:
        return CheckResult(True, None)
    if rhs is None:
        return CheckResult(
            False, "type lost: present in reference but absent in actual"
        )

    # Compare attributes that define type equivalence
    for attr in ("symbolic_shape", "dtype", "address_space"):
        lhs_val = getattr(lhs, attr, None)
        rhs_val = getattr(rhs, attr, None)
        if lhs_val != rhs_val:
            return CheckResult(False, f"{attr} mismatch: {lhs_val} vs {rhs_val}")

    return CheckResult(True, None)


def _check_index_equivalent(
    lhs: IndexDict | Sequence[IndexDict] | None,
    rhs: IndexDict | Sequence[IndexDict] | None,
    subs: Optional[dict[IndexSymbol, int]],
) -> CheckResult:
    """Compare index mappings or sequences for equivalence."""
    if lhs == rhs:
        return CheckResult(True, None)

    if _is_effectively_none(lhs) and _is_effectively_none(rhs):
        return CheckResult(True, None)

    if lhs is None or rhs is None:
        return CheckResult(False, f"index presence mismatch: {lhs} vs {rhs}")
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        return _check_index_mapping_equivalent(lhs, rhs, subs)
    if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
        if len(lhs) != len(rhs):
            return CheckResult(
                False, f"index list length mismatch: {len(lhs)} vs {len(rhs)}"
            )
        for idx, (lval, rval) in enumerate(zip(lhs, rhs)):
            if not (
                check_result := _check_index_mapping_equivalent(lval, rval, subs)
            ).success:
                return CheckResult(
                    False, f"index list mismatch at {idx}: {check_result.reason}"
                )
        return CheckResult(True, None)
    raise ValueError(f"Unsupported index types: {type(lhs)} vs {type(rhs)}")


def _check_nodes_equivalent(
    lhs_custom: CustomOp,
    rhs_custom: CustomOp,
    lhs_trace: CapturedTrace,
    rhs_trace: CapturedTrace,
    subs: Optional[dict[IndexSymbol, int]],
    node_map: dict[fx.Node, fx.Node],
) -> CheckResult:
    """
    Compare two CustomOp nodes for equivalence across traces.

    Performs a field-by-field comparison in the following order:

    1. Type identity
    2. Result types (directional; skipped for structural nodes)
    3. Dataclass fields with compare=True
    4. Index mappings (directional; skipped for structural nodes)
    5. Semantic attrs: vector_shapes, reduction_dim, iter_idx (directional)
    6. Subgraph equivalence (for NestedRegionOps)

    "Directional" means the reference (lhs) may be absent but the actual (rhs)
    must not lose information that the reference has. Particular nodes
    (Placeholder, Output -- but not IterArg) are exempt from checks
    2/4/5 as they don't participate in verify_nodes.
    """
    # Check type identity
    if type(lhs_custom) is not type(rhs_custom):
        return CheckResult(
            False,
            f"op mismatch: {type(lhs_custom).__name__} vs {type(rhs_custom).__name__}",
        )

    # Placeholder equivalence is handled by _match_root_placeholders (root
    # graphs) and _reconcile_lifted_placeholders (subgraphs).  Output nodes
    # don't carry types, indices, or semantic attrs; their return_vals field
    # is compared via the dataclass field check below.  IterArg is a
    # Placeholder subclass but is compared normally.
    _skip_node_types = isinstance(lhs_custom, (Placeholder, Output)) and not isinstance(
        lhs_custom, IterArg
    )

    # Check result types
    if not _skip_node_types:
        if not (
            check_result := _check_result_types_equivalent(
                lhs_custom.type, rhs_custom.type
            )
        ).success:
            return CheckResult(
                False,
                f"{type(lhs_custom).__name__}: {check_result.reason}",
            )

    # Check dataclass fields (automatically skips compare=False fields)
    if is_dataclass(lhs_custom):
        for f in dataclass_fields(lhs_custom):
            if not f.compare:
                continue

            lhs_val = getattr(lhs_custom, f.name, None)
            rhs_val = getattr(rhs_custom, f.name, None)

            if not (
                check_result := _check_payloads_equivalent(
                    lhs_val, rhs_val, subs, node_map
                )
            ).success:
                return CheckResult(
                    False,
                    f"field '{f.name}' mismatch: {check_result.reason}",
                )

    # Check dynamically-set index attribute.
    # Directional: reference absent is OK (index may not yet be set in early
    # pipeline stages), but reference present and actual absent means loss.
    if not _skip_node_types:
        lhs_index = getattr(lhs_custom, "index", None)
        rhs_index = getattr(rhs_custom, "index", None)

        # Allocate indices are derived via backward propagation
        # (set_derived_index) and may contain induction symbols that belong
        # to a nested Iterate scope rather than the Allocate's own scope.
        # The wave-to-MLIR conversion strips these before emitting (see
        # strip_out_of_scope_induction_symbols in symbol_utils.py).
        # Only the LHS (reference / source trace) needs cleanup; the RHS
        # comes from MLIR import where scoping is enforced structurally.
        if isinstance(lhs_custom, Allocate):
            if isinstance(lhs_index, dict):
                allowed = collect_allowed_induction_symbols(lhs_custom.fx_node)
                lhs_index = strip_out_of_scope_induction_symbols(lhs_index, allowed)

        if lhs_index is not None:
            if rhs_index is None:
                return CheckResult(
                    False,
                    f"index lost on {type(lhs_custom).__name__}: present in reference but absent in actual",
                )
            if not (
                check_result := _check_index_equivalent(lhs_index, rhs_index, subs)
            ).success:
                return check_result

    # Check node-attached semantic attributes (see _NODE_SEMANTIC_ATTRS).
    # Directional: lhs (reference) absent is OK (rhs may have enriched info),
    # but lhs present and rhs absent means information was lost.
    if not _skip_node_types:
        for attr_name in _ADDITIONAL_NODE_ATTRS:
            lhs_val = getattr(lhs_custom, attr_name, None)
            rhs_val = getattr(rhs_custom, attr_name, None)
            if lhs_val is None:
                continue
            if rhs_val is None:
                return CheckResult(
                    False,
                    f"attr '{attr_name}' lost on {type(lhs_custom).__name__}: present in reference but absent in actual",
                )
            if not (
                check_result := _check_payloads_equivalent(
                    lhs_val, rhs_val, subs, node_map
                )
            ).success:
                return CheckResult(
                    False, f"attr '{attr_name}' mismatch: {check_result.reason}"
                )

    # implicit_captures (compare=False) may differ in order (makeIsolated
    # can reorder) or length (the source trace may have pruned dead
    # captures after hoisting, while the MLIR-imported trace includes all
    # values that makeIsolated found referenced from the region).
    # Check that every LHS capture maps to a capture on the RHS.
    lhs_captures = getattr(lhs_custom, "implicit_captures", None)
    rhs_captures = getattr(rhs_custom, "implicit_captures", None)
    if lhs_captures is not None:
        if rhs_captures is None:
            return CheckResult(
                False, "implicit_captures present in LHS but absent in RHS"
            )
        rhs_capture_set = set(rhs_captures)
        for lc in lhs_captures:
            mapped = node_map.get(lc)
            if mapped is not None and mapped not in rhs_capture_set:
                return CheckResult(
                    False,
                    f"implicit_captures mismatch: LHS capture {lc} "
                    f"(mapped to {mapped}) not found in RHS captures",
                )

    # Check subgraphs for NestedRegionOps
    lhs_subgraph = getattr(lhs_custom, "subgraph_name", None)
    rhs_subgraph = getattr(rhs_custom, "subgraph_name", None)
    if (lhs_subgraph is None) != (rhs_subgraph is None):
        return CheckResult(False, "subgraph presence mismatch")
    if lhs_subgraph is not None:
        if not (
            check_result := _check_graphs_equivalent(
                lhs_trace, rhs_trace, subs, node_map, lhs_subgraph, rhs_subgraph
            )
        ).success:
            return CheckResult(
                False, f"subgraph '{lhs_subgraph}' mismatch: {check_result.reason}"
            )

    return CheckResult(True, None)


def _reconcile_lifted_placeholders(
    lhs_nodes: list[fx.Node],
    rhs_nodes: list[fx.Node],
    node_map: dict[fx.Node, fx.Node],
) -> tuple[list[fx.Node], list[fx.Node]]:
    """Reconcile lifted placeholders between two subgraphs.

    "Lifted" placeholders represent captured values from outer scopes.
    The fx_emitter calls `makeIsolated` on every Iterate node,
    which ensures captured values become explicit block arguments. They are
    reconstructed as placeholder nodes in the FX graph.  However,
    the source FX graph may have had them transformed away (e.g. by hoisting),
    so the two sides can differ in whether a lifted placeholder exists.
    This function updates `node_map` so that references resolve correctly
    regardless of which form each side uses, and returns the non-lifted nodes
    from each side for positional comparison.

    Three cases are handled:
    1. Both sides have a lifted placeholder for the same outer value --
       the two placeholders are mapped to each other.
    2. Only rhs has a lifted placeholder -- the lhs outer node is remapped
       to the rhs placeholder within this subgraph scope.
    3. Only lhs has a lifted placeholder -- it is mapped to the rhs outer node.
    """
    lhs_lifted_map = {
        n.meta["lifted"]: n
        for n in lhs_nodes
        if isinstance(get_custom(n), Placeholder) and "lifted" in n.meta
    }
    rhs_lifted_map = {
        n.meta["lifted"]: n
        for n in rhs_nodes
        if isinstance(get_custom(n), Placeholder) and "lifted" in n.meta
    }

    # Case 1: both sides have a lifted placeholder for equivalent outer values.
    for lhs_outer, lhs_ph in lhs_lifted_map.items():
        rhs_outer = node_map.get(lhs_outer)
        if rhs_outer is not None and rhs_outer in rhs_lifted_map:
            node_map[lhs_ph] = rhs_lifted_map[rhs_outer]

    # Case 2: rhs has a lifted placeholder but lhs references the outer directly.
    for rhs_outer, rhs_ph in rhs_lifted_map.items():
        lhs_outer = next((ln for ln, rn in node_map.items() if rn == rhs_outer), None)
        if lhs_outer is not None and lhs_outer not in lhs_lifted_map:
            node_map[lhs_outer] = rhs_ph

    # Case 3: lhs has a lifted placeholder but rhs references the outer directly.
    for lhs_outer, lhs_ph in lhs_lifted_map.items():
        rhs_outer = node_map.get(lhs_outer)
        if rhs_outer is not None and rhs_outer not in rhs_lifted_map:
            node_map[lhs_ph] = rhs_outer

    # Return non-lifted nodes for positional comparison.
    is_lifted = lambda n: isinstance(get_custom(n), Placeholder) and "lifted" in n.meta
    lhs_non_lifted = [n for n in lhs_nodes if not is_lifted(n)]
    rhs_non_lifted = [n for n in rhs_nodes if not is_lifted(n)]
    return lhs_non_lifted, rhs_non_lifted


def _match_root_placeholders(
    lhs_nodes: list[fx.Node],
    rhs_nodes: list[fx.Node],
    lhs_trace: CapturedTrace,
    rhs_trace: CapturedTrace,
    subs: Optional[dict[IndexSymbol, int]],
    node_map: dict[fx.Node, fx.Node],
) -> tuple[list[fx.Node], list[fx.Node], CheckResult | None]:
    """Match root-graph placeholders by order and return non-placeholder nodes.

    Placeholder names may differ (e.g. `a` vs `arg0`), but their order as
    function parameters is what matters semantically.  Each matched pair is
    checked for equivalence via `_check_nodes_equivalent`.  Returns the
    non-placeholder nodes from each side, or a `CheckResult` on failure.
    """
    lhs_phs = [
        (i, n)
        for i, n in enumerate(lhs_nodes)
        if isinstance(get_custom(n), Placeholder)
    ]
    rhs_phs = [
        (i, n)
        for i, n in enumerate(rhs_nodes)
        if isinstance(get_custom(n), Placeholder)
    ]

    if len(lhs_phs) != len(rhs_phs):
        return (
            [],
            [],
            CheckResult(
                False,
                f"placeholder count mismatch: {len(lhs_phs)} vs {len(rhs_phs)}",
            ),
        )

    lhs_ph_positions = set()
    rhs_ph_positions = set()
    for (lhs_pos, lhs_node), (rhs_pos, rhs_node) in zip(lhs_phs, rhs_phs):
        node_map[lhs_node] = rhs_node
        lhs_ph_positions.add(lhs_pos)
        rhs_ph_positions.add(rhs_pos)

        if not (
            check_result := _check_nodes_equivalent(
                get_custom(lhs_node),
                get_custom(rhs_node),
                lhs_trace,
                rhs_trace,
                subs,
                node_map,
            )
        ).success:
            return (
                [],
                [],
                CheckResult(False, f"placeholder mismatch: {check_result.reason}"),
            )

    lhs_non_ph = [n for i, n in enumerate(lhs_nodes) if i not in lhs_ph_positions]
    rhs_non_ph = [n for i, n in enumerate(rhs_nodes) if i not in rhs_ph_positions]
    return lhs_non_ph, rhs_non_ph, None


def _compare_node_lists(
    lhs_nodes: list[fx.Node],
    rhs_nodes: list[fx.Node],
    lhs_trace: CapturedTrace,
    rhs_trace: CapturedTrace,
    subs: Optional[dict[IndexSymbol, int]],
    node_map: dict[fx.Node, fx.Node],
) -> CheckResult:
    """Compare two node lists positionally using _check_nodes_equivalent."""
    if len(lhs_nodes) != len(rhs_nodes):
        return CheckResult(
            False,
            f"node count mismatch: {len(lhs_nodes)} vs {len(rhs_nodes)}",
        )
    for lhs_node, rhs_node in zip(lhs_nodes, rhs_nodes):
        node_map[lhs_node] = rhs_node
        if not (
            check_result := _check_nodes_equivalent(
                get_custom(lhs_node),
                get_custom(rhs_node),
                lhs_trace,
                rhs_trace,
                subs,
                node_map,
            )
        ).success:
            return check_result
    return CheckResult(True, None)


def _check_graphs_equivalent(
    lhs_trace: CapturedTrace,
    rhs_trace: CapturedTrace,
    subs: Optional[dict[IndexSymbol, int]],
    node_map: dict[fx.Node, fx.Node],
    lhs_graph_name: str | None = None,
    rhs_graph_name: str | None = None,
) -> CheckResult:
    """Compare two graphs node-by-node with shared node_map.

    Placeholder handling is delegated to helper functions:
    - Subgraphs: `_reconcile_lifted_placeholders` maps captured values.
    - Root graphs: `_match_root_placeholders` pairs parameters by order.

    After placeholders are resolved, non-placeholder nodes are compared
    positionally via `_compare_node_lists`.
    """
    lhs = (
        lhs_trace.get_root_graph()
        if lhs_graph_name is None
        else lhs_trace.get_subgraph(lhs_graph_name)
    )
    rhs = (
        rhs_trace.get_root_graph()
        if rhs_graph_name is None
        else rhs_trace.get_subgraph(rhs_graph_name)
    )

    lhs_nodes = list(lhs.nodes)
    rhs_nodes = list(rhs.nodes)

    if lhs_graph_name is not None:
        # Subgraph: reconcile lifted placeholders, then compare non-lifted.
        lhs_compare, rhs_compare = _reconcile_lifted_placeholders(
            lhs_nodes, rhs_nodes, node_map
        )
    else:
        # Root graph: match placeholders by order, then compare the rest.
        if len(lhs_nodes) != len(rhs_nodes):
            return CheckResult(False, "node count mismatch")
        lhs_compare, rhs_compare, err = _match_root_placeholders(
            lhs_nodes, rhs_nodes, lhs_trace, rhs_trace, subs, node_map
        )
        if err is not None:
            return err

    return _compare_node_lists(
        lhs_compare, rhs_compare, lhs_trace, rhs_trace, subs, node_map
    )


def assert_traces_equivalent(
    lhs: CapturedTrace,
    rhs: CapturedTrace,
    subs: Optional[dict[IndexSymbol, int]] = None,
) -> None:
    """Assert structural equivalence between two traces.

    Designed for MLIR roundtrip validation where `lhs` is the reference
    (source trace, possibly sparsely populated in early pipeline stages) and
    `rhs` is the actual (roundtrip result, eagerly populated from MLIR).

    The comparison is directional: `rhs` may carry additional information
    (e.g. indices or vector_shapes derived from constraints) that `lhs` does
    not yet have, but any information present in `lhs` must also appear in
    `rhs`.
    """
    node_map: dict[fx.Node, fx.Node] = {}
    check_result = _check_graphs_equivalent(lhs, rhs, subs, node_map)
    if not check_result.success:
        raise AssertionError(f"Traces are not equivalent: {check_result.reason}")


def assert_constraints_equivalent(
    lhs_constraints: Sequence[Any],
    rhs_constraints: Sequence[Any],
    custom_comparators: Optional[dict[type, callable]] = None,
) -> None:
    """
    Assert that two lists of constraints are equivalent (order-independent).

    Constraints use custom __eq__ methods that only compare semantically significant
    fields, making this suitable for testing MLIR roundtrip fidelity. Order does not
    matter as constraints form an unordered set.

    Args:
        lhs_constraints: First list of constraints
        rhs_constraints: Second list of constraints
        custom_comparators: Optional dict mapping constraint types to custom comparison
            functions. Each function should take (lhs, rhs) and return bool.
            If a constraint type is in this dict, its comparator is used instead of ==.

    Raises:
        AssertionError: If constraints are not equivalent
    """
    if len(lhs_constraints) != len(rhs_constraints):
        raise AssertionError(
            f"Constraint count mismatch: {len(lhs_constraints)} vs {len(rhs_constraints)}"
        )

    def constraints_equal(lhs_c, rhs_c) -> bool:
        """Check if two constraints are equal, using custom comparator if available."""
        if custom_comparators:
            for constraint_type, comparator in custom_comparators.items():
                if isinstance(lhs_c, constraint_type) and isinstance(
                    rhs_c, constraint_type
                ):
                    return comparator(lhs_c, rhs_c)
        return lhs_c == rhs_c

    # Check each LHS constraint has a matching RHS constraint (order-independent)
    for lhs_c in lhs_constraints:
        if not any(constraints_equal(lhs_c, rhs_c) for rhs_c in rhs_constraints):
            raise AssertionError(
                f"No matching constraint found for:\n  {lhs_c}\n"
                f"Available constraints:\n  "
                + "\n  ".join(str(c) for c in rhs_constraints)
            )


def move_node_after(src_node: fx.Node, anchor: fx.Node):
    """
    Moves src_node into a location after a given anchor node.
    This function will invalidate "src_node" and return the
    newly copied/"moved" node.
    """
    custom_src = get_custom(src_node)
    moved_src = custom_src.copy(anchor=(anchor)).fx_node
    custom_src.replace_all_uses_with(moved_src)
    src_name = src_node.name
    src_node.graph.erase_node(src_node)
    moved_src.name = src_name
    return moved_src


def remove_chained_getresult(trace: CapturedTrace):
    def is_chained_getresult(node: fx.Node) -> bool:
        custom = get_custom(node)
        return isinstance(custom, GetResult) and isinstance(
            get_custom(custom.value), GetResult
        )

    while removable_nodes := trace.walk(is_chained_getresult):
        for node in removable_nodes:
            get_custom(node).replace_all_uses_with(get_custom(node).value)
            get_custom(node).graph.erase_node(node)


def remove_chained_extractslice(trace: CapturedTrace):
    def is_chained_extractslice(node: fx.Node) -> bool:
        custom = get_custom(node)
        if not isinstance(custom, ExtractSlice):
            return False
        register = get_custom(custom.register_)
        if not isinstance(register, ExtractSlice):
            return False
        return custom.rank == register.rank

    while removable_nodes := trace.walk(is_chained_extractslice):
        for node in removable_nodes:
            dst_extract = get_custom(node)
            src_extract = get_custom(dst_extract.register_)
            dst_extract.register_ = src_extract.register_
            new_offset = [
                dst_i + src_i
                for dst_i, src_i in zip(dst_extract.offset, src_extract.offset)
            ]
            dst_extract.update_arg("register_", src_extract.register_)
            dst_extract.update_arg("offset", new_offset)
            if len(src_extract.fx_node.users) == 0:
                get_custom(node).graph.erase_node(src_extract.fx_node)


def erase_graph(graph: fx.Graph):
    """
    Erase all nodes in the graph.
    """
    for node in reversed(graph.nodes):
        for user in node.users:
            graph.erase_node(user)
        graph.erase_node(node)


def _placeholder_captures(placeholder_node: fx.Node, target: fx.Node) -> bool:
    """
    Recursively check if a placeholder (or chain of placeholders) captures the target node.
    """
    custom = get_custom(placeholder_node)
    if not isinstance(custom, Placeholder):
        return placeholder_node == target

    captured = custom.get_captured_fx_node()
    if captured is None:
        return False
    # Recursively check if what this placeholder captures is itself
    # a placeholder that captures target
    return _placeholder_captures(captured, target)


def get_users(
    node: fx.Node, region: fx.Node = None
) -> tuple[list[fx.Node], Optional[fx.Node]]:
    """
    Return the users of a node, propagating through reductions and conditionals.
    Returns (users, region_node) where region_node is either an Iterate or Conditional node.
    """
    users = []
    for user in node.users:
        custom = user
        if not isinstance(custom, CustomOp):
            custom = get_custom(user)
        if isinstance(custom, Iterate):
            # Map init arg to iter arg
            region = custom
            graph = custom.get_root_graph().subgraphs[custom.subgraph_name]
            if node in custom.init_args:
                init_arg_idx = custom.init_args.index(node)
                users.append(custom.iter_args(graph)[init_arg_idx])
            elif node in custom.implicit_captures:
                for outside_node in graph.nodes:
                    if outside_node.meta.get("lifted", None) == node:
                        users.append(outside_node)
                        break
            else:
                # Check if any placeholder in implicit_captures captures this node (recursively)
                for capture in custom.implicit_captures:
                    if _placeholder_captures(capture, node):
                        for outside_node in graph.nodes:
                            if outside_node.meta.get("lifted", None) == capture:
                                users.append(outside_node)
                                break
                        break
            continue
        if isinstance(custom, Output):
            # Map output to get result
            return_vals = custom.return_vals[0]
            parent_region = custom.graph.parent_op
            if not isinstance(return_vals, (list, tuple)):
                if parent_region.users:
                    users.append(next(iter(parent_region.users)))
            else:
                # Handles case where DCE eliminate unused GetResult.
                get_results = {
                    get_custom(x).res_idx: x
                    for x in parent_region.users
                    if isinstance(get_custom(x), GetResult)
                }
                output_idx = return_vals.index(node)
                # Sometime IterArg only used within the tkw.Reduction region
                if output_idx in get_results:
                    users.append(get_results[output_idx])
            continue
        if isinstance(custom, Conditional):
            region = custom
            if node == custom.condition:
                users.append(user)
            elif custom.init_args is not None and node in custom.init_args:
                # For init_args, the users are the iter_args in the subgraph
                # and the conditional itself (as it returns the value)
                subgraph = custom.get_root_graph().subgraphs[custom.subgraph_name]
                iter_args = custom.iter_args(subgraph)
                # Find the iter_arg that corresponds to this init_arg
                init_arg_idx = custom.init_args.index(node)
                if init_arg_idx < len(iter_args):
                    for u in iter_args[init_arg_idx].users:
                        users.append(u)
                # The conditional node itself is also a user if it returns values
                users.append(user)
            else:
                subgraph = custom.get_root_graph().subgraphs[custom.subgraph_name]
                var = custom.get_captured_fx_node(subgraph, node)
                assert var is not None, "Invalid captured var"
                for u in var.users:
                    users.append(u)

            continue

        users.append(user)
    return users, region


def propagate_placeholders(n: fx.Node | tuple | None) -> fx.Node | tuple | None:
    """
    Returns the captured node of a placeholder if it exists.
    Handles tuples by recursively propagating each element.
    """
    if n is None:
        return None
    if isinstance(n, tuple):
        return (propagate_placeholders(elem) for elem in n)
    c = get_custom(n)
    if isinstance(c, Placeholder):
        p = c.get_captured_fx_node()
        if p is not None:
            return p
    return n


def propagate_loop_carried_vars(n: fx.Node, depth: int = 0) -> fx.Node:
    """
    Propagates node through placeholders and loop-carried vars.

    `depth` is the number of iterations to propagate through.
    For example, if `depth` is 1, then the function will propagate through the
    loop-carried vars to the previous iteration.
    If `depth` is 0, then the function will propagate to the loop init_args.
    """
    c = get_custom(n)
    if isinstance(c, IterArg):
        idx = c.iter_idx
        iterate = c.parent_op()
        assert isinstance(iterate, Iterate), f"Expected Iterate, but got {iterate}"
        args = iterate.init_args if depth == 0 else iterate.outputs()
        assert idx < len(
            args
        ), f"IterArg index {idx} out of range for {args}, depth={depth}"
        depth = max(depth - 1, 0)
        return propagate_loop_carried_vars(args[idx], depth)
    elif isinstance(c, Placeholder):
        p = c.get_captured_fx_node()

        # Top level placeholders correspond to kernel arguments and don't have
        # captured nodes.
        if p is not None:
            return p
    elif isinstance(c, GetResult):
        iterate = get_custom(c.value)
        assert isinstance(iterate, Iterate), f"Expected Iterate, but got {iterate}"
        idx = c.res_idx
        args = iterate.init_args if depth == 0 else iterate.outputs()
        assert idx < len(
            args
        ), f"GetResult index {idx} out of range for {args}, depth={depth}"
        depth = max(depth - 1, 0)
        return propagate_loop_carried_vars(args[idx], depth)

    return n


def get_inputs(
    node: fx.Node, region: fx.Node = None
) -> tuple[list[fx.Node], Optional[fx.Node]]:
    """
    Return the inputs of a node, propagating through reductions and conditionals.
    Returns (inputs, region_node) where region_node is either an Iterate or Conditional node.
    """
    inputs = []
    custom = get_custom(node)
    if isinstance(custom, IterArg):
        # Map iter args to init args
        if region is None:
            parent_op = custom.parent_op()
            if isinstance(parent_op, (Iterate, Conditional)):
                region = parent_op
        iter_arg_idx = custom.iter_idx
        if region and region.init_args:
            inputs.append(region.init_args[iter_arg_idx])
    elif isinstance(custom, GetResult):
        assert custom.value is not None, f"GetResult node {custom} has no value"
        parent_op = get_custom(custom.value)
        if isinstance(parent_op, TopkOp):
            region = None
            inputs += node.all_input_nodes
        elif isinstance(parent_op, Iterate):
            region = parent_op
            # Map get result to output
            iteration_subgraph = region.get_root_graph().subgraphs[region.subgraph_name]
            if len(region.init_args) == 1:
                outputs = region.outputs(iteration_subgraph)
                if isinstance(outputs, Sequence):
                    inputs += outputs
                else:
                    inputs.append(outputs)
            else:
                inputs.append(region.outputs(iteration_subgraph)[custom.res_idx])
        elif isinstance(parent_op, Conditional):
            region = parent_op
            # Map get result to output
            conditional_subgraph = region.get_root_graph().subgraphs[
                region.subgraph_name
            ]
            outputs = region.outputs(conditional_subgraph)
            if isinstance(outputs, Sequence):
                if len(outputs) == 1:
                    inputs.append(outputs[0])
                else:
                    inputs.append(outputs[custom.res_idx])
            else:
                inputs.append(outputs)
        else:
            raise ValueError(
                f"GetResult must be using an Iterate or Conditional, but\n{custom}\nis using\n{parent_op}"
            )
    elif isinstance(custom, Iterate):
        iteration_subgraph = custom.get_root_graph().subgraphs[custom.subgraph_name]
        inputs.append(custom.outputs(iteration_subgraph))
    elif isinstance(custom, Conditional):
        conditional_subgraph = custom.get_root_graph().subgraphs[custom.subgraph_name]
        inputs.append(custom.outputs(conditional_subgraph))
    else:
        # Default handling for other ops.
        for input in node.all_input_nodes:
            inputs.append(input)

    inputs = [propagate_placeholders(i) for i in inputs if i is not None]
    # Flatten any sequences in inputs and filter out None values
    flattened_inputs = []
    for inp in inputs:
        if isinstance(inp, Sequence):
            flattened_inputs.extend(x for x in inp if x is not None)
        elif inp is not None:
            flattened_inputs.append(inp)
    return flattened_inputs, region


def bfs(
    node: fx.Node,
    get_neighbors: Callable[[fx.Node, fx.Node], list[fx.Node]],
    filter_fn: Callable[[fx.node], bool],
) -> set[fx.Node]:
    """
    Run BFS on the graph. The filter function is not applied to
    the incoming node.
    """
    visited: set[fx.Node] = set()
    queue: list[fx.Node] = []
    visited.add(node)
    queue.append(node)
    region = None
    while queue:
        s = queue.pop(0)
        neighbors, region = get_neighbors(s, region)
        for neighbor in neighbors:
            if neighbor not in visited and filter_fn(neighbor):
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def capture_forward_slice(
    node: fx.Node,
    filter_fn: Callable[[fx.node], bool] = lambda x: True,
) -> set[fx.Node]:
    """
    Run BFS on the graph to capture the forward slice of a node.
    """
    return bfs(node, lambda x, y: get_users(x, y), filter_fn)


def capture_backward_slice(
    node: fx.Node, filter_fn: Callable[[fx.node], bool] = lambda x: True
) -> set[fx.Node]:
    """
    Capture backward slice from a node and return the tree.
    Assumes graph is directed.
    """
    return bfs(node, lambda x, y: get_inputs(x, y), filter_fn)


def capture_mma_slices(mma: MMA) -> dict[IndexSymbol, list[fx.Node]]:
    """
    Given an index sequence, specialize it to a LHS, RHS or ACC index sequence
    based on whether the node is used as the LHS, RHS or ACC in the MMA node.
    """
    mma_slices = {x: [] for x in [MMA_LHS, MMA_RHS, MMA_ACC]}
    is_not_mma = lambda x: not isinstance(get_custom(x), MMA)
    mma_slices[MMA_LHS] += capture_backward_slice(mma.lhs, is_not_mma)
    mma_slices[MMA_RHS] += capture_backward_slice(mma.rhs, is_not_mma)
    mma_slices[MMA_ACC] += capture_forward_slice(mma.fx_node, is_not_mma).union(
        capture_backward_slice(mma.acc, is_not_mma)
    )
    return mma_slices


def graph_copy(graph: fx.Graph) -> tuple[fx.Graph, dict[fx.Node, fx.Node]]:
    """
    Copy the graph and return the new graph with the nodes in node_map.
    Also return the mapping of old nodes to new nodes.
    """
    new_graph = fx.Graph()
    node_map = {}
    for node in graph.nodes:
        custom = get_custom(node)
        new_node = custom.copy(
            new_graph=new_graph,
            arg_transform=lambda x: node_map[x] if x in node_map else x,
        )
        node_map[node] = new_node.fx_node
    return new_graph, node_map


def replace_uses_in(users: dict[fx.Node, list[CustomOp]], old: CustomOp, new: fx.Node):
    """
    Replace all uses of `old` with `new` in the list of users.
    """
    for user in users[old]:
        for i, arg in enumerate(user.fx_node.args):
            if arg == old.fx_node:
                user.update_arg(i, new)


def is_iterate_subgraph(graph: fx.Graph):
    """
    Check that graph is a subgraph that is owned by ReductionOp.
    """
    if not hasattr(graph, "parent_op"):
        return False
    return isinstance(get_custom(graph.parent_op), Iterate)


def initialize_iter_args(trace: CapturedTrace) -> None:
    """
    Initializes the IterArgs in each reduction with an index
    based on their location in the graph.
    Also handles arguments to Conditional nodes.
    """

    reductions = trace.walk(
        lambda node: isinstance(get_custom(node), (Iterate, Conditional))
    )
    for reduction in reductions:
        reduction_graph = trace.get_subgraph(get_custom(reduction).subgraph_name)
        count = 0
        for node in reduction_graph.nodes:
            custom = get_custom(node)
            if isinstance(custom, IterArg):
                custom.iter_idx = count
                count += 1


def get_outer_node(outer_node: fx.Node) -> fx.Node:
    while "lifted" in outer_node.meta:
        outer_node = outer_node.meta["lifted"]
    return outer_node


def is_barrier_between_same_graph(
    src: fx.Node, dst: fx.Node, barId: int = -1, barrier_check: set = None
) -> Optional[fx.Node]:
    """
    Checks if there is a barrier between the source and destination nodes,
    assuming that they are in the same graph.
    """
    next_node = src.next
    if barrier_check is None:
        barrier_check = set()

    while next_node != dst and next_node.next.op != "root":
        custom_next_node = get_custom(next_node)

        # Check for SharedMemoryBarrier (amdgpu.lds_barrier)
        if isinstance(custom_next_node, SharedMemoryBarrier):
            return next_node

        # Check for split barriers (signal/wait)
        if isinstance(custom_next_node, SharedMemoryBarrierSignal):
            barrier_check.add(custom_next_node.barId)
        if isinstance(custom_next_node, SharedMemoryBarrierWait):
            if custom_next_node.barId == barId and barId in barrier_check:
                return next_node

        # Check for MemoryCounterWaitBarrier (amdgpu.memory_counter_wait + rocdl.s.barrier)
        if isinstance(custom_next_node, MemoryCounterWaitBarrier):
            return next_node

        next_node = next_node.next

    return None


def _get_parent_chain(node: fx.Node) -> list[tuple[fx.Node, fx.Graph]]:
    """
    Get the chain of parent graphs and their parent_op nodes from node up to the root graph.
    Returns a list of (parent_op, graph) tuples, ordered from innermost to outermost.
    The node's own graph is not included, only its ancestors.
    """
    chain = []
    current_graph = node.graph
    while hasattr(current_graph, "parent_op"):
        parent_op = current_graph.parent_op
        parent_graph = parent_op.graph
        chain.append((parent_op, parent_graph))
        current_graph = parent_graph
    return chain


def _find_common_ancestor(
    src: fx.Node, dst: fx.Node
) -> tuple[Optional[fx.Graph], int, int]:
    """
    Find the closest common ancestor graph for src and dst nodes.
    Returns (common_ancestor_graph, src_depth, dst_depth) where:
    - common_ancestor_graph: The closest common ancestor graph (or None if src.graph == dst.graph)
    - src_depth: Number of levels from src to common ancestor (0 if src is in common ancestor)
    - dst_depth: Number of levels from dst to common ancestor (0 if dst is in common ancestor)
    """
    if src.graph == dst.graph:
        return None, 0, 0

    src_chain = _get_parent_chain(src)
    dst_chain = _get_parent_chain(dst)

    # Check if src is in dst's parent chain
    for depth, (parent_op, parent_graph) in enumerate(dst_chain):
        if src.graph == parent_graph:
            return parent_graph, 0, depth + 1

    # Check if dst is in src's parent chain
    for depth, (parent_op, parent_graph) in enumerate(src_chain):
        if dst.graph == parent_graph:
            return parent_graph, depth + 1, 0

    # Find common ancestor in both chains
    # Reverse chains to go from root to leaf
    src_chain_rev = list(reversed(src_chain))
    dst_chain_rev = list(reversed(dst_chain))

    # Find the deepest common graph
    common_ancestor = None
    common_depth = 0
    for i, ((src_op, src_g), (dst_op, dst_g)) in enumerate(
        zip(src_chain_rev, dst_chain_rev)
    ):
        if src_g == dst_g:
            common_ancestor = src_g
            common_depth = i
        else:
            break

    if common_ancestor is None:
        # No common ancestor found, they must be in different root graphs
        return None, len(src_chain), len(dst_chain)

    # Calculate depth from each node to the common ancestor
    # common_depth is the index in the reversed chain where we found the common ancestor
    # The depth is the total chain length minus the position of the common ancestor
    src_depth = len(src_chain) - common_depth
    dst_depth = len(dst_chain) - common_depth

    return common_ancestor, src_depth, dst_depth


def is_barrier_between(
    src: fx.Node, dst: fx.Node, barId: int = -1
) -> Optional[fx.Node]:
    """
    Checks if there is a barrier between the source and destination nodes.
    """
    barriers = set()

    if src.graph == dst.graph:
        # The following cases are handled when src and dst are in same graph:
        # 1. src and dst is on the same iteration step and src < dst (topographic).
        # 2. src and dst are on different iteration step and src > dst (topographic).

        # Case 1:
        if dst >= src:
            return is_barrier_between_same_graph(src, dst, barId)

        # Case 2:
        if dst < src:
            # Check between src and end of loop.
            if node := is_barrier_between_same_graph(
                src, list(src.graph.nodes)[-1], barId, barriers
            ):
                return node
            # If cannot find between src to end of loop,
            # then, we check beginning of loop to dst.
            return is_barrier_between_same_graph(
                list(dst.graph.nodes)[0], dst, barId, barriers
            )
    else:
        # General algorithm for nodes in different graphs:
        # 1. Get parent chains and find common ancestor
        # 2. Check from src up to common ancestor (src to graph outputs)
        # 3. Check in common ancestor (between ancestor nodes)
        # 4. Check from common ancestor down to dst (graph inputs to dst)

        common_ancestor, src_depth, dst_depth = _find_common_ancestor(src, dst)

        if common_ancestor is None:
            return None

        # Step 2: Check from src up to common ancestor
        # For each nested graph containing src, check from src to output
        current_node = src
        current_graph = src.graph
        src_chain = _get_parent_chain(src)
        for depth in range(src_depth):
            if node := is_barrier_between_same_graph(
                current_node, list(current_graph.nodes)[-1], barId, barriers
            ):
                return node
            if depth < len(src_chain):
                parent_op, parent_graph = src_chain[depth]
                current_node = parent_op
                current_graph = parent_graph

        # Step 3: Check in common ancestor graph
        src_ancestor = (
            current_node  # This is the node in common_ancestor representing src's path
        )
        dst_chain = _get_parent_chain(dst)

        if dst_depth > 0:
            dst_ancestor = dst_chain[dst_depth - 1][0]  # parent_op at the right level
        else:
            dst_ancestor = dst
        if node := is_barrier_between_same_graph(
            src_ancestor, dst_ancestor, barId, barriers
        ):
            return node

        # Step 4: Check from common ancestor down to dst
        current_node = dst
        current_graph = dst.graph
        for depth in range(dst_depth):
            if node := is_barrier_between_same_graph(
                list(current_graph.nodes)[0], current_node, barId, barriers
            ):
                return node
            if depth < len(dst_chain):
                parent_op, parent_graph = dst_chain[depth]
                current_node = parent_op
                current_graph = parent_graph

    return None


def update_sort_keys(
    trace: CapturedTrace, graph: fx.Graph, prefix: Optional[tuple] = ()
):
    """
    Update the sort keys of the graph so that
    consecutive nodes have consecutive sort keys.
    Also, broadcast the sort keys for ops in nested graphs.
    After this pass, the sort keys are unique and monotonically increasing.
    For example, if we have a graph with nodes [a, b, c, d], and c is a nested
    region with ops [e, f, g], then the sort keys will be:

    a: (0,)
    b: (1,)
    c: (2,)
        e: (2, 0)
        f: (2, 1)
        g: (2, 2)
    d: (3,)

    so that we can always say that a < b < c < e < f < g < d.
    """
    for i, node in enumerate(graph.nodes):
        node._sort_key = prefix + (i,)
        custom = get_custom(node)
        if isinstance(custom, NestedRegionOp):
            update_sort_keys(
                trace,
                trace.region_graph.subgraphs[custom.subgraph_name],
                node._sort_key,
            )


def get_graph_node(
    custom: CustomOp,
    graph: fx.Graph,
    location: Optional[CapturedLocation] = None,
) -> fx.Node:
    """Add a CustomOp to a graph and return its fx_node."""
    custom.add_to_graph(graph, loc=location)
    return custom.fx_node


def prepare_subgraph_for_conditional(
    subgraph_name: str,
    captured_nodes: list[fx.Node],
    memory_nodes: list[fx.Node] = None,
) -> tuple[fx.Graph, list[fx.Node], dict[fx.Node, fx.Node]]:
    """
    Prepare a subgraph with placeholders for captured nodes.
    Intended for use with finish_conditional_subgraph.
    When creating a subgraph, the nodes within the subgraph can't just refer to nodes in the outer graph.
    So to allow communication, we create placeholder nodes in the subgraph to represent the outer nodes.
    The captured_nodes argument specifies which outer nodes the inner graph needs to represent.

    This returns the subgraph, the captures list (to pass to
    finish_conditional_subgraph, the output list is not the same as the input
    capture captured_nodes list), and a dictionary for placeholders.
    When adding nodes to the subgraph, you can use the dictionary to map from outer nodes to their placeholders.
    Eg.

    ```
        subgraph, captures, placeholders = prepare_subgraph_for_conditional(name, [arg1, arg2], [arg2])
        write_val = Write(placeholders[arg1], placeholders[arg2], 1).add_to_graph(subgraph)
        finish_conditional_subgraph(trace, graph, condition, subgraph, captures)
    ```


    Args:
        subgraph_name: Name for the subgraph
        captured_nodes: Nodes from outer graph that need to be accessible in subgraph
        memory_nodes: Subset of captured_nodes that are memory allocations (get "lifted" metadata)

    Returns:
        (subgraph, implicit_captures, placeholders_map)
    """
    subgraph = fx.Graph()
    subgraph._name = subgraph_name

    memory_nodes = set(memory_nodes or [])
    placeholders = {}
    implicit_captures = []

    for node in captured_nodes:
        # Create placeholder in subgraph
        custom = get_custom(node) if hasattr(node, "tkw_op") else node
        placeholder = get_graph_node(Placeholder.from_fx_node(custom), subgraph)
        placeholder.type = custom.type

        # Mark memory allocations with "lifted" metadata
        if node in memory_nodes:
            placeholder.meta["lifted"] = node

        placeholders[node] = placeholder
        implicit_captures.append(get_outer_node(node))

    return subgraph, implicit_captures, placeholders


def finish_conditional_subgraph(
    trace: CapturedTrace,
    main_graph: fx.Graph,
    condition_node: fx.Node,
    subgraph: fx.Graph,
    implicit_captures: list[fx.Node],
    location: Optional[CapturedLocation] = None,
) -> fx.Node:
    """Create conditional node and register subgraph with trace.

    Args:
        trace: Trace to register subgraph with
        main_graph: Main graph to add conditional to
        condition_node: Boolean condition for the conditional
        subgraph: The prepared subgraph
        implicit_captures: List of captured nodes

    Returns:
        The conditional node
    """
    conditional = get_graph_node(
        Conditional(
            condition_node,
            subgraph_name=subgraph._name,
            implicit_captures=implicit_captures,
        ),
        main_graph,
        location,
    )

    # Register subgraph with trace
    subgraph.parent_op = conditional
    trace.add_subgraph(subgraph._name, subgraph)
    trace.get_root_graph().subgraphs[subgraph._name] = subgraph

    return conditional
