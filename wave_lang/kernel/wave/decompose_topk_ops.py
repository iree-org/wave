# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from typing import Callable, Optional

import torch.fx as fx

import wave_lang.kernel.lang as tkl

from .._support.location import CapturedLocation
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexSequence, IndexExpr
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    Broadcast,
    Eq,
    Extract,
    GetResult,
    Gt,
    Maximum,
    NewScalar,
    Reshape,
    SelectOp,
    SelfIndex,
    ShuffleOp,
    TopkOp,
    get_custom,
)
from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    TilingConstraint,
)
from .utils.classes import ShuffleMode
from .utils.general_utils import all_equal
from .utils.graph_utils import (
    DCE,
    get_graph_node,
)
from .utils.symbol_utils import subs_idxc
from .decompose_reduce_ops import determine_shuffle_config
from .analysis.index_sequence_analysis import resolve_broadcasting_for_op


def construct_self_index_index_sequence(
    reduction_dim, elements_per_thread, constraints, register_idx, src_register_size
):
    hw_constraint = next(c for c in constraints if isinstance(c, HardwareConstraint))
    thread_offset = hw_constraint.linearized_thread_id

    # Calculate offset within this thread based on register iteration
    # Each register processes src_register_size elements sequentially
    within_thread_offset = register_idx * src_register_size
    total_offset = thread_offset * elements_per_thread + within_thread_offset
    return {
        reduction_dim: IndexSequence(
            total_offset,
            src_register_size,
            1,
        )
    }


def emit_sources_topk_reduction(
    binary_fn: Callable,
    src_values: list[fx.Node],
    src_indices: list[fx.Node],
    graph: fx.Graph,
    location: Optional[CapturedLocation],
) -> tuple[fx.Node, fx.Node]:
    """
    Does topk reduction over lists of values and indices, returning both the reduced value and its index.
    """
    init_val = src_values[0]
    init_idx = src_indices[0]

    for i in range(1, len(src_values)):
        cmp_result = get_graph_node(Gt(src_values[i], init_val), graph, location)
        init_val = get_graph_node(
            SelectOp(cmp_result, src_values[i], init_val), graph, location
        )
        init_idx = get_graph_node(
            SelectOp(cmp_result, src_indices[i], init_idx), graph, location
        )
        get_custom(init_val).index = src_values[0].index
        get_custom(init_idx).index = src_indices[0].index
        resolve_broadcasting_for_op(
            get_custom(init_val), ["cond", "if_true", "if_false"]
        )
        resolve_broadcasting_for_op(
            get_custom(init_idx), ["cond", "if_true", "if_false"]
        )
    return init_val, init_idx


def emit_variable_topk_reduction(
    binary_fn: Callable,
    src_val: fx.Node,
    src_idx: fx.Node,
    graph: fx.Graph,
    local_reduction_size: int,
    location: Optional[CapturedLocation],
) -> tuple[fx.Node, fx.Node]:
    """
    Does topk reduction over singular value and index variables.
    """
    init_val = get_graph_node(Extract(src_val, [0]), graph, location)
    init_idx = get_graph_node(Extract(src_idx, [0]), graph, location)
    get_custom(init_val).index = get_custom(src_val).index
    get_custom(init_idx).index = get_custom(src_idx).index
    # From here on out we are dealing with scalar values
    init_val.type.symbolic_shape = ()
    init_idx.type.symbolic_shape = ()

    for i in range(1, local_reduction_size):
        cur_val = get_graph_node(Extract(src_val, [i]), graph, location)
        cur_idx = get_graph_node(Extract(src_idx, [i]), graph, location)
        cur_val.type.symbolic_shape = ()
        cur_idx.type.symbolic_shape = ()

        cmp_result = get_graph_node(Gt(cur_val, init_val), graph, location)
        init_val = get_graph_node(
            SelectOp(cmp_result, cur_val, init_val), graph, location
        )
        init_idx = get_graph_node(
            SelectOp(cmp_result, cur_idx, init_idx), graph, location
        )
        get_custom(init_val).index = get_custom(cur_val).index
        get_custom(init_idx).index = get_custom(cur_idx).index

    return init_val, init_idx


def emit_local_topk_reduction(
    binary_fn: Callable,
    reduction_src: list[fx.Node],
    reduction_indices: list[fx.Node],
    graph: fx.Graph,
    local_reduction_size,
    location: Optional[CapturedLocation],
) -> tuple[fx.Node, fx.Node]:
    """
    Does topk reduction over all elements carried along by TopkOp at local
    thread level. This returns both the top value and its index.
    """
    src_val, src_idx = emit_sources_topk_reduction(
        binary_fn, reduction_src, reduction_indices, graph, location
    )

    local_val, local_idx = emit_variable_topk_reduction(
        binary_fn, src_val, src_idx, graph, local_reduction_size, location
    )
    return local_val, local_idx


def emit_intrawave_topk_reduction(
    binary_fn: Callable,
    src_val: fx.Node,
    src_idx: fx.Node,
    graph: fx.Graph,
    subgroup_size: int,
    cluster_size: int,
    cluster_stride: int,
    location: Optional[CapturedLocation],
) -> tuple[fx.Node, fx.Node]:
    """
    Reduce data across threads in a wave by doing butterfly shuffle for topk.
    """
    init_val = src_val
    init_idx = src_idx
    num_steps = int(math.log2(float(cluster_size)))

    for _ in range(num_steps):
        # Shuffle both values and indices
        # TODO - if the values and indices are small types, we could pack them together and do a single shuffle.
        shuffle_val = ShuffleOp(
            init_val, cluster_stride, subgroup_size, ShuffleMode.XOR
        )
        shuffle_val_node = get_graph_node(shuffle_val, graph, location)
        shuffle_idx = ShuffleOp(
            init_idx, cluster_stride, subgroup_size, ShuffleMode.XOR
        )
        shuffle_idx_node = get_graph_node(shuffle_idx, graph, location)

        get_custom(shuffle_val_node).index = get_custom(init_val).index
        get_custom(shuffle_idx_node).index = get_custom(init_idx).index

        val_greater = get_graph_node(Gt(shuffle_val_node, init_val), graph, location)

        init_val = get_graph_node(
            SelectOp(val_greater, shuffle_val_node, init_val),
            graph,
            location,
        )
        init_idx = get_graph_node(
            SelectOp(val_greater, shuffle_idx_node, init_idx),
            graph,
            location,
        )
        get_custom(init_val).index = get_custom(shuffle_val_node).index
        get_custom(init_idx).index = get_custom(shuffle_idx_node).index

        resolve_broadcasting_for_op(
            get_custom(init_val), ["cond", "if_true", "if_false"]
        )
        resolve_broadcasting_for_op(
            get_custom(init_idx), ["cond", "if_true", "if_false"]
        )

        cluster_stride <<= 1

    # Broadcast the final index result to all lanes.  If there was a tie for the
    # top value, then another lane has a different index that it thinks is the
    # maximum index, and multiple elements will be masked for the next iteration
    # if we don't communicate a canonical answer.
    final_idx_broadcast = ShuffleOp(init_idx, 0, subgroup_size, ShuffleMode.IDX)
    final_idx_broadcast_node = get_graph_node(final_idx_broadcast, graph, location)
    get_custom(final_idx_broadcast_node).index = get_custom(init_idx).index

    return init_val, final_idx_broadcast_node


def validate_topk_sources(
    topk_src: list[fx.Node],
    reduction_dim,
) -> int:
    """
    Validation checks for topk sources.

    Args:
        topk_src: List of topk source nodes
        reduction_dim: The dimension to reduce over

    Returns:
        local_reduce_size: The size of local reduction
    """
    if not get_custom(topk_src[0]).type.symbolic_shape:
        raise ValueError(f"No symbolic shape found for topk source {topk_src[0]}")
    src_fastest_dims = [get_custom(arg).type.symbolic_shape[-1] for arg in topk_src]
    if not all_equal(src_fastest_dims):
        raise NotImplementedError("NYI: Expect all topk_src to have same fastest dim.")
    if reduction_dim is not src_fastest_dims[0]:
        raise NotImplementedError(
            f"Only implemented topk on fastest dimension. Got {reduction_dim} and {src_fastest_dims}."
        )

    get_thread_shape = lambda index: max(subs_idxc(x.size) for x in index.values())
    local_reduce_sizes = []
    for arg in topk_src:
        thread_shape = get_thread_shape(get_custom(arg).index)
        local_reduce_sizes.append(thread_shape)

    if not all_equal(local_reduce_sizes):
        raise NotImplementedError(
            "NYI: Expect all topk_src to have same local reduce size."
        )

    return local_reduce_sizes[0]


def generate_initial_indices(
    topk_src: list[fx.Node],
    reduction_dim,
    local_reduce_size: int,
    constraints: list[Constraint],
    graph: fx.Graph,
    location: Optional[CapturedLocation],
) -> list[fx.Node]:
    """
    Generate initial indices using SelfIndex.

    Args:
        topk_src: List of topk source nodes
        reduction_dim: The dimension to reduce over
        local_reduce_size: Size of local reduction
        constraints: List of constraints from the trace
        graph: FX graph to add nodes to

    Returns:
        List of self index nodes
    """
    self_indices = []
    for register_idx, src in enumerate(topk_src):
        src_custom = get_custom(src)
        self_index = SelfIndex(dim=reduction_dim, dtype=tkl.i32)
        index_node = get_graph_node(self_index, graph, location)
        src_register_size = subs_idxc(src_custom.index[reduction_dim].size)
        get_custom(index_node).index = construct_self_index_index_sequence(
            reduction_dim,
            local_reduce_size,
            constraints,
            register_idx,
            src_register_size,
        )
        self_indices.append(index_node)
    return self_indices


def perform_topk_reductions(
    working_src: list[fx.Node],
    self_indices: list[fx.Node],
    k_size: int,
    reduction_dim,
    local_reduce_size: int,
    binary_fn: Callable,
    graph: fx.Graph,
    node: fx.Node,
    subgroup_size: int,
    induction_vars: list,
    location: Optional[CapturedLocation],
) -> tuple[list[fx.Node], list[fx.Node]]:
    """
    For each k from 0 to K-1:
      a. Local TopK: Each thread finds the local maximum and its index.
      b. Intrawave TopK: Each thread finds the global maximum across threads
         using butterfly shuffle, tracking both values and indices.
      c. Keep track of the top value and index register nodes.
      d. Mask out the value at the found index to exclude it from future iterations.

    Args:
        working_src: Working copy of source nodes (will be modified)
        self_indices: Self index nodes
        k_size: Number of top-k elements to find
        reduction_dim: The dimension to reduce over
        local_reduce_size: Size of local reduction
        binary_fn: Binary function for comparison
        graph: FX graph to add nodes to
        node: Original topk node (for vector_shapes)
        subgroup_size: Number of threads per wave
        induction_vars: Induction variables from tiling constraints

    Returns:
        Tuple of (top_value_registers, top_index_registers)
    """
    # Collect the top values and registers for later packing.
    top_value_registers = []
    top_index_registers = []

    cluster_size, cluster_stride = (None, None)

    # We emit the reduction K times to get the top K elements.  This
    # requires a static K.  A better next approach would be to emit a
    # loop.
    for k in range(k_size):
        local_val, local_idx = emit_local_topk_reduction(
            binary_fn,
            working_src,
            self_indices,
            graph,
            local_reduce_size,
            location,
        )

        if not cluster_size:
            cluster_size, cluster_stride = determine_shuffle_config(
                working_src[0].index,
                reduction_dim,
                node.vector_shapes,
                subgroup_size,
                induction_vars,
            )
        global_val, global_idx = emit_intrawave_topk_reduction(
            binary_fn,
            local_val,
            local_idx,
            graph,
            subgroup_size,
            cluster_size,
            cluster_stride,
            location,
        )

        top_value_registers.append(global_val)
        top_index_registers.append(global_idx)

        # Masking: set the found maximum value to negative infinity for next iteration
        if k < k_size - 1:
            dtype = get_custom(working_src[0]).type.dtype
            neg_inf_constant = NewScalar(dtype=dtype, value=float("-inf"))
            neg_inf_node = get_graph_node(neg_inf_constant, graph, location)

            for i, src in enumerate(working_src):
                src_custom = get_custom(src)
                global_idx_broadcast = get_graph_node(
                    Broadcast(global_idx, (reduction_dim,)), graph, location
                )
                get_custom(global_idx_broadcast).index = get_custom(
                    self_indices[i]
                ).index
                is_at_max_index = get_graph_node(
                    Eq(self_indices[i], global_idx_broadcast), graph, location
                )
                is_at_max_index.index = self_indices[i].index

                src = get_graph_node(
                    SelectOp(is_at_max_index, neg_inf_node, src), graph, location
                )
                get_custom(src).index = get_custom(working_src[i]).index
                resolve_broadcasting_for_op(
                    get_custom(src), ["cond", "if_true", "if_false"]
                )

                working_src[i] = src

    return top_value_registers, top_index_registers


def pack_topk_results(
    top_value_registers: list[fx.Node],
    top_index_registers: list[fx.Node],
    topk_src: list[fx.Node],
    custom,
    k_size: int,
    graph: fx.Graph,
    location: Optional[CapturedLocation],
) -> tuple[fx.Node, fx.Node]:
    """
    Pack the collected register nodes into vector registers of K elements.

    Args:
        top_value_registers: List of top value nodes
        top_index_registers: List of top index nodes
        topk_src: Original topk source nodes
        custom: Custom op for the topk node
        k_size: Number of top-k elements
        graph: FX graph to add nodes to

    Returns:
        Tuple of (final_values, final_indices)
    """
    # Pack the collected register nodes into vector registers of K elements
    target_shape = {custom.k_dim: k_size}
    final_values = get_graph_node(
        Reshape(top_value_registers, target_shape), graph, location
    )
    final_indices = get_graph_node(
        Reshape(top_index_registers, target_shape), graph, location
    )

    # Create index using same keys as original input topk_src
    original_index = get_custom(topk_src[0]).index
    final_values_index = {}
    for dim in original_index.keys():
        final_values_index[dim] = IndexSequence(0, k_size, 1)
    get_custom(final_values).index = final_values_index
    get_custom(final_indices).index = final_values_index.copy()

    custom.type = [
        get_custom(final_values).type,
        get_custom(final_indices).type,
    ]

    # Store the computed results on the fx node for handle_topk_get_results
    custom.fx_node.decomposed_values = final_values
    custom.fx_node.decomposed_indices = final_indices

    return final_values, final_indices


def decompose_topk_op(
    node: fx.Node,
    constraints: list[Constraint],
    subgroup_size: int,
    induction_vars: list[IndexExpr | None],
) -> None:
    """
    Decompose a single topk operation.

      1. Validation checks.
      2. Generate initial indices using SelfIndex.
      3. Do topk reduction to a lists of single-element registers for indices and values.
      4. Pack the collected register nodes into vector registers of K elements.

    Args:
        node: The topk node to decompose
        constraints: List of constraints from the trace
        subgroup_size: Number of threads per wave
        induction_vars: Induction variables from tiling constraints
    """
    custom = get_custom(node)
    with custom.graph.inserting_before(custom.fx_node):
        topk_src = custom.arg
        k_size = subs_idxc(custom.k_dim)
        reduction_dim = custom.dim
        # TODO - can add arg for max/min
        binary_fn = Maximum

        if not isinstance(topk_src, (list, tuple)):
            topk_src = [topk_src]

        working_src = list(topk_src)

        local_reduce_size = validate_topk_sources(topk_src, reduction_dim)

        self_indices = generate_initial_indices(
            topk_src,
            reduction_dim,
            local_reduce_size,
            constraints,
            custom.graph,
            custom.location,
        )

        top_value_registers, top_index_registers = perform_topk_reductions(
            working_src,
            self_indices,
            k_size,
            reduction_dim,
            local_reduce_size,
            binary_fn,
            custom.graph,
            node,
            subgroup_size,
            induction_vars,
            custom.location,
        )

        pack_topk_results(
            top_value_registers,
            top_index_registers,
            topk_src,
            custom,
            k_size,
            custom.graph,
            custom.location,
        )

        # Don't delete the TopkOp yet - instead we handle the GetResult ops
        # that target them, and remove them together.


def decompose_topk_ops(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    The lowering for topk operations.  The meat is in `decompose_topk_op`.
    """
    topk_nodes = trace.walk(lambda node: isinstance(get_custom(node), TopkOp))
    if not topk_nodes:
        return

    hardware_constraint = next(
        c for c in constraints if isinstance(c, HardwareConstraint)
    )
    induction_vars = [
        c.induction_var for c in constraints if isinstance(c, TilingConstraint)
    ]

    subgroup_size = hardware_constraint.threads_per_wave

    for node in topk_nodes:
        decompose_topk_op(node, constraints, subgroup_size, induction_vars)

    # After decomposing topk ops, handle GetResult operations
    handle_topk_get_results(trace)
    remove_topk_ops(trace)
    DCE(trace)


def handle_topk_get_results(trace: CapturedTrace):
    """
    Handle GetResult operations on decomposed TopkOp nodes by replacing them
    with the appropriate computed values or indices.
    """
    get_result_nodes = trace.walk(lambda node: isinstance(get_custom(node), GetResult))

    for node in get_result_nodes:
        custom = get_custom(node)
        parent_op = get_custom(custom.value)

        if isinstance(parent_op, TopkOp):
            with custom.graph.inserting_before(custom.fx_node):
                if custom.res_idx == 0:
                    replacement = custom.value.decomposed_values
                elif custom.res_idx == 1:
                    replacement = custom.value.decomposed_indices
                else:
                    raise ValueError(
                        f"TopkOp only returns 2 values, but GetResult requested index {custom.res_idx}"
                    )
                custom.replace_all_uses_with(replacement)
                custom.fx_node.graph.erase_node(custom.fx_node)


def remove_topk_ops(trace: CapturedTrace):
    """
    Remove all TopkOp nodes from the graph after they have been decomposed
    and their GetResult operations have been resolved.
    """
    topk_nodes = trace.walk(lambda node: isinstance(get_custom(node), TopkOp))

    for node in topk_nodes:
        custom = get_custom(node)
        if custom.fx_node.users:
            remaining_users = [str(get_custom(user)) for user in custom.fx_node.users]
            raise RuntimeError(
                f"TopkOp {custom} still has users after decomposition: {remaining_users}."
                "All GetResult operations should have been resolved first."
            )
        custom.fx_node.graph.erase_node(custom.fx_node)
