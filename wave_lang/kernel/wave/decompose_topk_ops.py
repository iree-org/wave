# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from typing import Callable

import torch.fx as fx

import wave_lang.kernel.lang as tkl

from .._support.tracing import CapturedTrace
from .._support.indexing import IndexSequence, IndexingContext
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    Allocate,
    Conditional,
    CustomOp,
    Extract,
    GetResult,
    Gt,
    Maximum,
    NewScalar,
    Placeholder,
    Read,
    SelectOp,
    SelfIndex,
    TopkOp,
    ShuffleOp,
    Write,
    Eq,
    Broadcast,
    get_custom,
)
import wave_lang.kernel.lang as tkl
from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    TilingConstraint,
    WaveConstraint,
    WorkgroupConstraint,
)
from .utils.classes import ShuffleMode
from .utils.general_utils import all_equal
from .utils.graph_utils import DCE
from .utils.symbol_utils import subs_idxc
from .utils.graph_utils import get_outer_node
from .decompose_reduce_ops import determine_shuffle_config
from ..lang.kernel_buffer import AddressSpace
from .analysis.index_sequence_analysis import resolve_broadcasting_for_op


def get_graph_node(custom: CustomOp, graph: fx.Graph) -> fx.Node:
    custom.add_to_graph(graph)
    custom = custom.fx_node
    return custom


def construct_self_index_index_sequence(
    reduction_dim, elements_per_thread, constraints, register_idx, src_register_size
):
    idxc = IndexingContext.current()
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
) -> tuple[fx.Node, fx.Node]:
    """
    Does topk reduction over lists of values and indices, returning both the reduced value and its index.
    """
    init_val = src_values[0]
    init_idx = src_indices[0]

    for i in range(1, len(src_values)):
        cmp_result = get_graph_node(Gt(src_values[i], init_val), graph)
        init_val = get_graph_node(SelectOp(cmp_result, src_values[i], init_val), graph)
        init_idx = get_graph_node(SelectOp(cmp_result, src_indices[i], init_idx), graph)
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
) -> tuple[fx.Node, fx.Node]:
    """
    Does topk reduction over singular value and index variables.
    """
    init_val = get_graph_node(Extract(src_val, [0]), graph)
    init_idx = get_graph_node(Extract(src_idx, [0]), graph)
    get_custom(init_val).index = get_custom(src_val).index
    get_custom(init_idx).index = get_custom(src_idx).index
    # From here on out we are dealing with scalar values
    init_val.type.symbolic_shape = ()
    init_idx.type.symbolic_shape = ()

    for i in range(1, local_reduction_size):
        cur_val = get_graph_node(Extract(src_val, [i]), graph)
        cur_idx = get_graph_node(Extract(src_idx, [i]), graph)
        cur_val.type.symbolic_shape = ()
        cur_idx.type.symbolic_shape = ()

        cmp_result = get_graph_node(Gt(cur_val, init_val), graph)
        init_val = get_graph_node(SelectOp(cmp_result, cur_val, init_val), graph)
        init_idx = get_graph_node(SelectOp(cmp_result, cur_idx, init_idx), graph)
        get_custom(init_val).index = get_custom(cur_val).index
        get_custom(init_idx).index = get_custom(cur_idx).index

    return init_val, init_idx


def emit_local_topk_reduction(
    binary_fn: Callable,
    reduction_src: list[fx.Node],
    reduction_indices: list[fx.Node],
    graph: fx.Graph,
    local_reduction_size,
) -> tuple[fx.Node, fx.Node]:
    """
    Does topk reduction over all elements carried along by TopkOp at local
    thread level. This returns both the top value and its index.
    """
    src_val, src_idx = emit_sources_topk_reduction(
        binary_fn, reduction_src, reduction_indices, graph
    )

    local_val, local_idx = emit_variable_topk_reduction(
        binary_fn, src_val, src_idx, graph, local_reduction_size
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
) -> tuple[fx.Node, fx.Node]:
    """
    Reduce data across threads in a wave by doing butterfly shuffle for topk.
    """
    init_val = src_val
    init_idx = src_idx
    num_steps = int(math.log2(float(cluster_size)))

    for _ in range(num_steps):
        # Shuffle both values and indices
        shuffle_val = ShuffleOp(
            init_val, cluster_stride, subgroup_size, ShuffleMode.XOR
        )
        shuffle_val_node = get_graph_node(shuffle_val, graph)
        shuffle_idx = ShuffleOp(
            init_idx, cluster_stride, subgroup_size, ShuffleMode.XOR
        )
        shuffle_idx_node = get_graph_node(shuffle_idx, graph)

        get_custom(shuffle_val_node).index = get_custom(init_val).index
        get_custom(shuffle_idx_node).index = get_custom(init_idx).index

        val_greater = get_graph_node(Gt(shuffle_val_node, init_val), graph)

        init_val = get_graph_node(
            SelectOp(val_greater, shuffle_val_node, init_val),
            graph,
        )
        init_idx = get_graph_node(
            SelectOp(val_greater, shuffle_idx_node, init_idx),
            graph,
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
    final_idx_broadcast_node = get_graph_node(final_idx_broadcast, graph)
    get_custom(final_idx_broadcast_node).index = get_custom(init_idx).index

    return init_val, final_idx_broadcast_node


def decompose_topk_ops(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    The lowering for topk operations:
      1. Generate initial indices using SelfIndex for each source.
      2. Allocate shared memory buffers to store K top values and K top indices.
      3. For each k from 0 to K-1:
         a. Local TopK: Each thread finds the local maximum and its index.
         b. Intrawave TopK: Each thread finds the global maximum across threads
            using butterfly shuffle, tracking both values and indices.
         c. Store the top value and index in shared memory at position k.
         d. Mask out the value at the found index to exclude it from future iterations.
      4. Read and return vectors of K top values and K top indices from shared memory.
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

    wave_constraint_map = {
        c.dim: c for c in constraints if isinstance(c, WaveConstraint)
    }
    workgroup_constraint_map = {
        c.dim: c for c in constraints if isinstance(c, WorkgroupConstraint)
    }
    subgroup_size = hardware_constraint.threads_per_wave

    lane_id = (
        hardware_constraint.linearized_thread_id % hardware_constraint.threads_per_wave
    )

    for node in topk_nodes:
        custom = get_custom(node)
        with custom.graph.inserting_before(custom.fx_node):
            topk_src = custom.arg
            k_size = subs_idxc(custom.k_dim)
            reduction_dim = custom.dim_to_reduce
            # TODO - can add arg for max/min
            binary_fn = Maximum

            if not isinstance(topk_src, (list, tuple)):
                topk_src = [topk_src]

            working_src = list(topk_src)

            # Will generate initial indices for each source using SelfIndex after calculating local_reduce_sizes
            self_indices = []

            if not get_custom(topk_src[0]).type.symbolic_shape:
                raise ValueError(
                    f"No symbolic shape found for topk source {topk_src[0]}"
                )
            src_fastest_dims = [
                get_custom(arg).type.symbolic_shape[-1] for arg in topk_src
            ]
            if not all_equal(src_fastest_dims):
                raise NotImplementedError(
                    "NYI: Expect all topk_src to have same fastest dim."
                )
            if reduction_dim is not src_fastest_dims[0]:
                raise NotImplementedError(
                    f"Only implemented topk on fastest dimension. Got {reduction_dim} and {src_fastest_dims}."
                    f"\n{custom}"
                )

            get_thread_shape = lambda index: max(
                subs_idxc(x.size) for x in index.values()
            )
            local_reduce_sizes = []
            for arg in topk_src:
                try:
                    op = get_custom(arg)

                    thread_shape = get_thread_shape(op.index)
                    local_reduce_sizes.append(thread_shape)
                except Exception as e:
                    index_str = "\n".join(f"{k}: {v}" for k, v in op.index.items())
                    raise RuntimeError(
                        f"Error in decompose_topk_ops: {arg} with index\n"
                        f"{index_str}\n{topk_src=}\n{reduction_dim=}"
                    ) from e

            if not all_equal(local_reduce_sizes):
                raise NotImplementedError(
                    "NYI: Expect all topk_src to have same local reduce size."
                )

            for register_idx, src in enumerate(topk_src):
                src_custom = get_custom(src)
                self_index = SelfIndex(dim=reduction_dim, dtype=tkl.i32)
                index_node = get_graph_node(self_index, custom.graph)
                src_register_size = subs_idxc(src_custom.index[reduction_dim].size)
                get_custom(index_node).index = construct_self_index_index_sequence(
                    reduction_dim,
                    local_reduce_sizes[0],
                    constraints,
                    register_idx,
                    src_register_size,
                )
                self_indices.append(index_node)

            dtype = get_custom(topk_src[0]).type.dtype

            top_values_alloc = Allocate(
                shape=(k_size,),
                distributed_shape=(k_size,),
                dtype=dtype,
                address_space=AddressSpace.SHARED_MEMORY,
            )
            top_values_alloc.add_to_graph(custom.graph)

            top_indices_alloc = Allocate(
                shape=(k_size,),
                distributed_shape=(k_size,),
                dtype=tkl.i32,
                address_space=AddressSpace.SHARED_MEMORY,
            )
            top_indices_alloc.add_to_graph(custom.graph)

            # Track all conditional writes for final read dependency
            all_conditional_writes = []

            cluster_size, cluster_stride = (None, None)

            # We emit the reduction K times to get the top K elements.  This
            # requires a static K.  A better next approach would be to emit a
            # loop.
            for k in range(k_size):
                local_val, local_idx = emit_local_topk_reduction(
                    binary_fn,
                    working_src,
                    self_indices,
                    custom.graph,
                    local_reduce_sizes[0],
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
                    custom.graph,
                    subgroup_size,
                    cluster_size,
                    cluster_stride,
                )

                # We need to write the final answer for the top element into shared memory, but just one lane needs to do it.
                execute_on_lane0_graph = fx.Graph()
                subgraph_name = f"execute_on_lane0_topk_write_{k}"

                # TODO - this is all cargo-culted.  Why is this placeholder stuff necessary?
                placeholder_val = get_graph_node(
                    Placeholder.from_fx_node(global_val), execute_on_lane0_graph
                )
                placeholder_val.type = get_custom(global_val).type

                placeholder_idx = get_graph_node(
                    Placeholder.from_fx_node(global_idx), execute_on_lane0_graph
                )
                placeholder_idx.type = get_custom(global_idx).type

                placeholder_values_alloc = get_graph_node(
                    Placeholder.from_fx_node(get_custom(top_values_alloc.fx_node)),
                    execute_on_lane0_graph,
                )
                placeholder_values_alloc.type = get_custom(
                    top_values_alloc.fx_node
                ).type
                placeholder_values_alloc.meta["lifted"] = top_values_alloc.fx_node

                placeholder_indices_alloc = get_graph_node(
                    Placeholder.from_fx_node(get_custom(top_indices_alloc.fx_node)),
                    execute_on_lane0_graph,
                )
                placeholder_indices_alloc.type = get_custom(
                    top_indices_alloc.fx_node
                ).type
                placeholder_indices_alloc.meta["lifted"] = top_indices_alloc.fx_node

                write_val = Write(
                    placeholder_val, placeholder_values_alloc, 1
                ).add_to_graph(execute_on_lane0_graph)
                get_custom(write_val).index = {custom.k_dim: IndexSequence(k, 1, 1)}

                write_idx = Write(
                    placeholder_idx, placeholder_indices_alloc, 1
                ).add_to_graph(execute_on_lane0_graph)
                get_custom(write_idx).index = {custom.k_dim: IndexSequence(k, 1, 1)}

                implicit_capture_val = get_outer_node(global_val)
                implicit_capture_idx = get_outer_node(global_idx)
                lane_id_reg = get_graph_node(NewScalar(lane_id, tkl.i32), custom.graph)
                zero_reg = get_graph_node(NewScalar(0, tkl.i32), custom.graph)
                is_lane_0 = get_graph_node(Eq(lane_id_reg, zero_reg), custom.graph)

                execute_on_lane0 = get_graph_node(
                    Conditional(
                        is_lane_0,
                        subgraph_name=subgraph_name,
                        implicit_captures=[
                            implicit_capture_val,
                            implicit_capture_idx,
                            top_values_alloc.fx_node,
                            top_indices_alloc.fx_node,
                        ],
                    ),
                    custom.graph,
                )
                execute_on_lane0_graph.parent_op = execute_on_lane0
                trace.add_subgraph(subgraph_name, execute_on_lane0_graph)
                trace.get_root_graph().subgraphs[subgraph_name] = execute_on_lane0_graph
                all_conditional_writes.extend([execute_on_lane0, write_val, write_idx])

                # Masking: set the found maximum value to negative infinity for next iteration
                if k < k_size - 1:
                    dtype = get_custom(working_src[0]).type.dtype
                    neg_inf_constant = NewScalar(dtype=dtype, value=float("-inf"))
                    neg_inf_node = get_graph_node(neg_inf_constant, custom.graph)

                    for i, src in enumerate(working_src):
                        src_custom = get_custom(src)
                        global_idx_broadcast = get_graph_node(
                            Broadcast(global_idx, (reduction_dim,)), custom.graph
                        )
                        get_custom(global_idx_broadcast).index = get_custom(
                            self_indices[i]
                        ).index
                        is_at_max_index = get_graph_node(
                            Eq(self_indices[i], global_idx_broadcast), custom.graph
                        )
                        is_at_max_index.index = self_indices[i].index

                        src = get_graph_node(
                            SelectOp(is_at_max_index, neg_inf_node, src), custom.graph
                        )
                        get_custom(src).index = get_custom(working_src[i]).index
                        resolve_broadcasting_for_op(
                            get_custom(src), ["cond", "if_true", "if_false"]
                        )

                        working_src[i] = src

            # Read the final results from shared memory with write dependencies
            final_values = get_graph_node(
                Read(
                    top_values_alloc.fx_node,
                    elements_per_thread=k_size,
                    _write_dependency=all_conditional_writes,
                ),
                custom.graph,
            )
            get_custom(final_values).index = {custom.k_dim: IndexSequence(0, 1, 1)}
            final_indices = get_graph_node(
                Read(
                    top_indices_alloc.fx_node,
                    elements_per_thread=k_size,
                    _write_dependency=all_conditional_writes,
                ),
                custom.graph,
            )
            get_custom(final_indices).index = {custom.k_dim: IndexSequence(0, 1, 1)}

            final_values_type = get_custom(final_values).type
            final_indices_type = get_custom(final_indices).type
            custom.type = [final_values_type, final_indices_type]

            # Store the computed results on the fx node for handle_topk_get_results
            custom.fx_node.decomposed_values = final_values
            custom.fx_node.decomposed_indices = final_indices

            # Don't delete the TopkOp yet - instead we handle the GetResult ops
            # that target them, and remove them together.

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
