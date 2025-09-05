# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Dict, List

import torch.fx as fx
from torch.utils import _pytree as pytree

import wave_lang.kernel.lang as tkl

from .._support.tracing import CapturedTrace
from ..compiler.kernel_codegen import filter_fx_graph
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    MMA,
    Conditional,
    CustomOp,
    Ge,
    Iterate,
    Lt,
    NewScalar,
    ScaledMMA,
    SchedulingBarrier,
    SetWavePrio,
    SharedMemoryBarrier,
    WorkgroupBarrier,
    Write,
    get_custom,
)
from .constraints import (
    Constraint,
    HardwareConstraint,
    get_constrained_shape,
)
from .scheduling.schedule_enums import SchedulingType
from .utils.general_utils import (
    flatten_list,
    get_hardware_constraint,
    is_shared_read,
    topological_sort_with_dependencies,
)
from .utils.symbol_utils import subs_idxc
from .utils.classes import AttentionOperationType

##############################################################
# General graph helper functions
##############################################################

"""
Formatting for schedule reordering strategies:
Values: 0xAB where:
* A = Strategy types:
  * 0 = No reordering
  * 1 = Reordering w/o Ping Pong
  * 2 = Ping Pong reordering
* B = Bitwidth:
    * 0 = 4 bits
    * 1 = 8 bits
    * 2 = 16 bits
* C = Type of MMA.
    * 0 = MMA
    * 1 = ScaledMMA
* D enumerates different strategy that share the same 0xAB* bits.
"""


class SchedReorderStrategy(Enum):
    NONE = 0x00
    TWO_PP_CLUSTER = 0x220
    MXFP4_PP_CLUSTER = 0x101


def is_pingpong_strategy(strategy):
    """Convert enum value to hex string and checks A-bit"""
    return int(hex(strategy.value)[2]) == 2


"""
Track tile size requirements for different reordering strategies.
"""


@dataclass
class CompatibleBlockSize:
    block_m: int
    block_n: int
    block_k: int
    bitwidth: int
    mma_type: type


twoPPConfig = CompatibleBlockSize(128, 256, 64, 16, MMA)
MXFP4PPConfig = CompatibleBlockSize(256, 128, 256, 4, ScaledMMA)


class InsertionMode(Enum):
    BEFORE = 0
    AFTER = 1


class InsertionPoint(object):
    """
    Helper class to keep track of movements/insertion of ops into very
    precise/specific locations before or after an another op
    (referred here as "anchor op".)
    """

    def __init__(
        self, mode: InsertionMode, op: fx.Node, anchor_op: fx.Node | Iterable[fx.Node]
    ):
        # Initialize insertion mode.
        if not isinstance(mode, InsertionMode):
            raise ValueError("Unexpected insetion mode.")
        self.mode = mode

        # Initialize source location.
        if isinstance(anchor_op, fx.Node):
            self.anchor_op = anchor_op
        elif isinstance(anchor_op, Iterable):
            if mode == InsertionMode.AFTER:
                self.anchor_op = flatten_list(anchor_op)[-1]
            else:
                self.anchor_op = flatten_list(anchor_op)[0]
        else:
            raise ValueError("Unexpected src type")

        if not isinstance(op, fx.Node):
            raise ValueError("Unexpected op type.")
        self.op = op

    @property
    def graph(self):
        return self.anchor_op.graph


def insert_op_after(op, anchor_op):
    return InsertionPoint(InsertionMode.AFTER, op, anchor_op)


def insert_op_before(op, anchor_op):
    return InsertionPoint(InsertionMode.BEFORE, op, anchor_op)


def get_graph_node(custom: CustomOp, graph: fx.Graph) -> fx.Node:
    custom.add_to_graph(graph)
    custom = custom.fx_node
    return custom


def get_mma_bitwidth(mma_node):
    lhs_dtype = get_custom(mma_node).lhs_type.dtype
    rhs_dtype = get_custom(mma_node).rhs_type.dtype
    if lhs_dtype != rhs_dtype:
        raise NotImplementedError("Not expecting mixed lhs/rhs dtype.")
    return lhs_dtype.bitwidth()


def get_mma_tile_size(mma_nodes, constraints):
    # Check that all MMA nodes comes from single pre_expanded source.
    # Early exit if not.
    pre_expansion_ids = set([get_custom(node).pre_expansion_id for node in mma_nodes])
    if len(pre_expansion_ids) != 1:
        return None, None, None

    # Just using first one because we know it all came from same source,
    # and hence all the dims should be the same.
    mma_node = mma_nodes[0]
    filter_batch_dims = lambda dims: [
        dim for dim in dims if mma_node.vector_shapes[dim] != 0
    ]
    custom = get_custom(mma_node)
    lhs_dim = set(get_custom(custom.lhs).indexing_dims)
    rhs_dim = set(get_custom(custom.rhs).indexing_dims)
    acc_dim = set(get_custom(custom.acc).indexing_dims)

    m_dims = filter_batch_dims(list(lhs_dim - rhs_dim))
    n_dims = filter_batch_dims(list(rhs_dim - lhs_dim))
    # Subtract by acc dim to remove batch dims.
    k_dims = filter_batch_dims(list(rhs_dim.intersection(lhs_dim) - acc_dim))

    # Only expected single dim for each M,N,K.
    if len(m_dims) != 1 or len(n_dims) != 1 or len(k_dims) != 1:
        return None, None, None
    mnk_dim = [m_dims[0], n_dims[0], k_dims[0]]
    mnk_tile = [
        subs_idxc(tile_sym) for tile_sym in get_constrained_shape(mnk_dim, constraints)
    ]
    return mnk_tile


def schedule_nodes_to_graph(graph, node_map, nodes):

    for node in nodes:
        custom = get_custom(node)
        new_node = custom.copy(
            new_graph=graph, arg_transform=lambda x: node_map[x] if x in node_map else x
        )
        node_map[node] = new_node.fx_node


def insert_scheduling_nodes(reordered_original_nodes, insertion_points):
    """
    This function helps output a new list/order of nodes with scheduling
    nodes inserted in precise locations specified by insertion_points.
    """
    reordered_nodes = reordered_original_nodes.copy()
    for insertion_point in insertion_points:
        if insertion_point.mode == InsertionMode.BEFORE:
            insert_index = reordered_nodes.index(insertion_point.anchor_op)
        elif insertion_point.mode == InsertionMode.AFTER:
            insert_index = reordered_nodes.index(insertion_point.anchor_op) + 1
        else:
            raise ValueError("Invalid InsertionMode found!")
        reordered_nodes.insert(insert_index, insertion_point.op)
    return reordered_nodes


def reorder_graph(graph, clusters):
    node_list = list(graph.nodes)
    prune_duplicates = lambda x: list(dict.fromkeys(x))
    # This lambda filters away scheduling ops generated in this pass
    # such as SetWavePrio, WorkgroupBarrier, SchedulingBarriers.
    prune_reordering_nodes = lambda x: [
        x for x in flattened_cluster if not isinstance(x, InsertionPoint)
    ]

    flattened_cluster, _ = pytree.tree_flatten(clusters)
    flattened_cluster = prune_duplicates(flattened_cluster)

    # Get location of where cluster start and end.
    original_cluster_nodes = prune_reordering_nodes(flattened_cluster)
    ordered_cluster = sorted(original_cluster_nodes)
    earliest_cluster_node = ordered_cluster[0]
    latest_cluster_node = ordered_cluster[-1]

    # Slice node_list to get precluster nodes
    pre_cluster_nodes = [x for x in node_list if x < earliest_cluster_node]

    # Slice node_list to get post cluster nodes
    post_cluster_nodes = [x for x in node_list if x > latest_cluster_node]

    # Exhaustive clustered nodes
    exhaustive_cluster_nodes = [
        x for x in node_list if x >= earliest_cluster_node and x <= latest_cluster_node
    ]

    reordered_original_nodes = topological_sort_with_dependencies(
        original_cluster_nodes, exhaustive_cluster_nodes, pre_cluster_nodes
    )

    total_reordered_node = (
        len(pre_cluster_nodes) + len(reordered_original_nodes) + len(post_cluster_nodes)
    )

    # Reorederd should have same number of nodes with original.
    # Bail out/not do ping-pong if this analysis fails.
    # Sometime this could be impacted if use_scheduling_barriers=True,
    # since we get an unexpected workgroup barrier.
    if len(node_list) != total_reordered_node:
        return None

    insertion_points = [
        node for node in flattened_cluster if isinstance(node, InsertionPoint)
    ]

    # Insertion of scheduling nodes specified in this pass.
    # e.g set_wave_prio, wg_barrier, shared_memory_barrier.
    reordered_nodes = insert_scheduling_nodes(
        reordered_original_nodes, insertion_points
    )

    # Schedule pre-cluster, cluster, and post-cluster nodes in new graph.
    reordered_graph = fx.Graph()
    node_map = {}
    schedule_nodes_to_graph(reordered_graph, node_map, pre_cluster_nodes)
    schedule_nodes_to_graph(reordered_graph, node_map, reordered_nodes)
    schedule_nodes_to_graph(reordered_graph, node_map, post_cluster_nodes)

    return reordered_graph


##############################################################
# Ping Pong helper functions
##############################################################


def slice_mma(mma_nodes, lhs_nodes, rhs_nodes, num_slice):
    sliced_mma_nodes = [[] for _ in range(num_slice)]
    sliced_lhs_nodes = [[] for _ in range(num_slice)]
    sliced_rhs_nodes = [[] for _ in range(num_slice)]

    reduction_dim = get_custom(mma_nodes[0]).reduction_dim
    reduction_dim_ids = set(
        [get_custom(node).expanded_dims[reduction_dim] for node in mma_nodes]
    )

    # Checking that MMAs is valid.
    reduction_expand_size = len(reduction_dim_ids)
    assert reduction_expand_size >= num_slice and reduction_expand_size % num_slice == 0
    assert all(x in reduction_dim_ids for x in range(reduction_expand_size))

    size_of_slice = reduction_expand_size // num_slice
    for mma_node, lhs_node, rhs_node in zip(mma_nodes, lhs_nodes, rhs_nodes):
        custom = get_custom(mma_node)
        k_id = custom.expanded_dims[reduction_dim]
        slice_id = k_id // size_of_slice
        sliced_mma_nodes[slice_id].append(mma_node)
        sliced_lhs_nodes[slice_id].append(lhs_node)
        sliced_rhs_nodes[slice_id].append(rhs_node)
    return sliced_mma_nodes, sliced_lhs_nodes, sliced_rhs_nodes


def slice_scale_mma(
    mma_nodes, lhs_nodes, rhs_nodes, lhs_scale_nodes, rhs_scale_nodes, num_slice
):
    sliced_mma_nodes = [[] for _ in range(num_slice)]
    sliced_lhs_nodes = [[] for _ in range(num_slice)]
    sliced_rhs_nodes = [[] for _ in range(num_slice)]
    sliced_lhs_scale_nodes = [[] for _ in range(num_slice)]
    sliced_rhs_scale_nodes = [[] for _ in range(num_slice)]

    reduction_dim = get_custom(mma_nodes[0]).reduction_dim
    reduction_dim_ids = set(
        [get_custom(node).expanded_dims[reduction_dim] for node in mma_nodes]
    )

    # Checking that MMAs is valid.
    reduction_expand_size = len(reduction_dim_ids)
    assert reduction_expand_size >= num_slice and reduction_expand_size % num_slice == 0
    assert all(x in reduction_dim_ids for x in range(reduction_expand_size))

    size_of_slice = reduction_expand_size // num_slice
    for mma_node, lhs_node, rhs_node, lhs_scale_node, rhs_scale_node in zip(
        mma_nodes, lhs_nodes, rhs_nodes, lhs_scale_nodes, rhs_scale_nodes
    ):
        custom = get_custom(mma_node)
        k_id = custom.expanded_dims[reduction_dim]
        slice_id = k_id // size_of_slice
        sliced_mma_nodes[slice_id].append(mma_node)
        sliced_lhs_nodes[slice_id].append(lhs_node)
        sliced_rhs_nodes[slice_id].append(rhs_node)
        sliced_lhs_scale_nodes[slice_id].append(lhs_scale_node)
        sliced_rhs_scale_nodes[slice_id].append(rhs_scale_node)
    return (
        sliced_mma_nodes,
        sliced_lhs_nodes,
        sliced_rhs_nodes,
        sliced_lhs_scale_nodes,
        sliced_rhs_scale_nodes,
    )


def insert_cond_barrier(cond_reg, trace, graph):
    barrier_graph = fx.Graph()
    barrier_graph_name = f"barrier_graph_{cond_reg.name}"
    WorkgroupBarrier().add_to_graph(barrier_graph)
    cond_barrier = Conditional(
        cond_reg,
        subgraph_name=barrier_graph_name,
        implicit_captures=[],
    ).add_to_graph(graph)
    barrier_graph.parent_op = cond_barrier
    trace.add_subgraph(barrier_graph_name, barrier_graph)
    return cond_barrier


def add_conditional_barriers_to_loop(custom_iterate, trace, hardware_constraint):
    """
    This function wraps loop with two cond barriers. First, hold half of the wave
    (waveHi) in a block before the loop so the barriers in the loop synchronize
    waves at the different point per the warp groups. After the loop, hold
    proceeding waves (waveLo) by calling cond_barrier on them.
    """
    graph = custom_iterate.graph
    # Compute midwave, where if wave_id > mid_wave => wave_high.
    flat_wave_count = math.prod(hardware_constraint.waves_per_block)
    assert flat_wave_count % 2 == 0
    mid_wave = flat_wave_count // 2

    flat_id = hardware_constraint.linearized_thread_id
    wave_id = flat_id // hardware_constraint.threads_per_wave

    # Inserting condition computation into graph.
    with graph.inserting_before(custom_iterate.fx_node):
        mid_wave_reg = get_graph_node(NewScalar(mid_wave, tkl.i32), graph)
        wave_id_reg = get_graph_node(NewScalar(wave_id, tkl.i32), graph)
        is_wave_hi = get_graph_node(Ge(wave_id_reg, mid_wave_reg), graph)
        is_wave_lo = get_graph_node(Lt(wave_id_reg, mid_wave_reg), graph)

    # Generating and inserting cond_barriers to correct place in graph.
    with graph.inserting_before(custom_iterate.fx_node):
        insert_cond_barrier(is_wave_hi, trace, graph)
    with graph.inserting_after(custom_iterate.fx_node):
        insert_cond_barrier(is_wave_lo, trace, graph)
    return


def insert_prefetch_loop_barriers(custom_iterate, cluster_graph, clusters):
    """
    This function manually inserts first barrier right before for loop, and
    the barrier inside the loop close to the end of the loop. This ordering
    allow for optimal latency hiding of shared writes for kernels with
    scheduling strategy == SchedulingType.PREFETCH.
    """
    graph = custom_iterate.graph
    with graph.inserting_before(custom_iterate.fx_node):
        SharedMemoryBarrier().add_to_graph(graph)
    # TODO: Replace this with just lgmkcnt or change lowering of
    # SharedMemoryBarrier to only do lgmkcnt w/os sbarrier
    with cluster_graph.inserting_before(cluster_graph.output_node()):
        clusters.append(SharedMemoryBarrier().add_to_graph(cluster_graph))
    return


##############################################################
# Ping Pong Transformation.
##############################################################


def is_compatible_strategy(mTile, nTile, kTile, mma_bitwidth, mma_type, strategy):
    return (
        mTile % strategy.block_m == 0
        and nTile % strategy.block_n == 0
        and kTile % strategy.block_k == 0
        and mma_bitwidth == strategy.bitwidth
        and mma_type == strategy.mma_type
    )


def select_reorder_strategy(
    mma_type, mTile, nTile, kTile, mma_bitwidth, hardware_constraint
):
    flat_wave_count = math.prod(hardware_constraint.waves_per_block)
    if flat_wave_count != 8:
        return SchedReorderStrategy.NONE
    if is_compatible_strategy(mTile, nTile, kTile, mma_bitwidth, mma_type, twoPPConfig):
        return SchedReorderStrategy.TWO_PP_CLUSTER
    elif is_compatible_strategy(
        mTile, nTile, kTile, mma_bitwidth, mma_type, MXFP4PPConfig
    ):
        return SchedReorderStrategy.MXFP4_PP_CLUSTER
    else:
        return SchedReorderStrategy.NONE


def transform_two_PP_clusters(
    mma_nodes,
    local_load_lhs,
    local_load_rhs,
    global_load_lhs,
    global_load_rhs,
    local_write_lhs,
    local_write_rhs,
):
    num_slices = 2
    sliced_mma_nodes, sliced_local_load_lhs, sliced_local_load_rhs = slice_mma(
        mma_nodes, local_load_lhs, local_load_rhs, num_slice=num_slices
    )
    # Check that we have valid slice size for local_loads and mmas.
    assert len(sliced_mma_nodes) == len(sliced_local_load_rhs)
    assert len(sliced_mma_nodes) == len(sliced_local_load_lhs)
    assert len(sliced_mma_nodes) == num_slices

    clusters = []
    tmp_graph = fx.Graph()
    # 1st cluster interleaved local and global reads.
    clusters.append(sliced_local_load_lhs[0])
    clusters.append(sliced_local_load_rhs[0])
    clusters.append(
        insert_op_after(
            SchedulingBarrier([]).add_to_graph(tmp_graph), sliced_local_load_rhs[0]
        )
    )

    clusters.append(global_load_lhs)
    clusters.append(
        insert_op_after(SchedulingBarrier([]).add_to_graph(tmp_graph), global_load_lhs)
    )

    clusters.append(sliced_local_load_lhs[1])
    clusters.append(sliced_local_load_rhs[1])
    clusters.append(
        insert_op_after(
            SchedulingBarrier([]).add_to_graph(tmp_graph), sliced_local_load_rhs[1]
        )
    )

    clusters.append(global_load_rhs)
    clusters.append(
        insert_op_after(WorkgroupBarrier().add_to_graph(tmp_graph), global_load_rhs)
    )
    clusters.append(
        insert_op_after(SchedulingBarrier([]).add_to_graph(tmp_graph), clusters[-1].op)
    )

    # 2nd cluster mma_slice[0].
    clusters.append(
        insert_op_before(SetWavePrio(1).add_to_graph(tmp_graph), sliced_mma_nodes[0])
    )
    clusters.append(sliced_mma_nodes[0])
    clusters.append(
        insert_op_after(SetWavePrio(0).add_to_graph(tmp_graph), sliced_mma_nodes[0])
    )
    clusters.append(
        insert_op_after(SharedMemoryBarrier().add_to_graph(tmp_graph), clusters[-1].op)
    )
    clusters.append(
        insert_op_after(SchedulingBarrier([]).add_to_graph(tmp_graph), clusters[-1].op)
    )

    # 3rd cluster local writes.
    clusters.append(local_write_lhs)
    clusters.append(local_write_rhs)
    clusters.append(
        insert_op_after(WorkgroupBarrier().add_to_graph(tmp_graph), local_write_rhs)
    )
    clusters.append(
        insert_op_after(SchedulingBarrier([]).add_to_graph(tmp_graph), clusters[-1].op)
    )

    # 4th cluster mma_slice[1].
    clusters.append(
        insert_op_before(SetWavePrio(1).add_to_graph(tmp_graph), sliced_mma_nodes[1])
    )
    clusters.append(sliced_mma_nodes[1])
    clusters.append(
        insert_op_after(SetWavePrio(0).add_to_graph(tmp_graph), sliced_mma_nodes[1])
    )
    clusters.append(
        insert_op_after(SchedulingBarrier([]).add_to_graph(tmp_graph), clusters[-1].op)
    )

    return clusters


def transform_MXFP4_PP_clusters(
    mma_nodes,
    local_load_lhs,
    local_load_rhs,
    global_load_lhs,
    global_load_rhs,
    local_write_lhs,
    local_write_rhs,
    local_load_lhs_scale,
    local_load_rhs_scale,
    global_load_lhs_scale,
    global_load_rhs_scale,
    local_write_lhs_scale,
    local_write_rhs_scale,
):
    num_slices = 2
    (
        sliced_mma_nodes,
        sliced_local_load_lhs,
        sliced_local_load_rhs,
        sliced_local_load_lhs_scale,
        sliced_local_load_rhs_scale,
    ) = slice_scale_mma(
        mma_nodes,
        local_load_lhs,
        local_load_rhs,
        local_load_lhs_scale,
        local_load_rhs_scale,
        num_slice=num_slices,
    )

    # Check that we have valid slice size for local_loads and mmas.
    assert len(sliced_mma_nodes) == len(sliced_local_load_rhs)
    assert len(sliced_mma_nodes) == len(sliced_local_load_lhs)
    assert len(sliced_mma_nodes) == len(sliced_local_load_lhs_scale)
    assert len(sliced_mma_nodes) == len(sliced_local_load_rhs_scale)
    assert len(sliced_mma_nodes) == num_slices

    clusters = []

    clusters.append(global_load_rhs_scale)
    clusters.append(sliced_local_load_rhs_scale[0])
    clusters.append(sliced_local_load_rhs_scale[1])

    clusters.append(global_load_lhs_scale)
    clusters.append(sliced_local_load_lhs_scale[0])
    clusters.append(sliced_local_load_lhs_scale[1])

    clusters.append(global_load_rhs)
    clusters.append(sliced_local_load_rhs[0])
    clusters.append(sliced_local_load_rhs[1])

    clusters.append(global_load_lhs)
    clusters.append(sliced_local_load_lhs[0])
    clusters.append(sliced_local_load_lhs[1])

    clusters.append(sliced_mma_nodes[0])
    clusters.append(sliced_mma_nodes[1])
    clusters.append(local_write_lhs_scale)
    clusters.append(local_write_rhs_scale)
    clusters.append(local_write_lhs)
    clusters.append(local_write_rhs)

    return clusters


##############################################################
# Helper fn to classify/detect ops.
##############################################################


def get_closest_local_load(node: fx.Node):
    workqueue = deque([node])
    seen = set()
    can_extend = lambda x: isinstance(x, fx.Node) and x not in seen
    while workqueue:
        cur_node = workqueue.pop()
        if is_shared_read(get_custom(cur_node)):
            return cur_node
        child_nodes = [arg_node for arg_node in cur_node.args if can_extend(arg_node)]

        # Update ancestor and seen tracker, and workqueue with new child nodes.
        seen.update(child_nodes)
        workqueue.extend(child_nodes)
    return None


def get_local_loads(mma_nodes):
    local_load_lhs = []
    local_load_rhs = []
    for mma_node in mma_nodes:
        custom = get_custom(mma_node)
        lhs = get_closest_local_load(custom.lhs)
        rhs = get_closest_local_load(custom.rhs)
        if lhs == None or rhs == None:
            return None, None
        local_load_lhs.append(lhs)
        local_load_rhs.append(rhs)
    return local_load_lhs, local_load_rhs


def get_scale_local_loads(mma_nodes):
    local_load_lhs_scale = []
    local_load_rhs_scale = []
    for mma_node in mma_nodes:
        custom = get_custom(mma_node)
        lhs_scale = get_closest_local_load(custom.lhs_scale)
        rhs_scale = get_closest_local_load(custom.rhs_scale)
        if lhs_scale == None or rhs_scale == None:
            return None, None
        local_load_lhs_scale.append(lhs_scale)
        local_load_rhs_scale.append(rhs_scale)
    return local_load_lhs_scale, local_load_rhs_scale


def get_local_writes(local_loads):
    local_writes = set()
    for local_load in local_loads:
        custom = get_custom(local_load)
        cur_writes = [
            w
            for w in custom.memory.users
            if isinstance(get_custom(w), Write) and w.graph == custom.graph
        ]
        local_writes.update(cur_writes)
    return list(local_writes)


def get_global_loads(local_writes):
    global_loads = set()
    for local_write in local_writes:
        custom = get_custom(local_write)
        global_loads.add(custom.register_)
    return list(global_loads)


def schedule_reordering(
    trace: CapturedTrace,
    constraints: list[Constraint],
    scheduling_type: SchedulingType,
):
    """
    Ping Pong transformation is done by:
        1. Get Reduction/Iterate op
        2. Get MMAs inside reduction/iterate op that is tiled with reduction dim.
        3. Based on mma_node's expanded_dim and consumers we can classify global read, local read, mma, and local write.
    """
    if scheduling_type == SchedulingType.PREFETCH_ATTENTION:
        return attention_schedule_reordering(trace, constraints)
    if scheduling_type != SchedulingType.PREFETCH:
        return

    hardware_constraint = get_hardware_constraint(constraints)
    iterate_nodes = trace.walk(lambda node: isinstance(get_custom(node), Iterate))
    if not iterate_nodes:
        return
    for iterate_node in iterate_nodes:
        custom_iterate = get_custom(iterate_node)
        graph = trace.get_subgraph(custom_iterate.subgraph_name)
        iteration_dim = custom_iterate.axis
        # Get MMA nodes inside a for op, who's reduction dim is being tiled in the for op.
        mma_nodes = filter_fx_graph(
            graph,
            lambda node: isinstance(get_custom(node), (MMA, ScaledMMA))
            and get_custom(node).reduction_dim == iteration_dim,
        )
        # Early exit if no MMA found.
        if not mma_nodes:
            continue

        mma_types = set([type(get_custom(mma_node)) for mma_node in mma_nodes])
        # Only handle single MMA per kernel and need to have same type.s
        if len(mma_types) != 1:
            continue
        mma_type = mma_types.pop()
        local_load_lhs, local_load_rhs = get_local_loads(mma_nodes)
        # Early exit if cannot find either local loads
        if not local_load_lhs or not local_load_rhs:
            continue
        local_write_lhs = get_local_writes(local_load_lhs)
        local_write_rhs = get_local_writes(local_load_rhs)
        global_load_lhs = get_global_loads(local_write_lhs)
        global_load_rhs = get_global_loads(local_write_rhs)
        # Early exit if cannot find either operand's local write or global loads.
        if any(
            not memory_op
            for memory_op in [
                local_write_lhs,
                local_write_rhs,
                global_load_lhs,
                global_load_rhs,
            ]
        ):
            continue

        local_load_lhs_scale = None
        local_load_rhs_scale = None
        local_write_lhs_scale = None
        local_write_lhs_scale = None
        global_load_lhs_scale = None
        global_load_rhs_scale = None
        if mma_type == ScaledMMA:
            local_load_lhs_scale, local_load_rhs_scale = get_scale_local_loads(
                mma_nodes
            )
            local_write_lhs_scale = get_local_writes(local_load_lhs_scale)
            local_write_rhs_scale = get_local_writes(local_load_rhs_scale)
            global_load_lhs_scale = get_global_loads(local_write_lhs_scale)
            global_load_rhs_scale = get_global_loads(local_write_rhs_scale)
            # Early exit if cannot find any scale's local write or global loads.
            if any(
                not scale_memory_op
                for scale_memory_op in [
                    local_write_lhs_scale,
                    local_write_rhs_scale,
                    global_load_lhs_scale,
                    global_load_rhs_scale,
                ]
            ):
                continue

        # Heuristic to select reorder strategy.
        mTile, nTile, kTile = get_mma_tile_size(mma_nodes, constraints)
        mma_bitwidth = get_mma_bitwidth(mma_nodes[0])
        reorder_strategy = select_reorder_strategy(
            mma_type, mTile, nTile, kTile, mma_bitwidth, hardware_constraint
        )

        # Cannot find a suitable transform, early exit.
        if reorder_strategy == SchedReorderStrategy.NONE:
            continue
        elif reorder_strategy == SchedReorderStrategy.TWO_PP_CLUSTER:
            clusters = transform_two_PP_clusters(
                mma_nodes,
                local_load_lhs,
                local_load_rhs,
                global_load_lhs,
                global_load_rhs,
                local_write_lhs,
                local_write_rhs,
            )
            insert_prefetch_loop_barriers(custom_iterate, graph, clusters)
        elif reorder_strategy == SchedReorderStrategy.MXFP4_PP_CLUSTER:
            clusters = transform_MXFP4_PP_clusters(
                mma_nodes,
                local_load_lhs,
                local_load_rhs,
                global_load_lhs,
                global_load_rhs,
                local_write_lhs,
                local_write_rhs,
                local_load_lhs_scale,
                local_load_rhs_scale,
                global_load_lhs_scale,
                global_load_rhs_scale,
                local_write_lhs_scale,
                local_write_rhs_scale,
            )
            clusters = flatten_list(clusters)
        else:
            raise ValueError("Unhandled SchedReorderStrategy case.")
        reordered_graph = reorder_graph(graph, clusters)
        # Skip to next Iterate if fail to reorder graph.
        if reordered_graph is None:
            continue
        reordered_graph.parent_op = graph.parent_op
        reordered_subgraph_name = f"reoredered_{custom_iterate.subgraph_name}"
        trace.add_subgraph(reordered_subgraph_name, reordered_graph)
        trace.get_root_graph().subgraphs[reordered_subgraph_name] = reordered_graph
        custom_iterate.update_arg("subgraph_name", reordered_subgraph_name)
        if is_pingpong_strategy(reorder_strategy):
            add_conditional_barriers_to_loop(custom_iterate, trace, hardware_constraint)


def create_attention_clusters(
    mma0_nodes: List[fx.Node],
    local_load_k: List[fx.Node],
    local_write_k: List[fx.Node],
    global_load_k: List[fx.Node],
    mma1_nodes: List[fx.Node],
    local_load_v: List[fx.Node],
    local_write_v: List[fx.Node],
    global_load_v: List[fx.Node],
    softmax0_nodes: List[fx.Node],
    softmax1_nodes: List[fx.Node],
) -> List:
    """
    Create clusters for attention schedule reordering.

    Args:
        mma0_nodes: First set of MMA operations (QK computation)
        local_load_k: Local load operations for K
        local_write_k: Local write operations for K
        global_load_k: Global load operations for K
        mma1_nodes: Second set of MMA operations (PV computation)
        local_load_v: Local load operations for V
        local_write_v: Local write operations for V
        global_load_v: Global load operations for V
        softmax0_nodes: First set of softmax operations
        softmax1_nodes: Second set of softmax operations

    Returns:
        List of operation clusters for reordering
    """
    clusters = []
    tmp_graph = fx.Graph()

    # Cluster 0: QK computation and softmax1
    clusters.extend(_create_qk_softmax_cluster(mma0_nodes, softmax1_nodes, tmp_graph))

    # Cluster 1: K data movement (global load, local write, local load V)
    clusters.extend(
        _create_k_data_movement_cluster(
            global_load_k, local_write_k, local_load_v, tmp_graph
        )
    )

    # Cluster 2: PV computation and softmax0
    clusters.extend(_create_pv_softmax_cluster(mma1_nodes, softmax0_nodes, tmp_graph))

    # Cluster 3: V data movement (global load, local write, local load K)
    clusters.extend(
        _create_v_data_movement_cluster(
            global_load_v, local_write_v, local_load_k, tmp_graph
        )
    )

    return clusters


def _create_qk_softmax_cluster(
    mma0_nodes: List[fx.Node], softmax1_nodes: List[fx.Node], tmp_graph: fx.Graph
) -> List:
    """Create cluster for QK computation and softmax1."""
    clusters = []

    # Set wave priority for QK computation
    clusters.append(
        insert_op_before(SetWavePrio(1).add_to_graph(tmp_graph), mma0_nodes)
    )
    clusters.append(mma0_nodes)
    clusters.append(insert_op_after(SetWavePrio(0).add_to_graph(tmp_graph), mma0_nodes))

    # Add softmax1 operations
    clusters.append(softmax1_nodes)
    clusters.append(
        insert_op_after(WorkgroupBarrier().add_to_graph(tmp_graph), softmax1_nodes)
    )
    clusters.append(
        insert_op_after(SchedulingBarrier([]).add_to_graph(tmp_graph), clusters[-1].op)
    )

    return clusters


def _create_k_data_movement_cluster(
    global_load_k: List[fx.Node],
    local_write_k: List[fx.Node],
    local_load_v: List[fx.Node],
    tmp_graph: fx.Graph,
) -> List:
    """Create cluster for K data movement operations."""
    clusters = []

    # Global load K, local write K
    clusters.append(global_load_k)
    clusters.append(local_write_k)

    # Local load V and scheduling barrier
    clusters.append(local_load_v)
    clusters.append(
        insert_op_after(WorkgroupBarrier().add_to_graph(tmp_graph), local_load_v)
    )
    clusters.append(
        insert_op_after(SchedulingBarrier([]).add_to_graph(tmp_graph), clusters[-1].op)
    )

    return clusters


def _create_pv_softmax_cluster(
    mma1_nodes: List[fx.Node], softmax0_nodes: List[fx.Node], tmp_graph: fx.Graph
) -> List:
    """Create cluster for PV computation and softmax0."""
    clusters = []

    # Set wave priority for PV computation
    clusters.append(
        insert_op_before(SetWavePrio(1).add_to_graph(tmp_graph), mma1_nodes)
    )
    clusters.append(mma1_nodes)
    clusters.append(insert_op_after(SetWavePrio(0).add_to_graph(tmp_graph), mma1_nodes))

    # Add softmax0 operations
    clusters.append(softmax0_nodes)
    clusters.append(
        insert_op_after(WorkgroupBarrier().add_to_graph(tmp_graph), softmax0_nodes)
    )
    clusters.append(
        insert_op_after(SchedulingBarrier([]).add_to_graph(tmp_graph), clusters[-1].op)
    )

    return clusters


def _create_v_data_movement_cluster(
    global_load_v: List[fx.Node],
    local_write_v: List[fx.Node],
    local_load_k: List[fx.Node],
    tmp_graph: fx.Graph,
) -> List:
    """Create cluster for V data movement operations."""
    clusters = []

    # Global load V, local write V
    clusters.append(global_load_v)
    clusters.append(local_write_v)

    # Local load K and scheduling barrier
    clusters.append(local_load_k)
    clusters.append(
        insert_op_after(WorkgroupBarrier().add_to_graph(tmp_graph), local_load_k)
    )
    clusters.append(
        insert_op_after(SchedulingBarrier([]).add_to_graph(tmp_graph), clusters[-1].op)
    )

    return clusters


def _classify_attention_operations(
    graph: fx.Graph,
) -> Dict[AttentionOperationType, List[fx.Node]]:
    """
    Classify operations in the attention graph by their prefetch stage.

    Args:
        graph: The graph containing attention operations

    Returns:
        Dictionary mapping operation types to lists of nodes
    """
    operation_groups = {
        op_type: [] for op_type in AttentionOperationType.get_all_types()
    }

    for node in graph.nodes:
        prefetch_stage = node.meta.get("prefetch_stage", None)
        if prefetch_stage:
            try:
                op_type = AttentionOperationType.from_string(prefetch_stage)
                operation_groups[op_type].append(node)
            except:
                raise ValueError(f"Unknown prefetch stage: {prefetch_stage}")

    return operation_groups


def _validate_attention_operations(
    operation_groups: Dict[AttentionOperationType, List[fx.Node]],
) -> bool:
    """
    Validate that required attention operations are present.

    Args:
        operation_groups: Dictionary of operation groups

    Returns:
        True if validation passes, False otherwise
    """
    # All operation types are required
    for op_type in AttentionOperationType.get_all_types():
        if not operation_groups[op_type]:
            return False

    return True


def attention_schedule_reordering(
    trace: CapturedTrace, constraints: List[Constraint]
) -> None:
    """
    Perform attention schedule reordering for prefetch attention kernels.

    This function identifies attention operations in iterate nodes and reorders
    them to optimize memory access patterns and computation scheduling.

    Args:
        trace: Captured trace containing the computation graph
        constraints: List of constraints for the computation
    """
    hardware_constraint = get_hardware_constraint(constraints)
    flat_wave_count = math.prod(hardware_constraint.waves_per_block)
    # Only support 8 waves for now.
    if flat_wave_count != 8:
        return
    iterate_nodes = trace.walk(lambda node: isinstance(get_custom(node), Iterate))

    if not iterate_nodes:
        return

    for iterate_node in iterate_nodes:
        _process_attention_iterate_node(iterate_node, trace, hardware_constraint)


def _process_attention_iterate_node(
    iterate_node: fx.Node, trace: CapturedTrace, hardware_constraint: HardwareConstraint
) -> None:
    """
    Process a single iterate node for attention schedule reordering.

    Args:
        iterate_node: The iterate node to process
        trace: Captured trace containing the computation graph
        hardware_constraint: Hardware constraints for the computation
    """
    custom_iterate = get_custom(iterate_node)
    graph = trace.get_subgraph(custom_iterate.subgraph_name)

    # Classify operations by their prefetch stage
    operation_groups = _classify_attention_operations(graph)

    # Validate that all required operations are present
    if not _validate_attention_operations(operation_groups):
        return

    # Create operation clusters for reordering
    clusters = create_attention_clusters(
        mma0_nodes=operation_groups[AttentionOperationType.MMA_0],
        local_load_k=operation_groups[AttentionOperationType.LOCAL_LOAD_0],
        local_write_k=operation_groups[AttentionOperationType.LOCAL_STORE_0],
        global_load_k=operation_groups[AttentionOperationType.GLOBAL_LOAD_0],
        mma1_nodes=operation_groups[AttentionOperationType.MMA_1],
        local_load_v=operation_groups[AttentionOperationType.LOCAL_LOAD_1],
        local_write_v=operation_groups[AttentionOperationType.LOCAL_STORE_1],
        global_load_v=operation_groups[AttentionOperationType.GLOBAL_LOAD_1],
        softmax0_nodes=operation_groups[AttentionOperationType.SOFTMAX_0],
        softmax1_nodes=operation_groups[AttentionOperationType.SOFTMAX_1],
    )

    # Perform graph reordering
    reordered_graph = reorder_graph(graph, clusters)
    if reordered_graph is None:
        return

    # Update the trace with the reordered graph
    _update_trace_with_reordered_graph(trace, graph, reordered_graph, custom_iterate)

    # Add conditional barriers for ping-pong strategies
    add_conditional_barriers_to_loop(custom_iterate, trace, hardware_constraint)


def _update_trace_with_reordered_graph(
    trace: CapturedTrace,
    original_graph: fx.Graph,
    reordered_graph: fx.Graph,
    custom_iterate: Iterate,
) -> None:
    """
    Update the trace with the reordered graph.

    Args:
        trace: Captured trace containing the computation graph
        original_graph: Original graph before reordering
        reordered_graph: Reordered graph after optimization
        custom_iterate: The iterate operation being updated
    """
    # Set parent operation for the reordered graph
    reordered_graph.parent_op = original_graph.parent_op

    # Create new subgraph name and add to trace
    reordered_subgraph_name = f"reordered_{custom_iterate.subgraph_name}"
    trace.add_subgraph(reordered_subgraph_name, reordered_graph)
    trace.get_root_graph().subgraphs[reordered_subgraph_name] = reordered_graph

    # Update the iterate operation to use the new subgraph
    custom_iterate.update_arg("subgraph_name", reordered_subgraph_name)
