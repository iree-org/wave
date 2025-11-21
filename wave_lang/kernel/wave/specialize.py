# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

'''
Specialization in wave.

note. tensor store is not supported yet.

Analysis:
- identify 2~3 partitions
    - load partition
    - compute partition
    [- store partition]

Workers:
- 1 wave for load
- 1 wave for store
- other waves for compute

Shared resources:
- i in {1, 16} staging buffer (global -> LDS)
- 2^i memory barriers
    e.g., i=1
    - signal_empty:
        - compute partition signals load parition [data is consumed]
    - signal_ready:
        - load partition signals compute partition [data is ready]
    - wait_empty:
        - load partition waiting on compute to signal empty.
    - wait_ready:
        - compute partition waiting on load to signal ready.

Behavior:
1. Analysis -> get list of nodes for load_partition and compute_partition
2. Workers -> Wrap load_partition code in wave for load, compute_partition code for rest of the waves
3. Adjust index
4.

'''

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Dict, List, Optional

import torch.fx as fx
from torch.utils import _pytree as pytree

import wave_lang.kernel.lang as tkl

from .._support.tracing import CapturedTrace
from .._support.location import CapturedLocation
from ..compiler.kernel_codegen import filter_fx_graph
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    MMA,
    Conditional,
    CustomOp,
    Ge,
    Iterate,
    Lt,
    MemoryCounterWait,
    NewScalar,
    ScaledMMA,
    SchedulingBarrier,
    SetWavePrio,
    SharedMemoryBarrier,
    WorkgroupBarrier,
    get_custom,
)
from .compile_options import WaveCompileOptions
from .constraints import (
    Constraint,
    WaveConstraint,
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
from .utils.classes import AttentionOperationType, GemmOperationType
from .utils.print_utils import print_trace, print_graph

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
    ASYNC_TWO_PP_CLUSTER = 0x2201
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
    use_global_to_shared: bool
    mma_type: type


twoPPConfig = CompatibleBlockSize(128, 128, 64, 16, False, MMA)
asyncTwoPPConfig = CompatibleBlockSize(128, 128, 64, 16, True, MMA)
MXFP4PPConfig = CompatibleBlockSize(256, 128, 256, 4, False, ScaledMMA)


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


def get_graph_node(
    custom: CustomOp, graph: fx.Graph, location: Optional[CapturedLocation]
) -> fx.Node:
    custom.add_to_graph(graph)
    custom.location = location
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


def _slice_node_list(node_list, reduction_expand_size, num_slices, reduction_dim):
    """
    Slice a list of nodes based on reduction dimension into equal-sized slices.

    This function computes slice boundaries based on the sorted order of nodes
    (to determine how many nodes exist per reduction dimension), but returns
    slices from the original node_list order. This assumes node_list is already
    sorted by reduction dimension.

    Args:
        node_list: List of nodes to slice (assumed to be sorted by reduction_dim)
        reduction_expand_size: Number of unique reduction dimension IDs
        num_slices: Number of slices to create
        reduction_dim: The reduction dimension index

    Returns:
        List of lists, where each inner list contains nodes for one slice
    """
    # Sort to compute nodes_per_k_dim, but slice the original list
    # (which assumes it's already sorted by reduction_dim)
    sorted_nodes = sorted(node_list, key=lambda x: x.expanded_dims[reduction_dim])
    nodes_per_k_dim = len(sorted_nodes) // reduction_expand_size
    slice_size = reduction_expand_size // num_slices

    sliced_nodes = []
    for slice_id in range(num_slices):
        lower_bound_k_id = slice_id * slice_size
        upper_bound_k_id = (slice_id + 1) * slice_size

        start_idx = lower_bound_k_id * nodes_per_k_dim
        end_idx = upper_bound_k_id * nodes_per_k_dim

        sliced_nodes.append(sorted_nodes[start_idx:end_idx])

    return sliced_nodes


def slice_mma(mma_nodes, lhs_nodes, rhs_nodes, num_slice):
    """
    Slice MMA operations and their operand load nodes into equal-sized slices
    based on the reduction dimension.

    This function groups MMA nodes and their corresponding LHS/RHS load nodes
    into slices, where each slice contains operations for a contiguous range
    of reduction dimension IDs.

    Args:
        mma_nodes: List of MMA operation nodes
        lhs_nodes: List of left-hand side load nodes
        rhs_nodes: List of right-hand side load nodes
        num_slice: Number of slices to create (must evenly divide reduction_expand_size)

    Returns:
        Tuple of (sliced_mma_nodes, sliced_lhs_nodes, sliced_rhs_nodes),
        where each element is a list of lists, one per slice

    Raises:
        AssertionError: If slicing preconditions are not met
    """
    reduction_dim = get_custom(mma_nodes[0]).reduction_dim
    reduction_dim_ids = set(
        [get_custom(node).expanded_dims[reduction_dim] for node in mma_nodes]
    )
    # Checking that MMAs is valid.
    reduction_expand_size = len(reduction_dim_ids)
    assert reduction_expand_size >= num_slice and reduction_expand_size % num_slice == 0
    assert all(x in reduction_dim_ids for x in range(reduction_expand_size))

    # Slice each node list independently
    sliced_mma_nodes = _slice_node_list(
        mma_nodes, reduction_expand_size, num_slice, reduction_dim
    )
    sliced_lhs_nodes = _slice_node_list(
        lhs_nodes, reduction_expand_size, num_slice, reduction_dim
    )
    sliced_rhs_nodes = _slice_node_list(
        rhs_nodes, reduction_expand_size, num_slice, reduction_dim
    )

    return sliced_mma_nodes, sliced_lhs_nodes, sliced_rhs_nodes


def slice_scale_mma(
    mma_nodes, lhs_nodes, rhs_nodes, lhs_scale_nodes, rhs_scale_nodes, num_slice
):
    """
    Slice scaled MMA operations and their operand/scale load nodes into equal-sized slices
    based on the reduction dimension.

    This function groups scaled MMA nodes and their corresponding LHS/RHS load nodes
    (including scale nodes) into slices, where each slice contains operations for a
    contiguous range of reduction dimension IDs.

    Args:
        mma_nodes: List of MMA operation nodes
        lhs_nodes: List of left-hand side load nodes
        rhs_nodes: List of right-hand side load nodes
        lhs_scale_nodes: List of left-hand side scale load nodes
        rhs_scale_nodes: List of right-hand side scale load nodes
        num_slice: Number of slices to create (must evenly divide reduction_expand_size)

    Returns:
        Tuple of (sliced_mma_nodes, sliced_lhs_nodes, sliced_rhs_nodes,
                  sliced_lhs_scale_nodes, sliced_rhs_scale_nodes),
        where each element is a list of lists, one per slice

    Raises:
        AssertionError: If slicing preconditions are not met
    """
    reduction_dim = get_custom(mma_nodes[0]).reduction_dim
    reduction_dim_ids = set(
        [get_custom(node).expanded_dims[reduction_dim] for node in mma_nodes]
    )
    # Checking that MMAs is valid.
    reduction_expand_size = len(reduction_dim_ids)
    assert reduction_expand_size >= num_slice and reduction_expand_size % num_slice == 0
    assert all(x in reduction_dim_ids for x in range(reduction_expand_size))

    # Slice each node list independently
    sliced_mma_nodes = _slice_node_list(
        mma_nodes, reduction_expand_size, num_slice, reduction_dim
    )
    sliced_lhs_nodes = _slice_node_list(
        lhs_nodes, reduction_expand_size, num_slice, reduction_dim
    )
    sliced_rhs_nodes = _slice_node_list(
        rhs_nodes, reduction_expand_size, num_slice, reduction_dim
    )
    sliced_lhs_scale_nodes = _slice_node_list(
        lhs_scale_nodes, reduction_expand_size, num_slice, reduction_dim
    )
    sliced_rhs_scale_nodes = _slice_node_list(
        rhs_scale_nodes, reduction_expand_size, num_slice, reduction_dim
    )

    return (
        sliced_mma_nodes,
        sliced_lhs_nodes,
        sliced_rhs_nodes,
        sliced_lhs_scale_nodes,
        sliced_rhs_scale_nodes,
    )


def insert_cond_barrier(cond_reg, trace, graph, location: Optional[CapturedLocation]):
    barrier_graph = fx.Graph()
    barrier_graph_name = f"barrier_graph_{cond_reg.name}"
    barrier_node = WorkgroupBarrier().add_to_graph(barrier_graph)
    barrier_node.location = location
    cond_barrier = Conditional(
        cond_reg,
        subgraph_name=barrier_graph_name,
        implicit_captures=[],
    ).add_to_graph(graph)
    cond_barrier.location = location
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
        mid_wave_reg = get_graph_node(
            NewScalar(mid_wave, tkl.i32), graph, custom_iterate.location
        )
        wave_id_reg = get_graph_node(
            NewScalar(wave_id, tkl.i32), graph, custom_iterate.location
        )
        is_wave_hi = get_graph_node(
            Ge(wave_id_reg, mid_wave_reg), graph, custom_iterate.location
        )
        is_wave_lo = get_graph_node(
            Lt(wave_id_reg, mid_wave_reg), graph, custom_iterate.location
        )

    # Generating and inserting cond_barriers to correct place in graph.
    with graph.inserting_before(custom_iterate.fx_node):
        insert_cond_barrier(is_wave_hi, trace, graph, custom_iterate.location)
    with graph.inserting_after(custom_iterate.fx_node):
        insert_cond_barrier(is_wave_lo, trace, graph, custom_iterate.location)
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
        barrier = SharedMemoryBarrier().add_to_graph(graph, loc=custom_iterate.location)
    # TODO: Replace this with just lgmkcnt or change lowering of
    # SharedMemoryBarrier to only do lgmkcnt w/os sbarrier
    with cluster_graph.inserting_before(cluster_graph.output_node()):
        barrier = SharedMemoryBarrier().add_to_graph(
            cluster_graph, loc=custom_iterate.location
        )
        clusters.append(barrier)
    return


##############################################################
# Ping Pong Transformation.
##############################################################


def is_compatible_strategy(
    mTile, nTile, kTile, mma_bitwidth, use_global_to_shared, mma_type, strategy
):
    return (
        mTile % strategy.block_m == 0
        and nTile % strategy.block_n == 0
        and kTile % strategy.block_k == 0
        and mma_bitwidth == strategy.bitwidth
        and use_global_to_shared == strategy.use_global_to_shared
        and mma_type == strategy.mma_type
    )


def select_reorder_strategy(
    mma_type,
    mTile,
    nTile,
    kTile,
    mma_bitwidth,
    use_global_to_shared,
    hardware_constraint,
):
    flat_wave_count = math.prod(hardware_constraint.waves_per_block)
    if flat_wave_count != 8:
        return SchedReorderStrategy.NONE
    if is_compatible_strategy(
        mTile, nTile, kTile, mma_bitwidth, use_global_to_shared, mma_type, twoPPConfig
    ):
        return SchedReorderStrategy.TWO_PP_CLUSTER
    elif is_compatible_strategy(
        mTile,
        nTile,
        kTile,
        mma_bitwidth,
        use_global_to_shared,
        mma_type,
        asyncTwoPPConfig,
    ):
        return SchedReorderStrategy.ASYNC_TWO_PP_CLUSTER
    elif is_compatible_strategy(
        mTile, nTile, kTile, mma_bitwidth, use_global_to_shared, mma_type, MXFP4PPConfig
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

    context_location = mma_nodes and mma_nodes[0].location

    clusters = []
    tmp_graph = fx.Graph()
    # 1st cluster interleaved local and global reads.
    clusters.append(sliced_local_load_lhs[0])
    clusters.append(sliced_local_load_rhs[0])
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, sliced_local_load_rhs[0]))

    clusters.append(global_load_lhs)
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, global_load_lhs))

    clusters.append(sliced_local_load_lhs[1])
    clusters.append(sliced_local_load_rhs[1])
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, sliced_local_load_rhs[1]))

    clusters.append(global_load_rhs)
    barrier_op = WorkgroupBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, global_load_rhs))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    # 2nd cluster mma_slice[0].
    prio_op = SetWavePrio(1).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_before(prio_op, sliced_mma_nodes[0]))
    clusters.append(sliced_mma_nodes[0])
    prio_op = SetWavePrio(0).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_after(prio_op, sliced_mma_nodes[0]))
    barrier_op = SharedMemoryBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    # 3rd cluster local writes.
    clusters.append(local_write_lhs)
    clusters.append(local_write_rhs)
    barrier_op = WorkgroupBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, local_write_rhs))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    # 4th cluster mma_slice[1].
    prio_op = SetWavePrio(1).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_before(prio_op, sliced_mma_nodes[1]))
    clusters.append(sliced_mma_nodes[1])
    prio_op = SetWavePrio(0).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_after(prio_op, sliced_mma_nodes[1]))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    return clusters


def transform_async_two_PP_clusters(
    mma_nodes,
    local_load_lhs,
    local_load_rhs,
    global_to_shared_lhs,
    global_to_shared_rhs,
):
    num_slices = 2
    sliced_mma_nodes, sliced_local_load_lhs, sliced_local_load_rhs = slice_mma(
        mma_nodes, local_load_lhs, local_load_rhs, num_slice=num_slices
    )
    # Check that we have valid slice size for local_loads and mmas.
    assert len(sliced_mma_nodes) == len(sliced_local_load_rhs)
    assert len(sliced_mma_nodes) == len(sliced_local_load_lhs)
    assert len(sliced_mma_nodes) == num_slices

    context_location = mma_nodes and mma_nodes[0].location

    clusters = []
    tmp_graph = fx.Graph()
    # 1st cluster interleaved local and global reads.
    clusters.append(sliced_local_load_lhs[0])
    clusters.append(sliced_local_load_rhs[0])
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, sliced_local_load_rhs[0]))

    clusters.append(global_to_shared_lhs)
    clusters.append(global_to_shared_rhs)
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, global_to_shared_rhs))

    barrier_op = WorkgroupBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    # 2nd cluster mma_slice[0].
    prio_op = SetWavePrio(1).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_before(prio_op, sliced_mma_nodes[0]))
    clusters.append(sliced_mma_nodes[0])
    prio_op = SetWavePrio(0).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_after(prio_op, sliced_mma_nodes[0]))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    independent_global_count = len(global_to_shared_lhs + global_to_shared_rhs)
    barrier_op = MemoryCounterWait(load=independent_global_count).add_to_graph(
        tmp_graph
    )
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    barrier_op = WorkgroupBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    # 3rd cluster local load 2nd slice.
    clusters.append(sliced_local_load_lhs[1])
    clusters.append(sliced_local_load_rhs[1])
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, sliced_local_load_rhs[1]))

    barrier_op = MemoryCounterWait(load=0).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    barrier_op = WorkgroupBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    # 4th cluster mma_slice[1].
    prio_op = SetWavePrio(1).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_before(prio_op, sliced_mma_nodes[1]))
    clusters.append(sliced_mma_nodes[1])
    prio_op = SetWavePrio(0).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_after(prio_op, sliced_mma_nodes[1]))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

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


def get_ops_of_type(graph, operation_type):
    op_type_key = "prefetch_stage"
    return [
        node
        for node in graph.nodes
        if op_type_key in node.meta and node.meta[op_type_key] == operation_type
    ]


def schedule_reordering(
    trace: CapturedTrace,
    constraints: list[Constraint],
    scheduling_type: SchedulingType,
    use_global_to_shared: bool,
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
        local_load_lhs = get_ops_of_type(graph, GemmOperationType.LOCAL_LOAD_LHS)
        local_load_rhs = get_ops_of_type(graph, GemmOperationType.LOCAL_LOAD_RHS)
        # Early exit if cannot find either local loads
        if not local_load_lhs or not local_load_rhs:
            continue
        global_to_shared_lhs = get_ops_of_type(
            graph, GemmOperationType.GLOBAL_LOAD_TO_LDS_LHS
        )
        global_to_shared_rhs = get_ops_of_type(
            graph, GemmOperationType.GLOBAL_LOAD_TO_LDS_RHS
        )
        local_write_lhs = get_ops_of_type(graph, GemmOperationType.LOCAL_WRITE_LHS)
        local_write_rhs = get_ops_of_type(graph, GemmOperationType.LOCAL_WRITE_RHS)
        global_load_lhs = get_ops_of_type(graph, GemmOperationType.GLOBAL_LOAD_LHS)
        global_load_rhs = get_ops_of_type(graph, GemmOperationType.GLOBAL_LOAD_RHS)
        # Early exit if cannot find either operand's local write or global loads.
        if not use_global_to_shared and any(
            not memory_op
            for memory_op in [
                local_write_lhs,
                local_write_rhs,
                global_load_lhs,
                global_load_rhs,
            ]
        ):
            continue

        # Early exit if cannot find either operand's local write or global loads.
        if use_global_to_shared and any(
            not memory_op
            for memory_op in [
                global_to_shared_lhs,
                global_to_shared_rhs,
            ]
        ):
            continue

        local_load_lhs_scale = None
        local_load_rhs_scale = None
        local_write_lhs_scale = None
        local_write_lhs_scale = None
        global_load_lhs_scale = None
        global_load_rhs_scale = None
        global_to_shared_lhs_scale = None
        global_to_shared_rhs_scale = None
        if mma_type == ScaledMMA:
            local_load_lhs_scale = get_ops_of_type(
                graph, GemmOperationType.LOCAL_LOAD_LHS_SCALE
            )
            local_load_rhs_scale = get_ops_of_type(
                graph, GemmOperationType.LOCAL_LOAD_RHS_SCALE
            )
            global_to_shared_lhs_scale = get_ops_of_type(
                graph, GemmOperationType.GLOBAL_LOAD_TO_LDS_LHS_SCALE
            )
            global_to_shared_rhs_scale = get_ops_of_type(
                graph, GemmOperationType.GLOBAL_LOAD_TO_LDS_RHS_SCALE
            )
            local_write_lhs_scale = get_ops_of_type(
                graph, GemmOperationType.LOCAL_WRITE_LHS_SCALE
            )
            local_write_rhs_scale = get_ops_of_type(
                graph, GemmOperationType.LOCAL_WRITE_RHS_SCALE
            )
            global_load_lhs_scale = get_ops_of_type(
                graph, GemmOperationType.GLOBAL_LOAD_LHS_SCALE
            )
            global_load_rhs_scale = get_ops_of_type(
                graph, GemmOperationType.GLOBAL_LOAD_RHS_SCALE
            )
            # Early exit if cannot find any scale's local write or global loads.
            if not use_global_to_shared and any(
                not scale_memory_op
                for scale_memory_op in [
                    local_write_lhs_scale,
                    local_write_rhs_scale,
                    global_load_lhs_scale,
                    global_load_rhs_scale,
                ]
            ):
                continue
            if use_global_to_shared and any(
                not memory_op
                for memory_op in [
                    global_to_shared_lhs_scale,
                    global_to_shared_rhs_scale,
                ]
            ):
                continue

        # Heuristic to select reorder strategy.
        mTile, nTile, kTile = get_mma_tile_size(mma_nodes, constraints)
        mma_bitwidth = get_mma_bitwidth(mma_nodes[0])
        reorder_strategy = select_reorder_strategy(
            mma_type,
            mTile,
            nTile,
            kTile,
            mma_bitwidth,
            use_global_to_shared,
            hardware_constraint,
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
        elif reorder_strategy == SchedReorderStrategy.ASYNC_TWO_PP_CLUSTER:
            clusters = transform_async_two_PP_clusters(
                mma_nodes,
                local_load_lhs,
                local_load_rhs,
                global_to_shared_lhs,
                global_to_shared_rhs,
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
        original_subgraph_name = custom_iterate.subgraph_name
        reordered_subgraph_name = f"reoredered_{original_subgraph_name}"
        trace.add_subgraph(reordered_subgraph_name, reordered_graph)
        trace.get_root_graph().subgraphs[reordered_subgraph_name] = reordered_graph
        custom_iterate.update_arg("subgraph_name", reordered_subgraph_name)

        del trace.region_graph.subgraphs[original_subgraph_name]
        del trace.get_root_graph().subgraphs[original_subgraph_name]

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

    context_location = (mma0_nodes and get_custom(mma0_nodes[0]).location) or (
        softmax1_nodes and get_custom(softmax1_nodes[0]).location
    )

    # Set wave priority for QK computation
    prio_op = SetWavePrio(1).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_before(prio_op, mma0_nodes))
    clusters.append(mma0_nodes)
    prio_op = SetWavePrio(0).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_after(prio_op, mma0_nodes))

    # Add softmax1 operations
    clusters.append(softmax1_nodes)
    barrier_op = WorkgroupBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, softmax1_nodes))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    return clusters


def _create_k_data_movement_cluster(
    global_load_k: List[fx.Node],
    local_write_k: List[fx.Node],
    local_load_v: List[fx.Node],
    tmp_graph: fx.Graph,
) -> List:
    """Create cluster for K data movement operations."""
    clusters = []

    context_location = (
        (global_load_k and get_custom(global_load_k[0]).location)
        or (local_write_k and get_custom(local_write_k[0]).location)
        or (local_load_v and get_custom(local_load_v[0]).location)
    )

    # Global load K, local write K
    clusters.append(global_load_k)
    clusters.append(local_write_k)

    # Local load V and scheduling barrier
    clusters.append(local_load_v)
    barrier_op = WorkgroupBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, local_load_v))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    return clusters


def _create_pv_softmax_cluster(
    mma1_nodes: List[fx.Node], softmax0_nodes: List[fx.Node], tmp_graph: fx.Graph
) -> List:
    """Create cluster for PV computation and softmax0."""
    clusters = []

    context_location = (mma1_nodes and get_custom(mma1_nodes[0]).location) or (
        softmax0_nodes and get_custom(softmax0_nodes[0]).location
    )

    # Set wave priority for PV computation
    prio_op = SetWavePrio(1).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_before(prio_op, mma1_nodes))
    clusters.append(mma1_nodes)
    prio_op = SetWavePrio(0).add_to_graph(tmp_graph)
    prio_op.location = context_location
    clusters.append(insert_op_after(prio_op, mma1_nodes))

    # Add softmax0 operations
    clusters.append(softmax0_nodes)
    barrier_op = WorkgroupBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, softmax0_nodes))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

    return clusters


def _create_v_data_movement_cluster(
    global_load_v: List[fx.Node],
    local_write_v: List[fx.Node],
    local_load_k: List[fx.Node],
    tmp_graph: fx.Graph,
) -> List:
    """Create cluster for V data movement operations."""
    clusters = []

    context_location = (
        (global_load_v and get_custom(global_load_v[0]).location)
        or (local_write_v and get_custom(local_write_v[0]).location)
        or (local_load_k and get_custom(local_load_k[0]).location)
    )

    # Global load V, local write V
    clusters.append(global_load_v)
    clusters.append(local_write_v)

    # Local load K and scheduling barrier
    clusters.append(local_load_k)
    barrier_op = WorkgroupBarrier().add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, local_load_k))
    barrier_op = SchedulingBarrier([]).add_to_graph(tmp_graph)
    barrier_op.location = context_location
    clusters.append(insert_op_after(barrier_op, clusters[-1].op))

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

def specialize_kernel(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
):
    """
    0. Check requirement:
        1) option.use_global_to_shared is set
        2) target is gfx1250
    1. Build 1-many mapping of GLOBAL_READ: SHARED_WRITE_X ... #a
    2. Build descriptors for tensor.load.to.lds with proper IndexSequence offsets.
    3. Replace #a with tensor_load_to_shared op.
    4. Update write dependencies.
    """
    return

    if not options.specialize:
        return

    if "gfx12" not in options.target:
        return

    id_to_read_write = defaultdict(list)
    for read in trace.walk(is_valid_read):
        read = get_custom(read)
        for write in read.users:
            if not is_valid_write(write):
                continue

            key = (read.pre_expansion_id, write.pre_expansion_id)
            id_to_read_write[key].append((read, write))

    if not id_to_read_write:
        return

    hardware_constraint = get_hardware_constraint(constraints)
    threads_per_wave = hardware_constraint.threads_per_wave
    waves_per_block = hardware_constraint.waves_per_block

    # uniform shared memory write base address by aligning thread indexing position.
    # $T0 // wave size -> wave id
    wave_subs = {
        THREAD_0: (
            (THREAD_0 // threads_per_wave) * threads_per_wave
            if waves_per_block[0] > 1
            else 0
        ),
        THREAD_1: THREAD_1 if waves_per_block[1] > 1 else 0,
        THREAD_2: THREAD_2 if waves_per_block[2] > 1 else 0,
    }

    return
