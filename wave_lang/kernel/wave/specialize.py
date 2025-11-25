# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
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

"""

import math
from typing import Optional

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
    NewScalar,
    get_custom,
)
from .compile_options import WaveCompileOptions
from .constraints import (
    Constraint,
)
from .utils.general_utils import (
    get_hardware_constraint,
    get_wave_constraints,
)
from .utils.classes import GemmOperationType
from .utils.graph_utils import update_sort_keys

from .scheduling.scheduler_utils import (
    GemmScheduler,
)
from .scheduling.resources import annotate_resource_usage

##############################################################
# General graph helper functions
##############################################################


def get_graph_node(
    custom: CustomOp, graph: fx.Graph, location: Optional[CapturedLocation]
) -> fx.Node:
    custom.add_to_graph(graph)
    custom.location = location
    custom = custom.fx_node
    return custom


def insert_service_cond(
    cond_reg, trace, iterate_op, location: Optional[CapturedLocation], nodes
):
    # service subgraph
    service_graph = fx.Graph()
    service_graph_name = f"service_graph_{cond_reg.name}"

    for node in nodes:
        op = get_custom(node)
        new_op = op.copy(new_graph=service_graph)
        op.replace_all_uses_with(new_op)
        op.erase()

    # add conditional at parent graph
    pgraph = trace.get_subgraph(iterate_op.subgraph_name)
    is_service_cond = Conditional(
        cond_reg,
        subgraph_name=service_graph_name,
        implicit_captures=iterate_op.implicit_captures,
    ).add_to_graph(pgraph)

    is_service_cond.location = location
    service_graph.parent_op = is_service_cond

    # update trace
    trace.add_subgraph(service_graph_name, service_graph)

    # update root graph
    get_custom(is_service_cond).get_root_graph().subgraphs[
        service_graph_name
    ] = service_graph

    return is_service_cond


def insert_compute_cond(
    cond_reg, trace, iterate_op, location: Optional[CapturedLocation], nodes
):
    # compute subgraph
    compute_graph = fx.Graph()
    compute_graph_name = f"compute_graph_{cond_reg.name}"

    for node in nodes:
        op = get_custom(node)
        new_op = op.copy(new_graph=compute_graph)
        op.replace_all_uses_with(new_op)
        op.erase()

    # add conditional at parent graph
    pgraph = trace.get_subgraph(iterate_op.subgraph_name)
    is_compute_cond = Conditional(
        cond_reg,
        subgraph_name=compute_graph_name,
        implicit_captures=[],
    ).add_to_graph(pgraph)

    is_compute_cond.location = location
    compute_graph.parent_op = is_compute_cond

    # update trace
    trace.add_subgraph(compute_graph_name, compute_graph)

    # update root graph
    get_custom(is_compute_cond).get_root_graph().subgraphs[
        compute_graph_name
    ] = compute_graph

    return is_compute_cond


def add_specialized_conditions(
    custom_iterate,
    trace,
    hardware_constraint,
    wave_constraints,
    load_partition,
    compute_partition,
):
    """ """
    graph = trace.get_subgraph(custom_iterate.subgraph_name)

    # calculate service wid
    physical_wid = (
        math.prod(hardware_constraint.waves_per_block)
        + hardware_constraint.n_service_waves
    )
    compute_wid = math.prod(map(lambda c: c.waves_per_block, wave_constraints))

    # calculate physical wid
    flat_id = hardware_constraint.linearized_thread_id
    wave_id = flat_id // hardware_constraint.threads_per_wave

    # Inserting wid is service_wid condition in graph
    # - get condition: is physcial wid > compute wid?
    # - get wid register
    # - get `is_load_wave` condition:
    #       is_load_wave = physical_wid >= compute_wid
    # - get `is_compute_wave` condition:
    #       is_compute_wave = physical_wid < compute_wid
    with graph.inserting_before(list(graph.nodes)[0]):
        compute_wid_reg = get_graph_node(
            NewScalar(compute_wid, tkl.i32), graph, custom_iterate.location
        )
        wave_id_reg = get_graph_node(
            NewScalar(wave_id, tkl.i32), graph, custom_iterate.location
        )
        is_load_wave = get_graph_node(
            Ge(wave_id_reg, compute_wid_reg), graph, custom_iterate.location
        )
        is_compute_wave = get_graph_node(
            Lt(wave_id_reg, compute_wid_reg), graph, custom_iterate.location
        )

    # get insertion point
    load_flattened, _ = pytree.tree_flatten(load_partition)
    load_flattened.reverse()
    compute_flattened, _ = pytree.tree_flatten(compute_partition)
    compute_flattened.reverse()

    # Generating and inserting cond_barriers to correct place in graph.
    insert_compute_cond(
        is_compute_wave,
        trace,
        custom_iterate,
        compute_flattened[0].location,
        compute_flattened,
    )
    insert_service_cond(
        is_load_wave, trace, custom_iterate, load_flattened[0].location, load_flattened
    )
    return


def get_ops_of_type(graph, operation_type):
    op_type_key = "specialize"
    return [
        node
        for node in graph.nodes
        if op_type_key in node.meta and node.meta[op_type_key] == operation_type
    ]


def specialize_kernel(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
):
    """
    n_service_waves: int, Number of service waves specified by user, this is the number of specialized waves only doing load workloads

    service wave index is calculated as:
    - number of service waves \times
    - number of threads per wave \times
    - number of `compute waves` in a block
    >>> n_service_waves * threads_per_wave * compute_wave_per_block (@M dimension)

    * assume graph is canonicalize such that all mma operations are after LW *
    * assume SchedulingType.NONE *

    0. walk the graph, get all shared memory write nodes -> this is the `load partition`
    1. add condition around the cluster -> only if wave_id is identified as service waves
    2. add condition around the rest of the graph -> else, these are compute waves' workload
    3. add `signal empty` and
    """
    if not options.specialize:
        return

    hardware_constraint = get_hardware_constraint(constraints)
    wave_constraints = get_wave_constraints(constraints)
    physical_wid = hardware_constraint.waves_per_block
    tpw = hardware_constraint.threads_per_wave

    # uniform shared memory write base address by aligning thread indexing position.
    # $T0 // wave size -> wave id
    wave_subs = {
        THREAD_0: ((THREAD_0 // tpw) * tpw if physical_wid[0] > 1 else 0),
        THREAD_1: THREAD_1 if physical_wid[1] > 1 else 0,
        THREAD_2: THREAD_2 if physical_wid[2] > 1 else 0,
    }

    iterate_nodes = trace.walk(lambda node: isinstance(get_custom(node), Iterate))
    if not iterate_nodes:
        return
    for iterate_node in iterate_nodes:
        custom_iterate = get_custom(iterate_node)

        graph = trace.get_subgraph(custom_iterate.subgraph_name)
        ignore_nodes, iter_args, output = annotate_resource_usage(graph)

        update_sort_keys(trace, graph)

        specialist = Specialist(
            graph=graph, edges=None, meta_name="specialize", resources=None
        )

        # Global load to lds will be handle by service_waves

        # 0) Get all local write and global read.
        load_partition = []
        load_partition.extend(
            get_ops_of_type(graph, GemmOperationType.GLOBAL_LOAD_TO_LDS_LHS)
        )
        load_partition.extend(
            get_ops_of_type(graph, GemmOperationType.GLOBAL_LOAD_TO_LDS_RHS)
        )

        load_partition.extend(get_ops_of_type(graph, GemmOperationType.LOCAL_WRITE_LHS))
        load_partition.extend(get_ops_of_type(graph, GemmOperationType.LOCAL_WRITE_RHS))

        load_partition.extend(get_ops_of_type(graph, GemmOperationType.GLOBAL_LOAD_LHS))
        load_partition.extend(get_ops_of_type(graph, GemmOperationType.GLOBAL_LOAD_RHS))

        # Local load and mma will be handle by compute waves
        compute_partition = []

        # 0) Get all local read and wmma
        # Get MMA nodes inside a for op, who's reduction dim is being tiled in the for op.
        mma_nodes = filter_fx_graph(
            graph,
            lambda node: isinstance(get_custom(node), MMA)
            and get_custom(node).reduction_dim == custom_iterate.axis,
        )

        compute_partition.extend(mma_nodes)
        compute_partition.extend(
            get_ops_of_type(graph, GemmOperationType.LOCAL_LOAD_LHS)
        )
        compute_partition.extend(
            get_ops_of_type(graph, GemmOperationType.LOCAL_LOAD_RHS)
        )

        # 1) Early exit if cannot find either operand's local write or global loads.
        if not load_partition or not compute_partition:
            continue

        # 2) Encapsulate the load with condition
        add_specialized_conditions(
            custom_iterate,
            trace,
            hardware_constraint,
            wave_constraints,
            load_partition,
            compute_partition,
        )

    print("done")
    return


class Specialist(GemmScheduler):
    """
    Specialist is responsible for converting basic gemm form like:
        for i = 0 to N:
            a = READ_GLOBAL i
            WRITE_SHARED a
            barrier
            b = READ_SHARED
            COMPUTE b

    into specialize form:
        for i = 0 to N:
            if wave_id == service_waves:
                a = READ_GLOBAL i
                wait empty
                WRITE_SHARED a
                signal ready
            else:
                wait ready
                b = READ_SHARED
                signal empty
                COMPUTE b
    """

    def partition(self):
        return
