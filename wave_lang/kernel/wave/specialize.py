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
from typing import Optional, List
from collections import defaultdict

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
    Output,
    Ge,
    GetResult,
    Iterate,
    Lt,
    NewScalar,
    Read,
    SharedMemoryBarrierSignal,
    SharedMemoryBarrierWait,
    Write,
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
from .utils.symbol_utils import get_induction_symbol

from .scheduling.scheduler_utils import (
    GemmScheduler,
)
from .scheduling.graph_utils import Edge

from .minimize_global_loads import update_write_dependencies

##############################################################
# General graph helper functions
##############################################################


def add_output_to_cond(
    op: CustomOp, return_vals: List, subgraph, parent_graph, parent_loc
):
    assert isinstance(
        op, Conditional
    ), "Expect to add output node only on condition ops"
    if any(isinstance(get_custom(node), Output) for node in subgraph.nodes):
        return

    # add arguments to condition yield
    output_op = Output(return_vals).add_to_graph(subgraph, loc=op.location)
    output_node = get_custom(output_op)

    # replace uses of compute result by conditional returns
    with parent_graph.inserting_after(op.fx_node):
        for i, return_val in enumerate(return_vals):
            gr_node = GetResult(op.fx_node, i).add_to_graph(
                parent_graph, loc=parent_loc
            )
            gr_node.index = return_val.index
            get_custom(return_val).replace_all_uses_with_except(gr_node, [output_node])


def set_specialied_conditions(
    graph,
    hardware_constraint,
    wave_constraints,
) -> (fx.Node, fx.Node):
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
    anchor = next(iter(graph.nodes))
    with graph.inserting_before(anchor):
        compute_wid_reg = NewScalar(compute_wid, tkl.i32).add_to_graph(
            graph, loc=anchor.location
        )
        wave_id_reg = NewScalar(wave_id, tkl.i32).add_to_graph(
            graph, loc=anchor.location
        )
        is_load_wave = Ge(wave_id_reg, compute_wid_reg).add_to_graph(
            graph, loc=anchor.location
        )
        is_compute_wave = Lt(wave_id_reg, compute_wid_reg).add_to_graph(
            graph, loc=anchor.location
        )

    return (is_load_wave, is_compute_wave, wave_id)


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
    if hardware_constraint.n_service_waves == 0:
        return

    wave_constraints = get_wave_constraints(constraints)

    is_load_wave, is_compute_wave, wave_id = set_specialied_conditions(
        trace.get_root_graph(), hardware_constraint, wave_constraints
    )

    iterate_nodes = trace.walk(lambda node: isinstance(get_custom(node), Iterate))
    if not iterate_nodes:
        return

    for iterate_node in iterate_nodes:
        iterate_op = get_custom(iterate_node)
        caller_args = list(iterate_op.implicit_captures)
        caller_args.extend([is_load_wave, is_compute_wave])
        iterate_op.implicit_captures = caller_args

        graph = trace.get_subgraph(iterate_op.subgraph_name)

        update_sort_keys(trace, graph)

        specialist = Specialist(
            trace=trace,
            graph=graph,
            edges=None,
            resources=None,
            wave_id=wave_id,
            hw=hardware_constraint,
        )

        load_partition, compute_partition = specialist.partition(iterate_op)

        # 1) Early exit if cannot find either operand's local write or global loads.
        if not load_partition or not compute_partition:
            continue

        # 2) Encapsulate the load with condition
        specialist.transform(
            iterate_op,
            is_load_wave,
            load_partition,
            is_compute_wave,
            compute_partition,
        )

    # TODO(megan.kuo) assume one iterate per graph
    specialist.add_epilog_guard(iterate_op, is_compute_wave)
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

    def __init__(
        self,
        trace: CapturedTrace,
        graph: fx.Graph,
        edges: list[Edge],
        resources: list[int],
        meta_name: str = "specialize",
        wave_id=None,
        hw=None,
    ) -> None:
        super().__init__(graph, edges, resources, meta_name)
        assert wave_id is not None, "Wave ID should be provided to specilist"

        self.trace = trace
        self.graph = graph
        self.wave_id = wave_id
        self.hw = hw
        self.barUB = math.prod(self.hw.waves_per_block) + 1

    def add_nodes_to_graph(self, graph, nodes, subs, new_writes, write_record):
        def new_index(index, shift_subs):
            return {k: v.subs(shift_subs) for k, v in index.items()}

        node_map = dict()
        for node in nodes:
            custom = get_custom(node)
            new_node = custom.copy(
                new_graph=graph,
                arg_transform=lambda x: node_map[x] if x in node_map else x,
            )
            new_node.index = new_index(node.index, subs)
            node_map[node] = new_node.fx_node

            if isinstance(custom, Write):
                new_writes[custom.memory].append(new_node.fx_node)
                write_record.append(new_node.fx_node)

    def get_ops_of_type(self, operation_type):
        return [
            node
            for node in self.graph.nodes
            if self.meta_name in node.meta
            and node.meta[self.meta_name] == operation_type
        ]

    def partition(self, iterate_op):
        # Global load to lds will be handle by service_waves

        # 0) Get all local write and global read.
        load_partition = []
        load_partition.extend(
            self.get_ops_of_type(GemmOperationType.GLOBAL_LOAD_TO_LDS_LHS)
        )
        load_partition.extend(
            self.get_ops_of_type(GemmOperationType.GLOBAL_LOAD_TO_LDS_RHS)
        )

        load_partition.extend(self.get_ops_of_type(GemmOperationType.LOCAL_WRITE_LHS))
        load_partition.extend(self.get_ops_of_type(GemmOperationType.LOCAL_WRITE_RHS))

        load_partition.extend(self.get_ops_of_type(GemmOperationType.GLOBAL_LOAD_LHS))
        load_partition.extend(self.get_ops_of_type(GemmOperationType.GLOBAL_LOAD_RHS))

        # Local load and mma will be handle by compute waves
        compute_partition = []

        # 0) Get all local read and wmma
        # Get MMA nodes inside a for op, who's reduction dim is being tiled in the for op.
        mma_nodes = filter_fx_graph(
            self.graph,
            lambda node: isinstance(get_custom(node), MMA)
            and get_custom(node).reduction_dim == iterate_op.axis,
        )

        compute_partition.extend(mma_nodes)
        compute_partition.extend(self.get_ops_of_type(GemmOperationType.LOCAL_LOAD_LHS))
        compute_partition.extend(self.get_ops_of_type(GemmOperationType.LOCAL_LOAD_RHS))

        return load_partition, compute_partition

    def transform(
        self,
        iterate_op,
        is_load_wave,
        load_partition,
        is_compute_wave,
        compute_partition,
    ):
        """ """
        # get insertion point
        load_flattened, _ = pytree.tree_flatten(load_partition)
        load_flattened.reverse()
        compute_flattened, _ = pytree.tree_flatten(compute_partition)
        compute_flattened.reverse()

        # Generating and inserting cond_barriers to correct place in graph.
        new_writes = self.insert_service_cond(
            is_load_wave, iterate_op, load_flattened[0].location, load_flattened
        )

        self.insert_compute_cond(
            is_compute_wave,
            iterate_op,
            compute_flattened[-1].location,
            compute_flattened,
        )

        update_write_dependencies(new_writes, self.trace)
        return

    def insert_compute_cond(
        self, cond_reg, iterate_op, location: Optional[CapturedLocation], nodes
    ):
        # declare new subgraph
        compute_graph = fx.Graph()
        compute_graph_name = f"compute_graph_{cond_reg.name}"

        # add nodes to compute graph
        last_shared_read = None
        for node in nodes:
            op = get_custom(node)
            new_op = op.copy(new_graph=compute_graph)
            if isinstance(op, Read):
                last_shared_read = new_op.fx_node
            op.replace_all_uses_with(new_op)
            op.erase()

        # get parent graph
        pgraph = self.graph

        # add conditional nodes to parent graph
        with pgraph.inserting_before(pgraph.output_node()):
            is_compute_cond = Conditional(
                cond_reg,
                subgraph_name=compute_graph_name,
                implicit_captures=[],
                else_return=iterate_op.init_args,
            ).add_to_graph(pgraph, loc=location)

        compute_graph.parent_op = is_compute_cond

        # add return node in compute graph and get result in parent graph
        add_output_to_cond(
            get_custom(is_compute_cond),
            iterate_op.outputs(),
            compute_graph,
            pgraph,
            location,
        )

        # add wait before compute
        self.add_compute_split_barrier(compute_graph, last_shared_read)

        # update trace
        self.trace.add_subgraph(compute_graph_name, compute_graph)

        # update root graph
        get_custom(is_compute_cond).get_root_graph().subgraphs[
            compute_graph_name
        ] = compute_graph

    def insert_service_cond(
        self, cond_reg, iterate_op, location: Optional[CapturedLocation], nodes
    ):
        # service subgraph
        service_graph = fx.Graph()
        service_graph_name = f"service_graph_{cond_reg.name}"

        # add conditional at parent graph
        pgraph = self.graph

        with pgraph.inserting_before(pgraph.output_node()):
            is_service_cond = Conditional(
                cond_reg,
                subgraph_name=service_graph_name,
                implicit_captures=iterate_op.implicit_captures,
            ).add_to_graph(pgraph)

        is_service_cond.location = location
        service_graph.parent_op = is_service_cond

        # duplicate nodes
        new_writes = defaultdict(list)
        for w in range(self.hw.waves_per_block[0], 0, -1):
            write_record = []

            # TODO(megan.kuo)
            shift_subs = {THREAD_1: THREAD_1 - w}

            self.add_nodes_to_graph(
                service_graph, nodes, shift_subs, new_writes, write_record
            )

            # add signal after load
            dup_times = self.hw.waves_per_block[0] - w
            self.add_load_split_barrier(
                service_graph, iterate_op, dup_times, write_record[0], write_record[-1]
            )

        # close the graph with empty output
        service_graph.output(None)

        # update trace
        self.trace.add_subgraph(service_graph_name, service_graph)

        # update root graph
        get_custom(is_service_cond).get_root_graph().subgraphs[
            service_graph_name
        ] = service_graph

        return new_writes

    def add_compute_split_barrier(self, subgraph, last_shared_read):
        """
        add split barriers for compute partition
        ---- Example ----
        Before:
            if is compute partition:
                LOCAL_READ
                LOCAL_READ
                MMA
        After:
            if is compute partition:
                if wave id == 0:
                    wait 0
                if wave id == 1:
                    wait 1
                ...

                LOCAL_READ
                LOCAL_READ
                MMA

                if wave id == 0:
                    signal 0
                if wave id == 1:
                    signal 1
                ...
        """
        first_node = next(iter(subgraph.nodes))
        location = get_custom(first_node).location

        with subgraph.inserting_before(first_node):
            for i in range(1, self.barUB):
                # declare graph
                wid_wait_graph = fx.Graph()
                wid_wait_graph_name = f"compute_wait_{i}_graph"

                # update trace
                self.trace.add_subgraph(wid_wait_graph_name, wid_wait_graph)

                # add wait node to wid_wait_graph
                SharedMemoryBarrierWait(barId=i).add_to_graph(
                    region_graph=wid_wait_graph, loc=location
                )

                # add condition entry to parent graph (wave id is 0-based)
                cond_expr = sympy.Eq(self.wave_id, i - 1)
                wait_cond_op = Conditional(
                    cond_expr,
                    subgraph_name=wid_wait_graph_name,
                    implicit_captures=[],
                ).add_to_graph(subgraph, loc=location)

                wid_wait_graph.parent_op = wait_cond_op

                # update root graph
                get_custom(wait_cond_op).get_root_graph().subgraphs[
                    wid_wait_graph_name
                ] = wid_wait_graph

        signal_location = get_custom(last_shared_read).location
        with subgraph.inserting_after(last_shared_read):
            for i in range(1, self.barUB):
                # declare graph
                wid_signal_graph = fx.Graph()
                wid_signal_graph_name = f"compute_signal_{i}_graph"

                # update trace
                self.trace.add_subgraph(wid_signal_graph_name, wid_signal_graph)

                # add signal node to wid_signal_graph
                SharedMemoryBarrierSignal(barId=i).add_to_graph(
                    region_graph=wid_signal_graph, loc=signal_location
                )

                # add condition entry to parent graph
                cond_expr = sympy.Eq(self.wave_id, i - 1)
                signal_cond_op = Conditional(
                    cond_expr,
                    subgraph_name=wid_signal_graph_name,
                    implicit_captures=[],
                ).add_to_graph(subgraph, loc=signal_location)

                wid_signal_graph.parent_op = signal_cond_op

                # update root graph
                get_custom(signal_cond_op).get_root_graph().subgraphs[
                    wid_signal_graph_name
                ] = wid_signal_graph

    def add_load_split_barrier(
        self, subgraph, iterate_op, dup_times, first_lw, last_lw
    ):
        """
        add split barriers for load partition
        ---- Example ----
        Before:
            if is load partition:
                GLOBAL_READ
                GLOBAL_READ
                LOCAL_WRITE     <- first_lw
                LOCAL_WRITE     <- last_lw
        After:
            if is load partition:
                GLOBAL_READ
                GLOBAL_READ

                if not first round and wave id == 0:
                    wait 0

                LOCAL_WRITE
                LOCAL_WRITE

                if wave id == 0:
                    signal 0

                GLOBAL_READ
                GLOBAL_READ

                if not first round and wave id == 1:
                    wait 1

                LOCAL_WRITE
                LOCAL_WRITE

                if wave id == 1:
                    signal 1
                ...
        """

        # first load wave id
        start_load_wid = math.prod(self.hw.waves_per_block)

        # induction symbol
        iv = get_induction_symbol(iterate_op.axis)

        with subgraph.inserting_before(first_lw):
            location = get_custom(first_lw).location
            for i in range(1, self.barUB):
                # declare graph
                wid_wait_graph = fx.Graph()
                wid_wait_graph_name = f"load_wait_{i}_graph"

                # update trace
                self.trace.add_subgraph(wid_wait_graph_name, wid_wait_graph)

                # add wait node to wid_wait_graph
                SharedMemoryBarrierWait(barId=i).add_to_graph(
                    region_graph=wid_wait_graph, loc=location
                )

                # calculate which compute wid this load wid is helping with
                compute_wid = (
                    self.wave_id % start_load_wid
                ) + dup_times * self.hw.waves_per_block[0]

                # add condition entry to parent graph
                cond_expr = sympy.And(sympy.Eq(compute_wid, i - 1), sympy.Ne(iv, 0))

                wait_cond_op = Conditional(
                    cond_expr,
                    subgraph_name=wid_wait_graph_name,
                    implicit_captures=[],
                ).add_to_graph(subgraph, loc=location)

                wid_wait_graph.parent_op = wait_cond_op

                # update root graph
                get_custom(wait_cond_op).get_root_graph().subgraphs[
                    wid_wait_graph_name
                ] = wid_wait_graph

        with subgraph.inserting_after(last_lw):
            location = get_custom(last_lw).location
            for i in range(1, self.barUB):
                # declare graph
                wid_signal_graph = fx.Graph()
                wid_signal_graph_name = f"load_signal_{i}_graph"

                # update trace
                self.trace.add_subgraph(wid_signal_graph_name, wid_signal_graph)

                # add signal node to wid_signal_graph
                SharedMemoryBarrierSignal(barId=i).add_to_graph(
                    region_graph=wid_signal_graph, loc=location
                )

                # calculate which compute wid this load wid is helping with
                compute_wid = (
                    self.wave_id % start_load_wid
                ) + dup_times * self.hw.waves_per_block[0]

                # add condition entry to parent graph
                cond_expr = sympy.Eq(compute_wid, i - 1)

                signal_cond_op = Conditional(
                    cond_expr,
                    subgraph_name=wid_signal_graph_name,
                    implicit_captures=[],
                ).add_to_graph(subgraph, loc=location)

                wid_signal_graph.parent_op = signal_cond_op

                # update root graph
                get_custom(signal_cond_op).get_root_graph().subgraphs[
                    wid_signal_graph_name
                ] = wid_signal_graph

    def add_epilog_guard(self, iterate_op, cond_reg):
        # declare new subgraph
        store_graph = fx.Graph()
        store_graph_name = f"store_graph_{cond_reg.name}"

        # add nodes to store graph
        post_iterate_node = iterate_op.graph.output_node().prev
        with store_graph.inserting_before():
            while post_iterate_node != iterate_op.fx_node:
                op = get_custom(post_iterate_node)
                new_op = op.copy(new_graph=store_graph)
                op.replace_all_uses_with(new_op)
                op.erase()
                post_iterate_node = post_iterate_node.prev

        # get parent graph
        pgraph = iterate_op.graph

        # add conditional nodes to parent graph
        with pgraph.inserting_before(pgraph.output_node()):
            is_compute_cond = Conditional(
                cond_reg,
                subgraph_name=store_graph_name,
                implicit_captures=[],
            ).add_to_graph(pgraph, loc=get_custom(pgraph.output_node()).location)

        store_graph.parent_op = is_compute_cond

        # update trace
        self.trace.add_subgraph(store_graph_name, store_graph)

        # update root graph
        iterate_op.get_root_graph().subgraphs[store_graph_name] = store_graph
