# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from typing import Sequence

import torch.fx as fx

from .graph_utils import Edge, sort_graph_by_edge_weight
from .resources import Operation
from .scheduler_utils import get_scheduling_stage, BaseScheduler
from ...ops.wave_ops import (
    get_custom,
    Read,
    Write,
    MMA,
    ScaledMMA,
    IterArg,
    Reshape,
    Extract,
)
from ..utils.graph_utils import capture_backward_slice

import logging

logger = logging.getLogger(__name__)


class PrefetchStage(Enum):
    GLOBAL_LOAD = 0
    LOCAL_STORE = 1
    LOCAL_LOAD = 2
    COMPUTE = 3

    @staticmethod
    def is_valid_transition(
        from_stage: "PrefetchStage", to_stage: "PrefetchStage"
    ) -> bool:
        if from_stage == to_stage:
            return True

        return (from_stage, to_stage) in _prefetch_stage_transition_table


_prefetch_stage_transition_table = {
    (PrefetchStage.GLOBAL_LOAD, PrefetchStage.LOCAL_STORE),
    (PrefetchStage.LOCAL_STORE, PrefetchStage.LOCAL_LOAD),
    # GLOBAL_TO_SHARED combines both GLOBAL_LOAD and LOCAL_STORE
    (PrefetchStage.GLOBAL_LOAD, PrefetchStage.LOCAL_LOAD),
    (PrefetchStage.LOCAL_LOAD, PrefetchStage.COMPUTE),
    (PrefetchStage.COMPUTE, PrefetchStage.GLOBAL_LOAD),
}

_operation_stage_table = {
    Operation.READ_SHARED: PrefetchStage.LOCAL_LOAD,
    Operation.WRITE_SHARED: PrefetchStage.LOCAL_STORE,
    Operation.READ_GLOBAL: PrefetchStage.GLOBAL_LOAD,
    Operation.GLOBAL_TO_SHARED: PrefetchStage.GLOBAL_LOAD,
    Operation.MMA: PrefetchStage.COMPUTE,
    Operation.NOOP: PrefetchStage.COMPUTE,
    Operation.VALU: PrefetchStage.COMPUTE,
    Operation.SHUFFLE: PrefetchStage.COMPUTE,
}


class PrefetchScheduler(BaseScheduler):
    """
    Prefetch Scheduler

    Convert vanilla schedule of:
        for i = 0 to N:
            a = READ_GLOBAL i
            WRITE_SHARED a
            barrier
            b = READ_SHARED
            COMPUTE b

    into prefetch schedule:
        a_0 = READ_GLOBAL 0
        WRITE_SHARED a_0
        for i = 0 to N - 1:
            a_{i+1} = READ_GLOBAL i + 1
            // a_{i+1} is NOT blocked by this barrier because barriers only block shared memory transfers
            barrier
            b_i = READ_SHARED
            COMPUTE b_i
            barrier
            WRITE_SHARED a_{i+1}
        barrier
        b_N = READ_SHARED
        COMPUTE b_N
    """

    def prefetch_scheduling(self, graph: fx.Graph, edges: list[Edge]):
        """
        Classify node to different stages. Based on it's stage,
        program schedules clock for each node. This function also checks
        that sorted node "contiguously" move between stages.
        """
        sorted_nodes = sort_graph_by_edge_weight(graph.nodes, edges)
        schedule = {}
        current_stage = get_scheduling_stage(sorted_nodes[0], _operation_stage_table)
        for node in sorted_nodes:
            node_stage = get_scheduling_stage(node, _operation_stage_table)
            logger.info(f"Node {node} is in stage {node_stage}")
            if node_stage == current_stage:
                schedule[node] = node_stage.value
            elif PrefetchStage.is_valid_transition(current_stage, node_stage):
                schedule[node] = node_stage.value
                current_stage = node_stage
            else:
                # Node do not move contigously through stages.
                logger.warning(
                    f"No valid transition from {current_stage} to {node_stage} for node {node}"
                )
                return {}, False
        return schedule, True

    def schedule_graph(self) -> tuple[dict[fx.Node, int], bool]:
        """
        1. Identify which nodes are part of the global_read/local_write/local_read/compute phase
        2. Set nodes to clock (0,1,2,3) based on phase.
        2. Set initiation interval to generate valid 2 stage prefetch.
        """
        self.schedule, success = self.prefetch_scheduling(self.graph, self.edges)
        if not success:
            logger.warning("Prefetch scheduling failed")
            return {}, False

        logger.info(f"Schedule: {self.schedule}")
        assert self.schedule, "Schedule is empty"
        self._initiation_interval = 2
        if self.num_stages != self._initiation_interval:
            logger.warning(
                f"Initiation interval {self._initiation_interval} does not match number of stages {self.num_stages}"
            )
            return {}, False
        return self.schedule, success


class MMAGroup:
    def __init__(self, mma_ops: list[fx.Node]):
        self.global_reads = set()
        self.shared_reads = set()
        self.shared_writes = set()
        self.mma_ops = set(mma_ops)

    def add_nodes(self, nodes: list[fx.Node]):
        for node in nodes:
            custom = get_custom(node)
            if isinstance(custom, Read):
                if custom.memory is None:
                    self.global_reads.add(node)
                else:
                    self.shared_reads.add(node)
            elif isinstance(custom, Write):
                self.shared_writes.add(node)

    def __repr__(self):
        return f"MMAGroup(\nglobal_reads={self.global_reads},\nshared_reads={self.shared_reads},\nshared_writes={self.shared_writes},\nmma_ops={self.mma_ops})"


def _set_cycle(nodes: list[fx.Node], cycle: int, schedule: dict[fx.Node, int]):
    for node in nodes:
        schedule[node] = cycle
    return schedule


class PrefetchAttentionScheduler(BaseScheduler):

    def schedule_graph(self) -> tuple[dict[fx.Node, int], bool]:
        """
        Implements attention-specific prefetch scheduling with the following cycle assignments:
        - Cycle 0: Global loads and shared writes for GEMM0
        - Cycle 1: Shared reads for GEMM0, shared writes for GEMM1, global loads for GEMM1
        - Cycle 2: GEMM0 operations and softmax0 operations
        - Cycle 3: GEMM1 operations, softmax1 operations, and shared reads for GEMM1 operands
        """
        schedule = {}

        # Identify all MMAs.
        mmas = []
        for node in self.graph.nodes:
            custom_op = get_custom(node)
            if isinstance(custom_op, (MMA, ScaledMMA)):
                mmas.append(node)

        # Partition MMAs into GEMM0 and GEMM1.
        # One of the GEMM0 operands should be hoisted out of the loop.
        mma0, mma1 = self._partition_mmas(mmas)

        # Find and partition softmax operations.
        softmax_ops = self._identify_softmax_operations(mma0)
        softmax_ops.update(
            self._identify_softmax_operations(
                [get_custom(x).acc for x in mma1], backward=True
            )
        )
        softmax_ops = sorted(softmax_ops, key=lambda x: x._sort_key)
        # TODO: There is sensitivity to the split index. For certain choices of split index,
        # the scheduler will incorrectly schedule the graph. Investigate this further.
        split_index = softmax_ops.index(
            [x for x in softmax_ops if get_custom(x).tkw_op_name == "sub"][-1]
        )
        softmax0 = softmax_ops[:split_index]
        softmax1 = softmax_ops[split_index:]

        # Safety check: ensure we have operations to schedule
        if not mmas and not softmax_ops:
            logger.warning("No MMAs or softmax operations found in graph")
            self.schedule = {}
            self._initiation_interval = 1
            return {}, False

        # Analyze dependencies for both MMA groups
        mma0_group = self._analyze_mma_dependencies(mma0)
        mma1_group = self._analyze_mma_dependencies(
            mma1,
            exclude_shared_reads=mma0_group.shared_reads,
        )

        # Cycle 0, 1: Global loads and shared writes for GEMM0
        schedule = _set_cycle(mma0_group.global_reads, 0, schedule)
        schedule = _set_cycle(mma0_group.shared_writes, 1, schedule)

        # Cycle 2, 3: Shared reads for GEMM0, global loads for GEMM1, shared writes for GEMM1
        schedule = _set_cycle(mma0_group.shared_reads, 2, schedule)
        schedule = _set_cycle(mma1_group.global_reads, 2, schedule)
        schedule = _set_cycle(mma1_group.shared_writes, 3, schedule)

        # Cycle 2: QK and softmax0 operations
        schedule = _set_cycle(mma0_group.mma_ops, 4, schedule)
        schedule = _set_cycle(softmax0, 5, schedule)

        # Cycle 3: PV, shared reads for GEMM1 operands, and softmax1 operations
        schedule = _set_cycle(softmax1, 6, schedule)
        schedule = _set_cycle(mma1_group.shared_reads, 6, schedule)
        schedule = _set_cycle(mma1_group.mma_ops, 7, schedule)

        # Schedule remaining nodes by inheriting cycles from their arguments
        schedule = self._schedule_remaining_nodes(schedule)

        # Set initiation interval and number of stages
        self.schedule = schedule
        self._initiation_interval = 2

        logger.info(f"PrefetchAttentionScheduler: Scheduled {len(schedule)} nodes")
        logger.info(f"GEMM0: {len(mma0)} operations, GEMM1: {len(mma1)} operations")
        logger.info(
            f"Softmax0: {len(softmax0)} operations, Softmax1: {len(softmax1)} operations"
        )
        logger.info(f"Initiation interval: {self._initiation_interval}")

        return schedule, True

    def _partition_mmas(
        self, mmas: list[fx.Node]
    ) -> tuple[list[fx.Node], list[fx.Node]]:
        """
        Partitions MMAs into two groups: GEMM0 and GEMM1.
        GEMM0 is identified by checking if the lhs and rhs of any MMA are in different graphs.
        The other MMA is GEMM1.
        """
        mma0 = []
        mma1 = []
        for mma_node in mmas:
            mma = get_custom(mma_node)
            lhs = get_custom(mma.lhs)
            rhs = get_custom(mma.rhs)
            if lhs.graph != rhs.graph:
                mma0.append(mma_node)
            else:
                mma1.append(mma_node)
        return mma0, mma1

    def _identify_softmax_operations(
        self, mma0_nodes: list[fx.Node], backward: bool = False
    ) -> list[fx.Node]:
        """
        Identifies softmax operations that are guaranteed to not be reads, writes, MMAs, or iter args.
        These operations are guaranteed to be after the GEMM0 operations and before the GEMM1 operations.
        """
        from torch.utils import _pytree as pytree

        softmax_ops = set()
        end_node_types = (Read, Write, MMA, ScaledMMA, IterArg)
        start_nodes = set(mma0_nodes)
        visited = set()
        while start_nodes:
            node = start_nodes.pop()
            if node in visited:
                continue
            visited.add(node)
            neighbors = node.users.keys()
            if backward:
                neighbors, _ = pytree.tree_flatten(
                    list(get_custom(node).node_args.values())
                )
                neighbors = [x.fx_node for x in neighbors]
            for user in neighbors:
                if user not in visited and not isinstance(
                    get_custom(user), end_node_types
                ):
                    softmax_ops.add(user)
                    start_nodes.add(user)
        return softmax_ops

    def _schedule_remaining_nodes(
        self, schedule: dict[fx.Node, int]
    ) -> dict[fx.Node, int]:
        """
        Schedules remaining nodes by having them inherit cycles from their arguments.
        Uses an iterative approach to handle dependencies that may not be resolved in the first pass.
        """
        # The max iters controls the maximum depth of the dependency chain that we will traverse.
        max_iters = 4
        iter_count = 0

        while iter_count < max_iters:
            for node in self.graph.nodes:
                if node in schedule:
                    continue

                custom = get_custom(node)
                for arg in custom.node_args.values():
                    if isinstance(arg, Sequence):
                        for elem in arg:
                            if elem.fx_node in schedule:
                                schedule[node] = schedule[elem.fx_node]
                                break
                    else:
                        if arg.fx_node in schedule:
                            schedule[node] = schedule[arg.fx_node]
                            break

            iter_count += 1

        # Verify all nodes were scheduled
        not_scheduled = [node for node in self.graph.nodes if node not in schedule]
        assert not not_scheduled, "Not all nodes were scheduled"

        return schedule

    def _analyze_mma_dependencies(
        self,
        mma_ops: list[fx.Node],
        exclude_shared_reads: set[fx.Node] = set(),
    ) -> MMAGroup:
        """
        Analyzes dependencies for a single MMA group to determine global reads and shared reads.
        """
        mma_group = MMAGroup(mma_ops)

        for mma_node in mma_ops:
            mma = get_custom(mma_node)
            lhs = mma.lhs
            rhs = mma.rhs

            # Add global reads for the MMA's operands if they are not shared reads
            if lhs not in exclude_shared_reads and lhs.graph == mma.graph:
                backward_slice = capture_backward_slice(
                    lhs,
                    lambda x: isinstance(
                        get_custom(x), (Read, Write, Reshape, Extract)
                    ),
                )
                mma_group.add_nodes(backward_slice)

            if rhs not in exclude_shared_reads and rhs.graph == mma.graph:
                backward_slice = capture_backward_slice(
                    rhs,
                    lambda x: isinstance(
                        get_custom(x), (Read, Write, Reshape, Extract)
                    ),
                )
                mma_group.add_nodes(backward_slice)

        return mma_group
