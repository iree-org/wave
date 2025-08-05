# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum

import torch.fx as fx

from .graph_utils import Edge, sort_graph_by_edge_weight
from .resources import Operation
from .scheduler_utils import get_scheduling_stage, BaseScheduler

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
