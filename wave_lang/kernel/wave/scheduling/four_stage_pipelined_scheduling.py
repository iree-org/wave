import torch.fx as fx
from .graph_utils import Edge, sort_graph_by_edge_weight
from .resources import Operation
from enum import Enum, auto
from .scheduler_utils import (
    get_scheduling_stage,
    BaseScheduler,
    is_single_mma_source,
    is_mma_node,
)
import logging

logger = logging.getLogger(__name__)


class FourStageStage(Enum):
    GLOBAL_LOAD = auto()
    LOCAL_STORE = auto()
    LOCAL_LOAD = auto()
    COMPUTE = auto()
    SCHEDULING_NOOP = -1

    # Helper function to check next stage follows from current.
    @staticmethod
    def is_valid_transition(
        from_stage: "FourStageStage", to_stage: "FourStageStage"
    ) -> bool:
        if from_stage == to_stage:
            return True
        return (from_stage, to_stage) in _four_stage_stage_transition_table


_four_stage_stage_transition_table = {
    (FourStageStage.GLOBAL_LOAD, FourStageStage.LOCAL_STORE),
    (FourStageStage.LOCAL_STORE, FourStageStage.LOCAL_LOAD),
    # GLOBAL_TO_SHARED combines both GLOBAL_LOAD and LOCAL_STORE
    (FourStageStage.GLOBAL_LOAD, FourStageStage.LOCAL_LOAD),
    (FourStageStage.LOCAL_LOAD, FourStageStage.COMPUTE),
    (FourStageStage.COMPUTE, FourStageStage.GLOBAL_LOAD),
}
_operation_stage_table = {
    Operation.READ_SHARED: FourStageStage.LOCAL_LOAD,
    Operation.WRITE_SHARED: FourStageStage.LOCAL_STORE,
    Operation.READ_GLOBAL: FourStageStage.GLOBAL_LOAD,
    Operation.GLOBAL_TO_SHARED: FourStageStage.GLOBAL_LOAD,
    Operation.MMA: FourStageStage.COMPUTE,
    Operation.NOOP: FourStageStage.SCHEDULING_NOOP,
    Operation.VALU: FourStageStage.COMPUTE,
    Operation.SHUFFLE: FourStageStage.COMPUTE,
    Operation.WRITE_GLOBAL: FourStageStage.COMPUTE,
}


class FourStageScheduler(BaseScheduler):
    """
    Four Stage Pipelined Scheduler

    Precondition: Only a single MMA instruction group is allowed for this scheduling approach

    Convert vanilla schedule of:
        for i = 0 to N:
            a = READ_GLOBAL i
            WRITE_SHARED a
            barrier
            b = READ_SHARED
            COMPUTE b

    let SM be shared memory, then SM[0] SM[1] are the multibuffers
    into mega pipelined schedule:
        a_0 = READ_GLOBAL 0

        WRITE_SHARED a_0 SM[0]
        a_1 = READ_GLOBAL 1

        b_0 = READ_SHARED SM[0]
        WRITE_SHARED a_1 SM[1]
        a_2 = READ_GLOBAL 2


        for i = 0 to N -3:
            COMPUTE b_i
            b_{i+1} = READ_SHARED SM[i+1 %2]
            WRITE_SHARED a_{i+2} SM[i%2]
            a_{i+3} = READ_GLOBAL i+3
            barrier


        COMPUTE b_{n-2}
        b_{n-1} = READ_SHARED SM[n-1 %2]
        WRITE_SHARED a_{n} SM[n % 2]

        COMPUTE b_{n-1}
        b_{n} = READ_SHARED SM[n %2]

        COMPUTE b_n

    """

    def four_stage_scheduling(
        self, graph: fx.Graph, edges: list[Edge]
    ) -> tuple[dict[fx.Node, int], bool]:
        """
        Classify node to different stages. Based on its stage,
        program schedules the node to a specific cycle for  each node.
        This function also checks that the sorted nodes move contiguously through
        expected stages.
        """
        sorted_nodes = sort_graph_by_edge_weight(graph.nodes, edges)
        schedule = {}
        current_stage = get_scheduling_stage(sorted_nodes[0], _operation_stage_table)

        all_mma_nodes = list()
        current_stage_idx = 0
        for node in sorted_nodes:
            if is_mma_node(node):
                all_mma_nodes.append(node)

            node_stage = get_scheduling_stage(node, _operation_stage_table)
            if node_stage in [current_stage, FourStageStage.SCHEDULING_NOOP]:
                schedule[node] = current_stage_idx
            elif FourStageStage.is_valid_transition(current_stage, node_stage):
                current_stage_idx += 1
                schedule[node] = current_stage_idx
                current_stage = node_stage
            else:
                logger.warning(
                    f"No valid transition from {current_stage} to {node_stage} for node {node}"
                )
                # Node does not move contigously through stages.
                return {}, False
        if not is_single_mma_source(all_mma_nodes):
            logger.warning(
                "Structure of kernel is different than expected, only one MMA is present"
            )
            return {}, False

        return schedule, True

    def schedule_graph(self) -> tuple[dict[fx.Node, int], bool]:
        """
        1. Identify which nodes are part of the global_read/local_write/local_read/compute phase
        2. Set nodes to clock (0,1,2,3) based on phase.
        3. Set initiation interval to 1.
        """
        self.schedule, success = self.four_stage_scheduling(self.graph, self.edges)
        self._initiation_interval = 1
        return self.schedule, success
