import torch.fx as fx
from .graph_utils import Edge, sort_graph_by_edge_weight
from .resources import Operation
from enum import Enum
from .scheduler_utils import get_scheduling_stage, BaseScheduler


class GemmFourStageStage(Enum):
    GLOBAL_LOAD = 0
    LOCAL_STORE = 1
    LOCAL_LOAD = 2
    COMPUTE = 3
    SCHEDULING_NOOP = -1

    # Helper function to get next stage from the current.
    # If at stage 3 returns itself to prevent crash
    # since it is final stage.
    def next(self):

        if self.value == 3:
            return GemmFourStageStage(3)
        v = self.value + 1
        return GemmFourStageStage(v)


operation_stage_table = {
    Operation.READ_SHARED: GemmFourStageStage.LOCAL_LOAD,
    Operation.WRITE_SHARED: GemmFourStageStage.LOCAL_STORE,
    Operation.READ_GLOBAL: GemmFourStageStage.GLOBAL_LOAD,
    Operation.MMA: GemmFourStageStage.COMPUTE,
    Operation.NOOP: GemmFourStageStage.SCHEDULING_NOOP,
    Operation.VALU: GemmFourStageStage.COMPUTE,
    Operation.SHUFFLE: GemmFourStageStage.COMPUTE,
    Operation.WRITE_GLOBAL: GemmFourStageStage.COMPUTE,
}


class GemmFourStageScheduler(BaseScheduler):
    """
    GEMM Four Stage Pipelined Scheduler

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

    def mega_pipelined_scheduling(self, graph: fx.Graph, edges: list[Edge]):
        """
        Classify node to different stages. Based on it's stage,
        program schedules clock for each node. This function also checks
        that sorted node "contiguously" move between stages.
        """
        sorted_nodes = sort_graph_by_edge_weight(graph.nodes, edges)
        schedule = {}
        current_stage = GemmFourStageStage.GLOBAL_LOAD

        for node in sorted_nodes:
            node_stage = get_scheduling_stage(node, operation_stage_table)
            next_stage = current_stage.next()
            if node_stage in [current_stage, GemmFourStageStage.SCHEDULING_NOOP]:
                schedule[node] = current_stage.value
            elif node_stage == next_stage:
                schedule[node] = next_stage.value
                current_stage = next_stage
            else:
                # Node do not move contigously through stages.
                return {}, False
        return schedule, True

    def schedule_graph(self) -> tuple[dict[fx.Node, int], bool]:
        """
        1. Identify which nodes are part of the global_read/local_write/local_read/compute phase
        2. Set nodes to clock (0,1,2,3) based on phase.
        3. Set initiation interval to 1.
        """
        self.schedule, success = self.mega_pipelined_scheduling(self.graph, self.edges)
        self._initiation_interval = 1
        return self.schedule, success
