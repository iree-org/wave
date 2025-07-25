from .resources import get_custom_operation_type
import torch.fx as fx
from enum import Enum
from .graph_utils import Edge
from ...ops.wave_ops import get_custom
import math

ScheduleStage = Enum  # alias


def get_scheduling_stage(
    op: fx.Node, operation_stage_table: dict[ScheduleStage:int]
) -> ScheduleStage:
    op_ty = get_custom_operation_type(get_custom(op))
    assert op_ty is not None, f"get_custom_operation_type returned None for {op}"
    if op_ty not in operation_stage_table:
        raise NotImplementedError(f"Cannot find {op_ty} in operation_stage_table")
    return operation_stage_table[op_ty]


class BaseScheduler:
    def __init__(
        self,
        graph: fx.Graph,
        edges: list[Edge],
        resources: list[int],
    ) -> None:
        # assert False
        self.graph = graph
        self.edges = edges
        self.resources = resources
        self.seed = 2024

    @property
    def initiation_interval(self) -> int:
        """
        Returns the initiation interval of the schedule.
        """
        return self._initiation_interval

    @property
    def num_stages(self) -> int:
        """
        Returns the number of stages in the kernel of the pipelined loop.
        """
        max_cycle = max([t + 1 for t in self.schedule.values()])
        return math.ceil(max_cycle / self.initiation_interval)
