from .resources import get_custom_operation_type
import torch.fx as fx
from enum import Enum
from .graph_utils import Edge
from ...ops.wave_ops import get_custom
import math
from typing import TypeVar
from enum import Enum
from .resources import Operation
from typing import List

ScheduleStage = TypeVar("ScheduleStage", bound=Enum)


def get_scheduling_stage(
    op: fx.Node, operation_stage_table: dict[Operation, ScheduleStage]
) -> ScheduleStage:
    op_ty = get_custom_operation_type(get_custom(op))
    assert op_ty is not None, f"get_custom_operation_type returned None for {op}"
    if op_ty not in operation_stage_table:
        raise NotImplementedError(f"Cannot find {op_ty} in operation_stage_table")
    return operation_stage_table[op_ty]


def is_single_mma_source(all_mma_nodes: List[fx.Node]):
    pre_expansion_ids = set(
        [get_custom(node).pre_expansion_id for node in all_mma_nodes]
    )
    return len(pre_expansion_ids) == 1


def is_mma_node(possible_mma_node: fx.Node):
    op_ty = get_custom_operation_type(get_custom(possible_mma_node))
    return op_ty == Operation.MMA


class BaseScheduler:
    def __init__(
        self,
        graph: fx.Graph,
        edges: list[Edge],
        resources: list[int],
    ) -> None:
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
        max_cycle = max(t + 1 for t in self.schedule.values())
        return math.ceil(max_cycle / self.initiation_interval)
