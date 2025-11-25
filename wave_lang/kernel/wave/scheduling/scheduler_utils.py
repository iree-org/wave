from .resources import get_custom_operation_type
import torch.fx as fx
from enum import Enum
from .graph_utils import Edge
from ..utils.general_utils import is_shared_read
from ...ops.wave_ops import (
    TensorLoadToLDS,
    get_custom,
    CustomOp,
    Read,
    Write,
    GatherToLDS,
    TensorLoadToLDS,
    MMA,
    ScaledMMA,
)
import math
from typing import Iterable, TypeVar
from enum import Enum
from .resources import Operation
from typing import List
from collections import deque
from ..utils.classes import GemmOperationType
from ...compiler.kernel_codegen import filter_fx_graph

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


class GemmScheduler(BaseScheduler):
    def __init__(
        self,
        graph: fx.Graph,
        edges: list[Edge],
        resources: list[int],
        meta_name: str = "prefetch_stage",
    ) -> None:
        super().__init__(graph, edges, resources)
        self.meta_name = meta_name
        self.annotate_gemm_operation_type(self.graph)

    def get_closest_local_load(self, node: fx.Node):
        workqueue = deque([node])
        seen = set()
        can_extend = lambda x: isinstance(x, fx.Node) and x not in seen
        while workqueue:
            cur_node = workqueue.pop()
            if is_shared_read(get_custom(cur_node)):
                return cur_node
            child_nodes = [
                arg_node for arg_node in cur_node.args if can_extend(arg_node)
            ]

            # Update ancestor and seen tracker, and workqueue with new child nodes.
            seen.update(child_nodes)
            workqueue.extend(child_nodes)
        return None

    def get_local_loads(self, mma_nodes):
        local_load_lhs = []
        local_load_rhs = []
        for mma_node in mma_nodes:
            custom = get_custom(mma_node)
            lhs = self.get_closest_local_load(custom.lhs)
            rhs = self.get_closest_local_load(custom.rhs)
            if lhs == None or rhs == None:
                return None, None
            local_load_lhs.append(lhs)
            local_load_rhs.append(rhs)
        return local_load_lhs, local_load_rhs

    def get_scale_local_loads(self, mma_nodes):
        local_load_lhs_scale = []
        local_load_rhs_scale = []
        for mma_node in mma_nodes:
            custom = get_custom(mma_node)
            lhs_scale = self.get_closest_local_load(custom.lhs_scale)
            rhs_scale = self.get_closest_local_load(custom.rhs_scale)
            if lhs_scale == None or rhs_scale == None:
                return None, None
            local_load_lhs_scale.append(lhs_scale)
            local_load_rhs_scale.append(rhs_scale)
        return local_load_lhs_scale, local_load_rhs_scale

    def get_local_writes(self, local_loads: Iterable[Read]) -> Iterable[Write]:
        local_writes = dict()
        for local_load in local_loads:
            custom = get_custom(local_load)
            cur_writes = dict.fromkeys(
                w
                for w in custom.memory.users
                if isinstance(get_custom(w), Write) and w.graph == custom.graph
            )
            local_writes.update(cur_writes)
        return local_writes.keys()

    def get_global_loads_to_lds(
        self, local_loads: Iterable[Read]
    ) -> Iterable[GatherToLDS | TensorLoadToLDS]:
        lds_gathers = dict()
        for local_load in local_loads:
            custom = get_custom(local_load)
            # Get direct users and users from rotated registers.
            memory_users = dict.fromkeys(g for g in custom.memory.users)
            # Filter users for GatherToLDS or TensorLoadToLDS
            cur_gathers = dict.fromkeys(
                g
                for g in memory_users
                if isinstance(get_custom(g), (GatherToLDS, TensorLoadToLDS))
                and g.graph == custom.graph
            )
            lds_gathers.update(cur_gathers)
            cur_gathers = dict.fromkeys(
                g
                for g in memory_users
                if isinstance(get_custom(g), TensorLoadToLDS)
                and g.graph == custom.graph
            )
            lds_gathers.update(cur_gathers)
        return lds_gathers.keys()

    def get_global_loads(self, local_writes: Iterable[Write]) -> Iterable[Read]:
        global_loads = dict()
        for local_write in local_writes:
            custom = get_custom(local_write)
            if isinstance(get_custom(custom.register_), Read):
                global_loads[custom.register_] = None
        return global_loads.keys()

    def annotate_op_with_gemm_operation_type(self, nodes, gemm_operation_type):
        for node in nodes:
            if isinstance(node, CustomOp):
                node = node.fx_node
            node.meta[self.meta_name] = gemm_operation_type

    def annotate_gemm_operation_type(self, graph):
        mma_nodes = filter_fx_graph(
            graph,
            lambda node: isinstance(get_custom(node), (MMA, ScaledMMA)),
        )
        # Early exit if no MMA found.
        if not mma_nodes:
            return

        mma_types = set([type(get_custom(mma_node)) for mma_node in mma_nodes])
        # Only handle single MMA per kernel and need to have same type.s
        if len(mma_types) != 1:
            return

        mma_type = mma_types.pop()
        local_load_lhs, local_load_rhs = self.get_local_loads(mma_nodes)
        # Early exit if cannot find either local loads
        if not local_load_lhs or not local_load_rhs:
            return
        global_to_shared_lhs = self.get_global_loads_to_lds(local_load_lhs)
        global_to_shared_rhs = self.get_global_loads_to_lds(local_load_rhs)
        local_write_lhs = self.get_local_writes(local_load_lhs)
        local_write_rhs = self.get_local_writes(local_load_rhs)
        global_load_lhs = self.get_global_loads(local_write_lhs)
        global_load_rhs = self.get_global_loads(local_write_rhs)

        self.annotate_op_with_gemm_operation_type(
            local_load_lhs, GemmOperationType.LOCAL_LOAD_LHS
        )
        self.annotate_op_with_gemm_operation_type(
            local_load_rhs, GemmOperationType.LOCAL_LOAD_RHS
        )
        self.annotate_op_with_gemm_operation_type(
            global_to_shared_lhs, GemmOperationType.GLOBAL_LOAD_TO_LDS_LHS
        )
        self.annotate_op_with_gemm_operation_type(
            global_to_shared_rhs, GemmOperationType.GLOBAL_LOAD_TO_LDS_RHS
        )
        self.annotate_op_with_gemm_operation_type(
            local_write_lhs, GemmOperationType.LOCAL_WRITE_LHS
        )
        self.annotate_op_with_gemm_operation_type(
            local_write_rhs, GemmOperationType.LOCAL_WRITE_RHS
        )
        self.annotate_op_with_gemm_operation_type(
            global_load_lhs, GemmOperationType.GLOBAL_LOAD_LHS
        )
        self.annotate_op_with_gemm_operation_type(
            global_load_rhs, GemmOperationType.GLOBAL_LOAD_RHS
        )

        if mma_type == ScaledMMA:
            local_load_lhs_scale, local_load_rhs_scale = self.get_scale_local_loads(
                mma_nodes
            )
            global_to_shared_lhs_scale = self.get_global_loads_to_lds(
                local_load_lhs_scale
            )
            global_to_shared_rhs_scale = self.get_global_loads_to_lds(
                local_load_rhs_scale
            )
            local_write_lhs_scale = self.get_local_writes(local_load_lhs_scale)
            local_write_rhs_scale = self.get_local_writes(local_load_rhs_scale)
            global_load_lhs_scale = self.get_global_loads(local_write_lhs_scale)
            global_load_rhs_scale = self.get_global_loads(local_write_rhs_scale)

            self.annotate_op_with_gemm_operation_type(
                local_load_lhs_scale, GemmOperationType.LOCAL_LOAD_LHS_SCALE
            )
            self.annotate_op_with_gemm_operation_type(
                local_load_rhs_scale, GemmOperationType.LOCAL_LOAD_RHS_SCALE
            )
            self.annotate_op_with_gemm_operation_type(
                global_to_shared_lhs_scale,
                GemmOperationType.GLOBAL_LOAD_TO_LDS_LHS_SCALE,
            )
            self.annotate_op_with_gemm_operation_type(
                global_to_shared_rhs_scale,
                GemmOperationType.GLOBAL_LOAD_TO_LDS_RHS_SCALE,
            )
            self.annotate_op_with_gemm_operation_type(
                local_write_lhs_scale, GemmOperationType.LOCAL_WRITE_LHS_SCALE
            )
            self.annotate_op_with_gemm_operation_type(
                local_write_rhs_scale, GemmOperationType.LOCAL_WRITE_RHS_SCALE
            )
            self.annotate_op_with_gemm_operation_type(
                global_load_lhs_scale, GemmOperationType.GLOBAL_LOAD_LHS_SCALE
            )
            self.annotate_op_with_gemm_operation_type(
                global_load_rhs_scale, GemmOperationType.GLOBAL_LOAD_RHS_SCALE
            )
        self.annotate_op_with_gemm_operation_type(mma_nodes, GemmOperationType.MMA)
