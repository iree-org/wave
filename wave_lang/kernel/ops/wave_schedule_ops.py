from dataclasses import dataclass
from typing import Any, Sequence, TYPE_CHECKING
import torch.fx as fx
from ..ops.wave_ops import get_custom, Read, Write
from ..wave.constraints import Constraint
from .base import define_schedule_op
import logging

if TYPE_CHECKING:
    from .._support.tracing import CapturedTrace

logger = logging.getLogger(__name__)


def get_proxy_result(proxy):
    """
    Get the real result for a proxy from the current schedule context.

    Args:
        proxy: The proxy to resolve

    Returns:
        The real result if found in the context, otherwise None
    """
    from .._support.tracing import ScheduleContext

    current_context = ScheduleContext.current()
    if current_context is not None:
        return current_context.get_proxy_result(proxy)
    return None


def create_schedule_proxy(
    region_graph,
    real_value: Any,
    op_name: str = "schedule_op",
):
    """
    Create a proxy for a schedule operation that embeds the real value.

    Args:
        region_graph: The region graph to create the proxy in
        real_value: The real value to embed in the proxy
        op_name: The name of the operation for debugging

    Returns:
        The created proxy
    """

    # Create a proxy function that returns the embedded real value
    def proxy_func(*proxy_args, **proxy_kwargs):
        return real_value

    # Set the function name for better debugging
    proxy_func.__name__ = op_name

    try:
        proxy = region_graph.create_proxy(
            "call_function",
            proxy_func,
            (),
            {},
        )

        from .._support.tracing import ScheduleContext

        current_context = ScheduleContext.current()
        if current_context is not None:
            current_context.proxy_to_results[proxy] = real_value

        return proxy

    except Exception as e:
        logger.exception("Failed to create schedule proxy for op '%s': %s", op_name, e)
        raise ValueError(
            f"Failed to create schedule proxy for op '{op_name}': {e}"
        ) from e


def empty_proxy(name: str = "empty_proxy"):
    from .._support.tracing import ScheduleContext

    current_context = ScheduleContext.current()
    if current_context is not None:
        return create_schedule_proxy(
            current_context.region_graph,
            None,
            name,
        )


# Stubs to enable type checking of the custom schedule ops - decorated with @define_op for dispatch
@define_schedule_op
def get_node_by_tag(tag: str): ...


@define_schedule_op
def get_node_by_tag_and_type(tag: str, node_type: Any): ...


@define_schedule_op
def partition_by_address_space(node: Any, address_space: Any): ...


@define_schedule_op
def pipeline(iterate: Sequence[fx.Node]): ...


@define_schedule_op
def getitem(obj: Any, index: int): ...


def get_node_by_tag_helper(kernel_trace, tag: str):
    logger.info(f"Getting node by tag: {tag}")
    nodes = kernel_trace.walk(lambda x: get_custom(x).tag == tag)
    logger.info(f"Found {len(nodes)} nodes by tag: {tag}")
    return nodes


@dataclass
class CustomScheduleOp:
    """Base class for custom schedule operations."""

    pass


@dataclass
class GetNodeByTag(CustomScheduleOp):
    tag: str
    schedule_op_name = "get_node_by_tag"

    @classmethod
    def handle(
        cls, region_graph, kernel_trace, constraints: list[Constraint], tag: str
    ):
        # Always execute the real logic during tracing to apply scheduling
        real_result = get_node_by_tag_helper(kernel_trace, tag)

        # Create a proxy that embeds the real result
        return create_schedule_proxy(
            region_graph,
            real_result,
            cls.schedule_op_name,
        )


@dataclass
class GetNodeByTagAndType(CustomScheduleOp):
    tag: str
    node_type: Any
    schedule_op_name = "get_node_by_tag_and_type"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        tag: str,
        node_type: Any,
    ):
        assert constraints is not None, "Constraints are required"

        nodes = get_node_by_tag_helper(kernel_trace, tag)
        nodes = [node for node in nodes if isinstance(get_custom(node), node_type)]
        logger.info(f"Found {len(nodes)} nodes by tag: {tag} and type: {node_type}")

        return create_schedule_proxy(
            region_graph,
            nodes,
            cls.schedule_op_name,
        )


@dataclass
class PartitionByAddressSpace(CustomScheduleOp):
    node: Any
    address_space: Any
    schedule_op_name = "partition_by_address_space"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        nodes: Sequence[fx.Node],
        address_space: Any,
    ):
        matched, unmatched = [], []

        assert hasattr(
            nodes, "node"
        ), f"Expected 'nodes' to be a proxy object with a 'node' attribute, but got type: {type(nodes).__name__}"
        nodes = get_proxy_result(nodes.node)
        assert nodes is not None, "Nodes must have a result"
        assert len(nodes) > 0, "Nodes must have at least one element"

        assert all(
            [
                isinstance(get_custom(node), Read)
                or isinstance(get_custom(node), Write)
                for node in nodes
            ]
        ), "Nodes to partition must be Read or Write"
        matched = [
            node
            for node in nodes
            if get_custom(node).memory_type.address_space == address_space
        ]
        unmatched = [node for node in nodes if node not in matched]

        return create_schedule_proxy(
            region_graph,
            (matched, unmatched),
            cls.schedule_op_name,
        )


class PipelinedLoop:
    def __init__(
        self,
        iterate: Sequence[fx.Node],
        kernel_trace: "CapturedTrace",
        constraints: list[Constraint],
    ):
        self.iterate = iterate
        self.kernel_trace = kernel_trace
        self.constraints = constraints
        self.initiation_interval = None
        self.num_stages = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from ..wave.scheduling.schedule import (
            propagate_scheduling_parameters_to_iter_args,
            apply_pipelined_schedule,
        )
        from ..wave.scheduling.schedule_enums import SchedulingType

        result = get_proxy_result(self.iterate)
        assert result is not None, "Iterate must have a result"
        assert isinstance(result, Sequence), "Iterate must be a sequence"
        assert len(result) == 1, "Iterate must have exactly one element"

        custom_iterate = get_custom(result[0])
        propagate_scheduling_parameters_to_iter_args(
            self.kernel_trace.get_subgraph(custom_iterate.subgraph_name),
            self.initiation_interval,
        )
        apply_pipelined_schedule(
            custom_iterate,
            self.kernel_trace.get_subgraph(custom_iterate.subgraph_name),
            self.kernel_trace,
            self.constraints,
            use_scheduling_barriers=True,
            num_stages=self.num_stages,
            initiation_interval=self.initiation_interval,
            scheduling_type=SchedulingType.MANUAL,
            visualize=False,
            multi_buffer_count=None,
        )

    def set_stage(self, nodes: Sequence[fx.Node]):
        stage = self.num_stages
        if self.initiation_interval is None:
            self.initiation_interval = len(nodes)
        else:
            assert self.initiation_interval == len(
                nodes
            ), "The number of clusters must be the same across stages"
        result_clusters = []
        for cluster in nodes:
            result_nodes = []
            for node in cluster:
                node_result = get_proxy_result(node)
                assert node_result is not None, "Nodes must have a result"
                result_nodes.append(node_result)
            result_clusters.append(tuple(result_nodes))

        for i, cluster in enumerate(result_clusters):
            for nodes in cluster:
                for node in nodes:
                    custom = get_custom(node)
                    custom.scheduling_parameters = {
                        "absolute_cycle": stage * self.initiation_interval + i,
                        "stage": stage,
                        "initiation_interval": self.initiation_interval,
                        "prefetch_stage": None,
                    }
                    custom.scheduling_parameters["cycle"] = (
                        custom.scheduling_parameters["absolute_cycle"]
                        % self.initiation_interval
                    )

        self.num_stages += 1
        # During tracing, return a proxy for the set_stage operation
        return empty_proxy("set_stage")


@dataclass
class GetItem(CustomScheduleOp):
    obj: Any
    index: int
    schedule_op_name = "getitem"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        obj: Any,
        index: int,
    ):
        if hasattr(obj, "node"):
            source_result = get_proxy_result(obj.node)
            if isinstance(source_result, (list, tuple)) and len(source_result) > index:
                real_result = source_result[index]
            else:
                raise ValueError(f"Index {index} out of bounds for {source_result}")
        else:
            raise ValueError(f"Object {obj} is not a proxy")

        return create_schedule_proxy(
            region_graph,
            real_result,
            cls.schedule_op_name,
        )


@dataclass
class Pipeline(CustomScheduleOp):
    schedule_op_name = "pipeline"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        iterate: Sequence[fx.Node],
    ):
        real_pipelined_loop = PipelinedLoop(iterate, kernel_trace, constraints)

        create_schedule_proxy(
            region_graph,
            real_pipelined_loop,
            cls.schedule_op_name,
        )

        # For context manager support, return the real object
        return real_pipelined_loop
