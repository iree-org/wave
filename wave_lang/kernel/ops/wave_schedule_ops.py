from dataclasses import dataclass
from typing import Any, Sequence, TYPE_CHECKING
import torch.fx as fx
from ..ops.wave_ops import (
    get_custom,
    Read,
    Write,
    SharedMemoryBarrier,
    WorkgroupBarrier,
    SchedulingBarrier,
)
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
def get_node_by_tag(tag: str, subgraph: Any = None): ...


@define_schedule_op
def get_node_by_tag_and_type(tag: str, node_type: Any, subgraph: Any = None): ...


@define_schedule_op
def partition_by_address_space(node: Any, address_space: Any): ...


@define_schedule_op
def partition_by_dim(nodes: Any, dim: Any, factor: int): ...


@define_schedule_op
def cluster(ops: Any, barriers_before: str = "", barriers_after: str = ""): ...


@define_schedule_op
def reorder_graph(loop: Any, clusters: Any): ...


@define_schedule_op
def pipeline(iterate: Sequence[fx.Node]): ...


@define_schedule_op
def getitem(obj: Any, index: int): ...


def get_node_by_tag_helper(kernel_trace, tag: str):
    logger.info(f"Getting node by tag: {tag}")
    nodes = kernel_trace.walk(lambda x: get_custom(x).tag == tag)
    logger.info(f"Found {len(nodes)} nodes by tag: {tag}")
    return nodes


def add_op_before(op, subgraph: fx.Graph, anchor: fx.Node, location=None):
    """Insert a scheduling operation before the anchor node."""
    with subgraph.inserting_before(anchor):
        new_op = op.add_to_graph(subgraph)
    if location:
        new_op.location = location
    return new_op


def add_op_after(op, subgraph: fx.Graph, anchor: fx.Node, location=None):
    """Insert a scheduling operation after the anchor node."""
    with subgraph.inserting_after(anchor):
        new_op = op.add_to_graph(subgraph)
    if location:
        new_op.location = location
    return new_op


def extract_proxy_nodes(item):
    """Extract fx.Node(s) from a proxy or return direct node."""
    if isinstance(item, fx.Proxy):
        result = get_proxy_result(item.node)
        return result if isinstance(result, list) else [result]
    return [item]


@dataclass
class CustomScheduleOp:
    """Base class for custom schedule operations."""

    pass


@dataclass
class GetNodeByTag(CustomScheduleOp):
    tag: str
    subgraph: Any = None
    schedule_op_name = "get_node_by_tag"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        tag: str,
        subgraph: Any = None,
    ):
        # If subgraph is provided, search within that subgraph
        if subgraph is not None:
            assert hasattr(
                subgraph, "node"
            ), f"Expected 'subgraph' to be a proxy object with a 'node' attribute"
            subgraph_result = get_proxy_result(subgraph.node)
            assert subgraph_result is not None, "Subgraph node must have a result"
            assert (
                len(subgraph_result) > 0
            ), "Subgraph node must have at least one element"

            parent_node = subgraph_result[0]
            custom_parent = get_custom(parent_node)
            target_subgraph = kernel_trace.get_subgraph(custom_parent.subgraph_name)

            # Search within the subgraph
            nodes = [
                node
                for node in target_subgraph.nodes
                if hasattr(get_custom(node), "tag") and get_custom(node).tag == tag
            ]
            logger.info(
                f"Found {len(nodes)} nodes in subgraph '{custom_parent.subgraph_name}' with tag='{tag}'"
            )
        else:
            # Search the entire trace (original behavior)
            nodes = get_node_by_tag_helper(kernel_trace, tag)

        # Create a proxy that embeds the real result
        return create_schedule_proxy(
            region_graph,
            nodes,
            cls.schedule_op_name,
        )


@dataclass
class GetNodeByTagAndType(CustomScheduleOp):
    tag: str
    node_type: Any
    subgraph: Any = None
    schedule_op_name = "get_node_by_tag_and_type"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        tag: str,
        node_type: Any,
        subgraph: Any = None,
    ):
        assert constraints is not None, "Constraints are required"

        # If subgraph is provided, search within that subgraph
        if subgraph is not None:
            assert hasattr(
                subgraph, "node"
            ), f"Expected 'subgraph' to be a proxy object with a 'node' attribute"
            subgraph_result = get_proxy_result(subgraph.node)
            assert subgraph_result is not None, "Subgraph node must have a result"
            assert (
                len(subgraph_result) > 0
            ), "Subgraph node must have at least one element"

            parent_node = subgraph_result[0]
            custom_parent = get_custom(parent_node)
            target_subgraph = kernel_trace.get_subgraph(custom_parent.subgraph_name)

            # Search within the subgraph
            nodes = [
                node
                for node in target_subgraph.nodes
                if hasattr(get_custom(node), "tag")
                and get_custom(node).tag == tag
                and isinstance(get_custom(node), node_type)
            ]
            logger.info(
                f"Found {len(nodes)} nodes in subgraph '{custom_parent.subgraph_name}' with tag='{tag}' and type: {node_type}"
            )
        else:
            # Search the entire trace (original behavior)
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


@dataclass
class Cluster(CustomScheduleOp):
    ops: Any
    barriers_before: str = ""
    barriers_after: str = ""
    schedule_op_name = "cluster"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        ops: Any,
        barriers_before: str = "",
        barriers_after: str = "",
    ):
        from ..ops.wave_ops import SetWavePrio

        assert isinstance(ops, (list, tuple)), "ops must be a list"

        # Helper: Check if item is a scheduling op
        def is_scheduling_op(item):
            return isinstance(
                item,
                (SchedulingBarrier, WorkgroupBarrier, SharedMemoryBarrier, SetWavePrio),
            )

        # Find first proxy to get subgraph and context
        first_proxy_node = None
        for item in ops:
            if isinstance(item, fx.Proxy):
                first_proxy_node = extract_proxy_nodes(item)[0]
                break

        if first_proxy_node is None:
            raise ValueError("Cluster must have at least one proxy operation")

        subgraph = first_proxy_node.graph
        context_location = getattr(get_custom(first_proxy_node), "location", None)

        # Parse barrier string: "scheduling, workgroup" -> [SchedulingBarrier(), WorkgroupBarrier()]
        barrier_map = {
            "scheduling": lambda: SchedulingBarrier([]),
            "workgroup": lambda: WorkgroupBarrier(),
            "shared": lambda: SharedMemoryBarrier(),
        }

        result_nodes = []
        barriers_before_list = []
        barriers_after_list = []

        # Insert barriers_before BEFORE the first proxy node
        prev_node = None
        for b in barriers_before.split(","):
            b = b.strip()
            if b:
                assert b in barrier_map, f"Unknown barrier type: {b}"
                # First barrier goes before the first proxy, rest chain after each other
                if prev_node is None:
                    new_node = add_op_before(
                        barrier_map[b](), subgraph, first_proxy_node, context_location
                    )
                else:
                    new_node = add_op_after(
                        barrier_map[b](), subgraph, prev_node, context_location
                    )
                barriers_before_list.append(new_node)
                prev_node = new_node

        result_nodes.extend(barriers_before_list)

        # Track the last node for sequential insertion of ops
        last_anchor = prev_node if prev_node else None
        first_proxy_encountered = False

        # Process ops sequentially, inserting scheduling ops relative to proxies
        for item in ops:
            if isinstance(item, fx.Proxy):
                proxy_nodes = extract_proxy_nodes(item)
                result_nodes.extend(proxy_nodes)
                # Update anchor to the last proxy node
                last_anchor = proxy_nodes[-1]
                first_proxy_encountered = True
                # Update context location
                context_location = getattr(
                    get_custom(proxy_nodes[-1]), "location", None
                )
            else:
                # If we haven't encountered a proxy yet, insert before first proxy
                # Otherwise, insert after the last anchor
                if not first_proxy_encountered:
                    new_node = add_op_before(
                        item, subgraph, first_proxy_node, context_location
                    )
                else:
                    new_node = add_op_after(
                        item, subgraph, last_anchor, context_location
                    )
                result_nodes.append(new_node)
                last_anchor = new_node

        # Insert barriers_after AFTER the last operation
        for b in barriers_after.split(","):
            b = b.strip()
            if b:
                assert b in barrier_map, f"Unknown barrier type: {b}"
                new_node = add_op_after(
                    barrier_map[b](), subgraph, last_anchor, context_location
                )
                barriers_after_list.append(new_node)
                last_anchor = new_node

        result_nodes.extend(barriers_after_list)

        # this works but probably sending the result back as a list is also okay
        return create_schedule_proxy(region_graph, result_nodes, cls.schedule_op_name)


@dataclass
class ReorderGraph(CustomScheduleOp):
    loop: Any
    clusters: Any
    schedule_op_name = "reorder_graph"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        loop: Any,
        clusters: Any,
    ):
        from ..wave.schedule_reordering import reorder_graph as reorder_graph_impl

        # Get the iterate node from the proxy
        assert hasattr(
            loop, "node"
        ), f"Expected 'loop' to be a proxy object with a 'node' attribute, but got type: {type(loop).__name__}"
        loop_result = get_proxy_result(loop.node)
        assert loop_result is not None, "Loop must have a result"
        assert len(loop_result) > 0, "Loop must have at least one element"

        # Get the iterate's subgraph (this will be the KERNEL stage after pipelining)
        iterate_node = loop_result[0]
        custom_iterate = get_custom(iterate_node)
        subgraph = kernel_trace.get_subgraph(custom_iterate.subgraph_name)

        # Extract cluster nodes from proxies
        cluster_nodes = []

        assert isinstance(clusters, (list, tuple)), "Clusters must be a list or tuple"
        for item in clusters:
            cluster_nodes.extend(extract_proxy_nodes(item))

        logger.info(f"Reordering with {len(cluster_nodes)} cluster items")

        # Apply the reordering to the subgraph
        reordered_subgraph = reorder_graph_impl(subgraph, cluster_nodes)

        if reordered_subgraph is None:
            logger.warning("Failed to reorder graph, skipping reordering")
            return empty_proxy("reorder_graph_failed")

        # Replace the old subgraph with the reordered one
        reordered_subgraph.parent_op = subgraph.parent_op
        original_subgraph_name = custom_iterate.subgraph_name
        reordered_subgraph_name = f"reordered_{original_subgraph_name}"

        # Add the new subgraph and update references
        kernel_trace.add_subgraph(reordered_subgraph_name, reordered_subgraph)
        kernel_trace.get_root_graph().subgraphs[
            reordered_subgraph_name
        ] = reordered_subgraph
        custom_iterate.update_arg("subgraph_name", reordered_subgraph_name)

        # Remove the old subgraph
        del kernel_trace.region_graph.subgraphs[original_subgraph_name]
        del kernel_trace.get_root_graph().subgraphs[original_subgraph_name]

        logger.info(
            f"Successfully reordered graph: {original_subgraph_name} -> {reordered_subgraph_name}"
        )

        return empty_proxy("reorder_graph_success")


@dataclass
class PartitionByDim(CustomScheduleOp):
    nodes: Any
    dim: Any
    factor: int
    schedule_op_name = "partition_by_dim"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        nodes: Any,
        dim: Any,
        factor: int,
    ):
        # Get the actual nodes from the proxy
        assert hasattr(
            nodes, "node"
        ), f"Expected 'nodes' to be a proxy object with a 'node' attribute, but got type: {type(nodes).__name__}"
        nodes_list = get_proxy_result(nodes.node)
        assert nodes_list is not None, "Nodes must have a result"
        assert len(nodes_list) > 0, "Nodes must have at least one element"

        partitioned_nodes = [[] for _ in range(factor)]

        # Get all unique dimension IDs for the specified dimension
        dim_ids = set()
        for node in nodes_list:
            custom = get_custom(node)
            if custom.expanded_dims and dim in custom.expanded_dims:
                dim_ids.add(custom.expanded_dims[dim])

        # Validate that the dimension can be partitioned by the factor
        dim_expand_size = len(dim_ids)
        assert dim_expand_size >= factor and dim_expand_size % factor == 0, (
            f"Dimension {dim} has size {dim_expand_size} which cannot be evenly "
            f"partitioned by factor {factor}"
        )
        assert all(
            x in dim_ids for x in range(dim_expand_size)
        ), f"Dimension {dim} IDs are not contiguous: {sorted(dim_ids)}"

        size_of_partition = dim_expand_size // factor
        for node in nodes_list:
            custom = get_custom(node)
            if custom.expanded_dims and dim in custom.expanded_dims:
                dim_id = custom.expanded_dims[dim]
                partition_id = dim_id // size_of_partition
                partitioned_nodes[partition_id].append(node)
            else:
                # If node doesn't have the dimension, add to all partitions
                # This matches behavior for nodes that aren't expanded along this dim
                for partition in partitioned_nodes:
                    partition.append(node)

        # Return tuple of partitioned node lists
        return create_schedule_proxy(
            region_graph,
            tuple(partitioned_nodes),
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
        self.barrier_insertions = []

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
        subgraph = self.kernel_trace.get_subgraph(custom_iterate.subgraph_name)

        propagate_scheduling_parameters_to_iter_args(
            subgraph,
            self.initiation_interval,
        )

        # Apply barrier insertions before constructing the pipelined loop
        self._apply_barrier_insertions(subgraph)

        apply_pipelined_schedule(
            custom_iterate,
            subgraph,
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

    def insert_barrier_after(self, proxy, barrier_type="shared_memory"):
        """
        Insert a barrier after the operations in proxy.
        """

        # assert that barrier type is one of "shared_memory" or "workgroup" or "scheduling"
        assert barrier_type in [
            "shared_memory",
            "workgroup",
            "scheduling",
        ], "Invalid barrier type"
        self.barrier_insertions.append(
            {
                "proxy": proxy,
                "barrier_type": barrier_type,
                "position": "after",
            }
        )
        return self

    def _apply_barrier_insertions(self, subgraph: fx.Graph):
        """
        Apply all queued barrier insertions to the subgraph.
        """

        for insertion in self.barrier_insertions:
            # Get the actual nodes from the proxy
            target_nodes = get_proxy_result(insertion["proxy"])
            if not target_nodes:
                continue

            if not isinstance(target_nodes, (list, tuple)):
                target_nodes = [target_nodes]

            # Find the last node in the list (to insert after)
            last_node = target_nodes[-1]

            # Get scheduling parameters from the last target node
            custom = get_custom(last_node)
            if (
                not hasattr(custom, "scheduling_parameters")
                or custom.scheduling_parameters is None
            ):
                print(
                    f"Warning: Node {last_node} has no scheduling parameters, skipping barrier insertion"
                )
                continue

            # Insert barrier after the last target node in the graph
            barrier = None
            with subgraph.inserting_after(last_node):
                if insertion["barrier_type"] == "scheduling":
                    barrier = SchedulingBarrier([]).add_to_graph(
                        subgraph, loc=custom.location
                    )
                elif insertion["barrier_type"] == "workgroup":
                    barrier = WorkgroupBarrier().add_to_graph(
                        subgraph, loc=custom.location
                    )
                else:
                    barrier = SharedMemoryBarrier().add_to_graph(
                        subgraph, loc=custom.location
                    )

            # Copy scheduling parameters from the target node (for the rest of the passes)
            barrier_custom = get_custom(barrier)
            barrier_custom.scheduling_parameters = custom.scheduling_parameters.copy()


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
        if isinstance(obj, fx.Proxy):
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
