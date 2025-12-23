from dataclasses import dataclass
from typing import Any, Sequence, TYPE_CHECKING
import torch.fx as fx
from ..ops.wave_ops import (
    get_custom,
    Read,
    Write,
    Placeholder,
)
from ..wave.constraints import Constraint
from .base import define_schedule_op
import logging

if TYPE_CHECKING:
    from .._support.tracing import CapturedTrace

logger = logging.getLogger(__name__)


# Stubs to enable type checking of the custom schedule ops - decorated with @define_op for dispatch
@define_schedule_op
def get_node_by_tag(tag: str): ...


@define_schedule_op
def get_node_by_tag_and_type(tag: str, node_type: Any): ...


@define_schedule_op
def partition_by_address_space(node: Any, address_space: Any): ...


@define_schedule_op
def partition_by_dim(nodes: Any, dim: Any, num_partitions: int): ...


@define_schedule_op
def cluster(ops: Any): ...


@define_schedule_op
def insert_before(target: Any, op: Any): ...


@define_schedule_op
def insert_after(target: Any, op: Any): ...


@define_schedule_op
def insert_at_start(subgraph_ref: Any, op: Any): ...


@define_schedule_op
def insert_at_end(subgraph_ref: Any, op: Any): ...


@define_schedule_op
def reorder_graph(loop: Any, clusters: Any): ...


@define_schedule_op
def pipeline(iterate: Sequence[fx.Node]): ...


@define_schedule_op
def getitem(obj: Any, index: int): ...


@define_schedule_op
def stagger(loop: Any): ...


@define_schedule_op
def filter_nodes(nodes: Any, subgraph: Any = None, node_type: Any = None): ...


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


def extract_nodes(item):
    """Extract fx.Node(s) from a list or return direct node."""
    if isinstance(item, (list, tuple)):
        # Direct list of nodes
        return list(item)
    # Single node
    return [item]


def get_nodes_from_ref(ref):
    """
    Get the actual nodes from a reference (PipelineStageRef or list).
    """
    if isinstance(ref, PipelineStageRef):
        # For PipelineStageRef, return the pipelined iterate node directly
        return [ref.pipelined_iterate_node]
    elif isinstance(ref, (list, tuple)):
        # Direct list of nodes
        return list(ref)
    else:
        raise ValueError(f"Expected PipelineStageRef or list, got {type(ref)}")


@dataclass
class PipelineStageRef:
    """Reference to a specific pipeline stage (KERNEL, PROLOGUE, or EPILOGUE)."""

    pipelined_iterate_node: Any
    stage: Any


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

        # Return nodes directly instead of wrapping in proxy
        return real_result


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

        # Return nodes directly instead of wrapping in proxy
        return nodes


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

        # Expect a list or tuple of nodes
        if not isinstance(nodes, (list, tuple)):
            raise ValueError(
                f"Expected 'nodes' to be a list or tuple, but got type: {type(nodes).__name__}"
            )
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

        # Return tuple of lists directly instead of wrapping in proxy
        return (matched, unmatched)


@dataclass
class Cluster(CustomScheduleOp):
    ops: Any
    schedule_op_name = "cluster"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        ops: Any,
    ):

        assert isinstance(ops, (list, tuple)), "ops must be a list"

        # Find first node from a list to get subgraph and context
        first_node = None
        for item in ops:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                # Direct list of nodes
                first_node = item[0]
                break
            elif isinstance(item, fx.Node):
                # Single node
                first_node = item
                break

        if first_node is None:
            raise ValueError("Cluster must have at least one node")

        subgraph = first_node.graph
        context_location = getattr(get_custom(first_node), "location", None)

        result_nodes = []

        # Track the last node for sequential insertion of ops
        last_anchor = None
        first_node_encountered = False

        # Process ops sequentially, inserting scheduling ops relative to nodes or lists
        for item in ops:
            if isinstance(item, fx.Node):
                # Single node
                result_nodes.append(item)
                last_anchor = item
                first_node_encountered = True
                context_location = getattr(get_custom(item), "location", None)
            elif isinstance(item, (list, tuple)):
                # Direct list of nodes
                result_nodes.extend(item)
                # Update anchor to the last node
                last_anchor = item[-1]
                first_node_encountered = True
                # Update context location
                context_location = getattr(get_custom(item[-1]), "location", None)
            else:
                # If we haven't encountered a node yet, insert before first node
                # Otherwise, insert after the last anchor
                if not first_node_encountered:
                    new_node = add_op_before(
                        item, subgraph, first_node, context_location
                    )
                else:
                    new_node = add_op_after(
                        item, subgraph, last_anchor, context_location
                    )
                result_nodes.append(new_node)
                last_anchor = new_node

        # Return list of nodes directly instead of wrapping in proxy
        return result_nodes


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

        # Get the iterate node from the reference (PipelineStageRef or list)
        loop_result = get_nodes_from_ref(loop)
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
            cluster_nodes.extend(extract_nodes(item))

        logger.info(f"Reordering with {len(cluster_nodes)} cluster items")

        # Apply the reordering to the subgraph
        reordered_subgraph = reorder_graph_impl(subgraph, cluster_nodes)

        if reordered_subgraph is None:
            logger.warning("Failed to reorder graph, skipping reordering")
            return None

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

        return None


@dataclass
class PartitionByDim(CustomScheduleOp):
    nodes: Any
    dim: Any
    num_partitions: int
    schedule_op_name = "partition_by_dim"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        nodes: Any,
        dim: Any,
        num_partitions: int,
    ):
        """
        Partition a list of nodes by dimension into equal-sized partitions.

        This function sorts nodes by the specified dimension, then partitions them
        into equal-sized groups. Each partition contains nodes with contiguous
        dimension IDs.

        For example, with M,N,K = 2,8,4 and num_partitions=2, partitioning by K dimension
        creates two partitions: first with K IDs 0-1, second with K IDs 2-3.

        """
        # Expect a list or tuple of nodes
        if not isinstance(nodes, (list, tuple)):
            raise ValueError(
                f"Expected 'nodes' to be a list or tuple, but got type: {type(nodes).__name__}"
            )
        nodes_list = list(nodes)
        assert len(nodes_list) > 0, "Nodes must have at least one element"

        # Get all unique dimension IDs for the specified dimension
        dim_ids = set()
        for node in nodes_list:
            custom = get_custom(node)
            if custom.expanded_dims and dim in custom.expanded_dims:
                dim_ids.add(custom.expanded_dims[dim])

        # Validate that the dimension can be partitioned by the num_partitions
        dim_expand_size = len(dim_ids)
        assert (
            dim_expand_size >= num_partitions and dim_expand_size % num_partitions == 0
        ), (
            f"Dimension {dim} has {dim_expand_size} unique IDs which cannot be evenly "
            f"partitioned by num_partitions {num_partitions}"
        )

        # Sort nodes by dimension and compute partition boundaries
        sorted_nodes_list = sorted(
            nodes_list, key=lambda x: get_custom(x).expanded_dims[dim]
        )
        nodes_per_dim = len(sorted_nodes_list) // dim_expand_size
        partition_size = dim_expand_size // num_partitions

        # Partition nodes by dimension ID ranges
        partitioned_nodes = []
        for partition_id in range(num_partitions):
            lower_bound_dim_id = partition_id * partition_size
            upper_bound_dim_id = (partition_id + 1) * partition_size

            start_idx = lower_bound_dim_id * nodes_per_dim
            end_idx = upper_bound_dim_id * nodes_per_dim

            partitioned_nodes.append(sorted_nodes_list[start_idx:end_idx])

        # Return tuple of partitioned node lists directly instead of wrapping in proxy
        return tuple(partitioned_nodes)


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

        # Access options from the current ScheduleContext
        from .._support.tracing import ScheduleContext

        current_context = ScheduleContext.current()
        self.use_scheduling_barriers = (
            current_context.use_scheduling_barriers if current_context else False
        )

        self.initiation_interval = None
        self.num_stages = 0

        # These will be set after pipelining is applied
        self._pipelined_iterate_node = None
        self._KERNEL = None
        self._PROLOGUE = None
        self._EPILOGUE = None

        # Track lists used during set_stage to auto-update after pipelining
        # This allows us to mutate lists in-place with their pipelined nodes
        self._tracked_lists = {}

        # Direct mapping from original nodes to their pipelined copies
        self._node_mapping = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from ..wave.scheduling.schedule import (
            propagate_scheduling_parameters_to_iter_args,
            apply_pipelined_schedule,
        )
        from ..wave.scheduling.schedule_enums import SchedulingType

        # Expect a list or tuple
        if not isinstance(self.iterate, (list, tuple)):
            raise ValueError(
                f"Expected iterate to be a list or tuple, got {type(self.iterate)}"
            )

        result = self.iterate
        assert isinstance(result, Sequence), "Iterate must be a sequence"
        assert len(result) == 1, "Iterate must have exactly one element"

        custom_iterate = get_custom(result[0])
        subgraph = self.kernel_trace.get_subgraph(custom_iterate.subgraph_name)

        propagate_scheduling_parameters_to_iter_args(
            subgraph,
            self.initiation_interval,
        )

        # Get use_scheduling_barriers from options if available
        pipelining_result = apply_pipelined_schedule(
            custom_iterate,
            subgraph,
            self.kernel_trace,
            self.constraints,
            use_scheduling_barriers=self.use_scheduling_barriers,
            num_stages=self.num_stages,
            initiation_interval=self.initiation_interval,
            scheduling_type=SchedulingType.MANUAL,
            visualize=False,
            multi_buffer_count=None,
        )

        # Store the pipelined iterate node and node mapping, then create proxies for the stages
        if pipelining_result is not None:
            assert (
                len(pipelining_result) == 2
            ), "Pipelining result must be a tuple of length 2"
            pipelined_iterate_node, node_mapping = pipelining_result
            assert isinstance(
                pipelined_iterate_node, fx.Node
            ), "Pipelined iterate node must be a fx.Node"
            assert isinstance(node_mapping, dict), "Node mapping must be a dictionary"
            self._pipelined_iterate_node = pipelined_iterate_node
            self._node_mapping = node_mapping
            self._create_stage_refs()
            self._update_kernel_node_mapping()

            # Store node mapping in context for auto-update of direct node references
            from .._support.tracing import ScheduleContext

            ctx = ScheduleContext.current()
            if ctx is not None:
                ctx.node_mapping.update(node_mapping)
                logger.info(
                    f"Stored {len(node_mapping)} node mappings in context for auto-update"
                )
        else:
            logger.warning(
                "Pipelining failed, KERNEL/PROLOGUE/EPILOGUE properties will not be available"
            )

    def _create_stage_refs(self):
        """Create references for KERNEL, PROLOGUE, and EPILOGUE stages."""
        if self._pipelined_iterate_node is None:
            return

        # Import locally to avoid circular import
        from ..wave.scheduling.loop_reconstruction import PipelineStage

        # Create simple stage references
        self._KERNEL = PipelineStageRef(
            self._pipelined_iterate_node, PipelineStage.KERNEL
        )
        self._PROLOGUE = PipelineStageRef(
            self._pipelined_iterate_node, PipelineStage.PROLOGUE
        )
        self._EPILOGUE = PipelineStageRef(
            self._pipelined_iterate_node, PipelineStage.EPILOGUE
        )

    @property
    def KERNEL(self):
        """Get a reference to the KERNEL stage (pipelined iterate subgraph)."""
        return self._KERNEL

    @property
    def PROLOGUE(self):
        """Get a reference to the PROLOGUE stage (nodes before pipelined iterate)."""
        return self._PROLOGUE

    @property
    def EPILOGUE(self):
        """Get a reference to the EPILOGUE stage (nodes after pipelined iterate)."""
        return self._EPILOGUE

    def _update_kernel_node_mapping(self):
        """
        Auto-update tracked lists to contain all mapped nodes (prologue, kernel, epilogue).

        Since lists are mutable, we update them in-place so user variables automatically
        reflect the pipelined nodes.
        """
        from .._support.tracing import ScheduleContext

        ctx = ScheduleContext.current()
        if ctx is None:
            logger.warning("No schedule context, cannot update node mappings")
            return

        if self._node_mapping is None:
            logger.warning("No node mapping available, cannot update lists")
            return

        # Update each tracked list by mutating it in-place
        updated_count = 0
        for tracked_list in self._tracked_lists.values():
            if not isinstance(tracked_list, list):
                continue

            # Save original nodes
            original_nodes = list(tracked_list)

            # Collect all mapped versions using direct mapping
            mapped_nodes = []
            for orig_node in original_nodes:
                if orig_node in self._node_mapping:
                    mapped_nodes.extend(self._node_mapping[orig_node])

            if mapped_nodes:
                # Mutate the list in-place - this updates the user's variable!
                tracked_list.clear()
                tracked_list.extend(mapped_nodes)
                logger.debug(
                    f"Updated list in-place: {len(original_nodes)} original -> {len(mapped_nodes)} mapped nodes"
                )
                updated_count += 1

        logger.info(f"Auto-updated {updated_count} lists in-place with pipelined nodes")

    def get_kernel_nodes(self, tag: str, node_type=None):
        """
        Get nodes from KERNEL stage by tag, optionally filtered by type.
        This is a convenience method to avoid re-querying after pipelining.

        Args:
            tag: The tag to search for
            node_type: Optional node type to filter by (e.g., tkw.Read, tkw.Write)

        Returns:
            A list containing the nodes with the specified tag in the KERNEL stage
        """
        if not hasattr(self, "_kernel_nodes_by_tag"):
            logger.warning("Kernel node mapping not available, returning empty list")
            nodes = []
        else:
            nodes = self._kernel_nodes_by_tag.get(tag, [])

            if node_type is not None:
                nodes = [n for n in nodes if isinstance(get_custom(n), node_type)]

        # Return list directly
        return nodes

    def set_stage(self, nodes: Sequence[fx.Node]):
        """
        Set scheduling stage for nodes.

        Tracks mutable lists to enable in-place mutation after pipelining.
        This ensures that user variables automatically reflect pipelined nodes.
        """
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
                # Handle lists and tuples
                if isinstance(node, list):
                    # It's a mutable list - track it for mutation after pipelining
                    node_result = node
                    # Track this list so we can mutate it after pipelining
                    list_id = id(node)
                    if list_id not in self._tracked_lists:
                        self._tracked_lists[list_id] = node
                        logger.debug(
                            f"Tracking list with {len(node)} original nodes for auto-update"
                        )
                elif isinstance(node, tuple):
                    # Convert tuple to list so we can track and mutate it
                    node_result = list(node)
                    logger.warning(
                        "Converting tuple to list for mutation - consider using lists directly"
                    )
                else:
                    # Assume it's a single node
                    node_result = [node]

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
        # Return None - no proxy needed
        return None


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
        # Handle lists and tuples
        if not isinstance(obj, (list, tuple)):
            raise ValueError(
                f"Object must be a list or tuple, got {type(obj).__name__}"
            )

        if len(obj) > index:
            return obj[index]
        else:
            raise ValueError(
                f"Index {index} out of bounds for list of length {len(obj)}"
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

        # Return the real object directly (no proxy needed)
        return real_pipelined_loop


@dataclass
class Stagger(CustomScheduleOp):
    schedule_op_name = "stagger"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        loop: Any,
    ):
        """
        Implements stagger scheduling by adding conditional barriers around a loop that blocks waves such that
        waves execute clusters in a staggered manner for better overlap of computation and memory access.

        For 2 waves (default):
        at T0:
           wave 0 runs cluster 0
           wave 1 blocked
        at T1:
           wave 0 runs cluster 1
           wave 1 runs cluster 0
        at T2:
           wave 0 runs cluster 2
           wave 1 runs cluster 1

        This pattern continues, allowing N waves to execute clusters in parallel with a stagger offset.
        """
        from ..wave.schedule_reordering import add_conditional_barriers_to_loop
        from ..wave.utils.general_utils import get_hardware_constraint

        # Get the iterate node from the reference (PipelineStageRef or list)
        loop_result = get_nodes_from_ref(loop)
        assert loop_result is not None, "Loop must have a result"
        assert len(loop_result) > 0, "Loop must have at least one element"

        iterate_node = loop_result[0]
        custom_iterate = get_custom(iterate_node)

        # Get hardware constraints
        hardware_constraint = get_hardware_constraint(constraints)
        if hardware_constraint is None:
            raise ValueError("Stagger requires HardwareConstraint")

        add_conditional_barriers_to_loop(
            custom_iterate, kernel_trace, hardware_constraint
        )

        logger.info(f"Applied 2-way stagger scheduling to loop")

        return None


@dataclass
class InsertAtStart(CustomScheduleOp):
    schedule_op_name = "insert_at_start"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        subgraph_ref: Any,
        op: Any,
    ):
        """
        Inserts an operation at the beginning of a subgraph (after placeholders).

        Args:
            subgraph_ref: Reference to a subgraph (e.g., iterate node, conditional, or pipeline stage)
            op: The operation to insert (e.g., SharedMemoryBarrier, SchedulingBarrier)
        """

        # Get the subgraph from the reference
        nodes = get_nodes_from_ref(subgraph_ref)
        assert nodes is not None, "Subgraph reference must have a result"
        assert len(nodes) > 0, "Subgraph reference must have at least one element"
        custom = get_custom(nodes[0])

        # Get the subgraph
        assert hasattr(
            custom, "subgraph_name"
        ), f"Node must have a subgraph (got {custom.__class__.__name__})"
        subgraph = kernel_trace.get_subgraph(custom.subgraph_name)

        # Find the first non-placeholder node
        first_node = None
        for node in subgraph.nodes:
            node_custom = get_custom(node)
            if not isinstance(node_custom, Placeholder):
                first_node = node
                break

        assert (
            first_node is not None
        ), "Could not find first non-placeholder node in subgraph"

        # Insert before the first non-placeholder node
        add_op_before(op, subgraph, first_node, custom.location)

        logger.info(f"Inserted {op.__class__.__name__} at start of subgraph")
        return None


@dataclass
class InsertAtEnd(CustomScheduleOp):
    schedule_op_name = "insert_at_end"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        subgraph_ref: Any,
        op: Any,
    ):
        """
        Inserts an operation at the end of a subgraph (before the output node).

        Args:
            subgraph_ref: Reference to a subgraph (e.g., iterate node, conditional, or pipeline stage)
            op: The operation to insert (e.g., SharedMemoryBarrier, SchedulingBarrier)
        """
        # Get the subgraph from the reference
        nodes = get_nodes_from_ref(subgraph_ref)
        assert nodes is not None, "Subgraph reference must have a result"
        assert len(nodes) > 0, "Subgraph reference must have at least one element"
        custom = get_custom(nodes[0])

        # Get the subgraph
        assert hasattr(
            custom, "subgraph_name"
        ), f"Node must have a subgraph (got {custom.__class__.__name__})"
        subgraph = kernel_trace.get_subgraph(custom.subgraph_name)

        # Get the output node
        output_node = subgraph.output_node()
        assert output_node is not None, "Subgraph must have an output node"

        # Insert before the output node
        add_op_before(op, subgraph, output_node, custom.location)

        logger.info(f"Inserted {op.__class__.__name__} at end of subgraph")
        return None


@dataclass
class InsertBefore(CustomScheduleOp):
    schedule_op_name = "insert_before"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        target: Any,
        op: Any,
    ):
        """
        Inserts an operation before the target node in the graph.

        Args:
            target: The target reference (e.g., pipelined_loop.KERNEL or list)
            op: The operation to insert (e.g., SharedMemoryBarrier, SchedulingBarrier)
        """
        # Get the iterate node from the reference
        nodes = get_nodes_from_ref(target)
        assert nodes is not None, "Nodes must have a result"
        assert len(nodes) > 0, "Nodes must have at least one element"
        custom = get_custom(nodes[0])

        # Use the helper function
        add_op_before(op, custom.graph, custom.fx_node, custom.location)

        return None


@dataclass
class InsertAfter(CustomScheduleOp):
    schedule_op_name = "insert_after"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        target: Any,
        op: Any,
    ):
        """
        Inserts an operation after the target node in the graph.

        Args:
            target: The target reference (e.g., pipelined_loop.KERNEL or list)
            op: The operation to insert (e.g., SharedMemoryBarrier, SchedulingBarrier)
        """
        # Get the iterate node from the reference
        nodes = get_nodes_from_ref(target)
        custom = get_custom(nodes[0])

        # Use the helper function
        add_op_after(op, custom.graph, custom.fx_node, custom.location)

        return None


@dataclass
class FilterNodes(CustomScheduleOp):
    schedule_op_name = "filter_nodes"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        nodes: Any,
        subgraph: Any = None,
        node_type: Any = None,
    ):
        """
        Filter nodes by pipeline stage and/or node type.

        Args:
            nodes: The nodes list to filter
            subgraph: Optional PipelineStageRef (e.g., pipelined_loop.KERNEL) to filter by stage
            node_type: Optional node type (e.g., tkw.Read, tkw.Write) to filter by type

        Returns:
            A list containing the filtered nodes
        """

        # Expect a list or tuple
        if not isinstance(nodes, (list, tuple)):
            raise ValueError(
                f"Expected 'nodes' to be a list or tuple, but got type: {type(nodes).__name__}"
            )

        nodes_list = list(nodes)
        assert len(nodes_list) > 0, "FilterNodes: Nodes must have at least one element"

        if not nodes_list:
            return []

        filtered_nodes = nodes_list

        # Filter by pipeline stage if specified
        if subgraph is not None:
            if isinstance(subgraph, PipelineStageRef):
                filtered_nodes = [
                    node
                    for node in filtered_nodes
                    if node.meta.get("pipeline_stage") == subgraph.stage
                ]
                logger.info(
                    f"Filtered to {len(filtered_nodes)} nodes in {subgraph.stage.name} stage"
                )
            else:
                logger.warning(
                    f"Subgraph filter must be a PipelineStageRef, got {type(subgraph)}"
                )

        # Filter by node type if specified
        if node_type is not None:
            filtered_nodes = [
                node
                for node in filtered_nodes
                if isinstance(get_custom(node), node_type)
            ]
            logger.info(
                f"Filtered to {len(filtered_nodes)} nodes of type {node_type.__name__}"
            )

        # Return list directly
        return filtered_nodes
