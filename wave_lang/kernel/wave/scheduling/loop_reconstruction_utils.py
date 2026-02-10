import random
from collections import defaultdict
from typing import Optional, Sequence

import torch.fx as fx

from wave_lang.support.logging import get_logger
from ..utils.general_utils import (
    is_shared_write,
    get_shared_memory_operands,
    ceildiv,
    propagate_loop_carried_vars,
)
from ...ops.wave_ops import (
    GatherToLDS,
    GetResult,
    IterArg,
    TensorLoadToLDS,
    Write,
    get_custom,
)

logger = get_logger("wave.scheduling.loop_reconstruction_utils")


class ArgumentContext:
    """
    The argument context is used to store the mapping of arguments
    for each modulo pipelining stage.
    """

    def __init__(
        self,
        results: list[fx.Node],
        iter_args: list[fx.Node],
        init_args: list[fx.Node],
        num_stages: int,
        num_iterations: Optional[int] = None,
    ) -> None:
        if num_iterations is None:
            num_iterations = num_stages
        self.argument_map: list[list[dict[fx.Node, fx.Node]]] = [
            [{} for _ in range(num_stages)] for _ in range(num_iterations)
        ]
        self.results = results
        self.iter_args = iter_args
        self.init_args = init_args
        self.num_stages = num_stages
        self.num_iterations = num_iterations
        self.result_to_iter_arg: dict[fx.Node, fx.Node] = {}
        self.result_to_init_arg: dict[fx.Node, fx.Node] = {}

        for result, iter_arg in zip(results, iter_args):
            self.result_to_iter_arg[result] = iter_arg
        for result, init_arg in zip(results, init_args):
            self.result_to_init_arg[result] = init_arg

    def map_arg_all(self, from_: fx.Node, to_: fx.Node | Sequence[fx.Node]) -> None:
        """
        Maps the given argument from one to another into the argument context for all stages
        and for all iterations.
        """
        if isinstance(to_, Sequence):
            count = len(to_)
            for iteration in range(self.num_iterations):
                for stage in range(self.num_stages):
                    self.argument_map[iteration][stage][from_] = to_[iteration % count]
        else:
            for iteration in range(self.num_iterations):
                for stage in range(self.num_stages):
                    self.argument_map[iteration][stage][from_] = to_

    def map_arg_all_after_iteration(
        self, from_: fx.Node, to_: fx.Node, iteration: int
    ) -> None:
        """
        Maps the given argument from one to another into the argument context for all stages
        after the specified iteration.
        """
        for iteration in range(iteration + 1, self.num_iterations):
            for stage in range(self.num_stages):
                self.argument_map[iteration][stage][from_] = to_

    def map_arg_all_iterations(self, stage: int, from_: fx.Node, to_: fx.Node) -> None:
        """
        Maps the given argument from one to another into the argument context for all stages
        and for all iterations.
        """
        for iteration in range(self.num_iterations):
            self.argument_map[iteration][stage][from_] = to_

    def get_mapped_results(self, get_results: list[GetResult]) -> list[fx.Node]:
        """
        Gets the mapped results from the last iteration. If the result is not
        in the last iteration, then get it from the get result nodes.
        """
        mapped_results = []
        for result, get_result in zip(self.results, get_results):
            stage = result.scheduling_parameters["stage"]
            if result not in self.argument_map[self.num_iterations - 1][stage]:
                mapped_results.append(get_result.fx_node)
            else:
                mapped_results.append(
                    self.argument_map[self.num_iterations - 1][stage][result]
                )
        return mapped_results

    def get_kernel_iteration(self, stage: int) -> int:
        """
        Get the iteration from the stage for the kernel.
        """
        return self.num_stages - 1 - stage

    def get_kernel_results(self) -> list[fx.Node]:
        """
        Gets the mapped results for the kernel. Here there
        exists a fixed relationship between the iteration and stage.
        """
        mapped_results = []
        for result in self.results:
            stage = result.scheduling_parameters["stage"]
            iteration = self.get_kernel_iteration(stage)
            mapped_results.append(self.argument_map[iteration][stage][result])
        return mapped_results

    def __setitem__(self, key: tuple[int, fx.Node], value: fx.Node) -> None:
        """
        Sets the argument mapping for the given stage.
        """
        assert isinstance(key, tuple), "Argument context key must be a tuple"
        iteration, stage, from_ = key
        assert iteration < len(
            self.argument_map
        ), f"Iteration {iteration} not yet initialized"
        assert stage < len(self.argument_map), f"Stage {stage} not yet initialized"
        self.argument_map[iteration][stage][from_] = value

    def __getitem__(self, value: tuple[int, fx.Node]) -> fx.Node:
        """
        Gets the argument mapping for the given stage.
        """
        assert isinstance(value, tuple), "Argument context key must be a tuple"
        iteration, stage, key = value
        assert iteration < len(
            self.argument_map
        ), f"Iteration {iteration} not yet initialized"
        assert stage < len(self.argument_map), f"Stage {stage} not yet initialized"
        return self.argument_map[iteration][stage].get(key, None)

    def __contains__(self, key: fx.Node | tuple[int, fx.Node]) -> bool:
        """
        Checks if the argument context contains the given node at a specified
        iteration and stage or at all iterations and stages.
        """
        if isinstance(key, tuple):
            iteration, stage, key = key
            return key in self.argument_map[iteration][stage]
        return any(
            key in self.argument_map[iteration][stage]
            for iteration in range(self.num_iterations)
            for stage in range(self.num_stages)
        )

    def lookup(self, key: fx.Node) -> Optional[fx.Node]:
        """
        Looks up the argument mapping for the given node.
        """
        for iteration in range(self.num_iterations - 1, -1, -1):
            for stage in range(self.num_stages):
                if key in self.argument_map[iteration][stage]:
                    return self.argument_map[iteration][stage][key]
        return None

    def contains_in_iteration(self, iteration: int, key: fx.Node) -> bool:
        """
        Checks if the argument context contains the given node at a specified
        iteration.
        """
        return any(
            key in self.argument_map[iteration][stage]
            for stage in range(self.num_stages)
        )

    def get_from_iteration(self, iteration: int, key: fx.Node, stage: int) -> fx.Node:
        """
        Gets the argument mapping for the given iteration with
        preference to the given stage.
        """

        if stage and key in self.argument_map[iteration][stage]:
            return self.argument_map[iteration][stage][key]

        for stage in range(self.num_stages):
            if key in self.argument_map[iteration][stage]:
                return self.argument_map[iteration][stage][key]
        return None

    def dump(self):
        """
        Dump the argument context to the logger.
        """
        for iteration in range(self.num_iterations):
            for stage in range(self.num_stages):
                logger.debug(f"Iteration: {iteration}, Stage: {stage}")
                for key, value in self.argument_map[iteration][stage].items():
                    logger.debug(f"  {key} -> {value}")


def create_fill_stage_schedule(n: int) -> list[list[int]]:
    """
    Create the schedule of which stages need to be interleaved for the prologue (fill).
    This looks like:
    [0 None None None]
    [1    0 None None]
    [2    1    0 None]
    """
    schedule = []
    for i in range(n - 1):
        row = list(range(i, -1, -1))
        row.extend([None] * (n - i - 1))
        schedule.append(row)
    return schedule


def create_drain_stage_schedule(n: int) -> list[list[int]]:
    """
    Create the schedule of which stages need to be interleaved for the epilogue (drain).
    This looks like:
    [None    3    2 1]
    [None None    3 2]
    [None None None 3]
    """
    schedule = []
    for i in range(n - 1):
        row = [None] * (i + 1)
        row.extend(range(n - 1, i, -1))
        schedule.append(row)
    return schedule


def compute_lifetime(
    graph: fx.Graph, use_absolute_cycle: bool = False
) -> dict[fx.Node, int]:
    """
    Compute number of clocks each node result needs to be alive.
    """
    lifetime: dict[fx.Node, int] = defaultdict(int)
    name = "absolute_cycle" if use_absolute_cycle else "stage"
    for node in graph.nodes:
        custom = get_custom(node)
        if custom.scheduling_parameters is None:
            continue

        node_stage = custom.scheduling_parameters[name]
        for user in custom.users:
            if user.scheduling_parameters is None:
                continue

            user_stage = user.scheduling_parameters[name]
            user_lifetime = user_stage - node_stage

            logger.debug(
                f"Node: {node}, User: {user.fx_node}, lifetime: {user_lifetime}"
            )
            lifetime[node] = max(user_lifetime, lifetime[node])

    return lifetime


def liveness_analysis(graph: fx.Graph) -> dict[fx.Node, int]:
    """
    Perform liveness analysis on the graph to determine the live ranges of
    variables and use that to deduce how many rotating registers we need.
    """
    lifetime: dict[fx.Node, int] = compute_lifetime(graph, use_absolute_cycle=False)

    # Determine how many copies we need for each node. If the lifetime of a node
    # is l clocks and the initiation interval is T, then only ceil(l/T) values
    # of the node can be live at the same time. We need to create copies of only
    # those nodes that are live at more than one stage.
    num_rotating_registers: dict[fx.Node, int] = {}
    for node, l in lifetime.items():
        if node in num_rotating_registers:
            continue
        custom = get_custom(node)
        if is_shared_write(custom):
            continue

        if isinstance(custom, (GatherToLDS, TensorLoadToLDS)):
            continue

        if l > 0:
            num_rotating_registers[node] = l

    return num_rotating_registers


def compute_multi_buffer_count(
    graph: fx.Graph, initiation_interval: int, multi_buffer_count: Optional[int] = None
) -> dict[fx.Node, int]:
    """
    Compute the number of buffers needed for each node.
    """
    lifetime: dict[fx.Node, int] = compute_lifetime(graph, use_absolute_cycle=True)
    result: dict[fx.Node, int] = defaultdict(int)
    for node in graph.nodes:
        if not isinstance(get_custom(node), (Write, GatherToLDS, TensorLoadToLDS)):
            continue

        shared_memory_operands = get_shared_memory_operands(node)
        for shared_memory_operand in shared_memory_operands:
            shared_memory_operand = propagate_loop_carried_vars(shared_memory_operand)
            if multi_buffer_count:
                result[shared_memory_operand] = multi_buffer_count
                continue

            assert node in lifetime, f"Node {node} not found in lifetime"
            # Lifetime returns 0 if node result only used on same clock, 1 if it used on next clock, etc,
            # so we need to add 1 to the lifetime to get the number of clocks the result is live.
            # Ceildiv is required for cases like (lifetime=3, initiation_interval=2) which would otherwise
            # result in buffer_count=1:
            # 000
            #   111
            #     222
            buffer_count = ceildiv(lifetime[node] + 1, initiation_interval)
            logger.debug(f"Node: {node}, Buffer count: {buffer_count}")
            if buffer_count < 2:
                continue

            result[shared_memory_operand] = max(
                result[shared_memory_operand], buffer_count
            )

    return result


def partition_graph_by_stage(
    graph: fx.Graph, num_stages: int
) -> list[dict[int, list[fx.Node]]]:
    """
    Partition the graph into stages based on the scheduling parameters.
    """
    partitioned_graph: list[dict[int, list[fx.Node]]] = [
        defaultdict(list) for _ in range(num_stages)
    ]
    for stage in range(num_stages):
        for node in graph.nodes:
            custom = get_custom(node)
            if custom.scheduling_parameters is None:
                continue
            if isinstance(custom, IterArg):
                continue
            if custom.scheduling_parameters["stage"] == stage:
                cycle = custom.scheduling_parameters["cycle"]
                partitioned_graph[stage][cycle].append(node)
    return partitioned_graph


def filter_partitioned_graph_by_prefetch(
    partitioned_graph: list[dict[int, list[fx.Node]]],
    min_extra_offset: int,
) -> list[dict[int, list[fx.Node]]]:
    """
    Filter a partitioned graph to only include nodes with
    prefetch_extra_offset >= min_extra_offset in their scheduling parameters.

    This is used to create a filtered version of the partitioned graph
    for extra prefetch iterations in the prologue, where only deeply-
    prefetched nodes should be included.

    Args:
        partitioned_graph: The full partitioned graph (list of stage dicts).
        min_extra_offset: Minimum prefetch_extra_offset value to include.

    Returns:
        A new partitioned graph containing only the filtered nodes.
    """
    filtered = [defaultdict(list) for _ in range(len(partitioned_graph))]
    for stage_idx, stage_dict in enumerate(partitioned_graph):
        for cycle, nodes in stage_dict.items():
            for node in nodes:
                custom = get_custom(node)
                extra_offset = custom.scheduling_parameters.get(
                    "prefetch_extra_offset", 0
                )
                if extra_offset >= min_extra_offset:
                    filtered[stage_idx][cycle].append(node)
    return filtered


def filter_out_deep_prefetch(
    partitioned_graph: list[dict[int, list[fx.Node]]],
) -> list[dict[int, list[fx.Node]]]:
    """
    Create a partitioned graph with deeply-prefetched nodes removed from
    stage 0. All other stages are preserved unchanged.

    This is used in the epilogue drain, where deeply-prefetched stage-0 nodes
    have already completed their work (they were prefetched further ahead by
    the extra prologue and the kernel's shifted indices). Only non-deeply-
    prefetched stage-0 nodes need to be drained.

    Args:
        partitioned_graph: The full partitioned graph (list of stage dicts).

    Returns:
        A new partitioned graph with deeply-prefetched nodes excluded from stage 0.
    """
    filtered = [defaultdict(list) for _ in range(len(partitioned_graph))]
    for stage_idx, stage_dict in enumerate(partitioned_graph):
        for cycle, nodes in stage_dict.items():
            for node in nodes:
                custom = get_custom(node)
                extra_offset = custom.scheduling_parameters.get(
                    "prefetch_extra_offset", 0
                )
                # For stage 0, exclude deeply-prefetched nodes
                if stage_idx == 0 and extra_offset > 0:
                    continue
                filtered[stage_idx][cycle].append(node)
    return filtered


def create_extended_drain_stage_schedule(
    num_stages: int, max_extra_depth: int
) -> list[list[Optional[int]]]:
    """
    Create an extended drain schedule that accounts for deeply-prefetched nodes.

    With max_extra_depth > 0, extra drain iterations are needed because
    non-deeply-prefetched stage-0 nodes haven't completed their last iterations
    (they were only 1 iteration ahead, while the kernel ran fewer iterations
    due to extra prologue fills).

    For num_stages=2, max_extra_depth=1, the schedule is:
        [None, 1, 0]     -> compute at iter 1, non-deep prefetch at iter 2
        [None, None, 1]   -> compute at iter 2

    This is derived from create_drain_stage_schedule(num_stages + max_extra_depth)
    with virtual stage indices remapped to real stage indices:
        virtual_stage -> real_stage = max(0, virtual_stage - max_extra_depth)

    Args:
        num_stages: Number of real pipeline stages (e.g., 2).
        max_extra_depth: Extra depth from deeply-prefetched nodes (e.g., 1).

    Returns:
        Extended drain schedule with (num_stages - 1 + max_extra_depth) rows.
    """
    if max_extra_depth == 0:
        return create_drain_stage_schedule(num_stages)

    effective_stages = num_stages + max_extra_depth
    virtual_schedule = create_drain_stage_schedule(effective_stages)

    # Remap virtual stage indices to real stage indices.
    # Virtual stages [0, max_extra_depth) map to real stage 0.
    # Virtual stages [max_extra_depth, effective_stages) map to
    # real stages [0, num_stages).
    remapped_schedule = []
    for row in virtual_schedule:
        remapped_row = []
        for entry in row:
            if entry is None:
                remapped_row.append(None)
            else:
                real_stage = max(0, entry - max_extra_depth)
                remapped_row.append(real_stage)
        remapped_schedule.append(remapped_row)

    return remapped_schedule


def interleave_instructions(instructions: list[tuple[int, int, fx.Node]]):
    """
    Interleave the instructions that are scheduled in the same cycle.
    Currently, we just randomly shuffle them, but we could also sort
    them based on some criteria.
    """
    rng = random.Random(0)
    # rng.shuffle(instructions)
