# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import torch.fx as fx
import sympy

import wave_lang.kernel.lang as tkl
from wave_lang.support.logging import get_logger

from ..._support.tracing import CapturedTrace
from ...ops.wave_ops import (
    Allocate,
    Conditional,
    CustomOp,
    Ge,
    GetResult,
    IterArg,
    Iterate,
    NewScalar,
    Output,
    get_custom,
)
from ..constraints import Constraint
from ..utils.general_utils import (
    get_tiling_constraint,
)
from ..utils.graph_utils import (
    erase_graph,
    get_graph_node,
    graph_copy,
    prepare_subgraph_for_conditional,
    update_sort_keys,
)

# Import moved to function level to avoid circular import
from ..utils.symbol_utils import (
    subs_idxc,
)
from ..visualization import visualize_edges, visualize_graph, visualize_schedule
from .graph_utils import Edge, create_scheduling_edges
from .loop_reconstruction import construct_pipelined_loop
from .modulo_scheduling import ModuloScheduler
from .prefetch_scheduling import PrefetchScheduler, PrefetchAttentionScheduler
from .four_stage_pipelined_scheduling import FourStageScheduler
from .resources import (
    annotate_resource_usage,
    get_available_resources,
    get_resource_names,
)
from .schedule_enums import SchedulingType
from typing import Optional

logger = get_logger("wave.scheduling.schedule")


def is_solver_based(scheduling_type: SchedulingType):
    scheduling_ty_val = scheduling_type.value
    return scheduling_ty_val >= 0x10 and scheduling_ty_val < 0x20


def visualize_scheduling_graph(edges: list[Edge]):
    visualize_edges(edges, "reduction_graph.png")


def schedule_reduction(
    reduction: Iterate,
    trace: CapturedTrace,
    constraints: list[Constraint],
    use_scheduling_barriers: bool = False,
    scheduling_type: SchedulingType = SchedulingType.NONE,
    override_schedule_file: str = None,
    dump_schedule_file: Optional[str] = None,
    multi_buffer_count: Optional[int] = None,
):
    """
    Clones the reduction graph and does the following:
    1. Annotates resource usage for each node.
    2. Creates edges between outputs and return args for scheduling
       and assigns weights to all edges.
    Does scheduling on the cloned graph and applies the schedule
    to the original graph. Finally, erases the cloned graph.

    Args:
        override_schedule_file: If provided, load schedule from this file instead of computing it
        dump_schedule_file: If provided, dump the computed schedule to this file
    """
    if scheduling_type == SchedulingType.NONE:
        return {}
    reduction_graph = trace.get_subgraph(reduction.subgraph_name)
    graph, node_map = graph_copy(reduction_graph)
    ignore_nodes, iter_args, output = annotate_resource_usage(graph)
    edges = create_scheduling_edges(graph, ignore_nodes, iter_args, output)

    update_sort_keys(trace, graph)

    if override_schedule_file:
        if not os.path.exists(override_schedule_file):
            raise ValueError(
                f"Override schedule file {override_schedule_file} does not exist"
            )
        # Import here to avoid circular import
        from ..utils.print_utils import load_schedule

        (
            schedule,
            initiation_interval,
            num_stages,
            _,
            edges,
            _,
            _,
        ) = load_schedule(override_schedule_file, graph)
        success = True
    else:
        if is_solver_based(scheduling_type):
            scheduler = ModuloScheduler(graph, edges, get_available_resources())
        elif scheduling_type == SchedulingType.PREFETCH:
            scheduler = PrefetchScheduler(graph, edges, get_available_resources())
        elif scheduling_type == SchedulingType.PREFETCH_ATTENTION:
            scheduler = PrefetchAttentionScheduler(
                graph, edges, get_available_resources()
            )
        elif scheduling_type == SchedulingType.FOUR_STAGE:
            scheduler = FourStageScheduler(graph, edges, get_available_resources())
        else:
            raise ValueError("Unknown scheduling type")

        schedule, success = scheduler.schedule_graph()
        if not success:
            raise ValueError("Scheduling failed.")

        initiation_interval = scheduler.initiation_interval
        num_stages = scheduler.num_stages

        if dump_schedule_file:
            # Import here to avoid circular import
            from ..utils.print_utils import dump_schedule

            dump_schedule(
                schedule,
                initiation_interval,
                num_stages,
                dump_schedule_file,
                scheduler.resource_reservations,
                get_resource_names(),
            )

    visualize = False
    if visualize:
        if is_solver_based(scheduling_type):
            visualize_scheduling_graph(edges)
            visualize_graph(graph, "scheduling_fx_graph.png")
        visualize_schedule(schedule, initiation_interval, "schedule.html")

    # Apply schedule to original graph, specifying the stage
    # that each node is scheduled in as well as the cycle in
    # each stage when the node should be issued.
    inverse_node_map = {v: k for k, v in node_map.items()}
    iter_args: list[CustomOp] = []
    for node, cycle in schedule.items():
        if node not in inverse_node_map:
            continue
        custom = get_custom(inverse_node_map[node])
        custom.scheduling_parameters = {
            "absolute_cycle": cycle,
            "cycle": cycle % initiation_interval,
            "stage": cycle // initiation_interval,
            "initiation_interval": initiation_interval,
            "prefetch_stage": node.meta.get("prefetch_stage", None),
        }
        # Erase edges between outputs and iter args.
        if isinstance(get_custom(node), IterArg):
            node.args = ()
            iter_args.append(custom)

    for custom in iter_args:
        cycle = min(x.scheduling_parameters["absolute_cycle"] for x in custom.users)
        custom.scheduling_parameters = {
            "absolute_cycle": cycle,
            "cycle": cycle % initiation_interval,
            "stage": cycle // initiation_interval,
            "initiation_interval": initiation_interval,
            "prefetch_stage": custom.fx_node.meta.get("prefetch_stage", None),
        }

    erase_graph(graph)
    apply_pipelined_schedule(
        reduction,
        reduction_graph,
        trace,
        constraints,
        use_scheduling_barriers,
        num_stages,
        initiation_interval,
        scheduling_type,
        visualize,
        multi_buffer_count,
    )


def build_guarded_pipeline_with_remainder(
    trace: CapturedTrace,
    reduction: Iterate,
    reduction_graph: fx.Graph,
    constraints: list[Constraint],
    num_stages: int,
    initiation_interval: int,
    max_induction_variable,
    visualize: bool = False,
    use_scheduling_barriers: bool = False,
    multi_buffer_count: Optional[int] = None,
    max_extra_depth: int = 0,
):
    """
    Build conditional + pipelined loop + remainder loop for dynamic shapes.

    Structure:
        if (max_induction_variable >= num_stages):
            pipelined_result = pipelined_loop_with_prologue_epilogue()
        else:
            pipelined_result = init_values
        final_result = remainder_loop(init=pipelined_result)

    The conditional ensures prologue/epilogue only run when there are enough iterations.
    """
    # Save properties before any transformations
    original_index = reduction.index
    original_init_args = reduction.init_args
    main_graph = reduction.graph

    # Create condition: max_induction_variable >= num_stages
    with main_graph.inserting_before(reduction.fx_node):
        num_stages_scalar = get_graph_node(
            NewScalar(num_stages, tkl.i32), main_graph, reduction.location
        )
        num_iters_scalar = get_graph_node(
            NewScalar(max_induction_variable, tkl.i32), main_graph, reduction.location
        )
        condition = get_graph_node(
            Ge(num_iters_scalar, num_stages_scalar), main_graph, reduction.location
        )

    # Prepare conditional subgraph
    captured_nodes = list(reduction.init_args) + list(reduction.implicit_captures)
    memory_nodes = [
        node for node in captured_nodes if isinstance(get_custom(node), Allocate)
    ]

    subgraph_name = f"pipelined_conditional_{reduction.fx_node.name}"
    conditional_subgraph, implicit_captures, placeholders = (
        prepare_subgraph_for_conditional(
            subgraph_name, captured_nodes, memory_nodes=memory_nodes
        )
    )

    # Compute the number of iterations the pipelined loop should process:
    # This ensures the pipelined loop only processes complete pipeline stages
    if isinstance(max_induction_variable, (int, float)):
        pipelined_iterations = (int(max_induction_variable) // num_stages) * num_stages
    else:
        pipelined_iterations = (max_induction_variable // num_stages) * num_stages

    conditional_body_graph, _ = graph_copy(reduction_graph)
    placeholder_init_args = [placeholders[arg] for arg in reduction.init_args]
    placeholder_captures = [placeholders[cap] for cap in reduction.implicit_captures]

    # Register the body graph in the conditional subgraph
    temp_body_name = f"{reduction.subgraph_name}_cond"
    if not hasattr(conditional_subgraph, "subgraphs"):
        conditional_subgraph.subgraphs = {}
    conditional_subgraph.subgraphs[temp_body_name] = conditional_body_graph

    # Create temporary reduction in conditional subgraph
    temp_reduction = Iterate(
        reduction.axis,
        init_args=placeholder_init_args,
        step=reduction.step,
        subgraph_name=temp_body_name,
        implicit_captures=placeholder_captures,
    ).add_to_graph(conditional_subgraph, type=reduction.type, loc=reduction.location)

    conditional_body_graph.parent_op = temp_reduction
    get_custom(temp_reduction).index = original_index
    get_custom(temp_reduction).count = pipelined_iterations

    # Create GetResult nodes for the temp reduction
    # These are required by construct_pipelined_loop's epilogue construction,
    # which expects to find GetResult nodes as users of the pipelined reduction
    for i in range(len(placeholder_init_args)):
        GetResult(temp_reduction, i).add_to_graph(
            conditional_subgraph,
            type=(
                placeholder_init_args[i].type
                if hasattr(placeholder_init_args[i], "type")
                else None
            ),
            loc=reduction.location,
        )

    pipelined_node, node_mapping, final_results = construct_pipelined_loop(
        trace,
        get_custom(temp_reduction),
        conditional_body_graph,
        constraints,
        num_stages,
        initiation_interval,
        pipelined_iterations,  # Pass the actual number of iterations to process
        visualize,
        use_scheduling_barriers,
        multi_buffer_count,
        max_extra_depth,
    )

    # Set the count for the pipelined loop
    # With step > 1 (e.g., from unrolling), we need to reduce the count by more
    # to prevent out-of-bounds access. The last kernel iteration's stage 0 loads
    # data for the "next" iteration (offset by step), so we need to ensure
    # that stays within bounds.
    step = get_custom(pipelined_node).step
    # Reduce loop count by max_extra_depth: the extra prologue fills
    # consume additional iterations, so the kernel runs fewer.
    get_custom(pipelined_node).count = (
        pipelined_iterations - (num_stages - 1 + max_extra_depth) * step
    )

    # Verify we have the right number of results
    assert len(final_results) == len(
        original_init_args
    ), f"Expected {len(original_init_args)} results but found {len(final_results)}"

    # Add Output to the conditional subgraph using the final results from the epilogue
    Output(final_results).add_to_graph(conditional_subgraph, loc=reduction.location)

    # Create the Conditional node in the main graph
    # Register the conditional subgraph
    if not hasattr(main_graph, "subgraphs"):
        main_graph.subgraphs = {}
    main_graph.subgraphs[subgraph_name] = conditional_subgraph
    trace.region_graph.subgraphs[subgraph_name] = conditional_subgraph

    with main_graph.inserting_before(reduction.fx_node):
        conditional_op = Conditional(
            condition,
            subgraph_name=subgraph_name,
            implicit_captures=implicit_captures,
            else_return=reduction.init_args,  # Else: return init values unchanged
        ).add_to_graph(main_graph, type=reduction.type, loc=reduction.location)

    conditional_subgraph.parent_op = conditional_op

    # Extract results from conditional
    conditional_results = []
    with main_graph.inserting_before(reduction.fx_node):
        for i in range(len(original_init_args)):
            result = GetResult(conditional_op, i).add_to_graph(
                main_graph,
                type=(
                    original_init_args[i].type
                    if hasattr(original_init_args[i], "type")
                    else None
                ),
                loc=reduction.location,
            )
            conditional_results.append(result)

    # Create remainder loop using a copy of the original body
    # The remainder loop should start where the pipelined loop ended
    remainder_graph, _ = graph_copy(reduction_graph)
    remainder_subgraph_name = f"{reduction.subgraph_name}_remainder"

    main_graph.subgraphs[remainder_subgraph_name] = remainder_graph
    trace.region_graph.subgraphs[remainder_subgraph_name] = remainder_graph

    # Create a scalar node for the starting iteration (where pipelined loop ended)
    # This will be pipelined_iterations
    with main_graph.inserting_before(reduction.fx_node):
        start_iter = get_graph_node(
            NewScalar(pipelined_iterations, tkl.index), main_graph, reduction.location
        )

        remainder_reduction = Iterate(
            reduction.axis,
            init_args=conditional_results,
            step=reduction.step,
            subgraph_name=remainder_subgraph_name,
            implicit_captures=reduction.implicit_captures,
            start=start_iter,  # Start where pipelined loop ended
        ).add_to_graph(main_graph, type=reduction.type, loc=reduction.location)

    remainder_graph.parent_op = remainder_reduction
    # Set the index and count for the remainder loop
    # The start parameter already handles the offset, so index uses the original
    get_custom(remainder_reduction).index = original_index
    get_custom(remainder_reduction).count = max_induction_variable

    # Replace all uses of the original reduction with the remainder loop
    # This will update all GetResult nodes automatically
    reduction.replace_all_uses_with(get_custom(remainder_reduction))

    # Erase the original reduction (now safe since it has no users)
    reduction.erase()

    # Return the pipelined node and node mapping so manual schedules can reference it
    return pipelined_node, node_mapping


def construct_pipelined_loop_adaptive(
    trace: CapturedTrace,
    reduction: Iterate,
    reduction_graph: fx.Graph,
    constraints: list[Constraint],
    num_stages: int,
    initiation_interval: int,
    max_induction_variable,
    visualize: bool = False,
    use_scheduling_barriers: bool = False,
    multi_buffer_count: Optional[int] = None,
    max_extra_depth: int = 0,
):
    """
    Constructs a pipelined loop wrapped in a conditional, followed by a remainder loop.

    Structure:
        if (num_iterations >= num_stages):
            prologue
            pipelined_loop
            epilogue
            return (iterations_done, result_values...)
        else:
            return (0, init_values...)
        remainder_loop(start=iterations_done, end=total_iterations)
    """
    # Check if we have a dynamic shape (max_induction_variable is symbolic)
    is_dynamic = not (
        isinstance(max_induction_variable, (int, float))
        or (
            hasattr(max_induction_variable, "is_number")
            and max_induction_variable.is_number
        )
    )

    if not is_dynamic:
        # For static shapes, use the old implementation
        new_reduction, node_mapping, _ = construct_pipelined_loop(
            trace,
            reduction,
            reduction_graph,
            constraints,
            num_stages,
            initiation_interval,
            int(max_induction_variable),
            visualize,
            use_scheduling_barriers,
            multi_buffer_count,
            max_extra_depth,
        )
        if new_reduction:
            step = get_custom(new_reduction).step
            # Reduce loop count by max_extra_depth: the extra prologue fills
            # consume additional iterations, so the kernel runs fewer.
            get_custom(new_reduction).count = (
                max_induction_variable - (num_stages - 1 + max_extra_depth) * step
            )
        return new_reduction, node_mapping

    # For dynamic shapes, emit conditional + pipelined loop + remainder loop
    # Call helper function to build the conditional structure
    return build_guarded_pipeline_with_remainder(
        trace,
        reduction,
        reduction_graph,
        constraints,
        num_stages,
        initiation_interval,
        max_induction_variable,
        visualize,
        use_scheduling_barriers,
        multi_buffer_count,
        max_extra_depth,
    )


def apply_pipelined_schedule(
    reduction: Iterate,
    reduction_graph: fx.Graph,
    trace: CapturedTrace,
    constraints: list[Constraint],
    use_scheduling_barriers: bool = False,
    num_stages: int = None,
    initiation_interval: int = None,
    scheduling_type: SchedulingType = SchedulingType.NONE,
    visualize: bool = False,
    multi_buffer_count: Optional[int] = None,
    max_extra_depth: int = 0,
) -> Optional[tuple[fx.Node, dict]]:

    # After scheduling has completed, we have enough information to decide
    # whether to pipeline the loop. For pipelining to be possible, we need
    # to have atleast N iterations of the loop where N > num_stages - 1 (because
    # we will be peeling off num_stages iterations from the loop).
    tiling_constraint = get_tiling_constraint(reduction, constraints)
    max_induction_variable = subs_idxc(tiling_constraint.count)

    # Try to evaluate the expression to a concrete number if possible
    if isinstance(max_induction_variable, sympy.Basic):
        # First simplify
        max_induction_variable = max_induction_variable.simplify()
        # If it's now a number, convert to int
        if max_induction_variable.is_number:
            max_induction_variable = int(max_induction_variable)

    # For dynamic shapes, we emit a conditional that checks at runtime
    # whether we have enough iterations to pipeline, and a remainder loop
    # to handle any leftover iterations.
    # For static shapes, the same structure is emitted and later compiler
    # passes will optimize away the conditional or remainder loop as needed.

    return construct_pipelined_loop_adaptive(
        trace,
        reduction,
        reduction_graph,
        constraints,
        num_stages,
        initiation_interval,
        max_induction_variable,
        visualize,
        use_scheduling_barriers,
        multi_buffer_count,
        max_extra_depth,
    )


def propagate_scheduling_parameters_to_iter_args(
    graph: fx.Graph, initiation_interval: int
):
    """
    Sets the scheduling parameters for the iter args based on the users.
    """
    iter_args = []
    for node in graph.nodes:
        custom = get_custom(node)
        if isinstance(custom, IterArg):
            iter_args.append(custom)

    for custom in iter_args:
        cycle = min(x.scheduling_parameters["absolute_cycle"] for x in custom.users)
        custom.scheduling_parameters = {
            "absolute_cycle": cycle,
            "cycle": cycle % initiation_interval,
            "stage": cycle // initiation_interval,
            "initiation_interval": initiation_interval,
            "prefetch_stage": None,
        }


def schedule_graph(
    trace: CapturedTrace,
    constraints: list[Constraint],
    use_scheduling_barriers: bool = False,
    scheduling_type: SchedulingType = SchedulingType.NONE,
    override_schedule: str = None,
    dump_schedule: str = None,
    multi_buffer_count: Optional[int] = None,
):
    """
    Given a graph, pipelines the reductions in the graph.

    Args:
        override_schedule: If provided, load schedule from this file instead of computing it
        dump_schedule: If provided, dump the computed schedule to this file
    """
    if scheduling_type == SchedulingType.NONE:
        return

    def is_reduction(node: fx.Node) -> bool:
        return isinstance(get_custom(node), Iterate)

    reduction_nodes = trace.walk(is_reduction)
    if not reduction_nodes:
        return

    for reduction_node in reduction_nodes:
        schedule_reduction(
            get_custom(reduction_node),
            trace,
            constraints,
            use_scheduling_barriers,
            scheduling_type,
            override_schedule,
            dump_schedule,
            multi_buffer_count,
        )
