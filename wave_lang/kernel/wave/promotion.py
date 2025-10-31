# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

from wave_lang.support.logging import get_logger

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import *
from .constraints import Constraint, get_constrained_shape
from .utils.classes import KernelLaunchInfo
from .utils.graph_utils import move_node_after
from .utils.symbol_utils import subs_idxc

logger = get_logger("wave.promotion")


def apply_padding(
    shape: tuple[IndexSymbol | int], dtype: DataType
) -> tuple[int, tuple[IndexSymbol | int]]:
    """
    When accessing shared memory, we need to be cognizant of bank conflicts
    that can have a significant impact on performance. One way to mitigate
    these conflicts is by applying padding to the shared memory allocation.
    This function applies padding of 64 bits to the shared memory allocation.
    While this approach accomplishes the goal of reducing bank conflicts, it
    is inefficient in terms of memory usage. A more sophisticated approach
    would involve swizzling of the shared memory access patterns.
    """
    padding = 64 // dtype.bitwidth()
    return padding, tuple(
        value + padding if i == len(shape) - 1 else value
        for i, value in enumerate(shape)
    )


def apply_promotion_pattern(
    custom_node: Read | Write,
    allocate_node: Allocate,
    last_write_to_shared: fx.Node,
    reorder_allocs: bool = True,
):
    """
    Decompose and reorders read_from_global -> write_to_shared -> read_from_shared sequence.
    In the previous naive way, the generated instruction ordering used to look like:
    ```
    read_from_global lhs
    write_to_shared lhs
    shared_barrier
    read_from_shared lhs
    read_from_global rhs
    write_to_shared rhs
    shared_barrier
    read_from_shared rhs
    ```
    For this simple example, we have 2 shared barriers.

    Currently, this pass keep track of the last_write_to_shared, S.T
    read_from_global -> write_to_shared -> read_from_shared sequence
    can be inserted after the `last_write_to_shared` and before the
    last read from shared. This ensures that we isolate all the
    read_from_shared to be located one after another. This allows
    us to only need 1 shared barrier as seen below:
    ```
    read_from_global lhs
    write_to_shared lhs
    read_from_global rhs
    write_to_shared rhs
    shared_barrier
    read_from_shared lhs
    read_from_shared rhs
    ```
    """
    match custom_node:
        case Read(memory, elements_per_thread) if (
            get_custom(memory).type.address_space != allocate_node.address_space
        ):
            # Moves memory to top of graph after allocate to avoid non-dominating operands.
            move_node_after(custom_node.memory, allocate_node.fx_node)
            # We move CustomOp/Read up to the last write_to_shared_mem S.T
            # all reads from shared mem happens only after all read from globals
            # and write to shared mem happen. Which will minimize lds_barrier count.
            # If we have multiple nested region ops in the graph, then the last_write_to_shared
            # can be in a different graph, so we add a check to ensure that the last_write_to_shared
            # is always in the same graph as the custom_node.
            # While this minimizes lds_barrier count, it increases the live ranges of
            # the allocated memory and prevents reuse of the allocated memory. So we only
            # move the custom_node if minimize shared memory allocation pass is disabled.
            if reorder_allocs:
                if (
                    last_write_to_shared
                    and last_write_to_shared.graph == custom_node.graph
                ):
                    moved_src = move_node_after(
                        custom_node.fx_node, last_write_to_shared
                    )
                    custom_node = get_custom(moved_src)
            with custom_node.graph.inserting_after(custom_node.fx_node):
                promoted_read = Read(
                    allocate_node.fx_node, elements_per_thread
                ).add_to_graph(
                    custom_node.graph, loc=custom_node.location, tag=custom_node.tag
                )
            custom_node.replace_all_uses_with(promoted_read)
            with custom_node.graph.inserting_before(promoted_read):
                promoted_write = Write(
                    custom_node.fx_node, allocate_node.fx_node, elements_per_thread
                ).add_to_graph(
                    custom_node.graph, loc=custom_node.location, tag=custom_node.tag
                )
                custom_read = get_custom(promoted_read)
                custom_read.write_dependency = [promoted_write]
            custom_node.memory_type.address_space = GLOBAL_ADDRESS_SPACE
            last_write_to_shared = promoted_write
            return last_write_to_shared


def fix_manual_allocate_dependencies(trace: CapturedTrace):
    """
    Fix write dependencies for user-created manual allocations.
    When users manually allocate shared memory and write to it, the subsequent
    reads need write dependencies set to prevent DCE from removing the writes.
    """
    root_graph = trace.get_root_graph()
    for subgraph in root_graph.subgraphs.values():
        node_list = list(subgraph.nodes)

        # Group writes by which allocate they target
        writes_by_allocate = {}
        for node in node_list:
            custom = get_custom(node)

            # look for the write nodes that target a placeholder
            if not isinstance(custom, Write):
                continue

            memory_custom = get_custom(custom.memory)
            if not isinstance(memory_custom, Placeholder):
                continue

            # ensure that the placeholder is an allocate
            # can't differentiate between compiler generated allocate and user-created allocate
            captured_node = memory_custom.get_captured_fx_node()
            if not (captured_node and isinstance(get_custom(captured_node), Allocate)):
                continue

            # map this write to the allocate
            # can we have multiple writes to the same memory (?)
            writes_by_allocate.setdefault(custom.memory, []).append(node)

        # set write dependencies on reads from those allocates
        for read_idx, node in enumerate(node_list):
            custom = get_custom(node)
            if not isinstance(custom, Read):
                continue
            if custom._write_dependency is not None:
                continue
            if custom.memory not in writes_by_allocate:
                continue

            # Find writes that came before this read to the same allocate
            # the we read from. In case of multiple writes, return the writes that occur before the read
            writes_before = [
                w
                for w in writes_by_allocate[custom.memory]
                if node_list.index(w) < read_idx
            ]
            if writes_before:
                custom.update_arg("_write_dependency", writes_before)
                logger.debug(f"Set write dependency for {node} to {writes_before}")


def promote_node(
    node: Read | Write,
    last_write_to_shared: fx.Node,
    address_space: IndexSymbol,
    constraints: list[Constraint],
    reorder_allocs: bool = True,
):
    """Promotes the given operand in the provided graph
    to the specified address space.

    The process of promotion involves allocating memory
    in the new address space and writing to the new
    memory location and subsequent uses reading from there.
    """

    assert isinstance(node, Read) or isinstance(node, Write)
    # If the read is a gather, then we should use the memory type instead of the
    # type, when determining the shape of the promoted memory.
    symbolic_shape = node.type.symbolic_shape
    with node.graph.inserting_after(node.graph._root):
        constrained_shape = get_constrained_shape(symbolic_shape, constraints)
        # If the read/write operation already has a set distributed shape at the kernel
        # we use that for allocation. Otherwise deduce the shape from constraints.
        memory_node = get_custom(node.memory)
        if isinstance(memory_node, Allocate) and memory_node.distributed_shape:
            constrained_shape = memory_node.distributed_shape
        padding, padded_shape = apply_padding(constrained_shape, node.type.dtype)
        allocate_node = Allocate(
            symbolic_shape, padded_shape, node.type.dtype, address_space, padding
        )
        allocate_node.add_to_graph(node.graph, loc=node.location)
    last_write_to_shared = apply_promotion_pattern(
        node, allocate_node, last_write_to_shared, reorder_allocs
    )
    return last_write_to_shared


def promote_placeholders(
    graph: CapturedTrace,
    constraints: list[Constraint],
    reorder_allocs: bool = True,
):
    read_or_write_nodes = graph.walk(
        lambda node: isinstance(get_custom(node), Read)
        or isinstance(get_custom(node), Write)
    )
    last_write_to_shared = None
    for node in read_or_write_nodes:
        custom = get_custom(node)
        if not custom.memory_type:
            continue
        address_space = subs_idxc(custom.memory_type.address_space)
        if address_space == SHARED_ADDRESS_SPACE:
            last_write_to_shared = promote_node(
                custom,
                last_write_to_shared,
                address_space,
                constraints,
                reorder_allocs,
            )

    # Fix write dependencies for user-created allocations
    # When users manually allocate shared memory, the reads from those allocations
    # need write dependencies set to prevent DCE from removing the writes
    fix_manual_allocate_dependencies(graph)


def compute_shared_memory_usage(
    graph: CapturedTrace, kernel_launch_info: KernelLaunchInfo
):
    """
    Compute the amount of shared memory used in bytes by iterating over all allocate
    nodes and summing up their distributed shapes.
    """
    is_allocate = lambda x: isinstance(get_custom(x), Allocate)
    for alloc in graph.walk(is_allocate):
        custom_alloc = get_custom(alloc)
        # Ignore allocations that are slices of a larger allocation.
        if custom_alloc.parent is not None:
            continue
        shape = subs_idxc(math.prod(custom_alloc.distributed_shape))
        bits = custom_alloc.type.dtype.bitwidth()
        kernel_launch_info.shared_memory_bytes += int((shape * bits) // 8)
