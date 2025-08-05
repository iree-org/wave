# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations


from ..._support.indexing import (
    IndexSymbol,
)
from ..._support.tracing import CapturedTrace
from ...compiler.base import CodegenError
from ...lang.global_symbols import SHARED_ADDRESS_SPACE
from ...lang.wave_types import IndexMapping
from ...ops.wave_ops import (
    CustomOp,
    Iterate,
    Read,
    Write,
    get_custom,
)
import copy
from ..utils.mapping_utils import get_dict_with_updated_key
from typing import Optional


def heuristically_multi_buffer_shared_memory(
    new_node: CustomOp,
    induction_variable: IndexSymbol,
    reduction: Iterate,
    multi_buffer_count: Optional[int] = None,
):
    """
    This function adjusts the offset to a shared memory buffer to multibuffer.
    This enables the removal of LDS barriers and enables greater parallelism
    during software pipelining and schedule reordering techniques

    The induction variable used in this function is the induction variable before
    it's remapped during schedule reordering. (See update_node_index in loop_reconstuction.py)

    This enables index to be set to buffer_size * (original_induction_variable % multi_buffer_count)
    Regardless of current node stage, because it will be remapped correctly later depending on if
    prologue, epilogue, or kernel stage
    """
    assert multi_buffer_count is not None
    is_read_write = isinstance(new_node, Read | Write)
    original_buffer = (
        get_custom(new_node.memory) if is_read_write else get_custom(new_node.dst)
    )
    reduction_axis = reduction.axis
    reduction_dim_indices = [
        i for i, dim in enumerate(original_buffer.shape) if dim == reduction_axis
    ]
    for i, dim in enumerate(original_buffer.shape):
        new_node_idx = copy.deepcopy(
            new_node.index if is_read_write else new_node.dst_index
        )
        if i in reduction_dim_indices or dim not in new_node_idx:
            continue
        block_size = original_buffer.distributed_shape[i]
        which_buffer = induction_variable % multi_buffer_count
        offset = block_size * which_buffer
        new_node_idx[dim].start += offset

        # Update the mapping for the operation as the keys for the
        # mapping have to match the shape of memory location the

        # operation reads from / writes to, which we change below.
        new_node_idx = get_dict_with_updated_key(
            new_node_idx, dim, dim * multi_buffer_count
        )
        if not is_read_write:
            new_node.dst_index = new_node_idx
            return
        new_node.index = new_node_idx

        if isinstance(new_node.mapping, IndexMapping):
            input_mapping = new_node.mapping.input_mapping
            output_mapping = new_node.mapping.output_mapping
            if dim in input_mapping:
                new_node.mapping.input_mapping = get_dict_with_updated_key(
                    input_mapping, dim, dim * multi_buffer_count
                )
            if dim in output_mapping:
                new_node.mapping.output_mapping = get_dict_with_updated_key(
                    output_mapping, dim, dim * multi_buffer_count
                )

        return


def multi_buffer(trace: CapturedTrace, multi_buffer_count: Optional[int] = None):
    """Perform multi buffering for all supported shared memory locations"""

    # Find all reductions
    reductions = trace.walk(lambda node: isinstance(get_custom(node), Iterate))

    # Get reduction dimension from first reduction
    if not reductions or len(reductions) != 1:
        raise CodegenError(
            f"Unexpected number of reductions found in graph: {len(reductions)} vs 1"
        )

    reduction_axis = get_custom(reductions[0]).axis

    # Find reads and writes operating on shared memory
    reads_and_writes = []
    for node in trace.get_subgraph(get_custom(reductions[0]).subgraph_name).nodes:
        custom = get_custom(node)
        if (
            isinstance(custom, Read | Write)
            and custom.memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            reads_and_writes.append(custom)

    # Partition reads and writes by memory location
    memory_to_shared_memory_op = _partition_by_memory(reads_and_writes)
    # Perform multi buffering for all collected memory locations
    for memory_location in set(memory_to_shared_memory_op):
        _increase_dimensions_for_multibuffering(
            original_buffer=memory_location,
            reduction_axis=reduction_axis,
            buffer_count=multi_buffer_count,
        )


def _increase_dimensions_for_multibuffering(
    original_buffer: CustomOp, reduction_axis: IndexSymbol, buffer_count: int
):
    # Create new shape with increased non-reduction dimensions
    reduction_dim_indices = [
        i for i, dim in enumerate(original_buffer.shape) if dim == reduction_axis
    ]
    new_shape = []
    new_distributed_shape = []
    increased_multi_buffer_dim = False

    for i, dim in enumerate(original_buffer.shape):
        if increased_multi_buffer_dim or i in reduction_dim_indices:
            # Keep reduction dimensions as is
            new_shape.append(dim)
            new_distributed_shape.append(original_buffer.distributed_shape[i])
        else:
            # Increase only the multibuffered first non-reduction dimension
            increased_multi_buffer_dim = True
            new_shape.append(dim * buffer_count)
            new_distributed_shape.append(
                original_buffer.distributed_shape[i] * buffer_count
            )

    original_buffer.update_arg(0, tuple(new_shape))
    original_buffer.update_arg(1, tuple(new_distributed_shape))


def _partition_by_memory(nodes: list[CustomOp]) -> dict[CustomOp, list[CustomOp]]:
    """
    Partitions reads or writes by their source memory location.
    Returns a dict mapping memory nodes to lists of read or write operations from that memory.
    """
    memory_mapping: dict[CustomOp, list[CustomOp]] = {}

    for node in nodes:
        memory_node = get_custom(node.memory)

        if memory_node not in memory_mapping:
            memory_mapping[memory_node] = []

        memory_mapping[memory_node].append(node)

    return memory_mapping
