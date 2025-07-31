# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import Read, Write, get_custom
from ..lang.wave_types import IndexMapping, index_symbol, IndexSymbol
from ..wave.constraints import Constraint, IteratorBindings


def construct_index_mapping(
    trace: CapturedTrace, constraints: list[Constraint]
) -> None:
    """
    Walk the trace and find all read and write nodes that have a source and target specified.
    For those, create an IndexMapping and update the read/write node so that the mapping is set
    and update the type.
    """
    nodes = trace.walk(lambda x: isinstance(get_custom(x), (Read, Write)))
    iterator_bindings = [c for c in constraints if isinstance(c, IteratorBindings)]
    assert (
        len(iterator_bindings) <= 1
    ), " Cannot have more than one IteratorBindings constraint"
    for node in nodes:
        custom_op = get_custom(node)
        if custom_op.source is not None and custom_op.target is not None:
            mapping = _create_index_mapping_for_node(
                custom_op, iterator_bindings[0].bindings
            )
            custom_op.update_arg("mapping", mapping)
            if isinstance(custom_op, Read):
                custom_op.type.symbolic_shape = [*mapping.output_shape]
            else:
                custom_op.type.symbolic_shape = [*mapping.input_shape]


def _create_index_mapping_for_node(
    node: Read | Write, bindings: dict[IndexSymbol, IndexSymbol]
) -> IndexMapping:
    """
    Create an IndexMapping for a Read or Write node based on its source and target.

    Args:
        node: The Read or Write node with source and target specified

    Returns:
        IndexMapping object representing the mapping between source and target
    """
    # Get the source and target expressions
    source_exprs = node.source
    target_exprs = node.target

    if source_exprs is None or target_exprs is None:
        raise ValueError("Node must have both source and target specified")

    # For reads, we have an output identity mapping, for writes, we have an input identity mapping.
    num_iterators = len(target_exprs) if isinstance(node, Read) else len(source_exprs)
    iterators = [index_symbol(f"$index{i}") for i in range(num_iterators)]

    if isinstance(node, Read):
        output_mapping, iterator_map = _create_mapping_using_bindings(
            node, target_exprs, iterators, bindings
        )
        input_mapping = _create_mapping_using_memory_type(
            node, source_exprs, iterator_map
        )
    else:
        input_mapping, iterator_map = _create_mapping_using_bindings(
            node, source_exprs, iterators, bindings
        )
        output_mapping = _create_mapping_using_memory_type(
            node, target_exprs, iterator_map
        )

    return IndexMapping(
        num_iterators=num_iterators, inputs=input_mapping, outputs=output_mapping
    )


def _create_mapping_using_memory_type(
    node: Read | Write,
    source_exprs: tuple,
    iterator_map: dict[IndexSymbol, IndexSymbol],
) -> dict:
    """
    Create input mapping for a Read/Write node using the memory type.
    """
    # Get memory shape from the memory operand
    memory_type = node.memory_type
    memory_shape = memory_type.symbolic_shape

    mapping = {}
    for mem_dim, source_expr in zip(memory_shape, source_exprs):
        mapping[mem_dim] = source_expr.subs(iterator_map)
    return mapping


def _create_mapping_using_bindings(
    node: Read | Write,
    target_exprs: tuple,
    iterators: list[IndexSymbol],
    bindings: dict[IndexSymbol, IndexSymbol],
) -> dict:
    """
    Create output mapping for a Read/Write node using the current bindings.
    """
    mapping = {}
    iterator_map = {}
    for i, target_expr in enumerate(target_exprs):
        assert target_expr in bindings
        mapping[bindings[target_expr]] = iterators[i]
        iterator_map[target_expr] = iterators[i]

    return mapping, iterator_map
