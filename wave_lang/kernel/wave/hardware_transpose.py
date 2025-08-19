# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import copy
import sympy
import torch.fx as fx

from .minimize_global_loads import is_transposed_read, materialize_shape
from .utils.general_utils import get_hardware_constraint
from .utils.symbol_utils import safe_subs

from .global_to_shared_gathers import update_read_mapping_dynamic_values
from ..lang.global_symbols import THREAD_0, SHARED_ADDRESS_SPACE
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import Read, Reshape, get_custom
from ..wave.constraints import (
    Constraint,
    TilingConstraint,
    WorkgroupConstraint,
)
from ..wave.utils.run_utils import get_default_arch


def meets_hw_transpose_requirements(read: Read, constraints: list[Constraint]):
    if not get_default_arch() == "gfx950":
        return False

    if read.has_identity_mapping():
        return False

    if len(list(read.index.keys())) != 2:
        return False

    bitwidth = read.type.dtype.bitwidth()
    if bitwidth != 8 and bitwidth != 16:
        return False

    bits = read.elements_per_thread * bitwidth
    if bits == 0 or bits % 64 != 0:
        return False

    if read.memory_type.address_space != SHARED_ADDRESS_SPACE:
        return False

    if read.mapping_dynamic_vals:
        return False

    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, (TilingConstraint, WorkgroupConstraint))
    }

    materialized_shape = materialize_shape(
        constraint_tile_size, read.type.symbolic_shape, read.vector_shapes
    )

    if any(s > 1 for s in materialized_shape[:-2]) or any(
        s <= 1 for s in materialized_shape[-2:]
    ):
        return False

    hardware_constraint = get_hardware_constraint(constraints)
    return hardware_constraint.threads_per_wave >= 16


def fetch_delinearized_indices(shape, dtype_width, thread_id):
    if shape == (16, 32) and dtype_width == 8:
        return [
            (thread_id % 2) * 8 + (sympy.floor(thread_id / 16) % 2) * 16,
            sympy.floor((thread_id % 64) / 32) * 8 + sympy.floor((thread_id % 16) / 2),
        ]

    if shape == (32, 16) and dtype_width == 8:
        return [
            (thread_id % 2) * 8,
            sympy.floor((thread_id % 64) / 2),
        ]

    if shape == (16, 16) and dtype_width == 16:
        return [
            (thread_id % 4) * 4,
            sympy.floor((thread_id % 64) / 4),
        ]

    if shape == (8, 32) and dtype_width == 16:
        return [
            (thread_id % 4) * 4 + sympy.floor((thread_id % 32) / 16) * 16,
            sympy.floor((thread_id % 16) / 4) + sympy.floor((thread_id % 64) / 32) * 4,
        ]

    if shape == (32, 16) and dtype_width == 16:
        return [
            (thread_id % 4) * 4,
            sympy.floor((thread_id % 64) / 4) + 4 * sympy.floor((thread_id % 64) / 16),
        ]

    if shape == (16, 32) and dtype_width == 16:
        return [
            (thread_id % 4) * 4 + sympy.floor((thread_id % 32) / 16) * 16,
            sympy.floor((thread_id % 16) / 4) + sympy.floor((thread_id % 64) / 32) * 8,
        ]

    # XXX: Uncomment following line for debugging
    # assert False, "Unhandled shape and datatype!"

    # Signal to caller that we should not use the transposed load operation
    return None


def modify_index(index, elems_per_thread, delinearized):
    new_index = {key: index[key].subs({THREAD_0: 0}) for key in index}

    for i, key in enumerate(index.keys()):
        new_index[key].start += delinearized[i]
        new_index[key].size = elems_per_thread if i == len(index.keys()) - 1 else 1
        new_index[key].stride = 1
    return new_index


def rewrite_node(read, custom_node, elems_per_thread, delinearized):
    bits = custom_node.elements_per_thread * custom_node.type.dtype.bitwidth()

    # If a single transpose operation will suffice, then just modify the index
    if bits == 64:
        custom_node.index = modify_index(read.index, elems_per_thread, delinearized)
        custom_node.update_arg("transpose", True)
        return

    # Otherwise, generate smaller read operations, each of which will read 64 bits
    factor = bits // 64
    read_ops = [
        Read(
            custom_node.memory,
            custom_node.elements_per_thread // factor,
            mapping=custom_node.mapping,
            mapping_dynamic_vals=custom_node.mapping_dynamic_vals,
        ).add_to_graph(custom_node.graph)
        for _ in range(factor)
    ]

    for idx, op in enumerate(read_ops):
        # Adjust the start and size so that each new read operation reads a
        # smaller chunk from a particular offset
        new_index = copy.deepcopy(read.index)
        for key in read.index.keys():
            new_index[key].stride = read.index[key].stride

            if read.index[key].size > 1:
                offset = idx * read.index[key].size // factor
                new_index[key].size = offset
                new_index[key].start = read.index[key].start + offset
            else:
                new_index[key].size = read.index[key].size
                new_index[key].start = read.index[key].start

        op.index = new_index
        custom_op = get_custom(op)
        custom_op.infer_type()
        if custom_node.mapping_dynamic_vals:
            update_read_mapping_dynamic_values(custom_op)

        custom_op.index = modify_index(op.index, elems_per_thread, delinearized)
        custom_op.update_arg("transpose", True)

    concat = Reshape(read_ops, read.vector_shapes).add_to_graph(custom_node.graph)
    custom_node.replace_all_uses_with(concat)


def mark_hardware_transpose_candidates(
    trace: CapturedTrace, constraints: list[Constraint]
):
    hardware_constraint = get_hardware_constraint(constraints)
    thread_id = hardware_constraint.linearized_thread_id

    def transpose_wrapper(node: fx.Node) -> bool:
        read = get_custom(node)
        if not isinstance(read, Read) or len(read.type.symbolic_shape) <= 1:
            return False
        return is_transposed_read(read)

    # Get concrete shape of read result
    sub = lambda x: safe_subs(x, read.vector_shapes)

    for read in trace.walk(transpose_wrapper):
        custom_node = get_custom(read)
        if meets_hw_transpose_requirements(custom_node, constraints):
            mem_type = custom_node.memory_type
            width = mem_type.dtype.bitwidth()
            concrete_shape = tuple(map(sub, custom_node.memory_type.symbolic_shape))
            maybe_indices = fetch_delinearized_indices(concrete_shape, width, thread_id)
            if maybe_indices:
                with custom_node.graph.inserting_before(read):
                    elems_per_thread = hardware_constraint.max_elems_per_load(
                        mem_type.dtype
                    )
                    rewrite_node(read, custom_node, elems_per_thread, maybe_indices)
