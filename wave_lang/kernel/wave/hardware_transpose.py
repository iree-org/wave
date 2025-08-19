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

from .global_to_shared_gathers import update_read_mapping_dynamic_values
from ..lang.global_symbols import THREAD_0, SHARED_ADDRESS_SPACE
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import Read, Reshape, get_custom
from ..wave.constraints import (
    Constraint,
    MMAType,
    TilingConstraint,
    WorkgroupConstraint,
)
from ..wave.utils.run_utils import get_default_arch


def meets_hw_transpose_requirements(read: Read, constraints: list[Constraint]):
    if not get_default_arch() == "gfx950":
        return False

    if len(list(read.index.keys())) != 2:
        return False

    if read.type.dtype.bitwidth() > 16:
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


def fetch_delinearized_indices(hardware_constraint):
    mma_type = hardware_constraint.mma_type
    thread_id = hardware_constraint.linearized_thread_id

    if mma_type == MMAType.I32_16x16x32_I8:
        return [
            (thread_id % 2) * 8,
            sympy.floor((thread_id % 64) / 2),
        ]

    if mma_type == MMAType.F32_16x16x16_F16:
        return [
            (thread_id % 4) * 4,
            sympy.floor((thread_id % 64) / 4),
        ]

    if mma_type == MMAType.F32_32x32x8_F16:
        return [
            (thread_id % 4) * 4 + sympy.floor((thread_id % 32) / 16) * 16,
            sympy.floor((thread_id % 16) / 4) + sympy.floor((thread_id % 64) / 32) * 4,
        ]

    if mma_type == MMAType.F32_16x16x32_F16 or mma_type == MMAType.F32_16x16x32_BF16:
        return [
            (thread_id % 4) * 4,
            sympy.floor((thread_id % 64) / 4) + 4 * sympy.floor((thread_id % 64) / 16),
        ]

    if mma_type == MMAType.F32_32x32x16_F16 or mma_type == MMAType.F32_32x32x16_BF16:
        return [
            (thread_id % 4) * 4 + sympy.floor((thread_id % 32) / 16) * 16,
            sympy.floor((thread_id % 16) / 4) + sympy.floor((thread_id % 64) / 32) * 8,
        ]

    assert False, "unhandled MMA type"


def modify_index(index, hardware_constraint, elem_type):
    delinearized = fetch_delinearized_indices(hardware_constraint)
    new_index = {key: index[key].subs({THREAD_0: 0}) for key in index}
    load_elems_per_thread = hardware_constraint.max_elems_per_load(elem_type)

    for i, key in enumerate(index.keys()):
        new_index[key].start += delinearized[i]
        new_index[key].size = load_elems_per_thread if i == len(index.keys()) - 1 else 1
        new_index[key].stride = 1
    return new_index


def lhs_index(index):
    new_index = copy.deepcopy(index)
    for key in index.keys():
        new_index[key].start = index[key].start
        new_index[key].stride = index[key].stride

        if index[key].size > 1:
            new_index[key].size = index[key].size // 2
        else:
            new_index[key].size = index[key].size
    return new_index


def rhs_index(index, vector_shapes):
    new_index = copy.deepcopy(index)
    for key in index.keys():
        new_index[key].stride = index[key].stride

        if index[key].size > 1:
            offset = index[key].size // 2
            new_index[key].size = offset
            new_index[key].start = index[key].start + offset
        else:
            new_index[key].size = index[key].size
            new_index[key].start = index[key].start
    return new_index


def rewrite_split_node(read, custom_node, hardware_constraint):
    read_lhs = Read(
        custom_node.memory,
        custom_node.elements_per_thread // 2,
        mapping=custom_node.mapping,
        mapping_dynamic_vals=custom_node.mapping_dynamic_vals,
    ).add_to_graph(custom_node.graph)
    read_lhs.index = lhs_index(read.index)
    custom_lhs = get_custom(read_lhs)
    custom_lhs.infer_type()
    if custom_node.mapping_dynamic_vals:
        update_read_mapping_dynamic_values(custom_lhs)

    read_rhs = Read(
        custom_node.memory,
        custom_node.elements_per_thread // 2,
        mapping=custom_node.mapping,
        mapping_dynamic_vals=custom_node.mapping_dynamic_vals,
    ).add_to_graph(custom_node.graph)
    read_rhs.index = rhs_index(read.index, read.vector_shapes)
    custom_rhs = get_custom(read_rhs)
    custom_rhs.infer_type()
    if custom_node.mapping_dynamic_vals:
        update_read_mapping_dynamic_values(custom_rhs)

    concat = Reshape([read_lhs, read_rhs], read.vector_shapes).add_to_graph(
        custom_node.graph
    )
    custom_node.replace_all_uses_with(concat)

    custom_lhs.index = modify_index(
        read_lhs.index, hardware_constraint, custom_lhs.type.dtype
    )
    custom_lhs.update_arg("transpose", True)

    custom_rhs.index = modify_index(
        read_rhs.index, hardware_constraint, custom_rhs.type.dtype
    )
    custom_rhs.update_arg("transpose", True)


def rewrite_non_split_node(read, custom_node, hardware_constraint):
    custom_node.index = modify_index(
        read.index, hardware_constraint, custom_node.type.dtype
    )
    custom_node.update_arg("transpose", True)


def rewrite_node(read, custom_node, hardware_constraint):
    bits = custom_node.elements_per_thread * custom_node.type.dtype.bitwidth()
    if bits == 128:
        rewrite_split_node(read, custom_node, hardware_constraint)
    else:
        rewrite_non_split_node(read, custom_node, hardware_constraint)


def mark_hardware_transpose_candidates(
    trace: CapturedTrace, constraints: list[Constraint]
):
    hardware_constraint = get_hardware_constraint(constraints)
    supported_mfma_types = [
        MMAType.I32_16x16x32_I8,
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
        MMAType.F32_16x16x32_F16,
        MMAType.F32_16x16x32_BF16,
        MMAType.F32_32x32x16_F16,
        MMAType.F32_32x32x16_BF16,
    ]

    if not hardware_constraint.mma_type in supported_mfma_types:
        return

    def transpose_wrapper(node: fx.Node) -> bool:
        read = get_custom(node)
        if not isinstance(read, Read) or len(read.type.symbolic_shape) <= 1:
            return False
        return is_transposed_read(read)

    for read in trace.walk(transpose_wrapper):
        custom_node = get_custom(read)
        if meets_hw_transpose_requirements(custom_node, constraints):
            with custom_node.graph.inserting_before(read):
                rewrite_node(read, custom_node, hardware_constraint)
