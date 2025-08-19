# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import sympy
import torch.fx as fx

from .minimize_global_loads import is_transposed_read, materialize_shape
from .utils.general_utils import get_hardware_constraint

from ..lang.global_symbols import THREAD_0, SHARED_ADDRESS_SPACE
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import Read, get_custom
from ..wave.constraints import Constraint, MMAType, TilingConstraint, WorkgroupConstraint
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


def modify_index(index, hardware_constraint, elem_type):
    thread_id = hardware_constraint.linearized_thread_id
    load_elems_per_thread = hardware_constraint.max_elems_per_load(elem_type)
    mul_factor = 4 * (16 // elem_type.bitwidth())
    div_factor = 4 // (16 // elem_type.bitwidth())

    thread_ids = [THREAD_0]
    new_index = {key: index[key].subs({t: 0 for t in thread_ids}) for key in index}
    mul = 4 if elem_type.bitwidth() == 16 else 2
    delinearized = [(thread_id % div_factor) * mul_factor, sympy.floor((thread_id % 64) / div_factor)]
    for i, key in enumerate(index.keys()):
        new_index[key].start += delinearized[i]
        new_index[key].size = load_elems_per_thread if i == len(index.keys()) - 1 else 1
        new_index[key].stride = 1
    return new_index


def mark_hardware_transpose_candidates(
    trace: CapturedTrace, constraints: list[Constraint]
):
    hardware_constraint = get_hardware_constraint(constraints)
    supported_mfma_types = [
        MMAType.I32_16x16x32_I8, MMAType.F32_16x16x16_F16,
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
            custom_node.index = modify_index(read.index, hardware_constraint, custom_node.type.dtype)
            custom_node.update_arg("transpose", True)
