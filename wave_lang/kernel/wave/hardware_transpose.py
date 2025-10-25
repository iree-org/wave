# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import copy
import sympy

from .utils.general_utils import get_hardware_constraint
from .utils.symbol_utils import safe_subs

from .global_to_shared_gathers import update_read_mapping_dynamic_values
from ..lang.global_symbols import THREAD_0
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    Read,
    Reshape,
    get_custom,
    read_meets_hw_transpose_requirements,
)
from ..wave.constraints import Constraint
from .compile_options import WaveCompileOptions


def _calculate_wide_tile_offset(Tm, dtype_width, vw):
    """Calculate offset for wide tile transpose operations."""
    if Tm > 8 and dtype_width == 16:
        return vw * 2
    return vw


def compute_transpose_indices(shape, dtype_width, thread_id, threads_per_wave):
    """
    Uses tile classification to determine the appropriate index pattern.
    Classifies tiles as wide, tall, or base based on shape parameters.
    """
    Tm, Tk = shape

    # Hardware constants
    TILE_THRESHOLD = 16

    # Vector width in elements (4 for 16-bit, 8 for 8-bit)
    vw = threads_per_wave // dtype_width

    # Group size for first dimension cycling (4 for 16-bit, 2 for 8-bit)
    group_size = vw if dtype_width == 16 else vw // 4

    # Precompute common thread ID modulo operations
    tid_mod_group = thread_id % group_size
    tid_mod_16 = thread_id % 16
    tid_mod_32 = thread_id % 32
    tid_mod_wave = thread_id % threads_per_wave

    # Base pattern components
    base_first = tid_mod_group * vw
    base_second = sympy.floor(tid_mod_wave / group_size)

    # Tile classification
    is_wide_tile = Tm <= TILE_THRESHOLD and Tk > TILE_THRESHOLD
    is_tall_tile = Tm > TILE_THRESHOLD and Tk <= vw * group_size and dtype_width == 16

    if is_wide_tile:
        # Wide tiles: multiple wave-groups horizontally
        first_dim = (
            base_first + sympy.floor(tid_mod_32 / TILE_THRESHOLD) * TILE_THRESHOLD
        )
        wide_second_base = sympy.floor(tid_mod_16 / group_size)
        offset = _calculate_wide_tile_offset(Tm, dtype_width, vw)
        second_dim = wide_second_base + sympy.floor(tid_mod_wave / 32) * offset
    elif is_tall_tile:
        # Tall tiles: multiple wave-groups vertically
        first_dim = base_first
        second_dim = base_second + group_size * sympy.floor(
            tid_mod_wave / TILE_THRESHOLD
        )
    else:
        # Base case: single wave-group (16x16 or smaller)
        first_dim = base_first
        second_dim = base_second

    return [first_dim, second_dim]


def fetch_delinearized_indices(shape, dtype_width, thread_id, threads_per_wave):
    """
    Returns delinearized indices for hardware transpose operations.

    Now uses unified formula with tile classification instead of 6 hardcoded cases.
    All shapes meeting basic alignment constraints are supported.
    """
    # Validate supported configurations
    if dtype_width not in (8, 16):
        return None

    Tm, Tk = shape
    vw = threads_per_wave // dtype_width

    # Check if shape is supported (must be compatible with base unit)
    if Tm % 4 != 0 or Tk % vw != 0:
        return None

    # All aligned shapes now supported via tile classification
    return compute_transpose_indices(shape, dtype_width, thread_id, threads_per_wave)


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
        return

    # Otherwise, generate smaller read operations, each of which will read 64 bits
    factor = bits // 64
    read_ops = [
        Read(
            custom_node.memory,
            custom_node.elements_per_thread // factor,
            mapping=custom_node.mapping,
            mapping_dynamic_vals=custom_node.mapping_dynamic_vals,
        ).add_to_graph(custom_node.graph, loc=custom_node.location)
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

    concat = Reshape(read_ops, read.vector_shapes).add_to_graph(
        custom_node.graph, loc=custom_node.location
    )
    custom_node.replace_all_uses_with(concat)


def mark_hardware_transpose_candidates(
    trace: CapturedTrace, constraints: list[Constraint], options: WaveCompileOptions
):
    """
    This pass attempts to rewrite transposed read operations on MI350 machines.
    It first detects whether the operation satisfies the prerequisites, before
    rewriting the index field to the form that can work with the native
    transposed load operation.
    """

    if "gfx95" not in options.target:
        return

    hardware_constraint = get_hardware_constraint(constraints)
    thread_id = hardware_constraint.linearized_thread_id

    # Get concrete shape of read result
    sub = lambda x: safe_subs(x, read.vector_shapes)

    # Filter read operations
    is_read = lambda x: isinstance(get_custom(x), Read)

    for read in trace.walk(is_read):
        custom_node = get_custom(read)
        if not read_meets_hw_transpose_requirements(custom_node, constraints):
            continue

        mem_type = custom_node.memory_type
        width = mem_type.dtype.bitwidth()
        concrete_shape = tuple(map(sub, custom_node.memory_type.symbolic_shape))
        maybe_indices = fetch_delinearized_indices(
            concrete_shape, width, thread_id, hardware_constraint.threads_per_wave
        )
        if not maybe_indices:
            continue

        with custom_node.graph.inserting_before(read):
            elems_per_thread = hardware_constraint.max_elems_per_load(mem_type.dtype)
            rewrite_node(read, custom_node, elems_per_thread, maybe_indices)
