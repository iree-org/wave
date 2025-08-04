# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import Optional

import sympy
import torch.fx as fx

from .._support.indexing import IndexExpr, IndexSequence, IndexSymbol, xor
from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    CustomOp,
    GatherToLDS,
    IndexSequence,
    Read,
    Write,
    get_custom,
)
from ..wave.constraints import (
    Constraint,
    TilingConstraint,
    WorkgroupConstraint,
)
from ..wave.utils.graph_utils import DCE
from .compile_options import WaveCompileOptions
from .minimize_global_loads import (
    materialize_shape,
    update_write_dependencies,
)
from .utils.general_utils import (
    ceildiv,
    delinearize_index,
    find_index_bounds,
    get_hardware_constraint,
    infer_dim,
    remove_thread_indexing,
    remove_global_indexing,
)
from .utils.graph_utils import DCE
from .utils.symbol_utils import subs_idxc

logger = logging.getLogger(__name__)


def is_valid_read(node: fx.Node) -> bool:
    read = get_custom(node)
    if not isinstance(read, Read):
        return False

    if subs_idxc(read.memory_type.address_space) != GLOBAL_ADDRESS_SPACE:
        return False

    return True


def is_valid_write(write: CustomOp) -> bool:
    if not isinstance(write, Write):
        return False

    if subs_idxc(write.memory_type.address_space) != SHARED_ADDRESS_SPACE:
        return False

    if not write.has_identity_mapping():
        return False

    return True


def combine_index(
    index1: dict[IndexSymbol, IndexSequence],
    index2: dict[IndexSymbol, IndexSequence],
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function takes two index sequences and combines them.
    """
    assert set(index1.keys()) == set(index2.keys())
    return {
        key: IndexSequence(
            index1[key].start + index2[key].start,
            sympy.Max(index1[key].size, index2[key].size),
            1,
        )
        for key in index2
    }


@dataclass
class GatherToSharedConfig:
    materialized_shape: list[IndexSymbol]
    elements_per_thread: int
    expected_number_of_loads: int


def get_gather_to_shared_config(
    read: Read,
    constraint_tile_size: dict[IndexSymbol, int],
    total_number_of_threads,
    element_type: "DataType",
    supported_load_widths: list[int],
    hardware_constraint: "HardwareConstraint",
    fastest_dim_bound: Optional[IndexExpr],
) -> Optional[GatherToSharedConfig]:
    """
    Get the gather to shared config for the given read and write.
    """
    logger.info(f"fastest_dim_bound={fastest_dim_bound}")

    bitwidth = element_type.bitwidth()
    logger.info(f"element_type={element_type}, bitwidth={bitwidth}")

    symbolic_shape = read.type.symbolic_shape
    logger.info(f"symbolic_shape={symbolic_shape}")

    store_elems_per_thread = hardware_constraint.max_elems_per_load(element_type)
    max_elements_per_store = total_number_of_threads * store_elems_per_thread
    logger.info(
        f"store_elems_per_thread={store_elems_per_thread}, "
        f"max_elements_per_store={max_elements_per_store}"
    )

    materialized_shape = materialize_shape(constraint_tile_size, symbolic_shape)
    logger.info(f"materialized_shape={materialized_shape}")

    total_number_of_elements = prod(materialized_shape)
    logger.info(f"total_number_of_elements={total_number_of_elements}")
    expected_number_of_loads = ceildiv(total_number_of_elements, max_elements_per_store)
    logger.info(f"expected_number_of_loads={expected_number_of_loads}")
    elements_per_thread = ceildiv(
        total_number_of_elements, expected_number_of_loads * total_number_of_threads
    )
    logger.info(f"elements_per_thread={elements_per_thread}")

    vector_width = elements_per_thread * bitwidth
    logger.info(f"vector_width={vector_width}")
    total_vector_width = total_number_of_elements * bitwidth
    logger.info(f"total_vector_width={total_vector_width}")
    if fastest_dim_bound is not None:
        fastest_dim_bound = fastest_dim_bound * bitwidth

    # `total_vector_width` is the total vector width all threads load.
    # `fastest_dim_bound` is the bound against which we are checking our loads,
    # it is larger than `vector_width` but it must be aligned with chosen load width.
    load_width = get_load_width(
        supported_load_widths,
        total_vector_width,
        total_number_of_threads,
        fastest_dim_bound,
    )
    if load_width is None:
        logger.info(
            f"No supported load width found for width={total_vector_width}, "
            f"fastest_dim_bound={subs_idxc(fastest_dim_bound)}"
        )
        return None

    logger.info(f"load_width={load_width}")

    # We need to adjust the number of loads and elements per thread if the
    # deduced GatherToLDS width is not equal to the original vector width.
    if vector_width > load_width:
        ratio = vector_width // load_width
        logger.info(f"ratio={ratio}")
        expected_number_of_loads *= ratio
        elements_per_thread //= ratio
    else:
        ratio = load_width // vector_width
        logger.info(f"ratio={1/ratio}")
        expected_number_of_loads = ceildiv(expected_number_of_loads, ratio)
        elements_per_thread *= ratio

    if materialized_shape[-1] % elements_per_thread != 0:
        logger.info(
            f"materialized_shape[-1]={materialized_shape[-1]} is not divisible by "
            f"elements_per_thread={elements_per_thread}"
        )
        return None

    return GatherToSharedConfig(
        materialized_shape,
        elements_per_thread,
        expected_number_of_loads,
    )


def emit_global_to_lds(
    read: Read,
    write: Write,
    materialized_shape: list[IndexSymbol],
    elements_per_thread: int,
    expected_number_of_loads: int,
    total_number_of_threads: int,
    thread_id: IndexExpr,
    symbolic_shape: list[IndexSymbol],
    bounds: dict[IndexSymbol, IndexExpr],
    wave_subs: dict[IndexSymbol, IndexExpr],
    element_type: "DataType",
) -> defaultdict[fx.Node, list[Write]]:
    """
    Emit `GatherToLDS` for the given read and write.
    """
    elements_per_wave = elements_per_thread * total_number_of_threads
    logger.info(f"elements_per_wave={elements_per_wave}")

    # For index delinearization, assume our shape in `elements_per_thread` chunks.
    materialized_shape_adjusted = list(materialized_shape)
    materialized_shape_adjusted[-1] = materialized_shape[-1] // elements_per_thread

    logger.info(f"materialized_shape_adjusted={materialized_shape_adjusted}")

    # GatherToLDS writes `elements_per_wave` elements contiguously to LDS, so we
    # cannot have any padding if it crosses a array row boundary.
    drop_padding = materialized_shape[-1] % elements_per_wave != 0
    tail_padding = elements_per_wave - prod(materialized_shape) % elements_per_wave
    logger.info(f"tail_padding={tail_padding}")

    global_index = remove_thread_indexing(read.index)
    logger.info(f"global_index={global_index}")

    new_writes = defaultdict(list)

    commmon_id = None
    for i in range(expected_number_of_loads):
        # As we adjusted our shape to be in `elements_per_thread` chunks, each
        # subsequent load will be `total_number_of_threads` elements apart.
        thread_id_adjusted = thread_id + i * total_number_of_threads
        nd_index = delinearize_index(thread_id_adjusted, materialized_shape_adjusted)
        logger.info(f"nd_index={nd_index}")
        write_index = {}
        for bound_expr, idx in zip(symbolic_shape, nd_index):
            last = bound_expr == symbolic_shape[-1]
            dim = infer_dim(bound_expr)

            idx = idx * elements_per_thread if last else idx
            size = elements_per_thread if last else 1
            stride = 1
            write_index[dim] = IndexSequence(idx, size, stride)

        read_index = combine_index(global_index, write_index)

        # GatherToLDS only uses write index from the first thread in wave,
        # so make the index wave-uniform, simplifying the calculation.
        write_index = {k: v.subs(wave_subs) for k, v in write_index.items()}

        logger.info(f"read_index={read_index}")
        logger.info(f"write_index={write_index}")
        with write.graph.inserting_before(write.fx_node):
            new_write = GatherToLDS(
                read.memory,
                write.memory,
                read_index,
                write_index,
                element_type,
                elements_per_thread,
                read.mapping,
                write.mapping,
                bounds,
            ).add_to_graph(write.graph)

        if i == 0:
            commmon_id = id(new_write)

        # Set `pre_expansion_id` for newly created `GatherToLDS` ops so we can find
        # they are part of the same group later.
        new_write.pre_expansion_id = commmon_id

        new_writes[write.memory].append(new_write)
        if drop_padding:
            custom_memory = get_custom(write.memory)
            padding = custom_memory.padding
            if padding != 0:
                custom_memory.update_arg("padding", 0)
                new_distributed_shape = list(custom_memory.distributed_shape)
                new_distributed_shape[-1] -= padding
                custom_memory.update_arg(
                    "distributed_shape", tuple(new_distributed_shape)
                )

        if tail_padding != 0:
            custom_memory = get_custom(write.memory)
            custom_memory.update_arg("tail_padding", tail_padding)

    return new_writes


def get_load_width(
    supported_load_widths: list[int],
    target_width: int,
    total_number_of_threads: int,
    fastest_dim_bound: Optional[IndexExpr],
) -> Optional[int]:
    """
    Get the largest suitable load width for the given bitwidth.
    """
    remainder = None
    load_width = None
    for width in supported_load_widths[::-1]:
        # Fastest dim must be divisible by load width as we cannot mask it to
        # do a partial load.
        if fastest_dim_bound is not None:
            # `subs_idxc` can return symbolic values which will also be != 0,
            # so we need to check `not (subs_idxc(...) == 0)`.
            if not (subs_idxc(fastest_dim_bound % width) == 0):
                continue

        load_width = width * total_number_of_threads
        rem = target_width % load_width
        if rem == 0:
            return width

        # Possible padding to be added to the end of the shared memory buffer.
        rem = load_width - rem

        # Choose the size which minimizes the shared memory waste.
        if remainder is None or rem < remainder:
            remainder = rem
            load_width = width

    return load_width


def gather_to_shared(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
):
    """
    This pass enables direct memory load from global to lds without passing
    through register reducing the data movement. This instruction is supported
    only on specific architectures (gfx950).
    """
    if not options.use_global_to_shared:
        return

    logger.info("gather_to_shared")

    if "gfx94" not in options.target and "gfx95" not in options.target:
        logger.info("gather_to_shared not supported on this architecture")
        return

    id_to_read_write = defaultdict(list)
    for read in trace.walk(is_valid_read):
        read = get_custom(read)
        for write in read.users:
            if not is_valid_write(write):
                continue

            key = (read.pre_expansion_id, write.pre_expansion_id)
            id_to_read_write[key].append((read, write))

    if not id_to_read_write:
        return

    hardware_constraint = get_hardware_constraint(constraints)
    threads_per_wave = hardware_constraint.threads_per_wave
    waves_per_block = hardware_constraint.waves_per_block
    threads_per_block = hardware_constraint.threads_per_block
    total_number_of_threads = prod(threads_per_block)
    logger.info(f"total_number_of_threads={total_number_of_threads}")

    thread_id = hardware_constraint.linearized_thread_id

    # Make LDS write index to be wave-uniform by doing (THREAD_0 // 64) * 64
    wave_subs = {
        THREAD_0: (
            ((THREAD_0 // threads_per_wave) * threads_per_wave)
            if waves_per_block[0] > 1
            else 0
        ),
        THREAD_1: THREAD_1 if waves_per_block[1] > 1 else 0,
        THREAD_2: THREAD_2 if waves_per_block[2] > 1 else 0,
    }

    supported_load_widths = [32]

    if "gfx95" in options.target:
        supported_load_widths += [96, 128]

    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, TilingConstraint) or isinstance(c, WorkgroupConstraint)
    }

    for reads_writes in id_to_read_write.values():
        read, write = reads_writes[0]
        logger.info(f"processing read={read}, write={write}")

        if not read.has_identity_mapping():
            logger.info("non-identity read mapping is not supported yet")
            continue

        assert read.index == write.index

        element_type = read.type.dtype

        symbolic_shape = read.type.symbolic_shape
        if read.bounds:
            bounds = read.bounds
        else:
            vector_shapes = read.vector_shapes or hardware_constraint.vector_shapes
            bounds = find_index_bounds(
                constraints, read.index, vector_shapes, symbolic_shape
            )

        logger.info(f"bounds={bounds}")
        fastest_dim_bound = bounds.get(symbolic_shape[-1], None) if bounds else None

        config = get_gather_to_shared_config(
            read,
            constraint_tile_size,
            total_number_of_threads,
            element_type,
            supported_load_widths,
            hardware_constraint,
            fastest_dim_bound,
        )

        if config is None:
            logger.info("no gather to shared config found")
            continue

        materialized_shape = config.materialized_shape
        elements_per_thread = config.elements_per_thread
        expected_number_of_loads = config.expected_number_of_loads

        new_writes = emit_global_to_lds(
            read,
            write,
            materialized_shape,
            elements_per_thread,
            expected_number_of_loads,
            total_number_of_threads,
            thread_id,
            symbolic_shape,
            bounds,
            wave_subs,
            element_type,
        )

        update_write_dependencies(new_writes, trace)

    DCE(trace)


def gather_to_shared_swizzling(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
):
    """
    This pass is used to swizzle the gather to shared op global index and
    corresponding LDS load index to reduce LDS bank conflicts.

    The formula for swizzling is:
    ```
    new_col = xor(row % max_phase, col // elements_per_thread) * elements_per_thread
    ```
    """
    if "gfx95" not in options.target:
        logger.info("gather_to_shared_swizzling not supported on this architecture")
        return

    logger.info("gather_to_shared_swizzling")

    id_to_gather = defaultdict(list)
    for gather in trace.walk(lambda x: isinstance(get_custom(x), GatherToLDS)):
        gather = get_custom(gather)
        key = gather.pre_expansion_id
        id_to_gather[key].append(gather)

    if not id_to_gather:
        return

    for gathers in id_to_gather.values():
        mem = gathers[0].dst
        reads = [
            get_custom(read) for read in mem.users if isinstance(get_custom(read), Read)
        ]
        gather = gathers[0]
        read = reads[0]
        if (
            gather.dtype != read.dtype
            or gather.elements_per_thread != read.elements_per_thread
        ):
            logger.info(
                "mismatched gather and read thread shapes: "
                f"gather.dtype={gather.dtype}, read.dtype={read.dtype}, "
                f"gather.elements_per_thread={gather.elements_per_thread}, "
                f"read.elements_per_thread={read.elements_per_thread}"
            )
            continue

        elements_per_thread = gather.elements_per_thread
        logger.info(f"elements_per_thread={elements_per_thread}")

        if elements_per_thread * gather.dtype.bitwidth() != 128:
            logger.info(
                f"gather to shared swizzling only supported for 128-bit elements, got {elements_per_thread}x{gather.dtype}"
            )
            continue

        shape = get_custom(mem).type.symbolic_shape
        if len(shape) < 2:
            logger.info(f"shape={shape} must be at least 2D")
            continue

        col_dim = infer_dim(shape[-1])
        row_dim = infer_dim(shape[-2])

        max_phase = 8

        for read in reads:
            index = remove_global_indexing(read.index, constraints)
            col_seq = index[col_dim]
            row_seq = index[row_dim]
            col = col_seq.start // elements_per_thread
            row = row_seq.start % max_phase
            col = xor(row, col) * elements_per_thread
            index[col_dim] = IndexSequence(col, col_seq.size, col_seq.stride)
            logger.info(f"read.index={read.index} -> {index}")
            read.index = index

        for gather in gathers:
            # Only apply swizzling to the thread part of the index and keep the
            # global part of the index unchanged.
            index = dict(gather.src_index)
            global_index = remove_thread_indexing(index)
            local_index = remove_global_indexing(index, constraints)
            col_seq = local_index[col_dim]
            row_seq = local_index[row_dim]
            col = col_seq.start // elements_per_thread
            row = row_seq.start % max_phase
            col = xor(row, col) * elements_per_thread
            col = global_index[col_dim].start + col
            index[col_dim] = IndexSequence(col, col_seq.size, col_seq.stride)
            logger.info(f"gather.src_index={gather.src_index} -> {index}")
            gather.update_arg("src_index", index)
