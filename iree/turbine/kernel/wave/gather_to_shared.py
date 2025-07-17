# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    CustomOp,
    GatherToLDS,
    Read,
    Write,
    get_custom,
)
from ..ops.wave_ops import IndexSequence
from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr
from ..wave.constraints import (
    Constraint,
    TilingConstraint,
    WorkgroupConstraint,
)
from ..wave.utils.graph_utils import DCE
from .utils.general_utils import (
    get_hardware_constraint,
    ceildiv,
    delinearize_index,
)
from .utils.graph_utils import DCE
from .utils.mapping_utils import transform_index_on_mapping
from .utils.symbol_utils import (
    safe_subs,
    subs_idxc,
)
from .minimize_global_loads import (
    materialize_shape,
    update_write_dependencies,
)
from typing import Optional
from math import prod
import logging
import torch.fx as fx
from collections import defaultdict
import sympy
from .compile_options import WaveCompileOptions

logger = logging.getLogger(__name__)


gather_to_shared_supported_arch = ["gfx950"]


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


def get_write_node_consumers(read_custom: CustomOp) -> list[fx.Node]:
    write_node = []

    for user in read_custom.users:
        if (
            isinstance(user, Write)
            and subs_idxc(user.memory_type.address_space) == SHARED_ADDRESS_SPACE
        ):
            write_node.append(user)

    return write_node


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


def remove_thread_indexing(
    index: dict[IndexSymbol, IndexSequence],
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function takes the index sequence for a global read and removes all
    thread level indexing.
    """
    subs = {t: 0 for t in [THREAD_0, THREAD_1, THREAD_2, GPR_NUM]}
    return {key: safe_subs(index[key], subs) for key in index}


def get_load_width(supported_load_widths: list[int], bitwidth: int) -> Optional[int]:
    for width in supported_load_widths[::-1]:
        if bitwidth % width == 0:
            return width
    return None


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
    logger.info("gather_to_shared")

    # if get_default_arch() not in gather_to_shared_supported_arch:
    #     return

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

    # Make LDS write index to be wave-uniform.
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

        index = read.index
        assert index == write.index

        # fastest_dim = get_fastest_index(index)
        # last_dim = list(index)[fastest_dim]

        element_type = read.type.dtype
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
        expected_number_of_loads = ceildiv(
            total_number_of_elements, max_elements_per_store
        )
        logger.info(f"expected_number_of_loads={expected_number_of_loads}")
        elements_per_thread = ceildiv(
            total_number_of_elements, expected_number_of_loads * total_number_of_threads
        )
        logger.info(f"elements_per_thread={elements_per_thread}")

        vector_width = elements_per_thread * bitwidth
        load_width = get_load_width(supported_load_widths, vector_width)
        if load_width is None:
            logger.error(f"No supported load width found for bitwidth={vector_width}")
            continue

        logger.info(f"load_width={load_width}")

        ratio = vector_width // load_width
        logger.info(f"ratio={ratio}")
        expected_number_of_loads *= ratio
        elements_per_thread //= ratio

        elements_per_wave = elements_per_thread * total_number_of_threads
        logger.info(f"elements_per_wave={elements_per_wave}")
        drop_padding = materialized_shape[-1] % elements_per_wave != 0

        global_index = remove_thread_indexing(read.index)
        logger.info(f"global_index={global_index}")

        materialized_shape_adjusted = list(materialized_shape)
        materialized_shape_adjusted[-1] = sympy.ceiling(
            materialized_shape[-1] / elements_per_thread
        )
        logger.info(f"materialized_shape_adjusted={materialized_shape_adjusted}")

        new_writes = defaultdict(list)
        for i in range(expected_number_of_loads):
            nd_index = delinearize_index(
                thread_id + i * total_number_of_threads,
                materialized_shape_adjusted,
            )
            logger.info(f"nd_index={nd_index}")
            write_index = {}
            for dim, idx in zip(symbolic_shape, nd_index):
                last = dim == symbolic_shape[-1]

                idx = idx * elements_per_thread if last else idx
                size = elements_per_thread if last else 1
                stride = 1
                write_index[dim] = IndexSequence(idx, size, stride)

            read_index = combine_index(global_index, write_index)

            write_index = {k: v.subs(wave_subs) for k, v in write_index.items()}

            logger.info(f"read_index={read_index}")
            logger.info(f"write_index={write_index}")
            with write.graph.inserting_before(write.fx_node):
                new_write = GatherToLDS(
                    read.memory,
                    write.memory,
                    read_index,
                    write_index,
                    read.mapping,
                    write.mapping,
                    element_type,
                    elements_per_thread,
                ).add_to_graph(write.graph)

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

        update_write_dependencies(new_writes, trace)

    DCE(trace)
