# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Transform shared memory layout for preshuffle scale buffers.

After promote_placeholders + minimize_global_loads, preshuffle scale data
flows through shared memory as:

    Global Read (ept=N, preshuffle mapping)
      -> Shared Write (ept=N, identity)
        -> 8 x Shared Read (ept=1, scattered positions)

The default [K_groups, M + pad] layout makes the shared reads scattered
(one ds_read_u8 per byte), because the MMA access pattern spans multiple
K-group rows and M-tile columns.

This pass keeps the global read mapping intact (so data arrives correctly
unshuffled in logical order) but transforms the LDS write indices using
the preshuffle formula so that data is stored in physical (preshuffle)
order in LDS.  This places each MFMA thread's 4 scale bytes at
consecutive addresses (lane_id * 4), enabling ds_read_b32 reads.

The preshuffle formula for logical (i=k_scale, j=m_row) is:
    flat = (j//32)*chunk + (i%4)*64 + (j%16)*4 + (i//4)*2 + ((j//16)%2)
where chunk = (K_SCALE_SHUFFLED // 8) * 256.
"""

from collections import defaultdict
from copy import deepcopy

import sympy
import torch.fx as fx

from wave_lang.support.logging import get_logger

from .._support.indexing import IndexSequence
from ..ops.wave_ops import (
    ExtractSlice,
    Read,
    Write,
    get_custom,
)
from .._support.tracing import CapturedTrace
from .constraints import Constraint
from .utils.general_utils import infer_dim, remove_global_indexing
from .utils.symbol_utils import subs_idxc

logger = get_logger("wave.preshuffle_scale_to_shared")


def _is_preshuffle_mapping(mapping) -> bool:
    """Check if a mapping is a preshuffle-style mapping:
    output is identity (logical coords pass through) but input is non-identity
    (physical coords are shuffled)."""
    if mapping is None:
        return False
    return mapping.is_output_identity() and not mapping.is_input_identity()


def _create_wide_read_1d(
    col_map: dict[int, tuple],
    alloc_node: fx.Node,
    dim_0,
    dim_1,
    flat_start,
    write_nodes: list[fx.Node],
):
    """Replace 4 individual byte reads with one 4-byte read + ExtractSlice ops.

    Works with the 1D-like LDS layout (total_bytes x 1).
    """
    sample_node = col_map[0][0]
    _, sample_read, _, _ = col_map[0]

    all_write_deps = []
    seen_deps = set()
    for c in range(4):
        node, read, _, _ = col_map[c]
        if read._write_dependency:
            for dep in read._write_dependency:
                if id(dep) not in seen_deps:
                    all_write_deps.append(dep)
                    seen_deps.add(id(dep))
    if not all_write_deps:
        all_write_deps = write_nodes if write_nodes else None

    earliest_node = sample_node
    for c in range(1, 4):
        candidate = col_map[c][0]
        for n in sample_read.graph.nodes:
            if n is candidate:
                earliest_node = candidate
                break
            if n is earliest_node:
                break

    with sample_read.graph.inserting_before(earliest_node):
        wide_read_node = Read(
            alloc_node,
            elements_per_thread=4,
            mapping=None,
            _write_dependency=all_write_deps,
            flags=sample_read.flags,
        ).add_to_graph(sample_read.graph, loc=sample_read.location)
        wide_custom = get_custom(wide_read_node)
        wide_custom.index = {
            dim_0: IndexSequence(flat_start, 4, 1),
            dim_1: IndexSequence(0, 1, 1),
        }
        if hasattr(earliest_node, "vector_shapes"):
            wide_read_node.vector_shapes = deepcopy(earliest_node.vector_shapes)

    for c in range(4):
        node, read, _, _ = col_map[c]
        with read.graph.inserting_before(node):
            extract = ExtractSlice(wide_read_node, [c], [1], [1]).add_to_graph(
                read.graph, loc=read.location
            )
            if hasattr(node, "vector_shapes"):
                extract.vector_shapes = deepcopy(node.vector_shapes)

        read.replace_all_uses_with(extract)
        if read.mapping is not None:
            read.update_arg("mapping", None)
        read.erase()


def preshuffle_scale_to_shared(trace: CapturedTrace, constraints: list[Constraint]):
    """Transform shared memory layout for preshuffle scale buffers.

    Runs after minimize_global_loads and before gather_to_shared.
    """
    from ..lang.global_symbols import (
        GLOBAL_ADDRESS_SPACE,
        SHARED_ADDRESS_SPACE,
    )

    writes_to_process = []
    for node in trace.walk(lambda n: isinstance(get_custom(n), Write)):
        write = get_custom(node)
        if subs_idxc(write.memory_type.address_space) != SHARED_ADDRESS_SPACE:
            continue

        input_read = get_custom(write.register_)
        if not isinstance(input_read, Read):
            continue
        if subs_idxc(input_read.memory_type.address_space) != GLOBAL_ADDRESS_SPACE:
            continue
        if not _is_preshuffle_mapping(input_read.mapping):
            continue
        if input_read.mapping_dynamic_vals:
            continue

        writes_to_process.append((node, write, input_read))

    if not writes_to_process:
        return

    memory_groups: dict[fx.Node, list] = defaultdict(list)
    for node, write, input_read in writes_to_process:
        memory_groups[write.memory].append((node, write, input_read))

    for alloc_node, group in memory_groups.items():
        _transform_scale_memory(alloc_node, group, trace, constraints)


def _preshuffle_flat(k, m, k_scale_shuffled):
    """Compute the flat preshuffle offset for logical scale coords (k, m).

    This is the CK/AITER preShuffleScaleBuffer formula for
    v_mfma_scale_f32_16x16x128_f8f6f4.
    """
    chunk_size = (k_scale_shuffled // 8) * 256
    return (
        (m // 32) * chunk_size
        + (k % 4) * 64
        + (m % 16) * 4
        + (k // 4) * 2
        + ((m // 16) % 2)
    )


def _transform_scale_memory(
    alloc_node: fx.Node,
    write_group: list,
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """Transform one scale memory's LDS layout to preshuffle order.

    Strategy:
    1. KEEP the global read mapping (data arrives in correct logical order)
    2. Transform WRITE indices: each logical byte at (k, m) is written to
       its preshuffle physical position in a 1D LDS
    3. Transform READ indices: constant_base + lane_id * 4
    4. Skip gather_to_shared since we manage the LDS layout ourselves
    """
    from ..lang.global_symbols import SHARED_ADDRESS_SPACE, THREAD_0

    alloc = get_custom(alloc_node)
    symbolic_shape = alloc.shape

    dim_0 = infer_dim(symbolic_shape[0])  # K/32 (outer dimension)
    dim_1 = infer_dim(symbolic_shape[1])  # M (inner dimension)

    old_dist = list(alloc.distributed_shape)
    dist_0 = int(subs_idxc(old_dist[0]))  # K/32 count
    dist_1 = int(subs_idxc(old_dist[1]))  # M count + padding
    padding = int(subs_idxc(alloc.padding)) if alloc.padding > 0 else 0
    m_count = dist_1 - padding
    k_count = dist_0
    k_scale_shuffled = k_count

    total_bytes = k_count * m_count
    alloc.update_arg("distributed_shape", (total_bytes, 1))
    alloc.update_arg("padding", 0)

    # Transform writes
    # The global read KEEPS its preshuffle mapping, so data arrives in
    # logical order.  We transform the write index so each logical byte
    # at (k, m) is stored at its preshuffle physical offset in the 1D LDS.
    all_new_writes = []
    for write_node, write, input_read in write_group:
        input_read.fx_node.meta["skip_gather_to_shared"] = True

        local_index = remove_global_indexing(write.index, constraints)
        seq_d0 = local_index.get(dim_0)  # K/32 index
        seq_d1 = local_index.get(dim_1)  # M index
        if seq_d0 is None or seq_d1 is None:
            all_new_writes.append(write_node)
            continue

        k_size = int(subs_idxc(seq_d0.size))
        m_size = int(subs_idxc(seq_d1.size))
        ept = max(k_size, m_size)

        if ept == 1:
            k_val = subs_idxc(seq_d0.start)
            m_val = subs_idxc(seq_d1.start)
            flat = _preshuffle_flat(k_val, m_val, k_scale_shuffled)
            write.index = {
                dim_0: IndexSequence(flat, 1, 1),
                dim_1: IndexSequence(0, 1, 1),
            }
            all_new_writes.append(write_node)
            continue

        read_result = write.register_
        new_writes_for_this = []
        for i in range(ept):
            with write.graph.inserting_before(write_node):
                extract = ExtractSlice(read_result, [i], [1], [1]).add_to_graph(
                    write.graph, loc=write.location, tag=write.tag
                )
                if m_size > 1:
                    m_val = subs_idxc(seq_d1.start) + i
                    k_val = subs_idxc(seq_d0.start)
                else:
                    m_val = subs_idxc(seq_d1.start)
                    k_val = subs_idxc(seq_d0.start) + i

                flat = _preshuffle_flat(k_val, m_val, k_scale_shuffled)
                new_w = Write(extract, alloc_node, 1).add_to_graph(
                    write.graph, loc=write.location, tag=write.tag
                )
                new_w_custom = get_custom(new_w)
                new_w_custom.index = {
                    dim_0: IndexSequence(flat, 1, 1),
                    dim_1: IndexSequence(0, 1, 1),
                }
                if hasattr(write_node, "vector_shapes"):
                    new_w.vector_shapes = write_node.vector_shapes
                if hasattr(write_node, "pre_expansion_id"):
                    new_w.pre_expansion_id = write_node.pre_expansion_id
                new_writes_for_this.append(new_w)

        all_new_writes.extend(new_writes_for_this)

        for user in list(write_node.users):
            user_custom = get_custom(user)
            if (
                isinstance(user_custom, Read)
                and user_custom._write_dependency is not None
            ):
                new_deps = [d for d in user_custom._write_dependency if d != write_node]
                new_deps.extend(new_writes_for_this)
                user_custom.update_arg("_write_dependency", new_deps)

        get_custom(write_node).erase()

    # --- Transform reads ---
    # LDS data is now in preshuffle physical order.  Each MMA read needs
    # one scale byte at logical (k, m).  The preshuffle formula decomposes
    # into constant_base + lane_id * 4 when k_offset is a multiple of 4
    # and m_offset is a multiple of 16 (guaranteed by MMA tiling).
    read_infos = []
    for node in trace.walk(lambda n: isinstance(get_custom(n), Read)):
        read = get_custom(node)
        if read.memory != alloc_node:
            continue
        if subs_idxc(read.memory_type.address_space) != SHARED_ADDRESS_SPACE:
            continue

        local_index = remove_global_indexing(read.index, constraints)
        seq_d0 = local_index.get(dim_0)
        seq_d1 = local_index.get(dim_1)
        if seq_d0 is None or seq_d1 is None:
            continue

        k_start = subs_idxc(seq_d0.start)
        m_start = subs_idxc(seq_d1.start)

        probe_subs = {s: 0 for s in k_start.free_symbols | m_start.free_symbols}
        k_offset = int(k_start.subs(probe_subs))
        m_offset = int(m_start.subs(probe_subs))

        chunk_size = (k_scale_shuffled // 8) * 256
        constant_base = (
            (m_offset // 32) * chunk_size + (k_offset // 4) * 2 + ((m_offset // 16) % 2)
        )

        lane_id = sympy.Mod(THREAD_0, 64)
        flat_lds = constant_base + lane_id * 4

        read_infos.append((node, read, flat_lds, constant_base))

    # Group by dword-aligned base for wide 4-byte reads.
    row_groups: dict[int, list] = defaultdict(list)
    for info in read_infos:
        _, _, _, cbase = info
        dword_base = (cbase // 4) * 4
        row_groups[dword_base].append(info)

    remapped = 0
    wide_reads_created = 0
    for dword_base, group in row_groups.items():
        if len(group) == 4:
            col_map = {}
            for node, read, flat_lds, cbase in group:
                byte_within_dword = cbase % 4
                col_map[byte_within_dword] = (node, read, flat_lds, cbase)
            if set(col_map.keys()) == {0, 1, 2, 3}:
                wide_flat = dword_base + sympy.Mod(THREAD_0, 64) * 4
                _create_wide_read_1d(
                    col_map, alloc_node, dim_0, dim_1, wide_flat, all_new_writes
                )
                remapped += 4
                wide_reads_created += 1
                continue

        for node, read, flat_lds, cbase in group:
            read.index = {
                dim_0: IndexSequence(flat_lds, 1, 1),
                dim_1: IndexSequence(0, 1, 1),
            }
            if read.mapping is not None:
                read.update_arg("mapping", None)
            remapped += 1

    logger.info(
        f"Remapped {remapped} shared reads "
        f"({wide_reads_created} wide 4-byte reads created)"
    )
