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

This pass replaces the global Read + shared Write pair with direct
GatherToLDS (DMA) operations.  The e8m0_shuffle buffer layout places
each 32-row M-block's k-group data in 256-byte contiguous regions:

    flat = B * block_stride + G * 256 + K4 * 64 + M16 * 4 + P * 2 + H

where block_stride = K32_full * 32, B = m//32, G = k//8.  Each DMA op
loads one such 256-byte region (64 lanes x 4 bytes) directly into LDS
in preshuffle order.

The shared reads are transformed to use preshuffle addressing
(constant_base + lane_id * 4), enabling ds_read_b32 reads.

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
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    ExtractSlice,
    GatherToLDS,
    Read,
    Write,
    get_custom,
)
from .constraints import Constraint
from .utils.general_utils import (
    get_hardware_constraint,
    infer_dim,
    make_index_uniform_per_wave,
    remove_global_indexing,
    remove_thread_indexing,
)
from .utils.symbol_utils import subs_idxc

logger = get_logger("wave.preshuffle_scale_to_shared")


def _is_preshuffle_mapping(mapping) -> bool:
    """Check if a mapping is a preshuffle-style mapping:
    output is identity (logical coords pass through) but input is non-identity
    (physical coords are shuffled)."""
    if mapping is None:
        return False
    if len(mapping.output_mapping) != 2:
        return False
    if len(mapping.input_mapping) != 2:
        return False
    if not (mapping.is_output_identity() and not mapping.is_input_identity()):
        return False
    # Preshuffle input expressions always contain floor and Mod from the
    # e8m0 shuffle formula.
    input_atoms = set()
    for expr in mapping.input_mapping.values():
        input_atoms.update(type(a) for a in sympy.preorder_traversal(expr))
    return sympy.floor in input_atoms and sympy.Mod in input_atoms


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
            dim_0: IndexSequence(0, 1, 1),
            dim_1: IndexSequence(flat_start, 4, 1),
        }
        if hasattr(earliest_node, "vector_shapes"):
            wide_read_node.vector_shapes = deepcopy(earliest_node.vector_shapes)
        if sample_read.tag:
            wide_custom.tag = sample_read.tag
        if sample_read.expanded_dims:
            wide_custom.expanded_dims = deepcopy(sample_read.expanded_dims)

    for c in range(4):
        node, read, _, _ = col_map[c]
        with read.graph.inserting_before(node):
            extract = ExtractSlice(wide_read_node, [c], [1], [1]).add_to_graph(
                read.graph, loc=read.location
            )
            extract_custom = get_custom(extract)
            if hasattr(node, "vector_shapes"):
                extract.vector_shapes = deepcopy(node.vector_shapes)
            if read.tag:
                extract_custom.tag = read.tag
            if read.expanded_dims:
                extract_custom.expanded_dims = deepcopy(read.expanded_dims)

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
    1. Remove the preshuffle mapping from the global read (the shuffled buffer
       already has data in preshuffle order within each 256-byte region)
    2. Emit GatherToLDS ops that DMA 256-byte contiguous regions from global
       memory directly into LDS in preshuffle order
    3. Transform shared READ indices: constant_base + lane_id * 4
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

    hw = get_hardware_constraint(constraints)
    threads_per_wave = hw.threads_per_wave
    waves_per_block = hw.waves_per_block
    total_threads = int(sympy.prod(hw.threads_per_block))
    elements_per_thread = 4
    num_waves = total_threads // threads_per_wave
    num_m_blocks = m_count // 32

    # ensures we schedule enough DMA ops to cover all m-blocks even when num_m_blocks doesn't divide evenly by num_waves
    num_dma_ops = max(1, (num_m_blocks + num_waves - 1) // num_waves)

    logger.info(
        f"DMA config: {num_waves} waves x {threads_per_wave} lanes, "
        f"{elements_per_thread} ept, "
        f"{num_m_blocks} m-blocks, {num_dma_ops} DMA ops"
    )

    all_new_writes = []
    erased_reads = set()

    sample_read = write_group[0][2]
    global_mem_node = sample_read.memory
    global_type = sample_read.memory_type
    global_symbolic_shape = global_type.symbolic_shape
    global_m_dim = infer_dim(global_symbolic_shape[0])
    global_k_dim = infer_dim(global_symbolic_shape[1])
    global_k32_full = subs_idxc(global_symbolic_shape[1])
    element_type = sample_read.type.dtype

    global_index = remove_thread_indexing(sample_read.index)
    m_wg = global_index[global_m_dim].start
    k_tile = global_index[global_k_dim].start
    bounds = sample_read.bounds

    sample_write_node = write_group[0][0]

    wave_id = hw.linearized_thread_id // threads_per_wave
    lane_id = hw.linearized_thread_id % threads_per_wave

    block_stride = global_k32_full * 32
    g_offset = sympy.floor(k_tile / 8) * 256

    common_id = None
    for op_i in range(num_dma_ops):
        b_index = op_i * num_waves + wave_id

        # ensure we don't overflow the m-blocks
        # Eg: num_m_blocks = 2 and 4 waves, b_index without clamping would be: 0, 1, 2, 3
        # which would cause us to access m-blocks 2 and 3, which don't exist.
        b_index = sympy.Min(b_index, num_m_blocks - 1)

        src_flat = (
            (sympy.floor(m_wg / 32) + b_index) * block_stride
            + g_offset
            + lane_id * elements_per_thread
        )

        src_row = sympy.floor(src_flat / global_k32_full)
        src_col = sympy.Mod(src_flat, global_k32_full)

        src_index = {
            global_m_dim: IndexSequence(src_row, elements_per_thread, 1),
            global_k_dim: IndexSequence(src_col, elements_per_thread, 1),
        }

        lds_base = b_index * 256
        dst_index = {
            dim_0: IndexSequence(lds_base, elements_per_thread, 1),
            dim_1: IndexSequence(sympy.Integer(0), 1, 1),
        }
        dst_index = make_index_uniform_per_wave(
            dst_index, threads_per_wave, waves_per_block
        )

        with sample_read.graph.inserting_before(sample_write_node):
            new_write = GatherToLDS(
                global_mem_node,
                alloc_node,
                src_index,
                dst_index,
                element_type,
                elements_per_thread,
                None,
                None,
                bounds,
                (),
                (),
            ).add_to_graph(sample_read.graph, loc=sample_read.location)

        if op_i == 0:
            common_id = id(new_write)
        new_write.pre_expansion_id = common_id

        if hasattr(sample_write_node, "tag"):
            tag = get_custom(sample_write_node).tag
            if tag:
                new_write.tag = tag

        all_new_writes.append(new_write)

    for write_node, write, input_read in write_group:
        for user in list(write_node.users):
            user_custom = get_custom(user)
            if (
                isinstance(user_custom, Read)
                and user_custom._write_dependency is not None
            ):
                new_deps = [d for d in user_custom._write_dependency if d != write_node]
                new_deps.extend(all_new_writes)
                user_custom.update_arg("_write_dependency", new_deps)

        get_custom(write_node).erase()

        read_id = id(input_read.fx_node)
        if read_id not in erased_reads:
            erased_reads.add(read_id)
            input_read.update_arg("mapping", None)
            get_custom(input_read.fx_node).erase()

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
    for info in read_infos:
        node, read, flat_lds, cbase = info
        read.index = {
            dim_0: IndexSequence(0, 1, 1),
            dim_1: IndexSequence(flat_lds, 1, 1),
        }
        if read.mapping is not None:
            read.update_arg("mapping", None)
        remapped += 1

    logger.info(f"Remapped {remapped} shared reads to preshuffle addressing")
