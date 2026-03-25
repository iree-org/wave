# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Graph pass that coalesces epilogue stores by routing bf16 output writes
through LDS to enable wide global stores.

Each thread writes its bf16 values to LDS, a barrier synchronizes,
then threads read back contiguous 8-element chunks from LDS and
write them to global memory. The actual global store width depends
on the Write node codegen (typically buffer_store_dwordx4 for 8 bf16).
"""

import sympy

from .._support.indexing import IndexSequence, IndexSymbol
from .._support.tracing import CapturedTrace
from ..lang import Register, Memory
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    Allocate,
    Read,
    SharedMemoryBarrier,
    Write,
    get_custom,
)
from .constraints import Constraint, WorkgroupConstraint
from .region_canonicalization import RegionFormat, requires_region_format
from .utils.symbol_utils import subs_idxc, simplify

# 8 bf16 = 16 bytes = 128 bits = ds_write_b128 / buffer_store_dwordx4.
# Maximum single-op LDS write and global store width on GCN/CDNA/RDNA.
# Hardware-universal constant, not a target-specific knob.
COALESCED_STORE_WIDTH = 8

LDS_DIM = IndexSymbol("$LDS_FLAT")


def _get_tile_dims(
    constraints: list[Constraint], output_shape: tuple[IndexSymbol, ...]
) -> tuple[int, int]:
    """Extract BLOCK_M and BLOCK_N from WorkgroupConstraint.
    Tile sizes are always concrete integers (only problem dimensions
    M/N/K can be dynamic, not block sizes)."""
    tile_sizes = {}
    for c in constraints:
        if isinstance(c, WorkgroupConstraint):
            resolved = subs_idxc(c.tile_size)
            tile_sizes[c.dim] = int(resolved)
    assert (
        output_shape[0] in tile_sizes and output_shape[1] in tile_sizes
    ), f"Missing WorkgroupConstraint for output dims: {output_shape[:2]}"
    return tile_sizes[output_shape[0]], tile_sizes[output_shape[1]]


def _get_wg_offsets(
    constraints: list[Constraint], output_shape: tuple[IndexSymbol, ...]
) -> tuple[IndexSymbol, IndexSymbol]:
    """Get workgroup base offset expressions for M and N dims."""
    wg_offsets = {}
    for c in constraints:
        if isinstance(c, WorkgroupConstraint):
            wg_dim_sym = [WORKGROUP_0, WORKGROUP_1, WORKGROUP_2][c.workgroup_dim]
            wg_offsets[c.dim] = wg_dim_sym * subs_idxc(c.tile_size)
    m_wg = wg_offsets.get(output_shape[0], sympy.Integer(0))
    n_wg = wg_offsets.get(output_shape[1], sympy.Integer(0))
    return m_wg, n_wg


def _get_hardware_info(constraints: list[Constraint]):
    """Return HardwareConstraint and derived thread counts (tx, ty, total)."""
    from .utils.general_utils import get_hardware_constraint

    hw = get_hardware_constraint(constraints)
    tpb = hw.threads_per_block
    tx = int(subs_idxc(tpb[0]))
    ty = int(subs_idxc(tpb[1]))
    return hw, tx, ty, tx * ty


@requires_region_format(RegionFormat.SCHEDULE_SIGNATURE_PLACEHOLDERS)
def coalesce_epilogue_stores(
    trace: CapturedTrace,
    constraints: list[Constraint],
    epilogue_lds_budget: int = 49152,
):
    """Replace epilogue bf16 global writes with LDS-transpose wide stores.

    Identifies Write nodes that:
      - Target global memory
      - Have bf16 dtype
      - Are in the root graph (outside any loop)

    Replaces each with Write-to-LDS. Then appends Read-from-LDS (wide)
    and Write-to-global (wide) nodes. Barriers are inserted automatically
    by the add_shared_memory_barriers pass that runs later.
    """
    import wave_lang.kernel.lang as tkl

    root_graph = trace.get_root_graph()

    # Collect epilogue writes: global bf16 stores in the root graph.
    epilogue_writes = []
    for node in root_graph.nodes:
        if node.op != "call_function":
            continue
        custom = get_custom(node)
        if not isinstance(custom, Write):
            continue
        mem_type = custom.memory_type
        if (
            subs_idxc(mem_type.address_space) == GLOBAL_ADDRESS_SPACE
            and mem_type.dtype == tkl.bf16
        ):
            epilogue_writes.append(node)

    if not epilogue_writes:
        return

    output_node = None
    for node in root_graph.nodes:
        if node.op == "output":
            output_node = node
            break

    first_write = get_custom(epilogue_writes[0])
    output_shape = first_write.memory_type.symbolic_shape
    output_memory_node = first_write.memory

    tile_m, tile_n = _get_tile_dims(constraints, output_shape)

    m_wg, n_wg = _get_wg_offsets(constraints, output_shape)
    hw, tx, ty, num_threads = _get_hardware_info(constraints)

    m_sym, n_sym = output_shape[0], output_shape[1]

    # Compute how many rows of the tile fit in the epilogue LDS budget.
    # Reserve padding space for OOB writes from the shuffle-pack path.
    lds_row_stride = tile_n
    elem_bytes = first_write.memory_type.dtype.bitwidth() // 8
    padding_bytes = COALESCED_STORE_WIDTH * elem_bytes
    usable_budget = epilogue_lds_budget - padding_bytes
    chunk_rows = min(tile_m, usable_budget // (lds_row_stride * elem_bytes))
    # Align to MMA tile height so chunks don't split MMA output tiles.
    mma_m = hw.mma_matrix_shapes(hw.mma_type)[0]
    chunk_rows = (chunk_rows // mma_m) * mma_m
    assert chunk_rows > 0, (
        f"tile_n={tile_n} too large: epilogue LDS budget ({epilogue_lds_budget} B) "
        f"cannot fit even 16 rows of {tile_n * 2} bytes each"
    )
    # +COALESCED_STORE_WIDTH padding: safe landing zone for OOB thread writes.
    lds_elems = chunk_rows * lds_row_stride + COALESCED_STORE_WIDTH

    with root_graph.inserting_before(output_node):
        lds_alloc = Allocate(
            (sympy.Integer(lds_elems),),
            (sympy.Integer(lds_elems),),
            tkl.bf16,
            SHARED_ADDRESS_SPACE,
        ).add_to_graph(root_graph)
        lds_alloc.location = first_write.location

        num_chunks = (tile_m + chunk_rows - 1) // chunk_rows

        all_lds_write_nodes = []
        all_read_nodes = []

        # Linearized thread ID across the workgroup.
        flat_tid = THREAD_0 + THREAD_1 * tx

        # Drain outstanding async LDS ops (buffer_load ... lds) from the
        # GEMM loop before the epilogue writes to LDS. Without this,
        # epilogue writes can race with the loop's last async loads when
        # minimize_shared_allocs aliases their LDS regions.
        initial_barrier = SharedMemoryBarrier(wait_async_ops=True).add_to_graph(
            root_graph
        )
        initial_barrier.location = first_write.location

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_rows
            chunk_end = min(chunk_start + chunk_rows, tile_m)
            actual_rows = chunk_end - chunk_start
            chunk_elems = actual_rows * tile_n

            # Barrier between chunks: previous read-back must complete
            # before next chunk's LDS writes begin.
            if chunk_idx > 0:
                inter_barrier = SharedMemoryBarrier().add_to_graph(root_graph)
                inter_barrier.location = first_write.location

            # Phase 1: Each thread writes its bf16 values to LDS row-major.
            for w_node in epilogue_writes:
                w = get_custom(w_node)
                write_ept = int(subs_idxc(w.elements_per_thread))
                idx = w_node.index
                m_idx_start = (
                    idx[m_sym].start
                    if isinstance(idx[m_sym], IndexSequence)
                    else idx[m_sym]
                )
                n_idx_start = (
                    idx[n_sym].start
                    if isinstance(idx[n_sym], IndexSequence)
                    else idx[n_sym]
                )

                # Convert global indices to tile-local coordinates.
                m_local = m_idx_start - m_wg
                n_local = n_idx_start - n_wg

                # Flatten (m_local, n_local) to 1D LDS offset within this chunk.
                lds_flat = (m_local - chunk_start) * lds_row_stride + n_local
                # Threads outside the current chunk write to the padding zone.
                oob_pos = lds_elems

                in_chunk = sympy.Piecewise(
                    (lds_flat, (m_local >= chunk_start) & (m_local < chunk_end)),
                    (oob_pos, True),
                )

                lds_write = Write(
                    w.register_, lds_alloc, elements_per_thread=write_ept
                ).add_to_graph(root_graph)
                lds_write.index = {LDS_DIM: IndexSequence(in_chunk, write_ept, 1)}
                lds_write.location = w.location
                lds_write.type = Memory[
                    (sympy.Integer(lds_elems),), SHARED_ADDRESS_SPACE, tkl.bf16
                ]
                # TODO: Replace ad-hoc attribute with a formal Write node attribute
                # (e.g. packing_mode enum) so it survives graph transformations
                # and is visible in IR dumps.
                lds_write._shuffle_xor_pack = True
                all_lds_write_nodes.append(lds_write)

            # Barrier: all LDS writes must complete before reads begin.
            barrier = SharedMemoryBarrier().add_to_graph(root_graph)
            barrier.location = first_write.location

            # Phase 2: Threads read back contiguous 8-element chunks from
            # LDS and write them wide to global memory.
            chunk_lds_elems = actual_rows * lds_row_stride
            assert chunk_lds_elems % COALESCED_STORE_WIDTH == 0, (
                f"chunk elements ({chunk_lds_elems}) not divisible by "
                f"store width ({COALESCED_STORE_WIDTH})"
            )
            stores_total = chunk_lds_elems // COALESCED_STORE_WIDTH
            stores_per_thread = stores_total // num_threads

            # Main stores: evenly distributed across threads.
            for s in range(stores_per_thread):
                read_start = (flat_tid * stores_per_thread + s) * COALESCED_STORE_WIDTH

                lds_read = Read(
                    lds_alloc,
                    elements_per_thread=COALESCED_STORE_WIDTH,
                    _write_dependency=list(all_lds_write_nodes),
                ).add_to_graph(root_graph)
                lds_read.index = {
                    LDS_DIM: IndexSequence(read_start, COALESCED_STORE_WIDTH, 1)
                }
                lds_read.location = first_write.location
                lds_read.type = Register[sympy.Integer(COALESCED_STORE_WIDTH), tkl.bf16]
                all_read_nodes.append(lds_read)

                # Convert flat LDS offset back to (row, col) in the tile.
                flat_elem = read_start
                row_in_tile = (
                    simplify(sympy.floor(flat_elem / lds_row_stride)) + chunk_start
                )
                col_in_tile = simplify(sympy.Mod(flat_elem, lds_row_stride))

                global_m = m_wg + row_in_tile
                global_n = n_wg + col_in_tile

                global_write = Write(
                    lds_read,
                    output_memory_node,
                    elements_per_thread=COALESCED_STORE_WIDTH,
                ).add_to_graph(root_graph)
                global_write.index = {
                    m_sym: IndexSequence(simplify(global_m), 1, 1),
                    n_sym: IndexSequence(simplify(global_n), COALESCED_STORE_WIDTH, 1),
                }
                global_write.location = first_write.location
                global_write.type = first_write.type

            # Remainder stores: when stores_total isn't divisible by
            # num_threads, the leftover stores are handled by the
            # lowest-numbered threads. OOB threads read from index 0
            # (harmless duplicate).
            remainder = stores_total % num_threads
            if remainder > 0:
                extra_idx = stores_per_thread * num_threads + flat_tid
                read_start = extra_idx * COALESCED_STORE_WIDTH

                in_bounds_read = sympy.Piecewise(
                    (read_start, extra_idx < stores_total),
                    (sympy.Integer(0), True),
                )

                lds_read = Read(
                    lds_alloc,
                    elements_per_thread=COALESCED_STORE_WIDTH,
                    _write_dependency=list(all_lds_write_nodes),
                ).add_to_graph(root_graph)
                lds_read.index = {
                    LDS_DIM: IndexSequence(in_bounds_read, COALESCED_STORE_WIDTH, 1)
                }
                lds_read.location = first_write.location
                lds_read.type = Register[sympy.Integer(COALESCED_STORE_WIDTH), tkl.bf16]
                all_read_nodes.append(lds_read)

                flat_elem = in_bounds_read
                row_in_tile = (
                    simplify(sympy.floor(flat_elem / lds_row_stride)) + chunk_start
                )
                col_in_tile = simplify(sympy.Mod(flat_elem, lds_row_stride))

                global_m = m_wg + row_in_tile
                global_n = n_wg + col_in_tile

                global_write = Write(
                    lds_read,
                    output_memory_node,
                    elements_per_thread=COALESCED_STORE_WIDTH,
                ).add_to_graph(root_graph)
                global_write.index = {
                    m_sym: IndexSequence(simplify(global_m), 1, 1),
                    n_sym: IndexSequence(simplify(global_n), COALESCED_STORE_WIDTH, 1),
                }
                global_write.location = first_write.location
                global_write.type = first_write.type

    # Remove the original narrow global writes.
    for w_node in epilogue_writes:
        w_node.replace_all_uses_with(None)
        root_graph.erase_node(w_node)
