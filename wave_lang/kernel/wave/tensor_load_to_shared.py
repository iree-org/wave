# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
In this pass:

* Dimension / Shapes
Shape is measured in array elements (not bytes).
    - dim0 corresponds to movements along the X axis.
    - dim1 corresponds to movements along the Y axis.

* Example:
For a tensor with shape M x K:
    - dim0 refers to K (the X-axis / inner dimension).
    - dim1 refers to M (the Y-axis / outer dimension).

* Strides
Each stride is measured in array elements (not bytes).
    - stride for dim0 = number of elements to advance one step in dim1.
    - stride for dim1 = number of elements to advance one step in dim2.
    - etc.
note. This definition is derived from the formula: addr = x + y * stride0 + z * stride1 + a * stride2 + b * stride3

Example:
For a tensor with shape M x K, and K is the contiguous dimension:
    - In order to advance one step in dim M, number of elements have to skip = shape[K]

* Offsets
    ** Global offset (Index Sequence unit : elements)
        global offset is calculated by only perserving global index sequence with wave index sequence.
        this is valid because tensor load instruction expects global base address of a tile.
    ** Shared offset (Allocation size unit: bytes)
        shared offset is calculated by materializing the distributed shape from a "write_shared" node.

Example:
For loading tensors with shape M x K to alloc0 (smem), and N x K to alloc1 (smem),
with tile size = BLOCK_M * BLOCK_K, BLOCK_N x BLOCK_K, and K is the contiguous dimension:
    - global offset perserves BLOCK index and WAVE_ID: $WG0 * BLOCK_M + BLOCK_M * ($T0 // 32)
    - shared offset:
        - for alloc0 = 0
        - for alloc1 = BLOCKM * BLOCK_K
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch.fx as fx

from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr, is_literal
from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    CustomOp,
    TensorLoadToLDS,
    SharedMemoryBarrier,
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
    find_index_bounds,
    get_hardware_constraint,
    infer_dim,
    is_pow2,
)
from .utils.symbol_utils import subs_idxc

from .memory_analysis.minimize_shared_allocs import get_alloc_info
from .memory_analysis.solver import determine_allocations_offsets

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


@dataclass
class TensorLoadConfig:
    """
    element_type
    tensor_tile_shapes : [tile dim 0 shape, tile dim 1 shape]
    shared_tile_index (bytes)
    global_tile_index (IndexSequence)
    bounds

    note. base address will be represented as pointers in codegen.
    """

    element_type: "DataType"
    distributed_shape: list[IndexExpr]
    shared_tile_index: int
    global_tile_index: dict[IndexSymbol, IndexSequence]
    bounds: dict[IndexSymbol, IndexExpr]

    def __iter__(self):
        yield self.element_type
        yield self.distributed_shape
        yield self.shared_tile_index
        yield self.global_tile_index
        yield self.bounds


def get_global_element_offset(
    node: CustomOp, wave_subs
) -> dict[IndexSymbol, IndexSequence]:
    """
    Global address = global mem buffer + tile offset in bytes
    This function returns the address by removing threads offset within a tile.
    """
    assert isinstance(node, Read), "Expect Read custom node as caller argument"

    index = {k: v.subs(wave_subs) for k, v in node.index.items()}
    return {key: IndexSequence(index[key].start, 1, 1) for key in index.keys()}


def get_shared_tile_byte_offset(node: fx.Node, alloc_offset_map) -> int:
    """
    LDS address = Shared mem buffer + tile offset in bytes
    This function returns the tile offset.
    """
    offset_sym = alloc_offset_map[node.memory]
    return int(offset_sym)


def get_tensor_load_descriptor_config(
    read: Read,
    constraints: list[Constraint],
    write: Write,
    constraint_tile_size: dict[IndexSymbol, int],
    element_type: "DataType",
    wave_subs,
    hardware_constraint: "HardwareConstraint",
    alloc_offset_map,
) -> TensorLoadConfig:
    """
    Get the tensor to shared config for the given read and write.
    """

    symbolic_shape = read.type.symbolic_shape

    if read.bounds:
        bounds = read.bounds
    else:
        vector_shapes = read.vector_shapes or hardware_constraint.vector_shapes
        bounds = find_index_bounds(
            constraints, read.index, vector_shapes, symbolic_shape
        )

    # Descriptor always expect bounds for all dimensions so if no bounds are provided, set to full shape
    if bounds is None:
        bounds = {infer_dim(v): v for v in symbolic_shape}
    else:
        bounds = {infer_dim(v): bounds.get(infer_dim(v), v) for v in symbolic_shape}

    distributed_shape = materialize_shape(constraint_tile_size, symbolic_shape)

    # get LDS byte offset
    shared_tile_index = get_shared_tile_byte_offset(write, alloc_offset_map)

    # get global tile addr
    global_tile_index = get_global_element_offset(read, wave_subs)

    return TensorLoadConfig(
        element_type,
        distributed_shape,
        shared_tile_index,
        global_tile_index,
        bounds,
    )


def emit_tensor_load_to_shared(
    read: Read,
    write: Write,
    config: TensorLoadConfig,
) -> defaultdict[fx.Node, list[Write]]:
    """
    Emit `TensorLoadToLDS` for the given read and write.
    """

    tensor_writes = defaultdict(list)

    with write.graph.inserting_before(write.fx_node):
        tensor_write = TensorLoadToLDS(read.memory, write.memory, *config).add_to_graph(
            write.graph, loc=write.location
        )
        SharedMemoryBarrier(tensor_wait=True).add_to_graph(
            write.graph, loc=tensor_write.location
        )

    tensor_write.pre_expansion_id = id(tensor_write)

    tensor_writes[write.memory].append(tensor_write)

    return tensor_writes


def is_supported_padding(padding: int, unpadded_dim: IndexExpr) -> bool:
    # Padding must be divisible by DWORD (4 bytes) and at most 128 DWORDs (512 bytes)
    if padding % 4 != 0 or padding > (128 * 4):
        return False

    # Unpadded dim must be a constant
    unpadded_dim = subs_idxc(unpadded_dim)
    if not is_literal(unpadded_dim):
        return False

    unpadded_dim = int(unpadded_dim)

    # Unpadded dimension must be power of 2, at least 8 bytes (2 DWORDs), at most 1024 bytes (256 DWORDs)
    if not is_pow2(unpadded_dim) or unpadded_dim < (2 * 4) or unpadded_dim > (256 * 4):
        return False

    return True


def clear_padding(write: Write):
    """
    Clear shared memory padding if it's not supported by tensor op.
    """
    custom_memory = get_custom(write.memory)
    padding = custom_memory.padding
    if padding == 0:
        return

    bytewidth = custom_memory.dtype.bitwidth() // 8

    unpadded_dim = custom_memory.unpadded_shape[-1] * bytewidth

    if is_supported_padding(padding * bytewidth, unpadded_dim):
        return

    custom_memory.update_arg("padding", 0)
    new_distributed_shape = list(custom_memory.distributed_shape)
    new_distributed_shape[-1] -= padding
    custom_memory.update_arg("distributed_shape", tuple(new_distributed_shape))


def get_allocation_offsets(trace) -> dict[fx.Node, int]:
    allocs, _, alloc_info = get_alloc_info(trace)
    offsets, _ = determine_allocations_offsets(alloc_info)
    allocs_to_offsets = {allocs[i]: offsets[i] for i in range(len(allocs))}
    return allocs_to_offsets


def tensor_load_to_shared(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
):
    """
    0. Check requirement:
        1) option.use_global_to_shared is set
        2) target is gfx1250
    1. Build 1-many mapping of GLOBAL_READ: SHARED_WRITE_X ... #a
    2. Get shared memory allocation information.
    3. Build descriptors for tensor.load.to.lds.
    4. Replace #a with tensor_load_to_shared op.
    5. Update write dependencies.
    """
    if not options.use_global_to_shared:
        return

    if "gfx1250" not in options.target:
        logger.info("tensor_load_to_shared is not supported on this architecture")
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

    # uniform shared memory write base address by aligning thread indexing position.
    # $T0 // wave size -> wave id
    wave_subs = {
        THREAD_0: (
            (THREAD_0 // threads_per_wave) * threads_per_wave
            if waves_per_block[0] > 1
            else 0
        ),
        THREAD_1: THREAD_1 if waves_per_block[1] > 1 else 0,
        THREAD_2: THREAD_2 if waves_per_block[2] > 1 else 0,
    }

    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, TilingConstraint) or isinstance(c, WorkgroupConstraint)
    }

    for _writes in id_to_read_write.values():
        _, write = _writes[0]
        clear_padding(write)

    allocate_offsets = get_allocation_offsets(trace)

    for reads_writes in id_to_read_write.values():
        read, write = reads_writes[0]

        assert (
            read.index == write.index
        ), "Bug: Read Write for a same shared space has different access pattern."

        element_type = read.type.dtype

        config = get_tensor_load_descriptor_config(
            read,
            constraints,
            write,
            constraint_tile_size,
            element_type,
            wave_subs,
            hardware_constraint,
            allocate_offsets,
        )

        if config is None:
            logger.info("no gather to shared config found")
            continue

        tensor_writes = emit_tensor_load_to_shared(
            read,
            write,
            config,
        )

        update_write_dependencies(tensor_writes, trace)

    DCE(trace)
