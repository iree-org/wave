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
    ** Shared offset (Index Sequence unit: elements)
        shared offset is calculated by preserving the index sequence from a "write_shared" node,
        removing thread offsets within a tile, similar to global offset.

Example:
For loading tensors with shape M x K to alloc0 (smem), and N x K to alloc1 (smem),
with tile size = BLOCK_M * BLOCK_K, BLOCK_N x BLOCK_K, and K is the contiguous dimension:
    - global offset preserves BLOCK index and WAVE_ID: $WG0 * BLOCK_M + BLOCK_M * ($T0 // 32)
    - shared offset preserves tile-level index: similar structure to global offset
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import torch.fx as fx

from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr, is_literal
from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    CustomOp,
    TensorLoadToLDS,
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
    delinearize_index,
    divide_shape_into_chunks,
    find_index_bounds,
    get_hardware_constraint,
    infer_dim,
    is_pow2,
    remove_thread_indexing,
)
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


@dataclass
class TensorLoadConfig:
    """
    element_type
    distributed_shape : {dim0: tile dim 0 shape, dim1: tile dim 1 shape}
    shared_tile_index (IndexSequence)
    global_tile_index (IndexSequence)
    bounds

    note. base address will be represented as pointers in codegen.
    """

    element_type: "DataType"
    distributed_shape: dict[IndexSymbol, IndexExpr]
    shared_tile_index: dict[IndexSymbol, IndexSequence]
    global_tile_index: dict[IndexSymbol, IndexSequence]
    bounds: dict[IndexSymbol, IndexExpr]

    def __iter__(self):
        yield self.element_type
        yield self.distributed_shape
        yield self.shared_tile_index
        yield self.global_tile_index
        yield self.bounds


def get_global_element_index(
    node: CustomOp, thread_index: dict[IndexSymbol, IndexSequence]
) -> dict[IndexSymbol, IndexSequence]:
    """
    Global address = global mem buffer + tile offset in bytes
    This function returns the address by removing threads offset within a tile.
    """
    assert isinstance(node, Read), "Expect Read custom node as caller argument"
    index = remove_thread_indexing(node.index)

    return {
        key: IndexSequence(index[key].start + thread_index[key], 1, 1)
        for key in index.keys()
    }


def get_tensor_load_descriptor_config(
    read: Read,
    constraints: list[Constraint],
    constraint_tile_size: dict[IndexSymbol, int],
    element_type: "DataType",
    hardware_constraint: "HardwareConstraint",
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

    distributed_shape = materialize_shape(
        constraint_tile_size, symbolic_shape, read.vector_shapes
    )

    # Some waves can copy duplicated data, take the full tile shape and
    # redistribute across the waves using linearized wave_id.
    total_waves = math.prod(hardware_constraint.waves_per_block)
    linearized_wave_id = (
        hardware_constraint.linearized_thread_id // hardware_constraint.threads_per_wave
    )
    chunks_per_dim, chunk_shape = divide_shape_into_chunks(
        distributed_shape, total_waves
    )
    delinearized_wave_id = delinearize_index(linearized_wave_id, chunks_per_dim)
    delinearized_index = {
        sym: v * s
        for v, s, sym in zip(delinearized_wave_id, chunk_shape, symbolic_shape)
    }

    distributed_shape = {v: chunk_shape[i] for i, v in enumerate(symbolic_shape)}

    shared_tile_index = {
        sym: IndexSequence(v, 1, 1) for sym, v in delinearized_index.items()
    }

    global_tile_index = get_global_element_index(read, delinearized_index)

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
        tensor_write = TensorLoadToLDS(
            [read.memory], [write.memory], *config
        ).add_to_graph(write.graph, loc=write.location)

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
    2. Build descriptors for tensor.load.to.lds with proper IndexSequence offsets.
    3. Replace #a with tensor_load_to_shared op.
    4. Update write dependencies.
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

    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, TilingConstraint) or isinstance(c, WorkgroupConstraint)
    }

    for _writes in id_to_read_write.values():
        _, write = _writes[0]
        clear_padding(write)

    for reads_writes in id_to_read_write.values():
        read, write = reads_writes[0]

        assert (
            read.index == write.index
        ), "Bug: Read Write for a same shared space has different access pattern."

        element_type = read.type.dtype

        config = get_tensor_load_descriptor_config(
            read,
            constraints,
            constraint_tile_size,
            element_type,
            hardware_constraint,
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
