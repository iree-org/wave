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
from dataclasses import dataclass, field

import sympy
import torch.fx as fx

from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr
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
    get_hardware_constraint,
    get_workgroup_constraints,
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
    tensor_shapes : [tensor dim 0 shape, tensor dim 1 shape]
    tensor_strides
    element_type
    tensor_tile_shapes : [tile dim 0 shape, tile dim 1 shape]
    shared_tile_index (bytes)
    global_tile_index (IndexSequence)
    bounds

    note. base address will be represented as pointers in codegen.
    """

    tensor_shapes: list[IndexExpr] = None
    tensor_strides: list[int] = None
    element_type: "DataType" = None
    tensor_tile_shapes: list[int] = None
    shared_tile_index: int = None
    global_tile_index: dict[IndexSymbol, IndexSequence] = None
    bounds: list[int] = field(default_factory=list)

    def __iter__(self):
        yield self.tensor_shapes
        yield self.tensor_strides
        yield self.element_type
        yield self.tensor_tile_shapes
        yield self.shared_tile_index
        yield self.global_tile_index
        yield self.bounds


def get_tensor_tile_shapes(
    read: Read, constraint_tile_size: dict[IndexSymbol, int]
) -> list[int]:
    """
    0. Get symbolic shape from Read node.
    1. Materialize the tile from constraints.
    2. Return [tile dim 0 shape, tile dim 1 shape]
    """
    symbolic_shapes = read.type.symbolic_shape
    tensor_tile_shapes = materialize_shape(constraint_tile_size, symbolic_shapes)
    return list(reversed(tensor_tile_shapes))


def get_tensor_shapes(read: Read) -> list[IndexExpr]:
    """
    0. Get symbolic shape from Read node.
    1. Materialize the `data shape` using index subs
    2. Return [tensor dim 0 shape, tensor dim 1 shape]
    """
    tensor_shapes = []
    symbolic_shapes = read.type.symbolic_shape
    for sym_dim in reversed(symbolic_shapes):
        tensor_shapes.append(subs_idxc(sym_dim))

    assert all(
        [type(shape) is sympy.core.numbers.Integer for shape in tensor_shapes]
    ), "Unknown or dynamic dimension is not currently supported for tensor load to shared."
    return tensor_shapes


def get_tensor_strides(tensor_shapes) -> list[int]:
    """
    Formula: x + y * stride0 + z * stride1 + a * stride2 + b * stride3
    - stride 0 = dim x
    - stride 1 = stride 0 * dim y
    - stride 2 = stride 1 * dim z
    """

    strides = [tensor_shapes[0]]
    for i in range(1, len(tensor_shapes)):
        base = strides[-1]
        strides.append(base * tensor_shapes[i])

    return strides


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
    write: Write,
    constraint_tile_size: dict[IndexSymbol, int],
    waves_per_block,
    element_type: "DataType",
    wave_subs,
    hardware_constraint: "HardwareConstraint",
    alloc_offset_map,
) -> TensorLoadConfig:
    """
    Get the tensor to shared config for the given read and write.
    """

    # get data shape
    tensor_shapes = get_tensor_shapes(read)

    # get data strides
    tensor_strides = get_tensor_strides(tensor_shapes)

    # get tile shape
    tensor_tile_shapes = get_tensor_tile_shapes(read, constraint_tile_size)

    # get LDS byte offset
    shared_tile_index = get_shared_tile_byte_offset(write, alloc_offset_map)

    # get global tile addr
    global_tile_index = get_global_element_offset(read, wave_subs)

    return TensorLoadConfig(
        tensor_shapes,
        tensor_strides,
        element_type,
        tensor_tile_shapes,
        shared_tile_index,
        global_tile_index,
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
        barrier = SharedMemoryBarrier(tensor_wait=True).add_to_graph(
            write.graph, loc=tensor_write.location
        )

    tensor_write.pre_expansion_id = id(tensor_write)

    tensor_writes[write.memory].append(tensor_write)

    return tensor_writes


def get_allocation_offsets(trace) -> dict[fx.Node, int]:
    allocs, _, alloc_info = get_alloc_info(trace)
    offsets, allocation_size = determine_allocations_offsets(alloc_info)
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
    wg_constraints = get_workgroup_constraints(constraints)
    threads_per_wave = hardware_constraint.threads_per_wave
    waves_per_block = hardware_constraint.waves_per_block
    threads_per_block = hardware_constraint.threads_per_block

    thread_id = hardware_constraint.linearized_thread_id

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

    # clear all padding
    for _writes in id_to_read_write.values():
        _, write = _writes[0]
        custom_memory = get_custom(write.memory)
        padding = custom_memory.padding
        if padding != 0:
            custom_memory.update_arg("padding", 0)
            new_distributed_shape = list(custom_memory.distributed_shape)
            new_distributed_shape[-1] -= padding
            custom_memory.update_arg("distributed_shape", tuple(new_distributed_shape))

    allocate_offsets = get_allocation_offsets(trace)

    for reads_writes in id_to_read_write.values():
        read, write = reads_writes[0]

        assert (
            read.index == write.index
        ), "Bug: Read Write for a same shared space has different access pattern."

        element_type = read.type.dtype

        config = get_tensor_load_descriptor_config(
            read,
            write,
            constraint_tile_size,
            waves_per_block,
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
