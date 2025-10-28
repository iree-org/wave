import logging
from collections import defaultdict
from dataclasses import dataclass

import sympy
import torch.fx as fx

from .._support.indexing import IndexSequence, IndexSymbol
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
    remove_thread_indexing,
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
    tensor_shapes : [M, N]
    tensor_strides: [N, M*N]
    tensor_data_size: 0: 1 bytes, 1: 2 bytes, 2: 4 bytes, 4: 8 bytes
    tensor_tile_shapes: [BLOCK_M, BLOCK_N]
    global_tile_index
    Note. shared addresses will be calculated during MLIR codegen
    """

    tensor_shapes: list = None
    tensor_strides: list = None
    tensor_data_size: int = 0
    tensor_tile_shapes: list = None
    global_tile_index: IndexSequence = None
    shared_tile_index: IndexSequence = None


def get_tensor_data_size(tensor_type: "DataType" = None):
    if not tensor_type:
        return 0
    byte_size = tensor_type.bitwidth() >> 3
    match byte_size:
        case 1:
            return 0
        case 2:
            return 1
        case 4:
            return 2
        case 8:
            return 3
        case _:
            return 0
    return 0


def get_tensor_tile_shapes(read: Read, constraint_tile_size: dict[IndexSymbol, int]):
    """
    0. Get symbolic shape from Read node.
    1. Materialize the tile from constraints.
    e.g., for BLK_MxBLK_K, tile dim 0 is BLK_K and tile dim 1 is BLK_M
    """
    symbolic_shapes = read.type.symbolic_shape
    tensor_tile_shapes = materialize_shape(constraint_tile_size, symbolic_shapes)
    return list(reversed(tensor_tile_shapes))


def get_tensor_shapes(read: Read):
    """
    0. Get symbolic shape from Read node.
    1. Materialize the `data shape` using index subs
    e.g., for MxK, tensor dim 0 is K and tensor dim 1 is M
    """
    tensor_shapes = []
    symbolic_shapes = read.type.symbolic_shape
    for sym_dim in reversed(symbolic_shapes):
        tensor_shapes.append(subs_idxc(sym_dim))

    assert all(
        [type(shape) is sympy.core.numbers.Integer for shape in tensor_shapes]
    ), "Unknown or dynamic dimension is not currently supported for tensor load to shared."
    return tensor_shapes


def get_tensor_strides(tensor_shapes):
    """
    formula: x + y * stride0 + z * stride1 + a * stride2 + b * stride3
    - stride 0 = dim x
    - stride 1 = stride 0 * dim y
    - stride 2 = stride 1 * dim z
    """

    strides = [tensor_shapes[0]]
    for i in range(1, len(tensor_shapes)):
        base = strides[-1]
        strides.append(base * tensor_shapes[i])

    return strides


def get_global_indexing(node_index):
    base = remove_thread_indexing(node_index)
    return {key: IndexSequence(base[key].start, 1, 1) for key in base.keys()}


def get_thread_indexing(node_index, global_index):
    return {
        key: IndexSequence(node_index[key].start - global_index[key].start, 1, 1)
        for key in node_index.keys()
    }

    return {node_index.start - global_index.start, 1, 1}


def get_global_tile_byte_offset(
    node: CustomOp, wave_subs, constraints, waves_per_block
):
    """
    : node is an instance of Read
    expect address for TDM (:byte): base_address + tile offset
    this function returns the tile offset
    """
    assert isinstance(node, Read), "Expect Read custom node as caller argument"

    index = {k: v.subs(wave_subs) for k, v in node.index.items()}
    return {key: IndexSequence(index[key].start, 1, 1) for key in index.keys()}


def get_shared_tile_byte_offset(node: fx.Node, alloc_offset_map):
    """
    Allocation space offset + tile offset in bytes
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
    Get the gather to shared config for the given read and write.
    """

    # get data shape
    tensor_shapes = get_tensor_shapes(read)

    # get data strides
    tensor_strides = get_tensor_strides(tensor_shapes)

    # get data size
    tensor_data_size = get_tensor_data_size(element_type)

    # get tile shape
    tensor_tile_shapes = get_tensor_tile_shapes(read, constraint_tile_size)

    # get LDS base address
    shared_tile_index = get_shared_tile_byte_offset(write, alloc_offset_map)

    # get global tile addr
    global_tile_index = get_global_tile_byte_offset(
        read, wave_subs, constraint_tile_size, waves_per_block
    )

    return TensorLoadConfig(
        tensor_shapes,
        tensor_strides,
        tensor_data_size,
        tensor_tile_shapes,
        global_tile_index,
        shared_tile_index,
    )


def build_tensor_descriptors(config: TensorLoadConfig):
    """
    Constructor descriptor groups, the comment below should follow bit order (from high to low)

    Group0: 4xi32
        - _: i32
            - 2 for image mode -> deafult by FM
        - global address: IndexSequence,
        - lds address: MLIR codegen
        - valid tensor: 1

    Group1: 8xi32
        - tensor dim 1 stride: i48
        - tensor dim 0 stride: i48
        - _: i16
        - tensor tile shape 1: i16
        - tensor tile shape 0: i16
        - tensor dim 1 shape: 32
        - tensor dim 0 shape: 32
        - _: i16
        - element: i16
            - _: i14
            - data size: i2
        - _: i16

    Returns: [g0, g1, g2, g3], where gx has type list
    """
    group0 = [2, config.global_tile_index, config.shared_tile_index, 1]
    group1 = [
        config.tensor_strides[1],
        config.tensor_strides[0],
        0,
        config.tensor_tile_shapes[1],
        config.tensor_tile_shapes[0],
        config.tensor_shapes[1],
        config.tensor_shapes[0],
        0,
        0,
        config.tensor_data_size,
        0,
    ]
    group2 = [0, 0, 0, 0]
    group3 = [0, 0, 0, 0]

    return [group0, group1, group2, group3]


def emit_tensor_load_to_shared(
    read: Read,
    write: Write,
    config: TensorLoadConfig,
) -> defaultdict[fx.Node, list[Write]]:
    """
    Emit `GatherToLDS` for the given read and write.
    """

    descriptors = build_tensor_descriptors(config)

    tensor_writes = defaultdict(list)

    common_id = None

    with write.graph.inserting_before(write.fx_node):
        tensor_write = TensorLoadToLDS(
            read.memory, write.memory, descriptors
        ).add_to_graph(write.graph, loc=write.location)
        barrier = SharedMemoryBarrier(tensor_wait=True).add_to_graph(
            write.graph, loc=tensor_write.location
        )

    # Set `pre_expansion_id` for newly created `GatherToLDS` ops so we can find
    # they are part of the same group later.
    tensor_write.pre_expansion_id = id(tensor_write)

    tensor_writes[write.memory].append(tensor_write)

    return tensor_writes


def get_allocation_offsets(trace):
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
        1) option.tensor_load is set
        2) target is gfx1250
    1. Build 1-many mapping of GLOBAL_READ: SHARED_WRITE_X ... #a
    2. Materialize workgroup tile size, per wave tile size
        - workgroup tile size = BLOCK_M * BLOCK_D
        - wave tile size = BLOCK_M / wave_M * BLOCK_D / wave_D
    3. Calculate global address and shared address accessed by a wave
    4. Build descriptors for tensor.load.to.lds
    5. Replace #a with tensor_load_to_shared op.
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
