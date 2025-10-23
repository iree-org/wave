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
    ceildiv,
    delinearize_index,
    find_index_bounds,
    get_hardware_constraint,
    infer_dim,
    remove_thread_indexing,
    remove_global_indexing,
)
from .utils.general_utils import is_gather
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


@dataclass
class TensorLoadConfig:
    '''
    tensor_shapes : [M, N]
    tensor_tile_stride: [N, 1]
    tensor_data_size: 0: 1 bytes, 1: 2 bytes, 2: 4 bytes, 4: 8 bytes
    tensor_tile_shapes: [BLOCK_M, BLOCK_N]
    global_mem_base: remove_thread_indexing
    shared_mem_base:
    '''
    tensor_shapes: list = None
    tensor_strides: list = None
    tensor_data_size: int = 0
    tensor_tile_shapes: list = None
    global_mem_base: IndexSequence = None
    shared_mem_base: IndexSequence = None


def get_tensor_data_size(tensor_type: "DataType" = None):
    if not tensor_type: return 0
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
    '''
    0. Get symbolic shape from Read node.
    1. Materialize the tile from constraints.
    '''
    symbolic_shapes = read.type.symbolic_shape
    tensor_tile_shapes = materialize_shape(
        constraint_tile_size, symbolic_shapes
    )
    return tensor_tile_shapes

def get_tensor_shapes(read: Read):
    '''
    0. Get symbolic shape from Read node.
    1. Materialize the `data shape` using index subs
    '''
    tensor_shapes = []
    symbolic_shapes = read.type.symbolic_shape
    for sym_dim in symbolic_shapes:
        tensor_shapes.append(subs_idxc(sym_dim))

    assert all([type(shape) is sympy.core.numbers.Integer for shape in tensor_shapes]), "Unknown or dynamic dimension is not currently supported for tensor load to shared."
    return tensor_shapes

def get_tensor_strides(tensor_shapes):
    base = 1
    strides = []
    for shape in tensor_shapes:
        strides.append(base)
        base *= shape

    return strides


def get_write_shared_base_address(read: Read, write: Write, wave_subs, hardware_constraint, tile_shape):
    symbolic_shapes = read.type.symbolic_shape
    tid = hardware_constraint.linearized_thread_id
    nd_index = delinearize_index(tid, tile_shape)
    write_index = {k: v.subs(wave_subs) for k, v in zip(symbolic_shapes, nd_index)}
    return write_index

def get_tensor_load_descriptor_config(
    read: Read,
    write: Write,
    constraint_tile_size: dict[IndexSymbol, int],
    total_number_of_threads,
    element_type: "DataType",
    wave_subs,
    hardware_constraint: "HardwareConstraint",
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

    # get global base addr
    global_index = remove_thread_indexing(read.index)

    # get shared base addr
    shared_index = get_write_shared_base_address(read, write, wave_subs, hardware_constraint, tensor_tile_shapes)

    return TensorLoadConfig(
        tensor_shapes,
        tensor_strides,
        tensor_data_size,
        tensor_tile_shapes,
        global_index,
        shared_index
    )

def build_tensor_descriptors(config: TensorLoadConfig):
    '''
    Constructor descriptor groups, the comment below should follow bit order (from high to low)

    Group0: i128
        - global address: i64,
        - lds address: i32,
        - others: i32

    Group1: i256
        - tensor dim 1 stride: i48
        - tensor dim 0 stride: i48
        - _: i16
        - tensor tile shape 1: i16
        - tensor tile shape 0: i16
        - tensor dim 1 shape: 32
        - tensor dim 0 shape: 32
        - _: i16
        - _: i14
        - data size: i2
        - _: i16

    Returns: [g0, g1, g2, g3], where gx has type list
    '''
    group0 = [0, config.global_mem_base, config.shared_mem_base, 0]
    group1 = [config.tensor_strides[1], config.tensor_strides[0], 0, config.tensor_tile_shapes[1], config.tensor_tile_shapes[0], config.tensor_shapes[1], config.tensor_shapes[0], 0, 0, config.tensor_data_size, 0]
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
            read.memory,
            write.memory,
            read.index,
            write.index,
            read.mapping,
            write.mapping,
            read.mapping_dynamic_vals,
            write.mapping_dynamic_vals,
            descriptors,
        ).add_to_graph(write.graph, loc=write.location)

    # Set `pre_expansion_id` for newly created `GatherToLDS` ops so we can find
    # they are part of the same group later.
    tensor_write.pre_expansion_id = id(tensor_write)

    tensor_writes[write.memory].append(tensor_write)

    return tensor_writes



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
    2. Materialize workgroup data size, wave tile data size
    3. Calculate global base address and shared base address
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
    threads_per_wave = hardware_constraint.threads_per_wave
    waves_per_block = hardware_constraint.waves_per_block
    threads_per_block = hardware_constraint.threads_per_block
    total_number_of_threads = prod(threads_per_block)

    thread_id = hardware_constraint.linearized_thread_id

    # uniform shared memory write base address by aligning thread indexing position.
    # $T0 // wave size -> wave id
    # wave id * wave size -> wave start position
    wave_subs = {
        THREAD_0: (
            ((THREAD_0 // threads_per_wave) * threads_per_wave)
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

    for reads_writes in id_to_read_write.values():
        read, write = reads_writes[0]

        assert read.index == write.index, "Bug: Read Write for a same shared space has different access pattern."

        element_type = read.type.dtype

        config = get_tensor_load_descriptor_config(
            read,
            write,
            constraint_tile_size,
            total_number_of_threads,
            element_type,
            wave_subs,
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

