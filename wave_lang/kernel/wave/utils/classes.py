# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import List
from enum import Enum


@dataclass
class KernelLaunchInfo:
    grid: tuple[int] = None
    blocks: tuple[int] = None
    cluster_dims: tuple[int] = None
    shared_memory_bytes: int = 0
    func_name: str = ""
    grid_str: str = ""


############################################
# Wave Ops related Utils
############################################


# GPU shuffle modes
class ShuffleMode(Enum):
    XOR = 0
    DOWN = 1
    UP = 2
    IDX = 3


class AttentionOperationType(Enum):
    """Enumeration of attention operation types for prefetch stages."""

    MMA_0 = "mma_0"
    MMA_1 = "mma_1"
    SOFTMAX_0 = "softmax_0"
    SOFTMAX_1 = "softmax_1"
    LOCAL_LOAD_0 = "local_load_0"
    GLOBAL_LOAD_0 = "global_load_0"
    LOCAL_STORE_0 = "local_store_0"
    LOCAL_LOAD_1 = "local_load_1"
    GLOBAL_LOAD_1 = "global_load_1"
    LOCAL_STORE_1 = "local_store_1"

    @classmethod
    def get_all_types(cls) -> List["AttentionOperationType"]:
        """Get all operation types as a list."""
        return list(cls)

    @classmethod
    def from_string(cls, value: str) -> "AttentionOperationType":
        """Create an enum value from a string, with error handling."""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Unknown attention operation type: {value}")

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value


class GemmOperationType(Enum):
    """Enumeration of GEMM operation types"""

    MMA = "mma"
    GLOBAL_LOAD_LHS = "global_load_lhs"
    GLOBAL_LOAD_RHS = "global_load_rhs"
    LOCAL_WRITE_LHS = "local_write_lhs"
    LOCAL_WRITE_RHS = "local_write_rhs"
    GLOBAL_LOAD_TO_LDS_LHS = "global_load_to_lds_lhs"
    GLOBAL_LOAD_TO_LDS_RHS = "global_load_to_lds_rhs"
    LOCAL_LOAD_LHS = "local_load_lhs"
    LOCAL_LOAD_RHS = "local_load_rhs"

    GLOBAL_LOAD_LHS_SCALE = "global_load_lhs_scale"
    GLOBAL_LOAD_RHS_SCALE = "global_load_rhs_scale"
    LOCAL_WRITE_LHS_SCALE = "local_write_lhs_scale"
    LOCAL_WRITE_RHS_SCALE = "local_write_rhs_scale"
    GLOBAL_LOAD_TO_LDS_LHS_SCALE = "global_load_to_lds_lhs_scale"
    GLOBAL_LOAD_TO_LDS_RHS_SCALE = "global_load_to_lds_rhs_scale"
    LOCAL_LOAD_LHS_SCALE = "local_load_lhs_scale"
    LOCAL_LOAD_RHS_SCALE = "local_load_rhs_scale"

    @classmethod
    def get_all_types(cls) -> List["GemmOperationType"]:
        """Get all operation types as a list."""
        return list(cls)

    @classmethod
    def from_string(cls, value: str) -> "GemmOperationType":
        """Create an enum value from a string, with error handling."""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Unknown attention operation type: {value}")

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value


class CoalescingType(Enum):
    """
    CoalescingType.LINEAR:

    Most Naive and straightforward coalescing type/access pattern.
    In this mode, we simply flatten our N-d data, and flatten our M-d workers
    and simply distribute len(flatten_data) / flatten(flatten_worker) amount of data
    to each worker. Then we delinearize to get back N-d shape.

    CoalescingType.Linear is pretty good to minimize # of global loads, however it is
    not aware of wave data ownership.  This means every load only reads data for certain
    waves and not all waves/threads; hence stalling the process until data for all the
    wave is ready, as waves typically move together.
    To optimize that we introduce another strategy below.

    CoalescingType.WAVE_TILE_ALIGNED:

    More Wave data ownership aware coalescing strategy. We further divide data into
    different "wave tiles" which are based on data owned by each waves. Now that we have
    tiles, where each tile owns some set of waves, we can do something similar to
    linear distribution to maximize number of data each thread will load.
    Concrete examples will follow below.


    Example:

    Imagine you are working on a matmul of BLOCK_M=128, BLOCK_N=128, BLOCK_K=64.
    wave0-3 will load A[0:128, 0:64] @ B[0:64, 0:64]
    wave4-7 will load A[0:128, 0:64] @ B[0:64, 64:128]

    Wave0-3 and Wave4-7 will need to load same data for A.

    Assumptions:

    data to load: A matrix
    workgroup_shape = [BLOCK_M, BLOCK_N] = 128x64xf16
    wave_shape = 32x64xf16
    threads_per_wave = 64
    max_width_read = 128 bitwidth = 8 f16 elems
    num_waves = 8
    workgroup_threads = (64, 8)

    wave0 data ownership: data[0:32, :64]
    wave1 data ownership: data[32:64, :64]
    wave2 data ownership: data[64:96, :64]
    wave3 data ownership: data[96:128, :64]

    wave4 data ownership: data[0:32, :64]
    wave5 data ownership: data[32:64, :64]
    wave6 data ownership: data[64:96, :64]
    wave7 data ownership: data[96:128, :64]

    LINEAR distribution:
    flatten_data = 128*64 = 8192
    flatten_workers = 64*8 = 512
    data_per_worker =  flatten_data // flatten_workers = 16
    num_loads =  data_per_worker // max_width_read = 2

    Here we have two 128 bitwidth/8(f16) elems read. Where each thread/worker will own
    16 contiguous data, where neighboring threads will own contiguous data to each other.
    -> first load = read(data[0:64,:64]) -> useful only for wave0, wave1, wave4, wave5
    -> second load = read(data[64:128,:64]) -> useful only for wave2, wave3, wave6, wave7

    Things to note, Linear distribution does not take into account wave ownership of data
    and first loads data for only wave0 and wave1. Second load, loads only for wave2 and wave3.
    This means, we need to wait for all the loads to finish before we can proceed with further computations.


    WAVE_TILE_ALIGNED distribution:
    flatten_data = 128*64 = 8192
    flatten_wave_data = 32*64 = 2048
    flatten_workers = 64*8 = 512
    num_tiles = flatten_data // flatten_tile_data = 4
    waves_per_tile = num_waves / num_tile = 2
    tile_id = wave_id // waves_per_tile => tile0 owns wave0/wave1, tile1 owns wave2/wave3, ...

    Iff threads_per_wave * waves_per_tile * num_loads * max_width_read  == flatten_wave_data,
    then we can use WAVE_TILE_ALIGNED, since it is proven that with this access pattern we can also have minimal
    number of loads. In this case, since it is true, we can then have:

    -> first load = read(data[0:32, 0:32], data[32:64, 0:32], data[64:96, 0:32], data[96:128, 0:32])
    -> second load = read(data[0:32, 32:64], data[32:64, 32:64], data[64:96, 32:64], data[96:128, 32:64])

    As you can see, first load now loads half of the data owned by all the waves. and second load loads the remaining half.
    Since all the waves now have half their data ready, they can start processing doing computation/matmul as they wait on the
    second half.


    """

    LINEAR = 0x00
    WAVE_TILE_ALIGNED = 0x10
