# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from typing import Sequence, Optional

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)


def get_gemm_kernel(
    shape: tuple[int, int, int],
    dynamic_dims: bool | tuple[bool, bool, bool],
    mfma_variant: MMAType,
    dtype: torch.dtype = torch.float16,
    threads_per_wave: int = 64,
    block_shape: Optional[tuple[int, int, int]] = None,
    waves_per_block: Optional[tuple[int, int]] = None,
    n_service_waves: int = 0,
):
    if not isinstance(dynamic_dims, Sequence):
        dynamic_dims = (dynamic_dims,) * 3

    if not block_shape:
        # BLOCK_M, BLOCK_N, BLOCK_K
        block_shape = (64, 64, 32)

    if not waves_per_block:
        # WAVE_M, WAVE_N
        waves_per_block = (2, 2)

    assert len(block_shape) == 3, "block_shape needs to be rank 3 for M, N, K."
    assert len(waves_per_block) == 2, "waves_per_block neds to be rank 2 for M, N."

    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    dtype = torch_dtype_to_wave(dtype)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / waves_per_block[0])]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / waves_per_block[1])]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            mma_type=mfma_variant,
            n_service_waves=n_service_waves,
        )
    ]

    # With dynamic dimensions, we need to add an assumption on how big
    # the iterate dimension is to determine whether we can schedule or not.
    if dynamic_dims[2]:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, dtype],
        b: tkl.Memory[N, K, ADDRESS_SPACE, dtype],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, dtype]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, dtype]
            b_reg = tkw.read(b)

            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_shape[0],
        BLOCK_N: block_shape[1],
        BLOCK_K: block_shape[2],
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = []
    if dynamic_dims[0]:
        dynamic_symbols.append(M)
        del hyperparams[M]

    if dynamic_dims[1]:
        dynamic_symbols.append(N)
        del hyperparams[N]

    if dynamic_dims[2]:
        dynamic_symbols.append(K)
        del hyperparams[K]

    return gemm, hyperparams, dynamic_symbols


def get_gemm_kernel_transpose_a_b(
    shape: tuple[int, int, int],
    dynamic_dims: bool | tuple[bool, bool, bool],
    mfma_variant: MMAType,
    dtype: torch.dtype = torch.float16,
    threads_per_wave: int = 64,
):
    if not isinstance(dynamic_dims, Sequence):
        dynamic_dims = (dynamic_dims,) * 3

    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Iterator symbols
    m = tkl.sym.m
    n = tkl.sym.n
    k = tkl.sym.k

    dtype = torch_dtype_to_wave(dtype)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.IteratorBindings({m: M, n: N, k: K})]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=threads_per_wave, mma_type=mfma_variant)
    ]

    if dynamic_dims[2]:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[K, M, ADDRESS_SPACE, dtype],
        b: tkl.Memory[K, N, ADDRESS_SPACE, dtype],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, source=(k, m), target=(m, k))
            b_reg = tkw.read(b, source=(k, n), target=(n, k))
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = []
    if dynamic_dims[0]:
        dynamic_symbols.append(M)
        del hyperparams[M]

    if dynamic_dims[1]:
        dynamic_symbols.append(N)
        del hyperparams[N]

    if dynamic_dims[2]:
        dynamic_symbols.append(K)
        del hyperparams[K]

    return gemm, hyperparams, dynamic_symbols


def get_persistent_gemm_kernel(
    shape: tuple[int, int, int],
    mfma_variant: MMAType,
    threads_per_wave: int = 64,
    block_shape: Optional[tuple[int, int, int]] = None,
    waves_per_block: Optional[tuple[int, int, int]] = None,
    num_ctas: Optional[int] = None,
):
    """
    Creates a persistent GEMM kernel that uses a single workgroup dimension
    with linearized CTA dims. Each CTA iterates over multiple output tiles.

    """
    if not block_shape:
        block_shape = (128, 256, 64)

    if not waves_per_block:
        waves_per_block = (4, 1)

    m, n, k = shape
    block_m, block_n, block_k = block_shape

    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    total_tiles = m_tiles * n_tiles

    if num_ctas is None:
        num_ctas = 304  # Or number of compute units depending on the device

    # Symbols
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    TOTAL_TILES = tkl.sym.TOTAL_TILES
    NUM_CTAS = tkl.sym.NUM_CTAS
    TILE_IDX = tkl.sym.TILE_IDX
    CTA_M_OFFSET = tkl.sym.CTA_M_OFFSET
    CTA_N_OFFSET = tkl.sym.CTA_N_OFFSET
    N_TILES = tkl.sym.N_TILES

    # Index mappings for manual offset control
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    a_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i + CTA_M_OFFSET, K: j},
        outputs={M: i, K: j},
    )

    b_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={N: i + CTA_N_OFFSET, K: j},
        outputs={N: i, K: j},
    )

    c_write_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i + CTA_M_OFFSET, N: j + CTA_N_OFFSET},
    )

    constraints = [
        tkw.GridConstraint(NUM_CTAS),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.TilingConstraint(TILE_IDX),
        tkw.WaveConstraint(M, BLOCK_M / waves_per_block[0]),
        tkw.WaveConstraint(N, BLOCK_N / waves_per_block[1]),
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            mma_type=mfma_variant,
            vector_shapes={TILE_IDX: 0},
        ),
    ]

    @tkw.wave(constraints)
    def persistent_gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        condition = TILE_IDX < TOTAL_TILES
        init_tile_id = tkw.scalar(WORKGROUP_0, tkl.i32)

        @tkw.iterate(TILE_IDX, start=init_tile_id, condition=condition, init_args=[])
        def persistent_loop():
            tile_id = tkw.scalar(TILE_IDX, tkl.i32)
            m_offset = (tile_id // tkw.scalar(N_TILES, tkl.i32)) * tkw.scalar(
                BLOCK_M, tkl.i32
            )
            n_offset = (tile_id % tkw.scalar(N_TILES, tkl.i32)) * tkw.scalar(
                BLOCK_N, tkl.i32
            )
            tkw.set_symbol(CTA_M_OFFSET, m_offset)
            tkw.set_symbol(CTA_N_OFFSET, n_offset)

            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            @tkw.iterate(axis=K, init_args=[c_reg])
            def k_loop(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkw.read(a, mapping=a_read_mapping)
                b_reg = tkw.read(b, mapping=b_read_mapping)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(k_loop, c, mapping=c_write_mapping)

            num_ctas_scalar = tkw.scalar(NUM_CTAS, tkl.i32)
            next_idx = tile_id + num_ctas_scalar
            tkw.set_symbol(TILE_IDX, next_idx)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        TOTAL_TILES: total_tiles,
        N_TILES: n_tiles,
        NUM_CTAS: num_ctas,
    }

    return persistent_gemm, hyperparams


def get_streamk_gemm_kernel(
    shape: tuple[int, int, int],
    mfma_variant: MMAType,
    threads_per_wave: int = 64,
    block_shape: Optional[tuple[int, int, int]] = None,
    num_ctas: Optional[int] = None,
):
    """
    Creates a Stream-K GEMM kernel that distributes work more evenly across CTAs.
    Partitions work at the granularity of K-loop iterations
    Spinlock approach

    references used:
    https://github.com/ROCm/tritonBLAS/blob/main/include/tritonblas/kernels/streamk_gemm.py
    https://arxiv.org/abs/2301.03598
    https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/

    """
    import sympy
    from wave_lang.kernel._support.indexing import sym
    from wave_lang.kernel._support.dtype import f16, f32, i32
    from wave_lang.kernel.lang.wave_types import Memory, Register

    if not block_shape:
        block_shape = (128, 256, 64)

    m, n, k = shape
    # note: block_m must equal 128, since it matches with number threads along the cta dim i.e. (128, 2, 1)
    # this is because in when reading from partial_buffer, 1 thread is responsible for one row (so 128 threads to load 128 rows)
    block_m, block_n, block_k = block_shape

    num_tiles_m = (m + block_m - 1) // block_m
    num_tiles_n = (n + block_n - 1) // block_n
    total_tiles = num_tiles_m * num_tiles_n

    if num_ctas is None:
        num_ctas = 304  # TODO: Should be decided by a heuristic

    # each output tile is either split or not
    # if its split it will be a streamk tile
    # if its not it will be a data parallel tile
    # if you split a tile some ctas will get extra iterations if the total amount of iterations is not evenly distributed
    iters_per_tile = (k + block_k - 1) // block_k
    streamk_tiles = total_tiles
    total_streamk_iters = streamk_tiles * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // num_ctas
    streamk_extra_iters = total_streamk_iters % num_ctas
    data_parallel_tiles = total_tiles - streamk_tiles

    # Symbols
    M = sym.M
    N = sym.N
    K = sym.K
    BLOCK_M = sym.BLOCK_M
    BLOCK_N = sym.BLOCK_N
    BLOCK_K = sym.BLOCK_K
    ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
    ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
    ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C
    NUM_CTAS = sym.NUM_CTAS
    DP_TILE_ID_AXIS = sym.DP_TILE_ID_AXIS
    TOTAL_TILES = sym.TOTAL_TILES
    LOCK_DIM = sym.LOCK_DIM
    SPINLOCK_WAIT_FLAG = sym.SPINLOCK_WAIT_IDX
    CTA_ID = sym.CTA_ID
    CTA_ID_AXIS = sym.CTA_ID_AXIS
    THREAD_ID = sym.THREAD_ID
    NUM_K_TILES = sym.NUM_K_TILES
    START_K_TILE = sym.START_K_TILE
    ITERS_PER_OUTPUT_TILE = sym.ITERS_PER_OUTPUT_TILE
    WORK_UNIT_START = sym.WORK_UNIT_START
    WORK_UNIT_END = sym.WORK_UNIT_END
    CTA_M_OFFSET = sym.CTA_M_OFFSET
    CTA_N_OFFSET = sym.CTA_N_OFFSET
    N_TILES = sym.N_TILES
    NEW_CTA_K_END = sym.NEW_CTA_K_END
    OUTPUT_TILE_ITER_END = sym.OUTPUT_TILE_ITER_END
    TOTAL_STREAMK_ITERS = sym.TOTAL_STREAMK_ITERS
    STREAMK_ITERS_PCU = sym.STREAMK_ITERS_PCU
    STREAMK_EXTRA_ITERS = sym.STREAMK_EXTRA_ITERS
    STREAMK_TILES = sym.STREAMK_TILES
    DATA_PARALLEL_TILES = sym.DATA_PARALLEL_TILES
    OUTPUT_TILE_ID = sym.OUTPUT_TILE_ID
    NUM_CTAS_SK_TILES = sym.NUM_CTAS_SK_TILES
    SCALAR_DIM = sym.SCALAR_DIM
    CTA_K_END = sym.CTA_K_END

    # Index mappings
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    a_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i + CTA_M_OFFSET, K: j},
        outputs={M: i, K: j},
    )

    b_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={N: i + CTA_N_OFFSET, K: j},
        outputs={N: i, K: j},
    )

    partial_buffer_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            NUM_CTAS_SK_TILES: CTA_ID * STREAMK_TILES + OUTPUT_TILE_ID,
            M: THREAD_ID,
            N: j,
        },
        outputs={M: i, N: j},
    )

    partial_buffer_write_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={
            NUM_CTAS_SK_TILES: CTA_ID * STREAMK_TILES + OUTPUT_TILE_ID,
            M: i,
            N: j,
        },
    )

    lock_iter = tkw.IndexMapping.iterator(0)
    lock_buffer_read_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={
            NUM_CTAS_SK_TILES: CTA_ID * STREAMK_TILES + OUTPUT_TILE_ID,
            LOCK_DIM: lock_iter,
        },
        outputs={LOCK_DIM: lock_iter},
    )

    lock_buffer_write_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={LOCK_DIM: lock_iter},
        outputs={
            NUM_CTAS_SK_TILES: CTA_ID * STREAMK_TILES + OUTPUT_TILE_ID,
            LOCK_DIM: lock_iter,
        },
    )

    c_write_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: THREAD_ID + CTA_M_OFFSET, N: j + CTA_N_OFFSET},
    )

    constraints = [
        tkw.GridConstraint(NUM_CTAS),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(
            K, BLOCK_K, iters=NUM_K_TILES, start=START_K_TILE * BLOCK_K
        ),
        tkw.TilingConstraint(DP_TILE_ID_AXIS),
        tkw.TilingConstraint(WORK_UNIT_START),
        tkw.TilingConstraint(CTA_ID_AXIS),
        tkw.TilingConstraint(SPINLOCK_WAIT_FLAG),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            mma_type=mfma_variant,
            vector_shapes={
                DP_TILE_ID_AXIS: 0,
                LOCK_DIM: 1,
                SCALAR_DIM: 1,
                CTA_ID: 0,
                WORK_UNIT_START: 0,
                SPINLOCK_WAIT_FLAG: 0,
                CTA_ID_AXIS: 0,
                # M,N have to be 16 to match w/ the mfma variant where M=16,N=16 in F32_16x16x16_F16
                M: 16,
                N: 16,
            },
        ),
    ]

    @tkw.wave(constraints)
    def streamk_gemm(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],
        b: Memory[N, K, ADDRESS_SPACE_B, f16],
        partial_buffer: Memory[
            NUM_CTAS_SK_TILES,
            M,
            N,
            GLOBAL_ADDRESS_SPACE,
            f32,
            tkl.MemoryLayout(shape=(NUM_CTAS_SK_TILES, BLOCK_M, BLOCK_N)),
        ],
        lock_buffer: Memory[NUM_CTAS_SK_TILES, LOCK_DIM, GLOBAL_ADDRESS_SPACE, f32],
        c: Memory[M, N, ADDRESS_SPACE_C, f32],
    ):
        # StreamK part
        cta_id = tkw.scalar(WORKGROUP_0, i32)
        tkw.set_symbol(CTA_ID, cta_id)
        iters_per_output_tile = tkw.scalar(ITERS_PER_OUTPUT_TILE, i32)
        data_parallel_tiles = tkw.scalar(DATA_PARALLEL_TILES, i32)
        sk_iters_pcu = tkw.scalar(
            STREAMK_ITERS_PCU, i32
        )  # number of streamk iterations a compute unit is responsible
        sk_extra_iters = tkw.scalar(
            STREAMK_EXTRA_ITERS, i32
        )  # if there are remainder iterations i.e. a cta is responsible for more iterations than some

        extra_iter = tkw.minimum(cta_id, sk_extra_iters)

        # work unit range i.e. the amount of work a cta is responsible for across all waves/timesteps
        # where in a single wave a cta computes a single work unit
        # a single work unit can span multiple output tiles
        work_unit_start = (
            data_parallel_tiles * iters_per_output_tile
            + (cta_id * sk_iters_pcu)
            + extra_iter
        )
        next_extra_iter = tkw.minimum(cta_id + tkw.scalar(1, i32), sk_extra_iters)
        work_unit_end = (
            data_parallel_tiles * iters_per_output_tile
            + ((cta_id + tkw.scalar(1, i32)) * sk_iters_pcu)
            + next_extra_iter
        )

        tkw.set_symbol(WORK_UNIT_END, work_unit_end)
        sk_condition = WORK_UNIT_START < WORK_UNIT_END

        # loops through a single cta's work unit(s)
        @tkw.iterate(
            WORK_UNIT_START, start=work_unit_start, condition=sk_condition, init_args=[]
        )
        def sk_loop():
            # get the range of a cta within an output tile to do dynamic range for mma
            # where cta_k_start is the global cta start across all output tiles
            # i.e. if 3 output tiles with 10 iterations each
            # cta_k_start ranges from 0-29
            # while remainder ranges within 0-9
            cta_k_start = tkw.scalar(WORK_UNIT_START, i32)
            remainder = (
                cta_k_start % iters_per_output_tile
            )  # remainder is the relative index within the output tile itself
            cta_k_end = tkw.minimum(  # also relative end within an output tile
                cta_k_start + (iters_per_output_tile - remainder), work_unit_end
            )
            output_tile_id = cta_k_start // iters_per_output_tile
            tkw.set_symbol(OUTPUT_TILE_ID, output_tile_id)
            tkw.set_symbol(CTA_K_END, cta_k_end)

            m_offset = (output_tile_id // tkw.scalar(N_TILES, i32)) * tkw.scalar(
                BLOCK_M, i32
            )
            n_offset = (output_tile_id % tkw.scalar(N_TILES, i32)) * tkw.scalar(
                BLOCK_N, i32
            )
            tkw.set_symbol(CTA_M_OFFSET, m_offset)
            tkw.set_symbol(CTA_N_OFFSET, n_offset)

            tkw.set_symbol(START_K_TILE, remainder)
            tkw.set_symbol(NUM_K_TILES, cta_k_end - cta_k_start)

            c_reg = Register[M, N, f32](0.0)

            @tkw.iterate(K, init_args=[c_reg])
            def mac_loop(acc: Register[M, N, f32]) -> Register[M, N, f32]:
                a_reg = tkw.read(a, mapping=a_read_mapping)
                b_reg = tkw.read(b, mapping=b_read_mapping)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            output_tile_iter_start = output_tile_id * iters_per_output_tile
            is_first_split = cta_k_start == output_tile_iter_start

            # cta responsible for the first half of an output tile always does the reduction
            # since its easy to identify who is responsible for the first split of an output tile
            @tkw.conditional(~is_first_split)
            def store_partial():
                one_flag = Register[LOCK_DIM, f32](1.0)
                tkw.write(
                    mac_loop,
                    partial_buffer,
                    mapping=partial_buffer_write_mapping,
                    flags=tkw.MemoryAccessFlags.VOLATILE,
                )
                # memory release
                tkw.write(one_flag, lock_buffer, mapping=lock_buffer_write_mapping)

            @tkw.conditional(is_first_split)
            def reduce_partial():
                loop_start = cta_id + tkw.scalar(1, i32)
                output_tile_iter_end = output_tile_iter_start + iters_per_output_tile

                tkw.set_symbol(NEW_CTA_K_END, cta_k_end)
                new_cta_k_end = tkw.scalar(NEW_CTA_K_END, i32)

                tid = tkw.scalar(THREAD_0, i32)
                tkw.set_symbol(THREAD_ID, tid)

                # TODO: unnecessary read/write, right now it reads/writes to partial buffer
                # to change the element distribution per thread from 4 -> 16
                # Ideally everything should be kept in registers
                # i.e. simply do mac_loop + peer_p_reg
                # will also allow partial buffer and lock buffer to be of shape (NUM_CTAS,) rather than (NUM_CTAS*STREAMK_TILES,)
                # also reduction CTA should never write to global or else it will overwrite its data, my workaround solves this issue, but problems arise with hybrid
                tkw.write(
                    mac_loop, partial_buffer, mapping=partial_buffer_write_mapping
                )
                curr_acc = tkw.read(
                    partial_buffer,
                    mapping=partial_buffer_read_mapping,
                    elements_per_thread=16,  # we do elements_per_thread=16 since from our IndexMapping since the iterator strides by 16 elements from vector_shape (num of elements per wave)
                    # so then within each stride we want to load all 16 elements per thread
                )
                tkw.set_symbol(OUTPUT_TILE_ITER_END, output_tile_iter_end)

                # essentially if we are not at the end of an output tile there has to be one more cta w/ a higher index that has worked on this output tile too
                # so we continue until both conditions are satisfied
                aggregrate_partial_condition = (
                    sympy.Lt(GET_ITER_ARG(0), OUTPUT_TILE_ITER_END)
                ) & (CTA_ID_AXIS < NUM_CTAS)

                @tkw.iterate(
                    CTA_ID_AXIS,
                    start=loop_start,
                    condition=aggregrate_partial_condition,
                    init_args=[new_cta_k_end, curr_acc],
                )
                def aggregate_partials_loop(new_cta_k_end, acc):
                    curr_cta = tkw.scalar(CTA_ID_AXIS, i32)
                    not_ready = tkw.scalar(0, i32)
                    condition = sympy.Eq(SPINLOCK_WAIT_FLAG, 0)

                    # get the id of peer cta and collect their partial results
                    tkw.set_symbol(CTA_ID, curr_cta)

                    # wait until the partial result from another cta is ready
                    @tkw.iterate(
                        SPINLOCK_WAIT_FLAG,
                        start=not_ready,
                        condition=condition,
                        init_args=[],
                    )
                    def spinlock_wait():
                        # memory acquire
                        lock_val = tkw.read(
                            lock_buffer,
                            mapping=lock_buffer_read_mapping,
                            flags=tkw.MemoryAccessFlags.VOLATILE,
                        )
                        one_val = Register[LOCK_DIM, f32](1.0)
                        is_ready = lock_val == one_val
                        one_int = Register[LOCK_DIM, i32](1)
                        zero_int = Register[LOCK_DIM, i32](0)
                        # exit spinlock until lock value = 1
                        ready_flag = tkw.select(is_ready, one_int, zero_int)
                        tkw.set_symbol(SPINLOCK_WAIT_FLAG, ready_flag)

                    peer_p_reg = tkw.read(
                        partial_buffer,
                        mapping=partial_buffer_read_mapping,
                        elements_per_thread=16,
                        flags=tkw.MemoryAccessFlags.VOLATILE,
                    )
                    new_acc = acc + peer_p_reg

                    new_cta_k_end += sk_iters_pcu + tkw.select(
                        curr_cta < sk_extra_iters,
                        tkw.scalar(1, i32),
                        tkw.scalar(0, i32),
                    )
                    next_cta_val = curr_cta + tkw.scalar(1, i32)

                    tkw.set_symbol(CTA_ID_AXIS, next_cta_val)
                    return (new_cta_k_end, new_acc)

                final_k_end, final_acc = aggregate_partials_loop
                tkw.write(final_acc, c, mapping=c_write_mapping, elements_per_thread=16)

            new_cta_k_start = cta_k_end
            tkw.set_symbol(WORK_UNIT_START, new_cta_k_start)

    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        N_TILES: num_tiles_n,
        NUM_CTAS: num_ctas,
        NUM_CTAS_SK_TILES: max(1, num_ctas * streamk_tiles),
        LOCK_DIM: 1,
        SCALAR_DIM: 1,
        ITERS_PER_OUTPUT_TILE: iters_per_tile,
        TOTAL_STREAMK_ITERS: total_streamk_iters,
        STREAMK_ITERS_PCU: streamk_iters_pcu,
        STREAMK_EXTRA_ITERS: streamk_extra_iters,
        TOTAL_TILES: total_tiles,
        # num of streamk_tiles/data parallel tiles should be decided by a heuristic
        # as a good ratio will use data parallel as long as possible, and activate streamk when a given time step/wave of tiles needs
        # to be split for better utilization (hipblas/triton blas use origami as their heuristic)
        STREAMK_TILES: streamk_tiles,
        DATA_PARALLEL_TILES: data_parallel_tiles,
    }

    return streamk_gemm, hyperparams


def get_hybrid_streamk_gemm_kernel(
    shape: tuple[int, int, int],
    mfma_variant: MMAType,
    threads_per_wave: int = 64,
    block_shape: Optional[tuple[int, int, int]] = None,
    num_ctas: Optional[int] = None,
    streamk_tiles: Optional[int] = None,
):
    """
    Creates a hybrid Stream-K GEMM kernel that combines:
     Persistent GEMM and StreamK GEMM

    """
    import sympy
    from wave_lang.kernel._support.indexing import sym
    from wave_lang.kernel._support.dtype import f16, f32, i32
    from wave_lang.kernel.lang.wave_types import Memory, Register

    if not block_shape:
        block_shape = (128, 256, 64)

    m, n, k = shape
    block_m, block_n, block_k = block_shape

    num_tiles_m = (m + block_m - 1) // block_m
    num_tiles_n = (n + block_n - 1) // block_n
    total_tiles = num_tiles_m * num_tiles_n

    if num_ctas is None:
        num_ctas = 304

    if streamk_tiles is None:
        streamk_tiles = min(2, total_tiles)
    else:
        streamk_tiles = min(streamk_tiles, total_tiles)

    iters_per_tile = (k + block_k - 1) // block_k
    total_data_parallel_tiles = total_tiles - streamk_tiles
    total_streamk_iters = streamk_tiles * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // num_ctas
    streamk_extra_iters = total_streamk_iters % num_ctas

    # Symbols
    M = sym.M
    N = sym.N
    K = sym.K
    BLOCK_M = sym.BLOCK_M
    BLOCK_N = sym.BLOCK_N
    BLOCK_K = sym.BLOCK_K
    ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
    ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
    ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C
    NUM_CTAS = sym.NUM_CTAS
    DP_TILE_ID_AXIS = sym.DP_TILE_ID_AXIS
    TOTAL_TILES = sym.TOTAL_TILES
    LOCK_DIM = sym.LOCK_DIM
    SPINLOCK_WAIT_FLAG = sym.SPINLOCK_WAIT_IDX
    CTA_ID = sym.CTA_ID
    CTA_ID_AXIS = sym.CTA_ID_AXIS
    THREAD_ID = sym.THREAD_ID
    NUM_K_TILES = sym.NUM_K_TILES
    START_K_TILE = sym.START_K_TILE
    ITERS_PER_OUTPUT_TILE = sym.ITERS_PER_OUTPUT_TILE
    WORK_UNIT_START = sym.WORK_UNIT_START
    WORK_UNIT_END = sym.WORK_UNIT_END
    CTA_M_OFFSET = sym.CTA_M_OFFSET
    CTA_N_OFFSET = sym.CTA_N_OFFSET
    N_TILES = sym.N_TILES
    NEW_CTA_K_END = sym.NEW_CTA_K_END
    OUTPUT_TILE_ITER_END = sym.OUTPUT_TILE_ITER_END
    TOTAL_STREAMK_ITERS = sym.TOTAL_STREAMK_ITERS
    STREAMK_ITERS_PCU = sym.STREAMK_ITERS_PCU
    STREAMK_EXTRA_ITERS = sym.STREAMK_EXTRA_ITERS
    STREAMK_TILES = sym.STREAMK_TILES
    DATA_PARALLEL_TILES = sym.DATA_PARALLEL_TILES
    OUTPUT_TILE_ID = sym.OUTPUT_TILE_ID
    NUM_CTAS_SK_TILES = sym.NUM_CTAS_SK_TILES
    SCALAR_DIM = sym.SCALAR_DIM
    CTA_K_END = sym.CTA_K_END
    STREAMK_TILE_ID = sym.STREAMK_TILE_ID

    # Index mappings
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    a_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i + CTA_M_OFFSET, K: j},
        outputs={M: i, K: j},
    )

    b_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={N: i + CTA_N_OFFSET, K: j},
        outputs={N: i, K: j},
    )

    # For hybrid: use STREAMK_TILE_ID instead of OUTPUT_TILE_ID # since now an output tile can be dedicated to a streamk or data-parallel tile
    # (i.e. not all output tiles are streamk tiles)
    partial_buffer_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            NUM_CTAS_SK_TILES: CTA_ID * STREAMK_TILES + STREAMK_TILE_ID,
            M: THREAD_ID,
            N: j,
        },
        outputs={M: i, N: j},
    )

    partial_buffer_write_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={
            NUM_CTAS_SK_TILES: CTA_ID * STREAMK_TILES + STREAMK_TILE_ID,
            M: i,
            N: j,
        },
    )

    lock_iter = tkw.IndexMapping.iterator(0)
    lock_buffer_read_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={
            NUM_CTAS_SK_TILES: CTA_ID * STREAMK_TILES + STREAMK_TILE_ID,
            LOCK_DIM: lock_iter,
        },
        outputs={LOCK_DIM: lock_iter},
    )

    lock_buffer_write_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={LOCK_DIM: lock_iter},
        outputs={
            NUM_CTAS_SK_TILES: CTA_ID * STREAMK_TILES + STREAMK_TILE_ID,
            LOCK_DIM: lock_iter,
        },
    )

    c_write_mapping_sk = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: THREAD_ID + CTA_M_OFFSET, N: j + CTA_N_OFFSET},
    )

    c_write_mapping_dp = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i + CTA_M_OFFSET, N: j + CTA_N_OFFSET},
    )

    constraints = [
        tkw.GridConstraint(NUM_CTAS),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(
            K, BLOCK_K, iters=NUM_K_TILES, start=START_K_TILE * BLOCK_K
        ),
        tkw.TilingConstraint(DP_TILE_ID_AXIS),
        tkw.TilingConstraint(WORK_UNIT_START),
        tkw.TilingConstraint(CTA_ID_AXIS),
        tkw.TilingConstraint(SPINLOCK_WAIT_FLAG),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            mma_type=mfma_variant,
            vector_shapes={
                DP_TILE_ID_AXIS: 0,
                LOCK_DIM: 1,
                SCALAR_DIM: 1,
                CTA_ID: 0,
                WORK_UNIT_START: 0,
                SPINLOCK_WAIT_FLAG: 0,
                CTA_ID_AXIS: 0,
                M: 16,
                N: 16,
            },
        ),
    ]

    @tkw.wave(constraints)
    def hybrid_streamk_gemm(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],
        b: Memory[N, K, ADDRESS_SPACE_B, f16],
        partial_buffer: Memory[
            NUM_CTAS_SK_TILES,
            M,
            N,
            GLOBAL_ADDRESS_SPACE,
            f32,
            tkl.MemoryLayout(shape=(NUM_CTAS_SK_TILES, BLOCK_M, BLOCK_N)),
        ],
        lock_buffer: Memory[NUM_CTAS_SK_TILES, LOCK_DIM, GLOBAL_ADDRESS_SPACE, f32],
        c: Memory[M, N, ADDRESS_SPACE_C, f32],
    ):
        ### DATA PARALLEL PART
        condition = DP_TILE_ID_AXIS < DATA_PARALLEL_TILES
        init_tile_id = tkw.scalar(WORKGROUP_0, tkl.i32)

        @tkw.iterate(
            DP_TILE_ID_AXIS, start=init_tile_id, condition=condition, init_args=[]
        )
        def persistent_loop():
            tile_id = tkw.scalar(DP_TILE_ID_AXIS, tkl.i32)
            m_offset = (tile_id // tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_M, i32)
            n_offset = (tile_id % tkw.scalar(N_TILES, i32)) * tkw.scalar(BLOCK_N, i32)
            tkw.set_symbol(CTA_M_OFFSET, m_offset)
            tkw.set_symbol(CTA_N_OFFSET, n_offset)

            tkw.set_symbol(START_K_TILE, tkw.scalar(0, i32))
            tkw.set_symbol(NUM_K_TILES, tkw.scalar(ITERS_PER_OUTPUT_TILE, i32))

            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            @tkw.iterate(axis=K, init_args=[c_reg])
            def k_loop(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkw.read(a, mapping=a_read_mapping)
                b_reg = tkw.read(b, mapping=b_read_mapping)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(k_loop, c, mapping=c_write_mapping_dp)

            num_ctas_scalar = tkw.scalar(NUM_CTAS, tkl.i32)
            next_idx = tile_id + num_ctas_scalar
            tkw.set_symbol(DP_TILE_ID_AXIS, next_idx)

        ### STREAM-K PART
        cta_id = tkw.scalar(WORKGROUP_0, i32)
        tkw.set_symbol(CTA_ID, cta_id)
        iters_per_output_tile = tkw.scalar(ITERS_PER_OUTPUT_TILE, i32)
        data_parallel_tiles = tkw.scalar(DATA_PARALLEL_TILES, i32)
        sk_iters_pcu = tkw.scalar(STREAMK_ITERS_PCU, i32)
        sk_extra_iters = tkw.scalar(STREAMK_EXTRA_ITERS, i32)

        extra_iter = tkw.minimum(cta_id, sk_extra_iters)
        work_unit_start = (
            data_parallel_tiles * iters_per_output_tile
            + (cta_id * sk_iters_pcu)
            + extra_iter
        )
        next_extra_iter = tkw.minimum(cta_id + tkw.scalar(1, i32), sk_extra_iters)
        work_unit_end = (
            data_parallel_tiles * iters_per_output_tile
            + ((cta_id + tkw.scalar(1, i32)) * sk_iters_pcu)
            + next_extra_iter
        )

        tkw.set_symbol(WORK_UNIT_END, work_unit_end)
        sk_condition = WORK_UNIT_START < WORK_UNIT_END

        @tkw.iterate(
            WORK_UNIT_START, start=work_unit_start, condition=sk_condition, init_args=[]
        )
        def sk_loop():
            cta_k_start = tkw.scalar(WORK_UNIT_START, i32)
            remainder = cta_k_start % iters_per_output_tile
            cta_k_end = tkw.minimum(
                cta_k_start + (iters_per_output_tile - remainder), work_unit_end
            )
            output_tile_id = cta_k_start // iters_per_output_tile
            tkw.set_symbol(OUTPUT_TILE_ID, output_tile_id)
            tkw.set_symbol(CTA_K_END, cta_k_end)

            m_offset = (output_tile_id // tkw.scalar(N_TILES, i32)) * tkw.scalar(
                BLOCK_M, i32
            )
            n_offset = (output_tile_id % tkw.scalar(N_TILES, i32)) * tkw.scalar(
                BLOCK_N, i32
            )
            tkw.set_symbol(CTA_M_OFFSET, m_offset)
            tkw.set_symbol(CTA_N_OFFSET, n_offset)

            tkw.set_symbol(START_K_TILE, remainder)
            tkw.set_symbol(NUM_K_TILES, cta_k_end - cta_k_start)

            c_reg = Register[M, N, f32](0.0)

            @tkw.iterate(K, init_args=[c_reg])
            def mac_loop(acc: Register[M, N, f32]) -> Register[M, N, f32]:
                a_reg = tkw.read(a, mapping=a_read_mapping)
                b_reg = tkw.read(b, mapping=b_read_mapping)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            output_tile_iter_start = output_tile_id * iters_per_output_tile
            is_first_split = cta_k_start == output_tile_iter_start
            sk_tile_id = output_tile_id - tkw.scalar(DATA_PARALLEL_TILES, i32)
            tkw.set_symbol(STREAMK_TILE_ID, sk_tile_id)

            @tkw.conditional(~is_first_split)
            def store_partial():
                one_flag = Register[LOCK_DIM, f32](1.0)
                tkw.write(
                    mac_loop,
                    partial_buffer,
                    mapping=partial_buffer_write_mapping,
                    flags=tkw.MemoryAccessFlags.VOLATILE,
                )
                tkw.write(one_flag, lock_buffer, mapping=lock_buffer_write_mapping)

            @tkw.conditional(is_first_split)
            def reduce_and_write():
                loop_start = cta_id + tkw.scalar(1, i32)
                output_tile_iter_end = output_tile_iter_start + iters_per_output_tile

                tkw.set_symbol(NEW_CTA_K_END, cta_k_end)
                new_cta_k_end_scalar = tkw.scalar(NEW_CTA_K_END, i32)

                tid = tkw.scalar(THREAD_0, i32)
                tkw.set_symbol(THREAD_ID, tid)
                tkw.write(
                    mac_loop, partial_buffer, mapping=partial_buffer_write_mapping
                )
                curr_acc = tkw.read(
                    partial_buffer,
                    mapping=partial_buffer_read_mapping,
                    elements_per_thread=16,
                )

                tkw.set_symbol(OUTPUT_TILE_ITER_END, output_tile_iter_end)

                aggregrate_partial_condition = (
                    sympy.Lt(GET_ITER_ARG(0), OUTPUT_TILE_ITER_END)
                ) & (CTA_ID_AXIS < NUM_CTAS)

                @tkw.iterate(
                    CTA_ID_AXIS,
                    start=loop_start,
                    condition=aggregrate_partial_condition,
                    init_args=[new_cta_k_end_scalar, curr_acc],
                )
                def aggregate_partials_loop(loop_cta_k_end, acc):
                    curr_cta = tkw.scalar(CTA_ID_AXIS, i32)
                    not_ready = tkw.scalar(0, i32)
                    condition = sympy.Eq(SPINLOCK_WAIT_FLAG, 0)

                    tkw.set_symbol(CTA_ID, curr_cta)

                    @tkw.iterate(
                        SPINLOCK_WAIT_FLAG,
                        start=not_ready,
                        condition=condition,
                        init_args=[],
                    )
                    def spinlock_wait():
                        lock_val = tkw.read(
                            lock_buffer,
                            mapping=lock_buffer_read_mapping,
                            flags=tkw.MemoryAccessFlags.VOLATILE,
                        )
                        one_val = Register[LOCK_DIM, f32](1.0)
                        is_ready = lock_val == one_val
                        one_int = Register[LOCK_DIM, i32](1)
                        zero_int = Register[LOCK_DIM, i32](0)
                        ready_flag = tkw.select(is_ready, one_int, zero_int)
                        tkw.set_symbol(SPINLOCK_WAIT_FLAG, ready_flag)

                    tid2 = tkw.scalar(THREAD_0, i32)
                    tkw.set_symbol(THREAD_ID, tid2)
                    peer_p_reg = tkw.read(
                        partial_buffer,
                        mapping=partial_buffer_read_mapping,
                        elements_per_thread=16,
                        flags=tkw.MemoryAccessFlags.VOLATILE,
                    )
                    new_acc = acc + peer_p_reg

                    updated_k_end = (
                        loop_cta_k_end
                        + sk_iters_pcu
                        + tkw.select(
                            curr_cta < sk_extra_iters,
                            tkw.scalar(1, i32),
                            tkw.scalar(0, i32),
                        )
                    )
                    next_cta_val = curr_cta + tkw.scalar(1, i32)

                    tkw.set_symbol(CTA_ID_AXIS, next_cta_val)
                    return (updated_k_end, new_acc)

                tid3 = tkw.scalar(THREAD_0, i32)
                tkw.set_symbol(THREAD_ID, tid3)
                final_k_end, final_acc = aggregate_partials_loop
                tkw.write(
                    final_acc,
                    c,
                    mapping=c_write_mapping_sk,
                    elements_per_thread=16,
                )

            new_cta_k_start = cta_k_end
            tkw.set_symbol(WORK_UNIT_START, new_cta_k_start)

    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        N_TILES: num_tiles_n,
        NUM_CTAS: num_ctas,
        NUM_CTAS_SK_TILES: max(1, num_ctas * streamk_tiles),
        LOCK_DIM: 1,
        SCALAR_DIM: 1,
        ITERS_PER_OUTPUT_TILE: iters_per_tile,
        TOTAL_STREAMK_ITERS: total_streamk_iters,
        STREAMK_ITERS_PCU: streamk_iters_pcu,
        STREAMK_EXTRA_ITERS: streamk_extra_iters,
        TOTAL_TILES: total_tiles,
        STREAMK_TILES: streamk_tiles,
        DATA_PARALLEL_TILES: total_data_parallel_tiles,
    }

    return hybrid_streamk_gemm, hyperparams
