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
        num_ctas = 304

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
            tile_id = tkw.self_index(TILE_IDX, tkl.i32)
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
