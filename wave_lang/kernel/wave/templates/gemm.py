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
    waves_per_block: Optional[tuple[int, int, int]] = None,
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
            threads_per_wave=threads_per_wave, mma_type=mfma_variant, n_service_waves=1
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
