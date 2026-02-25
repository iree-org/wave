# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tagged MXFP4 Scaled GEMM kernel templates for CDNA4 (GFX950).

All ops are tagged for use with MXFP4 schedule functions (e.g. get_mxfp4_dbuf_schedule).

Provides:
  - get_tagged_mxfp4_gemm:                  vanilla (A, B via LDS)
  - get_tagged_mxfp4_gemm_preshuffle_b:     B + B_scale preshuffled (direct global reads)

Required tags: k_loop, read_a, read_a_scale, read_b, read_b_scale,
bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale, scaled_mma.
"""

from sympy import Piecewise, ceiling, floor, Max

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params


def get_tagged_mxfp4_gemm(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block_shape: tuple[int, int, int] = (256, 256, 256),
    wave_shape: tuple[int, int] = (2, 2),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    a_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
    b_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
    reorder_workgroups=True,
    group_size_n=32,
):
    """Return a tagged MXFP4 scaled GEMM kernel + compile options for CDNA4.

    All ops are tagged for use with MXFP4 schedule functions.

    Args:
        shape: (M, N, K) problem dimensions.
        block_shape: (BLOCK_M, BLOCK_N, BLOCK_K) tile sizes.
        mfma_variant: Scaled MMA instruction type.
        wave_shape: (WAVE_M, WAVE_N) waves per workgroup.

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    A_ADDRESS_SPACE = tkl.sym.A_ADDRESS_SPACE
    B_ADDRESS_SPACE = tkl.sym.B_ADDRESS_SPACE
    C_ADDRESS_SPACE = tkl.sym.C_ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [tkw.WaveConstraint(M, BLOCK_M / wave_shape[0])]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / wave_shape[1])]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    if reorder_workgroups:
        new_wg0, new_wg1 = _reorder_mxfp4_workgroups(
            M, N, BLOCK_M, BLOCK_N, group_size_n
        )
        constraints += [tkw.ReorderingConstraint(new_wg0, 0)]
        constraints += [tkw.ReorderingConstraint(new_wg1, 1)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, A_ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, A_ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, B_ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, B_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, C_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn, tag="bitcast_a")
            a_scale_reg = tkw.read(a_scale, tag="read_a_scale")
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu, tag="bitcast_a_scale")
            b_reg = tkw.read(b, tag="read_b")
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn, tag="bitcast_b")
            b_scale_reg = tkw.read(b_scale, tag="read_b_scale")
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu, tag="bitcast_b_scale")
            acc = tkw.scaled_mma(
                a_reg, a_scale_reg, b_reg, b_scale_reg, acc, tag="scaled_mma"
            )
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        A_ADDRESS_SPACE: a_address_space,
        B_ADDRESS_SPACE: b_address_space,
        C_ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: block_shape[0],
        BLOCK_N: block_shape[1],
        BLOCK_K: block_shape[2],
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_global_to_shared=True,
        minimize_shared_allocs=False,
    )

    return gemm, options


def get_tagged_mxfp4_gemm_preshuffle_b(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block_shape: tuple[int, int, int] = (256, 256, 256),
    wave_shape: tuple[int, int] = (2, 2),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    a_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
    reorder_workgroups=True,
    group_size_n=32,
):
    """Return a tagged MXFP4 scaled GEMM kernel with preshuffled B and B_scale.

    B data is read directly from global memory using a preshuffle mapping
    (aiter shuffle_weight permutation).  B scales are also read from global
    memory using an e8m0 scale preshuffle mapping.  A and A_scale go through
    shared memory (LDS) as usual.

    All ops are tagged for use with MXFP4 schedule functions.

    Args:
        shape: (M, N, K) problem dimensions.
        block_shape: (BLOCK_M, BLOCK_N, BLOCK_K) tile sizes.
        wave_shape: (WAVE_M, WAVE_N) waves per workgroup.
        mfma_variant: Scaled MMA instruction type.
        a_address_space: Address space for A and A_scale (typically SHARED).

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    GROUP_SIZE_N = tkl.sym.GROUP_SIZE_N
    A_ADDRESS_SPACE = tkl.sym.A_ADDRESS_SPACE
    C_ADDRESS_SPACE = tkl.sym.C_ADDRESS_SPACE
    K_PACKED = tkl.sym.K_PACKED
    K_SCALE_SHUFFLED = tkl.sym.K_SCALE_SHUFFLED

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [tkw.WaveConstraint(M, BLOCK_M / wave_shape[0])]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / wave_shape[1])]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    if reorder_workgroups:
        new_wg0, new_wg1 = _reorder_mxfp4_workgroups(
            M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_N
        )
        constraints += [tkw.ReorderingConstraint(new_wg0, 0)]
        constraints += [tkw.ReorderingConstraint(new_wg1, 1)]

    # --- B data preshuffle mapping (aiter shuffle_weight) ---
    # Each 16-row x 32-byte tile is reordered from [n, k_sub, k_elem] to
    # [k_sub, n, k_elem] so a contiguous 256-byte read fetches one K-chunk
    # for all 16 N-rows.
    n_it = tkw.IndexMapping.iterator(0)
    k_it = tkw.IndexMapping.iterator(1)

    within_nblk = (
        (k_it // 32) * 512 + ((k_it // 16) % 2) * 256 + (n_it % 16) * 16 + k_it % 16
    )

    b_preshuffle_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: (n_it // 16) * 16 + within_nblk // K_PACKED,
            K: within_nblk % K_PACKED,
        },
        outputs={N: n_it, K: k_it},
    )

    # --- A scale preshuffle mapping (e8m0_shuffle) ---
    # Maps logical (K/32, M) scale coordinates to the shuffled physical layout.
    # Same e8m0_shuffle permutation as B scale but over the M dimension.
    i_a = tkw.IndexMapping.iterator(0)
    j_a = tkw.IndexMapping.iterator(1)

    a_scale_flat = (
        (j_a // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
        + (i_a // 8) * 256
        + ((i_a % 8) % 4) * 64
        + ((j_a % 32) % 16) * 4
        + (((i_a % 8) // 4) * 2)
        + ((j_a % 32) // 16)
    )

    a_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            M: a_scale_flat // K_SCALE_SHUFFLED,
            K: a_scale_flat % K_SCALE_SHUFFLED,
        },
        outputs={K: i_a, M: j_a},
    )

    # --- B scale preshuffle mapping (e8m0_shuffle) ---
    # Maps logical (N, K/32) scale coordinates to the shuffled physical layout.
    # The e8m0_shuffle does:
    #   view(N//32, 2, 16, Ks//8, 2, 4).permute(0,3,5,2,4,1)
    # where Ks = K_SCALE_SHUFFLED = ceil(K/32, 8).
    k_s = tkw.IndexMapping.iterator(0)
    n_s = tkw.IndexMapping.iterator(1)

    b_scale_flat = (
        (n_s // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
        + (k_s // 8) * 256
        + ((k_s % 8) % 4) * 64
        + ((n_s % 32) % 16) * 4
        + (((k_s % 8) // 4) * 2)
        + ((n_s % 32) // 16)
    )

    b_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: b_scale_flat // K_SCALE_SHUFFLED,
            K: b_scale_flat % K_SCALE_SHUFFLED,
        },
        outputs={K: k_s, N: n_s},
    )

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, A_ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, A_ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, GLOBAL_ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, C_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn, tag="bitcast_a")
            a_scale_reg = tkw.read(a_scale, mapping=a_scale_mapping, tag="read_a_scale")
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu, tag="bitcast_a_scale")
            b_reg = tkw.read(b, mapping=b_preshuffle_mapping, tag="read_b")
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn, tag="bitcast_b")
            b_scale_reg = tkw.read(b_scale, mapping=b_scale_mapping, tag="read_b_scale")
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu, tag="bitcast_b_scale")
            acc = tkw.scaled_mma(
                a_reg, a_scale_reg, b_reg, b_scale_reg, acc, tag="scaled_mma"
            )
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        A_ADDRESS_SPACE: a_address_space,
        C_ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: block_shape[0],
        BLOCK_N: block_shape[1],
        BLOCK_K: block_shape[2],
        GROUP_SIZE_N: group_size_n,
        M: shape[0],
        N: shape[1],
        K: shape[2],
        K_PACKED: shape[2] // 2,
        K_SCALE_SHUFFLED: (((shape[2] // 32) + 7) // 8) * 8,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_global_to_shared=True,
        minimize_shared_allocs=False,
    )

    return gemm, options


def _reorder_mxfp4_workgroups(m, n, block_m, block_n, group_size_n):
    """Remap workgroup indices to a new order based on group_size_n along N dimension.

    Example (3x5 grid, group_size_n=2): column-major dispatch order becomes
    full groups of 2 along N, then tail:
      0  3  6  9 12       |0 1| | 6  7| 12
      1  4  7 10 13  -->  |2 3| | 8  9| 13
      2  5  8 11 14       |4 5| |10 11| 14

    Args:
        m: Problem dimension M.
        n: Problem dimension N.
        block_m: Tile size along M dimension.
        block_n: Tile size along N dimension.
        group_size_n: Number of N-tiles per group.

    Returns:
        (new_wg0, new_wg1): New workgroup indices along M and N dimensions.
    """
    wg0, wg1 = WORKGROUP_0, WORKGROUP_1
    num_wg_0 = ceiling(m / block_m)
    num_wg_1 = ceiling(n / block_n)

    # Flatten in column-major order
    flat_wg_index = wg0 + wg1 * num_wg_0
    group_index = flat_wg_index // group_size_n

    # Main case, forming full groups of GROUP_SIZE_N tiles along N
    main_new_wg0 = group_index % num_wg_0
    main_new_wg1 = (
        group_index // num_wg_0
    ) * group_size_n + flat_wg_index % group_size_n

    # Tailing case, when N tiles is not a multiple of GROUP_SIZE_N
    full_tiles_n = floor(num_wg_1 / group_size_n) * group_size_n
    tail_tiles_n = num_wg_1 - full_tiles_n
    total_full = full_tiles_n * num_wg_0
    tail_linear = flat_wg_index - total_full
    tail_new_wg0 = tail_linear // Max(1, tail_tiles_n)
    tail_new_wg1 = full_tiles_n + tail_linear % Max(1, tail_tiles_n)

    # Select tail path if we can no longer form full groups
    new_wg0 = Piecewise(
        (tail_new_wg0, (flat_wg_index >= total_full) & (tail_tiles_n > 0)),
        (main_new_wg0, True),
    )
    new_wg1 = Piecewise(
        (tail_new_wg1, (flat_wg_index >= total_full) & (tail_tiles_n > 0)),
        (main_new_wg1, True),
    )

    return new_wg0, new_wg1
