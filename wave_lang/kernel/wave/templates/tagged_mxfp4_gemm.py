# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tagged MXFP4 Scaled GEMM kernel templates for CDNA4 (GFX950).

All ops are tagged for use with MXFP4 schedule functions (e.g. get_mxfp4_dbuf_schedule).

Provides:
  - get_tagged_mxfp4_gemm:                              vanilla (A, B via LDS)
  - get_tagged_mxfp4_gemm_preshuffle_b:                 B + B_scale preshuffled (direct global reads)
  - get_tagged_splitk_mxfp4_gemm:                       split-K (A, B, scales via LDS)
  - get_tagged_splitk_mxfp4_gemm_preshuffle_scales:     split-K with preshuffled scales
  - get_tagged_splitk_mxfp4_gemm_preshuffle_b:          split-K with preshuffled B + scales

Required tags: k_loop, read_a, read_a_scale, read_b, read_b_scale,
bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale, scaled_mma.
"""

import math

import sympy
from sympy import Eq, Piecewise, ceiling, floor, Max

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
    output_dtype=tkl.f32,
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
    assert output_dtype in [
        tkl.f32,
        tkl.bf16,
    ], f"Unsupported output dtype: {output_dtype}"

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
        c: tkl.Memory[M, N, C_ADDRESS_SPACE, output_dtype],
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

        if output_dtype == tkl.bf16:
            repeat = tkw.cast(repeat, tkl.bf16)
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


def _get_tagged_mxfp4_gemm_preshuffle_scales_impl(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    wave_shape: tuple[int, int],
    mfma_variant: ScaledMMAType,
    a_address_space: tkl.AddressSpace,
    b_address_space: tkl.AddressSpace | None = None,
    *,
    b_preshuffled: bool = False,
    reorder_workgroups: bool = False,
    group_size_n=32,
):
    """Shared implementation: preshuffle scales only, or scales + B data.

    When b_preshuffled is False: B uses the regular (non-preshuffled) read path.
    When b_preshuffled is True: B reads use the preshuffle mapping.

    Whether A/B are read directly from global to VGPR or staged through shared memory
    is controlled by the selected address spaces (`a_address_space` and
    `b_address_space`).

    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    GROUP_SIZE_N = tkl.sym.GROUP_SIZE_N
    A_ADDRESS_SPACE = tkl.sym.A_ADDRESS_SPACE
    B_ADDRESS_SPACE = tkl.sym.B_ADDRESS_SPACE
    C_ADDRESS_SPACE = tkl.sym.C_ADDRESS_SPACE
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

    if b_preshuffled:
        K_PACKED = tkl.sym.K_PACKED
        # --- B data preshuffle mapping (aiter shuffle_weight) ---
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
        a_scale: tkl.Memory[M, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, B_ADDRESS_SPACE, tkl.i8],
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
            if b_preshuffled:
                b_reg = tkw.read(b, mapping=b_preshuffle_mapping, tag="read_b")
            else:
                b_reg = tkw.read(b, tag="read_b")
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
        B_ADDRESS_SPACE: (
            b_address_space
            if b_address_space is not None
            else (GLOBAL_ADDRESS_SPACE if b_preshuffled else a_address_space)
        ),
        C_ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: block_shape[0],
        BLOCK_N: block_shape[1],
        BLOCK_K: block_shape[2],
        GROUP_SIZE_N: group_size_n,
        M: shape[0],
        N: shape[1],
        K: shape[2],
        K_SCALE_SHUFFLED: (((shape[2] // 32) + 7) // 8) * 8,
    }
    if b_preshuffled:
        hyperparams[K_PACKED] = K // 2

    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_global_to_shared=True,
        minimize_shared_allocs=False,
    )
    return gemm, options


def get_tagged_mxfp4_gemm_preshuffle_scales(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block_shape: tuple[int, int, int] = (256, 256, 256),
    wave_shape: tuple[int, int] = (2, 2),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    a_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
):
    """Return a tagged MXFP4 scaled GEMM kernel with preshuffled B and B_scale.

    A and B are loaded from global to shared.
    A_scales and B_scales are read from global memory using an e8m0 scale preshuffle mapping and directly stored to VGPRs.

    All ops are tagged for use with MXFP4 schedule functions.

    Args:
        shape: (M, N, K) problem dimensions.
        block_shape: (BLOCK_M, BLOCK_N, BLOCK_K) tile sizes.
        wave_shape: (WAVE_M, WAVE_N) waves per workgroup.
        mfma_variant: Scaled MMA instruction type.
        a_address_space: Address space for A.

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    return _get_tagged_mxfp4_gemm_preshuffle_scales_impl(
        shape,
        block_shape,
        wave_shape,
        mfma_variant,
        a_address_space,
        b_preshuffled=False,
    )


def get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block_shape: tuple[int, int, int] = (256, 256, 256),
    wave_shape: tuple[int, int] = (2, 2),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    a_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
    b_address_space: tkl.AddressSpace | None = None,
):
    """Return a tagged MXFP4 scaled GEMM kernel with preshuffled B and B_scale.

    You can specify the address space to which A and B are loaded.
    A and B scales are read from global memory using an e8m0 scale preshuffle mapping and directly stored to VGPRs.

    All ops are tagged for use with MXFP4 schedule functions.

    Args:
        shape: (M, N, K) problem dimensions.
        block_shape: (BLOCK_M, BLOCK_N, BLOCK_K) tile sizes.
        wave_shape: (WAVE_M, WAVE_N) waves per workgroup.
        mfma_variant: Scaled MMA instruction type.
        a_address_space: Address space for A.
        b_address_space: Address space for B.
    Returns:
        (kernel_function, WaveCompileOptions)
    """
    return _get_tagged_mxfp4_gemm_preshuffle_scales_impl(
        shape,
        block_shape,
        wave_shape,
        mfma_variant,
        a_address_space,
        b_address_space,
        b_preshuffled=True,
    )


def get_tagged_mxfp4_gemm_preshuffle_b(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block_shape: tuple[int, int, int] = (256, 256, 256),
    wave_shape: tuple[int, int] = (2, 2),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    a_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
    a_scale_preshuffle: bool = True,
    reorder_workgroups=True,
    group_size_n=32,
    output_dtype=tkl.f32,
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

    # Divisibility assumptions for M, N, K (no effect for static shapes).
    constraints += [tkw.Assumption(Eq(M % 32, 0))]
    constraints += [tkw.Assumption(Eq(N % 32, 0))]
    constraints += [tkw.Assumption(Eq(K % 256, 0))]

    # K is always large enough for software pipelining.
    constraints += [tkw.Assumption(K > BLOCK_K * 6)]

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

    if a_scale_preshuffle:
        a_scale_mapping = tkw.IndexMapping(
            num_iterators=2,
            inputs={
                M: a_scale_flat // K_SCALE_SHUFFLED,
                K: a_scale_flat % K_SCALE_SHUFFLED,
            },
            outputs={K: i_a, M: j_a},
        )
    else:
        a_scale_mapping = None

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
        c: tkl.Memory[M, N, C_ADDRESS_SPACE, output_dtype],
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

        if output_dtype == tkl.bf16:
            repeat = tkw.cast(repeat, tkl.bf16)
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
        K_PACKED: K // 2,
        K_SCALE_SHUFFLED: (((K // 32) + 7) // 8) * 8,
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


def _get_tagged_splitk_mxfp4_gemm_impl(
    shape: tuple[int, int, int],
    num_splits: int,
    block_shape: tuple[int, int, int],
    wave_shape: tuple[int, int],
    mfma_variant: ScaledMMAType,
    a_address_space: tkl.AddressSpace,
    *,
    preshuffle_scales: bool = False,
    preshuffle_B: bool = False,
    output_type: "tkl.DataType" = tkl.f32,
):
    """Shared implementation for tagged split-K MXFP4 GEMM kernels.

    When preshuffle_scales is False and preshuffle_B is False:
        all data and scales go through shared memory (LDS).
    When preshuffle_scales is True:
        A and B data go through shared memory; A and B scales are read
        from global memory using e8m0 preshuffle mappings directly to VGPRs.
    When preshuffle_B is True:
        A and A_scale go through shared memory (A_scale with e8m0 mapping).
        B data is preshuffled and read from global.
        B_scale uses e8m0 preshuffle mapping from global.
    """
    m, n, k = shape
    k_per_split = math.ceil(k / num_splits)
    if k_per_split < block_shape[2]:
        raise ValueError(
            f"K per split ({k_per_split}) is less than BLOCK_K ({block_shape[2]}). "
            f"Reduce num_splits or BLOCK_K so that each split has at least BLOCK_K elements."
        )

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    S = tkl.sym.S
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    BLOCK_S = tkl.sym.BLOCK_S
    K_SPLIT_OFF = tkl.sym.K_SPLIT_OFF
    K_SPLIT_LEN = tkl.sym.K_SPLIT_LEN
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    B_ADDRESS_SPACE = tkl.sym.B_ADDRESS_SPACE
    K_PACKED = tkl.sym.K_PACKED
    K_SCALE_SHUFFLED = tkl.sym.K_SCALE_SHUFFLED

    k_packed_val = k // 2
    k_scale_shuffled_val = (((k // 32) + 7) // 8) * 8

    b_preshuffle_mapping = None
    if preshuffle_B:
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

    a_scale_mapping = None
    b_scale_mapping = None
    needs_scale_mappings = preshuffle_scales or preshuffle_B
    if needs_scale_mappings:
        i = tkw.IndexMapping.iterator(0)
        j = tkw.IndexMapping.iterator(1)
        _flat_a = (
            (j // 32) * ((k_scale_shuffled_val // 8) * 256)
            + (i // 8) * 256
            + ((i % 8) % 4) * 64
            + ((j % 32) % 16) * 4
            + (((i % 8) // 4) * 2)
            + ((j % 32) // 16)
        )
        a_scale_mapping = tkw.IndexMapping(
            num_iterators=2,
            inputs={
                M: _flat_a // k_scale_shuffled_val,
                K: _flat_a % k_scale_shuffled_val,
            },
            outputs={K: i, M: j},
        )
        kk = tkw.IndexMapping.iterator(0)
        n_s = tkw.IndexMapping.iterator(1)
        _flat_b = (
            (n_s // 32) * ((k_scale_shuffled_val // 8) * 256)
            + (kk // 8) * 256
            + ((kk % 8) % 4) * 64
            + ((n_s % 32) % 16) * 4
            + (((kk % 8) // 4) * 2)
            + ((n_s % 32) // 16)
        )
        b_scale_mapping = tkw.IndexMapping(
            num_iterators=2,
            inputs={
                N: _flat_b // k_scale_shuffled_val,
                K: _flat_b % k_scale_shuffled_val,
            },
            outputs={K: kk, N: n_s},
        )

    # preshuffle_scales: both scale reads bypass LDS, go directly from global.
    # preshuffle_B: A_scale stays in LDS (with e8m0 mapping applied at the
    #   shared-memory read), B_scale reads from global (like the non-splitk
    #   preshuffle_b template).
    if preshuffle_scales:
        a_scale_space = GLOBAL_ADDRESS_SPACE
    else:
        a_scale_space = ADDRESS_SPACE
    b_scale_space = (
        GLOBAL_ADDRESS_SPACE if (preshuffle_scales or preshuffle_B) else ADDRESS_SPACE
    )

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(S, BLOCK_S, 2),
        tkw.TilingConstraint(
            K,
            BLOCK_K,
            iters=sympy.ceiling(K_SPLIT_LEN / BLOCK_K),
            start=K_SPLIT_OFF,
        ),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / wave_shape[0])),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / wave_shape[1])),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={S: 0},
        ),
    ]

    @tkw.wave(constraints)
    def splitk_gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, a_scale_space, tkl.i8],
        b: tkl.Memory[N, K / 2, B_ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, b_scale_space, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, output_type],
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

        repeat_out = tkw.cast(repeat, output_type)
        tkw.atomic_add(repeat_out, c)

    hyperparams = {
        ADDRESS_SPACE: a_address_space,
        B_ADDRESS_SPACE: (GLOBAL_ADDRESS_SPACE if preshuffle_B else a_address_space),
        BLOCK_M: block_shape[0],
        BLOCK_N: block_shape[1],
        BLOCK_K: block_shape[2],
        BLOCK_S: 1,
        M: m,
        N: n,
        K: k,
        S: num_splits,
        K_SPLIT_OFF: WORKGROUP_2 * k_per_split,
        K_SPLIT_LEN: sympy.Min(K, (WORKGROUP_2 + 1) * k_per_split) - K_SPLIT_OFF,
    }
    for key, value in hyperparams.items():
        if isinstance(value, sympy.Expr):
            hyperparams[key] = value.subs(hyperparams)

    if preshuffle_B:
        hyperparams[K_PACKED] = k_packed_val
    if preshuffle_scales or preshuffle_B:
        hyperparams[K_SCALE_SHUFFLED] = k_scale_shuffled_val

    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_global_to_shared=True,
        minimize_shared_allocs=False,
    )

    return splitk_gemm, options


def get_tagged_splitk_mxfp4_gemm(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    num_splits: int = 2,
    block_shape: tuple[int, int, int] = (128, 128, 256),
    wave_shape: tuple[int, int] = (2, 2),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    a_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
    output_type: "tkl.DataType" = tkl.f32,
):
    """Return a tagged split-K MXFP4 GEMM kernel + compile options.

    Split-K parallelizes the K dimension across multiple workgroups using
    atomic_add accumulation.  The caller must zero-initialize C before launch.

    All data and scales go through shared memory (LDS).
    All ops are tagged for use with MXFP4 schedule functions
    (e.g. get_mxfp4_dbuf_schedule).

    Args:
        shape: (M, N, K) problem dimensions.
        num_splits: Number of splits along the K dimension.
        block_shape: (BLOCK_M, BLOCK_N, BLOCK_K) tile sizes.
        wave_shape: (WAVE_M, WAVE_N) waves per workgroup.
        mfma_variant: Scaled MMA instruction type.
        a_address_space: Address space for A and A_scale (typically SHARED).
        output_type: Element type of output tensor C.

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    return _get_tagged_splitk_mxfp4_gemm_impl(
        shape,
        num_splits,
        block_shape,
        wave_shape,
        mfma_variant,
        a_address_space,
        preshuffle_scales=False,
        output_type=output_type,
    )


def get_tagged_splitk_mxfp4_gemm_preshuffle_scales(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    num_splits: int = 2,
    block_shape: tuple[int, int, int] = (128, 128, 256),
    wave_shape: tuple[int, int] = (2, 2),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    a_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
    output_type: "tkl.DataType" = tkl.f32,
):
    """Return a tagged split-K MXFP4 GEMM kernel with preshuffled scales.

    Split-K parallelizes the K dimension across multiple workgroups using
    atomic_add accumulation.  The caller must zero-initialize C before launch.

    A and B data go through shared memory (LDS).
    A and B scales are read from global memory using e8m0 preshuffle mappings
    directly to VGPRs.

    All ops are tagged for use with MXFP4 schedule functions
    (e.g. get_mxfp4_dbuf_pingpong_schedule).

    Args:
        shape: (M, N, K) problem dimensions.
        num_splits: Number of splits along the K dimension.
        block_shape: (BLOCK_M, BLOCK_N, BLOCK_K) tile sizes.
        wave_shape: (WAVE_M, WAVE_N) waves per workgroup.
        mfma_variant: Scaled MMA instruction type.
        a_address_space: Address space for A data (typically SHARED).
        output_type: Element type of output tensor C.

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    return _get_tagged_splitk_mxfp4_gemm_impl(
        shape,
        num_splits,
        block_shape,
        wave_shape,
        mfma_variant,
        a_address_space,
        preshuffle_scales=True,
        output_type=output_type,
    )


def get_tagged_splitk_mxfp4_gemm_preshuffle_b(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    num_splits: int = 2,
    block_shape: tuple[int, int, int] = (128, 128, 256),
    wave_shape: tuple[int, int] = (2, 2),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    a_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
    output_type: "tkl.DataType" = tkl.f32,
):
    """Return a tagged split-K MXFP4 GEMM kernel with preshuffled B and scales.

    Split-K parallelizes the K dimension across multiple workgroups using
    atomic_add accumulation.  The caller must zero-initialize C before launch.

    A goes through shared memory (LDS).  B data is preshuffled and read from
    global memory.  A and B scales use e8m0 preshuffle mappings from global.

    All ops are tagged for use with MXFP4 schedule functions
    (e.g. get_mxfp4_asymmetric_schedule).

    Args:
        shape: (M, N, K) problem dimensions.
        num_splits: Number of splits along the K dimension.
        block_shape: (BLOCK_M, BLOCK_N, BLOCK_K) tile sizes.
        wave_shape: (WAVE_M, WAVE_N) waves per workgroup.
        mfma_variant: Scaled MMA instruction type.
        a_address_space: Address space for A data (typically SHARED).
        output_type: Element type of output tensor C.

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    return _get_tagged_splitk_mxfp4_gemm_impl(
        shape,
        num_splits,
        block_shape,
        wave_shape,
        mfma_variant,
        a_address_space,
        preshuffle_B=True,
        output_type=output_type,
    )


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
