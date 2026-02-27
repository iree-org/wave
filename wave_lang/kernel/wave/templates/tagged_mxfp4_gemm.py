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

from math import ceil
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
        reorder_workgroups: Enable N-dim workgroup reordering. When True,
        compute_best_group_size_n() is called to auto-select the optimal
        group size and decide whether reordering is actually beneficial.
        group_size_n: Number of N-tiles per reordering group.

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    # Auto-select group_size_n (and whether reordering helps) if not specified.
    if reorder_workgroups:
        group_size_n, reorder_workgroups = compute_best_group_size_n(
            shape[0], shape[1], shape[2], block_shape[0], block_shape[1]
        )
        print(
            f"[workgroup_reorder] enabled={reorder_workgroups}, group_size_n={group_size_n}"
        )

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
        reorder_workgroups: Enable N-dim workgroup reordering. When True,
        compute_best_group_size_n() is called to auto-select the optimal
        group size and decide whether reordering is actually beneficial.
        group_size_n: Number of N-tiles per reordering group.

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    # Auto-select group_size_n (and whether reordering helps) if not specified.
    if reorder_workgroups:
        group_size_n, reorder_workgroups = compute_best_group_size_n(
            shape[0], shape[1], shape[2], block_shape[0], block_shape[1]
        )
        print(
            f"[workgroup_reorder] enabled={reorder_workgroups}, group_size_n={group_size_n}"
        )

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


def compute_best_group_size_n(
    M: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    num_xcds: int = 8,
    cus_per_xcd: int = 32,
) -> tuple[int, bool]:
    """Auto-select group_size_n and decide whether N-dim reordering is beneficial.

    Dispatch model (MI300X / MI350):
        Hardware assigns flat workgroup indices round-robin to XCDs.
        Each XCD runs cus_per_xcd CUs in parallel, forming a "batch" of
        cus_per_xcd concurrent workgroups.

        Each batch covers U_A unique M-tiles × U_B unique N-tiles.
        Per K-iteration DRAM fetches = U_A + U_B.
        Minimise U_A + U_B subject to U_A × U_B ≈ cus_per_xcd (= 32).
        Optimal: (U_A, U_B) = (4, 8) or (8, 4) → sum = 12.

        WITHOUT N-reordering:
            U_B_natural ≈ (cus_per_xcd × num_xcds) / num_wg_0 = 256 / num_wg_0
            sum_natural  = U_A_natural + U_B_natural

        WITH N-reordering (group_size_n = gsn, multiple of num_xcds):
            U_B = gsn / num_xcds
            (cost function) sum_gsn = cus_per_xcd × num_xcds / gsn + gsn / num_xcds
                     = 256 / gsn + gsn / 8

        Optimal gsn (derivation from cost function set to zero and solved for gsn)
        ≈ num_xcds × √cus_per_xcd ≈ 45 → closest power of two: gsn=32 and gsn=64

    Worked examples (block_m = block_n = 256, MI300X defaults):

        Shape (M, N)      num_wg_0  U_B_natural  sum_natural  best_gsn  enable
        (4096,   57344)      16          16            18         32      YES  ← num_wg_0 < 32
        (8192,   57344)      32           8            12         --       NO  ← already optimal
        (16384,  16384)      64           4            12         --       NO  ← already optimal
        (32768,  16384)     128           2            18         64      YES  ← num_wg_0 > 64

    group_size_n selection:
        Both gsn=32 (U_A=8, U_B=4) and gsn=64 (U_A=4, U_B=8) achieve sum=12.
        Tie-breaking:
          • Exact divisors of num_wg_1 are preferred (no tail group).
          • B-heavy shapes (num_wg_1 >= num_wg_0): prefer gsn=32 (lower U_B →
            more concurrent B sharing per batch).
          • A-heavy shapes (num_wg_0 > num_wg_1): prefer gsn=64 (lower U_A →
            more concurrent A sharing per batch).

    Args:
        M, N, K:          Problem dimensions (K is accepted for API consistency
                          but does not affect the batch balance model).
        block_m, block_n: Tile sizes along M and N.
        num_xcds:         XCD count (MI300X / MI350: 8).
        cus_per_xcd:      CUs per XCD (MI300X / MI350: 32).

    Returns:
        (best_group_size_n, reorder_enabled)
        reorder_enabled=False means column-major dispatch already achieves the
        optimal batch balance (sum=12); best_group_size_n is still returned
        (32) as a safe default.
    """
    num_wg_0 = ceil(M / block_m)  # M-tiles
    num_wg_1 = ceil(N / block_n)  # N-tiles

    candidates = [g for g in (16, 32, 64) if g % num_xcds == 0 and g <= num_wg_1]
    if not candidates:
        return num_xcds, False

    def ub(g: int) -> int:
        return g // num_xcds

    def ua(g: int) -> int:
        return cus_per_xcd // max(1, ub(g))

    def gsn_sum(g: int) -> int:
        return ua(g) + ub(g)

    # Natural batch composition (no reordering)
    u_b_nat = max(1, min(cus_per_xcd, cus_per_xcd * num_xcds // max(1, num_wg_0)))
    u_a_nat = max(1, cus_per_xcd // u_b_nat)
    sum_natural = u_a_nat + u_b_nat

    best_sum = min(gsn_sum(g) for g in candidates)
    reorder_enabled = best_sum < sum_natural

    if not reorder_enabled:
        return 32, False

    optimal = [g for g in candidates if gsn_sum(g) == best_sum]
    exact = [g for g in optimal if num_wg_1 % g == 0]
    pool = exact if exact else optimal

    # Tie-break: B-heavy → smaller gsn (more B sharing); A-heavy → larger gsn.
    return (min(pool) if num_wg_1 >= num_wg_0 else max(pool)), True


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
