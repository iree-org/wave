# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tagged MXFP4 Scaled GEMM kernel template for CDNA4 (GFX950).

All ops are tagged for use with MXFP4 schedule functions (e.g. get_mxfp4_dbuf_schedule).

Required tags: k_loop, read_a, read_a_scale, read_b, read_b_scale,
bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale, scaled_mma.
"""

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
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    num_waves: int = 8,
):
    """Return a tagged MXFP4 scaled GEMM kernel + compile options for CDNA4.

    All ops are tagged for use with MXFP4 schedule functions.

    Args:
        shape: (M, N, K) problem dimensions.
        block_shape: (BLOCK_M, BLOCK_N, BLOCK_K) tile sizes.
        mfma_variant: Scaled MMA instruction type.
        num_waves: Waves per workgroup (4 or 8).

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    if num_waves == 8:
        # 8 waves: 4 M-tiles x 2 N-tiles
        constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    else:
        # 4 waves: 2 M-tiles x 2 N-tiles
        constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
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
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
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


def get_preshuffle_kernel(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block_shape: tuple[int, int, int] = (256, 256, 256),
):
    """Return the pre-shuffled MXFP4 GEMM kernel definition with IndexMapping for shuffled scales."""
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Calculate shuffled dimensions for pre-shuffle kernel
    k_scale_shuffled = (((shape[2] // 32) + 7) // 8) * 8
    K_SCALE_SHUFFLED = tkl.sym.K_SCALE_SHUFFLED

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 4),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=ScaledMMAType.F32_16x16x128_F8F6F4,
        ),
    ]

    # Create IndexMapping for shuffled A scales
    # The e8m0_shuffle coordinate transformation maps logical (K, M) iterators
    # to physical shuffled memory layout
    i = tkw.IndexMapping.iterator(0)  # K iterator
    j = tkw.IndexMapping.iterator(1)  # M iterator

    a_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            M: (
                (
                    (j // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                    + (i // 8) * 256
                    + ((i % 8) % 4) * 64
                    + ((j % 32) % 16) * 4
                    + (((i % 8) // 4) * 2)
                    + ((j % 32) // 16)
                )
                // K_SCALE_SHUFFLED
            ),
            K: (
                (
                    (j // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                    + (i // 8) * 256
                    + ((i % 8) % 4) * 64
                    + ((j % 32) % 16) * 4
                    + (((i % 8) // 4) * 2)
                    + ((j % 32) // 16)
                )
                % K_SCALE_SHUFFLED
            ),
        },
        outputs={
            K: i,
            M: j,
        },
    )

    # Create IndexMapping for shuffled B scales
    k = tkw.IndexMapping.iterator(0)  # K iterator
    n = tkw.IndexMapping.iterator(1)  # N iterator

    b_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: (
                (
                    (n // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                    + (k // 8) * 256
                    + ((k % 8) % 4) * 64
                    + ((n % 32) % 16) * 4
                    + (((k % 8) // 4) * 2)
                    + ((n % 32) // 16)
                )
                // K_SCALE_SHUFFLED
            ),
            K: (
                (
                    (n // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                    + (k // 8) * 256
                    + ((k % 8) % 4) * 64
                    + ((n % 32) % 16) * 4
                    + (((k % 8) // 4) * 2)
                    + ((n % 32) // 16)
                )
                % K_SCALE_SHUFFLED
            ),
        },
        outputs={
            K: k,
            N: n,
        },
    )

    @tkw.wave(constraints)
    def mxfp4_gemm_preshuffle(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn, tag="bitcast_a")
            a_scale_reg = tkw.read(a_scale, mapping=a_scale_mapping, tag="read_a_scale")
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu, tag="bitcast_a_scale")

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
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_shape[0],
        BLOCK_N: block_shape[1],
        BLOCK_K: block_shape[2],
        M: shape[0],
        N: shape[1],
        K: shape[2],
        K_SCALE_SHUFFLED: k_scale_shuffled,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_global_to_shared=True,
        minimize_shared_allocs=True,
    )
    return mxfp4_gemm_preshuffle, options
