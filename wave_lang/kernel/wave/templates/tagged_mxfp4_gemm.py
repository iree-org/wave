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
    wave_shape: tuple[int, int] = (2, 2),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    a_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
    b_address_space: tkl.AddressSpace = SHARED_ADDRESS_SPACE,
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
