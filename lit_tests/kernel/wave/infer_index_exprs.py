# REQUIRES: water
# RUN: python %s
# The point of this test is to avoid crashing or asserting, so just run it under lit.

# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

from wave_lang.kernel.wave.templates import AttentionShape
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config


def _get_matrix_add_kernel():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ADDRESS_SPACE = tkl.sym.GLOBAL_ADDRESS_SPACE
    dtype = tkl.f16

    constraints = [
        tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 4, N: 1}),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
    ]

    @tkw.wave(constraints)
    def matrix_add(
        a: tkl.Memory[M, N, ADDRESS_SPACE, dtype],
        b: tkl.Memory[M, N, ADDRESS_SPACE, dtype],
        c: tkl.Memory[M, N, ADDRESS_SPACE, dtype],
    ):
        a_reg = tkw.read(a)
        b_reg = tkw.read(b)
        c_reg = a_reg + b_reg
        tkw.write(c_reg, c)

    hyperparams = {
        M: 128,
        N: 128,
        BLOCK_M: 16,
        BLOCK_N: 16,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
    }
    return matrix_add, hyperparams


def _get_mma_chain_kernel():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    P = tkl.sym.P
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    dtype = tkl.f16

    constraints = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=MMAType.F32_16x16x16_F16,
            waves_per_block=(1, 2, 2),
        ),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
    ]

    @tkw.wave(constraints)
    def mma_chain(
        a: tkl.Memory[M, K, GLOBAL_ADDRESS_SPACE, dtype],
        b: tkl.Memory[N, K, GLOBAL_ADDRESS_SPACE, dtype],
        c: tkl.Memory[M, P, GLOBAL_ADDRESS_SPACE, tkl.f32],
        d: tkl.Memory[P, N, GLOBAL_ADDRESS_SPACE, dtype],
        storage: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, dtype],
    ):
        a_read = tkw.read(a)
        b_read = tkw.read(b)
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        mma1 = tkw.mma(a_read, b_read, c_reg)
        mma1_casted = tkw.cast(mma1, tkl.f16)
        tkw.write(mma1_casted, storage)
        reloaded = tkw.read(storage)
        d_read = tkw.read(d)
        c_reg2 = tkl.Register[M, P, tkl.f32](0.0)
        mma2 = tkw.mma(reloaded, d_read, c_reg2)
        tkw.write(mma2, c)

    hyperparams = {
        M: 128,
        N: 128,
        K: 128,
        P: 128,
        BLOCK_M: 16,
        BLOCK_N: 16,
    }
    return mma_chain, hyperparams


def testMatrixAdd():
    kernel, params = _get_matrix_add_kernel()
    options = WaveCompileOptions(
        subs=params,
        run_bench=False,
        check_water_analysis=True,
    )
    compiled_kernel = wave_compile(options, kernel)
    assert compiled_kernel is not None


def testGemm():
    relevant_hyperparams = [
        tkl.sym.M,
        tkl.sym.N,
        tkl.sym.K,
        tkl.sym.BLOCK_M,
        tkl.sym.BLOCK_N,
        tkl.sym.BLOCK_K,
        tkl.sym.ADDRESS_SPACE,
    ]

    for use_shmem in [True, False]:
        for mfma_variant, target in [
            (MMAType.F32_32x32x16_F16, "gfx950"),
            (MMAType.F32_16x16x16_F16, "gfx942"),
        ]:
            print(f"Testing {mfma_variant} on {target} with LDS={use_shmem}")
            gemm, hyperparams, _ = get_gemm_kernel(
                shape=(1024, 1024, 1024), dynamic_dims=False, mfma_variant=mfma_variant
            )

            # Override usage of shared memory if not requested as the template always uses it.
            if not use_shmem:
                hyperparams[tkl.sym.ADDRESS_SPACE] = GLOBAL_ADDRESS_SPACE
            # Avoid unused hyperparameter warnings
            hyperparams = {
                s: v for s, v in hyperparams.items() if s in relevant_hyperparams
            }
            options = WaveCompileOptions(
                subs=hyperparams,
                run_bench=False,
                check_water_analysis=True,
                target=target,
            )
            compiled_gemm = wave_compile(options, gemm)
            assert compiled_gemm is not None


def testMmaChain():
    kernel, params = _get_mma_chain_kernel()
    options = WaveCompileOptions(
        subs=params,
        run_bench=False,
        check_water_analysis=True,
    )
    compiled_kernel = wave_compile(options, kernel)
    assert compiled_kernel is not None


def testAttention():
    attention, hyperparams, _ = get_vanilla_attention_kernel(
        AttentionShape(
            num_query_heads=8,
            num_kv_heads=2,
            query_seq_len=256,
            head_size_kv=64,
            head_size=64,
            kv_seq_len=256,
        ),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        dynamic_dims=False,
    )

    options_mlir = WaveCompileOptions(
        subs=hyperparams,
        run_bench=False,
        # check_water_analysis=True,
        print_mlir_before_water=True,
        print_ir_after="all",
        # TODO(#982): this pass creates IR that appears malformed, though pywave
        # manages to execute it.
        enable_mark_hardware_transpose_candidates=False,
    )
    options_mlir = set_default_run_config(options_mlir)
    compiled_kernel = wave_compile(options_mlir, attention)
    assert compiled_kernel is not None


if __name__ == "__main__":
    testAttention()
    testMatrixAdd()
    testMmaChain()
    testGemm()
