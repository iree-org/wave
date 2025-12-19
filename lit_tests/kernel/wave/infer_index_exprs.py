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
from wave_lang.kernel.wave.wave import LaunchableWave
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.utils.general_utils import torch_dtype_to_wave

import torch


# TODO: use the generic template, currently blocked by water not handling wave constraints.
def _get_gemm_kernel(
    shape: tuple[int, int, int],
    mfma_variant: MMAType,
    dtype: torch.dtype = torch.float16,
    block_shape: tuple[int, int, int] | None = None,
    waves_per_block: tuple[int, int] | None = None,
) -> tuple[LaunchableWave, dict[tkl.IndexSymbol, tkl.IndexExpr]]:
    if not block_shape:
        # BLOCK_M, BLOCK_N, BLOCK_K
        block_shape = (64, 64, 32)

    if not waves_per_block:
        # WAVE_M, WAVE_N
        waves_per_block = (2, 2)

    assert len(block_shape) == 3, "block_shape needs to be rank 3 for M, N, K."
    assert len(waves_per_block) == 2, "waves_per_block needs to be rank 2 for M, N."

    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.GLOBAL_ADDRESS_SPACE
    dtype = torch_dtype_to_wave(dtype)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    # TODO: dialect expects waves_per_block to be rank 3, so we append a 1 to the end.
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            waves_per_block=waves_per_block + (1,),
        )
    ]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, GLOBAL_ADDRESS_SPACE, dtype],
        b: tkl.Memory[N, K, GLOBAL_ADDRESS_SPACE, dtype],
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
    return gemm, hyperparams


def testGemm():
    gemm, hyperparams = _get_gemm_kernel(
        shape=(1024, 1024, 1024), mfma_variant=MMAType.F32_16x16x16_F16
    )
    options = WaveCompileOptions(
        subs=hyperparams,
        run_bench=False,
        check_water_analysis=True,
        print_mlir_after_water=True,
    )
    compiled_gemm = wave_compile(options, gemm)
    assert compiled_gemm is not None


if __name__ == "__main__":
    testGemm()
