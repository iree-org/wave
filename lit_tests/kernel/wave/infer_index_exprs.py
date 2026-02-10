# REQUIRES: water
# RUN: python %s
# The point of this test is to avoid crashing or asserting, so just run it under lit.

# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType


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


if __name__ == "__main__":
    testGemm()
