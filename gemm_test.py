# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from torch.testing import assert_close


enable_scheduling = SchedulingType.PREFETCH
shape = (4096, 4096, 4096)
mfma_variant = MMAType.RDNA4_WAVE32_F32_16x16x16_F16
threads_per_wave = 32
datatype = torch.float16
dynamic_dims = False
run_bench = False


def testPureGemm():
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape, dynamic_dims, mfma_variant, datatype, threads_per_wave=threads_per_wave
    )

    multibuffer = enable_scheduling in [
        SchedulingType.FOUR_STAGE,
        SchedulingType.MODULO,
    ]
    UNROLL_FACTOR = tkl.sym.UNROLL_FACTOR
    hyperparams[UNROLL_FACTOR] = 2 if multibuffer else 1

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        # print_mlir=True
        # dump_intermediates="./dump/intermediates-pp/"
    )
    options.postprocess = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
            %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.loop.unroll %0 { factor = %%UNROLL_FACTOR%% } : !transform.any_op
            transform.yield
        }
    }
    """
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=datatype)
    b = device_randn(shape[1], shape[2], dtype=datatype)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False, equal_nan=True)


if __name__ == "__main__":
    testPureGemm()
