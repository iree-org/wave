# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
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
from typing import Sequence
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from torch.testing import assert_close
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)


enable_scheduling = SchedulingType.PREFETCH
shape = (4096, 4096, 4096)
mfma_variant = MMAType.RDNA4_WAVE32_F32_16x16x16_F16
threads_per_wave = 32
datatype = torch.float16
dynamic_dims = False
run_bench = False


def gemm_kernel(
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
    dtype = torch_dtype_to_wave(dtype)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=threads_per_wave, mma_type=mfma_variant)
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
        BLOCK_M: 128,
        BLOCK_N: 256,
        BLOCK_K: 64,
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


def testPureGemm():
    gemm, hyperparams, dynamic_symbols = gemm_kernel(
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

    # iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    # generate_iree_ref("mmt", [a, b], [iree_ref], options)
    # assert_close(c, iree_ref, check_device=False, equal_nan=True)


if __name__ == "__main__":
    testPureGemm()
