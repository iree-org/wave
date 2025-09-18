# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_randint,
    device_zeros,
    device_ones,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from tests.kernel.wave.common.utils import (
    require_e2e,
    require_cdna2,
    require_cdna3,
    require_cdna4,
    require_cdna_3_or_4,
    perf_test,
    param_bool,
)
from wave_lang.kernel.wave.constraints import MMAType, MMAOperand, GenericDot
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.lang import DataType
import os
import json
from torch.testing import assert_close

# Add test shapes for validation and performance testing.
default_test_shapes = {}
default_test_shapes["test_gemm"] = [
    (1024, 5120, 640),
    (2048, 10240, 1280),
    (4096, 20480, 2560),
]
default_test_shapes["test_gemm"] += [
    perf_test(x) for x in default_test_shapes["test_gemm"]
]
default_test_shapes["test_batched_gemm"] = [(8, 256, 128, 192), (32, 1024, 512, 768)]


user_specified_test_shapes = ""

test_params_path = os.environ.get("TEST_PARAMS_PATH", None)

if test_params_path:
    with open(test_params_path, "r") as file:
        user_specified_test_shapes = json.load(file)


def get_test_shapes(test_name: str) -> list[tuple[int]]:
    if test_name in user_specified_test_shapes:
        return user_specified_test_shapes[test_name]
    return default_test_shapes[test_name]


@require_e2e
@pytest.mark.parametrize("shape", [(32, 32, 32)] + get_test_shapes("test_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
    ],
)
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant, threads_per_wave, run_bench",
    [
#        (MMAType.F32_16x16x16_F16, 64),
#        (MMAType.F32_32x32x8_F16, 64),
        (MMAType.RDNA4_WAVE32_F32_16x16x16_F16, 32, True),
    ],
)
def testGemmSmallTiles(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    threads_per_wave: int,
    run_bench: bool
):
    tmp_path = "./.wave-tmp"
    perf_filename_tk = tmp_path + "/wave_gemm_bench.txt"
    perf_filename_iree = tmp_path + "/iree_gemm_bench.txt"
    # Test gemm with tiles smaller than MMA vector sizes.
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

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=threads_per_wave, mma_type=mfma_variant)]

    # With dynamic dimensions, we need to add an assumption on how big
    # the iterate dimension is to determine whether we can schedule or not.
    if dynamic_dims:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = []
    if dynamic_dims:
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(K)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[K]

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        use_buffer_ops=True,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_ones(shape[0], shape[2], dtype=torch.float16)
    b = device_ones(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)

