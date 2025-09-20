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
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from .common.utils import (
    require_e2e,
    require_cdna2,
    require_cdna3,
    require_cdna4,
    require_cdna_3_or_4,
    require_cdna_2_or_3_or_4,
    require_rdna4,
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
@require_cdna_3_or_4
def testGemmBench(tmp_path):
    shape = (64, 64, 64)
    perf_filename_tk = tmp_path / "wave_gemm_bench.txt"
    perf_filename_iree = tmp_path / "iree_gemm_bench.txt"
    enable_scheduling = SchedulingType.NONE
    dynamic_dims = False
    mfma_variant = MMAType.F32_16x16x16_F16
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape, dynamic_dims, mfma_variant, torch.float16
    )

    assert not perf_filename_tk.exists()

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=True,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)
    assert perf_filename_tk.exists()
    assert "real_time" in perf_filename_tk.read_text()

    assert not perf_filename_iree.exists()
    options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert perf_filename_iree.exists()
    assert "real_time" in perf_filename_iree.read_text()


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
        SchedulingType.PREFETCH,
        SchedulingType.FOUR_STAGE,
        SchedulingType.MODULO,
    ],
)
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
@pytest.mark.parametrize("datatype", [torch.float16])
def testPureGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    datatype: torch.dtype,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape, dynamic_dims, mfma_variant, datatype
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
        benchmark_results_file=perf_filename_tk,
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

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


_xfail = lambda *a: pytest.param(*a, marks=pytest.mark.xfail)


_global_to_lds_shapes = [(17, 23, 32), (15, 13, 4)]


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize(
    "shape", _global_to_lds_shapes + get_test_shapes("test_gemm")[:1]
)
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
        SchedulingType.PREFETCH,
        SchedulingType.FOUR_STAGE,
        SchedulingType.MODULO,
    ],
)
@pytest.mark.parametrize(
    "dynamic_dims",
    [
        (False, False, False),
        (True, True, False),
    ],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
@pytest.mark.parametrize("datatype", [torch.float16])
def testGemmGatherToLDS(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: tuple[bool, bool, bool],
    mfma_variant: MMAType,
    datatype: torch.dtype,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape, dynamic_dims, mfma_variant, datatype
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
        use_global_to_shared=True,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=datatype)
    b = device_randn(shape[1], shape[2], dtype=datatype)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    asm = gemm(a, b, c)

    assert "amdgpu.gather_to_lds" in asm, "gather_to_lds not found in asm"

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


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
    "mfma_variant, threads_per_wave",
    [
        pytest.param(MMAType.F32_16x16x16_F16, 64, marks=require_cdna_2_or_3_or_4),
        pytest.param(MMAType.F32_32x32x8_F16, 64, marks=require_cdna_2_or_3_or_4),
        pytest.param(MMAType.RDNA4_WAVE32_F32_16x16x16_F16, 32, marks=require_rdna4),
    ],
)
def testGemmSmallTiles(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    threads_per_wave: int,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
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

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=threads_per_wave, mma_type=mfma_variant)
    ]

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

    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False, atol=2e-4, rtol=1e-5)


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
        _xfail(SchedulingType.PREFETCH),
        SchedulingType.FOUR_STAGE,
        SchedulingType.MODULO,
    ],
)
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testNonTransposeGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    if dynamic_dims:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    # Transpose during read for expected shape: (M, K) @ (N, K) -> (M, N)
    b_mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, K: j}, outputs={N: i, K: j}
    )

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[K, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, tkl.f16]; data is transposed [K, N] -> [N, K] from b_mapping
            b_reg = tkw.read(b, mapping=b_mapping)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 128,  # bigger tile size for in-thread transpose to kick-in
        BLOCK_K: 32,
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
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)
    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[2], shape[1], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    # TODO: switch to comparison against generated iree_ref
    torch_ref = torch.matmul(a, b)
    assert_close(
        c.to(torch.float16), torch_ref, atol=1e-2, rtol=1e-2, check_device=False
    )


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize("shape", [(4096, 4096, 4096)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testPingPongGemm(
    shape: tuple[int],
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

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
        BLOCK_M: 128,
        BLOCK_N: 256,
        BLOCK_K: 64,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=SchedulingType.PREFETCH,
        use_scheduling_barriers=False,
        dynamic_symbols=[],
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", [get_test_shapes("test_gemm")[0]])
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.MODULO])
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize("mfma_variant", [MMAType.F32_16x16x16_F16])
@pytest.mark.parametrize("datatype", [torch.float16, torch.bfloat16])
def testGemmDumpOverrideSchedule(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    datatype: DataType,
    tmp_path,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape, dynamic_dims, mfma_variant, datatype
    )
    schedule_path = tmp_path / "schedule.txt"
    assert not schedule_path.exists()
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
        dump_schedule=schedule_path,
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=datatype)
    b = device_randn(shape[1], shape[2], dtype=datatype)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)

    compiled_gemm(a, b, c)
    assert schedule_path.exists()

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)

    # Now reload the schedule and run the kernel again.
    # The results should be the same.
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape, dynamic_dims, mfma_variant, datatype
    )
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
        override_schedule=schedule_path,
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)
    c_new = device_zeros(shape[0], shape[1], dtype=torch.float32)
    asm = compiled_gemm(a, b, c_new)
    assert_close(c, c_new, check_device=False)
    assert_close(c_new, iree_ref, check_device=False)


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize("shape", [(64, 64, 64)])
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        GenericDot(k_vec_size=4, along_dim=MMAOperand.M),
        GenericDot(k_mult=4, along_dim=MMAOperand.M),
        _xfail(GenericDot(out_vec_size=4, along_dim=MMAOperand.M)),
        GenericDot(),
        GenericDot(k_mult=2),
        GenericDot(k_mult=4),
        GenericDot(k_vec_size=1),
        GenericDot(k_vec_size=2),
        GenericDot(k_vec_size=4),
        GenericDot(out_vec_size=1),
        _xfail(GenericDot(out_vec_size=2)),
        _xfail(GenericDot(out_vec_size=4)),
    ],
)
def testGemmDot(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
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

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

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

        # This microkernel encodes the fact that if the reduction
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
        BLOCK_M: 4 if mfma_variant.along_dim == MMAOperand.N else 64,
        BLOCK_N: 64 if mfma_variant.along_dim == MMAOperand.N else 4,
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
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    torch.manual_seed(3)
    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)

    gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False, atol=1e-3, rtol=1e-3)


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE, SchedulingType.MODULO, SchedulingType.FOUR_STAGE],
)
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_32x32x16_K8_F16,
        MMAType.F32_16x16x32_K8_F16,
    ],
)
def testVMFMAGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

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
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
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
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, atol=2e-4, rtol=3e-4, check_device=False)


@require_e2e
@require_cdna2
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE, SchedulingType.MODULO, SchedulingType.FOUR_STAGE],
)
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.I32_16x16x16_I8,
        MMAType.I32_32x32x8_I8,
    ],
)
def testCDNA2IntGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

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
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        c_reg = tkl.Register[M, N, tkl.i32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.i32]) -> tkl.Register[M, N, tkl.i32]:
            # a_reg: tkw.Register[M, K, tkl.i8]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, tkl.i8]
            b_reg = tkw.read(b)
            # acc: tkw.Register[M, N, tkl.i32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
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
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    randint_hi = 4
    a = device_randint(randint_hi, (shape[0], shape[2]), dtype=torch.int8)
    b = device_randint(randint_hi, (shape[1], shape[2]), dtype=torch.int8)
    c = device_zeros(shape[0], shape[1], dtype=torch.int32)
    gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.int32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
        SchedulingType.MODULO,
        SchedulingType.FOUR_STAGE,
    ],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.I32_16x16x32_I8,
        MMAType.I32_32x32x16_I8,
    ],
)
def testCDNA3IntGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        c_reg = tkl.Register[M, N, tkl.i32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.i32]) -> tkl.Register[M, N, tkl.i32]:
            # a_reg: tkw.Register[M, K, tkl.i8]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, tkl.i8]
            b_reg = tkw.read(b)
            # acc: tkw.Register[M, N, tkl.i32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    randint_hi = 4
    a = device_randint(randint_hi, (shape[0], shape[2]), dtype=torch.int8)
    b = device_randint(randint_hi, (shape[1], shape[2]), dtype=torch.int8)
    c = device_zeros(shape[0], shape[1], dtype=torch.int32)
    gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.int32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE, SchedulingType.MODULO, SchedulingType.FOUR_STAGE],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x32_F8,
        MMAType.F32_16x16x32_K4_F8,
        MMAType.F32_32x32x16_F8,
        MMAType.F32_32x32x16_K4_F8,
    ],
)
def testF8Gemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.cast(a_reg, tkl.f8e4m3fnuz)
            b_reg = tkw.read(b)
            b_reg = tkw.cast(b_reg, tkl.f8e4m3fnuz)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt_f8", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, atol=3e-5, rtol=3e-4, check_device=False)


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
    ],
)
@param_bool("dynamic_dims", "dyn", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testPackedGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
    # TODO: Convert this to i8 -> bitcast f16 gemm
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    # With dynamic dimensions, we need to add an assumption on how big
    # the iterate dimension is to determine whether we can schedule or not.
    if dynamic_dims:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    # Wave-level micro-kernel.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)  # tkw.Register[M, K/2, tkl.i32]
            a_reg = tkw.bitcast(a_reg, tkl.f16)  # tkw.Register[M, K, tkl.f16]
            b_reg = tkw.read(b)  # tkw.Register[N, K/2, tkl.i32]
            b_reg = tkw.bitcast(b_reg, tkl.f16)  # tkw.Register[M, K, tkl.f16]
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
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
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a.view(torch.int32), b.view(torch.int32), c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


@require_e2e
@require_cdna_2_or_3_or_4
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
    ],
)
@param_bool("dynamic_dims", "dyn", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testPackedNonTransposeGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
    # TODO: Convert this to i8 -> bitcast f16 gemm
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    # With dynamic dimensions, we need to add an assumption on how big
    # the iterate dimension is to determine whether we can schedule or not.
    if dynamic_dims:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    # Transpose during read for expected shape: (M, K) @ (N, K) -> (M, N)
    b_mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, K: j}, outputs={N: i, K: j}
    )

    # Wave-level micro-kernel.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[K / 2, N, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)  # tkw.Register[M, K/2, tkl.i32]
            a_reg = tkw.bitcast(a_reg, tkl.f16)  # tkw.Register[M, K, tkl.f16]
            b_reg = tkw.read(b, mapping=b_mapping)  # tkw.Register[N, K/2, tkl.i32]
            b_reg = tkw.bitcast(b_reg, tkl.f16)  # tkw.Register[M, K, tkl.f16]
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
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
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a.view(torch.int32), b.view(torch.int32).T.contiguous(), c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


@require_e2e
@require_cdna_2_or_3_or_4
@pytest.mark.parametrize("shape", get_test_shapes("test_batched_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
        SchedulingType.MODULO,
        SchedulingType.FOUR_STAGE,
    ],
)
def testBatchedGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={B: 0})]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K: shape[3],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    batched_gemm = wave_compile(options, batched_gemm)

    a = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    b = device_randn(shape[0], shape[2], shape[3], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    batched_gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    generate_iree_ref("bmmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)

    torch_ref = torch.matmul(a, b.transpose(-2, -1))
    assert_close(c.to(torch.float16), torch_ref, atol=1e-3, rtol=5e-3)


@require_e2e
@require_cdna_2_or_3_or_4
@pytest.mark.parametrize("shape", get_test_shapes("test_batched_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE],
)
def testSequentialBatchedGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.TilingConstraint(B, BLOCK_B)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={B: 0})]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        @tkw.iterate(B, init_args=[])
        def body():
            c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

            @tkw.iterate(K, init_args=[c_reg])
            def repeat(
                acc: tkl.Register[B, M, N, tkl.f32],
            ) -> tkl.Register[B, M, N, tkl.f32]:
                a_reg = tkw.read(a)
                b_reg = tkw.read(b)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K: shape[3],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    batched_gemm = wave_compile(options, batched_gemm)

    a = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    b = device_randn(shape[0], shape[2], shape[3], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    batched_gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    generate_iree_ref("bmmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)

    torch_ref = torch.matmul(a, b.transpose(-2, -1))
    assert_close(c.to(torch.float16), torch_ref, atol=1e-3, rtol=5e-3)


@require_cdna3
@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_batched_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE],
)
def testSequentialBatchedGemmWhile(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    run_bench,
    perf_filename_tk,
):
    # Input sizes
    B = tkl.sym.B
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(B)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={B: 0})]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        init_value: tkl.i32,  # type: ignore
    ):
        @tkw.iterate(B, start=init_value, condition=B < shape[0], init_args=[])
        def body():
            c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

            @tkw.iterate(K, init_args=[c_reg])
            def repeat(
                acc: tkl.Register[B, M, N, tkl.f32],
            ) -> tkl.Register[B, M, N, tkl.f32]:
                a_reg = tkw.read(a)
                b_reg = tkw.read(b)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(repeat, c)

            # Set the next value for the iteration.
            # In this case, we are using a simple increment operation,
            # but this can be replaced with any other operation.
            index_b = tkw.self_index(B, tkl.i32)
            next_value = tkw.apply_expr(index_b, lambda x: x + 1)
            tkw.set_symbol(B, next_value)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K: shape[3],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    batched_gemm = wave_compile(options, batched_gemm)

    a = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    b = device_randn(shape[0], shape[2], shape[3], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    batched_gemm(a, b, c, 0)

    torch_ref = torch.matmul(a, b.transpose(-2, -1))
    assert_close(c.to(torch.float16), torch_ref, atol=1e-3, rtol=5e-3)


@require_cdna3
@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_batched_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE],
)
def testSequentialBatchedGemmWhileWithOutputSum(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    run_bench,
    perf_filename_tk,
):
    # Input sizes
    B = tkl.sym.B
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(B)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={B: 0})]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output_sum: tkl.Memory[N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
        init_value: tkl.i32,  # type: ignore
    ):
        o_reg = tkl.Register[N, M, tkl.f32](0.0)

        @tkw.iterate(B, start=init_value, condition=B < shape[0], init_args=[o_reg])
        def body(outer_acc: tkl.Register[N, M, tkl.f32]):
            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            @tkw.iterate(K, init_args=[c_reg])
            def repeat(
                acc: tkl.Register[B, M, N, tkl.f32],
            ) -> tkl.Register[B, M, N, tkl.f32]:
                a_reg = tkw.read(a)
                b_reg = tkw.read(b)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(repeat, c)
            permuted = tkw.permute(repeat, target_shape=[N, M])
            outer_acc += permuted

            # Set the next value for the iteration.
            # In this case, we are using a simple increment operation,
            # but this can be replaced with any other operation.
            index_b = tkw.self_index(B, tkl.i32)
            next_value = tkw.apply_expr(index_b, lambda x: x + 1)
            tkw.set_symbol(B, next_value)

            return outer_acc

        tkw.write(body, output_sum)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K: shape[3],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    batched_gemm = wave_compile(options, batched_gemm)

    a = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    b = device_randn(shape[0], shape[2], shape[3], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    d = device_zeros(shape[2], shape[1], dtype=torch.float32)
    batched_gemm(a, b, c, d, 0)

    torch_ref = torch.matmul(a, b.transpose(-2, -1))
    assert_close(c.to(torch.float16), torch_ref, atol=1e-3, rtol=5e-3)
    assert_close(d.T, torch.sum(c, dim=0), atol=1e-3, rtol=5e-3)


@require_e2e
@require_cdna_2_or_3_or_4
@pytest.mark.parametrize("shape", get_test_shapes("test_batched_gemm"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE],
)
def testBatchedGemmWithPermute(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    run_bench,
    perf_filename_tk,
):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Load and store sizes
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={B: 0})]

    @tkw.wave(constraints)
    def batched_gemm_with_permute(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, B, N, ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        res = tkw.permute(repeat, target_shape=[M, B, N])
        tkw.write(res, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 16,
        BLOCK_B: 1,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K: shape[3],
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    batched_gemm_with_permute = wave_compile(options, batched_gemm_with_permute)

    a = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    b = device_randn(shape[0], shape[2], shape[3], dtype=torch.float16)
    c = device_zeros(shape[1], shape[0], shape[2], dtype=torch.float32)
    batched_gemm_with_permute(a, b, c)

    torch_ref = (
        torch.bmm(a, b.permute(0, 2, 1).contiguous()).permute(1, 0, 2).contiguous()
    )
    assert_close(c.to(torch.float16), torch_ref, atol=1e-3, rtol=5e-3)


@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(4096, 4096, 4096)])
@pytest.mark.parametrize("datatype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x32_BF16,
        MMAType.F32_32x32x16_BF16,
        MMAType.F32_16x16x32_F16,
        MMAType.F32_32x32x16_F16,
    ],
)
def test_cdna4_mfma(shape: tuple[int], datatype: torch.dtype, mfma_variant: MMAType):
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape, False, mfma_variant, datatype
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        wave_runtime=True,
        canonicalize=True,
        use_scheduling_barriers=False,
        dynamic_symbols=[],
    )
    options = set_default_run_config(options)

    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], device="cuda", dtype=datatype)
    b = device_randn(shape[1], shape[2], device="cuda", dtype=datatype)
    c = device_randn(shape[0], shape[1], device="cuda", dtype=torch.float32)
    asm = gemm(a, b, c)

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)
