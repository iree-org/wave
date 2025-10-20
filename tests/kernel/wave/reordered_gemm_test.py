import pytest
import torch
from torch.testing import assert_close

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.templates.reordered_gemm import get_reordered_matmul
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)

from .common.utils import (
    require_cdna_2_or_3_or_4,
    require_e2e,
)

reordered_gemm_test_shapes = [
    (8192, 8192, 8192),
    (256, 256, 256),
    (2048, 1280, 5120),
]


@require_e2e
@require_cdna_2_or_3_or_4
@pytest.mark.parametrize("shape", reordered_gemm_test_shapes)
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.PREFETCH],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [MMAType.F32_16x16x16_F16],
)
def testReorderedPingPongGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
    # Input sizes
    M = shape[0]
    N = shape[1]
    K = shape[2]
    # Workgroup tile sizes
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    # Group size
    GROUP_SIZE_M = 16

    reordered_gemm, hyperparams = get_reordered_matmul(
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, mfma_variant
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        use_buffer_ops=True,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    reordered_gemm = wave_compile(options, reordered_gemm)
    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[1], shape[2], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    reordered_gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


@require_e2e
@pytest.mark.parametrize("shape", [(8192, 8192, 8192)])
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.FOUR_STAGE, SchedulingType.NONE],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [MMAType.F32_16x16x16_F16],
)
@pytest.mark.parametrize("transpose", ["NN", "NT", "TN"])
def testReorderedGemmTranspose(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
    transpose: str,
):
    tA, tB = transpose
    m, n, k = shape

    # Input sizes
    M = m
    N = n
    K = k
    # Workgroup tile sizes
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    # Group size
    GROUP_SIZE_M = 16

    reordered_gemm, hyperparams = get_reordered_matmul(
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_SIZE_M,
        mfma_variant,
        input_dtype=torch.float16,
        output_dtype=torch.float32,
        tA=tA,
        tB=tB,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        use_buffer_ops=True,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)

    reordered_gemm = wave_compile(options, reordered_gemm)

    a_shape = (m, k) if tA == "N" else (k, m)
    b_shape = (k, n) if tB == "N" else (n, k)
    c_shape = (m, n)

    a = device_randn(*a_shape, dtype=torch.float16)
    b = device_randn(*b_shape, dtype=torch.float16)
    c = device_zeros(*c_shape, dtype=torch.float32)
    reordered_gemm(a, b, c)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

    iree_ref = device_zeros(*c_shape, dtype=torch.float32)
    generate_iree_ref(f"mm_{tA}{tB}", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)
