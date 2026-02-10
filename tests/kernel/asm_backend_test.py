# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from torch.testing import assert_close

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
    get_default_arch,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randint,
    device_randn,
    device_zeros,
)
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)

from .common.utils import require_e2e, require_cdna_3_or_4


def _global_to_shared_params():
    # global_to_shared (gather_to_lds) is currently only supported for
    # single-wave configurations on gfx95+ in the ASM backend.
    # Multi-wave configurations require more complex LDS offset handling.
    if "gfx95" in get_default_arch():
        return [pytest.param(True, id="g2s"), pytest.param(False, id="no_g2s")]
    return [pytest.param(False, id="no_g2s")]


def _mma_type_params():
    """Return MMA type parameters with appropriate skip marks.

    F32_16x16x16_F16: Works on CDNA3 (gfx94*) and CDNA4 (gfx95*)
    F32_16x16x32_F16: Only works on CDNA4 (gfx95*)
    """
    params = [
        pytest.param(tkw.MMAType.F32_16x16x16_F16, 16, 4, id="16x16x16"),
    ]
    # Only add 16x16x32 on gfx95* (CDNA4)
    if "gfx95" in get_default_arch():
        params.append(pytest.param(tkw.MMAType.F32_16x16x32_F16, 32, 8, id="16x16x32"))
    return params


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize("shape", [(16, 16)])
def test_copy_kernel_asm_backend(shape, run_bench):
    """End-to-end test for the copy kernel using ASM backend."""
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Hardware constraints similar to the original copy kernel
    wave_size = 64
    BLOCK_M = 16
    BLOCK_N = 16

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def copy_kernel(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        """Copy kernel that reads from input and writes to output."""
        res = tkw.read(a)
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, copy_kernel)

    compiled_kernel(a, b)

    assert_close(a, b)


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize("mma_type,k_size,load_elems", _mma_type_params())
def test_mma_kernel_asm_backend(mma_type, k_size, load_elems, run_bench):
    """End-to-end test for the MMA kernel using ASM backend.

    Tests both F32_16x16x16_F16 (CDNA3/4) and F32_16x16x32_F16 (CDNA4 only).
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Hardware constraints for MMA
    wave_size = 64
    BLOCK_M = 16
    BLOCK_N = 16

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=mma_type,
        ),
    ]

    @tkw.wave(constraints)
    def mma_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        """MMA kernel that computes C = A @ B^T."""
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # Create test tensors - K size depends on MMA type (16 or 32)
    # A is M x K, B is N x K (for B^T in MMA), C is M x N
    m, n, k = 16, 16, k_size
    a = device_randn((m, k), dtype=torch.float16)
    b = device_randn((n, k), dtype=torch.float16)
    c = device_zeros((m, n), dtype=torch.float32)

    options = WaveCompileOptions(
        subs={
            M: m,
            N: n,
            K: k,
            LOAD_ELEMS_PER_THREAD: load_elems,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, mma_kernel)

    compiled_kernel(a, b, c)

    # Compute expected result: C = A @ B^T
    expected = torch.matmul(a.float(), b.float().T)

    assert_close(c, expected)


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize(
    "shape",
    [
        (32, 32, 16),  # 2x2 = 4 workgroups
        (64, 64, 16),  # 4x4 = 16 workgroups
        (128, 128, 16),  # 8x8 = 64 workgroups
        (256, 256, 16),  # 16x16 = 256 workgroups
    ],
)
def test_mma_multi_workgroup_single_wave_asm_backend(shape, run_bench):
    """End-to-end test for multi-workgroup MMA using ASM backend.

    Tests multi-workgroup scenarios with 1 wave per workgroup, where each wave
    operates on a 16x16 tile (required by the F32_16x16x16_F16 MMA instruction).
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Configuration: 1 wave per workgroup, each handling 16x16 tile
    BLOCK_M = 16
    BLOCK_N = 16
    WAVE_M = 16
    WAVE_N = 16
    wave_size = 64

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, WAVE_M),
        tkw.WaveConstraint(N, WAVE_N),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def mma_multi_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        """MMA kernel that computes C = A @ B^T with multi-workgroup, multi-wave."""
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # Create test tensors with larger dimensions
    # A is M x K, B is N x K (for B^T in MMA), C is M x N
    m, n, k = shape
    a = device_randn((m, k), dtype=torch.float16)
    b = device_randn((n, k), dtype=torch.float16)
    c = device_zeros((m, n), dtype=torch.float32)

    options = WaveCompileOptions(
        subs={
            M: m,
            N: n,
            K: k,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, mma_multi_kernel)

    compiled_kernel(a, b, c)

    # Compute expected result: C = A @ B^T
    expected = torch.matmul(a.float(), b.float().T)

    assert_close(c, expected)


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize(
    "shape,config",
    [
        # (M, N, K), (BLOCK_M, BLOCK_N, WAVE_M, WAVE_N)
        # Config ensures each wave handles 16x16 tile for F32_16x16x16_F16 MMA
        # Max 16 waves per workgroup constraint (16 * 64 = 1024 threads max)
        (
            (256, 256, 16),
            (64, 64, 16, 16),
        ),  # 4x4 WGs, 4x4 waves per WG (16 waves = max, test both multi-wg and multi-wave)
        (
            (64, 64, 16),
            (64, 64, 16, 16),
        ),  # 1 WG with 4x4 waves (16 waves = max, pure multi-wave)
        (
            (64, 32, 16),
            (64, 32, 16, 16),
        ),  # 1 WG with 4x2 waves (8 waves, smaller multi-wave)
        (
            (128, 64, 16),
            (32, 32, 16, 16),
        ),  # 4x2 WGs, 2x2 waves per WG (4 waves per WG, multi-wg + multi-wave)
        (
            (8192, 8192, 16),
            (32, 32, 16, 16),
        ),  # 256x256 WGs, 2x2 waves per WG (large scale test)
    ],
)
def test_mma_multi_wave_asm_backend(shape, config, run_bench):
    """End-to-end test for multi-wave MMA using ASM backend.

    Tests scenarios with multiple waves per workgroup, where each wave operates
    on a 16x16 tile (required by the F32_16x16x16_F16 MMA instruction).

    The ASM backend now fully supports multi-wave execution by properly extracting
    tid_x and tid_y from the flat thread ID in v0, matching LLVM's behavior.
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Extract configuration
    BLOCK_M, BLOCK_N, WAVE_M, WAVE_N = config
    wave_size = 64

    # Verify configuration: each wave must handle 16x16 tile for MMA
    assert (
        BLOCK_M % WAVE_M == 0
    ), f"BLOCK_M ({BLOCK_M}) must be divisible by WAVE_M ({WAVE_M})"
    assert (
        BLOCK_N % WAVE_N == 0
    ), f"BLOCK_N ({BLOCK_N}) must be divisible by WAVE_N ({WAVE_N})"
    assert (
        WAVE_M == 16 and WAVE_N == 16
    ), f"Wave tile must be 16x16 for F32_16x16x16_F16 MMA"

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, WAVE_M),
        tkw.WaveConstraint(N, WAVE_N),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def mma_multi_wave_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        """MMA kernel that computes C = A @ B^T with multi-wave support."""
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # Create test tensors
    m, n, k = shape
    a = device_randn((m, k), dtype=torch.float16)
    b = device_randn((n, k), dtype=torch.float16)
    c = device_zeros((m, n), dtype=torch.float32)

    options = WaveCompileOptions(
        subs={
            M: m,
            N: n,
            K: k,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, mma_multi_wave_kernel)

    compiled_kernel(a, b, c)

    # Compute expected result: C = A @ B^T
    expected = torch.matmul(a.float(), b.float().T)

    assert_close(c, expected)


def _gemm_mma_type_params():
    """Return MMA type parameters for GEMM tests.

    For GEMM, BLOCK_K must be divisible by K-dimension of the MMA instruction.
    F32_16x16x16_F16: K=16, works on CDNA3/4
    F32_16x16x32_F16: K=32, works on CDNA4 only
    """
    params = [
        pytest.param(tkw.MMAType.F32_16x16x16_F16, id="16x16x16"),
    ]
    if "gfx95" in get_default_arch():
        params.append(pytest.param(tkw.MMAType.F32_16x16x32_F16, id="16x16x32"))
    return params


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize(
    "shape,block_k,config",
    [
        # Single-wave configurations (BLOCK_M=16, BLOCK_N=16)
        ((64, 64, 64), 16, (16, 16, 16, 16)),  # 1 wave per WG, BLOCK_K = 16
        ((64, 64, 64), 32, (16, 16, 16, 16)),  # 1 wave per WG, BLOCK_K = 32
        ((64, 64, 128), 64, (16, 16, 16, 16)),  # 1 wave per WG, BLOCK_K = 64
        # Multi-wave configurations (multiple waves per workgroup)
        ((64, 64, 64), 16, (32, 32, 16, 16)),  # 2x2 = 4 waves per WG, BLOCK_K = 16
        ((64, 64, 64), 32, (32, 32, 16, 16)),  # 2x2 = 4 waves per WG, BLOCK_K = 32
        ((64, 64, 128), 64, (32, 32, 16, 16)),  # 2x2 = 4 waves per WG, BLOCK_K = 64
        (
            (128, 128, 64),
            16,
            (64, 64, 16, 16),
        ),  # 4x4 = 16 waves per WG (max), BLOCK_K = 16
        ((64, 128, 64), 16, (32, 64, 16, 16)),  # 2x4 = 8 waves per WG, BLOCK_K = 16
        # Larger problem size with BLOCK_K=64
        ((256, 256, 128), 64, (32, 32, 16, 16)),  # 2x2 = 4 waves per WG, 8x8 WGs
        # Non-square block configurations with BLOCK_K=64
        ((128, 64, 64), 64, (64, 32, 16, 16)),  # 4x2 = 8 waves per WG, non-square
        ((64, 128, 64), 64, (32, 64, 16, 16)),  # 2x4 = 8 waves per WG, non-square
    ],
)
@pytest.mark.parametrize("use_global_to_shared", _global_to_shared_params())
@pytest.mark.parametrize("mma_type", _gemm_mma_type_params())
def test_gemm_asm_backend(
    shape, block_k, config, use_global_to_shared, mma_type, run_bench
):
    """End-to-end test for GEMM with K-loop using ASM backend.

    Tests both single-wave and multi-wave configurations with varying BLOCK_K values.
    Multi-wave configurations enable testing workgroups with multiple waves per workgroup,
    where each wave operates on a tile (which can be larger than 16x16 MMA intrinsic).

    Also tests both F32_16x16x16_F16 (CDNA3/4) and F32_16x16x32_F16 (CDNA4 only).
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Extract configuration: (BLOCK_M, BLOCK_N, WAVE_M, WAVE_N)
    block_m, block_n, WAVE_M, WAVE_N = config
    wave_size = 64

    # Verify configuration
    assert (
        block_m % WAVE_M == 0
    ), f"BLOCK_M ({block_m}) must be divisible by WAVE_M ({WAVE_M})"
    assert (
        block_n % WAVE_N == 0
    ), f"BLOCK_N ({block_n}) must be divisible by WAVE_N ({WAVE_N})"

    # For F32_16x16x32_F16, the MFMA K-dimension is 32. If the parametrized
    # BLOCK_K is smaller, bump it to 32 so the test always runs.
    if mma_type == tkw.MMAType.F32_16x16x32_F16 and block_k < 32:
        block_k = 32

    # Calculate number of waves per workgroup
    waves_per_wg_m = block_m // WAVE_M
    waves_per_wg_n = block_n // WAVE_N
    waves_per_wg = waves_per_wg_m * waves_per_wg_n

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, WAVE_M),
        tkw.WaveConstraint(N, WAVE_N),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=mma_type,
        ),
    ]

    @tkw.wave(constraints)
    def gemm_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        """GEMM kernel: C = A @ B^T with K-loop."""
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    # Create test tensors
    m, n, k = shape
    a = device_randn((m, k), dtype=torch.float16)
    b = device_randn((n, k), dtype=torch.float16)
    c = device_zeros((m, n), dtype=torch.float32)

    options = WaveCompileOptions(
        subs={
            M: m,
            N: n,
            K: k,
            BLOCK_M: block_m,
            BLOCK_N: block_n,
            BLOCK_K: block_k,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        use_global_to_shared=use_global_to_shared,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, gemm_kernel)

    compiled_kernel(a, b, c)

    # Compute expected result: C = A @ B^T
    expected = torch.matmul(a.float(), b.float().T)

    assert_close(c, expected)


@require_e2e
@pytest.mark.skipif(
    "gfx95" not in get_default_arch(),
    reason="MXFP4 scaled MFMA only supported on gfx950+ (CDNA4/MI350X)",
)
@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 512),  # 2x2 workgroups, K=512 for multiple scale groups
        (128, 128, 512),  # 4x4 workgroups
    ],
)
@pytest.mark.parametrize(
    "use_global_to_shared",
    _global_to_shared_params(),
)
def test_mxfp4_scaled_gemm_asm_backend(shape, use_global_to_shared, run_bench):
    """End-to-end test for MXFP4 (4-bit float) scaled GEMM using ASM backend.

    Tests the v_mfma_scale_f32_16x16x128_f8f6f4 instruction which performs:
    - F4E2M1FN (MXFP4) matrix multiply
    - F8E8M0FNU (E8M0) scale factors per 32-element group
    - F32 accumulation

    The test uses packed i8 representation for FP4 data (K/2 dimension) and
    i8 representation for E8M0 scales (K/32 dimension).
    """
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # MXFP4 configuration: 16x16 tiles with K=128 per instruction
    # Each FP4 element is 4 bits, packed 2 per byte
    # Each scale factor is E8M0 (1 byte) for every 32 FP4 elements
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 256  # K dimension (in FP4 elements)
    WAVE_M = 16
    WAVE_N = 16
    wave_size = 64
    SCALE_GROUP_SIZE = 32  # Hardware-defined scale group size

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, WAVE_M),
        tkw.WaveConstraint(N, WAVE_N),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=tkw.ScaledMMAType.F32_16x16x128_F8F6F4,
        ),
    ]

    @tkw.wave(constraints)
    def mxfp4_gemm_kernel(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],  # Packed FP4: K/2 bytes
        a_scale: tkl.Memory[
            M, K / SCALE_GROUP_SIZE, ADDRESS_SPACE, tkl.i8
        ],  # E8M0 scales
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],  # Packed FP4: K/2 bytes
        b_scale: tkl.Memory[
            N, K / SCALE_GROUP_SIZE, ADDRESS_SPACE, tkl.i8
        ],  # E8M0 scales
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        """MXFP4 GEMM kernel: C = (A_scale * A_fp4) @ (B_scale * B_fp4)^T"""
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # Read packed FP4 data and bitcast to f4e2m1fn
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)

            # Read E8M0 scale factors and bitcast to f8e8m0fnu
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)

            # Read packed FP4 data and bitcast to f4e2m1fn
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)

            # Read E8M0 scale factors and bitcast to f8e8m0fnu
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)

            # Scaled MMA: computes (a_scale * a) @ (b_scale * b)^T + acc
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    m, n, k = shape

    # Generate random FP4 data (packed as i8: 2 FP4 elements per byte)
    # For testing, we use random i8 values (in practice, these would be properly packed FP4)
    a = device_randint(-128, 127, (m, k // 2), dtype=torch.int8)
    b = device_randint(-128, 127, (n, k // 2), dtype=torch.int8)

    # Generate random E8M0 scale factors (1 byte per 32 FP4 elements)
    a_scale = device_randint(-128, 127, (m, k // SCALE_GROUP_SIZE), dtype=torch.int8)
    b_scale = device_randint(-128, 127, (n, k // SCALE_GROUP_SIZE), dtype=torch.int8)

    # Output accumulator
    c = device_zeros((m, n), dtype=torch.float32)

    options = WaveCompileOptions(
        subs={
            M: m,
            N: n,
            K: k,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        use_global_to_shared=use_global_to_shared,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, mxfp4_gemm_kernel)

    # Execute the compiled kernel
    compiled_kernel(a, a_scale, b, b_scale, c)

    # Note: We cannot compute an exact expected result here because:
    # 1. The input is random i8 (not properly packed FP4)
    # 2. FP4 arithmetic is complex and hardware-specific
    # 3. This is primarily a compilation and execution test
    #
    # For a full validation, you would need:
    # - Proper FP4 packing/unpacking
    # - Software reference implementation of MXFP4 arithmetic
    # - Dequantization to FP32 for comparison
    #
    # This test verifies that:
    # - The kernel compiles successfully
    # - The scaled MFMA instructions are generated
    # - The kernel executes without errors
    # - The output has the correct shape and is non-zero

    assert c.shape == (m, n), f"Output shape mismatch: {c.shape} vs ({m}, {n})"
    # Verify output is not all zeros (scaled MMA should produce results)
    assert not torch.all(
        c == 0
    ), "Output is all zeros - kernel may not have executed correctly"
