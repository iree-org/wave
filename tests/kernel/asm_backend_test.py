# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sympy
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
@pytest.mark.parametrize("shape", [(16, 16, 16)])
def test_mma_kernel_asm_backend(shape, run_bench):
    """End-to-end test for the MMA kernel using ASM backend with 16x16x16 shape."""
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
            mma_type=tkw.MMAType.F32_16x16x16_F16,
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

    # Create test tensors
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
def test_gemm_asm_backend(shape, block_k, config, use_global_to_shared, run_bench):
    """End-to-end test for GEMM with K-loop using ASM backend.

    Tests both single-wave and multi-wave configurations with varying BLOCK_K values.
    Multi-wave configurations enable testing workgroups with multiple waves per workgroup,
    where each wave operates on a tile (which can be larger than 16x16 MMA intrinsic).
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
            mma_type=tkw.MMAType.F32_16x16x16_F16,
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
