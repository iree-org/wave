# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Comprehensive end-to-end tests for WaveASM C++ backend.

Run with:
    # Run all e2e tests (requires GPU)
    pytest tests/kernel/wave/asm/test_waveasm_e2e.py -v --run-e2e

    # Dump assembly files to /tmp for debugging
    pytest tests/kernel/wave/asm/test_waveasm_e2e.py -v --run-e2e --dump-asm

    # Run specific test
    pytest tests/kernel/wave/asm/test_waveasm_e2e.py::test_gemm_cpp_backend -v --run-e2e

Environment variables:
    WAVEASM_TRANSLATE: Path to waveasm-translate (default: auto-detect in build/)
    WAVE_DEFAULT_ARCH: Target architecture (default: auto-detect from GPU)
    ROCM_PATH: ROCm installation path (default: /opt/rocm)
    KEEP_TEMP_FILES: Set to 1 to keep temporary files for debugging
"""

import os
import re

import pytest

from tests.kernel.common.utils import param_bool, require_cdna4
from wave_lang.kernel.wave.utils.run_utils import get_default_arch
from waveasm.waveasm_e2e import (
    WaveASMCompiler,
    capture_wave_kernel_info,
    run_with_wave_runtime,
)

# All tests require waveasm-translate, a GPU, and CDNA4 (gfx950/MI350X).
# The C++ ASM backend does not support CDNA3 or earlier.
pytestmark = [pytest.mark.require_waveasm, pytest.mark.require_e2e, require_cdna4]

# =============================================================================
# Test Configuration
# =============================================================================


def is_cdna4() -> bool:
    """Check if running on CDNA4 (gfx95*)."""
    return "gfx95" in get_default_arch()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def compiler():
    """Create compiler and cleanup after test."""
    c = WaveASMCompiler(
        target=get_default_arch(),
        keep_temp_files=bool(os.environ.get("KEEP_TEMP_FILES")),
    )
    yield c
    c.cleanup()


# =============================================================================
# Test: Copy Kernel
# =============================================================================


@pytest.mark.parametrize("shape", [(16, 16)])
def test_copy_kernel_cpp_backend(shape, compiler):
    """End-to-end test for the copy kernel using C++ ASM backend.

    Mirrors: test_copy_kernel_asm_backend from asm_backend_test.py
    """

    import torch
    from torch.testing import assert_close

    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 16
    BLOCK_N = 16

    constraints = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        ),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
    ]

    @tkw.wave(constraints)
    def copy_kernel(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
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
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
    )
    options = set_default_run_config(options)

    # Capture MLIR and kernel info (workgroup size, etc.)
    kernel_info = capture_wave_kernel_info(options, copy_kernel)

    # Compile with C++ backend
    result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
    if not result.success:
        pytest.fail(f"Compilation failed: {result.error_message}")

    # Extract launch parameters - use Wave compiler's values for accuracy
    kernel_name = result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info (blocks, lds_size, grid) which is authoritative
    block = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size  # From Wave compiler, not assembly parsing
    grid = kernel_info.grid_size

    # Execute on GPU
    run_with_wave_runtime(
        binary_path=result.binary_path,
        inputs=[a],
        outputs=[b],
        grid=grid,
        block=block,
        shared_memory_bytes=lds_size,
        func_name=kernel_name,
    )

    # Validate
    assert_close(a, b)


# =============================================================================
# Test: MMA Kernel (Single Tile)
# =============================================================================


def _mma_type_params():
    """Return MMA type parameters with appropriate skip marks."""
    import wave_lang.kernel.wave as tkw

    params = [
        pytest.param(tkw.MMAType.F32_16x16x16_F16, 16, 4, id="16x16x16"),
    ]
    if is_cdna4():
        params.append(pytest.param(tkw.MMAType.F32_16x16x32_F16, 32, 8, id="16x16x32"))
    return params


@pytest.mark.parametrize("mma_type,k_size,load_elems", _mma_type_params())
def test_mma_kernel_cpp_backend(mma_type, k_size, load_elems, compiler):
    """End-to-end test for the MMA kernel using C++ ASM backend.

    Mirrors: test_mma_kernel_asm_backend from asm_backend_test.py
    """

    import torch
    from torch.testing import assert_close

    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.lang.global_symbols import (
        GLOBAL_ADDRESS_SPACE,
        SHARED_ADDRESS_SPACE,
    )
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    wave_size = 64
    BLOCK_M = 16
    BLOCK_N = 16

    constraints = [
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
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

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
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
    )
    options = set_default_run_config(options)

    # Capture MLIR and kernel info
    kernel_info = capture_wave_kernel_info(options, mma_kernel)

    # Compile with C++ backend
    result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
    if not result.success:
        pytest.fail(f"Compilation failed: {result.error_message}")

    # Extract launch parameters - use Wave compiler's values for accuracy
    kernel_name = result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info (blocks, lds_size) which is authoritative
    block = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size  # From Wave compiler, not assembly parsing

    # Execute on GPU
    run_with_wave_runtime(
        binary_path=result.binary_path,
        inputs=[a, b],
        outputs=[c],
        grid=(1, 1, 1),
        block=block,
        shared_memory_bytes=lds_size,
        func_name=kernel_name,
    )

    # Validate: C = A @ B^T
    expected = torch.matmul(a.float(), b.float().T)
    assert_close(c, expected, atol=1e-4, rtol=1e-4)


# =============================================================================
# Test: MMA Multi-Workgroup Single Wave
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32, 16),  # 2x2 = 4 workgroups
        (64, 64, 16),  # 4x4 = 16 workgroups
        (128, 128, 16),  # 8x8 = 64 workgroups
        (256, 256, 16),  # 16x16 = 256 workgroups
    ],
)
def test_mma_multi_workgroup_single_wave_cpp_backend(shape, compiler):
    """End-to-end test for multi-workgroup MMA using C++ ASM backend.

    Mirrors: test_mma_multi_workgroup_single_wave_asm_backend from asm_backend_test.py
    """

    import torch
    from torch.testing import assert_close

    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.lang.global_symbols import (
        GLOBAL_ADDRESS_SPACE,
        SHARED_ADDRESS_SPACE,
    )
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    BLOCK_M = 16
    BLOCK_N = 16
    WAVE_M = 16
    WAVE_N = 16
    wave_size = 64

    constraints = [
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
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

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
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
    )
    options = set_default_run_config(options)

    # Capture MLIR and kernel info
    kernel_info = capture_wave_kernel_info(options, mma_multi_kernel)

    # Compile with C++ backend
    result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
    if not result.success:
        pytest.fail(f"Compilation failed: {result.error_message}")

    # Extract launch parameters - use Wave compiler's values for accuracy
    kernel_name = result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info (blocks, lds_size) which is authoritative
    block = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size  # From Wave compiler, not assembly parsing
    grid = kernel_info.grid_size

    # Execute on GPU
    run_with_wave_runtime(
        binary_path=result.binary_path,
        inputs=[a, b],
        outputs=[c],
        grid=grid,
        block=block,
        shared_memory_bytes=lds_size,
        func_name=kernel_name,
    )

    # Validate: C = A @ B^T
    expected = torch.matmul(a.float(), b.float().T)
    assert_close(c, expected, atol=1e-4, rtol=1e-4)


# =============================================================================
# Test: MMA Multi-Wave
# =============================================================================


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
def test_mma_multi_wave_cpp_backend(shape, config, compiler):
    """End-to-end test for multi-wave MMA using C++ ASM backend.

    Mirrors: test_mma_multi_wave_asm_backend from asm_backend_test.py
    """

    import torch
    from torch.testing import assert_close

    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.lang.global_symbols import (
        GLOBAL_ADDRESS_SPACE,
        SHARED_ADDRESS_SPACE,
    )
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    BLOCK_M, BLOCK_N, WAVE_M, WAVE_N = config
    wave_size = 64

    # Verify configuration
    assert BLOCK_M % WAVE_M == 0
    assert BLOCK_N % WAVE_N == 0
    assert WAVE_M == 16 and WAVE_N == 16

    constraints = [
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
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

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
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
    )
    options = set_default_run_config(options)

    # Capture MLIR and kernel info
    kernel_info = capture_wave_kernel_info(options, mma_multi_wave_kernel)

    # Compile with C++ backend
    result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
    if not result.success:
        pytest.fail(f"Compilation failed: {result.error_message}")

    # Extract launch parameters - use Wave compiler's values for accuracy
    kernel_name = result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info (blocks, lds_size) which is authoritative
    block = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size  # From Wave compiler, not assembly parsing
    grid = kernel_info.grid_size

    # Execute on GPU
    run_with_wave_runtime(
        binary_path=result.binary_path,
        inputs=[a, b],
        outputs=[c],
        grid=grid,
        block=block,
        shared_memory_bytes=lds_size,
        func_name=kernel_name,
    )

    # Validate: C = A @ B^T (compute on CPU to avoid GPU OOM for large shapes)
    expected = torch.matmul(a.float().cpu(), b.float().cpu().T)
    assert_close(c.cpu(), expected, atol=1e-4, rtol=1e-4)


# =============================================================================
# Test: GEMM with K-loop
# =============================================================================


def _gemm_mma_type_params():
    """Return MMA type parameters for GEMM tests."""
    import wave_lang.kernel.wave as tkw

    params = [
        pytest.param(tkw.MMAType.F32_16x16x16_F16, id="16x16x16"),
    ]
    if is_cdna4():
        params.append(pytest.param(tkw.MMAType.F32_16x16x32_F16, id="16x16x32"))
    return params


def _global_to_shared_params():
    """Return global_to_shared (gather_to_lds) parameters."""
    if is_cdna4():
        return [
            pytest.param(True, id="g2s"),
            pytest.param(False, id="no_g2s"),
        ]
    return [pytest.param(False, id="no_g2s")]


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
        # Larger wave tile configurations (WAVE_M=32, WAVE_N=32)
        # Each wave does 2x2 MMA ops to cover 32x32 tile with 16x16 MMA intrinsic
        ((256, 256, 128), 64, (64, 64, 32, 32)),  # 2x2 = 4 waves per WG, WAVE=32x32
    ],
)
@pytest.mark.parametrize("use_global_to_shared", _global_to_shared_params())
@pytest.mark.parametrize("mma_type", _gemm_mma_type_params())
def test_gemm_cpp_backend(
    shape, block_k, config, use_global_to_shared, mma_type, compiler, dump_asm
):
    """End-to-end test for GEMM with K-loop using C++ ASM backend.

    Use --dump-asm to dump assembly files to /tmp
    """

    import torch
    from torch.testing import assert_close

    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw

    from wave_lang.kernel.lang.global_symbols import (
        GLOBAL_ADDRESS_SPACE,
        SHARED_ADDRESS_SPACE,
    )
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M_SYM = tkl.sym.BLOCK_M
    BLOCK_N_SYM = tkl.sym.BLOCK_N
    BLOCK_K_SYM = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    block_m, block_n, WAVE_M, WAVE_N = config
    wave_size = 64

    # Verify configuration
    assert block_m % WAVE_M == 0
    assert block_n % WAVE_N == 0

    # For F32_16x16x32_F16, bump BLOCK_K to at least 32
    if mma_type == tkw.MMAType.F32_16x16x32_F16 and block_k < 32:
        block_k = 32

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M_SYM, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N_SYM, 1),
        tkw.TilingConstraint(K, BLOCK_K_SYM),
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
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    m, n, k = shape
    a = device_randn((m, k), dtype=torch.float16)
    b = device_randn((n, k), dtype=torch.float16)
    c = device_zeros((m, n), dtype=torch.float32)

    options = WaveCompileOptions(
        subs={
            M: m,
            N: n,
            K: k,
            BLOCK_M_SYM: block_m,
            BLOCK_N_SYM: block_n,
            BLOCK_K_SYM: block_k,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=use_global_to_shared,
    )
    options = set_default_run_config(options)

    # Capture MLIR and kernel info
    kernel_info = capture_wave_kernel_info(options, gemm_kernel)

    # Test ID for file naming - include MMA type to avoid overwrites
    g2s_str = "g2s" if use_global_to_shared else "no_g2s"
    mma_str = "16x16x32" if mma_type == tkw.MMAType.F32_16x16x32_F16 else "16x16x16"
    test_id = f"gemm_{m}x{n}x{k}_bk{block_k}_{block_m}x{block_n}_{mma_str}_{g2s_str}"

    # Compile with C++ backend
    cpp_result = compiler.compile_full(
        kernel_info.mlir_text, kernel_info.workgroup_size
    )
    if not cpp_result.success:
        pytest.fail(f"C++ compilation failed: {cpp_result.error_message}")

    # Dump assemblies if requested
    if dump_asm:
        with open(f"/tmp/{test_id}_mlir.txt", "w") as f:
            f.write(kernel_info.mlir_text)
        if cpp_result.asm_text:
            with open(f"/tmp/{test_id}_cpp.s", "w") as f:
                f.write(cpp_result.asm_text)

    binary_path = cpp_result.binary_path
    kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info (blocks, lds_size) which is authoritative
    block = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size
    grid = kernel_info.grid_size

    # Execute on GPU
    run_with_wave_runtime(
        binary_path=binary_path,
        inputs=[a, b],
        outputs=[c],
        grid=grid,
        block=block,
        shared_memory_bytes=lds_size,
        func_name=kernel_name,
    )

    # Validate: C = A @ B^T
    expected = torch.matmul(a.float(), b.float().T)
    assert_close(c, expected, atol=1e-4, rtol=1e-4)


# =============================================================================
# Test: GEMM with K-loop Unrolling (Postprocess)
# =============================================================================

# Postprocess transform to unroll K-loop by factor of 2
K_LOOP_UNROLL_POSTPROCESS = """
module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.loop.unroll %0 { factor = 2 } : !transform.any_op
        transform.yield
    }
}
"""


@pytest.mark.parametrize(
    "shape,block_k,config",
    [
        # Test configuration with K-loop that can be unrolled
        # K=128, BLOCK_K=64 means 2 iterations -> fully unrollable by factor 2
        ((64, 64, 128), 64, (16, 16, 16, 16)),  # 1 wave per WG
        ((256, 256, 128), 64, (64, 64, 32, 32)),  # 2x2 waves per WG, WAVE=32x32
    ],
)
@pytest.mark.parametrize("use_global_to_shared", [True])  # Only test with g2s=True
def test_gemm_cpp_backend_with_k_unroll(
    shape, block_k, config, use_global_to_shared, compiler, dump_asm
):
    """End-to-end test for GEMM with K-loop UNROLLING using C++ backend.

    This test verifies that the C++ backend correctly handles MLIR where
    the K-loop has been unrolled via postprocess transform.

    The postprocess unrolls the scf.for loop by factor of 2, which for
    K=128, BLOCK_K=64 results in full unrolling (0 scf.for loops remaining).
    """

    import torch
    from torch.testing import assert_close

    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw

    from wave_lang.kernel.lang.global_symbols import (
        GLOBAL_ADDRESS_SPACE,
        SHARED_ADDRESS_SPACE,
    )
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M_SYM = tkl.sym.BLOCK_M
    BLOCK_N_SYM = tkl.sym.BLOCK_N
    BLOCK_K_SYM = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    block_m, block_n, WAVE_M, WAVE_N = config
    wave_size = 64

    # Verify configuration
    assert block_m % WAVE_M == 0
    assert block_n % WAVE_N == 0

    # Use MMA type that works with our test configuration
    mma_type = tkw.MMAType.F32_16x16x32_F16

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M_SYM, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N_SYM, 1),
        tkw.TilingConstraint(K, BLOCK_K_SYM),
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
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    m, n, k = shape
    a = device_randn((m, k), dtype=torch.float16)
    b = device_randn((n, k), dtype=torch.float16)
    c = device_zeros((m, n), dtype=torch.float32)

    # Create options WITH postprocess (K-loop unrolling)
    options = WaveCompileOptions(
        subs={
            M: m,
            N: n,
            K: k,
            BLOCK_M_SYM: block_m,
            BLOCK_N_SYM: block_n,
            BLOCK_K_SYM: block_k,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=use_global_to_shared,
    )
    options = set_default_run_config(options)

    # Apply K-loop unrolling postprocess
    options.postprocess = K_LOOP_UNROLL_POSTPROCESS

    # Capture MLIR and kernel info
    kernel_info = capture_wave_kernel_info(options, gemm_kernel)

    # Verify that the K-loop was unrolled (scf.for count should be 0)
    scf_for_count = kernel_info.mlir_text.count("scf.for")
    assert (
        scf_for_count == 0
    ), f"Expected K-loop to be fully unrolled, but found {scf_for_count} scf.for ops"

    # Test ID for file naming
    test_id = f"gemm_k_unroll_{m}x{n}x{k}_bk{block_k}_{block_m}x{block_n}"

    # Compile with C++ backend
    cpp_result = compiler.compile_full(
        kernel_info.mlir_text, kernel_info.workgroup_size
    )
    if not cpp_result.success:
        pytest.fail(f"C++ compilation failed: {cpp_result.error_message}")

    # Dump assembly if requested
    if dump_asm and cpp_result.asm_text:
        from pathlib import Path

        asm_file = Path("/tmp") / f"{test_id}_cpp.s"
        asm_file.write_text(cpp_result.asm_text)
        mlir_file = Path("/tmp") / f"{test_id}.mlir"
        mlir_file.write_text(kernel_info.mlir_text)

    kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info
    block = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size
    grid = kernel_info.grid_size

    # Execute on GPU
    run_with_wave_runtime(
        binary_path=cpp_result.binary_path,
        inputs=[a, b],
        outputs=[c],
        grid=grid,
        block=block,
        shared_memory_bytes=lds_size,
        func_name=kernel_name,
    )

    # Validate: C = A @ B^T
    expected = torch.matmul(a.float(), b.float().T)
    assert_close(c, expected, atol=1e-4, rtol=1e-4)


# =============================================================================
# Test: MXFP4 Scaled GEMM
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32, 128),  # Minimal MXFP4 GEMM (single K-tile, single workgroup)
        (64, 64, 512),  # Small MXFP4 GEMM
        (128, 128, 512),  # Medium MXFP4 GEMM
    ],
)
@pytest.mark.parametrize("use_global_to_shared", _global_to_shared_params())
def test_mxfp4_scaled_gemm_cpp_backend(shape, use_global_to_shared, compiler, dump_asm):
    """End-to-end test for MXFP4 (4-bit float) scaled GEMM with numerical validation.

    Uses generate_gemm_afp4wfp4_inputs for properly packed MXFP4 inputs
    and torchScaledGemmMXFP4 as the reference implementation. Validates
    exact numerical correctness against the software reference.

    MXFP4 uses:
    - FP4 (f4E2M1FN) data format - 4 bits per element, packed as uint8
    - E8M0 (f8E8M0FNU) scale factors - 1 byte per scale group of 32 elements
    - Scaled MFMA instruction: v_mfma_scale_f32_16x16x128_f8f6f4

    This test requires CDNA4 (gfx950+) architecture.
    """
    # Skip if not CDNA4 (gfx950+)
    if not is_cdna4():
        pytest.skip("MXFP4 scaled MFMA only supported on gfx950+ (CDNA4/MI350X)")

    import torch

    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.lang.global_symbols import (
        GLOBAL_ADDRESS_SPACE,
        SHARED_ADDRESS_SPACE,
    )
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.constraints import ScaledMMAType
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.mxfp_utils import (
        generate_gemm_afp4wfp4_inputs,
        torchScaledGemmMXFP4,
    )

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M_SYM = tkl.sym.BLOCK_M
    BLOCK_N_SYM = tkl.sym.BLOCK_N
    BLOCK_K_SYM = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # MXFP4 configuration
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 128

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M_SYM, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N_SYM, 1),
        tkw.TilingConstraint(K, BLOCK_K_SYM),
        tkw.WaveConstraint(M, BLOCK_M_SYM / 2),
        tkw.WaveConstraint(N, BLOCK_N_SYM / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=ScaledMMAType.F32_16x16x128_F8F6F4,
        ),
    ]

    @tkw.wave(constraints)
    def mxfp4_gemm_kernel(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    m, n, k = shape

    # Generate properly packed MXFP4 inputs and compute reference output
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    c = torch.zeros(m, n, dtype=torch.float32).cuda()

    options = WaveCompileOptions(
        subs={
            M: m,
            N: n,
            K: k,
            BLOCK_M_SYM: BLOCK_M,
            BLOCK_N_SYM: BLOCK_N,
            BLOCK_K_SYM: BLOCK_K,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=use_global_to_shared,
    )
    options = set_default_run_config(options)

    # Capture MLIR and kernel info
    kernel_info = capture_wave_kernel_info(options, mxfp4_gemm_kernel)

    # Verify MLIR contains scaled_mfma operation
    assert (
        "amdgpu.scaled_mfma" in kernel_info.mlir_text
    ), "Expected amdgpu.scaled_mfma operation in MLIR"

    # Test ID for file naming
    g2s_str = "g2s" if use_global_to_shared else "no_g2s"
    test_id = f"mxfp4_gemm_{m}x{n}x{k}_{g2s_str}"

    # Compile with C++ backend
    cpp_result = compiler.compile_full(
        kernel_info.mlir_text, kernel_info.workgroup_size
    )
    if not cpp_result.success:
        pytest.fail(f"C++ compilation failed: {cpp_result.error_message}")

    # Verify assembly contains scaled MFMA instruction
    assert (
        "v_mfma_scale_f32_16x16x128_f8f6f4" in cpp_result.asm_text
    ), "Expected v_mfma_scale_f32_16x16x128_f8f6f4 instruction in assembly"

    # Dump assemblies if requested
    if dump_asm:
        with open(f"/tmp/{test_id}_mlir.txt", "w") as f:
            f.write(kernel_info.mlir_text)
        if cpp_result.asm_text:
            with open(f"/tmp/{test_id}_cpp.s", "w") as f:
                f.write(cpp_result.asm_text)

    binary_path = cpp_result.binary_path
    kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info
    block = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size
    grid = kernel_info.grid_size

    # Execute on GPU
    # Kernel signature: (a, a_scale, b, b_scale, c)
    # b must be transposed to match the kernel's N x K/2 layout
    run_with_wave_runtime(
        binary_path=binary_path,
        inputs=[x, x_scales, w.T.contiguous(), w_scales],
        outputs=[c],
        grid=grid,
        block=block,
        shared_memory_bytes=lds_size,
        func_name=kernel_name,
    )

    # Numerical correctness validation against reference
    c_cpu = c.cpu()
    torch_out_cpu = torch_out.cpu() if torch_out.is_cuda else torch_out
    torch.testing.assert_close(
        torch_out_cpu,
        c_cpu,
        check_dtype=False,
        msg=f"MXFP4 GEMM {m}x{n}x{k} ({g2s_str}) failed numerical validation",
    )


# =============================================================================
# Test: MXFP4 Double-Buffered Scheduled GEMM (from 7.1_schedule.py)
# =============================================================================


def _dbuf_mxfp4_helper(
    shape,
    block,
    num_waves,
    use_stagger,
    compiler,
    dump_asm,
    dynamic_dims=False,
    use_buffer_ops=True,
    use_schedule=True,
    output_dtype="f32",
    wave_shape=None,
    reorder_workgroups=None,
    eliminate_epilogue=False,
):
    """Shared helper for double-buffered MXFP4 scheduled GEMM tests.

    Mirrors the 7.1_schedule.py examples using get_tagged_mxfp4_gemm
    (tagged kernel template) + get_mxfp4_dbuf_schedule (double-buffer
    schedule with 2-stage pipeline and K-partitioned clusters).

    This exercises the C++ ASM backend on MLIR produced by the full
    Wave scheduling pipeline (MANUAL schedule + wave_schedule clusters).

    Args:
        output_dtype: Output element type, one of "f32" or "bf16".
    """
    if not is_cdna4():
        pytest.skip("MXFP4 double-buffered GEMM only supported on gfx950+ (CDNA4)")

    import torch
    import wave_lang.kernel.lang as tkl

    from wave_lang.kernel.wave.templates import (
        get_tagged_mxfp4_gemm,
        get_tagged_mxfp4_gemm_preshuffle_b,
    )
    from wave_lang.kernel.wave.schedules import (
        get_mxfp4_dbuf_schedule,
        get_mxfp4_asymmetric_schedule,
    )
    from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.mxfp_utils import (
        generate_gemm_afp4wfp4_inputs,
        torchScaledGemmMXFP4,
        b_preshuffle,
        e8m0_shuffle,
    )

    dtype_map = {
        "f32": (tkl.f32, torch.float32),
        "bf16": (tkl.bf16, torch.bfloat16),
    }
    tkl_dtype, torch_dtype = dtype_map[output_dtype]

    # Get tagged kernel + options (same as 7.1_schedule.py)
    # Use preshuffle B + asymmetric schedule with wave_shape=(1,4) for 4-wave,
    # standard double-buffer with wave_shape=(4,2) for 8-wave.
    if reorder_workgroups is None:
        reorder_workgroups = not dynamic_dims

    if num_waves <= 4:
        gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
            shape,
            block,
            wave_shape=wave_shape,
            reorder_workgroups=reorder_workgroups,
            output_dtype=tkl_dtype,
        )
        options.eliminate_epilogue = eliminate_epilogue
        if use_schedule:
            schedule = get_mxfp4_asymmetric_schedule(
                eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
            )
        else:
            schedule = None
            options.schedule = SchedulingType.NONE
    else:
        gemm, options = get_tagged_mxfp4_gemm(
            shape,
            block,
            wave_shape=(4, 2),
            output_dtype=tkl_dtype,
        )
        if use_schedule:
            schedule = get_mxfp4_dbuf_schedule(use_stagger=use_stagger)
        else:
            schedule = None
            options.schedule = SchedulingType.NONE

    # Override to ASM backend for C++ compilation path
    options.backend = "asm"
    options.wave_runtime = True
    options.compile_to_mlir = False
    options.use_buffer_ops = use_buffer_ops
    options = set_default_run_config(options)

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    m, n, k = shape

    dynamic_symbols = []
    dynamic_values = {}
    if dynamic_dims:
        dynamic_symbols = [M, N]
        dynamic_values = {M: m, N: n}
        del options.subs[M]
        del options.subs[N]
        dynamic_symbols.append(K)
        dynamic_values[K] = k
        del options.subs[K]
        options.dynamic_symbols = dynamic_symbols

    # Generate MXFP4 inputs and reference output
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    c = torch.zeros(shape[0], shape[1], dtype=torch_dtype).cuda()

    # Capture MLIR with schedule applied
    kernel_info = capture_wave_kernel_info(
        options, gemm, schedule=schedule, dynamic_values=dynamic_values
    )

    # Verify MLIR contains scaled_mfma operation
    assert (
        "amdgpu.scaled_mfma" in kernel_info.mlir_text
    ), "Expected amdgpu.scaled_mfma operation in MLIR"
    if dynamic_dims:
        expected_idx = r"function_type = \([^)]*index, index, index\) -> \(\)"
        expected_msg = "M, N, and K"
        assert re.search(expected_idx, kernel_info.mlir_text), (
            f"Expected dynamic-dims MLIR signature to carry trailing dynamic "
            f"{expected_msg} index arguments, got:\n{kernel_info.mlir_text[:400]}"
        )

    waves_str = f"{num_waves}wave"
    stagger_str = "stagger" if use_stagger else "no_stagger"
    m, n, k = shape
    test_id = f"mxfp4_dbuf_{waves_str}_{m}x{n}x{k}_{stagger_str}"

    # Compile with C++ backend
    cpp_result = compiler.compile_full(
        kernel_info.mlir_text, kernel_info.workgroup_size
    )
    if not cpp_result.success:
        pytest.fail(f"C++ compilation failed: {cpp_result.error_message}")

    # Verify assembly contains scaled MFMA instruction
    assert (
        "v_mfma_scale_f32_16x16x128_f8f6f4" in cpp_result.asm_text
    ), "Expected v_mfma_scale_f32_16x16x128_f8f6f4 instruction in assembly"

    # Dump assembly if requested
    if dump_asm and cpp_result.asm_text:
        with open(f"/tmp/{test_id}_cpp.s", "w") as f:
            f.write(cpp_result.asm_text)
        with open(f"/tmp/{test_id}.mlir", "w") as f:
            f.write(kernel_info.mlir_text)

    kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info
    block_size = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size
    grid = kernel_info.grid_size

    # Execute on GPU
    # Kernel signature: (a, a_scale, b, b_scale, c)
    # For preshuffle B: transform all inputs to match kernel expectations.
    # a_scale_preshuffle=True (default) means a_scales must also be shuffled.
    if num_waves <= 4:
        ws = wave_shape or (1, 4)
        x_scales = e8m0_shuffle(x_scales).contiguous()
        w_input = b_preshuffle(w.T.contiguous()).contiguous()
        w_scales_input = e8m0_shuffle(w_scales).contiguous()
    else:
        w_input = w.T.contiguous()
        w_scales_input = w_scales
    run_with_wave_runtime(
        binary_path=cpp_result.binary_path,
        inputs=[x, x_scales, w_input, w_scales_input],
        outputs=[c],
        grid=grid,
        block=block_size,
        shared_memory_bytes=lds_size,
        func_name=kernel_name,
        dynamic_dims=[dynamic_values[s] for s in dynamic_symbols],
    )

    # Numerical correctness validation (same tolerance as existing MXFP4 test)
    c_cpu = c.cpu()
    torch_out_cpu = torch_out.cpu() if torch_out.is_cuda else torch_out
    torch.testing.assert_close(
        torch_out_cpu,
        c_cpu,
        check_dtype=False,
        msg=f"MXFP4 double-buffered {waves_str} GEMM {m}x{n}x{k} "
        f"({stagger_str}) failed numerical validation",
    )


@param_bool("dynamic_dims", "dyn")
@param_bool("use_buffer_ops", "bufops")
@param_bool("use_schedule", "sched")
@pytest.mark.parametrize("eliminate_epilogue", [True, False], ids=["ee", "no_ee"])
@pytest.mark.parametrize("output_dtype", ["f32", "bf16"])
@pytest.mark.parametrize(
    "shape,block,wave_shape",
    [
        pytest.param((1024, 1024, 8192), (128, 256, 256), (1, 4), id="128x256x256"),
        pytest.param((1024, 1024, 8192), (128, 32, 256), (2, 2), id="128x32x256"),
        pytest.param((896, 640, 8192), (224, 160, 256), (2, 2), id="224x160x256"),
        pytest.param((1024, 768, 8192), (256, 192, 256), (1, 4), id="256x192x256"),
        pytest.param((1024, 640, 8192), (256, 160, 256), (2, 2), id="256x160x256"),
        pytest.param((1024, 896, 8192), (256, 224, 256), (2, 2), id="256x224x256"),
    ],
)
def test_dbuf_4wave_mxfp4_gemm_cpp_backend(
    shape,
    block,
    wave_shape,
    dynamic_dims,
    use_buffer_ops,
    use_schedule,
    eliminate_epilogue,
    output_dtype,
    compiler,
    dump_asm,
):
    """End-to-end test for asymmetric MXFP4 GEMM with 4 waves.

    Uses get_mxfp4_asymmetric_schedule() with preshuffle B and various
    block configurations.  The wave_shape is chosen per block so that
    each wave's tile dimensions are divisible by the MFMA tile width (16).
    Tests both eliminate_epilogue=True and False paths.
    """
    block_id = f"{block[0]}x{block[1]}x{block[2]}"

    # Epilogue elimination + unscheduled pipeline + dynamic dims: the
    # unscheduled path makes K dynamic, but EE's validBytes clamping is
    # only wired for the pipelined (scheduled) loop structure.
    if eliminate_epilogue and not use_schedule and dynamic_dims:
        pytest.skip(
            "Epilogue elimination with dynamic K (unscheduled + dynamic dims) "
            "not yet supported"
        )

    # VGPR overflow: 256x224x256 scheduled pipeline exceeds 256 VGPR limit.
    if block_id == "256x224x256" and use_schedule:
        pytest.xfail("C++ ASM backend exceeds VGPR limit with scheduled pipeline")

    # VGPR overflow: 224x160x256 scheduled pipeline exceeds 256 VGPR limit
    # without epilogue elimination; with ee=True it fits for static dims
    # but dynamic dims adds enough extra VGPRs to overflow again.
    if block_id == "224x160x256" and use_schedule:
        if not eliminate_epilogue:
            pytest.xfail(
                "C++ ASM backend exceeds VGPR limit with scheduled pipeline "
                "(ee=False) for 224x160x256"
            )
        elif dynamic_dims:
            pytest.xfail(
                "C++ ASM backend exceeds VGPR limit with ee=True + dynamic "
                "dims for 224x160x256"
            )
        else:
            # TODO (Gaurav/Sanket): should be passing after all the cherry-picks
            pytest.xfail(
                "VGPR overflow: 224x160x256 ee + scheduled pipeline + static "
                "dims exceeds register limit (register index is out of range)"
            )
    # VGPR overflow: 256x160x256 scheduled pipeline exceeds 256 VGPR limit
    # without epilogue elimination; with ee=True it fits for static dims
    # but dynamic dims adds enough extra VGPRs to overflow again.
    if block_id == "256x160x256" and use_schedule:
        if not eliminate_epilogue:
            pytest.xfail(
                "C++ ASM backend exceeds VGPR limit with scheduled pipeline "
                "(ee=False) for 256x160x256"
            )
        elif dynamic_dims:
            pytest.xfail(
                "C++ ASM backend exceeds VGPR limit with ee=True + dynamic "
                "dims for 256x160x256"
            )
        else:
            # TODO (Gaurav/Sanket): should be passing after all the cherry-picks
            pytest.xfail(
                "VGPR overflow: 256x160x256 ee + scheduled pipeline + static "
                "dims exceeds register limit (register index is out of range)"
            )

    # VGPR overflow for 256x192x256: ee=True reduces register pressure
    # enough to pass with static dims; ee=False and dynamic dims still overflow.
    if block_id == "256x192x256" and use_schedule:
        if not eliminate_epilogue:
            pytest.xfail(
                "C++ ASM backend exceeds VGPR limit with scheduled pipeline "
                "(ee=False); ee=True resolves this for 256x192x256"
            )
        elif dynamic_dims:
            pytest.xfail(
                "C++ ASM backend exceeds VGPR limit with ee=True + dynamic "
                "dims for 256x192x256"
            )
        else:
            # TODO (Gaurav/Sanket): should be passing after all the cherry-picks
            pytest.xfail(
                "VGPR overflow: 256x192x256 ee + scheduled pipeline + static "
                "dims exceeds register limit (register index is out of range)"
            )

    # (2,2) wave shape + schedule: numerical mismatch (with or without
    # dynamic dims when ee=False; dynamic-dims-only when ee=True).
    if block_id == "128x32x256" and use_schedule:
        if not eliminate_epilogue:
            pytest.xfail(
                "Numerical mismatch on (2,2) wave shape with scheduled "
                "pipeline (ee=False)"
            )
        elif dynamic_dims:
            pytest.xfail("Numerical mismatch with dynamic dims on (2,2) wave shape")
        else:
            # TODO (Gaurav/Sanket): should be passing after all the cherry-picks
            pytest.xfail(
                "128x32x256 (2,2) wave shape ee + scheduled pipeline + "
                "static dims: numerical mismatch (no_bufops) or SGPR "
                "overflow s103 not available (bufops)"
            )

    _dbuf_mxfp4_helper(
        shape=shape,
        block=block,
        num_waves=4,
        use_stagger=False,
        compiler=compiler,
        dump_asm=dump_asm,
        dynamic_dims=dynamic_dims,
        use_buffer_ops=use_buffer_ops,
        use_schedule=use_schedule,
        output_dtype=output_dtype,
        wave_shape=wave_shape,
        eliminate_epilogue=eliminate_epilogue,
    )


@pytest.mark.parametrize(
    "shape,block,wave_shape",
    [
        pytest.param((1024, 1024, 8192), (128, 256, 256), (1, 4), id="128x256x256"),
    ],
)
def test_dbuf_4wave_mxfp4_dynamic_mn_reorder_cpp_backend(
    shape, block, wave_shape, compiler, dump_asm
):
    """MXFP4 GEMM with dynamic M,N and workgroup reordering enabled.

    Regression test for illegal memory access caused by:
    - SRD base adjustment using unsigned multiply (S_MUL_HI_U32 vs S_MUL_HI_I32)
    - Swizzle SRDs with wrong num_records (0xFFFFFFFF vs source copy)
    - Trans pipeline forwarding hazard (missing s_nop)

    These issues only manifested when workgroup reordering was enabled with
    dynamic dimensions, because negative byte offsets appeared in the SRD
    base adjustment.
    """
    _dbuf_mxfp4_helper(
        shape=shape,
        block=block,
        num_waves=4,
        use_stagger=False,
        compiler=compiler,
        dump_asm=dump_asm,
        dynamic_dims=True,
        use_buffer_ops=True,
        use_schedule=True,
        wave_shape=wave_shape,
        reorder_workgroups=True,
    )


@pytest.mark.skip(
    reason="C++ backend allocates 264 VGPRs for 8-wave MXFP4, exceeding the "
    "256 VGPR hardware limit. Running this test causes a fatal "
    "HSA_STATUS_ERROR_INVALID_ISA abort that kills the entire test process.",
)
def test_dbuf_8wave_mxfp4_gemm_cpp_backend(compiler, dump_asm):
    """End-to-end test for double-buffered MXFP4 GEMM with 8 waves.

    Mirrors: test_dbuf_8wave_mxfp_gemm from examples/python/7.1_schedule.py

    8-wave configuration (4 M-tiles x 2 N-tiles), with stagger.
    Uses tagged kernel template + double-buffer schedule with 2-stage
    pipeline, K-dimension partitioning, and wave staggering for better
    overlap.
    """
    _dbuf_mxfp4_helper(
        shape=(1024, 1024, 8192),
        block=(256, 256, 256),
        num_waves=8,
        use_stagger=True,
        compiler=compiler,
        dump_asm=dump_asm,
    )


# =============================================================================
# Test: GEMM with Pipelining (PREFETCH Schedule)
# =============================================================================


def _pipelined_g2s_params():
    """Return global_to_shared params for pipelined tests.

    The C++ ASM backend fully supports pipelined GEMM with g2s:
    - memref iter_args carried as SGPR LDS offsets (double-buffering)
    - gather_to_lds in prologue/epilogue (outside scf.for)
    - SGPR rotation (swap) at loop tail for ping-pong pattern
    - Dynamic SGPR-carried LDS offsets in vector.load/store and gather_to_lds
    - LDS base offset propagation to loop results for epilogue correctness
    """
    params = [pytest.param(False, id="no_g2s")]
    if is_cdna4():
        params.append(pytest.param(True, id="g2s"))
    return params


@pytest.mark.parametrize(
    "shape,block_k,config",
    [
        # Single-wave: 64x64x128 with BLOCK_K=16 -> 8 iterations, pipelined to 7+epilogue
        ((64, 64, 128), 16, (16, 16, 16, 16)),
        # Multi-wave: 2x2=4 waves per WG
        ((64, 64, 128), 16, (32, 32, 16, 16)),
    ],
)
@pytest.mark.parametrize("use_global_to_shared", _pipelined_g2s_params())
@pytest.mark.parametrize("use_buffer_ops", [False, True])
def test_gemm_pipelined_cpp_backend(
    shape,
    block_k,
    config,
    use_global_to_shared,
    use_buffer_ops,
    compiler,
    dump_asm,
):
    """End-to-end test for GEMM with PREFETCH pipelining using C++ ASM backend.

    Software pipelining (PREFETCH schedule) restructures the K-loop into:
    1. Prologue: Load first tile from global -> shared memory
    2. Loop body (N-1 iterations):
       - barrier + load current tile from shared
       - load NEXT tile from global (overlapped with compute)
       - MFMA compute on current tile
       - barrier + store next tile to shared
    3. Epilogue: barrier + load last tile from shared + final MFMA

    This overlaps global memory loads with compute to hide latency.
    """

    import torch
    from torch.testing import assert_close

    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw

    from wave_lang.kernel.lang.global_symbols import (
        GLOBAL_ADDRESS_SPACE,
        SHARED_ADDRESS_SPACE,
        READ_SHARED_DELAY,
        WRITE_SHARED_DELAY,
        READ_GLOBAL_DELAY,
        WRITE_GLOBAL_DELAY,
        MMA_DELAY,
        VALU_DELAY,
        SHUFFLE_DELAY,
        SHARED_MEMORY_UNITS,
        GLOBAL_MEMORY_UNITS,
        MMA_UNITS,
        VALU_UNITS,
        SHUFFLE_UNITS,
    )
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M_SYM = tkl.sym.BLOCK_M
    BLOCK_N_SYM = tkl.sym.BLOCK_N
    BLOCK_K_SYM = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    block_m, block_n, WAVE_M, WAVE_N = config
    wave_size = 64

    # Verify configuration
    assert block_m % WAVE_M == 0
    assert block_n % WAVE_N == 0

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M_SYM, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N_SYM, 1),
        tkw.TilingConstraint(K, BLOCK_K_SYM),
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
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    m, n, k = shape
    a = device_randn((m, k), dtype=torch.float16)
    b = device_randn((n, k), dtype=torch.float16)
    c = device_zeros((m, n), dtype=torch.float32)

    options = WaveCompileOptions(
        subs={
            M: m,
            N: n,
            K: k,
            BLOCK_M_SYM: block_m,
            BLOCK_N_SYM: block_n,
            BLOCK_K_SYM: block_k,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            VALU_DELAY: 1,
            SHUFFLE_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
            VALU_UNITS: 8,
            SHUFFLE_UNITS: 8,
        },
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=use_global_to_shared,
        use_buffer_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)

    # Capture MLIR and kernel info
    kernel_info = capture_wave_kernel_info(options, gemm_kernel)

    # Test ID for file naming
    g2s_str = "g2s" if use_global_to_shared else "no_g2s"
    buf_str = "bufops" if use_buffer_ops else "flat"
    test_id = f"gemm_pipelined_{m}x{n}x{k}_bk{block_k}_{block_m}x{block_n}_{g2s_str}_{buf_str}"

    # Compile with C++ backend
    cpp_result = compiler.compile_full(
        kernel_info.mlir_text, kernel_info.workgroup_size
    )
    if not cpp_result.success:
        pytest.fail(f"C++ compilation failed: {cpp_result.error_message}")

    # Dump assembly if requested
    if dump_asm and cpp_result.asm_text:
        from pathlib import Path

        asm_file = Path("/tmp") / f"{test_id}_cpp.s"
        asm_file.write_text(cpp_result.asm_text)
        mlir_file = Path("/tmp") / f"{test_id}.mlir"
        mlir_file.write_text(kernel_info.mlir_text)

    kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info
    block = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size
    grid = kernel_info.grid_size

    # Execute on GPU
    run_with_wave_runtime(
        binary_path=cpp_result.binary_path,
        inputs=[a, b],
        outputs=[c],
        grid=grid,
        block=block,
        shared_memory_bytes=lds_size,
        func_name=kernel_name,
    )

    # Validate: C = A @ B^T
    expected = torch.matmul(a.float(), b.float().T)
    assert_close(c, expected, atol=1e-4, rtol=1e-4)
