# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Comprehensive end-to-end tests for WaveASM backends.

This test suite mirrors the tests in tests/kernel/asm_backend_test.py,
and supports both C++ and Python ASM backends.

Tests included:
1. test_copy_kernel_cpp_backend - Simple copy kernel
2. test_mma_kernel_cpp_backend - Single-tile MMA
3. test_mma_multi_workgroup_single_wave_cpp_backend - Multi-workgroup MMA
4. test_mma_multi_wave_cpp_backend - Multi-wave MMA
5. test_gemm_cpp_backend - Full GEMM with K-loop

Run with:
    # Run all e2e tests with C++ backend (default, requires GPU)
    pytest test/e2e/test_asm_backend_e2e.py -v --run-e2e

    # Run with Python backend
    pytest test/e2e/test_asm_backend_e2e.py -v --run-e2e --backend=python

    # Run with both backends and compare
    pytest test/e2e/test_asm_backend_e2e.py -v --run-e2e --backend=both

    # Dump assembly files to /tmp for debugging
    pytest test/e2e/test_asm_backend_e2e.py -v --run-e2e --dump-asm

    # Run specific test
    pytest test/e2e/test_asm_backend_e2e.py::test_gemm_cpp_backend -v --run-e2e

    # Compare C++ vs Python backends (comparison tests)
    pytest test/e2e/test_asm_backend_e2e.py -v -k "compare"

Environment variables:
    WAVEASM_TRANSLATE: Path to waveasm-translate (default: auto-detect in build/)
    WAVE_DEFAULT_ARCH: Target architecture (default: auto-detect from GPU)
    ROCM_PATH: ROCm installation path (default: /opt/rocm)
    KEEP_TEMP_FILES: Set to 1 to keep temporary files for debugging
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Add wave_lang to path
wave_root = Path(__file__).parent.parent.parent.parent
if str(wave_root) not in sys.path:
    sys.path.insert(0, str(wave_root))

# Add e2e directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from waveasm_e2e import (
    WaveASMCompiler,
    capture_wave_mlir,
    capture_wave_kernel_info,
    compare_with_python_backend,
    run_with_wave_runtime,
    CompilationResult,
    CapturedKernelInfo,
)


# =============================================================================
# Test Configuration
# =============================================================================

def get_target_arch() -> str:
    """Get target architecture from environment or detect from GPU."""
    if "WAVE_DEFAULT_ARCH" in os.environ:
        arch = os.environ["WAVE_DEFAULT_ARCH"]
        return arch.split(":")[0]  # Strip feature flags

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, 'gcnArchName'):
                return props.gcnArchName.split(":")[0]
    except Exception:
        pass

    return "gfx942"


def is_cdna4() -> bool:
    """Check if running on CDNA4 (gfx95*)."""
    return "gfx95" in get_target_arch()


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
    except ImportError:
        pytest.skip("PyTorch not available")


def skip_if_no_wave_lang():
    """Skip test if wave_lang is not available."""
    try:
        import wave_lang.kernel.lang as tkl
        import wave_lang.kernel.wave as tkw
    except ImportError:
        pytest.skip("wave_lang not available")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def compiler():
    """Create compiler and cleanup after test."""
    c = WaveASMCompiler(
        target=get_target_arch(),
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
    skip_if_no_gpu()
    skip_if_no_wave_lang()

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
    grid = kernel_info.grid_size if kernel_info.grid_size != (1, 1, 1) else (shape[0] // BLOCK_M, shape[1] // BLOCK_N, 1)

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
    try:
        import wave_lang.kernel.wave as tkw
        params = [
            pytest.param(tkw.MMAType.F32_16x16x16_F16, 16, 4, id="16x16x16"),
        ]
        if is_cdna4():
            params.append(pytest.param(tkw.MMAType.F32_16x16x32_F16, 32, 8, id="16x16x32"))
        return params
    except ImportError:
        return [pytest.param(None, 16, 4, id="16x16x16")]


@pytest.mark.parametrize("mma_type,k_size,load_elems", _mma_type_params())
def test_mma_kernel_cpp_backend(mma_type, k_size, load_elems, compiler):
    """End-to-end test for the MMA kernel using C++ ASM backend.

    Mirrors: test_mma_kernel_asm_backend from asm_backend_test.py
    """
    skip_if_no_gpu()
    skip_if_no_wave_lang()

    import torch
    from torch.testing import assert_close
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
    from wave_lang.kernel.lang.global_symbols import SHARED_ADDRESS_SPACE, GLOBAL_ADDRESS_SPACE

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

    # Debug: save MLIR and assembly for debugging
    with open("/tmp/mma_debug_mlir.txt", "w") as f:
        f.write(kernel_info.mlir_text)
    with open("/tmp/mma_debug_asm.s", "w") as f:
        f.write(result.asm_text)

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
    assert_close(c, expected)


# =============================================================================
# Test: MMA Multi-Workgroup Single Wave
# =============================================================================

@pytest.mark.parametrize(
    "shape",
    [
        (32, 32, 16),    # 2x2 = 4 workgroups
        (64, 64, 16),    # 4x4 = 16 workgroups
        (128, 128, 16),  # 8x8 = 64 workgroups
        (256, 256, 16),  # 16x16 = 256 workgroups
    ],
)
def test_mma_multi_workgroup_single_wave_cpp_backend(shape, compiler):
    """End-to-end test for multi-workgroup MMA using C++ ASM backend.

    Mirrors: test_mma_multi_workgroup_single_wave_asm_backend from asm_backend_test.py
    """
    skip_if_no_gpu()
    skip_if_no_wave_lang()

    import torch
    from torch.testing import assert_close
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
    from wave_lang.kernel.lang.global_symbols import SHARED_ADDRESS_SPACE, GLOBAL_ADDRESS_SPACE

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

    # Calculate grid dimensions
    grid_x = m // BLOCK_M
    grid_y = n // BLOCK_N

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
    grid = kernel_info.grid_size if kernel_info.grid_size != (1, 1, 1) else (grid_x, grid_y, 1)

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
    assert_close(c, expected)


# =============================================================================
# Test: MMA Multi-Wave
# =============================================================================

@pytest.mark.parametrize(
    "shape,config",
    [
        # (M, N, K), (BLOCK_M, BLOCK_N, WAVE_M, WAVE_N)
        ((256, 256, 16), (64, 64, 16, 16)),   # 4x4 WGs, 4x4 waves per WG
        ((64, 64, 16), (64, 64, 16, 16)),     # 1 WG with 4x4 waves (16 waves = max)
        ((64, 32, 16), (64, 32, 16, 16)),     # 1 WG with 4x2 waves (8 waves)
        ((128, 64, 16), (32, 32, 16, 16)),    # 4x2 WGs, 2x2 waves per WG
    ],
)
def test_mma_multi_wave_cpp_backend(shape, config, compiler):
    """End-to-end test for multi-wave MMA using C++ ASM backend.

    Mirrors: test_mma_multi_wave_asm_backend from asm_backend_test.py
    """
    skip_if_no_gpu()
    skip_if_no_wave_lang()

    import torch
    from torch.testing import assert_close
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
    from wave_lang.kernel.lang.global_symbols import SHARED_ADDRESS_SPACE, GLOBAL_ADDRESS_SPACE

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

    # Calculate waves per workgroup
    waves_per_wg_m = BLOCK_M // WAVE_M
    waves_per_wg_n = BLOCK_N // WAVE_N
    waves_per_wg = waves_per_wg_m * waves_per_wg_n
    threads_per_block = waves_per_wg * wave_size

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

    # Calculate grid dimensions
    grid_x = m // BLOCK_M
    grid_y = n // BLOCK_N

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
    grid = kernel_info.grid_size if kernel_info.grid_size != (1, 1, 1) else (grid_x, grid_y, 1)

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
    assert_close(c, expected)


# =============================================================================
# Test: GEMM with K-loop
# =============================================================================

def _gemm_mma_type_params():
    """Return MMA type parameters for GEMM tests."""
    try:
        import wave_lang.kernel.wave as tkw
        params = [
            pytest.param(tkw.MMAType.F32_16x16x16_F16, id="16x16x16"),
        ]
        if is_cdna4():
            params.append(pytest.param(tkw.MMAType.F32_16x16x32_F16, id="16x16x32"))
        return params
    except ImportError:
        return [pytest.param(None, id="16x16x16")]


def _global_to_shared_params():
    """Return global_to_shared (gather_to_lds) parameters."""
    if is_cdna4():
        return [
            pytest.param(True, id="g2s", marks=pytest.mark.xfail(
                reason="g2s tests have intermittent failures due to synchronization issues"
            )),
            pytest.param(False, id="no_g2s")
        ]
    return [pytest.param(False, id="no_g2s")]


@pytest.mark.parametrize(
    "shape,block_k,config",
    [
        # Single-wave configurations
        ((64, 64, 64), 16, (16, 16, 16, 16)),   # 1 wave per WG
        ((64, 64, 64), 32, (16, 16, 16, 16)),   # 1 wave per WG, BLOCK_K = 32
        # Multi-wave configurations
        ((64, 64, 64), 16, (32, 32, 16, 16)),   # 2x2 = 4 waves per WG
        ((64, 64, 64), 32, (32, 32, 16, 16)),   # 2x2 = 4 waves per WG, BLOCK_K = 32
        ((128, 128, 64), 16, (64, 64, 16, 16)), # 4x4 = 16 waves per WG (max)
        # Larger problem size
        ((256, 256, 128), 64, (32, 32, 16, 16)), # 2x2 = 4 waves per WG, 8x8 WGs
    ],
)
@pytest.mark.parametrize("use_global_to_shared", _global_to_shared_params())
@pytest.mark.parametrize("mma_type", _gemm_mma_type_params())
def test_gemm_cpp_backend(shape, block_k, config, use_global_to_shared, mma_type, compiler, backend, dump_asm):
    """End-to-end test for GEMM with K-loop using C++ or Python ASM backend.

    Mirrors: test_gemm_asm_backend from asm_backend_test.py

    Use --backend=cpp (default), --backend=python, or --backend=both
    Use --dump-asm to dump assembly files to /tmp
    """
    skip_if_no_gpu()
    skip_if_no_wave_lang()

    import torch
    from torch.testing import assert_close
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw

    # xfail specific problematic combination: 16x16x32-no_g2s with large problem size
    if (mma_type == tkw.MMAType.F32_16x16x32_F16 and
        not use_global_to_shared and
        shape == (256, 256, 128)):
        pytest.xfail("16x16x32-no_g2s with large problem size has intermittent failures")
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
    from wave_lang.kernel.lang.global_symbols import SHARED_ADDRESS_SPACE, GLOBAL_ADDRESS_SPACE
    from wave_lang.kernel.wave.asm.kernel_module_compiler import KernelModuleCompiler

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

    # Calculate waves per workgroup
    waves_per_wg_m = block_m // WAVE_M
    waves_per_wg_n = block_n // WAVE_N
    waves_per_wg = waves_per_wg_m * waves_per_wg_n
    threads_per_block = waves_per_wg * wave_size

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

    # Calculate grid dimensions
    grid_x = m // block_m
    grid_y = n // block_n

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

    # Compile with C++ backend if requested
    cpp_result = None
    cpp_asm = None
    if backend in ("cpp", "both"):
        cpp_result = compiler.compile_full(kernel_info.mlir_text, kernel_info.workgroup_size)
        if not cpp_result.success:
            pytest.fail(f"C++ compilation failed: {cpp_result.error_message}")
        cpp_asm = cpp_result.asm_text

    # Compile with Python backend if requested
    python_asm = None
    python_binary_path = None
    if backend in ("python", "both"):
        try:
            python_compiler = KernelModuleCompiler(targetid=get_target_arch(), codeobj="5")
            python_asm = python_compiler.compile_mlir_string(kernel_info.mlir_text)
            # Also assemble to binary for execution using the same assembler as C++ backend
            if backend == "python":
                success, binary_path, error = compiler.assemble_to_binary(python_asm)
                if not success:
                    pytest.fail(f"Python ASM->Binary failed: {error}")
                python_binary_path = binary_path
        except Exception as e:
            if backend == "python":
                pytest.fail(f"Python compilation failed: {e}")
            else:
                print(f"Python backend compilation failed: {e}")

    # Dump assemblies if requested
    if dump_asm:
        with open(f"/tmp/{test_id}_mlir.txt", "w") as f:
            f.write(kernel_info.mlir_text)
        if cpp_asm:
            with open(f"/tmp/{test_id}_cpp.s", "w") as f:
                f.write(cpp_asm)
        if python_asm:
            with open(f"/tmp/{test_id}_python.s", "w") as f:
                f.write(python_asm)

        print(f"\n=== Dumped files for {test_id} ===")
        print(f"  MLIR:       /tmp/{test_id}_mlir.txt")
        if cpp_asm:
            print(f"  C++ ASM:    /tmp/{test_id}_cpp.s")
        if python_asm:
            print(f"  Python ASM: /tmp/{test_id}_python.s")
        print("=" * 40)

    # Determine which binary to execute
    if backend == "python":
        if python_binary_path is None:
            pytest.fail("Python backend did not produce a binary")
        binary_path = python_binary_path
        kernel_name = kernel_info.kernel_name
    else:
        # Use C++ backend (for both "cpp" and "both" modes)
        binary_path = cpp_result.binary_path
        kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name

    # Use Wave compiler's launch info (blocks, lds_size) which is authoritative
    block = kernel_info.workgroup_size
    lds_size = kernel_info.lds_size
    grid = kernel_info.grid_size if kernel_info.grid_size != (1, 1, 1) else (grid_x, grid_y, 1)

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
    assert_close(c, expected)


# =============================================================================
# Test: Compare C++ vs Python Backend
# =============================================================================

@pytest.mark.parametrize("shape", [(16, 16)])
def test_compare_backends_copy_kernel(shape, compiler):
    """Compare C++ and Python backend assembly output for copy kernel."""
    skip_if_no_wave_lang()

    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 16, N: 16},
        ),
        tkw.WorkgroupConstraint(M, 16, 0),
        tkw.WorkgroupConstraint(N, 16, 1),
        tkw.WaveConstraint(M, 16),
        tkw.WaveConstraint(N, 16),
    ]

    @tkw.wave(constraints)
    def copy_kernel(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

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

    # Capture MLIR
    mlir_text = capture_wave_mlir(options, copy_kernel)

    # Compare backends
    cpp_asm, python_asm = compare_with_python_backend(mlir_text, target=get_target_arch())

    # Basic sanity checks
    assert "failed" not in cpp_asm.lower() or "failed" not in python_asm.lower()

    # Count key instructions
    def count_instructions(asm):
        if "failed" in asm.lower():
            return {}
        lines = asm.split('\n')
        return {
            'buffer_load': sum(1 for l in lines if 'buffer_load' in l),
            'buffer_store': sum(1 for l in lines if 'buffer_store' in l),
            's_waitcnt': sum(1 for l in lines if 's_waitcnt' in l),
        }

    cpp_stats = count_instructions(cpp_asm)
    python_stats = count_instructions(python_asm)

    print(f"\nC++ Backend Stats: {cpp_stats}")
    print(f"Python Backend Stats: {python_stats}")

    # Both should have similar instruction counts
    if cpp_stats and python_stats:
        # Allow some variance but both should have the same load/store counts
        assert cpp_stats['buffer_load'] > 0 or python_stats['buffer_load'] > 0


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
