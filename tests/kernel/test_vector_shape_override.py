# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test that demonstrates vector_shape override behavior.

This test verifies that when vector_shapes are explicitly specified in
HardwareConstraint, they override the default MMA matrix shapes.
"""

import pytest
import torch
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
from .common.utils import require_cdna4, require_e2e


@require_e2e
@require_cdna4
def test_vector_shape_override_k():
    """
    Test that vector_shapes can override the K dimension.

    We use MMAType.F32_16x16x32_F16 which has default K=32,
    but override K=64 (double the default).
    """
    # Define symbolic dimensions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # MMAType.F32_16x16x32_F16 normally has 16x16x32 shapes,
    # but we override K to be 64
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x32_F16,
            # Override: K is set to 64 instead of default 32
            vector_shapes={M: 16, N: 16, K: 64},
        ),
    ]

    @tkw.wave(constraints)
    def gemm_with_k_override(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    # Test dimensions - BLOCK_K must match the overridden K vector_shape
    m, n, k = 128, 128, 128
    block_m, block_n, block_k = 16, 16, 64

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
    }

    # Create input tensors
    torch.manual_seed(42)
    a = device_randn(m, k, dtype=torch.float16)
    b = device_randn(n, k, dtype=torch.float16)
    c = device_zeros(m, n, dtype=torch.float32)

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
    )
    options = set_default_run_config(options)
    compiled = wave_compile(options, gemm_with_k_override)

    # Run the compiled kernel
    compiled(a, b, c)

    # Compute reference using PyTorch
    ref = torch.matmul(a.to(torch.float32), b.T.to(torch.float32))

    # Verify results
    torch.testing.assert_close(c, ref, rtol=1e-3, atol=1e-3)


@require_e2e
@require_cdna4
def test_vector_shape_override_m_n_multiples():
    """
    Test vector_shapes override with values that are multiples of defaults.

    We use MMAType.F32_16x16x16_F16 which has default shapes of 16x16x16,
    but override M=32 and N=32 (2x the default).
    """
    # Define symbolic dimensions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # MMAType.F32_16x16x16_F16 normally has 16x16x16 shapes,
    # but we override M=32 (2x default) and N=32 (2x default)
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            # Override: M=32 (2x16), N=32 (2x16), K stays at default 16
            vector_shapes={M: 32, N: 32, K: 16},
        ),
    ]

    @tkw.wave(constraints)
    def gemm_with_multiples(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    # Test dimensions - BLOCK_M/N must match the overridden vector_shapes
    m, n, k = 128, 128, 64
    block_m, block_n, block_k = 32, 32, 16

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
    }

    # Create input tensors
    torch.manual_seed(42)
    a = device_randn(m, k, dtype=torch.float16)
    b = device_randn(n, k, dtype=torch.float16)
    c = device_zeros(m, n, dtype=torch.float32)

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
    )
    options = set_default_run_config(options)
    compiled = wave_compile(options, gemm_with_multiples)

    # Run the compiled kernel
    compiled(a, b, c)

    # Compute reference using PyTorch
    ref = torch.matmul(a.to(torch.float32), b.T.to(torch.float32))

    # Verify results
    torch.testing.assert_close(c, ref, rtol=1e-3, atol=1e-3)


@require_e2e
@require_cdna4
@pytest.mark.skip(reason="known to give incorrect results")
def test_vector_shape_override_non_multiples():
    """
    Test vector_shapes override with values that are NOT multiples of MMA shapes.

    We use MMAType.F32_16x16x16_F16 which has default shapes of 16x16x16,
    but override M=20 and N=24 (NOT multiples of 16). This is an important
    edge case to ensure the override mechanism works with arbitrary values.
    """
    # Define symbolic dimensions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # MMAType.F32_16x16x16_F16 normally has 16x16x16 shapes,
    # but we override M=20, N=24 (NOT multiples of 16)
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            # Override: M=20, N=24 (NOT multiples of default 16)
            vector_shapes={M: 20, N: 24, K: 16},
        ),
    ]

    @tkw.wave(constraints)
    def gemm_non_multiples(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    # Test dimensions - BLOCK_M/N must match the overridden vector_shapes
    # m=160 is 8x20, n=120 is 5x24
    m, n, k = 160, 120, 64
    block_m, block_n, block_k = 20, 24, 16

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
    }

    # Create input tensors
    torch.manual_seed(42)
    a = device_randn(m, k, dtype=torch.float16)
    b = device_randn(n, k, dtype=torch.float16)
    c = device_zeros(m, n, dtype=torch.float32)

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
    )
    options = set_default_run_config(options)
    compiled = wave_compile(options, gemm_non_multiples)

    # Run the compiled kernel
    compiled(a, b, c)

    # Compute reference using PyTorch
    ref = torch.matmul(a.to(torch.float32), b.T.to(torch.float32))

    # Verify results
    torch.testing.assert_close(c, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main()
