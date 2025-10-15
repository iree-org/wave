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
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)

from .common.utils import require_e2e, require_cdna3


@require_e2e
@require_cdna3
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
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, copy_kernel)

    compiled_kernel(a, b)

    assert_close(a, b)
