# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import wave_lang.kernel as tk
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.templates.moe import (
    get_silu_and_mul_kernel,
)
from wave_lang.kernel.lang import DataType
import torch.nn.functional as F

torch.manual_seed(0)


def silu_and_mul_ref(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Reference implementation of SiLU and Mul operation"""
    return F.silu(x1) * x2


def test_silu_and_mul_kernel(
    m: int = 32,
    n: int = 64,
    dtype: torch.dtype = torch.float32,
):
    """Test the SiLU and Mul kernel against PyTorch reference"""
    device = "cuda"

    # Create test inputs
    x1 = torch.randn(m, n, dtype=dtype, device=device)
    x2 = torch.randn(m, n, dtype=dtype, device=device)

    # Reference implementation
    ref_output = silu_and_mul_ref(x1, x2)

    # Kernel implementation
    output = torch.zeros(m, n, dtype=dtype, device=device)

    # Get and compile the kernel
    silu_and_mul, symbols = get_silu_and_mul_kernel(m, n, tkl.f32)
    symbols.update(get_default_scheduling_params())
    options = WaveCompileOptions(subs=symbols)
    options = set_default_run_config(options)
    silu_and_mul = wave_compile(options, silu_and_mul)

    # Run the kernel
    silu_and_mul(x1, x2, output)

    # Compare results
    rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(
        output, ref_output, rtol=rtol, atol=atol, msg="SiLU and Mul output mismatch"
    )

    print(f"SiLU and Mul test passed for shape [{m}, {n}] with dtype {dtype}")


# Test parameters
m_values = [64]
n_values = [128]
dtypes = [torch.float32]


@pytest.mark.parametrize("m", m_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("dtype", dtypes)
def test_silu_and_mul_parametrized(m: int, n: int, dtype: torch.dtype):
    """Parametrized test for SiLU and Mul kernel"""
    test_silu_and_mul_kernel(m, n, dtype)


if __name__ == "__main__":
    # Run a simple test when script is executed directly
    test_silu_and_mul_kernel()
    print("All SiLU and Mul tests passed!")
