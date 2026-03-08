# Copyright 2026 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GEMM through the water+waveasm pipeline (LLVM dialect → WaveASM → binary)."""

import pytest
import torch
from torch.testing import assert_close

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

from ..common.utils import require_cdna_3_or_4, require_e2e


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize(
    "shape,block_shape,waves_per_block",
    [
        ((64, 64, 64), (32, 32, 16), (1, 1)),
    ],
    ids=["64x64x64"],
)
def test_gemm_water_waveasm(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    waves_per_block: tuple[int, int],
) -> None:
    """Test GEMM through the water+waveasm pipeline."""
    m, n, k = shape

    gemm, hyperparams, _ = get_gemm_kernel(
        shape=shape,
        dynamic_dims=False,
        mfma_variant=MMAType.F32_16x16x16_F16,
        block_shape=block_shape,
        waves_per_block=waves_per_block,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        use_water_backend=True,
        use_buffer_ops=True,
        backend="asm",
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    compiled = wave_compile(options, gemm)

    a = device_randn((m, k), dtype=torch.float16)
    b = device_randn((n, k), dtype=torch.float16)
    c = device_zeros((m, n), dtype=torch.float32)
    compiled(a, b, c)

    expected = torch.matmul(a, b.T).float()
    assert_close(c, expected, rtol=1e-3, atol=1e-3)
