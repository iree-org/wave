# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import pytest

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from wave_lang.kernel.wave.utils.mxfp_utils import (
    SCALE_GROUP_SIZE,
    f32_to_mxfp4,
    mxfp4_to_f32,
    e8m0_to_f32,
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
)
from wave_lang.kernel.wave.templates.dynamic_mxfp4_quant import (
    compute_mxfp4_scales,
    get_dynamic_mxfp4_quant_kernel,
    pack_mxfp4_codes,
)
from wave_lang.kernel.wave.templates import (
    get_tagged_mxfp4_gemm,
)
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_dbuf_schedule,
)

from .common.utils import (
    require_cdna4,
    require_e2e,
)


# ---------------------------------------------------------------------------
# Unit test: Wave quant kernel vs PyTorch reference
# ---------------------------------------------------------------------------


@require_e2e
@pytest.mark.parametrize(
    "shape",
    [
        (4, 64),
        (128, 256),
        (1, 1024),
        (64, 8192),
    ],
)
def test_dynamic_mxfp4_quant(shape):
    """Compile and run the Wave dynamic MXFP4 quant kernel, then compare
    against the pure-PyTorch ``f32_to_mxfp4`` reference."""
    M_val, K_val = shape

    quant, options = get_dynamic_mxfp4_quant_kernel(shape, block_m=2)
    options = set_default_run_config(options)
    quant = wave_compile(options, quant)

    torch.manual_seed(42)
    x = device_randn(shape, dtype=torch.float32)

    # Compute scales on GPU via PyTorch
    qs, bs_e8m0, _ = compute_mxfp4_scales(x)

    # Snapshot values to CPU *before* the IREE kernel call, which may
    # invalidate GPU tensors that share the PyTorch CUDA caching allocator.
    x_cpu = x.cpu().clone()
    bs_cpu = bs_e8m0.cpu().clone()

    # Allocate output and run Wave kernel
    codes = device_zeros(shape, dtype=torch.int8)
    quant(x, qs, codes)

    # PyTorch reference (runs on CPU)
    ref_fp4, ref_scales = f32_to_mxfp4(x_cpu.float())

    # Pack the Wave codes for comparison
    wave_fp4 = pack_mxfp4_codes(codes.cpu())

    torch.testing.assert_close(
        wave_fp4.to(torch.uint8), ref_fp4, atol=0, rtol=0
    )
    torch.testing.assert_close(bs_cpu, ref_scales, atol=0, rtol=0)

    # Cross-check via dequantize round-trip
    wave_deq = mxfp4_to_f32(wave_fp4.to(torch.uint8))
    wave_s = e8m0_to_f32(
        bs_cpu.repeat_interleave(SCALE_GROUP_SIZE, dim=1).float()
    )
    ref_deq = mxfp4_to_f32(ref_fp4)
    ref_s = e8m0_to_f32(
        ref_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).float()
    )
    torch.testing.assert_close(wave_deq * wave_s, ref_deq * ref_s)


# ---------------------------------------------------------------------------
# End-to-end test: Wave quant -> MXFP4 GEMM
# ---------------------------------------------------------------------------


@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(256, 256, 1024)])
def test_dynamic_mxfp4_quant_gemm_e2e(shape):
    """Quantise f32 activations with the Wave kernel, feed packed results into
    the MXFP4 scaled GEMM, and compare against the reference pipeline."""
    M_val, N_val, K_val = shape

    # -- Step 1: compile the quant kernel --
    quant, q_options = get_dynamic_mxfp4_quant_kernel(
        (M_val, K_val), block_m=2
    )
    q_options = set_default_run_config(q_options)
    quant = wave_compile(q_options, quant)

    # -- Step 2: compile the GEMM kernel --
    block_shape = (32, 32, 256)
    gemm, g_options = get_tagged_mxfp4_gemm(
        shape,
        block_shape,
        wave_shape=(2, 2),
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    schedule = get_mxfp4_dbuf_schedule()
    g_options = set_default_run_config(g_options)
    gemm = wave_compile(g_options, gemm, schedule)

    # -- Step 3: generate test data --
    torch.manual_seed(7)
    a = device_randn((M_val, K_val), dtype=torch.float32)
    _, w, _, w_scales = generate_gemm_afp4wfp4_inputs(shape)

    # -- Step 4: run Wave quant --
    qs, a_scales, _ = compute_mxfp4_scales(a)
    codes = device_zeros((M_val, K_val), dtype=torch.int8)
    quant(a, qs, codes)
    a_fp4 = pack_mxfp4_codes(codes)

    # -- Step 5: run GEMM --
    out = device_zeros(M_val, N_val, dtype=torch.float32)
    w_t = w.T.contiguous()
    gemm(a_fp4, a_scales, w_t, w_scales, out)

    # -- Step 6: reference pipeline --
    ref_fp4, ref_scales = f32_to_mxfp4(a.cpu().float())
    ref_fp4 = ref_fp4.to(a.device)
    ref_scales = ref_scales.to(a.device)
    ref_out = torchScaledGemmMXFP4(ref_fp4, w, ref_scales, w_scales)

    torch.testing.assert_close(ref_out, out, check_dtype=False)
