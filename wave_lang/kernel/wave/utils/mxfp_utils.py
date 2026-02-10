# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MXFP format utilities: input generation, type conversion, and reference kernels.
"""

import torch
from torch import Tensor

from .torch_utils import get_default_device

# HW-specified scale group size for MXFP formats.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(
    shape: tuple[int, int, int], device: torch.device = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate random packed MXFP4 inputs and e8m0 scales for scaled GEMM."""
    if device is None:
        device = get_default_device()
    M, N, K = shape
    torch.manual_seed(5)
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x = x_low | (x_high << 4)
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w = w_low | (w_high << 4)
    w = w.T
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device=device
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=device
    )
    x_scales = x_scales.T.contiguous()
    w_scales = w_scales.T.contiguous()
    return x, w, x_scales, w_scales


def mxfp4_to_f32(x: Tensor) -> Tensor:
    """Convert packed MXFP4 (e2m1) uint8 values to f32 via lookup table."""
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    lut = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    return torch.tensor(lut, dtype=torch.float32, device=x.device)[x.long()]


def e8m0_to_f32(x: Tensor) -> Tensor:
    """Convert e8m0 scale values to f32."""
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def torchScaledGemmMXFP4(
    x: Tensor, w: Tensor, x_scales: Tensor, w_scales: Tensor
) -> Tensor:
    """Reference scaled MXFP4 GEMM: dequantize inputs/scales then torch.mm."""
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.T).T
    x_scales_f32 = e8m0_to_f32(
        x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    )
    w_scales_f32 = e8m0_to_f32(
        w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    )
    return torch.mm(x_f32 * x_scales_f32, w_f32 * w_scales_f32.T)
