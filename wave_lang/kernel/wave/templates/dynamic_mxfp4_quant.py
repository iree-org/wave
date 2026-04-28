# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Dynamic MXFP4 quantization kernel template.

Quantizes f32 activations to per-element E2M1 codes (unpacked, 0-15 in i8)
given precomputed per-element quantization scales.

The full dynamic quantization pipeline is:

1. **Scale computation** (``compute_mxfp4_scales``, pure PyTorch on GPU):
   per-32-element block amax → E8M0 scale bytes + broadcasted quant_scale.
2. **FP4 encoding** (Wave kernel): comparison-based E2M1 encoding of
   ``x * quant_scale`` into i8 codes [0..15].
3. **Nibble packing** (``pack_mxfp4_codes``): pairs of i8 codes → uint8 bytes.

The scale computation uses a wave-incompatible 32-element reduction (AMD wave64
requires ≥ 64 elements for a wave-level reduce), so it stays in PyTorch.
"""

import torch
from torch import Tensor

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE
from wave_lang.kernel.wave.compile import WaveCompileOptions


SCALE_GROUP_SIZE = 32


def compute_mxfp4_scales(
    x: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-block E8M0 scales and broadcasted quant_scale for MXFP4.

    Runs entirely on the same device as *x* using standard PyTorch ops.

    Args:
        x: float32 tensor ``[M, K]``.  *K* must be divisible by 32.

    Returns:
        quant_scale: ``f32[M, K]`` -- per-element scale factor (broadcasted)
        bs_e8m0:     ``uint8[M, K // 32]`` -- biased E8M0 scale bytes
        scale_exp:   ``f32[M, K // 32, 1]`` -- unbiased exponents (for debug)
    """
    M, K = x.shape
    x_blocked = x.reshape(M, K // SCALE_GROUP_SIZE, SCALE_GROUP_SIZE)

    amax = x_blocked.abs().amax(dim=-1, keepdim=True)

    amax_bits = amax.contiguous().view(torch.int32)
    amax_bits = amax_bits + 0x200000
    mask = torch.tensor(-8388608, dtype=torch.int32, device=x.device)
    amax_bits = amax_bits & mask
    amax_pow2 = amax_bits.contiguous().view(torch.float32)

    scale_exp = torch.log2(amax_pow2).floor() - 2
    scale_exp = scale_exp.clamp(-127, 127)

    quant_scale_group = torch.exp2(-scale_exp)
    quant_scale = quant_scale_group.expand_as(x_blocked).reshape(M, K)

    bs_e8m0 = (scale_exp + 127).to(torch.uint8).squeeze(-1)

    return quant_scale, bs_e8m0, scale_exp


def get_dynamic_mxfp4_quant_kernel(
    shape: tuple[int, int],
    block_m: int = 2,
):
    """Return a Wave kernel + compile options for MXFP4 FP4 encoding.

    The kernel reads ``x[M, K]`` (f32) and precomputed ``quant_scale[M, K]``
    (f32), multiplies each element by its scale, and comparison-encodes the
    result to a 4-bit E2M1 code (stored as i8 in [0, 15]).

    Args:
        shape: ``(M, K)`` problem size.  *K* must be divisible by 32.
        block_m: Rows per workgroup tile.  ``block_m * 32`` must equal 64.

    Returns:
        ``(kernel_fn, WaveCompileOptions)``
    """
    M_val, K_val = shape
    assert K_val % 32 == 0
    block_k = 32

    M = tkl.sym.M
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(K, BLOCK_K, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(K, BLOCK_K),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: BLOCK_M, K: BLOCK_K},
        ),
    ]

    @tkw.wave(constraints)
    def quant(
        x: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f32],
        quant_scale: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f32],
        x_codes: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
    ):
        x_reg = tkw.read(x)
        qs = tkw.read(quant_scale)
        qx = x_reg * qs
        abs_qx = tkw.abs(qx)

        T025 = tkl.Register[M, K, tkl.f32](0.25)
        T075 = tkl.Register[M, K, tkl.f32](0.75)
        T125 = tkl.Register[M, K, tkl.f32](1.25)
        T175 = tkl.Register[M, K, tkl.f32](1.75)
        T250 = tkl.Register[M, K, tkl.f32](2.5)
        T350 = tkl.Register[M, K, tkl.f32](3.5)
        T500 = tkl.Register[M, K, tkl.f32](5.0)

        mag = (
            tkw.cast(abs_qx >= T025, tkl.f32)
            + tkw.cast(abs_qx >= T075, tkl.f32)
            + tkw.cast(abs_qx >= T125, tkl.f32)
            + tkw.cast(abs_qx >= T175, tkl.f32)
            + tkw.cast(abs_qx >= T250, tkl.f32)
            + tkw.cast(abs_qx >= T350, tkl.f32)
            + tkw.cast(abs_qx >= T500, tkl.f32)
        )

        ZERO_F32 = tkl.Register[M, K, tkl.f32](0.0)
        EIGHT_F32 = tkl.Register[M, K, tkl.f32](8.0)
        sign = tkw.select(qx < ZERO_F32, EIGHT_F32, ZERO_F32)

        codes = tkw.cast(mag + sign, tkl.i8)
        tkw.write(codes, x_codes)

    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_K: block_k,
        M: M_val,
        K: K_val,
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
    )

    return quant, options


def pack_mxfp4_codes(codes):
    """Pack unpacked i8 E2M1 codes [M, K] into nibble-packed uint8 [M, K/2].

    Even-indexed codes go into the low nibble, odd-indexed into the high nibble,
    matching the layout expected by ``mxfp4_to_f32`` and the MXFP4 GEMM kernel.
    """
    c = codes.to(torch.uint8)
    return c[:, 0::2] + c[:, 1::2] * 16
