# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tagged Attention Kernel for use with custom wave schedules.

Memory Layout: BSHD - Q[B,N_Q,H,D_Q], K[B,N_KV,H_KV,D_Q], V[B,N_KV,H_KV,D_KV], C[B,N_Q,H,D_KV]

Tags enable schedulers to identify operations:
- read_q/read_k/read_v, mma_qk/mma_pv, write_output
- softmax0_*: ops before last sub (max, exp_delta)
- softmax1_*: ops from last sub onward (sub_x, exp, sum, cast, scale)
"""

import math

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType

from .attention_common import AttentionShape


def get_tagged_bshd_attention_kernel(
    shape: AttentionShape,
    mfma_variant: list[MMAType],
    dynamic_dims: bool,
    is_causal: bool = False,
    num_waves: int = 8,
):
    """
    Create a tagged BSHD attention kernel for use with custom schedules.

    Args:
        shape: AttentionShape containing dimension sizes
        mfma_variant: Tuple of MMA types for QK and PV GEMMs
        dynamic_dims: Whether to use dynamic dimensions
        is_causal: Whether to apply causal masking
        num_waves: Number of waves (default 8 for ping-pong scheduling)

    Returns:
        Tuple of (kernel_function, hyperparams, dynamic_symbols)
    """
    B = tkl.sym.B
    N_Q = tkl.sym.N_Q
    N_KV = tkl.sym.N_KV
    D_Q = tkl.sym.D_Q
    D_KV = tkl.sym.D_KV
    H = tkl.sym.H
    H_KV = tkl.sym.H_KV

    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_N_Q = tkl.sym.BLOCK_N_Q
    BLOCK_N_KV = tkl.sym.BLOCK_N_KV
    BLOCK_D_KV = tkl.sym.BLOCK_D_KV
    BLOCK_H = tkl.sym.BLOCK_H
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(N_Q, BLOCK_N_Q, 0)]
    constraints += [tkw.WorkgroupConstraint(D_KV, BLOCK_D_KV, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 3)]
    constraints += [tkw.WorkgroupConstraint(H_KV, BLOCK_H, 3, primary=False)]
    constraints += [tkw.TilingConstraint(N_KV, BLOCK_N_KV)]
    constraints += [tkw.WaveConstraint(N_Q, BLOCK_N_Q / num_waves)]
    constraints += [tkw.WaveConstraint(D_KV, BLOCK_D_KV / 1)]

    if mfma_variant[1] == MMAType.F32_16x16x16_F16:
        Mvec, Nvec, TPW = 16, 16, 64
    elif mfma_variant[1] == MMAType.F32_16x16x32_F16:
        Mvec, Nvec, TPW = 16, 16, 64
    elif mfma_variant[1] == MMAType.F32_32x32x8_F16:
        Mvec, Nvec, TPW = 32, 32, 64
    elif mfma_variant[1] == MMAType.RDNA4_WAVE32_F32_16x16x16_F16:
        Mvec, Nvec, TPW = 16, 16, 32
    else:
        raise ValueError(f"Unsupported MMA variant: {mfma_variant[1]}")

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=TPW,
            mma_type=mfma_variant[1],
            vector_shapes={B: 0, H: 0, H_KV: 0, N_Q: Mvec, D_KV: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(N_KV > BLOCK_N_KV * 4)]

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    scale = dk_sqrt * log2e

    # GQA validation: num_query_heads must be divisible by num_kv_heads
    if shape.num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {shape.num_kv_heads}")
    if shape.num_query_heads % shape.num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads ({shape.num_query_heads}) must be divisible by "
            f"num_kv_heads ({shape.num_kv_heads}) for GQA"
        )
    head_ratio = shape.num_query_heads // shape.num_kv_heads

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)

    output_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, D_KV: k, N_Q: l},
        outputs={B: i, N_Q: l, H: j, D_KV: k},
    )
    q_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, N_Q: k, D_Q: l},
        outputs={B: i, H: j, N_Q: k, D_Q: l},
    )
    # GQA: K/V use j // head_ratio to map query head to KV head
    k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j // head_ratio, N_KV: k, D_Q: l},
        outputs={B: i, H_KV: j, N_KV: k, D_Q: l},
    )
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j // head_ratio, D_KV: k, N_KV: l},
        outputs={B: i, H_KV: j, D_KV: k, N_KV: l},
    )

    @tkw.wave(constraints)
    def tagged_attention(
        q: tkl.Memory[B, N_Q, H, D_Q, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, N_KV, H_KV, D_Q, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N_KV, H_KV, D_KV, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, N_Q, H, D_KV, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, H, D_KV, N_Q, tkl.f32](0.0)
        init_sum = tkl.Register[B, H, N_Q, tkl.f32](0.0)
        init_max = tkl.Register[B, H, N_Q, tkl.f32](-1e6)
        qkv_scaling = tkl.Register[B, H, N_Q, D_Q, tkl.f16](scale)
        ZEROF = tkl.Register[N_Q, N_KV, tkl.f32](0.0)
        MIN_INF = tkl.Register[N_Q, N_KV, tkl.f32](-1e6)

        @tkw.iterate(N_KV, init_args=[init_max, init_sum, c_reg], tag="n_kv_loop")
        def repeat(
            partial_max: tkl.Register[B, H, N_Q, tkl.f32],
            partial_sum: tkl.Register[B, H, N_Q, tkl.f32],
            acc: tkl.Register[B, H, D_KV, N_Q, tkl.f32],
        ):
            # GEMM0: Q @ K^T
            imm_reg = tkw.register((B, H, N_KV, N_Q), tkl.f32, 0.0, tag="mma_qk_init")
            q_reg = tkw.read(q, mapping=q_mapping, tag="read_q")
            q_reg = tkw.tag(q_reg * qkv_scaling, "softmax0_scale")
            k_reg = tkw.read(k, mapping=k_mapping, tag="read_k")
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0], tag="mma_qk")
            x_j = tkw.permute(
                inner_acc, target_shape=[B, H, N_Q, N_KV], tag="softmax0_permute"
            )

            # Masking
            n_kv_index = tkw.self_index(N_KV, tkl.i32, tag="softmax0_self_index_kv")
            mask = tkw.apply_expr(
                n_kv_index, lambda x: x < N_KV, tag="softmax0_apply_expr"
            )
            mask = tkw.broadcast(
                mask, target_shape=[N_Q, N_KV], tag="softmax0_broadcast_mask"
            )
            if is_causal:
                n_q_index = tkw.self_index(N_Q, tkl.i32, tag="softmax0_self_index_q")
                n_q_index = tkw.broadcast(
                    n_q_index, target_shape=[N_Q, N_KV], tag="softmax0_broadcast_q"
                )
                mask = tkw.tag(n_q_index >= n_kv_index, "softmax0_causal_cmp")
                mask = tkw.tag(mask & mask, "softmax0_causal_and")
            mask = tkw.cast(mask, tkw.i1, tag="softmax0_cast_mask")
            bias = tkw.select(mask, ZEROF, MIN_INF, tag="softmax0_select_bias")
            x_j = tkw.tag(x_j + bias, "softmax0_add_bias")

            # Softmax0: ops before last sub
            m_j = tkw.max(x_j, partial_max, dim=N_KV, tag="softmax0_max")
            delta_max_sub = tkw.tag(partial_max - m_j, "softmax0_sub_delta")
            e_delta_max = tkw.exp2(delta_max_sub, tag="softmax0_exp_delta")

            # Softmax1: from last sub onward
            x_sub = tkw.tag(x_j - m_j, "softmax1_sub_x")
            e_delta = tkw.exp2(x_sub, tag="softmax1_exp")
            e_init = tkw.tag(partial_sum * e_delta_max, "softmax1_mul_init")
            d_j = tkw.sum(e_delta, e_init, dim=N_KV, tag="softmax1_sum")
            imm_f16 = tkw.cast(e_delta, tkl.f16, tag="softmax1_cast")
            new_acc = tkw.tag(acc * e_delta_max, "softmax1_scale")

            # GEMM1: softmax(QK) @ V
            v_reg = tkw.read(v, mapping=v_mapping, tag="read_v")
            acc = tkw.mma(v_reg, imm_f16, new_acc, mfma_variant[1], tag="mma_pv")
            return m_j, d_j, acc

        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum, tag="epilog_reciprocal")
        res = tkw.tag(res_mm * reciprocal_sum, "epilog_normalize")
        tkw.write(res, c, mapping=output_mapping, tag="write_output")

    # BLOCK_N_Q scales with num_waves: BASE_BLOCK_N_Q elements per WAVES_PER_BLOCK_FACTOR waves
    # For 4 waves: BLOCK_N_Q=128, for 8 waves: BLOCK_N_Q=256
    WAVES_PER_BLOCK_FACTOR = 4
    BASE_BLOCK_N_Q = 128

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_H: 1,
        BLOCK_N_Q: BASE_BLOCK_N_Q * (num_waves // WAVES_PER_BLOCK_FACTOR),
        BLOCK_D_KV: shape.head_size_kv,
        BLOCK_N_KV: 64,
        B: 1,
        H: shape.num_query_heads,
        H_KV: shape.num_kv_heads,
        N_Q: shape.query_seq_len,
        D_KV: shape.head_size_kv,
        D_Q: shape.head_size,
        N_KV: shape.kv_seq_len,
    }

    dynamic_symbols = []
    if dynamic_dims:
        dynamic_symbols.append(N_Q)
        dynamic_symbols.append(D_KV)
        dynamic_symbols.append(B)
        dynamic_symbols.append(N_KV)
        del hyperparams[N_Q]
        del hyperparams[D_KV]
        del hyperparams[B]
        del hyperparams[N_KV]

    return tagged_attention, hyperparams, dynamic_symbols
