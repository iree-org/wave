# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
E2E tests for manually scheduled attention kernels.

These tests verify that the tagged attention kernel with manual scheduling
produces correct results and can be successfully compiled and executed
on CDNA4 (gfx95*) hardware.
"""

import pytest
import torch
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.templates.tagged_attention import (
    get_tagged_bshd_attention_kernel,
)
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.schedules.attention_prefetch import (
    get_attention_prefetch_schedule,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from torch.testing import assert_close
from ..common.utils import require_e2e, require_cdna4


def reference_attention(q, k, v, is_causal=False):
    """Reference attention using PyTorch scaled_dot_product_attention."""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=is_causal
    )


# Test configurations: (num_query_heads, num_kv_heads, query_seq_len, kv_seq_len, head_size, head_size_kv, num_waves, schedule_type)
# BLOCK_N_Q = 128 * (num_waves // 4), so:
#   - num_waves=4: BLOCK_N_Q=128, requires query_seq_len >= 128
#   - num_waves=8: BLOCK_N_Q=256, requires query_seq_len >= 256
test_configs = [
    # No scheduling (baseline) - smaller config with num_waves=4
    (1, 1, 128, 128, 64, 64, 4, SchedulingType.NONE),
    # Manual scheduling with various shapes
    (1, 1, 256, 256, 64, 64, 8, SchedulingType.MANUAL),
    (4, 4, 256, 256, 64, 64, 8, SchedulingType.MANUAL),
    (8, 2, 256, 256, 64, 64, 8, SchedulingType.MANUAL),  # GQA
    # Larger sequence to test pipelining
    (4, 4, 512, 512, 64, 64, 8, SchedulingType.MANUAL),
]

# MMA variants: (mma_qk, mma_pv)
mma_variants = [
    (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
    (MMAType.F32_16x16x32_F16, MMAType.F32_16x16x16_F16),
]


@require_e2e
@require_cdna4
@pytest.mark.parametrize("config", test_configs)
@pytest.mark.parametrize("mma_variant", mma_variants)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_tagged_attention(
    config: tuple,
    mma_variant: tuple[MMAType, MMAType],
    is_causal: bool,
    dtype: torch.dtype,
    run_bench,
    perf_filename_tk,
):
    """
    Test the tagged attention kernel with various configurations.

    Tests both unscheduled (NONE) and manually scheduled (MANUAL) variants
    with different shapes, sequence lengths, causal modes, and MMA types.
    """
    (
        num_query_heads,
        num_kv_heads,
        query_seq_len,
        kv_seq_len,
        head_size,
        head_size_kv,
        num_waves,
        schedule_type,
    ) = config

    attention_shape = AttentionShape(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        head_size=head_size,
        head_size_kv=head_size_kv,
    )

    mfma_variant = mma_variant

    tagged_attention, hyperparams, _ = get_tagged_bshd_attention_kernel(
        attention_shape,
        mfma_variant,
        dynamic_dims=False,
        is_causal=is_causal,
        num_waves=num_waves,
    )
    hyperparams.update(get_default_scheduling_params())

    # Get schedule if using manual scheduling
    attention_schedule = (
        get_attention_prefetch_schedule()
        if schedule_type == SchedulingType.MANUAL
        else None
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=schedule_type,
        use_scheduling_barriers=False,
        use_global_to_shared=True,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
    )

    options = set_default_run_config(options)
    compiled_attention = (
        wave_compile(options, tagged_attention, attention_schedule)
        if attention_schedule
        else wave_compile(options, tagged_attention)
    )

    # Create input tensors - BSHD layout: [B, N, H, D]
    B = 1
    q = device_randn((B, query_seq_len, num_query_heads, head_size), dtype=dtype)
    k = device_randn((B, kv_seq_len, num_kv_heads, head_size), dtype=dtype)
    v = device_randn((B, kv_seq_len, num_kv_heads, head_size_kv), dtype=dtype)
    output = device_zeros(
        (B, query_seq_len, num_query_heads, head_size_kv), dtype=torch.float32
    )

    compiled_attention(q, k, v, output)

    # Compute reference - PyTorch expects BHND layout
    q_ref = q.permute(0, 2, 1, 3)
    k_ref = k.permute(0, 2, 1, 3)
    v_ref = v.permute(0, 2, 1, 3)

    # Handle GQA by repeating KV heads for reference
    if num_query_heads != num_kv_heads:
        repeat_factor = num_query_heads // num_kv_heads
        k_ref = k_ref.repeat_interleave(repeat_factor, dim=1)
        v_ref = v_ref.repeat_interleave(repeat_factor, dim=1)

    ref_output = reference_attention(q_ref, k_ref, v_ref, is_causal=is_causal)
    ref_output = ref_output.permute(0, 2, 1, 3)

    assert_close(output, ref_output, check_dtype=False, atol=1e-3, rtol=1e-3)
