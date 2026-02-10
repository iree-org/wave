"""
Attention Scheduling: Custom 4-Cluster Ping-Pong Schedule

This example demonstrates how to use a custom wave schedule for attention that
implements a 4-cluster ping-pong pattern matching create_attention_clusters
from schedule_reordering.py.

The schedule and tagged kernel template are defined in:
- wave_lang.kernel.wave.templates.tagged_attention - BSHD attention kernel with tags
- wave_lang.kernel.wave.schedules.attention_prefetch - 4-cluster ping-pong schedule

The schedule uses 4 clusters with workgroup barriers for synchronization:

- Cluster 0: QK computation + softmax1
  - SetWavePrio(1), MMA0 (QK), SetWavePrio(0)
  - softmax1 operations (sub_x, exp, mul_init, sum, cast, scale)
  - WorkgroupBarrier, SchedulingBarrier

- Cluster 1: K data movement + V shared load
  - GatherToLDS K (global -> shared)
  - Shared load V (from previous iteration's prefetch)
  - WorkgroupBarrier, SchedulingBarrier

- Cluster 2: PV computation + softmax0
  - SetWavePrio(1), MMA1 (PV), SetWavePrio(0)
  - softmax0 operations (max, sub_delta, exp_delta, masking)
  - WorkgroupBarrier, SchedulingBarrier

- Cluster 3: V data movement + K shared load
  - GatherToLDS V (global -> shared)
  - Shared load K (from current iteration's prefetch)
  - WorkgroupBarrier, SchedulingBarrier

The stagger() call enables ping-pong execution where wave groups alternate
between clusters, allowing one group to compute while another prefetches.

IMPORTANT: This schedule requires 8 waves (num_waves=8) to work correctly
since it is a ping-pong schedule using multi-buffering with wave staggering.
"""

import math

import torch
from utils import list_tests, parse_args, run_test

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.schedules.attention_prefetch import (
    get_attention_prefetch_schedule,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType

# Import templates and schedules
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.templates.tagged_attention import (
    get_tagged_bshd_attention_kernel,
)
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config


def test_attention_manual_schedule(is_debug=False):
    """
    Attention with custom 4-cluster ping-pong schedule.

    This demonstrates using the tagged attention kernel template with the
    4-cluster ping-pong schedule from attention_prefetch.py. The kernel uses
    BSHD layout:
    - Q: [B, N_Q, H, D_Q] - Query tensor (Batch, QuerySeq, Heads, HeadDim)
    - K: [B, N_KV, H_KV, D_Q] - Key tensor
    - V: [B, N_KV, H_KV, D_KV] - Value tensor
    - C: [B, N_Q, H, D_KV] - Output tensor

    The 4-cluster schedule pattern:
    - Cluster 0: QK MMA + softmax1 (compute on K data)
    - Cluster 1: GatherToLDS K + shared_load V (K prefetch, V consume)
    - Cluster 2: PV MMA + softmax0 (compute on V data)
    - Cluster 3: GatherToLDS V + shared_load K (V prefetch, K consume)

    Wave staggering enables ping-pong execution between wave groups, allowing
    one group to compute while another prefetches data.
    """
    shape = AttentionShape(
        num_query_heads=64,
        num_kv_heads=64,
        query_seq_len=16384,
        head_size_kv=128,
        head_size=128,
        kv_seq_len=16384,
    )
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2

    # Get the tagged BSHD attention kernel
    tagged_attention, hyperparams, _ = get_tagged_bshd_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims=False,
        is_causal=False,
        num_waves=8,  # 8 waves for ping-pong scheduling
    )
    hyperparams.update(get_default_scheduling_params())

    # set the unroll factor
    UNROLL_FACTOR = tkl.sym.UNROLL_FACTOR
    hyperparams[UNROLL_FACTOR] = 4

    # Get the prefetch schedule
    attention_prefetch_schedule = get_attention_prefetch_schedule()

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_global_to_shared=True,  # Enable GatherToLDS
        print_ir_after="all" if is_debug else [],
        use_buffer_ops=True,
        postprocess="""
        module attributes {transform.with_named_sequence} {
            transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
                %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
                transform.loop.unroll %0 { factor = %%UNROLL_FACTOR%% } : !transform.any_op
                transform.yield
            }
        }
        """,
        linearize_shared_access=False,  # This impacts the VGPR spills
    )

    options = set_default_run_config(options)

    # Compile with the custom schedule
    compiled_attention = wave_compile(
        options, tagged_attention, attention_prefetch_schedule
    )

    if is_debug:
        with open("attention_manual_schedule.asm", "w") as f:
            f.write(compiled_attention.asm)
        print("=== MANUAL ATTENTION SCHEDULE (GatherToLDS) ===")
        print(compiled_attention.asm)

    # Create test data - BSHD layout: [Batch, Seq, Heads, Dim]
    # For the tagged kernel, B=1, H=num_query_heads, H_KV=num_kv_heads
    batch = 1
    num_query_heads = shape.num_query_heads
    num_kv_heads = shape.num_kv_heads
    seq_q = shape.query_seq_len
    seq_kv = shape.kv_seq_len
    head_size = shape.head_size
    head_size_kv = shape.head_size_kv

    q = torch.randn(
        batch, seq_q, num_query_heads, head_size, dtype=torch.float16, device="cuda"
    )
    k = torch.randn(
        batch, seq_kv, num_kv_heads, head_size, dtype=torch.float16, device="cuda"
    )
    v = torch.randn(
        batch, seq_kv, num_kv_heads, head_size_kv, dtype=torch.float16, device="cuda"
    )
    c = torch.zeros(
        batch, seq_q, num_query_heads, head_size_kv, dtype=torch.float32, device="cuda"
    )

    # Run the kernel
    compiled_attention(q, k, v, c)

    # Compute reference using PyTorch
    # Reshape for standard attention computation
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / head_size)
    scale = dk_sqrt * log2e

    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()

    # Transpose for attention: [B, H, S, D]
    q_t = q_f32.permute(0, 2, 1, 3)  # [B, H, N_Q, D_Q]
    k_t = k_f32.permute(0, 2, 1, 3)  # [B, H_KV, N_KV, D_Q]
    v_t = v_f32.permute(0, 2, 1, 3)  # [B, H_KV, N_KV, D_KV]

    # For GQA, repeat K/V heads to match Q heads
    head_ratio = num_query_heads // num_kv_heads
    if head_ratio > 1:
        k_t = k_t.repeat_interleave(head_ratio, dim=1)  # [B, H, N_KV, D_Q]
        v_t = v_t.repeat_interleave(head_ratio, dim=1)  # [B, H, N_KV, D_KV]

    # Attention: Q @ K^T
    attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    # Our kernel uses exp2, so adjust softmax
    attn_probs = torch.softmax(attn_scores * math.log(2), dim=-1)
    # Output: attn @ V
    output = torch.matmul(attn_probs, v_t)  # [B, H, N_Q, D_KV]

    # Transpose back to BSHD: [B, N_Q, H, D_KV]
    expected = output.permute(0, 2, 1, 3)

    assert torch.allclose(c.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)

    print("Attention manual schedule test passed!")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(args.test, globals(), args.debug, args.repeat)
    exit(0 if success else 1)
