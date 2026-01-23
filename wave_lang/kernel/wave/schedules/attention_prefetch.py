# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Attention Prefetch Schedule

This module provides a custom wave schedule for attention that implements
a 4-cluster ping-pong pattern matching create_attention_clusters from
schedule_reordering.py.

IMPORTANT: This schedule requires 8 waves (num_waves=8) to work correctly
since it is a ping-pong schedule using multi-buffering with wave staggering.

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

This schedule expects a kernel with the following tags:
- "n_kv_loop": The main iteration loop over KV sequence
- "read_k": K read - uses GatherToLDS (global->shared) + Read (shared load)
- "read_v": V read - uses GatherToLDS (global->shared) + Read (shared load)
- "mma_qk", "mma_qk_init": QK MMA operations (GEMM0)
- "mma_pv": PV MMA operation (GEMM1)
- "softmax0_*": Softmax0 ops (permute, masking, scale, max, sub_delta, exp_delta)
- "softmax1_*": Softmax1 ops (sub_x, exp, mul_init, sum, cast, scale)
- Requires use_global_to_shared=True in WaveCompileOptions
- Requires num_waves=8 in the kernel creation
"""

import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *


def _get_node_by_tag_optional(tag: str):
    """
    Get nodes by tag, returning None if no nodes found.
    This allows handling optional operations like causal masking.
    """
    result = tkw.get_node_by_tag(tag)
    # get_node_by_tag returns empty list if no nodes found
    if result is None or (isinstance(result, (list, tuple)) and len(result) == 0):
        return None
    return result


def get_attention_prefetch_schedule():
    """
    Returns a schedule function that implements the prefetch attention pattern
    with 4-stage pipelining (8 cycles, II=2) using GatherToLDS.

    Returns:
        A wave_schedule decorated function
    """

    @wave_schedule.wave_schedule()
    def attention_prefetch_schedule():
        """
        Custom schedule implementing 4-cluster ping-pong attention pattern.

        Cluster ordering matches create_attention_clusters from schedule_reordering.py:
        - Cluster 0: QK MMA + softmax1 (compute on K data)
        - Cluster 1: GatherToLDS K + shared_load V (K prefetch, V consume)
        - Cluster 2: PV MMA + softmax0 (compute on V data)
        - Cluster 3: GatherToLDS V + shared_load K (V prefetch, K consume)

        Wave staggering enables ping-pong execution between wave groups.
        """
        # Get the main iteration loop
        n_kv_loop = tkw.get_node_by_tag("n_kv_loop")

        # ==================== Get K read operations (GEMM0) ====================
        all_read_k = tkw.get_node_by_tag("read_k")
        global_to_shared_k = tkw.filter_nodes(all_read_k, node_type=tkw.GatherToLDS)
        shared_load_k = tkw.filter_nodes(all_read_k, node_type=tkw.Read)

        # ==================== Get V read operations (GEMM1) ====================
        all_read_v = tkw.get_node_by_tag("read_v")
        global_to_shared_v = tkw.filter_nodes(all_read_v, node_type=tkw.GatherToLDS)
        shared_load_v = tkw.filter_nodes(all_read_v, node_type=tkw.Read)

        # ==================== Get MMA operations ====================
        mma_qk_init = tkw.get_node_by_tag("mma_qk_init")
        mma_qk = tkw.get_node_by_tag("mma_qk")
        mma_pv = tkw.get_node_by_tag("mma_pv")

        # ==================== Get softmax0 operations (before last sub) ====================
        # Includes masking operations which are grouped with softmax0
        softmax0_permute = tkw.get_node_by_tag("softmax0_permute")
        softmax0_self_index_kv = tkw.get_node_by_tag("softmax0_self_index_kv")
        softmax0_apply_expr = tkw.get_node_by_tag("softmax0_apply_expr")
        softmax0_broadcast_mask = tkw.get_node_by_tag("softmax0_broadcast_mask")
        # Causal operations are optional (only present when is_causal=True)
        softmax0_self_index_q = _get_node_by_tag_optional("softmax0_self_index_q")
        softmax0_broadcast_q = _get_node_by_tag_optional("softmax0_broadcast_q")
        softmax0_causal_cmp = _get_node_by_tag_optional("softmax0_causal_cmp")
        softmax0_causal_and = _get_node_by_tag_optional("softmax0_causal_and")
        softmax0_cast_mask = tkw.get_node_by_tag("softmax0_cast_mask")
        softmax0_select_bias = tkw.get_node_by_tag("softmax0_select_bias")
        softmax0_add_bias = tkw.get_node_by_tag("softmax0_add_bias")
        softmax0_scale = tkw.get_node_by_tag("softmax0_scale")
        softmax0_max = tkw.get_node_by_tag("softmax0_max")
        softmax0_sub_delta = tkw.get_node_by_tag("softmax0_sub_delta")
        softmax0_exp_delta = tkw.get_node_by_tag("softmax0_exp_delta")

        # ==================== Get softmax1 operations (from last sub onward) ====================
        softmax1_sub_x = tkw.get_node_by_tag("softmax1_sub_x")
        softmax1_exp = tkw.get_node_by_tag("softmax1_exp")
        softmax1_mul_init = tkw.get_node_by_tag("softmax1_mul_init")
        softmax1_sum = tkw.get_node_by_tag("softmax1_sum")
        softmax1_cast = tkw.get_node_by_tag("softmax1_cast")
        softmax1_scale = tkw.get_node_by_tag("softmax1_scale")

        # ==================== Create 4-stage pipeline ====================
        # Matches PrefetchAttentionScheduler's 8-cycle, II=2 structure
        # Use simple stage assignments without barriers (barriers are added later via reorder_graph)
        pipeline_loop = tkw.pipeline(n_kv_loop)

        # Build softmax0 ops list (excluding None for non-causal)
        softmax0_ops = [
            softmax0_permute,
            softmax0_self_index_kv,
            softmax0_apply_expr,
            softmax0_broadcast_mask,
            softmax0_self_index_q,
            softmax0_broadcast_q,
            softmax0_causal_cmp,
            softmax0_causal_and,
            softmax0_cast_mask,
            softmax0_select_bias,
            softmax0_add_bias,
            softmax0_scale,
            softmax0_max,
            softmax0_sub_delta,
            softmax0_exp_delta,
        ]
        softmax0_ops = [op for op in softmax0_ops if op is not None and op]

        # Build softmax1 ops list
        softmax1_ops = [
            softmax1_sub_x,
            softmax1_exp,
            softmax1_mul_init,
            softmax1_sum,
            softmax1_cast,
            softmax1_scale,
        ]

        with pipeline_loop as pl:
            # Stage 0 (cycles 0-1): GatherToLDS K
            pl.set_stage(
                [
                    (global_to_shared_k,),  # cycle 0: K global->shared
                    (),  # cycle 1
                ],
            )
            # Stage 1 (cycles 2-3): shared_load K + GatherToLDS V
            pl.set_stage(
                [
                    (shared_load_k,),  # cycle 0: K shared load
                    (global_to_shared_v,),  # cycle 1: V global->shared
                ],
            )
            # Stage 2 (cycles 4-5): MMA0, softmax0 (including masking ops)
            pl.set_stage(
                [
                    (mma_qk_init, mma_qk),  # cycle 0: MMA0 (QK)
                    tuple(softmax0_ops),  # cycle 1: softmax0
                ],
            )
            # Stage 3 (cycles 6-7): softmax1, shared_load V, MMA1
            pl.set_stage(
                [
                    tuple(softmax1_ops),  # cycle 0: softmax1
                    (shared_load_v, mma_pv),  # cycle 1: V shared read + MMA1 (PV)
                ],
            )

        # ==================== Apply ping-pong cluster ordering and staggering ====================
        # Filter nodes for the KERNEL stage only
        global_to_shared_k_kernel = tkw.filter_nodes(
            global_to_shared_k, subgraph=pipeline_loop.KERNEL
        )
        shared_load_k_kernel = tkw.filter_nodes(
            shared_load_k, subgraph=pipeline_loop.KERNEL
        )
        global_to_shared_v_kernel = tkw.filter_nodes(
            global_to_shared_v, subgraph=pipeline_loop.KERNEL
        )
        shared_load_v_kernel = tkw.filter_nodes(
            shared_load_v, subgraph=pipeline_loop.KERNEL
        )
        mma_qk_init_kernel = tkw.filter_nodes(
            mma_qk_init, subgraph=pipeline_loop.KERNEL
        )
        mma_qk_kernel = tkw.filter_nodes(mma_qk, subgraph=pipeline_loop.KERNEL)
        mma_pv_kernel = tkw.filter_nodes(mma_pv, subgraph=pipeline_loop.KERNEL)
        softmax0_ops_kernel = [
            tkw.filter_nodes(op, subgraph=pipeline_loop.KERNEL) for op in softmax0_ops
        ]
        softmax1_ops_kernel = [
            tkw.filter_nodes(op, subgraph=pipeline_loop.KERNEL) for op in softmax1_ops
        ]

        # Filter out empty lists from softmax ops (some ops may not be in KERNEL stage)
        # Note: Empty softmax ops lists are valid - some operations may be scheduled
        # in PROLOGUE or EPILOGUE stages depending on the pipeline structure.
        # Empty clusters with only barriers are allowed and handled correctly.
        softmax0_ops_kernel = [op for op in softmax0_ops_kernel if op]
        softmax1_ops_kernel = [op for op in softmax1_ops_kernel if op]

        # Create cluster ordering matching create_attention_clusters from schedule_reordering.py
        # This is the ping-pong pattern with 4 clusters:
        # - Cluster 0: QK computation + softmax1
        # - Cluster 1: K data movement + local load V
        # - Cluster 2: PV computation + softmax0
        # - Cluster 3: V data movement + local load K
        clusters = [
            # Cluster 0: QK computation and softmax1
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_qk_init_kernel,
                    mma_qk_kernel,
                    tkw.SetWavePrio(0),
                    *softmax1_ops_kernel,
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 1: K data movement (global_to_shared_k) + local load V
            tkw.cluster(
                [
                    global_to_shared_k_kernel,
                    shared_load_v_kernel,
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: PV computation and softmax0
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_pv_kernel,
                    tkw.SetWavePrio(0),
                    *softmax0_ops_kernel,
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: V data movement (global_to_shared_v) + local load K
            tkw.cluster(
                [
                    global_to_shared_v_kernel,
                    shared_load_k_kernel,
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)
        tkw.stagger(pipeline_loop.KERNEL)

    return attention_prefetch_schedule
