# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MXFP4 Scaled GEMM Double Buffer Schedule for CDNA4 (GFX950)

Reusable 2-stage pipeline schedule for MXFP4 scaled GEMM on GFX950.
Handles 4 input tensors (A data, A scale, B data, B scale) with bitcasts.

Stage 0: GatherToLDS async prefetch | Stage 1: shared loads + bitcasts + MMA
K-dimension partitioned into 2 halves for memory/compute interleaving.

Required kernel tags: k_loop, read_a, read_a_scale, read_b, read_b_scale,
bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale, scaled_mma.
Requires use_global_to_shared=True and threads_per_wave=64.
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.wave_schedule as wave_schedule


def get_mxfp4_dbuf_schedule(use_stagger: bool = True):
    """Return a double-buffered MXFP4 schedule for wave_compile().

    Args:
        use_stagger: Enable wave staggering + WorkgroupBarrier in cluster 0.
            Recommended for 8-wave configs; disable for 4-wave.
    """
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix A data - GatherToLDS (global->shared) + Read (shared load)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        global_to_shared_a_scale = tkw.filter_nodes(
            all_read_a_scale, node_type=tkw.GatherToLDS
        )
        shared_load_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        # Matrix B data
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Matrix B scale
        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        global_to_shared_b_scale = tkw.filter_nodes(
            all_read_b_scale, node_type=tkw.GatherToLDS
        )
        shared_load_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Global-to-shared prefetch via GatherToLDS (no fusion)
            pl.set_stage(
                [
                    (
                        global_to_shared_a,
                        global_to_shared_a_scale,
                        global_to_shared_b,
                        global_to_shared_b_scale,
                    ),
                    (),
                    (),
                ],
            )
            # Stage 1: Shared memory loads + bitcasts + compute
            pl.set_stage(
                [
                    (
                        shared_load_a,
                        shared_load_b,
                        shared_load_a_scale,
                        shared_load_b_scale,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        loop_global_to_shared = (
            tkw.filter_nodes(global_to_shared_a, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_a_scale, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)
            + tkw.filter_nodes(global_to_shared_b_scale, subgraph=pipeline_loop.KERNEL)
        )

        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b = tkw.filter_nodes(
            shared_load_b, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_a_scale = tkw.filter_nodes(
            shared_load_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b_scale = tkw.filter_nodes(
            shared_load_b_scale, subgraph=pipeline_loop.KERNEL
        )

        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        # Partition by K dimension for interleaving compute with memory ops.
        # NOTE: Bitcasts MUST also be partitioned by K to match their producer
        # shared loads, otherwise reorder_graph fails with
        # "Cannot find producer(s)" because bitcasts in an earlier cluster
        # would depend on shared loads in a later cluster.
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_0, loop_shared_load_a_1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_b_0, loop_shared_load_b_1 = tkw.partition_by_dim(
            loop_shared_load_b, dim=K, num_partitions=2
        )
        loop_shared_load_a_scale_0, loop_shared_load_a_scale_1 = tkw.partition_by_dim(
            loop_shared_load_a_scale, dim=K, num_partitions=2
        )
        loop_shared_load_b_scale_0, loop_shared_load_b_scale_1 = tkw.partition_by_dim(
            loop_shared_load_b_scale, dim=K, num_partitions=2
        )
        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=K, num_partitions=2
        )
        loop_bitcast_a_scale_0, loop_bitcast_a_scale_1 = tkw.partition_by_dim(
            loop_bitcast_a_scale, dim=K, num_partitions=2
        )
        loop_bitcast_b_0, loop_bitcast_b_1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=K, num_partitions=2
        )
        loop_bitcast_b_scale_0, loop_bitcast_b_scale_1 = tkw.partition_by_dim(
            loop_bitcast_b_scale, dim=K, num_partitions=2
        )

        independent_global_count = len(loop_global_to_shared)

        # Build cluster 0: first K-partition loads + bitcasts + GatherToLDS
        cluster_0_ops = [
            loop_shared_load_a_0,
            loop_shared_load_a_scale_0,
            loop_shared_load_b_0,
            loop_shared_load_b_scale_0,
            loop_bitcast_a_0,
            loop_bitcast_a_scale_0,
            loop_bitcast_b_0,
            loop_bitcast_b_scale_0,
            tkw.SchedulingBarrier([]),
            loop_global_to_shared,
            tkw.SchedulingBarrier([]),
        ]
        if use_stagger:
            cluster_0_ops.extend(
                [
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ]
            )

        clusters = [
            # Cluster 0: First K-partition shared loads/bitcasts + async GatherToLDS
            tkw.cluster(cluster_0_ops),
            # Cluster 1: First K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=independent_global_count),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: Second K-partition shared loads/bitcasts
            tkw.cluster(
                [
                    loop_shared_load_a_1,
                    loop_shared_load_a_scale_1,
                    loop_shared_load_b_1,
                    loop_shared_load_b_scale_1,
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    loop_bitcast_b_1,
                    loop_bitcast_b_scale_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Second K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert shared memory barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply wave staggering for better overlap
        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return mxfp4_dbuf_schedule


def get_mxfp4_asymmetric_schedule():
    """Return an asymmetric-prefetch MXFP4 schedule for wave_compile().

    Asymmetric data paths:
      - A (data + scale): global -> LDS -> VGPRs, prefetch depth 2
        (triple-buffered in LDS).
      - B (data + scale): global -> VGPRs directly (no LDS).

    3-stage pipeline:
      Stage 0: Async global-to-LDS prefetch for A and A_scale.
      Stage 1: Global-to-VGPR loads for B and B_scale;
               LDS-to-VGPR loads for first M-partition of A.
      Stage 2: LDS-to-VGPR loads for second M-partition of A;
               bitcasts; scaled MMA accumulation.

    The main loop interleaves MMA with memory operations:
      First MMA half: interleaved with B loads and second-partition A reads.
      Second MMA half: interleaved with B_scale loads and next-iteration
                       first-partition A reads (plus G2S for the iteration
                       after next).
    """
    M = tkl.sym.M

    @wave_schedule.wave_schedule()
    def mxfp4_dbuf_schedule():
        # =====================================================================
        # Get tagged nodes from the kernel
        # =====================================================================
        k_loop = tkw.get_node_by_tag("k_loop")

        # Matrix A data - GatherToLDS (global->shared) + Read (shared load)
        all_read_a = tkw.get_node_by_tag("read_a")
        g2s_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        s2v_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Matrix A scale
        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        g2s_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.GatherToLDS)
        s2v_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        # partition s2v_a by M
        s2v_a_0, s2v_a_1 = tkw.partition_by_dim(s2v_a, dim=M, num_partitions=2)
        s2v_a_scale_0, s2v_a_scale_1 = tkw.partition_by_dim(
            s2v_a_scale, dim=M, num_partitions=2
        )

        # Matrix B data and B scale - Global to Vector
        g2v_b = tkw.get_node_by_tag("read_b")
        g2v_b_scale = tkw.get_node_by_tag("read_b_scale")

        # Bitcast operations (needed alongside compute)
        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        # Scaled MMA
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =====================================================================
        # Create 2-stage pipeline (double buffering)
        # =====================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (
                        g2s_a,
                        g2s_a_scale,
                    ),
                    (),
                    (),
                ],
            )
            # Stage 0: Global-to-shared prefetch via GatherToLDS (no fusion)
            pl.set_stage(
                [
                    (
                        g2v_b,
                        g2v_b_scale,
                    ),
                    (
                        s2v_a_0,
                        s2v_a_scale_0,
                    ),
                    (),
                ],
            )
            # Stage 1: Shared memory loads + bitcasts + compute
            pl.set_stage(
                [
                    (
                        s2v_a_1,
                        s2v_a_scale_1,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # Constants derived from schedule structure
        num_m_partitions = (
            2  # we are dividing the M dimension into 2 paritions per loop iteration
        )
        num_pf_iters = (
            2  # prefetch depth of A and A_scale is 2 iterations (triple buffer)
        )

        # =====================================================================
        # Prologue: G2S_A + G2S_A_scale + G2V_B + G2V_B_scale + vmcnt(25) + s2v_a_0 + s2v_a_scale_0
        # =====================================================================
        prologue_g2s_a = tkw.filter_nodes(g2s_a, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2s_a_scale = tkw.filter_nodes(
            g2s_a_scale, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_g2v_b = tkw.filter_nodes(g2v_b, subgraph=pipeline_loop.PROLOGUE)
        prologue_g2v_b_scale = tkw.filter_nodes(
            g2v_b_scale, subgraph=pipeline_loop.PROLOGUE
        )
        prologue_s2v_a_0 = tkw.filter_nodes(s2v_a_0, subgraph=pipeline_loop.PROLOGUE)
        prologue_s2v_a_scale_0 = tkw.filter_nodes(
            s2v_a_scale_0, subgraph=pipeline_loop.PROLOGUE
        )
        g2s_per_iter = (len(prologue_g2s_a) + len(prologue_g2s_a_scale)) // num_pf_iters
        must_complete = g2s_per_iter // num_m_partitions
        total_vmcnt = (
            len(prologue_g2s_a)
            + len(prologue_g2s_a_scale)
            + len(prologue_g2v_b)
            + len(prologue_g2v_b_scale)
        )
        prologue_vmcnt = total_vmcnt - must_complete
        prologue_clusters = [
            tkw.cluster(
                [
                    prologue_g2s_a,
                    prologue_g2s_a_scale,
                    prologue_g2v_b,
                    prologue_g2v_b_scale,
                    tkw.MemoryCounterWaitBarrier(load=prologue_vmcnt),
                    prologue_s2v_a_0,
                    prologue_s2v_a_scale_0,
                ],
            )
        ]

        # =====================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =====================================================================

        # Filter nodes for KERNEL stage
        loop_g2s_a = tkw.filter_nodes(g2s_a, subgraph=pipeline_loop.KERNEL)
        loop_g2s_a_scale = tkw.filter_nodes(g2s_a_scale, subgraph=pipeline_loop.KERNEL)

        loop_g2v_b = tkw.filter_nodes(g2v_b, subgraph=pipeline_loop.KERNEL)
        loop_g2v_b_scale = tkw.filter_nodes(g2v_b_scale, subgraph=pipeline_loop.KERNEL)

        loop_shared_load_a_0 = tkw.filter_nodes(s2v_a_0, subgraph=pipeline_loop.KERNEL)
        loop_shared_load_a_scale_0 = tkw.filter_nodes(
            s2v_a_scale_0, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_a_1 = tkw.filter_nodes(s2v_a_1, subgraph=pipeline_loop.KERNEL)
        loop_shared_load_a_scale_1 = tkw.filter_nodes(
            s2v_a_scale_1, subgraph=pipeline_loop.KERNEL
        )

        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)
        # Partition by K dimension for interleaving compute with memory ops.
        # NOTE: Bitcasts MUST also be partitioned by K to match their producer
        # shared loads, otherwise reorder_graph fails with
        # "Cannot find producer(s)" because bitcasts in an earlier cluster
        # would depend on shared loads in a later cluster.
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=M, num_partitions=2
        )
        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=M, num_partitions=2
        )
        loop_bitcast_a_scale_0, loop_bitcast_a_scale_1 = tkw.partition_by_dim(
            loop_bitcast_a_scale, dim=M, num_partitions=2
        )

        # Barrier count calculations
        g2s_per_iter = (len(prologue_g2s_a) + len(prologue_g2s_a_scale)) // num_pf_iters
        n_b_loads = len(loop_g2v_b) + len(loop_g2v_b_scale)

        # First A sub-block that must land in LDS before ds_read
        a_sub_block = g2s_per_iter // num_m_partitions

        total_prologue_loads = (
            len(prologue_g2s_a)
            + len(prologue_g2s_a_scale)
            + len(prologue_g2v_b)
            + len(prologue_g2v_b_scale)
        )

        prologue_wait_load = total_prologue_loads - a_sub_block

        # ===========================================================
        # Loop vmcnt: wait for first A sub-block + ALL B loads
        # ===========================================================
        loop_wait_load = prologue_wait_load - a_sub_block - n_b_loads

        # Insert MemoryCounterWaitBarrier at the start of the kernel
        tkw.insert_at_start(
            pipeline_loop.KERNEL, tkw.MemoryCounterWaitBarrier(load=loop_wait_load)
        )

        # Interleave MFMAs with memory ops (matching aiter f4gemm pattern).
        # First half: g2v_b (buffer_load_dwordx4) and shared_load_a_1
        # (ds_read_b128) interleaved every 4 MFMAs.
        interleaved_mma_0 = tkw.interleave_operations(
            base_ops=loop_scaled_mma_0,
            interleaved_ops=[
                loop_g2v_b,
                loop_shared_load_a_1,
                loop_shared_load_a_scale_1,
                loop_g2v_b_scale,
            ],
            intervals=[4, 4, 2, 4],
            start_offsets=[0, 3, 2, 0],
            start_after_groups=[[], [], [1], [0]],
        )

        interleaved_mma_1 = tkw.interleave_operations(
            base_ops=loop_scaled_mma_1,
            interleaved_ops=[
                loop_g2s_a,
                loop_shared_load_a_0,
                loop_shared_load_a_scale_0,
                loop_g2s_a_scale,
            ],
            intervals=[4, 4, 2, 4],
            start_offsets=[0, 3, 2, 0],
            start_after_groups=[[], [], [1], [0]],
        )

        clusters = [
            tkw.cluster(
                [
                    loop_bitcast_a_0,
                    loop_bitcast_a_scale_0,
                    loop_bitcast_b,
                    loop_bitcast_b_scale,
                    tkw.SchedulingBarrier([]),
                    interleaved_mma_0,
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    tkw.SchedulingBarrier([]),
                    interleaved_mma_1,
                    tkw.MemoryCounterWaitBarrier(load=n_b_loads, ds=0),
                    loop_shared_load_a_0,
                    loop_shared_load_a_scale_0,
                    tkw.SchedulingBarrier([]),
                ]
            ),
        ]

        #################### EPILOGUE ####################

        # Filter nodes for EPILOGUE stage

        epilogue_g2v_b = tkw.filter_nodes(g2v_b, subgraph=pipeline_loop.EPILOGUE)
        epilogue_g2v_b_scale = tkw.filter_nodes(
            g2v_b_scale, subgraph=pipeline_loop.EPILOGUE
        )
        epilogue_s2v_a_0 = tkw.filter_nodes(s2v_a_0, subgraph=pipeline_loop.EPILOGUE)
        epilogue_s2v_a_scale_0 = tkw.filter_nodes(
            s2v_a_scale_0, subgraph=pipeline_loop.EPILOGUE
        )
        epilogue_s2v_a_1 = tkw.filter_nodes(s2v_a_1, subgraph=pipeline_loop.EPILOGUE)
        epilogue_s2v_a_scale_1 = tkw.filter_nodes(
            s2v_a_scale_1, subgraph=pipeline_loop.EPILOGUE
        )
        epilogue_bitcast_a = tkw.filter_nodes(
            bitcast_a, subgraph=pipeline_loop.EPILOGUE
        )
        epilogue_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.EPILOGUE
        )
        epilogue_bitcast_b = tkw.filter_nodes(
            bitcast_b, subgraph=pipeline_loop.EPILOGUE
        )
        epilogue_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.EPILOGUE
        )

        epilogue_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.EPILOGUE)

        def split_by_iteration(nodes, key="name"):
            # TODO: Replace name-based splitting with a pipeline_drain_iteration
            # attribute (analogous to unroll_iteration). expanded_dims can't be
            # used here because loop_reconstruction copies them verbatim for
            # both drain iterations.
            itr0 = []
            itr1 = []
            for node in nodes:
                value = getattr(node, key)
                if "1_2" in value:
                    itr0.append(node)
                elif "2_2" in value:
                    itr1.append(node)
                else:
                    raise ValueError(f"Unknown {key} for node: {value}")
            return itr0, itr1

        epilogue_mma_itr0, epilogue_mma_itr1 = split_by_iteration(epilogue_mma)
        epilogue_s2v_a_1_itr0, epilogue_s2v_a_1_itr1 = split_by_iteration(
            epilogue_s2v_a_1
        )
        epilogue_s2v_a_scale_1_itr0, epilogue_s2v_a_scale_1_itr1 = split_by_iteration(
            epilogue_s2v_a_scale_1
        )
        epilogue_bitcast_a_itr0, epilogue_bitcast_a_itr1 = split_by_iteration(
            epilogue_bitcast_a
        )
        epilogue_bitcast_a_scale_itr0, epilogue_bitcast_a_scale_itr1 = (
            split_by_iteration(epilogue_bitcast_a_scale)
        )
        epilogue_bitcast_b_itr0, epilogue_bitcast_b_itr1 = split_by_iteration(
            epilogue_bitcast_b
        )
        epilogue_bitcast_b_scale_itr0, epilogue_bitcast_b_scale_itr1 = (
            split_by_iteration(epilogue_bitcast_b_scale)
        )

        epilogue_mma_itr0_0, epilogue_mma_itr0_1 = tkw.partition_by_dim(
            epilogue_mma_itr0, dim=M, num_partitions=2
        )
        epilogue_bitcast_a_itr0_0, epilogue_bitcast_a_itr0_1 = tkw.partition_by_dim(
            epilogue_bitcast_a_itr0, dim=M, num_partitions=2
        )
        epilogue_bitcast_a_scale_itr0_0, epilogue_bitcast_a_scale_itr0_1 = (
            tkw.partition_by_dim(epilogue_bitcast_a_scale_itr0, dim=M, num_partitions=2)
        )

        epilogue_mma_itr1_0, epilogue_mma_itr1_1 = tkw.partition_by_dim(
            epilogue_mma_itr1, dim=M, num_partitions=2
        )
        epilogue_bitcast_a_itr1_0, epilogue_bitcast_a_itr1_1 = tkw.partition_by_dim(
            epilogue_bitcast_a_itr1, dim=M, num_partitions=2
        )
        epilogue_bitcast_a_scale_itr1_0, epilogue_bitcast_a_scale_itr1_1 = (
            tkw.partition_by_dim(epilogue_bitcast_a_scale_itr1, dim=M, num_partitions=2)
        )

        epilogue_clusters_itr0 = [
            tkw.cluster(
                [
                    epilogue_bitcast_a_itr0_0,
                    epilogue_bitcast_a_scale_itr0_0,
                    epilogue_bitcast_b_itr0,
                    epilogue_bitcast_b_scale_itr0,
                    tkw.SchedulingBarrier([]),
                    epilogue_mma_itr0_0,
                    epilogue_g2v_b,
                    epilogue_s2v_a_1_itr0,
                    epilogue_g2v_b_scale,
                    epilogue_s2v_a_scale_1_itr0,
                    epilogue_bitcast_a_itr0_1,
                    epilogue_bitcast_a_scale_itr0_1,
                ],
            ),
            tkw.cluster(
                [
                    epilogue_mma_itr0_1,
                    tkw.SchedulingBarrier([]),
                    epilogue_s2v_a_0,
                    epilogue_s2v_a_scale_0,
                ],
            ),
            tkw.cluster(
                [
                    epilogue_bitcast_a_itr1_0,
                    epilogue_bitcast_a_scale_itr1_0,
                    epilogue_bitcast_b_itr1,
                    epilogue_bitcast_b_scale_itr1,
                    tkw.SchedulingBarrier([]),
                    epilogue_mma_itr1_0,
                    epilogue_s2v_a_1_itr1,
                    epilogue_s2v_a_scale_1_itr1,
                ],
            ),
            tkw.cluster(
                [
                    epilogue_bitcast_a_itr1_1,
                    epilogue_bitcast_a_scale_itr1_1,
                    epilogue_mma_itr1_1,
                ],
            ),
        ]

        clusters += epilogue_clusters_itr0
        clusters += prologue_clusters
        tkw.reorder_graph(pipeline_loop.EPILOGUE, clusters)

    return mxfp4_dbuf_schedule
