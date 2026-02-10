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
