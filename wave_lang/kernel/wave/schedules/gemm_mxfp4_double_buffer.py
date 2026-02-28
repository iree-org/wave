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


def get_mxfp4_dbuf_pingpong_schedule(use_stagger: bool = True, shape: tuple = None):
    """Return a double-buffered MXFP4 schedule for wave_compile().

    Args:
        use_stagger: Enable wave staggering + WorkgroupBarrier in cluster 0.
            Recommended for 8-wave configs; disable for 4-wave.
        shape: Tuple of (M, N, K) dimensions. If provided and bigger than
            (1024, 1024, 1024), an extra WorkgroupBarrier will be added
            after the first SchedulingBarrier in cluster 0.
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
        print("loop_shared_load_a_0")
        print(loop_shared_load_a_0)
        print("loop_scaled_mma_0")
        print(loop_scaled_mma_0)

        # Check if shape is bigger than threshold (1024, 1024, 1024)
        use_extra_barrier = False
        if shape is not None and len(shape) >= 3:
            threshold = (1024, 1024, 1024)
            use_extra_barrier = (
                shape[0] > threshold[0]
                and shape[1] > threshold[1]
                and shape[2] > threshold[2]
            )

        # Build cluster 0: first K-partition loads + bitcasts + GatherToLDS
        cluster_0_ops = [
            tkw.SchedulingBarrier([]),
        ]
        if use_extra_barrier:
            cluster_0_ops.append(tkw.WorkgroupBarrier())
        cluster_0_ops.extend(
            [
                tkw.MemoryCounterWait(load=0),
                tkw.WorkgroupBarrier(),
                loop_global_to_shared,
                tkw.SchedulingBarrier([]),
                loop_shared_load_a_0,
                loop_shared_load_a_scale_0,
                loop_shared_load_b_0,
                loop_shared_load_b_scale_0,
                loop_bitcast_a_0,
                loop_bitcast_a_scale_0,
                loop_bitcast_b_0,
                loop_bitcast_b_scale_0,
                tkw.SchedulingBarrier([]),
            ]
        )
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
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: Second K-partition shared loads/bitcasts
            tkw.cluster(
                [
                    tkw.SchedulingBarrier([]),
                    loop_shared_load_a_1,
                    loop_shared_load_a_scale_1,
                    loop_shared_load_b_1,
                    loop_shared_load_b_scale_1,
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    loop_bitcast_b_1,
                    loop_bitcast_b_scale_1,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
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

        # Insert barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.WorkgroupBarrier())
        tkw.insert_after(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        # tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply wave staggering for better overlap
        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return mxfp4_dbuf_schedule


def get_mxfp4_dbuf_mixed_pingpong_schedule(use_stagger: bool = True):
    """Return a double-buffered MXFP4 schedule for wave_compile().

    Hides the latency of the second WorkgroupBarrier by issuing a "safe"
    subset of LDS vector.loads (rows owned by this wave) before the barrier,
    then interleaving the dependent loads with compute after it.

    Safe/dependent split (per K-partition):
      - A / A_scale : M:0,1 safe  |  M:2,3 dependent
      - B / B_scale : N:0,1,4,5 safe  |  N:2,3,6,7 dependent
      - MFMAs       : M:0,1 x N:0,1,4,5 safe  |  rest dependent

    Args:
        use_stagger: Enable wave staggering via tkw.stagger().
            Recommended for 8-wave configs; disable for 4-wave.
    """
    K = tkl.sym.K
    M = tkl.sym.M
    N = tkl.sym.N

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

        # Partition by K dimension first
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

        # =====================================================================
        # Split A / A_scale by M dimension: safe = M:0,1 | dependent = M:2,3
        # These correspond to rows owned by the "early" wave group (wave_idx<4).
        # After memory_counter_wait(N_gather), this wave's LDS writes are done,
        # so M:0,1 rows are readable before the global workgroup barrier.
        # =====================================================================
        # K=0 partition
        loop_sla_0_safe, loop_sla_0_dep = tkw.partition_by_dim(
            loop_shared_load_a_0, dim=M, num_partitions=2
        )
        loop_slascale_0_safe, loop_slascale_0_dep = tkw.partition_by_dim(
            loop_shared_load_a_scale_0, dim=M, num_partitions=2
        )
        loop_bca_0_safe, loop_bca_0_dep = tkw.partition_by_dim(
            loop_bitcast_a_0, dim=M, num_partitions=2
        )
        loop_bcascale_0_safe, loop_bcascale_0_dep = tkw.partition_by_dim(
            loop_bitcast_a_scale_0, dim=M, num_partitions=2
        )
        # K=1 partition
        loop_sla_1_safe, loop_sla_1_dep = tkw.partition_by_dim(
            loop_shared_load_a_1, dim=M, num_partitions=2
        )
        loop_slascale_1_safe, loop_slascale_1_dep = tkw.partition_by_dim(
            loop_shared_load_a_scale_1, dim=M, num_partitions=2
        )
        loop_bca_1_safe, loop_bca_1_dep = tkw.partition_by_dim(
            loop_bitcast_a_1, dim=M, num_partitions=2
        )
        loop_bcascale_1_safe, loop_bcascale_1_dep = tkw.partition_by_dim(
            loop_bitcast_a_scale_1, dim=M, num_partitions=2
        )

        # =====================================================================
        # Split B / B_scale by N dimension (4 partitions):
        #   safe       = N:0,1 (p0) + N:4,5 (p2)
        #   dependent  = N:2,3 (p1) + N:6,7 (p3)
        # =====================================================================
        # K=0 partition
        slb_0_p0, slb_0_p1, slb_0_p2, slb_0_p3 = tkw.partition_by_dim(
            loop_shared_load_b_0, dim=N, num_partitions=4
        )
        slbscale_0_p0, slbscale_0_p1, slbscale_0_p2, slbscale_0_p3 = (
            tkw.partition_by_dim(loop_shared_load_b_scale_0, dim=N, num_partitions=4)
        )
        bcb_0_p0, bcb_0_p1, bcb_0_p2, bcb_0_p3 = tkw.partition_by_dim(
            loop_bitcast_b_0, dim=N, num_partitions=4
        )
        bcbscale_0_p0, bcbscale_0_p1, bcbscale_0_p2, bcbscale_0_p3 = (
            tkw.partition_by_dim(loop_bitcast_b_scale_0, dim=N, num_partitions=4)
        )
        loop_slb_0_safe = slb_0_p0 + slb_0_p2  # N:0,1,4,5
        loop_slb_0_dep = slb_0_p1 + slb_0_p3  # N:2,3,6,7
        loop_slbscale_0_safe = slbscale_0_p0 + slbscale_0_p2
        loop_slbscale_0_dep = slbscale_0_p1 + slbscale_0_p3
        loop_bcb_0_safe = bcb_0_p0 + bcb_0_p2
        loop_bcb_0_dep = bcb_0_p1 + bcb_0_p3
        loop_bcbscale_0_safe = bcbscale_0_p0 + bcbscale_0_p2
        loop_bcbscale_0_dep = bcbscale_0_p1 + bcbscale_0_p3

        # K=1 partition
        slb_1_p0, slb_1_p1, slb_1_p2, slb_1_p3 = tkw.partition_by_dim(
            loop_shared_load_b_1, dim=N, num_partitions=4
        )
        slbscale_1_p0, slbscale_1_p1, slbscale_1_p2, slbscale_1_p3 = (
            tkw.partition_by_dim(loop_shared_load_b_scale_1, dim=N, num_partitions=4)
        )
        bcb_1_p0, bcb_1_p1, bcb_1_p2, bcb_1_p3 = tkw.partition_by_dim(
            loop_bitcast_b_1, dim=N, num_partitions=4
        )
        bcbscale_1_p0, bcbscale_1_p1, bcbscale_1_p2, bcbscale_1_p3 = (
            tkw.partition_by_dim(loop_bitcast_b_scale_1, dim=N, num_partitions=4)
        )
        loop_slb_1_safe = slb_1_p0 + slb_1_p2
        loop_slb_1_dep = slb_1_p1 + slb_1_p3
        loop_slbscale_1_safe = slbscale_1_p0 + slbscale_1_p2
        loop_slbscale_1_dep = slbscale_1_p1 + slbscale_1_p3
        loop_bcb_1_safe = bcb_1_p0 + bcb_1_p2
        loop_bcb_1_dep = bcb_1_p1 + bcb_1_p3
        loop_bcbscale_1_safe = bcbscale_1_p0 + bcbscale_1_p2
        loop_bcbscale_1_dep = bcbscale_1_p1 + bcbscale_1_p3

        # =====================================================================
        # Split MFMAs:
        #   safe       = M:0,1 x N:0,1,4,5   (8 MFMAs per K-partition)
        #   dep_B      = M:0,1 x N:2,3,6,7   (8 MFMAs) -- safe A, dep B
        #   dep_A      = M:2,3 x N:0..7       (16 MFMAs) -- dep A, all B
        # =====================================================================
        # K=0
        mma_0_M01, mma_0_M23 = tkw.partition_by_dim(
            loop_scaled_mma_0, dim=M, num_partitions=2
        )
        mma_0_M01_N01, mma_0_M01_N23, mma_0_M01_N45, mma_0_M01_N67 = (
            tkw.partition_by_dim(mma_0_M01, dim=N, num_partitions=4)
        )
        loop_mma_0_safe = mma_0_M01_N01 + mma_0_M01_N45  # M:0,1 x N:0,1,4,5
        loop_mma_0_dep_B = mma_0_M01_N23 + mma_0_M01_N67  # M:0,1 x N:2,3,6,7
        loop_mma_0_dep_A = mma_0_M23  # M:2,3 x all N

        # K=1
        mma_1_M01, mma_1_M23 = tkw.partition_by_dim(
            loop_scaled_mma_1, dim=M, num_partitions=2
        )
        mma_1_M01_N01, mma_1_M01_N23, mma_1_M01_N45, mma_1_M01_N67 = (
            tkw.partition_by_dim(mma_1_M01, dim=N, num_partitions=4)
        )
        loop_mma_1_safe = mma_1_M01_N01 + mma_1_M01_N45
        loop_mma_1_dep_B = mma_1_M01_N23 + mma_1_M01_N67
        loop_mma_1_dep_A = mma_1_M23

        # Number of async gather_to_lds ops issued per loop iteration.
        # Used as the memory_counter_wait threshold placed after gather_to_lds.
        independent_global_count = len(loop_global_to_shared)

        # Build clusters
        # Cluster 0
        cluster_0_ops = [
            tkw.SchedulingBarrier([]),
            tkw.WorkgroupBarrier(),
            loop_global_to_shared,
            tkw.SchedulingBarrier([]),
            tkw.MemoryCounterWait(load=independent_global_count),
            loop_sla_0_safe,
            loop_slascale_0_safe,
            loop_slb_0_safe,
            loop_slbscale_0_safe,
            loop_bca_0_safe,
            loop_bcascale_0_safe,
            loop_bcb_0_safe,
            loop_bcbscale_0_safe,
            tkw.SchedulingBarrier([]),
            tkw.WorkgroupBarrier(),
            tkw.SchedulingBarrier([]),
        ]

        clusters = [
            tkw.cluster(cluster_0_ops),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_mma_0_safe,
                    tkw.SetWavePrio(0),
                    loop_sla_0_dep,
                    loop_slascale_0_dep,
                    loop_slb_0_dep,
                    loop_slbscale_0_dep,
                    loop_bca_0_dep,
                    loop_bcascale_0_dep,
                    loop_bcb_0_dep,
                    loop_bcbscale_0_dep,
                    tkw.SetWavePrio(1),
                    loop_mma_0_dep_B,
                    loop_mma_0_dep_A,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: K=1 safe loads/bitcasts
            tkw.cluster(
                [
                    tkw.SchedulingBarrier([]),
                    loop_sla_1_safe,
                    loop_slascale_1_safe,
                    loop_slb_1_safe,
                    loop_slbscale_1_safe,
                    loop_bca_1_safe,
                    loop_bcascale_1_safe,
                    loop_bcb_1_safe,
                    loop_bcbscale_1_safe,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: K=1 safe MFMAs, dep loads/bitcasts, dep MFMAs
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_mma_1_safe,
                    tkw.SetWavePrio(0),
                    loop_sla_1_dep,
                    loop_slascale_1_dep,
                    loop_slb_1_dep,
                    loop_slbscale_1_dep,
                    loop_bca_1_dep,
                    loop_bcascale_1_dep,
                    loop_bcb_1_dep,
                    loop_bcbscale_1_dep,
                    tkw.SetWavePrio(1),
                    loop_mma_1_dep_B,
                    loop_mma_1_dep_A,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.WorkgroupBarrier())
        tkw.insert_after(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply wave staggering for better overlap
        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return mxfp4_dbuf_schedule


def get_mxfp4_asymmetric_schedule(is_bscale_shuffled: bool = False):
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

        # This forces the pipeline to use double buffering
        pipeline_loop.multi_buffer_count = 2

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

        if is_bscale_shuffled:
            b_scale_shuffling_factor = 4
        else:
            b_scale_shuffling_factor = 1

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

        # A is prefetched twice in the prologue, we want to wait for just the first prefetch
        A_g2s_total = len(prologue_g2s_a) + len(prologue_g2s_a_scale)
        A_g2s_per_iter = A_g2s_total // num_pf_iters
        B_g2v_prologue = len(prologue_g2v_b) + (
            len(prologue_g2v_b_scale) // b_scale_shuffling_factor
        )

        prologue_clusters = [
            tkw.cluster(
                [
                    prologue_g2s_a,
                    prologue_g2s_a_scale,
                    prologue_g2v_b,
                    tkw.SchedulingBarrier([]),
                    prologue_g2v_b_scale,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=0),
                    tkw.SchedulingBarrier([]),
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

        loop_B_g2v_bs = len(loop_g2v_b) + (
            len(loop_g2v_b_scale) // b_scale_shuffling_factor
        )
        loop_A_s2v_bs = len(loop_g2s_a) + len(loop_g2s_a_scale)
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
                    tkw.MemoryCounterWaitBarrier(load=loop_B_g2v_bs, ds=0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    tkw.SchedulingBarrier([]),
                    interleaved_mma_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=loop_A_s2v_bs, ds=0),
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
                    tkw.MemoryCounterWaitBarrier(load=0, ds=0),
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
                    tkw.MemoryCounterWaitBarrier(load=0, ds=0),
                    epilogue_s2v_a_0,
                    epilogue_s2v_a_scale_0,
                ],
            ),
            tkw.cluster(
                [
                    tkw.MemoryCounterWaitBarrier(load=0, ds=0),
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

        tkw.reorder_graph(pipeline_loop.PROLOGUE, prologue_clusters)
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)
        # tkw.reorder_graph(pipeline_loop.EPILOGUE, epilogue_clusters_itr0)
        unroll_factor = 2
        tkw.unroll(pipeline_loop.KERNEL, unroll_factor)

        tkw.insert_at_start(
            pipeline_loop.KERNEL,
            tkw.MemoryCounterWaitBarrier(load=A_g2s_per_iter, ds=0),
        )
        tkw.insert_after(
            pipeline_loop.KERNEL, tkw.MemoryCounterWaitBarrier(load=0, ds=0)
        )

    return mxfp4_dbuf_schedule
