# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *


def get_async_two_cluster_triple_buffer():
    """
    Returns a schedule function that implements a 3-stage pipelined prefetch
    with async global_to_shared operations, cluster-based reordering, and wave staggering.
    This schedule uses GatherToLDS (global_to_shared) to combine global load + shared write
    This schedule expects a kernel with the following tags:
    - "k_loop": The reduction loop to pipeline
    - "read_a", "read_b": Read operations (will include both Read and GatherToLDS nodes)
    - "mma": MMA operations to partition
    - Requires use_global_to_shared=True in WaveCompileOptions
    Returns:
        A wave_schedule decorated function

    Refer to docs/wave_schedule_2.rst for further details on the implementation logic of this schedule.
    """
    # Symbol definitions (needed for partitioning)
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def async_two_cluster_three_stage_schedule():
        # Get nodes to be manipulated in the schedule
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get all nodes with tag "read_a" - includes both Read and GatherToLDS nodes
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Get all nodes with tag "read_b" - includes both Read and GatherToLDS nodes
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        mma = tkw.get_node_by_tag("mma")

        # First, create the 3-stage pipeline
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (global_to_shared_a, global_to_shared_b),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # Now apply advanced scheduling to the KERNEL stage
        # Filter nodes to only include those in the KERNEL stage
        global_to_shared_a = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        )
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        global_to_shared_b = tkw.filter_nodes(
            global_to_shared_b, subgraph=pipeline_loop.KERNEL
        )
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        # Partition node lists by K dimension for fine-grained scheduling
        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )

        independent_global_count = len(global_to_shared_a) + len(global_to_shared_b)

        clusters = [
            tkw.cluster(
                [
                    tkw.MemoryCounterWaitBarrier(load=independent_global_count),
                    tkw.WorkgroupBarrier(),
                    shared_load_a_0,
                    shared_load_b_0,
                    tkw.SchedulingBarrier([]),
                    global_to_shared_a,
                    global_to_shared_b,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    shared_load_a_1,
                    shared_load_b_1,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply staggering waves scheduling to allow two waves to execute clusters in parallel with a stagger offset
        tkw.stagger(pipeline_loop.KERNEL)

        # Unroll factor requires per-GEMM tuning:
        # Over-unrolling leads to bloated code and instruction cache pressure.
        # Under-unrolling prevents the backend from resolving memory aliasing
        # metadata
        unroll_factor = 2
        tkw.unroll(pipeline_loop.KERNEL, unroll_factor)

    return async_two_cluster_three_stage_schedule


def get_gfx1250_tbuf_gemm_schedule():
    """
    Returns a schedule function that implements a 3-stage pipelined prefetch
    with async global_to_shared operations, cluster-based reordering, and wave staggering.

    This schedule uses TensorLoadToLDS (global_to_shared) to combine global load + shared write.
    It implements triple buffering with wave staggering for optimal memory/compute overlap.

    Schedule structure:
    - Prologue: Prefetch 2 iterations worth of data with barrier synchronization
    - Main loop:
      * Stage 0: Load from shared memory (i iteration)
      * Stage 1: Async load i+2 iteration + Compute current data (i) (overlapped)
    - Epilogue: Continue staggering with chunked loads/MMAs

    This schedule expects a kernel with the following tags:
    - "k_loop": The reduction loop to pipeline
    - "read_a", "read_b": Read operations (will include both Read and TensorLoadToLDS nodes)
    - "mma": MMA operations to partition
    - Requires use_global_to_shared=True in WaveCompileOptions
    """

    @wave_schedule.wave_schedule()
    def gfx1250_tbuf_gemm_schedule():
        # Get nodes to be manipulated in the schedule
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get all nodes with tag "read_a" - includes both Read and TensorLoadToLDS nodes
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.TensorLoadToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Get all nodes with tag "read_b" - includes both Read and TensorLoadToLDS nodes
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.TensorLoadToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        global_to_shared_fused = tkw.get_node_by_tag("read_a,read_b")

        if len(global_to_shared_fused) == 0:
            global_to_shared_fused.extend(global_to_shared_a)
            global_to_shared_fused.extend(global_to_shared_b)

        mma = tkw.get_node_by_tag("mma")

        # First, create the 3-stage pipeline
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (global_to_shared_fused,),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # Filter nodes for PROLOGUE stage (before the loop)
        prologue_global_to_shared_fused = tkw.filter_nodes(
            global_to_shared_fused, subgraph=pipeline_loop.PROLOGUE
        )

        # Prologue cluster: tensor_load_to_lds + wait.tensorcnt(1) + barrier.signal + barrier.wait
        prologue_clusters = [
            tkw.cluster(
                [
                    prologue_global_to_shared_fused,
                    tkw.TensorCounterWait(1),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SharedMemoryBarrierWait(-1),
                ],
            )
        ]

        # Create wave stagger condition using hardware constraint
        hw = tkw.get_hardware_constraint()
        mid_wave = math.prod(hw.waves_per_block) // 2
        wave_hi = hw.wave_id >= mid_wave

        tkw.insert_cond_barrier_before(wave_hi, pipeline_loop.KERNEL)
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SetWavePrio(1))

        # Filter nodes for KERNEL stage
        global_to_shared_fused = tkw.filter_nodes(
            global_to_shared_fused, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b = tkw.filter_nodes(
            shared_load_b, subgraph=pipeline_loop.KERNEL
        )
        loop_mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        # Create cluster ordering with async operations
        # Pattern inside the loop:
        # 1. shared loads (A and B)
        # 2. SetWavePrio(0) + wait.tensorcnt(1) + wait.dscnt(0) + barrier.signal + sched.barrier + barrier.wait
        # 3. tensor_load_to_lds (global to shared)
        # 4. MMAs
        # 5. SetWavePrio(1) + barrier.signal + barrier.wait
        clusters = [
            tkw.cluster(
                [
                    # Shared memory loads
                    loop_shared_load_a,
                    loop_shared_load_b,
                    # Barrier pattern after shared loads
                    tkw.SetWavePrio(0),
                    tkw.TensorCounterWait(0),  # rocdl.s.wait.tensorcnt 0
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=True),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                    # Global to shared (tensor_load_to_lds)
                    global_to_shared_fused,
                    # MMAs
                    loop_mma,
                    # End of loop barrier pattern
                    tkw.SetWavePrio(1),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SharedMemoryBarrierWait(-1),
                ],
            ),
        ]

        epilogue_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.EPILOGUE
        )
        epilogue_shared_load_b = tkw.filter_nodes(
            shared_load_b, subgraph=pipeline_loop.EPILOGUE
        )
        epilogue_mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.EPILOGUE)

        # Divide them into 2 chunks to stagger the loads and MMAs
        epilogue_shared_load_a_chunks = [epilogue_shared_load_a[i::2] for i in range(2)]
        epilogue_shared_load_b_chunks = [epilogue_shared_load_b[i::2] for i in range(2)]
        epilogue_mma_chunks = [epilogue_mma[i::2] for i in range(2)]

        # Epilogue pattern:
        # 1. First set of loads (both A and B)
        # 2. SetWavePrio(0) + barrier.signal(-1) + sched.barrier + barrier.wait(-1)
        # 3. First set of MMAs
        # 4. SetWavePrio(1) + wait.tensorcnt(0) + barrier.signal(-1) + sched.barrier + barrier.wait(-1)
        # 5. Second set of loads (both A and B)
        # 6. SetWavePrio(0) + barrier.signal(-1) + sched.barrier + barrier.wait(-1)
        # 7. Second set of MMAs
        # 8. SetWavePrio(1)
        # 9. Conditional barrier (placed by insert_cond_barrier_after)
        # 10. barrier.signal(-1) + barrier.wait(-1)
        epilogue_clusters = [
            tkw.cluster(
                [
                    # First set of loads (B and A together)
                    tkw.TensorCounterWait(1),
                    epilogue_shared_load_b_chunks[0],
                    epilogue_shared_load_a_chunks[0],
                    # Stagger barrier before first MMAs (no ds_wait)
                    tkw.SetWavePrio(0),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                    # First set of MMAs
                    epilogue_mma_chunks[0],
                    # Stagger barrier before second loads
                    tkw.SetWavePrio(1),
                    tkw.TensorCounterWait(0),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                    # Second set of loads (B and A together)
                    epilogue_shared_load_b_chunks[1],
                    epilogue_shared_load_a_chunks[1],
                    # Stagger barrier before second MMAs (no ds_wait)
                    tkw.SetWavePrio(0),
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SchedulingBarrier([]),
                    tkw.SharedMemoryBarrierWait(-1),
                    # Second set of MMAs
                    epilogue_mma_chunks[1],
                    # Final signal/wait (after conditional barrier)
                    tkw.SharedMemoryBarrierSignal(-1, ds_wait=False),
                    tkw.SharedMemoryBarrierWait(-1),
                ],
            )
        ]

        # Prepend prologue clusters
        all_clusters = prologue_clusters + clusters + epilogue_clusters

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, all_clusters)

        # Get reference to last epilogue MMA for conditional barrier placement
        last_epilogue_mma = epilogue_mma_chunks[1][-1]
        # Apply stagger with custom placement for post-loop conditional barrier
        wave_lo = hw.wave_id < mid_wave
        tkw.insert_cond_barrier_after(wave_lo, last_epilogue_mma)
        tkw.insert_after(last_epilogue_mma, tkw.SetWavePrio(1))

    return gfx1250_tbuf_gemm_schedule
