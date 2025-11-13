# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType


def get_gemm_two_pp_cluster_schedule(
    shape: tuple[int, int, int] = (128, 256, 1024),
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x16_F16,
    compile_to_mlir: bool = False,
):
    """
    Returns a GEMM kernel with advanced prefetch scheduling including:
    - Pipelining with 2 stages
    - Cluster-based reordering for optimal latency hiding
    - 2-way wave staggering

    This is a reference implementation demonstrating the full manual scheduling.
    """
    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Basic constraints needed for compilation
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
        )
    ]

    # Define the kernel
    @tkw.wave(constraints)
    def gemm_prefetch_reorder(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    # Define the schedule with advanced reordering and staggering
    @wave_schedule.wave_schedule()
    def prefetch_reorder_schedule():
        # Get nodes to be manipulated in the schedule
        k_loop = tkw.get_node_by_tag("k_loop")
        load_a = tkw.get_node_by_tag_and_type("read_a", tkw.Read)
        global_load_a, shared_load_a = tkw.partition_by_address_space(
            load_a, GLOBAL_ADDRESS_SPACE
        )
        shared_write_a = tkw.get_node_by_tag_and_type("read_a", tkw.Write)
        load_b = tkw.get_node_by_tag_and_type("read_b", tkw.Read)
        global_load_b, shared_load_b = tkw.partition_by_address_space(
            load_b, GLOBAL_ADDRESS_SPACE
        )
        shared_write_b = tkw.get_node_by_tag_and_type("read_b", tkw.Write)
        mma = tkw.get_node_by_tag("mma")

        # Create a pipeline with 2 stages and specify the operations that are overlapping
        with tkw.pipeline(k_loop) as pipelined_loop:
            pipelined_loop.set_stage(
                [
                    (global_load_a, global_load_b),
                    (shared_write_a, shared_write_b),
                ],
            )
            pipelined_loop.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # Now get all the nodes again, but only from the pipelined kernel subgraph
        k_loop = tkw.get_node_by_tag("k_loop")

        # Use subgraph parameter to get nodes only from the KERNEL stage
        load_a = tkw.get_node_by_tag_and_type("read_a", tkw.Read, subgraph=k_loop)
        global_load_a, shared_load_a = tkw.partition_by_address_space(
            load_a, GLOBAL_ADDRESS_SPACE
        )
        shared_write_a = tkw.get_node_by_tag_and_type(
            "read_a", tkw.Write, subgraph=k_loop
        )
        load_b = tkw.get_node_by_tag_and_type("read_b", tkw.Read, subgraph=k_loop)
        global_load_b, shared_load_b = tkw.partition_by_address_space(
            load_b, GLOBAL_ADDRESS_SPACE
        )
        shared_write_b = tkw.get_node_by_tag_and_type(
            "read_b", tkw.Write, subgraph=k_loop
        )
        mma = tkw.get_node_by_tag("mma", subgraph=k_loop)

        # Partition node lists by K dimension for fine-grained scheduling
        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )

        # Create cluster ordering
        clusters = [
            tkw.cluster(
                [
                    shared_load_a_0,
                    shared_load_b_0,
                    tkw.SchedulingBarrier([]),
                    global_load_a,
                    tkw.SchedulingBarrier([]),
                    shared_load_a_1,
                    shared_load_b_1,
                    tkw.SchedulingBarrier([]),
                    global_load_b,
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SharedMemoryBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    shared_write_a,
                    shared_write_b,
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

        # Insert prefetch barriers before the loop
        tkw.insert_before(k_loop, tkw.SharedMemoryBarrier())
        output_node = tkw.get_output_node(k_loop)
        tkw.insert_before(output_node, tkw.SharedMemoryBarrier())

        # Apply reordering to the KERNEL stage only
        tkw.reorder_graph(k_loop, clusters)

        # Apply 2-way stagger scheduling (adds conditional barriers around loop)
        tkw.stagger(k_loop)

    # Define compile options
    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 128,
            BLOCK_N: 256,
            BLOCK_K: 64,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            VALU_DELAY: 1,
            SHUFFLE_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
            VALU_UNITS: 8,
            SHUFFLE_UNITS: 8,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_scheduling_barriers=False,
        compile_to_mlir=compile_to_mlir,
    )

    return gemm_prefetch_reorder, prefetch_reorder_schedule, options
