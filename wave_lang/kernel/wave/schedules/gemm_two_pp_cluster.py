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
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params


def get_tagged_gemm(
    shape: tuple[int, int, int] = (128, 256, 1024),
    block_shape: tuple[int, int, int] = (128, 256, 64),
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x16_F16,
    compile_to_mlir: bool = False,
    use_global_to_shared: bool = False,
):
    """
    Returns a tagged GEMM kernel with compile options.

    The kernel includes tags on key operations for scheduling:
    - "k_loop": The main reduction loop
    - "read_a": Reading from matrix A
    - "read_b": Reading from matrix B
    - "mma": Matrix multiply-accumulate operations

    Args:
        shape: (M, N, K) dimensions for the GEMM
        mfma_variant: The MMA type to use for matrix multiplication
        compile_to_mlir: Whether to compile to MLIR or continue to binary
        use_global_to_shared: Whether to use async GatherToLDS (global_to_shared) operations

    Returns:
        Tuple of (kernel_function, compile_options)
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

    # Define the kernel with tags for scheduling
    @tkw.wave(constraints)
    def gemm(
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

    hyperparams = {
        M: shape[0],
        N: shape[1],
        K: shape[2],
        BLOCK_M: block_shape[0],
        BLOCK_N: block_shape[1],
        BLOCK_K: block_shape[2],
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
    }
    hyperparams.update(get_default_scheduling_params())

    # Define compile options
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_scheduling_barriers=False,
        compile_to_mlir=compile_to_mlir,
        use_global_to_shared=use_global_to_shared,
    )

    return gemm, options


def get_two_pp_cluster_schedule():
    """
    Returns a schedule function that implements a 2-stage pipelined prefetch
    with cluster-based reordering and wave staggering.

    This schedule expects a kernel with the following tags:
    - "k_loop": The reduction loop to pipeline
    - "read_a", "read_b": Read operations to partition
    - "mma": MMA operations to partition

    Returns:
        A wave_schedule decorated function
    """
    # Symbol definitions (needed for partitioning)
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def two_pp_cluster_schedule():
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
        pipeline_loop = tkw.pipeline(k_loop)

        # First, create the basic 2-stage pipeline
        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (global_load_a, global_load_b),
                    (shared_write_a, shared_write_b),
                ],
            )
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # Now apply advanced scheduling to the KERNEL stage
        # Filter nodes to only include those in the KERNEL stage, ignoring the other stages (PROLOGUE and EPILOGUE)
        global_load_a = tkw.filter_nodes(global_load_a, subgraph=pipeline_loop.KERNEL)
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        shared_write_a = tkw.filter_nodes(shared_write_a, subgraph=pipeline_loop.KERNEL)
        global_load_b = tkw.filter_nodes(global_load_b, subgraph=pipeline_loop.KERNEL)
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        shared_write_b = tkw.filter_nodes(shared_write_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

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

        # Insert prefetch barriers before the for loop and at the end of the for loop
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply reordering to the KERNEL stage only
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply 2-way stagger scheduling (adds conditional barriers around loop)
        tkw.stagger(pipeline_loop.KERNEL)

    return two_pp_cluster_schedule


def get_async_two_pp_clusters():
    """
    Returns a schedule function that implements a 2-stage pipelined prefetch
    with async global_to_shared operations, cluster-based reordering, and wave staggering.

    This schedule uses GatherToLDS (global_to_shared) to combine global load + shared write

    This schedule expects a kernel with the following tags:
    - "k_loop": The reduction loop to pipeline
    - "read_a", "read_b": Read operations (will include both Read and GatherToLDS nodes)
    - "mma": MMA operations to partition
    - Requires use_global_to_shared=True in WaveCompileOptions

    Returns:
        A wave_schedule decorated function
    """
    # Symbol definitions (needed for partitioning)
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def async_two_pp_cluster_schedule():
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

        # Create a pipeline with 2 stages
        pipeline_loop = tkw.pipeline(k_loop)

        # First, create the basic 2-stage pipeline
        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (global_to_shared_a, global_to_shared_b),
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

        # Calculate the number of independent global_to_shared operations for MemoryCounterWait
        independent_global_count = tkw.get_node_count(
            global_to_shared_a
        ) + tkw.get_node_count(global_to_shared_b)

        # Create cluster ordering with async operations
        clusters = [
            tkw.cluster(
                [
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
                    tkw.MemoryCounterWait(load=independent_global_count),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    shared_load_a_1,
                    shared_load_b_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWait(load=0),
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

        # Insert barriers before the for loop and at the end of the for loop
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply staggering waves scheduling to allow two waves to execute clusters in parallel with a stagger offset
        tkw.stagger(pipeline_loop.KERNEL)

    return async_two_pp_cluster_schedule
