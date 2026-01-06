"""
GEMM Scheduling Part 2: Advanced Scheduling with Reordering and Staggering Waves

This example demonstrates advanced scheduling techniques for optimizing GPU performance:
1. Partitioning operations by dimensions to interleave compute with memory ops
2. Creating instruction clusters for optimal ordering
3. Wave priority manipulation (SetWavePrio) to prioritize compute waves over memory waves
4. Staggering waves for better overlap of computation and memory access
5. Scheduling barriers for fine-grained control
"""

import torch

import wave_lang.kernel.wave as tkw
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType

from utils import parse_args, list_tests, run_test


def test_gemm_advanced_scheduling(is_debug=False):
    """
    Advanced GEMM scheduling with cluster reordering and ping-pong buffering.

    This example demonstrates sophisticated scheduling techniques:

    1. Partitioning: Split MMA operations by K dimension to interleave compute with memory ops
    2. Clustering: Group instructions for optimal execution order
    3. Wave Priority: Use SetWavePrio to prioritize compute waves over memory waves
    4. Ping-Pong (Stagger): Double-buffer shared memory to avoid stalls
    5. Barriers: Insert explicit synchronization points

    The resulting schedule maximizes instruction-level parallelism and hides memory latency
    by carefully interleaving memory operations with compute operations from different
    iterations.
    """
    shape: tuple[int, int, int] = (128, 256, 1024)
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x16_F16

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

    # Define the kernel (same as before, but we'll apply advanced scheduling)
    @tkw.wave(constraints)
    def gemm_prefetch(
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

    # Define the advanced schedule
    @wave_schedule.wave_schedule()
    def advanced_schedule():
        """
        Advanced scheduling with cluster-based reordering and ping-pong buffering.

        The schedule creates a sophisticated instruction ordering that:
        1. Interleaves compute (MMA) with memory operations
        2. Uses wave priorities to ensure compute waves get resources when needed
        3. Implements ping-pong buffering via stagger() for double buffering
        4. Carefully places barriers to ensure correctness while maximizing parallelism
        """
        # Get nodes to be manipulated in the schedule.
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
        global_load_a = tkw.filter_nodes(global_load_a, subgraph=pipeline_loop.KERNEL)
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        shared_write_a = tkw.filter_nodes(shared_write_a, subgraph=pipeline_loop.KERNEL)
        global_load_b = tkw.filter_nodes(global_load_b, subgraph=pipeline_loop.KERNEL)
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        shared_write_b = tkw.filter_nodes(shared_write_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        # Partition MMA operations by K dimension into 2 groups
        # This allows us to interleave the first half of MMA with prefetch for next iteration
        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)

        # Similarly partition the shared memory loads
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )

        # Create instruction clusters that define the execution order
        # Each cluster groups instructions that should execute together
        clusters = [
            # Cluster 1: First half of loads + prefetch for next iteration
            tkw.cluster(
                [
                    shared_load_a_0,  # Load first half of A from shared memory
                    shared_load_b_0,  # Load first half of B from shared memory
                    tkw.SchedulingBarrier([]),  # Barrier for scheduling control
                    global_load_a,  # Prefetch A for next iteration (overlapped!)
                    tkw.SchedulingBarrier([]),
                    shared_load_a_1,  # Load second half of A
                    shared_load_b_1,  # Load second half of B
                    tkw.SchedulingBarrier([]),
                    global_load_b,  # Prefetch B for next iteration (overlapped!)
                    tkw.WorkgroupBarrier(),  # Ensure all waves complete loads
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: First half of MMA operations with high priority
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),  # Increase priority for compute
                    mma_0,  # Execute first half of MMA operations
                    tkw.SetWavePrio(0),  # Reset priority
                    tkw.SharedMemoryBarrier(),  # Sync shared memory
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Write prefetched data to shared memory
            tkw.cluster(
                [
                    shared_write_a,  # Write prefetched A to shared memory
                    shared_write_b,  # Write prefetched B to shared memory
                    tkw.WorkgroupBarrier(),  # Ensure writes complete
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 4: Second half of MMA operations
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),  # Increase priority for compute
                    mma_1,  # Execute second half of MMA operations
                    tkw.SetWavePrio(0),  # Reset priority
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
        print_ir_after="all" if is_debug else [],
    )

    # Set runtime configuration for execution
    options = set_default_run_config(options)

    # Compile the kernel with the advanced schedule
    gemm_prefetch = wave_compile(options, gemm_prefetch, advanced_schedule)

    # Create test data
    a = torch.randn(shape[0], shape[2], dtype=torch.float16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.float16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    # Run the kernel
    gemm_prefetch(a, b, c)

    if is_debug:
        print(gemm_prefetch.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("GEMM advanced scheduling test passed!")


def test_async_gemm_schedule(is_debug=False):
    """
    GEMM scheduling with async global_to_shared operations and ping-pong buffering.

    This example uses the following scheduling techniques with GatherToLDS:

    1. Async Global-to-Shared: Uses GatherToLDS to combine global load + shared write into a single async operation
    2. Partitioning: Splits MMA operations by K dimension to interleave compute with memory ops
    3. Clustering: Groups instructions to define execution order
    4. Wave Priority: Uses SetWavePrio to adjust compute wave priorities
    5. Stagger: Stagger the waves to allow two waves to execute clusters in parallel with a stagger offset
    6. Barriers: Inserts explicit synchronization and memory counter waits

    The schedule uses async operations to overlap global-to-shared transfers with compute operations
    from different iterations.
    """
    shape: tuple[int, int, int] = (128, 256, 1024)
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x16_F16

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

    # Define the kernel (same as before, but we'll apply advanced scheduling)
    @tkw.wave(constraints)
    def gemm_prefetch(
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

    # Define the schedule
    @wave_schedule.wave_schedule()
    def async_gemm_schedule():
        """
        Scheduling with cluster-based reordering and ping-pong buffering.

        The schedule creates an instruction ordering that:
        1. Uses async global_to_shared operations (GatherToLDS) to overlap global load + shared write
        2. Interleaves compute (MMA) with memory operations
        3. Uses wave priorities to adjust compute wave priorities
        4. Implements ping-pong buffering via stagger() for double buffering
        5. Places barriers for synchronization
        """
        # Get nodes to be manipulated in the schedule.
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
        global_to_shared_a = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        )
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        global_to_shared_b = tkw.filter_nodes(
            global_to_shared_b, subgraph=pipeline_loop.KERNEL
        )
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

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
        print_ir_after="all" if is_debug else [],
        use_global_to_shared=True,
    )

    # Set runtime configuration for execution
    options = set_default_run_config(options)

    # Compile the kernel with the advanced schedule
    gemm_prefetch = wave_compile(options, gemm_prefetch, async_gemm_schedule)

    # Create test data
    a = torch.randn(shape[0], shape[2], dtype=torch.float16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.float16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    # Run the kernel
    gemm_prefetch(a, b, c)

    if is_debug:
        print(gemm_prefetch.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("Async GEMM schedule with global_to_shared test passed!")


def test_async_gemm_schedule_triple_buffering(is_debug=False):
    """
    GEMM scheduling with async global_to_shared operations and triple buffering..

    This example uses the following scheduling techniques with GatherToLDS:

    1. Async Global-to-Shared: Uses GatherToLDS to combine global load + shared write into a single async operation
    2. Partitioning: Splits MMA operations by K dimension to interleave compute with memory ops
    3. Clustering: Groups instructions to define execution order
    4. Wave Priority: Uses SetWavePrio to adjust compute wave priorities
    5. Stagger: Stagger the waves to allow two waves to execute clusters in parallel with a stagger offset
    6. Barriers: Inserts explicit synchronization and memory counter waits

    The schedule uses async operations to overlap global-to-shared transfers with compute operations
    from different iterations. It also uses a 3 stage pipeline to triple buffer the shared memory which allows for 2 memory prefetches before the loop starts.
    """
    shape: tuple[int, int, int] = (128, 256, 1024)
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x16_F16

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

    # Define the kernel (same as before, but we'll apply advanced scheduling)
    @tkw.wave(constraints)
    def gemm_prefetch(
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

    # Define the schedule
    @wave_schedule.wave_schedule()
    def async_gemm_schedule_triple_buffering():
        """
        Scheduling with cluster-based reordering and ping-pong buffering.

        The schedule creates an instruction ordering that:
        1. Uses async global_to_shared operations (GatherToLDS) to overlap global load + shared write
        2. Interleaves compute (MMA) with memory operations
        3. Uses wave priorities to adjust compute wave priorities
        4. Implements ping-pong buffering via stagger() for double buffering
        5. Places barriers for synchronization
        """
        # Get nodes to be manipulated in the schedule.
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

        pipeline_loop = tkw.pipeline(k_loop)
        # First, create the basic 3-stage pipeline
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
        global_to_shared_a = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        )
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        global_to_shared_b = tkw.filter_nodes(
            global_to_shared_b, subgraph=pipeline_loop.KERNEL
        )
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )

        clusters = [
            tkw.cluster(
                [
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
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        print_ir_after="all" if is_debug else [],
        use_global_to_shared=True,
        minimize_shared_allocs=False,
    )

    # Set runtime configuration for execution
    options = set_default_run_config(options)

    # Compile the kernel with the advanced schedule
    gemm_prefetch = wave_compile(
        options, gemm_prefetch, async_gemm_schedule_triple_buffering
    )

    # Create test data
    a = torch.randn(shape[0], shape[2], dtype=torch.float16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.float16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    # Run the kernel
    gemm_prefetch(a, b, c)

    if is_debug:
        print(gemm_prefetch.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print(
        "Async GEMM schedule using triple buffering with global_to_shared test passed!"
    )


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
