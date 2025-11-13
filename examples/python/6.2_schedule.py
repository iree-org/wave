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

        # First, create the basic 2-stage pipeline
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

        # Now apply advanced scheduling to the KERNEL stage
        # Get all nodes again, but only from the pipelined kernel subgraph
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

        # Insert barriers before and after the loop for ping-pong buffering
        tkw.insert_before(k_loop, tkw.SharedMemoryBarrier())
        output_node = tkw.get_output_node(k_loop)
        tkw.insert_before(output_node, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(k_loop, clusters)

        # Apply staggering waves scheduling to allow two waves to execute clusters in parallel with a stagger offset
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
