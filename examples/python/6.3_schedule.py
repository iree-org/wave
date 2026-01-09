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


def test_gfx1250_optim_gemm(is_debug=False):
    """
    GEMM scheduling matching GFX1250 optimization kernel pattern.

    - Prologue: Prefetch 2 iterations worth of data
    - Main loop:
      * Stage 0: Load from shared memory (ds_load)
      * Stage 1: Async load next iteration + Compute current data (overlapped)
    - Epilogue: Drain pipeline with remaining compute

    Key configuration:
    - 256x256x64 blocks
    - 1024x1024x1024 problem size
    - Double buffering (NUM_BUFFERS=2)
    - F32_16x16x32_F16 MMA (32 elements in K)
    """
    shape: tuple[int, int, int] = (1024, 1024, 1024)
    mfma_variant: tkw.MMAType = tkw.MMAType.GFX1250_F32_16x16x32_F16

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K

    m_symbol = tkl.sym.m_symbol
    n_symbol = tkl.sym.n_symbol
    k_symbol = tkl.sym.k_symbol

    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Constraints needed for compilation
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]  # 256/4 = 64
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]  # 256/2 = 128
    constraints += [tkw.IteratorBindings({m_symbol: M, n_symbol: N, k_symbol: K})]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=32,
            mma_type=mfma_variant,
        )
    ]

    # Define the kernel
    @tkw.wave(constraints)
    def gemm_gfx1250_optim(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[N, M, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[N, M, tkl.f32],
        ) -> tkl.Register[N, M, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(b_reg, a_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c, source=(n_symbol, m_symbol), target=(m_symbol, n_symbol))

    # Define the GFX1250 optimization schedule
    @wave_schedule.wave_schedule()
    def gfx1250_optim_gemm_schedule():
        """
        Schedule matching GFX1250 optimization kernel pattern.
        """
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

        mma = tkw.get_node_by_tag("mma")

        pipeline_loop = tkw.pipeline(k_loop)

        # Create 2-stage pipeline
        with pipeline_loop as pl:
            # Stage 0: Load from shared memory (ds_load)
            pl.set_stage(
                [
                    (global_to_shared_a, global_to_shared_b),
                    (),
                ],
            )

            # Stage 1: Async load next iteration + Compute current
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),  # ds_load operations
                    (mma,),  # Compute with CURRENT iteration's data (from stage 0)
                ],
            )

        # Now apply reordering to the KERNEL stage
        global_to_shared_a = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        )
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        global_to_shared_b = tkw.filter_nodes(
            global_to_shared_b, subgraph=pipeline_loop.KERNEL
        )
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        # Create clusters:
        # Phase 1: All loads + barrier → Phase 2: Async loads → Phase 3: All compute + final barrier
        clusters = [
            # Cluster 1: ALL shared memory loads + barrier
            tkw.cluster(
                [
                    shared_load_a,  # All ds_loads for A together
                    shared_load_b,  # All ds_loads for B together
                    tkw.SharedMemoryBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    global_to_shared_a,  # Start async loads for next iteration
                    global_to_shared_b,
                    mma,
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        tkw.insert_after(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering to create the desired instruction pattern
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)
        # tkw.stagger(pipeline_loop.KERNEL)

    # Define compile options
    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 256,
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
        dump_binaries="./",
        dump_intermediates="./",
    )

    # Set runtime configuration for execution
    options = set_default_run_config(options)

    # Compile the kernel with the gfx1250 optimization schedule
    gemm_gfx1250_optim = wave_compile(
        options, gemm_gfx1250_optim, gfx1250_optim_gemm_schedule
    )
    with open("gemm_gfx1250_optim.asm", "w") as f:
        f.write(gemm_gfx1250_optim.asm)

    # Create test data
    datatype = torch.float16
    a = torch.randn(shape[0], shape[2], dtype=datatype).cuda()
    b = torch.randn(shape[1], shape[2], dtype=datatype).cuda()
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32).cuda()

    # Run the kernel
    gemm_gfx1250_optim(a, b, c)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("GFX1250 optimization GEMM schedule test passed!")


def test_gfx1250_tbuf_gemm(is_debug=False):
    """
    GEMM scheduling matching GFX1250 optimization kernel pattern.

    - Prologue: Prefetch 2 iterations worth of data
    - Main loop:
      * Stage 0: Load from shared memory (ds_load)
      * Stage 1: Async load next iteration + Compute current data (overlapped)
    - Epilogue: Drain pipeline with remaining compute

    Key configuration:
    - 256x256x64 blocks
    - 1024x1024x1024 problem size
    - Double buffering (NUM_BUFFERS=2)
    - F32_16x16x32_F16 MMA (32 elements in K)
    """
    shape: tuple[int, int, int] = (1024, 1024, 1024)
    mfma_variant: tkw.MMAType = tkw.MMAType.GFX1250_F32_16x16x32_F16

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K

    m_symbol = tkl.sym.m_symbol
    n_symbol = tkl.sym.n_symbol
    k_symbol = tkl.sym.k_symbol

    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Constraints needed for compilation
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]  # 256/4 = 64
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]  # 256/2 = 128
    constraints += [tkw.IteratorBindings({m_symbol: M, n_symbol: N, k_symbol: K})]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=32,
            mma_type=mfma_variant,
        )
    ]

    # Define the kernel
    @tkw.wave(constraints)
    def gemm_gfx1250_optim(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[N, M, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[N, M, tkl.f32],
        ) -> tkl.Register[N, M, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(b_reg, a_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c, source=(n_symbol, m_symbol), target=(m_symbol, n_symbol))

    @wave_schedule.wave_schedule()
    def gfx1250_optim_tbuf_gemm_schedule():
        # Get nodes to be manipulated in the schedule
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get all nodes with tag "read_a" - includes both Read and GatherToLDS nodes
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.TensorLoadToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Get all nodes with tag "read_b" - includes both Read and GatherToLDS nodes
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

        # Now apply advanced scheduling to the KERNEL stage
        # Filter nodes to only include those in the KERNEL stage
        global_to_shared_fused = tkw.filter_nodes(
            global_to_shared_fused, subgraph=pipeline_loop.KERNEL
        )
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        # Create cluster ordering with async operations
        clusters = [
            tkw.cluster(
                [
                    shared_load_a,
                    shared_load_b,
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    global_to_shared_fused,
                    tkw.SchedulingBarrier([]),
                    mma,
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]
        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)
        tkw.stagger(pipeline_loop.KERNEL)

    # Define compile options
    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 256,
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

    # Compile the kernel with the gfx1250 optimization schedule
    gemm_gfx1250_optim = wave_compile(
        options, gemm_gfx1250_optim, gfx1250_optim_tbuf_gemm_schedule
    )

    if is_debug:
        with open("gemm_gfx1250_optim_tbuf.asm", "w") as f:
            f.write(gemm_gfx1250_optim.asm)

    # Create test data
    datatype = torch.float16
    a = torch.randn(shape[0], shape[2], dtype=datatype).cuda()
    b = torch.randn(shape[1], shape[2], dtype=datatype).cuda()
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32).cuda()

    # Run the kernel
    gemm_gfx1250_optim(a, b, c)

    expected = torch.matmul(a.cpu(), b.cpu().T).to(torch.float32)
    assert torch.allclose(c.cpu(), expected, rtol=1e-2, atol=1e-2)

    print("GFX1250 optimization Tbuf GEMM schedule test passed!")


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
