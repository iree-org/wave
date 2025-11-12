"""
GEMM Scheduling Examples

Demonstrates manual scheduling with pipelining, reordering, and wave priority control
for optimizing matrix multiplication performance.
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


def test_default_gemm_prefetch(is_debug=False):
    """Basic matrix multiplication kernel with prefetch schedule."""

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE_A = tkl.sym.ADDRESS_SPACE_A
    ADDRESS_SPACE_B = tkl.sym.ADDRESS_SPACE_B
    ADDRESS_SPACE_C = tkl.sym.ADDRESS_SPACE_C

    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 4),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def wave_kernel(
        a: Memory[M, K, ADDRESS_SPACE_A, tkl.f16],  # Input matrix A
        b: Memory[N, K, ADDRESS_SPACE_B, tkl.f16],  # Input matrix B
        c: Memory[M, N, ADDRESS_SPACE_C, tkl.f32],  # Output matrix C
    ):
        # Initialize the accumulator register with zeros
        c_reg = Register[M, N, tkl.f32](0.0)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, tkl.f32]) -> Register[M, N, tkl.f32]:
            # Load elements from A and B
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Store the final result to C
        tkw.write(repeat, c)

    # Create test matrices
    m, n, k = 128, 256, 1024

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_B: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        M: m,
        N: n,
        K: k,
        BLOCK_M: 128,
        BLOCK_N: 256,
        BLOCK_K: 64,
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
    }

    # Compile the kernel
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.PREFETCH,
        print_ir_after="all" if is_debug else [],
    )
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, wave_kernel)

    if is_debug:
        with open("default_gemm_prefetch.mlir", "w") as f:
            f.write(compiled_gemm.asm)

    # Run the GEMM kernel
    compiled_gemm(a, b, c)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b.t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")


def test_gemm_schedule(is_debug=False):
    """
    Returns a GEMM kernel with prefetch schedule that demonstrates pipelining.
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

    # Define the kernel
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
    def prefetch_schedule():
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

        # Create a pipeline with 2 stages and specify the operations that are overlapping.
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

    # Compile the kernel with the schedule
    gemm_prefetch = wave_compile(options, gemm_prefetch, prefetch_schedule)

    # Create test data
    a = torch.randn(shape[0], shape[2], dtype=torch.float16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.float16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    # Run the kernel
    gemm_prefetch(a, b, c)

    if is_debug:
        with open("gemm_prefetch.mlir", "w") as f:
            f.write(gemm_prefetch.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("GEMM prefetch test passed!")


def test_gemm_prefetch_reorder(is_debug=False):
    """Reference GEMM with improved prefetch scheduling using cluster-based reordering."""
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

    # Define the kernel
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
    def prefetch_schedule():
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

        # Create a pipeline with 2 stages and specify the operations that are overlapping.
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

        # now get all the nodes again, but only from the pipelined kernel subgraph
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

        # Partition node lists by K dimension
        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )

        # Create cluster ordering (similar to two_pp_cluster)
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

        # insert prefetch barriers before the loop
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
        use_scheduling_barriers=True,
        print_ir_after="all" if is_debug else [],
    )

    # Set runtime configuration for execution
    options = set_default_run_config(options)

    # Compile the kernel with the schedule
    gemm_prefetch = wave_compile(options, gemm_prefetch, prefetch_schedule)

    # Create test data
    a = torch.randn(shape[0], shape[2], dtype=torch.float16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.float16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    # Run the kernel
    gemm_prefetch(a, b, c)

    if is_debug:
        with open("gemm_prefetch_ref.mlir", "w") as f:
            f.write(gemm_prefetch.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("GEMM reference prefetch test passed!")


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
