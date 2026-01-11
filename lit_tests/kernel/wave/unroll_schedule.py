# RUN: python %s | FileCheck %s

"""
Test for the tkw.unroll schedule operation.

This test verifies that the `tkw.unroll(loop, factor)` schedule op correctly
unrolls an iterate node, modifying the emitted MLIR scf.for loop:
- The step is multiplied by the unroll factor
- The upper bound (count) is divided by the unroll factor
- The loop body is duplicated within each iteration
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.general_utils import run_test

# Symbol definitions
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_unroll_schedule_op():
    """
    Test that tkw.unroll in a manual schedule correctly modifies the iterate loop.

    The kernel has K=256 and BLOCK_K=16, so the iterate count is 16.
    Unrolling by factor 2 should result in:
    - count = 8 (256/16/2 = 8)
    - step = 2
    - Each iteration processes 2x the original work (2 MMAs per iteration)
    """
    # Constraints for a simple GEMM
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    @tkw.wave(constraints)
    def gemm_unroll(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # Tag the iterate so we can reference it in the schedule
        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    # Define the schedule that unrolls the k_loop by factor 2
    @wave_schedule.wave_schedule()
    def unroll_schedule():
        k_loop = tkw.get_node_by_tag("k_loop")
        tkw.unroll(k_loop, 2)

    # Compile options
    # K=256, BLOCK_K=16 => original count=16
    # After unroll by 2: count=8, step=2
    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            K: 256,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        compile_to_mlir=True,
    )

    gemm_unroll = wave_compile(options, gemm_unroll, unroll_schedule)
    print(gemm_unroll.asm)

    # CHECK-LABEL: func.func @gemm_unroll
    #
    # Verify that the loop has been unrolled:
    # - Original: K=256, BLOCK_K=16 => count=16, step=1 => scf.for ... to %c16 step %c1
    # - After unroll by 2: count=16, step=2 => scf.for ... to %c16 step %c2
    #   (8 loop iterations, each doing 2 iterations worth of work)
    #
    # CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    # CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
    #
    # The unrolled loop should iterate from 0 to 16 with step 2
    # CHECK: scf.for %{{.*}} = %[[C0]] to %[[C16]] step %[[C2]]
    #
    # Inside the loop body, we should see 2 MMA operations (unrolled)
    # CHECK: amdgpu.mfma
    # CHECK: amdgpu.mfma
    #
    # CHECK: scf.yield


@run_test
def test_unroll_schedule_op_factor_4():
    """
    Test unrolling with factor 4.

    The kernel has K=512 and BLOCK_K=16, so the iterate count is 32.
    Unrolling by factor 4 should result in:
    - count = 32 (unchanged), step = 4
    - scf.for ... to 32 step 4 => 8 loop iterations
    - Each iteration processes 4x the original work (4 MMAs per iteration)
    """
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    @tkw.wave(constraints)
    def gemm_unroll_4(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    @wave_schedule.wave_schedule()
    def unroll_schedule_4():
        k_loop = tkw.get_node_by_tag("k_loop")
        tkw.unroll(k_loop, 4)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            K: 512,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        compile_to_mlir=True,
    )

    gemm_unroll_4 = wave_compile(options, gemm_unroll_4, unroll_schedule_4)
    print(gemm_unroll_4.asm)

    # CHECK-LABEL: func.func @gemm_unroll_4
    #
    # After unroll by 4: count=32, step=4 => scf.for ... to %c32 step %c4
    # CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
    # CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
    #
    # CHECK: scf.for %{{.*}} = %[[C0]] to %[[C32]] step %[[C4]]
    #
    # Inside the loop body, we should see 4 MMA operations (unrolled)
    # CHECK: amdgpu.mfma
    # CHECK: amdgpu.mfma
    # CHECK: amdgpu.mfma
    # CHECK: amdgpu.mfma
    #
    # CHECK: scf.yield


@run_test
def test_unroll_and_reorder():
    """
    Test unrolling followed by reordering of unrolled operations.

    This test demonstrates that after unrolling, the duplicated operations
    can be accessed and reordered in the schedule. We unroll by 2, which
    creates 2 sets of reads and MMAs, then reorder them so that:
    - All reads (a and b) come first
    - Then all MMAs

    This shows that unroll + reorder compose correctly in the schedule DSL.
    """
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    @tkw.wave(constraints)
    def gemm_unroll_reorder(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    @wave_schedule.wave_schedule()
    def unroll_and_reorder_schedule():
        # First, unroll the loop by 2
        k_loop = tkw.get_node_by_tag("k_loop")
        tkw.unroll(k_loop, 2)

        # After unrolling, we can access all the duplicated ops by tag
        # Now there are 2 of each due to unrolling (original + _unrolled0 suffix)
        all_reads_a = tkw.get_node_by_tag("read_a")
        all_reads_b = tkw.get_node_by_tag("read_b")
        all_mmas = tkw.get_node_by_tag("mma")

        # Create a single cluster with all ops in desired order:
        # All reads first (grouped), then all MMAs
        # This reorders operations so reads are batched before MMAs
        clusters = [
            tkw.cluster(
                [
                    # All reads from A (both iterations)
                    all_reads_a,
                    # All reads from B (both iterations)
                    all_reads_b,
                    # Scheduling barrier between reads and MMAs
                    tkw.SchedulingBarrier([]),
                    # All MMAs (both iterations)
                    all_mmas,
                ],
            ),
        ]

        # Apply reordering to the iterate subgraph
        tkw.reorder_graph(k_loop, clusters)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            K: 256,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        compile_to_mlir=True,
    )

    gemm_unroll_reorder = wave_compile(
        options, gemm_unroll_reorder, unroll_and_reorder_schedule
    )
    print(gemm_unroll_reorder.asm)

    # CHECK-LABEL: func.func @gemm_unroll_reorder
    #
    # Verify that the loop structure is correct (unrolled by 2)
    # K=256, BLOCK_K=16 => count=16, after unroll by 2: count=16, step=2
    # CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    # CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
    #
    # CHECK: scf.for %{{.*}} = %[[C0]] to %[[C16]] step %[[C2]]
    #
    # Verify the reordering: all reads come first, then barrier, then MMAs
    # The reads (4 total after unroll by 2: 2 for A, 2 for B) should precede MMAs
    #
    # All reads first (4 vector.load ops)
    # CHECK: vector.load
    # CHECK: vector.load
    # CHECK: vector.load
    # CHECK: vector.load
    #
    # Scheduling barrier between reads and MMAs
    # CHECK: rocdl.sched.barrier
    #
    # Then all MMAs (2 mfma ops after unroll by 2)
    # CHECK: amdgpu.mfma
    # CHECK: amdgpu.mfma
    #
    # CHECK: scf.yield


@run_test
def test_get_node_by_tag_and_iteration():
    """
    Test the get_node_by_tag_and_iteration API for accessing unrolled operations.

    After unrolling by factor 2, operations from each iteration can be accessed:
    - iteration=0: Original operations (no _unrolled suffix)
    - iteration=1: Unrolled operations (with _unrolled0 suffix)

    This test demonstrates that we can access and reorder operations from
    specific unrolled iterations independently.
    """
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    @tkw.wave(constraints)
    def gemm_iter_access(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    @wave_schedule.wave_schedule()
    def iter_access_schedule():
        # First, unroll the loop by 2
        k_loop = tkw.get_node_by_tag("k_loop")
        tkw.unroll(k_loop, 2)

        # Access operations from specific iterations using get_node_by_tag_and_iteration
        # iteration=0: Original operations
        reads_a_iter0 = tkw.get_node_by_tag_and_iteration("read_a", iteration=0)
        reads_b_iter0 = tkw.get_node_by_tag_and_iteration("read_b", iteration=0)
        mma_iter0 = tkw.get_node_by_tag_and_iteration("mma", iteration=0)

        # iteration=1: Unrolled operations (with _unrolled0 suffix)
        reads_a_iter1 = tkw.get_node_by_tag_and_iteration("read_a", iteration=1)
        reads_b_iter1 = tkw.get_node_by_tag_and_iteration("read_b", iteration=1)
        mma_iter1 = tkw.get_node_by_tag_and_iteration("mma", iteration=1)

        # Reorder: iteration 0 reads, then iteration 1 reads, barrier, then all MMAs
        clusters = [
            tkw.cluster(
                [
                    reads_a_iter0,
                    reads_b_iter0,
                    reads_a_iter1,
                    reads_b_iter1,
                    tkw.SchedulingBarrier([]),
                    mma_iter0,
                    mma_iter1,
                ],
            ),
        ]

        tkw.reorder_graph(k_loop, clusters)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            K: 256,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        compile_to_mlir=True,
    )

    gemm_iter_access = wave_compile(options, gemm_iter_access, iter_access_schedule)
    print(gemm_iter_access.asm)

    # CHECK-LABEL: func.func @gemm_iter_access
    #
    # Verify that the loop structure is correct (unrolled by 2)
    # K=256, BLOCK_K=16 => count=16, step=2
    # CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    # CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
    #
    # CHECK: scf.for %{{.*}} = %[[C0]] to %[[C16]] step %[[C2]]
    #
    # Verify the reordering: reads from both iterations first, then barrier, then MMAs
    # CHECK: vector.load
    # CHECK: vector.load
    # CHECK: vector.load
    # CHECK: vector.load
    #
    # Scheduling barrier between reads and MMAs
    # CHECK: rocdl.sched.barrier
    #
    # Then all MMAs
    # CHECK: amdgpu.mfma
    # CHECK: amdgpu.mfma
    #
    # CHECK: scf.yield


@run_test
def test_unroll_then_pipeline():
    """
    Test unrolling an iterate node BEFORE pipelining it.

    This test demonstrates: unroll first, then pipeline.
    1. First unroll the iterate by factor 2
    2. Then pipeline the unrolled iterate with 2 stages

    This shows that unroll and pipeline compose correctly when unroll comes first.
    """
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    @tkw.wave(constraints)
    def gemm_unroll_then_pipeline(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    @wave_schedule.wave_schedule()
    def unroll_then_pipeline_schedule():
        # Get the iterate node
        k_loop = tkw.get_node_by_tag("k_loop")

        # Step 1: Unroll first (before pipelining)
        tkw.unroll(k_loop, 2)

        # Get nodes for pipelining (after unroll, tags are propagated to copies)
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

        # Step 2: Pipeline the unrolled iterate
        pipeline_loop = tkw.pipeline(k_loop)
        with pipeline_loop as pl:
            # Stage 0: Global loads and shared writes
            pl.set_stage(
                [
                    (global_load_a, global_load_b),
                    (shared_write_a, shared_write_b),
                ],
            )
            # Stage 1: Shared loads and MMA
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

    options = WaveCompileOptions(
        subs={
            M: 128,
            N: 256,
            K: 1024,
            BLOCK_M: 128,
            BLOCK_N: 256,
            BLOCK_K: 64,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        compile_to_mlir=True,
    )

    gemm_unroll_then_pipeline = wave_compile(
        options, gemm_unroll_then_pipeline, unroll_then_pipeline_schedule
    )
    print(gemm_unroll_then_pipeline.asm)

    # CHECK-LABEL: func.func @gemm_unroll_then_pipeline
    #
    # The loop should have step=2 due to unrolling
    # CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    #
    # Check for the main scf.for loop with step 2
    # CHECK: scf.for {{.*}} step %[[C2]]
    #
    # Verify multiple MFMAs in the loop body (from unroll + MMA expansion)
    # CHECK: amdgpu.mfma
    # CHECK: amdgpu.mfma
    #
    # CHECK: scf.yield


@run_test
def test_pipeline_then_unroll():
    """
    Test pipelining an iterate node BEFORE unrolling it.

    This test demonstrates: pipeline first, then unroll.
    1. First pipeline the iterate with 2 stages
    2. Then unroll the pipelined iterate by factor 2

    This shows that pipeline and unroll can compose in this order as well.
    """
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    @tkw.wave(constraints)
    def gemm_pipeline_then_unroll(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    @wave_schedule.wave_schedule()
    def pipeline_then_unroll_schedule():
        # Get the iterate node
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get nodes for pipelining
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

        # Step 1: Pipeline first
        pipeline_loop = tkw.pipeline(k_loop)
        with pipeline_loop as pl:
            # Stage 0: Global loads and shared writes
            pl.set_stage(
                [
                    (global_load_a, global_load_b),
                    (shared_write_a, shared_write_b),
                ],
            )
            # Stage 1: Shared loads and MMA
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # Step 2: Unroll after pipelining - unroll the KERNEL stage
        tkw.unroll(pipeline_loop.KERNEL, 2)

    # K=1088, BLOCK_K=64 => count=17, after 2-stage pipeline => count=16 (divisible by 2)
    options = WaveCompileOptions(
        subs={
            M: 128,
            N: 256,
            K: 1088,
            BLOCK_M: 128,
            BLOCK_N: 256,
            BLOCK_K: 64,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        compile_to_mlir=True,
    )

    gemm_pipeline_then_unroll = wave_compile(
        options, gemm_pipeline_then_unroll, pipeline_then_unroll_schedule
    )
    print(gemm_pipeline_then_unroll.asm)

    # CHECK-LABEL: func.func @gemm_pipeline_then_unroll
    #
    # The loop should have step=2 due to unrolling
    # CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    #
    # Check for the main scf.for loop with step 2
    # CHECK: scf.for {{.*}} step %[[C2]]
    #
    # Verify multiple MFMAs in the loop body (from unroll + MMA expansion)
    # CHECK: amdgpu.mfma
    # CHECK: amdgpu.mfma
    #
    # CHECK: scf.yield
