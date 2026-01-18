# RUN: python %s | FileCheck %s

import logging

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel._support.indexing import IndexingContext
from wave_lang.kernel._support.tracing import CapturedTrace
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.ops.wave_ops import *
from wave_lang.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from wave_lang.kernel.wave.barriers import add_shared_memory_barriers
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.expansion.expansion import add_get_results, expand_graph
from wave_lang.kernel.wave.hoisting import hoist_loop_invariant_ops
from wave_lang.kernel.wave.promotion import promote_node
from wave_lang.kernel.wave.schedule_reordering import schedule_reordering
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType, schedule_graph
from wave_lang.kernel.wave.templates.attention_common import (
    AttentionShape,
)
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_bshd_attention_kernel,
)
from wave_lang.kernel.wave.type_inference import infer_types
from wave_lang.kernel.wave.utils.barriers_utils import (
    get_barriers_analysis,
)
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    run_test,
)
from wave_lang.kernel.wave.utils.graph_utils import (
    initialize_iter_args,
)
from wave_lang.kernel.wave.utils.print_utils import (
    print_trace,
)

logger = logging.getLogger(__name__)


def get_read_nodes(graph: fx.Graph) -> list[CustomOp]:
    custom_nodes: list[CustomOp] = [get_custom(node) for node in graph.nodes]
    return [node for node in custom_nodes if isinstance(node, Read)]


def tweak_index(graph: fx.Graph):
    promoted_read_nodes = [
        node for node in get_read_nodes(graph) if node.write_dependency is not None
    ]
    # Modify the write dependency index to trigger a barrier.
    for promoted_read_node in promoted_read_nodes:
        write_dependency = promoted_read_node.write_dependency[0]
        for key, value in write_dependency.index.items():
            write_dependency.index[key].start = value.start + 1


def count_barriers(graph: fx.Graph) -> int:
    """Count the number of SharedMemoryBarrier nodes in the graph."""
    count = 0
    for node in graph.nodes:
        custom = get_custom(node)
        if isinstance(custom, SharedMemoryBarrier):
            count += 1
    return count


def insert_barrier_before_first_shared_read(graph: fx.Graph):
    """
    Find the first read from shared memory and insert a barrier before it.
    This simulates a pre-existing barrier in the graph.
    """
    for node in graph.nodes:
        custom = get_custom(node)
        if isinstance(custom, Read) and custom.write_dependency is not None:
            # This is a read from shared memory (has write dependency)
            with graph.inserting_before(node):
                SharedMemoryBarrier(wait_async_ops=False).add_to_graph(
                    graph, loc=custom.location
                )
            return True
    return False


# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K

# Address space (for GPU, shared(1) or global(0))
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
ADDRESS_SPACE_1 = tkl.sym.ADDRESS_SPACE_1

# Induction variable for dimension K
ARGK = tkl.sym.ARGK


@tkw.wave_trace_only()
def gemm(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.iterate(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=4)


@run_test
def test_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]
    subs = {
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 32,
    }
    with IndexingContext() as idxc:
        idxc.subs = subs
        trace: CapturedTrace = gemm()
        graph: fx.Graph = trace.get_subgraph("region_0")
        idxc.finalize()
        initialize_iter_args(trace)
        add_get_results(trace)
        infer_types(trace)
        read_nodes = get_read_nodes(graph)
        for read_node in read_nodes:
            promote_node(read_node, None, SHARED_ADDRESS_SPACE, constraints)
        set_node_indices(trace, constraints)
        expand_graph(trace, constraints)
        set_post_expansion_indices(trace, constraints)
        tweak_index(graph)
        hoist_loop_invariant_ops(trace, constraints)
        add_shared_memory_barriers(trace)
        print_trace(trace, False)

    # CHECK-LABEL: test_gemm


@run_test
def test_split_barriers():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [
        tkw.HardwareConstraint(threads_per_wave=32, waves_per_block=(1, 1, 1))
    ]
    subs = {
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 128,
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        SHARED_MEMORY_UNITS: 2,
        GLOBAL_MEMORY_UNITS: 2,
        MMA_UNITS: 2,
        VALU_DELAY: 1,
        VALU_UNITS: 2,
        SHUFFLE_DELAY: 1,
        SHUFFLE_UNITS: 2,
    }

    enable_scheduling = SchedulingType.MODULO
    use_global_to_shared = False

    with IndexingContext() as idxc:
        idxc.subs = subs
        trace: CapturedTrace = gemm()
        graph: fx.Graph = trace.get_subgraph("region_0")
        idxc.finalize()
        initialize_iter_args(trace)
        add_get_results(trace)
        infer_types(trace)
        read_nodes = get_read_nodes(graph)
        for read_node in read_nodes:
            promote_node(read_node, None, SHARED_ADDRESS_SPACE, constraints)
        set_node_indices(trace, constraints)
        expand_graph(trace, constraints)
        set_post_expansion_indices(trace, constraints)
        tweak_index(graph)
        hoist_loop_invariant_ops(trace, constraints)
        schedule_graph(trace, constraints, True, enable_scheduling)
        schedule_reordering(trace, constraints, enable_scheduling, use_global_to_shared)
        add_shared_memory_barriers(trace, target="gfx1201")
        print_trace(trace, False)

    # CHECK-LABEL: test_split_barriers


@run_test
def test_find_intersecting_interval_strategy():
    shape = AttentionShape(
        num_query_heads=8,
        num_kv_heads=8,
        query_seq_len=128,
        head_size_kv=128,
        head_size=64,
        kv_seq_len=256,
    )

    mfma_variant = (tkw.MMAType.RDNA4_WAVE32_F32_16x16x16_F16,) * 2

    base_attention, hyperparams, _ = get_bshd_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims=False,
    )
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.MODULO,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
        target="gfx1201",
    )
    base_attention = wave_compile(options, base_attention)
    print(base_attention.asm)

    # CHECK-LABEL: test_find_intersecting_interval_strategy
    # CHECK-COUNT-3: rocdl.s.barrier.signal id = -1
    # CHECK-COUNT-3: rocdl.s.barrier.wait id = -1


@run_test
def test_minimum_placement_strategy():
    shape = AttentionShape(
        num_query_heads=8,
        num_kv_heads=8,
        query_seq_len=128,
        head_size_kv=128,
        head_size=64,
        kv_seq_len=256,
    )

    mfma_variant = (tkw.MMAType.RDNA4_WAVE32_F32_16x16x16_F16,) * 2

    base_attention, hyperparams, _ = get_bshd_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims=False,
    )
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.MODULO,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
        target="gfx942",
    )
    base_attention = wave_compile(options, base_attention)
    print(base_attention.asm)

    # CHECK-LABEL: test_minimum_placement_strategy
    # CHECK-COUNT-5: amdgpu.lds_barrier


@tkw.wave_trace_only()
def read_write_same_size(
    a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    tkw.write(a_reg, c, elements_per_thread=4)


@run_test
def test_existing_barrier_not_duplicated():
    """
    Test that when a barrier already exists between producer and consumer,
    the barrier placement pass does not insert a duplicate.
    """
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]

    subs = {
        BLOCK_M: 32,
        BLOCK_N: 32,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
    }
    with IndexingContext() as idxc:
        idxc.subs = subs
        trace: CapturedTrace = read_write_same_size()
        graph: fx.Graph = trace.get_root_graph()
        read_node = get_read_nodes(graph)[0]
        idxc.finalize()
        add_get_results(trace)
        infer_types(trace)
        promote_node(read_node, None, SHARED_ADDRESS_SPACE, constraints)
        set_node_indices(trace, constraints)
        expand_graph(trace, constraints)
        set_post_expansion_indices(trace, constraints)
        tweak_index(graph)

        # Manually insert a barrier before calling add_shared_memory_barriers
        # This barrier will be between the write_shared and read_shared operations
        inserted = insert_barrier_before_first_shared_read(graph)
        assert inserted, "Failed to insert pre-existing barrier"

        # Count barriers before
        barriers_before = count_barriers(graph)

        # Now run barrier placement - it should detect the existing barrier
        # and NOT insert duplicates
        add_shared_memory_barriers(trace)

        # Count barriers after
        barriers_after = count_barriers(graph)

        print_trace(trace, False)

        # Print summary for test verification
        print(f"Barriers before: {barriers_before}")
        print(f"Barriers after: {barriers_after}")

    # The graph should have barriers between writes and reads
    # But since we pre-inserted one, we should have only 1 barrier (not duplicated)
    # CHECK: %allocate
    # CHECK: %write_1_shared_M:0_N:0
    # CHECK: %write_1_shared_M:0_N:1
    # CHECK: %write_1_shared_M:1_N:0
    # CHECK: %write_1_shared_M:1_N:1
    # CHECK: %shared_memory_barrier
    # CHECK-NEXT: %read_1_shared_M:0_N:0
    # CHECK: %read_1_shared_M:0_N:1
    # CHECK: %read_1_shared_M:1_N:0
    # CHECK: %read_1_shared_M:1_N:1

    # Verify only one barrier exists (the pre-existing one, no duplicate added)
    # CHECK: Barriers before: 1
    # CHECK: Barriers after: 1


@run_test
def test_manual_barriers_prevent_sync_regions():
    """
    Test that manual barriers prevent creation of sync regions.
    """

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 16, N: 16},
        )
    ]

    subs = {
        BLOCK_M: 32,
        BLOCK_N: 32,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
    }

    with IndexingContext() as idxc:
        idxc.subs = subs
        trace: CapturedTrace = read_write_same_size()
        graph: fx.Graph = trace.get_root_graph()
        read_node = get_read_nodes(graph)[0]
        idxc.finalize()
        add_get_results(trace)
        infer_types(trace, constraints)
        promote_node(read_node, None, SHARED_ADDRESS_SPACE, constraints)
        set_node_indices(trace, constraints, subs)
        expand_graph(trace, constraints)
        set_post_expansion_indices(trace, constraints)

        # force a hazard
        logger.debug("Tweaking indices to create hazard")
        for node in graph.nodes:
            custom = get_custom(node)
            if isinstance(custom, Read) and custom.write_dependency is not None:
                write_dep = custom.write_dependency[0]
                for key, value in write_dep.index.items():
                    write_dep.index[key].start = value.start + 1

        target_arch = None

        # get sync regions without manual barriers
        logger.debug("Getting sync regions without manual barriers")
        sync_regions_no_barriers = get_barriers_analysis(trace, target_arch)

        num_expected_barriers = len(sync_regions_no_barriers)
        logger.info(
            f"Detected {num_expected_barriers} sync regions without manual barriers"
        )
        assert (
            num_expected_barriers > 0
        ), "Expected at least one sync region (hazard detected)"

        # Manually insert barriers at the exact locations where they are needed
        logger.debug("Inserting manual barriers")
        for node in graph.nodes:
            custom = get_custom(node)
            if isinstance(custom, Read) and custom.write_dependency is not None:
                with graph.inserting_before(node):
                    SharedMemoryBarrierSignal(barId=-1).add_to_graph(
                        graph, loc=custom.location
                    )
                    SharedMemoryBarrierWait(barId=-1).add_to_graph(
                        graph, loc=custom.location
                    )
                break

        # Get sync regions with manual barriers
        logger.debug("Getting sync regions with manual barriers")
        sync_regions = get_barriers_analysis(
            trace, target_arch, check_existing_barriers=True
        )

        # Assertion: should have fewer sync regions (0) because manual barriers are detected
        logger.info(
            f"Sync regions without barrier vs with barrier: {len(sync_regions_no_barriers)} vs {len(sync_regions)}"
        )

        print(f"Sync regions without manual barriers: {len(sync_regions_no_barriers)}")
        print("Manual barriers inserted: 1 signal + 1 wait")
        print(f"Sync regions with manual barriers: {len(sync_regions)}")

        if len(sync_regions) < len(sync_regions_no_barriers):
            print(f"VALIDATION SUCCESS: Manual barriers prevented sync region creation")
        else:
            print(
                f"VALIDATION FAIL - Manual barriers did NOT prevent sync region creation"
            )
            print(
                f"Expected fewer sync regions, but got: {len(sync_regions_no_barriers)} vs {len(sync_regions)}"
            )
            assert False, (
                f"Manual barriers were NOT respected! "
                f"Sync regions should be 0 with manual barriers, but got {len(sync_regions)}. "
            )

    # CHECK-LABEL: test_manual_barriers_prevent_sync_regions
    # CHECK: Sync regions without manual barriers: {{[1-9]}}
    # CHECK: Manual barriers inserted: 1 signal + 1 wait
    # CHECK: Sync regions with manual barriers: {{[0-9]}}
    # CHECK: VALIDATION SUCCESS: Manual barriers prevented sync region creation
    # CHECK-NOT: VALIDATION FAIL


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
