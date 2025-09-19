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
from wave_lang.kernel.wave.expansion.expansion import add_get_results, expand_graph
from wave_lang.kernel.wave.hoisting import hoist_loop_invariant_ops
from wave_lang.kernel.wave.promotion import promote_node
from wave_lang.kernel.wave.type_inference import infer_types
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
)
from wave_lang.kernel.wave.utils.graph_utils import (
    initialize_iter_args,
)
from wave_lang.kernel.wave.utils.print_utils import (
    print_trace,
)


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
def read_write_same_size(
    a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    tkw.write(a_reg, c, elements_per_thread=4)


@run_test
def test_read_write_equal_sizes():
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
        add_shared_memory_barriers(trace)
        print_trace(trace, False)
    # CHECK: %allocate
    # CHECK-SAME: ((M, N), (BLOCK_M, BLOCK_N + 4), f16, $SHARED_ADDRESS_SPACE, 4, None, None, 0)
    # CHECK-NEXT: %a
    # CHECK-NEXT: %c
    # CHECK-NEXT: %read_M:0_N:0
    # CHECK-SAME: (%a, 4, None, (), None, None, None, None)
    # CHECK-NEXT: %read_M:0_N:1
    # CHECK-SAME: (%a, 4, None, (), None, None, None, None)
    # CHECK-NEXT: %read_M:1_N:0
    # CHECK-SAME: (%a, 4, None, (), None, None, None, None)
    # CHECK-NEXT: %read_M:1_N:1
    # CHECK-SAME: (%a, 4, None, (), None, None, None, None)
    # CHECK-NEXT: %write_1_shared_M:0_N:0
    # CHECK-SAME: (%read_M:0_N:0, %allocate, 4, None, (), None, None, None)
    # CHECK-NEXT: %write_1_shared_M:0_N:1
    # CHECK-SAME: (%read_M:0_N:1, %allocate, 4, None, (), None, None, None)
    # CHECK-NEXT: %write_1_shared_M:1_N:0
    # CHECK-SAME: (%read_M:1_N:0, %allocate, 4, None, (), None, None, None)
    # CHECK-NEXT: %write_1_shared_M:1_N:1
    # CHECK-SAME: (%read_M:1_N:1, %allocate, 4, None, (), None, None, None)
    # CHECK-NEXT: %shared_memory_barrier
    # CHECK-NEXT: %read_1_shared_M:0_N:0
    # CHECK-SAME: (%allocate, 4, None, (), None, None, None, [%write_1_shared_M:0_N:0]
    # CHECK-NEXT: %read_1_shared_M:0_N:1
    # CHECK-SAME: (%allocate, 4, None, (), None, None, None, [%write_1_shared_M:0_N:1]
    # CHECK-NEXT: %read_1_shared_M:1_N:0
    # CHECK-SAME: (%allocate, 4, None, (), None, None, None, [%write_1_shared_M:1_N:0]
    # CHECK-NEXT: %read_1_shared_M:1_N:1
    # CHECK-SAME: (%allocate, 4, None, (), None, None, None, [%write_1_shared_M:1_N:1]
    # CHECK-NEXT: %write_M:0_N:0
    # CHECK-SAME: (%read_1_shared_M:0_N:0, %c, 4, None, (), None, None, None)
    # CHECK-NEXT: %write_M:0_N:1
    # CHECK-SAME: (%read_1_shared_M:0_N:1, %c, 4, None, (), None, None, None)
    # CHECK-NEXT: %write_M:1_N:0
    # CHECK-SAME: (%read_1_shared_M:1_N:0, %c, 4, None, (), None, None, None)
    # CHECK-NEXT: %write_M:1_N:1
    # CHECK-SAME: (%read_1_shared_M:1_N:1, %c, 4, None, (), None, None, None)
    # CHECK-NEXT: return None

    # CHECK: -----


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
    # Root graph:
    # CHECK: %a
    # CHECK-NEXT: %b
    # CHECK-NEXT: %c
    # CHECK-NEXT: %register_M:0_N:0_K:0
    # CHECK-NEXT: %register_M:0_N:1_K:0
    # CHECK-NEXT: %register_M:1_N:0_K:0
    # CHECK-NEXT: %register_M:1_N:1_K:0
    # CHECK-NEXT: %allocate_1
    # CHECK-SAME: ((N, K), (BLOCK_N, BLOCK_K + 4), f16, $SHARED_ADDRESS_SPACE, 4, None, None, 0)
    # CHECK-NEXT: %allocate
    # CHECK-SAME: ((M, K), (BLOCK_M, BLOCK_K + 4), f16, $SHARED_ADDRESS_SPACE, 4, None, None, 0)
    # CHECK-NEXT: iterate
    # CHECK-SAME: (K, [%register_M:0_N:0_K:0, %register_M:0_N:1_K:0, %register_M:1_N:0_K:0, %register_M:1_N:1_K:0]
    # CHECK-NEXT: %get_result_M:0_N:0_K:0
    # CHECK-SAME: (%iterate, 0)
    # CHECK-NEXT: %get_result_M:0_N:1_K:0
    # CHECK-SAME: (%iterate, 1)
    # CHECK-NEXT: %get_result_M:1_N:0_K:0
    # CHECK-SAME: (%iterate, 2)
    # CHECK-NEXT: %get_result_M:1_N:1_K:0
    # CHECK-SAME: (%iterate, 3)
    # CHECK-NEXT: %write_M:0_N:0_K:0
    # CHECK-SAME: (%get_result_M:0_N:0_K:0, %c, 4, None, (), None, None, None)
    # CHECK-NEXT: %write_M:0_N:1_K:0
    # CHECK-SAME: (%get_result_M:0_N:1_K:0, %c, 4, None, (), None, None, None)
    # CHECK-NEXT: %write_M:1_N:0
    # CHECK-SAME: (%get_result_M:1_N:0_K:0, %c, 4, None, (), None, None, None)
    # CHECK-NEXT: %write_M:1_N:1_K:0
    # CHECK-SAME: (%get_result_M:1_N:1_K:0, %c, 4, None, (), None, None, None)
    # CHECK-NEXT: return None

    # iterate subgraph:
    # CHECK: %b
    # CHECK: %a
    # CHECK-NEXT: %acc_M:0_N:0_K:0
    # CHECK-NEXT: %acc_M:0_N:1_K:0
    # CHECK-NEXT: %acc_M:1_N:0_K:0
    # CHECK-NEXT: %acc_M:1_N:1_K:0
    # CHECK-NEXT: %read_M:0_N:0_K:0
    # CHECK-NEXT: %read_M:0_N:0_K:1
    # CHECK-NEXT: %read_M:1_N:0_K:0
    # CHECK-NEXT: %read_M:1_N:0_K:1
    # We don't actually need a berrier here as the reads from the prev iteration are to the different buffer
    # CHECK-NEXT: %write_shared_M:0_N:0_K:0
    # CHECK-NEXT: %write_shared_M:0_N:0_K:1
    # CHECK-NEXT: %write_shared_M:1_N:0_K:0
    # CHECK-NEXT: %write_shared_M:1_N:0_K:1
    # CHECK-NEXT: %shared_memory_barrier
    # CHECK-NEXT: %read_2_shared_M:0_N:0_K:0
    # CHECK-NEXT: %read_2_shared_M:0_N:0_K:1
    # CHECK-NEXT: %read_2_shared_M:1_N:0_K:0
    # CHECK-NEXT: %read_2_shared_M:1_N:0_K:1
    # CHECK-NEXT: %read_1_M:0_N:0_K:0
    # CHECK-NEXT: %read_1_M:0_N:0_K:1
    # CHECK-NEXT: %read_1_M:0_N:1_K:0
    # CHECK-NEXT: %read_1_M:0_N:1_K:1
    # We don't actually need a berrier here as reads and writes to the different buffers
    # CHECK-NEXT: %write_1_shared_M:0_N:0_K:0
    # CHECK-NEXT: %write_1_shared_M:0_N:0_K:1
    # CHECK-NEXT: %write_1_shared_M:0_N:1_K:0
    # CHECK-NEXT: %write_1_shared_M:0_N:1_K:1
    # CHECK-NEXT: %shared_memory_barrier
    # CHECK-NEXT: %read_3_shared_M:0_N:0_K:0
    # CHECK-NEXT: %read_3_shared_M:0_N:0_K:1
    # CHECK-NEXT: %read_3_shared_M:0_N:1_K:0
    # CHECK-NEXT: %read_3_shared_M:0_N:1_K:1
    # CHECK-NEXT: %mma_M:0_N:0_K:0
    # CHECK-NEXT: %mma_M:0_N:0_K:1
    # CHECK-NEXT: %mma_M:0_N:1_K:0
    # CHECK-NEXT: %mma_M:0_N:1_K:1
    # CHECK-NEXT: %mma_M:1_N:0_K:0
    # CHECK-NEXT: %mma_M:1_N:0_K:1
    # CHECK-NEXT: %mma_M:1_N:1_K:0
    # CHECK-NEXT: %mma_M:1_N:1_K:1


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
