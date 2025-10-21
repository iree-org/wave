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
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.schedule_reordering import schedule_reordering
from wave_lang.kernel.wave.scheduling.schedule import schedule_graph

from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.templates.attention_common import (
    AttentionShape,
)
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_bshd_attention_kernel,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile


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
        schedule_reordering(trace, constraints, enable_scheduling)
        add_shared_memory_barriers(trace, target="gfx1201")
        print_trace(trace, False)

    # MODULO - Root graph:
    # CHECK: %a
    # CHECK-NEXT: %b
    # CHECK-NEXT: %c
    # CHECK-DAG:  %register_
    # CHECK-DAG: %allocate
    # CHECK-DAG:  %iterate
    # CHECK-DAG:  %get_result
    # CHECK-DAG:  %write
    # CHECK:      return None

    # region_0:
    # CHECK: %b
    # CHECK: %a
    # CHECK-DAG: %acc_
    # CHECK-DAG: %read_
    # CHECK-DAG: %write_shared_

    # CHECK-DAG:  %shared_memory_barrier_signal
    # CHECK-NEXT: %shared_memory_barrier_wait

    # CHECK-DAG: %read_2_shared_
    # CHECK-DAG: %read_
    # CHECK-DAG: %write_1_shared_

    # CHECK-DAG:  %shared_memory_barrier_signal
    # CHECK-NEXT: %shared_memory_barrier_wait

    # CHECK-NEXT: %read_3_shared_
    # CHECK-DAG:  %mma
    # CHECK: return [


@run_test
def test_bshd_attention_pipelined():
    """
    When reduction graphs are introduced, a pair of signal and wait barrier
    is required to be inserted at the prolog and the epilog of a subgraph.
    Other than that, the insertion pattern is simple:
    - Insert a wait before a consumer
    - Insert a signal after a producer
    """
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

    # CHECK-LABEL:  func.func @base_attention
    # CHECK-DAG:      memref.alloc()
    # CHECK-DAG:      %[[V0:.*]] = memref.view {{.*}} #gpu.address_space<workgroup>>
    # CHECK-DAG:      %[[V1:.*]] = memref.view {{.*}} #gpu.address_space<workgroup>>
    # CHECK-DAG:      vector.load

    # CHECK-DAG:      vector.store {{.*}} %[[V1]]{{.*}} #gpu.address_space<workgroup>>
    # CHECK-DAG:      rocdl.s.wait.dscnt 0
    # CHECK-DAG:      rocdl.s.barrier.signal -1

    # CHECK-DAG:      vector.extract

    # CHECK-DAG:      rocdl.s.barrier.wait -1
    # CHECK-DAG:      vector.load %[[V1]]{{.*}} #gpu.address_space<workgroup>>

    ### loads and stores are operating on differnt parts of shared buffers -> no barriers need to be inserted here.

    # CHECK-DAG:      vector.store {{.*}} %[[V0]]{{.*}} #gpu.address_space<workgroup>>
    # CHECK-DAG:      rocdl.s.wait.dscnt 0
    # CHECK-DAG:      rocdl.s.barrier.signal -1

    # CHECK-DAG:      amdgpu.wmma

    # CHECK-DAG:      rocdl.s.barrier.wait -1
    # CHECK-DAG:      vector.load %[[V0]]{{.*}} #gpu.address_space<workgroup>>
    # CHECK-DAG:      vector.load %[[V1]]{{.*}} #gpu.address_space<workgroup>>
    # CHECK-DAG:      amdgpu.wmma

    # CHECK-DAG:    rocdl.s.wait.dscnt 0
    # CHECK-DAG:    rocdl.s.barrier.signal -1
    # CHECK-NEXT:     scf.for
    # CHECK-DAG:        amdgpu.wmma
    # CHECK-DAG:        vector.load

    # CHECK-DAG:        rocdl.s.barrier.wait -1
    # CHECK-DAG:        vector.store {{.*}} %[[V1]]{{.*}} #gpu.address_space<workgroup>>

    ### signal write to buffer 1 completes.

    # CHECK-DAG:        rocdl.s.wait.dscnt 0
    # CHECK-DAG:        rocdl.s.barrier.signal -1

    # CHECK-DAG:        vector.load

    # CHECK-DAG:        rocdl.s.barrier.wait -1
    # CHECK-DAG:        vector.load %[[V1]]{{.*}} #gpu.address_space<workgroup>>
    # CHECK-DAG:        vector.store {{.*}} %[[V0]]{{.*}} #gpu.address_space<workgroup>>

    ### signal here represents 2 things: read from buffer 1 completes, write to buffer 0 completes.

    # CHECK-DAG:        rocdl.s.wait.dscnt 0
    # CHECK-DAG:        rocdl.s.barrier.signal -1

    # CHECK-DAG:        amdgpu.wmma

    ### wait here then waits for read from buffer 1 completes, write to buffer 0 completes.

    # CHECK-DAG:        rocdl.s.barrier.wait -1
    # CHECK-DAG:        vector.load %[[V0]]{{.*}} #gpu.address_space<workgroup>>
    # CHECK-DAG:        vector.load %[[V1]]{{.*}} #gpu.address_space<workgroup>>
    # CHECK-DAG:        amdgpu.wmma
    # CHECK-DAG:        vector.load %[[V0]]{{.*}} #gpu.address_space<workgroup>>

    ### signal read from buffer 0 completes.

    # CHECK-DAG:        rocdl.s.wait.dscnt 0
    # CHECK-DAG:        rocdl.s.barrier.signal -1

    # CHECK-DAG:        arith.maximumf
    # CHECK-DAG:        amdgpu.wmma

    # CHECK-DAG:      scf.yield
    # CHECK-NEXT:     }
    # CHECK-NEXT:   rocdl.s.barrier.wait -1


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
