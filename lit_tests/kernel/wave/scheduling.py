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
from wave_lang.kernel.wave.expansion.expansion import add_get_results, expand_graph
from wave_lang.kernel.wave.hoisting import hoist_loop_invariant_ops
from wave_lang.kernel.wave.minimize_global_loads import minimize_global_loads
from wave_lang.kernel.wave.promotion import promote_placeholders
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType, schedule_graph
from wave_lang.kernel.wave.shared_memory_indexing import (
    apply_shared_memory_indexing_corrections,
)
from wave_lang.kernel.wave.type_inference import infer_types
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.utils.graph_utils import initialize_iter_args
from wave_lang.kernel.wave.utils.print_utils import print_subgraph

# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K

# Address space
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

# Induction variable for dimension K
ARGK = tkl.sym.ARGK


@tkw.wave_trace_only()
def gemm_pipelined(
    a: tkl.Memory[M, K, ADDRESS_SPACE_0, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE_0, tkl.f16],
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
def test_gemm_pipelined():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
    ]
    subs = {
        M: 128,
        N: 256,
        K: 128,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_0: SHARED_ADDRESS_SPACE,
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
    with IndexingContext() as idxc:
        idxc.subs = subs
        trace: CapturedTrace = gemm_pipelined()
        idxc.finalize()
        initialize_iter_args(trace)
        add_get_results(trace)
        infer_types(trace)
        promote_placeholders(trace, constraints)
        set_node_indices(trace, constraints)
        expand_graph(trace, constraints)
        set_post_expansion_indices(trace, constraints)
        hoist_loop_invariant_ops(trace, constraints)
        minimize_global_loads(trace, constraints)
        apply_shared_memory_indexing_corrections(trace, constraints)
        schedule_graph(
            trace, constraints, True, SchedulingType.MODULO, multi_buffer_count=2
        )

    print_subgraph(trace, "pipelined_iterate", False)
    # CHECK: %acc_m_0_n_0_k_0
    # CHECK-NEXT: %acc_m_0_n_1_k_0
    # CHECK-NEXT: %acc_m_1_n_0_k_0
    # CHECK-NEXT: %acc_m_1_n_1_k_0
    # CHECK-NEXT: %rotating_reg_0
    # CHECK-NEXT: %rotating_reg_1
    # CHECK-NEXT: %rotating_reg_2
    # CHECK-NEXT: %rotating_reg_3
    # CHECK-NEXT: %outer_rotating_reg_8
    # CHECK-NEXT: %outer_rotating_reg_9
    # CHECK-NEXT: %outer_rotating_reg_10
    # CHECK-NEXT: %outer_rotating_reg_11
    # CHECK-NEXT: %read_4_shared_M:0_N:0_K:0_mapped_1_1
    # CHECK-NEXT: %read_2_shared_M:0_N:0_K:0_mapped_1_1
    # CHECK-NEXT: %read_21
    # CHECK-NEXT: %read_22
    # CHECK-NEXT: %scheduling_group_barrier
    # CHECK-SAME: ({Operation.READ_SHARED: 2, Operation.READ_GLOBAL: 2}, 0)
    # CHECK-NEXT: %read_4_shared_M:0_N:1_K:0
    # CHECK-NEXT: %read_4_shared_M:0_N:1_K:1
    # CHECK-NEXT: %mma_M:0_N:0_K:0
    # CHECK-SAME: (%read_2_shared_M:0_N:0_K:0_mapped_1_1, %read_4_shared_M:0_N:0_K:0_mapped_1_1, %acc_m_0_n_0_k_0, None)
    # CHECK-NEXT: %mma_M:1_N:0_K:0
    # CHECK-SAME: (%rotating_reg_2, %read_4_shared_M:0_N:0_K:0_mapped_1_1, %acc_m_1_n_0_k_0, None)
    # CHECK-NEXT: %scheduling_group_barrier_1
    # CHECK-SAME: ({Operation.READ_SHARED: 2, Operation.MMA: 2}, 0)
    # CHECK-NEXT: %mma_M:0_N:0_K:1
    # CHECK-SAME: (%rotating_reg_1, %rotating_reg_0, %mma_M:0_N:0_K:0_mapped_1_1, None)
    # CHECK-NEXT: %mma_M:1_N:0_K:1
    # CHECK-SAME: (%rotating_reg_3, %rotating_reg_0, %mma_M:1_N:0_K:0_mapped_1_1, None)
    # CHECK-NEXT: %write_10
    # CHECK-NEXT: %write_11
    # CHECK-NEXT: %scheduling_group_barrier_2
    # CHECK-SAME: ({Operation.MMA: 2, Operation.WRITE_SHARED: 2}, 0)
    # CHECK-NEXT: %mma_M:0_N:1_K:0
    # CHECK-SAME: (%read_2_shared_M:0_N:0_K:0_mapped_1_1, %read_4_shared_M:0_N:1_K:0_mapped_1_1, %acc_m_0_n_1_k_0, None)
    # CHECK-NEXT: %mma_M:1_N:1_K:0
    # CHECK-SAME: (%rotating_reg_2, %read_4_shared_M:0_N:1_K:0_mapped_1_1, %acc_m_1_n_1_k_0, None), kwargs = {})
    # CHECK-NEXT: %read_2_shared_M:1_N:0_K:0
    # CHECK-NEXT: %read_2_shared_M:1_N:0_K:1
    # CHECK-NEXT: %scheduling_group_barrier_3
    # CHECK-SAME: ({Operation.MMA: 2, Operation.READ_SHARED: 2}, 0)
    # CHECK-NEXT: %mma_M:0_N:1_K:1
    # CHECK-SAME: (%rotating_reg_1, %read_4_shared_M:0_N:1_K:1_mapped_1_1, %mma_M:0_N:1_K:0_mapped_1_1, None)
    # CHECK-NEXT: %mma_M:1_N:1_K:1
    # CHECK-SAME: (%rotating_reg_3, %read_4_shared_M:0_N:1_K:1_mapped_1_1, %mma_M:1_N:1_K:0_mapped_1_1, None)
    # CHECK-NEXT: %read_4_shared_M:0_N:0_K:1
    # CHECK-NEXT: %read_2_shared_M:0_N:0_K:1
    # CHECK-NEXT: %scheduling_group_barrier_4
    # CHECK-SAME: ({Operation.MMA: 2, Operation.READ_SHARED: 2}, 0)
    # CHECK-NEXT: [mma_M:0_N:0_K:1_mapped_1_1, mma_M:0_N:1_K:1_mapped_1_1, mma_M:1_N:0_K:1_mapped_1_1, mma_M:1_N:1_K:1_mapped_1_1, read_4_shared_M:0_N:0_K:1_mapped_2_0, read_2_shared_M:0_N:0_K:1_mapped_2_0, read_2_shared_M:1_N:0_K:0_mapped_2_0, read_2_shared_M:1_N:0_K:1_mapped_2_0, outer_rotating_reg_9, outer_rotating_reg_8, outer_rotating_reg_11, outer_rotating_reg_10]

    print_subgraph(trace, "region_1", False)
    # CHECK: %a
    # CHECK-NEXT: %b
    # CHECK-NEXT: %c
    # CHECK-NEXT: %register_M:0_N:0_K:0
    # CHECK-NEXT: %register_M:0_N:1_K:0
    # CHECK-NEXT: %register_M:1_N:0_K:0
    # CHECK-NEXT: %register_M:1_N:1_K:0
    # CHECK-NEXT: %allocate_1_multi_buffer_1
    # CHECK-NEXT: %allocate_1_multi_buffer_0
    # CHECK-NEXT: %allocate_multi_buffer_1
    # CHECK-NEXT: %allocate_multi_buffer_0
    # CHECK-NEXT: %read_21
    # CHECK-NEXT: %read_22
    # CHECK-NEXT: %write_10
    # CHECK-NEXT: %write_11
    # CHECK-NEXT: %read_2_shared_M:1_N:0_K:0
    # CHECK-NEXT: %read_2_shared_M:1_N:0_K:1
    # CHECK-NEXT: %read_4_shared_M:0_N:0_K:1
    # CHECK-NEXT: %read_2_shared_M:0_N:0_K:1
    # CHECK-NEXT: %read_4_shared_M:0_N:0_K:0
    # CHECK-NEXT: %read_2_shared_M:0_N:0_K:0
    # CHECK-NEXT: %read_21
    # CHECK-NEXT: %read_22
    # CHECK-NEXT: %read_4_shared_M:0_N:1_K:0
    # CHECK-NEXT: %read_4_shared_M:0_N:1_K:1

    # CHECK-NEXT: %mma_M:0_N:0_K:0
    # CHECK-SAME: (%read_2_shared_M:0_N:0_K:0_mapped_0_1, %read_4_shared_M:0_N:0_K:0_mapped_0_1, %register_M:0_N:0_K:0, None)
    # CHECK-NEXT: %mma_M:1_N:0_K:0
    # CHECK-SAME: (%read_2_shared_M:1_N:0_K:0_mapped_0_0, %read_4_shared_M:0_N:0_K:0_mapped_0_1, %register_M:1_N:0_K:0, None)
    # CHECK-NEXT: %mma_M:0_N:0_K:1
    # CHECK-SAME: (%read_2_shared_M:0_N:0_K:1_mapped_0_0, %read_4_shared_M:0_N:0_K:1_mapped_0_0, %mma_M:0_N:0_K:0_mapped_0_1, None)
    # CHECK-NEXT: %mma_M:1_N:0_K:1
    # CHECK-SAME: (%read_2_shared_M:1_N:0_K:1_mapped_0_0, %read_4_shared_M:0_N:0_K:1_mapped_0_0, %mma_M:1_N:0_K:0_mapped_0_1, None)

    # CHECK-NEXT: %write_10
    # CHECK-NEXT: %write_11

    # CHECK-NEXT: %mma_M:0_N:1_K:0
    # CHECK-SAME: (%read_2_shared_M:0_N:0_K:0_mapped_0_1, %read_4_shared_M:0_N:1_K:0_mapped_0_1, %register_M:0_N:1_K:0, None)
    # CHECK-NEXT: %mma_M:1_N:1_K:0
    # CHECK-SAME: (%read_2_shared_M:1_N:0_K:0_mapped_0_0, %read_4_shared_M:0_N:1_K:0_mapped_0_1, %register_M:1_N:1_K:0, None)

    # CHECK-NEXT: %read_2_shared_M:1_N:0_K:0
    # CHECK-NEXT: %read_2_shared_M:1_N:0_K:1

    # CHECK-NEXT: %mma_M:0_N:1_K:1
    # CHECK-SAME: (%read_2_shared_M:0_N:0_K:1_mapped_0_0, %read_4_shared_M:0_N:1_K:1_mapped_0_1, %mma_M:0_N:1_K:0_mapped_0_1, None)
    # CHECK-NEXT: %mma_M:1_N:1_K:1
    # CHECK-SAME: (%read_2_shared_M:1_N:0_K:1_mapped_0_0, %read_4_shared_M:0_N:1_K:1_mapped_0_1, %mma_M:1_N:1_K:0_mapped_0_1, None)

    # CHECK-NEXT: %read_4_shared_M:0_N:0_K:1
    # CHECK-NEXT: %read_2_shared_M:0_N:0_K:1

    # CHECK-NEXT: %iterate_1
    # CHECK-SAME: (K, [%mma_M:0_N:0_K:1_mapped_0_1, %mma_M:0_N:1_K:1_mapped_0_1, %mma_M:1_N:0_K:1_mapped_0_1, %mma_M:1_N:1_K:1_mapped_0_1, %read_4_shared_M:0_N:0_K:1_mapped_1_0, %read_2_shared_M:0_N:0_K:1_mapped_1_0, %read_2_shared_M:1_N:0_K:0_mapped_1_0, %read_2_shared_M:1_N:0_K:1_mapped_1_0, %allocate_multi_buffer_0, %allocate_multi_buffer_1, %allocate_1_multi_buffer_0, %allocate_1_multi_buffer_1], pipelined_iterate, [%a, %b], 1, None, None)

    # CHECK-NEXT: %get_result_M:0_N:0_K:0
    # CHECK-SAME: (%iterate_1, 0)
    # CHECK-NEXT: %get_result_M:0_N:1_K:0
    # CHECK-SAME: (%iterate_1, 1)
    # CHECK-NEXT: %get_result_M:1_N:0_K:0
    # CHECK-SAME: (%iterate_1, 2)
    # CHECK-NEXT: %get_result_M:1_N:1_K:0
    # CHECK-SAME: (%iterate_1, 3)
    # CHECK-NEXT: %get_result_9
    # CHECK-SAME: (%iterate_1, 4)
    # CHECK-NEXT: %get_result_10
    # CHECK-SAME: (%iterate_1, 5)
    # CHECK-NEXT: %get_result_11
    # CHECK-SAME: (%iterate_1, 6)
    # CHECK-NEXT: %get_result_12
    # CHECK-SAME: (%iterate_1, 7)
    # CHECK-NEXT: %get_result_13
    # CHECK-SAME: (%iterate_1, 8)
    # CHECK-NEXT: %get_result_14
    # CHECK-SAME: (%iterate_1, 9)
    # CHECK-NEXT: %get_result_15
    # CHECK-SAME: (%iterate_1, 10)
    # CHECK-NEXT: %get_result_16
    # CHECK-SAME: (%iterate_1, 11)

    # CHECK-NEXT: %read_4_shared_M:0_N:0_K:0
    # CHECK-NEXT: %read_2_shared_M:0_N:0_K:0
    # CHECK-NEXT: %read_4_shared_M:0_N:1_K:0
    # CHECK-NEXT: %read_4_shared_M:0_N:1_K:1

    # CHECK-NEXT: %mma_M:0_N:0_K:0
    # CHECK-SAME: (%read_2_shared_M:0_N:0_K:0_mapped_2_1, %read_4_shared_M:0_N:0_K:0_mapped_2_1, %get_result_M:0_N:0_K:0, None)
    # CHECK-NEXT: %mma_M:1_N:0_K:0
    # CHECK-SAME: (%get_result_11, %read_4_shared_M:0_N:0_K:0_mapped_2_1, %get_result_M:1_N:0_K:0, None)
    # CHECK-NEXT: %mma_M:0_N:0_K:1
    # CHECK-SAME: (%get_result_10, %get_result_9, %mma_M:0_N:0_K:0_mapped_2_1, None)
    # CHECK-NEXT: %mma_M:1_N:0_K:1
    # CHECK-SAME: (%get_result_12, %get_result_9, %mma_M:1_N:0_K:0_mapped_2_1, None)
    # CHECK-NEXT: %mma_M:0_N:1_K:0
    # CHECK-SAME: (%read_2_shared_M:0_N:0_K:0_mapped_2_1, %read_4_shared_M:0_N:1_K:0_mapped_2_1, %get_result_M:0_N:1_K:0, None)
    # CHECK-NEXT: %mma_M:1_N:1_K:0
    # CHECK-SAME: (%get_result_11, %read_4_shared_M:0_N:1_K:0_mapped_2_1, %get_result_M:1_N:1_K:0, None)
    # CHECK-NEXT: %mma_M:0_N:1_K:1
    # CHECK-SAME: (%get_result_10, %read_4_shared_M:0_N:1_K:1_mapped_2_1, %mma_M:0_N:1_K:0_mapped_2_1, None)
    # CHECK-NEXT: %mma_M:1_N:1_K:1
    # CHECK-SAME: (%get_result_12, %read_4_shared_M:0_N:1_K:1_mapped_2_1, %mma_M:1_N:1_K:0_mapped_2_1, None)

    # CHECK-NEXT: %write_M:0_N:0_K:0
    # CHECK-SAME: (%mma_M:0_N:0_K:1
    # CHECK-NEXT: %write_M:0_N:1_K:0
    # CHECK-SAME: (%mma_M:0_N:1_K:1
    # CHECK-NEXT: %write_M:1_N:0_K:0
    # CHECK-SAME: (%mma_M:1_N:0_K:1
    # CHECK-NEXT: %write_M:1_N:1_K:0
    # CHECK-SAME: (%mma_M:1_N:1_K:1

    # CHECK-NEXT: return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
