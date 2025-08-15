# RUN: python %s | FileCheck %s

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
)
from wave_lang.kernel._support.indexing import IndexingContext
from wave_lang.kernel.wave.hoisting import hoist_loop_invariant_ops
from wave_lang.kernel.wave.utils.print_utils import print_trace, print_subgraph
from wave_lang.kernel.wave.expansion.expansion import (
    expand_graph,
    add_get_results,
)
from wave_lang.kernel.wave.type_inference import infer_types
from wave_lang.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from wave_lang.kernel.wave.utils.graph_utils import (
    remove_chained_getresult,
    initialize_iter_args,
)


@run_test
def test_simple_hoisting():
    """Test that the hoisting pass correctly identifies loop-invariant operations
    in a simple kernel's iterate node.

    Specifically, we want to verify that:
    1. The read operation (a_reg = tkw.read(a))
    2. The multiplication (a_reg *= scale_reg)

    are present in the iterate subgraph and could potentially be hoisted
    since they don't depend on the loop induction variable.
    """
    # Define symbolic variables
    N = tkl.sym.N
    M = tkl.sym.M
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ARGK = tkl.sym.ARGK

    # Simple constraints
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K, ARGK),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 16, N: 16, K: 16},
        ),
    ]

    @tkw.wave_trace_only(constraints)
    def simple_kernel(
        a: tkl.Memory[M, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        d: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        acc_reg = tkl.Register[M, N, tkl.f32](2.0)
        scale_reg = tkl.Register[M, N, tkl.f32](2.0)

        @tkw.iterate(K, init_args=[acc_reg])
        def loop(acc: tkl.Register[M, N, tkl.f32]):
            # This read and multiplication should be hoisted out of the loop
            # since they don't depend on the loop induction variable
            c_reg = tkw.read(c)
            c_reg *= scale_reg

            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc += c_reg
            acc += tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(loop, d)

    # Get the trace
    with IndexingContext() as idxc:
        idxc.subs = {
            M: 16,
            N: 16,
            K: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            BLOCK_K: 16,
        }
        trace = simple_kernel()
        idxc.finalize()
        initialize_iter_args(trace)
        add_get_results(trace)
        infer_types(trace)
        set_node_indices(trace, constraints)
        expand_graph(trace, constraints)
        set_post_expansion_indices(trace, constraints)
        remove_chained_getresult(trace)
        hoist_loop_invariant_ops(trace, constraints)

    print("=== Root Graph ===")
    print_trace(trace)

    print("\n=== Iterate Subgraph ===")
    print_subgraph(trace, "region_0", False)

    # CHECK: === Root Graph ===
    # CHECK: %a
    # CHECK: %b
    # CHECK: %c
    # CHECK: %d
    # CHECK: %register_M:0_N:0_K:0
    # CHECK: %register_1
    # CHECK: %read_M_0_N_0_K_0
    # CHECK: %mul_M_0_N_0_K_0
    # CHECK: %iterate
    # CHECK: %get_result_M:0_N:0_K:0
    # CHECK: %write_M:0_N:0_K:0

    # CHECK: === Iterate Subgraph ===
    # CHECK: %acc_M:0_N:0_K:0
    # CHECK: %a
    # CHECK: %read_1_M:0_N:0_K:0
    # CHECK: %b
    # CHECK: %read_2_M:0_N:0_K:0
    # CHECK: %add_M:0_N:0_K:0
    # CHECK: %mma_M:0_N:0_K:0
    # CHECK: %add_1_M:0_N:0_K:0
