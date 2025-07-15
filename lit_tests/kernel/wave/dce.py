# RUN: python %s | FileCheck %s

import logging
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.promotion import promote_placeholders
from iree.turbine.kernel.wave.expansion.expansion import expand_graph, add_get_results
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.ops.wave_ops import *
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.utils.print_utils import (
    print_trace,
)
from iree.turbine.kernel.wave.utils.graph_utils import (
    initialize_iter_args,
    DCE,
)
from iree.turbine.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
)

# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K

# Induction variable for dimension K
ARGK = tkl.sym.ARGK


@tkw.wave_trace_only()
def gemm(
    a: tkl.Memory[M, K, SHARED_ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, SHARED_ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.iterate(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        @tkw.conditional(tkw.scalar(THREAD_0, tkl.i32) == tkw.scalar(0, tkl.i32))
        def then():
            # set_wave_prio as some side-effecting op
            tkw.set_wave_prio(1)

        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=4)


@run_test
def test_dce():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
    ]
    with tk.gen.TestLaunchContext(
        {
            M: 128,
            N: 256,
            K: 64,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 64,
        }
    ):
        trace: CapturedTrace = gemm()
        visualize = False
        IndexingContext.current().finalize()
        initialize_iter_args(trace)
        add_get_results(trace)
        infer_types(trace)
        promote_placeholders(trace, constraints)
        set_node_indices(trace, constraints)
        DCE(trace)
        print_trace(trace)

        # Ensure that the conditional with the side-effecting op remains after DCE
        # CHECK: Custom format:
        # CHECK: conditional(condition=eq, subgraph_name=[[REGION:[a-z_0-9]*]],{{.*}})
        # CHECK: [[REGION]]:
        # CHECK: Custom format:
        # CHECK-NEXT: set_wave_prio
