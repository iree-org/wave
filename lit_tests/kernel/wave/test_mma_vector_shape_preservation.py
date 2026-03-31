# RUN: python %s | FileCheck %s

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as wave
from wave_lang.kernel._support.indexing import IndexingContext
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.ops.wave_ops import get_custom
from wave_lang.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
)
from wave_lang.kernel.wave.expansion.expansion import add_get_results
from wave_lang.kernel.wave.type_inference import infer_types
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.utils.graph_utils import initialize_iter_args
from wave_lang.kernel.wave.utils.print_utils import print_trace

# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B  # Batch dimension

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B

# Address spaces
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

# Thread IDs
THREAD_0 = tkl.sym.THREAD_0
THREAD_1 = tkl.sym.THREAD_1


@wave.wave_trace_only()
def batched_mma(
    a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[B, M, N, tkl.f32](0.0)
    a_reg = wave.read(a, elements_per_thread=4)
    b_reg = wave.read(b, elements_per_thread=4)
    acc = wave.mma(a_reg, b_reg, c_reg)
    wave.write(acc, c, elements_per_thread=4)


@run_test
def test_mma_vector_shape_not_overridden():
    """
    Test that MMA instruction-derived vector shapes (M=16, N=16, K=16) are
    preserved and not overridden by hardware constraint vector_shape..
    """
    constraints: list[wave.Constraint] = [wave.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [wave.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [wave.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [wave.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [wave.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]
    constraints += [
        wave.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            # M, N, K intentionally dimensions don't match those needed for MMA.
            vector_shapes={B: 0, M: 32, N: 27, K: 15},
            mma_type=wave.MMAType.F32_16x16x16_F16,
        )
    ]

    subs = {
        M: 64,
        N: 64,
        K: 16,
        B: 1,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 16,
        BLOCK_B: 1,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
    }

    with IndexingContext() as idxc:
        idxc.subs = subs
        trace = batched_mma()
        idxc.finalize()
        initialize_iter_args(trace, canonicalize_output=False)
        add_get_results(trace, canonicalize_output=False)
        infer_types(trace, canonicalize_output=False)
        set_node_indices(trace, constraints, canonicalize_output=False)

        # Verify that MMA nodes have the correct vector_shapes
        # This is the key verification that the fix is working
        def is_mma(node):
            custom = get_custom(node)
            return custom is not None and hasattr(custom, "mma_type")

        mma_nodes = trace.walk(is_mma)
        for mma_node in mma_nodes:
            custom = get_custom(mma_node)
            # MMA instruction should have M=16, N=16, K=16 from the instruction type
            # and B=0 added from hardware constraint
            assert M in custom.vector_shapes, "M dimension should be in vector_shapes"
            assert N in custom.vector_shapes, "N dimension should be in vector_shapes"
            assert K in custom.vector_shapes, "K dimension should be in vector_shapes"
            assert B in custom.vector_shapes, "B dimension should be in vector_shapes"

            # The key assertion: MMA shapes should be 16x16x16, NOT overridden
            assert (
                custom.vector_shapes[M] == 16
            ), f"M should be 16 from MMA type, got {custom.vector_shapes[M]}"
            assert (
                custom.vector_shapes[N] == 16
            ), f"N should be 16 from MMA type, got {custom.vector_shapes[N]}"
            assert (
                custom.vector_shapes[K] == 16
            ), f"K should be 16 from MMA type, got {custom.vector_shapes[K]}"
            # Batch dimension should be added from hardware constraint
            assert (
                custom.vector_shapes[B] == 0
            ), f"B should be 0 from hardware constraint, got {custom.vector_shapes[B]}"

        print_trace(trace)

    # CHECK-LABEL: test_mma_vector_shape_not_overridden

    # The register indices should reflect the MMA-derived shapes (16x16x16)
    # If the shapes were incorrectly overridden, the stride/size patterns would differ
    # The key is that M dimension has size 4 and stride 16 (from 16x16x16 MMA)
    # CHECK: register({{.*}}index={B: $WG2*BLOCK_B : 1 : 1, M: {{.*}} : 4 : 16, N: {{.*}} : 1 : 1})

    # Verify MMA node reads have correct strides based on 16x16x16 shapes
    # CHECK: read(memory=a, elements_per_thread=4,
    # CHECK-SAME: K: {{.*}} : 4 : 1})

    # CHECK: read(memory=b, elements_per_thread=4,
    # CHECK-SAME: K: {{.*}} : 4 : 1})
