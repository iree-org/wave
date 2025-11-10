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
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
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
from wave_lang.kernel.wave.utils.print_utils import print_trace

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





@run_test
def test_gemm_pipelined_trace():
    """
    Test pipelined GEMM trace output.
    This shows the intermediate representation after scheduling.
    """
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    # constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    # constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    # constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, 0)]
    # constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, 1)]
    # constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    # constraints += [
    #     tkw.HardwareConstraint(threads_per_wave=64,
    #                            #waves_per_block=(2, 2, 1)
    #                            )
    # ]
    @tkw.wave(constraints)
    def gemm(
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

    compile_options = WaveCompileOptions(
        subs=subs,
        canonicalize=True,
        compile_to_mlir=True,
        schedule=SchedulingType.PREFETCH,
    )
    print("test_gemm_pipelined_trace")
    k = wave_compile(compile_options, gemm)
    print(k.asm)

    # CHECK-LABEL: test_gemm_pipelined_trace
    # Verify that the pipelined loop is generated with scf.for
    # CHECK: scf.for
    # CHECK: amdgpu.mfma
    # CHECK: scf.yield




@run_test
def test_gemm_dynamic_pipelined_trace():
    """
    Test dynamic pipelined GEMM trace output.
    This shows the intermediate representation after scheduling with dynamic shapes.
    """
    # constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    # constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    # constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, 0)]
    # constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, 1)]
    # constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    # constraints += [
    #     tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
    # ]

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
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

    # Use dynamic K (not substituted) to trigger dynamic pipelining
    subs = {
        M: 128,
        N: 256,
        #K: 128, # Do not provide K, force it to be dynamic.
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

    compile_options = WaveCompileOptions(
        subs=subs,
        dynamic_symbols=[K],
        canonicalize=True,
        compile_to_mlir=True,
        schedule=SchedulingType.PREFETCH,
    )
    print("test_gemm_dynamic_pipelined_trace")
    k = wave_compile(compile_options, gemm)
    print(k.asm)

    # CHECK-LABEL: test_gemm_dynamic_pipelined_trace
    # Verify that conditional branching is generated for dynamic pipelining
    # The pipelined conditional should check if we have enough iterations
    # CHECK: arith.cmpi sgt
    # CHECK: scf.if
    #
    # Inside the pipelined conditional, we should have:
    # - Initial reads (prologue)
    # - A pipelined loop (scf.for)
    # - Final MMAs (epilogue)
    # CHECK: scf.for
    # CHECK: amdgpu.mfma
    # CHECK: scf.yield %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}
    # CHECK: scf.yield %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}
    # CHECK-NEXT: else
    # CHECK-NEXT: scf.yield %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}
    #
    # After the pipelined conditional, there should be another conditional
    # for the remainder loop
    # CHECK: arith.cmpi sgt
    # CHECK: scf.if
    #
    # Inside the remainder conditional, we should have:
    # - A loop that processes remaining iterations
    # CHECK: scf.for
    # CHECK: amdgpu.mfma
    # CHECK: scf.yield %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}
    # CHECK: scf.yield %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}
    # CHECK-NEXT: else
    # CHECK-NEXT: scf.yield %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}, %{{[a-zA-Z0-9_#]+}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
