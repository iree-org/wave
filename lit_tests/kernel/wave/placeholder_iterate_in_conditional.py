# RUN: python %s | FileCheck %s

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ITER_IDX = tkl.sym.ITER_IDX


@run_test
def test_iterate_result_placeholder():
    """
    Test using an iterate result inside a conditional as init_arg to another iterate.
    This was a bug that was part of the reason for needing an extra write/read pair in the initial StreamK GEMM implementation.
    """
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16, ITER_IDX: 0},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.TilingConstraint(ITER_IDX)]

    @tkw.wave(constraints)
    def iter_placeholder_in_conditional_iter(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        partial: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        is_first: tkl.i32,
    ):
        # MMA accumulation loop
        init_acc = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[init_acc])
        def mac_loop(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Simple conditional
        one = tkw.scalar(1, tkl.i32)
        cond = is_first == one

        # Inside conditional: use mac_loop result as init_arg (triggers bug)
        @tkw.conditional(cond)
        def branch():
            @tkw.iterate(ITER_IDX, init_args=[mac_loop])
            def aggregate_loop(
                acc: tkl.Register[M, N, tkl.f32],
            ) -> tkl.Register[M, N, tkl.f32]:
                peer = tkw.read(partial)
                return acc + peer

            tkw.write(aggregate_loop, c)

        @tkw.conditional(~cond)
        def other_branch():
            tkw.write(mac_loop, partial)

    options = WaveCompileOptions(
        subs={
            M: 32,
            N: 32,
            K: 64,
            ITER_IDX: 2,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    kernel = wave_compile(options, iter_placeholder_in_conditional_iter)
    print(kernel.asm)

    # Check for expected structure: MMA loop, conditional, nested iterate
    # CHECK: scf.for
    # CHECK: scf.if
    # CHECK: scf.for
    # CHECK: arith.addf
