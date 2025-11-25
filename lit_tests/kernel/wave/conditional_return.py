# RUN: python %s | FileCheck %s

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)

M = tkl.sym.M
N = tkl.sym.N
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_conditional_with_return():
    """
    This test demonstrates a conditional operation with return values.
    The conditional checks if a threshold is met and either processes data
    or returns the initial value unchanged.
    """
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def conditional_kernel(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f16],
        threshold: tkl.f32,
    ):
        result_reg = tkl.Register[M, N, tkl.f32](0.0)

        threshold_scalar = tkw.scalar(0.5, tkl.f32)

        @tkw.conditional(threshold > threshold_scalar, else_return=[result_reg])
        def process_data(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            a_f32 = tkw.cast(a_reg, tkl.f32)
            result = acc + a_f32
            return result

        tkw.write(process_data, b, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            BLOCK_M: 64,
            BLOCK_N: 64,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    kernel = wave_compile(options, conditional_kernel)
    print(kernel.asm)

    # CHECK: scf.if
    # CHECK: scf.yield %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}
    # CHECK: else
    # CHECK: scf.yield %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}
