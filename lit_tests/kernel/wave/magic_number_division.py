# RUN: python %s | FileCheck %s

from sympy import ceiling

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
)

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
GROUP_SIZE_N = tkl.sym.GROUP_SIZE_N
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_magic_number_div():
    """Test that floordiv/mod by dynamic (runtime) divisors are lowered
    to the magic-number multiply-high trick instead of expensive hardware
    division.

    When kernel dimensions are dynamic, the compiler cannot fold
    floordiv/mod into compile-time constants. The magic-number
    optimisation precomputes ``ceil(2^32 / d)`` once per unique divisor
    and replaces every subsequent division with a 64-bit multiply + shift,
    which is significantly cheaper on GPU.

    We use a GEMM with GROUP_SIZE_N workgroup reordering to exercise
    this: the reordering delinearises the flat workgroup id via
    ``ceildiv(M, BLOCK_M)``, and the GEMM's multiple memory accesses
    (read A, read B, write C) each independently compute reordered
    indices, producing enough dynamic floordiv/mod expressions to
    demonstrate that the expensive magic-number precomputation
    (a single divui) is performed once per divisor and then reused
    by multiple cheap multiply-and-shift sequences.
    """
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    wg0, wg1 = WORKGROUP_0, WORKGROUP_1
    num_wg_0 = ceiling(M / BLOCK_M)

    flat_wg_index = wg1 * num_wg_0 + wg0
    num_wg_group = GROUP_SIZE_N * num_wg_0
    group_id = flat_wg_index // num_wg_group
    first_wg_id_1 = group_id * GROUP_SIZE_N
    new_wg0 = (flat_wg_index % num_wg_group) // GROUP_SIZE_N
    new_wg1 = first_wg_id_1 + (flat_wg_index % num_wg_group) % GROUP_SIZE_N

    constraints += [tkw.ReorderingConstraint(new_wg0, 0)]
    constraints += [tkw.ReorderingConstraint(new_wg1, 1)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    options = WaveCompileOptions(
        subs={
            M: 512,
            N: 1024,
            K: 256,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            GROUP_SIZE_N: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        magic_number_div=True,
    )

    options.dynamic_symbols = [M, N, K]
    for sym in options.dynamic_symbols:
        del options.subs[sym]

    gemm = wave_compile(options, gemm)
    print(gemm.asm)

    # CHECK-LABEL: func.func @gemm
    # CHECK-DAG:     arith.constant 4294967295 : i64
    # CHECK-DAG:     %[[C32:.*]] = arith.constant 32 : i64
    #
    # Magic precomputation: one divui per unique dynamic divisor.
    # CHECK:         arith.divui {{.*}} : i64
    # CHECK:         arith.divui {{.*}} : i64
    #
    # Multiply-high (shrui >> 32) reusing precomputed magic numbers.
    # CHECK:         arith.shrui {{.*}}, %[[C32]] : i64
    #
    # Amortised: mulhi reusing a previously computed magic number
    # with a different dividend — no new divui needed.
    # CHECK-NOT:     arith.divui
    # CHECK-NOT:     arith.divsi
    # CHECK:         arith.shrui {{.*}}, %[[C32]] : i64
    # CHECK-NOT:     arith.divui
    # CHECK-NOT:     arith.divsi
    # CHECK:         return
