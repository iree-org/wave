# RUN: python %s | FileCheck %s

# Tests that generate_bounds_exprs substitutes reordered workgroup indices
# into bound expressions.  When the wave-tile along N (here 24 = 48/2) is
# not divisible by the MMA vector size (16), the WaveConstraint produces a
# bound expression containing WORKGROUP_1.  With workgroup reordering, that
# symbol must be replaced by the reordered expression (which involves both
# WORKGROUP_0 and WORKGROUP_1), so the emitted affine.apply that computes
# the bound references both %block_id_x and %block_id_y.
#
# Dynamic M, N, K mirrors the real use case (e.g. 256x224x256 block size
# with varying problem shapes).

from sympy import ceiling

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test

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
def test_bounds_respect_workgroup_reordering():
    # BLOCK_N=48 with waves_per_block=(2,2,1) gives wave_tile_N = 24.
    # 24 % 16 (MMA vector size) = 8 != 0, triggering a bound expression
    # that contains WORKGROUP_1.
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
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    options = WaveCompileOptions(
        subs={
            BLOCK_M: 64,
            BLOCK_N: 48,
            BLOCK_K: 16,
            GROUP_SIZE_N: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        dynamic_symbols=(M, N, K),
        compile_to_mlir=True,
    )
    gemm = wave_compile(options, gemm)
    print(gemm.asm)

    # The N-dimension bound expression comes from WaveConstraint.get_index_bound
    # because wave_tile_N=24 is not divisible by the MMA vector size 16.
    # The raw bound is  WORKGROUP_1 * BLOCK_N + wave_id * 24 + 24.
    # With workgroup reordering, WORKGROUP_1 is replaced by an expression
    # involving both block_id_x and block_id_y, so the affine.apply that
    # computes the bound must reference both block IDs.
    #
    # Without the fix, the bound would use only %block_id_y (the raw WG_1),
    # giving incorrect masking when workgroups are reordered.

    # CHECK-LABEL:   test_bounds_respect_workgroup_reordering

    # The reordered N-dimension bound uses %arg3 (dynamic M, needed for
    # ceiling(M/BLOCK_M) in the swizzle), plus both block IDs and thread_id_y.
    # Without the fix the affine.apply would reference only %block_id_y and
    # %thread_id_y.
    # CHECK:         %[[N_BOUND:.+]] = affine.apply #{{.+}}()[%arg3, %block_id_y, %block_id_x, %thread_id_y]
    # CHECK-NEXT:    %[[N_BOUND_CLAMPED:.+]] = arith.minsi %[[N_BOUND]], %arg4 : index

    # The clamped bound is used in a comparison for masking.
    # CHECK:         arith.cmpi slt, %{{.+}}, %[[N_BOUND_CLAMPED]] : index
