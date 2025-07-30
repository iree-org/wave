from sympy import ceiling, Min

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)


# This function provides a GEMM kernel where the workgroups executing the kernel are re-arranged
# to provide potential L2 Cache Optimizations. More details can be found in docs/wave/workgroup_reordering.rst
def get_reordered_matmul(
    m_size: int,
    n_size: int,
    k_size: int,
    block_m_size: int,
    block_n_size: int,
    block_k_size: int,
    group_m_size: int,
    mfma_variant,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # The grouping factor to group columns by in our reordering scheme
    GROUP_SIZE_M = tkl.sym.GROUP_SIZE_M
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M // 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N // 2)]

    # Global symbols representing the symbolic coordinates of each workgroup ie: (wg0, wg1)
    wg0, wg1 = WORKGROUP_0, WORKGROUP_1
    num_wg_0 = ceiling(M / BLOCK_M)
    num_wg_1 = ceiling(N / BLOCK_N)

    num_wgs_total = num_wg_0 * num_wg_1
    # 8 XCDs on MI300s
    num_xcds = 8

    # flatten workgroup index in column-major order since on hardware,
    # workgroup dim 0 is the fastest dimension
    flat_wg_index = wg1 * num_wg_0 + wg0
    # create XCD-based index (wgs are assigned round-robin to XCDs)
    extra_wgs = num_wgs_total % num_xcds
    xcd_wg_index = (
        (flat_wg_index % num_xcds) * (num_wgs_total // num_xcds)
        + Min(flat_wg_index % num_xcds, extra_wgs)
        + (flat_wg_index // num_xcds)
    )
    # num_wg_group is how many workgroups are in each group
    num_wg_group = GROUP_SIZE_M * num_wg_1
    group_id = xcd_wg_index // num_wg_group
    first_wg_id_0 = group_id * GROUP_SIZE_M
    group_size_m = Min(num_wg_0 - first_wg_id_0, GROUP_SIZE_M)
    new_wg0 = first_wg_id_0 + ((xcd_wg_index % num_wg_group) % group_size_m)
    new_wg1 = (xcd_wg_index % num_wg_group) // group_size_m

    constraints += [tkw.ReorderingConstraint(new_wg0, 0)]
    constraints += [tkw.ReorderingConstraint(new_wg1, 1)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        M: m_size,
        N: n_size,
        K: k_size,
        BLOCK_M: block_m_size,
        BLOCK_N: block_n_size,
        BLOCK_K: block_k_size,
        GROUP_SIZE_M: group_m_size,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
    }
    hyperparams.update(get_default_scheduling_params())
    return gemm, hyperparams
