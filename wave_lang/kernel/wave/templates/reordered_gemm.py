from sympy import Max, Min, ceiling
import torch
from typing import Literal, Optional

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)
from wave_lang.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
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
    mfma_variant: tuple[MMAType, MMAType],
    input_dtype: torch.dtype = torch.float16,
    output_dtype: torch.dtype = torch.float32,
    quantized_dtype: Optional[torch.dtype] = None,
    tA: Literal["N", "T"] = "N",
    tB: Literal["N", "T"] = "T",
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
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    input_dtype = torch_dtype_to_wave(input_dtype)
    output_dtype = torch_dtype_to_wave(output_dtype)
    if quantized_dtype:
        quantized_dtype = torch_dtype_to_wave(quantized_dtype)

    if not isinstance(mfma_variant, tuple):
        mfma_variant = (mfma_variant, mfma_variant)

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

    # Clamping group_size_m to be >= 1 to prevent empty groups, or, groups of size 0.
    # This ensures that every division or mod has a valid, nonzero divisor and stays within bounds.
    # (we were seeing OOB accesses otherwise for certain test cases)
    # Resolved Issue: https://github.com/iree-org/wave/issues/315 contains the list of failed TC.
    group_size_m = Max(1, Min(num_wg_0 - first_wg_id_0, GROUP_SIZE_M))

    new_wg0 = first_wg_id_0 + ((xcd_wg_index % num_wg_group) % group_size_m)
    new_wg1 = (xcd_wg_index % num_wg_group) // group_size_m

    constraints += [tkw.ReorderingConstraint(new_wg0, 0)]
    constraints += [tkw.ReorderingConstraint(new_wg1, 1)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant[0])
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    a_shape = (K, M) if tA == "T" else (M, K)
    b_shape = (N, K) if tB == "T" else (K, N)

    a_mapping = tkw.IndexMapping(
        num_iterators=2, inputs={M: i, K: j}, outputs={M: i, K: j}
    )
    b_mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, K: j}, outputs={N: i, K: j}
    )

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[a_shape[0], a_shape[1], ADDRESS_SPACE, input_dtype],
        b: tkl.Memory[b_shape[0], b_shape[1], ADDRESS_SPACE, input_dtype],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, output_dtype],
    ):
        c_reg = tkl.Register[M, N, output_dtype](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[M, N, output_dtype],
        ) -> tkl.Register[M, N, output_dtype]:
            # a_reg: tkw.Register[M, K, input_dtype]
            a_reg = tkw.read(
                a, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=a_mapping
            )
            # b_reg: tkw.Register[N, K, input_dtype]
            b_reg = tkw.read(
                b, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=b_mapping
            )
            if quantized_dtype:
                a_reg = tkw.cast(a_reg, quantized_dtype)
                b_reg = tkw.cast(b_reg, quantized_dtype)
            # acc: tkw.Register[M, N, output_dtype]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        M: m_size,
        N: n_size,
        K: k_size,
        BLOCK_M: block_m_size,
        BLOCK_N: block_n_size,
        BLOCK_K: block_k_size,
        GROUP_SIZE_M: group_m_size,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant[0]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
    }
    hyperparams.update(get_default_scheduling_params())
    return gemm, hyperparams
