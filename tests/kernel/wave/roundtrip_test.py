# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.support.ir_imports import Module, Context
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from wave_lang.kernel.wave.water import _deiree
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from torch.testing import assert_close
from .common.utils import (
    require_e2e,
    require_cdna_2_or_3_or_4,
    require_cdna3,
    param_bool,
)
from .common.shapes import get_test_shapes


@require_e2e
@require_cdna_2_or_3_or_4
@pytest.mark.parametrize("shape", get_test_shapes("chained_gemm"))
@param_bool("enable_scheduling", "sched", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testChainedGemm(
    shape: tuple[int],
    enable_scheduling: bool,
    mfma_variant: MMAType,
    run_bench,
    perf_filename_tk,
):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M: j, N: k}, outputs={B: i, N: k, M: j}
    )

    @tkw.wave(constraints)
    def chained_gemm(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(qk_cast_reg, v_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(
            repeat, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    hyperparams = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        B: batch,
        M: q_seq_len,
        N: v_head_dim,
        K1: qk_head_dim,
        K2: kv_seq_len,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
    )
    options = set_default_run_config(options)
    chained_gemm_ref = wave_compile(options, chained_gemm)

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
    v = device_randn(batch, v_head_dim, kv_seq_len, dtype=torch.float16)
    output = device_zeros(batch, v_head_dim, q_seq_len, dtype=torch.float32)
    chained_gemm_ref(q, k, v, output)

    torch_qk = torch.matmul(q, k.transpose(-1, -2))
    torch_ref = torch.matmul(torch_qk, v.transpose(-1, -2))
    output_for_cmp = output.transpose(-1, -2).to(torch.float16)
    assert_close(output_for_cmp, torch_ref, atol=5e-2, rtol=5e-3)

    with Context() as ctx:
        options.override_mlir = _deiree(Module.parse(chained_gemm_ref.asm))

    # Inject module that contains upstream dialects only, wave_compile() will take care of iree-fying it.
    chained_gemm_deiree = wave_compile(options, chained_gemm)
    output_deiree = device_zeros(batch, v_head_dim, q_seq_len, dtype=torch.float32)
    chained_gemm_deiree(q, k, v, output_deiree)
    output_for_cmp = output_deiree.transpose(-1, -2).to(torch.float16)
    assert_close(output_for_cmp, torch_ref, atol=5e-2, rtol=5e-3)

    # Inject module that is already iree-fied, wave_compile() will just compile to vmfb without any extra transformations.
    options.override_mlir = chained_gemm_ref.asm
    chained_gemm_override = wave_compile(options, chained_gemm)
    output_override = device_zeros(batch, v_head_dim, q_seq_len, dtype=torch.float32)
    chained_gemm_override(q, k, v, output_override)
    output_for_cmp = output_override.transpose(-1, -2).to(torch.float16)
    assert_close(output_for_cmp, torch_ref, atol=5e-2, rtol=5e-3)
