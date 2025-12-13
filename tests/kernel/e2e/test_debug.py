# Copyright 2024-2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sympy
import torch
from torch.testing import assert_close

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

from ..common.utils import (
    param_bool,
    require_e2e,
    require_cdna_2_or_3_or_4,
    require_rdna4,
)


@require_e2e
@param_bool("dynamic_dims", "dyn")
def test_debug_log_core(dynamic_dims: bool):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    shape = (256, 128)

    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i // 2, N: j},
        outputs={M: i, N: j},
    )
    write_mapping = tkw.IndexMapping(
        num_iterators=read_mapping.num_iterators,
        inputs=read_mapping.output_mapping,
        outputs=read_mapping.input_mapping,
    )

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]

    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    printer_args = None
    handler_arg = None

    def printer(*args):
        nonlocal printer_args
        printer_args = args

    def handler(arg):
        nonlocal handler_arg
        handler_arg = arg

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a)
        rhs = tkw.read(b)
        tkw.debug_log(lhs, printer=printer)
        tkw.debug_log(rhs, label="rhslog")
        lhs_mapped = tkw.read(a, mapping=read_mapping)
        tkw.debug_log(lhs_mapped, label="lhs_mapped", mapping=write_mapping)
        res = lhs + rhs
        tkw.debug_log(res, label="res", handler=handler)
        tkw.write(res, c)

    a = device_randn(shape, dtype=torch.float16)
    b = device_randn(shape, dtype=torch.float16)
    c = device_zeros(shape, dtype=torch.float16)
    ref = a + b

    subs = {
        ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
    }
    dynamic_symbols = []
    if dynamic_dims:
        dynamic_symbols = [M, N]
    else:
        subs[M] = shape[0]
        subs[N] = shape[1]
    options = WaveCompileOptions(
        subs=subs,
        dynamic_symbols=dynamic_symbols,
    )
    options = set_default_run_config(options)

    test = wave_compile(options, test)

    debug_logs = {}
    test(a, b, c, debug_logs=debug_logs)
    assert_close(a, debug_logs["debug_log_output_0"]["value"])
    assert printer_args[0] == "debug_log_output_0"
    assert printer_args[1] is debug_logs["debug_log_output_0"]["value"]
    assert_close(b, debug_logs["rhslog"]["value"])
    assert_close(c, debug_logs["res"]["value"])
    # with the input mapping the rows of the first half are duplicated, with the
    # output mapping the duplicate rows are written back only to the first half.
    assert_close(
        a[0 : shape[0] // 2, :], debug_logs["lhs_mapped"]["value"][0 : shape[0] // 2, :]
    )
    assert handler_arg == debug_logs


@require_e2e
@pytest.mark.parametrize(
    "mfma_variant, threads_per_wave",
    [
        pytest.param(tkw.MMAType.F32_16x16x16_F16, 64, marks=require_cdna_2_or_3_or_4),
        pytest.param(
            tkw.MMAType.RDNA4_WAVE32_F32_16x16x16_F16, 32, marks=require_rdna4
        ),
    ],
)
def test_debug_log_iteration_dims(mfma_variant, threads_per_wave):
    iterations = 4

    # Basic gemm since it has an iteration.
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave, mma_type=mfma_variant
        ),
    ]

    debug_logs = None

    def handler(logs):
        nonlocal debug_logs
        debug_logs = logs

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkw.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkw.Register[M, N, tkl.f32]) -> tkw.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            tkw.debug_log(
                acc,
                "acc",
                extra_iteration_dimensions=[(tkl.sym.iter, K, iterations)],
                handler=handler,
            )
            return acc

        tkw.write(repeat, c)

    def test_gemm():
        m, n, k = 128, 256, 128  # Small dimensions for testing

        torch.manual_seed(0)
        a = torch.randn(m, k, dtype=torch.float16, device="cuda")
        b = torch.randn(n, k, dtype=torch.float16, device="cuda")
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

        hyperparams = {
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            M: m,
            N: n,
            K: k,
        }

        options = WaveCompileOptions(
            subs=hyperparams,
        )
        options = set_default_run_config(options)
        compiled_gemm = wave_compile(options, gemm)

        compiled_gemm(a, b, c)

        assert len(debug_logs["acc"]["value"]) == iterations
        assert debug_logs["acc"]["iteration_dimensions"] == [tkl.sym.iter]
        # The last element of the acc log should be the final iteration, and
        # thus equal to the final output.
        assert torch.equal(debug_logs["acc"]["value"][-1], c)
        # Meanwhile the first element should be quite different.
        assert not torch.allclose(debug_logs["acc"]["value"][0], c)

    test_gemm()
