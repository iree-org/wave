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
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import check_leaks
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

from ..common.utils import param_bool, require_e2e, use_water_backend_bool
from ._test_util import get_test_shapes


def get_copy_template(
    shape: tuple[int, int],
    use_dynamic_dims: bool = False,
    run_bench: bool = False,
    use_buffer_ops: bool = False,
    use_water_backend: bool = False,
) -> tuple[WaveCompileOptions, "LaunchableWave"]:
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

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

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    if use_dynamic_dims:
        subs = {
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        }
    else:
        subs = {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        }

    options = WaveCompileOptions(
        subs=subs,
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
        use_water_backend=use_water_backend,
    )
    if use_dynamic_dims:
        options.dynamic_symbols = [M, N]
    return options, test


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
@use_water_backend_bool("use_water_backend")
@check_leaks
def test_copy(
    shape: tuple[int, int],
    use_buffer_ops: bool,
    run_bench: bool,
    use_water_backend: bool,
) -> None:
    options, test = get_copy_template(
        shape,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
        use_water_backend=use_water_backend,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)
    test(a, b)
    assert_close(a, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
@use_water_backend_bool("use_water_backend")
def test_dynamic_copy(
    shape: tuple[int, int],
    use_buffer_ops: bool,
    run_bench: bool,
    use_water_backend: bool,
) -> None:
    options, test = get_copy_template(
        shape,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
        use_water_backend=use_water_backend,
        use_dynamic_dims=True,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)
    test(a, b)
    assert_close(a, b)
