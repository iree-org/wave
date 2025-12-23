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
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_randint,
    device_randperm,
    device_zeros,
    to_default_device,
)
from wave_lang.kernel.wave.utils.general_utils import ceildiv

from ..common.utils import param_bool, require_e2e
from ._test_util import get_test_shapes, mark_shapes_skip


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
def test_read_write_same(shape, use_buffer_ops, run_bench):
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
    def double(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        res = tkw.read(a)
        double = res + res
        tkw.write(double, a)

    a = device_randn(shape, dtype=torch.float16)
    ref = a + a
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    double = wave_compile(options, double)

    double(a)
    assert_close(a, ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
def test_offset_read(shape, use_buffer_ops, run_bench):
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

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: k, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i, N: j},
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    off = device_randint(shape[0], shape, dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.take_along_dim(a, off.to(torch.long), dim=0)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
def test_offset_read_one(shape, use_buffer_ops, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    N1 = tkl.sym.N1
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_N // wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N, N1: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: k, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i, N1: j // ELEMS_PER_THREAD},
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N1, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    count = int(ELEMS_PER_THREAD)
    n1 = ceildiv(shape[1], count)
    off = device_randint(shape[0], (shape[0], n1), dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            N1: n1,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    off_expanded = off.repeat_interleave(count, dim=1)[:, : shape[1]].to(torch.long)
    out_ref = torch.take_along_dim(a, off_expanded, dim=0)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
def test_offset_write(shape, use_buffer_ops, run_bench):
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

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i, N: k},
        dynamic_val_mappings={M: i, N: j},
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(a)
        tkw.write(
            res,
            b,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )

    a = device_randn(shape, dtype=torch.float16)
    off = (
        device_randperm(shape[1], dtype=torch.int32)
        .reshape((1, shape[1]))
        .repeat(shape[0], 1)
    )
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.zeros_like(out)
    out_ref = out_ref.scatter(1, off.to(torch.long), a)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize(
    "shape",
    mark_shapes_skip(get_test_shapes("test_copy"), [(111, 813)], "TODO: OOB scatter"),
)
@param_bool("use_buffer_ops", "buf_ops")
def test_offset_write_one(shape, use_buffer_ops, run_bench):
    M = tkl.sym.M
    N = tkl.sym.N
    N1 = tkl.sym.N1
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_N // wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N, N1: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i, N: k + j % ELEMS_PER_THREAD},
        dynamic_val_mappings={M: i, N1: j // ELEMS_PER_THREAD},
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N1, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(a)
        tkw.write(
            res,
            b,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )

    a = device_randn(shape, dtype=torch.float16)
    count = int(ELEMS_PER_THREAD)
    n1 = ceildiv(shape[1], count)
    off = (
        device_randperm(n1, dtype=torch.int32).reshape((1, n1)).repeat(shape[0], 1)
        * count
    )
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            N1: n1,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.zeros_like(out)
    off_expanded = off.repeat_interleave(count, dim=1)
    off_expanded = off_expanded + to_default_device(
        torch.arange(count, dtype=torch.int32)
    ).reshape((1, count)).repeat(shape[0], n1)
    off_expanded = off_expanded[:, : shape[1]].to(torch.long)
    out_ref = out_ref.scatter(1, off_expanded.to(torch.long), a)

    assert_close(out, out_ref)
