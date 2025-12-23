# Copyright 2024-2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from torch.testing import assert_close

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.templates.conv import get_igemm_conv2d
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
from wave_lang.kernel.wave.wave_sim import wave_sim

from ..common.utils import (
    param_bool,
    perf_test,
    require_e2e,
    require_cdna_2_or_3_or_4,
    require_cdna3,
)


@require_e2e
def test_im2col(run_bench):
    # TODO: we don't support unaligned access at the moment so all sizes must
    # be aligned to WG/Wave sizes, c * hw * wf == 8 and number of windows == 64.
    n, c, h, w = 1, 2, 9, 9  # Image.
    cf, hf, wf = c, 2, 2  # Filters.
    padding = 0
    stride = 1

    sym = tkl.sym
    ADDRESS_SPACE = sym.ADDRESS_SPACE

    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF

    H_OUT = (H + 2 * padding - HF) // stride + 1
    W_OUT = (W + 2 * padding - WF) // stride + 1
    SZ_OUT = H_OUT * W_OUT

    # K = HF * WF * C
    # M = SZ_OUT * N
    M = sym.M
    K = sym.K

    # We unroll K dimension according to ELEMS_PER_THREAD value.
    # i.e. for K==8 we will have 2 vector.gather's.
    # Each WG will process 64 windows.
    wave_size = 64
    BLOCK_K = hf * wf * c
    BLOCK_M = 64
    ELEMS_PER_THREAD = 4

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j // (HF * WF),
            H: (i % SZ_OUT) // W_OUT * stride + (j % (HF * WF)) // WF,
            W: (i % SZ_OUT) % W_OUT * stride + (j % (HF * WF)) % WF,
        },
        outputs={M: i, K: j},
    )

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, K: ELEMS_PER_THREAD},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(K, BLOCK_K)]
    # TODO: TilingConstraint doesn't work without actual iterate loop, instead
    # we treat K as WG '1' dimension, but corresponding WG size will be always
    # equal to 1.
    # constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[N, C, H, W, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, mapping=mapping, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (h_out * w_out * n, hf * wf * c)
    a = device_randn((n, c, h, w), dtype=torch.float16)
    b = device_zeros(res_shape, dtype=torch.float16)

    im2col = torch.nn.Unfold(kernel_size=(hf, wf), padding=padding, stride=stride)
    expected = im2col(a)[0, :, :].T

    options = WaveCompileOptions(
        subs={
            N: n,
            C: c,
            W: w,
            H: h,
            WF: wf,
            HF: hf,
            M: res_shape[0],
            K: res_shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b)
    assert_close(b, expected)


# TODO: Fix test for CDNA2. CDNA2 seem to have worse accuracy, atol=0.0094, rtol=10.2405
@require_e2e
@require_cdna_2_or_3_or_4
def test_im2col_mma(run_bench):
    # igemm without final col2im
    n, c, h, w = 1, 4, 9, 9  # Image.
    nf, cf, hf, wf = 64, c, 2, 2  # Filters.
    padding = 0  # TODO: only pad=0 is supported for now
    stride = 1

    x = torch.randn(n, c, h, w, dtype=torch.float16)
    we = torch.randn(nf, cf, hf, wf, dtype=torch.float16)

    convRef = torch.nn.Conv2d(c, nf, hf, stride=stride, padding=padding, bias=False)
    convRef.weight = torch.nn.Parameter(we)
    out_ref = convRef(x).detach()

    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF

    H_OUT = (H + 2 * padding - HF) // stride + 1
    W_OUT = (W + 2 * padding - WF) // stride + 1
    SZ_OUT = H_OUT * W_OUT

    K = HF * WF * C
    M = SZ_OUT * N

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    x_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j // (HF * WF),
            H: (i % SZ_OUT) % W_OUT * stride + (j % (HF * WF)) % WF,
            W: (i % SZ_OUT) // W_OUT * stride + (j % (HF * WF)) // WF,
        },
        outputs={M: i, K: j},
    )
    w_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NF: i % NF, C: j // (HF * WF), HF: j % WF, WF: (j % (HF * WF)) // WF},
        outputs={NF: i, K: j},
    )

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # BLOCK_K = tkl.sym.BLOCK_K
    BLOCK_K = K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
        )
    ]

    def func(
        x: tkl.Memory[N, C, H, W, ADDRESS_SPACE, tkl.f16],
        we: tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, tkl.f16],
        out: tkl.Memory[M, NF, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, NF, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, NF, tkl.f32]) -> tkl.Register[M, NF, tkl.f32]:
            a_reg = tkw.read(
                x,
                mapping=x_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, out, elements_per_thread=ELEMS_PER_THREAD)

    sim_func = wave_sim(constraints)(func)
    gpu_func = tkw.wave(constraints)(func)

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (h_out * w_out * n, nf)
    out_ref = torch.zeros(res_shape, dtype=torch.float32)
    sim_func(x, we, out_ref)

    out = torch.zeros_like(out_ref)

    options = WaveCompileOptions(
        subs={
            N: n,
            C: c,
            W: w,
            H: h,
            NF: nf,
            WF: wf,
            HF: hf,
            BLOCK_M: 64,
            BLOCK_N: 64,
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    gpu_func = wave_compile(options, gpu_func)

    x = x.to("cuda")
    we = we.to("cuda")
    out = out.to("cuda")
    gpu_func(x, we, out)
    assert_close(out, out_ref, rtol=1e-05, atol=1e-05, check_device=False)


_igemm_cases = [
    (1, 5, 5, 10, 2, 2, 2, 2),
    (2, 5, 5, 3, 2, 2, 1, 1),
    (4, 5, 5, 10, 2, 2, 2, 1),
    (2, 5, 5, 10, 2, 2, 1, 1),
    (2, 5, 5, 10, 2, 2, 2, 1),
    (1, 5, 5, 10, 2, 2, 16, 1),
    (1, 5, 5, 10, 2, 2, 1, 2),
    (1, 5, 5, 4, 2, 2, 2, 1),
    (4, 5, 5, 10, 2, 2, 2, 3),
    (4, 5, 5, 10, 2, 2, 1, 3),
    (4, 5, 5, 10, 2, 2, 16, 2),
    (1, 5, 5, 3, 2, 2, 2, 2),
    (4, 5, 5, 10, 2, 2, 16, 1),
    (4, 5, 5, 4, 2, 2, 16, 1),
    (2, 5, 5, 4, 2, 2, 1, 3),
    (2, 5, 5, 4, 2, 2, 2, 1),
    (1, 5, 5, 10, 2, 2, 16, 3),
    (4, 5, 5, 4, 2, 2, 16, 2),
    (4, 5, 5, 10, 2, 2, 2, 1),
    (4, 5, 5, 3, 2, 2, 1, 1),
    (4, 5, 5, 4, 2, 2, 2, 1),
    (4, 5, 5, 3, 2, 2, 2, 1),
    (2, 5, 5, 1, 2, 2, 1, 3),
    (2, 5, 5, 4, 2, 2, 2, 1),
    (2, 5, 5, 10, 2, 2, 16, 1),
    (1, 5, 5, 1, 3, 3, 1, 1),
]

validation_test = lambda *a: pytest.param(*a, marks=pytest.mark.validate_only)

_igemm_cases += [
    perf_test(2, 128, 128, 16, 3, 3, 320, 1),
    perf_test(2, 128, 128, 320, 1, 1, 640, 1),
    perf_test(2, 128, 128, 320, 1, 1, 960, 1),
    perf_test(2, 128, 128, 320, 3, 3, 16, 1),
    perf_test(2, 128, 128, 320, 3, 3, 320, 1),
    perf_test(2, 32, 32, 1280, 1, 1, 1920, 1),
    perf_test(2, 32, 32, 1280, 1, 1, 2560, 1),
    perf_test(2, 32, 32, 1280, 1, 1, 640, 1),
    perf_test(2, 32, 32, 1280, 3, 3, 1280, 1),
    perf_test(2, 32, 32, 1280, 3, 3, 1920, 1),
    perf_test(2, 32, 32, 1280, 3, 3, 2560, 1),
    perf_test(2, 32, 32, 1280, 3, 3, 640, 1),
    perf_test(2, 32, 32, 640, 3, 3, 640, 1),
    perf_test(2, 64, 64, 320, 3, 3, 320, 1),
    perf_test(2, 64, 64, 640, 1, 1, 1280, 1),
    perf_test(2, 64, 64, 640, 1, 1, 1920, 1),
    perf_test(2, 64, 64, 640, 1, 1, 320, 1),
    perf_test(2, 64, 64, 640, 1, 1, 960, 1),
    perf_test(2, 64, 64, 640, 3, 3, 320, 1),
    perf_test(2, 64, 64, 640, 3, 3, 640, 1),
]

_mem_spaces = [
    pytest.param(GLOBAL_ADDRESS_SPACE, id="global", marks=pytest.mark.validate_only),
    pytest.param(SHARED_ADDRESS_SPACE, id="shared"),
]

_layouts = [
    pytest.param("nchw_fchw", marks=pytest.mark.validate_only),
    pytest.param("nhwc_hwcf"),
]


@require_e2e
@require_cdna3
@pytest.mark.parametrize("n, h, w, c, hf, wf, nf, stride", _igemm_cases)
@pytest.mark.parametrize("mem_space", _mem_spaces)
@pytest.mark.parametrize("layout", _layouts)
@pytest.mark.parametrize("optimization_level", [False, True])
@param_bool("use_buffer_ops", "buf_ops")
def test_igemm_conv(
    n,
    h,
    w,
    c,
    hf,
    wf,
    nf,
    stride,
    mem_space,
    layout,
    optimization_level,
    use_buffer_ops,
    run_bench,
    perf_filename_tk,
    perf_filename_iree,
):
    cf = c
    padding = 0  # TODO: only pad=0 is supported for now

    torch.manual_seed(1)
    x = device_randn(n, c, h, w, dtype=torch.float16)
    we = device_randn(nf, cf, hf, wf, dtype=torch.float16)

    convRef = torch.nn.Conv2d(c, nf, hf, stride=stride, padding=padding, bias=False)
    convRef.weight = torch.nn.Parameter(we)
    out_ref = convRef(x).detach().to(torch.float32)

    if layout == "nchw_fchw":
        pass  # Nothing
    elif layout == "nhwc_hwcf":
        x = torch.permute(x, (0, 2, 3, 1)).contiguous()
        we = torch.permute(we, (2, 3, 1, 0)).contiguous()
        out_ref = torch.permute(out_ref, (0, 2, 3, 1)).contiguous()
    else:
        raise ValueError(f"Invalid layout: {layout}")

    conv, hyperparams = get_igemm_conv2d(
        layout=layout,
        n=n,
        h=h,
        w=w,
        c=c,
        hf=hf,
        wf=wf,
        nf=nf,
        stride=stride,
        input_dtype=tkl.f16,
        output_dtype=tkl.f32,
        mem_space=mem_space,
    )
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_ops=use_buffer_ops,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=perf_filename_tk,
        optimization_level=optimization_level,
    )
    options = set_default_run_config(options)
    conv = wave_compile(options, conv)

    out = torch.zeros_like(out_ref)
    conv(x, we, out)
    assert_close(out, out_ref, rtol=1e-03, atol=1e-03)

    if run_bench:
        options.benchmark_results_file = perf_filename_iree

        options.iree_preprocessing_pass_pipeline = "builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)"
        iree_ref = torch.zeros_like(out_ref)
        generate_iree_ref(
            "conv_2d_" + layout,
            [x, we],
            [iree_ref],
            options,
        )
