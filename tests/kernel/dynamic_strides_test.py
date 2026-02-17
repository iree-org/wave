import torch
from torch.testing import assert_close

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)

from .common.utils import require_cdna_2_or_3_or_4


def _compile_gemm(shape):
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape,
        dynamic_dims=False,
        mfma_variant=MMAType.F32_16x16x16_F16,
    )
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        dynamic_symbols=dynamic_symbols,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, gemm), options


@require_cdna_2_or_3_or_4
def test_gemm_dynamic_strides():
    shape = (1024, 1024, 1024)
    gemm, options = _compile_gemm(shape)

    m, n, k = shape
    a = device_randn(m, k * 4, dtype=torch.float16)
    b = device_randn(n, k * 2, dtype=torch.float16)

    a = a[:, :k]
    b = b[:, :k]

    assert not a.is_contiguous() and not b.is_contiguous()

    c = device_randn(m, n, dtype=torch.float32)
    gemm(a, b, c)

    iree_ref = device_randn(m, n, dtype=torch.float32)
    generate_iree_ref("mmt", [a.contiguous(), b.contiguous()], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


@require_cdna_2_or_3_or_4
def test_gemm_dynamic_strides_non_contiguous_output():
    shape = (1024, 512, 512)
    gemm, options = _compile_gemm(shape)

    m, n, k = shape
    a = device_randn(m, k, dtype=torch.float16)
    b = device_randn(n, k, dtype=torch.float16)
    c_big = device_zeros(m, n * 3, dtype=torch.float32)
    c = c_big[:, :n]

    assert c.shape == (m, n) and not c.is_contiguous()

    gemm(a, b, c)

    iree_ref = device_zeros(m, n, dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


@require_cdna_2_or_3_or_4
def test_gemm_dynamic_strides_non_contiguous_input_and_output():
    shape = (512, 1024, 2048)
    gemm, options = _compile_gemm(shape)

    m, n, k = shape
    a = device_randn(m, k * 2, dtype=torch.float16)[:, :k]
    b = device_randn(n, k * 2, dtype=torch.float16)[:, :k]
    c_big = device_zeros(m, n * 2, dtype=torch.float32)
    c = c_big[:, :n]

    assert not a.is_contiguous() and not b.is_contiguous() and not c.is_contiguous()

    gemm(a, b, c)

    iree_ref = device_zeros(m, n, dtype=torch.float32)
    generate_iree_ref("mmt", [a.contiguous(), b.contiguous()], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)


@require_cdna_2_or_3_or_4
def test_gemm_dynamic_strides_offset():
    shape = (4096, 2048, 4096)
    gemm, options = _compile_gemm(shape)

    m, n, k = shape
    offset_m, offset_k = 4, 8
    # a is a view with row and column offset: logical shape (m, k)
    a_big = device_randn(m + offset_m, k + offset_k, dtype=torch.float16)
    a = a_big[offset_m:, offset_k:]

    offset_n = 2
    b_big = device_randn(n + offset_n, k + offset_k, dtype=torch.float16)
    b = b_big[offset_n:, offset_k:]

    assert a.shape == (m, k) and b.shape == (n, k)
    assert not a.is_contiguous() and not b.is_contiguous()

    c = device_zeros(m, n, dtype=torch.float32)
    gemm(a, b, c)

    iree_ref = device_zeros(m, n, dtype=torch.float32)
    generate_iree_ref("mmt", [a.contiguous(), b.contiguous()], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)
