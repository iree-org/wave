import pytest
import torch
from torch.testing import assert_close

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.templates.reordered_gemm import get_reordered_matmul
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)

from .common.utils import require_cdna_2_or_3_or_4


@require_cdna_2_or_3_or_4
def test_gemm_dynamic_strides():
    shape = (1024, 1024, 1024)
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
        use_dynamic_strides=True,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)
    with open("gemm_dynamic_strides.mlir", "w") as f:
        f.write(gemm.asm)

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
