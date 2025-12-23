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
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_randint,
    device_zeros,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from .common.utils import (
    require_e2e,
    require_cdna2,
    require_cdna3,
    require_cdna_3_or_4,
    require_cdna_2_or_3_or_4,
    require_gpus,
    perf_test,
    param_bool,
)
from wave_lang.kernel.wave.constraints import MMAType, MMAOperand, GenericDot
from wave_lang.kernel.wave.templates.dist_gemm import get_dist_gemm_kernel
from wave_lang.kernel.lang import DataType
import os
import json
from torch.testing import assert_close

default_test_shapes = {}
default_test_shapes["test_gemm"] = [
    (1024, 5120, 640),
    (2048, 10240, 1280),
    (4096, 20480, 2560),
]


@require_e2e
@require_cdna_2_or_3_or_4
@pytest.mark.parametrize("shape", default_test_shapes["test_gemm"])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
@pytest.mark.parametrize(
    "devices",
    [
        pytest.param((1, 1), marks=require_gpus(1)),
        pytest.param((2, 1), marks=require_gpus(2)),
        pytest.param((4, 1), marks=require_gpus(4)),
        pytest.param((8, 1), marks=require_gpus(8)),
        pytest.param((1, 2), marks=require_gpus(2)),
        pytest.param((1, 4), marks=require_gpus(4)),
        pytest.param((1, 8), marks=require_gpus(8)),
        pytest.param((2, 2), marks=require_gpus(4)),
        pytest.param((2, 4), marks=require_gpus(8)),
        pytest.param((4, 2), marks=require_gpus(8)),
    ],
)
@pytest.mark.parametrize("datatype", [torch.float16])
def testDistGemm(
    shape: tuple[int],
    mfma_variant: MMAType,
    datatype: torch.dtype,
    devices: tuple[int, int],
):
    device_m, device_n = devices
    gemm, hyperparams, dynamic_symbols = get_dist_gemm_kernel(
        shape, False, mfma_variant, datatype, device_m, device_n
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        iree_launch_async=False,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)
    a = device_randn(shape[0], shape[2], dtype=datatype)
    b = device_randn(shape[1], shape[2], dtype=datatype)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

    iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
    generate_iree_ref("mmt", [a, b], [iree_ref], options)
    assert_close(c, iree_ref, check_device=False)
