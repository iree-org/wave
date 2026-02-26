import torch
import pytest
from pathlib import Path

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randint,
    device_randn,
    device_zeros,
    get_default_device,
    to_default_device,
)
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.templates.test_kernels import (
    get_broadcasted_scale_gemm_mxfp4,
)
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.constraints import (
    ScaledMMAType,
)
from wave_lang.kernel.wave.templates import (
    get_tagged_mxfp4_gemm,
    get_tagged_mxfp4_gemm_preshuffle_b,
)
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_dbuf_schedule,
    get_mxfp4_asymmetric_schedule,
)
from wave_lang.kernel.wave.utils.mxfp_utils import (
    SCALE_GROUP_SIZE,
    generate_gemm_afp4wfp4_inputs,
    mxfp4_to_f32,
    e8m0_to_f32,
    torchScaledGemmMXFP4,
    b_preshuffle,
    e8m0_shuffle,
)

from .common.utils import (
    extract_kernel_metadata,
    glob_asm_files,
    param_bool,
    require_cdna4,
    require_e2e,
    require_gfx1250,
    use_water_backend_bool,
)


def generate_gemm_afp8wfp8_inputs(shape):
    M, N, K = shape
    torch.manual_seed(5)
    # 34 is two packed e2m1 values 0010 which is 1.0.
    x = device_randn((M, K), dtype=torch.float32).to(torch.float8_e5m2)
    w = device_randn((N, K), dtype=torch.float32).to(torch.float8_e5m2)
    w = w.T
    # Scale of 1.0 in e8m0, bias 127.
    x_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8)
    w_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8)
    x_scales = x_scales.T
    w_scales = w_scales.T

    return x, w, x_scales, w_scales


def torchScaledGemmMXFP8(x, w, x_scales, w_scales):
    # First convert the x and w inputs to f32.
    x_f32 = x.to(torch.float32)
    w_f32 = w.to(torch.float32)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32.T
    return torch.mm(x_f32, w_f32)


@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (8192, 8192, 8192)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@param_bool("use_global_to_shared")
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
        SchedulingType.PREFETCH,
        SchedulingType.MODULO,
        SchedulingType.FOUR_STAGE,
    ],
)
def testScaledGemmMXFP4(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
    use_global_to_shared: bool,
    dynamic_dims: bool,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = []
    if dynamic_dims:
        dynamic_symbols = [M, N, K]
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[K]

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
        use_global_to_shared=use_global_to_shared,
        dynamic_symbols=dynamic_symbols,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    out = device_zeros(x.shape[0], w.shape[1], dtype=torch.float32)

    w_t = w.T.contiguous()
    gemm(x, x_scales, w_t, w_scales, out)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    torch.testing.assert_close(torch_out, out, check_dtype=False)


# BMK @ NK -> BMN represents Linear Layer style BMM.
def get_batched_scaled_gemm_template(
    shape: tuple[int, int, int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
) -> tuple[WaveCompileOptions, "LaunchableWave"]:
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[B, M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 256,
        BLOCK_N: 128,
        BLOCK_K: 256,
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = [B, M]
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
        use_buffer_ops=True,
        linearize_shared_access=True,
        dynamic_symbols=dynamic_symbols,
        minimize_shared_allocs=False,
        use_global_to_shared=(enable_scheduling == SchedulingType.PREFETCH),
    )
    return options, batched_gemm


@require_e2e
@require_cdna4
@pytest.mark.parametrize("batch", [4, 8])
@pytest.mark.parametrize(
    "shape",
    [(1024, 1024, 1024), (8192, 8192, 8192), (16384, 16384, 16384), (1, 16384, 1664)],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.PREFETCH,
        SchedulingType.FOUR_STAGE,
    ],
)
@use_water_backend_bool("use_water_backend")
def testScaledBatchedGemmMXFP4(
    batch: int,
    shape: tuple[int, int, int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
    use_water_backend: bool,
):
    options, batched_gemm = get_batched_scaled_gemm_template(
        shape, mfma_variant, enable_scheduling
    )
    options = set_default_run_config(options)
    options.use_water_backend = use_water_backend
    batched_gemm = wave_compile(options, batched_gemm)

    linearized_shape = (batch * shape[0], shape[1], shape[2])
    flat_x, w, flat_x_scales, w_scales = generate_gemm_afp4wfp4_inputs(linearized_shape)
    w_t = w.T.contiguous()

    x = flat_x.view(batch, shape[0], shape[2] // 2)
    x_scales = flat_x_scales.view(batch, shape[0], shape[2] // 32)
    w_t = w_t.view(shape[1], shape[2] // 2)
    w_scales = w_scales.view(shape[1], shape[2] // 32)
    out = device_zeros(batch, shape[0], shape[1], dtype=torch.float32)

    batched_gemm(x, x_scales, w_t, w_scales, out)
    torch_flat_out = torchScaledGemmMXFP4(flat_x, w, flat_x_scales, w_scales)
    torch_out = torch_flat_out.view(batch, shape[0], shape[1])
    torch.testing.assert_close(torch_out, out)


@use_water_backend_bool("use_water_backend")
def testScaledBatchedGemmMXFP4Codegen(use_water_backend: bool, tmp_path: Path):
    shape = (16384, 16384, 16384)
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4
    enable_scheduling = SchedulingType.PREFETCH
    options, batched_gemm = get_batched_scaled_gemm_template(
        shape, mfma_variant, enable_scheduling
    )
    options.target = "gfx950"
    options.dump_intermediates = tmp_path
    options.use_water_backend = use_water_backend
    wave_compile(options, batched_gemm)
    asm_files = glob_asm_files(tmp_path)

    assert len(asm_files) == 1, "Expected 1 ASM file"
    text = asm_files[0].read_text()

    metadata = extract_kernel_metadata(text)

    # We encode the exact registers and wait counts as we want to know if
    # they suddenly change due to backend or upstream MLIR changes.
    if use_water_backend:
        vgpr_count = 164
        vgpr_spill_count = 0
        sgpr_count = 57
        sgpr_spill_count = 0
        waitcounts = [
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt vmcnt(0)",
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "s_waitcnt vmcnt(0)",
            "s_waitcnt lgkmcnt(7)",
            "s_waitcnt lgkmcnt(5)",
            "s_waitcnt lgkmcnt(4)",
            "s_waitcnt lgkmcnt(3)",
            "s_waitcnt lgkmcnt(1)",
            "s_waitcnt lgkmcnt(0)",
        ]
    else:
        vgpr_count = 164
        vgpr_spill_count = 0
        sgpr_count = 59
        sgpr_spill_count = 0
        waitcounts = [
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt vmcnt(0)",
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "s_waitcnt vmcnt(0)",
            "s_waitcnt lgkmcnt(1)",
            "s_waitcnt lgkmcnt(5)",
            "s_waitcnt lgkmcnt(4)",
            "s_waitcnt lgkmcnt(3)",
            "s_waitcnt lgkmcnt(1)",
            "s_waitcnt lgkmcnt(0)",
        ]

    assert (
        metadata.vgpr_count == vgpr_count
    ), f"Expected {vgpr_count} VGPRs, got {metadata.vgpr_count}"
    assert (
        metadata.vgpr_spill_count == vgpr_spill_count
    ), f"Expected {vgpr_spill_count} VGPR spills, got {metadata.vgpr_spill_count}"
    assert (
        metadata.sgpr_count == sgpr_count
    ), f"Expected {sgpr_count} SGPRs, got {metadata.sgpr_count}"
    assert (
        metadata.sgpr_spill_count == sgpr_spill_count
    ), f"Expected {sgpr_spill_count} SGPR spills, got {metadata.sgpr_spill_count}"
    assert (
        metadata.waitcnt_ops == waitcounts
    ), f"Expected {waitcounts} waitcnt operations, got {metadata.waitcnt_ops}"


@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (8192, 8192, 8192)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
def testScaledGemmMXFP8(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f8e5m2)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f8e5m2)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    x, w, x_scales, w_scales = generate_gemm_afp8wfp8_inputs(shape)
    out = device_zeros(x.shape[0], w.shape[1], dtype=torch.float32)

    w_t = w.T.contiguous()
    gemm(x.view(torch.int8), x_scales, w_t.view(torch.int8), w_scales, out)
    torch_out = torchScaledGemmMXFP8(x, w, x_scales, w_scales)
    torch.testing.assert_close(torch_out, out, atol=2e-3, rtol=1e-3, check_dtype=False)


@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (8192, 8192, 8192)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.PREFETCH])
def testScaledGemmPrefetechMXFP4(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def prefetch_gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 256,
        BLOCK_N: 256,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
    )
    options = set_default_run_config(options)
    prefetch_gemm = wave_compile(options, prefetch_gemm)

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    out = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=torch.float32)

    w_t = w.T.contiguous()
    prefetch_gemm(x, x_scales, w_t, w_scales, out)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    torch.testing.assert_close(torch_out, out, check_dtype=False)


# This test can slowly updated to have fused dynamic quantization.
@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(512, 512, 512)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
def testBroadcastedScaleGemmMXFP4(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
):
    broadcasted_scale_scaled_mma, hyperparams = get_broadcasted_scale_gemm_mxfp4(
        shape, mfma_variant
    )
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.NONE,
    )
    options = set_default_run_config(options)
    broadcasted_scale_scaled_mma = wave_compile(options, broadcasted_scale_scaled_mma)

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    out = device_zeros(x.shape[0], w.shape[1], dtype=torch.float32)
    # Taking only first value of scales to test broadcasting.
    # [1024, 32] -> [1024]
    compacted_x_scales = x_scales[:, 0].contiguous()

    w_t = w.T.contiguous()
    broadcasted_scale_scaled_mma(x, compacted_x_scales, w_t, w_scales, out)

    # Re-broadcast/copy along old dimension
    # [1024] -> [1024, 32]
    broadcasted_compacted_x_scales = (
        compacted_x_scales.view(-1, 1).repeat(1, x_scales.shape[-1]).contiguous()
    )
    torch_out = torchScaledGemmMXFP4(x, w, broadcasted_compacted_x_scales, w_scales)

    torch.testing.assert_close(torch_out, out, check_dtype=False)


@require_e2e
@require_cdna4
@pytest.mark.parametrize(
    "shape",
    [(1024, 1024, 8192)],
)
@pytest.mark.parametrize(
    "block_shape",
    [(256, 256, 256)],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [ScaledMMAType.F32_16x16x128_F8F6F4],
)
@pytest.mark.parametrize(
    "wave_shape,use_stagger",
    [
        ((2, 2), False),
        ((4, 2), True),
    ],
)
def testScaledGemmMXFP4ManualDoubleBuf(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    mfma_variant: ScaledMMAType,
    wave_shape: tuple[int, int],
    use_stagger: bool,
):
    """End-to-end test for CDNA4 MXFP4 scaled GEMM with manual double-buffer schedule."""
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block_shape, wave_shape=wave_shape, mfma_variant=mfma_variant
    )
    schedule = get_mxfp4_dbuf_schedule(use_stagger=use_stagger)
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    out = device_zeros(x.shape[0], w.shape[1], dtype=torch.float32)

    w_t = w.T.contiguous()
    gemm(x, x_scales, w_t, w_scales, out)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    torch.testing.assert_close(torch_out, out, check_dtype=False)


@require_e2e
@require_cdna4
@pytest.mark.parametrize(
    "shape",
    [(1024, 1024, 8192)],
)
@pytest.mark.parametrize(
    "block_shape",
    [(256, 256, 256)],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [ScaledMMAType.F32_16x16x128_F8F6F4],
)
@use_water_backend_bool("use_water_backend")
def testScaledGemmMXFP4AsymmetricSchedule(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    mfma_variant: ScaledMMAType,
    use_water_backend: bool,
):
    """End-to-end test for asymmetric MXFP4 GEMM: A through LDS, B direct from global."""
    gemm, options = get_tagged_mxfp4_gemm(
        shape,
        block_shape,
        wave_shape=(1, 4),
        mfma_variant=mfma_variant,
        b_address_space=GLOBAL_ADDRESS_SPACE,
    )
    schedule = get_mxfp4_asymmetric_schedule()
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.use_water_backend = use_water_backend
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    out = device_zeros(x.shape[0], w.shape[1], dtype=torch.float32)

    w_t = w.T.contiguous()
    gemm(x, x_scales, w_t, w_scales, out)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    torch.testing.assert_close(torch_out, out, check_dtype=False)


@require_e2e
@require_cdna4
@pytest.mark.parametrize(
    "shape",
    [(1024, 1024, 8192)],
)
@pytest.mark.parametrize(
    "block_shape",
    [(256, 256, 256)],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [ScaledMMAType.F32_16x16x128_F8F6F4],
)
@use_water_backend_bool("use_water_backend")
def testScaledGemmMXFP4PreshuffleB(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    mfma_variant: ScaledMMAType,
    use_water_backend: bool,
):
    """End-to-end test for MXFP4 GEMM with preshuffled B data and B scales."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block_shape,
        wave_shape=(1, 4),
        mfma_variant=mfma_variant,
    )
    schedule = get_mxfp4_asymmetric_schedule()
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.use_water_backend = use_water_backend
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    w_t = w.T.contiguous()
    w_t_ps = b_preshuffle(w_t)
    x_scales_ps = e8m0_shuffle(x_scales)
    w_scales_ps = e8m0_shuffle(w_scales)

    out = device_zeros(x.shape[0], w_t_ps.shape[0], dtype=torch.float32)
    gemm(x, x_scales_ps, w_t_ps, w_scales_ps, out)

    torch.testing.assert_close(torch_out, out, check_dtype=False)


def get_gfx1250_scaled_gemm_mxfp4_template(
    shape: tuple[int], mfma_variant: ScaledMMAType, enable_scheduling: SchedulingType
) -> tuple[WaveCompileOptions, "LaunchableWave"]:
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    # GFX1250 uses wave32.
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=32,
            mma_type=mfma_variant,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
    )
    return options, gemm


@use_water_backend_bool("use_water_backend")
def testGFX1250ScaledGemmMXFP4Codegen(use_water_backend: bool, tmp_path: Path):
    """Codegen test for GFX1250 scaled WMMA with MXFP4."""
    shape = (1024, 1024, 1024)
    mfma_variant = ScaledMMAType.GFX1250_F32_16x16x128_F8F6F4
    options, gemm = get_gfx1250_scaled_gemm_mxfp4_template(
        shape, mfma_variant, SchedulingType.NONE
    )
    options.target = "gfx1250"
    options.dump_intermediates = tmp_path
    options.use_water_backend = use_water_backend
    wave_compile(options, gemm)

    asm_files = glob_asm_files(tmp_path)
    assert len(asm_files) == 1, "Expected 1 ASM file"
    text = asm_files[0].read_text()

    # Verify scaled WMMA intrinsic is present.
    assert "v_wmma_scale" in text, "Expected v_wmma_scale instruction in assembly"

    metadata = extract_kernel_metadata(text)
    assert (
        metadata.vgpr_spill_count == 0
    ), f"Expected 0 VGPR spills, got {metadata.vgpr_spill_count}"
    assert (
        metadata.sgpr_spill_count == 0
    ), f"Expected 0 SGPR spills, got {metadata.sgpr_spill_count}"


@require_e2e
@require_gfx1250
@pytest.mark.parametrize("shape", [(256, 256, 256), (1024, 1024, 1024)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.GFX1250_F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize(
    "enable_scheduling",
    [
        SchedulingType.NONE,
    ],
)
def testGFX1250ScaledGemmMXFP4(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
):
    """End-to-end test for GFX1250 scaled WMMA with MXFP4."""
    # Input sizes
    options, gemm = get_gfx1250_scaled_gemm_mxfp4_template(
        shape, mfma_variant, enable_scheduling
    )
    options = set_default_run_config(options)
    options.wave_runtime = True
    gemm = wave_compile(options, gemm)

    # Generate inputs and compute reference on CPU.
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(
        shape, device=torch.device("cpu")
    )
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    # Move inputs to GPU.
    x = to_default_device(x)
    w = to_default_device(w)
    x_scales = to_default_device(x_scales)
    w_scales = to_default_device(w_scales)
    out = device_zeros(x.shape[0], w.shape[1], dtype=torch.float32)

    w_t = w.T.contiguous()
    gemm(x, x_scales, w_t, w_scales, out)

    torch.testing.assert_close(torch_out, out.cpu(), check_dtype=False)
