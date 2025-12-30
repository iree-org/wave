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
    device_randn,
    device_randint,
    device_tensor,
    device_zeros,
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

from .common.utils import (
    extract_kernel_metadata,
    glob_asm_files,
    param_bool,
    require_cdna4,
    require_e2e,
    use_water_backend_bool,
)

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(shape):
    M, N, K = shape
    torch.manual_seed(5)
    # 34 is two packed e2m1 values 0010 which is 1.0.
    x_low = device_randint(0, 16, (M, K // 2), dtype=torch.uint8)
    x_high = device_randint(0, 16, (M, K // 2), dtype=torch.uint8)
    x = x_low | x_high << 4
    w_low = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w_high = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w = w_low | w_high << 4
    w = w.T
    # Scale of 1.0 in e8m0, bias 127.
    x_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8)
    w_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8)
    x_scales = x_scales.T.contiguous()
    w_scales = w_scales.T.contiguous()

    return x, w, x_scales, w_scales


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


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = device_tensor(mxfp4_list, dtype=torch.float32)
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def torchScaledGemmMXFP4(x, w, x_scales, w_scales):
    # First convert the x and w inputs to f32.
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.T)
    w_f32 = w_f32.T
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32.T
    return torch.mm(x_f32, w_f32)


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

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
        use_global_to_shared=use_global_to_shared,
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
        vgpr_count = 140
        vgpr_spill_count = 0
        sgpr_count = 58
        sgpr_spill_count = 0
        waitcounts = [
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt vmcnt(0)",
            "s_waitcnt lgkmcnt(10)",
            "s_waitcnt lgkmcnt(1)",
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt lgkmcnt(1)",
            "s_waitcnt lgkmcnt(1)",
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "s_waitcnt vmcnt(0)",
            "s_waitcnt lgkmcnt(8)",
            "s_waitcnt lgkmcnt(6)",
            "s_waitcnt lgkmcnt(5)",
            "s_waitcnt lgkmcnt(3)",
            "s_waitcnt lgkmcnt(1)",
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt lgkmcnt(1)",
            "s_waitcnt lgkmcnt(0)",
        ]
    else:
        vgpr_count = 162
        vgpr_spill_count = 0
        sgpr_count = 59
        sgpr_spill_count = 0
        waitcounts = [
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt vmcnt(0)",
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "s_waitcnt vmcnt(0)",
            "s_waitcnt lgkmcnt(7)",
            "s_waitcnt lgkmcnt(6)",
            "s_waitcnt lgkmcnt(5)",
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
