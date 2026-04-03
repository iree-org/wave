# RUN: python %s | FileCheck %s

"""
Test linearization of GatherToLDS src indices to 1-D physical offsets.

Uses print_ir_after to dump TorchFX IR after flatten_read_indices and
annotate_iv_strides and verifies that GatherToLDS operations have their
src_index flattened to a single $LINEAR_INDEX key, just like regular
Read operations.
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.utils.general_utils import run_test

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_linearize_g2l_simple():
    """Simple GEMM with use_global_to_shared: GatherToLDS gets $LINEAR_INDEX."""
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        use_global_to_shared=True,
        target="gfx950",
    )
    options.print_ir_after = ["flatten_read_indices"]
    wave_compile(options, gemm)

    # CHECK-LABEL: test_linearize_g2l_simple

    # After flatten: GatherToLDS src_index should have $LINEAR_INDEX
    # CHECK-LABEL: After flatten_read_indices

    # GatherToLDS for a (M x K): src_index now has $LINEAR_INDEX
    # CHECK: gather_to_lds(src=a, {{.*}}src_index={$LINEAR_INDEX: {{.*}} : 2 : 1}

    # GatherToLDS for b (N x K): src_index now has $LINEAR_INDEX
    # CHECK: gather_to_lds(src=b, {{.*}}src_index={$LINEAR_INDEX: {{.*}} : 2 : 1}


@run_test
def test_linearize_g2l_preshuffle():
    """Preshuffle MXFP4 GEMM: A data and A scale GatherToLDS get $LINEAR_INDEX
    with concrete strides extracted by annotate_iv_strides."""
    shape = (1024, 1024, 8192)
    block = (128, 256, 256)
    kernel, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(1, 4),
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    schedule = get_mxfp4_asymmetric_schedule(is_bscale_shuffled=True)
    options.print_ir_after = ["flatten_read_indices", "annotate_iv_strides"]
    options.use_buffer_ops = True
    options.compile_to_mlir = True
    options.device = "hip"
    options.target = "gfx950"
    wave_compile(options, kernel, schedule)

    # CHECK-LABEL: test_linearize_g2l_preshuffle

    # ---------------------------------------------------------------
    # 1. After flatten: GatherToLDS and Read ops get $LINEAR_INDEX
    # ---------------------------------------------------------------
    # CHECK-LABEL: After flatten_read_indices

    # A data GatherToLDS: src_index has $LINEAR_INDEX (A goes through shared).
    # Stride field is 1 (not yet extracted).
    # CHECK: gather_to_lds(src=a, {{.*}}src_index={$LINEAR_INDEX: {{.*}} : 16 : 1}

    # A scale GatherToLDS: src_index has $LINEAR_INDEX
    # CHECK: gather_to_lds(src=a_scale, {{.*}}src_index={$LINEAR_INDEX: {{.*}} : 4 : 1}

    # B data Read (from global): also linearized
    # CHECK: read(memory=b, {{.*}}index={$LINEAR_INDEX: {{.*}} : 16 : 1})

    # ---------------------------------------------------------------
    # 2. After annotate_iv_strides: concrete strides extracted
    # ---------------------------------------------------------------
    # CHECK-LABEL: After annotate_iv_strides

    # A data GatherToLDS: stride = K/2 / step = 4096/2 / 2 = 128
    # (K=8192, K/2=4096 is the packed dimension stride, step=2)
    # CHECK: gather_to_lds(src=a, {{.*}}src_index={$LINEAR_INDEX: 128*$ARGK + {{.*}} : 16 : 128}

    # A scale GatherToLDS: stride = 256
    # CHECK: gather_to_lds(src=a_scale, {{.*}}src_index={$LINEAR_INDEX: 256*$ARGK + {{.*}} : 4 : 256}

    # B data Read: stride already extracted (same as linearize_read_indices test)
    # CHECK: read(memory=b, {{.*}}index={$LINEAR_INDEX: 2048*$ARGK + {{.*}} : 16 : 2048})
