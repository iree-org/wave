# RUN: python %s | FileCheck %s

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test

M = tkl.sym.M
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


@run_test
def test_dynamic_mxfp4_quant():
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(K, BLOCK_K, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(K, BLOCK_K),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: BLOCK_M, K: BLOCK_K},
        ),
    ]

    @tkw.wave(constraints)
    def quant(
        x: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f32],
        quant_scale: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f32],
        x_codes: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
    ):
        x_reg = tkw.read(x)
        qs = tkw.read(quant_scale)
        qx = x_reg * qs
        abs_qx = tkw.abs(qx)

        T025 = tkl.Register[M, K, tkl.f32](0.25)
        T075 = tkl.Register[M, K, tkl.f32](0.75)
        T125 = tkl.Register[M, K, tkl.f32](1.25)
        T175 = tkl.Register[M, K, tkl.f32](1.75)
        T250 = tkl.Register[M, K, tkl.f32](2.5)
        T350 = tkl.Register[M, K, tkl.f32](3.5)
        T500 = tkl.Register[M, K, tkl.f32](5.0)

        mag = (
            tkw.cast(abs_qx >= T025, tkl.f32)
            + tkw.cast(abs_qx >= T075, tkl.f32)
            + tkw.cast(abs_qx >= T125, tkl.f32)
            + tkw.cast(abs_qx >= T175, tkl.f32)
            + tkw.cast(abs_qx >= T250, tkl.f32)
            + tkw.cast(abs_qx >= T350, tkl.f32)
            + tkw.cast(abs_qx >= T500, tkl.f32)
        )

        ZERO_F32 = tkl.Register[M, K, tkl.f32](0.0)
        EIGHT_F32 = tkl.Register[M, K, tkl.f32](8.0)
        sign = tkw.select(qx < ZERO_F32, EIGHT_F32, ZERO_F32)

        codes = tkw.cast(mag + sign, tkl.i8)
        tkw.write(codes, x_codes)

    options = WaveCompileOptions(
        subs={
            M: 4,
            K: 64,
            BLOCK_M: 2,
            BLOCK_K: 32,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )

    quant = wave_compile(options, quant)
    print(quant.asm)

    # CHECK-LABEL: test_dynamic_mxfp4_quant
    # CHECK:       func.func @quant
    # CHECK:         math.absf {{.*}} : vector<{{.*}}xf32>
    # CHECK:         arith.cmpf oge, {{.*}} : vector<{{.*}}xf32>
    # CHECK:         arith.select {{.*}} : vector<{{.*}}xi1>, vector<{{.*}}xf32>
    # CHECK:         arith.cmpf olt, {{.*}} : vector<{{.*}}xf32>
    # CHECK:         arith.select {{.*}} : vector<{{.*}}xi1>, vector<{{.*}}xf32>
    # CHECK:         arith.fptosi {{.*}} : vector<{{.*}}xf32> to vector<{{.*}}xi8>
