# RUN: python %s | FileCheck %s

"""
Test dynamic-shapes support for preshuffle-B MXFP4 GEMM.

When M, N, K are dynamic (not substituted at compile time), the compiler
emits a pipelined kernel with gather_to_lds prefetch and software-pipelined
scf.for loop.  Index simplification (factoring out divisor-multiples from
floor/Mod expressions) enables the scheduler to prove the pipeline guard
is always satisfied, eliminating the scf.if entirely.

Key structural invariants verified:
  1. Function signature accepts dynamic index arguments for M, N, K.
  2. Prologue gather_to_lds prefetch loads.
  3. scf.for main loop with amdgpu.scaled_mfma.
  4. Epilogue scaled_mfma after the loop.
"""

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.utils.general_utils import run_test
import wave_lang.kernel.lang as tkl


@run_test
def test_dynamic_preshuffle_b_mxfp4():
    shape = (1024, 1024, 8192)
    block = (128, 256, 256)
    kernel, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(1, 4),
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    schedule = get_mxfp4_asymmetric_schedule(is_bscale_shuffled=True)
    options.use_buffer_ops = True
    options.compile_to_mlir = True
    options.device = "hip"
    options.target = "gfx950"
    result = wave_compile(options, kernel, schedule)
    print(result.asm)

    # CHECK-LABEL: test_dynamic_preshuffle_b_mxfp4

    # 1. Dynamic index arguments for M, N, K in function signature.
    # CHECK: func.func @gemm(%arg0: {{.*}}, %arg1: {{.*}}, %arg2: {{.*}}, %arg3: {{.*}}, %arg4: {{.*}}, %arg5: index, %arg6: index, %arg7: index)

    # 2. No scf.if guard — simplification proves the pipeline guard
    #    is always satisfied.
    # CHECK-NOT: scf.if

    # 3. Prologue gather_to_lds prefetch.
    # CHECK: amdgpu.gather_to_lds
    # CHECK: amdgpu.gather_to_lds

    # 4. Main pipelined loop with scaled_mfma.
    # CHECK: scf.for
    # CHECK: amdgpu.scaled_mfma
    # CHECK: scf.yield

    # 5. Epilogue scaled_mfma after the loop.
    # CHECK: amdgpu.scaled_mfma


@run_test
def test_dynamic_preshuffle_b_mxfp4_eliminate_epilogue():
    shape = (1024, 1024, 8192)
    block = (128, 256, 256)
    kernel, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(1, 4),
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.eliminate_epilogue = True
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=True, is_bscale_shuffled=True
    )
    options.use_buffer_ops = True
    options.compile_to_mlir = True
    options.device = "hip"
    options.target = "gfx950"
    result = wave_compile(options, kernel, schedule)
    print(result.asm)

    # CHECK-LABEL: test_dynamic_preshuffle_b_mxfp4_eliminate_epilogue

    # 1. Dynamic index arguments for M, N, K in function signature.
    # CHECK: func.func @gemm(%arg0: {{.*}}, %arg1: {{.*}}, %arg2: {{.*}}, %arg3: {{.*}}, %arg4: {{.*}}, %arg5: index, %arg6: index, %arg7: index)

    # 2. No scf.if guard — simplification proves it always satisfied.
    # CHECK-NOT: scf.if

    # 3. Dynamic valid bytes: runtime total clamped to hw SRD limit
    #    (appears in prologue, before the loop).
    # CHECK: arith.minui %{{.*}}, %c2147483646_i64 : i64

    # 4. Pipelined loop steps by 2 (unroll factor 2, no epilogue to peel off).
    # CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c2

    # 5. Loop carries vector iter_args (epilogue folded in).
    # CHECK-SAME: vector<4xf32>

    # 6. OOB guard: arith.select chooses clamped validBytes vs 0 for
    #    out-of-range iterations, so the hardware returns zeros on OOB loads.
    # CHECK: arith.select %{{.*}}, %{{.*}}, %c0_i64 : i64

    # 7. fat_raw_buffer_cast uses the dynamically selected validBytes.
    # CHECK: amdgpu.fat_raw_buffer_cast %{{.*}} validBytes(%{{.*}})

    # 8. scaled_mfma inside the pipelined loop.
    # CHECK: amdgpu.scaled_mfma

    # 9. Loop body ends; no epilogue mfma between loop end and scf.yield.
    # CHECK: scf.yield
    # CHECK-NEXT: }
