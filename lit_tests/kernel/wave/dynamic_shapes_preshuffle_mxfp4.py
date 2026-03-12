# RUN: python %s | FileCheck %s

"""
Test dynamic-shapes support for preshuffle-B MXFP4 GEMM.

When M, N, K are dynamic (not substituted at compile time), the compiler
must emit a runtime guard (scf.if) that checks whether K is large enough
for the pipelined kernel path (prologue + scf.for + epilogue).  If K is
too small, the else branch yields zero accumulators and falls through to
a non-pipelined epilogue loop.

Key structural invariants verified:
  1. Function signature accepts dynamic index arguments for M, N, K.
  2. scf.if guard selects between pipelined and fallback paths.
  3. The pipelined "then" branch contains:
     - gather_to_lds prefetch (prologue)
     - scf.for main loop with amdgpu.scaled_mfma
     - epilogue scaled_mfma after the loop
  4. The "else" branch yields zero accumulators (scf.yield with %cst).
  5. A second scf.for after the scf.if handles remaining K iterations.
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

    # 2. scf.if guard: pipelined path vs fallback.
    # CHECK: scf.if %{{.*}} ->

    # 3a. Pipelined "then" branch: prologue gather_to_lds prefetch.
    # CHECK: amdgpu.gather_to_lds
    # CHECK: amdgpu.gather_to_lds

    # 3b. Pipelined "then" branch: main loop with scaled_mfma.
    # CHECK: scf.for
    # CHECK: amdgpu.scaled_mfma
    # CHECK: scf.yield

    # 3c. Pipelined "then" branch: epilogue scaled_mfma after loop.
    # CHECK: amdgpu.scaled_mfma

    # 4. Else branch: fallback yields zero accumulators.
    # CHECK: } else {
    # CHECK-NEXT: scf.yield %cst

    # 5. Non-pipelined epilogue scf.for after the scf.if.
    # CHECK: scf.for
    # CHECK: amdgpu.lds_barrier
    # CHECK: amdgpu.gather_to_lds
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

    # 2. scf.if guard: pipelined path vs fallback.
    # CHECK: scf.if %{{.*}} ->

    # 3. Pipelined loop steps by 1 (no epilogue to peel off).
    # CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1

    # 4. Loop carries shared-memory buffers as iter_args (epilogue folded in).
    # CHECK-SAME: memref<{{.*}}, #gpu.address_space<workgroup>>

    # 5. OOB guard: arith.select chooses real validBytes vs 0 for out-of-range
    #    iterations, so the hardware returns zeros on OOB loads.
    # CHECK: arith.select %{{.*}}, %c2147483646_i64, %c0_i64 : i64

    # 6. fat_raw_buffer_cast uses the dynamically selected validBytes.
    # CHECK: amdgpu.fat_raw_buffer_cast %{{.*}} validBytes(%{{.*}})

    # 7. scaled_mfma inside the pipelined loop.
    # CHECK: amdgpu.scaled_mfma

    # 8. Loop body ends; no epilogue mfma between loop end and scf.yield.
    # CHECK: scf.yield
    # CHECK-NEXT: }

    # 9. Then-branch yields loop results directly (no epilogue ops).
    # CHECK: scf.yield

    # 10. Else branch: fallback yields zero accumulators.
    # CHECK: } else {
    # CHECK-NEXT: scf.yield %cst

    # 11. Remainder scf.for after the scf.if.
    # CHECK: scf.for
    # CHECK: amdgpu.scaled_mfma
