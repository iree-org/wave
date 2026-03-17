# RUN: env WAVE_CACHE_ON=0 python %s | FileCheck %s

"""
Test that IV stride extraction fires for dynamic preshuffle MXFP4 GEMM.

The preshuffle B-data mapping produces floor/Mod index expressions that
the old symbolic extraction cannot handle.  The annotate_iv_strides pass
uses numerical probing to pre-compute the IV stride, which codegen then
uses via the fast path: a simple arith.muli(iv, stride) instead of a
complex floor/Mod tree inside the loop body.

Key invariants verified:
  1. The pipelined scf.for loop exists.
  2. Inside the loop, IV offsets are computed via arith.muli (not floor/Mod).
  3. No arith.floordivsi or arith.remsi inside the loop body -- these
     would indicate the IV stride extraction did NOT fire.
  4. amdgpu.fat_raw_buffer_cast uses dynamically selected validBytes.
"""

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.utils.general_utils import run_test
import wave_lang.kernel.lang as tkl


@run_test
def test_iv_stride_extraction_preshuffle():
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

    # CHECK-LABEL: test_iv_stride_extraction_preshuffle

    # 1. Pipelined scf.for loop with step 1.
    # CHECK: scf.for [[IV:%.*]] = %c0 to %{{.*}} step %c1

    # 2. Inside the loop, IV stride is applied via simple multiply.
    #    The precomputed stride (constant integer) times the IV.
    # CHECK: arith.muli [[IV]], %c{{[0-9]+}} overflow<nsw>

    # 3. The IV offset is added to the base offset.
    # CHECK: arith.addi %{{.*}}, %{{.*}} overflow<nsw>

    # 4. No floor/mod integer division inside the loop body.
    #    These would mean IV stride extraction did NOT fire.
    # CHECK-NOT: arith.floordivsi
    # CHECK-NOT: arith.remsi

    # 5. OOB masking: arith.select chooses validBytes vs 0.
    # CHECK: arith.select %{{.*}}, %c2147483646_i64, %c0_i64 : i64

    # 6. fat_raw_buffer_cast with dynamic validBytes.
    # CHECK: amdgpu.fat_raw_buffer_cast %{{.*}} validBytes(%{{.*}})

    # 7. scaled_mfma inside the loop.
    # CHECK: amdgpu.scaled_mfma

    # 8. Loop ends.
    # CHECK: scf.yield
