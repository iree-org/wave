# RUN: python %s | FileCheck %s

"""
Test that linearization hoists loop-invariant index bases out of scf.for.

When M, N, K are dynamic (not substituted at compile time), the pipelined
loop body should use hoisted ``arith.muli``/``arith.addi`` for GatherToLDS
index computation rather than monolithic ``affine.apply`` ops with many
operands.

Reads whose flat index is truly affine in the IV (A-data GatherToLDS)
get their stride annotated and codegen emits ``base + IV * stride``.
Reads with complex preshuffle mappings that contain floor/Mod terms
interacting with the IV (B-data, B-scale reads) may still use
``affine.apply`` because the expression is not decomposable as
``base + IV * stride`` in the symbolic domain.

Two wave shapes are tested to cover different workgroup tiling patterns:
  (1, 4):  64 threads, 1 wave in M, 4 in N -- block (256, 192, 256)
  (2, 2):  64 threads, 2 waves in M, 2 in N -- block (256, 224, 256)
"""

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.utils.general_utils import run_test
import wave_lang.kernel.lang as tkl


def _compile_dynamic_preshuffle_b(shape, block, wave_shape):
    """Compile an mxfp4 preshuffle-B gemm with dynamic M, N, K."""
    kernel, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=wave_shape,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.eliminate_epilogue = False
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=False, is_bscale_shuffled=True
    )
    options.compile_to_mlir = True
    options.device = "hip"
    options.target = "gfx950"
    result = wave_compile(options, kernel, schedule)
    print(result.asm)


@run_test
def test_hoisted_indices_in_loop_1x4():
    """Dynamic 256x192x256 block, (1,4) wave: GatherToLDS uses hoisted arith."""
    _compile_dynamic_preshuffle_b((8192, 3072, 8192), (256, 192, 256), (1, 4))

    # CHECK-LABEL: test_hoisted_indices_in_loop_1x4
    # CHECK:       func.func @
    #              Inside the pipelined scf.for, GatherToLDS reads should use
    #              hoisted base + IV * stride (arith.muli + arith.addi), not
    #              monolithic affine.apply with 6+ operands.
    #              B-data reads (complex preshuffle) may still use affine.apply
    #              because the expression is not decomposable as base + IV * stride.
    # CHECK:       scf.for
    # CHECK:       arith.muli
    # CHECK:       arith.addi
    # CHECK:       gather_to_lds
    #              KEY CHECK -- do not remove.  The affine.apply count and
    #              CHECK-NOT verify that hoisting is working: only the
    #              residual B-data/B-scale preshuffle reads (with floor/Mod
    #              IV interaction) remain as affine.apply in the loop body.
    # CHECK-COUNT-12: affine.apply
    # CHECK-NOT:   affine.apply
    # CHECK:       scf.yield


@run_test
def test_hoisted_indices_in_loop_2x2():
    """Dynamic 256x224x256 block, (2,2) wave: GatherToLDS uses hoisted arith."""
    _compile_dynamic_preshuffle_b((8192, 3584, 8192), (256, 224, 256), (2, 2))

    # CHECK-LABEL: test_hoisted_indices_in_loop_2x2
    # CHECK:       func.func @
    # CHECK:       scf.for
    # CHECK:       arith.muli
    # CHECK:       arith.addi
    # CHECK:       gather_to_lds
    #              KEY CHECK -- do not remove.  The affine.apply count and
    #              CHECK-NOT verify that hoisting is working: only the
    #              residual B-data/B-scale preshuffle reads (with floor/Mod
    #              IV interaction) remain as affine.apply in the loop body.
    # CHECK-COUNT-28: affine.apply
    # CHECK-NOT:   affine.apply
    # CHECK:       scf.yield
