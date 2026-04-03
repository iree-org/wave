# RUN: python %s | FileCheck %s

"""
Test linearization of complex N-D read indices to 1-D physical offsets.

Uses print_ir_after to dump TorchFX IR after each pass and verifies:

  1. After flatten_read_indices: global-memory reads that originally had
     complex 2-D preshuffle mappings now carry a single $LINEAR_INDEX
     key instead of per-dimension {N: ..., K: ...} indices.

  2. After annotate_iv_strides: the IndexSequence stride field (third
     component of "start : size : stride") is a concrete integer --
     4096 for B data (= K/2) and 512 for B scale.  The start has the
     form STRIDE*$ARGK + BASE, i.e. the loop variable is multiplied by
     the extracted stride and all other address terms are loop-invariant.
"""

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.utils.general_utils import run_test


@run_test
def test_linearize_preshuffle_static():
    """Static shapes: flatten to LINEAR_INDEX, then extract concrete strides."""
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

    # CHECK-LABEL: test_linearize_preshuffle_static

    # ---------------------------------------------------------------
    # 1. After flatten: 2-D mapped reads are now $LINEAR_INDEX
    # ---------------------------------------------------------------
    # CHECK-LABEL: After flatten_read_indices

    # B data: was {N: ..., K/2: ...} with preshuffle mapping;
    # now a single $LINEAR_INDEX.  Stride field is 1 (not yet extracted).
    # CHECK: read(memory=b, {{.*}}index={$LINEAR_INDEX: {{.*}} : 16 : 1})

    # B scale: was {K/32: ..., N: ...} with e8m0_shuffle mapping;
    # also linearized.
    # CHECK: read(memory=b_scale, {{.*}}index={$LINEAR_INDEX: {{.*}} : 1 : 1})

    # ---------------------------------------------------------------
    # 2. After annotate_iv_strides: concrete strides, base+IV*stride form
    # ---------------------------------------------------------------
    # CHECK-LABEL: After annotate_iv_strides

    # B data: per-unit stride = K/2 / step = 4096/2 = 2048.
    # Start is "2048*$ARGK + <loop-invariant base>".
    # IndexSequence prints as "start : size : stride".
    # CHECK: read(memory=b, {{.*}}index={$LINEAR_INDEX: 2048*$ARGK + {{.*}} : 16 : 2048})

    # B scale: per-unit stride = 512/step = 256 (after merge_contiguous_reads).
    # CHECK: read(memory=b_scale, {{.*}}index={$LINEAR_INDEX: 256*$ARGK + {{.*}} : 4 : 256})
