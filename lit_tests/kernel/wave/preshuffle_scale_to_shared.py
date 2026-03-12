# RUN: python %s | FileCheck %s

"""
Test preshuffle_scale_to_shared pass for a_scale in MXFP4 GEMM.

When a_scale goes through shared memory (LDS) with a preshuffle mapping
(e8m0_shuffle), the preshuffle_scale_to_shared pass:

  1. Replaces the global Read + shared Write pair with GatherToLDS DMA ops
     that load directly from global to LDS in preshuffle order.
  2. Reshapes the a_scale LDS buffer from 2D [K/32, M] to flat 1D [N, 1].
  3. Re-indexes LDS reads to use preshuffle addressing (constant_base + lane*4).
  4. The merge_contiguous_reads pass then coalesces 4 individual byte reads
     into dword (vector<4xi8>) loads from LDS.
"""

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.utils.general_utils import run_test


@run_test
def test_preshuffle_scale_a_gather_to_lds():
    """a_scale DMA: global to LDS via gather_to_lds, no ds_write."""
    shape = (1024, 1024, 8192)
    block = (256, 256, 256)
    kernel, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(1, 4),
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    schedule = get_mxfp4_asymmetric_schedule()
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.compile_to_mlir = True
    options.device = "hip"
    options.target = "gfx950"
    result = wave_compile(options, kernel, schedule)
    print(result.asm)

    # CHECK-LABEL: test_preshuffle_scale_a_gather_to_lds

    # a_scale LDS buffers are reshaped to flat 1D (Nx1).
    # Three buffers for triple-buffering.
    # CHECK-DAG: memref<2048x1xi8, #gpu.address_space<workgroup>>

    # a_scale uses gather_to_lds DMA with dword (4-byte) transfers.
    # Prologue loads two pipeline stages.
    # CHECK: amdgpu.gather_to_lds {{.*}} : vector<4xi8>, {{.*}}, memref<2048x1xi8, #gpu.address_space<workgroup>>
    # CHECK: amdgpu.gather_to_lds {{.*}} : vector<4xi8>, {{.*}}, memref<2048x1xi8, #gpu.address_space<workgroup>>

    # a_scale LDS reads use dword loads (vector<4xi8>) after merge pass.
    # The 1D buffer is reinterpret_cast to memref<2048xi8> for flat access.
    # CHECK: vector.load {{.*}} : memref<2048xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK: vector.load {{.*}} : memref<2048xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK: vector.load {{.*}} : memref<2048xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK: vector.load {{.*}} : memref<2048xi8, #gpu.address_space<workgroup>>, vector<4xi8>

    # No unmerged byte loads from a_scale LDS (regression guard).
    # CHECK-NOT: vector.load {{.*}} : memref<2048xi8, #gpu.address_space<workgroup>>, vector<1xi8>

    # Main loop present with scaled_mfma.
    # CHECK: scf.for
    # CHECK: amdgpu.scaled_mfma 16x16x128

    # Inside loop: gather_to_lds for next-iteration a_scale prefetch.
    # CHECK: amdgpu.gather_to_lds {{.*}} : vector<4xi8>, {{.*}}, memref<2048x1xi8, #gpu.address_space<workgroup>>

    # Inside loop: a_scale LDS reads are still dword loads.
    # CHECK: vector.load {{.*}} : memref<2048xi8, #gpu.address_space<workgroup>>, vector<4xi8>

    # No byte loads inside loop either.
    # CHECK-NOT: vector.load {{.*}} : memref<2048xi8, #gpu.address_space<workgroup>>, vector<1xi8>
