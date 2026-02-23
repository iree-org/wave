# RUN: python %s | FileCheck %s

"""
Test merge_contiguous_reads pass on pre-shuffled (e8m0_shuffle) scale
reads for MXFP4 GEMM.

The e8m0_shuffle index mapping rearranges scale data so that each thread's
scale elements land in contiguous groups in physical memory.  The merge pass
should combine the expanded scalar reads into wider vector loads:

  BLOCK_K=128 -> 4 scale elements -> 2 groups of 2 -> vector<2xi8>
  BLOCK_K=256 -> 8 scale elements -> 2 groups of 4 -> vector<4xi8>

The shuffle layout requires K/32 >= 64 (i.e. K >= 2048) for the groups to
land contiguously in the row-major [M, K/32] scale tensor.

Also verifies that the opsel_scaled_mfma pass enables byte selection in
amdgpu.scaled_mfma, replacing scalar scale operands with vector operands
and scalesIdxA/scalesIdxB attributes for efficient hardware extraction.
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    run_test,
)

# Symbols shared by all tests.
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
K_SCALE_SHUFFLED = tkl.sym.K_SCALE_SHUFFLED


def get_preshuffle_kernel():
    """Return the pre-shuffled MXFP4 GEMM kernel with e8m0_shuffle mappings."""
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=ScaledMMAType.F32_16x16x128_F8F6F4,
        ),
    ]

    # e8m0_shuffle index mapping: logical (iter0, iter1) -> physical (row, col).
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    shuffle_expr = (
        (j // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
        + (i // 8) * 256
        + ((i % 8) % 4) * 64
        + ((j % 32) % 16) * 4
        + (((i % 8) // 4) * 2)
        + ((j % 32) // 16)
    )

    a_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            M: shuffle_expr // K_SCALE_SHUFFLED,
            K: shuffle_expr % K_SCALE_SHUFFLED,
        },
        outputs={K: i, M: j},
    )

    k = tkw.IndexMapping.iterator(0)
    n = tkw.IndexMapping.iterator(1)

    shuffle_expr_b = (
        (n // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
        + (k // 8) * 256
        + ((k % 8) % 4) * 64
        + ((n % 32) % 16) * 4
        + (((k % 8) // 4) * 2)
        + ((n % 32) // 16)
    )

    b_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: shuffle_expr_b // K_SCALE_SHUFFLED,
            K: shuffle_expr_b % K_SCALE_SHUFFLED,
        },
        outputs={K: k, N: n},
    )

    @tkw.wave(constraints)
    def preshuffle_scaled_mma(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale, mapping=a_scale_mapping)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale, mapping=b_scale_mapping)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    return preshuffle_scaled_mma


def compile_and_print(m, n, k, block_k):
    """Compile the preshuffle kernel with given dimensions and print MLIR."""
    k_scale_shuffled = (((k // 32) + 7) // 8) * 8
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 128,
        BLOCK_N: 128,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        K_SCALE_SHUFFLED: k_scale_shuffled,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        device="hip",
        target="gfx950",
        compile_to_mlir=True,
        use_global_to_shared=True,
    )
    kernel = get_preshuffle_kernel()
    result = wave_compile(options, kernel)
    print(result.asm)


@run_test
def test_preshuffle_scale_merge_block_k_128():
    # BLOCK_K=128: 4 scale elements per thread -> 2 groups of 2 -> vector<2xi8>.
    compile_and_print(m=512, n=512, k=2048, block_k=128)

    # CHECK-LABEL: test_preshuffle_scale_merge_block_k_128

    # Each scale tensor produces 2 merged vector<2xi8> loads from global.
    # CHECK:      vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<2xi8>
    # CHECK:      vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<2xi8>
    # CHECK:      vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<2xi8>
    # CHECK:      vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<2xi8>

    # No unmerged scalar scale loads from global should remain.
    # CHECK-NOT:  vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<1xi8>


@run_test
def test_preshuffle_scale_merge_block_k_256():
    # BLOCK_K=256: 8 scale elements per thread -> 2 groups of 4 -> vector<4xi8>.
    compile_and_print(m=512, n=512, k=2048, block_k=256)

    # CHECK-LABEL: test_preshuffle_scale_merge_block_k_256

    # Each scale tensor produces 2 merged vector<4xi8> loads from global.
    # CHECK:      vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<4xi8>
    # CHECK:      vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<4xi8>
    # CHECK:      vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<4xi8>
    # CHECK:      vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<4xi8>

    # No unmerged scalar scale loads from global should remain.
    # CHECK-NOT:  vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<1xi8>

    # Check that amdgpu.scaled_mfma uses opsel (indexed access into scale values)
    # The key indicator is the [N] indexing syntax on f8E8M0FNU scale operands.  Check %REG[1] as a simple check that we are doing a non-zero index
    # CHECK: amdgpu.scaled_mfma {{.*}} (%{{.*}}[1] * %{{.*}}) * (%{{.*}}[1] * %{{.*}}) + %{{.*}} : vector<4xf8E8M0FNU>, vector<{{.*}}xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<{{.*}}xf4E2M1FN>, vector<4xf32>

    # Verify that we're not using scalar scale extracts (the old pattern)
    # If opsel is working, we should NOT see vector.extract before scaled_mfma
    # CHECK-NOT: vector.extract %{{.*}}[0] : f8E8M0FNU


@run_test
def test_preshuffle_b_opsel():
    """Verify opsel pass on B scales from the preshuffle_b template kernel.

    Uses get_tagged_mxfp4_gemm_preshuffle_b (B and B_scale read directly
    from global with preshuffle mappings).  With BLOCK_K=256, B scale loads
    are merged into vector<4xi8>.  The opsel pass should convert them to
    vector<4xf8E8M0FNU> used directly in amdgpu.scaled_mfma with indexed
    byte selection, eliminating all extract_strided_slice and scalar
    vector.extract ops on f8E8M0FNU.
    """
    shape = (1024, 1024, 8192)
    block = (256, 256, 256)
    kernel, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(1, 4)
    )
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.compile_to_mlir = True
    options.device = "hip"
    options.target = "gfx950"
    schedule = get_mxfp4_asymmetric_schedule()
    result = wave_compile(options, kernel, schedule)
    print(result.asm)

    # CHECK-LABEL: test_preshuffle_b_opsel

    # B scale (3rd type) should be vector<4xf8E8M0FNU> from the opsel pass.
    # A scale remains scalar because A goes through LDS (not direct global loads).
    # Type signature: A_scale, A_src, B_scale, B_src, acc

    # No scaled_mfma before the first match should have scalar B scale.
    # CHECK-NOT: amdgpu.scaled_mfma {{.*}} : {{.*}}, vector<{{.*}}xf4E2M1FN>, f8E8M0FNU,

    # At least one scaled_mfma has the correct vector<4xf8E8M0FNU> B scale.
    # CHECK: amdgpu.scaled_mfma {{.*}} : {{.*}}, vector<{{.*}}xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<{{.*}}xf4E2M1FN>, vector<4xf32>

    # No remaining scaled_mfma should have scalar B scale either.
    # CHECK-NOT: amdgpu.scaled_mfma {{.*}} : {{.*}}, vector<{{.*}}xf4E2M1FN>, f8E8M0FNU,
