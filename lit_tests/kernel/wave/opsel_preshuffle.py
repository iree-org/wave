# RUN: python %s | FileCheck %s

"""
Test that pre-shuffled (e8m0_shuffle) scale reads with MXFP4 GEMM
use opsel (byte selection) in amdgpu.scaled_mfma at the MLIR level.

The opsel_scaled_mfma pass should replace:
  - Scalar scale operands (f8E8M0FNU)
  - With vector scale operands (vector<4xf8E8M0FNU>)
  - And set scalesIdxA/scalesIdxB attributes to select the byte

This allows hardware to efficiently extract the correct scale byte
from a packed VGPR instead of using separate scalar extracts.
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
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


@run_test
def test_preshuffle_opsel():
    """Test that opsel is applied to preshuffle scale reads."""
    m, n, k = 512, 512, 2048
    block_k = 256
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

    # CHECK-LABEL: test_preshuffle_opsel

    # Check that scales are loaded as vector<4xi8> from global memory
    # CHECK: vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<4xi8>

    # Check that the vector<4xi8> is bitcast to vector<4xf8E8M0FNU>
    # CHECK: %[[SCALE_VEC_A:.+]] = vector.bitcast %{{.*}} : vector<4xi8> to vector<4xf8E8M0FNU>

    # Check that another scale vector is loaded and bitcast
    # CHECK: vector.load %{{.*}} : memref<{{.*}}xi8, strided<[{{.*}}, 1]>>, vector<4xi8>
    # CHECK: %[[SCALE_VEC_B:.+]] = vector.bitcast %{{.*}} : vector<4xi8> to vector<4xf8E8M0FNU>

    # Check that amdgpu.scaled_mfma uses vector scales with opsel (indexed access)
    # The key indicator is the [N] indexing syntax on vector<4xf8E8M0FNU> scales
    # CHECK: amdgpu.scaled_mfma {{.*}} (%{{.*}}[{{[0-9]+}}] * %{{.*}}) * (%{{.*}}[{{[0-9]+}}] * %{{.*}}) + %{{.*}} : vector<4xf8E8M0FNU>, vector<{{.*}}xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<{{.*}}xf4E2M1FN>, vector<4xf32>

    # Verify that we're not using scalar scale extracts (the old pattern)
    # If opsel is working, we should NOT see vector.extract before scaled_mfma
    # CHECK-NOT: vector.extract %{{.*}}[0] : f8E8M0FNU
