# RUN: python %s | FileCheck %s


import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
)

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
B_KV = tkl.sym.B_KV
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
GROUP_SIZE_N = tkl.sym.GROUP_SIZE_N
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_gather_to_shared():
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
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
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
    gemm = wave_compile(options, gemm)
    print(gemm.asm)

    # CHECK-LABEL:    test_gather_to_shared
    # CHECK:          func.func @gemm
    # CHECK-COUNT-1:    memref.alloc()
    # CHECK:            scf.for
    # CHECK:              amdgpu.lds_barrier
    # CHECK-COUNT-2:      amdgpu.gather_to_lds {{.*}} vector<2xf16>
    # CHECK:              rocdl.s.waitcnt
    # CHECK:              amdgpu.lds_barrier
    # CHECK-COUNT-2:      vector.load
    # CHECK:              amdgpu.mfma
    # CHECK-COUNT-4:    vector.store


@run_test
def test_gather_to_shared_cant_k():
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
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 63,
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
    gemm = wave_compile(options, gemm)
    print(gemm.asm)

    # CHECK-LABEL:    test_gather_to_shared_cant_k
    # CHECK-NOT:        amdgpu.gather_to_lds


@run_test
def test_gather_to_shared_dyn_k():
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
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
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
        dynamic_symbols=[K],
    )
    gemm = wave_compile(options, gemm)
    print(gemm.asm)

    # CHECK-LABEL:    test_gather_to_shared_dyn_k
    # CHECK-NOT:        amdgpu.gather_to_lds


@run_test
def test_gather_to_shared_scaled_dims():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.ScaledMMAType.F32_16x16x128_F8F6F4,
        )
    ]

    @tkw.wave(constraints)
    def scaled_gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    options = WaveCompileOptions(
        subs={
            M: 1024,
            N: 1024,
            K: 1024,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 256,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        use_global_to_shared=True,
        target="gfx950",
    )
    scaled_gemm = wave_compile(options, scaled_gemm)
    print(scaled_gemm.asm)

    # CHECK-LABEL:    test_gather_to_shared_scaled_dims
    # CHECK:          func.func @scaled_gemm
    # CHECK-COUNT-1:    memref.alloc()
    # CHECK:            scf.for
    # CHECK:              amdgpu.lds_barrier
    # CHECK-COUNT-4:      amdgpu.gather_to_lds {{.*}}
    # CHECK-NOT:          vector.load
    # CHECK-NOT:          vector.store
    # CHECK:              rocdl.s.waitcnt
    # CHECK:              amdgpu.lds_barrier
    # CHECK-COUNT-8:      vector.load
    # CHECK-COUNT-2:      amdgpu.scaled_mfma
    # CHECK-COUNT-4:    vector.store
