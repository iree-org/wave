# RUN: python %s | FileCheck %s

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import (
    ScaledMMAType,
)
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.templates.test_kernels import (
    get_broadcasted_scale_gemm_mxfp4,
)
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    run_test,
)

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


@run_test
def test_scaled_gemm_mxfp4():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

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

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 128,
        M: 1024,
        N: 1024,
        K: 1024,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.NONE,
        device="hip",
        target="gfx950",
        compile_to_mlir=True,
    )

    scaled_gemm = wave_compile(options, scaled_gemm)
    print(scaled_gemm.asm)

    # CHECK-LABEL: test_scaled_gemm_mxfp4
    # CHECK-DAG:    #map = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #map1 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #map2 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #map3 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #map4 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #map5 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #map6 = affine_map<()[s0, s1] -> (s0 * 64 + ((s1 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #map7 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #map8 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:    #map9 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:    #map10 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:    #map11 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @scaled_gemm
    # CHECK-COUNT-1:    memref.alloc()
    # CHECK:            scf.for
    # CHECK:              vector.load
    # CHECK:              amdgpu.lds_barrier
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              amdgpu.lds_barrier
    # CHECK-COUNT-4:      vector.load
    # CHECK:              amdgpu.scaled_mfma
    # CHECK-COUNT-4:    vector.store


@run_test
def test_scaled_gemm_mxfp8():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def scaled_gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f8e5m2],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f8e5m2],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_scale_reg = tkw.read(a_scale)
            b_reg = tkw.read(b)
            b_scale_reg = tkw.read(b_scale)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 128,
        M: 1024,
        N: 1024,
        K: 1024,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.NONE,
        device="hip",
        target="gfx950",
        compile_to_mlir=True,
    )

    scaled_gemm = wave_compile(options, scaled_gemm)
    print(scaled_gemm.asm)

    # CHECK-LABEL: test_scaled_gemm_mxfp8
    # CHECK-DAG:    #map = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #map1 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #map2 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #map3 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
    # CHECK-DAG:    #map4 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #map5 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #map6 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #map7 = affine_map<()[s0, s1] -> (s0 * 128 + ((s1 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #map8 = affine_map<()[s0, s1] -> (s0 * 128 + ((s1 mod 64) floordiv 16) * 16 + 64)>
    # CHECK-DAG:    #map9 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #map10 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:    #map11 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:    #map12 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:    #map13 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @scaled_gemm
    # CHECK-COUNT-1:    memref.alloc()
    # CHECK:            scf.for
    # CHECK:              vector.load
    # CHECK:              amdgpu.lds_barrier
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              amdgpu.lds_barrier
    # CHECK:              memref.load
    # CHECK-COUNT-2:      vector.load
    # CHECK:              memref.load
    # CHECK-COUNT-2:      vector.load
    # CHECK:              amdgpu.scaled_mfma
    # CHECK-COUNT-4:    vector.store


@run_test
def packed_mxfp4_test():
    mfma_variant = tkw.ScaledMMAType.F32_16x16x128_F8F6F4
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(4, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def gemm_mxfp4_prefetch(
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

    shape = (1024, 1024, 1024)
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 256,
        BLOCK_N: 256,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 8,
        SHUFFLE_UNITS: 8,
    }
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        compile_to_mlir=True,
    )
    gemm_mxfp4_prefetch = wave_compile(options, gemm_mxfp4_prefetch)
    print(gemm_mxfp4_prefetch.asm)

    # CHECK-LABEL:    gemm_mxfp4_prefetch

    # Prologue Global Read
    # CHECK-COUNT-4:  vector.load {{.*}} : memref<1024x512xi8, strided<[512, 1], offset: ?>>, vector<16xi8>
    # CHECK:          vector.load {{.*}} : memref<1024x32xi8, strided<[32, 1], offset: ?>>, vector<4xi8>
    # CHECK-COUNT-4:  vector.load {{.*}} : memref<1024x512xi8, strided<[512, 1], offset: ?>>, vector<16xi8>
    # CHECK:          vector.load {{.*}} : memref<1024x32xi8, strided<[32, 1], offset: ?>>, vector<4xi8>

    # Prologue Local Write
    # CHECK-COUNT-4:  vector.store {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK:          vector.store {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK-COUNT-4:  vector.store {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK:          vector.store {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>

    # Steady State
    # CHECK:          scf.for

    # Steady State global_load_rhs_scale
    # CHECK:            vector.load %{{.*}} : memref<1024x32xi8, strided<[32, 1], offset: ?>>, vector<4xi8>
    # Steady State local_load_rhs_scale
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>

    # Steady State global_load_lhs_scale
    # CHECK:            vector.load %{{.*}} : memref<1024x32xi8, strided<[32, 1], offset: ?>>, vector<4xi8>
    # Steady State local_load_lhs_scale
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>

    # Steady State global_load_rhs
    # CHECK-COUNT-4:    vector.load %{{.*}} : memref<1024x512xi8, strided<[512, 1], offset: ?>>, vector<16xi8>
    # Steady State local_load_rhs
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>

    # Steady State global_load_lhs
    # CHECK-COUNT-4:    vector.load %{{.*}} : memref<1024x512xi8, strided<[512, 1], offset: ?>>, vector<16xi8>
    # Steady State local_load_lhs
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>

    # Steady State MFMA
    # CHECK-COUNT-64:   amdgpu.scaled_mfma

    # Steady State Local Write (lhs_scale, rhs_scale, lhs, rhs)
    # CHECK-COUNT-2:    vector.store {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK-COUNT-8:    vector.store {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK:            scf.yield
    # CHECK:          }

    # Epilogue Local Read
    # CHECK-COUNT-16: vector.load {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
    # CHECK-COUNT-16: vector.load {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-COUNT-8:  vector.load {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
    # CHECK-COUNT-8:  vector.load {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>

    # Epilogue MFMA
    # CHECK-COUNT-64: amdgpu.scaled_mfma


@run_test
def packed_mxfp4_global_to_lds_test():
    mfma_variant = tkw.ScaledMMAType.F32_16x16x128_F8F6F4
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(4, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def gemm_mxfp4_global_to_lds_prefetch(
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

    shape = (1024, 1024, 1024)
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 256,
        BLOCK_N: 256,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        GLOBAL_TO_SHARED_DELAY: 2,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 8,
        SHUFFLE_UNITS: 8,
    }
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        compile_to_mlir=True,
        use_global_to_shared=True,
    )
    gemm_mxfp4_global_to_lds_prefetch = wave_compile(
        options, gemm_mxfp4_global_to_lds_prefetch
    )
    print(gemm_mxfp4_global_to_lds_prefetch.asm)

    # This test is important to protect efficiency and correctness of gather_to_lds version of MXFP4 prefetch GEMM.
    # The thing to look out for in this LIT test is we want to ensure the instruction ordering within the steady state
    # is correct, to protect from unexpected changes in schedule/instruction reordering.

    # CHECK-LABEL:    packed_mxfp4_global_to_lds_test

    # Prologue global load to lds (global read + local write)
    # CHECK-COUNT-34: amdgpu.gather_to_lds

    # Steady State
    # CHECK:          scf.for
    # CHECK:            rocdl.s.waitcnt
    # CHECK:            amdgpu.lds_barrier

    # Steady state local loads
    # CHECK-COUNT-48:   vector.load{{.*}} memref<{{.*}}, #gpu.address_space<workgroup>>

    # Steady State global load to lds
    # CHECK-COUNT-34:   amdgpu.gather_to_lds

    # Steady State unpack to mxfp4
    # CHECK-COUNT-48:   vector.bitcast

    # Steady state mfma
    # CHECK-COUNT-64:   amdgpu.scaled_mfma
    # CHECK:            scf.yield


@run_test
def batched_prefetch_mxfp4_test():
    mfma_variant = tkw.ScaledMMAType.F32_16x16x128_F8F6F4
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, vector_shapes={B: 0}, mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def batched_gemm_mxfp4_prefetch(
        a: tkl.Memory[B, M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[B, M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
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

    shape = (8, 1024, 1024, 1024)
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 256,
        BLOCK_N: 256,
        BLOCK_K: 256,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K: shape[3],
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 8,
        SHUFFLE_UNITS: 8,
    }
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        use_buffer_ops=True,
        linearize_shared_access=True,
        compile_to_mlir=True,
    )
    batched_gemm_mxfp4_prefetch = wave_compile(options, batched_gemm_mxfp4_prefetch)
    print(batched_gemm_mxfp4_prefetch.asm)

    # CHECK-LABEL:    batched_gemm_mxfp4_prefetch

    # CHECK-DAG:      %[[C32_I14:.+]] = arith.constant 32 : i14
    # CHECK-DAG:      %[[C512_I14:.+]] = arith.constant 512 : i14

    # Prologue Global Read
    # CHECK:          memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [%{{.*}}], strides: [1] : memref<i8> to memref<?xi8, strided<[1], offset: ?>>
    # CHECK:          amdgpu.fat_raw_buffer_cast %{{.*}} validBytes(%{{.*}}) cacheSwizzleStride(%[[C512_I14]]) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
    # CHECK-COUNT-4:  vector.load {{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    # CHECK:          memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [%{{.*}}], strides: [1] : memref<i8> to memref<?xi8, strided<[1], offset: ?>>
    # CHECK:          amdgpu.fat_raw_buffer_cast %{{.*}} validBytes(%{{.*}}) cacheSwizzleStride(%[[C32_I14]]) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
    # CHECK:          vector.load {{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    # CHECK-COUNT-4:  vector.load {{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    # CHECK:          vector.load {{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>

    # Prologue Linearize shared memory + Local Write
    # CHECK:          memref.reinterpret_cast {{.*}} to offset: [0], sizes: [34816], strides: [1] : memref<1x256x136xi8, #gpu.address_space<workgroup>> to memref<34816xi8, #gpu.address_space<workgroup>>
    # CHECK-COUNT-4:  vector.store {{.*}} : memref<34816xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK:          memref.reinterpret_cast {{.*}} to offset: [0], sizes: [4096], strides: [1] : memref<1x256x16xi8, #gpu.address_space<workgroup>> to memref<4096xi8, #gpu.address_space<workgroup>>
    # CHECK:          vector.store {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK:          memref.reinterpret_cast {{.*}} to offset: [0], sizes: [34816], strides: [1] : memref<256x136xi8, #gpu.address_space<workgroup>> to memref<34816xi8, #gpu.address_space<workgroup>>
    # CHECK-COUNT-4:  vector.store {{.*}} : memref<34816xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK:          memref.reinterpret_cast {{.*}} to offset: [0], sizes: [4096], strides: [1] : memref<256x16xi8, #gpu.address_space<workgroup>> to memref<4096xi8, #gpu.address_space<workgroup>>
    # CHECK:          vector.store {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>>, vector<4xi8>

    # Steady State
    # CHECK:          scf.for

    # Steady State global_load_rhs_scale
    # CHECK:            vector.load %{{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    # Steady State local_load_rhs_scale
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<4096xi8, #gpu.address_space<workgroup>>, vector<1xi8>

    # Steady State global_load_lhs_scale
    # CHECK:            vector.load %{{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    # Steady State local_load_lhs_scale
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<4096xi8, #gpu.address_space<workgroup>>, vector<1xi8>

    # Steady State global_load_rhs
    # CHECK-COUNT-4:    vector.load %{{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    # Steady State local_load_rhs
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<34816xi8, #gpu.address_space<workgroup>>, vector<16xi8>

    # Steady State global_load_lhs
    # CHECK-COUNT-4:    vector.load %{{.*}} : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    # Steady State local_load_lhs
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<34816xi8, #gpu.address_space<workgroup>>, vector<16xi8>

    # Steady State MFMA
    # CHECK-COUNT-64:   amdgpu.scaled_mfma

    # Steady State Local Write (lhs_scale, rhs_scale, lhs, rhs)
    # CHECK:            vector.store {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK:            vector.store {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK-COUNT-4:    vector.store {{.*}} : memref<34816xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-COUNT-4:    vector.store {{.*}} : memref<34816xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK:            scf.yield
    # CHECK:          }

    # Epilogue Local Read
    # CHECK-COUNT-16: vector.load {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>>, vector<1xi8>
    # CHECK-COUNT-16: vector.load {{.*}} : memref<34816xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-COUNT-8:  vector.load {{.*}} : memref<4096xi8, #gpu.address_space<workgroup>>, vector<1xi8>
    # CHECK-COUNT-8:  vector.load {{.*}} : memref<34816xi8, #gpu.address_space<workgroup>>, vector<16xi8>

    # Epilogue MFMA
    # CHECK-COUNT-64: amdgpu.scaled_mfma


@run_test
def test_unaligned_scaled_gemm_mxfp4():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def unaligned_scaled_gemm(
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

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 128,
        M: 1024,
        N: 1024,
        K: 192,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.NONE,
        device="hip",
        target="gfx950",
        compile_to_mlir=True,
    )

    unaligned_scaled_gemm = wave_compile(options, unaligned_scaled_gemm)
    print(unaligned_scaled_gemm.asm)

    # Importance of this test is to ensure unaligned K dim works. The main
    # thing to observe is the bounds that is used to compute the mask
    # for vector.maskedload is indeed scaled.

    # We check that for logits (K: K/2) the bounds is 192/2 = 96.
    # We also ensure for scales (K: K/32) the bound is 192/32 = 6.

    # CHECK-LABEL: unaligned_scaled_gemm

    # CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0> : vector<1xi8>
    # CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0> : vector<16xi8>
    # CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
    # CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
    # CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:    %[[SCALED_SCALES_BOUND:.+]] = arith.constant 6 : index
    # CHECK-DAG:    %[[SCALED_LOGITS_BOUND:.+]] = arith.constant dense<96> : vector<16xindex>
    # CHECK:        scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]]
    # CHECK:          %[[SCALED_LOGITS_MASK:.+]] = arith.cmpi slt, %{{.*}}, %[[SCALED_LOGITS_BOUND]] : vector<16xindex>
    # CHECK:          vector.maskedload {{.*}}, %[[SCALED_LOGITS_MASK]], %[[CST_0]] : memref<1024x96xi8, strided<[96, 1], offset: ?>>, vector<16xi1>, vector<16xi8> into vector<16xi8>
    # CHECK:          %[[SCALED_SCALES_MASK_VAL:.+]] = arith.cmpi slt, %{{.*}}, %[[SCALED_SCALES_BOUND]] : index
    # CHECK:          %[[SCALED_SCALES_MASK:.+]] = vector.broadcast %[[SCALED_SCALES_MASK_VAL]] : i1 to vector<1xi1>
    # CHECK:          vector.maskedload {{.*}}, %[[SCALED_SCALES_MASK]], %[[CST]] : memref<1024x6xi8, strided<[6, 1], offset: ?>>, vector<1xi1>, vector<1xi8> into vector<1xi8>
    # CHECK:          vector.maskedload {{.*}}, %[[SCALED_LOGITS_MASK]], %[[CST_0]] : memref<1024x96xi8, strided<[96, 1], offset: ?>>, vector<16xi1>, vector<16xi8> into vector<16xi8>
    # CHECK:          vector.maskedload {{.*}}, %[[SCALED_SCALES_MASK]], %[[CST]] : memref<1024x6xi8, strided<[6, 1], offset: ?>>, vector<1xi1>, vector<1xi8> into vector<1xi8>
    # CHECK:          amdgpu.scaled_mfma
    # CHECK:          scf.yield
    # CHECK:         }


@run_test
def test_mxfp4_scaled_mma_unaligned_16x16x128():
    shape = (16384, 16384, 16384)
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4
    enable_scheduling = SchedulingType.PREFETCH
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[B, M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
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

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 256,
        BLOCK_N: 128,
        BLOCK_K: 256,
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = [B, M]
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        compile_to_mlir=True,
        schedule=enable_scheduling,
        use_buffer_ops=True,
        dynamic_symbols=dynamic_symbols,
    )
    from wave_lang.kernel.wave.utils.run_utils import (
        set_default_run_config,
    )

    options = set_default_run_config(options)
    options.target = "gfx950"
    batched_gemm = wave_compile(options, batched_gemm)
    print(batched_gemm.asm)

    # This test checks the boundary condition for unaligned shapes.

    # CHECK-LABEL:  test_mxfp4_scaled_mma_unaligned_16x16x128
    # CHECK-DAG:    #[[MAP2:.*]] = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 8) * 128)>
    # CHECK-DAG:    #[[MAP3:.*]] = affine_map<()[s0] -> (s0 * 8192)>
    # CHECK-DAG:    #[[MAP6:.*]] = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
    # CHECK-DAG:    #[[MAP17:.*]] = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
    # CHECK:        func.func @batched_gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: index, %arg6: index) attributes {translation_info = #translation} {
    # CHECK-DAG:        %[[CST1:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
    # CHECK-DAG:        %[[CST2:.*]] = arith.constant dense<2147483647> : vector<16xindex>
    # CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C8192:.*]] = arith.constant 8192 : index
    # CHECK-DAG:        %[[C2147483646_I64:.*]] = arith.constant 2147483646 : i64
    # CHECK-DAG:        %[[C2147483646:.*]] = arith.constant 2147483646 : index
    # CHECK-DAG:        %[[C_NEG_8192_I14:.*]] = arith.constant -8192 : i14
    # CHECK-DAG:        %[[BLOCK_ID_X:.*]] = gpu.block_id  x
    # CHECK-DAG:        %[[BLOCK_ID_Z:.*]] = gpu.block_id  z
    # CHECK-DAG:        %[[THREAD_ID_X:.*]] = gpu.thread_id  x upper_bound 256
    # CHECK-DAG:        %[[THREAD_ID_Y:.*]] = gpu.thread_id  y upper_bound 2
    # CHECK:            %[[AFFINE_APPLY2:.*]] = affine.apply #[[MAP3]]()[%arg6]
    # CHECK:            %[[VIEW6:.*]] = memref.view
    # CHECK:            %[[AFFINE_APPLY1:.*]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK:            %[[MUL1:.*]] = arith.muli %[[BLOCK_ID_Z]], %[[AFFINE_APPLY2]] overflow<nsw> : index
    # CHECK:            %{{.*}}, %[[OFFSET_TO_TENSOR:.+]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %{{.*}} : memref<?x?x8192xi8, strided<[?, 8192, 1], offset: ?>> -> memref<i8>, index, index, index, index, index, index, index
    # CHECK:            %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%[[OFFSET_TO_TENSOR]]], sizes: [%[[C2147483646]]], strides: [1] : memref<i8> to memref<?xi8, strided<[1], offset: ?>>
    # CHECK:            %[[BUFF_CAST:.*]] = amdgpu.fat_raw_buffer_cast %[[REINTERPRET_CAST]] validBytes(%[[C2147483646_I64]]) cacheSwizzleStride(%[[C_NEG_8192_I14]]) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
    # CHECK:            %[[AFFINE_APPLY3:.*]] = affine.apply #[[MAP6]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]], %[[BLOCK_ID_X]]]
    # CHECK:            %[[CMP1:.*]] = arith.cmpi slt, %[[AFFINE_APPLY3]], %arg6 : index
    # CHECK:            %[[BROADCAST1:.*]] = vector.broadcast %[[CMP1]] : i1 to vector<16xi1>
    # CHECK:            %[[MUL2:.*]] = arith.muli %[[AFFINE_APPLY3]], %[[C8192]] overflow<nsw> : index
    # CHECK:            %[[ADD1:.*]] = arith.addi  %[[MUL1]], %[[MUL2]] overflow<nsw> : index
    # CHECK:            %[[AFFINE_APPLY4:.*]] = affine.apply #[[MAP17]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK:            scf.for %{{.*}}
    # CHECK:                %[[ADD2:.*]] = arith.addi %[[ADD1]], %{{.*}} overflow<nsw> : index
    # CHECK:                %[[IDX_CAST1:.*]] = arith.index_cast %[[ADD2]] : index to i32
    # CHECK:                %[[BROADCAST2:.*]] = vector.broadcast %[[IDX_CAST1]] : i32 to vector<16xi32>
    # CHECK:                %[[ADD3:.*]] = arith.addi %[[BROADCAST2]], %[[CST1]] : vector<16xi32>
    # CHECK:                %[[IDX_CAST4:.*]] = arith.index_cast %[[ADD3]] : vector<16xi32> to vector<16xindex>
    # CHECK:                %[[SELECT2:.*]] = arith.select %[[BROADCAST1]], %[[IDX_CAST4]], %[[CST2]] : vector<16xi1>, vector<16xindex>
    # CHECK:                %[[EXTRACT2:.*]] = vector.extract %[[SELECT2]][0] : index from vector<16xindex>
    # CHECK:                %[[LOAD2:.*]] = vector.load %[[BUFF_CAST]][%[[EXTRACT2]]] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>


@run_test
def test_mxfp4_broadcasted_scale_scaled_mma_16x16x128():
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4
    shape = (32, 32, 256)
    broadcasted_scale_scaled_mma, hyperparams = get_broadcasted_scale_gemm_mxfp4(
        shape, mfma_variant
    )
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        device="hip",
        target="gfx950",
        compile_to_mlir=True,
    )
    broadcasted_scale_scaled_mma = wave_compile(options, broadcasted_scale_scaled_mma)
    print(broadcasted_scale_scaled_mma.asm)

    # This test is important to check that broadcasting on scaled dimension works.
    # The thing to look out for in this test is the same lhs_scale is being used on the two different mfmas.

    # CHECK-LABEL:  test_mxfp4_broadcasted_scale_scaled_mma_16x16x128
    # CHECK:   func.func @broadcasted_scale_scaled_mma(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
    # CHECK:            %[[LHS_SCALE:.+]] = vector.load {{.*}} : memref<40xi8, #gpu.address_space<workgroup>>, vector<1xi8>
    # CHECK:            %[[LHS_SCALE_BITCAST:.+]] = vector.bitcast %[[LHS_SCALE]] : vector<1xi8> to vector<1xf8E8M0FNU>
    # CHECK:            %[[LHS_SCALE_EXTRACT:.+]] = vector.extract %[[LHS_SCALE_BITCAST]][0] : f8E8M0FNU from vector<1xf8E8M0FNU>
    # CHECK-COUNT-2:    amdgpu.scaled_mfma(%[[LHS_SCALE_EXTRACT]][0] * %{{.*}}) * (%{{.*}}[0] * %{{.*}}) + %{{.*}} {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
