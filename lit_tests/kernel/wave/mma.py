# RUN: python %s | FileCheck %s

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_mma():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    compile_options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 16,
            BLOCK_M: 32,
            BLOCK_N: 32,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    mma = wave_compile(compile_options, mma)
    print(mma.asm)

    # CHECK-LABEL: test_mma
    # CHECK-DAG:        #[[MAP0:.*]] = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #[[MAP1:.*]] = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:        #[[MAP2:.*]] = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:        #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @mma
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK-DAG:        %[[workgroup_id_0:.*]] = gpu.block_id x
    # CHECK-DAG:        %[[thread_id_x:.*]] = gpu.thread_id  x

    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref<2560xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1280:.+]] = arith.constant 1280 : index
    # CHECK-DAG:        %[[ALLOC:.+]] = memref.view %[[BASE_ALLOC]][%[[C0]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x20xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.view %[[BASE_ALLOC]][%[[C1280]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x20xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]][{{.*}}, {{.*}}] : memref<32x20xf16,
    # CHECK-DAG:        %[[D20:.+]] = vector.load %[[ALLOC_0]][{{.*}}, {{.*}}] : memref<32x20xf16,
    # CHECK:            %[[D21:.+]] = amdgpu.mfma 16x16x16 %[[D20]] * %[[D12]] + %[[CST]] blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>

    # CHECK:            %[[OFF0:.*]] = affine.apply #[[MAP0]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[D22]], {{.*}}[%[[OFF0]], {{.*}}]  : memref<64x128xf32, strided<[128, 1]>>, vector<1xf32>
    # CHECK:            %[[D26:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [1], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>
    # CHECK:            %[[OFF1:.*]] = affine.apply #[[MAP1]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[D26]], {{.*}}[%[[OFF1]], {{.*}}] : memref<64x128xf32, strided<[128, 1]>>, vector<1xf32>
    # CHECK:            %[[D28:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [2], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>
    # CHECK:            %[[OFF2:.*]] = affine.apply #[[MAP2]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[D28]], {{.*}}[%[[OFF2]], {{.*}}] : memref<64x128xf32, strided<[128, 1]>>, vector<1xf32>
    # CHECK:            %[[D30:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [3], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>
    # CHECK:            %[[OFF3:.*]] = affine.apply #[[MAP3]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[D30]], {{.*}}[%[[OFF3]], {{.*}}] : memref<64x128xf32, strided<[128, 1]>>, vector<1xf32>


@run_test
def test_mma_32x32x8():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_32x32x8_F16,
        )
    ]

    @tkw.wave(constraints)
    def mma_32x32x8(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    compile_options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K: 8,
            BLOCK_M: 64,
            BLOCK_N: 64,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    mma_32x32x8 = wave_compile(compile_options, mma_32x32x8)
    print(mma_32x32x8.asm)

    # CHECK-LABEL:    test_mma_32x32x8
    # CHECK-DAG:        #[[MAP0:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4)>
    # CHECK-DAG:        #[[MAP1:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 1)>
    # CHECK-DAG:        #[[MAP2:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 2)>
    # CHECK-DAG:        #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 3)>
    # CHECK-DAG:        #[[MAP4:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 8)>
    # CHECK-DAG:        #[[MAP5:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 9)>
    # CHECK-DAG:        #[[MAP6:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 10)>
    # CHECK-DAG:        #[[MAP7:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 11)>
    # CHECK-DAG:        #[[MAP8:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 16)>
    # CHECK-DAG:        #[[MAP9:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 17)>
    # CHECK-DAG:        #[[MAP10:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 18)>
    # CHECK-DAG:        #[[MAP11:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 19)>
    # CHECK-DAG:        #[[MAP12:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 24)>
    # CHECK-DAG:        #[[MAP13:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 25)>
    # CHECK-DAG:        #[[MAP14:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 26)>
    # CHECK-DAG:        #[[MAP15:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 27)>
    # CHECK:          func.func @mma_32x32x8
    # CHECK-DAG:        %[[workgroup_id_0:.*]] = gpu.block_id x
    # CHECK-DAG:        %[[thread_id_x:.*]] = gpu.thread_id  x

    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref<3072xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1536:.+]] = arith.constant 1536 : index
    # CHECK-DAG:        %[[SUBSPAN:.*]] = stream.binding.subspan %arg2[%[[C0]]] : !stream.binding -> memref<f32>
    # CHECK-DAG:        %[[DST:.*]] = memref.reinterpret_cast %[[SUBSPAN]] to offset: [0], sizes: [128, 128], strides: [128, 1] : memref<f32> to memref<128x128xf32, strided<[128, 1]>>
    # CHECK-DAG:        %[[ALLOC:.+]] = memref.view %[[BASE_ALLOC]][%[[C0]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<64x12xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.view %[[BASE_ALLOC]][%[[C1536]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<64x12xf16, #gpu.address_space<workgroup>>
    # CHECK:            %[[D12:.+]] = vector.load %[[ALLOC]]{{.*}} : memref<64x12xf16,
    # CHECK:            %[[D20:.+]] = vector.load %[[ALLOC_0]]{{.*}} : memref<64x12xf16,
    # CHECK:            %[[D21:.+]] = amdgpu.mfma 32x32x8 %[[D20]] * %[[D12]] + %[[CST]] blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
    # CHECK:            %[[VAL0:.*]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :

    # CHECK:            %[[OFF0:.*]] = affine.apply #[[MAP0]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL0]], %[[DST]][%[[OFF0]], %{{.*}}]
    # CHECK:            %[[VAL1:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [1], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF1:.*]] = affine.apply #[[MAP1]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL1]], %[[DST]][%[[OFF1]], %{{.*}}]
    # CHECK:            %[[VAL2:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [2], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF2:.*]] = affine.apply #[[MAP2]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL2]], %[[DST]][%[[OFF2]], %{{.*}}]
    # CHECK:            %[[VAL3:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [3], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF3:.*]] = affine.apply #[[MAP3]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL3]], %[[DST]][%[[OFF3]], %{{.*}}]
    # CHECK:            %[[VAL4:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [4], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF4:.*]] = affine.apply #[[MAP4]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL4]], %[[DST]][%[[OFF4]], %{{.*}}]
    # CHECK:            %[[VAL5:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [5], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF5:.*]] = affine.apply #[[MAP5]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL5]], %[[DST]][%[[OFF5]], %{{.*}}]
    # CHECK:            %[[VAL6:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [6], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF6:.*]] = affine.apply #[[MAP6]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL6]], %[[DST]][%[[OFF6]], %{{.*}}]
    # CHECK:            %[[VAL7:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [7], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF7:.*]] = affine.apply #[[MAP7]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL7]], %[[DST]][%[[OFF7]], %{{.*}}]
    # CHECK:            %[[VAL8:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [8], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF8:.*]] = affine.apply #[[MAP8]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL8]], %[[DST]][%[[OFF8]], %{{.*}}]
    # CHECK:            %[[VAL9:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [9], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF9:.*]] = affine.apply #[[MAP9]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL9]], %[[DST]][%[[OFF9]], %{{.*}}]
    # CHECK:            %[[VAL10:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [10], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF10:.*]] = affine.apply #[[MAP10]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL10]], %[[DST]][%[[OFF10]], %{{.*}}]
    # CHECK:            %[[VAL11:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [11], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF11:.*]] = affine.apply #[[MAP11]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL11]], %[[DST]][%[[OFF11]], %{{.*}}]
    # CHECK:            %[[VAL12:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [12], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF12:.*]] = affine.apply #[[MAP12]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL12]], %[[DST]][%[[OFF12]], %{{.*}}]
    # CHECK:            %[[VAL13:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [13], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF13:.*]] = affine.apply #[[MAP13]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL13]], %[[DST]][%[[OFF13]], %{{.*}}]
    # CHECK:            %[[VAL14:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [14], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF14:.*]] = affine.apply #[[MAP14]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL14]], %[[DST]][%[[OFF14]], %{{.*}}]
    # CHECK:            %[[VAL15:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [15], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF15:.*]] = affine.apply #[[MAP15]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL15]], %[[DST]][%[[OFF15]], %{{.*}}]


@run_test
def test_mma_32x32x16():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_32x32x16_F8,
        )
    ]

    @tkw.wave(constraints)
    def mma_32x32x16(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    compile_options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K: 16,
            BLOCK_M: 64,
            BLOCK_N: 64,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    mma_32x32x16 = wave_compile(compile_options, mma_32x32x16)
    print(mma_32x32x16.asm)

    # CHECK-LABEL:    test_mma_32x32x16
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 1)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 2)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 3)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 8)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 9)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 10)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 11)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 17)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 18)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 19)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 24)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 25)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 26)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 27)>
    # CHECK:          func.func @mma_32x32x16

    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref<3072xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1536:.+]] = arith.constant 1536 : index
    # CHECK-DAG:        %[[ALLOC:.+]] = memref.view %[[BASE_ALLOC]][%[[C0]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<64x24xf8E4M3FNUZ, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.view %[[BASE_ALLOC]][%[[C1536]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<64x24xf8E4M3FNUZ, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]]{{.*}} : memref<64x24xf8E4M3FNUZ,
    # CHECK:            %[[D20:.+]] = vector.load %[[ALLOC_0]]{{.*}} : memref<64x24xf8E4M3FNUZ,
    # CHECK:            %[[D21:.+]] = amdgpu.mfma 32x32x16 %[[D20]] * %[[D12]] + %[[CST]] blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>
    # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<16xf32> to vector<1xf32>

    # CHECK-COUNT-16:   vector.store


@run_test
def test_mma_16x16x32():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x32_F8,
        )
    ]

    @tkw.wave(constraints)
    def mma_16x16x32(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    compile_options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K: 32,
            BLOCK_M: 32,
            BLOCK_N: 32,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    mma_16x16x32 = wave_compile(compile_options, mma_16x16x32)
    print(mma_16x16x32.asm)

    # CHECK-LABEL:    test_mma_16x16x32
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @mma_16x16x32
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>

    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref<2560xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1280:.+]] = arith.constant 1280 : index
    # CHECK-DAG:        %[[ALLOC:.+]] = memref.view %[[BASE_ALLOC]][%[[C0]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x40xf8E4M3FNUZ, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.view %[[BASE_ALLOC]][%[[C1280]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x40xf8E4M3FNUZ, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]][{{.*}}, {{.*}}] : memref<32x40xf8E4M3FNUZ,
    # CHECK-DAG:        %[[D20:.+]] = vector.load %[[ALLOC_0]][{{.*}}, {{.*}}] : memref<32x40xf8E4M3FNUZ,
    # CHECK:            %[[D21:.+]] = amdgpu.mfma 16x16x32 %[[D20]] * %[[D12]] + %[[CST]] blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>
    # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>

    # CHECK-COUNT-4:    vector.store


@run_test
def test_mma_with_linear_shared_access():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def mma_with_linear_shared_access(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    compile_options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 16,
            BLOCK_M: 32,
            BLOCK_N: 32,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        linearize_shared_access=True,
        canonicalize=True,
        compile_to_mlir=True,
    )
    mma_with_linear_shared_access = wave_compile(
        compile_options, mma_with_linear_shared_access
    )
    print(mma_with_linear_shared_access.asm)

    # CHECK-LABEL: test_mma_with_linear_shared_access
    # CHECK:          func.func @mma_with_linear_shared_access
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref<2560xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1280:.+]] = arith.constant 1280 : index
    # CHECK-DAG:        %[[ALLOC:.+]] = memref.view %[[BASE_ALLOC]][%[[C0]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x20xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.view %[[BASE_ALLOC]][%[[C1280]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x20xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[LINEAR_ALLOC_0:.+]] = memref.reinterpret_cast %[[ALLOC_0]] to offset: [0], sizes: [640], strides: [1] : memref<32x20xf16, #gpu.address_space<workgroup>> to memref<640xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[LINEAR_ALLOC:.+]] = memref.reinterpret_cast %[[ALLOC]] to offset: [0], sizes: [640], strides: [1] : memref<32x20xf16, #gpu.address_space<workgroup>> to memref<640xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[D12:.+]] = vector.load %[[LINEAR_ALLOC]][{{.*}}] : memref<640xf16,
    # CHECK-DAG:        %[[D20:.+]] = vector.load %[[LINEAR_ALLOC_0]][{{.*}}] : memref<640xf16,
    # CHECK:            %[[D21:.+]] = amdgpu.mfma 16x16x16 %[[D20]] * %[[D12]] + %[[CST]] blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>


@run_test
def test_wmma_f32_16x16x16_f16_w32():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=32, mma_type=tkw.MMAType.RDNA4_WAVE32_F32_16x16x16_F16
        )
    ]

    @tkw.wave(constraints)
    def mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    compile_options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 8,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        target="gfx1201",
    )
    mma = wave_compile(compile_options, mma)
    print(mma.asm)

    # CHECK-LABEL: test_wmma_f32_16x16x16_f16_w32
    # CHECK:          func.func @mma
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<8xf32>

    # CHECK-DAG:        %[[BID_X:.+]] = gpu.block_id  x upper_bound 4
    # CHECK-DAG:        %[[BID_Y:.+]] = gpu.block_id  y upper_bound 8
    # CHECK-DAG:        %[[TID_X:.+]] = gpu.thread_id  x upper_bound 32

    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref
    # CHECK-DAG:        %[[VIEW0:.+]] = memref.view %[[BASE_ALLOC]]
    # CHECK-DAG:        %[[VIEW1:.+]] = memref.view %[[BASE_ALLOC]]

    # CHECK-DAG:        %[[V1:.+]] = affine.apply #map()[%[[TID_X]], %[[BID_X]]]
    # CHECK-DAG:        %[[V2:.+]] = affine.apply #map1()[%[[TID_X]]]
    # CHECK-DAG:        %[[R1:.+]] = vector.load %{{.*}}[%[[V1]], %[[V2]]] {{.*}} vector<8xf16>

    # CHECK-DAG:        %[[V4:.+]] = affine.apply #map2()[%[[TID_X]]]
    # CHECK-DAG:        vector.store %[[R1]], %[[VIEW1]][%[[V4]], %[[V2]]] {{.*}} vector<8xf16>

    # CHECK-DAG:        %[[V6:.+]] = affine.apply #map3()[%[[TID_X]], %[[BID_Y]]]
    # CHECK-DAG:        %[[R2:.+]] = vector.load %{{.*}}[%[[V6]], %[[V2]]] {{.*}} vector<8xf16>

    # CHECK-DAG:        %[[V8:.+]] = affine.apply #map4()[%[[TID_X]]]
    # CHECK-DAG:        vector.store %[[R2]], %[[VIEW0]][%[[V8]], %[[V2]]] {{.*}} vector<8xf16>

    # CHECK-DAG:        rocdl.s.wait.dscnt
    # CHECK-DAG:        rocdl.s.barrier.signal
    # CHECK-DAG:        rocdl.s.barrier.wait

    # CHECK-DAG:        %[[R3:.+]] = vector.load %[[VIEW0]][%[[V8]], %[[V2]]] {{.*}} vector<8xf16>
    # CHECK-DAG:        %[[R4:.+]] = vector.load %[[VIEW1]][%[[V4]], %[[V2]]] {{.*}} vector<8xf16>

    # CHECK-DAG:        %[[V11:.+]] = amdgpu.wmma 16x16x16 %[[R4]] * %[[R3]] + %[[CST]] : vector<8xf16>, vector<8xf16>, vector<8xf32>

    # CHECK-DAG:        %[[E1:.+]] = vector.extract_strided_slice %[[V11]] {offsets = [0], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>

    # CHECK-DAG:        %[[V14:.*]] = affine.apply #map5()[%[[TID_X]], %[[BID_X]]]
    # CHECK-DAG:        vector.store %[[E1]], %{{.*}}[%[[V14]], %[[V6]]] {{.*}} vector<1xf32>

    # CHECK-DAG:        %[[E2:.+]] = vector.extract_strided_slice %[[V11]] {offsets = [1], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
    # CHECK-DAG:        %[[V16:.*]] = affine.apply #map6()[%[[TID_X]], %[[BID_X]]]
    # CHECK-DAG:        vector.store %[[E2]], {{.*}}[%[[V16]], %[[V6]]] {{.*}} vector<1xf32>

    # CHECK-DAG:        %[[E3:.+]] = vector.extract_strided_slice %[[V11]] {offsets = [2], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
    # CHECK-DAG:        %[[V18:.*]] = affine.apply #map7()[%[[TID_X]], %[[BID_X]]]
    # CHECK-DAG:        vector.store %[[E3]], %{{.*}}[%[[V18]], %[[V6]]] {{.*}} vector<1xf32>

    # CHECK-DAG:        %[[E4:.+]] = vector.extract_strided_slice %[[V11]] {offsets = [3], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
    # CHECK-DAG:        %[[V20:.*]] = affine.apply #map8()[%[[TID_X]], %[[BID_X]]]
    # CHECK-DAG:        vector.store %[[E4]], %{{.*}}[%[[V20]], %[[V6]]] {{.*}} vector<1xf32>

    # CHECK-DAG:        %[[E5:.+]] = vector.extract_strided_slice %[[V11]] {offsets = [4], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
    # CHECK-DAG:        %[[V22:.*]] = affine.apply #map9()[%[[TID_X]], %[[BID_X]]]
    # CHECK-DAG:        vector.store %[[E5]], %{{.*}}[%[[V22]], %[[V6]]] {{.*}} vector<1xf32>

    # CHECK-DAG:        %[[E6:.+]] = vector.extract_strided_slice %[[V11]] {offsets = [5], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
    # CHECK-DAG:        %[[V24:.*]] = affine.apply #map10()[%[[TID_X]], %[[BID_X]]]
    # CHECK-DAG:        vector.store %[[E6]], %{{.*}}[%[[V24]], %[[V6]]] {{.*}} vector<1xf32>

    # CHECK-DAG:        %[[E7:.+]] = vector.extract_strided_slice %[[V11]] {offsets = [6], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
    # CHECK-DAG:        %[[V26:.*]] = affine.apply #map11()[%[[TID_X]], %[[BID_X]]]
    # CHECK-DAG:        vector.store %[[E7]], %{{.*}}[%[[V26]], %[[V6]]] {{.*}} vector<1xf32>

    # CHECK-DAG:        %[[E8:.+]] = vector.extract_strided_slice %[[V11]] {offsets = [7], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
    # CHECK-DAG:        %[[V28:.*]] = affine.apply #map12()[%[[TID_X]], %[[BID_X]]]
    # CHECK-DAG:        vector.store %[[E8]], %{{.*}}[%[[V28]], %[[V6]]] {{.*}} vector<1xf32>

    # CHECK:            return


@run_test
def test_wmma_f32_16x16x32_f16():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=32, mma_type=tkw.MMAType.GFX1250_F32_16x16x32_F16
        )
    ]

    @tkw.wave(constraints)
    def mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a)
        b_reg = tkw.read(b)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c)

    compile_options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 32,
            BLOCK_M: 16,
            BLOCK_N: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        target="gfx1250",
    )
    mma = wave_compile(compile_options, mma)
    print(mma.asm)

    # CHECK-LABEL: test_wmma_f32_16x16x32_f16
    # CHECK:          func.func @mma

    # CHECK:            rocdl.wmma.f32.16x16x32.f16 {{.*}} : (i1, vector<16xf16>, i1, vector<16xf16>, i16, vector<8xf32>, i1, i1) -> vector<8xf32>


@run_test
def test_wmma_with_tensor_load():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=32, mma_type=tkw.MMAType.GFX1250_F32_16x16x32_F16
        )
    ]

    @tkw.wave(constraints)
    def mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a)
        b_reg = tkw.read(b)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c)

    compile_options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 32,
            BLOCK_M: 16,
            BLOCK_N: 16,
            BLOCK_K: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        target="gfx1250",
        use_global_to_shared=True,
    )
    mma = wave_compile(compile_options, mma)
    print(mma.asm)

    # CHECK-LABEL: test_wmma_with_tensor_load
    # CHECK:          func.func @mma

    ### global buffer is bound to %0, %1 and %2 : MK, NK, MN
    # CHECK:        %[[SUBSPAN0:.*]] = stream.binding.subspan
    # CHECK:        %[[SUBSPAN1:.*]] = stream.binding.subspan
    # CHECK:        %[[SUBSPAN2:.*]] = stream.binding.subspan

    # CHECK:        %[[CAST_0:.*]] = memref.reinterpret_cast %[[SUBSPAN0]]
    # CHECK:        %[[CAST_1:.*]] = memref.reinterpret_cast %[[SUBSPAN1]]
    # CHECK:        %[[CAST_2:.*]] = memref.reinterpret_cast %[[SUBSPAN2]]

    ### shared memory alloc
    #   Make sure the shared memory allocation is padded
    # CHECK:        %[[SMEM:.*]] = memref.alloc() : memref<2304xi8, #gpu.address_space<workgroup>>
    # CHECK:        %[[VIEW0:.*]] = memref.view %[[SMEM]][{{.*}}] : memref<2304xi8, #gpu.address_space<workgroup>> to memref<16x36xf16, #gpu.address_space<workgroup>>
    # CHECK:        %[[VIEW1:.*]] = memref.view %[[SMEM]][{{.*}}] : memref<2304xi8, #gpu.address_space<workgroup>> to memref<16x36xf16, #gpu.address_space<workgroup>>

    ### get global buffer pointer
    # CHECK:        %[[INT_PTR_0:.+]] = memref.extract_aligned_pointer_as_index

    ### get shared buffer pointer
    # CHECK:        %[[CAST_3:.*]] = memref.reinterpret_cast %[[VIEW1]]
    # CHECK:        %[[INT_PTR_1:.+]] = memref.extract_aligned_pointer_as_index %[[CAST_3]]

    ### pack descriptors and invoke tensor load
    # CHECK:        %[[D0:.*]] = vector.from_elements
    # CHECK:        %[[TENSOR_DESC_0:.*]] = vector.from_elements
    # CHECK:        llvm.call_intrinsic "llvm.amdgcn.tensor.load.to.lds"(%[[D0]], %[[TENSOR_DESC_0]], {{.*}} : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, i32) -> ()
    # CHECK-NOT:    llvm.call_intrinsic "llvm.amdgcn.s.wait.tensorcnt"
    # CHECK-NOT:    amdgpu.lds_barrier

    ### get shared buffer pointer
    # CHECK:        %[[CAST_4:.*]] = memref.reinterpret_cast %[[VIEW0]]
    # CHECK:        %[[INT_PTR_2:.+]] = memref.extract_aligned_pointer_as_index %[[CAST_4]]
    # CHECK:        %[[INT_PTR_2_CAST:.+]] = arith.index_cast %[[INT_PTR_2]] : index to i32

    ### pack descriptors and invoke tensor load
    # CHECK:        %[[D1:.*]] = vector.from_elements %{{.*}}, %[[INT_PTR_2_CAST]], %{{.*}}, %{{.*}} : vector<4xi32>
    # CHECK:        %[[TENSOR_DESC_1:.*]] = vector.from_elements

    ### resource provider
    # CHECK:        llvm.call_intrinsic "llvm.amdgcn.tensor.load.to.lds"(%[[D1]], %[[TENSOR_DESC_1]], {{.*}} : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, i32) -> ()
    # CHECK:        llvm.call_intrinsic "llvm.amdgcn.s.wait.tensorcnt"
    # CHECK:        rocdl.s.wait.dscnt 0
    # CHECK:        rocdl.s.barrier.signal -1

    ### resource consumer
    # CHECK-NEXT:   rocdl.s.barrier.wait -1
    # CHECK-NEXT:   affine.apply
    # CHECK-NEXT:   affine.apply
    # CHECK-NEXT:   vector.load %[[VIEW0]]
    # CHECK-NEXT:   affine.apply
    # CHECK-NEXT:   vector.load %[[VIEW1]]

    ### wmma
    # CHECK:        rocdl.wmma.f32.16x16x32.f16 {{.*}} : (i1, vector<16xf16>, i1, vector<16xf16>, i16, vector<8xf32>, i1, i1) -> vector<8xf32>
