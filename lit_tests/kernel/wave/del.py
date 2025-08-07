# RUN: python %s | FileCheck %s

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    run_test,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


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
        schedule=enable_scheduling,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        use_stride_cache_swizzle=True,
        dynamic_symbols=dynamic_symbols,
    )
    from wave_lang.kernel.wave.utils.run_utils import (
        set_default_run_config,
    )

    options = set_default_run_config(options)
    options.target = "gfx950"
    batched_gemm = wave_compile(options, batched_gemm)
    print(batched_gemm.asm)

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
    # CHECK-DAG:        %[[C2147483646_I32:.*]] = arith.constant 2147483646 : i32
    # CHECK-DAG:        %[[C2147483646:.*]] = arith.constant 2147483646 : index
    # CHECK-DAG:        %[[C_NEG_8192_I14:.*]] = arith.constant -8192 : i14
    # CHECK-DAG:        %[[BLOCK_ID_X:.*]] = gpu.block_id  x
    # CHECK-DAG:        %[[BLOCK_ID_Z:.*]] = gpu.block_id  z
    # CHECK-DAG:        %[[THREAD_ID_X:.*]] = gpu.thread_id  x upper_bound 256
    # CHECK-DAG:        %[[THREAD_ID_Y:.*]] = gpu.thread_id  y upper_bound 2
    # CHECK:            %[[SPAN0:.*]] = stream.binding.subspan %arg0[%[[C0]]] : !stream.binding -> memref<?x?x8192xi8, strided<[?, 8192, 1], offset: ?>>{%arg5, %arg6}
    # CHECK:            %[[VIEW6:.*]] = memref.view
    # CHECK:            %[[AFFINE_APPLY1:.*]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK:            %[[AFFINE_APPLY2:.*]] = affine.apply #[[MAP3]]()[%arg6]
    # CHECK:            %[[MUL1:.*]] = arith.muli %[[BLOCK_ID_Z]], %[[AFFINE_APPLY2]] overflow<nsw> : index
    # CHECK:            %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[SPAN0]] to offset: [%[[C0]]], sizes: [%[[C2147483646]]], strides: [1] : memref<?x?x8192xi8, strided<[?, 8192, 1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
    # CHECK:            %[[BUFF_CAST:.*]] = amdgpu.fat_raw_buffer_cast %[[REINTERPRET_CAST]] validBytes(%[[C2147483646_I32]]) cacheSwizzleStride(%[[C_NEG_8192_I14]]) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
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

if __name__ == "__main__":
    test_mxfp4_scaled_mma_unaligned_16x16x128()