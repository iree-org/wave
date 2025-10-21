# RUN: python %s | FileCheck %s

import sympy

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


@run_test
def test_topk():
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N, K: K},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def topk(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        values: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        indices: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i32],
    ):
        src = tkw.read(a)
        topk_values, topk_indices = tkw.topk(src, K, N)
        tkw.write(topk_values, values, elements_per_thread=K)
        tkw.write(topk_indices, indices, elements_per_thread=K)

    compile_options = WaveCompileOptions(
        subs={
            M: 32,
            N: 64,
            K: 2,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    topk = wave_compile(compile_options, topk)
    print(topk.asm)

    # CHECK-LABEL: test_topk
    # CHECK: func.func @topk
    # CHECK-DAG: %[[THREAD_ID:.*]] = gpu.thread_id x
    # CHECK-DAG: %[[NEG_INF:.*]] = arith.constant dense<0xFC00> : vector<1xf16>
    # CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
    # CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
    # CHECK-DAG: %[[C64_I32:.*]] = arith.constant 64 : i32
    # CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index

    # CHECK-DAG: %[[INPUT:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%[[C0]]], sizes: [32, 64], strides: [64, 1] : memref<f16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
    # CHECK-DAG: %[[VALUES_OUT:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%[[C0]]], sizes: [32, 2], strides: [2, 1] : memref<f16> to memref<32x2xf16, strided<[2, 1], offset: ?>>
    # CHECK-DAG: %[[INDICES_OUT:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%[[C0]]], sizes: [32, 2], strides: [2, 1] : memref<i32> to memref<32x2xi32, strided<[2, 1], offset: ?>>

    # Check for read operation
    # CHECK: %[[LOADED:.*]] = vector.load %[[INPUT]]

    # Check for thread index broadcast for self_index
    # CHECK: %[[THREAD_IDX_I32:.*]] = arith.index_cast %[[THREAD_ID]] : index to i32
    # CHECK: %[[IDX_VEC:.*]] = vector.broadcast %[[THREAD_IDX_I32]] : i32 to vector<1xi32>

    # First topk iteration - intrawave reduction with butterfly shuffle
    # CHECK: gpu.shuffle  xor {{.*}}, %[[C1_I32]], %[[C64_I32]] : vector<1xf16>
    # CHECK: gpu.shuffle  xor {{.*}}, %[[C1_I32]], %[[C64_I32]] : vector<1xi32>
    # CHECK: arith.cmpf ogt{{.*}} : vector<1xf16>
    # CHECK: arith.select{{.*}} : vector<1xi1>, vector<1xf16>
    # CHECK: arith.select{{.*}} : vector<1xi1>, vector<1xi32>

    # More shuffle stages (log2(64) = 6 stages with strides 1,2,4,8,16,32)
    # CHECK: gpu.shuffle  xor {{.*}}, %{{.*}}, %[[C64_I32]] : vector<1xf16>
    # CHECK: arith.cmpf ogt{{.*}} : vector<1xf16>
    # CHECK: arith.select

    # Broadcast the final index to all threads
    # CHECK: gpu.shuffle  idx {{.*}}, %[[C0_I32]], %[[C64_I32]] : vector<1xi32>

    # Masking: Check for equality to found max index and replace with -inf
    # CHECK: %[[IS_MAX_IDX:.*]] = arith.cmpi eq{{.*}} : vector<1xi32>
    # CHECK: %[[MASKED:.*]] = arith.select %[[IS_MAX_IDX]], %[[NEG_INF]]{{.*}} : vector<1xi1>, vector<1xf16>

    # Second topk iteration (K=2, so we do this twice)
    # CHECK: gpu.shuffle  xor {{.*}} : vector<1xf16>
    # CHECK: arith.cmpf ogt{{.*}} : vector<1xf16>
    # CHECK: arith.select{{.*}} : vector<1xi1>, vector<1xf16>

    # Pack results into final vectors
    # CHECK: vector.from_elements{{.*}} : vector<2xf16>
    # CHECK: vector.from_elements{{.*}} : vector<2xi32>

    # Write operations for both values and indices
    # CHECK: vector.store{{.*}}, %[[VALUES_OUT]]{{.*}} : memref<32x2xf16{{.*}}>, vector<2xf16>
    # CHECK: vector.store{{.*}}, %[[INDICES_OUT]]{{.*}} : memref<32x2xi32{{.*}}>, vector<2xi32>
