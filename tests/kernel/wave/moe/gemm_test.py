import triton
import triton.language as tl
import torch
import itertools

import triton.compiler as tc

from torch.nn import functional as F
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.templates.reordered_gemm import get_reordered_matmul

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang import DataType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
    enable_scheduling_barriers,
    dump_generated_mlir,
    check_individual_kernels,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType

@triton.jit
def matmul_abt_kernel(
    C, A, B,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)         # (BM,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)         # (BN,)
    offs_k = tl.arange(0, BLOCK_K)                           # (BK,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K                          

        # A tile pointers: (BM, BK)
        A_ptrs = (A
                  + offs_m[:, None] * stride_am              
                  + (k0 + offs_k)[None, :] * stride_ak)     
        A_tile = tl.load(A_ptrs,
                         mask=(offs_m[:, None] < M) & k_mask[None, :],
                         other=0.0)

        # B tile pointers: (BN, BK)  <-- FIXED SHAPES
        B_ptrs = (B
                  + offs_n[:, None] * stride_bn            
                  + (k0 + offs_k)[None, :] * stride_bk)    
        B_tile_NK = tl.load(B_ptrs,
                            mask=(offs_n[:, None] < N) & k_mask[None, :],
                            other=0.0)                        
        B_tile = tl.trans(B_tile_NK)                          

        acc += tl.dot(A_tile, B_tile)

    C_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul_abt(A: torch.Tensor, B: torch.Tensor, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2):
    """
    C = A @ B.T
    A: (M,K)  B: (N,K)  -> C: (M,N)
    """
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    N, Kb = B.shape
    assert K == Kb, "K mismatch"
    # result in fp32 to match your Wave kernel’s accum type
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_abt_kernel[grid](
        C, A, B,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C



def testReorderedPingPongGemm(shape: tuple[int, int, int],dtype: torch.dtype, dynamic_dims: bool | tuple[bool, bool, bool],mfma_variant: MMAType):

    asm = """
        #map = affine_map<()[s0, s1] -> ((s0 * 2 + s1) mod 8)>
        #map1 = affine_map<()[s0, s1, s2] -> (((s0 * 132 + s1 * 66 + s2 - ((s0 * 2 + s1) floordiv 8) * 527) floordiv 4256) * -16 + 2)>
        #map2 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + ((s2 * 132 + s3 * 66 + s4 - ((s2 * 2 + s3) floordiv 8) * 527) floordiv 4256) * 2048 + (((s2 * 132 + s3 * 66 + s5 - ((s2 * 2 + s3) floordiv 8) * 527) mod 4256) mod s6) * 128)>
        #map3 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 8) * 64)>
        #map4 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + ((s2 * 132 + s3 * 66 + s4 - ((s2 * 2 + s3) floordiv 8) * 527) floordiv 4256) * 2048 + (((s2 * 132 + s3 * 66 + s5 - ((s2 * 2 + s3) floordiv 8) * 527) mod 4256) mod s6) * 128 + 64)>
        #map5 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256)>
        #map6 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 64)>
        #map7 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 128)>
        #map8 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 192)>
        #map9 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 128)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>
        #map11 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
        #map12 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map13 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map14 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map16 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
        #map17 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8)>
        #map18 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 16)>
        #map19 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32)>
        #map20 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 32)>
        #map21 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 64)>
        #map22 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 96)>
        #map23 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 32)>
        #map24 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 48)>
        #map25 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 - (s1 floordiv 8) * 64 + 64)>
        #map26 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256)>
        #map27 = affine_map<()[s0, s1, s2, s3, s4] -> (((s0 * 132 + s1 * 66 + s2 - ((s0 * 2 + s1) floordiv 8) * 527) floordiv 4256) * 2048 + (((s0 * 132 + s1 * 66 + s3 - ((s0 * 2 + s1) floordiv 8) * 527) mod 4256) mod s4) * 128)>
        #map28 = affine_map<()[s0, s1, s2, s3] -> ((((s0 * 66 + s1 * 132 + s2 - ((s0 + s1 * 2) floordiv 8) * 527) mod 4256) floordiv s3) * 256)>
        #map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
        #map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
        #map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
        #map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
        #map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
        #map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
        #map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
        #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
        #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
        #map45 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 32)>
        #map46 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 64)>
        #map47 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 96)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c2 = arith.constant 2 : index
            %c266 = arith.constant 266 : index
            %c1 = arith.constant 1 : index
            stream.return %c2, %c266, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
                %cst_0 = arith.constant dense<1073741823> : vector<8xindex>
                %c1024_i14 = arith.constant 1024 : i14
                %c536870911 = arith.constant 536870911 : index
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c536870910 = arith.constant 536870910 : index
                %c0_i32 = arith.constant 0 : i32
                %c15 = arith.constant 15 : index
                %c68032 = arith.constant 68032 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c1024 = arith.constant 1024 : index
                %c1 = arith.constant 1 : index
                %c4 = arith.constant 4 : index
                %c34816 = arith.constant 34816 : index
                %cst_1 = arith.constant dense<0.000000e+00> : vector<16xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<f16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<f16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 2
                %block_id_y = gpu.block_id  y upper_bound 266
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<f16> to memref<256x1024xf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_2 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [68032, 1024], strides: [1024, 1] : memref<f16> to memref<68032x1024xf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_3 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 68032], strides: [68032, 1] : memref<f32> to memref<256x68032xf32, strided<[68032, 1], offset: ?>>
                %alloc = memref.alloc() : memref<52224xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c0][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<256x68xf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c34816][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<128x68xf16, #gpu.address_space<workgroup>>
                %3 = affine.apply #map()[%block_id_y, %block_id_x]
                %4 = arith.minsi %3, %c4 : index
                %5 = affine.apply #map1()[%block_id_y, %block_id_x, %4]
                %6 = arith.maxsi %5, %c1 : index
                %7 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %8 = affine.apply #map3()[%thread_id_x]
                %9 = arith.muli %7, %c1024 overflow<nsw> : index
                %10 = arith.addi %9, %8 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xf16, strided<[1024, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<f16> to memref<?xf16, strided<[1], offset: ?>>
                %11 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>
                %66 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                //%12 = vector.load %11[%10] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                amdgpu.gather_to_lds %11[%10], %view_4[%66, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x68xf16, #gpu.address_space<workgroup>>
                //vector.store %12, %view_4[%66, %8] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>

                %13 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %14 = arith.muli %13, %c1024 overflow<nsw> : index
                %15 = arith.addi %14, %8 overflow<nsw> : index
                //%16 = vector.load %11[%15] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                %67 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                //vector.store %16, %view_4[%67, %8] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                amdgpu.gather_to_lds %11[%15], %view_4[%67, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x68xf16, #gpu.address_space<workgroup>>

                %17 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %18 = arith.cmpi slt, %17, %c68032 : index
                %19 = vector.broadcast %18 : i1 to vector<8xi1>
                %20 = arith.muli %17, %c1024 overflow<nsw> : index
                %21 = arith.addi %20, %8 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_2 : memref<68032x1024xf16, strided<[1024, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<f16> to memref<?xf16, strided<[1], offset: ?>>
                %22 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>
                %23 = arith.index_cast %21 : index to i32
                %24 = vector.broadcast %23 : i32 to vector<8xi32>
                %25 = arith.addi %24, %cst : vector<8xi32>
                %26 = arith.index_cast %25 : vector<8xi32> to vector<8xindex>
                %27 = arith.select %19, %26, %cst_0 : vector<8xi1>, vector<8xindex>
                %28 = vector.extract %27[0] : index from vector<8xindex>
                //%29 = vector.load %22[%28] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                %68 = affine.apply #map11()[%thread_id_x, %thread_id_y]
                //vector.store %29, %view[%68, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                amdgpu.gather_to_lds %22[%28] , %view[%68, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x68xf16, #gpu.address_space<workgroup>>
                %30 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %31 = arith.cmpi slt, %30, %c68032 : index
                %32 = vector.broadcast %31 : i1 to vector<8xi1>
                %33 = arith.muli %30, %c1024 overflow<nsw> : index
                %34 = arith.addi %33, %8 overflow<nsw> : index
                %35 = arith.index_cast %34 : index to i32
                %36 = vector.broadcast %35 : i32 to vector<8xi32>
                %37 = arith.addi %36, %cst : vector<8xi32>
                %38 = arith.index_cast %37 : vector<8xi32> to vector<8xindex>
                %39 = arith.select %32, %38, %cst_0 : vector<8xi1>, vector<8xindex>
                %40 = vector.extract %39[0] : index from vector<8xindex>
                //%41 = vector.load %22[%40] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                %69 = affine.apply #map12()[%thread_id_x, %thread_id_y]
                //vector.store %41, %view[%69, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                amdgpu.gather_to_lds %22[%40] , %view[%69, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x68xf16, #gpu.address_space<workgroup>>
                %42 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %43 = arith.cmpi slt, %42, %c68032 : index
                %44 = vector.broadcast %43 : i1 to vector<8xi1>
                %45 = arith.muli %42, %c1024 overflow<nsw> : index
                %46 = arith.addi %45, %8 overflow<nsw> : index
                %47 = arith.index_cast %46 : index to i32
                %48 = vector.broadcast %47 : i32 to vector<8xi32>
                %49 = arith.addi %48, %cst : vector<8xi32>
                %50 = arith.index_cast %49 : vector<8xi32> to vector<8xindex>
                %51 = arith.select %44, %50, %cst_0 : vector<8xi1>, vector<8xindex>
                %52 = vector.extract %51[0] : index from vector<8xindex>
                //%53 = vector.load %22[%52] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                %70 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                //vector.store %53, %view[%70, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                amdgpu.gather_to_lds %22[%52] , %view[%70, %8] : vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x68xf16, #gpu.address_space<workgroup>>
                %54 = affine.apply #map8()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %55 = arith.cmpi slt, %54, %c68032 : index
                %56 = vector.broadcast %55 : i1 to vector<8xi1>
                %57 = arith.muli %54, %c1024 overflow<nsw> : index
                %58 = arith.addi %57, %8 overflow<nsw> : index
                %59 = arith.index_cast %58 : index to i32
                %60 = vector.broadcast %59 : i32 to vector<8xi32>
                %61 = arith.addi %60, %cst : vector<8xi32>
                %62 = arith.index_cast %61 : vector<8xi32> to vector<8xindex>
                %63 = arith.select %56, %62, %cst_0 : vector<8xi1>, vector<8xindex>
                %64 = vector.extract %63[0] : index from vector<8xindex>
                //%65 = vector.load %22[%64] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                %71 = affine.apply #map14()[%thread_id_x, %thread_id_y]
                //vector.store %65, %view[%71, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                amdgpu.gather_to_lds%22[%64] ,  %view[%71, %8]: vector<8xf16>, memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x68xf16, #gpu.address_space<workgroup>>
                amdgpu.lds_barrier
                %72 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %73 = arith.index_cast %72 : index to i32
                %74 = arith.cmpi sge, %73, %c4_i32 : i32
                %75 = arith.cmpi slt, %73, %c4_i32 : i32
                scf.if %74 {
                rocdl.s.barrier
                }
                %76 = affine.apply #map16()[%thread_id_x]
                %77 = affine.apply #map17()[%thread_id_x]
                %78 = affine.apply #map18()[%thread_id_x]
                %79 = affine.apply #map19()[%thread_id_x, %thread_id_y]
                %80 = affine.apply #map20()[%thread_id_x, %thread_id_y]
                %81 = affine.apply #map21()[%thread_id_x, %thread_id_y]
                %82 = affine.apply #map22()[%thread_id_x, %thread_id_y]
                %83 = affine.apply #map23()[%thread_id_x]
                %84 = affine.apply #map24()[%thread_id_x]
                %85:4 = scf.for %arg3 = %c0 to %c15 step %c1 iter_args(%arg4 = %cst_1, %arg5 = %cst_1, %arg6 = %cst_1, %arg7 = %cst_1) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
                %369 = vector.load %view_4[%76, %77] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %370 = vector.load %view_4[%76, %78] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %371 = vector.load %view[%79, %77] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %372 = vector.load %view[%79, %78] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %373 = vector.load %view[%80, %77] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %374 = vector.load %view[%80, %78] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %375 = vector.load %view[%81, %77] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %376 = vector.load %view[%81, %78] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %377 = vector.load %view[%82, %77] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %378 = vector.load %view[%82, %78] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %379 = affine.apply #map25()[%arg3, %thread_id_x]
                %380 = arith.addi %14, %379 overflow<nsw> : index
                %381 = vector.load %11[%380] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                %382 = arith.addi %9, %379 overflow<nsw> : index
                %383 = vector.load %11[%382] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %384 = vector.load %view_4[%76, %83] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %385 = vector.load %view_4[%76, %84] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %386 = vector.load %view[%79, %83] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %387 = vector.load %view[%79, %84] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %388 = vector.load %view[%80, %83] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %389 = vector.load %view[%80, %84] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %390 = vector.load %view[%81, %83] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %391 = vector.load %view[%81, %84] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %392 = vector.load %view[%82, %83] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %393 = vector.load %view[%82, %84] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %394 = arith.addi %33, %379 overflow<nsw> : index
                %395 = arith.index_cast %394 : index to i32
                %396 = vector.broadcast %395 : i32 to vector<8xi32>
                %397 = arith.addi %396, %cst : vector<8xi32>
                %398 = arith.index_cast %397 : vector<8xi32> to vector<8xindex>
                %399 = arith.select %32, %398, %cst_0 : vector<8xi1>, vector<8xindex>
                %400 = vector.extract %399[0] : index from vector<8xindex>
                %401 = vector.load %22[%400] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                %402 = arith.addi %45, %379 overflow<nsw> : index
                %403 = arith.index_cast %402 : index to i32
                %404 = vector.broadcast %403 : i32 to vector<8xi32>
                %405 = arith.addi %404, %cst : vector<8xi32>
                %406 = arith.index_cast %405 : vector<8xi32> to vector<8xindex>
                %407 = arith.select %44, %406, %cst_0 : vector<8xi1>, vector<8xindex>
                %408 = vector.extract %407[0] : index from vector<8xindex>
                %409 = vector.load %22[%408] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                %410 = arith.addi %20, %379 overflow<nsw> : index
                %411 = arith.index_cast %410 : index to i32
                %412 = vector.broadcast %411 : i32 to vector<8xi32>
                %413 = arith.addi %412, %cst : vector<8xi32>
                %414 = arith.index_cast %413 : vector<8xi32> to vector<8xindex>
                %415 = arith.select %19, %414, %cst_0 : vector<8xi1>, vector<8xindex>
                %416 = vector.extract %415[0] : index from vector<8xindex>
                %417 = vector.load %22[%416] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                %418 = arith.addi %57, %379 overflow<nsw> : index
                %419 = arith.index_cast %418 : index to i32
                %420 = vector.broadcast %419 : i32 to vector<8xi32>
                %421 = arith.addi %420, %cst : vector<8xi32>
                %422 = arith.index_cast %421 : vector<8xi32> to vector<8xindex>
                %423 = arith.select %56, %422, %cst_0 : vector<8xi1>, vector<8xindex>
                %424 = vector.extract %423[0] : index from vector<8xindex>
                %425 = vector.load %22[%424] : memref<?xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %426 = amdgpu.mfma %369 * %371 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %427 = amdgpu.mfma %370 * %372 + %426 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %428 = amdgpu.mfma %369 * %373 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %429 = amdgpu.mfma %370 * %374 + %428 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %430 = amdgpu.mfma %369 * %375 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %431 = amdgpu.mfma %370 * %376 + %430 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %432 = amdgpu.mfma %369 * %377 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %433 = amdgpu.mfma %370 * %378 + %432 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                rocdl.s.setprio 0
                amdgpu.lds_barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                vector.store %383, %view_4[%66, %8] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                vector.store %381, %view_4[%67, %8] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                vector.store %409, %view[%70, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                vector.store %425, %view[%71, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                vector.store %417, %view[%68, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                vector.store %401, %view[%69, %8] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %434 = amdgpu.mfma %384 * %386 + %427 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %435 = amdgpu.mfma %385 * %387 + %434 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %436 = amdgpu.mfma %384 * %388 + %429 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %437 = amdgpu.mfma %385 * %389 + %436 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %438 = amdgpu.mfma %384 * %390 + %431 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %439 = amdgpu.mfma %385 * %391 + %438 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %440 = amdgpu.mfma %384 * %392 + %433 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %441 = amdgpu.mfma %385 * %393 + %440 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                rocdl.s.setprio 0
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.lds_barrier
                scf.yield %435, %437, %439, %441 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
                }
                scf.if %75 {
                rocdl.s.barrier
                }
                %86 = affine.apply #map19()[%thread_id_x, %thread_id_y]
                %87 = affine.apply #map17()[%thread_id_x]
                %88 = vector.load %view[%86, %87] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %89 = affine.apply #map18()[%thread_id_x]
                %90 = vector.load %view[%86, %89] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %91 = affine.apply #map23()[%thread_id_x]
                %92 = vector.load %view[%86, %91] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %93 = affine.apply #map24()[%thread_id_x]
                %94 = vector.load %view[%86, %93] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %95 = affine.apply #map20()[%thread_id_x, %thread_id_y]
                %96 = vector.load %view[%95, %87] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %97 = vector.load %view[%95, %89] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %98 = vector.load %view[%95, %91] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %99 = vector.load %view[%95, %93] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %100 = affine.apply #map21()[%thread_id_x, %thread_id_y]
                %101 = vector.load %view[%100, %87] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %102 = vector.load %view[%100, %89] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %103 = vector.load %view[%100, %91] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %104 = vector.load %view[%100, %93] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %105 = affine.apply #map22()[%thread_id_x, %thread_id_y]
                %106 = vector.load %view[%105, %87] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %107 = vector.load %view[%105, %89] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %108 = vector.load %view[%105, %91] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %109 = vector.load %view[%105, %93] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %110 = affine.apply #map16()[%thread_id_x]
                %111 = vector.load %view_4[%110, %87] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %112 = vector.load %view_4[%110, %89] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %113 = vector.load %view_4[%110, %91] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %114 = vector.load %view_4[%110, %93] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
                %115 = amdgpu.mfma %111 * %88 + %85#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %116 = amdgpu.mfma %112 * %90 + %115 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %117 = amdgpu.mfma %113 * %92 + %116 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %118 = amdgpu.mfma %114 * %94 + %117 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %119 = amdgpu.mfma %111 * %96 + %85#1 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %120 = amdgpu.mfma %112 * %97 + %119 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %121 = amdgpu.mfma %113 * %98 + %120 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %122 = amdgpu.mfma %114 * %99 + %121 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %123 = amdgpu.mfma %111 * %101 + %85#2 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %124 = amdgpu.mfma %112 * %102 + %123 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %125 = amdgpu.mfma %113 * %103 + %124 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %126 = amdgpu.mfma %114 * %104 + %125 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %127 = amdgpu.mfma %111 * %106 + %85#3 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %128 = amdgpu.mfma %112 * %107 + %127 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %129 = amdgpu.mfma %113 * %108 + %128 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %130 = amdgpu.mfma %114 * %109 + %129 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf16>, vector<8xf16>, vector<16xf32>
                %131 = vector.extract_strided_slice %118 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %132 = affine.apply #map26()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %133 = arith.cmpi slt, %132, %c68032 : index
                %134 = affine.apply #map27()[%block_id_y, %block_id_x, %4, %4, %6]
                %135 = affine.apply #map28()[%block_id_x, %block_id_y, %4, %6]
                %136 = affine.apply #map29()[%thread_id_x]
                %137 = arith.muli %134, %c68032 overflow<nsw> : index
                %138 = arith.muli %136, %c68032 overflow<nsw> : index
                %139 = arith.addi %137, %135 overflow<nsw> : index
                %140 = arith.addi %138, %86 overflow<nsw> : index
                %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %reinterpret_cast_3 : memref<256x68032xf32, strided<[68032, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
                %141 = arith.addi %139, %offset_12 overflow<nsw> : index
                %reinterpret_cast_15 = memref.reinterpret_cast %2 to offset: [%141], sizes: [%c536870910], strides: [1] : memref<f32> to memref<?xf32, strided<[1], offset: ?>>
                %142 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_15 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                %143 = arith.select %133, %140, %c536870911 : index
                vector.store %131, %142[%143] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %144 = vector.extract_strided_slice %118 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %145 = affine.apply #map30()[%thread_id_x]
                %146 = arith.muli %145, %c68032 overflow<nsw> : index
                %147 = arith.addi %146, %86 overflow<nsw> : index
                %148 = arith.select %133, %147, %c536870911 : index
                vector.store %144, %142[%148] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %149 = vector.extract_strided_slice %118 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %150 = affine.apply #map31()[%thread_id_x]
                %151 = arith.muli %150, %c68032 overflow<nsw> : index
                %152 = arith.addi %151, %86 overflow<nsw> : index
                %153 = arith.select %133, %152, %c536870911 : index
                vector.store %149, %142[%153] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %154 = vector.extract_strided_slice %118 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %155 = affine.apply #map32()[%thread_id_x]
                %156 = arith.muli %155, %c68032 overflow<nsw> : index
                %157 = arith.addi %156, %86 overflow<nsw> : index
                %158 = arith.select %133, %157, %c536870911 : index
                vector.store %154, %142[%158] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %159 = vector.extract_strided_slice %118 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %160 = affine.apply #map33()[%thread_id_x]
                %161 = arith.muli %160, %c68032 overflow<nsw> : index
                %162 = arith.addi %161, %86 overflow<nsw> : index
                %163 = arith.select %133, %162, %c536870911 : index
                vector.store %159, %142[%163] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %164 = vector.extract_strided_slice %118 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %165 = affine.apply #map34()[%thread_id_x]
                %166 = arith.muli %165, %c68032 overflow<nsw> : index
                %167 = arith.addi %166, %86 overflow<nsw> : index
                %168 = arith.select %133, %167, %c536870911 : index
                vector.store %164, %142[%168] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %169 = vector.extract_strided_slice %118 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %170 = affine.apply #map35()[%thread_id_x]
                %171 = arith.muli %170, %c68032 overflow<nsw> : index
                %172 = arith.addi %171, %86 overflow<nsw> : index
                %173 = arith.select %133, %172, %c536870911 : index
                vector.store %169, %142[%173] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %174 = vector.extract_strided_slice %118 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %175 = affine.apply #map36()[%thread_id_x]
                %176 = arith.muli %175, %c68032 overflow<nsw> : index
                %177 = arith.addi %176, %86 overflow<nsw> : index
                %178 = arith.select %133, %177, %c536870911 : index
                vector.store %174, %142[%178] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %179 = vector.extract_strided_slice %118 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %180 = affine.apply #map37()[%thread_id_x]
                %181 = arith.muli %180, %c68032 overflow<nsw> : index
                %182 = arith.addi %181, %86 overflow<nsw> : index
                %183 = arith.select %133, %182, %c536870911 : index
                vector.store %179, %142[%183] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %184 = vector.extract_strided_slice %118 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %185 = affine.apply #map38()[%thread_id_x]
                %186 = arith.muli %185, %c68032 overflow<nsw> : index
                %187 = arith.addi %186, %86 overflow<nsw> : index
                %188 = arith.select %133, %187, %c536870911 : index
                vector.store %184, %142[%188] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %189 = vector.extract_strided_slice %118 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %190 = affine.apply #map39()[%thread_id_x]
                %191 = arith.muli %190, %c68032 overflow<nsw> : index
                %192 = arith.addi %191, %86 overflow<nsw> : index
                %193 = arith.select %133, %192, %c536870911 : index
                vector.store %189, %142[%193] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %194 = vector.extract_strided_slice %118 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %195 = affine.apply #map40()[%thread_id_x]
                %196 = arith.muli %195, %c68032 overflow<nsw> : index
                %197 = arith.addi %196, %86 overflow<nsw> : index
                %198 = arith.select %133, %197, %c536870911 : index
                vector.store %194, %142[%198] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %199 = vector.extract_strided_slice %118 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %200 = affine.apply #map41()[%thread_id_x]
                %201 = arith.muli %200, %c68032 overflow<nsw> : index
                %202 = arith.addi %201, %86 overflow<nsw> : index
                %203 = arith.select %133, %202, %c536870911 : index
                vector.store %199, %142[%203] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %204 = vector.extract_strided_slice %118 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %205 = affine.apply #map42()[%thread_id_x]
                %206 = arith.muli %205, %c68032 overflow<nsw> : index
                %207 = arith.addi %206, %86 overflow<nsw> : index
                %208 = arith.select %133, %207, %c536870911 : index
                vector.store %204, %142[%208] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %209 = vector.extract_strided_slice %118 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %210 = affine.apply #map43()[%thread_id_x]
                %211 = arith.muli %210, %c68032 overflow<nsw> : index
                %212 = arith.addi %211, %86 overflow<nsw> : index
                %213 = arith.select %133, %212, %c536870911 : index
                vector.store %209, %142[%213] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %214 = vector.extract_strided_slice %118 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %215 = affine.apply #map44()[%thread_id_x]
                %216 = arith.muli %215, %c68032 overflow<nsw> : index
                %217 = arith.addi %216, %86 overflow<nsw> : index
                %218 = arith.select %133, %217, %c536870911 : index
                vector.store %214, %142[%218] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %219 = vector.extract_strided_slice %122 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %220 = affine.apply #map45()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %221 = arith.cmpi slt, %220, %c68032 : index
                %222 = arith.addi %138, %95 overflow<nsw> : index
                %223 = arith.select %221, %222, %c536870911 : index
                vector.store %219, %142[%223] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %224 = vector.extract_strided_slice %122 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %225 = arith.addi %146, %95 overflow<nsw> : index
                %226 = arith.select %221, %225, %c536870911 : index
                vector.store %224, %142[%226] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %227 = vector.extract_strided_slice %122 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %228 = arith.addi %151, %95 overflow<nsw> : index
                %229 = arith.select %221, %228, %c536870911 : index
                vector.store %227, %142[%229] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %230 = vector.extract_strided_slice %122 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %231 = arith.addi %156, %95 overflow<nsw> : index
                %232 = arith.select %221, %231, %c536870911 : index
                vector.store %230, %142[%232] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %233 = vector.extract_strided_slice %122 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %234 = arith.addi %161, %95 overflow<nsw> : index
                %235 = arith.select %221, %234, %c536870911 : index
                vector.store %233, %142[%235] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %236 = vector.extract_strided_slice %122 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %237 = arith.addi %166, %95 overflow<nsw> : index
                %238 = arith.select %221, %237, %c536870911 : index
                vector.store %236, %142[%238] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %239 = vector.extract_strided_slice %122 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %240 = arith.addi %171, %95 overflow<nsw> : index
                %241 = arith.select %221, %240, %c536870911 : index
                vector.store %239, %142[%241] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %242 = vector.extract_strided_slice %122 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %243 = arith.addi %176, %95 overflow<nsw> : index
                %244 = arith.select %221, %243, %c536870911 : index
                vector.store %242, %142[%244] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %245 = vector.extract_strided_slice %122 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %246 = arith.addi %181, %95 overflow<nsw> : index
                %247 = arith.select %221, %246, %c536870911 : index
                vector.store %245, %142[%247] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %248 = vector.extract_strided_slice %122 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %249 = arith.addi %186, %95 overflow<nsw> : index
                %250 = arith.select %221, %249, %c536870911 : index
                vector.store %248, %142[%250] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %251 = vector.extract_strided_slice %122 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %252 = arith.addi %191, %95 overflow<nsw> : index
                %253 = arith.select %221, %252, %c536870911 : index
                vector.store %251, %142[%253] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %254 = vector.extract_strided_slice %122 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %255 = arith.addi %196, %95 overflow<nsw> : index
                %256 = arith.select %221, %255, %c536870911 : index
                vector.store %254, %142[%256] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %257 = vector.extract_strided_slice %122 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %258 = arith.addi %201, %95 overflow<nsw> : index
                %259 = arith.select %221, %258, %c536870911 : index
                vector.store %257, %142[%259] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %260 = vector.extract_strided_slice %122 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %261 = arith.addi %206, %95 overflow<nsw> : index
                %262 = arith.select %221, %261, %c536870911 : index
                vector.store %260, %142[%262] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %263 = vector.extract_strided_slice %122 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %264 = arith.addi %211, %95 overflow<nsw> : index
                %265 = arith.select %221, %264, %c536870911 : index
                vector.store %263, %142[%265] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %266 = vector.extract_strided_slice %122 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %267 = arith.addi %216, %95 overflow<nsw> : index
                %268 = arith.select %221, %267, %c536870911 : index
                vector.store %266, %142[%268] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %269 = vector.extract_strided_slice %126 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %270 = affine.apply #map46()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %271 = arith.cmpi slt, %270, %c68032 : index
                %272 = arith.addi %138, %100 overflow<nsw> : index
                %273 = arith.select %271, %272, %c536870911 : index
                vector.store %269, %142[%273] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %274 = vector.extract_strided_slice %126 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %275 = arith.addi %146, %100 overflow<nsw> : index
                %276 = arith.select %271, %275, %c536870911 : index
                vector.store %274, %142[%276] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %277 = vector.extract_strided_slice %126 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %278 = arith.addi %151, %100 overflow<nsw> : index
                %279 = arith.select %271, %278, %c536870911 : index
                vector.store %277, %142[%279] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %280 = vector.extract_strided_slice %126 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %281 = arith.addi %156, %100 overflow<nsw> : index
                %282 = arith.select %271, %281, %c536870911 : index
                vector.store %280, %142[%282] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %283 = vector.extract_strided_slice %126 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %284 = arith.addi %161, %100 overflow<nsw> : index
                %285 = arith.select %271, %284, %c536870911 : index
                vector.store %283, %142[%285] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %286 = vector.extract_strided_slice %126 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %287 = arith.addi %166, %100 overflow<nsw> : index
                %288 = arith.select %271, %287, %c536870911 : index
                vector.store %286, %142[%288] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %289 = vector.extract_strided_slice %126 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %290 = arith.addi %171, %100 overflow<nsw> : index
                %291 = arith.select %271, %290, %c536870911 : index
                vector.store %289, %142[%291] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %292 = vector.extract_strided_slice %126 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %293 = arith.addi %176, %100 overflow<nsw> : index
                %294 = arith.select %271, %293, %c536870911 : index
                vector.store %292, %142[%294] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %295 = vector.extract_strided_slice %126 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %296 = arith.addi %181, %100 overflow<nsw> : index
                %297 = arith.select %271, %296, %c536870911 : index
                vector.store %295, %142[%297] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %298 = vector.extract_strided_slice %126 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %299 = arith.addi %186, %100 overflow<nsw> : index
                %300 = arith.select %271, %299, %c536870911 : index
                vector.store %298, %142[%300] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %301 = vector.extract_strided_slice %126 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %302 = arith.addi %191, %100 overflow<nsw> : index
                %303 = arith.select %271, %302, %c536870911 : index
                vector.store %301, %142[%303] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %304 = vector.extract_strided_slice %126 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %305 = arith.addi %196, %100 overflow<nsw> : index
                %306 = arith.select %271, %305, %c536870911 : index
                vector.store %304, %142[%306] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %307 = vector.extract_strided_slice %126 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %308 = arith.addi %201, %100 overflow<nsw> : index
                %309 = arith.select %271, %308, %c536870911 : index
                vector.store %307, %142[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %310 = vector.extract_strided_slice %126 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %311 = arith.addi %206, %100 overflow<nsw> : index
                %312 = arith.select %271, %311, %c536870911 : index
                vector.store %310, %142[%312] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %313 = vector.extract_strided_slice %126 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %314 = arith.addi %211, %100 overflow<nsw> : index
                %315 = arith.select %271, %314, %c536870911 : index
                vector.store %313, %142[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %316 = vector.extract_strided_slice %126 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %317 = arith.addi %216, %100 overflow<nsw> : index
                %318 = arith.select %271, %317, %c536870911 : index
                vector.store %316, %142[%318] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %319 = vector.extract_strided_slice %130 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %320 = affine.apply #map47()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %321 = arith.cmpi slt, %320, %c68032 : index
                %322 = arith.addi %138, %105 overflow<nsw> : index
                %323 = arith.select %321, %322, %c536870911 : index
                vector.store %319, %142[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %324 = vector.extract_strided_slice %130 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %325 = arith.addi %146, %105 overflow<nsw> : index
                %326 = arith.select %321, %325, %c536870911 : index
                vector.store %324, %142[%326] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %327 = vector.extract_strided_slice %130 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %328 = arith.addi %151, %105 overflow<nsw> : index
                %329 = arith.select %321, %328, %c536870911 : index
                vector.store %327, %142[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %330 = vector.extract_strided_slice %130 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %331 = arith.addi %156, %105 overflow<nsw> : index
                %332 = arith.select %321, %331, %c536870911 : index
                vector.store %330, %142[%332] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %333 = vector.extract_strided_slice %130 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %334 = arith.addi %161, %105 overflow<nsw> : index
                %335 = arith.select %321, %334, %c536870911 : index
                vector.store %333, %142[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %336 = vector.extract_strided_slice %130 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %337 = arith.addi %166, %105 overflow<nsw> : index
                %338 = arith.select %321, %337, %c536870911 : index
                vector.store %336, %142[%338] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %339 = vector.extract_strided_slice %130 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %340 = arith.addi %171, %105 overflow<nsw> : index
                %341 = arith.select %321, %340, %c536870911 : index
                vector.store %339, %142[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %342 = vector.extract_strided_slice %130 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %343 = arith.addi %176, %105 overflow<nsw> : index
                %344 = arith.select %321, %343, %c536870911 : index
                vector.store %342, %142[%344] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %345 = vector.extract_strided_slice %130 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %346 = arith.addi %181, %105 overflow<nsw> : index
                %347 = arith.select %321, %346, %c536870911 : index
                vector.store %345, %142[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %348 = vector.extract_strided_slice %130 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %349 = arith.addi %186, %105 overflow<nsw> : index
                %350 = arith.select %321, %349, %c536870911 : index
                vector.store %348, %142[%350] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %351 = vector.extract_strided_slice %130 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %352 = arith.addi %191, %105 overflow<nsw> : index
                %353 = arith.select %321, %352, %c536870911 : index
                vector.store %351, %142[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %354 = vector.extract_strided_slice %130 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %355 = arith.addi %196, %105 overflow<nsw> : index
                %356 = arith.select %321, %355, %c536870911 : index
                vector.store %354, %142[%356] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %357 = vector.extract_strided_slice %130 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %358 = arith.addi %201, %105 overflow<nsw> : index
                %359 = arith.select %321, %358, %c536870911 : index
                vector.store %357, %142[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %360 = vector.extract_strided_slice %130 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %361 = arith.addi %206, %105 overflow<nsw> : index
                %362 = arith.select %321, %361, %c536870911 : index
                vector.store %360, %142[%362] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %363 = vector.extract_strided_slice %130 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %364 = arith.addi %211, %105 overflow<nsw> : index
                %365 = arith.select %321, %364, %c536870911 : index
                vector.store %363, %142[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %366 = vector.extract_strided_slice %130 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %367 = arith.addi %216, %105 overflow<nsw> : index
                %368 = arith.select %321, %367, %c536870911 : index
                vector.store %366, %142[%368] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<68032x1024xf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x68032xf32>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xf16>, tensor<68032x1024xf16>, tensor<256x68032xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x68032xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x68032xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
        """
   
    asm_bf16="""
        #map = affine_map<()[s0, s1] -> ((s0 * 2 + s1) mod 8)>
        #map1 = affine_map<()[s0, s1, s2] -> (((s0 * 132 + s1 * 66 + s2 - ((s0 * 2 + s1) floordiv 8) * 527) floordiv 4256) * -16 + 2)>
        #map2 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + ((s2 * 132 + s3 * 66 + s4 - ((s2 * 2 + s3) floordiv 8) * 527) floordiv 4256) * 2048 + (((s2 * 132 + s3 * 66 + s5 - ((s2 * 2 + s3) floordiv 8) * 527) mod 4256) mod s6) * 128)>
        #map3 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 8) * 64)>
        #map4 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + ((s2 * 132 + s3 * 66 + s4 - ((s2 * 2 + s3) floordiv 8) * 527) floordiv 4256) * 2048 + (((s2 * 132 + s3 * 66 + s5 - ((s2 * 2 + s3) floordiv 8) * 527) mod 4256) mod s6) * 128 + 64)>
        #map5 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256)>
        #map6 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 64)>
        #map7 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 128)>
        #map8 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 192)>
        #map9 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 128)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>
        #map11 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
        #map12 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map13 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map14 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map16 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
        #map17 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8)>
        #map18 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 16)>
        #map19 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32)>
        #map20 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 32)>
        #map21 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 64)>
        #map22 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 96)>
        #map23 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 32)>
        #map24 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 48)>
        #map25 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 - (s1 floordiv 8) * 64 + 64)>
        #map26 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256)>
        #map27 = affine_map<()[s0, s1, s2, s3, s4] -> (((s0 * 132 + s1 * 66 + s2 - ((s0 * 2 + s1) floordiv 8) * 527) floordiv 4256) * 2048 + (((s0 * 132 + s1 * 66 + s3 - ((s0 * 2 + s1) floordiv 8) * 527) mod 4256) mod s4) * 128)>
        #map28 = affine_map<()[s0, s1, s2, s3] -> ((((s0 * 66 + s1 * 132 + s2 - ((s0 + s1 * 2) floordiv 8) * 527) mod 4256) floordiv s3) * 256)>
        #map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
        #map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
        #map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
        #map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
        #map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
        #map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
        #map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
        #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
        #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
        #map45 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 32)>
        #map46 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 64)>
        #map47 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 96)>
        #map_wave_offset = affine_map<()[s0, s1] -> ((s1 * 4 + s0 floordiv 64) * 8)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c2 = arith.constant 2 : index
            %c266 = arith.constant 266 : index
            %c1 = arith.constant 1 : index
            stream.return %c2, %c266, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
                %cst_0 = arith.constant dense<1073741823> : vector<8xindex>
                %c1024_i14 = arith.constant 1024 : i14
                %c536870911 = arith.constant 536870911 : index
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c536870910 = arith.constant 536870910 : index
                %c0_i32 = arith.constant 0 : i32
                %c15 = arith.constant 15 : index
                %c68032 = arith.constant 68032 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c1024 = arith.constant 1024 : index
                %c1 = arith.constant 1 : index
                %c4 = arith.constant 4 : index
                %c34816 = arith.constant 34816 : index
                %cst_1 = arith.constant dense<0.000000e+00> : vector<16xf32>
                %c0 = arith.constant 0 : index
                %c64 = arith.constant 64 : index
                %c128 = arith.constant 128 : index
                %c192 = arith.constant 192 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 2
                %block_id_y = gpu.block_id  y upper_bound 266
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<bf16> to memref<256x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_2 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [68032, 1024], strides: [1024, 1] : memref<bf16> to memref<68032x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_3 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 68032], strides: [68032, 1] : memref<f32> to memref<256x68032xf32, strided<[68032, 1], offset: ?>>
                %alloc = memref.alloc() : memref<52224xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c0][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<256x68xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c34816][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<128x68xbf16, #gpu.address_space<workgroup>>
                %tile4 = memref.subview %view_4[0, 0] [128, 64] [1, 1] : memref<128x68xbf16, #gpu.address_space<workgroup>> to memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
                %tile = memref.subview %view[0, 0] [256, 64] [1, 1] : memref<256x68xbf16, #gpu.address_space<workgroup>> to memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
        
                %3 = affine.apply #map()[%block_id_y, %block_id_x]
                %4 = arith.minsi %3, %c4 : index
                %5 = affine.apply #map1()[%block_id_y, %block_id_x, %4]
                %6 = arith.maxsi %5, %c1 : index
                %7 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %8 = affine.apply #map3()[%thread_id_x]
                %9 = arith.muli %7, %c1024 overflow<nsw> : index
                %10 = arith.addi %9, %8 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %11 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %66 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                
                %67 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                //%12 = vector.load %11[%10] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                //vector.store %12, %view_4[%66, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset = affine.apply #map_wave_offset()[%thread_id_x, %thread_id_y]
                amdgpu.gather_to_lds %11[%10], %tile4[%wave_offset, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                %13 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %14 = arith.muli %13, %c1024 overflow<nsw> : index
                %15 = arith.addi %14, %8 overflow<nsw> : index
                //%16 = vector.load %11[%15] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                //vector.store %16, %view_4[%67, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset64= arith.addi %wave_offset, %c64 overflow<nsw> : index
                amdgpu.gather_to_lds %11[%15], %tile4[%wave_offset64, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                %17 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %18 = arith.cmpi slt, %17, %c68032 : index
                %19 = vector.broadcast %18 : i1 to vector<8xi1>
                %20 = arith.muli %17, %c1024 overflow<nsw> : index
                %21 = arith.addi %20, %8 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_2 : memref<68032x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %22 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %23 = arith.index_cast %21 : index to i32
                %24 = vector.broadcast %23 : i32 to vector<8xi32>
                %25 = arith.addi %24, %cst : vector<8xi32>
                %26 = arith.index_cast %25 : vector<8xi32> to vector<8xindex>
                %27 = arith.select %19, %26, %cst_0 : vector<8xi1>, vector<8xindex>
                %28 = vector.extract %27[0] : index from vector<8xindex>
                %29 = vector.load %22[%28] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %68 = affine.apply #map11()[%thread_id_x, %thread_id_y]
                vector.store %29, %view[%68, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                //amdgpu.gather_to_lds %22[%28], %tile[%wave_offset, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                %30 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %31 = arith.cmpi slt, %30, %c68032 : index
                %32 = vector.broadcast %31 : i1 to vector<8xi1>
                %33 = arith.muli %30, %c1024 overflow<nsw> : index
                %34 = arith.addi %33, %8 overflow<nsw> : index
                %35 = arith.index_cast %34 : index to i32
                %36 = vector.broadcast %35 : i32 to vector<8xi32>
                %37 = arith.addi %36, %cst : vector<8xi32>
                %38 = arith.index_cast %37 : vector<8xi32> to vector<8xindex>
                %39 = arith.select %32, %38, %cst_0 : vector<8xi1>, vector<8xindex>
                %40 = vector.extract %39[0] : index from vector<8xindex>
                %41 = vector.load %22[%40] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %69 = affine.apply #map12()[%thread_id_x, %thread_id_y]
                vector.store %41, %view[%69, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                //amdgpu.gather_to_lds %22[%40], %tile[%wave_offset64, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                %42 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %43 = arith.cmpi slt, %42, %c68032 : index
                %44 = vector.broadcast %43 : i1 to vector<8xi1>
                %45 = arith.muli %42, %c1024 overflow<nsw> : index
                %46 = arith.addi %45, %8 overflow<nsw> : index
                %47 = arith.index_cast %46 : index to i32
                %48 = vector.broadcast %47 : i32 to vector<8xi32>
                %49 = arith.addi %48, %cst : vector<8xi32>
                %50 = arith.index_cast %49 : vector<8xi32> to vector<8xindex>
                %51 = arith.select %44, %50, %cst_0 : vector<8xi1>, vector<8xindex>
                %52 = vector.extract %51[0] : index from vector<8xindex>
                %53 = vector.load %22[%52] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %70 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                vector.store %53, %view[%70, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset128 = arith.addi %wave_offset, %c128 overflow<nsw> : index
                //amdgpu.gather_to_lds %22[%52], %tile[%wave_offset128, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                %54 = affine.apply #map8()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4, %6]
                %55 = arith.cmpi slt, %54, %c68032 : index
                %56 = vector.broadcast %55 : i1 to vector<8xi1>
                %57 = arith.muli %54, %c1024 overflow<nsw> : index
                %58 = arith.addi %57, %8 overflow<nsw> : index
                %59 = arith.index_cast %58 : index to i32
                %60 = vector.broadcast %59 : i32 to vector<8xi32>
                %61 = arith.addi %60, %cst : vector<8xi32>
                %62 = arith.index_cast %61 : vector<8xi32> to vector<8xindex>
                %63 = arith.select %56, %62, %cst_0 : vector<8xi1>, vector<8xindex>
                %64 = vector.extract %63[0] : index from vector<8xindex>
                %65 = vector.load %22[%64] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %71 = affine.apply #map14()[%thread_id_x, %thread_id_y]
                vector.store %65, %view[%71, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset192 = arith.addi %wave_offset, %c192 overflow<nsw> : index

                //amdgpu.gather_to_lds %22[%64], %tile[%wave_offset192, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<256x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
                amdgpu.lds_barrier
                %72 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %73 = arith.index_cast %72 : index to i32
                %74 = arith.cmpi sge, %73, %c4_i32 : i32
                %75 = arith.cmpi slt, %73, %c4_i32 : i32
                scf.if %74 {
                rocdl.s.barrier
                }
                %76 = affine.apply #map16()[%thread_id_x]
                %77 = affine.apply #map17()[%thread_id_x]
                %78 = affine.apply #map18()[%thread_id_x]
                %79 = affine.apply #map19()[%thread_id_x, %thread_id_y]
                %80 = affine.apply #map20()[%thread_id_x, %thread_id_y]
                %81 = affine.apply #map21()[%thread_id_x, %thread_id_y]
                %82 = affine.apply #map22()[%thread_id_x, %thread_id_y]
                %83 = affine.apply #map23()[%thread_id_x]
                %84 = affine.apply #map24()[%thread_id_x]
                %85:4 = scf.for %arg3 = %c0 to %c15 step %c1 iter_args(%arg4 = %cst_1, %arg5 = %cst_1, %arg6 = %cst_1, %arg7 = %cst_1) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
                %369 = vector.load %view_4[%76, %77] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %370 = vector.load %view_4[%76, %78] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %371 = vector.load %view[%79, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %372 = vector.load %view[%79, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %373 = vector.load %view[%80, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %374 = vector.load %view[%80, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %375 = vector.load %view[%81, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %376 = vector.load %view[%81, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %377 = vector.load %view[%82, %77] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %378 = vector.load %view[%82, %78] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %379 = affine.apply #map25()[%arg3, %thread_id_x]
                %380 = arith.addi %14, %379 overflow<nsw> : index
                %381 = vector.load %11[%380] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %382 = arith.addi %9, %379 overflow<nsw> : index
                %383 = vector.load %11[%382] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %384 = vector.load %view_4[%76, %83] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %385 = vector.load %view_4[%76, %84] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %386 = vector.load %view[%79, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %387 = vector.load %view[%79, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %388 = vector.load %view[%80, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %389 = vector.load %view[%80, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %390 = vector.load %view[%81, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %391 = vector.load %view[%81, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %392 = vector.load %view[%82, %83] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %393 = vector.load %view[%82, %84] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %394 = arith.addi %33, %379 overflow<nsw> : index
                %395 = arith.index_cast %394 : index to i32
                %396 = vector.broadcast %395 : i32 to vector<8xi32>
                %397 = arith.addi %396, %cst : vector<8xi32>
                %398 = arith.index_cast %397 : vector<8xi32> to vector<8xindex>
                %399 = arith.select %32, %398, %cst_0 : vector<8xi1>, vector<8xindex>
                %400 = vector.extract %399[0] : index from vector<8xindex>
                %401 = vector.load %22[%400] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %402 = arith.addi %20, %379 overflow<nsw> : index
                %403 = arith.index_cast %402 : index to i32
                %404 = vector.broadcast %403 : i32 to vector<8xi32>
                %405 = arith.addi %404, %cst : vector<8xi32>
                %406 = arith.index_cast %405 : vector<8xi32> to vector<8xindex>
                %407 = arith.select %19, %406, %cst_0 : vector<8xi1>, vector<8xindex>
                %408 = vector.extract %407[0] : index from vector<8xindex>
                %409 = vector.load %22[%408] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %410 = arith.addi %57, %379 overflow<nsw> : index
                %411 = arith.index_cast %410 : index to i32
                %412 = vector.broadcast %411 : i32 to vector<8xi32>
                %413 = arith.addi %412, %cst : vector<8xi32>
                %414 = arith.index_cast %413 : vector<8xi32> to vector<8xindex>
                %415 = arith.select %56, %414, %cst_0 : vector<8xi1>, vector<8xindex>
                %416 = vector.extract %415[0] : index from vector<8xindex>
                %417 = vector.load %22[%416] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                %418 = arith.addi %45, %379 overflow<nsw> : index
                %419 = arith.index_cast %418 : index to i32
                %420 = vector.broadcast %419 : i32 to vector<8xi32>
                %421 = arith.addi %420, %cst : vector<8xi32>
                %422 = arith.index_cast %421 : vector<8xi32> to vector<8xindex>
                %423 = arith.select %44, %422, %cst_0 : vector<8xi1>, vector<8xindex>
                %424 = vector.extract %423[0] : index from vector<8xindex>
                %425 = vector.load %22[%424] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %426 = amdgpu.mfma %369 * %371 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %427 = amdgpu.mfma %370 * %372 + %426 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %428 = amdgpu.mfma %369 * %373 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %429 = amdgpu.mfma %370 * %374 + %428 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %430 = amdgpu.mfma %369 * %375 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %431 = amdgpu.mfma %370 * %376 + %430 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %432 = amdgpu.mfma %369 * %377 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %433 = amdgpu.mfma %370 * %378 + %432 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                rocdl.s.setprio 0
                amdgpu.lds_barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                vector.store %381, %view_4[%67, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %383, %view_4[%66, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %425, %view[%70, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %409, %view[%68, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %417, %view[%71, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                vector.store %401, %view[%69, %8] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %434 = amdgpu.mfma %384 * %386 + %427 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %435 = amdgpu.mfma %385 * %387 + %434 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %436 = amdgpu.mfma %384 * %388 + %429 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %437 = amdgpu.mfma %385 * %389 + %436 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %438 = amdgpu.mfma %384 * %390 + %431 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %439 = amdgpu.mfma %385 * %391 + %438 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %440 = amdgpu.mfma %384 * %392 + %433 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %441 = amdgpu.mfma %385 * %393 + %440 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                rocdl.s.setprio 0
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.lds_barrier
                scf.yield %435, %437, %439, %441 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
                }
                scf.if %75 {
                rocdl.s.barrier
                }
                %86 = affine.apply #map19()[%thread_id_x, %thread_id_y]
                %87 = affine.apply #map17()[%thread_id_x]
                %88 = vector.load %view[%86, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %89 = affine.apply #map18()[%thread_id_x]
                %90 = vector.load %view[%86, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %91 = affine.apply #map23()[%thread_id_x]
                %92 = vector.load %view[%86, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %93 = affine.apply #map24()[%thread_id_x]
                %94 = vector.load %view[%86, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %95 = affine.apply #map20()[%thread_id_x, %thread_id_y]
                %96 = vector.load %view[%95, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %97 = vector.load %view[%95, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %98 = vector.load %view[%95, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %99 = vector.load %view[%95, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %100 = affine.apply #map21()[%thread_id_x, %thread_id_y]
                %101 = vector.load %view[%100, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %102 = vector.load %view[%100, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %103 = vector.load %view[%100, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %104 = vector.load %view[%100, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %105 = affine.apply #map22()[%thread_id_x, %thread_id_y]
                %106 = vector.load %view[%105, %87] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %107 = vector.load %view[%105, %89] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %108 = vector.load %view[%105, %91] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %109 = vector.load %view[%105, %93] : memref<256x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %110 = affine.apply #map16()[%thread_id_x]
                %111 = vector.load %view_4[%110, %87] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %112 = vector.load %view_4[%110, %89] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %113 = vector.load %view_4[%110, %91] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %114 = vector.load %view_4[%110, %93] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %115 = amdgpu.mfma %111 * %88 + %85#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %116 = amdgpu.mfma %112 * %90 + %115 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %117 = amdgpu.mfma %113 * %92 + %116 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %118 = amdgpu.mfma %114 * %94 + %117 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %119 = amdgpu.mfma %111 * %96 + %85#1 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %120 = amdgpu.mfma %112 * %97 + %119 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %121 = amdgpu.mfma %113 * %98 + %120 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %122 = amdgpu.mfma %114 * %99 + %121 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %123 = amdgpu.mfma %111 * %101 + %85#2 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %124 = amdgpu.mfma %112 * %102 + %123 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %125 = amdgpu.mfma %113 * %103 + %124 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %126 = amdgpu.mfma %114 * %104 + %125 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %127 = amdgpu.mfma %111 * %106 + %85#3 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %128 = amdgpu.mfma %112 * %107 + %127 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %129 = amdgpu.mfma %113 * %108 + %128 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %130 = amdgpu.mfma %114 * %109 + %129 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %131 = vector.extract_strided_slice %118 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %132 = affine.apply #map26()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %133 = arith.cmpi slt, %132, %c68032 : index
                %134 = affine.apply #map27()[%block_id_y, %block_id_x, %4, %4, %6]
                %135 = affine.apply #map28()[%block_id_x, %block_id_y, %4, %6]
                %136 = affine.apply #map29()[%thread_id_x]
                %137 = arith.muli %134, %c68032 overflow<nsw> : index
                %138 = arith.muli %136, %c68032 overflow<nsw> : index
                %139 = arith.addi %137, %135 overflow<nsw> : index
                %140 = arith.addi %138, %86 overflow<nsw> : index
                %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %reinterpret_cast_3 : memref<256x68032xf32, strided<[68032, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
                %141 = arith.addi %139, %offset_12 overflow<nsw> : index
                %reinterpret_cast_15 = memref.reinterpret_cast %2 to offset: [%141], sizes: [%c536870910], strides: [1] : memref<f32> to memref<?xf32, strided<[1], offset: ?>>
                %142 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_15 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                %143 = arith.select %133, %140, %c536870911 : index
                vector.store %131, %142[%143] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %144 = vector.extract_strided_slice %118 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %145 = affine.apply #map30()[%thread_id_x]
                %146 = arith.muli %145, %c68032 overflow<nsw> : index
                %147 = arith.addi %146, %86 overflow<nsw> : index
                %148 = arith.select %133, %147, %c536870911 : index
                vector.store %144, %142[%148] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %149 = vector.extract_strided_slice %118 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %150 = affine.apply #map31()[%thread_id_x]
                %151 = arith.muli %150, %c68032 overflow<nsw> : index
                %152 = arith.addi %151, %86 overflow<nsw> : index
                %153 = arith.select %133, %152, %c536870911 : index
                vector.store %149, %142[%153] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %154 = vector.extract_strided_slice %118 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %155 = affine.apply #map32()[%thread_id_x]
                %156 = arith.muli %155, %c68032 overflow<nsw> : index
                %157 = arith.addi %156, %86 overflow<nsw> : index
                %158 = arith.select %133, %157, %c536870911 : index
                vector.store %154, %142[%158] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %159 = vector.extract_strided_slice %118 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %160 = affine.apply #map33()[%thread_id_x]
                %161 = arith.muli %160, %c68032 overflow<nsw> : index
                %162 = arith.addi %161, %86 overflow<nsw> : index
                %163 = arith.select %133, %162, %c536870911 : index
                vector.store %159, %142[%163] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %164 = vector.extract_strided_slice %118 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %165 = affine.apply #map34()[%thread_id_x]
                %166 = arith.muli %165, %c68032 overflow<nsw> : index
                %167 = arith.addi %166, %86 overflow<nsw> : index
                %168 = arith.select %133, %167, %c536870911 : index
                vector.store %164, %142[%168] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %169 = vector.extract_strided_slice %118 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %170 = affine.apply #map35()[%thread_id_x]
                %171 = arith.muli %170, %c68032 overflow<nsw> : index
                %172 = arith.addi %171, %86 overflow<nsw> : index
                %173 = arith.select %133, %172, %c536870911 : index
                vector.store %169, %142[%173] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %174 = vector.extract_strided_slice %118 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %175 = affine.apply #map36()[%thread_id_x]
                %176 = arith.muli %175, %c68032 overflow<nsw> : index
                %177 = arith.addi %176, %86 overflow<nsw> : index
                %178 = arith.select %133, %177, %c536870911 : index
                vector.store %174, %142[%178] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %179 = vector.extract_strided_slice %118 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %180 = affine.apply #map37()[%thread_id_x]
                %181 = arith.muli %180, %c68032 overflow<nsw> : index
                %182 = arith.addi %181, %86 overflow<nsw> : index
                %183 = arith.select %133, %182, %c536870911 : index
                vector.store %179, %142[%183] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %184 = vector.extract_strided_slice %118 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %185 = affine.apply #map38()[%thread_id_x]
                %186 = arith.muli %185, %c68032 overflow<nsw> : index
                %187 = arith.addi %186, %86 overflow<nsw> : index
                %188 = arith.select %133, %187, %c536870911 : index
                vector.store %184, %142[%188] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %189 = vector.extract_strided_slice %118 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %190 = affine.apply #map39()[%thread_id_x]
                %191 = arith.muli %190, %c68032 overflow<nsw> : index
                %192 = arith.addi %191, %86 overflow<nsw> : index
                %193 = arith.select %133, %192, %c536870911 : index
                vector.store %189, %142[%193] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %194 = vector.extract_strided_slice %118 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %195 = affine.apply #map40()[%thread_id_x]
                %196 = arith.muli %195, %c68032 overflow<nsw> : index
                %197 = arith.addi %196, %86 overflow<nsw> : index
                %198 = arith.select %133, %197, %c536870911 : index
                vector.store %194, %142[%198] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %199 = vector.extract_strided_slice %118 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %200 = affine.apply #map41()[%thread_id_x]
                %201 = arith.muli %200, %c68032 overflow<nsw> : index
                %202 = arith.addi %201, %86 overflow<nsw> : index
                %203 = arith.select %133, %202, %c536870911 : index
                vector.store %199, %142[%203] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %204 = vector.extract_strided_slice %118 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %205 = affine.apply #map42()[%thread_id_x]
                %206 = arith.muli %205, %c68032 overflow<nsw> : index
                %207 = arith.addi %206, %86 overflow<nsw> : index
                %208 = arith.select %133, %207, %c536870911 : index
                vector.store %204, %142[%208] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %209 = vector.extract_strided_slice %118 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %210 = affine.apply #map43()[%thread_id_x]
                %211 = arith.muli %210, %c68032 overflow<nsw> : index
                %212 = arith.addi %211, %86 overflow<nsw> : index
                %213 = arith.select %133, %212, %c536870911 : index
                vector.store %209, %142[%213] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %214 = vector.extract_strided_slice %118 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %215 = affine.apply #map44()[%thread_id_x]
                %216 = arith.muli %215, %c68032 overflow<nsw> : index
                %217 = arith.addi %216, %86 overflow<nsw> : index
                %218 = arith.select %133, %217, %c536870911 : index
                vector.store %214, %142[%218] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %219 = vector.extract_strided_slice %122 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %220 = affine.apply #map45()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %221 = arith.cmpi slt, %220, %c68032 : index
                %222 = arith.addi %138, %95 overflow<nsw> : index
                %223 = arith.select %221, %222, %c536870911 : index
                vector.store %219, %142[%223] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %224 = vector.extract_strided_slice %122 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %225 = arith.addi %146, %95 overflow<nsw> : index
                %226 = arith.select %221, %225, %c536870911 : index
                vector.store %224, %142[%226] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %227 = vector.extract_strided_slice %122 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %228 = arith.addi %151, %95 overflow<nsw> : index
                %229 = arith.select %221, %228, %c536870911 : index
                vector.store %227, %142[%229] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %230 = vector.extract_strided_slice %122 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %231 = arith.addi %156, %95 overflow<nsw> : index
                %232 = arith.select %221, %231, %c536870911 : index
                vector.store %230, %142[%232] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %233 = vector.extract_strided_slice %122 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %234 = arith.addi %161, %95 overflow<nsw> : index
                %235 = arith.select %221, %234, %c536870911 : index
                vector.store %233, %142[%235] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %236 = vector.extract_strided_slice %122 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %237 = arith.addi %166, %95 overflow<nsw> : index
                %238 = arith.select %221, %237, %c536870911 : index
                vector.store %236, %142[%238] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %239 = vector.extract_strided_slice %122 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %240 = arith.addi %171, %95 overflow<nsw> : index
                %241 = arith.select %221, %240, %c536870911 : index
                vector.store %239, %142[%241] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %242 = vector.extract_strided_slice %122 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %243 = arith.addi %176, %95 overflow<nsw> : index
                %244 = arith.select %221, %243, %c536870911 : index
                vector.store %242, %142[%244] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %245 = vector.extract_strided_slice %122 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %246 = arith.addi %181, %95 overflow<nsw> : index
                %247 = arith.select %221, %246, %c536870911 : index
                vector.store %245, %142[%247] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %248 = vector.extract_strided_slice %122 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %249 = arith.addi %186, %95 overflow<nsw> : index
                %250 = arith.select %221, %249, %c536870911 : index
                vector.store %248, %142[%250] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %251 = vector.extract_strided_slice %122 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %252 = arith.addi %191, %95 overflow<nsw> : index
                %253 = arith.select %221, %252, %c536870911 : index
                vector.store %251, %142[%253] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %254 = vector.extract_strided_slice %122 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %255 = arith.addi %196, %95 overflow<nsw> : index
                %256 = arith.select %221, %255, %c536870911 : index
                vector.store %254, %142[%256] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %257 = vector.extract_strided_slice %122 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %258 = arith.addi %201, %95 overflow<nsw> : index
                %259 = arith.select %221, %258, %c536870911 : index
                vector.store %257, %142[%259] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %260 = vector.extract_strided_slice %122 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %261 = arith.addi %206, %95 overflow<nsw> : index
                %262 = arith.select %221, %261, %c536870911 : index
                vector.store %260, %142[%262] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %263 = vector.extract_strided_slice %122 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %264 = arith.addi %211, %95 overflow<nsw> : index
                %265 = arith.select %221, %264, %c536870911 : index
                vector.store %263, %142[%265] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %266 = vector.extract_strided_slice %122 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %267 = arith.addi %216, %95 overflow<nsw> : index
                %268 = arith.select %221, %267, %c536870911 : index
                vector.store %266, %142[%268] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %269 = vector.extract_strided_slice %126 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %270 = affine.apply #map46()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %271 = arith.cmpi slt, %270, %c68032 : index
                %272 = arith.addi %138, %100 overflow<nsw> : index
                %273 = arith.select %271, %272, %c536870911 : index
                vector.store %269, %142[%273] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %274 = vector.extract_strided_slice %126 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %275 = arith.addi %146, %100 overflow<nsw> : index
                %276 = arith.select %271, %275, %c536870911 : index
                vector.store %274, %142[%276] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %277 = vector.extract_strided_slice %126 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %278 = arith.addi %151, %100 overflow<nsw> : index
                %279 = arith.select %271, %278, %c536870911 : index
                vector.store %277, %142[%279] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %280 = vector.extract_strided_slice %126 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %281 = arith.addi %156, %100 overflow<nsw> : index
                %282 = arith.select %271, %281, %c536870911 : index
                vector.store %280, %142[%282] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %283 = vector.extract_strided_slice %126 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %284 = arith.addi %161, %100 overflow<nsw> : index
                %285 = arith.select %271, %284, %c536870911 : index
                vector.store %283, %142[%285] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %286 = vector.extract_strided_slice %126 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %287 = arith.addi %166, %100 overflow<nsw> : index
                %288 = arith.select %271, %287, %c536870911 : index
                vector.store %286, %142[%288] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %289 = vector.extract_strided_slice %126 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %290 = arith.addi %171, %100 overflow<nsw> : index
                %291 = arith.select %271, %290, %c536870911 : index
                vector.store %289, %142[%291] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %292 = vector.extract_strided_slice %126 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %293 = arith.addi %176, %100 overflow<nsw> : index
                %294 = arith.select %271, %293, %c536870911 : index
                vector.store %292, %142[%294] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %295 = vector.extract_strided_slice %126 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %296 = arith.addi %181, %100 overflow<nsw> : index
                %297 = arith.select %271, %296, %c536870911 : index
                vector.store %295, %142[%297] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %298 = vector.extract_strided_slice %126 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %299 = arith.addi %186, %100 overflow<nsw> : index
                %300 = arith.select %271, %299, %c536870911 : index
                vector.store %298, %142[%300] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %301 = vector.extract_strided_slice %126 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %302 = arith.addi %191, %100 overflow<nsw> : index
                %303 = arith.select %271, %302, %c536870911 : index
                vector.store %301, %142[%303] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %304 = vector.extract_strided_slice %126 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %305 = arith.addi %196, %100 overflow<nsw> : index
                %306 = arith.select %271, %305, %c536870911 : index
                vector.store %304, %142[%306] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %307 = vector.extract_strided_slice %126 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %308 = arith.addi %201, %100 overflow<nsw> : index
                %309 = arith.select %271, %308, %c536870911 : index
                vector.store %307, %142[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %310 = vector.extract_strided_slice %126 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %311 = arith.addi %206, %100 overflow<nsw> : index
                %312 = arith.select %271, %311, %c536870911 : index
                vector.store %310, %142[%312] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %313 = vector.extract_strided_slice %126 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %314 = arith.addi %211, %100 overflow<nsw> : index
                %315 = arith.select %271, %314, %c536870911 : index
                vector.store %313, %142[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %316 = vector.extract_strided_slice %126 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %317 = arith.addi %216, %100 overflow<nsw> : index
                %318 = arith.select %271, %317, %c536870911 : index
                vector.store %316, %142[%318] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %319 = vector.extract_strided_slice %130 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %320 = affine.apply #map47()[%thread_id_x, %block_id_x, %block_id_y, %4, %6, %thread_id_y]
                %321 = arith.cmpi slt, %320, %c68032 : index
                %322 = arith.addi %138, %105 overflow<nsw> : index
                %323 = arith.select %321, %322, %c536870911 : index
                vector.store %319, %142[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %324 = vector.extract_strided_slice %130 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %325 = arith.addi %146, %105 overflow<nsw> : index
                %326 = arith.select %321, %325, %c536870911 : index
                vector.store %324, %142[%326] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %327 = vector.extract_strided_slice %130 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %328 = arith.addi %151, %105 overflow<nsw> : index
                %329 = arith.select %321, %328, %c536870911 : index
                vector.store %327, %142[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %330 = vector.extract_strided_slice %130 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %331 = arith.addi %156, %105 overflow<nsw> : index
                %332 = arith.select %321, %331, %c536870911 : index
                vector.store %330, %142[%332] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %333 = vector.extract_strided_slice %130 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %334 = arith.addi %161, %105 overflow<nsw> : index
                %335 = arith.select %321, %334, %c536870911 : index
                vector.store %333, %142[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %336 = vector.extract_strided_slice %130 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %337 = arith.addi %166, %105 overflow<nsw> : index
                %338 = arith.select %321, %337, %c536870911 : index
                vector.store %336, %142[%338] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %339 = vector.extract_strided_slice %130 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %340 = arith.addi %171, %105 overflow<nsw> : index
                %341 = arith.select %321, %340, %c536870911 : index
                vector.store %339, %142[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %342 = vector.extract_strided_slice %130 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %343 = arith.addi %176, %105 overflow<nsw> : index
                %344 = arith.select %321, %343, %c536870911 : index
                vector.store %342, %142[%344] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %345 = vector.extract_strided_slice %130 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %346 = arith.addi %181, %105 overflow<nsw> : index
                %347 = arith.select %321, %346, %c536870911 : index
                vector.store %345, %142[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %348 = vector.extract_strided_slice %130 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %349 = arith.addi %186, %105 overflow<nsw> : index
                %350 = arith.select %321, %349, %c536870911 : index
                vector.store %348, %142[%350] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %351 = vector.extract_strided_slice %130 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %352 = arith.addi %191, %105 overflow<nsw> : index
                %353 = arith.select %321, %352, %c536870911 : index
                vector.store %351, %142[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %354 = vector.extract_strided_slice %130 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %355 = arith.addi %196, %105 overflow<nsw> : index
                %356 = arith.select %321, %355, %c536870911 : index
                vector.store %354, %142[%356] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %357 = vector.extract_strided_slice %130 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %358 = arith.addi %201, %105 overflow<nsw> : index
                %359 = arith.select %321, %358, %c536870911 : index
                vector.store %357, %142[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %360 = vector.extract_strided_slice %130 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %361 = arith.addi %206, %105 overflow<nsw> : index
                %362 = arith.select %321, %361, %c536870911 : index
                vector.store %360, %142[%362] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %363 = vector.extract_strided_slice %130 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %364 = arith.addi %211, %105 overflow<nsw> : index
                %365 = arith.select %321, %364, %c536870911 : index
                vector.store %363, %142[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %366 = vector.extract_strided_slice %130 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %367 = arith.addi %216, %105 overflow<nsw> : index
                %368 = arith.select %321, %367, %c536870911 : index
                vector.store %366, %142[%368] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<68032x1024xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x68032xf32>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xbf16>, tensor<68032x1024xbf16>, tensor<256x68032xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x68032xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x68032xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
        """
    asm_debug= """
        #map = affine_map<()[s0, s1] -> ((s0 * 2 + s1) mod 8)>
        #map1 = affine_map<()[s0, s1, s2] -> (((s0 * 132 + s1 * 66 + s2 - ((s0 * 2 + s1) floordiv 8) * 527) floordiv 4256) * -16 + 2)>
        #map2 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + ((s2 * 132 + s3 * 66 + s4 - ((s2 * 2 + s3) floordiv 8) * 527) floordiv 4256) * 2048 + (((s2 * 132 + s3 * 66 + s5 - ((s2 * 2 + s3) floordiv 8) * 527) mod 4256) mod s6) * 128)>
        #map3 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 8) * 64)>
        #map4 = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + ((s2 * 132 + s3 * 66 + s4 - ((s2 * 2 + s3) floordiv 8) * 527) floordiv 4256) * 2048 + (((s2 * 132 + s3 * 66 + s5 - ((s2 * 2 + s3) floordiv 8) * 527) mod 4256) mod s6) * 128 + 64)>
        #map5 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256)>
        #map6 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 64)>
        #map7 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 128)>
        #map8 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + (((s2 * 66 + s3 * 132 + s4 - ((s2 + s3 * 2) floordiv 8) * 527) mod 4256) floordiv s5) * 256 + 192)>
        #map9 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 128)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>
        #map11 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
        #map12 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map13 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map14 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map16 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
        #map17 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8)>
        #map18 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 16)>
        #map19 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32)>
        #map20 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 32)>
        #map21 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 64)>
        #map22 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 32) * 32 + 96)>
        #map23 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 32)>
        #map24 = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 8 + 48)>
        #map25 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 - (s1 floordiv 8) * 64 + 64)>
        #map26 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256)>
        #map27 = affine_map<()[s0, s1, s2, s3, s4] -> (((s0 * 132 + s1 * 66 + s2 - ((s0 * 2 + s1) floordiv 8) * 527) floordiv 4256) * 2048 + (((s0 * 132 + s1 * 66 + s3 - ((s0 * 2 + s1) floordiv 8) * 527) mod 4256) mod s4) * 128)>
        #map28 = affine_map<()[s0, s1, s2, s3] -> ((((s0 * 66 + s1 * 132 + s2 - ((s0 + s1 * 2) floordiv 8) * 527) mod 4256) floordiv s3) * 256)>
        #map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
        #map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
        #map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
        #map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
        #map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
        #map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
        #map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
        #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
        #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
        #map45 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 32)>
        #map46 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 64)>
        #map47 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 mod 32 + s5 * 128 + (((s1 * 66 + s2 * 132 + s3 - ((s1 + s2 * 2) floordiv 8) * 527) mod 4256) floordiv s4) * 256 + 96)>
        #map_wave_offset = affine_map<()[s0, s1] -> ((s1 * 4 + s0 floordiv 64) * 8)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c2 = arith.constant 2 : index
            %c266 = arith.constant 266 : index
            %c1 = arith.constant 1 : index
            stream.return %c2, %c266, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
                %cst_0 = arith.constant dense<1073741823> : vector<8xindex>
                %c1024_i14 = arith.constant 1024 : i14
                %c536870911 = arith.constant 536870911 : index
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c536870910 = arith.constant 536870910 : index
                %c0_i32 = arith.constant 0 : i32
                %c15 = arith.constant 15 : index
                %c68032 = arith.constant 68032 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c1024 = arith.constant 1024 : index
                %c1 = arith.constant 1 : index
                %c4 = arith.constant 4 : index
                %c34816 = arith.constant 34816 : index
                %cst_1 = arith.constant dense<0.000000e+00> : vector<16xf32>
                %c0 = arith.constant 0 : index
                %c64 = arith.constant 64 : index
                %c128 = arith.constant 128 : index
                %c192 = arith.constant 192 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 1
                %block_id_y = gpu.block_id  y upper_bound 1
                %thread_id_x = gpu.thread_id  x upper_bound 64
                %thread_id_y = gpu.thread_id  y upper_bound 1
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<bf16> to memref<256x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_2 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [68032, 1024], strides: [1024, 1] : memref<bf16> to memref<68032x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_3 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 68032], strides: [68032, 1] : memref<f32> to memref<256x68032xf32, strided<[68032, 1], offset: ?>>
                %alloc = memref.alloc() : memref<52224xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c0][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<256x68xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c34816][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<128x68xbf16, #gpu.address_space<workgroup>>
                %tile4 = memref.subview %view_4[0, 0] [128, 64] [1, 1] : memref<128x68xbf16, #gpu.address_space<workgroup>> to memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>
        
                %3 = affine.apply #map()[%block_id_y, %block_id_x]
                %4 = arith.minsi %3, %c4 : index
                %5 = affine.apply #map1()[%block_id_y, %block_id_x, %4]
                %6 = arith.maxsi %5, %c1 : index
                %7 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
                %8 = affine.apply #map3()[%thread_id_x]
                %9 = arith.muli %7, %c1024 overflow<nsw> : index
                %10 = arith.addi %9, %8 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %11 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %66 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                
                %67 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                //%12 = vector.load %11[%10] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                //vector.store %12, %view_4[%66, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %wave_offset = affine.apply #map_wave_offset()[%thread_id_x, %thread_id_y]
                %c65 = arith.constant 65 : index
                amdgpu.gather_to_lds %11[%10], %tile4[%wave_offset, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                //%13 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y, %block_id_x, %4, %4, %6]
               // %14 = arith.muli %13, %c1024 overflow<nsw> : index
                //%15 = arith.addi %14, %8 overflow<nsw> : index
                //%16 = vector.load %11[%15] : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
                //vector.store %16, %view_4[%67, %8] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
               // %wave_offset64= arith.addi %wave_offset, %c64 overflow<nsw> : index
                //amdgpu.gather_to_lds %11[%15], %tile4[%wave_offset64, %c0] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, strided<[68, 1]>, #gpu.address_space<workgroup>>

                  // Read back from LDS
                %loaded_from_lds = vector.load %view_4[%thread_id_x, %c0] : memref<128x68xbf16, #gpu.address_space<workgroup>>, vector<68xbf16>
                
                // Reinterpret output as 2D: [128, 1024] 
                //%reinterpret_cast_out = memref.reinterpret_cast %1 to offset: [%c0], sizes: [128, 1024], strides: [1024, 1] : memref<bf16> to memref<128x1024xbf16, strided<[1024, 1], offset: ?>>
                
                // Write to global output (using wave_offset as row, column based on thread)
                vector.store %loaded_from_lds, %reinterpret_cast_2[%thread_id_x, %c0] : memref<68032x1024xbf16, strided<[1024, 1], offset: ?>>, vector<68xbf16>
                
                
               
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<68032x1024xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x68032xf32>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xbf16>, tensor<68032x1024xbf16>, tensor<256x68032xf32>) -> %1
            %4 = hal.tensor.barrier join(%3 : tensor<68032x1024xbf16>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<68032x1024xbf16> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
    
    """
    ##Vectorized stores by first stoing to LDS and then to
    asm_vectorstore="""
        #map = affine_map<()[s0, s1] -> (((s0 * 133 + s1 * 266 - ((s0 + s1 * 2) floordiv 8) * 1063) floordiv 8512) * -16 + 2)>
        #map1 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + ((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) floordiv 8512) * 2048 + (((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) mod 8512) mod s4) * 128)>
        #map2 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map3 = affine_map<()[s0] -> (s0 mod 8)>
        #map4 = affine_map<()[s0] -> (s0 * 8)>
        #map5 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 128)>
        #map6 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + ((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) floordiv 8512) * 2048 + (((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) mod 8512) mod s4) * 128 + 64)>
        #map7 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 128 + 64)>
        #map8 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + (((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) mod 8512) floordiv s4) * 128)>
        #map9 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + (((s2 * 133 + s3 * 266 - ((s2 + s3 * 2) floordiv 8) * 1063) mod 8512) floordiv s4) * 128 + 64)>
        #map10 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 32) * 32)>
        #map11 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32)>
        #map12 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 2)>
        #map13 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 4)>
        #map14 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 6)>
        #map15 = affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 32) * 32 + 32)>
        #map16 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
        #map17 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 + 64)>
        #map18 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 mod 32 + s4 * 64 + (((s1 * 133 + s2 * 266 - ((s1 + s2 * 2) floordiv 8) * 1063) mod 8512) floordiv s3) * 128)>
        #map19 = affine_map<()[s0, s1, s2] -> (((s0 * 133 + s1 * 266 - ((s0 + s1 * 2) floordiv 8) * 1063) floordiv 8512) * 2048 + (((s0 * 133 + s1 * 266 - ((s0 + s1 * 2) floordiv 8) * 1063) mod 8512) mod s2) * 128)>
        #map20 = affine_map<()[s0, s1, s2] -> ((((s0 * 133 + s1 * 266 - ((s0 + s1 * 2) floordiv 8) * 1063) mod 8512) floordiv s2) * 128)>
        #map21 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
        #map22 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
        #map23 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
        #map24 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
        #map25 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
        #map26 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
        #map27 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
        #map28 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
        #map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
        #map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
        #map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
        #map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
        #map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
        #map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
        #map35 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
        #intermediaterow = affine_map<()[s0, s1] -> (s0 floordiv 4 + s1 * 64)>
        #intermediatecol = affine_map<()[s0, s1] -> ((s0 mod 4) * 32)>
        #map37 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 mod 32 + s4 * 64 + (((s1 * 133 + s2 * 266 - ((s1 + s2 * 2) floordiv 8) * 1063) mod 8512) floordiv s3) * 128 + 32)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c2 = arith.constant 2 : index
            %c532 = arith.constant 532 : index
            %c1 = arith.constant 1 : index
            stream.return %c2, %c532, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c1024_i14 = arith.constant 1024 : i14
                %c0_i32 = arith.constant 0 : i32
                %c536870911 = arith.constant 536870911 : index
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c536870910 = arith.constant 536870910 : index
                %c15 = arith.constant 15 : index
                %c1073741823 = arith.constant 1073741823 : index
                %c68032 = arith.constant 68032 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c1024 = arith.constant 1024 : index
                %c1 = arith.constant 1 : index
                %c98304 = arith.constant 98304 : index
                %c73728 = arith.constant 73728 : index
                %c49152 = arith.constant 49152 : index
                %c24576 = arith.constant 24576 : index
                %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 2
                %block_id_y = gpu.block_id  y upper_bound 532
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 1024], strides: [1024, 1] : memref<bf16> to memref<256x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [68032, 1024], strides: [1024, 1] : memref<bf16> to memref<68032x1024xbf16, strided<[1024, 1], offset: ?>>
                %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 68032], strides: [68032, 1] : memref<f32> to memref<256x68032xf32, strided<[68032, 1], offset: ?>>
                %alloc = memref.alloc() : memref<131584xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c0][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %view_2 = memref.view %alloc[%c24576][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %view_3 = memref.view %alloc[%c49152][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c73728][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %shared_output = memref.view %alloc[%c98304][] : memref<131584xi8, #gpu.address_space<workgroup>> to memref<128x130xf16, #gpu.address_space<workgroup>>

                %3 = affine.apply #map()[%block_id_x, %block_id_y]
                %4 = arith.maxsi %3, %c1 : index
                %5 = affine.apply #map1()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %6 = affine.apply #map2()[%thread_id_x]
                %7 = affine.apply #map3()[%thread_id_x]
                %8 = arith.xori %7, %6 : index
                %9 = affine.apply #map4()[%8]
                %10 = affine.apply #map5()[%thread_id_x, %thread_id_y]
                %11 = arith.index_cast %10 : index to i32
                %12 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %13 = arith.index_cast %12 : i32 to index
                %14 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %15 = arith.index_cast %14 : i32 to index
                %16 = arith.muli %5, %c1024 overflow<nsw> : index
                %17 = arith.addi %16, %9 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %18 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %18[%17], %view_4[%13, %15] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %19 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %20 = affine.apply #map7()[%thread_id_x, %thread_id_y]
                %21 = arith.index_cast %20 : index to i32
                %22 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %23 = arith.index_cast %22 : i32 to index
                %24 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %25 = arith.index_cast %24 : i32 to index
                %26 = arith.muli %19, %c1024 overflow<nsw> : index
                %27 = arith.addi %26, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%27], %view_4[%23, %25] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %28 = affine.apply #map8()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %29 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %30 = arith.index_cast %29 : i32 to index
                %31 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %32 = arith.index_cast %31 : i32 to index
                %33 = arith.muli %28, %c1024 overflow<nsw> : index
                %34 = arith.addi %33, %9 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<68032x1024xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %35 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %36 = arith.cmpi slt, %28, %c68032 : index
                %37 = arith.select %36, %34, %c1073741823 : index
                amdgpu.gather_to_lds %35[%37], %view_2[%30, %32] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %38 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %39 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %40 = arith.index_cast %39 : i32 to index
                %41 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %42 = arith.index_cast %41 : i32 to index
                %43 = arith.muli %38, %c1024 overflow<nsw> : index
                %44 = arith.addi %43, %9 overflow<nsw> : index
                %45 = arith.cmpi slt, %38, %c68032 : index
                %46 = arith.select %45, %44, %c1073741823 : index
                amdgpu.gather_to_lds %35[%46], %view_2[%40, %42] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %47 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %48 = arith.index_cast %47 : i32 to index
                %49 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %50 = arith.index_cast %49 : i32 to index
                %51 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %52 = arith.index_cast %51 : i32 to index
                %53 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %54 = arith.index_cast %53 : i32 to index
                %55 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %56 = arith.index_cast %55 : i32 to index
                %57 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %58 = arith.index_cast %57 : i32 to index
                %59 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %60 = arith.index_cast %59 : i32 to index
                %61 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %62 = arith.index_cast %61 : i32 to index
                %63 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %64 = affine.apply #map11()[%thread_id_x]
                %65 = arith.xori %64, %7 : index
                %66 = affine.apply #map4()[%65]
                %67 = affine.apply #map12()[%thread_id_x]
                %68 = arith.xori %67, %7 : index
                %69 = affine.apply #map4()[%68]
                %70 = affine.apply #map13()[%thread_id_x]
                %71 = arith.xori %70, %7 : index
                %72 = affine.apply #map4()[%71]
                %73 = affine.apply #map14()[%thread_id_x]
                %74 = arith.xori %73, %7 : index
                %75 = affine.apply #map4()[%74]
                %76 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %77 = affine.apply #map16()[%thread_id_x]
                %78:6 = scf.for %arg3 = %c0 to %c15 step %c1 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %view_4, %arg7 = %view_3, %arg8 = %view_2, %arg9 = %view) -> (vector<16xf32>, vector<16xf32>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>) {
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %252 = vector.load %arg8[%63, %66] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %253 = vector.load %arg8[%63, %69] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %254 = vector.load %arg8[%63, %72] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %255 = vector.load %arg8[%63, %75] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %256 = vector.load %arg8[%76, %66] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %257 = vector.load %arg8[%76, %69] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %258 = vector.load %arg8[%76, %72] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %259 = vector.load %arg8[%76, %75] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %260 = vector.load %arg6[%77, %66] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %261 = vector.load %arg6[%77, %69] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %262 = vector.load %arg6[%77, %72] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %263 = vector.load %arg6[%77, %75] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                amdgpu.lds_barrier
                %264 = affine.apply #map17()[%arg3, %8]
                %265 = arith.addi %16, %264 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%265], %arg7[%48, %50] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %266 = arith.addi %26, %264 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%266], %arg7[%52, %54] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %267 = arith.addi %33, %264 overflow<nsw> : index
                %268 = arith.select %36, %267, %c1073741823 : index
                amdgpu.gather_to_lds %35[%268], %arg9[%56, %58] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %269 = arith.addi %43, %264 overflow<nsw> : index
                %270 = arith.select %45, %269, %c1073741823 : index
                amdgpu.gather_to_lds %35[%270], %arg9[%60, %62] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %271 = amdgpu.mfma %260 * %252 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %272 = amdgpu.mfma %261 * %253 + %271 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %273 = amdgpu.mfma %262 * %254 + %272 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %274 = amdgpu.mfma %263 * %255 + %273 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %275 = amdgpu.mfma %260 * %256 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %276 = amdgpu.mfma %261 * %257 + %275 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %277 = amdgpu.mfma %262 * %258 + %276 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %278 = amdgpu.mfma %263 * %259 + %277 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                scf.yield %274, %278, %arg7, %arg6, %arg9, %arg8 : vector<16xf32>, vector<16xf32>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                }
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %79 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %80 = affine.apply #map11()[%thread_id_x]
                %81 = arith.xori %80, %7 : index
                %82 = affine.apply #map4()[%81]
                %83 = vector.load %78#4[%79, %82] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %84 = affine.apply #map12()[%thread_id_x]
                %85 = arith.xori %84, %7 : index
                %86 = affine.apply #map4()[%85]
                %87 = vector.load %78#4[%79, %86] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %88 = affine.apply #map13()[%thread_id_x]
                %89 = arith.xori %88, %7 : index
                %90 = affine.apply #map4()[%89]
                %91 = vector.load %78#4[%79, %90] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %92 = affine.apply #map14()[%thread_id_x]
                %93 = arith.xori %92, %7 : index
                %94 = affine.apply #map4()[%93]
                %95 = vector.load %78#4[%79, %94] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %96 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %97 = vector.load %78#4[%96, %82] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %98 = vector.load %78#4[%96, %86] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %99 = vector.load %78#4[%96, %90] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %100 = vector.load %78#4[%96, %94] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %101 = affine.apply #map16()[%thread_id_x]
                %102 = vector.load %78#2[%101, %82] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %103 = vector.load %78#2[%101, %86] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %104 = vector.load %78#2[%101, %90] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %105 = vector.load %78#2[%101, %94] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %106 = amdgpu.mfma %102 * %83 + %78#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %107 = amdgpu.mfma %103 * %87 + %106 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %108 = amdgpu.mfma %104 * %91 + %107 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %109 = amdgpu.mfma %105 * %95 + %108 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %110 = amdgpu.mfma %102 * %97 + %78#1 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %111 = amdgpu.mfma %103 * %98 + %110 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %112 = amdgpu.mfma %104 * %99 + %111 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %113 = amdgpu.mfma %105 * %100 + %112 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %114 = vector.extract_strided_slice %109 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %115 = affine.apply #map18()[%thread_id_x, %block_id_x, %block_id_y, %4, %thread_id_y]
                %116 = arith.cmpi slt, %115, %c68032 : index
                %117 = affine.apply #map19()[%block_id_x, %block_id_y, %4]
                %118 = affine.apply #map20()[%block_id_x, %block_id_y, %4]
                %119 = affine.apply #map21()[%thread_id_x]
                %120 = arith.muli %117, %c68032 overflow<nsw> : index
                %121 = arith.muli %119, %c68032 overflow<nsw> : index
                %122 = arith.addi %120, %118 overflow<nsw> : index
                %123 = arith.addi %121, %79 overflow<nsw> : index
                %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %reinterpret_cast_1 : memref<256x68032xf32, strided<[68032, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
                %124 = arith.addi %122, %offset_12 overflow<nsw> : index
                %reinterpret_cast_15 = memref.reinterpret_cast %2 to offset: [%124], sizes: [%c536870910], strides: [1] : memref<f32> to memref<?xf32, strided<[1], offset: ?>>
                %125 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_15 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                %126 = arith.select %116, %123, %c536870911 : index

                %114_t = arith.truncf %114 : vector<1xf32> to vector<1xf16>
                vector.store %114_t, %shared_output[%119, %79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %127 = vector.extract_strided_slice %109 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %128 = affine.apply #map22()[%thread_id_x]
                %127_t = arith.truncf %127 : vector<1xf32> to vector<1xf16>
                vector.store %127_t, %shared_output[%128, %79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %132 = vector.extract_strided_slice %109 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %133 = affine.apply #map23()[%thread_id_x]
                %132_t = arith.truncf %132 : vector<1xf32> to vector<1xf16>
                vector.store %132_t, %shared_output[%133, %79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>


                %137 = vector.extract_strided_slice %109 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %138 = affine.apply #map24()[%thread_id_x]
                %137_t = arith.truncf %137 : vector<1xf32> to vector<1xf16>
                vector.store %137_t, %shared_output[%138,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>


                %142 = vector.extract_strided_slice %109 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %143 = affine.apply #map25()[%thread_id_x]
                %142_t = arith.truncf %142 : vector<1xf32> to vector<1xf16>
                vector.store %142_t, %shared_output[%143,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %147 = vector.extract_strided_slice %109 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %148 = affine.apply #map26()[%thread_id_x]

                %147_t = arith.truncf %147 : vector<1xf32> to vector<1xf16>
                vector.store %147_t, %shared_output[%148,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %152 = vector.extract_strided_slice %109 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %153 = affine.apply #map27()[%thread_id_x]
                %152_t = arith.truncf %152 : vector<1xf32> to vector<1xf16>
                vector.store %152_t, %shared_output[%153,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>


                %157 = vector.extract_strided_slice %109 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %158 = affine.apply #map28()[%thread_id_x]
                %157_t = arith.truncf %157 : vector<1xf32> to vector<1xf16>
                vector.store %157_t, %shared_output[%158,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %162 = vector.extract_strided_slice %109 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %163 = affine.apply #map29()[%thread_id_x]
                %162_t = arith.truncf %162 : vector<1xf32> to vector<1xf16>
                vector.store %162_t, %shared_output[%163,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>


                %167 = vector.extract_strided_slice %109 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %168 = affine.apply #map30()[%thread_id_x]
                %167_t = arith.truncf %167 : vector<1xf32> to vector<1xf16>
                vector.store %167_t, %shared_output[%168,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %172 = vector.extract_strided_slice %109 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %173 = affine.apply #map31()[%thread_id_x]
                %172_t = arith.truncf %172 : vector<1xf32> to vector<1xf16>
                vector.store %172_t, %shared_output[%173,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %177 = vector.extract_strided_slice %109 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %178 = affine.apply #map32()[%thread_id_x]
                %177_t = arith.truncf %177 : vector<1xf32> to vector<1xf16>
                vector.store %177_t, %shared_output[%178,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %182 = vector.extract_strided_slice %109 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %183 = affine.apply #map33()[%thread_id_x]
                %182_t = arith.truncf %182 : vector<1xf32> to vector<1xf16>
                vector.store %182_t, %shared_output[%183,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %187 = vector.extract_strided_slice %109 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %188 = affine.apply #map34()[%thread_id_x]
                %187_t = arith.truncf %187 : vector<1xf32> to vector<1xf16>
                vector.store %187_t, %shared_output[%188,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %192 = vector.extract_strided_slice %109 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %193 = affine.apply #map35()[%thread_id_x]
                %192_t = arith.truncf %192 : vector<1xf32> to vector<1xf16>
                vector.store %192_t, %shared_output[%193,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %197 = vector.extract_strided_slice %109 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %198 = affine.apply #map36()[%thread_id_x]
                %197_t = arith.truncf %197 : vector<1xf32> to vector<1xf16>
                vector.store %197_t , %shared_output[%198,%79] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %202 = vector.extract_strided_slice %113 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %203 = affine.apply #map37()[%thread_id_x, %block_id_x, %block_id_y, %4, %thread_id_y]
                %204 = arith.cmpi slt, %203, %c68032 : index
                %205 = arith.addi %121, %96 overflow<nsw> : index
                %206 = arith.select %204, %205, %c536870911 : index
                %202_t = arith.truncf %202 : vector<1xf32> to vector<1xf16>
                vector.store %202_t , %shared_output[%119,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %207 = vector.extract_strided_slice %113 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %207_t = arith.truncf %207 : vector<1xf32> to vector<1xf16>
                vector.store %207_t , %shared_output[%128,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %210 = vector.extract_strided_slice %113 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %210_t = arith.truncf %210 : vector<1xf32> to vector<1xf16>
                vector.store %210_t , %shared_output[%133,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %213 = vector.extract_strided_slice %113 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>

                %213_t = arith.truncf %213 : vector<1xf32> to vector<1xf16>
                vector.store %213_t , %shared_output[%138,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %216 = vector.extract_strided_slice %113 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            
                %216_t = arith.truncf %216 : vector<1xf32> to vector<1xf16>
                vector.store %216_t, %shared_output[%143,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %219 = vector.extract_strided_slice %113 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>

                %219_t = arith.truncf %219 : vector<1xf32> to vector<1xf16>
                vector.store %219_t, %shared_output[%148,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %222 = vector.extract_strided_slice %113 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            
                %222_t = arith.truncf %222 : vector<1xf32> to vector<1xf16>
                vector.store %222_t, %shared_output[%153,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %225 = vector.extract_strided_slice %113 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>

                %225_t = arith.truncf %225 : vector<1xf32> to vector<1xf16>
                vector.store %225_t, %shared_output[%158,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %228 = vector.extract_strided_slice %113 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
        
                %228_t = arith.truncf %228 : vector<1xf32> to vector<1xf16>
                vector.store %228_t, %shared_output[%163,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %231 = vector.extract_strided_slice %113 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %231_t = arith.truncf %231 : vector<1xf32> to vector<1xf16>
                vector.store %231_t, %shared_output[%168,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %234 = vector.extract_strided_slice %113 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
            
                %234_t = arith.truncf %234 : vector<1xf32> to vector<1xf16>
                vector.store %234_t, %shared_output[%173,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %237 = vector.extract_strided_slice %113 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %237_t = arith.truncf %237 : vector<1xf32> to vector<1xf16>
                vector.store %237_t, %shared_output[%178,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %240 = vector.extract_strided_slice %113 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>

                %240_t = arith.truncf %240 : vector<1xf32> to vector<1xf16>
                vector.store %240_t, %shared_output[%183,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %243 = vector.extract_strided_slice %113 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>

                %243_t = arith.truncf %243 : vector<1xf32> to vector<1xf16>
                vector.store %243_t, %shared_output[%188,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %246 = vector.extract_strided_slice %113 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>

                %246_t = arith.truncf %246 : vector<1xf32> to vector<1xf16>
                vector.store %246_t , %shared_output[%193,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %249 = vector.extract_strided_slice %113 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %249_t = arith.truncf %249 : vector<1xf32> to vector<1xf16>

                vector.store %249_t, %shared_output[%198,%96] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<1xf16>

                %rowint = affine.apply #intermediaterow()[%thread_id_x, %thread_id_y]
                %colint = affine.apply #intermediatecol()[%thread_id_x, %thread_id_y]
              

                %250 = vector.load %shared_output[%rowint, %colint] : memref<128x130xf16, #gpu.address_space<workgroup>>, vector<32xf16>
                %data = arith.extf %250 : vector<32xf16> to vector<32xf32>
                
                %globalrow = arith.muli %rowint, %c68032 overflow<nsw> : index
                %globalindex = arith.addi %globalrow, %colint overflow<nsw> : index

                vector.store %data, %125[%globalindex] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<32xf32>

                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x1024xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<68032x1024xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x68032xf32>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x1024xbf16>, tensor<68032x1024xbf16>, tensor<256x68032xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x68032xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x68032xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }

    """

    asm_256x256x64_tile_128x64x64 = """
        #map = affine_map<()[s0, s1] -> (((s0 + s1 * 2 - ((s0 + s1 * 2) floordiv 8) * 7) floordiv 64) * -16 + 2)>
        #map1 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128 + ((s2 + s3 * 2 - ((s2 + s3 * 2) floordiv 8) * 7) floordiv 64) * 2048 + (((s2 + s3 * 2 - ((s2 + s3 * 2) floordiv 8) * 7) mod 64) mod s4) * 128)>
        #map2 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map3 = affine_map<()[s0] -> (s0 mod 8)>
        #map4 = affine_map<()[s0] -> (s0 * 8)>
        #map5 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 128)>
        #map6 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + ((s2 + s3 * 2 - ((s2 + s3 * 2) floordiv 8) * 7) floordiv 64) * 2048 + (((s2 + s3 * 2 - ((s2 + s3 * 2) floordiv 8) * 7) mod 64) mod s4) * 128 + 64)>
        #map7 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 128 + 64)>
        #map8 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 floordiv 8 + s1 * 32 - ((s1 * 32 + s0 floordiv 8) floordiv 64) * 64 + (((s2 + s3 * 2 - ((s2 + s3 * 2) floordiv 8) * 7) mod 64) floordiv s4) * 64)>
        #map9 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 64)>
        #map10 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 32) * 32)>
        #map11 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32)>
        #map12 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 2)>
        #map13 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 4)>
        #map14 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 6)>
        #map15 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
        #map16 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 + 64)>
        #map17 = affine_map<()[s0, s1, s2] -> (((s0 + s1 * 2 - ((s0 + s1 * 2) floordiv 8) * 7) floordiv 64) * 2048 + (((s0 + s1 * 2 - ((s0 + s1 * 2) floordiv 8) * 7) mod 64) mod s2) * 128)>
        #map18 = affine_map<()[s0, s1, s2] -> ((((s0 + s1 * 2 - ((s0 + s1 * 2) floordiv 8) * 7) mod 64) floordiv s2) * 64)>
        #map19 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
        #map20 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
        #map21 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
        #map22 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
        #map23 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
        #map24 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
        #map25 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
        #map26 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
        #map27 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
        #map28 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
        #map29 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
        #map30 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
        #map31 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
        #map32 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
        #map33 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
        #map34 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c2 = arith.constant 2 : index
            %c4 = arith.constant 4 : index
            %c1 = arith.constant 1 : index
            stream.return %c2, %c4, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c256_i14 = arith.constant 256 : i14
                %c232_i14 = arith.constant 232 : i14
                %c0_i32 = arith.constant 0 : i32
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c536870910 = arith.constant 536870910 : index
                %c256 = arith.constant 256 : index
                %c3 = arith.constant 3 : index
                %c1073741823 = arith.constant 1073741823 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c232 = arith.constant 232 : index
                %c1 = arith.constant 1 : index
                %c16384 = arith.constant 16384 : index
                %c40960 = arith.constant 40960 : index
                %c32768 = arith.constant 32768 : index
                %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 2
                %block_id_y = gpu.block_id  y upper_bound 4
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 256], strides: [256, 1] : memref<f32> to memref<256x256xf32, strided<[256, 1], offset: ?>>
                %alloc = memref.alloc() : memref<49152xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c32768][] : memref<49152xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_2 = memref.view %alloc[%c40960][] : memref<49152xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_3 = memref.view %alloc[%c0][] : memref<49152xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c16384][] : memref<49152xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %3 = affine.apply #map()[%block_id_x, %block_id_y]
                %4 = arith.maxsi %3, %c1 : index
                %5 = affine.apply #map1()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %6 = affine.apply #map2()[%thread_id_x]
                %7 = affine.apply #map3()[%thread_id_x]
                %8 = arith.xori %7, %6 : index
                %9 = affine.apply #map4()[%8]
                %10 = affine.apply #map5()[%thread_id_x, %thread_id_y]
                %11 = arith.index_cast %10 : index to i32
                %12 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %13 = arith.index_cast %12 : i32 to index
                %14 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %15 = arith.index_cast %14 : i32 to index
                %16 = arith.muli %5, %c232 overflow<nsw> : index
                %17 = arith.addi %16, %9 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %18 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %19 = arith.cmpi slt, %9, %c232 : index
                %20 = arith.select %19, %17, %c1073741823 : index
                amdgpu.gather_to_lds %18[%20], %view_4[%13, %15] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %21 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %22 = affine.apply #map7()[%thread_id_x, %thread_id_y]
                %23 = arith.index_cast %22 : index to i32
                %24 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%23) : (i32) -> i32
                %25 = arith.index_cast %24 : i32 to index
                %26 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %27 = arith.index_cast %26 : i32 to index
                %28 = arith.muli %21, %c232 overflow<nsw> : index
                %29 = arith.addi %28, %9 overflow<nsw> : index
                %30 = arith.select %19, %29, %c1073741823 : index
                amdgpu.gather_to_lds %18[%30], %view_4[%25, %27] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %31 = affine.apply #map8()[%thread_id_x, %thread_id_y, %block_id_x, %block_id_y, %4]
                %32 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                %33 = arith.index_cast %32 : index to i32
                %34 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%33) : (i32) -> i32
                %35 = arith.index_cast %34 : i32 to index
                %36 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %37 = arith.index_cast %36 : i32 to index
                %38 = arith.muli %31, %c232 overflow<nsw> : index
                %39 = arith.addi %38, %9 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %40 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %41 = arith.select %19, %39, %c1073741823 : index
                amdgpu.gather_to_lds %40[%41], %view_2[%35, %37] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                %42 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%11) : (i32) -> i32
                %43 = arith.index_cast %42 : i32 to index
                %44 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %45 = arith.index_cast %44 : i32 to index
                %46 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%23) : (i32) -> i32
                %47 = arith.index_cast %46 : i32 to index
                %48 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %49 = arith.index_cast %48 : i32 to index
                %50 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%33) : (i32) -> i32
                %51 = arith.index_cast %50 : i32 to index
                %52 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %53 = arith.index_cast %52 : i32 to index
                %54 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %55 = affine.apply #map11()[%thread_id_x]
                %56 = arith.xori %55, %7 : index
                %57 = affine.apply #map4()[%56]
                %58 = affine.apply #map12()[%thread_id_x]
                %59 = arith.xori %58, %7 : index
                %60 = affine.apply #map4()[%59]
                %61 = affine.apply #map13()[%thread_id_x]
                %62 = arith.xori %61, %7 : index
                %63 = affine.apply #map4()[%62]
                %64 = affine.apply #map14()[%thread_id_x]
                %65 = arith.xori %64, %7 : index
                %66 = affine.apply #map4()[%65]
                %67 = affine.apply #map15()[%thread_id_x]
                %68:5 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %cst, %arg5 = %view_4, %arg6 = %view_3, %arg7 = %view_2, %arg8 = %view) -> (vector<16xf32>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>) {
                    rocdl.s.waitcnt 16368
                    amdgpu.lds_barrier
                    %165 = vector.load %arg7[%54, %57] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %166 = vector.load %arg7[%54, %60] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %169 = vector.load %arg5[%67, %57] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %170 = vector.load %arg5[%67, %60] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %173 = affine.apply #map16()[%arg3, %8]
                    %174 = arith.addi %16, %173 overflow<nsw> : index
                    %175 = arith.cmpi slt, %173, %c232 : index
                    %176 = arith.select %175, %174, %c1073741823 : index
                    amdgpu.gather_to_lds %18[%176], %arg6[%43, %45] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                    %177 = arith.addi %28, %173 overflow<nsw> : index
                    %178 = arith.select %175, %177, %c1073741823 : index
                    amdgpu.gather_to_lds %18[%178], %arg6[%47, %49] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                    %179 = arith.addi %38, %173 overflow<nsw> : index
                    %180 = arith.select %175, %179, %c1073741823 : index
                    amdgpu.gather_to_lds %40[%180], %arg8[%51, %53] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %181 = amdgpu.mfma %169 * %165 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                    %182 = amdgpu.mfma %170 * %166 + %181 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %167 = vector.load %arg7[%54, %63] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %168 = vector.load %arg7[%54, %66] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %171 = vector.load %arg5[%67, %63] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %172 = vector.load %arg5[%67, %66] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %183 = amdgpu.mfma %171 * %167 + %182 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                    %184 = amdgpu.mfma %172 * %168 + %183 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                    scf.yield %184, %arg6, %arg5, %arg8, %arg7 : vector<16xf32>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                }
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %69 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %70 = affine.apply #map11()[%thread_id_x]
                %71 = arith.xori %70, %7 : index
                %72 = affine.apply #map4()[%71]
                %73 = vector.load %68#3[%69, %72] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %74 = affine.apply #map12()[%thread_id_x]
                %75 = arith.xori %74, %7 : index
                %76 = affine.apply #map4()[%75]
                %77 = vector.load %68#3[%69, %76] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %78 = affine.apply #map13()[%thread_id_x]
                %79 = arith.xori %78, %7 : index
                %80 = affine.apply #map4()[%79]
                %81 = vector.load %68#3[%69, %80] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %82 = affine.apply #map14()[%thread_id_x]
                %83 = arith.xori %82, %7 : index
                %84 = affine.apply #map4()[%83]
                %85 = vector.load %68#3[%69, %84] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %86 = affine.apply #map15()[%thread_id_x]
                %87 = vector.load %68#1[%86, %72] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %88 = vector.load %68#1[%86, %76] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %89 = vector.load %68#1[%86, %80] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %90 = vector.load %68#1[%86, %84] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %91 = amdgpu.mfma %87 * %73 + %68#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %92 = amdgpu.mfma %88 * %77 + %91 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %93 = amdgpu.mfma %89 * %81 + %92 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %94 = amdgpu.mfma %90 * %85 + %93 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %95 = vector.extract_strided_slice %94 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %96 = affine.apply #map17()[%block_id_x, %block_id_y, %4]
                %97 = affine.apply #map18()[%block_id_x, %block_id_y, %4]
                %98 = affine.apply #map19()[%thread_id_x]
                %99 = arith.muli %96, %c256 overflow<nsw> : index
                %100 = arith.muli %98, %c256 overflow<nsw> : index
                %101 = arith.addi %99, %97 overflow<nsw> : index
                %102 = arith.addi %100, %69 overflow<nsw> : index
                %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %reinterpret_cast_1 : memref<256x256xf32, strided<[256, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
                %103 = arith.addi %101, %offset_12 overflow<nsw> : index
                %reinterpret_cast_15 = memref.reinterpret_cast %2 to offset: [%103], sizes: [%c536870910], strides: [1] : memref<f32> to memref<?xf32, strided<[1], offset: ?>>
                %104 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_15 validBytes(%c2147483643_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                vector.store %95, %104[%102] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %105 = vector.extract_strided_slice %94 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %106 = affine.apply #map20()[%thread_id_x]
                %107 = arith.muli %106, %c256 overflow<nsw> : index
                %108 = arith.addi %107, %69 overflow<nsw> : index
                vector.store %105, %104[%108] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %109 = vector.extract_strided_slice %94 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %110 = affine.apply #map21()[%thread_id_x]
                %111 = arith.muli %110, %c256 overflow<nsw> : index
                %112 = arith.addi %111, %69 overflow<nsw> : index
                vector.store %109, %104[%112] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %113 = vector.extract_strided_slice %94 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %114 = affine.apply #map22()[%thread_id_x]
                %115 = arith.muli %114, %c256 overflow<nsw> : index
                %116 = arith.addi %115, %69 overflow<nsw> : index
                vector.store %113, %104[%116] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %117 = vector.extract_strided_slice %94 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %118 = affine.apply #map23()[%thread_id_x]
                %119 = arith.muli %118, %c256 overflow<nsw> : index
                %120 = arith.addi %119, %69 overflow<nsw> : index
                vector.store %117, %104[%120] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %121 = vector.extract_strided_slice %94 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %122 = affine.apply #map24()[%thread_id_x]
                %123 = arith.muli %122, %c256 overflow<nsw> : index
                %124 = arith.addi %123, %69 overflow<nsw> : index
                vector.store %121, %104[%124] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %125 = vector.extract_strided_slice %94 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %126 = affine.apply #map25()[%thread_id_x]
                %127 = arith.muli %126, %c256 overflow<nsw> : index
                %128 = arith.addi %127, %69 overflow<nsw> : index
                vector.store %125, %104[%128] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %129 = vector.extract_strided_slice %94 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %130 = affine.apply #map26()[%thread_id_x]
                %131 = arith.muli %130, %c256 overflow<nsw> : index
                %132 = arith.addi %131, %69 overflow<nsw> : index
                vector.store %129, %104[%132] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %133 = vector.extract_strided_slice %94 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %134 = affine.apply #map27()[%thread_id_x]
                %135 = arith.muli %134, %c256 overflow<nsw> : index
                %136 = arith.addi %135, %69 overflow<nsw> : index
                vector.store %133, %104[%136] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %137 = vector.extract_strided_slice %94 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %138 = affine.apply #map28()[%thread_id_x]
                %139 = arith.muli %138, %c256 overflow<nsw> : index
                %140 = arith.addi %139, %69 overflow<nsw> : index
                vector.store %137, %104[%140] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %141 = vector.extract_strided_slice %94 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %142 = affine.apply #map29()[%thread_id_x]
                %143 = arith.muli %142, %c256 overflow<nsw> : index
                %144 = arith.addi %143, %69 overflow<nsw> : index
                vector.store %141, %104[%144] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %145 = vector.extract_strided_slice %94 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %146 = affine.apply #map30()[%thread_id_x]
                %147 = arith.muli %146, %c256 overflow<nsw> : index
                %148 = arith.addi %147, %69 overflow<nsw> : index
                vector.store %145, %104[%148] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %149 = vector.extract_strided_slice %94 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %150 = affine.apply #map31()[%thread_id_x]
                %151 = arith.muli %150, %c256 overflow<nsw> : index
                %152 = arith.addi %151, %69 overflow<nsw> : index
                vector.store %149, %104[%152] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %153 = vector.extract_strided_slice %94 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %154 = affine.apply #map32()[%thread_id_x]
                %155 = arith.muli %154, %c256 overflow<nsw> : index
                %156 = arith.addi %155, %69 overflow<nsw> : index
                vector.store %153, %104[%156] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %157 = vector.extract_strided_slice %94 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %158 = affine.apply #map33()[%thread_id_x]
                %159 = arith.muli %158, %c256 overflow<nsw> : index
                %160 = arith.addi %159, %69 overflow<nsw> : index
                vector.store %157, %104[%160] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %161 = vector.extract_strided_slice %94 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %162 = affine.apply #map34()[%thread_id_x]
                %163 = arith.muli %162, %c256 overflow<nsw> : index
                %164 = arith.addi %163, %69 overflow<nsw> : index
                vector.store %161, %104[%164] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x232xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<256x232xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x256xf32>
            %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<256x232xbf16>, tensor<256x232xbf16>, tensor<256x256xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x256xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x256xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
    """

    asm_256x256x64_tile_128x64x64_pingpong = """
        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 128 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 8)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 128)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 128 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>
        #map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 128 + 64)>
        #map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 64) * 64)>
        #map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 64)>
        #map9 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map10 = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
        #map11 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32)>
        #map12 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 2)>
        #map13 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 32) * 32)>
        #map14 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 4)>
        #map15 = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 6)>
        #map16 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 + 64)>
        #map17 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4)>
        #map18 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 32) * 32)>
        #map19 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 1)>
        #map20 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 2)>
        #map21 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 3)>
        #map22 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 8)>
        #map23 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 9)>
        #map24 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 10)>
        #map25 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 11)>
        #map26 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 16)>
        #map27 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 17)>
        #map28 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 18)>
        #map29 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 19)>
        #map30 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 24)>
        #map31 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 25)>
        #map32 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 26)>
        #map33 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 27)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm_prefetch {
            stream.executable.export public @gemm_prefetch workgroups() -> (index, index, index) {
            %c2 = arith.constant 2 : index
            %c4 = arith.constant 4 : index
            %c1 = arith.constant 1 : index
            stream.return %c2, %c4, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm_prefetch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c232_i14 = arith.constant 232 : i14
                %c0_i32 = arith.constant 0 : i32
                %c3 = arith.constant 3 : index
                %c1073741823 = arith.constant 1073741823 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c232 = arith.constant 232 : index
                %c1 = arith.constant 1 : index
                %c16384 = arith.constant 16384 : index
                %c40960 = arith.constant 40960 : index
                %c32768 = arith.constant 32768 : index
                %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 2
                %block_id_y = gpu.block_id  y upper_bound 4
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 256], strides: [256, 1] : memref<f32> to memref<256x256xf32, strided<[256, 1], offset: ?>>
                %alloc = memref.alloc() : memref<49152xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c32768][] : memref<49152xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_2 = memref.view %alloc[%c40960][] : memref<49152xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_3 = memref.view %alloc[%c0][] : memref<49152xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c16384][] : memref<49152xi8, #gpu.address_space<workgroup>> to memref<128x64xbf16, #gpu.address_space<workgroup>>
                %3 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %4 = affine.apply #map1()[%thread_id_x]
                %5 = affine.apply #map2()[%thread_id_x]
                %6 = arith.xori %5, %4 : index
                %7 = affine.apply #map3()[%6]
                %8 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %9 = arith.index_cast %8 : index to i32
                %10 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%9) : (i32) -> i32
                %11 = arith.index_cast %10 : i32 to index
                %12 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %13 = arith.index_cast %12 : i32 to index
                %14 = arith.muli %3, %c232 overflow<nsw> : index
                %15 = arith.addi %14, %7 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %16 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %17 = arith.cmpi slt, %7, %c232 : index
                %18 = arith.select %17, %15, %c1073741823 : index
                amdgpu.gather_to_lds %16[%18], %view_4[%11, %13] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %19 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
                %20 = affine.apply #map6()[%thread_id_x, %thread_id_y]
                %21 = arith.index_cast %20 : index to i32
                %22 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %23 = arith.index_cast %22 : i32 to index
                %24 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %25 = arith.index_cast %24 : i32 to index
                %26 = arith.muli %19, %c232 overflow<nsw> : index
                %27 = arith.addi %26, %7 overflow<nsw> : index
                %28 = arith.select %17, %27, %c1073741823 : index
                amdgpu.gather_to_lds %16[%28], %view_4[%23, %25] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %29 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
                %30 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                %31 = arith.index_cast %30 : index to i32
                %32 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%31) : (i32) -> i32
                %33 = arith.index_cast %32 : i32 to index
                %34 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %35 = arith.index_cast %34 : i32 to index
                %36 = arith.muli %29, %c232 overflow<nsw> : index
                %37 = arith.addi %36, %7 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %38 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %39 = arith.select %17, %37, %c1073741823 : index
                amdgpu.gather_to_lds %38[%39], %view_2[%33, %35] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                amdgpu.lds_barrier
                %40 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                %41 = arith.index_cast %40 : index to i32
                %42 = arith.cmpi sge, %41, %c4_i32 : i32
                %43 = arith.cmpi slt, %41, %c4_i32 : i32
                scf.if %42 {
                rocdl.s.barrier
                }
                %44 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%9) : (i32) -> i32
                %45 = arith.index_cast %44 : i32 to index
                %46 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %47 = arith.index_cast %46 : i32 to index
                %48 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%21) : (i32) -> i32
                %49 = arith.index_cast %48 : i32 to index
                %50 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %51 = arith.index_cast %50 : i32 to index
                %52 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%31) : (i32) -> i32
                %53 = arith.index_cast %52 : i32 to index
                %54 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %55 = arith.index_cast %54 : i32 to index
                %56 = affine.apply #map10()[%thread_id_x]
                %57 = affine.apply #map11()[%thread_id_x]
                %58 = arith.xori %57, %5 : index
                %59 = affine.apply #map3()[%58]
                %60 = affine.apply #map12()[%thread_id_x]
                %61 = arith.xori %60, %5 : index
                %62 = affine.apply #map3()[%61]
                %63 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                %64 = affine.apply #map14()[%thread_id_x]
                %65 = arith.xori %64, %5 : index
                %66 = affine.apply #map3()[%65]
                %67 = affine.apply #map15()[%thread_id_x]
                %68 = arith.xori %67, %5 : index
                %69 = affine.apply #map3()[%68]
                %70:5 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %cst, %arg5 = %view_4, %arg6 = %view_3, %arg7 = %view_2, %arg8 = %view) -> (vector<16xf32>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>) {
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %130 = vector.load %arg5[%56, %59] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %131 = vector.load %arg5[%56, %62] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %132 = vector.load %arg7[%63, %59] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %133 = vector.load %arg7[%63, %62] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %134 = affine.apply #map16()[%arg3, %6]
                %135 = arith.addi %14, %134 overflow<nsw> : index
                %136 = arith.cmpi slt, %134, %c232 : index
                %137 = arith.select %136, %135, %c1073741823 : index
                amdgpu.gather_to_lds %16[%137], %arg6[%45, %47] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %138 = arith.addi %26, %134 overflow<nsw> : index
                %139 = arith.select %136, %138, %c1073741823 : index
                amdgpu.gather_to_lds %16[%139], %arg6[%49, %51] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<128x64xbf16, #gpu.address_space<workgroup>>
                %140 = arith.addi %36, %134 overflow<nsw> : index
                %141 = arith.select %136, %140, %c1073741823 : index
                amdgpu.gather_to_lds %38[%141], %arg8[%53, %55] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %142 = amdgpu.mfma %130 * %132 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %143 = amdgpu.mfma %131 * %133 + %142 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                rocdl.s.setprio 0
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.memory_counter_wait load(3)
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %144 = vector.load %arg5[%56, %66] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %145 = vector.load %arg5[%56, %69] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %146 = vector.load %arg7[%63, %66] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %147 = vector.load %arg7[%63, %69] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.memory_counter_wait load(0)
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %148 = amdgpu.mfma %144 * %146 + %143 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %149 = amdgpu.mfma %145 * %147 + %148 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                rocdl.s.setprio 0
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.lds_barrier
                scf.yield %149, %arg6, %arg5, %arg8, %arg7 : vector<16xf32>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<128x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                }
                scf.if %43 {
                rocdl.s.barrier
                }
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %71 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                %72 = affine.apply #map11()[%thread_id_x]
                %73 = arith.xori %72, %5 : index
                %74 = affine.apply #map3()[%73]
                %75 = vector.load %70#3[%71, %74] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %76 = affine.apply #map12()[%thread_id_x]
                %77 = arith.xori %76, %5 : index
                %78 = affine.apply #map3()[%77]
                %79 = vector.load %70#3[%71, %78] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %80 = affine.apply #map14()[%thread_id_x]
                %81 = arith.xori %80, %5 : index
                %82 = affine.apply #map3()[%81]
                %83 = vector.load %70#3[%71, %82] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %84 = affine.apply #map15()[%thread_id_x]
                %85 = arith.xori %84, %5 : index
                %86 = affine.apply #map3()[%85]
                %87 = vector.load %70#3[%71, %86] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %88 = affine.apply #map10()[%thread_id_x]
                %89 = vector.load %70#1[%88, %74] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %90 = vector.load %70#1[%88, %78] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %91 = vector.load %70#1[%88, %82] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %92 = vector.load %70#1[%88, %86] : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %93 = amdgpu.mfma %89 * %75 + %70#0 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %94 = amdgpu.mfma %90 * %79 + %93 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %95 = amdgpu.mfma %91 * %83 + %94 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %96 = amdgpu.mfma %92 * %87 + %95 {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
                %97 = vector.extract_strided_slice %96 {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %98 = affine.apply #map17()[%block_id_x, %thread_id_x]
                %99 = affine.apply #map18()[%thread_id_x, %block_id_y, %thread_id_y]
                vector.store %97, %reinterpret_cast_1[%98, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %100 = vector.extract_strided_slice %96 {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %101 = affine.apply #map19()[%block_id_x, %thread_id_x]
                vector.store %100, %reinterpret_cast_1[%101, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %102 = vector.extract_strided_slice %96 {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %103 = affine.apply #map20()[%block_id_x, %thread_id_x]
                vector.store %102, %reinterpret_cast_1[%103, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %104 = vector.extract_strided_slice %96 {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %105 = affine.apply #map21()[%block_id_x, %thread_id_x]
                vector.store %104, %reinterpret_cast_1[%105, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %106 = vector.extract_strided_slice %96 {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %107 = affine.apply #map22()[%block_id_x, %thread_id_x]
                vector.store %106, %reinterpret_cast_1[%107, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %108 = vector.extract_strided_slice %96 {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %109 = affine.apply #map23()[%block_id_x, %thread_id_x]
                vector.store %108, %reinterpret_cast_1[%109, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %110 = vector.extract_strided_slice %96 {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %111 = affine.apply #map24()[%block_id_x, %thread_id_x]
                vector.store %110, %reinterpret_cast_1[%111, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %112 = vector.extract_strided_slice %96 {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %113 = affine.apply #map25()[%block_id_x, %thread_id_x]
                vector.store %112, %reinterpret_cast_1[%113, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %114 = vector.extract_strided_slice %96 {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %115 = affine.apply #map26()[%block_id_x, %thread_id_x]
                vector.store %114, %reinterpret_cast_1[%115, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %116 = vector.extract_strided_slice %96 {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %117 = affine.apply #map27()[%block_id_x, %thread_id_x]
                vector.store %116, %reinterpret_cast_1[%117, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %118 = vector.extract_strided_slice %96 {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %119 = affine.apply #map28()[%block_id_x, %thread_id_x]
                vector.store %118, %reinterpret_cast_1[%119, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %120 = vector.extract_strided_slice %96 {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %121 = affine.apply #map29()[%block_id_x, %thread_id_x]
                vector.store %120, %reinterpret_cast_1[%121, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %122 = vector.extract_strided_slice %96 {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %123 = affine.apply #map30()[%block_id_x, %thread_id_x]
                vector.store %122, %reinterpret_cast_1[%123, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %124 = vector.extract_strided_slice %96 {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %125 = affine.apply #map31()[%block_id_x, %thread_id_x]
                vector.store %124, %reinterpret_cast_1[%125, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %126 = vector.extract_strided_slice %96 {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %127 = affine.apply #map32()[%block_id_x, %thread_id_x]
                vector.store %126, %reinterpret_cast_1[%127, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %128 = vector.extract_strided_slice %96 {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
                %129 = affine.apply #map33()[%block_id_x, %thread_id_x]
                vector.store %128, %reinterpret_cast_1[%129, %99] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x232xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<256x232xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x256xf32>
            %3 = flow.dispatch @gemm_prefetch::@gemm_prefetch(%0, %1, %2) : (tensor<256x232xbf16>, tensor<256x232xbf16>, tensor<256x256xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x256xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x256xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
    """

    asm_256x256x64_tile_64x32x64_pingpong_mfma_16x16x32= """
        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 64) * 64)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 8)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 64)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 32 - ((s1 * 8 + s0 floordiv 32) floordiv 32) * 32)>
        #map6 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 32) * 64)>
        #map7 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 2 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 32)>
        #map8 = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 32 - ((s1 * 8 + s0 floordiv 32 + 16) floordiv 32) * 32 + 16)>
        #map9 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 2 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 32 + 16)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map11 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
        #map12 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map13 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
        #map14 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 8)>
        #map15 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map16 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 8 + 32)>
        #map17 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 + 64)>
        #map18 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 2 - (s1 floordiv 32) * 64 + 64)>
        #map19 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
        #map20 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
        #map21 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
        #map22 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
        #map23 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm_prefetch {
            stream.executable.export public @gemm_prefetch workgroups() -> (index, index, index) {
            %c4 = arith.constant 4 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            stream.return %c4, %c8, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm_prefetch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c232_i14 = arith.constant 232 : i14
                %c0_i32 = arith.constant 0 : i32
                %c3 = arith.constant 3 : index
                %c1073741823 = arith.constant 1073741823 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c232 = arith.constant 232 : index
                %c1 = arith.constant 1 : index
                %c8192 = arith.constant 8192 : index
                %c20480 = arith.constant 20480 : index
                %c16384 = arith.constant 16384 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 4
                %block_id_y = gpu.block_id  y upper_bound 8
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 256], strides: [256, 1] : memref<f32> to memref<256x256xf32, strided<[256, 1], offset: ?>>
                %alloc = memref.alloc() : memref<24576xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c16384][] : memref<24576xi8, #gpu.address_space<workgroup>> to memref<32x64xbf16, #gpu.address_space<workgroup>>
                %view_2 = memref.view %alloc[%c20480][] : memref<24576xi8, #gpu.address_space<workgroup>> to memref<32x64xbf16, #gpu.address_space<workgroup>>
                %view_3 = memref.view %alloc[%c0][] : memref<24576xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c8192][] : memref<24576xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %3 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %4 = affine.apply #map1()[%thread_id_x]
                %5 = affine.apply #map2()[%thread_id_x]
                %6 = arith.xori %5, %4 : index
                %7 = affine.apply #map3()[%6]
                %8 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %9 = arith.index_cast %8 : index to i32
                %10 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%9) : (i32) -> i32
                %11 = arith.index_cast %10 : i32 to index
                %12 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %13 = arith.index_cast %12 : i32 to index
                %14 = arith.muli %3, %c232 overflow<nsw> : index
                %15 = arith.addi %14, %7 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %16 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %17 = arith.cmpi slt, %7, %c232 : index
                %18 = arith.select %17, %15, %c1073741823 : index
                amdgpu.gather_to_lds %16[%18], %view_4[%11, %13] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                %19 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                %20 = affine.apply #map6()[%thread_id_x]
                %21 = affine.apply #map7()[%thread_id_x, %thread_id_y]
                %22 = arith.index_cast %21 : index to i32
                %23 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%22) : (i32) -> i32
                %24 = arith.index_cast %23 : i32 to index
                %25 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %26 = arith.index_cast %25 : i32 to index
                %27 = arith.muli %19, %c232 overflow<nsw> : index
                %28 = arith.addi %27, %20 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %29 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %30 = arith.cmpi slt, %20, %c232 : index
                %31 = arith.select %30, %28, %c1073741823 : index
                amdgpu.gather_to_lds %29[%31], %view_2[%24, %26] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                %32 = affine.apply #map8()[%thread_id_x, %thread_id_y, %block_id_y]
                %33 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                %34 = arith.index_cast %33 : index to i32
                %35 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%34) : (i32) -> i32
                %36 = arith.index_cast %35 : i32 to index
                %37 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %38 = arith.index_cast %37 : i32 to index
                %39 = arith.muli %32, %c232 overflow<nsw> : index
                %40 = arith.addi %39, %20 overflow<nsw> : index
                %41 = arith.select %30, %40, %c1073741823 : index
                amdgpu.gather_to_lds %29[%41], %view_2[%36, %38] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                amdgpu.lds_barrier
                %42 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %43 = arith.index_cast %42 : index to i32
                %44 = arith.cmpi sge, %43, %c4_i32 : i32
                %45 = arith.cmpi slt, %43, %c4_i32 : i32
                scf.if %44 {
                rocdl.s.barrier
                }
                %46 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%9) : (i32) -> i32
                %47 = arith.index_cast %46 : i32 to index
                %48 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %49 = arith.index_cast %48 : i32 to index
                %50 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%22) : (i32) -> i32
                %51 = arith.index_cast %50 : i32 to index
                %52 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %53 = arith.index_cast %52 : i32 to index
                %54 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%34) : (i32) -> i32
                %55 = arith.index_cast %54 : i32 to index
                %56 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %57 = arith.index_cast %56 : i32 to index
                %58 = affine.apply #map11()[%thread_id_x]
                %59 = affine.apply #map12()[%thread_id_x]
                %60 = arith.xori %59, %5 : index
                %61 = affine.apply #map3()[%60]
                %62 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                %63 = affine.apply #map14()[%thread_id_x]
                %64 = affine.apply #map15()[%thread_id_x]
                %65 = arith.xori %64, %5 : index
                %66 = affine.apply #map3()[%65]
                %67 = affine.apply #map16()[%thread_id_x]
                %68:5 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %cst, %arg5 = %view_4, %arg6 = %view_3, %arg7 = %view_2, %arg8 = %view) -> (vector<4xf32>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>) {
                    rocdl.s.waitcnt 16368
                    amdgpu.lds_barrier
                    %94 = vector.load %arg5[%58, %61] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %95 = vector.load %arg7[%62, %63] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %96 = affine.apply #map17()[%arg3, %6]
                    %97 = arith.addi %14, %96 overflow<nsw> : index
                    %98 = arith.cmpi slt, %96, %c232 : index
                    %99 = arith.select %98, %97, %c1073741823 : index
                    amdgpu.gather_to_lds %16[%99], %arg6[%47, %49] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                    %100 = affine.apply #map18()[%arg3, %thread_id_x]
                    %101 = arith.addi %27, %100 overflow<nsw> : index
                    %102 = arith.cmpi slt, %100, %c232 : index
                    %103 = arith.select %102, %101, %c1073741823 : index
                    amdgpu.gather_to_lds %29[%103], %arg8[%51, %53] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                    %104 = arith.addi %39, %100 overflow<nsw> : index
                    %105 = arith.select %102, %104, %c1073741823 : index
                    amdgpu.gather_to_lds %29[%105], %arg8[%55, %57] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.barrier
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.setprio 1
                    %106 = amdgpu.mfma %94 * %95 + %arg4 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                    rocdl.s.setprio 0
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    amdgpu.memory_counter_wait load(3)
                    rocdl.s.barrier
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %107 = vector.load %arg5[%58, %66] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %108 = vector.load %arg7[%62, %67] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    amdgpu.memory_counter_wait load(0)
                    rocdl.s.barrier
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.setprio 1
                    %109 = amdgpu.mfma %107 * %108 + %106 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                    rocdl.s.setprio 0
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    amdgpu.lds_barrier
                scf.yield %109, %arg6, %arg5, %arg8, %arg7 : vector<4xf32>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                }
                scf.if %45 {
                rocdl.s.barrier
                }
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %69 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                %70 = affine.apply #map14()[%thread_id_x]
                %71 = vector.load %68#3[%69, %70] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %72 = affine.apply #map16()[%thread_id_x]
                %73 = vector.load %68#3[%69, %72] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %74 = affine.apply #map11()[%thread_id_x]
                %75 = affine.apply #map12()[%thread_id_x]
                %76 = arith.xori %75, %5 : index
                %77 = affine.apply #map3()[%76]
                %78 = vector.load %68#1[%74, %77] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %79 = affine.apply #map15()[%thread_id_x]
                %80 = arith.xori %79, %5 : index
                %81 = affine.apply #map3()[%80]
                %82 = vector.load %68#1[%74, %81] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %83 = amdgpu.mfma %78 * %71 + %68#0 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                %84 = amdgpu.mfma %82 * %73 + %83 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                %85 = vector.extract_strided_slice %84 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %86 = affine.apply #map19()[%block_id_x, %thread_id_x]
                %87 = affine.apply #map20()[%thread_id_x, %block_id_y, %thread_id_y]
                vector.store %85, %reinterpret_cast_1[%86, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %88 = vector.extract_strided_slice %84 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %89 = affine.apply #map21()[%block_id_x, %thread_id_x]
                vector.store %88, %reinterpret_cast_1[%89, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %90 = vector.extract_strided_slice %84 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %91 = affine.apply #map22()[%block_id_x, %thread_id_x]
                vector.store %90, %reinterpret_cast_1[%91, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %92 = vector.extract_strided_slice %84 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %93 = affine.apply #map23()[%block_id_x, %thread_id_x]
                vector.store %92, %reinterpret_cast_1[%93, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x232xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<256x232xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x256xf32>
            %3 = flow.dispatch @gemm_prefetch::@gemm_prefetch(%0, %1, %2) : (tensor<256x232xbf16>, tensor<256x232xbf16>, tensor<256x256xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x256xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x256xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }

    """

    asm_256x256x64_tile_64x32x64_pingpong_mfma_16x16x32_triple_buffer= """
        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 64) * 64)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 8)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 64)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 32 - ((s1 * 8 + s0 floordiv 32) floordiv 32) * 32)>
        #map6 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 32) * 64)>
        #map7 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 2 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 32)>
        #map8 = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 32 - ((s1 * 8 + s0 floordiv 32 + 16) floordiv 32) * 32 + 16)>
        #map9 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 2 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 32 + 16)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map11 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
        #map12 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map13 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
        #map14 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 8)>
        #map15 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map16 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 8 + 32)>
        #map17 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 + 64)>
        #map18 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 2 - (s1 floordiv 32) * 64 + 64)>
        #map19 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
        #map20 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
        #map21 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
        #map22 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
        #map23 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm_prefetch {
            stream.executable.export public @gemm_prefetch workgroups() -> (index, index, index) {
            %c4 = arith.constant 4 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            stream.return %c4, %c8, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm_prefetch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c232_i14 = arith.constant 232 : i14
                %c0_i32 = arith.constant 0 : i32
                %c3 = arith.constant 3 : index
                %c1073741823 = arith.constant 1073741823 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c232 = arith.constant 232 : index
                %c1 = arith.constant 1 : index
                %c2 = arith.constant 2 : index
                %c8192 = arith.constant 8192 : index
                %c20480 = arith.constant 20480 : index
                %c16384 = arith.constant 16384 : index
                %c24576 = arith.constant 24576 : index
                %c28672 = arith.constant 28672 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 4
                %block_id_y = gpu.block_id  y upper_bound 8
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 256], strides: [256, 1] : memref<f32> to memref<256x256xf32, strided<[256, 1], offset: ?>>
                %alloc = memref.alloc() : memref<36864xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c16384][] : memref<36864xi8, #gpu.address_space<workgroup>> to memref<32x64xbf16, #gpu.address_space<workgroup>>
                %view_2 = memref.view %alloc[%c20480][] : memref<36864xi8, #gpu.address_space<workgroup>> to memref<32x64xbf16, #gpu.address_space<workgroup>>
                %view_3 = memref.view %alloc[%c0][] : memref<36864xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c8192][] : memref<36864xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_5 = memref.view %alloc[%c28672][] : memref<36864xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_6 = memref.view %alloc[%c24576][] : memref<36864xi8, #gpu.address_space<workgroup>> to memref<32x64xbf16, #gpu.address_space<workgroup>>

                %3 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %4 = affine.apply #map1()[%thread_id_x]
                %5 = affine.apply #map2()[%thread_id_x]
                %6 = arith.xori %5, %4 : index
                %7 = affine.apply #map3()[%6]
                %8 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %9 = arith.index_cast %8 : index to i32
                %10 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%9) : (i32) -> i32
                %11 = arith.index_cast %10 : i32 to index
                %12 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %13 = arith.index_cast %12 : i32 to index
                %14 = arith.muli %3, %c232 overflow<nsw> : index
                %15 = arith.addi %14, %7 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %16 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %17 = arith.cmpi slt, %7, %c232 : index
                %18 = arith.select %17, %15, %c1073741823 : index
                
                //iteration 0
                amdgpu.gather_to_lds %16[%18], %view_4[%11, %13] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                %19 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                %20 = affine.apply #map6()[%thread_id_x]
                %21 = affine.apply #map7()[%thread_id_x, %thread_id_y]
                %22 = arith.index_cast %21 : index to i32
                %23 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%22) : (i32) -> i32
                %24 = arith.index_cast %23 : i32 to index
                %25 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %26 = arith.index_cast %25 : i32 to index
                %27 = arith.muli %19, %c232 overflow<nsw> : index
                %28 = arith.addi %27, %20 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %29 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %30 = arith.cmpi slt, %20, %c232 : index
                %31 = arith.select %30, %28, %c1073741823 : index
                amdgpu.gather_to_lds %29[%31], %view_2[%24, %26] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                %32 = affine.apply #map8()[%thread_id_x, %thread_id_y, %block_id_y]
                %33 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                %34 = arith.index_cast %33 : index to i32
                %35 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%34) : (i32) -> i32
                %36 = arith.index_cast %35 : i32 to index
                %37 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %38 = arith.index_cast %37 : i32 to index
                %39 = arith.muli %32, %c232 overflow<nsw> : index
                %40 = arith.addi %39, %20 overflow<nsw> : index
                %41 = arith.select %30, %40, %c1073741823 : index
                amdgpu.gather_to_lds %29[%41], %view_2[%36, %38] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>

                // iteration 1
                %110 = affine.apply #map17()[%c0, %6]
                //%14 = arith.muli %3, %c232 overflow<nsw> : index
                %111 = arith.addi %14, %110 overflow<nsw> : index
                %112 = arith.cmpi slt, %110, %c232 : index
                %113 = arith.select %112, %111, %c1073741823 : index
                amdgpu.gather_to_lds %16[%113], %view_3[%11, %13] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                
                %120 = affine.apply #map18()[%c0, %thread_id_x]
                %122 = arith.addi %27, %120 overflow<nsw> : index
                %121 = arith.cmpi slt, %120, %c232 : index
                %123 = arith.select %121, %122, %c1073741823 : index
                amdgpu.gather_to_lds %29[%123], %view[%24, %26] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                
                %125 = arith.addi %39, %120 overflow<nsw> : index
                %126 = arith.select %121, %125, %c1073741823 : index
                amdgpu.gather_to_lds %29[%126], %view[%36, %38] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>

                //amdgpu.lds_barrier
                %42 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %43 = arith.index_cast %42 : index to i32
                %44 = arith.cmpi sge, %43, %c4_i32 : i32
                %45 = arith.cmpi slt, %43, %c4_i32 : i32
                scf.if %44 {
                rocdl.s.barrier
                }
                %46 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%9) : (i32) -> i32
                %47 = arith.index_cast %46 : i32 to index
                %48 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %49 = arith.index_cast %48 : i32 to index
                %50 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%22) : (i32) -> i32
                %51 = arith.index_cast %50 : i32 to index
                %52 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %53 = arith.index_cast %52 : i32 to index
                %54 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%34) : (i32) -> i32
                %55 = arith.index_cast %54 : i32 to index
                %56 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %57 = arith.index_cast %56 : i32 to index
                %58 = affine.apply #map11()[%thread_id_x]
                %59 = affine.apply #map12()[%thread_id_x]
                %60 = arith.xori %59, %5 : index
                %61 = affine.apply #map3()[%60]
                %62 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                %63 = affine.apply #map14()[%thread_id_x]
                %64 = affine.apply #map15()[%thread_id_x]
                %65 = arith.xori %64, %5 : index
                %66 = affine.apply #map3()[%65]
                %67 = affine.apply #map16()[%thread_id_x]
                %68:7 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %cst,%a_curr = %view_4, %a_ready = %view_3, %a_fetch = %view_5, %b_curr = %view_2, %b_ready = %view, %b_fetch = %view_6) -> (vector<4xf32>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>,memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>) {
                    rocdl.s.waitcnt 16371    // lowers to vmcnt(3)
                    amdgpu.lds_barrier
                    %arg_30= arith.addi %arg3, %c1 : index
                    %94 = vector.load %a_curr[%58, %61] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %95 = vector.load %b_curr[%62, %63] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %96 = affine.apply #map17()[%arg_30, %6]
                    %97 = arith.addi %14, %96 overflow<nsw> : index
                    %98 = arith.cmpi slt, %96, %c232 : index
                    %99 = arith.select %98, %97, %c1073741823 : index
                    amdgpu.gather_to_lds %16[%99], %a_fetch[%47, %49] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                    %100 = affine.apply #map18()[%arg_30, %thread_id_x]
                    %101 = arith.addi %27, %100 overflow<nsw> : index
                    %102 = arith.cmpi slt, %100, %c232 : index
                    %103 = arith.select %102, %101, %c1073741823 : index
                    amdgpu.gather_to_lds %29[%103], %b_fetch[%51, %53] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                    %104 = arith.addi %39, %100 overflow<nsw> : index
                    %105 = arith.select %102, %104, %c1073741823 : index
                    amdgpu.gather_to_lds %29[%105], %b_fetch[%55, %57] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.barrier
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.setprio 1
                    %106 = amdgpu.mfma %94 * %95 + %arg4 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                    rocdl.s.setprio 0
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    amdgpu.memory_counter_wait load(3)
                    rocdl.s.barrier
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %107 = vector.load %a_curr[%58, %66] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %108 = vector.load %b_curr[%62, %67] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    amdgpu.memory_counter_wait load(0)
                    rocdl.s.barrier
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.setprio 1
                    %109 = amdgpu.mfma %107 * %108 + %106 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                    rocdl.s.setprio 0
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    //amdgpu.lds_barrier

                    scf.yield %109, %a_ready, %a_fetch, %a_curr, %b_ready, %b_fetch, %b_curr : vector<4xf32>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>,memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>,memref<32x64xbf16, #gpu.address_space<workgroup>>
                }
                scf.if %45 {
                rocdl.s.barrier
                }
                rocdl.s.waitcnt 16371
                amdgpu.lds_barrier

                %69 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                %70 = affine.apply #map14()[%thread_id_x]
                %71 = vector.load %68#4[%69, %70] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %72 = affine.apply #map16()[%thread_id_x]
                %73 = vector.load %68#4[%69, %72] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %74 = affine.apply #map11()[%thread_id_x]
                %75 = affine.apply #map12()[%thread_id_x]
                %76 = arith.xori %75, %5 : index
                %77 = affine.apply #map3()[%76]
                %78 = vector.load %68#1[%74, %77] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %79 = affine.apply #map15()[%thread_id_x]
                %80 = arith.xori %79, %5 : index
                %81 = affine.apply #map3()[%80]
                %82 = vector.load %68#1[%74, %81] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %83 = amdgpu.mfma %78 * %71 + %68#0 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                %84 = amdgpu.mfma %82 * %73 + %83 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                
                rocdl.s.waitcnt 16368    
               
                %94 = vector.load %68#5[%69, %70] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %95 = vector.load %68#5[%69, %72] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
       
                %96 = vector.load %68#2[%74, %77] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %97 = vector.load %68#2[%74, %81] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %98 = amdgpu.mfma %96 * %94 + %84 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                %99 = amdgpu.mfma %97 * %95 + %98 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
        
                %85 = vector.extract_strided_slice %99 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                
                %86 = affine.apply #map19()[%block_id_x, %thread_id_x]
                %87 = affine.apply #map20()[%thread_id_x, %block_id_y, %thread_id_y]
                vector.store %85, %reinterpret_cast_1[%86, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %88 = vector.extract_strided_slice %99 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %89 = affine.apply #map21()[%block_id_x, %thread_id_x]
                vector.store %88, %reinterpret_cast_1[%89, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %90 = vector.extract_strided_slice %99 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %91 = affine.apply #map22()[%block_id_x, %thread_id_x]
                vector.store %90, %reinterpret_cast_1[%91, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %92 = vector.extract_strided_slice %99 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %93 = affine.apply #map23()[%block_id_x, %thread_id_x]
                vector.store %92, %reinterpret_cast_1[%93, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x232xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<256x232xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x256xf32>
            %3 = flow.dispatch @gemm_prefetch::@gemm_prefetch(%0, %1, %2) : (tensor<256x232xbf16>, tensor<256x232xbf16>, tensor<256x256xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x256xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x256xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
        
    
    """
    asm_256x256x64_tile_64x32x64_pingpong_mfma_16x16x16= """
        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 64) * 64)>
        #map1 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 8) * 64)>
        #map2 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 64)>
        #map3 = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 32 - ((s1 * 8 + s0 floordiv 32) floordiv 32) * 32)>
        #map4 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 32) * 64)>
        #map5 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 2 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 32)>
        #map6 = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 32 - ((s1 * 8 + s0 floordiv 32 + 16) floordiv 32) * 32 + 16)>
        #map7 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 2 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 32 + 16)>
        #map8 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map9 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
        #map10 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
        #map11 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 16)>
        #map12 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
        #map13 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 32)>
        #map14 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 48)>
        #map15 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 - (s1 floordiv 8) * 64 + 64)>
        #map16 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 2 - (s1 floordiv 32) * 64 + 64)>
        #map17 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
        #map18 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
        #map19 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
        #map20 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
        #map21 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm_prefetch {
            stream.executable.export public @gemm_prefetch workgroups() -> (index, index, index) {
            %c4 = arith.constant 4 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            stream.return %c4, %c8, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm_prefetch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c232_i14 = arith.constant 232 : i14
                %c0_i32 = arith.constant 0 : i32
                %c3 = arith.constant 3 : index
                %c1073741823 = arith.constant 1073741823 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c232 = arith.constant 232 : index
                %c1 = arith.constant 1 : index
                %c8192 = arith.constant 8192 : index
                %c20480 = arith.constant 20480 : index
                %c16384 = arith.constant 16384 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 4
                %block_id_y = gpu.block_id  y upper_bound 8
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 256], strides: [256, 1] : memref<f32> to memref<256x256xf32, strided<[256, 1], offset: ?>>
                %alloc = memref.alloc() : memref<24576xi8, #gpu.address_space<workgroup>>
                %view = memref.view %alloc[%c16384][] : memref<24576xi8, #gpu.address_space<workgroup>> to memref<32x64xbf16, #gpu.address_space<workgroup>>
                %view_2 = memref.view %alloc[%c20480][] : memref<24576xi8, #gpu.address_space<workgroup>> to memref<32x64xbf16, #gpu.address_space<workgroup>>
                %view_3 = memref.view %alloc[%c0][] : memref<24576xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.view %alloc[%c8192][] : memref<24576xi8, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
                %3 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %4 = affine.apply #map1()[%thread_id_x]
                %5 = affine.apply #map2()[%thread_id_x, %thread_id_y]
                %6 = arith.index_cast %5 : index to i32
                %7 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%6) : (i32) -> i32
                %8 = arith.index_cast %7 : i32 to index
                %9 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %10 = arith.index_cast %9 : i32 to index
                %11 = arith.muli %3, %c232 overflow<nsw> : index
                %12 = arith.addi %11, %4 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %13 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %14 = arith.cmpi slt, %4, %c232 : index
                %15 = arith.select %14, %12, %c1073741823 : index
                amdgpu.gather_to_lds %13[%15], %view_4[%8, %10] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                %16 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_y]
                %17 = affine.apply #map4()[%thread_id_x]
                %18 = affine.apply #map5()[%thread_id_x, %thread_id_y]
                %19 = arith.index_cast %18 : index to i32
                %20 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%19) : (i32) -> i32
                %21 = arith.index_cast %20 : i32 to index
                %22 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %23 = arith.index_cast %22 : i32 to index
                %24 = arith.muli %16, %c232 overflow<nsw> : index
                %25 = arith.addi %24, %17 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %26 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %27 = arith.cmpi slt, %17, %c232 : index
                %28 = arith.select %27, %25, %c1073741823 : index
                amdgpu.gather_to_lds %26[%28], %view_2[%21, %23] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                %29 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_y]
                %30 = affine.apply #map7()[%thread_id_x, %thread_id_y]
                %31 = arith.index_cast %30 : index to i32
                %32 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%31) : (i32) -> i32
                %33 = arith.index_cast %32 : i32 to index
                %34 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %35 = arith.index_cast %34 : i32 to index
                %36 = arith.muli %29, %c232 overflow<nsw> : index
                %37 = arith.addi %36, %17 overflow<nsw> : index
                %38 = arith.select %27, %37, %c1073741823 : index
                amdgpu.gather_to_lds %26[%38], %view_2[%33, %35] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                amdgpu.lds_barrier
                %39 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                %40 = arith.index_cast %39 : index to i32
                %41 = arith.cmpi sge, %40, %c4_i32 : i32
                %42 = arith.cmpi slt, %40, %c4_i32 : i32
                scf.if %41 {
                rocdl.s.barrier
                }
                %43 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%6) : (i32) -> i32
                %44 = arith.index_cast %43 : i32 to index
                %45 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %46 = arith.index_cast %45 : i32 to index
                %47 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%19) : (i32) -> i32
                %48 = arith.index_cast %47 : i32 to index
                %49 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %50 = arith.index_cast %49 : i32 to index
                %51 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%31) : (i32) -> i32
                %52 = arith.index_cast %51 : i32 to index
                %53 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %54 = arith.index_cast %53 : i32 to index
                %55 = affine.apply #map9()[%thread_id_x]
                %56 = affine.apply #map10()[%thread_id_x]
                %57 = affine.apply #map11()[%thread_id_x]
                %58 = affine.apply #map12()[%thread_id_x, %thread_id_y]
                %59 = affine.apply #map13()[%thread_id_x]
                %60 = affine.apply #map14()[%thread_id_x]
                %61:5 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %cst, %arg5 = %view_4, %arg6 = %view_3, %arg7 = %view_2, %arg8 = %view) -> (vector<4xf32>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>) {
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %89 = vector.load %arg5[%55, %56] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %90 = vector.load %arg5[%55, %57] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %91 = vector.load %arg7[%58, %56] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %92 = vector.load %arg7[%58, %57] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %93 = affine.apply #map15()[%arg3, %thread_id_x]
                %94 = arith.addi %11, %93 overflow<nsw> : index
                %95 = arith.cmpi slt, %93, %c232 : index
                %96 = arith.select %95, %94, %c1073741823 : index
                amdgpu.gather_to_lds %13[%96], %arg6[%44, %46] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                %97 = affine.apply #map16()[%arg3, %thread_id_x]
                %98 = arith.addi %24, %97 overflow<nsw> : index
                %99 = arith.cmpi slt, %97, %c232 : index
                %100 = arith.select %99, %98, %c1073741823 : index
                amdgpu.gather_to_lds %26[%100], %arg8[%48, %50] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                %101 = arith.addi %36, %97 overflow<nsw> : index
                %102 = arith.select %99, %101, %c1073741823 : index
                amdgpu.gather_to_lds %26[%102], %arg8[%52, %54] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %103 = amdgpu.mfma %89 * %91 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
                %104 = amdgpu.mfma %90 * %92 + %103 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
                rocdl.s.setprio 0
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.memory_counter_wait load(3)
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                %105 = vector.load %arg5[%55, %59] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %106 = vector.load %arg5[%55, %60] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %107 = vector.load %arg7[%58, %59] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %108 = vector.load %arg7[%58, %60] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.memory_counter_wait load(0)
                rocdl.s.barrier
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                rocdl.s.setprio 1
                %109 = amdgpu.mfma %105 * %107 + %104 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
                %110 = amdgpu.mfma %106 * %108 + %109 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
                rocdl.s.setprio 0
                llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                amdgpu.lds_barrier
                scf.yield %110, %arg6, %arg5, %arg8, %arg7 : vector<4xf32>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                }
                scf.if %42 {
                rocdl.s.barrier
                }
                rocdl.s.waitcnt 16368
                amdgpu.lds_barrier
                %62 = affine.apply #map12()[%thread_id_x, %thread_id_y]
                %63 = affine.apply #map10()[%thread_id_x]
                %64 = vector.load %61#3[%62, %63] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %65 = affine.apply #map11()[%thread_id_x]
                %66 = vector.load %61#3[%62, %65] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %67 = affine.apply #map13()[%thread_id_x]
                %68 = vector.load %61#3[%62, %67] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %69 = affine.apply #map14()[%thread_id_x]
                %70 = vector.load %61#3[%62, %69] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %71 = affine.apply #map9()[%thread_id_x]
                %72 = vector.load %61#1[%71, %63] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %73 = vector.load %61#1[%71, %65] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %74 = vector.load %61#1[%71, %67] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %75 = vector.load %61#1[%71, %69] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
                %76 = amdgpu.mfma %72 * %64 + %61#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
                %77 = amdgpu.mfma %73 * %66 + %76 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
                %78 = amdgpu.mfma %74 * %68 + %77 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
                %79 = amdgpu.mfma %75 * %70 + %78 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
                %80 = vector.extract_strided_slice %79 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %81 = affine.apply #map17()[%block_id_x, %thread_id_x]
                %82 = affine.apply #map18()[%thread_id_x, %block_id_y, %thread_id_y]
                vector.store %80, %reinterpret_cast_1[%81, %82] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %83 = vector.extract_strided_slice %79 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %84 = affine.apply #map19()[%block_id_x, %thread_id_x]
                vector.store %83, %reinterpret_cast_1[%84, %82] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %85 = vector.extract_strided_slice %79 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %86 = affine.apply #map20()[%block_id_x, %thread_id_x]
                vector.store %85, %reinterpret_cast_1[%86, %82] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %87 = vector.extract_strided_slice %79 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %88 = affine.apply #map21()[%block_id_x, %thread_id_x]
                vector.store %87, %reinterpret_cast_1[%88, %82] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x232xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<256x232xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x256xf32>
            %3 = flow.dispatch @gemm_prefetch::@gemm_prefetch(%0, %1, %2) : (tensor<256x232xbf16>, tensor<256x232xbf16>, tensor<256x256xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x256xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x256xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
    """

    # Input sizes
    M = shape[0]
    N = shape[1]
    K = shape[2]
    # Workgroup tile sizes
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64
    # Group sizes
    GROUP_SIZE_M = 16

    reordered_gemm, hyperparams = get_reordered_matmul(
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, mfma_variant, torch.bfloat16, torch.float32
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        use_buffer_ops=True,
        print_mlir=True,
        use_global_to_shared=True,
        override_mlir=asm_256x256x64_tile_64x32x64_pingpong_mfma_16x16x32_triple_buffer,
    )

    options = set_default_run_config(options)
    return wave_compile(options, reordered_gemm)


def torch_compile_matmul(a, b):
    # Simple wrapper around matmul
    def fn(x, y):
        return torch.mm(x, y.t())   # same convention as your GEMM test

    compiled_fn = torch.compile(fn)     # JIT-compile it with TorchDynamo+Inductor
    return compiled_fn(a, b)


def calculate_diff_gemm(M, N, K, dtype=torch.bfloat16):
    # Random test matrices
    # A = torch.ones(M, K, dtype=dtype, device="cuda")
    # B = torch.zeros(N, K, dtype=dtype, device="cuda")  # careful: ABᵀ → shape (M,N)

    #A = torch.arange(M * K, dtype=dtype, device="cuda").reshape(M, K)
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(N, K, dtype=dtype, device="cuda")
    C = torch.empty((M, N), dtype=torch.float32, device="cuda")

    # ---- WAVE ----
    # wave_kernel = get_wave_gemm((M,N,K), dtype, [False,False,False],MMAType.F32_32x32x16_K8_F16)  # <- your Wave GEMM builder
    # wave_kernel(A.clone(), B.clone(),C) 
    
    # ---- WAVE PIPELINED ----
    #warmup
    wave_gemm = testReorderedPingPongGemm((M,N,K), dtype, [False,False,False], MMAType.F32_32x32x16_F16)
    for i in range(209):
        wave_gemm(A,B,C) 

    wave_gemm(A,B,C) 

    # ---- TRITON ----
    output_triton = triton_matmul_abt(A.clone(), B.clone())  

    # ---- TORCH ----
    # GEMM ABᵀ → (M,K) * (N,K)ᵀ = (M,N)
    #output_torch = torch.matmul(A, B.t())

    # ---- Compare ----
    print(f"Wave output shape:   {C.shape}")
    print(f"Triton output shape: {output_triton.shape}")
   # print(f"Torch output shape:  {output_torch.shape}")

    torch.set_printoptions(
    threshold=float("inf"),  # no summarization
    linewidth=200,           # wider line before wrapping
    edgeitems=8,             # how many items to show at each edge
    precision=3,
    sci_mode=False,
    )


    print("wave A:")
    print(A[:2, :16])
    print("wave B:")
    print(B[:8, :68])
    print("wave output:")
    print(C[:4, :16])
    print("ref output")
    print(output_triton[:4, :16])

    if torch.allclose(C, output_triton.to(torch.float32), atol=1e-2, rtol=1e-2):
        print("✅ Wave and Triton implementations match")
    else:
        print("❌ Wave and Triton implementations differ")
        max_diff = (C - output_triton.to(torch.float32)).abs().max().item()
        print(f"Max diff Wave vs Triton: {max_diff}")


# Pick a grid to match what you want to compare with Wave
# M_vals = [64,128,256]
# N_vals = [128 ,128,256]
# K_vals = [511,511,511 ]
# M_vals = [64]
# N_vals = [128]
# K_vals = [512]

# M_vals = [16384]
# N_vals = [32768]
# K_vals = [6144]

M_vals = [256]
N_vals = [256]
K_vals = [232]

# M_vals = [256]
# N_vals = [256]
# K_vals = [512]

configs = list(itertools.product(M_vals, N_vals, K_vals))

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        # line_vals=["wave", "reordered_gemm","wave_pipelined_16x16","wave_pipelined_32x32", "harsh_version_32x32x8","triton"],
        # line_names=["wave_16x16x16", "reordered_gemm","wave_pipelined_16x16_16","wave_pipelined_32x32_16","harsh_version_32x32x8","Triton"],
        line_vals=["reordered_gemm"],
        line_names=["reordered_gemm"],
        styles=[("blue","-"), ("red","-"),("green","-"),("orange","-"),("black","-"),("black","-")],
        ylabel="ms",
        plot_name="gemm-abt-performance",
        args={},
    )
)

def bench(M, N, K, provider, ):
    dtype = torch.bfloat16
    dtype32=torch.float32
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(N, K, dtype=dtype, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        # warmup
        triton_matmul_abt(A, B)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul_abt(A, B),
            quantiles=quantiles,
        )       
    elif provider == "reordered_gemm":
        # plug your compiled wave GEMM here; it should compute C in fp32
        wave_gemm = testReorderedPingPongGemm( (M,N,K), dtype, [False,False,False], MMAType.F32_32x32x16_F16)
        C = torch.empty((M, N), dtype=dtype32, device="cuda")
        _ = wave_gemm(A, B, C)   # warmup; expect A(M,K), B(N,K), C(M,N)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: wave_gemm(A, B, C),
            quantiles=quantiles,
        )
    elif provider == "torch_compile":
        ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: torch_compile_matmul(A, B),
        quantiles=quantiles,
    )
    elif provider == "torch":
        ref = lambda: torch.mm(A, B.t()).float()
        _ = ref()
        ms, min_ms, max_ms = triton.testing.do_bench(ref, quantiles=quantiles)

    else:
        raise ValueError(provider)

    return ms, min_ms, max_ms

if __name__ == "__main__":
    
    # M_vals = [16384]
    # N_vals = [32768]
    # K_vals = [6144]
    
    # perf sweep
    #bench.run(print_data=True, show_plots=False)
    
    #calculate_diff_gemm(256, 256, 1024, torch.bfloat16)
    #calculate_diff_gemm(256, 512, 40960, torch.bfloat16)
    calculate_diff_gemm(256, 256, 232, torch.bfloat16)

    #calculate_diff_gemm(1280, 32, 192, torch.bfloat16)

    #calculate_diff_gemm(256, 256, 512)
    #calculate_diff_gemm(64, 128, 511)
    #calculate_diff_gemm(16384, 32768, 6144,torch.bfloat16)








