"""
MXFP4 Scaled GEMM Scheduling for GFX950 (MI350)

Double-buffered MXFP4 GEMM with 4-wave and 8-wave configurations.
Uses get_tagged_mxfp4_gemm (templates) + get_mxfp4_dbuf_schedule (schedules).

Usage:
    python 7.1_schedule.py --test test_dbuf_4wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm --debug
    python 7.1_schedule.py --list_tests
"""

import torch

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm, get_preshuffle_kernel
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_dbuf_schedule,
    get_mxfp4_dbuf_schedule_shuffle,
)
from wave_lang.kernel.wave.schedules import get_mxfp4_triplebuf_schedule
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
)

from utils import parse_args, list_tests, run_test


def e8m0_shuffle(scale):
    """
    Shuffle the scale tensor for e8m0 format.

    This particular shuffle is taken from
    https://github.com/ROCm/rocm-libraries/blob/4348901528fe100a84975b89c247eece553a2a2d/shared/mxdatagenerator/lib/include/mxDataGenerator/PreSwizzle.hpp#L403

    The e8m0_shuffle operation transforms a matrix with shape (m, n) as follows:
    1. Pads to shape ((m+255)//256*256, (n+7)//8*8)
    2. Reshapes to (sm//32, 2, 16, sn//8, 2, 4)
    3. Permutes dimensions: (0, 3, 5, 2, 4, 1)
    4. Flattens back to (sm, sn)

    Args:
        scale: A 2D tensor to be shuffled

    Returns:
        Shuffled tensor with the same padded shape
    """
    if scale is None:
        return scale
    if scale.dtype == torch.float32:
        return scale
    assert scale.ndim == 2, "scale must be a 2D tensor"
    m, n = scale.shape
    scale_padded = torch.zeros(
        (m + 255) // 256 * 256,
        (n + 7) // 8 * 8,
        dtype=scale.dtype,
        device=scale.device,
    )

    scale_padded[:m, :n] = scale
    scale = scale_padded
    sm, sn = scale.shape
    scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
    scale = scale.view(sm, sn)
    return scale


def _run_mxfp_gemm(gemm, shape, shuffle_scales=False):
    """Run compiled GEMM kernel and verify against reference.

    Args:
        gemm: Compiled GEMM kernel function.
        shape: (M, N, K) problem dimensions.
        shuffle_scales: If True, shuffle the scale tensors using e8m0_shuffle.
    """
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    if shuffle_scales:
        # x_scales = e8m0_shuffle(x_scales)
        w_scales = e8m0_shuffle(w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    for i in range(100):
        gemm(x, x_scales, w.T.contiguous(), w_scales, out)

    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def test_dbuf_4wave_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 4 waves, no stagger."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, num_waves=4)

    schedule = get_mxfp4_dbuf_schedule(use_stagger=False)

    options.print_ir_after = "all" if is_debug else []
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave.mlir"
    options.print_mlir = True
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 4-wave test passed!")


def test_dbuf_8wave_mxfp_gemm(
    is_debug=False, shape=(4096, 57344, 16384), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 8 waves, with stagger."""

    mlir = """
        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 16)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
        #map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
        #map9 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
        #map11 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
        #map12 = affine_map<()[s0] -> ((s0 floordiv 2) mod 2)>
        #map13 = affine_map<()[s0] -> (s0 mod 2)>
        #map14 = affine_map<()[s0] -> (s0 * 4)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 32 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 256)>
        #map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
        #map18 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
        #map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
        #map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
        #map22 = affine_map<()[s0] -> (s0 * 4 + (s0 mod 64) floordiv 16 - (s0 floordiv 2) * 8)>
        #map23 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
        #map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
        #map25 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
        #map26 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
        #map27 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
        #map28 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
        #map29 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
        #map30 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
        #map31 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map32 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
        #map33 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 128)>
        #map34 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 + 8)>
        #map35 = affine_map<()[s0] -> (s0 * 256)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
        #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
        #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
        #map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
        #map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
        #map47 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
        #map48 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
        #map49 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
        #map50 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
        #map51 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c64 = arith.constant 64 : index
            %c1 = arith.constant 1 : index
            stream.return %c64, %c64, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c512_i14 = arith.constant 512 : i14
                %c-8192_i14 = arith.constant -8192 : i14
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c16384 = arith.constant 16384 : index
                %c63 = arith.constant 63 : index
                %c512 = arith.constant 512 : index
                %c2147483646_i64 = arith.constant 2147483646 : i64
                %c8192 = arith.constant 8192 : index
                %c1 = arith.constant 1 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
                %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
                %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 64
                %block_id_y = gpu.block_id  y upper_bound 64
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %alloc = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_0 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_3 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_4 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_5 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_6 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %5 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %6 = affine.apply #map1()[%thread_id_x]
                %7 = affine.apply #map2()[%thread_id_x]
                %8 = arith.xori %7, %6 : index
                %9 = affine.apply #map3()[%8]
                %10 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %11 = gpu.subgroup_broadcast %10,  first_active_lane : index
                %12 = gpu.subgroup_broadcast %c0,  first_active_lane : index
                %13 = arith.muli %5, %c8192 overflow<nsw> : index
                %14 = arith.addi %13, %9 overflow<nsw> : index
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast = memref.cast %reinterpret_cast : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %15 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %15[%14], %alloc_6[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %16 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
                %17 = affine.apply #map6()[%thread_id_x, %thread_id_y]
                %18 = gpu.subgroup_broadcast %17,  first_active_lane : index
                %19 = arith.muli %16, %c8192 overflow<nsw> : index
                %20 = arith.addi %19, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%20], %alloc_6[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %21 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
                %22 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                %23 = gpu.subgroup_broadcast %22,  first_active_lane : index
                %24 = arith.muli %21, %c8192 overflow<nsw> : index
                %25 = arith.addi %24, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%25], %alloc_6[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %26 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
                %27 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %28 = gpu.subgroup_broadcast %27,  first_active_lane : index
                %29 = arith.muli %26, %c8192 overflow<nsw> : index
                %30 = arith.addi %29, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%30], %alloc_6[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %31 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_x]
                %32 = affine.apply #map12()[%thread_id_x]
                %33 = affine.apply #map13()[%thread_id_x]
                %34 = arith.xori %33, %32 : index
                %35 = affine.apply #map14()[%34]
                %36 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %37 = gpu.subgroup_broadcast %36,  first_active_lane : index
                %38 = arith.muli %31, %c512 overflow<nsw> : index
                %39 = arith.addi %38, %35 overflow<nsw> : index
                %reinterpret_cast_7 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %40 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %40[%39], %alloc_4[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                %41 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
                %42 = arith.muli %41, %c8192 overflow<nsw> : index
                %43 = arith.addi %42, %9 overflow<nsw> : index
                %reinterpret_cast_9 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_10 = memref.cast %reinterpret_cast_9 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %44 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %44[%43], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %45 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                %46 = arith.muli %45, %c8192 overflow<nsw> : index
                %47 = arith.addi %46, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%47], %alloc_2[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %48 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
                %49 = arith.muli %48, %c8192 overflow<nsw> : index
                %50 = arith.addi %49, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%50], %alloc_2[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %51 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_y]
                %52 = arith.muli %51, %c8192 overflow<nsw> : index
                %53 = arith.addi %52, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%53], %alloc_2[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %54 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_y]
                %55 = arith.muli %54, %c512 overflow<nsw> : index
                %56 = arith.addi %55, %35 overflow<nsw> : index
                %reinterpret_cast_11 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_12 = memref.cast %reinterpret_cast_11 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %57 = amdgpu.fat_raw_buffer_cast %cast_12 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %57[%56], %alloc_0[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.s.barrier
                %58 = affine.apply #map16()[%thread_id_x, %thread_id_y]
                %59 = arith.index_cast %58 : index to i32
                %60 = arith.cmpi sge, %59, %c4_i32 : i32
                %61 = arith.cmpi slt, %59, %c4_i32 : i32
                scf.if %60 {
                rocdl.s.barrier
                }
                %62 = affine.apply #map17()[%thread_id_x]
                %63 = affine.apply #map18()[%thread_id_x]
                %64 = arith.xori %63, %7 : index
                %65 = affine.apply #map3()[%64]
                %66 = affine.apply #map19()[%thread_id_x]
                %67 = affine.apply #map20()[%thread_id_x]
                %68 = affine.apply #map21()[%thread_id_x]
                %69 = affine.apply #map22()[%thread_id_x]
                %70 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %71 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %72 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %73 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %74 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %75 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %76 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %77 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %78 = affine.apply #map31()[%thread_id_x]
                %79 = arith.xori %78, %7 : index
                %80 = affine.apply #map3()[%79]
                %81 = arith.xori %33, %c1 : index
                %82 = affine.apply #map32()[%thread_id_x, %81]
                %83:40 = scf.for %arg5 = %c0 to %c63 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %alloc_6, %arg39 = %alloc_5, %arg40 = %alloc_4, %arg41 = %alloc_3, %arg42 = %alloc_2, %arg43 = %alloc_1, %arg44 = %alloc_0, %arg45 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>) {
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(0)
                rocdl.s.barrier
                %582 = affine.apply #map33()[%arg5, %8]
                %583 = arith.addi %13, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%583], %arg39[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %584 = arith.addi %19, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%584], %arg39[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %585 = arith.addi %24, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%585], %arg39[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %586 = arith.addi %29, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%586], %arg39[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %587 = affine.apply #map34()[%arg5, %34]
                %588 = arith.addi %38, %587 overflow<nsw> : index
                amdgpu.gather_to_lds %40[%588], %arg41[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                %589 = arith.addi %42, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%589], %arg43[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %590 = arith.addi %46, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%590], %arg43[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %591 = arith.addi %49, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%591], %arg43[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %592 = arith.addi %52, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%592], %arg43[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %593 = arith.addi %55, %587 overflow<nsw> : index
                amdgpu.gather_to_lds %57[%593], %arg45[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.sched.barrier 0
                %594 = vector.load %arg38[%62, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %595 = vector.load %arg38[%66, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %596 = vector.load %arg38[%67, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %597 = vector.load %arg38[%68, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %598 = vector.load %arg40[%62, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %599 = vector.load %arg40[%66, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %600 = vector.load %arg40[%67, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %601 = vector.load %arg40[%68, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %602 = vector.load %arg42[%70, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %603 = vector.load %arg42[%71, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %604 = vector.load %arg42[%72, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %605 = vector.load %arg42[%73, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %606 = vector.load %arg42[%74, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %607 = vector.load %arg42[%75, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %608 = vector.load %arg42[%76, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %609 = vector.load %arg42[%77, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %610 = vector.load %arg44[%70, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %611 = vector.load %arg44[%71, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %612 = vector.load %arg44[%72, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %613 = vector.load %arg44[%73, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %614 = vector.load %arg44[%74, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %615 = vector.load %arg44[%75, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %616 = vector.load %arg44[%76, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %617 = vector.load %arg44[%77, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %618 = vector.bitcast %594 : vector<16xi8> to vector<32xf4E2M1FN>
                %619 = vector.bitcast %595 : vector<16xi8> to vector<32xf4E2M1FN>
                %620 = vector.bitcast %596 : vector<16xi8> to vector<32xf4E2M1FN>
                %621 = vector.bitcast %597 : vector<16xi8> to vector<32xf4E2M1FN>
                %622 = vector.bitcast %598 : vector<1xi8> to vector<1xf8E8M0FNU>
                %623 = vector.bitcast %599 : vector<1xi8> to vector<1xf8E8M0FNU>
                %624 = vector.bitcast %600 : vector<1xi8> to vector<1xf8E8M0FNU>
                %625 = vector.bitcast %601 : vector<1xi8> to vector<1xf8E8M0FNU>
                %626 = vector.bitcast %602 : vector<16xi8> to vector<32xf4E2M1FN>
                %627 = vector.bitcast %603 : vector<16xi8> to vector<32xf4E2M1FN>
                %628 = vector.bitcast %604 : vector<16xi8> to vector<32xf4E2M1FN>
                %629 = vector.bitcast %605 : vector<16xi8> to vector<32xf4E2M1FN>
                %630 = vector.bitcast %606 : vector<16xi8> to vector<32xf4E2M1FN>
                %631 = vector.bitcast %607 : vector<16xi8> to vector<32xf4E2M1FN>
                %632 = vector.bitcast %608 : vector<16xi8> to vector<32xf4E2M1FN>
                %633 = vector.bitcast %609 : vector<16xi8> to vector<32xf4E2M1FN>
                %634 = vector.bitcast %610 : vector<1xi8> to vector<1xf8E8M0FNU>
                %635 = vector.bitcast %611 : vector<1xi8> to vector<1xf8E8M0FNU>
                %636 = vector.bitcast %612 : vector<1xi8> to vector<1xf8E8M0FNU>
                %637 = vector.bitcast %613 : vector<1xi8> to vector<1xf8E8M0FNU>
                %638 = vector.bitcast %614 : vector<1xi8> to vector<1xf8E8M0FNU>
                %639 = vector.bitcast %615 : vector<1xi8> to vector<1xf8E8M0FNU>
                %640 = vector.bitcast %616 : vector<1xi8> to vector<1xf8E8M0FNU>
                %641 = vector.bitcast %617 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %642 = vector.extract %622[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %643 = vector.extract %634[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %644 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%643[0] * %626) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %645 = vector.extract %635[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %646 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%645[0] * %627) + %arg7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %647 = vector.extract %636[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %648 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%647[0] * %628) + %arg8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %649 = vector.extract %637[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %650 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%649[0] * %629) + %arg9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %651 = vector.extract %638[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %652 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%651[0] * %630) + %arg10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %653 = vector.extract %639[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %654 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%653[0] * %631) + %arg11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %655 = vector.extract %640[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %656 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%655[0] * %632) + %arg12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %657 = vector.extract %641[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %658 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%657[0] * %633) + %arg13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %659 = vector.extract %623[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %660 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%643[0] * %626) + %arg14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %661 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%645[0] * %627) + %arg15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %662 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%647[0] * %628) + %arg16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %663 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%649[0] * %629) + %arg17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %664 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%651[0] * %630) + %arg18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %665 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%653[0] * %631) + %arg19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %666 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%655[0] * %632) + %arg20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %667 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%657[0] * %633) + %arg21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %668 = vector.extract %624[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %669 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%643[0] * %626) + %arg22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %670 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%645[0] * %627) + %arg23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %671 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%647[0] * %628) + %arg24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %672 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%649[0] * %629) + %arg25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %673 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%651[0] * %630) + %arg26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %674 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%653[0] * %631) + %arg27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %675 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%655[0] * %632) + %arg28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %676 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%657[0] * %633) + %arg29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %677 = vector.extract %625[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %678 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%643[0] * %626) + %arg30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %679 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%645[0] * %627) + %arg31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %680 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%647[0] * %628) + %arg32 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %681 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%649[0] * %629) + %arg33 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %682 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%651[0] * %630) + %arg34 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %683 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%653[0] * %631) + %arg35 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %684 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%655[0] * %632) + %arg36 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %685 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%657[0] * %633) + %arg37 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                %686 = vector.load %arg38[%62, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %687 = vector.load %arg38[%66, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %688 = vector.load %arg38[%67, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %689 = vector.load %arg38[%68, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %690 = vector.load %arg40[%62, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %691 = vector.load %arg40[%66, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %692 = vector.load %arg40[%67, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %693 = vector.load %arg40[%68, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %694 = vector.load %arg42[%70, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %695 = vector.load %arg42[%71, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %696 = vector.load %arg42[%72, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %697 = vector.load %arg42[%73, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %698 = vector.load %arg42[%74, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %699 = vector.load %arg42[%75, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %700 = vector.load %arg42[%76, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %701 = vector.load %arg42[%77, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %702 = vector.load %arg44[%70, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %703 = vector.load %arg44[%71, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %704 = vector.load %arg44[%72, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %705 = vector.load %arg44[%73, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %706 = vector.load %arg44[%74, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %707 = vector.load %arg44[%75, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %708 = vector.load %arg44[%76, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %709 = vector.load %arg44[%77, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %710 = vector.bitcast %686 : vector<16xi8> to vector<32xf4E2M1FN>
                %711 = vector.bitcast %687 : vector<16xi8> to vector<32xf4E2M1FN>
                %712 = vector.bitcast %688 : vector<16xi8> to vector<32xf4E2M1FN>
                %713 = vector.bitcast %689 : vector<16xi8> to vector<32xf4E2M1FN>
                %714 = vector.bitcast %690 : vector<1xi8> to vector<1xf8E8M0FNU>
                %715 = vector.bitcast %691 : vector<1xi8> to vector<1xf8E8M0FNU>
                %716 = vector.bitcast %692 : vector<1xi8> to vector<1xf8E8M0FNU>
                %717 = vector.bitcast %693 : vector<1xi8> to vector<1xf8E8M0FNU>
                %718 = vector.bitcast %694 : vector<16xi8> to vector<32xf4E2M1FN>
                %719 = vector.bitcast %695 : vector<16xi8> to vector<32xf4E2M1FN>
                %720 = vector.bitcast %696 : vector<16xi8> to vector<32xf4E2M1FN>
                %721 = vector.bitcast %697 : vector<16xi8> to vector<32xf4E2M1FN>
                %722 = vector.bitcast %698 : vector<16xi8> to vector<32xf4E2M1FN>
                %723 = vector.bitcast %699 : vector<16xi8> to vector<32xf4E2M1FN>
                %724 = vector.bitcast %700 : vector<16xi8> to vector<32xf4E2M1FN>
                %725 = vector.bitcast %701 : vector<16xi8> to vector<32xf4E2M1FN>
                %726 = vector.bitcast %702 : vector<1xi8> to vector<1xf8E8M0FNU>
                %727 = vector.bitcast %703 : vector<1xi8> to vector<1xf8E8M0FNU>
                %728 = vector.bitcast %704 : vector<1xi8> to vector<1xf8E8M0FNU>
                %729 = vector.bitcast %705 : vector<1xi8> to vector<1xf8E8M0FNU>
                %730 = vector.bitcast %706 : vector<1xi8> to vector<1xf8E8M0FNU>
                %731 = vector.bitcast %707 : vector<1xi8> to vector<1xf8E8M0FNU>
                %732 = vector.bitcast %708 : vector<1xi8> to vector<1xf8E8M0FNU>
                %733 = vector.bitcast %709 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %734 = vector.extract %714[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %735 = vector.extract %726[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %736 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%735[0] * %718) + %644 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %737 = vector.extract %727[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %738 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%737[0] * %719) + %646 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %739 = vector.extract %728[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %740 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%739[0] * %720) + %648 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %741 = vector.extract %729[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %742 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%741[0] * %721) + %650 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %743 = vector.extract %730[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %744 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%743[0] * %722) + %652 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %745 = vector.extract %731[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %746 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%745[0] * %723) + %654 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %747 = vector.extract %732[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %748 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%747[0] * %724) + %656 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %749 = vector.extract %733[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %750 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%749[0] * %725) + %658 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %751 = vector.extract %715[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %752 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%735[0] * %718) + %660 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %753 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%737[0] * %719) + %661 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %754 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%739[0] * %720) + %662 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %755 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%741[0] * %721) + %663 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %756 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%743[0] * %722) + %664 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %757 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%745[0] * %723) + %665 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %758 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%747[0] * %724) + %666 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %759 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%749[0] * %725) + %667 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %760 = vector.extract %716[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %761 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%735[0] * %718) + %669 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %762 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%737[0] * %719) + %670 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %763 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%739[0] * %720) + %671 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %764 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%741[0] * %721) + %672 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %765 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%743[0] * %722) + %673 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %766 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%745[0] * %723) + %674 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %767 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%747[0] * %724) + %675 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %768 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%749[0] * %725) + %676 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %769 = vector.extract %717[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %770 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%735[0] * %718) + %678 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %771 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%737[0] * %719) + %679 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %772 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%739[0] * %720) + %680 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %773 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%741[0] * %721) + %681 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %774 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%743[0] * %722) + %682 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %775 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%745[0] * %723) + %683 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %776 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%747[0] * %724) + %684 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %777 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%749[0] * %725) + %685 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                scf.yield %736, %738, %740, %742, %744, %746, %748, %750, %752, %753, %754, %755, %756, %757, %758, %759, %761, %762, %763, %764, %765, %766, %767, %768, %770, %771, %772, %773, %774, %775, %776, %777, %arg39, %arg38, %arg41, %arg40, %arg43, %arg42, %arg45, %arg44 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                }
                amdgpu.lds_barrier
                scf.if %61 {
                rocdl.s.barrier
                }
                %84 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %85 = affine.apply #map22()[%thread_id_x]
                %86 = vector.load %83#38[%84, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %87 = arith.xori %33, %c1 : index
                %88 = affine.apply #map32()[%thread_id_x, %87]
                %89 = vector.load %83#38[%84, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %90 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %91 = vector.load %83#38[%90, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %92 = vector.load %83#38[%90, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %93 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %94 = vector.load %83#38[%93, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %95 = vector.load %83#38[%93, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %96 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %97 = vector.load %83#38[%96, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %98 = vector.load %83#38[%96, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %99 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %100 = vector.load %83#38[%99, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %101 = vector.load %83#38[%99, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %102 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %103 = vector.load %83#38[%102, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %104 = vector.load %83#38[%102, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %105 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %106 = vector.load %83#38[%105, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %107 = vector.load %83#38[%105, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %108 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %109 = vector.load %83#38[%108, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %110 = vector.load %83#38[%108, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %111 = affine.apply #map18()[%thread_id_x]
                %112 = arith.xori %111, %7 : index
                %113 = affine.apply #map3()[%112]
                %114 = vector.load %83#36[%84, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %115 = affine.apply #map31()[%thread_id_x]
                %116 = arith.xori %115, %7 : index
                %117 = affine.apply #map3()[%116]
                %118 = vector.load %83#36[%84, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %119 = vector.load %83#36[%90, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %120 = vector.load %83#36[%90, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %121 = vector.load %83#36[%93, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %122 = vector.load %83#36[%93, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %123 = vector.load %83#36[%96, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %124 = vector.load %83#36[%96, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %125 = vector.load %83#36[%99, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %126 = vector.load %83#36[%99, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %127 = vector.load %83#36[%102, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %128 = vector.load %83#36[%102, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %129 = vector.load %83#36[%105, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %130 = vector.load %83#36[%105, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %131 = vector.load %83#36[%108, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %132 = vector.load %83#36[%108, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %133 = affine.apply #map17()[%thread_id_x]
                %134 = vector.load %83#34[%133, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %135 = vector.load %83#34[%133, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %136 = affine.apply #map19()[%thread_id_x]
                %137 = vector.load %83#34[%136, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %138 = vector.load %83#34[%136, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %139 = affine.apply #map20()[%thread_id_x]
                %140 = vector.load %83#34[%139, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %141 = vector.load %83#34[%139, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %142 = affine.apply #map21()[%thread_id_x]
                %143 = vector.load %83#34[%142, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %144 = vector.load %83#34[%142, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %145 = vector.load %83#32[%133, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %146 = vector.load %83#32[%133, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %147 = vector.load %83#32[%136, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %148 = vector.load %83#32[%136, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %149 = vector.load %83#32[%139, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %150 = vector.load %83#32[%139, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %151 = vector.load %83#32[%142, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %152 = vector.load %83#32[%142, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %153 = vector.bitcast %145 : vector<16xi8> to vector<32xf4E2M1FN>
                %154 = vector.bitcast %146 : vector<16xi8> to vector<32xf4E2M1FN>
                %155 = vector.bitcast %147 : vector<16xi8> to vector<32xf4E2M1FN>
                %156 = vector.bitcast %148 : vector<16xi8> to vector<32xf4E2M1FN>
                %157 = vector.bitcast %149 : vector<16xi8> to vector<32xf4E2M1FN>
                %158 = vector.bitcast %150 : vector<16xi8> to vector<32xf4E2M1FN>
                %159 = vector.bitcast %151 : vector<16xi8> to vector<32xf4E2M1FN>
                %160 = vector.bitcast %152 : vector<16xi8> to vector<32xf4E2M1FN>
                %161 = vector.bitcast %134 : vector<1xi8> to vector<1xf8E8M0FNU>
                %162 = vector.bitcast %135 : vector<1xi8> to vector<1xf8E8M0FNU>
                %163 = vector.bitcast %137 : vector<1xi8> to vector<1xf8E8M0FNU>
                %164 = vector.bitcast %138 : vector<1xi8> to vector<1xf8E8M0FNU>
                %165 = vector.bitcast %140 : vector<1xi8> to vector<1xf8E8M0FNU>
                %166 = vector.bitcast %141 : vector<1xi8> to vector<1xf8E8M0FNU>
                %167 = vector.bitcast %143 : vector<1xi8> to vector<1xf8E8M0FNU>
                %168 = vector.bitcast %144 : vector<1xi8> to vector<1xf8E8M0FNU>
                %169 = vector.bitcast %114 : vector<16xi8> to vector<32xf4E2M1FN>
                %170 = vector.bitcast %118 : vector<16xi8> to vector<32xf4E2M1FN>
                %171 = vector.bitcast %119 : vector<16xi8> to vector<32xf4E2M1FN>
                %172 = vector.bitcast %120 : vector<16xi8> to vector<32xf4E2M1FN>
                %173 = vector.bitcast %121 : vector<16xi8> to vector<32xf4E2M1FN>
                %174 = vector.bitcast %122 : vector<16xi8> to vector<32xf4E2M1FN>
                %175 = vector.bitcast %123 : vector<16xi8> to vector<32xf4E2M1FN>
                %176 = vector.bitcast %124 : vector<16xi8> to vector<32xf4E2M1FN>
                %177 = vector.bitcast %125 : vector<16xi8> to vector<32xf4E2M1FN>
                %178 = vector.bitcast %126 : vector<16xi8> to vector<32xf4E2M1FN>
                %179 = vector.bitcast %127 : vector<16xi8> to vector<32xf4E2M1FN>
                %180 = vector.bitcast %128 : vector<16xi8> to vector<32xf4E2M1FN>
                %181 = vector.bitcast %129 : vector<16xi8> to vector<32xf4E2M1FN>
                %182 = vector.bitcast %130 : vector<16xi8> to vector<32xf4E2M1FN>
                %183 = vector.bitcast %131 : vector<16xi8> to vector<32xf4E2M1FN>
                %184 = vector.bitcast %132 : vector<16xi8> to vector<32xf4E2M1FN>
                %185 = vector.bitcast %86 : vector<1xi8> to vector<1xf8E8M0FNU>
                %186 = vector.bitcast %89 : vector<1xi8> to vector<1xf8E8M0FNU>
                %187 = vector.bitcast %91 : vector<1xi8> to vector<1xf8E8M0FNU>
                %188 = vector.bitcast %92 : vector<1xi8> to vector<1xf8E8M0FNU>
                %189 = vector.bitcast %94 : vector<1xi8> to vector<1xf8E8M0FNU>
                %190 = vector.bitcast %95 : vector<1xi8> to vector<1xf8E8M0FNU>
                %191 = vector.bitcast %97 : vector<1xi8> to vector<1xf8E8M0FNU>
                %192 = vector.bitcast %98 : vector<1xi8> to vector<1xf8E8M0FNU>
                %193 = vector.bitcast %100 : vector<1xi8> to vector<1xf8E8M0FNU>
                %194 = vector.bitcast %101 : vector<1xi8> to vector<1xf8E8M0FNU>
                %195 = vector.bitcast %103 : vector<1xi8> to vector<1xf8E8M0FNU>
                %196 = vector.bitcast %104 : vector<1xi8> to vector<1xf8E8M0FNU>
                %197 = vector.bitcast %106 : vector<1xi8> to vector<1xf8E8M0FNU>
                %198 = vector.bitcast %107 : vector<1xi8> to vector<1xf8E8M0FNU>
                %199 = vector.bitcast %109 : vector<1xi8> to vector<1xf8E8M0FNU>
                %200 = vector.bitcast %110 : vector<1xi8> to vector<1xf8E8M0FNU>
                %201 = vector.extract %161[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %202 = vector.extract %185[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %203 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%202[0] * %169) + %83#0 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %204 = vector.extract %162[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %205 = vector.extract %186[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %206 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%205[0] * %170) + %203 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %207 = vector.extract %187[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %208 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%207[0] * %171) + %83#1 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %209 = vector.extract %188[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %210 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%209[0] * %172) + %208 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %211 = vector.extract %189[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %212 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%211[0] * %173) + %83#2 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %213 = vector.extract %190[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %214 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%213[0] * %174) + %212 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %215 = vector.extract %191[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %216 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%215[0] * %175) + %83#3 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %217 = vector.extract %192[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %218 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%217[0] * %176) + %216 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %219 = vector.extract %193[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %220 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%219[0] * %177) + %83#4 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %221 = vector.extract %194[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %222 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%221[0] * %178) + %220 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %223 = vector.extract %195[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %224 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%223[0] * %179) + %83#5 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %225 = vector.extract %196[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %226 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%225[0] * %180) + %224 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %227 = vector.extract %197[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %228 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%227[0] * %181) + %83#6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %229 = vector.extract %198[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %230 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%229[0] * %182) + %228 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %231 = vector.extract %199[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %232 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%231[0] * %183) + %83#7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %233 = vector.extract %200[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %234 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%233[0] * %184) + %232 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %235 = vector.extract %163[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %236 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%202[0] * %169) + %83#8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %237 = vector.extract %164[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %238 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%205[0] * %170) + %236 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %239 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%207[0] * %171) + %83#9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %240 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%209[0] * %172) + %239 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %241 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%211[0] * %173) + %83#10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %242 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%213[0] * %174) + %241 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %243 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%215[0] * %175) + %83#11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %244 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%217[0] * %176) + %243 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %245 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%219[0] * %177) + %83#12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %246 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%221[0] * %178) + %245 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %247 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%223[0] * %179) + %83#13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %248 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%225[0] * %180) + %247 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %249 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%227[0] * %181) + %83#14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %250 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%229[0] * %182) + %249 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %251 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%231[0] * %183) + %83#15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %252 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%233[0] * %184) + %251 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %253 = vector.extract %165[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %254 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%202[0] * %169) + %83#16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %255 = vector.extract %166[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %256 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%205[0] * %170) + %254 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %257 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%207[0] * %171) + %83#17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %258 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%209[0] * %172) + %257 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %259 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%211[0] * %173) + %83#18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %260 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%213[0] * %174) + %259 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %261 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%215[0] * %175) + %83#19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %262 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%217[0] * %176) + %261 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %263 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%219[0] * %177) + %83#20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %264 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%221[0] * %178) + %263 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %265 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%223[0] * %179) + %83#21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %266 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%225[0] * %180) + %265 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %267 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%227[0] * %181) + %83#22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %268 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%229[0] * %182) + %267 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %269 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%231[0] * %183) + %83#23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %270 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%233[0] * %184) + %269 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %271 = vector.extract %167[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %272 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%202[0] * %169) + %83#24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %273 = vector.extract %168[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %274 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%205[0] * %170) + %272 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %275 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%207[0] * %171) + %83#25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %276 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%209[0] * %172) + %275 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %277 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%211[0] * %173) + %83#26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %278 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%213[0] * %174) + %277 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %279 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%215[0] * %175) + %83#27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %280 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%217[0] * %176) + %279 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %281 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%219[0] * %177) + %83#28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %282 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%221[0] * %178) + %281 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %283 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%223[0] * %179) + %83#29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %284 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%225[0] * %180) + %283 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %285 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%227[0] * %181) + %83#30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %286 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%229[0] * %182) + %285 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %287 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%231[0] * %183) + %83#31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %288 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%233[0] * %184) + %287 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %289 = vector.extract_strided_slice %206 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %290 = affine.apply #map35()[%block_id_x]
                %291 = affine.apply #map35()[%block_id_y]
                %292 = affine.apply #map36()[%thread_id_x]
                %293 = arith.muli %290, %c16384 overflow<nsw> : index
                %294 = arith.muli %292, %c16384 overflow<nsw> : index
                %295 = arith.addi %293, %291 overflow<nsw> : index
                %296 = arith.addi %294, %84 overflow<nsw> : index
                %reinterpret_cast_13 = memref.reinterpret_cast %4 to offset: [%295], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
                %cast_14 = memref.cast %reinterpret_cast_13 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
                %297 = amdgpu.fat_raw_buffer_cast %cast_14 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                vector.store %289, %297[%296] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %298 = vector.extract_strided_slice %206 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %299 = affine.apply #map37()[%thread_id_x]
                %300 = arith.muli %299, %c16384 overflow<nsw> : index
                %301 = arith.addi %300, %84 overflow<nsw> : index
                vector.store %298, %297[%301] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %302 = vector.extract_strided_slice %206 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %303 = affine.apply #map38()[%thread_id_x]
                %304 = arith.muli %303, %c16384 overflow<nsw> : index
                %305 = arith.addi %304, %84 overflow<nsw> : index
                vector.store %302, %297[%305] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %306 = vector.extract_strided_slice %206 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %307 = affine.apply #map39()[%thread_id_x]
                %308 = arith.muli %307, %c16384 overflow<nsw> : index
                %309 = arith.addi %308, %84 overflow<nsw> : index
                vector.store %306, %297[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %310 = vector.extract_strided_slice %210 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %311 = arith.addi %294, %90 overflow<nsw> : index
                vector.store %310, %297[%311] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %312 = vector.extract_strided_slice %210 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %313 = arith.addi %300, %90 overflow<nsw> : index
                vector.store %312, %297[%313] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %314 = vector.extract_strided_slice %210 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %315 = arith.addi %304, %90 overflow<nsw> : index
                vector.store %314, %297[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %316 = vector.extract_strided_slice %210 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %317 = arith.addi %308, %90 overflow<nsw> : index
                vector.store %316, %297[%317] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %318 = vector.extract_strided_slice %214 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %319 = arith.addi %294, %93 overflow<nsw> : index
                vector.store %318, %297[%319] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %320 = vector.extract_strided_slice %214 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %321 = arith.addi %300, %93 overflow<nsw> : index
                vector.store %320, %297[%321] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %322 = vector.extract_strided_slice %214 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %323 = arith.addi %304, %93 overflow<nsw> : index
                vector.store %322, %297[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %324 = vector.extract_strided_slice %214 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %325 = arith.addi %308, %93 overflow<nsw> : index
                vector.store %324, %297[%325] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %326 = vector.extract_strided_slice %218 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %327 = arith.addi %294, %96 overflow<nsw> : index
                vector.store %326, %297[%327] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %328 = vector.extract_strided_slice %218 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %329 = arith.addi %300, %96 overflow<nsw> : index
                vector.store %328, %297[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %330 = vector.extract_strided_slice %218 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %331 = arith.addi %304, %96 overflow<nsw> : index
                vector.store %330, %297[%331] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %332 = vector.extract_strided_slice %218 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %333 = arith.addi %308, %96 overflow<nsw> : index
                vector.store %332, %297[%333] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %334 = vector.extract_strided_slice %222 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %335 = arith.addi %294, %99 overflow<nsw> : index
                vector.store %334, %297[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %336 = vector.extract_strided_slice %222 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %337 = arith.addi %300, %99 overflow<nsw> : index
                vector.store %336, %297[%337] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %338 = vector.extract_strided_slice %222 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %339 = arith.addi %304, %99 overflow<nsw> : index
                vector.store %338, %297[%339] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %340 = vector.extract_strided_slice %222 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %341 = arith.addi %308, %99 overflow<nsw> : index
                vector.store %340, %297[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %342 = vector.extract_strided_slice %226 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %343 = arith.addi %294, %102 overflow<nsw> : index
                vector.store %342, %297[%343] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %344 = vector.extract_strided_slice %226 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %345 = arith.addi %300, %102 overflow<nsw> : index
                vector.store %344, %297[%345] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %346 = vector.extract_strided_slice %226 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %347 = arith.addi %304, %102 overflow<nsw> : index
                vector.store %346, %297[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %348 = vector.extract_strided_slice %226 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %349 = arith.addi %308, %102 overflow<nsw> : index
                vector.store %348, %297[%349] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %350 = vector.extract_strided_slice %230 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %351 = arith.addi %294, %105 overflow<nsw> : index
                vector.store %350, %297[%351] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %352 = vector.extract_strided_slice %230 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %353 = arith.addi %300, %105 overflow<nsw> : index
                vector.store %352, %297[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %354 = vector.extract_strided_slice %230 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %355 = arith.addi %304, %105 overflow<nsw> : index
                vector.store %354, %297[%355] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %356 = vector.extract_strided_slice %230 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %357 = arith.addi %308, %105 overflow<nsw> : index
                vector.store %356, %297[%357] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %358 = vector.extract_strided_slice %234 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %359 = arith.addi %294, %108 overflow<nsw> : index
                vector.store %358, %297[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %360 = vector.extract_strided_slice %234 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %361 = arith.addi %300, %108 overflow<nsw> : index
                vector.store %360, %297[%361] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %362 = vector.extract_strided_slice %234 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %363 = arith.addi %304, %108 overflow<nsw> : index
                vector.store %362, %297[%363] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %364 = vector.extract_strided_slice %234 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %365 = arith.addi %308, %108 overflow<nsw> : index
                vector.store %364, %297[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %366 = vector.extract_strided_slice %238 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %367 = affine.apply #map40()[%thread_id_x]
                %368 = arith.muli %367, %c16384 overflow<nsw> : index
                %369 = arith.addi %368, %84 overflow<nsw> : index
                vector.store %366, %297[%369] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %370 = vector.extract_strided_slice %238 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %371 = affine.apply #map41()[%thread_id_x]
                %372 = arith.muli %371, %c16384 overflow<nsw> : index
                %373 = arith.addi %372, %84 overflow<nsw> : index
                vector.store %370, %297[%373] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %374 = vector.extract_strided_slice %238 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %375 = affine.apply #map42()[%thread_id_x]
                %376 = arith.muli %375, %c16384 overflow<nsw> : index
                %377 = arith.addi %376, %84 overflow<nsw> : index
                vector.store %374, %297[%377] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %378 = vector.extract_strided_slice %238 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %379 = affine.apply #map43()[%thread_id_x]
                %380 = arith.muli %379, %c16384 overflow<nsw> : index
                %381 = arith.addi %380, %84 overflow<nsw> : index
                vector.store %378, %297[%381] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %382 = vector.extract_strided_slice %240 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %383 = arith.addi %368, %90 overflow<nsw> : index
                vector.store %382, %297[%383] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %384 = vector.extract_strided_slice %240 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %385 = arith.addi %372, %90 overflow<nsw> : index
                vector.store %384, %297[%385] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %386 = vector.extract_strided_slice %240 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %387 = arith.addi %376, %90 overflow<nsw> : index
                vector.store %386, %297[%387] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %388 = vector.extract_strided_slice %240 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %389 = arith.addi %380, %90 overflow<nsw> : index
                vector.store %388, %297[%389] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %390 = vector.extract_strided_slice %242 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %391 = arith.addi %368, %93 overflow<nsw> : index
                vector.store %390, %297[%391] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %392 = vector.extract_strided_slice %242 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %393 = arith.addi %372, %93 overflow<nsw> : index
                vector.store %392, %297[%393] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %394 = vector.extract_strided_slice %242 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %395 = arith.addi %376, %93 overflow<nsw> : index
                vector.store %394, %297[%395] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %396 = vector.extract_strided_slice %242 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %397 = arith.addi %380, %93 overflow<nsw> : index
                vector.store %396, %297[%397] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %398 = vector.extract_strided_slice %244 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %399 = arith.addi %368, %96 overflow<nsw> : index
                vector.store %398, %297[%399] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %400 = vector.extract_strided_slice %244 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %401 = arith.addi %372, %96 overflow<nsw> : index
                vector.store %400, %297[%401] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %402 = vector.extract_strided_slice %244 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %403 = arith.addi %376, %96 overflow<nsw> : index
                vector.store %402, %297[%403] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %404 = vector.extract_strided_slice %244 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %405 = arith.addi %380, %96 overflow<nsw> : index
                vector.store %404, %297[%405] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %406 = vector.extract_strided_slice %246 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %407 = arith.addi %368, %99 overflow<nsw> : index
                vector.store %406, %297[%407] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %408 = vector.extract_strided_slice %246 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %409 = arith.addi %372, %99 overflow<nsw> : index
                vector.store %408, %297[%409] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %410 = vector.extract_strided_slice %246 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %411 = arith.addi %376, %99 overflow<nsw> : index
                vector.store %410, %297[%411] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %412 = vector.extract_strided_slice %246 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %413 = arith.addi %380, %99 overflow<nsw> : index
                vector.store %412, %297[%413] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %414 = vector.extract_strided_slice %248 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %415 = arith.addi %368, %102 overflow<nsw> : index
                vector.store %414, %297[%415] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %416 = vector.extract_strided_slice %248 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %417 = arith.addi %372, %102 overflow<nsw> : index
                vector.store %416, %297[%417] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %418 = vector.extract_strided_slice %248 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %419 = arith.addi %376, %102 overflow<nsw> : index
                vector.store %418, %297[%419] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %420 = vector.extract_strided_slice %248 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %421 = arith.addi %380, %102 overflow<nsw> : index
                vector.store %420, %297[%421] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %422 = vector.extract_strided_slice %250 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %423 = arith.addi %368, %105 overflow<nsw> : index
                vector.store %422, %297[%423] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %424 = vector.extract_strided_slice %250 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %425 = arith.addi %372, %105 overflow<nsw> : index
                vector.store %424, %297[%425] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %426 = vector.extract_strided_slice %250 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %427 = arith.addi %376, %105 overflow<nsw> : index
                vector.store %426, %297[%427] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %428 = vector.extract_strided_slice %250 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %429 = arith.addi %380, %105 overflow<nsw> : index
                vector.store %428, %297[%429] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %430 = vector.extract_strided_slice %252 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %431 = arith.addi %368, %108 overflow<nsw> : index
                vector.store %430, %297[%431] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %432 = vector.extract_strided_slice %252 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %433 = arith.addi %372, %108 overflow<nsw> : index
                vector.store %432, %297[%433] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %434 = vector.extract_strided_slice %252 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %435 = arith.addi %376, %108 overflow<nsw> : index
                vector.store %434, %297[%435] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %436 = vector.extract_strided_slice %252 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %437 = arith.addi %380, %108 overflow<nsw> : index
                vector.store %436, %297[%437] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %438 = vector.extract_strided_slice %256 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %439 = affine.apply #map44()[%thread_id_x]
                %440 = arith.muli %439, %c16384 overflow<nsw> : index
                %441 = arith.addi %440, %84 overflow<nsw> : index
                vector.store %438, %297[%441] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %442 = vector.extract_strided_slice %256 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %443 = affine.apply #map45()[%thread_id_x]
                %444 = arith.muli %443, %c16384 overflow<nsw> : index
                %445 = arith.addi %444, %84 overflow<nsw> : index
                vector.store %442, %297[%445] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %446 = vector.extract_strided_slice %256 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %447 = affine.apply #map46()[%thread_id_x]
                %448 = arith.muli %447, %c16384 overflow<nsw> : index
                %449 = arith.addi %448, %84 overflow<nsw> : index
                vector.store %446, %297[%449] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %450 = vector.extract_strided_slice %256 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %451 = affine.apply #map47()[%thread_id_x]
                %452 = arith.muli %451, %c16384 overflow<nsw> : index
                %453 = arith.addi %452, %84 overflow<nsw> : index
                vector.store %450, %297[%453] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %454 = vector.extract_strided_slice %258 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %455 = arith.addi %440, %90 overflow<nsw> : index
                vector.store %454, %297[%455] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %456 = vector.extract_strided_slice %258 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %457 = arith.addi %444, %90 overflow<nsw> : index
                vector.store %456, %297[%457] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %458 = vector.extract_strided_slice %258 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %459 = arith.addi %448, %90 overflow<nsw> : index
                vector.store %458, %297[%459] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %460 = vector.extract_strided_slice %258 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %461 = arith.addi %452, %90 overflow<nsw> : index
                vector.store %460, %297[%461] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %462 = vector.extract_strided_slice %260 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %463 = arith.addi %440, %93 overflow<nsw> : index
                vector.store %462, %297[%463] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %464 = vector.extract_strided_slice %260 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %465 = arith.addi %444, %93 overflow<nsw> : index
                vector.store %464, %297[%465] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %466 = vector.extract_strided_slice %260 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %467 = arith.addi %448, %93 overflow<nsw> : index
                vector.store %466, %297[%467] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %468 = vector.extract_strided_slice %260 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %469 = arith.addi %452, %93 overflow<nsw> : index
                vector.store %468, %297[%469] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %470 = vector.extract_strided_slice %262 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %471 = arith.addi %440, %96 overflow<nsw> : index
                vector.store %470, %297[%471] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %472 = vector.extract_strided_slice %262 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %473 = arith.addi %444, %96 overflow<nsw> : index
                vector.store %472, %297[%473] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %474 = vector.extract_strided_slice %262 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %475 = arith.addi %448, %96 overflow<nsw> : index
                vector.store %474, %297[%475] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %476 = vector.extract_strided_slice %262 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %477 = arith.addi %452, %96 overflow<nsw> : index
                vector.store %476, %297[%477] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %478 = vector.extract_strided_slice %264 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %479 = arith.addi %440, %99 overflow<nsw> : index
                vector.store %478, %297[%479] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %480 = vector.extract_strided_slice %264 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %481 = arith.addi %444, %99 overflow<nsw> : index
                vector.store %480, %297[%481] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %482 = vector.extract_strided_slice %264 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %483 = arith.addi %448, %99 overflow<nsw> : index
                vector.store %482, %297[%483] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %484 = vector.extract_strided_slice %264 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %485 = arith.addi %452, %99 overflow<nsw> : index
                vector.store %484, %297[%485] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %486 = vector.extract_strided_slice %266 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %487 = arith.addi %440, %102 overflow<nsw> : index
                vector.store %486, %297[%487] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %488 = vector.extract_strided_slice %266 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %489 = arith.addi %444, %102 overflow<nsw> : index
                vector.store %488, %297[%489] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %490 = vector.extract_strided_slice %266 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %491 = arith.addi %448, %102 overflow<nsw> : index
                vector.store %490, %297[%491] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %492 = vector.extract_strided_slice %266 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %493 = arith.addi %452, %102 overflow<nsw> : index
                vector.store %492, %297[%493] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %494 = vector.extract_strided_slice %268 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %495 = arith.addi %440, %105 overflow<nsw> : index
                vector.store %494, %297[%495] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %496 = vector.extract_strided_slice %268 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %497 = arith.addi %444, %105 overflow<nsw> : index
                vector.store %496, %297[%497] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %498 = vector.extract_strided_slice %268 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %499 = arith.addi %448, %105 overflow<nsw> : index
                vector.store %498, %297[%499] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %500 = vector.extract_strided_slice %268 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %501 = arith.addi %452, %105 overflow<nsw> : index
                vector.store %500, %297[%501] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %502 = vector.extract_strided_slice %270 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %503 = arith.addi %440, %108 overflow<nsw> : index
                vector.store %502, %297[%503] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %504 = vector.extract_strided_slice %270 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %505 = arith.addi %444, %108 overflow<nsw> : index
                vector.store %504, %297[%505] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %506 = vector.extract_strided_slice %270 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %507 = arith.addi %448, %108 overflow<nsw> : index
                vector.store %506, %297[%507] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %508 = vector.extract_strided_slice %270 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %509 = arith.addi %452, %108 overflow<nsw> : index
                vector.store %508, %297[%509] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %510 = vector.extract_strided_slice %274 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %511 = affine.apply #map48()[%thread_id_x]
                %512 = arith.muli %511, %c16384 overflow<nsw> : index
                %513 = arith.addi %512, %84 overflow<nsw> : index
                vector.store %510, %297[%513] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %514 = vector.extract_strided_slice %274 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %515 = affine.apply #map49()[%thread_id_x]
                %516 = arith.muli %515, %c16384 overflow<nsw> : index
                %517 = arith.addi %516, %84 overflow<nsw> : index
                vector.store %514, %297[%517] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %518 = vector.extract_strided_slice %274 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %519 = affine.apply #map50()[%thread_id_x]
                %520 = arith.muli %519, %c16384 overflow<nsw> : index
                %521 = arith.addi %520, %84 overflow<nsw> : index
                vector.store %518, %297[%521] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %522 = vector.extract_strided_slice %274 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %523 = affine.apply #map51()[%thread_id_x]
                %524 = arith.muli %523, %c16384 overflow<nsw> : index
                %525 = arith.addi %524, %84 overflow<nsw> : index
                vector.store %522, %297[%525] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %526 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %527 = arith.addi %512, %90 overflow<nsw> : index
                vector.store %526, %297[%527] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %528 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %529 = arith.addi %516, %90 overflow<nsw> : index
                vector.store %528, %297[%529] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %530 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %531 = arith.addi %520, %90 overflow<nsw> : index
                vector.store %530, %297[%531] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %532 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %533 = arith.addi %524, %90 overflow<nsw> : index
                vector.store %532, %297[%533] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %534 = vector.extract_strided_slice %278 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %535 = arith.addi %512, %93 overflow<nsw> : index
                vector.store %534, %297[%535] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %536 = vector.extract_strided_slice %278 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %537 = arith.addi %516, %93 overflow<nsw> : index
                vector.store %536, %297[%537] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %538 = vector.extract_strided_slice %278 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %539 = arith.addi %520, %93 overflow<nsw> : index
                vector.store %538, %297[%539] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %540 = vector.extract_strided_slice %278 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %541 = arith.addi %524, %93 overflow<nsw> : index
                vector.store %540, %297[%541] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %542 = vector.extract_strided_slice %280 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %543 = arith.addi %512, %96 overflow<nsw> : index
                vector.store %542, %297[%543] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %544 = vector.extract_strided_slice %280 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %545 = arith.addi %516, %96 overflow<nsw> : index
                vector.store %544, %297[%545] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %546 = vector.extract_strided_slice %280 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %547 = arith.addi %520, %96 overflow<nsw> : index
                vector.store %546, %297[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %548 = vector.extract_strided_slice %280 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %549 = arith.addi %524, %96 overflow<nsw> : index
                vector.store %548, %297[%549] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %550 = vector.extract_strided_slice %282 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %551 = arith.addi %512, %99 overflow<nsw> : index
                vector.store %550, %297[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %552 = vector.extract_strided_slice %282 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %553 = arith.addi %516, %99 overflow<nsw> : index
                vector.store %552, %297[%553] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %554 = vector.extract_strided_slice %282 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %555 = arith.addi %520, %99 overflow<nsw> : index
                vector.store %554, %297[%555] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %556 = vector.extract_strided_slice %282 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %557 = arith.addi %524, %99 overflow<nsw> : index
                vector.store %556, %297[%557] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %558 = vector.extract_strided_slice %284 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %559 = arith.addi %512, %102 overflow<nsw> : index
                vector.store %558, %297[%559] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %560 = vector.extract_strided_slice %284 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %561 = arith.addi %516, %102 overflow<nsw> : index
                vector.store %560, %297[%561] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %562 = vector.extract_strided_slice %284 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %563 = arith.addi %520, %102 overflow<nsw> : index
                vector.store %562, %297[%563] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %564 = vector.extract_strided_slice %284 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %565 = arith.addi %524, %102 overflow<nsw> : index
                vector.store %564, %297[%565] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %566 = vector.extract_strided_slice %286 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %567 = arith.addi %512, %105 overflow<nsw> : index
                vector.store %566, %297[%567] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %568 = vector.extract_strided_slice %286 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %569 = arith.addi %516, %105 overflow<nsw> : index
                vector.store %568, %297[%569] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %570 = vector.extract_strided_slice %286 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %571 = arith.addi %520, %105 overflow<nsw> : index
                vector.store %570, %297[%571] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %572 = vector.extract_strided_slice %286 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %573 = arith.addi %524, %105 overflow<nsw> : index
                vector.store %572, %297[%573] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %574 = vector.extract_strided_slice %288 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %575 = arith.addi %512, %108 overflow<nsw> : index
                vector.store %574, %297[%575] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %576 = vector.extract_strided_slice %288 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %577 = arith.addi %516, %108 overflow<nsw> : index
                vector.store %576, %297[%577] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %578 = vector.extract_strided_slice %288 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %579 = arith.addi %520, %108 overflow<nsw> : index
                vector.store %578, %297[%579] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %580 = vector.extract_strided_slice %288 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %581 = arith.addi %524, %108 overflow<nsw> : index
                vector.store %580, %297[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<16384x8192xi8>
            %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<16384x512xi8>
            %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<16384x8192xi8>
            %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<16384x512xi8>
            %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<16384x16384xf32>
            %5 = flow.dispatch @gemm::@gemm(%0, %1, %2, %3, %4) : (tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x16384xf32>) -> %4
            %6 = hal.tensor.barrier join(%5 : tensor<16384x16384xf32>) => %arg6 : !hal.fence
            %7 = hal.tensor.export %6 : tensor<16384x16384xf32> -> !hal.buffer_view
            return %7 : !hal.buffer_view
        }
        }
    """
    mlir2 = """
        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 16)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
        #map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
        #map9 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
        #map11 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
        #map12 = affine_map<()[s0] -> ((s0 floordiv 2) mod 2)>
        #map13 = affine_map<()[s0] -> (s0 mod 2)>
        #map14 = affine_map<()[s0] -> (s0 * 4)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 32 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 256)>
        #map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
        #map18 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
        #map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
        #map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
        #map22 = affine_map<()[s0] -> (s0 * 4 + (s0 mod 64) floordiv 16 - (s0 floordiv 2) * 8)>
        #map23 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
        #map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
        #map25 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
        #map26 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
        #map27 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
        #map28 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
        #map29 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
        #map30 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
        #map31 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map32 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
        #map33 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 128)>
        #map34 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 + 8)>
        #map35 = affine_map<()[s0] -> (s0 * 256)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
        #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
        #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
        #map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
        #map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
        #map47 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
        #map48 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
        #map49 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
        #map50 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
        #map51 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c64 = arith.constant 64 : index
            %c1 = arith.constant 1 : index
            stream.return %c64, %c64, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c512_i14 = arith.constant 512 : i14
                %c-8192_i14 = arith.constant -8192 : i14
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c16384 = arith.constant 16384 : index
                %c63 = arith.constant 63 : index
                %c512 = arith.constant 512 : index
                %c2147483646_i64 = arith.constant 2147483646 : i64
                %c8192 = arith.constant 8192 : index
                %c1 = arith.constant 1 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
                %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
                %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 64
                %block_id_y = gpu.block_id  y upper_bound 64
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %alloc = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_0 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_3 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_4 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_5 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_6 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %5 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %6 = affine.apply #map1()[%thread_id_x]
                %7 = affine.apply #map2()[%thread_id_x]
                %8 = arith.xori %7, %6 : index
                %9 = affine.apply #map3()[%8]
                %10 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %11 = gpu.subgroup_broadcast %10,  first_active_lane : index
                %12 = gpu.subgroup_broadcast %c0,  first_active_lane : index
                %13 = arith.muli %5, %c8192 overflow<nsw> : index
                %14 = arith.addi %13, %9 overflow<nsw> : index
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast = memref.cast %reinterpret_cast : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %15 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %15[%14], %alloc_6[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %16 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
                %17 = affine.apply #map6()[%thread_id_x, %thread_id_y]
                %18 = gpu.subgroup_broadcast %17,  first_active_lane : index
                %19 = arith.muli %16, %c8192 overflow<nsw> : index
                %20 = arith.addi %19, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%20], %alloc_6[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %21 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
                %22 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                %23 = gpu.subgroup_broadcast %22,  first_active_lane : index
                %24 = arith.muli %21, %c8192 overflow<nsw> : index
                %25 = arith.addi %24, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%25], %alloc_6[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %26 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
                %27 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %28 = gpu.subgroup_broadcast %27,  first_active_lane : index
                %29 = arith.muli %26, %c8192 overflow<nsw> : index
                %30 = arith.addi %29, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%30], %alloc_6[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %31 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_x]
                %32 = affine.apply #map12()[%thread_id_x]
                %33 = affine.apply #map13()[%thread_id_x]
                %34 = arith.xori %33, %32 : index
                %35 = affine.apply #map14()[%34]
                %36 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %37 = gpu.subgroup_broadcast %36,  first_active_lane : index
                %38 = arith.muli %31, %c512 overflow<nsw> : index
                %39 = arith.addi %38, %35 overflow<nsw> : index
                %reinterpret_cast_7 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %40 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %40[%39], %alloc_4[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                %41 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
                %42 = arith.muli %41, %c8192 overflow<nsw> : index
                %43 = arith.addi %42, %9 overflow<nsw> : index
                %reinterpret_cast_9 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_10 = memref.cast %reinterpret_cast_9 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %44 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %44[%43], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %45 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                %46 = arith.muli %45, %c8192 overflow<nsw> : index
                %47 = arith.addi %46, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%47], %alloc_2[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %48 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
                %49 = arith.muli %48, %c8192 overflow<nsw> : index
                %50 = arith.addi %49, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%50], %alloc_2[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %51 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_y]
                %52 = arith.muli %51, %c8192 overflow<nsw> : index
                %53 = arith.addi %52, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%53], %alloc_2[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %54 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_y]
                %55 = arith.muli %54, %c512 overflow<nsw> : index
                %56 = arith.addi %55, %35 overflow<nsw> : index
                %reinterpret_cast_11 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_12 = memref.cast %reinterpret_cast_11 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %57 = amdgpu.fat_raw_buffer_cast %cast_12 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %57[%56], %alloc_0[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.s.barrier
                %58 = affine.apply #map16()[%thread_id_x, %thread_id_y]
                %59 = arith.index_cast %58 : index to i32
                %60 = arith.cmpi sge, %59, %c4_i32 : i32
                %61 = arith.cmpi slt, %59, %c4_i32 : i32
                scf.if %60 {
                rocdl.s.barrier
                }
                %62 = affine.apply #map17()[%thread_id_x]
                %63 = affine.apply #map18()[%thread_id_x]
                %64 = arith.xori %63, %7 : index
                %65 = affine.apply #map3()[%64]
                %66 = affine.apply #map19()[%thread_id_x]
                %67 = affine.apply #map20()[%thread_id_x]
                %68 = affine.apply #map21()[%thread_id_x]
                %69 = affine.apply #map22()[%thread_id_x]
                %70 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %71 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %72 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %73 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %74 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %75 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %76 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %77 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %78 = affine.apply #map31()[%thread_id_x]
                %79 = arith.xori %78, %7 : index
                %80 = affine.apply #map3()[%79]
                %81 = arith.xori %33, %c1 : index
                %82 = affine.apply #map32()[%thread_id_x, %81]
                %83:40 = scf.for %arg5 = %c0 to %c63 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %alloc_6, %arg39 = %alloc_5, %arg40 = %alloc_4, %arg41 = %alloc_3, %arg42 = %alloc_2, %arg43 = %alloc_1, %arg44 = %alloc_0, %arg45 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>) {
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.s.barrier
                //amdgpu.lds_barrier
                %582 = affine.apply #map33()[%arg5, %8]
                %583 = arith.addi %13, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%583], %arg39[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %584 = arith.addi %19, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%584], %arg39[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %585 = arith.addi %24, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%585], %arg39[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %586 = arith.addi %29, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%586], %arg39[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %587 = affine.apply #map34()[%arg5, %34]
                %588 = arith.addi %38, %587 overflow<nsw> : index
                amdgpu.gather_to_lds %40[%588], %arg41[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                %589 = arith.addi %42, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%589], %arg43[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %590 = arith.addi %46, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%590], %arg43[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %591 = arith.addi %49, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%591], %arg43[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %592 = arith.addi %52, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%592], %arg43[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %593 = arith.addi %55, %587 overflow<nsw> : index
                amdgpu.gather_to_lds %57[%593], %arg45[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(10)
                //rocdl.s.waitcnt 16368
                //amdgpu.lds_barrier
                %594 = vector.load %arg38[%62, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %595 = vector.load %arg38[%66, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %596 = vector.load %arg38[%67, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %597 = vector.load %arg38[%68, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %598 = vector.load %arg40[%62, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %599 = vector.load %arg40[%66, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %600 = vector.load %arg40[%67, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %601 = vector.load %arg40[%68, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %602 = vector.load %arg42[%70, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %603 = vector.load %arg42[%71, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %604 = vector.load %arg42[%72, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %605 = vector.load %arg42[%73, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %606 = vector.load %arg42[%74, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %607 = vector.load %arg42[%75, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %608 = vector.load %arg42[%76, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %609 = vector.load %arg42[%77, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %610 = vector.load %arg44[%70, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %611 = vector.load %arg44[%71, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %612 = vector.load %arg44[%72, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %613 = vector.load %arg44[%73, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %614 = vector.load %arg44[%74, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %615 = vector.load %arg44[%75, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %616 = vector.load %arg44[%76, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %617 = vector.load %arg44[%77, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %618 = vector.bitcast %594 : vector<16xi8> to vector<32xf4E2M1FN>
                %619 = vector.bitcast %595 : vector<16xi8> to vector<32xf4E2M1FN>
                %620 = vector.bitcast %596 : vector<16xi8> to vector<32xf4E2M1FN>
                %621 = vector.bitcast %597 : vector<16xi8> to vector<32xf4E2M1FN>
                %622 = vector.bitcast %598 : vector<1xi8> to vector<1xf8E8M0FNU>
                %623 = vector.bitcast %599 : vector<1xi8> to vector<1xf8E8M0FNU>
                %624 = vector.bitcast %600 : vector<1xi8> to vector<1xf8E8M0FNU>
                %625 = vector.bitcast %601 : vector<1xi8> to vector<1xf8E8M0FNU>
                %626 = vector.bitcast %602 : vector<16xi8> to vector<32xf4E2M1FN>
                %627 = vector.bitcast %603 : vector<16xi8> to vector<32xf4E2M1FN>
                %628 = vector.bitcast %604 : vector<16xi8> to vector<32xf4E2M1FN>
                %629 = vector.bitcast %605 : vector<16xi8> to vector<32xf4E2M1FN>
                %630 = vector.bitcast %606 : vector<16xi8> to vector<32xf4E2M1FN>
                %631 = vector.bitcast %607 : vector<16xi8> to vector<32xf4E2M1FN>
                %632 = vector.bitcast %608 : vector<16xi8> to vector<32xf4E2M1FN>
                %633 = vector.bitcast %609 : vector<16xi8> to vector<32xf4E2M1FN>
                %634 = vector.bitcast %610 : vector<1xi8> to vector<1xf8E8M0FNU>
                %635 = vector.bitcast %611 : vector<1xi8> to vector<1xf8E8M0FNU>
                %636 = vector.bitcast %612 : vector<1xi8> to vector<1xf8E8M0FNU>
                %637 = vector.bitcast %613 : vector<1xi8> to vector<1xf8E8M0FNU>
                %638 = vector.bitcast %614 : vector<1xi8> to vector<1xf8E8M0FNU>
                %639 = vector.bitcast %615 : vector<1xi8> to vector<1xf8E8M0FNU>
                %640 = vector.bitcast %616 : vector<1xi8> to vector<1xf8E8M0FNU>
                %641 = vector.bitcast %617 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %642 = vector.extract %622[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %643 = vector.extract %634[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %644 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%643[0] * %626) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %645 = vector.extract %635[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %646 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%645[0] * %627) + %arg7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %647 = vector.extract %636[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %648 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%647[0] * %628) + %arg8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %649 = vector.extract %637[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %650 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%649[0] * %629) + %arg9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %651 = vector.extract %638[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %652 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%651[0] * %630) + %arg10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %653 = vector.extract %639[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %654 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%653[0] * %631) + %arg11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %655 = vector.extract %640[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %656 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%655[0] * %632) + %arg12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %657 = vector.extract %641[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %658 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%657[0] * %633) + %arg13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %659 = vector.extract %623[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %660 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%643[0] * %626) + %arg14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %661 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%645[0] * %627) + %arg15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %662 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%647[0] * %628) + %arg16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %663 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%649[0] * %629) + %arg17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %664 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%651[0] * %630) + %arg18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %665 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%653[0] * %631) + %arg19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %666 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%655[0] * %632) + %arg20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %667 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%657[0] * %633) + %arg21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %668 = vector.extract %624[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %669 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%643[0] * %626) + %arg22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %670 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%645[0] * %627) + %arg23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %671 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%647[0] * %628) + %arg24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %672 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%649[0] * %629) + %arg25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %673 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%651[0] * %630) + %arg26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %674 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%653[0] * %631) + %arg27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %675 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%655[0] * %632) + %arg28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %676 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%657[0] * %633) + %arg29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %677 = vector.extract %625[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %678 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%643[0] * %626) + %arg30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %679 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%645[0] * %627) + %arg31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %680 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%647[0] * %628) + %arg32 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %681 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%649[0] * %629) + %arg33 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %682 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%651[0] * %630) + %arg34 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %683 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%653[0] * %631) + %arg35 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %684 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%655[0] * %632) + %arg36 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %685 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%657[0] * %633) + %arg37 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                %686 = vector.load %arg38[%62, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %687 = vector.load %arg38[%66, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %688 = vector.load %arg38[%67, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %689 = vector.load %arg38[%68, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %690 = vector.load %arg40[%62, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %691 = vector.load %arg40[%66, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %692 = vector.load %arg40[%67, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %693 = vector.load %arg40[%68, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %694 = vector.load %arg42[%70, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %695 = vector.load %arg42[%71, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %696 = vector.load %arg42[%72, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %697 = vector.load %arg42[%73, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %698 = vector.load %arg42[%74, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %699 = vector.load %arg42[%75, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %700 = vector.load %arg42[%76, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %701 = vector.load %arg42[%77, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %702 = vector.load %arg44[%70, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %703 = vector.load %arg44[%71, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %704 = vector.load %arg44[%72, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %705 = vector.load %arg44[%73, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %706 = vector.load %arg44[%74, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %707 = vector.load %arg44[%75, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %708 = vector.load %arg44[%76, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %709 = vector.load %arg44[%77, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %710 = vector.bitcast %686 : vector<16xi8> to vector<32xf4E2M1FN>
                %711 = vector.bitcast %687 : vector<16xi8> to vector<32xf4E2M1FN>
                %712 = vector.bitcast %688 : vector<16xi8> to vector<32xf4E2M1FN>
                %713 = vector.bitcast %689 : vector<16xi8> to vector<32xf4E2M1FN>
                %714 = vector.bitcast %690 : vector<1xi8> to vector<1xf8E8M0FNU>
                %715 = vector.bitcast %691 : vector<1xi8> to vector<1xf8E8M0FNU>
                %716 = vector.bitcast %692 : vector<1xi8> to vector<1xf8E8M0FNU>
                %717 = vector.bitcast %693 : vector<1xi8> to vector<1xf8E8M0FNU>
                %718 = vector.bitcast %694 : vector<16xi8> to vector<32xf4E2M1FN>
                %719 = vector.bitcast %695 : vector<16xi8> to vector<32xf4E2M1FN>
                %720 = vector.bitcast %696 : vector<16xi8> to vector<32xf4E2M1FN>
                %721 = vector.bitcast %697 : vector<16xi8> to vector<32xf4E2M1FN>
                %722 = vector.bitcast %698 : vector<16xi8> to vector<32xf4E2M1FN>
                %723 = vector.bitcast %699 : vector<16xi8> to vector<32xf4E2M1FN>
                %724 = vector.bitcast %700 : vector<16xi8> to vector<32xf4E2M1FN>
                %725 = vector.bitcast %701 : vector<16xi8> to vector<32xf4E2M1FN>
                %726 = vector.bitcast %702 : vector<1xi8> to vector<1xf8E8M0FNU>
                %727 = vector.bitcast %703 : vector<1xi8> to vector<1xf8E8M0FNU>
                %728 = vector.bitcast %704 : vector<1xi8> to vector<1xf8E8M0FNU>
                %729 = vector.bitcast %705 : vector<1xi8> to vector<1xf8E8M0FNU>
                %730 = vector.bitcast %706 : vector<1xi8> to vector<1xf8E8M0FNU>
                %731 = vector.bitcast %707 : vector<1xi8> to vector<1xf8E8M0FNU>
                %732 = vector.bitcast %708 : vector<1xi8> to vector<1xf8E8M0FNU>
                %733 = vector.bitcast %709 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %734 = vector.extract %714[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %735 = vector.extract %726[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %736 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%735[0] * %718) + %644 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %737 = vector.extract %727[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %738 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%737[0] * %719) + %646 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %739 = vector.extract %728[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %740 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%739[0] * %720) + %648 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %741 = vector.extract %729[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %742 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%741[0] * %721) + %650 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %743 = vector.extract %730[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %744 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%743[0] * %722) + %652 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %745 = vector.extract %731[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %746 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%745[0] * %723) + %654 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %747 = vector.extract %732[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %748 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%747[0] * %724) + %656 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %749 = vector.extract %733[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %750 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%749[0] * %725) + %658 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %751 = vector.extract %715[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %752 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%735[0] * %718) + %660 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %753 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%737[0] * %719) + %661 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %754 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%739[0] * %720) + %662 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %755 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%741[0] * %721) + %663 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %756 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%743[0] * %722) + %664 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %757 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%745[0] * %723) + %665 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %758 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%747[0] * %724) + %666 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %759 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%749[0] * %725) + %667 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %760 = vector.extract %716[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %761 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%735[0] * %718) + %669 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %762 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%737[0] * %719) + %670 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %763 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%739[0] * %720) + %671 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %764 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%741[0] * %721) + %672 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %765 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%743[0] * %722) + %673 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %766 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%745[0] * %723) + %674 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %767 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%747[0] * %724) + %675 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %768 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%749[0] * %725) + %676 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %769 = vector.extract %717[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %770 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%735[0] * %718) + %678 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %771 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%737[0] * %719) + %679 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %772 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%739[0] * %720) + %680 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %773 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%741[0] * %721) + %681 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %774 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%743[0] * %722) + %682 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %775 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%745[0] * %723) + %683 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %776 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%747[0] * %724) + %684 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %777 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%749[0] * %725) + %685 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                scf.yield %736, %738, %740, %742, %744, %746, %748, %750, %752, %753, %754, %755, %756, %757, %758, %759, %761, %762, %763, %764, %765, %766, %767, %768, %770, %771, %772, %773, %774, %775, %776, %777, %arg39, %arg38, %arg41, %arg40, %arg43, %arg42, %arg45, %arg44 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                }
                scf.if %61 {
                rocdl.s.barrier
                }
                amdgpu.lds_barrier
                %84 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %85 = affine.apply #map22()[%thread_id_x]
                %86 = vector.load %83#38[%84, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %87 = arith.xori %33, %c1 : index
                %88 = affine.apply #map32()[%thread_id_x, %87]
                %89 = vector.load %83#38[%84, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %90 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %91 = vector.load %83#38[%90, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %92 = vector.load %83#38[%90, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %93 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %94 = vector.load %83#38[%93, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %95 = vector.load %83#38[%93, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %96 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %97 = vector.load %83#38[%96, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %98 = vector.load %83#38[%96, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %99 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %100 = vector.load %83#38[%99, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %101 = vector.load %83#38[%99, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %102 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %103 = vector.load %83#38[%102, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %104 = vector.load %83#38[%102, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %105 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %106 = vector.load %83#38[%105, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %107 = vector.load %83#38[%105, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %108 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %109 = vector.load %83#38[%108, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %110 = vector.load %83#38[%108, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %111 = affine.apply #map18()[%thread_id_x]
                %112 = arith.xori %111, %7 : index
                %113 = affine.apply #map3()[%112]
                %114 = vector.load %83#36[%84, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %115 = affine.apply #map31()[%thread_id_x]
                %116 = arith.xori %115, %7 : index
                %117 = affine.apply #map3()[%116]
                %118 = vector.load %83#36[%84, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %119 = vector.load %83#36[%90, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %120 = vector.load %83#36[%90, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %121 = vector.load %83#36[%93, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %122 = vector.load %83#36[%93, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %123 = vector.load %83#36[%96, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %124 = vector.load %83#36[%96, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %125 = vector.load %83#36[%99, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %126 = vector.load %83#36[%99, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %127 = vector.load %83#36[%102, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %128 = vector.load %83#36[%102, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %129 = vector.load %83#36[%105, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %130 = vector.load %83#36[%105, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %131 = vector.load %83#36[%108, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %132 = vector.load %83#36[%108, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %133 = affine.apply #map17()[%thread_id_x]
                %134 = vector.load %83#34[%133, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %135 = vector.load %83#34[%133, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %136 = affine.apply #map19()[%thread_id_x]
                %137 = vector.load %83#34[%136, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %138 = vector.load %83#34[%136, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %139 = affine.apply #map20()[%thread_id_x]
                %140 = vector.load %83#34[%139, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %141 = vector.load %83#34[%139, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %142 = affine.apply #map21()[%thread_id_x]
                %143 = vector.load %83#34[%142, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %144 = vector.load %83#34[%142, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %145 = vector.load %83#32[%133, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %146 = vector.load %83#32[%133, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %147 = vector.load %83#32[%136, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %148 = vector.load %83#32[%136, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %149 = vector.load %83#32[%139, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %150 = vector.load %83#32[%139, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %151 = vector.load %83#32[%142, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %152 = vector.load %83#32[%142, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %153 = vector.bitcast %145 : vector<16xi8> to vector<32xf4E2M1FN>
                %154 = vector.bitcast %146 : vector<16xi8> to vector<32xf4E2M1FN>
                %155 = vector.bitcast %147 : vector<16xi8> to vector<32xf4E2M1FN>
                %156 = vector.bitcast %148 : vector<16xi8> to vector<32xf4E2M1FN>
                %157 = vector.bitcast %149 : vector<16xi8> to vector<32xf4E2M1FN>
                %158 = vector.bitcast %150 : vector<16xi8> to vector<32xf4E2M1FN>
                %159 = vector.bitcast %151 : vector<16xi8> to vector<32xf4E2M1FN>
                %160 = vector.bitcast %152 : vector<16xi8> to vector<32xf4E2M1FN>
                %161 = vector.bitcast %134 : vector<1xi8> to vector<1xf8E8M0FNU>
                %162 = vector.bitcast %135 : vector<1xi8> to vector<1xf8E8M0FNU>
                %163 = vector.bitcast %137 : vector<1xi8> to vector<1xf8E8M0FNU>
                %164 = vector.bitcast %138 : vector<1xi8> to vector<1xf8E8M0FNU>
                %165 = vector.bitcast %140 : vector<1xi8> to vector<1xf8E8M0FNU>
                %166 = vector.bitcast %141 : vector<1xi8> to vector<1xf8E8M0FNU>
                %167 = vector.bitcast %143 : vector<1xi8> to vector<1xf8E8M0FNU>
                %168 = vector.bitcast %144 : vector<1xi8> to vector<1xf8E8M0FNU>
                %169 = vector.bitcast %114 : vector<16xi8> to vector<32xf4E2M1FN>
                %170 = vector.bitcast %118 : vector<16xi8> to vector<32xf4E2M1FN>
                %171 = vector.bitcast %119 : vector<16xi8> to vector<32xf4E2M1FN>
                %172 = vector.bitcast %120 : vector<16xi8> to vector<32xf4E2M1FN>
                %173 = vector.bitcast %121 : vector<16xi8> to vector<32xf4E2M1FN>
                %174 = vector.bitcast %122 : vector<16xi8> to vector<32xf4E2M1FN>
                %175 = vector.bitcast %123 : vector<16xi8> to vector<32xf4E2M1FN>
                %176 = vector.bitcast %124 : vector<16xi8> to vector<32xf4E2M1FN>
                %177 = vector.bitcast %125 : vector<16xi8> to vector<32xf4E2M1FN>
                %178 = vector.bitcast %126 : vector<16xi8> to vector<32xf4E2M1FN>
                %179 = vector.bitcast %127 : vector<16xi8> to vector<32xf4E2M1FN>
                %180 = vector.bitcast %128 : vector<16xi8> to vector<32xf4E2M1FN>
                %181 = vector.bitcast %129 : vector<16xi8> to vector<32xf4E2M1FN>
                %182 = vector.bitcast %130 : vector<16xi8> to vector<32xf4E2M1FN>
                %183 = vector.bitcast %131 : vector<16xi8> to vector<32xf4E2M1FN>
                %184 = vector.bitcast %132 : vector<16xi8> to vector<32xf4E2M1FN>
                %185 = vector.bitcast %86 : vector<1xi8> to vector<1xf8E8M0FNU>
                %186 = vector.bitcast %89 : vector<1xi8> to vector<1xf8E8M0FNU>
                %187 = vector.bitcast %91 : vector<1xi8> to vector<1xf8E8M0FNU>
                %188 = vector.bitcast %92 : vector<1xi8> to vector<1xf8E8M0FNU>
                %189 = vector.bitcast %94 : vector<1xi8> to vector<1xf8E8M0FNU>
                %190 = vector.bitcast %95 : vector<1xi8> to vector<1xf8E8M0FNU>
                %191 = vector.bitcast %97 : vector<1xi8> to vector<1xf8E8M0FNU>
                %192 = vector.bitcast %98 : vector<1xi8> to vector<1xf8E8M0FNU>
                %193 = vector.bitcast %100 : vector<1xi8> to vector<1xf8E8M0FNU>
                %194 = vector.bitcast %101 : vector<1xi8> to vector<1xf8E8M0FNU>
                %195 = vector.bitcast %103 : vector<1xi8> to vector<1xf8E8M0FNU>
                %196 = vector.bitcast %104 : vector<1xi8> to vector<1xf8E8M0FNU>
                %197 = vector.bitcast %106 : vector<1xi8> to vector<1xf8E8M0FNU>
                %198 = vector.bitcast %107 : vector<1xi8> to vector<1xf8E8M0FNU>
                %199 = vector.bitcast %109 : vector<1xi8> to vector<1xf8E8M0FNU>
                %200 = vector.bitcast %110 : vector<1xi8> to vector<1xf8E8M0FNU>
                %201 = vector.extract %161[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %202 = vector.extract %185[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %203 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%202[0] * %169) + %83#0 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %204 = vector.extract %162[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %205 = vector.extract %186[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %206 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%205[0] * %170) + %203 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %207 = vector.extract %187[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %208 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%207[0] * %171) + %83#1 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %209 = vector.extract %188[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %210 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%209[0] * %172) + %208 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %211 = vector.extract %189[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %212 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%211[0] * %173) + %83#2 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %213 = vector.extract %190[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %214 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%213[0] * %174) + %212 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %215 = vector.extract %191[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %216 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%215[0] * %175) + %83#3 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %217 = vector.extract %192[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %218 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%217[0] * %176) + %216 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %219 = vector.extract %193[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %220 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%219[0] * %177) + %83#4 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %221 = vector.extract %194[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %222 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%221[0] * %178) + %220 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %223 = vector.extract %195[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %224 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%223[0] * %179) + %83#5 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %225 = vector.extract %196[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %226 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%225[0] * %180) + %224 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %227 = vector.extract %197[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %228 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%227[0] * %181) + %83#6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %229 = vector.extract %198[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %230 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%229[0] * %182) + %228 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %231 = vector.extract %199[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %232 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%231[0] * %183) + %83#7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %233 = vector.extract %200[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %234 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%233[0] * %184) + %232 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %235 = vector.extract %163[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %236 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%202[0] * %169) + %83#8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %237 = vector.extract %164[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %238 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%205[0] * %170) + %236 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %239 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%207[0] * %171) + %83#9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %240 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%209[0] * %172) + %239 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %241 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%211[0] * %173) + %83#10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %242 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%213[0] * %174) + %241 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %243 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%215[0] * %175) + %83#11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %244 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%217[0] * %176) + %243 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %245 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%219[0] * %177) + %83#12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %246 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%221[0] * %178) + %245 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %247 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%223[0] * %179) + %83#13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %248 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%225[0] * %180) + %247 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %249 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%227[0] * %181) + %83#14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %250 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%229[0] * %182) + %249 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %251 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%231[0] * %183) + %83#15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %252 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%233[0] * %184) + %251 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %253 = vector.extract %165[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %254 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%202[0] * %169) + %83#16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %255 = vector.extract %166[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %256 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%205[0] * %170) + %254 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %257 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%207[0] * %171) + %83#17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %258 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%209[0] * %172) + %257 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %259 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%211[0] * %173) + %83#18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %260 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%213[0] * %174) + %259 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %261 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%215[0] * %175) + %83#19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %262 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%217[0] * %176) + %261 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %263 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%219[0] * %177) + %83#20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %264 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%221[0] * %178) + %263 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %265 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%223[0] * %179) + %83#21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %266 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%225[0] * %180) + %265 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %267 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%227[0] * %181) + %83#22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %268 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%229[0] * %182) + %267 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %269 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%231[0] * %183) + %83#23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %270 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%233[0] * %184) + %269 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %271 = vector.extract %167[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %272 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%202[0] * %169) + %83#24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %273 = vector.extract %168[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %274 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%205[0] * %170) + %272 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %275 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%207[0] * %171) + %83#25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %276 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%209[0] * %172) + %275 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %277 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%211[0] * %173) + %83#26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %278 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%213[0] * %174) + %277 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %279 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%215[0] * %175) + %83#27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %280 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%217[0] * %176) + %279 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %281 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%219[0] * %177) + %83#28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %282 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%221[0] * %178) + %281 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %283 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%223[0] * %179) + %83#29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %284 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%225[0] * %180) + %283 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %285 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%227[0] * %181) + %83#30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %286 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%229[0] * %182) + %285 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %287 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%231[0] * %183) + %83#31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %288 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%233[0] * %184) + %287 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %289 = vector.extract_strided_slice %206 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %290 = affine.apply #map35()[%block_id_x]
                %291 = affine.apply #map35()[%block_id_y]
                %292 = affine.apply #map36()[%thread_id_x]
                %293 = arith.muli %290, %c16384 overflow<nsw> : index
                %294 = arith.muli %292, %c16384 overflow<nsw> : index
                %295 = arith.addi %293, %291 overflow<nsw> : index
                %296 = arith.addi %294, %84 overflow<nsw> : index
                %reinterpret_cast_13 = memref.reinterpret_cast %4 to offset: [%295], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
                %cast_14 = memref.cast %reinterpret_cast_13 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
                %297 = amdgpu.fat_raw_buffer_cast %cast_14 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                vector.store %289, %297[%296] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %298 = vector.extract_strided_slice %206 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %299 = affine.apply #map37()[%thread_id_x]
                %300 = arith.muli %299, %c16384 overflow<nsw> : index
                %301 = arith.addi %300, %84 overflow<nsw> : index
                vector.store %298, %297[%301] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %302 = vector.extract_strided_slice %206 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %303 = affine.apply #map38()[%thread_id_x]
                %304 = arith.muli %303, %c16384 overflow<nsw> : index
                %305 = arith.addi %304, %84 overflow<nsw> : index
                vector.store %302, %297[%305] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %306 = vector.extract_strided_slice %206 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %307 = affine.apply #map39()[%thread_id_x]
                %308 = arith.muli %307, %c16384 overflow<nsw> : index
                %309 = arith.addi %308, %84 overflow<nsw> : index
                vector.store %306, %297[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %310 = vector.extract_strided_slice %210 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %311 = arith.addi %294, %90 overflow<nsw> : index
                vector.store %310, %297[%311] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %312 = vector.extract_strided_slice %210 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %313 = arith.addi %300, %90 overflow<nsw> : index
                vector.store %312, %297[%313] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %314 = vector.extract_strided_slice %210 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %315 = arith.addi %304, %90 overflow<nsw> : index
                vector.store %314, %297[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %316 = vector.extract_strided_slice %210 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %317 = arith.addi %308, %90 overflow<nsw> : index
                vector.store %316, %297[%317] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %318 = vector.extract_strided_slice %214 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %319 = arith.addi %294, %93 overflow<nsw> : index
                vector.store %318, %297[%319] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %320 = vector.extract_strided_slice %214 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %321 = arith.addi %300, %93 overflow<nsw> : index
                vector.store %320, %297[%321] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %322 = vector.extract_strided_slice %214 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %323 = arith.addi %304, %93 overflow<nsw> : index
                vector.store %322, %297[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %324 = vector.extract_strided_slice %214 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %325 = arith.addi %308, %93 overflow<nsw> : index
                vector.store %324, %297[%325] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %326 = vector.extract_strided_slice %218 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %327 = arith.addi %294, %96 overflow<nsw> : index
                vector.store %326, %297[%327] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %328 = vector.extract_strided_slice %218 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %329 = arith.addi %300, %96 overflow<nsw> : index
                vector.store %328, %297[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %330 = vector.extract_strided_slice %218 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %331 = arith.addi %304, %96 overflow<nsw> : index
                vector.store %330, %297[%331] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %332 = vector.extract_strided_slice %218 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %333 = arith.addi %308, %96 overflow<nsw> : index
                vector.store %332, %297[%333] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %334 = vector.extract_strided_slice %222 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %335 = arith.addi %294, %99 overflow<nsw> : index
                vector.store %334, %297[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %336 = vector.extract_strided_slice %222 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %337 = arith.addi %300, %99 overflow<nsw> : index
                vector.store %336, %297[%337] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %338 = vector.extract_strided_slice %222 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %339 = arith.addi %304, %99 overflow<nsw> : index
                vector.store %338, %297[%339] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %340 = vector.extract_strided_slice %222 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %341 = arith.addi %308, %99 overflow<nsw> : index
                vector.store %340, %297[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %342 = vector.extract_strided_slice %226 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %343 = arith.addi %294, %102 overflow<nsw> : index
                vector.store %342, %297[%343] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %344 = vector.extract_strided_slice %226 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %345 = arith.addi %300, %102 overflow<nsw> : index
                vector.store %344, %297[%345] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %346 = vector.extract_strided_slice %226 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %347 = arith.addi %304, %102 overflow<nsw> : index
                vector.store %346, %297[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %348 = vector.extract_strided_slice %226 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %349 = arith.addi %308, %102 overflow<nsw> : index
                vector.store %348, %297[%349] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %350 = vector.extract_strided_slice %230 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %351 = arith.addi %294, %105 overflow<nsw> : index
                vector.store %350, %297[%351] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %352 = vector.extract_strided_slice %230 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %353 = arith.addi %300, %105 overflow<nsw> : index
                vector.store %352, %297[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %354 = vector.extract_strided_slice %230 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %355 = arith.addi %304, %105 overflow<nsw> : index
                vector.store %354, %297[%355] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %356 = vector.extract_strided_slice %230 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %357 = arith.addi %308, %105 overflow<nsw> : index
                vector.store %356, %297[%357] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %358 = vector.extract_strided_slice %234 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %359 = arith.addi %294, %108 overflow<nsw> : index
                vector.store %358, %297[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %360 = vector.extract_strided_slice %234 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %361 = arith.addi %300, %108 overflow<nsw> : index
                vector.store %360, %297[%361] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %362 = vector.extract_strided_slice %234 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %363 = arith.addi %304, %108 overflow<nsw> : index
                vector.store %362, %297[%363] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %364 = vector.extract_strided_slice %234 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %365 = arith.addi %308, %108 overflow<nsw> : index
                vector.store %364, %297[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %366 = vector.extract_strided_slice %238 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %367 = affine.apply #map40()[%thread_id_x]
                %368 = arith.muli %367, %c16384 overflow<nsw> : index
                %369 = arith.addi %368, %84 overflow<nsw> : index
                vector.store %366, %297[%369] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %370 = vector.extract_strided_slice %238 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %371 = affine.apply #map41()[%thread_id_x]
                %372 = arith.muli %371, %c16384 overflow<nsw> : index
                %373 = arith.addi %372, %84 overflow<nsw> : index
                vector.store %370, %297[%373] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %374 = vector.extract_strided_slice %238 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %375 = affine.apply #map42()[%thread_id_x]
                %376 = arith.muli %375, %c16384 overflow<nsw> : index
                %377 = arith.addi %376, %84 overflow<nsw> : index
                vector.store %374, %297[%377] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %378 = vector.extract_strided_slice %238 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %379 = affine.apply #map43()[%thread_id_x]
                %380 = arith.muli %379, %c16384 overflow<nsw> : index
                %381 = arith.addi %380, %84 overflow<nsw> : index
                vector.store %378, %297[%381] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %382 = vector.extract_strided_slice %240 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %383 = arith.addi %368, %90 overflow<nsw> : index
                vector.store %382, %297[%383] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %384 = vector.extract_strided_slice %240 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %385 = arith.addi %372, %90 overflow<nsw> : index
                vector.store %384, %297[%385] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %386 = vector.extract_strided_slice %240 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %387 = arith.addi %376, %90 overflow<nsw> : index
                vector.store %386, %297[%387] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %388 = vector.extract_strided_slice %240 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %389 = arith.addi %380, %90 overflow<nsw> : index
                vector.store %388, %297[%389] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %390 = vector.extract_strided_slice %242 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %391 = arith.addi %368, %93 overflow<nsw> : index
                vector.store %390, %297[%391] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %392 = vector.extract_strided_slice %242 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %393 = arith.addi %372, %93 overflow<nsw> : index
                vector.store %392, %297[%393] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %394 = vector.extract_strided_slice %242 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %395 = arith.addi %376, %93 overflow<nsw> : index
                vector.store %394, %297[%395] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %396 = vector.extract_strided_slice %242 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %397 = arith.addi %380, %93 overflow<nsw> : index
                vector.store %396, %297[%397] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %398 = vector.extract_strided_slice %244 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %399 = arith.addi %368, %96 overflow<nsw> : index
                vector.store %398, %297[%399] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %400 = vector.extract_strided_slice %244 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %401 = arith.addi %372, %96 overflow<nsw> : index
                vector.store %400, %297[%401] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %402 = vector.extract_strided_slice %244 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %403 = arith.addi %376, %96 overflow<nsw> : index
                vector.store %402, %297[%403] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %404 = vector.extract_strided_slice %244 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %405 = arith.addi %380, %96 overflow<nsw> : index
                vector.store %404, %297[%405] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %406 = vector.extract_strided_slice %246 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %407 = arith.addi %368, %99 overflow<nsw> : index
                vector.store %406, %297[%407] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %408 = vector.extract_strided_slice %246 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %409 = arith.addi %372, %99 overflow<nsw> : index
                vector.store %408, %297[%409] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %410 = vector.extract_strided_slice %246 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %411 = arith.addi %376, %99 overflow<nsw> : index
                vector.store %410, %297[%411] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %412 = vector.extract_strided_slice %246 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %413 = arith.addi %380, %99 overflow<nsw> : index
                vector.store %412, %297[%413] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %414 = vector.extract_strided_slice %248 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %415 = arith.addi %368, %102 overflow<nsw> : index
                vector.store %414, %297[%415] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %416 = vector.extract_strided_slice %248 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %417 = arith.addi %372, %102 overflow<nsw> : index
                vector.store %416, %297[%417] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %418 = vector.extract_strided_slice %248 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %419 = arith.addi %376, %102 overflow<nsw> : index
                vector.store %418, %297[%419] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %420 = vector.extract_strided_slice %248 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %421 = arith.addi %380, %102 overflow<nsw> : index
                vector.store %420, %297[%421] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %422 = vector.extract_strided_slice %250 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %423 = arith.addi %368, %105 overflow<nsw> : index
                vector.store %422, %297[%423] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %424 = vector.extract_strided_slice %250 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %425 = arith.addi %372, %105 overflow<nsw> : index
                vector.store %424, %297[%425] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %426 = vector.extract_strided_slice %250 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %427 = arith.addi %376, %105 overflow<nsw> : index
                vector.store %426, %297[%427] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %428 = vector.extract_strided_slice %250 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %429 = arith.addi %380, %105 overflow<nsw> : index
                vector.store %428, %297[%429] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %430 = vector.extract_strided_slice %252 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %431 = arith.addi %368, %108 overflow<nsw> : index
                vector.store %430, %297[%431] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %432 = vector.extract_strided_slice %252 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %433 = arith.addi %372, %108 overflow<nsw> : index
                vector.store %432, %297[%433] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %434 = vector.extract_strided_slice %252 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %435 = arith.addi %376, %108 overflow<nsw> : index
                vector.store %434, %297[%435] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %436 = vector.extract_strided_slice %252 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %437 = arith.addi %380, %108 overflow<nsw> : index
                vector.store %436, %297[%437] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %438 = vector.extract_strided_slice %256 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %439 = affine.apply #map44()[%thread_id_x]
                %440 = arith.muli %439, %c16384 overflow<nsw> : index
                %441 = arith.addi %440, %84 overflow<nsw> : index
                vector.store %438, %297[%441] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %442 = vector.extract_strided_slice %256 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %443 = affine.apply #map45()[%thread_id_x]
                %444 = arith.muli %443, %c16384 overflow<nsw> : index
                %445 = arith.addi %444, %84 overflow<nsw> : index
                vector.store %442, %297[%445] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %446 = vector.extract_strided_slice %256 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %447 = affine.apply #map46()[%thread_id_x]
                %448 = arith.muli %447, %c16384 overflow<nsw> : index
                %449 = arith.addi %448, %84 overflow<nsw> : index
                vector.store %446, %297[%449] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %450 = vector.extract_strided_slice %256 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %451 = affine.apply #map47()[%thread_id_x]
                %452 = arith.muli %451, %c16384 overflow<nsw> : index
                %453 = arith.addi %452, %84 overflow<nsw> : index
                vector.store %450, %297[%453] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %454 = vector.extract_strided_slice %258 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %455 = arith.addi %440, %90 overflow<nsw> : index
                vector.store %454, %297[%455] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %456 = vector.extract_strided_slice %258 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %457 = arith.addi %444, %90 overflow<nsw> : index
                vector.store %456, %297[%457] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %458 = vector.extract_strided_slice %258 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %459 = arith.addi %448, %90 overflow<nsw> : index
                vector.store %458, %297[%459] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %460 = vector.extract_strided_slice %258 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %461 = arith.addi %452, %90 overflow<nsw> : index
                vector.store %460, %297[%461] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %462 = vector.extract_strided_slice %260 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %463 = arith.addi %440, %93 overflow<nsw> : index
                vector.store %462, %297[%463] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %464 = vector.extract_strided_slice %260 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %465 = arith.addi %444, %93 overflow<nsw> : index
                vector.store %464, %297[%465] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %466 = vector.extract_strided_slice %260 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %467 = arith.addi %448, %93 overflow<nsw> : index
                vector.store %466, %297[%467] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %468 = vector.extract_strided_slice %260 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %469 = arith.addi %452, %93 overflow<nsw> : index
                vector.store %468, %297[%469] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %470 = vector.extract_strided_slice %262 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %471 = arith.addi %440, %96 overflow<nsw> : index
                vector.store %470, %297[%471] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %472 = vector.extract_strided_slice %262 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %473 = arith.addi %444, %96 overflow<nsw> : index
                vector.store %472, %297[%473] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %474 = vector.extract_strided_slice %262 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %475 = arith.addi %448, %96 overflow<nsw> : index
                vector.store %474, %297[%475] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %476 = vector.extract_strided_slice %262 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %477 = arith.addi %452, %96 overflow<nsw> : index
                vector.store %476, %297[%477] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %478 = vector.extract_strided_slice %264 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %479 = arith.addi %440, %99 overflow<nsw> : index
                vector.store %478, %297[%479] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %480 = vector.extract_strided_slice %264 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %481 = arith.addi %444, %99 overflow<nsw> : index
                vector.store %480, %297[%481] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %482 = vector.extract_strided_slice %264 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %483 = arith.addi %448, %99 overflow<nsw> : index
                vector.store %482, %297[%483] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %484 = vector.extract_strided_slice %264 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %485 = arith.addi %452, %99 overflow<nsw> : index
                vector.store %484, %297[%485] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %486 = vector.extract_strided_slice %266 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %487 = arith.addi %440, %102 overflow<nsw> : index
                vector.store %486, %297[%487] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %488 = vector.extract_strided_slice %266 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %489 = arith.addi %444, %102 overflow<nsw> : index
                vector.store %488, %297[%489] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %490 = vector.extract_strided_slice %266 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %491 = arith.addi %448, %102 overflow<nsw> : index
                vector.store %490, %297[%491] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %492 = vector.extract_strided_slice %266 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %493 = arith.addi %452, %102 overflow<nsw> : index
                vector.store %492, %297[%493] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %494 = vector.extract_strided_slice %268 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %495 = arith.addi %440, %105 overflow<nsw> : index
                vector.store %494, %297[%495] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %496 = vector.extract_strided_slice %268 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %497 = arith.addi %444, %105 overflow<nsw> : index
                vector.store %496, %297[%497] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %498 = vector.extract_strided_slice %268 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %499 = arith.addi %448, %105 overflow<nsw> : index
                vector.store %498, %297[%499] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %500 = vector.extract_strided_slice %268 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %501 = arith.addi %452, %105 overflow<nsw> : index
                vector.store %500, %297[%501] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %502 = vector.extract_strided_slice %270 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %503 = arith.addi %440, %108 overflow<nsw> : index
                vector.store %502, %297[%503] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %504 = vector.extract_strided_slice %270 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %505 = arith.addi %444, %108 overflow<nsw> : index
                vector.store %504, %297[%505] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %506 = vector.extract_strided_slice %270 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %507 = arith.addi %448, %108 overflow<nsw> : index
                vector.store %506, %297[%507] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %508 = vector.extract_strided_slice %270 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %509 = arith.addi %452, %108 overflow<nsw> : index
                vector.store %508, %297[%509] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %510 = vector.extract_strided_slice %274 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %511 = affine.apply #map48()[%thread_id_x]
                %512 = arith.muli %511, %c16384 overflow<nsw> : index
                %513 = arith.addi %512, %84 overflow<nsw> : index
                vector.store %510, %297[%513] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %514 = vector.extract_strided_slice %274 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %515 = affine.apply #map49()[%thread_id_x]
                %516 = arith.muli %515, %c16384 overflow<nsw> : index
                %517 = arith.addi %516, %84 overflow<nsw> : index
                vector.store %514, %297[%517] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %518 = vector.extract_strided_slice %274 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %519 = affine.apply #map50()[%thread_id_x]
                %520 = arith.muli %519, %c16384 overflow<nsw> : index
                %521 = arith.addi %520, %84 overflow<nsw> : index
                vector.store %518, %297[%521] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %522 = vector.extract_strided_slice %274 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %523 = affine.apply #map51()[%thread_id_x]
                %524 = arith.muli %523, %c16384 overflow<nsw> : index
                %525 = arith.addi %524, %84 overflow<nsw> : index
                vector.store %522, %297[%525] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %526 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %527 = arith.addi %512, %90 overflow<nsw> : index
                vector.store %526, %297[%527] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %528 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %529 = arith.addi %516, %90 overflow<nsw> : index
                vector.store %528, %297[%529] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %530 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %531 = arith.addi %520, %90 overflow<nsw> : index
                vector.store %530, %297[%531] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %532 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %533 = arith.addi %524, %90 overflow<nsw> : index
                vector.store %532, %297[%533] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %534 = vector.extract_strided_slice %278 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %535 = arith.addi %512, %93 overflow<nsw> : index
                vector.store %534, %297[%535] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %536 = vector.extract_strided_slice %278 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %537 = arith.addi %516, %93 overflow<nsw> : index
                vector.store %536, %297[%537] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %538 = vector.extract_strided_slice %278 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %539 = arith.addi %520, %93 overflow<nsw> : index
                vector.store %538, %297[%539] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %540 = vector.extract_strided_slice %278 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %541 = arith.addi %524, %93 overflow<nsw> : index
                vector.store %540, %297[%541] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %542 = vector.extract_strided_slice %280 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %543 = arith.addi %512, %96 overflow<nsw> : index
                vector.store %542, %297[%543] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %544 = vector.extract_strided_slice %280 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %545 = arith.addi %516, %96 overflow<nsw> : index
                vector.store %544, %297[%545] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %546 = vector.extract_strided_slice %280 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %547 = arith.addi %520, %96 overflow<nsw> : index
                vector.store %546, %297[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %548 = vector.extract_strided_slice %280 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %549 = arith.addi %524, %96 overflow<nsw> : index
                vector.store %548, %297[%549] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %550 = vector.extract_strided_slice %282 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %551 = arith.addi %512, %99 overflow<nsw> : index
                vector.store %550, %297[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %552 = vector.extract_strided_slice %282 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %553 = arith.addi %516, %99 overflow<nsw> : index
                vector.store %552, %297[%553] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %554 = vector.extract_strided_slice %282 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %555 = arith.addi %520, %99 overflow<nsw> : index
                vector.store %554, %297[%555] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %556 = vector.extract_strided_slice %282 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %557 = arith.addi %524, %99 overflow<nsw> : index
                vector.store %556, %297[%557] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %558 = vector.extract_strided_slice %284 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %559 = arith.addi %512, %102 overflow<nsw> : index
                vector.store %558, %297[%559] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %560 = vector.extract_strided_slice %284 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %561 = arith.addi %516, %102 overflow<nsw> : index
                vector.store %560, %297[%561] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %562 = vector.extract_strided_slice %284 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %563 = arith.addi %520, %102 overflow<nsw> : index
                vector.store %562, %297[%563] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %564 = vector.extract_strided_slice %284 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %565 = arith.addi %524, %102 overflow<nsw> : index
                vector.store %564, %297[%565] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %566 = vector.extract_strided_slice %286 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %567 = arith.addi %512, %105 overflow<nsw> : index
                vector.store %566, %297[%567] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %568 = vector.extract_strided_slice %286 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %569 = arith.addi %516, %105 overflow<nsw> : index
                vector.store %568, %297[%569] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %570 = vector.extract_strided_slice %286 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %571 = arith.addi %520, %105 overflow<nsw> : index
                vector.store %570, %297[%571] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %572 = vector.extract_strided_slice %286 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %573 = arith.addi %524, %105 overflow<nsw> : index
                vector.store %572, %297[%573] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %574 = vector.extract_strided_slice %288 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %575 = arith.addi %512, %108 overflow<nsw> : index
                vector.store %574, %297[%575] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %576 = vector.extract_strided_slice %288 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %577 = arith.addi %516, %108 overflow<nsw> : index
                vector.store %576, %297[%577] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %578 = vector.extract_strided_slice %288 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %579 = arith.addi %520, %108 overflow<nsw> : index
                vector.store %578, %297[%579] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %580 = vector.extract_strided_slice %288 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %581 = arith.addi %524, %108 overflow<nsw> : index
                vector.store %580, %297[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<16384x8192xi8>
            %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<16384x512xi8>
            %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<16384x8192xi8>
            %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<16384x512xi8>
            %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<16384x16384xf32>
            %5 = flow.dispatch @gemm::@gemm(%0, %1, %2, %3, %4) : (tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x8192xi8>, tensor<16384x512xi8>, tensor<16384x16384xf32>) -> %4
            %6 = hal.tensor.barrier join(%5 : tensor<16384x16384xf32>) => %arg6 : !hal.fence
            %7 = hal.tensor.export %6 : tensor<16384x16384xf32> -> !hal.buffer_view
            return %7 : !hal.buffer_view
        }
        }
    """
    ##cluster 0 loads most of it
    mlir_different_mapping = """

        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 16)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
        #map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
        #map9 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
        #map11 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
        #map12 = affine_map<()[s0] -> ((s0 floordiv 2) mod 2)>
        #map13 = affine_map<()[s0] -> (s0 mod 2)>
        #map14 = affine_map<()[s0] -> (s0 * 4)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 32 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 256)>
        #map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
        #map18 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
        #map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
        #map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
        #map22 = affine_map<()[s0] -> (s0 * 4 + (s0 mod 64) floordiv 16 - (s0 floordiv 2) * 8)>
        #map23 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
        #map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
        #map25 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
        #map26 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
        #map27 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
        #map28 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
        #map29 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
        #map30 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
        #map31 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map32 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
        #map33 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 128)>
        #map34 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 + 8)>
        #map35 = affine_map<()[s0] -> (s0 * 256)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
        #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
        #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
        #map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
        #map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
        #map47 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
        #map48 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
        #map49 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
        #map50 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
        #map51 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c16 = arith.constant 16 : index
            %c224 = arith.constant 224 : index
            %c1 = arith.constant 1 : index
            stream.return %c16, %c224, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c512_i14 = arith.constant 512 : i14
                %c-8192_i14 = arith.constant -8192 : i14
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c57344 = arith.constant 57344 : index
                %c63 = arith.constant 63 : index
                %c512 = arith.constant 512 : index
                %c2147483646_i64 = arith.constant 2147483646 : i64
                %c8192 = arith.constant 8192 : index
                %c1 = arith.constant 1 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
                %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
                %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 16
                %block_id_y = gpu.block_id  y upper_bound 224
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %alloc = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_0 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_3 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_4 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_5 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_6 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %c32_idx = arith.constant 32 : index
                %c128_idx = arith.constant 128 : index
                %c262144 = arith.constant 262144 : index
                %c65536 = arith.constant 65536 : index
                %is_cluster0 = arith.cmpi eq, %thread_id_y, %c0 : index
                %5 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %6 = affine.apply #map1()[%thread_id_x]
                %7 = affine.apply #map2()[%thread_id_x]
                %8 = arith.xori %7, %6 : index
                %9 = affine.apply #map3()[%8]
                %10 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %11 = gpu.subgroup_broadcast %10,  first_active_lane : index
                %12 = gpu.subgroup_broadcast %c0,  first_active_lane : index
                %13 = arith.muli %5, %c8192 overflow<nsw> : index
                %14 = arith.addi %13, %9 overflow<nsw> : index
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast = memref.cast %reinterpret_cast : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %15 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                // --- Address computations (all waves) ---
                %16 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
                %17 = affine.apply #map6()[%thread_id_x, %thread_id_y]
                %18 = gpu.subgroup_broadcast %17,  first_active_lane : index
                %19 = arith.muli %16, %c8192 overflow<nsw> : index
                %20 = arith.addi %19, %9 overflow<nsw> : index
                %21 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
                %22 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                %23 = gpu.subgroup_broadcast %22,  first_active_lane : index
                %24 = arith.muli %21, %c8192 overflow<nsw> : index
                %25 = arith.addi %24, %9 overflow<nsw> : index
                %26 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
                %27 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %28 = gpu.subgroup_broadcast %27,  first_active_lane : index
                %29 = arith.muli %26, %c8192 overflow<nsw> : index
                %30 = arith.addi %29, %9 overflow<nsw> : index
                %31 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_x]
                %32 = affine.apply #map12()[%thread_id_x]
                %33 = affine.apply #map13()[%thread_id_x]
                %34 = arith.xori %33, %32 : index
                %35 = affine.apply #map14()[%34]
                %36 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %37 = gpu.subgroup_broadcast %36,  first_active_lane : index
                %38 = arith.muli %31, %c512 overflow<nsw> : index
                %39 = arith.addi %38, %35 overflow<nsw> : index
                %reinterpret_cast_7 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %40 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                %41 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
                %42 = arith.muli %41, %c8192 overflow<nsw> : index
                %43 = arith.addi %42, %9 overflow<nsw> : index
                %reinterpret_cast_9 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_10 = memref.cast %reinterpret_cast_9 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %44 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                %45 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                %46 = arith.muli %45, %c8192 overflow<nsw> : index
                %47 = arith.addi %46, %9 overflow<nsw> : index
                %48 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
                %49 = arith.muli %48, %c8192 overflow<nsw> : index
                %50 = arith.addi %49, %9 overflow<nsw> : index
                %51 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_y]
                %52 = arith.muli %51, %c8192 overflow<nsw> : index
                %53 = arith.addi %52, %9 overflow<nsw> : index
                %54 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_y]
                %55 = arith.muli %54, %c512 overflow<nsw> : index
                %56 = arith.addi %55, %35 overflow<nsw> : index
                %reinterpret_cast_11 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_12 = memref.cast %reinterpret_cast_11 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %57 = amdgpu.fat_raw_buffer_cast %cast_12 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                // --- Cluster 0 only: A data (8), A scale (2), B data (8) gathers ---
                scf.if %is_cluster0 {
                // A data: 4 original gathers (ty=0 addresses)
                amdgpu.gather_to_lds %15[%14], %alloc_6[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %15[%20], %alloc_6[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %15[%25], %alloc_6[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %15[%30], %alloc_6[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                // A data: 4 extra gathers (ty=1 addresses: global +262144, LDS row +32)
                %ea_g0 = arith.addi %14, %c262144 overflow<nsw> : index
                %ea_l0 = arith.addi %11, %c32_idx overflow<nsw> : index
                amdgpu.gather_to_lds %15[%ea_g0], %alloc_6[%ea_l0, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %ea_g1 = arith.addi %20, %c262144 overflow<nsw> : index
                %ea_l1 = arith.addi %18, %c32_idx overflow<nsw> : index
                amdgpu.gather_to_lds %15[%ea_g1], %alloc_6[%ea_l1, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %ea_g2 = arith.addi %25, %c262144 overflow<nsw> : index
                %ea_l2 = arith.addi %23, %c32_idx overflow<nsw> : index
                amdgpu.gather_to_lds %15[%ea_g2], %alloc_6[%ea_l2, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %ea_g3 = arith.addi %30, %c262144 overflow<nsw> : index
                %ea_l3 = arith.addi %28, %c32_idx overflow<nsw> : index
                amdgpu.gather_to_lds %15[%ea_g3], %alloc_6[%ea_l3, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                // A scale: 1 original gather (ty=0)
                amdgpu.gather_to_lds %40[%39], %alloc_4[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                // A scale: 1 extra gather (ty=1: global +65536, LDS row +128)
                %eas_g0 = arith.addi %39, %c65536 overflow<nsw> : index
                %eas_l0 = arith.addi %37, %c128_idx overflow<nsw> : index
                amdgpu.gather_to_lds %40[%eas_g0], %alloc_4[%eas_l0, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                // B data: 4 original gathers (ty=0 addresses)
                amdgpu.gather_to_lds %44[%43], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %44[%47], %alloc_2[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %44[%50], %alloc_2[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %44[%53], %alloc_2[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                // B data: 4 extra gathers (ty=1: global +262144, LDS row +32)
                %eb_g0 = arith.addi %43, %c262144 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%eb_g0], %alloc_2[%ea_l0, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %eb_g1 = arith.addi %47, %c262144 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%eb_g1], %alloc_2[%ea_l1, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %eb_g2 = arith.addi %50, %c262144 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%eb_g2], %alloc_2[%ea_l2, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %eb_g3 = arith.addi %53, %c262144 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%eb_g3], %alloc_2[%ea_l3, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                }
                // B scale: unchanged (both clusters, already cluster-aligned)
                amdgpu.gather_to_lds %57[%56], %alloc_0[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.s.barrier
                %58 = affine.apply #map16()[%thread_id_x, %thread_id_y]
                %59 = arith.index_cast %58 : index to i32
                %60 = arith.cmpi sge, %59, %c4_i32 : i32
                %61 = arith.cmpi slt, %59, %c4_i32 : i32
                scf.if %60 {
                rocdl.s.barrier
                }
                %62 = affine.apply #map17()[%thread_id_x]
                %63 = affine.apply #map18()[%thread_id_x]
                %64 = arith.xori %63, %7 : index
                %65 = affine.apply #map3()[%64]
                %66 = affine.apply #map19()[%thread_id_x]
                %67 = affine.apply #map20()[%thread_id_x]
                %68 = affine.apply #map21()[%thread_id_x]
                %69 = affine.apply #map22()[%thread_id_x]
                %70 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %71 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %72 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %73 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %74 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %75 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %76 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %77 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %78 = affine.apply #map31()[%thread_id_x]
                %79 = arith.xori %78, %7 : index
                %80 = affine.apply #map3()[%79]
                %81 = arith.xori %33, %c1 : index
                %82 = affine.apply #map32()[%thread_id_x, %81]
                %83:40 = scf.for %arg5 = %c0 to %c63 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %alloc_6, %arg39 = %alloc_5, %arg40 = %alloc_4, %arg41 = %alloc_3, %arg42 = %alloc_2, %arg43 = %alloc_1, %arg44 = %alloc_0, %arg45 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>) {
                rocdl.sched.barrier 0
                rocdl.s.barrier
                // --- Address computations (all waves) ---
                %582 = affine.apply #map33()[%arg5, %8]
                %583 = arith.addi %13, %582 overflow<nsw> : index
                %584 = arith.addi %19, %582 overflow<nsw> : index
                %585 = arith.addi %24, %582 overflow<nsw> : index
                %586 = arith.addi %29, %582 overflow<nsw> : index
                %587 = affine.apply #map34()[%arg5, %34]
                %588 = arith.addi %38, %587 overflow<nsw> : index
                %589 = arith.addi %42, %582 overflow<nsw> : index
                %590 = arith.addi %46, %582 overflow<nsw> : index
                %591 = arith.addi %49, %582 overflow<nsw> : index
                %592 = arith.addi %52, %582 overflow<nsw> : index
                %593 = arith.addi %55, %587 overflow<nsw> : index
                // --- Cluster 0 only: A data (8), A scale (2), B data (8) gathers ---
                scf.if %is_cluster0 {
                    // A data: 4 original gathers (ty=0 addresses)
                    amdgpu.gather_to_lds %15[%583], %arg39[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %15[%584], %arg39[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %15[%585], %arg39[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %15[%586], %arg39[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    // A data: 4 extra gathers (ty=1: global +262144, LDS row +32)
                    %lea_g0 = arith.addi %583, %c262144 overflow<nsw> : index
                    %lea_l0 = arith.addi %11, %c32_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %15[%lea_g0], %arg39[%lea_l0, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %lea_g1 = arith.addi %584, %c262144 overflow<nsw> : index
                    %lea_l1 = arith.addi %18, %c32_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %15[%lea_g1], %arg39[%lea_l1, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %lea_g2 = arith.addi %585, %c262144 overflow<nsw> : index
                    %lea_l2 = arith.addi %23, %c32_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %15[%lea_g2], %arg39[%lea_l2, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %lea_g3 = arith.addi %586, %c262144 overflow<nsw> : index
                    %lea_l3 = arith.addi %28, %c32_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %15[%lea_g3], %arg39[%lea_l3, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    // A scale: 1 original gather (ty=0)
                    amdgpu.gather_to_lds %40[%588], %arg41[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                    // A scale: 1 extra gather (ty=1: global +65536, LDS row +128)
                    %leas_g0 = arith.addi %588, %c65536 overflow<nsw> : index
                    %leas_l0 = arith.addi %37, %c128_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %40[%leas_g0], %arg41[%leas_l0, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                    // B data: 4 original gathers (ty=0 addresses)
                    amdgpu.gather_to_lds %44[%589], %arg43[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %44[%590], %arg43[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %44[%591], %arg43[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %44[%592], %arg43[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    // B data: 4 extra gathers (ty=1: global +262144, LDS row +32)
                    %leb_g0 = arith.addi %589, %c262144 overflow<nsw> : index
                    amdgpu.gather_to_lds %44[%leb_g0], %arg43[%lea_l0, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %leb_g1 = arith.addi %590, %c262144 overflow<nsw> : index
                    amdgpu.gather_to_lds %44[%leb_g1], %arg43[%lea_l1, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %leb_g2 = arith.addi %591, %c262144 overflow<nsw> : index
                    amdgpu.gather_to_lds %44[%leb_g2], %arg43[%lea_l2, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %leb_g3 = arith.addi %592, %c262144 overflow<nsw> : index
                    amdgpu.gather_to_lds %44[%leb_g3], %arg43[%lea_l3, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                }
                // B scale: unchanged (both clusters)
                amdgpu.gather_to_lds %57[%593], %arg45[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(10)
                %594 = vector.load %arg38[%62, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %595 = vector.load %arg38[%66, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %596 = vector.load %arg38[%67, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %597 = vector.load %arg38[%68, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %598 = vector.load %arg40[%62, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %599 = vector.load %arg40[%66, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %600 = vector.load %arg40[%67, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %601 = vector.load %arg40[%68, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %602 = vector.load %arg42[%70, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %603 = vector.load %arg42[%71, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %604 = vector.load %arg42[%72, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %605 = vector.load %arg42[%73, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %606 = vector.load %arg42[%74, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %607 = vector.load %arg42[%75, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %608 = vector.load %arg42[%76, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %609 = vector.load %arg42[%77, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %610 = vector.load %arg44[%70, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %611 = vector.load %arg44[%71, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %612 = vector.load %arg44[%72, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %613 = vector.load %arg44[%73, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %614 = vector.load %arg44[%74, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %615 = vector.load %arg44[%75, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %616 = vector.load %arg44[%76, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %617 = vector.load %arg44[%77, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %618 = vector.bitcast %594 : vector<16xi8> to vector<32xf4E2M1FN>
                %619 = vector.bitcast %595 : vector<16xi8> to vector<32xf4E2M1FN>
                %620 = vector.bitcast %596 : vector<16xi8> to vector<32xf4E2M1FN>
                %621 = vector.bitcast %597 : vector<16xi8> to vector<32xf4E2M1FN>
                %622 = vector.bitcast %598 : vector<1xi8> to vector<1xf8E8M0FNU>
                %623 = vector.bitcast %599 : vector<1xi8> to vector<1xf8E8M0FNU>
                %624 = vector.bitcast %600 : vector<1xi8> to vector<1xf8E8M0FNU>
                %625 = vector.bitcast %601 : vector<1xi8> to vector<1xf8E8M0FNU>
                %626 = vector.bitcast %602 : vector<16xi8> to vector<32xf4E2M1FN>
                %627 = vector.bitcast %603 : vector<16xi8> to vector<32xf4E2M1FN>
                %628 = vector.bitcast %604 : vector<16xi8> to vector<32xf4E2M1FN>
                %629 = vector.bitcast %605 : vector<16xi8> to vector<32xf4E2M1FN>
                %630 = vector.bitcast %606 : vector<16xi8> to vector<32xf4E2M1FN>
                %631 = vector.bitcast %607 : vector<16xi8> to vector<32xf4E2M1FN>
                %632 = vector.bitcast %608 : vector<16xi8> to vector<32xf4E2M1FN>
                %633 = vector.bitcast %609 : vector<16xi8> to vector<32xf4E2M1FN>
                %634 = vector.bitcast %610 : vector<1xi8> to vector<1xf8E8M0FNU>
                %635 = vector.bitcast %611 : vector<1xi8> to vector<1xf8E8M0FNU>
                %636 = vector.bitcast %612 : vector<1xi8> to vector<1xf8E8M0FNU>
                %637 = vector.bitcast %613 : vector<1xi8> to vector<1xf8E8M0FNU>
                %638 = vector.bitcast %614 : vector<1xi8> to vector<1xf8E8M0FNU>
                %639 = vector.bitcast %615 : vector<1xi8> to vector<1xf8E8M0FNU>
                %640 = vector.bitcast %616 : vector<1xi8> to vector<1xf8E8M0FNU>
                %641 = vector.bitcast %617 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %642 = vector.extract %622[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %643 = vector.extract %634[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %644 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%643[0] * %626) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %645 = vector.extract %635[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %646 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%645[0] * %627) + %arg7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %647 = vector.extract %636[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %648 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%647[0] * %628) + %arg8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %649 = vector.extract %637[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %650 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%649[0] * %629) + %arg9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %651 = vector.extract %638[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %652 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%651[0] * %630) + %arg10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %653 = vector.extract %639[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %654 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%653[0] * %631) + %arg11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %655 = vector.extract %640[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %656 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%655[0] * %632) + %arg12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %657 = vector.extract %641[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %658 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%657[0] * %633) + %arg13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %659 = vector.extract %623[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %660 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%643[0] * %626) + %arg14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %661 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%645[0] * %627) + %arg15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %662 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%647[0] * %628) + %arg16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %663 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%649[0] * %629) + %arg17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %664 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%651[0] * %630) + %arg18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %665 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%653[0] * %631) + %arg19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %666 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%655[0] * %632) + %arg20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %667 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%657[0] * %633) + %arg21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %668 = vector.extract %624[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %669 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%643[0] * %626) + %arg22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %670 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%645[0] * %627) + %arg23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %671 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%647[0] * %628) + %arg24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %672 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%649[0] * %629) + %arg25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %673 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%651[0] * %630) + %arg26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %674 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%653[0] * %631) + %arg27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %675 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%655[0] * %632) + %arg28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %676 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%657[0] * %633) + %arg29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %677 = vector.extract %625[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %678 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%643[0] * %626) + %arg30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %679 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%645[0] * %627) + %arg31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %680 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%647[0] * %628) + %arg32 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %681 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%649[0] * %629) + %arg33 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %682 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%651[0] * %630) + %arg34 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %683 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%653[0] * %631) + %arg35 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %684 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%655[0] * %632) + %arg36 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %685 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%657[0] * %633) + %arg37 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.sched.barrier 0
                %686 = vector.load %arg38[%62, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %687 = vector.load %arg38[%66, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %688 = vector.load %arg38[%67, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %689 = vector.load %arg38[%68, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %690 = vector.load %arg40[%62, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %691 = vector.load %arg40[%66, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %692 = vector.load %arg40[%67, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %693 = vector.load %arg40[%68, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %694 = vector.load %arg42[%70, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %695 = vector.load %arg42[%71, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %696 = vector.load %arg42[%72, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %697 = vector.load %arg42[%73, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %698 = vector.load %arg42[%74, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %699 = vector.load %arg42[%75, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %700 = vector.load %arg42[%76, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %701 = vector.load %arg42[%77, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %702 = vector.load %arg44[%70, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %703 = vector.load %arg44[%71, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %704 = vector.load %arg44[%72, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %705 = vector.load %arg44[%73, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %706 = vector.load %arg44[%74, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %707 = vector.load %arg44[%75, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %708 = vector.load %arg44[%76, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %709 = vector.load %arg44[%77, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %710 = vector.bitcast %686 : vector<16xi8> to vector<32xf4E2M1FN>
                %711 = vector.bitcast %687 : vector<16xi8> to vector<32xf4E2M1FN>
                %712 = vector.bitcast %688 : vector<16xi8> to vector<32xf4E2M1FN>
                %713 = vector.bitcast %689 : vector<16xi8> to vector<32xf4E2M1FN>
                %714 = vector.bitcast %690 : vector<1xi8> to vector<1xf8E8M0FNU>
                %715 = vector.bitcast %691 : vector<1xi8> to vector<1xf8E8M0FNU>
                %716 = vector.bitcast %692 : vector<1xi8> to vector<1xf8E8M0FNU>
                %717 = vector.bitcast %693 : vector<1xi8> to vector<1xf8E8M0FNU>
                %718 = vector.bitcast %694 : vector<16xi8> to vector<32xf4E2M1FN>
                %719 = vector.bitcast %695 : vector<16xi8> to vector<32xf4E2M1FN>
                %720 = vector.bitcast %696 : vector<16xi8> to vector<32xf4E2M1FN>
                %721 = vector.bitcast %697 : vector<16xi8> to vector<32xf4E2M1FN>
                %722 = vector.bitcast %698 : vector<16xi8> to vector<32xf4E2M1FN>
                %723 = vector.bitcast %699 : vector<16xi8> to vector<32xf4E2M1FN>
                %724 = vector.bitcast %700 : vector<16xi8> to vector<32xf4E2M1FN>
                %725 = vector.bitcast %701 : vector<16xi8> to vector<32xf4E2M1FN>
                %726 = vector.bitcast %702 : vector<1xi8> to vector<1xf8E8M0FNU>
                %727 = vector.bitcast %703 : vector<1xi8> to vector<1xf8E8M0FNU>
                %728 = vector.bitcast %704 : vector<1xi8> to vector<1xf8E8M0FNU>
                %729 = vector.bitcast %705 : vector<1xi8> to vector<1xf8E8M0FNU>
                %730 = vector.bitcast %706 : vector<1xi8> to vector<1xf8E8M0FNU>
                %731 = vector.bitcast %707 : vector<1xi8> to vector<1xf8E8M0FNU>
                %732 = vector.bitcast %708 : vector<1xi8> to vector<1xf8E8M0FNU>
                %733 = vector.bitcast %709 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %734 = vector.extract %714[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %735 = vector.extract %726[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %736 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%735[0] * %718) + %644 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %737 = vector.extract %727[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %738 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%737[0] * %719) + %646 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %739 = vector.extract %728[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %740 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%739[0] * %720) + %648 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %741 = vector.extract %729[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %742 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%741[0] * %721) + %650 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %743 = vector.extract %730[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %744 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%743[0] * %722) + %652 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %745 = vector.extract %731[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %746 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%745[0] * %723) + %654 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %747 = vector.extract %732[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %748 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%747[0] * %724) + %656 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %749 = vector.extract %733[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %750 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%749[0] * %725) + %658 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %751 = vector.extract %715[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %752 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%735[0] * %718) + %660 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %753 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%737[0] * %719) + %661 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %754 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%739[0] * %720) + %662 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %755 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%741[0] * %721) + %663 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %756 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%743[0] * %722) + %664 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %757 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%745[0] * %723) + %665 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %758 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%747[0] * %724) + %666 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %759 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%749[0] * %725) + %667 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %760 = vector.extract %716[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %761 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%735[0] * %718) + %669 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %762 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%737[0] * %719) + %670 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %763 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%739[0] * %720) + %671 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %764 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%741[0] * %721) + %672 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %765 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%743[0] * %722) + %673 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %766 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%745[0] * %723) + %674 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %767 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%747[0] * %724) + %675 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %768 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%749[0] * %725) + %676 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %769 = vector.extract %717[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %770 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%735[0] * %718) + %678 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %771 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%737[0] * %719) + %679 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %772 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%739[0] * %720) + %680 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %773 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%741[0] * %721) + %681 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %774 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%743[0] * %722) + %682 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %775 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%745[0] * %723) + %683 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %776 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%747[0] * %724) + %684 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %777 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%749[0] * %725) + %685 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                scf.yield %736, %738, %740, %742, %744, %746, %748, %750, %752, %753, %754, %755, %756, %757, %758, %759, %761, %762, %763, %764, %765, %766, %767, %768, %770, %771, %772, %773, %774, %775, %776, %777, %arg39, %arg38, %arg41, %arg40, %arg43, %arg42, %arg45, %arg44 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                }
                scf.if %61 {
                rocdl.s.barrier
                }
                amdgpu.lds_barrier
                %84 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %85 = affine.apply #map22()[%thread_id_x]
                %86 = vector.load %83#38[%84, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %87 = arith.xori %33, %c1 : index
                %88 = affine.apply #map32()[%thread_id_x, %87]
                %89 = vector.load %83#38[%84, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %90 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %91 = vector.load %83#38[%90, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %92 = vector.load %83#38[%90, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %93 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %94 = vector.load %83#38[%93, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %95 = vector.load %83#38[%93, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %96 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %97 = vector.load %83#38[%96, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %98 = vector.load %83#38[%96, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %99 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %100 = vector.load %83#38[%99, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %101 = vector.load %83#38[%99, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %102 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %103 = vector.load %83#38[%102, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %104 = vector.load %83#38[%102, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %105 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %106 = vector.load %83#38[%105, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %107 = vector.load %83#38[%105, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %108 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %109 = vector.load %83#38[%108, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %110 = vector.load %83#38[%108, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %111 = affine.apply #map18()[%thread_id_x]
                %112 = arith.xori %111, %7 : index
                %113 = affine.apply #map3()[%112]
                %114 = vector.load %83#36[%84, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %115 = affine.apply #map31()[%thread_id_x]
                %116 = arith.xori %115, %7 : index
                %117 = affine.apply #map3()[%116]
                %118 = vector.load %83#36[%84, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %119 = vector.load %83#36[%90, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %120 = vector.load %83#36[%90, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %121 = vector.load %83#36[%93, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %122 = vector.load %83#36[%93, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %123 = vector.load %83#36[%96, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %124 = vector.load %83#36[%96, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %125 = vector.load %83#36[%99, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %126 = vector.load %83#36[%99, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %127 = vector.load %83#36[%102, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %128 = vector.load %83#36[%102, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %129 = vector.load %83#36[%105, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %130 = vector.load %83#36[%105, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %131 = vector.load %83#36[%108, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %132 = vector.load %83#36[%108, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %133 = affine.apply #map17()[%thread_id_x]
                %134 = vector.load %83#34[%133, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %135 = vector.load %83#34[%133, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %136 = affine.apply #map19()[%thread_id_x]
                %137 = vector.load %83#34[%136, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %138 = vector.load %83#34[%136, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %139 = affine.apply #map20()[%thread_id_x]
                %140 = vector.load %83#34[%139, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %141 = vector.load %83#34[%139, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %142 = affine.apply #map21()[%thread_id_x]
                %143 = vector.load %83#34[%142, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %144 = vector.load %83#34[%142, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %145 = vector.load %83#32[%133, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %146 = vector.load %83#32[%133, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %147 = vector.load %83#32[%136, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %148 = vector.load %83#32[%136, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %149 = vector.load %83#32[%139, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %150 = vector.load %83#32[%139, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %151 = vector.load %83#32[%142, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %152 = vector.load %83#32[%142, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %153 = vector.bitcast %145 : vector<16xi8> to vector<32xf4E2M1FN>
                %154 = vector.bitcast %146 : vector<16xi8> to vector<32xf4E2M1FN>
                %155 = vector.bitcast %147 : vector<16xi8> to vector<32xf4E2M1FN>
                %156 = vector.bitcast %148 : vector<16xi8> to vector<32xf4E2M1FN>
                %157 = vector.bitcast %149 : vector<16xi8> to vector<32xf4E2M1FN>
                %158 = vector.bitcast %150 : vector<16xi8> to vector<32xf4E2M1FN>
                %159 = vector.bitcast %151 : vector<16xi8> to vector<32xf4E2M1FN>
                %160 = vector.bitcast %152 : vector<16xi8> to vector<32xf4E2M1FN>
                %161 = vector.bitcast %134 : vector<1xi8> to vector<1xf8E8M0FNU>
                %162 = vector.bitcast %135 : vector<1xi8> to vector<1xf8E8M0FNU>
                %163 = vector.bitcast %137 : vector<1xi8> to vector<1xf8E8M0FNU>
                %164 = vector.bitcast %138 : vector<1xi8> to vector<1xf8E8M0FNU>
                %165 = vector.bitcast %140 : vector<1xi8> to vector<1xf8E8M0FNU>
                %166 = vector.bitcast %141 : vector<1xi8> to vector<1xf8E8M0FNU>
                %167 = vector.bitcast %143 : vector<1xi8> to vector<1xf8E8M0FNU>
                %168 = vector.bitcast %144 : vector<1xi8> to vector<1xf8E8M0FNU>
                %169 = vector.bitcast %114 : vector<16xi8> to vector<32xf4E2M1FN>
                %170 = vector.bitcast %118 : vector<16xi8> to vector<32xf4E2M1FN>
                %171 = vector.bitcast %119 : vector<16xi8> to vector<32xf4E2M1FN>
                %172 = vector.bitcast %120 : vector<16xi8> to vector<32xf4E2M1FN>
                %173 = vector.bitcast %121 : vector<16xi8> to vector<32xf4E2M1FN>
                %174 = vector.bitcast %122 : vector<16xi8> to vector<32xf4E2M1FN>
                %175 = vector.bitcast %123 : vector<16xi8> to vector<32xf4E2M1FN>
                %176 = vector.bitcast %124 : vector<16xi8> to vector<32xf4E2M1FN>
                %177 = vector.bitcast %125 : vector<16xi8> to vector<32xf4E2M1FN>
                %178 = vector.bitcast %126 : vector<16xi8> to vector<32xf4E2M1FN>
                %179 = vector.bitcast %127 : vector<16xi8> to vector<32xf4E2M1FN>
                %180 = vector.bitcast %128 : vector<16xi8> to vector<32xf4E2M1FN>
                %181 = vector.bitcast %129 : vector<16xi8> to vector<32xf4E2M1FN>
                %182 = vector.bitcast %130 : vector<16xi8> to vector<32xf4E2M1FN>
                %183 = vector.bitcast %131 : vector<16xi8> to vector<32xf4E2M1FN>
                %184 = vector.bitcast %132 : vector<16xi8> to vector<32xf4E2M1FN>
                %185 = vector.bitcast %86 : vector<1xi8> to vector<1xf8E8M0FNU>
                %186 = vector.bitcast %89 : vector<1xi8> to vector<1xf8E8M0FNU>
                %187 = vector.bitcast %91 : vector<1xi8> to vector<1xf8E8M0FNU>
                %188 = vector.bitcast %92 : vector<1xi8> to vector<1xf8E8M0FNU>
                %189 = vector.bitcast %94 : vector<1xi8> to vector<1xf8E8M0FNU>
                %190 = vector.bitcast %95 : vector<1xi8> to vector<1xf8E8M0FNU>
                %191 = vector.bitcast %97 : vector<1xi8> to vector<1xf8E8M0FNU>
                %192 = vector.bitcast %98 : vector<1xi8> to vector<1xf8E8M0FNU>
                %193 = vector.bitcast %100 : vector<1xi8> to vector<1xf8E8M0FNU>
                %194 = vector.bitcast %101 : vector<1xi8> to vector<1xf8E8M0FNU>
                %195 = vector.bitcast %103 : vector<1xi8> to vector<1xf8E8M0FNU>
                %196 = vector.bitcast %104 : vector<1xi8> to vector<1xf8E8M0FNU>
                %197 = vector.bitcast %106 : vector<1xi8> to vector<1xf8E8M0FNU>
                %198 = vector.bitcast %107 : vector<1xi8> to vector<1xf8E8M0FNU>
                %199 = vector.bitcast %109 : vector<1xi8> to vector<1xf8E8M0FNU>
                %200 = vector.bitcast %110 : vector<1xi8> to vector<1xf8E8M0FNU>
                %201 = vector.extract %161[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %202 = vector.extract %185[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %203 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%202[0] * %169) + %83#0 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %204 = vector.extract %162[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %205 = vector.extract %186[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %206 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%205[0] * %170) + %203 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %207 = vector.extract %187[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %208 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%207[0] * %171) + %83#1 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %209 = vector.extract %188[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %210 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%209[0] * %172) + %208 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %211 = vector.extract %189[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %212 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%211[0] * %173) + %83#2 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %213 = vector.extract %190[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %214 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%213[0] * %174) + %212 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %215 = vector.extract %191[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %216 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%215[0] * %175) + %83#3 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %217 = vector.extract %192[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %218 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%217[0] * %176) + %216 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %219 = vector.extract %193[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %220 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%219[0] * %177) + %83#4 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %221 = vector.extract %194[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %222 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%221[0] * %178) + %220 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %223 = vector.extract %195[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %224 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%223[0] * %179) + %83#5 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %225 = vector.extract %196[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %226 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%225[0] * %180) + %224 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %227 = vector.extract %197[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %228 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%227[0] * %181) + %83#6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %229 = vector.extract %198[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %230 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%229[0] * %182) + %228 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %231 = vector.extract %199[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %232 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%231[0] * %183) + %83#7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %233 = vector.extract %200[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %234 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%233[0] * %184) + %232 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %235 = vector.extract %163[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %236 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%202[0] * %169) + %83#8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %237 = vector.extract %164[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %238 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%205[0] * %170) + %236 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %239 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%207[0] * %171) + %83#9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %240 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%209[0] * %172) + %239 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %241 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%211[0] * %173) + %83#10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %242 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%213[0] * %174) + %241 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %243 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%215[0] * %175) + %83#11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %244 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%217[0] * %176) + %243 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %245 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%219[0] * %177) + %83#12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %246 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%221[0] * %178) + %245 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %247 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%223[0] * %179) + %83#13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %248 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%225[0] * %180) + %247 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %249 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%227[0] * %181) + %83#14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %250 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%229[0] * %182) + %249 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %251 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%231[0] * %183) + %83#15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %252 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%233[0] * %184) + %251 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %253 = vector.extract %165[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %254 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%202[0] * %169) + %83#16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %255 = vector.extract %166[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %256 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%205[0] * %170) + %254 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %257 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%207[0] * %171) + %83#17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %258 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%209[0] * %172) + %257 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %259 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%211[0] * %173) + %83#18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %260 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%213[0] * %174) + %259 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %261 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%215[0] * %175) + %83#19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %262 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%217[0] * %176) + %261 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %263 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%219[0] * %177) + %83#20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %264 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%221[0] * %178) + %263 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %265 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%223[0] * %179) + %83#21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %266 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%225[0] * %180) + %265 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %267 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%227[0] * %181) + %83#22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %268 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%229[0] * %182) + %267 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %269 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%231[0] * %183) + %83#23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %270 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%233[0] * %184) + %269 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %271 = vector.extract %167[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %272 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%202[0] * %169) + %83#24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %273 = vector.extract %168[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %274 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%205[0] * %170) + %272 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %275 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%207[0] * %171) + %83#25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %276 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%209[0] * %172) + %275 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %277 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%211[0] * %173) + %83#26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %278 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%213[0] * %174) + %277 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %279 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%215[0] * %175) + %83#27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %280 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%217[0] * %176) + %279 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %281 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%219[0] * %177) + %83#28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %282 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%221[0] * %178) + %281 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %283 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%223[0] * %179) + %83#29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %284 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%225[0] * %180) + %283 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %285 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%227[0] * %181) + %83#30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %286 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%229[0] * %182) + %285 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %287 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%231[0] * %183) + %83#31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %288 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%233[0] * %184) + %287 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %289 = vector.extract_strided_slice %206 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %290 = affine.apply #map35()[%block_id_x]
                %291 = affine.apply #map35()[%block_id_y]
                %292 = affine.apply #map36()[%thread_id_x]
                %293 = arith.muli %290, %c57344 overflow<nsw> : index
                %294 = arith.muli %292, %c57344 overflow<nsw> : index
                %295 = arith.addi %293, %291 overflow<nsw> : index
                %296 = arith.addi %294, %84 overflow<nsw> : index
                %reinterpret_cast_13 = memref.reinterpret_cast %4 to offset: [%295], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
                %cast_14 = memref.cast %reinterpret_cast_13 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
                %297 = amdgpu.fat_raw_buffer_cast %cast_14 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                vector.store %289, %297[%296] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %298 = vector.extract_strided_slice %206 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %299 = affine.apply #map37()[%thread_id_x]
                %300 = arith.muli %299, %c57344 overflow<nsw> : index
                %301 = arith.addi %300, %84 overflow<nsw> : index
                vector.store %298, %297[%301] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %302 = vector.extract_strided_slice %206 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %303 = affine.apply #map38()[%thread_id_x]
                %304 = arith.muli %303, %c57344 overflow<nsw> : index
                %305 = arith.addi %304, %84 overflow<nsw> : index
                vector.store %302, %297[%305] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %306 = vector.extract_strided_slice %206 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %307 = affine.apply #map39()[%thread_id_x]
                %308 = arith.muli %307, %c57344 overflow<nsw> : index
                %309 = arith.addi %308, %84 overflow<nsw> : index
                vector.store %306, %297[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %310 = vector.extract_strided_slice %210 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %311 = arith.addi %294, %90 overflow<nsw> : index
                vector.store %310, %297[%311] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %312 = vector.extract_strided_slice %210 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %313 = arith.addi %300, %90 overflow<nsw> : index
                vector.store %312, %297[%313] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %314 = vector.extract_strided_slice %210 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %315 = arith.addi %304, %90 overflow<nsw> : index
                vector.store %314, %297[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %316 = vector.extract_strided_slice %210 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %317 = arith.addi %308, %90 overflow<nsw> : index
                vector.store %316, %297[%317] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %318 = vector.extract_strided_slice %214 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %319 = arith.addi %294, %93 overflow<nsw> : index
                vector.store %318, %297[%319] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %320 = vector.extract_strided_slice %214 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %321 = arith.addi %300, %93 overflow<nsw> : index
                vector.store %320, %297[%321] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %322 = vector.extract_strided_slice %214 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %323 = arith.addi %304, %93 overflow<nsw> : index
                vector.store %322, %297[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %324 = vector.extract_strided_slice %214 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %325 = arith.addi %308, %93 overflow<nsw> : index
                vector.store %324, %297[%325] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %326 = vector.extract_strided_slice %218 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %327 = arith.addi %294, %96 overflow<nsw> : index
                vector.store %326, %297[%327] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %328 = vector.extract_strided_slice %218 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %329 = arith.addi %300, %96 overflow<nsw> : index
                vector.store %328, %297[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %330 = vector.extract_strided_slice %218 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %331 = arith.addi %304, %96 overflow<nsw> : index
                vector.store %330, %297[%331] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %332 = vector.extract_strided_slice %218 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %333 = arith.addi %308, %96 overflow<nsw> : index
                vector.store %332, %297[%333] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %334 = vector.extract_strided_slice %222 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %335 = arith.addi %294, %99 overflow<nsw> : index
                vector.store %334, %297[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %336 = vector.extract_strided_slice %222 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %337 = arith.addi %300, %99 overflow<nsw> : index
                vector.store %336, %297[%337] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %338 = vector.extract_strided_slice %222 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %339 = arith.addi %304, %99 overflow<nsw> : index
                vector.store %338, %297[%339] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %340 = vector.extract_strided_slice %222 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %341 = arith.addi %308, %99 overflow<nsw> : index
                vector.store %340, %297[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %342 = vector.extract_strided_slice %226 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %343 = arith.addi %294, %102 overflow<nsw> : index
                vector.store %342, %297[%343] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %344 = vector.extract_strided_slice %226 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %345 = arith.addi %300, %102 overflow<nsw> : index
                vector.store %344, %297[%345] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %346 = vector.extract_strided_slice %226 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %347 = arith.addi %304, %102 overflow<nsw> : index
                vector.store %346, %297[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %348 = vector.extract_strided_slice %226 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %349 = arith.addi %308, %102 overflow<nsw> : index
                vector.store %348, %297[%349] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %350 = vector.extract_strided_slice %230 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %351 = arith.addi %294, %105 overflow<nsw> : index
                vector.store %350, %297[%351] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %352 = vector.extract_strided_slice %230 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %353 = arith.addi %300, %105 overflow<nsw> : index
                vector.store %352, %297[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %354 = vector.extract_strided_slice %230 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %355 = arith.addi %304, %105 overflow<nsw> : index
                vector.store %354, %297[%355] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %356 = vector.extract_strided_slice %230 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %357 = arith.addi %308, %105 overflow<nsw> : index
                vector.store %356, %297[%357] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %358 = vector.extract_strided_slice %234 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %359 = arith.addi %294, %108 overflow<nsw> : index
                vector.store %358, %297[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %360 = vector.extract_strided_slice %234 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %361 = arith.addi %300, %108 overflow<nsw> : index
                vector.store %360, %297[%361] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %362 = vector.extract_strided_slice %234 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %363 = arith.addi %304, %108 overflow<nsw> : index
                vector.store %362, %297[%363] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %364 = vector.extract_strided_slice %234 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %365 = arith.addi %308, %108 overflow<nsw> : index
                vector.store %364, %297[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %366 = vector.extract_strided_slice %238 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %367 = affine.apply #map40()[%thread_id_x]
                %368 = arith.muli %367, %c57344 overflow<nsw> : index
                %369 = arith.addi %368, %84 overflow<nsw> : index
                vector.store %366, %297[%369] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %370 = vector.extract_strided_slice %238 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %371 = affine.apply #map41()[%thread_id_x]
                %372 = arith.muli %371, %c57344 overflow<nsw> : index
                %373 = arith.addi %372, %84 overflow<nsw> : index
                vector.store %370, %297[%373] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %374 = vector.extract_strided_slice %238 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %375 = affine.apply #map42()[%thread_id_x]
                %376 = arith.muli %375, %c57344 overflow<nsw> : index
                %377 = arith.addi %376, %84 overflow<nsw> : index
                vector.store %374, %297[%377] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %378 = vector.extract_strided_slice %238 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %379 = affine.apply #map43()[%thread_id_x]
                %380 = arith.muli %379, %c57344 overflow<nsw> : index
                %381 = arith.addi %380, %84 overflow<nsw> : index
                vector.store %378, %297[%381] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %382 = vector.extract_strided_slice %240 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %383 = arith.addi %368, %90 overflow<nsw> : index
                vector.store %382, %297[%383] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %384 = vector.extract_strided_slice %240 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %385 = arith.addi %372, %90 overflow<nsw> : index
                vector.store %384, %297[%385] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %386 = vector.extract_strided_slice %240 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %387 = arith.addi %376, %90 overflow<nsw> : index
                vector.store %386, %297[%387] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %388 = vector.extract_strided_slice %240 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %389 = arith.addi %380, %90 overflow<nsw> : index
                vector.store %388, %297[%389] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %390 = vector.extract_strided_slice %242 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %391 = arith.addi %368, %93 overflow<nsw> : index
                vector.store %390, %297[%391] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %392 = vector.extract_strided_slice %242 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %393 = arith.addi %372, %93 overflow<nsw> : index
                vector.store %392, %297[%393] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %394 = vector.extract_strided_slice %242 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %395 = arith.addi %376, %93 overflow<nsw> : index
                vector.store %394, %297[%395] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %396 = vector.extract_strided_slice %242 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %397 = arith.addi %380, %93 overflow<nsw> : index
                vector.store %396, %297[%397] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %398 = vector.extract_strided_slice %244 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %399 = arith.addi %368, %96 overflow<nsw> : index
                vector.store %398, %297[%399] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %400 = vector.extract_strided_slice %244 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %401 = arith.addi %372, %96 overflow<nsw> : index
                vector.store %400, %297[%401] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %402 = vector.extract_strided_slice %244 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %403 = arith.addi %376, %96 overflow<nsw> : index
                vector.store %402, %297[%403] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %404 = vector.extract_strided_slice %244 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %405 = arith.addi %380, %96 overflow<nsw> : index
                vector.store %404, %297[%405] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %406 = vector.extract_strided_slice %246 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %407 = arith.addi %368, %99 overflow<nsw> : index
                vector.store %406, %297[%407] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %408 = vector.extract_strided_slice %246 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %409 = arith.addi %372, %99 overflow<nsw> : index
                vector.store %408, %297[%409] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %410 = vector.extract_strided_slice %246 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %411 = arith.addi %376, %99 overflow<nsw> : index
                vector.store %410, %297[%411] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %412 = vector.extract_strided_slice %246 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %413 = arith.addi %380, %99 overflow<nsw> : index
                vector.store %412, %297[%413] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %414 = vector.extract_strided_slice %248 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %415 = arith.addi %368, %102 overflow<nsw> : index
                vector.store %414, %297[%415] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %416 = vector.extract_strided_slice %248 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %417 = arith.addi %372, %102 overflow<nsw> : index
                vector.store %416, %297[%417] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %418 = vector.extract_strided_slice %248 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %419 = arith.addi %376, %102 overflow<nsw> : index
                vector.store %418, %297[%419] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %420 = vector.extract_strided_slice %248 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %421 = arith.addi %380, %102 overflow<nsw> : index
                vector.store %420, %297[%421] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %422 = vector.extract_strided_slice %250 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %423 = arith.addi %368, %105 overflow<nsw> : index
                vector.store %422, %297[%423] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %424 = vector.extract_strided_slice %250 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %425 = arith.addi %372, %105 overflow<nsw> : index
                vector.store %424, %297[%425] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %426 = vector.extract_strided_slice %250 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %427 = arith.addi %376, %105 overflow<nsw> : index
                vector.store %426, %297[%427] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %428 = vector.extract_strided_slice %250 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %429 = arith.addi %380, %105 overflow<nsw> : index
                vector.store %428, %297[%429] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %430 = vector.extract_strided_slice %252 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %431 = arith.addi %368, %108 overflow<nsw> : index
                vector.store %430, %297[%431] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %432 = vector.extract_strided_slice %252 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %433 = arith.addi %372, %108 overflow<nsw> : index
                vector.store %432, %297[%433] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %434 = vector.extract_strided_slice %252 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %435 = arith.addi %376, %108 overflow<nsw> : index
                vector.store %434, %297[%435] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %436 = vector.extract_strided_slice %252 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %437 = arith.addi %380, %108 overflow<nsw> : index
                vector.store %436, %297[%437] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %438 = vector.extract_strided_slice %256 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %439 = affine.apply #map44()[%thread_id_x]
                %440 = arith.muli %439, %c57344 overflow<nsw> : index
                %441 = arith.addi %440, %84 overflow<nsw> : index
                vector.store %438, %297[%441] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %442 = vector.extract_strided_slice %256 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %443 = affine.apply #map45()[%thread_id_x]
                %444 = arith.muli %443, %c57344 overflow<nsw> : index
                %445 = arith.addi %444, %84 overflow<nsw> : index
                vector.store %442, %297[%445] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %446 = vector.extract_strided_slice %256 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %447 = affine.apply #map46()[%thread_id_x]
                %448 = arith.muli %447, %c57344 overflow<nsw> : index
                %449 = arith.addi %448, %84 overflow<nsw> : index
                vector.store %446, %297[%449] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %450 = vector.extract_strided_slice %256 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %451 = affine.apply #map47()[%thread_id_x]
                %452 = arith.muli %451, %c57344 overflow<nsw> : index
                %453 = arith.addi %452, %84 overflow<nsw> : index
                vector.store %450, %297[%453] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %454 = vector.extract_strided_slice %258 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %455 = arith.addi %440, %90 overflow<nsw> : index
                vector.store %454, %297[%455] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %456 = vector.extract_strided_slice %258 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %457 = arith.addi %444, %90 overflow<nsw> : index
                vector.store %456, %297[%457] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %458 = vector.extract_strided_slice %258 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %459 = arith.addi %448, %90 overflow<nsw> : index
                vector.store %458, %297[%459] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %460 = vector.extract_strided_slice %258 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %461 = arith.addi %452, %90 overflow<nsw> : index
                vector.store %460, %297[%461] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %462 = vector.extract_strided_slice %260 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %463 = arith.addi %440, %93 overflow<nsw> : index
                vector.store %462, %297[%463] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %464 = vector.extract_strided_slice %260 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %465 = arith.addi %444, %93 overflow<nsw> : index
                vector.store %464, %297[%465] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %466 = vector.extract_strided_slice %260 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %467 = arith.addi %448, %93 overflow<nsw> : index
                vector.store %466, %297[%467] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %468 = vector.extract_strided_slice %260 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %469 = arith.addi %452, %93 overflow<nsw> : index
                vector.store %468, %297[%469] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %470 = vector.extract_strided_slice %262 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %471 = arith.addi %440, %96 overflow<nsw> : index
                vector.store %470, %297[%471] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %472 = vector.extract_strided_slice %262 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %473 = arith.addi %444, %96 overflow<nsw> : index
                vector.store %472, %297[%473] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %474 = vector.extract_strided_slice %262 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %475 = arith.addi %448, %96 overflow<nsw> : index
                vector.store %474, %297[%475] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %476 = vector.extract_strided_slice %262 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %477 = arith.addi %452, %96 overflow<nsw> : index
                vector.store %476, %297[%477] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %478 = vector.extract_strided_slice %264 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %479 = arith.addi %440, %99 overflow<nsw> : index
                vector.store %478, %297[%479] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %480 = vector.extract_strided_slice %264 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %481 = arith.addi %444, %99 overflow<nsw> : index
                vector.store %480, %297[%481] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %482 = vector.extract_strided_slice %264 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %483 = arith.addi %448, %99 overflow<nsw> : index
                vector.store %482, %297[%483] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %484 = vector.extract_strided_slice %264 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %485 = arith.addi %452, %99 overflow<nsw> : index
                vector.store %484, %297[%485] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %486 = vector.extract_strided_slice %266 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %487 = arith.addi %440, %102 overflow<nsw> : index
                vector.store %486, %297[%487] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %488 = vector.extract_strided_slice %266 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %489 = arith.addi %444, %102 overflow<nsw> : index
                vector.store %488, %297[%489] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %490 = vector.extract_strided_slice %266 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %491 = arith.addi %448, %102 overflow<nsw> : index
                vector.store %490, %297[%491] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %492 = vector.extract_strided_slice %266 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %493 = arith.addi %452, %102 overflow<nsw> : index
                vector.store %492, %297[%493] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %494 = vector.extract_strided_slice %268 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %495 = arith.addi %440, %105 overflow<nsw> : index
                vector.store %494, %297[%495] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %496 = vector.extract_strided_slice %268 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %497 = arith.addi %444, %105 overflow<nsw> : index
                vector.store %496, %297[%497] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %498 = vector.extract_strided_slice %268 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %499 = arith.addi %448, %105 overflow<nsw> : index
                vector.store %498, %297[%499] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %500 = vector.extract_strided_slice %268 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %501 = arith.addi %452, %105 overflow<nsw> : index
                vector.store %500, %297[%501] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %502 = vector.extract_strided_slice %270 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %503 = arith.addi %440, %108 overflow<nsw> : index
                vector.store %502, %297[%503] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %504 = vector.extract_strided_slice %270 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %505 = arith.addi %444, %108 overflow<nsw> : index
                vector.store %504, %297[%505] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %506 = vector.extract_strided_slice %270 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %507 = arith.addi %448, %108 overflow<nsw> : index
                vector.store %506, %297[%507] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %508 = vector.extract_strided_slice %270 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %509 = arith.addi %452, %108 overflow<nsw> : index
                vector.store %508, %297[%509] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %510 = vector.extract_strided_slice %274 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %511 = affine.apply #map48()[%thread_id_x]
                %512 = arith.muli %511, %c57344 overflow<nsw> : index
                %513 = arith.addi %512, %84 overflow<nsw> : index
                vector.store %510, %297[%513] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %514 = vector.extract_strided_slice %274 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %515 = affine.apply #map49()[%thread_id_x]
                %516 = arith.muli %515, %c57344 overflow<nsw> : index
                %517 = arith.addi %516, %84 overflow<nsw> : index
                vector.store %514, %297[%517] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %518 = vector.extract_strided_slice %274 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %519 = affine.apply #map50()[%thread_id_x]
                %520 = arith.muli %519, %c57344 overflow<nsw> : index
                %521 = arith.addi %520, %84 overflow<nsw> : index
                vector.store %518, %297[%521] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %522 = vector.extract_strided_slice %274 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %523 = affine.apply #map51()[%thread_id_x]
                %524 = arith.muli %523, %c57344 overflow<nsw> : index
                %525 = arith.addi %524, %84 overflow<nsw> : index
                vector.store %522, %297[%525] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %526 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %527 = arith.addi %512, %90 overflow<nsw> : index
                vector.store %526, %297[%527] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %528 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %529 = arith.addi %516, %90 overflow<nsw> : index
                vector.store %528, %297[%529] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %530 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %531 = arith.addi %520, %90 overflow<nsw> : index
                vector.store %530, %297[%531] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %532 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %533 = arith.addi %524, %90 overflow<nsw> : index
                vector.store %532, %297[%533] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %534 = vector.extract_strided_slice %278 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %535 = arith.addi %512, %93 overflow<nsw> : index
                vector.store %534, %297[%535] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %536 = vector.extract_strided_slice %278 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %537 = arith.addi %516, %93 overflow<nsw> : index
                vector.store %536, %297[%537] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %538 = vector.extract_strided_slice %278 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %539 = arith.addi %520, %93 overflow<nsw> : index
                vector.store %538, %297[%539] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %540 = vector.extract_strided_slice %278 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %541 = arith.addi %524, %93 overflow<nsw> : index
                vector.store %540, %297[%541] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %542 = vector.extract_strided_slice %280 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %543 = arith.addi %512, %96 overflow<nsw> : index
                vector.store %542, %297[%543] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %544 = vector.extract_strided_slice %280 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %545 = arith.addi %516, %96 overflow<nsw> : index
                vector.store %544, %297[%545] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %546 = vector.extract_strided_slice %280 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %547 = arith.addi %520, %96 overflow<nsw> : index
                vector.store %546, %297[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %548 = vector.extract_strided_slice %280 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %549 = arith.addi %524, %96 overflow<nsw> : index
                vector.store %548, %297[%549] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %550 = vector.extract_strided_slice %282 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %551 = arith.addi %512, %99 overflow<nsw> : index
                vector.store %550, %297[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %552 = vector.extract_strided_slice %282 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %553 = arith.addi %516, %99 overflow<nsw> : index
                vector.store %552, %297[%553] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %554 = vector.extract_strided_slice %282 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %555 = arith.addi %520, %99 overflow<nsw> : index
                vector.store %554, %297[%555] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %556 = vector.extract_strided_slice %282 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %557 = arith.addi %524, %99 overflow<nsw> : index
                vector.store %556, %297[%557] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %558 = vector.extract_strided_slice %284 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %559 = arith.addi %512, %102 overflow<nsw> : index
                vector.store %558, %297[%559] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %560 = vector.extract_strided_slice %284 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %561 = arith.addi %516, %102 overflow<nsw> : index
                vector.store %560, %297[%561] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %562 = vector.extract_strided_slice %284 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %563 = arith.addi %520, %102 overflow<nsw> : index
                vector.store %562, %297[%563] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %564 = vector.extract_strided_slice %284 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %565 = arith.addi %524, %102 overflow<nsw> : index
                vector.store %564, %297[%565] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %566 = vector.extract_strided_slice %286 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %567 = arith.addi %512, %105 overflow<nsw> : index
                vector.store %566, %297[%567] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %568 = vector.extract_strided_slice %286 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %569 = arith.addi %516, %105 overflow<nsw> : index
                vector.store %568, %297[%569] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %570 = vector.extract_strided_slice %286 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %571 = arith.addi %520, %105 overflow<nsw> : index
                vector.store %570, %297[%571] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %572 = vector.extract_strided_slice %286 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %573 = arith.addi %524, %105 overflow<nsw> : index
                vector.store %572, %297[%573] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %574 = vector.extract_strided_slice %288 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %575 = arith.addi %512, %108 overflow<nsw> : index
                vector.store %574, %297[%575] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %576 = vector.extract_strided_slice %288 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %577 = arith.addi %516, %108 overflow<nsw> : index
                vector.store %576, %297[%577] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %578 = vector.extract_strided_slice %288 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %579 = arith.addi %520, %108 overflow<nsw> : index
                vector.store %578, %297[%579] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %580 = vector.extract_strided_slice %288 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %581 = arith.addi %524, %108 overflow<nsw> : index
                vector.store %580, %297[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<4096x8192xi8>
            %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<4096x512xi8>
            %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<57344x8192xi8>
            %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<57344x512xi8>
            %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<4096x57344xf32>
            %5 = flow.dispatch @gemm::@gemm(%0, %1, %2, %3, %4) : (tensor<4096x8192xi8>, tensor<4096x512xi8>, tensor<57344x8192xi8>, tensor<57344x512xi8>, tensor<4096x57344xf32>) -> %4
            %6 = hal.tensor.barrier join(%5 : tensor<4096x57344xf32>) => %arg6 : !hal.fence
            %7 = hal.tensor.export %6 : tensor<4096x57344xf32> -> !hal.buffer_view
            return %7 : !hal.buffer_view
        }
        }
    """
    # split into depdendent and independent loads
    mlir_pingpong_mixed = """
        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 16)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
        #map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
        #map9 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
        #map11 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
        #map12 = affine_map<()[s0] -> ((s0 floordiv 2) mod 2)>
        #map13 = affine_map<()[s0] -> (s0 mod 2)>
        #map14 = affine_map<()[s0] -> (s0 * 4)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 32 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 256)>
        #map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
        #map18 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
        #map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
        #map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
        #map22 = affine_map<()[s0] -> (s0 * 4 + (s0 mod 64) floordiv 16 - (s0 floordiv 2) * 8)>
        #map23 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
        #map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
        #map25 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
        #map26 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
        #map27 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
        #map28 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
        #map29 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
        #map30 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
        #map31 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map32 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
        #map33 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 128)>
        #map34 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 + 8)>
        #map35 = affine_map<()[s0] -> (s0 * 256)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
        #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
        #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
        #map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
        #map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
        #map47 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
        #map48 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
        #map49 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
        #map50 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
        #map51 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c16 = arith.constant 16 : index
            %c224 = arith.constant 224 : index
            %c1 = arith.constant 1 : index
            stream.return %c16, %c224, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c512_i14 = arith.constant 512 : i14
                %c-8192_i14 = arith.constant -8192 : i14
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c57344 = arith.constant 57344 : index
                %c63 = arith.constant 63 : index
                %c512 = arith.constant 512 : index
                %c2147483646_i64 = arith.constant 2147483646 : i64
                %c8192 = arith.constant 8192 : index
                %c1 = arith.constant 1 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
                %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
                %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 16
                %block_id_y = gpu.block_id  y upper_bound 224
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %alloc = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_0 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_3 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_4 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_5 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_6 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %5 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %6 = affine.apply #map1()[%thread_id_x]
                %7 = affine.apply #map2()[%thread_id_x]
                %8 = arith.xori %7, %6 : index
                %9 = affine.apply #map3()[%8]
                %10 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %11 = gpu.subgroup_broadcast %10,  first_active_lane : index
                %12 = gpu.subgroup_broadcast %c0,  first_active_lane : index
                %13 = arith.muli %5, %c8192 overflow<nsw> : index
                %14 = arith.addi %13, %9 overflow<nsw> : index
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast = memref.cast %reinterpret_cast : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %15 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %15[%14], %alloc_6[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %16 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
                %17 = affine.apply #map6()[%thread_id_x, %thread_id_y]
                %18 = gpu.subgroup_broadcast %17,  first_active_lane : index
                %19 = arith.muli %16, %c8192 overflow<nsw> : index
                %20 = arith.addi %19, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%20], %alloc_6[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %21 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
                %22 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                %23 = gpu.subgroup_broadcast %22,  first_active_lane : index
                %24 = arith.muli %21, %c8192 overflow<nsw> : index
                %25 = arith.addi %24, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%25], %alloc_6[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %26 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
                %27 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %28 = gpu.subgroup_broadcast %27,  first_active_lane : index
                %29 = arith.muli %26, %c8192 overflow<nsw> : index
                %30 = arith.addi %29, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%30], %alloc_6[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %31 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_x]
                %32 = affine.apply #map12()[%thread_id_x]
                %33 = affine.apply #map13()[%thread_id_x]
                %34 = arith.xori %33, %32 : index
                %35 = affine.apply #map14()[%34]
                %36 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %37 = gpu.subgroup_broadcast %36,  first_active_lane : index
                %38 = arith.muli %31, %c512 overflow<nsw> : index
                %39 = arith.addi %38, %35 overflow<nsw> : index
                %reinterpret_cast_7 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %40 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %40[%39], %alloc_4[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                %41 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
                %42 = arith.muli %41, %c8192 overflow<nsw> : index
                %43 = arith.addi %42, %9 overflow<nsw> : index
                %reinterpret_cast_9 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_10 = memref.cast %reinterpret_cast_9 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %44 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %44[%43], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %45 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                %46 = arith.muli %45, %c8192 overflow<nsw> : index
                %47 = arith.addi %46, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%47], %alloc_2[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %48 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
                %49 = arith.muli %48, %c8192 overflow<nsw> : index
                %50 = arith.addi %49, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%50], %alloc_2[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %51 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_y]
                %52 = arith.muli %51, %c8192 overflow<nsw> : index
                %53 = arith.addi %52, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%53], %alloc_2[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %54 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_y]
                %55 = arith.muli %54, %c512 overflow<nsw> : index
                %56 = arith.addi %55, %35 overflow<nsw> : index
                %reinterpret_cast_11 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_12 = memref.cast %reinterpret_cast_11 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %57 = amdgpu.fat_raw_buffer_cast %cast_12 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %57[%56], %alloc_0[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.s.barrier
                %58 = affine.apply #map16()[%thread_id_x, %thread_id_y]
                %59 = arith.index_cast %58 : index to i32
                %60 = arith.cmpi sge, %59, %c4_i32 : i32
                %61 = arith.cmpi slt, %59, %c4_i32 : i32
                scf.if %60 {
                rocdl.s.barrier
                }
                %62 = affine.apply #map17()[%thread_id_x]
                %63 = affine.apply #map18()[%thread_id_x]
                %64 = arith.xori %63, %7 : index
                %65 = affine.apply #map3()[%64]
                %66 = affine.apply #map19()[%thread_id_x]
                %67 = affine.apply #map20()[%thread_id_x]
                %68 = affine.apply #map21()[%thread_id_x]
                %69 = affine.apply #map22()[%thread_id_x]
                %70 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %71 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %72 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %73 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %74 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %75 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %76 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %77 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %78 = affine.apply #map31()[%thread_id_x]
                %79 = arith.xori %78, %7 : index
                %80 = affine.apply #map3()[%79]
                %81 = arith.xori %33, %c1 : index
                %82 = affine.apply #map32()[%thread_id_x, %81]
                %83:40 = scf.for %arg5 = %c0 to %c63 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %alloc_6, %arg39 = %alloc_5, %arg40 = %alloc_4, %arg41 = %alloc_3, %arg42 = %alloc_2, %arg43 = %alloc_1, %arg44 = %alloc_0, %arg45 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>) {
                rocdl.sched.barrier 0
                rocdl.s.barrier
                %582 = affine.apply #map33()[%arg5, %8]
                %583 = arith.addi %13, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%583], %arg39[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %584 = arith.addi %19, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%584], %arg39[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %585 = arith.addi %24, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%585], %arg39[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %586 = arith.addi %29, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%586], %arg39[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %587 = affine.apply #map34()[%arg5, %34]
                %588 = arith.addi %38, %587 overflow<nsw> : index
                amdgpu.gather_to_lds %40[%588], %arg41[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                %589 = arith.addi %42, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%589], %arg43[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %590 = arith.addi %46, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%590], %arg43[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %591 = arith.addi %49, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%591], %arg43[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %592 = arith.addi %52, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%592], %arg43[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %593 = arith.addi %55, %587 overflow<nsw> : index
                amdgpu.gather_to_lds %57[%593], %arg45[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(10)
                // --- SAFE vector.loads: A(M0,M1), Ascale(M0,M1), B(N0,N1,N4,N5), Bscale(N0,N1,N4,N5) ---
                %594 = vector.load %arg38[%62, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %595 = vector.load %arg38[%66, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %598 = vector.load %arg40[%62, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %599 = vector.load %arg40[%66, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %602 = vector.load %arg42[%70, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %603 = vector.load %arg42[%71, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %606 = vector.load %arg42[%74, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %607 = vector.load %arg42[%75, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %610 = vector.load %arg44[%70, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %611 = vector.load %arg44[%71, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %614 = vector.load %arg44[%74, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %615 = vector.load %arg44[%75, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                // --- SAFE bitcasts ---
                %618 = vector.bitcast %594 : vector<16xi8> to vector<32xf4E2M1FN>
                %619 = vector.bitcast %595 : vector<16xi8> to vector<32xf4E2M1FN>
                %622 = vector.bitcast %598 : vector<1xi8> to vector<1xf8E8M0FNU>
                %623 = vector.bitcast %599 : vector<1xi8> to vector<1xf8E8M0FNU>
                %626 = vector.bitcast %602 : vector<16xi8> to vector<32xf4E2M1FN>
                %627 = vector.bitcast %603 : vector<16xi8> to vector<32xf4E2M1FN>
                %630 = vector.bitcast %606 : vector<16xi8> to vector<32xf4E2M1FN>
                %631 = vector.bitcast %607 : vector<16xi8> to vector<32xf4E2M1FN>
                %634 = vector.bitcast %610 : vector<1xi8> to vector<1xf8E8M0FNU>
                %635 = vector.bitcast %611 : vector<1xi8> to vector<1xf8E8M0FNU>
                %638 = vector.bitcast %614 : vector<1xi8> to vector<1xf8E8M0FNU>
                %639 = vector.bitcast %615 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                // --- SAFE MFMAs: M0,M1 x N0,N1,N4,N5 (cluster 0 data only) ---
                %642 = vector.extract %622[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %643 = vector.extract %634[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %644 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%643[0] * %626) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %645 = vector.extract %635[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %646 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%645[0] * %627) + %arg7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %651 = vector.extract %638[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %652 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%651[0] * %630) + %arg10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %653 = vector.extract %639[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %654 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%653[0] * %631) + %arg11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %659 = vector.extract %623[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %660 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%643[0] * %626) + %arg14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %661 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%645[0] * %627) + %arg15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %664 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%651[0] * %630) + %arg18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %665 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%653[0] * %631) + %arg19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                // --- DEPENDENT vector.loads: A(M2,M3), Ascale(M2,M3), B(N2,N3,N6,N7), Bscale(N2,N3,N6,N7) ---
                %596 = vector.load %arg38[%67, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %597 = vector.load %arg38[%68, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %600 = vector.load %arg40[%67, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %601 = vector.load %arg40[%68, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %604 = vector.load %arg42[%72, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %605 = vector.load %arg42[%73, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %608 = vector.load %arg42[%76, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %609 = vector.load %arg42[%77, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %612 = vector.load %arg44[%72, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %613 = vector.load %arg44[%73, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %616 = vector.load %arg44[%76, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %617 = vector.load %arg44[%77, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                // --- DEPENDENT bitcasts ---
                %620 = vector.bitcast %596 : vector<16xi8> to vector<32xf4E2M1FN>
                %621 = vector.bitcast %597 : vector<16xi8> to vector<32xf4E2M1FN>
                %624 = vector.bitcast %600 : vector<1xi8> to vector<1xf8E8M0FNU>
                %625 = vector.bitcast %601 : vector<1xi8> to vector<1xf8E8M0FNU>
                %628 = vector.bitcast %604 : vector<16xi8> to vector<32xf4E2M1FN>
                %629 = vector.bitcast %605 : vector<16xi8> to vector<32xf4E2M1FN>
                %632 = vector.bitcast %608 : vector<16xi8> to vector<32xf4E2M1FN>
                %633 = vector.bitcast %609 : vector<16xi8> to vector<32xf4E2M1FN>
                %636 = vector.bitcast %612 : vector<1xi8> to vector<1xf8E8M0FNU>
                %637 = vector.bitcast %613 : vector<1xi8> to vector<1xf8E8M0FNU>
                %640 = vector.bitcast %616 : vector<1xi8> to vector<1xf8E8M0FNU>
                %641 = vector.bitcast %617 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.s.setprio 1
                // --- DEPENDENT MFMAs: M0,M1 x N2,N3,N6,N7 (cluster 1 B data) ---
                %647 = vector.extract %636[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %648 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%647[0] * %628) + %arg8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %649 = vector.extract %637[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %650 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%649[0] * %629) + %arg9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %655 = vector.extract %640[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %656 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%655[0] * %632) + %arg12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %657 = vector.extract %641[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %658 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%657[0] * %633) + %arg13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %662 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%647[0] * %628) + %arg16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %663 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%649[0] * %629) + %arg17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %666 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%655[0] * %632) + %arg20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %667 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%657[0] * %633) + %arg21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                // --- DEPENDENT MFMAs: M2 x all N (cluster 1 A data) ---
                %668 = vector.extract %624[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %669 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%643[0] * %626) + %arg22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %670 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%645[0] * %627) + %arg23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %671 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%647[0] * %628) + %arg24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %672 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%649[0] * %629) + %arg25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %673 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%651[0] * %630) + %arg26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %674 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%653[0] * %631) + %arg27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %675 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%655[0] * %632) + %arg28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %676 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%657[0] * %633) + %arg29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                // --- DEPENDENT MFMAs: M3 x all N (cluster 1 A data) ---
                %677 = vector.extract %625[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %678 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%643[0] * %626) + %arg30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %679 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%645[0] * %627) + %arg31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %680 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%647[0] * %628) + %arg32 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %681 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%649[0] * %629) + %arg33 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %682 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%651[0] * %630) + %arg34 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %683 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%653[0] * %631) + %arg35 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %684 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%655[0] * %632) + %arg36 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %685 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%657[0] * %633) + %arg37 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.sched.barrier 0
                // --- PHASE 2 SAFE vector.loads: A(M0,M1), Ascale(M0,M1), B(N0,N1,N4,N5), Bscale(N0,N1,N4,N5) ---
                %686 = vector.load %arg38[%62, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %687 = vector.load %arg38[%66, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %690 = vector.load %arg40[%62, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %691 = vector.load %arg40[%66, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %694 = vector.load %arg42[%70, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %695 = vector.load %arg42[%71, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %698 = vector.load %arg42[%74, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %699 = vector.load %arg42[%75, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %702 = vector.load %arg44[%70, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %703 = vector.load %arg44[%71, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %706 = vector.load %arg44[%74, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %707 = vector.load %arg44[%75, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                // --- PHASE 2 SAFE bitcasts ---
                %710 = vector.bitcast %686 : vector<16xi8> to vector<32xf4E2M1FN>
                %711 = vector.bitcast %687 : vector<16xi8> to vector<32xf4E2M1FN>
                %714 = vector.bitcast %690 : vector<1xi8> to vector<1xf8E8M0FNU>
                %715 = vector.bitcast %691 : vector<1xi8> to vector<1xf8E8M0FNU>
                %718 = vector.bitcast %694 : vector<16xi8> to vector<32xf4E2M1FN>
                %719 = vector.bitcast %695 : vector<16xi8> to vector<32xf4E2M1FN>
                %722 = vector.bitcast %698 : vector<16xi8> to vector<32xf4E2M1FN>
                %723 = vector.bitcast %699 : vector<16xi8> to vector<32xf4E2M1FN>
                %726 = vector.bitcast %702 : vector<1xi8> to vector<1xf8E8M0FNU>
                %727 = vector.bitcast %703 : vector<1xi8> to vector<1xf8E8M0FNU>
                %730 = vector.bitcast %706 : vector<1xi8> to vector<1xf8E8M0FNU>
                %731 = vector.bitcast %707 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                // --- PHASE 2 SAFE MFMAs: M0,M1 x N0,N1,N4,N5 (cluster 0 data only) ---
                %734 = vector.extract %714[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %735 = vector.extract %726[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %736 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%735[0] * %718) + %644 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %737 = vector.extract %727[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %738 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%737[0] * %719) + %646 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %743 = vector.extract %730[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %744 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%743[0] * %722) + %652 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %745 = vector.extract %731[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %746 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%745[0] * %723) + %654 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %751 = vector.extract %715[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %752 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%735[0] * %718) + %660 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %753 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%737[0] * %719) + %661 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %756 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%743[0] * %722) + %664 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %757 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%745[0] * %723) + %665 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                // --- PHASE 2 DEPENDENT vector.loads: A(M2,M3), Ascale(M2,M3), B(N2,N3,N6,N7), Bscale(N2,N3,N6,N7) ---
                %688 = vector.load %arg38[%67, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %689 = vector.load %arg38[%68, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %692 = vector.load %arg40[%67, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %693 = vector.load %arg40[%68, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %696 = vector.load %arg42[%72, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %697 = vector.load %arg42[%73, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %700 = vector.load %arg42[%76, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %701 = vector.load %arg42[%77, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %704 = vector.load %arg44[%72, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %705 = vector.load %arg44[%73, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %708 = vector.load %arg44[%76, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %709 = vector.load %arg44[%77, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                // --- PHASE 2 DEPENDENT bitcasts ---
                %712 = vector.bitcast %688 : vector<16xi8> to vector<32xf4E2M1FN>
                %713 = vector.bitcast %689 : vector<16xi8> to vector<32xf4E2M1FN>
                %716 = vector.bitcast %692 : vector<1xi8> to vector<1xf8E8M0FNU>
                %717 = vector.bitcast %693 : vector<1xi8> to vector<1xf8E8M0FNU>
                %720 = vector.bitcast %696 : vector<16xi8> to vector<32xf4E2M1FN>
                %721 = vector.bitcast %697 : vector<16xi8> to vector<32xf4E2M1FN>
                %724 = vector.bitcast %700 : vector<16xi8> to vector<32xf4E2M1FN>
                %725 = vector.bitcast %701 : vector<16xi8> to vector<32xf4E2M1FN>
                %728 = vector.bitcast %704 : vector<1xi8> to vector<1xf8E8M0FNU>
                %729 = vector.bitcast %705 : vector<1xi8> to vector<1xf8E8M0FNU>
                %732 = vector.bitcast %708 : vector<1xi8> to vector<1xf8E8M0FNU>
                %733 = vector.bitcast %709 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.s.setprio 1
                // --- PHASE 2 DEPENDENT MFMAs: M0,M1 x N2,N3,N6,N7 (cluster 1 B data) ---
                %739 = vector.extract %728[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %740 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%739[0] * %720) + %648 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %741 = vector.extract %729[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %742 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%741[0] * %721) + %650 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %747 = vector.extract %732[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %748 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%747[0] * %724) + %656 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %749 = vector.extract %733[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %750 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%749[0] * %725) + %658 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %754 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%739[0] * %720) + %662 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %755 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%741[0] * %721) + %663 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %758 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%747[0] * %724) + %666 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %759 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%749[0] * %725) + %667 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                // --- PHASE 2 DEPENDENT MFMAs: M2 x all N (cluster 1 A data) ---
                %760 = vector.extract %716[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %761 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%735[0] * %718) + %669 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %762 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%737[0] * %719) + %670 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %763 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%739[0] * %720) + %671 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %764 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%741[0] * %721) + %672 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %765 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%743[0] * %722) + %673 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %766 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%745[0] * %723) + %674 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %767 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%747[0] * %724) + %675 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %768 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%749[0] * %725) + %676 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                // --- PHASE 2 DEPENDENT MFMAs: M3 x all N (cluster 1 A data) ---
                %769 = vector.extract %717[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %770 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%735[0] * %718) + %678 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %771 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%737[0] * %719) + %679 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %772 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%739[0] * %720) + %680 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %773 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%741[0] * %721) + %681 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %774 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%743[0] * %722) + %682 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %775 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%745[0] * %723) + %683 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %776 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%747[0] * %724) + %684 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %777 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%749[0] * %725) + %685 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                scf.yield %736, %738, %740, %742, %744, %746, %748, %750, %752, %753, %754, %755, %756, %757, %758, %759, %761, %762, %763, %764, %765, %766, %767, %768, %770, %771, %772, %773, %774, %775, %776, %777, %arg39, %arg38, %arg41, %arg40, %arg43, %arg42, %arg45, %arg44 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                }
                scf.if %61 {
                rocdl.s.barrier
                }
                amdgpu.lds_barrier
                %84 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %85 = affine.apply #map22()[%thread_id_x]
                %86 = vector.load %83#38[%84, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %87 = arith.xori %33, %c1 : index
                %88 = affine.apply #map32()[%thread_id_x, %87]
                %89 = vector.load %83#38[%84, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %90 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %91 = vector.load %83#38[%90, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %92 = vector.load %83#38[%90, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %93 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %94 = vector.load %83#38[%93, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %95 = vector.load %83#38[%93, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %96 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %97 = vector.load %83#38[%96, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %98 = vector.load %83#38[%96, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %99 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %100 = vector.load %83#38[%99, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %101 = vector.load %83#38[%99, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %102 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %103 = vector.load %83#38[%102, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %104 = vector.load %83#38[%102, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %105 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %106 = vector.load %83#38[%105, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %107 = vector.load %83#38[%105, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %108 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %109 = vector.load %83#38[%108, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %110 = vector.load %83#38[%108, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %111 = affine.apply #map18()[%thread_id_x]
                %112 = arith.xori %111, %7 : index
                %113 = affine.apply #map3()[%112]
                %114 = vector.load %83#36[%84, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %115 = affine.apply #map31()[%thread_id_x]
                %116 = arith.xori %115, %7 : index
                %117 = affine.apply #map3()[%116]
                %118 = vector.load %83#36[%84, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %119 = vector.load %83#36[%90, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %120 = vector.load %83#36[%90, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %121 = vector.load %83#36[%93, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %122 = vector.load %83#36[%93, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %123 = vector.load %83#36[%96, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %124 = vector.load %83#36[%96, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %125 = vector.load %83#36[%99, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %126 = vector.load %83#36[%99, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %127 = vector.load %83#36[%102, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %128 = vector.load %83#36[%102, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %129 = vector.load %83#36[%105, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %130 = vector.load %83#36[%105, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %131 = vector.load %83#36[%108, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %132 = vector.load %83#36[%108, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %133 = affine.apply #map17()[%thread_id_x]
                %134 = vector.load %83#34[%133, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %135 = vector.load %83#34[%133, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %136 = affine.apply #map19()[%thread_id_x]
                %137 = vector.load %83#34[%136, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %138 = vector.load %83#34[%136, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %139 = affine.apply #map20()[%thread_id_x]
                %140 = vector.load %83#34[%139, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %141 = vector.load %83#34[%139, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %142 = affine.apply #map21()[%thread_id_x]
                %143 = vector.load %83#34[%142, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %144 = vector.load %83#34[%142, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %145 = vector.load %83#32[%133, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %146 = vector.load %83#32[%133, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %147 = vector.load %83#32[%136, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %148 = vector.load %83#32[%136, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %149 = vector.load %83#32[%139, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %150 = vector.load %83#32[%139, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %151 = vector.load %83#32[%142, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %152 = vector.load %83#32[%142, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %153 = vector.bitcast %145 : vector<16xi8> to vector<32xf4E2M1FN>
                %154 = vector.bitcast %146 : vector<16xi8> to vector<32xf4E2M1FN>
                %155 = vector.bitcast %147 : vector<16xi8> to vector<32xf4E2M1FN>
                %156 = vector.bitcast %148 : vector<16xi8> to vector<32xf4E2M1FN>
                %157 = vector.bitcast %149 : vector<16xi8> to vector<32xf4E2M1FN>
                %158 = vector.bitcast %150 : vector<16xi8> to vector<32xf4E2M1FN>
                %159 = vector.bitcast %151 : vector<16xi8> to vector<32xf4E2M1FN>
                %160 = vector.bitcast %152 : vector<16xi8> to vector<32xf4E2M1FN>
                %161 = vector.bitcast %134 : vector<1xi8> to vector<1xf8E8M0FNU>
                %162 = vector.bitcast %135 : vector<1xi8> to vector<1xf8E8M0FNU>
                %163 = vector.bitcast %137 : vector<1xi8> to vector<1xf8E8M0FNU>
                %164 = vector.bitcast %138 : vector<1xi8> to vector<1xf8E8M0FNU>
                %165 = vector.bitcast %140 : vector<1xi8> to vector<1xf8E8M0FNU>
                %166 = vector.bitcast %141 : vector<1xi8> to vector<1xf8E8M0FNU>
                %167 = vector.bitcast %143 : vector<1xi8> to vector<1xf8E8M0FNU>
                %168 = vector.bitcast %144 : vector<1xi8> to vector<1xf8E8M0FNU>
                %169 = vector.bitcast %114 : vector<16xi8> to vector<32xf4E2M1FN>
                %170 = vector.bitcast %118 : vector<16xi8> to vector<32xf4E2M1FN>
                %171 = vector.bitcast %119 : vector<16xi8> to vector<32xf4E2M1FN>
                %172 = vector.bitcast %120 : vector<16xi8> to vector<32xf4E2M1FN>
                %173 = vector.bitcast %121 : vector<16xi8> to vector<32xf4E2M1FN>
                %174 = vector.bitcast %122 : vector<16xi8> to vector<32xf4E2M1FN>
                %175 = vector.bitcast %123 : vector<16xi8> to vector<32xf4E2M1FN>
                %176 = vector.bitcast %124 : vector<16xi8> to vector<32xf4E2M1FN>
                %177 = vector.bitcast %125 : vector<16xi8> to vector<32xf4E2M1FN>
                %178 = vector.bitcast %126 : vector<16xi8> to vector<32xf4E2M1FN>
                %179 = vector.bitcast %127 : vector<16xi8> to vector<32xf4E2M1FN>
                %180 = vector.bitcast %128 : vector<16xi8> to vector<32xf4E2M1FN>
                %181 = vector.bitcast %129 : vector<16xi8> to vector<32xf4E2M1FN>
                %182 = vector.bitcast %130 : vector<16xi8> to vector<32xf4E2M1FN>
                %183 = vector.bitcast %131 : vector<16xi8> to vector<32xf4E2M1FN>
                %184 = vector.bitcast %132 : vector<16xi8> to vector<32xf4E2M1FN>
                %185 = vector.bitcast %86 : vector<1xi8> to vector<1xf8E8M0FNU>
                %186 = vector.bitcast %89 : vector<1xi8> to vector<1xf8E8M0FNU>
                %187 = vector.bitcast %91 : vector<1xi8> to vector<1xf8E8M0FNU>
                %188 = vector.bitcast %92 : vector<1xi8> to vector<1xf8E8M0FNU>
                %189 = vector.bitcast %94 : vector<1xi8> to vector<1xf8E8M0FNU>
                %190 = vector.bitcast %95 : vector<1xi8> to vector<1xf8E8M0FNU>
                %191 = vector.bitcast %97 : vector<1xi8> to vector<1xf8E8M0FNU>
                %192 = vector.bitcast %98 : vector<1xi8> to vector<1xf8E8M0FNU>
                %193 = vector.bitcast %100 : vector<1xi8> to vector<1xf8E8M0FNU>
                %194 = vector.bitcast %101 : vector<1xi8> to vector<1xf8E8M0FNU>
                %195 = vector.bitcast %103 : vector<1xi8> to vector<1xf8E8M0FNU>
                %196 = vector.bitcast %104 : vector<1xi8> to vector<1xf8E8M0FNU>
                %197 = vector.bitcast %106 : vector<1xi8> to vector<1xf8E8M0FNU>
                %198 = vector.bitcast %107 : vector<1xi8> to vector<1xf8E8M0FNU>
                %199 = vector.bitcast %109 : vector<1xi8> to vector<1xf8E8M0FNU>
                %200 = vector.bitcast %110 : vector<1xi8> to vector<1xf8E8M0FNU>
                %201 = vector.extract %161[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %202 = vector.extract %185[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %203 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%202[0] * %169) + %83#0 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %204 = vector.extract %162[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %205 = vector.extract %186[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %206 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%205[0] * %170) + %203 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %207 = vector.extract %187[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %208 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%207[0] * %171) + %83#1 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %209 = vector.extract %188[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %210 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%209[0] * %172) + %208 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %211 = vector.extract %189[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %212 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%211[0] * %173) + %83#2 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %213 = vector.extract %190[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %214 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%213[0] * %174) + %212 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %215 = vector.extract %191[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %216 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%215[0] * %175) + %83#3 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %217 = vector.extract %192[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %218 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%217[0] * %176) + %216 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %219 = vector.extract %193[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %220 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%219[0] * %177) + %83#4 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %221 = vector.extract %194[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %222 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%221[0] * %178) + %220 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %223 = vector.extract %195[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %224 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%223[0] * %179) + %83#5 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %225 = vector.extract %196[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %226 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%225[0] * %180) + %224 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %227 = vector.extract %197[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %228 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%227[0] * %181) + %83#6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %229 = vector.extract %198[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %230 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%229[0] * %182) + %228 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %231 = vector.extract %199[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %232 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%231[0] * %183) + %83#7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %233 = vector.extract %200[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %234 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%233[0] * %184) + %232 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %235 = vector.extract %163[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %236 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%202[0] * %169) + %83#8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %237 = vector.extract %164[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %238 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%205[0] * %170) + %236 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %239 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%207[0] * %171) + %83#9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %240 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%209[0] * %172) + %239 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %241 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%211[0] * %173) + %83#10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %242 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%213[0] * %174) + %241 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %243 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%215[0] * %175) + %83#11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %244 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%217[0] * %176) + %243 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %245 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%219[0] * %177) + %83#12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %246 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%221[0] * %178) + %245 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %247 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%223[0] * %179) + %83#13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %248 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%225[0] * %180) + %247 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %249 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%227[0] * %181) + %83#14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %250 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%229[0] * %182) + %249 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %251 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%231[0] * %183) + %83#15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %252 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%233[0] * %184) + %251 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %253 = vector.extract %165[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %254 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%202[0] * %169) + %83#16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %255 = vector.extract %166[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %256 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%205[0] * %170) + %254 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %257 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%207[0] * %171) + %83#17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %258 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%209[0] * %172) + %257 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %259 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%211[0] * %173) + %83#18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %260 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%213[0] * %174) + %259 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %261 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%215[0] * %175) + %83#19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %262 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%217[0] * %176) + %261 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %263 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%219[0] * %177) + %83#20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %264 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%221[0] * %178) + %263 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %265 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%223[0] * %179) + %83#21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %266 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%225[0] * %180) + %265 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %267 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%227[0] * %181) + %83#22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %268 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%229[0] * %182) + %267 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %269 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%231[0] * %183) + %83#23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %270 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%233[0] * %184) + %269 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %271 = vector.extract %167[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %272 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%202[0] * %169) + %83#24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %273 = vector.extract %168[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %274 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%205[0] * %170) + %272 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %275 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%207[0] * %171) + %83#25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %276 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%209[0] * %172) + %275 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %277 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%211[0] * %173) + %83#26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %278 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%213[0] * %174) + %277 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %279 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%215[0] * %175) + %83#27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %280 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%217[0] * %176) + %279 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %281 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%219[0] * %177) + %83#28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %282 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%221[0] * %178) + %281 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %283 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%223[0] * %179) + %83#29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %284 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%225[0] * %180) + %283 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %285 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%227[0] * %181) + %83#30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %286 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%229[0] * %182) + %285 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %287 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%231[0] * %183) + %83#31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %288 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%233[0] * %184) + %287 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %289 = vector.extract_strided_slice %206 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %290 = affine.apply #map35()[%block_id_x]
                %291 = affine.apply #map35()[%block_id_y]
                %292 = affine.apply #map36()[%thread_id_x]
                %293 = arith.muli %290, %c57344 overflow<nsw> : index
                %294 = arith.muli %292, %c57344 overflow<nsw> : index
                %295 = arith.addi %293, %291 overflow<nsw> : index
                %296 = arith.addi %294, %84 overflow<nsw> : index
                %reinterpret_cast_13 = memref.reinterpret_cast %4 to offset: [%295], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
                %cast_14 = memref.cast %reinterpret_cast_13 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
                %297 = amdgpu.fat_raw_buffer_cast %cast_14 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                vector.store %289, %297[%296] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %298 = vector.extract_strided_slice %206 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %299 = affine.apply #map37()[%thread_id_x]
                %300 = arith.muli %299, %c57344 overflow<nsw> : index
                %301 = arith.addi %300, %84 overflow<nsw> : index
                vector.store %298, %297[%301] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %302 = vector.extract_strided_slice %206 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %303 = affine.apply #map38()[%thread_id_x]
                %304 = arith.muli %303, %c57344 overflow<nsw> : index
                %305 = arith.addi %304, %84 overflow<nsw> : index
                vector.store %302, %297[%305] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %306 = vector.extract_strided_slice %206 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %307 = affine.apply #map39()[%thread_id_x]
                %308 = arith.muli %307, %c57344 overflow<nsw> : index
                %309 = arith.addi %308, %84 overflow<nsw> : index
                vector.store %306, %297[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %310 = vector.extract_strided_slice %210 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %311 = arith.addi %294, %90 overflow<nsw> : index
                vector.store %310, %297[%311] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %312 = vector.extract_strided_slice %210 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %313 = arith.addi %300, %90 overflow<nsw> : index
                vector.store %312, %297[%313] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %314 = vector.extract_strided_slice %210 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %315 = arith.addi %304, %90 overflow<nsw> : index
                vector.store %314, %297[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %316 = vector.extract_strided_slice %210 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %317 = arith.addi %308, %90 overflow<nsw> : index
                vector.store %316, %297[%317] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %318 = vector.extract_strided_slice %214 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %319 = arith.addi %294, %93 overflow<nsw> : index
                vector.store %318, %297[%319] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %320 = vector.extract_strided_slice %214 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %321 = arith.addi %300, %93 overflow<nsw> : index
                vector.store %320, %297[%321] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %322 = vector.extract_strided_slice %214 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %323 = arith.addi %304, %93 overflow<nsw> : index
                vector.store %322, %297[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %324 = vector.extract_strided_slice %214 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %325 = arith.addi %308, %93 overflow<nsw> : index
                vector.store %324, %297[%325] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %326 = vector.extract_strided_slice %218 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %327 = arith.addi %294, %96 overflow<nsw> : index
                vector.store %326, %297[%327] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %328 = vector.extract_strided_slice %218 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %329 = arith.addi %300, %96 overflow<nsw> : index
                vector.store %328, %297[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %330 = vector.extract_strided_slice %218 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %331 = arith.addi %304, %96 overflow<nsw> : index
                vector.store %330, %297[%331] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %332 = vector.extract_strided_slice %218 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %333 = arith.addi %308, %96 overflow<nsw> : index
                vector.store %332, %297[%333] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %334 = vector.extract_strided_slice %222 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %335 = arith.addi %294, %99 overflow<nsw> : index
                vector.store %334, %297[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %336 = vector.extract_strided_slice %222 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %337 = arith.addi %300, %99 overflow<nsw> : index
                vector.store %336, %297[%337] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %338 = vector.extract_strided_slice %222 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %339 = arith.addi %304, %99 overflow<nsw> : index
                vector.store %338, %297[%339] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %340 = vector.extract_strided_slice %222 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %341 = arith.addi %308, %99 overflow<nsw> : index
                vector.store %340, %297[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %342 = vector.extract_strided_slice %226 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %343 = arith.addi %294, %102 overflow<nsw> : index
                vector.store %342, %297[%343] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %344 = vector.extract_strided_slice %226 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %345 = arith.addi %300, %102 overflow<nsw> : index
                vector.store %344, %297[%345] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %346 = vector.extract_strided_slice %226 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %347 = arith.addi %304, %102 overflow<nsw> : index
                vector.store %346, %297[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %348 = vector.extract_strided_slice %226 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %349 = arith.addi %308, %102 overflow<nsw> : index
                vector.store %348, %297[%349] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %350 = vector.extract_strided_slice %230 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %351 = arith.addi %294, %105 overflow<nsw> : index
                vector.store %350, %297[%351] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %352 = vector.extract_strided_slice %230 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %353 = arith.addi %300, %105 overflow<nsw> : index
                vector.store %352, %297[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %354 = vector.extract_strided_slice %230 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %355 = arith.addi %304, %105 overflow<nsw> : index
                vector.store %354, %297[%355] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %356 = vector.extract_strided_slice %230 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %357 = arith.addi %308, %105 overflow<nsw> : index
                vector.store %356, %297[%357] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %358 = vector.extract_strided_slice %234 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %359 = arith.addi %294, %108 overflow<nsw> : index
                vector.store %358, %297[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %360 = vector.extract_strided_slice %234 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %361 = arith.addi %300, %108 overflow<nsw> : index
                vector.store %360, %297[%361] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %362 = vector.extract_strided_slice %234 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %363 = arith.addi %304, %108 overflow<nsw> : index
                vector.store %362, %297[%363] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %364 = vector.extract_strided_slice %234 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %365 = arith.addi %308, %108 overflow<nsw> : index
                vector.store %364, %297[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %366 = vector.extract_strided_slice %238 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %367 = affine.apply #map40()[%thread_id_x]
                %368 = arith.muli %367, %c57344 overflow<nsw> : index
                %369 = arith.addi %368, %84 overflow<nsw> : index
                vector.store %366, %297[%369] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %370 = vector.extract_strided_slice %238 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %371 = affine.apply #map41()[%thread_id_x]
                %372 = arith.muli %371, %c57344 overflow<nsw> : index
                %373 = arith.addi %372, %84 overflow<nsw> : index
                vector.store %370, %297[%373] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %374 = vector.extract_strided_slice %238 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %375 = affine.apply #map42()[%thread_id_x]
                %376 = arith.muli %375, %c57344 overflow<nsw> : index
                %377 = arith.addi %376, %84 overflow<nsw> : index
                vector.store %374, %297[%377] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %378 = vector.extract_strided_slice %238 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %379 = affine.apply #map43()[%thread_id_x]
                %380 = arith.muli %379, %c57344 overflow<nsw> : index
                %381 = arith.addi %380, %84 overflow<nsw> : index
                vector.store %378, %297[%381] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %382 = vector.extract_strided_slice %240 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %383 = arith.addi %368, %90 overflow<nsw> : index
                vector.store %382, %297[%383] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %384 = vector.extract_strided_slice %240 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %385 = arith.addi %372, %90 overflow<nsw> : index
                vector.store %384, %297[%385] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %386 = vector.extract_strided_slice %240 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %387 = arith.addi %376, %90 overflow<nsw> : index
                vector.store %386, %297[%387] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %388 = vector.extract_strided_slice %240 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %389 = arith.addi %380, %90 overflow<nsw> : index
                vector.store %388, %297[%389] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %390 = vector.extract_strided_slice %242 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %391 = arith.addi %368, %93 overflow<nsw> : index
                vector.store %390, %297[%391] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %392 = vector.extract_strided_slice %242 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %393 = arith.addi %372, %93 overflow<nsw> : index
                vector.store %392, %297[%393] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %394 = vector.extract_strided_slice %242 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %395 = arith.addi %376, %93 overflow<nsw> : index
                vector.store %394, %297[%395] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %396 = vector.extract_strided_slice %242 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %397 = arith.addi %380, %93 overflow<nsw> : index
                vector.store %396, %297[%397] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %398 = vector.extract_strided_slice %244 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %399 = arith.addi %368, %96 overflow<nsw> : index
                vector.store %398, %297[%399] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %400 = vector.extract_strided_slice %244 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %401 = arith.addi %372, %96 overflow<nsw> : index
                vector.store %400, %297[%401] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %402 = vector.extract_strided_slice %244 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %403 = arith.addi %376, %96 overflow<nsw> : index
                vector.store %402, %297[%403] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %404 = vector.extract_strided_slice %244 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %405 = arith.addi %380, %96 overflow<nsw> : index
                vector.store %404, %297[%405] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %406 = vector.extract_strided_slice %246 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %407 = arith.addi %368, %99 overflow<nsw> : index
                vector.store %406, %297[%407] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %408 = vector.extract_strided_slice %246 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %409 = arith.addi %372, %99 overflow<nsw> : index
                vector.store %408, %297[%409] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %410 = vector.extract_strided_slice %246 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %411 = arith.addi %376, %99 overflow<nsw> : index
                vector.store %410, %297[%411] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %412 = vector.extract_strided_slice %246 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %413 = arith.addi %380, %99 overflow<nsw> : index
                vector.store %412, %297[%413] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %414 = vector.extract_strided_slice %248 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %415 = arith.addi %368, %102 overflow<nsw> : index
                vector.store %414, %297[%415] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %416 = vector.extract_strided_slice %248 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %417 = arith.addi %372, %102 overflow<nsw> : index
                vector.store %416, %297[%417] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %418 = vector.extract_strided_slice %248 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %419 = arith.addi %376, %102 overflow<nsw> : index
                vector.store %418, %297[%419] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %420 = vector.extract_strided_slice %248 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %421 = arith.addi %380, %102 overflow<nsw> : index
                vector.store %420, %297[%421] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %422 = vector.extract_strided_slice %250 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %423 = arith.addi %368, %105 overflow<nsw> : index
                vector.store %422, %297[%423] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %424 = vector.extract_strided_slice %250 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %425 = arith.addi %372, %105 overflow<nsw> : index
                vector.store %424, %297[%425] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %426 = vector.extract_strided_slice %250 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %427 = arith.addi %376, %105 overflow<nsw> : index
                vector.store %426, %297[%427] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %428 = vector.extract_strided_slice %250 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %429 = arith.addi %380, %105 overflow<nsw> : index
                vector.store %428, %297[%429] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %430 = vector.extract_strided_slice %252 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %431 = arith.addi %368, %108 overflow<nsw> : index
                vector.store %430, %297[%431] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %432 = vector.extract_strided_slice %252 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %433 = arith.addi %372, %108 overflow<nsw> : index
                vector.store %432, %297[%433] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %434 = vector.extract_strided_slice %252 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %435 = arith.addi %376, %108 overflow<nsw> : index
                vector.store %434, %297[%435] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %436 = vector.extract_strided_slice %252 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %437 = arith.addi %380, %108 overflow<nsw> : index
                vector.store %436, %297[%437] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %438 = vector.extract_strided_slice %256 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %439 = affine.apply #map44()[%thread_id_x]
                %440 = arith.muli %439, %c57344 overflow<nsw> : index
                %441 = arith.addi %440, %84 overflow<nsw> : index
                vector.store %438, %297[%441] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %442 = vector.extract_strided_slice %256 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %443 = affine.apply #map45()[%thread_id_x]
                %444 = arith.muli %443, %c57344 overflow<nsw> : index
                %445 = arith.addi %444, %84 overflow<nsw> : index
                vector.store %442, %297[%445] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %446 = vector.extract_strided_slice %256 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %447 = affine.apply #map46()[%thread_id_x]
                %448 = arith.muli %447, %c57344 overflow<nsw> : index
                %449 = arith.addi %448, %84 overflow<nsw> : index
                vector.store %446, %297[%449] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %450 = vector.extract_strided_slice %256 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %451 = affine.apply #map47()[%thread_id_x]
                %452 = arith.muli %451, %c57344 overflow<nsw> : index
                %453 = arith.addi %452, %84 overflow<nsw> : index
                vector.store %450, %297[%453] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %454 = vector.extract_strided_slice %258 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %455 = arith.addi %440, %90 overflow<nsw> : index
                vector.store %454, %297[%455] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %456 = vector.extract_strided_slice %258 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %457 = arith.addi %444, %90 overflow<nsw> : index
                vector.store %456, %297[%457] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %458 = vector.extract_strided_slice %258 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %459 = arith.addi %448, %90 overflow<nsw> : index
                vector.store %458, %297[%459] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %460 = vector.extract_strided_slice %258 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %461 = arith.addi %452, %90 overflow<nsw> : index
                vector.store %460, %297[%461] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %462 = vector.extract_strided_slice %260 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %463 = arith.addi %440, %93 overflow<nsw> : index
                vector.store %462, %297[%463] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %464 = vector.extract_strided_slice %260 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %465 = arith.addi %444, %93 overflow<nsw> : index
                vector.store %464, %297[%465] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %466 = vector.extract_strided_slice %260 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %467 = arith.addi %448, %93 overflow<nsw> : index
                vector.store %466, %297[%467] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %468 = vector.extract_strided_slice %260 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %469 = arith.addi %452, %93 overflow<nsw> : index
                vector.store %468, %297[%469] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %470 = vector.extract_strided_slice %262 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %471 = arith.addi %440, %96 overflow<nsw> : index
                vector.store %470, %297[%471] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %472 = vector.extract_strided_slice %262 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %473 = arith.addi %444, %96 overflow<nsw> : index
                vector.store %472, %297[%473] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %474 = vector.extract_strided_slice %262 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %475 = arith.addi %448, %96 overflow<nsw> : index
                vector.store %474, %297[%475] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %476 = vector.extract_strided_slice %262 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %477 = arith.addi %452, %96 overflow<nsw> : index
                vector.store %476, %297[%477] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %478 = vector.extract_strided_slice %264 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %479 = arith.addi %440, %99 overflow<nsw> : index
                vector.store %478, %297[%479] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %480 = vector.extract_strided_slice %264 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %481 = arith.addi %444, %99 overflow<nsw> : index
                vector.store %480, %297[%481] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %482 = vector.extract_strided_slice %264 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %483 = arith.addi %448, %99 overflow<nsw> : index
                vector.store %482, %297[%483] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %484 = vector.extract_strided_slice %264 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %485 = arith.addi %452, %99 overflow<nsw> : index
                vector.store %484, %297[%485] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %486 = vector.extract_strided_slice %266 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %487 = arith.addi %440, %102 overflow<nsw> : index
                vector.store %486, %297[%487] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %488 = vector.extract_strided_slice %266 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %489 = arith.addi %444, %102 overflow<nsw> : index
                vector.store %488, %297[%489] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %490 = vector.extract_strided_slice %266 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %491 = arith.addi %448, %102 overflow<nsw> : index
                vector.store %490, %297[%491] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %492 = vector.extract_strided_slice %266 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %493 = arith.addi %452, %102 overflow<nsw> : index
                vector.store %492, %297[%493] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %494 = vector.extract_strided_slice %268 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %495 = arith.addi %440, %105 overflow<nsw> : index
                vector.store %494, %297[%495] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %496 = vector.extract_strided_slice %268 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %497 = arith.addi %444, %105 overflow<nsw> : index
                vector.store %496, %297[%497] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %498 = vector.extract_strided_slice %268 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %499 = arith.addi %448, %105 overflow<nsw> : index
                vector.store %498, %297[%499] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %500 = vector.extract_strided_slice %268 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %501 = arith.addi %452, %105 overflow<nsw> : index
                vector.store %500, %297[%501] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %502 = vector.extract_strided_slice %270 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %503 = arith.addi %440, %108 overflow<nsw> : index
                vector.store %502, %297[%503] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %504 = vector.extract_strided_slice %270 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %505 = arith.addi %444, %108 overflow<nsw> : index
                vector.store %504, %297[%505] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %506 = vector.extract_strided_slice %270 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %507 = arith.addi %448, %108 overflow<nsw> : index
                vector.store %506, %297[%507] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %508 = vector.extract_strided_slice %270 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %509 = arith.addi %452, %108 overflow<nsw> : index
                vector.store %508, %297[%509] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %510 = vector.extract_strided_slice %274 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %511 = affine.apply #map48()[%thread_id_x]
                %512 = arith.muli %511, %c57344 overflow<nsw> : index
                %513 = arith.addi %512, %84 overflow<nsw> : index
                vector.store %510, %297[%513] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %514 = vector.extract_strided_slice %274 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %515 = affine.apply #map49()[%thread_id_x]
                %516 = arith.muli %515, %c57344 overflow<nsw> : index
                %517 = arith.addi %516, %84 overflow<nsw> : index
                vector.store %514, %297[%517] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %518 = vector.extract_strided_slice %274 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %519 = affine.apply #map50()[%thread_id_x]
                %520 = arith.muli %519, %c57344 overflow<nsw> : index
                %521 = arith.addi %520, %84 overflow<nsw> : index
                vector.store %518, %297[%521] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %522 = vector.extract_strided_slice %274 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %523 = affine.apply #map51()[%thread_id_x]
                %524 = arith.muli %523, %c57344 overflow<nsw> : index
                %525 = arith.addi %524, %84 overflow<nsw> : index
                vector.store %522, %297[%525] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %526 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %527 = arith.addi %512, %90 overflow<nsw> : index
                vector.store %526, %297[%527] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %528 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %529 = arith.addi %516, %90 overflow<nsw> : index
                vector.store %528, %297[%529] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %530 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %531 = arith.addi %520, %90 overflow<nsw> : index
                vector.store %530, %297[%531] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %532 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %533 = arith.addi %524, %90 overflow<nsw> : index
                vector.store %532, %297[%533] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %534 = vector.extract_strided_slice %278 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %535 = arith.addi %512, %93 overflow<nsw> : index
                vector.store %534, %297[%535] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %536 = vector.extract_strided_slice %278 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %537 = arith.addi %516, %93 overflow<nsw> : index
                vector.store %536, %297[%537] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %538 = vector.extract_strided_slice %278 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %539 = arith.addi %520, %93 overflow<nsw> : index
                vector.store %538, %297[%539] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %540 = vector.extract_strided_slice %278 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %541 = arith.addi %524, %93 overflow<nsw> : index
                vector.store %540, %297[%541] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %542 = vector.extract_strided_slice %280 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %543 = arith.addi %512, %96 overflow<nsw> : index
                vector.store %542, %297[%543] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %544 = vector.extract_strided_slice %280 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %545 = arith.addi %516, %96 overflow<nsw> : index
                vector.store %544, %297[%545] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %546 = vector.extract_strided_slice %280 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %547 = arith.addi %520, %96 overflow<nsw> : index
                vector.store %546, %297[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %548 = vector.extract_strided_slice %280 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %549 = arith.addi %524, %96 overflow<nsw> : index
                vector.store %548, %297[%549] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %550 = vector.extract_strided_slice %282 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %551 = arith.addi %512, %99 overflow<nsw> : index
                vector.store %550, %297[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %552 = vector.extract_strided_slice %282 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %553 = arith.addi %516, %99 overflow<nsw> : index
                vector.store %552, %297[%553] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %554 = vector.extract_strided_slice %282 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %555 = arith.addi %520, %99 overflow<nsw> : index
                vector.store %554, %297[%555] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %556 = vector.extract_strided_slice %282 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %557 = arith.addi %524, %99 overflow<nsw> : index
                vector.store %556, %297[%557] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %558 = vector.extract_strided_slice %284 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %559 = arith.addi %512, %102 overflow<nsw> : index
                vector.store %558, %297[%559] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %560 = vector.extract_strided_slice %284 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %561 = arith.addi %516, %102 overflow<nsw> : index
                vector.store %560, %297[%561] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %562 = vector.extract_strided_slice %284 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %563 = arith.addi %520, %102 overflow<nsw> : index
                vector.store %562, %297[%563] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %564 = vector.extract_strided_slice %284 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %565 = arith.addi %524, %102 overflow<nsw> : index
                vector.store %564, %297[%565] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %566 = vector.extract_strided_slice %286 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %567 = arith.addi %512, %105 overflow<nsw> : index
                vector.store %566, %297[%567] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %568 = vector.extract_strided_slice %286 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %569 = arith.addi %516, %105 overflow<nsw> : index
                vector.store %568, %297[%569] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %570 = vector.extract_strided_slice %286 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %571 = arith.addi %520, %105 overflow<nsw> : index
                vector.store %570, %297[%571] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %572 = vector.extract_strided_slice %286 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %573 = arith.addi %524, %105 overflow<nsw> : index
                vector.store %572, %297[%573] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %574 = vector.extract_strided_slice %288 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %575 = arith.addi %512, %108 overflow<nsw> : index
                vector.store %574, %297[%575] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %576 = vector.extract_strided_slice %288 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %577 = arith.addi %516, %108 overflow<nsw> : index
                vector.store %576, %297[%577] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %578 = vector.extract_strided_slice %288 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %579 = arith.addi %520, %108 overflow<nsw> : index
                vector.store %578, %297[%579] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %580 = vector.extract_strided_slice %288 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %581 = arith.addi %524, %108 overflow<nsw> : index
                vector.store %580, %297[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<4096x8192xi8>
            %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<4096x512xi8>
            %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<57344x8192xi8>
            %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<57344x512xi8>
            %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<4096x57344xf32>
            %5 = flow.dispatch @gemm::@gemm(%0, %1, %2, %3, %4) : (tensor<4096x8192xi8>, tensor<4096x512xi8>, tensor<57344x8192xi8>, tensor<57344x512xi8>, tensor<4096x57344xf32>) -> %4
            %6 = hal.tensor.barrier join(%5 : tensor<4096x57344xf32>) => %arg6 : !hal.fence
            %7 = hal.tensor.export %6 : tensor<4096x57344xf32> -> !hal.buffer_view
            return %7 : !hal.buffer_view
        }
        }
    """

    mlir_claude = """
        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 16)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
        #map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
        #map9 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
        #map11 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
        #map12 = affine_map<()[s0] -> ((s0 floordiv 2) mod 2)>
        #map13 = affine_map<()[s0] -> (s0 mod 2)>
        #map14 = affine_map<()[s0] -> (s0 * 4)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 32 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 256)>
        #map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
        #map18 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
        #map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
        #map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
        #map22 = affine_map<()[s0] -> (s0 * 4 + (s0 mod 64) floordiv 16 - (s0 floordiv 2) * 8)>
        #map23 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
        #map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
        #map25 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
        #map26 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
        #map27 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
        #map28 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
        #map29 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
        #map30 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
        #map31 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map32 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
        #map33 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 128)>
        #map34 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 + 8)>
        #map35 = affine_map<()[s0] -> (s0 * 256)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
        #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
        #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
        #map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
        #map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
        #map47 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
        #map48 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
        #map49 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
        #map50 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
        #map51 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c16 = arith.constant 16 : index
            %c224 = arith.constant 224 : index
            %c1 = arith.constant 1 : index
            stream.return %c16, %c224, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c512_i14 = arith.constant 512 : i14
                %c-8192_i14 = arith.constant -8192 : i14
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c57344 = arith.constant 57344 : index
                %c63 = arith.constant 63 : index
                %c512 = arith.constant 512 : index
                %c2147483646_i64 = arith.constant 2147483646 : i64
                %c8192 = arith.constant 8192 : index
                %c1 = arith.constant 1 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
                %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
                %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 16
                %block_id_y = gpu.block_id  y upper_bound 224
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %alloc = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_0 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_3 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_4 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_5 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_6 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %c32_idx = arith.constant 32 : index
                %c128_idx = arith.constant 128 : index
                %c262144 = arith.constant 262144 : index
                %c65536 = arith.constant 65536 : index
                %is_cluster0 = arith.cmpi eq, %thread_id_y, %c0 : index
                %5 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %6 = affine.apply #map1()[%thread_id_x]
                %7 = affine.apply #map2()[%thread_id_x]
                %8 = arith.xori %7, %6 : index
                %9 = affine.apply #map3()[%8]
                %10 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %11 = gpu.subgroup_broadcast %10,  first_active_lane : index
                %12 = gpu.subgroup_broadcast %c0,  first_active_lane : index
                %13 = arith.muli %5, %c8192 overflow<nsw> : index
                %14 = arith.addi %13, %9 overflow<nsw> : index
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast = memref.cast %reinterpret_cast : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %15 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                // --- Address computations (all waves) ---
                %16 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
                %17 = affine.apply #map6()[%thread_id_x, %thread_id_y]
                %18 = gpu.subgroup_broadcast %17,  first_active_lane : index
                %19 = arith.muli %16, %c8192 overflow<nsw> : index
                %20 = arith.addi %19, %9 overflow<nsw> : index
                %21 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
                %22 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                %23 = gpu.subgroup_broadcast %22,  first_active_lane : index
                %24 = arith.muli %21, %c8192 overflow<nsw> : index
                %25 = arith.addi %24, %9 overflow<nsw> : index
                %26 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
                %27 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %28 = gpu.subgroup_broadcast %27,  first_active_lane : index
                %29 = arith.muli %26, %c8192 overflow<nsw> : index
                %30 = arith.addi %29, %9 overflow<nsw> : index
                %31 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_x]
                %32 = affine.apply #map12()[%thread_id_x]
                %33 = affine.apply #map13()[%thread_id_x]
                %34 = arith.xori %33, %32 : index
                %35 = affine.apply #map14()[%34]
                %36 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %37 = gpu.subgroup_broadcast %36,  first_active_lane : index
                %38 = arith.muli %31, %c512 overflow<nsw> : index
                %39 = arith.addi %38, %35 overflow<nsw> : index
                %reinterpret_cast_7 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %40 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                %41 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
                %42 = arith.muli %41, %c8192 overflow<nsw> : index
                %43 = arith.addi %42, %9 overflow<nsw> : index
                %reinterpret_cast_9 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_10 = memref.cast %reinterpret_cast_9 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %44 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                %45 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                %46 = arith.muli %45, %c8192 overflow<nsw> : index
                %47 = arith.addi %46, %9 overflow<nsw> : index
                %48 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
                %49 = arith.muli %48, %c8192 overflow<nsw> : index
                %50 = arith.addi %49, %9 overflow<nsw> : index
                %51 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_y]
                %52 = arith.muli %51, %c8192 overflow<nsw> : index
                %53 = arith.addi %52, %9 overflow<nsw> : index
                %54 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_y]
                %55 = arith.muli %54, %c512 overflow<nsw> : index
                %56 = arith.addi %55, %35 overflow<nsw> : index
                %reinterpret_cast_11 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_12 = memref.cast %reinterpret_cast_11 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %57 = amdgpu.fat_raw_buffer_cast %cast_12 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                // --- Cluster 0 only: A data (8), A scale (2), B data (8) gathers ---
                scf.if %is_cluster0 {
                // A data: 4 original gathers (ty=0 addresses)
                amdgpu.gather_to_lds %15[%14], %alloc_6[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %15[%20], %alloc_6[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %15[%25], %alloc_6[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %15[%30], %alloc_6[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                // A data: 4 extra gathers (ty=1 addresses: global +262144, LDS row +32)
                %ea_g0 = arith.addi %14, %c262144 overflow<nsw> : index
                %ea_l0 = arith.addi %11, %c32_idx overflow<nsw> : index
                amdgpu.gather_to_lds %15[%ea_g0], %alloc_6[%ea_l0, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %ea_g1 = arith.addi %20, %c262144 overflow<nsw> : index
                %ea_l1 = arith.addi %18, %c32_idx overflow<nsw> : index
                amdgpu.gather_to_lds %15[%ea_g1], %alloc_6[%ea_l1, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %ea_g2 = arith.addi %25, %c262144 overflow<nsw> : index
                %ea_l2 = arith.addi %23, %c32_idx overflow<nsw> : index
                amdgpu.gather_to_lds %15[%ea_g2], %alloc_6[%ea_l2, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %ea_g3 = arith.addi %30, %c262144 overflow<nsw> : index
                %ea_l3 = arith.addi %28, %c32_idx overflow<nsw> : index
                amdgpu.gather_to_lds %15[%ea_g3], %alloc_6[%ea_l3, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                // A scale: 1 original gather (ty=0)
                amdgpu.gather_to_lds %40[%39], %alloc_4[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                // A scale: 1 extra gather (ty=1: global +65536, LDS row +128)
                %eas_g0 = arith.addi %39, %c65536 overflow<nsw> : index
                %eas_l0 = arith.addi %37, %c128_idx overflow<nsw> : index
                amdgpu.gather_to_lds %40[%eas_g0], %alloc_4[%eas_l0, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                // B data: 4 original gathers (ty=0 addresses)
                amdgpu.gather_to_lds %44[%43], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %44[%47], %alloc_2[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %44[%50], %alloc_2[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                amdgpu.gather_to_lds %44[%53], %alloc_2[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                // B data: 4 extra gathers (ty=1: global +262144, LDS row +32)
                %eb_g0 = arith.addi %43, %c262144 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%eb_g0], %alloc_2[%ea_l0, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %eb_g1 = arith.addi %47, %c262144 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%eb_g1], %alloc_2[%ea_l1, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %eb_g2 = arith.addi %50, %c262144 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%eb_g2], %alloc_2[%ea_l2, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %eb_g3 = arith.addi %53, %c262144 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%eb_g3], %alloc_2[%ea_l3, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                }
                // B scale: unchanged (both clusters, already cluster-aligned)
                amdgpu.gather_to_lds %57[%56], %alloc_0[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.s.barrier
                %58 = affine.apply #map16()[%thread_id_x, %thread_id_y]
                %59 = arith.index_cast %58 : index to i32
                %60 = arith.cmpi sge, %59, %c4_i32 : i32
                %61 = arith.cmpi slt, %59, %c4_i32 : i32
                scf.if %60 {
                rocdl.s.barrier
                }
                %62 = affine.apply #map17()[%thread_id_x]
                %63 = affine.apply #map18()[%thread_id_x]
                %64 = arith.xori %63, %7 : index
                %65 = affine.apply #map3()[%64]
                %66 = affine.apply #map19()[%thread_id_x]
                %67 = affine.apply #map20()[%thread_id_x]
                %68 = affine.apply #map21()[%thread_id_x]
                %69 = affine.apply #map22()[%thread_id_x]
                %70 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %71 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %72 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %73 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %74 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %75 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %76 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %77 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %78 = affine.apply #map31()[%thread_id_x]
                %79 = arith.xori %78, %7 : index
                %80 = affine.apply #map3()[%79]
                %81 = arith.xori %33, %c1 : index
                %82 = affine.apply #map32()[%thread_id_x, %81]
                %83:40 = scf.for %arg5 = %c0 to %c63 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %alloc_6, %arg39 = %alloc_5, %arg40 = %alloc_4, %arg41 = %alloc_3, %arg42 = %alloc_2, %arg43 = %alloc_1, %arg44 = %alloc_0, %arg45 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>) {
                rocdl.sched.barrier 0
                rocdl.s.barrier
                // --- Address computations (all waves) ---
                %582 = affine.apply #map33()[%arg5, %8]
                %583 = arith.addi %13, %582 overflow<nsw> : index
                %584 = arith.addi %19, %582 overflow<nsw> : index
                %585 = arith.addi %24, %582 overflow<nsw> : index
                %586 = arith.addi %29, %582 overflow<nsw> : index
                %587 = affine.apply #map34()[%arg5, %34]
                %588 = arith.addi %38, %587 overflow<nsw> : index
                %589 = arith.addi %42, %582 overflow<nsw> : index
                %590 = arith.addi %46, %582 overflow<nsw> : index
                %591 = arith.addi %49, %582 overflow<nsw> : index
                %592 = arith.addi %52, %582 overflow<nsw> : index
                %593 = arith.addi %55, %587 overflow<nsw> : index
                // --- Cluster 0 only: A data (8), A scale (2), B data (8) gathers ---
                scf.if %is_cluster0 {
                    // A data: 4 original gathers (ty=0 addresses)
                    amdgpu.gather_to_lds %15[%583], %arg39[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %15[%584], %arg39[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %15[%585], %arg39[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %15[%586], %arg39[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    // A data: 4 extra gathers (ty=1: global +262144, LDS row +32)
                    %lea_g0 = arith.addi %583, %c262144 overflow<nsw> : index
                    %lea_l0 = arith.addi %11, %c32_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %15[%lea_g0], %arg39[%lea_l0, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %lea_g1 = arith.addi %584, %c262144 overflow<nsw> : index
                    %lea_l1 = arith.addi %18, %c32_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %15[%lea_g1], %arg39[%lea_l1, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %lea_g2 = arith.addi %585, %c262144 overflow<nsw> : index
                    %lea_l2 = arith.addi %23, %c32_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %15[%lea_g2], %arg39[%lea_l2, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %lea_g3 = arith.addi %586, %c262144 overflow<nsw> : index
                    %lea_l3 = arith.addi %28, %c32_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %15[%lea_g3], %arg39[%lea_l3, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    // A scale: 1 original gather (ty=0)
                    amdgpu.gather_to_lds %40[%588], %arg41[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                    // A scale: 1 extra gather (ty=1: global +65536, LDS row +128)
                    %leas_g0 = arith.addi %588, %c65536 overflow<nsw> : index
                    %leas_l0 = arith.addi %37, %c128_idx overflow<nsw> : index
                    amdgpu.gather_to_lds %40[%leas_g0], %arg41[%leas_l0, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                    // B data: 4 original gathers (ty=0 addresses)
                    amdgpu.gather_to_lds %44[%589], %arg43[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %44[%590], %arg43[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %44[%591], %arg43[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    amdgpu.gather_to_lds %44[%592], %arg43[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    // B data: 4 extra gathers (ty=1: global +262144, LDS row +32)
                    %leb_g0 = arith.addi %589, %c262144 overflow<nsw> : index
                    amdgpu.gather_to_lds %44[%leb_g0], %arg43[%lea_l0, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %leb_g1 = arith.addi %590, %c262144 overflow<nsw> : index
                    amdgpu.gather_to_lds %44[%leb_g1], %arg43[%lea_l1, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %leb_g2 = arith.addi %591, %c262144 overflow<nsw> : index
                    amdgpu.gather_to_lds %44[%leb_g2], %arg43[%lea_l2, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                    %leb_g3 = arith.addi %592, %c262144 overflow<nsw> : index
                    amdgpu.gather_to_lds %44[%leb_g3], %arg43[%lea_l3, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                }
                // B scale: unchanged (both clusters)
                amdgpu.gather_to_lds %57[%593], %arg45[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(10)
                %594 = vector.load %arg38[%62, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %595 = vector.load %arg38[%66, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %596 = vector.load %arg38[%67, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %597 = vector.load %arg38[%68, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %598 = vector.load %arg40[%62, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %599 = vector.load %arg40[%66, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %600 = vector.load %arg40[%67, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %601 = vector.load %arg40[%68, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %602 = vector.load %arg42[%70, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %603 = vector.load %arg42[%71, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %604 = vector.load %arg42[%72, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %605 = vector.load %arg42[%73, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %606 = vector.load %arg42[%74, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %607 = vector.load %arg42[%75, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %608 = vector.load %arg42[%76, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %609 = vector.load %arg42[%77, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %610 = vector.load %arg44[%70, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %611 = vector.load %arg44[%71, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %612 = vector.load %arg44[%72, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %613 = vector.load %arg44[%73, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %614 = vector.load %arg44[%74, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %615 = vector.load %arg44[%75, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %616 = vector.load %arg44[%76, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %617 = vector.load %arg44[%77, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %618 = vector.bitcast %594 : vector<16xi8> to vector<32xf4E2M1FN>
                %619 = vector.bitcast %595 : vector<16xi8> to vector<32xf4E2M1FN>
                %620 = vector.bitcast %596 : vector<16xi8> to vector<32xf4E2M1FN>
                %621 = vector.bitcast %597 : vector<16xi8> to vector<32xf4E2M1FN>
                %622 = vector.bitcast %598 : vector<1xi8> to vector<1xf8E8M0FNU>
                %623 = vector.bitcast %599 : vector<1xi8> to vector<1xf8E8M0FNU>
                %624 = vector.bitcast %600 : vector<1xi8> to vector<1xf8E8M0FNU>
                %625 = vector.bitcast %601 : vector<1xi8> to vector<1xf8E8M0FNU>
                %626 = vector.bitcast %602 : vector<16xi8> to vector<32xf4E2M1FN>
                %627 = vector.bitcast %603 : vector<16xi8> to vector<32xf4E2M1FN>
                %628 = vector.bitcast %604 : vector<16xi8> to vector<32xf4E2M1FN>
                %629 = vector.bitcast %605 : vector<16xi8> to vector<32xf4E2M1FN>
                %630 = vector.bitcast %606 : vector<16xi8> to vector<32xf4E2M1FN>
                %631 = vector.bitcast %607 : vector<16xi8> to vector<32xf4E2M1FN>
                %632 = vector.bitcast %608 : vector<16xi8> to vector<32xf4E2M1FN>
                %633 = vector.bitcast %609 : vector<16xi8> to vector<32xf4E2M1FN>
                %634 = vector.bitcast %610 : vector<1xi8> to vector<1xf8E8M0FNU>
                %635 = vector.bitcast %611 : vector<1xi8> to vector<1xf8E8M0FNU>
                %636 = vector.bitcast %612 : vector<1xi8> to vector<1xf8E8M0FNU>
                %637 = vector.bitcast %613 : vector<1xi8> to vector<1xf8E8M0FNU>
                %638 = vector.bitcast %614 : vector<1xi8> to vector<1xf8E8M0FNU>
                %639 = vector.bitcast %615 : vector<1xi8> to vector<1xf8E8M0FNU>
                %640 = vector.bitcast %616 : vector<1xi8> to vector<1xf8E8M0FNU>
                %641 = vector.bitcast %617 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %642 = vector.extract %622[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %643 = vector.extract %634[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %644 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%643[0] * %626) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %645 = vector.extract %635[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %646 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%645[0] * %627) + %arg7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %647 = vector.extract %636[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %648 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%647[0] * %628) + %arg8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %649 = vector.extract %637[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %650 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%649[0] * %629) + %arg9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %651 = vector.extract %638[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %652 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%651[0] * %630) + %arg10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %653 = vector.extract %639[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %654 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%653[0] * %631) + %arg11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %655 = vector.extract %640[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %656 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%655[0] * %632) + %arg12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %657 = vector.extract %641[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %658 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%657[0] * %633) + %arg13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %659 = vector.extract %623[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %660 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%643[0] * %626) + %arg14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %661 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%645[0] * %627) + %arg15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %662 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%647[0] * %628) + %arg16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %663 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%649[0] * %629) + %arg17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %664 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%651[0] * %630) + %arg18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %665 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%653[0] * %631) + %arg19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %666 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%655[0] * %632) + %arg20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %667 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%657[0] * %633) + %arg21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %668 = vector.extract %624[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %669 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%643[0] * %626) + %arg22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %670 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%645[0] * %627) + %arg23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %671 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%647[0] * %628) + %arg24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %672 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%649[0] * %629) + %arg25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %673 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%651[0] * %630) + %arg26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %674 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%653[0] * %631) + %arg27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %675 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%655[0] * %632) + %arg28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %676 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%657[0] * %633) + %arg29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %677 = vector.extract %625[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %678 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%643[0] * %626) + %arg30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %679 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%645[0] * %627) + %arg31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %680 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%647[0] * %628) + %arg32 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %681 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%649[0] * %629) + %arg33 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %682 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%651[0] * %630) + %arg34 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %683 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%653[0] * %631) + %arg35 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %684 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%655[0] * %632) + %arg36 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %685 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%657[0] * %633) + %arg37 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.sched.barrier 0
                %686 = vector.load %arg38[%62, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %687 = vector.load %arg38[%66, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %688 = vector.load %arg38[%67, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %689 = vector.load %arg38[%68, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %690 = vector.load %arg40[%62, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %691 = vector.load %arg40[%66, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %692 = vector.load %arg40[%67, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %693 = vector.load %arg40[%68, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %694 = vector.load %arg42[%70, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %695 = vector.load %arg42[%71, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %696 = vector.load %arg42[%72, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %697 = vector.load %arg42[%73, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %698 = vector.load %arg42[%74, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %699 = vector.load %arg42[%75, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %700 = vector.load %arg42[%76, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %701 = vector.load %arg42[%77, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %702 = vector.load %arg44[%70, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %703 = vector.load %arg44[%71, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %704 = vector.load %arg44[%72, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %705 = vector.load %arg44[%73, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %706 = vector.load %arg44[%74, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %707 = vector.load %arg44[%75, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %708 = vector.load %arg44[%76, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %709 = vector.load %arg44[%77, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %710 = vector.bitcast %686 : vector<16xi8> to vector<32xf4E2M1FN>
                %711 = vector.bitcast %687 : vector<16xi8> to vector<32xf4E2M1FN>
                %712 = vector.bitcast %688 : vector<16xi8> to vector<32xf4E2M1FN>
                %713 = vector.bitcast %689 : vector<16xi8> to vector<32xf4E2M1FN>
                %714 = vector.bitcast %690 : vector<1xi8> to vector<1xf8E8M0FNU>
                %715 = vector.bitcast %691 : vector<1xi8> to vector<1xf8E8M0FNU>
                %716 = vector.bitcast %692 : vector<1xi8> to vector<1xf8E8M0FNU>
                %717 = vector.bitcast %693 : vector<1xi8> to vector<1xf8E8M0FNU>
                %718 = vector.bitcast %694 : vector<16xi8> to vector<32xf4E2M1FN>
                %719 = vector.bitcast %695 : vector<16xi8> to vector<32xf4E2M1FN>
                %720 = vector.bitcast %696 : vector<16xi8> to vector<32xf4E2M1FN>
                %721 = vector.bitcast %697 : vector<16xi8> to vector<32xf4E2M1FN>
                %722 = vector.bitcast %698 : vector<16xi8> to vector<32xf4E2M1FN>
                %723 = vector.bitcast %699 : vector<16xi8> to vector<32xf4E2M1FN>
                %724 = vector.bitcast %700 : vector<16xi8> to vector<32xf4E2M1FN>
                %725 = vector.bitcast %701 : vector<16xi8> to vector<32xf4E2M1FN>
                %726 = vector.bitcast %702 : vector<1xi8> to vector<1xf8E8M0FNU>
                %727 = vector.bitcast %703 : vector<1xi8> to vector<1xf8E8M0FNU>
                %728 = vector.bitcast %704 : vector<1xi8> to vector<1xf8E8M0FNU>
                %729 = vector.bitcast %705 : vector<1xi8> to vector<1xf8E8M0FNU>
                %730 = vector.bitcast %706 : vector<1xi8> to vector<1xf8E8M0FNU>
                %731 = vector.bitcast %707 : vector<1xi8> to vector<1xf8E8M0FNU>
                %732 = vector.bitcast %708 : vector<1xi8> to vector<1xf8E8M0FNU>
                %733 = vector.bitcast %709 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %734 = vector.extract %714[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %735 = vector.extract %726[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %736 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%735[0] * %718) + %644 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %737 = vector.extract %727[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %738 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%737[0] * %719) + %646 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %739 = vector.extract %728[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %740 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%739[0] * %720) + %648 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %741 = vector.extract %729[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %742 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%741[0] * %721) + %650 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %743 = vector.extract %730[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %744 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%743[0] * %722) + %652 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %745 = vector.extract %731[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %746 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%745[0] * %723) + %654 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %747 = vector.extract %732[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %748 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%747[0] * %724) + %656 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %749 = vector.extract %733[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %750 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%749[0] * %725) + %658 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %751 = vector.extract %715[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %752 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%735[0] * %718) + %660 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %753 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%737[0] * %719) + %661 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %754 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%739[0] * %720) + %662 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %755 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%741[0] * %721) + %663 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %756 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%743[0] * %722) + %664 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %757 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%745[0] * %723) + %665 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %758 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%747[0] * %724) + %666 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %759 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%749[0] * %725) + %667 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %760 = vector.extract %716[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %761 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%735[0] * %718) + %669 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %762 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%737[0] * %719) + %670 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %763 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%739[0] * %720) + %671 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %764 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%741[0] * %721) + %672 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %765 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%743[0] * %722) + %673 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %766 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%745[0] * %723) + %674 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %767 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%747[0] * %724) + %675 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %768 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%749[0] * %725) + %676 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %769 = vector.extract %717[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %770 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%735[0] * %718) + %678 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %771 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%737[0] * %719) + %679 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %772 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%739[0] * %720) + %680 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %773 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%741[0] * %721) + %681 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %774 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%743[0] * %722) + %682 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %775 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%745[0] * %723) + %683 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %776 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%747[0] * %724) + %684 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %777 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%749[0] * %725) + %685 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                scf.yield %736, %738, %740, %742, %744, %746, %748, %750, %752, %753, %754, %755, %756, %757, %758, %759, %761, %762, %763, %764, %765, %766, %767, %768, %770, %771, %772, %773, %774, %775, %776, %777, %arg39, %arg38, %arg41, %arg40, %arg43, %arg42, %arg45, %arg44 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                }
                scf.if %61 {
                rocdl.s.barrier
                }
                amdgpu.lds_barrier
                %84 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %85 = affine.apply #map22()[%thread_id_x]
                %86 = vector.load %83#38[%84, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %87 = arith.xori %33, %c1 : index
                %88 = affine.apply #map32()[%thread_id_x, %87]
                %89 = vector.load %83#38[%84, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %90 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %91 = vector.load %83#38[%90, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %92 = vector.load %83#38[%90, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %93 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %94 = vector.load %83#38[%93, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %95 = vector.load %83#38[%93, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %96 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %97 = vector.load %83#38[%96, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %98 = vector.load %83#38[%96, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %99 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %100 = vector.load %83#38[%99, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %101 = vector.load %83#38[%99, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %102 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %103 = vector.load %83#38[%102, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %104 = vector.load %83#38[%102, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %105 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %106 = vector.load %83#38[%105, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %107 = vector.load %83#38[%105, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %108 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %109 = vector.load %83#38[%108, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %110 = vector.load %83#38[%108, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %111 = affine.apply #map18()[%thread_id_x]
                %112 = arith.xori %111, %7 : index
                %113 = affine.apply #map3()[%112]
                %114 = vector.load %83#36[%84, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %115 = affine.apply #map31()[%thread_id_x]
                %116 = arith.xori %115, %7 : index
                %117 = affine.apply #map3()[%116]
                %118 = vector.load %83#36[%84, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %119 = vector.load %83#36[%90, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %120 = vector.load %83#36[%90, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %121 = vector.load %83#36[%93, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %122 = vector.load %83#36[%93, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %123 = vector.load %83#36[%96, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %124 = vector.load %83#36[%96, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %125 = vector.load %83#36[%99, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %126 = vector.load %83#36[%99, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %127 = vector.load %83#36[%102, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %128 = vector.load %83#36[%102, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %129 = vector.load %83#36[%105, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %130 = vector.load %83#36[%105, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %131 = vector.load %83#36[%108, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %132 = vector.load %83#36[%108, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %133 = affine.apply #map17()[%thread_id_x]
                %134 = vector.load %83#34[%133, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %135 = vector.load %83#34[%133, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %136 = affine.apply #map19()[%thread_id_x]
                %137 = vector.load %83#34[%136, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %138 = vector.load %83#34[%136, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %139 = affine.apply #map20()[%thread_id_x]
                %140 = vector.load %83#34[%139, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %141 = vector.load %83#34[%139, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %142 = affine.apply #map21()[%thread_id_x]
                %143 = vector.load %83#34[%142, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %144 = vector.load %83#34[%142, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %145 = vector.load %83#32[%133, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %146 = vector.load %83#32[%133, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %147 = vector.load %83#32[%136, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %148 = vector.load %83#32[%136, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %149 = vector.load %83#32[%139, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %150 = vector.load %83#32[%139, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %151 = vector.load %83#32[%142, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %152 = vector.load %83#32[%142, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %153 = vector.bitcast %145 : vector<16xi8> to vector<32xf4E2M1FN>
                %154 = vector.bitcast %146 : vector<16xi8> to vector<32xf4E2M1FN>
                %155 = vector.bitcast %147 : vector<16xi8> to vector<32xf4E2M1FN>
                %156 = vector.bitcast %148 : vector<16xi8> to vector<32xf4E2M1FN>
                %157 = vector.bitcast %149 : vector<16xi8> to vector<32xf4E2M1FN>
                %158 = vector.bitcast %150 : vector<16xi8> to vector<32xf4E2M1FN>
                %159 = vector.bitcast %151 : vector<16xi8> to vector<32xf4E2M1FN>
                %160 = vector.bitcast %152 : vector<16xi8> to vector<32xf4E2M1FN>
                %161 = vector.bitcast %134 : vector<1xi8> to vector<1xf8E8M0FNU>
                %162 = vector.bitcast %135 : vector<1xi8> to vector<1xf8E8M0FNU>
                %163 = vector.bitcast %137 : vector<1xi8> to vector<1xf8E8M0FNU>
                %164 = vector.bitcast %138 : vector<1xi8> to vector<1xf8E8M0FNU>
                %165 = vector.bitcast %140 : vector<1xi8> to vector<1xf8E8M0FNU>
                %166 = vector.bitcast %141 : vector<1xi8> to vector<1xf8E8M0FNU>
                %167 = vector.bitcast %143 : vector<1xi8> to vector<1xf8E8M0FNU>
                %168 = vector.bitcast %144 : vector<1xi8> to vector<1xf8E8M0FNU>
                %169 = vector.bitcast %114 : vector<16xi8> to vector<32xf4E2M1FN>
                %170 = vector.bitcast %118 : vector<16xi8> to vector<32xf4E2M1FN>
                %171 = vector.bitcast %119 : vector<16xi8> to vector<32xf4E2M1FN>
                %172 = vector.bitcast %120 : vector<16xi8> to vector<32xf4E2M1FN>
                %173 = vector.bitcast %121 : vector<16xi8> to vector<32xf4E2M1FN>
                %174 = vector.bitcast %122 : vector<16xi8> to vector<32xf4E2M1FN>
                %175 = vector.bitcast %123 : vector<16xi8> to vector<32xf4E2M1FN>
                %176 = vector.bitcast %124 : vector<16xi8> to vector<32xf4E2M1FN>
                %177 = vector.bitcast %125 : vector<16xi8> to vector<32xf4E2M1FN>
                %178 = vector.bitcast %126 : vector<16xi8> to vector<32xf4E2M1FN>
                %179 = vector.bitcast %127 : vector<16xi8> to vector<32xf4E2M1FN>
                %180 = vector.bitcast %128 : vector<16xi8> to vector<32xf4E2M1FN>
                %181 = vector.bitcast %129 : vector<16xi8> to vector<32xf4E2M1FN>
                %182 = vector.bitcast %130 : vector<16xi8> to vector<32xf4E2M1FN>
                %183 = vector.bitcast %131 : vector<16xi8> to vector<32xf4E2M1FN>
                %184 = vector.bitcast %132 : vector<16xi8> to vector<32xf4E2M1FN>
                %185 = vector.bitcast %86 : vector<1xi8> to vector<1xf8E8M0FNU>
                %186 = vector.bitcast %89 : vector<1xi8> to vector<1xf8E8M0FNU>
                %187 = vector.bitcast %91 : vector<1xi8> to vector<1xf8E8M0FNU>
                %188 = vector.bitcast %92 : vector<1xi8> to vector<1xf8E8M0FNU>
                %189 = vector.bitcast %94 : vector<1xi8> to vector<1xf8E8M0FNU>
                %190 = vector.bitcast %95 : vector<1xi8> to vector<1xf8E8M0FNU>
                %191 = vector.bitcast %97 : vector<1xi8> to vector<1xf8E8M0FNU>
                %192 = vector.bitcast %98 : vector<1xi8> to vector<1xf8E8M0FNU>
                %193 = vector.bitcast %100 : vector<1xi8> to vector<1xf8E8M0FNU>
                %194 = vector.bitcast %101 : vector<1xi8> to vector<1xf8E8M0FNU>
                %195 = vector.bitcast %103 : vector<1xi8> to vector<1xf8E8M0FNU>
                %196 = vector.bitcast %104 : vector<1xi8> to vector<1xf8E8M0FNU>
                %197 = vector.bitcast %106 : vector<1xi8> to vector<1xf8E8M0FNU>
                %198 = vector.bitcast %107 : vector<1xi8> to vector<1xf8E8M0FNU>
                %199 = vector.bitcast %109 : vector<1xi8> to vector<1xf8E8M0FNU>
                %200 = vector.bitcast %110 : vector<1xi8> to vector<1xf8E8M0FNU>
                %201 = vector.extract %161[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %202 = vector.extract %185[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %203 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%202[0] * %169) + %83#0 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %204 = vector.extract %162[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %205 = vector.extract %186[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %206 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%205[0] * %170) + %203 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %207 = vector.extract %187[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %208 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%207[0] * %171) + %83#1 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %209 = vector.extract %188[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %210 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%209[0] * %172) + %208 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %211 = vector.extract %189[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %212 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%211[0] * %173) + %83#2 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %213 = vector.extract %190[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %214 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%213[0] * %174) + %212 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %215 = vector.extract %191[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %216 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%215[0] * %175) + %83#3 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %217 = vector.extract %192[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %218 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%217[0] * %176) + %216 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %219 = vector.extract %193[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %220 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%219[0] * %177) + %83#4 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %221 = vector.extract %194[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %222 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%221[0] * %178) + %220 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %223 = vector.extract %195[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %224 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%223[0] * %179) + %83#5 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %225 = vector.extract %196[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %226 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%225[0] * %180) + %224 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %227 = vector.extract %197[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %228 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%227[0] * %181) + %83#6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %229 = vector.extract %198[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %230 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%229[0] * %182) + %228 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %231 = vector.extract %199[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %232 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%231[0] * %183) + %83#7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %233 = vector.extract %200[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %234 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%233[0] * %184) + %232 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %235 = vector.extract %163[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %236 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%202[0] * %169) + %83#8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %237 = vector.extract %164[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %238 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%205[0] * %170) + %236 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %239 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%207[0] * %171) + %83#9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %240 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%209[0] * %172) + %239 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %241 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%211[0] * %173) + %83#10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %242 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%213[0] * %174) + %241 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %243 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%215[0] * %175) + %83#11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %244 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%217[0] * %176) + %243 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %245 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%219[0] * %177) + %83#12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %246 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%221[0] * %178) + %245 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %247 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%223[0] * %179) + %83#13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %248 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%225[0] * %180) + %247 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %249 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%227[0] * %181) + %83#14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %250 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%229[0] * %182) + %249 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %251 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%231[0] * %183) + %83#15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %252 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%233[0] * %184) + %251 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %253 = vector.extract %165[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %254 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%202[0] * %169) + %83#16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %255 = vector.extract %166[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %256 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%205[0] * %170) + %254 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %257 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%207[0] * %171) + %83#17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %258 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%209[0] * %172) + %257 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %259 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%211[0] * %173) + %83#18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %260 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%213[0] * %174) + %259 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %261 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%215[0] * %175) + %83#19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %262 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%217[0] * %176) + %261 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %263 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%219[0] * %177) + %83#20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %264 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%221[0] * %178) + %263 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %265 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%223[0] * %179) + %83#21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %266 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%225[0] * %180) + %265 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %267 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%227[0] * %181) + %83#22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %268 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%229[0] * %182) + %267 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %269 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%231[0] * %183) + %83#23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %270 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%233[0] * %184) + %269 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %271 = vector.extract %167[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %272 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%202[0] * %169) + %83#24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %273 = vector.extract %168[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %274 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%205[0] * %170) + %272 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %275 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%207[0] * %171) + %83#25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %276 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%209[0] * %172) + %275 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %277 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%211[0] * %173) + %83#26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %278 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%213[0] * %174) + %277 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %279 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%215[0] * %175) + %83#27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %280 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%217[0] * %176) + %279 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %281 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%219[0] * %177) + %83#28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %282 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%221[0] * %178) + %281 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %283 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%223[0] * %179) + %83#29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %284 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%225[0] * %180) + %283 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %285 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%227[0] * %181) + %83#30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %286 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%229[0] * %182) + %285 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %287 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%231[0] * %183) + %83#31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %288 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%233[0] * %184) + %287 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %289 = vector.extract_strided_slice %206 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %290 = affine.apply #map35()[%block_id_x]
                %291 = affine.apply #map35()[%block_id_y]
                %292 = affine.apply #map36()[%thread_id_x]
                %293 = arith.muli %290, %c57344 overflow<nsw> : index
                %294 = arith.muli %292, %c57344 overflow<nsw> : index
                %295 = arith.addi %293, %291 overflow<nsw> : index
                %296 = arith.addi %294, %84 overflow<nsw> : index
                %reinterpret_cast_13 = memref.reinterpret_cast %4 to offset: [%295], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
                %cast_14 = memref.cast %reinterpret_cast_13 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
                %297 = amdgpu.fat_raw_buffer_cast %cast_14 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                vector.store %289, %297[%296] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %298 = vector.extract_strided_slice %206 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %299 = affine.apply #map37()[%thread_id_x]
                %300 = arith.muli %299, %c57344 overflow<nsw> : index
                %301 = arith.addi %300, %84 overflow<nsw> : index
                vector.store %298, %297[%301] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %302 = vector.extract_strided_slice %206 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %303 = affine.apply #map38()[%thread_id_x]
                %304 = arith.muli %303, %c57344 overflow<nsw> : index
                %305 = arith.addi %304, %84 overflow<nsw> : index
                vector.store %302, %297[%305] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %306 = vector.extract_strided_slice %206 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %307 = affine.apply #map39()[%thread_id_x]
                %308 = arith.muli %307, %c57344 overflow<nsw> : index
                %309 = arith.addi %308, %84 overflow<nsw> : index
                vector.store %306, %297[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %310 = vector.extract_strided_slice %210 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %311 = arith.addi %294, %90 overflow<nsw> : index
                vector.store %310, %297[%311] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %312 = vector.extract_strided_slice %210 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %313 = arith.addi %300, %90 overflow<nsw> : index
                vector.store %312, %297[%313] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %314 = vector.extract_strided_slice %210 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %315 = arith.addi %304, %90 overflow<nsw> : index
                vector.store %314, %297[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %316 = vector.extract_strided_slice %210 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %317 = arith.addi %308, %90 overflow<nsw> : index
                vector.store %316, %297[%317] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %318 = vector.extract_strided_slice %214 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %319 = arith.addi %294, %93 overflow<nsw> : index
                vector.store %318, %297[%319] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %320 = vector.extract_strided_slice %214 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %321 = arith.addi %300, %93 overflow<nsw> : index
                vector.store %320, %297[%321] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %322 = vector.extract_strided_slice %214 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %323 = arith.addi %304, %93 overflow<nsw> : index
                vector.store %322, %297[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %324 = vector.extract_strided_slice %214 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %325 = arith.addi %308, %93 overflow<nsw> : index
                vector.store %324, %297[%325] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %326 = vector.extract_strided_slice %218 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %327 = arith.addi %294, %96 overflow<nsw> : index
                vector.store %326, %297[%327] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %328 = vector.extract_strided_slice %218 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %329 = arith.addi %300, %96 overflow<nsw> : index
                vector.store %328, %297[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %330 = vector.extract_strided_slice %218 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %331 = arith.addi %304, %96 overflow<nsw> : index
                vector.store %330, %297[%331] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %332 = vector.extract_strided_slice %218 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %333 = arith.addi %308, %96 overflow<nsw> : index
                vector.store %332, %297[%333] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %334 = vector.extract_strided_slice %222 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %335 = arith.addi %294, %99 overflow<nsw> : index
                vector.store %334, %297[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %336 = vector.extract_strided_slice %222 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %337 = arith.addi %300, %99 overflow<nsw> : index
                vector.store %336, %297[%337] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %338 = vector.extract_strided_slice %222 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %339 = arith.addi %304, %99 overflow<nsw> : index
                vector.store %338, %297[%339] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %340 = vector.extract_strided_slice %222 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %341 = arith.addi %308, %99 overflow<nsw> : index
                vector.store %340, %297[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %342 = vector.extract_strided_slice %226 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %343 = arith.addi %294, %102 overflow<nsw> : index
                vector.store %342, %297[%343] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %344 = vector.extract_strided_slice %226 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %345 = arith.addi %300, %102 overflow<nsw> : index
                vector.store %344, %297[%345] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %346 = vector.extract_strided_slice %226 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %347 = arith.addi %304, %102 overflow<nsw> : index
                vector.store %346, %297[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %348 = vector.extract_strided_slice %226 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %349 = arith.addi %308, %102 overflow<nsw> : index
                vector.store %348, %297[%349] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %350 = vector.extract_strided_slice %230 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %351 = arith.addi %294, %105 overflow<nsw> : index
                vector.store %350, %297[%351] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %352 = vector.extract_strided_slice %230 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %353 = arith.addi %300, %105 overflow<nsw> : index
                vector.store %352, %297[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %354 = vector.extract_strided_slice %230 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %355 = arith.addi %304, %105 overflow<nsw> : index
                vector.store %354, %297[%355] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %356 = vector.extract_strided_slice %230 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %357 = arith.addi %308, %105 overflow<nsw> : index
                vector.store %356, %297[%357] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %358 = vector.extract_strided_slice %234 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %359 = arith.addi %294, %108 overflow<nsw> : index
                vector.store %358, %297[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %360 = vector.extract_strided_slice %234 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %361 = arith.addi %300, %108 overflow<nsw> : index
                vector.store %360, %297[%361] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %362 = vector.extract_strided_slice %234 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %363 = arith.addi %304, %108 overflow<nsw> : index
                vector.store %362, %297[%363] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %364 = vector.extract_strided_slice %234 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %365 = arith.addi %308, %108 overflow<nsw> : index
                vector.store %364, %297[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %366 = vector.extract_strided_slice %238 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %367 = affine.apply #map40()[%thread_id_x]
                %368 = arith.muli %367, %c57344 overflow<nsw> : index
                %369 = arith.addi %368, %84 overflow<nsw> : index
                vector.store %366, %297[%369] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %370 = vector.extract_strided_slice %238 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %371 = affine.apply #map41()[%thread_id_x]
                %372 = arith.muli %371, %c57344 overflow<nsw> : index
                %373 = arith.addi %372, %84 overflow<nsw> : index
                vector.store %370, %297[%373] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %374 = vector.extract_strided_slice %238 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %375 = affine.apply #map42()[%thread_id_x]
                %376 = arith.muli %375, %c57344 overflow<nsw> : index
                %377 = arith.addi %376, %84 overflow<nsw> : index
                vector.store %374, %297[%377] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %378 = vector.extract_strided_slice %238 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %379 = affine.apply #map43()[%thread_id_x]
                %380 = arith.muli %379, %c57344 overflow<nsw> : index
                %381 = arith.addi %380, %84 overflow<nsw> : index
                vector.store %378, %297[%381] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %382 = vector.extract_strided_slice %240 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %383 = arith.addi %368, %90 overflow<nsw> : index
                vector.store %382, %297[%383] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %384 = vector.extract_strided_slice %240 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %385 = arith.addi %372, %90 overflow<nsw> : index
                vector.store %384, %297[%385] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %386 = vector.extract_strided_slice %240 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %387 = arith.addi %376, %90 overflow<nsw> : index
                vector.store %386, %297[%387] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %388 = vector.extract_strided_slice %240 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %389 = arith.addi %380, %90 overflow<nsw> : index
                vector.store %388, %297[%389] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %390 = vector.extract_strided_slice %242 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %391 = arith.addi %368, %93 overflow<nsw> : index
                vector.store %390, %297[%391] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %392 = vector.extract_strided_slice %242 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %393 = arith.addi %372, %93 overflow<nsw> : index
                vector.store %392, %297[%393] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %394 = vector.extract_strided_slice %242 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %395 = arith.addi %376, %93 overflow<nsw> : index
                vector.store %394, %297[%395] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %396 = vector.extract_strided_slice %242 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %397 = arith.addi %380, %93 overflow<nsw> : index
                vector.store %396, %297[%397] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %398 = vector.extract_strided_slice %244 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %399 = arith.addi %368, %96 overflow<nsw> : index
                vector.store %398, %297[%399] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %400 = vector.extract_strided_slice %244 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %401 = arith.addi %372, %96 overflow<nsw> : index
                vector.store %400, %297[%401] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %402 = vector.extract_strided_slice %244 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %403 = arith.addi %376, %96 overflow<nsw> : index
                vector.store %402, %297[%403] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %404 = vector.extract_strided_slice %244 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %405 = arith.addi %380, %96 overflow<nsw> : index
                vector.store %404, %297[%405] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %406 = vector.extract_strided_slice %246 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %407 = arith.addi %368, %99 overflow<nsw> : index
                vector.store %406, %297[%407] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %408 = vector.extract_strided_slice %246 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %409 = arith.addi %372, %99 overflow<nsw> : index
                vector.store %408, %297[%409] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %410 = vector.extract_strided_slice %246 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %411 = arith.addi %376, %99 overflow<nsw> : index
                vector.store %410, %297[%411] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %412 = vector.extract_strided_slice %246 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %413 = arith.addi %380, %99 overflow<nsw> : index
                vector.store %412, %297[%413] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %414 = vector.extract_strided_slice %248 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %415 = arith.addi %368, %102 overflow<nsw> : index
                vector.store %414, %297[%415] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %416 = vector.extract_strided_slice %248 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %417 = arith.addi %372, %102 overflow<nsw> : index
                vector.store %416, %297[%417] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %418 = vector.extract_strided_slice %248 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %419 = arith.addi %376, %102 overflow<nsw> : index
                vector.store %418, %297[%419] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %420 = vector.extract_strided_slice %248 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %421 = arith.addi %380, %102 overflow<nsw> : index
                vector.store %420, %297[%421] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %422 = vector.extract_strided_slice %250 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %423 = arith.addi %368, %105 overflow<nsw> : index
                vector.store %422, %297[%423] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %424 = vector.extract_strided_slice %250 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %425 = arith.addi %372, %105 overflow<nsw> : index
                vector.store %424, %297[%425] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %426 = vector.extract_strided_slice %250 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %427 = arith.addi %376, %105 overflow<nsw> : index
                vector.store %426, %297[%427] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %428 = vector.extract_strided_slice %250 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %429 = arith.addi %380, %105 overflow<nsw> : index
                vector.store %428, %297[%429] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %430 = vector.extract_strided_slice %252 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %431 = arith.addi %368, %108 overflow<nsw> : index
                vector.store %430, %297[%431] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %432 = vector.extract_strided_slice %252 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %433 = arith.addi %372, %108 overflow<nsw> : index
                vector.store %432, %297[%433] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %434 = vector.extract_strided_slice %252 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %435 = arith.addi %376, %108 overflow<nsw> : index
                vector.store %434, %297[%435] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %436 = vector.extract_strided_slice %252 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %437 = arith.addi %380, %108 overflow<nsw> : index
                vector.store %436, %297[%437] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %438 = vector.extract_strided_slice %256 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %439 = affine.apply #map44()[%thread_id_x]
                %440 = arith.muli %439, %c57344 overflow<nsw> : index
                %441 = arith.addi %440, %84 overflow<nsw> : index
                vector.store %438, %297[%441] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %442 = vector.extract_strided_slice %256 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %443 = affine.apply #map45()[%thread_id_x]
                %444 = arith.muli %443, %c57344 overflow<nsw> : index
                %445 = arith.addi %444, %84 overflow<nsw> : index
                vector.store %442, %297[%445] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %446 = vector.extract_strided_slice %256 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %447 = affine.apply #map46()[%thread_id_x]
                %448 = arith.muli %447, %c57344 overflow<nsw> : index
                %449 = arith.addi %448, %84 overflow<nsw> : index
                vector.store %446, %297[%449] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %450 = vector.extract_strided_slice %256 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %451 = affine.apply #map47()[%thread_id_x]
                %452 = arith.muli %451, %c57344 overflow<nsw> : index
                %453 = arith.addi %452, %84 overflow<nsw> : index
                vector.store %450, %297[%453] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %454 = vector.extract_strided_slice %258 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %455 = arith.addi %440, %90 overflow<nsw> : index
                vector.store %454, %297[%455] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %456 = vector.extract_strided_slice %258 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %457 = arith.addi %444, %90 overflow<nsw> : index
                vector.store %456, %297[%457] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %458 = vector.extract_strided_slice %258 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %459 = arith.addi %448, %90 overflow<nsw> : index
                vector.store %458, %297[%459] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %460 = vector.extract_strided_slice %258 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %461 = arith.addi %452, %90 overflow<nsw> : index
                vector.store %460, %297[%461] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %462 = vector.extract_strided_slice %260 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %463 = arith.addi %440, %93 overflow<nsw> : index
                vector.store %462, %297[%463] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %464 = vector.extract_strided_slice %260 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %465 = arith.addi %444, %93 overflow<nsw> : index
                vector.store %464, %297[%465] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %466 = vector.extract_strided_slice %260 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %467 = arith.addi %448, %93 overflow<nsw> : index
                vector.store %466, %297[%467] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %468 = vector.extract_strided_slice %260 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %469 = arith.addi %452, %93 overflow<nsw> : index
                vector.store %468, %297[%469] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %470 = vector.extract_strided_slice %262 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %471 = arith.addi %440, %96 overflow<nsw> : index
                vector.store %470, %297[%471] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %472 = vector.extract_strided_slice %262 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %473 = arith.addi %444, %96 overflow<nsw> : index
                vector.store %472, %297[%473] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %474 = vector.extract_strided_slice %262 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %475 = arith.addi %448, %96 overflow<nsw> : index
                vector.store %474, %297[%475] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %476 = vector.extract_strided_slice %262 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %477 = arith.addi %452, %96 overflow<nsw> : index
                vector.store %476, %297[%477] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %478 = vector.extract_strided_slice %264 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %479 = arith.addi %440, %99 overflow<nsw> : index
                vector.store %478, %297[%479] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %480 = vector.extract_strided_slice %264 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %481 = arith.addi %444, %99 overflow<nsw> : index
                vector.store %480, %297[%481] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %482 = vector.extract_strided_slice %264 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %483 = arith.addi %448, %99 overflow<nsw> : index
                vector.store %482, %297[%483] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %484 = vector.extract_strided_slice %264 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %485 = arith.addi %452, %99 overflow<nsw> : index
                vector.store %484, %297[%485] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %486 = vector.extract_strided_slice %266 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %487 = arith.addi %440, %102 overflow<nsw> : index
                vector.store %486, %297[%487] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %488 = vector.extract_strided_slice %266 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %489 = arith.addi %444, %102 overflow<nsw> : index
                vector.store %488, %297[%489] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %490 = vector.extract_strided_slice %266 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %491 = arith.addi %448, %102 overflow<nsw> : index
                vector.store %490, %297[%491] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %492 = vector.extract_strided_slice %266 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %493 = arith.addi %452, %102 overflow<nsw> : index
                vector.store %492, %297[%493] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %494 = vector.extract_strided_slice %268 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %495 = arith.addi %440, %105 overflow<nsw> : index
                vector.store %494, %297[%495] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %496 = vector.extract_strided_slice %268 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %497 = arith.addi %444, %105 overflow<nsw> : index
                vector.store %496, %297[%497] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %498 = vector.extract_strided_slice %268 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %499 = arith.addi %448, %105 overflow<nsw> : index
                vector.store %498, %297[%499] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %500 = vector.extract_strided_slice %268 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %501 = arith.addi %452, %105 overflow<nsw> : index
                vector.store %500, %297[%501] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %502 = vector.extract_strided_slice %270 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %503 = arith.addi %440, %108 overflow<nsw> : index
                vector.store %502, %297[%503] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %504 = vector.extract_strided_slice %270 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %505 = arith.addi %444, %108 overflow<nsw> : index
                vector.store %504, %297[%505] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %506 = vector.extract_strided_slice %270 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %507 = arith.addi %448, %108 overflow<nsw> : index
                vector.store %506, %297[%507] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %508 = vector.extract_strided_slice %270 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %509 = arith.addi %452, %108 overflow<nsw> : index
                vector.store %508, %297[%509] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %510 = vector.extract_strided_slice %274 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %511 = affine.apply #map48()[%thread_id_x]
                %512 = arith.muli %511, %c57344 overflow<nsw> : index
                %513 = arith.addi %512, %84 overflow<nsw> : index
                vector.store %510, %297[%513] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %514 = vector.extract_strided_slice %274 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %515 = affine.apply #map49()[%thread_id_x]
                %516 = arith.muli %515, %c57344 overflow<nsw> : index
                %517 = arith.addi %516, %84 overflow<nsw> : index
                vector.store %514, %297[%517] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %518 = vector.extract_strided_slice %274 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %519 = affine.apply #map50()[%thread_id_x]
                %520 = arith.muli %519, %c57344 overflow<nsw> : index
                %521 = arith.addi %520, %84 overflow<nsw> : index
                vector.store %518, %297[%521] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %522 = vector.extract_strided_slice %274 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %523 = affine.apply #map51()[%thread_id_x]
                %524 = arith.muli %523, %c57344 overflow<nsw> : index
                %525 = arith.addi %524, %84 overflow<nsw> : index
                vector.store %522, %297[%525] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %526 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %527 = arith.addi %512, %90 overflow<nsw> : index
                vector.store %526, %297[%527] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %528 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %529 = arith.addi %516, %90 overflow<nsw> : index
                vector.store %528, %297[%529] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %530 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %531 = arith.addi %520, %90 overflow<nsw> : index
                vector.store %530, %297[%531] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %532 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %533 = arith.addi %524, %90 overflow<nsw> : index
                vector.store %532, %297[%533] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %534 = vector.extract_strided_slice %278 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %535 = arith.addi %512, %93 overflow<nsw> : index
                vector.store %534, %297[%535] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %536 = vector.extract_strided_slice %278 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %537 = arith.addi %516, %93 overflow<nsw> : index
                vector.store %536, %297[%537] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %538 = vector.extract_strided_slice %278 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %539 = arith.addi %520, %93 overflow<nsw> : index
                vector.store %538, %297[%539] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %540 = vector.extract_strided_slice %278 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %541 = arith.addi %524, %93 overflow<nsw> : index
                vector.store %540, %297[%541] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %542 = vector.extract_strided_slice %280 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %543 = arith.addi %512, %96 overflow<nsw> : index
                vector.store %542, %297[%543] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %544 = vector.extract_strided_slice %280 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %545 = arith.addi %516, %96 overflow<nsw> : index
                vector.store %544, %297[%545] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %546 = vector.extract_strided_slice %280 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %547 = arith.addi %520, %96 overflow<nsw> : index
                vector.store %546, %297[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %548 = vector.extract_strided_slice %280 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %549 = arith.addi %524, %96 overflow<nsw> : index
                vector.store %548, %297[%549] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %550 = vector.extract_strided_slice %282 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %551 = arith.addi %512, %99 overflow<nsw> : index
                vector.store %550, %297[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %552 = vector.extract_strided_slice %282 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %553 = arith.addi %516, %99 overflow<nsw> : index
                vector.store %552, %297[%553] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %554 = vector.extract_strided_slice %282 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %555 = arith.addi %520, %99 overflow<nsw> : index
                vector.store %554, %297[%555] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %556 = vector.extract_strided_slice %282 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %557 = arith.addi %524, %99 overflow<nsw> : index
                vector.store %556, %297[%557] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %558 = vector.extract_strided_slice %284 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %559 = arith.addi %512, %102 overflow<nsw> : index
                vector.store %558, %297[%559] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %560 = vector.extract_strided_slice %284 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %561 = arith.addi %516, %102 overflow<nsw> : index
                vector.store %560, %297[%561] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %562 = vector.extract_strided_slice %284 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %563 = arith.addi %520, %102 overflow<nsw> : index
                vector.store %562, %297[%563] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %564 = vector.extract_strided_slice %284 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %565 = arith.addi %524, %102 overflow<nsw> : index
                vector.store %564, %297[%565] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %566 = vector.extract_strided_slice %286 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %567 = arith.addi %512, %105 overflow<nsw> : index
                vector.store %566, %297[%567] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %568 = vector.extract_strided_slice %286 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %569 = arith.addi %516, %105 overflow<nsw> : index
                vector.store %568, %297[%569] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %570 = vector.extract_strided_slice %286 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %571 = arith.addi %520, %105 overflow<nsw> : index
                vector.store %570, %297[%571] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %572 = vector.extract_strided_slice %286 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %573 = arith.addi %524, %105 overflow<nsw> : index
                vector.store %572, %297[%573] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %574 = vector.extract_strided_slice %288 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %575 = arith.addi %512, %108 overflow<nsw> : index
                vector.store %574, %297[%575] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %576 = vector.extract_strided_slice %288 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %577 = arith.addi %516, %108 overflow<nsw> : index
                vector.store %576, %297[%577] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %578 = vector.extract_strided_slice %288 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %579 = arith.addi %520, %108 overflow<nsw> : index
                vector.store %578, %297[%579] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %580 = vector.extract_strided_slice %288 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %581 = arith.addi %524, %108 overflow<nsw> : index
                vector.store %580, %297[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<4096x8192xi8>
            %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<4096x512xi8>
            %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<57344x8192xi8>
            %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<57344x512xi8>
            %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<4096x57344xf32>
            %5 = flow.dispatch @gemm::@gemm(%0, %1, %2, %3, %4) : (tensor<4096x8192xi8>, tensor<4096x512xi8>, tensor<57344x8192xi8>, tensor<57344x512xi8>, tensor<4096x57344xf32>) -> %4
            %6 = hal.tensor.barrier join(%5 : tensor<4096x57344xf32>) => %arg6 : !hal.fence
            %7 = hal.tensor.export %6 : tensor<4096x57344xf32> -> !hal.buffer_view
            return %7 : !hal.buffer_view
        }
        }
    """

    mlir_claude_rescheduled = """
                #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
                #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
                #map2 = affine_map<()[s0] -> (s0 mod 8)>
                #map3 = affine_map<()[s0] -> (s0 * 16)>
                #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
                #map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
                #map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
                #map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
                #map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
                #map9 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
                #map10 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
                #map11 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
                #map12 = affine_map<()[s0] -> ((s0 floordiv 2) mod 2)>
                #map13 = affine_map<()[s0] -> (s0 mod 2)>
                #map14 = affine_map<()[s0] -> (s0 * 4)>
                #map15 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 32 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 256)>
                #map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
                #map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
                #map18 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
                #map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
                #map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
                #map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
                #map22 = affine_map<()[s0] -> (s0 * 4 + (s0 mod 64) floordiv 16 - (s0 floordiv 2) * 8)>
                #map23 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
                #map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
                #map25 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
                #map26 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
                #map27 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
                #map28 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
                #map29 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
                #map30 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
                #map31 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
                #map32 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
                #map33 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 128)>
                #map34 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 + 8)>
                #map35 = affine_map<()[s0] -> (s0 * 256)>
                #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
                #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
                #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
                #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
                #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
                #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
                #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
                #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
                #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
                #map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
                #map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
                #map47 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
                #map48 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
                #map49 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
                #map50 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
                #map51 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
                #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
                module attributes {transform.with_named_sequence} {
                stream.executable private @gemm {
                    stream.executable.export public @gemm workgroups() -> (index, index, index) {
                    %c16 = arith.constant 16 : index
                    %c224 = arith.constant 224 : index
                    %c1 = arith.constant 1 : index
                    stream.return %c16, %c224, %c1 : index, index, index
                    }
                    builtin.module {
                    func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
                        %c4_i32 = arith.constant 4 : i32
                        %c512_i14 = arith.constant 512 : i14
                        %c-8192_i14 = arith.constant -8192 : i14
                        %c2147483643_i64 = arith.constant 2147483643 : i64
                        %c57344 = arith.constant 57344 : index
                        %c63 = arith.constant 63 : index
                        %c512 = arith.constant 512 : index
                        %c2147483646_i64 = arith.constant 2147483646 : i64
                        %c8192 = arith.constant 8192 : index
                        %c1 = arith.constant 1 : index
                        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                        %c0 = arith.constant 0 : index
                        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
                        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
                        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
                        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
                        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
                        %block_id_x = gpu.block_id  x upper_bound 16
                        %block_id_y = gpu.block_id  y upper_bound 224
                        %thread_id_x = gpu.thread_id  x upper_bound 256
                        %thread_id_y = gpu.thread_id  y upper_bound 2
                        %alloc = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                        %alloc_0 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                        %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                        %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                        %alloc_3 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                        %alloc_4 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                        %alloc_5 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                        %alloc_6 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                        %5 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                        %6 = affine.apply #map1()[%thread_id_x]
                        %7 = affine.apply #map2()[%thread_id_x]
                        %8 = arith.xori %7, %6 : index
                        %9 = affine.apply #map3()[%8]
                        %10 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                        %11 = gpu.subgroup_broadcast %10,  first_active_lane : index
                        %12 = gpu.subgroup_broadcast %c0,  first_active_lane : index
                        %13 = arith.muli %5, %c8192 overflow<nsw> : index
                        %14 = arith.addi %13, %9 overflow<nsw> : index
                        %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                        %cast = memref.cast %reinterpret_cast : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                        %15 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                        %16 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
                        %17 = affine.apply #map6()[%thread_id_x, %thread_id_y]
                        %18 = gpu.subgroup_broadcast %17,  first_active_lane : index
                        %19 = arith.muli %16, %c8192 overflow<nsw> : index
                        %20 = arith.addi %19, %9 overflow<nsw> : index
                        %21 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
                        %22 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                        %23 = gpu.subgroup_broadcast %22,  first_active_lane : index
                        %24 = arith.muli %21, %c8192 overflow<nsw> : index
                        %25 = arith.addi %24, %9 overflow<nsw> : index
                        %26 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
                        %27 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                        %28 = gpu.subgroup_broadcast %27,  first_active_lane : index
                        %29 = arith.muli %26, %c8192 overflow<nsw> : index
                        %30 = arith.addi %29, %9 overflow<nsw> : index
                        %31 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_x]
                        %32 = affine.apply #map12()[%thread_id_x]
                        %33 = affine.apply #map13()[%thread_id_x]
                        %34 = arith.xori %33, %32 : index
                        %35 = affine.apply #map14()[%34]
                        %36 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                        %37 = gpu.subgroup_broadcast %36,  first_active_lane : index
                        %38 = arith.muli %31, %c512 overflow<nsw> : index
                        %39 = arith.addi %38, %35 overflow<nsw> : index
                        %reinterpret_cast_7 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                        %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                        %40 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                        %41 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
                        %42 = arith.muli %41, %c8192 overflow<nsw> : index
                        %43 = arith.addi %42, %9 overflow<nsw> : index
                        %reinterpret_cast_9 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                        %cast_10 = memref.cast %reinterpret_cast_9 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                        %44 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                        %45 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                        %46 = arith.muli %45, %c8192 overflow<nsw> : index
                        %47 = arith.addi %46, %9 overflow<nsw> : index
                        %48 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
                        %49 = arith.muli %48, %c8192 overflow<nsw> : index
                        %50 = arith.addi %49, %9 overflow<nsw> : index
                        %51 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_y]
                        %52 = arith.muli %51, %c8192 overflow<nsw> : index
                        %53 = arith.addi %52, %9 overflow<nsw> : index
                        %54 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_y]
                        %55 = arith.muli %54, %c512 overflow<nsw> : index
                        %56 = arith.addi %55, %35 overflow<nsw> : index
                        %reinterpret_cast_11 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                        %cast_12 = memref.cast %reinterpret_cast_11 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                        %57 = amdgpu.fat_raw_buffer_cast %cast_12 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                        amdgpu.gather_to_lds %15[%14], %alloc_6[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %15[%20], %alloc_6[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %15[%25], %alloc_6[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %15[%30], %alloc_6[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %40[%39], %alloc_4[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %44[%43], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %44[%47], %alloc_2[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %44[%50], %alloc_2[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %44[%53], %alloc_2[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %57[%56], %alloc_0[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                        rocdl.s.barrier
                        %58 = affine.apply #map16()[%thread_id_x, %thread_id_y]
                        %59 = arith.index_cast %58 : index to i32
                        %60 = arith.cmpi sge, %59, %c4_i32 : i32
                        %61 = arith.cmpi slt, %59, %c4_i32 : i32
                        scf.if %60 {
                        rocdl.s.barrier
                        }
                        %62 = affine.apply #map17()[%thread_id_x]
                        %63 = affine.apply #map18()[%thread_id_x]
                        %64 = arith.xori %63, %7 : index
                        %65 = affine.apply #map3()[%64]
                        %66 = affine.apply #map19()[%thread_id_x]
                        %67 = affine.apply #map20()[%thread_id_x]
                        %68 = affine.apply #map21()[%thread_id_x]
                        %69 = affine.apply #map22()[%thread_id_x]
                        %70 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                        %71 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                        %72 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                        %73 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                        %74 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                        %75 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                        %76 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                        %77 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                        %78 = affine.apply #map31()[%thread_id_x]
                        %79 = arith.xori %78, %7 : index
                        %80 = affine.apply #map3()[%79]
                        %81 = arith.xori %33, %c1 : index
                        %82 = affine.apply #map32()[%thread_id_x, %81]
                        %83:40 = scf.for %arg5 = %c0 to %c63 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %alloc_6, %arg39 = %alloc_5, %arg40 = %alloc_4, %arg41 = %alloc_3, %arg42 = %alloc_2, %arg43 = %alloc_1, %arg44 = %alloc_0, %arg45 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>) {
                        rocdl.sched.barrier 0
                        rocdl.s.barrier
                        %582 = affine.apply #map33()[%arg5, %8]
                        %583 = arith.addi %13, %582 overflow<nsw> : index
                        %584 = arith.addi %19, %582 overflow<nsw> : index
                        %585 = arith.addi %24, %582 overflow<nsw> : index
                        %586 = arith.addi %29, %582 overflow<nsw> : index
                        %587 = affine.apply #map34()[%arg5, %34]
                        %588 = arith.addi %38, %587 overflow<nsw> : index
                        %589 = arith.addi %42, %582 overflow<nsw> : index
                        %590 = arith.addi %46, %582 overflow<nsw> : index
                        %591 = arith.addi %49, %582 overflow<nsw> : index
                        %592 = arith.addi %52, %582 overflow<nsw> : index
                        %593 = arith.addi %55, %587 overflow<nsw> : index
                        amdgpu.gather_to_lds %15[%583], %arg39[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %15[%584], %arg39[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %15[%585], %arg39[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %15[%586], %arg39[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %40[%588], %arg41[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %44[%589], %arg43[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %44[%590], %arg43[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %44[%591], %arg43[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %44[%592], %arg43[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                        amdgpu.gather_to_lds %57[%593], %arg45[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                        rocdl.sched.barrier 0
                        amdgpu.memory_counter_wait load(10)
                        %594 = vector.load %arg38[%62, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %595 = vector.load %arg38[%66, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %596 = vector.load %arg38[%67, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %597 = vector.load %arg38[%68, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %598 = vector.load %arg40[%62, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %599 = vector.load %arg40[%66, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %600 = vector.load %arg40[%67, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %601 = vector.load %arg40[%68, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %602 = vector.load %arg42[%70, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %603 = vector.load %arg42[%71, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %604 = vector.load %arg42[%72, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %605 = vector.load %arg42[%73, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %606 = vector.load %arg42[%74, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %607 = vector.load %arg42[%75, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %608 = vector.load %arg42[%76, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %609 = vector.load %arg42[%77, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %610 = vector.load %arg44[%70, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %611 = vector.load %arg44[%71, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %612 = vector.load %arg44[%72, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %613 = vector.load %arg44[%73, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %614 = vector.load %arg44[%74, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %615 = vector.load %arg44[%75, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %616 = vector.load %arg44[%76, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %617 = vector.load %arg44[%77, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %618 = vector.bitcast %594 : vector<16xi8> to vector<32xf4E2M1FN>
                        %619 = vector.bitcast %595 : vector<16xi8> to vector<32xf4E2M1FN>
                        %620 = vector.bitcast %596 : vector<16xi8> to vector<32xf4E2M1FN>
                        %621 = vector.bitcast %597 : vector<16xi8> to vector<32xf4E2M1FN>
                        %622 = vector.bitcast %598 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %623 = vector.bitcast %599 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %624 = vector.bitcast %600 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %625 = vector.bitcast %601 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %626 = vector.bitcast %602 : vector<16xi8> to vector<32xf4E2M1FN>
                        %627 = vector.bitcast %603 : vector<16xi8> to vector<32xf4E2M1FN>
                        %628 = vector.bitcast %604 : vector<16xi8> to vector<32xf4E2M1FN>
                        %629 = vector.bitcast %605 : vector<16xi8> to vector<32xf4E2M1FN>
                        %630 = vector.bitcast %606 : vector<16xi8> to vector<32xf4E2M1FN>
                        %631 = vector.bitcast %607 : vector<16xi8> to vector<32xf4E2M1FN>
                        %632 = vector.bitcast %608 : vector<16xi8> to vector<32xf4E2M1FN>
                        %633 = vector.bitcast %609 : vector<16xi8> to vector<32xf4E2M1FN>
                        %634 = vector.bitcast %610 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %635 = vector.bitcast %611 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %636 = vector.bitcast %612 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %637 = vector.bitcast %613 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %638 = vector.bitcast %614 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %639 = vector.bitcast %615 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %640 = vector.bitcast %616 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %641 = vector.bitcast %617 : vector<1xi8> to vector<1xf8E8M0FNU>
                        rocdl.sched.barrier 0
                        rocdl.s.barrier
                        rocdl.sched.barrier 0
                        rocdl.s.setprio 1
                        // --- SAFE MFMAs: M0,M1 x N0,N1,N4,N5 (cluster 0 data only) ---
                        %642 = vector.extract %622[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %643 = vector.extract %634[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %644 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%643[0] * %626) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %645 = vector.extract %635[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %646 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%645[0] * %627) + %arg7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %651 = vector.extract %638[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %652 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%651[0] * %630) + %arg10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %653 = vector.extract %639[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %654 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%653[0] * %631) + %arg11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %659 = vector.extract %623[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %660 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%643[0] * %626) + %arg14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %661 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%645[0] * %627) + %arg15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %664 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%651[0] * %630) + %arg18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %665 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%653[0] * %631) + %arg19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        // --- DEPENDENT MFMAs: M0,M1 x N2,N3,N6,N7 (cluster 1 B data) ---
                        %647 = vector.extract %636[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %648 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%647[0] * %628) + %arg8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %649 = vector.extract %637[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %650 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%649[0] * %629) + %arg9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %655 = vector.extract %640[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %656 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%655[0] * %632) + %arg12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %657 = vector.extract %641[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %658 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%657[0] * %633) + %arg13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %662 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%647[0] * %628) + %arg16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %663 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%649[0] * %629) + %arg17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %666 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%655[0] * %632) + %arg20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %667 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%657[0] * %633) + %arg21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        // --- DEPENDENT MFMAs: M2 x all N (cluster 1 A data) ---
                        %668 = vector.extract %624[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %669 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%643[0] * %626) + %arg22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %670 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%645[0] * %627) + %arg23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %671 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%647[0] * %628) + %arg24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %672 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%649[0] * %629) + %arg25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %673 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%651[0] * %630) + %arg26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %674 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%653[0] * %631) + %arg27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %675 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%655[0] * %632) + %arg28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %676 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%657[0] * %633) + %arg29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        // --- DEPENDENT MFMAs: M3 x all N (cluster 1 A data) ---
                        %677 = vector.extract %625[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %678 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%643[0] * %626) + %arg30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %679 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%645[0] * %627) + %arg31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %680 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%647[0] * %628) + %arg32 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %681 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%649[0] * %629) + %arg33 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %682 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%651[0] * %630) + %arg34 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %683 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%653[0] * %631) + %arg35 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %684 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%655[0] * %632) + %arg36 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %685 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%657[0] * %633) + %arg37 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        rocdl.s.setprio 0
                        rocdl.sched.barrier 0
                        rocdl.s.barrier
                        rocdl.sched.barrier 0
                        rocdl.sched.barrier 0
                        %686 = vector.load %arg38[%62, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %687 = vector.load %arg38[%66, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %688 = vector.load %arg38[%67, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %689 = vector.load %arg38[%68, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %690 = vector.load %arg40[%62, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %691 = vector.load %arg40[%66, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %692 = vector.load %arg40[%67, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %693 = vector.load %arg40[%68, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %694 = vector.load %arg42[%70, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %695 = vector.load %arg42[%71, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %696 = vector.load %arg42[%72, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %697 = vector.load %arg42[%73, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %698 = vector.load %arg42[%74, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %699 = vector.load %arg42[%75, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %700 = vector.load %arg42[%76, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %701 = vector.load %arg42[%77, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %702 = vector.load %arg44[%70, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %703 = vector.load %arg44[%71, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %704 = vector.load %arg44[%72, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %705 = vector.load %arg44[%73, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %706 = vector.load %arg44[%74, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %707 = vector.load %arg44[%75, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %708 = vector.load %arg44[%76, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %709 = vector.load %arg44[%77, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %710 = vector.bitcast %686 : vector<16xi8> to vector<32xf4E2M1FN>
                        %711 = vector.bitcast %687 : vector<16xi8> to vector<32xf4E2M1FN>
                        %712 = vector.bitcast %688 : vector<16xi8> to vector<32xf4E2M1FN>
                        %713 = vector.bitcast %689 : vector<16xi8> to vector<32xf4E2M1FN>
                        %714 = vector.bitcast %690 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %715 = vector.bitcast %691 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %716 = vector.bitcast %692 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %717 = vector.bitcast %693 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %718 = vector.bitcast %694 : vector<16xi8> to vector<32xf4E2M1FN>
                        %719 = vector.bitcast %695 : vector<16xi8> to vector<32xf4E2M1FN>
                        %720 = vector.bitcast %696 : vector<16xi8> to vector<32xf4E2M1FN>
                        %721 = vector.bitcast %697 : vector<16xi8> to vector<32xf4E2M1FN>
                        %722 = vector.bitcast %698 : vector<16xi8> to vector<32xf4E2M1FN>
                        %723 = vector.bitcast %699 : vector<16xi8> to vector<32xf4E2M1FN>
                        %724 = vector.bitcast %700 : vector<16xi8> to vector<32xf4E2M1FN>
                        %725 = vector.bitcast %701 : vector<16xi8> to vector<32xf4E2M1FN>
                        %726 = vector.bitcast %702 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %727 = vector.bitcast %703 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %728 = vector.bitcast %704 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %729 = vector.bitcast %705 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %730 = vector.bitcast %706 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %731 = vector.bitcast %707 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %732 = vector.bitcast %708 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %733 = vector.bitcast %709 : vector<1xi8> to vector<1xf8E8M0FNU>
                        rocdl.sched.barrier 0
                        rocdl.s.barrier
                        rocdl.sched.barrier 0
                        rocdl.s.setprio 1
                        // --- SAFE MFMAs: M0,M1 x N0,N1,N4,N5 (cluster 0 data only) ---
                        %734 = vector.extract %714[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %735 = vector.extract %726[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %736 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%735[0] * %718) + %644 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %737 = vector.extract %727[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %738 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%737[0] * %719) + %646 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %743 = vector.extract %730[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %744 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%743[0] * %722) + %652 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %745 = vector.extract %731[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %746 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%745[0] * %723) + %654 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %751 = vector.extract %715[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %752 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%735[0] * %718) + %660 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %753 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%737[0] * %719) + %661 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %756 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%743[0] * %722) + %664 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %757 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%745[0] * %723) + %665 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        // --- DEPENDENT MFMAs: M0,M1 x N2,N3,N6,N7 (cluster 1 B data) ---
                        %739 = vector.extract %728[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %740 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%739[0] * %720) + %648 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %741 = vector.extract %729[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %742 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%741[0] * %721) + %650 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %747 = vector.extract %732[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %748 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%747[0] * %724) + %656 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %749 = vector.extract %733[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %750 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%749[0] * %725) + %658 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %754 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%739[0] * %720) + %662 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %755 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%741[0] * %721) + %663 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %758 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%747[0] * %724) + %666 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %759 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%749[0] * %725) + %667 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        // --- DEPENDENT MFMAs: M2 x all N (cluster 1 A data) ---
                        %760 = vector.extract %716[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %761 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%735[0] * %718) + %669 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %762 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%737[0] * %719) + %670 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %763 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%739[0] * %720) + %671 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %764 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%741[0] * %721) + %672 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %765 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%743[0] * %722) + %673 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %766 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%745[0] * %723) + %674 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %767 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%747[0] * %724) + %675 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %768 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%749[0] * %725) + %676 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        // --- DEPENDENT MFMAs: M3 x all N (cluster 1 A data) ---
                        %769 = vector.extract %717[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %770 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%735[0] * %718) + %678 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %771 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%737[0] * %719) + %679 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %772 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%739[0] * %720) + %680 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %773 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%741[0] * %721) + %681 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %774 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%743[0] * %722) + %682 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %775 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%745[0] * %723) + %683 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %776 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%747[0] * %724) + %684 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %777 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%749[0] * %725) + %685 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        rocdl.s.setprio 0
                        rocdl.sched.barrier 0
                        scf.yield %736, %738, %740, %742, %744, %746, %748, %750, %752, %753, %754, %755, %756, %757, %758, %759, %761, %762, %763, %764, %765, %766, %767, %768, %770, %771, %772, %773, %774, %775, %776, %777, %arg39, %arg38, %arg41, %arg40, %arg43, %arg42, %arg45, %arg44 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                        }
                        scf.if %61 {
                        rocdl.s.barrier
                        }
                        amdgpu.lds_barrier
                        %84 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                        %85 = affine.apply #map22()[%thread_id_x]
                        %86 = vector.load %83#38[%84, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %87 = arith.xori %33, %c1 : index
                        %88 = affine.apply #map32()[%thread_id_x, %87]
                        %89 = vector.load %83#38[%84, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %90 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                        %91 = vector.load %83#38[%90, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %92 = vector.load %83#38[%90, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %93 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                        %94 = vector.load %83#38[%93, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %95 = vector.load %83#38[%93, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %96 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                        %97 = vector.load %83#38[%96, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %98 = vector.load %83#38[%96, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %99 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                        %100 = vector.load %83#38[%99, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %101 = vector.load %83#38[%99, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %102 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                        %103 = vector.load %83#38[%102, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %104 = vector.load %83#38[%102, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %105 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                        %106 = vector.load %83#38[%105, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %107 = vector.load %83#38[%105, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %108 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                        %109 = vector.load %83#38[%108, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %110 = vector.load %83#38[%108, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %111 = affine.apply #map18()[%thread_id_x]
                        %112 = arith.xori %111, %7 : index
                        %113 = affine.apply #map3()[%112]
                        %114 = vector.load %83#36[%84, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %115 = affine.apply #map31()[%thread_id_x]
                        %116 = arith.xori %115, %7 : index
                        %117 = affine.apply #map3()[%116]
                        %118 = vector.load %83#36[%84, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %119 = vector.load %83#36[%90, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %120 = vector.load %83#36[%90, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %121 = vector.load %83#36[%93, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %122 = vector.load %83#36[%93, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %123 = vector.load %83#36[%96, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %124 = vector.load %83#36[%96, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %125 = vector.load %83#36[%99, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %126 = vector.load %83#36[%99, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %127 = vector.load %83#36[%102, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %128 = vector.load %83#36[%102, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %129 = vector.load %83#36[%105, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %130 = vector.load %83#36[%105, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %131 = vector.load %83#36[%108, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %132 = vector.load %83#36[%108, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %133 = affine.apply #map17()[%thread_id_x]
                        %134 = vector.load %83#34[%133, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %135 = vector.load %83#34[%133, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %136 = affine.apply #map19()[%thread_id_x]
                        %137 = vector.load %83#34[%136, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %138 = vector.load %83#34[%136, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %139 = affine.apply #map20()[%thread_id_x]
                        %140 = vector.load %83#34[%139, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %141 = vector.load %83#34[%139, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %142 = affine.apply #map21()[%thread_id_x]
                        %143 = vector.load %83#34[%142, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %144 = vector.load %83#34[%142, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                        %145 = vector.load %83#32[%133, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %146 = vector.load %83#32[%133, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %147 = vector.load %83#32[%136, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %148 = vector.load %83#32[%136, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %149 = vector.load %83#32[%139, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %150 = vector.load %83#32[%139, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %151 = vector.load %83#32[%142, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %152 = vector.load %83#32[%142, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                        %153 = vector.bitcast %145 : vector<16xi8> to vector<32xf4E2M1FN>
                        %154 = vector.bitcast %146 : vector<16xi8> to vector<32xf4E2M1FN>
                        %155 = vector.bitcast %147 : vector<16xi8> to vector<32xf4E2M1FN>
                        %156 = vector.bitcast %148 : vector<16xi8> to vector<32xf4E2M1FN>
                        %157 = vector.bitcast %149 : vector<16xi8> to vector<32xf4E2M1FN>
                        %158 = vector.bitcast %150 : vector<16xi8> to vector<32xf4E2M1FN>
                        %159 = vector.bitcast %151 : vector<16xi8> to vector<32xf4E2M1FN>
                        %160 = vector.bitcast %152 : vector<16xi8> to vector<32xf4E2M1FN>
                        %161 = vector.bitcast %134 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %162 = vector.bitcast %135 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %163 = vector.bitcast %137 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %164 = vector.bitcast %138 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %165 = vector.bitcast %140 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %166 = vector.bitcast %141 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %167 = vector.bitcast %143 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %168 = vector.bitcast %144 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %169 = vector.bitcast %114 : vector<16xi8> to vector<32xf4E2M1FN>
                        %170 = vector.bitcast %118 : vector<16xi8> to vector<32xf4E2M1FN>
                        %171 = vector.bitcast %119 : vector<16xi8> to vector<32xf4E2M1FN>
                        %172 = vector.bitcast %120 : vector<16xi8> to vector<32xf4E2M1FN>
                        %173 = vector.bitcast %121 : vector<16xi8> to vector<32xf4E2M1FN>
                        %174 = vector.bitcast %122 : vector<16xi8> to vector<32xf4E2M1FN>
                        %175 = vector.bitcast %123 : vector<16xi8> to vector<32xf4E2M1FN>
                        %176 = vector.bitcast %124 : vector<16xi8> to vector<32xf4E2M1FN>
                        %177 = vector.bitcast %125 : vector<16xi8> to vector<32xf4E2M1FN>
                        %178 = vector.bitcast %126 : vector<16xi8> to vector<32xf4E2M1FN>
                        %179 = vector.bitcast %127 : vector<16xi8> to vector<32xf4E2M1FN>
                        %180 = vector.bitcast %128 : vector<16xi8> to vector<32xf4E2M1FN>
                        %181 = vector.bitcast %129 : vector<16xi8> to vector<32xf4E2M1FN>
                        %182 = vector.bitcast %130 : vector<16xi8> to vector<32xf4E2M1FN>
                        %183 = vector.bitcast %131 : vector<16xi8> to vector<32xf4E2M1FN>
                        %184 = vector.bitcast %132 : vector<16xi8> to vector<32xf4E2M1FN>
                        %185 = vector.bitcast %86 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %186 = vector.bitcast %89 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %187 = vector.bitcast %91 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %188 = vector.bitcast %92 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %189 = vector.bitcast %94 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %190 = vector.bitcast %95 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %191 = vector.bitcast %97 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %192 = vector.bitcast %98 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %193 = vector.bitcast %100 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %194 = vector.bitcast %101 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %195 = vector.bitcast %103 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %196 = vector.bitcast %104 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %197 = vector.bitcast %106 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %198 = vector.bitcast %107 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %199 = vector.bitcast %109 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %200 = vector.bitcast %110 : vector<1xi8> to vector<1xf8E8M0FNU>
                        %201 = vector.extract %161[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %202 = vector.extract %185[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %203 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%202[0] * %169) + %83#0 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %204 = vector.extract %162[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %205 = vector.extract %186[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %206 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%205[0] * %170) + %203 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %207 = vector.extract %187[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %208 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%207[0] * %171) + %83#1 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %209 = vector.extract %188[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %210 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%209[0] * %172) + %208 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %211 = vector.extract %189[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %212 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%211[0] * %173) + %83#2 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %213 = vector.extract %190[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %214 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%213[0] * %174) + %212 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %215 = vector.extract %191[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %216 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%215[0] * %175) + %83#3 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %217 = vector.extract %192[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %218 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%217[0] * %176) + %216 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %219 = vector.extract %193[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %220 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%219[0] * %177) + %83#4 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %221 = vector.extract %194[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %222 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%221[0] * %178) + %220 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %223 = vector.extract %195[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %224 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%223[0] * %179) + %83#5 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %225 = vector.extract %196[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %226 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%225[0] * %180) + %224 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %227 = vector.extract %197[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %228 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%227[0] * %181) + %83#6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %229 = vector.extract %198[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %230 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%229[0] * %182) + %228 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %231 = vector.extract %199[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %232 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%231[0] * %183) + %83#7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %233 = vector.extract %200[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %234 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%233[0] * %184) + %232 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %235 = vector.extract %163[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %236 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%202[0] * %169) + %83#8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %237 = vector.extract %164[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %238 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%205[0] * %170) + %236 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %239 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%207[0] * %171) + %83#9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %240 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%209[0] * %172) + %239 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %241 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%211[0] * %173) + %83#10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %242 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%213[0] * %174) + %241 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %243 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%215[0] * %175) + %83#11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %244 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%217[0] * %176) + %243 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %245 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%219[0] * %177) + %83#12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %246 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%221[0] * %178) + %245 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %247 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%223[0] * %179) + %83#13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %248 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%225[0] * %180) + %247 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %249 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%227[0] * %181) + %83#14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %250 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%229[0] * %182) + %249 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %251 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%231[0] * %183) + %83#15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %252 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%233[0] * %184) + %251 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %253 = vector.extract %165[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %254 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%202[0] * %169) + %83#16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %255 = vector.extract %166[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %256 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%205[0] * %170) + %254 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %257 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%207[0] * %171) + %83#17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %258 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%209[0] * %172) + %257 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %259 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%211[0] * %173) + %83#18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %260 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%213[0] * %174) + %259 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %261 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%215[0] * %175) + %83#19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %262 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%217[0] * %176) + %261 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %263 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%219[0] * %177) + %83#20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %264 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%221[0] * %178) + %263 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %265 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%223[0] * %179) + %83#21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %266 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%225[0] * %180) + %265 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %267 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%227[0] * %181) + %83#22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %268 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%229[0] * %182) + %267 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %269 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%231[0] * %183) + %83#23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %270 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%233[0] * %184) + %269 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %271 = vector.extract %167[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %272 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%202[0] * %169) + %83#24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %273 = vector.extract %168[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                        %274 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%205[0] * %170) + %272 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %275 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%207[0] * %171) + %83#25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %276 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%209[0] * %172) + %275 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %277 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%211[0] * %173) + %83#26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %278 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%213[0] * %174) + %277 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %279 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%215[0] * %175) + %83#27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %280 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%217[0] * %176) + %279 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %281 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%219[0] * %177) + %83#28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %282 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%221[0] * %178) + %281 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %283 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%223[0] * %179) + %83#29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %284 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%225[0] * %180) + %283 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %285 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%227[0] * %181) + %83#30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %286 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%229[0] * %182) + %285 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %287 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%231[0] * %183) + %83#31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %288 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%233[0] * %184) + %287 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                        %289 = vector.extract_strided_slice %206 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %290 = affine.apply #map35()[%block_id_x]
                        %291 = affine.apply #map35()[%block_id_y]
                        %292 = affine.apply #map36()[%thread_id_x]
                        %293 = arith.muli %290, %c57344 overflow<nsw> : index
                        %294 = arith.muli %292, %c57344 overflow<nsw> : index
                        %295 = arith.addi %293, %291 overflow<nsw> : index
                        %296 = arith.addi %294, %84 overflow<nsw> : index
                        %reinterpret_cast_13 = memref.reinterpret_cast %4 to offset: [%295], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
                        %cast_14 = memref.cast %reinterpret_cast_13 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
                        %297 = amdgpu.fat_raw_buffer_cast %cast_14 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                        vector.store %289, %297[%296] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %298 = vector.extract_strided_slice %206 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %299 = affine.apply #map37()[%thread_id_x]
                        %300 = arith.muli %299, %c57344 overflow<nsw> : index
                        %301 = arith.addi %300, %84 overflow<nsw> : index
                        vector.store %298, %297[%301] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %302 = vector.extract_strided_slice %206 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %303 = affine.apply #map38()[%thread_id_x]
                        %304 = arith.muli %303, %c57344 overflow<nsw> : index
                        %305 = arith.addi %304, %84 overflow<nsw> : index
                        vector.store %302, %297[%305] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %306 = vector.extract_strided_slice %206 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %307 = affine.apply #map39()[%thread_id_x]
                        %308 = arith.muli %307, %c57344 overflow<nsw> : index
                        %309 = arith.addi %308, %84 overflow<nsw> : index
                        vector.store %306, %297[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %310 = vector.extract_strided_slice %210 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %311 = arith.addi %294, %90 overflow<nsw> : index
                        vector.store %310, %297[%311] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %312 = vector.extract_strided_slice %210 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %313 = arith.addi %300, %90 overflow<nsw> : index
                        vector.store %312, %297[%313] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %314 = vector.extract_strided_slice %210 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %315 = arith.addi %304, %90 overflow<nsw> : index
                        vector.store %314, %297[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %316 = vector.extract_strided_slice %210 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %317 = arith.addi %308, %90 overflow<nsw> : index
                        vector.store %316, %297[%317] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %318 = vector.extract_strided_slice %214 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %319 = arith.addi %294, %93 overflow<nsw> : index
                        vector.store %318, %297[%319] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %320 = vector.extract_strided_slice %214 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %321 = arith.addi %300, %93 overflow<nsw> : index
                        vector.store %320, %297[%321] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %322 = vector.extract_strided_slice %214 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %323 = arith.addi %304, %93 overflow<nsw> : index
                        vector.store %322, %297[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %324 = vector.extract_strided_slice %214 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %325 = arith.addi %308, %93 overflow<nsw> : index
                        vector.store %324, %297[%325] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %326 = vector.extract_strided_slice %218 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %327 = arith.addi %294, %96 overflow<nsw> : index
                        vector.store %326, %297[%327] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %328 = vector.extract_strided_slice %218 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %329 = arith.addi %300, %96 overflow<nsw> : index
                        vector.store %328, %297[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %330 = vector.extract_strided_slice %218 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %331 = arith.addi %304, %96 overflow<nsw> : index
                        vector.store %330, %297[%331] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %332 = vector.extract_strided_slice %218 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %333 = arith.addi %308, %96 overflow<nsw> : index
                        vector.store %332, %297[%333] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %334 = vector.extract_strided_slice %222 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %335 = arith.addi %294, %99 overflow<nsw> : index
                        vector.store %334, %297[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %336 = vector.extract_strided_slice %222 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %337 = arith.addi %300, %99 overflow<nsw> : index
                        vector.store %336, %297[%337] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %338 = vector.extract_strided_slice %222 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %339 = arith.addi %304, %99 overflow<nsw> : index
                        vector.store %338, %297[%339] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %340 = vector.extract_strided_slice %222 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %341 = arith.addi %308, %99 overflow<nsw> : index
                        vector.store %340, %297[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %342 = vector.extract_strided_slice %226 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %343 = arith.addi %294, %102 overflow<nsw> : index
                        vector.store %342, %297[%343] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %344 = vector.extract_strided_slice %226 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %345 = arith.addi %300, %102 overflow<nsw> : index
                        vector.store %344, %297[%345] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %346 = vector.extract_strided_slice %226 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %347 = arith.addi %304, %102 overflow<nsw> : index
                        vector.store %346, %297[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %348 = vector.extract_strided_slice %226 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %349 = arith.addi %308, %102 overflow<nsw> : index
                        vector.store %348, %297[%349] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %350 = vector.extract_strided_slice %230 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %351 = arith.addi %294, %105 overflow<nsw> : index
                        vector.store %350, %297[%351] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %352 = vector.extract_strided_slice %230 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %353 = arith.addi %300, %105 overflow<nsw> : index
                        vector.store %352, %297[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %354 = vector.extract_strided_slice %230 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %355 = arith.addi %304, %105 overflow<nsw> : index
                        vector.store %354, %297[%355] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %356 = vector.extract_strided_slice %230 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %357 = arith.addi %308, %105 overflow<nsw> : index
                        vector.store %356, %297[%357] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %358 = vector.extract_strided_slice %234 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %359 = arith.addi %294, %108 overflow<nsw> : index
                        vector.store %358, %297[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %360 = vector.extract_strided_slice %234 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %361 = arith.addi %300, %108 overflow<nsw> : index
                        vector.store %360, %297[%361] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %362 = vector.extract_strided_slice %234 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %363 = arith.addi %304, %108 overflow<nsw> : index
                        vector.store %362, %297[%363] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %364 = vector.extract_strided_slice %234 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %365 = arith.addi %308, %108 overflow<nsw> : index
                        vector.store %364, %297[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %366 = vector.extract_strided_slice %238 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %367 = affine.apply #map40()[%thread_id_x]
                        %368 = arith.muli %367, %c57344 overflow<nsw> : index
                        %369 = arith.addi %368, %84 overflow<nsw> : index
                        vector.store %366, %297[%369] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %370 = vector.extract_strided_slice %238 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %371 = affine.apply #map41()[%thread_id_x]
                        %372 = arith.muli %371, %c57344 overflow<nsw> : index
                        %373 = arith.addi %372, %84 overflow<nsw> : index
                        vector.store %370, %297[%373] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %374 = vector.extract_strided_slice %238 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %375 = affine.apply #map42()[%thread_id_x]
                        %376 = arith.muli %375, %c57344 overflow<nsw> : index
                        %377 = arith.addi %376, %84 overflow<nsw> : index
                        vector.store %374, %297[%377] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %378 = vector.extract_strided_slice %238 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %379 = affine.apply #map43()[%thread_id_x]
                        %380 = arith.muli %379, %c57344 overflow<nsw> : index
                        %381 = arith.addi %380, %84 overflow<nsw> : index
                        vector.store %378, %297[%381] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %382 = vector.extract_strided_slice %240 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %383 = arith.addi %368, %90 overflow<nsw> : index
                        vector.store %382, %297[%383] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %384 = vector.extract_strided_slice %240 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %385 = arith.addi %372, %90 overflow<nsw> : index
                        vector.store %384, %297[%385] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %386 = vector.extract_strided_slice %240 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %387 = arith.addi %376, %90 overflow<nsw> : index
                        vector.store %386, %297[%387] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %388 = vector.extract_strided_slice %240 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %389 = arith.addi %380, %90 overflow<nsw> : index
                        vector.store %388, %297[%389] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %390 = vector.extract_strided_slice %242 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %391 = arith.addi %368, %93 overflow<nsw> : index
                        vector.store %390, %297[%391] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %392 = vector.extract_strided_slice %242 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %393 = arith.addi %372, %93 overflow<nsw> : index
                        vector.store %392, %297[%393] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %394 = vector.extract_strided_slice %242 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %395 = arith.addi %376, %93 overflow<nsw> : index
                        vector.store %394, %297[%395] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %396 = vector.extract_strided_slice %242 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %397 = arith.addi %380, %93 overflow<nsw> : index
                        vector.store %396, %297[%397] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %398 = vector.extract_strided_slice %244 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %399 = arith.addi %368, %96 overflow<nsw> : index
                        vector.store %398, %297[%399] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %400 = vector.extract_strided_slice %244 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %401 = arith.addi %372, %96 overflow<nsw> : index
                        vector.store %400, %297[%401] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %402 = vector.extract_strided_slice %244 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %403 = arith.addi %376, %96 overflow<nsw> : index
                        vector.store %402, %297[%403] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %404 = vector.extract_strided_slice %244 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %405 = arith.addi %380, %96 overflow<nsw> : index
                        vector.store %404, %297[%405] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %406 = vector.extract_strided_slice %246 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %407 = arith.addi %368, %99 overflow<nsw> : index
                        vector.store %406, %297[%407] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %408 = vector.extract_strided_slice %246 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %409 = arith.addi %372, %99 overflow<nsw> : index
                        vector.store %408, %297[%409] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %410 = vector.extract_strided_slice %246 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %411 = arith.addi %376, %99 overflow<nsw> : index
                        vector.store %410, %297[%411] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %412 = vector.extract_strided_slice %246 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %413 = arith.addi %380, %99 overflow<nsw> : index
                        vector.store %412, %297[%413] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %414 = vector.extract_strided_slice %248 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %415 = arith.addi %368, %102 overflow<nsw> : index
                        vector.store %414, %297[%415] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %416 = vector.extract_strided_slice %248 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %417 = arith.addi %372, %102 overflow<nsw> : index
                        vector.store %416, %297[%417] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %418 = vector.extract_strided_slice %248 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %419 = arith.addi %376, %102 overflow<nsw> : index
                        vector.store %418, %297[%419] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %420 = vector.extract_strided_slice %248 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %421 = arith.addi %380, %102 overflow<nsw> : index
                        vector.store %420, %297[%421] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %422 = vector.extract_strided_slice %250 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %423 = arith.addi %368, %105 overflow<nsw> : index
                        vector.store %422, %297[%423] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %424 = vector.extract_strided_slice %250 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %425 = arith.addi %372, %105 overflow<nsw> : index
                        vector.store %424, %297[%425] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %426 = vector.extract_strided_slice %250 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %427 = arith.addi %376, %105 overflow<nsw> : index
                        vector.store %426, %297[%427] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %428 = vector.extract_strided_slice %250 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %429 = arith.addi %380, %105 overflow<nsw> : index
                        vector.store %428, %297[%429] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %430 = vector.extract_strided_slice %252 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %431 = arith.addi %368, %108 overflow<nsw> : index
                        vector.store %430, %297[%431] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %432 = vector.extract_strided_slice %252 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %433 = arith.addi %372, %108 overflow<nsw> : index
                        vector.store %432, %297[%433] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %434 = vector.extract_strided_slice %252 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %435 = arith.addi %376, %108 overflow<nsw> : index
                        vector.store %434, %297[%435] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %436 = vector.extract_strided_slice %252 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %437 = arith.addi %380, %108 overflow<nsw> : index
                        vector.store %436, %297[%437] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %438 = vector.extract_strided_slice %256 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %439 = affine.apply #map44()[%thread_id_x]
                        %440 = arith.muli %439, %c57344 overflow<nsw> : index
                        %441 = arith.addi %440, %84 overflow<nsw> : index
                        vector.store %438, %297[%441] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %442 = vector.extract_strided_slice %256 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %443 = affine.apply #map45()[%thread_id_x]
                        %444 = arith.muli %443, %c57344 overflow<nsw> : index
                        %445 = arith.addi %444, %84 overflow<nsw> : index
                        vector.store %442, %297[%445] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %446 = vector.extract_strided_slice %256 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %447 = affine.apply #map46()[%thread_id_x]
                        %448 = arith.muli %447, %c57344 overflow<nsw> : index
                        %449 = arith.addi %448, %84 overflow<nsw> : index
                        vector.store %446, %297[%449] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %450 = vector.extract_strided_slice %256 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %451 = affine.apply #map47()[%thread_id_x]
                        %452 = arith.muli %451, %c57344 overflow<nsw> : index
                        %453 = arith.addi %452, %84 overflow<nsw> : index
                        vector.store %450, %297[%453] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %454 = vector.extract_strided_slice %258 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %455 = arith.addi %440, %90 overflow<nsw> : index
                        vector.store %454, %297[%455] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %456 = vector.extract_strided_slice %258 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %457 = arith.addi %444, %90 overflow<nsw> : index
                        vector.store %456, %297[%457] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %458 = vector.extract_strided_slice %258 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %459 = arith.addi %448, %90 overflow<nsw> : index
                        vector.store %458, %297[%459] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %460 = vector.extract_strided_slice %258 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %461 = arith.addi %452, %90 overflow<nsw> : index
                        vector.store %460, %297[%461] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %462 = vector.extract_strided_slice %260 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %463 = arith.addi %440, %93 overflow<nsw> : index
                        vector.store %462, %297[%463] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %464 = vector.extract_strided_slice %260 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %465 = arith.addi %444, %93 overflow<nsw> : index
                        vector.store %464, %297[%465] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %466 = vector.extract_strided_slice %260 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %467 = arith.addi %448, %93 overflow<nsw> : index
                        vector.store %466, %297[%467] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %468 = vector.extract_strided_slice %260 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %469 = arith.addi %452, %93 overflow<nsw> : index
                        vector.store %468, %297[%469] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %470 = vector.extract_strided_slice %262 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %471 = arith.addi %440, %96 overflow<nsw> : index
                        vector.store %470, %297[%471] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %472 = vector.extract_strided_slice %262 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %473 = arith.addi %444, %96 overflow<nsw> : index
                        vector.store %472, %297[%473] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %474 = vector.extract_strided_slice %262 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %475 = arith.addi %448, %96 overflow<nsw> : index
                        vector.store %474, %297[%475] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %476 = vector.extract_strided_slice %262 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %477 = arith.addi %452, %96 overflow<nsw> : index
                        vector.store %476, %297[%477] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %478 = vector.extract_strided_slice %264 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %479 = arith.addi %440, %99 overflow<nsw> : index
                        vector.store %478, %297[%479] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %480 = vector.extract_strided_slice %264 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %481 = arith.addi %444, %99 overflow<nsw> : index
                        vector.store %480, %297[%481] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %482 = vector.extract_strided_slice %264 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %483 = arith.addi %448, %99 overflow<nsw> : index
                        vector.store %482, %297[%483] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %484 = vector.extract_strided_slice %264 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %485 = arith.addi %452, %99 overflow<nsw> : index
                        vector.store %484, %297[%485] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %486 = vector.extract_strided_slice %266 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %487 = arith.addi %440, %102 overflow<nsw> : index
                        vector.store %486, %297[%487] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %488 = vector.extract_strided_slice %266 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %489 = arith.addi %444, %102 overflow<nsw> : index
                        vector.store %488, %297[%489] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %490 = vector.extract_strided_slice %266 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %491 = arith.addi %448, %102 overflow<nsw> : index
                        vector.store %490, %297[%491] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %492 = vector.extract_strided_slice %266 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %493 = arith.addi %452, %102 overflow<nsw> : index
                        vector.store %492, %297[%493] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %494 = vector.extract_strided_slice %268 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %495 = arith.addi %440, %105 overflow<nsw> : index
                        vector.store %494, %297[%495] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %496 = vector.extract_strided_slice %268 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %497 = arith.addi %444, %105 overflow<nsw> : index
                        vector.store %496, %297[%497] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %498 = vector.extract_strided_slice %268 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %499 = arith.addi %448, %105 overflow<nsw> : index
                        vector.store %498, %297[%499] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %500 = vector.extract_strided_slice %268 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %501 = arith.addi %452, %105 overflow<nsw> : index
                        vector.store %500, %297[%501] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %502 = vector.extract_strided_slice %270 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %503 = arith.addi %440, %108 overflow<nsw> : index
                        vector.store %502, %297[%503] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %504 = vector.extract_strided_slice %270 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %505 = arith.addi %444, %108 overflow<nsw> : index
                        vector.store %504, %297[%505] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %506 = vector.extract_strided_slice %270 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %507 = arith.addi %448, %108 overflow<nsw> : index
                        vector.store %506, %297[%507] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %508 = vector.extract_strided_slice %270 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %509 = arith.addi %452, %108 overflow<nsw> : index
                        vector.store %508, %297[%509] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %510 = vector.extract_strided_slice %274 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %511 = affine.apply #map48()[%thread_id_x]
                        %512 = arith.muli %511, %c57344 overflow<nsw> : index
                        %513 = arith.addi %512, %84 overflow<nsw> : index
                        vector.store %510, %297[%513] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %514 = vector.extract_strided_slice %274 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %515 = affine.apply #map49()[%thread_id_x]
                        %516 = arith.muli %515, %c57344 overflow<nsw> : index
                        %517 = arith.addi %516, %84 overflow<nsw> : index
                        vector.store %514, %297[%517] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %518 = vector.extract_strided_slice %274 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %519 = affine.apply #map50()[%thread_id_x]
                        %520 = arith.muli %519, %c57344 overflow<nsw> : index
                        %521 = arith.addi %520, %84 overflow<nsw> : index
                        vector.store %518, %297[%521] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %522 = vector.extract_strided_slice %274 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %523 = affine.apply #map51()[%thread_id_x]
                        %524 = arith.muli %523, %c57344 overflow<nsw> : index
                        %525 = arith.addi %524, %84 overflow<nsw> : index
                        vector.store %522, %297[%525] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %526 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %527 = arith.addi %512, %90 overflow<nsw> : index
                        vector.store %526, %297[%527] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %528 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %529 = arith.addi %516, %90 overflow<nsw> : index
                        vector.store %528, %297[%529] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %530 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %531 = arith.addi %520, %90 overflow<nsw> : index
                        vector.store %530, %297[%531] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %532 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %533 = arith.addi %524, %90 overflow<nsw> : index
                        vector.store %532, %297[%533] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %534 = vector.extract_strided_slice %278 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %535 = arith.addi %512, %93 overflow<nsw> : index
                        vector.store %534, %297[%535] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %536 = vector.extract_strided_slice %278 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %537 = arith.addi %516, %93 overflow<nsw> : index
                        vector.store %536, %297[%537] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %538 = vector.extract_strided_slice %278 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %539 = arith.addi %520, %93 overflow<nsw> : index
                        vector.store %538, %297[%539] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %540 = vector.extract_strided_slice %278 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %541 = arith.addi %524, %93 overflow<nsw> : index
                        vector.store %540, %297[%541] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %542 = vector.extract_strided_slice %280 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %543 = arith.addi %512, %96 overflow<nsw> : index
                        vector.store %542, %297[%543] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %544 = vector.extract_strided_slice %280 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %545 = arith.addi %516, %96 overflow<nsw> : index
                        vector.store %544, %297[%545] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %546 = vector.extract_strided_slice %280 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %547 = arith.addi %520, %96 overflow<nsw> : index
                        vector.store %546, %297[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %548 = vector.extract_strided_slice %280 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %549 = arith.addi %524, %96 overflow<nsw> : index
                        vector.store %548, %297[%549] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %550 = vector.extract_strided_slice %282 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %551 = arith.addi %512, %99 overflow<nsw> : index
                        vector.store %550, %297[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %552 = vector.extract_strided_slice %282 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %553 = arith.addi %516, %99 overflow<nsw> : index
                        vector.store %552, %297[%553] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %554 = vector.extract_strided_slice %282 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %555 = arith.addi %520, %99 overflow<nsw> : index
                        vector.store %554, %297[%555] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %556 = vector.extract_strided_slice %282 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %557 = arith.addi %524, %99 overflow<nsw> : index
                        vector.store %556, %297[%557] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %558 = vector.extract_strided_slice %284 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %559 = arith.addi %512, %102 overflow<nsw> : index
                        vector.store %558, %297[%559] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %560 = vector.extract_strided_slice %284 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %561 = arith.addi %516, %102 overflow<nsw> : index
                        vector.store %560, %297[%561] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %562 = vector.extract_strided_slice %284 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %563 = arith.addi %520, %102 overflow<nsw> : index
                        vector.store %562, %297[%563] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %564 = vector.extract_strided_slice %284 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %565 = arith.addi %524, %102 overflow<nsw> : index
                        vector.store %564, %297[%565] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %566 = vector.extract_strided_slice %286 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %567 = arith.addi %512, %105 overflow<nsw> : index
                        vector.store %566, %297[%567] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %568 = vector.extract_strided_slice %286 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %569 = arith.addi %516, %105 overflow<nsw> : index
                        vector.store %568, %297[%569] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %570 = vector.extract_strided_slice %286 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %571 = arith.addi %520, %105 overflow<nsw> : index
                        vector.store %570, %297[%571] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %572 = vector.extract_strided_slice %286 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %573 = arith.addi %524, %105 overflow<nsw> : index
                        vector.store %572, %297[%573] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %574 = vector.extract_strided_slice %288 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %575 = arith.addi %512, %108 overflow<nsw> : index
                        vector.store %574, %297[%575] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %576 = vector.extract_strided_slice %288 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %577 = arith.addi %516, %108 overflow<nsw> : index
                        vector.store %576, %297[%577] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %578 = vector.extract_strided_slice %288 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %579 = arith.addi %520, %108 overflow<nsw> : index
                        vector.store %578, %297[%579] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        %580 = vector.extract_strided_slice %288 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                        %581 = arith.addi %524, %108 overflow<nsw> : index
                        vector.store %580, %297[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                        return
                    }
                    }
                }
                func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
                    %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<4096x8192xi8>
                    %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<4096x512xi8>
                    %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<57344x8192xi8>
                    %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<57344x512xi8>
                    %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<4096x57344xf32>
                    %5 = flow.dispatch @gemm::@gemm(%0, %1, %2, %3, %4) : (tensor<4096x8192xi8>, tensor<4096x512xi8>, tensor<57344x8192xi8>, tensor<57344x512xi8>, tensor<4096x57344xf32>) -> %4
                    %6 = hal.tensor.barrier join(%5 : tensor<4096x57344xf32>) => %arg6 : !hal.fence
                    %7 = hal.tensor.export %6 : tensor<4096x57344xf32> -> !hal.buffer_view
                    return %7 : !hal.buffer_view
                }
                }
    """

    mlir_claude_rescheduled2 = """


        #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 16)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map6 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
        #map7 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map8 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
        #map9 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
        #map11 = affine_map<()[s0, s1, s2] -> (s1 * 128 + s2 * 256 + s0 floordiv 2 - ((s1 * 128 + s0 floordiv 2) floordiv 256) * 256)>
        #map12 = affine_map<()[s0] -> ((s0 floordiv 2) mod 2)>
        #map13 = affine_map<()[s0] -> (s0 mod 2)>
        #map14 = affine_map<()[s0] -> (s0 * 4)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 32 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 256)>
        #map16 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64)>
        #map18 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map19 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 16)>
        #map20 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 32)>
        #map21 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 64 + 48)>
        #map22 = affine_map<()[s0] -> (s0 * 4 + (s0 mod 64) floordiv 16 - (s0 floordiv 2) * 8)>
        #map23 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
        #map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
        #map25 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
        #map26 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
        #map27 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
        #map28 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
        #map29 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
        #map30 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
        #map31 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map32 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
        #map33 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 128)>
        #map34 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 + 8)>
        #map35 = affine_map<()[s0] -> (s0 * 256)>
        #map36 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
        #map37 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
        #map38 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
        #map39 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
        #map40 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
        #map41 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
        #map42 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
        #map43 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
        #map44 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
        #map45 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
        #map46 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
        #map47 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
        #map48 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
        #map49 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
        #map50 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
        #map51 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups() -> (index, index, index) {
            %c16 = arith.constant 16 : index
            %c224 = arith.constant 224 : index
            %c1 = arith.constant 1 : index
            stream.return %c16, %c224, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c512_i14 = arith.constant 512 : i14
                %c-8192_i14 = arith.constant -8192 : i14
                %c2147483643_i64 = arith.constant 2147483643 : i64
                %c57344 = arith.constant 57344 : index
                %c63 = arith.constant 63 : index
                %c512 = arith.constant 512 : index
                %c2147483646_i64 = arith.constant 2147483646 : i64
                %c8192 = arith.constant 8192 : index
                %c1 = arith.constant 1 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
                %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
                %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 16
                %block_id_y = gpu.block_id  y upper_bound 224
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %alloc = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_0 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_1 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_2 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_3 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_4 = memref.alloc() : memref<256x8xi8, #gpu.address_space<workgroup>>
                %alloc_5 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_6 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %5 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %6 = affine.apply #map1()[%thread_id_x]
                %7 = affine.apply #map2()[%thread_id_x]
                %8 = arith.xori %7, %6 : index
                %9 = affine.apply #map3()[%8]
                %10 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %11 = gpu.subgroup_broadcast %10,  first_active_lane : index
                %12 = gpu.subgroup_broadcast %c0,  first_active_lane : index
                %13 = arith.muli %5, %c8192 overflow<nsw> : index
                %14 = arith.addi %13, %9 overflow<nsw> : index
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast = memref.cast %reinterpret_cast : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %15 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %15[%14], %alloc_6[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %16 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
                %17 = affine.apply #map6()[%thread_id_x, %thread_id_y]
                %18 = gpu.subgroup_broadcast %17,  first_active_lane : index
                %19 = arith.muli %16, %c8192 overflow<nsw> : index
                %20 = arith.addi %19, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%20], %alloc_6[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %21 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_x]
                %22 = affine.apply #map8()[%thread_id_x, %thread_id_y]
                %23 = gpu.subgroup_broadcast %22,  first_active_lane : index
                %24 = arith.muli %21, %c8192 overflow<nsw> : index
                %25 = arith.addi %24, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%25], %alloc_6[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %26 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_x]
                %27 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %28 = gpu.subgroup_broadcast %27,  first_active_lane : index
                %29 = arith.muli %26, %c8192 overflow<nsw> : index
                %30 = arith.addi %29, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%30], %alloc_6[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %31 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_x]
                %32 = affine.apply #map12()[%thread_id_x]
                %33 = affine.apply #map13()[%thread_id_x]
                %34 = arith.xori %33, %32 : index
                %35 = affine.apply #map14()[%34]
                %36 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %37 = gpu.subgroup_broadcast %36,  first_active_lane : index
                %38 = arith.muli %31, %c512 overflow<nsw> : index
                %39 = arith.addi %38, %35 overflow<nsw> : index
                %reinterpret_cast_7 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %40 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %40[%39], %alloc_4[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                %41 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_y]
                %42 = arith.muli %41, %c8192 overflow<nsw> : index
                %43 = arith.addi %42, %9 overflow<nsw> : index
                %reinterpret_cast_9 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_10 = memref.cast %reinterpret_cast_9 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %44 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483646_i64) cacheSwizzleStride(%c-8192_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %44[%43], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %45 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                %46 = arith.muli %45, %c8192 overflow<nsw> : index
                %47 = arith.addi %46, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%47], %alloc_2[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %48 = affine.apply #map7()[%thread_id_x, %thread_id_y, %block_id_y]
                %49 = arith.muli %48, %c8192 overflow<nsw> : index
                %50 = arith.addi %49, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%50], %alloc_2[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %51 = affine.apply #map9()[%thread_id_x, %thread_id_y, %block_id_y]
                %52 = arith.muli %51, %c8192 overflow<nsw> : index
                %53 = arith.addi %52, %9 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%53], %alloc_2[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %54 = affine.apply #map11()[%thread_id_x, %thread_id_y, %block_id_y]
                %55 = arith.muli %54, %c512 overflow<nsw> : index
                %56 = arith.addi %55, %35 overflow<nsw> : index
                %reinterpret_cast_11 = memref.reinterpret_cast %3 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_12 = memref.cast %reinterpret_cast_11 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %57 = amdgpu.fat_raw_buffer_cast %cast_12 validBytes(%c2147483646_i64) cacheSwizzleStride(%c512_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %57[%56], %alloc_0[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.s.barrier
                %58 = affine.apply #map16()[%thread_id_x, %thread_id_y]
                %59 = arith.index_cast %58 : index to i32
                %60 = arith.cmpi sge, %59, %c4_i32 : i32
                %61 = arith.cmpi slt, %59, %c4_i32 : i32
                scf.if %60 {
                rocdl.s.barrier
                }
                %62 = affine.apply #map17()[%thread_id_x]
                %63 = affine.apply #map18()[%thread_id_x]
                %64 = arith.xori %63, %7 : index
                %65 = affine.apply #map3()[%64]
                %66 = affine.apply #map19()[%thread_id_x]
                %67 = affine.apply #map20()[%thread_id_x]
                %68 = affine.apply #map21()[%thread_id_x]
                %69 = affine.apply #map22()[%thread_id_x]
                %70 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %71 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %72 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %73 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %74 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %75 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %76 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %77 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %78 = affine.apply #map31()[%thread_id_x]
                %79 = arith.xori %78, %7 : index
                %80 = affine.apply #map3()[%79]
                %81 = arith.xori %33, %c1 : index
                %82 = affine.apply #map32()[%thread_id_x, %81]
                %83:40 = scf.for %arg5 = %c0 to %c63 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %alloc_6, %arg39 = %alloc_5, %arg40 = %alloc_4, %arg41 = %alloc_3, %arg42 = %alloc_2, %arg43 = %alloc_1, %arg44 = %alloc_0, %arg45 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>) {
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(0)
                rocdl.s.barrier
                //rocdl.s.barrier
                %582 = affine.apply #map33()[%arg5, %8]
                %583 = arith.addi %13, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%583], %arg39[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %584 = arith.addi %19, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%584], %arg39[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %585 = arith.addi %24, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%585], %arg39[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %586 = arith.addi %29, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %15[%586], %arg39[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %587 = affine.apply #map34()[%arg5, %34]
                %588 = arith.addi %38, %587 overflow<nsw> : index
                amdgpu.gather_to_lds %40[%588], %arg41[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                %589 = arith.addi %42, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%589], %arg43[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %590 = arith.addi %46, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%590], %arg43[%18, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %591 = arith.addi %49, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%591], %arg43[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %592 = arith.addi %52, %582 overflow<nsw> : index
                amdgpu.gather_to_lds %44[%592], %arg43[%28, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %593 = arith.addi %55, %587 overflow<nsw> : index
                amdgpu.gather_to_lds %57[%593], %arg45[%37, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                rocdl.sched.barrier 0
                //amdgpu.memory_counter_wait load(10)
                // --- SAFE vector.loads: A(M0,M1), Ascale(M0,M1), B(N0,N1,N4,N5), Bscale(N0,N1,N4,N5) ---
                %594 = vector.load %arg38[%62, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %595 = vector.load %arg38[%66, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %598 = vector.load %arg40[%62, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %599 = vector.load %arg40[%66, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %602 = vector.load %arg42[%70, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %603 = vector.load %arg42[%71, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %606 = vector.load %arg42[%74, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %607 = vector.load %arg42[%75, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %610 = vector.load %arg44[%70, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %611 = vector.load %arg44[%71, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %614 = vector.load %arg44[%74, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %615 = vector.load %arg44[%75, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                // --- SAFE bitcasts ---
                %618 = vector.bitcast %594 : vector<16xi8> to vector<32xf4E2M1FN>
                %619 = vector.bitcast %595 : vector<16xi8> to vector<32xf4E2M1FN>
                %622 = vector.bitcast %598 : vector<1xi8> to vector<1xf8E8M0FNU>
                %623 = vector.bitcast %599 : vector<1xi8> to vector<1xf8E8M0FNU>
                %626 = vector.bitcast %602 : vector<16xi8> to vector<32xf4E2M1FN>
                %627 = vector.bitcast %603 : vector<16xi8> to vector<32xf4E2M1FN>
                %630 = vector.bitcast %606 : vector<16xi8> to vector<32xf4E2M1FN>
                %631 = vector.bitcast %607 : vector<16xi8> to vector<32xf4E2M1FN>
                %634 = vector.bitcast %610 : vector<1xi8> to vector<1xf8E8M0FNU>
                %635 = vector.bitcast %611 : vector<1xi8> to vector<1xf8E8M0FNU>
                %638 = vector.bitcast %614 : vector<1xi8> to vector<1xf8E8M0FNU>
                %639 = vector.bitcast %615 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                // --- SAFE MFMAs: M0,M1 x N0,N1,N4,N5 (cluster 0 data only) ---
                %642 = vector.extract %622[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %643 = vector.extract %634[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %644 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%643[0] * %626) + %arg6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %645 = vector.extract %635[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %646 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%645[0] * %627) + %arg7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %651 = vector.extract %638[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %652 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%651[0] * %630) + %arg10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %653 = vector.extract %639[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %654 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%653[0] * %631) + %arg11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %659 = vector.extract %623[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %660 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%643[0] * %626) + %arg14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %661 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%645[0] * %627) + %arg15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %664 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%651[0] * %630) + %arg18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %665 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%653[0] * %631) + %arg19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                // --- DEPENDENT vector.loads: A(M2,M3), Ascale(M2,M3), B(N2,N3,N6,N7), Bscale(N2,N3,N6,N7) ---
                %596 = vector.load %arg38[%67, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %597 = vector.load %arg38[%68, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %600 = vector.load %arg40[%67, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %601 = vector.load %arg40[%68, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %604 = vector.load %arg42[%72, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %605 = vector.load %arg42[%73, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %608 = vector.load %arg42[%76, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %609 = vector.load %arg42[%77, %65] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %612 = vector.load %arg44[%72, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %613 = vector.load %arg44[%73, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %616 = vector.load %arg44[%76, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %617 = vector.load %arg44[%77, %69] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                // --- DEPENDENT bitcasts ---
                %620 = vector.bitcast %596 : vector<16xi8> to vector<32xf4E2M1FN>
                %621 = vector.bitcast %597 : vector<16xi8> to vector<32xf4E2M1FN>
                %624 = vector.bitcast %600 : vector<1xi8> to vector<1xf8E8M0FNU>
                %625 = vector.bitcast %601 : vector<1xi8> to vector<1xf8E8M0FNU>
                %628 = vector.bitcast %604 : vector<16xi8> to vector<32xf4E2M1FN>
                %629 = vector.bitcast %605 : vector<16xi8> to vector<32xf4E2M1FN>
                %632 = vector.bitcast %608 : vector<16xi8> to vector<32xf4E2M1FN>
                %633 = vector.bitcast %609 : vector<16xi8> to vector<32xf4E2M1FN>
                %636 = vector.bitcast %612 : vector<1xi8> to vector<1xf8E8M0FNU>
                %637 = vector.bitcast %613 : vector<1xi8> to vector<1xf8E8M0FNU>
                %640 = vector.bitcast %616 : vector<1xi8> to vector<1xf8E8M0FNU>
                %641 = vector.bitcast %617 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.s.setprio 1
                // --- DEPENDENT MFMAs: M0,M1 x N2,N3,N6,N7 (cluster 1 B data) ---
                %647 = vector.extract %636[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %648 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%647[0] * %628) + %arg8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %649 = vector.extract %637[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %650 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%649[0] * %629) + %arg9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %655 = vector.extract %640[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %656 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%655[0] * %632) + %arg12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %657 = vector.extract %641[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %658 = amdgpu.scaled_mfma 16x16x128 (%642[0] * %618) * (%657[0] * %633) + %arg13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %662 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%647[0] * %628) + %arg16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %663 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%649[0] * %629) + %arg17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %666 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%655[0] * %632) + %arg20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %667 = amdgpu.scaled_mfma 16x16x128 (%659[0] * %619) * (%657[0] * %633) + %arg21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                // --- DEPENDENT MFMAs: M2 x all N (cluster 1 A data) ---
                %668 = vector.extract %624[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %669 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%643[0] * %626) + %arg22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %670 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%645[0] * %627) + %arg23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %671 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%647[0] * %628) + %arg24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %672 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%649[0] * %629) + %arg25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %673 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%651[0] * %630) + %arg26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %674 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%653[0] * %631) + %arg27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %675 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%655[0] * %632) + %arg28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %676 = amdgpu.scaled_mfma 16x16x128 (%668[0] * %620) * (%657[0] * %633) + %arg29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                // --- DEPENDENT MFMAs: M3 x all N (cluster 1 A data) ---
                %677 = vector.extract %625[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %678 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%643[0] * %626) + %arg30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %679 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%645[0] * %627) + %arg31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %680 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%647[0] * %628) + %arg32 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %681 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%649[0] * %629) + %arg33 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %682 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%651[0] * %630) + %arg34 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %683 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%653[0] * %631) + %arg35 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %684 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%655[0] * %632) + %arg36 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %685 = amdgpu.scaled_mfma 16x16x128 (%677[0] * %621) * (%657[0] * %633) + %arg37 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.sched.barrier 0
                // --- PHASE 2 SAFE vector.loads: A(M0,M1), Ascale(M0,M1), B(N0,N1,N4,N5), Bscale(N0,N1,N4,N5) ---
                %686 = vector.load %arg38[%62, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %687 = vector.load %arg38[%66, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %690 = vector.load %arg40[%62, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %691 = vector.load %arg40[%66, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %694 = vector.load %arg42[%70, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %695 = vector.load %arg42[%71, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %698 = vector.load %arg42[%74, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %699 = vector.load %arg42[%75, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %702 = vector.load %arg44[%70, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %703 = vector.load %arg44[%71, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %706 = vector.load %arg44[%74, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %707 = vector.load %arg44[%75, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                // --- PHASE 2 SAFE bitcasts ---
                %710 = vector.bitcast %686 : vector<16xi8> to vector<32xf4E2M1FN>
                %711 = vector.bitcast %687 : vector<16xi8> to vector<32xf4E2M1FN>
                %714 = vector.bitcast %690 : vector<1xi8> to vector<1xf8E8M0FNU>
                %715 = vector.bitcast %691 : vector<1xi8> to vector<1xf8E8M0FNU>
                %718 = vector.bitcast %694 : vector<16xi8> to vector<32xf4E2M1FN>
                %719 = vector.bitcast %695 : vector<16xi8> to vector<32xf4E2M1FN>
                %722 = vector.bitcast %698 : vector<16xi8> to vector<32xf4E2M1FN>
                %723 = vector.bitcast %699 : vector<16xi8> to vector<32xf4E2M1FN>
                %726 = vector.bitcast %702 : vector<1xi8> to vector<1xf8E8M0FNU>
                %727 = vector.bitcast %703 : vector<1xi8> to vector<1xf8E8M0FNU>
                %730 = vector.bitcast %706 : vector<1xi8> to vector<1xf8E8M0FNU>
                %731 = vector.bitcast %707 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                // --- PHASE 2 SAFE MFMAs: M0,M1 x N0,N1,N4,N5 (cluster 0 data only) ---
                %734 = vector.extract %714[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %735 = vector.extract %726[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %736 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%735[0] * %718) + %644 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %737 = vector.extract %727[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %738 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%737[0] * %719) + %646 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %743 = vector.extract %730[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %744 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%743[0] * %722) + %652 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %745 = vector.extract %731[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %746 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%745[0] * %723) + %654 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %751 = vector.extract %715[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %752 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%735[0] * %718) + %660 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %753 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%737[0] * %719) + %661 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %756 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%743[0] * %722) + %664 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %757 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%745[0] * %723) + %665 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                // --- PHASE 2 DEPENDENT vector.loads: A(M2,M3), Ascale(M2,M3), B(N2,N3,N6,N7), Bscale(N2,N3,N6,N7) ---
                %688 = vector.load %arg38[%67, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %689 = vector.load %arg38[%68, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %692 = vector.load %arg40[%67, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %693 = vector.load %arg40[%68, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %696 = vector.load %arg42[%72, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %697 = vector.load %arg42[%73, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %700 = vector.load %arg42[%76, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %701 = vector.load %arg42[%77, %80] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %704 = vector.load %arg44[%72, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %705 = vector.load %arg44[%73, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %708 = vector.load %arg44[%76, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %709 = vector.load %arg44[%77, %82] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                // --- PHASE 2 DEPENDENT bitcasts ---
                %712 = vector.bitcast %688 : vector<16xi8> to vector<32xf4E2M1FN>
                %713 = vector.bitcast %689 : vector<16xi8> to vector<32xf4E2M1FN>
                %716 = vector.bitcast %692 : vector<1xi8> to vector<1xf8E8M0FNU>
                %717 = vector.bitcast %693 : vector<1xi8> to vector<1xf8E8M0FNU>
                %720 = vector.bitcast %696 : vector<16xi8> to vector<32xf4E2M1FN>
                %721 = vector.bitcast %697 : vector<16xi8> to vector<32xf4E2M1FN>
                %724 = vector.bitcast %700 : vector<16xi8> to vector<32xf4E2M1FN>
                %725 = vector.bitcast %701 : vector<16xi8> to vector<32xf4E2M1FN>
                %728 = vector.bitcast %704 : vector<1xi8> to vector<1xf8E8M0FNU>
                %729 = vector.bitcast %705 : vector<1xi8> to vector<1xf8E8M0FNU>
                %732 = vector.bitcast %708 : vector<1xi8> to vector<1xf8E8M0FNU>
                %733 = vector.bitcast %709 : vector<1xi8> to vector<1xf8E8M0FNU>
                rocdl.s.setprio 1
                // --- PHASE 2 DEPENDENT MFMAs: M0,M1 x N2,N3,N6,N7 (cluster 1 B data) ---
                %739 = vector.extract %728[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %740 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%739[0] * %720) + %648 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %741 = vector.extract %729[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %742 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%741[0] * %721) + %650 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %747 = vector.extract %732[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %748 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%747[0] * %724) + %656 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %749 = vector.extract %733[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %750 = amdgpu.scaled_mfma 16x16x128 (%734[0] * %710) * (%749[0] * %725) + %658 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %754 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%739[0] * %720) + %662 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %755 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%741[0] * %721) + %663 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %758 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%747[0] * %724) + %666 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %759 = amdgpu.scaled_mfma 16x16x128 (%751[0] * %711) * (%749[0] * %725) + %667 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                // --- PHASE 2 DEPENDENT MFMAs: (cluster 1 A data) ---
                %760 = vector.extract %716[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %761 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%735[0] * %718) + %669 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %762 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%737[0] * %719) + %670 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %763 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%739[0] * %720) + %671 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %764 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%741[0] * %721) + %672 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %765 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%743[0] * %722) + %673 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %766 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%745[0] * %723) + %674 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %767 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%747[0] * %724) + %675 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %768 = amdgpu.scaled_mfma 16x16x128 (%760[0] * %712) * (%749[0] * %725) + %676 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                // --- PHASE 2 DEPENDENT MFMAs:  (cluster 1 A data) ---
                %769 = vector.extract %717[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %770 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%735[0] * %718) + %678 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %771 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%737[0] * %719) + %679 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %772 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%739[0] * %720) + %680 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %773 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%741[0] * %721) + %681 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %774 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%743[0] * %722) + %682 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %775 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%745[0] * %723) + %683 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %776 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%747[0] * %724) + %684 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %777 = amdgpu.scaled_mfma 16x16x128 (%769[0] * %713) * (%749[0] * %725) + %685 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                scf.yield %736, %738, %740, %742, %744, %746, %748, %750, %752, %753, %754, %755, %756, %757, %758, %759, %761, %762, %763, %764, %765, %766, %767, %768, %770, %771, %772, %773, %774, %775, %776, %777, %arg39, %arg38, %arg41, %arg40, %arg43, %arg42, %arg45, %arg44 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>, memref<256x8xi8, #gpu.address_space<workgroup>>
                }
                scf.if %61 {
                rocdl.s.barrier
                }
                amdgpu.lds_barrier
                %84 = affine.apply #map23()[%thread_id_x, %thread_id_y]
                %85 = affine.apply #map22()[%thread_id_x]
                %86 = vector.load %83#38[%84, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %87 = arith.xori %33, %c1 : index
                %88 = affine.apply #map32()[%thread_id_x, %87]
                %89 = vector.load %83#38[%84, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %90 = affine.apply #map24()[%thread_id_x, %thread_id_y]
                %91 = vector.load %83#38[%90, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %92 = vector.load %83#38[%90, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %93 = affine.apply #map25()[%thread_id_x, %thread_id_y]
                %94 = vector.load %83#38[%93, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %95 = vector.load %83#38[%93, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %96 = affine.apply #map26()[%thread_id_x, %thread_id_y]
                %97 = vector.load %83#38[%96, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %98 = vector.load %83#38[%96, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %99 = affine.apply #map27()[%thread_id_x, %thread_id_y]
                %100 = vector.load %83#38[%99, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %101 = vector.load %83#38[%99, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %102 = affine.apply #map28()[%thread_id_x, %thread_id_y]
                %103 = vector.load %83#38[%102, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %104 = vector.load %83#38[%102, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %105 = affine.apply #map29()[%thread_id_x, %thread_id_y]
                %106 = vector.load %83#38[%105, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %107 = vector.load %83#38[%105, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %108 = affine.apply #map30()[%thread_id_x, %thread_id_y]
                %109 = vector.load %83#38[%108, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %110 = vector.load %83#38[%108, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %111 = affine.apply #map18()[%thread_id_x]
                %112 = arith.xori %111, %7 : index
                %113 = affine.apply #map3()[%112]
                %114 = vector.load %83#36[%84, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %115 = affine.apply #map31()[%thread_id_x]
                %116 = arith.xori %115, %7 : index
                %117 = affine.apply #map3()[%116]
                %118 = vector.load %83#36[%84, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %119 = vector.load %83#36[%90, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %120 = vector.load %83#36[%90, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %121 = vector.load %83#36[%93, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %122 = vector.load %83#36[%93, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %123 = vector.load %83#36[%96, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %124 = vector.load %83#36[%96, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %125 = vector.load %83#36[%99, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %126 = vector.load %83#36[%99, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %127 = vector.load %83#36[%102, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %128 = vector.load %83#36[%102, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %129 = vector.load %83#36[%105, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %130 = vector.load %83#36[%105, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %131 = vector.load %83#36[%108, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %132 = vector.load %83#36[%108, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %133 = affine.apply #map17()[%thread_id_x]
                %134 = vector.load %83#34[%133, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %135 = vector.load %83#34[%133, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %136 = affine.apply #map19()[%thread_id_x]
                %137 = vector.load %83#34[%136, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %138 = vector.load %83#34[%136, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %139 = affine.apply #map20()[%thread_id_x]
                %140 = vector.load %83#34[%139, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %141 = vector.load %83#34[%139, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %142 = affine.apply #map21()[%thread_id_x]
                %143 = vector.load %83#34[%142, %85] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %144 = vector.load %83#34[%142, %88] : memref<256x8xi8, #gpu.address_space<workgroup>>, vector<1xi8>
                %145 = vector.load %83#32[%133, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %146 = vector.load %83#32[%133, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %147 = vector.load %83#32[%136, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %148 = vector.load %83#32[%136, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %149 = vector.load %83#32[%139, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %150 = vector.load %83#32[%139, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %151 = vector.load %83#32[%142, %113] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %152 = vector.load %83#32[%142, %117] : memref<256x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %153 = vector.bitcast %145 : vector<16xi8> to vector<32xf4E2M1FN>
                %154 = vector.bitcast %146 : vector<16xi8> to vector<32xf4E2M1FN>
                %155 = vector.bitcast %147 : vector<16xi8> to vector<32xf4E2M1FN>
                %156 = vector.bitcast %148 : vector<16xi8> to vector<32xf4E2M1FN>
                %157 = vector.bitcast %149 : vector<16xi8> to vector<32xf4E2M1FN>
                %158 = vector.bitcast %150 : vector<16xi8> to vector<32xf4E2M1FN>
                %159 = vector.bitcast %151 : vector<16xi8> to vector<32xf4E2M1FN>
                %160 = vector.bitcast %152 : vector<16xi8> to vector<32xf4E2M1FN>
                %161 = vector.bitcast %134 : vector<1xi8> to vector<1xf8E8M0FNU>
                %162 = vector.bitcast %135 : vector<1xi8> to vector<1xf8E8M0FNU>
                %163 = vector.bitcast %137 : vector<1xi8> to vector<1xf8E8M0FNU>
                %164 = vector.bitcast %138 : vector<1xi8> to vector<1xf8E8M0FNU>
                %165 = vector.bitcast %140 : vector<1xi8> to vector<1xf8E8M0FNU>
                %166 = vector.bitcast %141 : vector<1xi8> to vector<1xf8E8M0FNU>
                %167 = vector.bitcast %143 : vector<1xi8> to vector<1xf8E8M0FNU>
                %168 = vector.bitcast %144 : vector<1xi8> to vector<1xf8E8M0FNU>
                %169 = vector.bitcast %114 : vector<16xi8> to vector<32xf4E2M1FN>
                %170 = vector.bitcast %118 : vector<16xi8> to vector<32xf4E2M1FN>
                %171 = vector.bitcast %119 : vector<16xi8> to vector<32xf4E2M1FN>
                %172 = vector.bitcast %120 : vector<16xi8> to vector<32xf4E2M1FN>
                %173 = vector.bitcast %121 : vector<16xi8> to vector<32xf4E2M1FN>
                %174 = vector.bitcast %122 : vector<16xi8> to vector<32xf4E2M1FN>
                %175 = vector.bitcast %123 : vector<16xi8> to vector<32xf4E2M1FN>
                %176 = vector.bitcast %124 : vector<16xi8> to vector<32xf4E2M1FN>
                %177 = vector.bitcast %125 : vector<16xi8> to vector<32xf4E2M1FN>
                %178 = vector.bitcast %126 : vector<16xi8> to vector<32xf4E2M1FN>
                %179 = vector.bitcast %127 : vector<16xi8> to vector<32xf4E2M1FN>
                %180 = vector.bitcast %128 : vector<16xi8> to vector<32xf4E2M1FN>
                %181 = vector.bitcast %129 : vector<16xi8> to vector<32xf4E2M1FN>
                %182 = vector.bitcast %130 : vector<16xi8> to vector<32xf4E2M1FN>
                %183 = vector.bitcast %131 : vector<16xi8> to vector<32xf4E2M1FN>
                %184 = vector.bitcast %132 : vector<16xi8> to vector<32xf4E2M1FN>
                %185 = vector.bitcast %86 : vector<1xi8> to vector<1xf8E8M0FNU>
                %186 = vector.bitcast %89 : vector<1xi8> to vector<1xf8E8M0FNU>
                %187 = vector.bitcast %91 : vector<1xi8> to vector<1xf8E8M0FNU>
                %188 = vector.bitcast %92 : vector<1xi8> to vector<1xf8E8M0FNU>
                %189 = vector.bitcast %94 : vector<1xi8> to vector<1xf8E8M0FNU>
                %190 = vector.bitcast %95 : vector<1xi8> to vector<1xf8E8M0FNU>
                %191 = vector.bitcast %97 : vector<1xi8> to vector<1xf8E8M0FNU>
                %192 = vector.bitcast %98 : vector<1xi8> to vector<1xf8E8M0FNU>
                %193 = vector.bitcast %100 : vector<1xi8> to vector<1xf8E8M0FNU>
                %194 = vector.bitcast %101 : vector<1xi8> to vector<1xf8E8M0FNU>
                %195 = vector.bitcast %103 : vector<1xi8> to vector<1xf8E8M0FNU>
                %196 = vector.bitcast %104 : vector<1xi8> to vector<1xf8E8M0FNU>
                %197 = vector.bitcast %106 : vector<1xi8> to vector<1xf8E8M0FNU>
                %198 = vector.bitcast %107 : vector<1xi8> to vector<1xf8E8M0FNU>
                %199 = vector.bitcast %109 : vector<1xi8> to vector<1xf8E8M0FNU>
                %200 = vector.bitcast %110 : vector<1xi8> to vector<1xf8E8M0FNU>
                %201 = vector.extract %161[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %202 = vector.extract %185[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %203 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%202[0] * %169) + %83#0 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %204 = vector.extract %162[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %205 = vector.extract %186[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %206 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%205[0] * %170) + %203 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %207 = vector.extract %187[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %208 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%207[0] * %171) + %83#1 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %209 = vector.extract %188[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %210 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%209[0] * %172) + %208 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %211 = vector.extract %189[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %212 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%211[0] * %173) + %83#2 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %213 = vector.extract %190[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %214 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%213[0] * %174) + %212 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %215 = vector.extract %191[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %216 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%215[0] * %175) + %83#3 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %217 = vector.extract %192[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %218 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%217[0] * %176) + %216 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %219 = vector.extract %193[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %220 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%219[0] * %177) + %83#4 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %221 = vector.extract %194[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %222 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%221[0] * %178) + %220 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %223 = vector.extract %195[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %224 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%223[0] * %179) + %83#5 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %225 = vector.extract %196[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %226 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%225[0] * %180) + %224 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %227 = vector.extract %197[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %228 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%227[0] * %181) + %83#6 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %229 = vector.extract %198[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %230 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%229[0] * %182) + %228 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %231 = vector.extract %199[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %232 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %153) * (%231[0] * %183) + %83#7 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %233 = vector.extract %200[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %234 = amdgpu.scaled_mfma 16x16x128 (%204[0] * %154) * (%233[0] * %184) + %232 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %235 = vector.extract %163[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %236 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%202[0] * %169) + %83#8 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %237 = vector.extract %164[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %238 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%205[0] * %170) + %236 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %239 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%207[0] * %171) + %83#9 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %240 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%209[0] * %172) + %239 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %241 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%211[0] * %173) + %83#10 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %242 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%213[0] * %174) + %241 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %243 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%215[0] * %175) + %83#11 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %244 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%217[0] * %176) + %243 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %245 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%219[0] * %177) + %83#12 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %246 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%221[0] * %178) + %245 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %247 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%223[0] * %179) + %83#13 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %248 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%225[0] * %180) + %247 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %249 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%227[0] * %181) + %83#14 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %250 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%229[0] * %182) + %249 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %251 = amdgpu.scaled_mfma 16x16x128 (%235[0] * %155) * (%231[0] * %183) + %83#15 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %252 = amdgpu.scaled_mfma 16x16x128 (%237[0] * %156) * (%233[0] * %184) + %251 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %253 = vector.extract %165[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %254 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%202[0] * %169) + %83#16 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %255 = vector.extract %166[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %256 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%205[0] * %170) + %254 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %257 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%207[0] * %171) + %83#17 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %258 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%209[0] * %172) + %257 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %259 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%211[0] * %173) + %83#18 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %260 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%213[0] * %174) + %259 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %261 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%215[0] * %175) + %83#19 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %262 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%217[0] * %176) + %261 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %263 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%219[0] * %177) + %83#20 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %264 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%221[0] * %178) + %263 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %265 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%223[0] * %179) + %83#21 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %266 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%225[0] * %180) + %265 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %267 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%227[0] * %181) + %83#22 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %268 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%229[0] * %182) + %267 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %269 = amdgpu.scaled_mfma 16x16x128 (%253[0] * %157) * (%231[0] * %183) + %83#23 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %270 = amdgpu.scaled_mfma 16x16x128 (%255[0] * %158) * (%233[0] * %184) + %269 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %271 = vector.extract %167[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %272 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%202[0] * %169) + %83#24 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %273 = vector.extract %168[0] : f8E8M0FNU from vector<1xf8E8M0FNU>
                %274 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%205[0] * %170) + %272 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %275 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%207[0] * %171) + %83#25 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %276 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%209[0] * %172) + %275 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %277 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%211[0] * %173) + %83#26 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %278 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%213[0] * %174) + %277 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %279 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%215[0] * %175) + %83#27 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %280 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%217[0] * %176) + %279 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %281 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%219[0] * %177) + %83#28 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %282 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%221[0] * %178) + %281 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %283 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%223[0] * %179) + %83#29 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %284 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%225[0] * %180) + %283 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %285 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%227[0] * %181) + %83#30 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %286 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%229[0] * %182) + %285 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %287 = amdgpu.scaled_mfma 16x16x128 (%271[0] * %159) * (%231[0] * %183) + %83#31 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %288 = amdgpu.scaled_mfma 16x16x128 (%273[0] * %160) * (%233[0] * %184) + %287 : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
                %289 = vector.extract_strided_slice %206 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %290 = affine.apply #map35()[%block_id_x]
                %291 = affine.apply #map35()[%block_id_y]
                %292 = affine.apply #map36()[%thread_id_x]
                %293 = arith.muli %290, %c57344 overflow<nsw> : index
                %294 = arith.muli %292, %c57344 overflow<nsw> : index
                %295 = arith.addi %293, %291 overflow<nsw> : index
                %296 = arith.addi %294, %84 overflow<nsw> : index
                %reinterpret_cast_13 = memref.reinterpret_cast %4 to offset: [%295], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
                %cast_14 = memref.cast %reinterpret_cast_13 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
                %297 = amdgpu.fat_raw_buffer_cast %cast_14 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
                vector.store %289, %297[%296] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %298 = vector.extract_strided_slice %206 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %299 = affine.apply #map37()[%thread_id_x]
                %300 = arith.muli %299, %c57344 overflow<nsw> : index
                %301 = arith.addi %300, %84 overflow<nsw> : index
                vector.store %298, %297[%301] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %302 = vector.extract_strided_slice %206 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %303 = affine.apply #map38()[%thread_id_x]
                %304 = arith.muli %303, %c57344 overflow<nsw> : index
                %305 = arith.addi %304, %84 overflow<nsw> : index
                vector.store %302, %297[%305] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %306 = vector.extract_strided_slice %206 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %307 = affine.apply #map39()[%thread_id_x]
                %308 = arith.muli %307, %c57344 overflow<nsw> : index
                %309 = arith.addi %308, %84 overflow<nsw> : index
                vector.store %306, %297[%309] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %310 = vector.extract_strided_slice %210 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %311 = arith.addi %294, %90 overflow<nsw> : index
                vector.store %310, %297[%311] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %312 = vector.extract_strided_slice %210 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %313 = arith.addi %300, %90 overflow<nsw> : index
                vector.store %312, %297[%313] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %314 = vector.extract_strided_slice %210 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %315 = arith.addi %304, %90 overflow<nsw> : index
                vector.store %314, %297[%315] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %316 = vector.extract_strided_slice %210 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %317 = arith.addi %308, %90 overflow<nsw> : index
                vector.store %316, %297[%317] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %318 = vector.extract_strided_slice %214 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %319 = arith.addi %294, %93 overflow<nsw> : index
                vector.store %318, %297[%319] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %320 = vector.extract_strided_slice %214 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %321 = arith.addi %300, %93 overflow<nsw> : index
                vector.store %320, %297[%321] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %322 = vector.extract_strided_slice %214 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %323 = arith.addi %304, %93 overflow<nsw> : index
                vector.store %322, %297[%323] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %324 = vector.extract_strided_slice %214 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %325 = arith.addi %308, %93 overflow<nsw> : index
                vector.store %324, %297[%325] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %326 = vector.extract_strided_slice %218 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %327 = arith.addi %294, %96 overflow<nsw> : index
                vector.store %326, %297[%327] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %328 = vector.extract_strided_slice %218 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %329 = arith.addi %300, %96 overflow<nsw> : index
                vector.store %328, %297[%329] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %330 = vector.extract_strided_slice %218 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %331 = arith.addi %304, %96 overflow<nsw> : index
                vector.store %330, %297[%331] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %332 = vector.extract_strided_slice %218 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %333 = arith.addi %308, %96 overflow<nsw> : index
                vector.store %332, %297[%333] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %334 = vector.extract_strided_slice %222 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %335 = arith.addi %294, %99 overflow<nsw> : index
                vector.store %334, %297[%335] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %336 = vector.extract_strided_slice %222 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %337 = arith.addi %300, %99 overflow<nsw> : index
                vector.store %336, %297[%337] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %338 = vector.extract_strided_slice %222 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %339 = arith.addi %304, %99 overflow<nsw> : index
                vector.store %338, %297[%339] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %340 = vector.extract_strided_slice %222 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %341 = arith.addi %308, %99 overflow<nsw> : index
                vector.store %340, %297[%341] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %342 = vector.extract_strided_slice %226 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %343 = arith.addi %294, %102 overflow<nsw> : index
                vector.store %342, %297[%343] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %344 = vector.extract_strided_slice %226 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %345 = arith.addi %300, %102 overflow<nsw> : index
                vector.store %344, %297[%345] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %346 = vector.extract_strided_slice %226 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %347 = arith.addi %304, %102 overflow<nsw> : index
                vector.store %346, %297[%347] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %348 = vector.extract_strided_slice %226 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %349 = arith.addi %308, %102 overflow<nsw> : index
                vector.store %348, %297[%349] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %350 = vector.extract_strided_slice %230 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %351 = arith.addi %294, %105 overflow<nsw> : index
                vector.store %350, %297[%351] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %352 = vector.extract_strided_slice %230 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %353 = arith.addi %300, %105 overflow<nsw> : index
                vector.store %352, %297[%353] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %354 = vector.extract_strided_slice %230 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %355 = arith.addi %304, %105 overflow<nsw> : index
                vector.store %354, %297[%355] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %356 = vector.extract_strided_slice %230 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %357 = arith.addi %308, %105 overflow<nsw> : index
                vector.store %356, %297[%357] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %358 = vector.extract_strided_slice %234 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %359 = arith.addi %294, %108 overflow<nsw> : index
                vector.store %358, %297[%359] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %360 = vector.extract_strided_slice %234 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %361 = arith.addi %300, %108 overflow<nsw> : index
                vector.store %360, %297[%361] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %362 = vector.extract_strided_slice %234 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %363 = arith.addi %304, %108 overflow<nsw> : index
                vector.store %362, %297[%363] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %364 = vector.extract_strided_slice %234 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %365 = arith.addi %308, %108 overflow<nsw> : index
                vector.store %364, %297[%365] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %366 = vector.extract_strided_slice %238 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %367 = affine.apply #map40()[%thread_id_x]
                %368 = arith.muli %367, %c57344 overflow<nsw> : index
                %369 = arith.addi %368, %84 overflow<nsw> : index
                vector.store %366, %297[%369] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %370 = vector.extract_strided_slice %238 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %371 = affine.apply #map41()[%thread_id_x]
                %372 = arith.muli %371, %c57344 overflow<nsw> : index
                %373 = arith.addi %372, %84 overflow<nsw> : index
                vector.store %370, %297[%373] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %374 = vector.extract_strided_slice %238 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %375 = affine.apply #map42()[%thread_id_x]
                %376 = arith.muli %375, %c57344 overflow<nsw> : index
                %377 = arith.addi %376, %84 overflow<nsw> : index
                vector.store %374, %297[%377] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %378 = vector.extract_strided_slice %238 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %379 = affine.apply #map43()[%thread_id_x]
                %380 = arith.muli %379, %c57344 overflow<nsw> : index
                %381 = arith.addi %380, %84 overflow<nsw> : index
                vector.store %378, %297[%381] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %382 = vector.extract_strided_slice %240 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %383 = arith.addi %368, %90 overflow<nsw> : index
                vector.store %382, %297[%383] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %384 = vector.extract_strided_slice %240 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %385 = arith.addi %372, %90 overflow<nsw> : index
                vector.store %384, %297[%385] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %386 = vector.extract_strided_slice %240 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %387 = arith.addi %376, %90 overflow<nsw> : index
                vector.store %386, %297[%387] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %388 = vector.extract_strided_slice %240 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %389 = arith.addi %380, %90 overflow<nsw> : index
                vector.store %388, %297[%389] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %390 = vector.extract_strided_slice %242 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %391 = arith.addi %368, %93 overflow<nsw> : index
                vector.store %390, %297[%391] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %392 = vector.extract_strided_slice %242 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %393 = arith.addi %372, %93 overflow<nsw> : index
                vector.store %392, %297[%393] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %394 = vector.extract_strided_slice %242 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %395 = arith.addi %376, %93 overflow<nsw> : index
                vector.store %394, %297[%395] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %396 = vector.extract_strided_slice %242 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %397 = arith.addi %380, %93 overflow<nsw> : index
                vector.store %396, %297[%397] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %398 = vector.extract_strided_slice %244 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %399 = arith.addi %368, %96 overflow<nsw> : index
                vector.store %398, %297[%399] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %400 = vector.extract_strided_slice %244 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %401 = arith.addi %372, %96 overflow<nsw> : index
                vector.store %400, %297[%401] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %402 = vector.extract_strided_slice %244 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %403 = arith.addi %376, %96 overflow<nsw> : index
                vector.store %402, %297[%403] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %404 = vector.extract_strided_slice %244 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %405 = arith.addi %380, %96 overflow<nsw> : index
                vector.store %404, %297[%405] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %406 = vector.extract_strided_slice %246 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %407 = arith.addi %368, %99 overflow<nsw> : index
                vector.store %406, %297[%407] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %408 = vector.extract_strided_slice %246 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %409 = arith.addi %372, %99 overflow<nsw> : index
                vector.store %408, %297[%409] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %410 = vector.extract_strided_slice %246 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %411 = arith.addi %376, %99 overflow<nsw> : index
                vector.store %410, %297[%411] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %412 = vector.extract_strided_slice %246 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %413 = arith.addi %380, %99 overflow<nsw> : index
                vector.store %412, %297[%413] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %414 = vector.extract_strided_slice %248 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %415 = arith.addi %368, %102 overflow<nsw> : index
                vector.store %414, %297[%415] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %416 = vector.extract_strided_slice %248 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %417 = arith.addi %372, %102 overflow<nsw> : index
                vector.store %416, %297[%417] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %418 = vector.extract_strided_slice %248 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %419 = arith.addi %376, %102 overflow<nsw> : index
                vector.store %418, %297[%419] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %420 = vector.extract_strided_slice %248 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %421 = arith.addi %380, %102 overflow<nsw> : index
                vector.store %420, %297[%421] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %422 = vector.extract_strided_slice %250 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %423 = arith.addi %368, %105 overflow<nsw> : index
                vector.store %422, %297[%423] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %424 = vector.extract_strided_slice %250 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %425 = arith.addi %372, %105 overflow<nsw> : index
                vector.store %424, %297[%425] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %426 = vector.extract_strided_slice %250 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %427 = arith.addi %376, %105 overflow<nsw> : index
                vector.store %426, %297[%427] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %428 = vector.extract_strided_slice %250 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %429 = arith.addi %380, %105 overflow<nsw> : index
                vector.store %428, %297[%429] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %430 = vector.extract_strided_slice %252 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %431 = arith.addi %368, %108 overflow<nsw> : index
                vector.store %430, %297[%431] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %432 = vector.extract_strided_slice %252 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %433 = arith.addi %372, %108 overflow<nsw> : index
                vector.store %432, %297[%433] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %434 = vector.extract_strided_slice %252 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %435 = arith.addi %376, %108 overflow<nsw> : index
                vector.store %434, %297[%435] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %436 = vector.extract_strided_slice %252 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %437 = arith.addi %380, %108 overflow<nsw> : index
                vector.store %436, %297[%437] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %438 = vector.extract_strided_slice %256 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %439 = affine.apply #map44()[%thread_id_x]
                %440 = arith.muli %439, %c57344 overflow<nsw> : index
                %441 = arith.addi %440, %84 overflow<nsw> : index
                vector.store %438, %297[%441] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %442 = vector.extract_strided_slice %256 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %443 = affine.apply #map45()[%thread_id_x]
                %444 = arith.muli %443, %c57344 overflow<nsw> : index
                %445 = arith.addi %444, %84 overflow<nsw> : index
                vector.store %442, %297[%445] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %446 = vector.extract_strided_slice %256 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %447 = affine.apply #map46()[%thread_id_x]
                %448 = arith.muli %447, %c57344 overflow<nsw> : index
                %449 = arith.addi %448, %84 overflow<nsw> : index
                vector.store %446, %297[%449] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %450 = vector.extract_strided_slice %256 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %451 = affine.apply #map47()[%thread_id_x]
                %452 = arith.muli %451, %c57344 overflow<nsw> : index
                %453 = arith.addi %452, %84 overflow<nsw> : index
                vector.store %450, %297[%453] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %454 = vector.extract_strided_slice %258 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %455 = arith.addi %440, %90 overflow<nsw> : index
                vector.store %454, %297[%455] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %456 = vector.extract_strided_slice %258 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %457 = arith.addi %444, %90 overflow<nsw> : index
                vector.store %456, %297[%457] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %458 = vector.extract_strided_slice %258 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %459 = arith.addi %448, %90 overflow<nsw> : index
                vector.store %458, %297[%459] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %460 = vector.extract_strided_slice %258 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %461 = arith.addi %452, %90 overflow<nsw> : index
                vector.store %460, %297[%461] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %462 = vector.extract_strided_slice %260 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %463 = arith.addi %440, %93 overflow<nsw> : index
                vector.store %462, %297[%463] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %464 = vector.extract_strided_slice %260 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %465 = arith.addi %444, %93 overflow<nsw> : index
                vector.store %464, %297[%465] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %466 = vector.extract_strided_slice %260 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %467 = arith.addi %448, %93 overflow<nsw> : index
                vector.store %466, %297[%467] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %468 = vector.extract_strided_slice %260 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %469 = arith.addi %452, %93 overflow<nsw> : index
                vector.store %468, %297[%469] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %470 = vector.extract_strided_slice %262 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %471 = arith.addi %440, %96 overflow<nsw> : index
                vector.store %470, %297[%471] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %472 = vector.extract_strided_slice %262 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %473 = arith.addi %444, %96 overflow<nsw> : index
                vector.store %472, %297[%473] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %474 = vector.extract_strided_slice %262 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %475 = arith.addi %448, %96 overflow<nsw> : index
                vector.store %474, %297[%475] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %476 = vector.extract_strided_slice %262 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %477 = arith.addi %452, %96 overflow<nsw> : index
                vector.store %476, %297[%477] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %478 = vector.extract_strided_slice %264 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %479 = arith.addi %440, %99 overflow<nsw> : index
                vector.store %478, %297[%479] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %480 = vector.extract_strided_slice %264 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %481 = arith.addi %444, %99 overflow<nsw> : index
                vector.store %480, %297[%481] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %482 = vector.extract_strided_slice %264 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %483 = arith.addi %448, %99 overflow<nsw> : index
                vector.store %482, %297[%483] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %484 = vector.extract_strided_slice %264 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %485 = arith.addi %452, %99 overflow<nsw> : index
                vector.store %484, %297[%485] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %486 = vector.extract_strided_slice %266 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %487 = arith.addi %440, %102 overflow<nsw> : index
                vector.store %486, %297[%487] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %488 = vector.extract_strided_slice %266 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %489 = arith.addi %444, %102 overflow<nsw> : index
                vector.store %488, %297[%489] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %490 = vector.extract_strided_slice %266 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %491 = arith.addi %448, %102 overflow<nsw> : index
                vector.store %490, %297[%491] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %492 = vector.extract_strided_slice %266 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %493 = arith.addi %452, %102 overflow<nsw> : index
                vector.store %492, %297[%493] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %494 = vector.extract_strided_slice %268 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %495 = arith.addi %440, %105 overflow<nsw> : index
                vector.store %494, %297[%495] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %496 = vector.extract_strided_slice %268 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %497 = arith.addi %444, %105 overflow<nsw> : index
                vector.store %496, %297[%497] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %498 = vector.extract_strided_slice %268 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %499 = arith.addi %448, %105 overflow<nsw> : index
                vector.store %498, %297[%499] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %500 = vector.extract_strided_slice %268 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %501 = arith.addi %452, %105 overflow<nsw> : index
                vector.store %500, %297[%501] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %502 = vector.extract_strided_slice %270 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %503 = arith.addi %440, %108 overflow<nsw> : index
                vector.store %502, %297[%503] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %504 = vector.extract_strided_slice %270 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %505 = arith.addi %444, %108 overflow<nsw> : index
                vector.store %504, %297[%505] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %506 = vector.extract_strided_slice %270 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %507 = arith.addi %448, %108 overflow<nsw> : index
                vector.store %506, %297[%507] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %508 = vector.extract_strided_slice %270 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %509 = arith.addi %452, %108 overflow<nsw> : index
                vector.store %508, %297[%509] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %510 = vector.extract_strided_slice %274 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %511 = affine.apply #map48()[%thread_id_x]
                %512 = arith.muli %511, %c57344 overflow<nsw> : index
                %513 = arith.addi %512, %84 overflow<nsw> : index
                vector.store %510, %297[%513] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %514 = vector.extract_strided_slice %274 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %515 = affine.apply #map49()[%thread_id_x]
                %516 = arith.muli %515, %c57344 overflow<nsw> : index
                %517 = arith.addi %516, %84 overflow<nsw> : index
                vector.store %514, %297[%517] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %518 = vector.extract_strided_slice %274 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %519 = affine.apply #map50()[%thread_id_x]
                %520 = arith.muli %519, %c57344 overflow<nsw> : index
                %521 = arith.addi %520, %84 overflow<nsw> : index
                vector.store %518, %297[%521] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %522 = vector.extract_strided_slice %274 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %523 = affine.apply #map51()[%thread_id_x]
                %524 = arith.muli %523, %c57344 overflow<nsw> : index
                %525 = arith.addi %524, %84 overflow<nsw> : index
                vector.store %522, %297[%525] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %526 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %527 = arith.addi %512, %90 overflow<nsw> : index
                vector.store %526, %297[%527] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %528 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %529 = arith.addi %516, %90 overflow<nsw> : index
                vector.store %528, %297[%529] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %530 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %531 = arith.addi %520, %90 overflow<nsw> : index
                vector.store %530, %297[%531] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %532 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %533 = arith.addi %524, %90 overflow<nsw> : index
                vector.store %532, %297[%533] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %534 = vector.extract_strided_slice %278 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %535 = arith.addi %512, %93 overflow<nsw> : index
                vector.store %534, %297[%535] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %536 = vector.extract_strided_slice %278 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %537 = arith.addi %516, %93 overflow<nsw> : index
                vector.store %536, %297[%537] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %538 = vector.extract_strided_slice %278 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %539 = arith.addi %520, %93 overflow<nsw> : index
                vector.store %538, %297[%539] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %540 = vector.extract_strided_slice %278 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %541 = arith.addi %524, %93 overflow<nsw> : index
                vector.store %540, %297[%541] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %542 = vector.extract_strided_slice %280 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %543 = arith.addi %512, %96 overflow<nsw> : index
                vector.store %542, %297[%543] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %544 = vector.extract_strided_slice %280 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %545 = arith.addi %516, %96 overflow<nsw> : index
                vector.store %544, %297[%545] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %546 = vector.extract_strided_slice %280 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %547 = arith.addi %520, %96 overflow<nsw> : index
                vector.store %546, %297[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %548 = vector.extract_strided_slice %280 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %549 = arith.addi %524, %96 overflow<nsw> : index
                vector.store %548, %297[%549] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %550 = vector.extract_strided_slice %282 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %551 = arith.addi %512, %99 overflow<nsw> : index
                vector.store %550, %297[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %552 = vector.extract_strided_slice %282 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %553 = arith.addi %516, %99 overflow<nsw> : index
                vector.store %552, %297[%553] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %554 = vector.extract_strided_slice %282 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %555 = arith.addi %520, %99 overflow<nsw> : index
                vector.store %554, %297[%555] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %556 = vector.extract_strided_slice %282 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %557 = arith.addi %524, %99 overflow<nsw> : index
                vector.store %556, %297[%557] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %558 = vector.extract_strided_slice %284 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %559 = arith.addi %512, %102 overflow<nsw> : index
                vector.store %558, %297[%559] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %560 = vector.extract_strided_slice %284 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %561 = arith.addi %516, %102 overflow<nsw> : index
                vector.store %560, %297[%561] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %562 = vector.extract_strided_slice %284 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %563 = arith.addi %520, %102 overflow<nsw> : index
                vector.store %562, %297[%563] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %564 = vector.extract_strided_slice %284 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %565 = arith.addi %524, %102 overflow<nsw> : index
                vector.store %564, %297[%565] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %566 = vector.extract_strided_slice %286 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %567 = arith.addi %512, %105 overflow<nsw> : index
                vector.store %566, %297[%567] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %568 = vector.extract_strided_slice %286 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %569 = arith.addi %516, %105 overflow<nsw> : index
                vector.store %568, %297[%569] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %570 = vector.extract_strided_slice %286 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %571 = arith.addi %520, %105 overflow<nsw> : index
                vector.store %570, %297[%571] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %572 = vector.extract_strided_slice %286 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %573 = arith.addi %524, %105 overflow<nsw> : index
                vector.store %572, %297[%573] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %574 = vector.extract_strided_slice %288 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %575 = arith.addi %512, %108 overflow<nsw> : index
                vector.store %574, %297[%575] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %576 = vector.extract_strided_slice %288 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %577 = arith.addi %516, %108 overflow<nsw> : index
                vector.store %576, %297[%577] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %578 = vector.extract_strided_slice %288 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %579 = arith.addi %520, %108 overflow<nsw> : index
                vector.store %578, %297[%579] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                %580 = vector.extract_strided_slice %288 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %581 = arith.addi %524, %108 overflow<nsw> : index
                vector.store %580, %297[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<4096x8192xi8>
            %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<4096x512xi8>
            %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<57344x8192xi8>
            %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<57344x512xi8>
            %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<4096x57344xf32>
            %5 = flow.dispatch @gemm::@gemm(%0, %1, %2, %3, %4) : (tensor<4096x8192xi8>, tensor<4096x512xi8>, tensor<57344x8192xi8>, tensor<57344x512xi8>, tensor<4096x57344xf32>) -> %4
            %6 = hal.tensor.barrier join(%5 : tensor<4096x57344xf32>) => %arg6 : !hal.fence
            %7 = hal.tensor.export %6 : tensor<4096x57344xf32> -> !hal.buffer_view
            return %7 : !hal.buffer_view
        }
        }
    """
    gemm, options = get_tagged_mxfp4_gemm(shape, block, num_waves=8)

    schedule = get_mxfp4_dbuf_schedule(use_stagger=True)

    options.use_buffer_ops = True
    options.specialize = True
    # options.override_mlir=mlir_claude_rescheduled2

    options.print_ir_after = "all" if is_debug else []
    # options.print_ir_after = "all"
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    print(gemm.asm)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 8-wave test passed!")


def test_dbuf_8wave_mxfp_gemm_shuffle(
    is_debug=False, shape=(4096, 57344, 16384), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 8 waves, with stagger."""

    gemm, options = get_preshuffle_kernel(shape, block)

    schedule = get_mxfp4_dbuf_schedule_shuffle(use_stagger=True)

    options.use_buffer_ops = True
    options.specialize = True
    # options.override_mlir=mlir_claude_rescheduled2

    options.print_ir_after = "all" if is_debug else []
    # options.print_ir_after = "all"
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    print(gemm.asm)

    _run_mxfp_gemm(gemm, shape, shuffle_scales=True)
    print("MXFP GEMM double-buffer 8-wave test passed!")


def test_triplebuf_8wave_mxfp_gemm(
    is_debug=False, shape=(16384, 16384, 16384), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 8 waves, with stagger."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, num_waves=8)
    schedule = get_mxfp4_triplebuf_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    print(gemm.asm)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM triple-buffer 8-wave test passed!")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(
        args.test, globals(), args.debug, args.repeat, args.shape, args.block
    )
    exit(0 if success else 1)
