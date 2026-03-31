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
from utils import list_tests, parse_args, run_test

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_asymmetric_schedule,
    get_mxfp4_dbuf_mixed_pingpong_schedule,
    get_mxfp4_dbuf_mixed_pingpong_shuffle_schedule,
    get_mxfp4_dbuf_pingpong_schedule,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds,
    get_mxfp4_dbuf_schedule,
)
from wave_lang.kernel.wave.templates import (
    get_tagged_mxfp4_gemm,
    get_tagged_mxfp4_gemm_preshuffle_b,
    get_tagged_mxfp4_gemm_preshuffle_b_wide_store,
    get_tagged_mxfp4_gemm_preshuffle_scales,
    get_tagged_mxfp4_gemm_preshuffle_scales_and_B,
)
from wave_lang.kernel.wave.utils.mxfp_utils import (
    b_preshuffle,
    e8m0_shuffle,
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
)
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config


def _run_mxfp_gemm(gemm, shape):
    """Run compiled GEMM kernel and verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    gemm(x, x_scales, w.T.contiguous(), w_scales, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def _run_mxfp_gemm_preshuffle(
    gemm, shape, all=False, only_scale=False, only_b=False, output_dtype=torch.float32
):
    """Run compiled GEMM kernel with preshuffled B and B_scale, verify against reference.

    Shuffling is applied based on the flags:
      all        - shuffle a_scale (x_scales), b_scale (w_scales), and b (w_t)
      only_scale - shuffle a_scale (x_scales) and b_scale (w_scales) only
      only_b     - shuffle b_scale (w_scales) only
    """
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    w_t = w.T.contiguous()

    # Apply b (w_t) preshuffle only when all=True
    w_t_ps = b_preshuffle(w_t) if all else w_t

    # Apply a_scale shuffle when all=True or only_scale=True
    x_scales_ps = e8m0_shuffle(x_scales) if (all or only_scale) else x_scales

    # Apply b_scale shuffle when all=True, only_scale=True, or only_b=True
    w_scales_ps = e8m0_shuffle(w_scales) if (all or only_scale or only_b) else w_scales

    x, w_t_ps = x.cuda(), w_t_ps.cuda()
    x_scales_ps, w_scales_ps = x_scales_ps.cuda(), w_scales_ps.cuda()
    out = torch.zeros(x.shape[0], w_t_ps.shape[0], dtype=output_dtype).cuda()

    gemm(x, x_scales_ps, w_t_ps, w_scales_ps, out)

    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def _get_8wave_shape_from_block(block):
    """Choose an 8-wave shape (4x2 or 2x4) from block M/N dims.

    If either block M or N is 32, force that corresponding wave dimension to 2.
    """
    m_blk, n_blk = block[0], block[1]
    if m_blk == 32 and n_blk == 32:
        raise ValueError(
            "Cannot satisfy both M and N=32 with an 8-wave shape constrained to (4, 2) or (2, 4)."
        )
    if m_blk == 32:
        return (2, 4)
    if n_blk == 32:
        return (4, 2)
    return (4, 2)


def test_dbuf_4wave_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 4 waves, no stagger."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(2, 2))
    schedule = get_mxfp4_dbuf_schedule(use_stagger=False)

    options.print_ir_after = "all" if is_debug else []
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave.mlir"
    options.print_mlir = True
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 4-wave test passed!")


def test_dbuf_8wave_pingpong_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    A and B are read from global memory directly to LDS.

    Note: for dynamic mode, keep block MxN at or below 128x256 or 256x128
    to avoid exceeding shared-memory limits.
    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape, block, wave_shape=wave_shape
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]

    schedule = get_mxfp4_dbuf_pingpong_schedule(use_stagger=True, shape=shape)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, only_scale=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scale shuffling ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    Same for B data. However, prefetching shuffled B directly to VGPR consumes too many VGPRs and causes spilling.
    A is read from global memory directly to LDS.
    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape, block, wave_shape=wave_shape
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled(use_stagger=True, shape=shape)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scale and B shuffling and B->VGPR ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle_lds(
    is_debug=False, shape=(8192, 6144, 8192), block=(256, 192, 256), dynamic=True
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    B data is preshuffled and loaded to LDS (shared memory), not directly to VGPRs.
    A data is read from global memory directly to LDS.
    """

    mlir_256x192 = """
        #map = affine_map<()[s0] -> (s0 ceildiv 256)>
        #map1 = affine_map<()[s0] -> (s0 ceildiv 192)>
        #map2 = affine_map<()[s0] -> (s0 floordiv 32)>
        #map3 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 32) * 256)>
        #map4 = affine_map<()[s0] -> (s0 floordiv 2)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
        #map6 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map7 = affine_map<()[s0] -> (s0 mod 8)>
        #map8 = affine_map<()[s0] -> (s0 * 16)>
        #map9 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 32) * 256 + 64)>
        #map10 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
        #map11 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 32) * 256 + 128)>
        #map12 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
        #map13 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 24) floordiv 32) * 256 + 192)>
        #map14 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
        #map15 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 24) * 192)>
        #map16 = affine_map<()[s0, s1] -> (s0 * 256 + (s1 floordiv 8) * 16 - ((s1 floordiv 8) floordiv 16) * 256)>
        #map17 = affine_map<()[s0, s1, s2, s3] -> (s0 + s1 * 192 + (((s3 * 32 + s2 floordiv 8) mod 192) floordiv 16) * 16)>
        #map18 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 24) * 192 + 64)>
        #map19 = affine_map<()[s0, s1, s2, s3] -> (s0 + s1 * 192 + s3 * 32 + ((s2 floordiv 8) floordiv 16) * 16 - ((s3 * 32 + s2 floordiv 8 + 64) floordiv 192) * 192 + 64)>
        #map20 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64 + 16) floordiv 24) * 192 + 128)>
        #map21 = affine_map<()[s0, s1, s2, s3] -> (s0 + s1 * 192 + s3 * 32 + ((s2 floordiv 8) floordiv 16) * 16 - ((s3 * 32 + s2 floordiv 8 + 128) floordiv 192) * 192 + 128)>
        #map22 = affine_map<()[s0] -> (s0 * 256)>
        #map23 = affine_map<()[s0] -> (s0 * 128 + ((s0 mod 64) floordiv 16) * 2048 - (s0 floordiv 16) * 2048)>
        #map24 = affine_map<()[s0, s1] -> (s0 + (s1 floordiv 64) * 64)>
        #map25 = affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 16) * 2048 + ((s0 mod 64) floordiv 16) * 2048)>
        #map26 = affine_map<()[s0, s1] -> (s0 + (s1 floordiv 64) * 64 + 32)>
        #map27 = affine_map<()[s0] -> (s0 * 192)>
        #map28 = affine_map<()[s0, s1] -> (s0 + s1 * 96)>
        #map29 = affine_map<()[s0, s1] -> (s0 + s1 * 96 + 32)>
        #map30 = affine_map<()[s0, s1] -> (s0 + s1 * 96 + 64)>
        #map31 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map32 = affine_map<()[s0] -> (s0 ceildiv 256 - 1)>
        #map33 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map34 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048)>
        #map35 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048 + 2048)>
        #map36 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048 + 4096)>
        #map37 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + (s0 floordiv 64) * 8192 - (s0 floordiv 16) * 2048 + 6144)>
        #map38 = affine_map<()[s0, s1, s2] -> (s0 * 12288 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048)>
        #map39 = affine_map<()[s0, s1, s2] -> (s0 * 12288 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 2048)>
        #map40 = affine_map<()[s0, s1, s2] -> (s0 * 12288 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 4096)>
        #map41 = affine_map<()[s0, s1, s2] -> (s0 * 12288 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 6144)>
        #map42 = affine_map<()[s0, s1, s2] -> (s0 * 12288 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 8192)>
        #map43 = affine_map<()[s0, s1, s2] -> (s0 * 12288 + s1 * 128 + s2 * 16 - (s1 floordiv 16) * 2048 + 10240)>
        #map44 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map45 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 128)>
        #map46 = affine_map<()[s0, s1, s2] -> (s0 * 2048 + s1 * 256 + (s2 floordiv 8) * 16 - ((s2 floordiv 8) floordiv 16) * 256 + 2048)>
        #map47 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 128 + ((s1 mod 64) floordiv 16) * 2048 - (s1 floordiv 16) * 2048 + 8192)>
        #map48 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 8192 - (s0 floordiv 16) * 2048 + ((s0 mod 64) floordiv 16) * 2048 + 8192)>
        #map49 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4)>
        #map50 = affine_map<()[s0, s1] -> (s0 + s1 * 96 - (s0 floordiv 16) * 16)>
        #map51 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 1)>
        #map52 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 2)>
        #map53 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 3)>
        #map54 = affine_map<()[s0, s1] -> (s0 + s1 * 96 - (s0 floordiv 16) * 16 + 16)>
        #map55 = affine_map<()[s0, s1] -> (s0 + s1 * 96 - (s0 floordiv 16) * 16 + 32)>
        #map56 = affine_map<()[s0, s1] -> (s0 + s1 * 96 - (s0 floordiv 16) * 16 + 48)>
        #map57 = affine_map<()[s0, s1] -> (s0 + s1 * 96 - (s0 floordiv 16) * 16 + 64)>
        #map58 = affine_map<()[s0, s1] -> (s0 + s1 * 96 - (s0 floordiv 16) * 16 + 80)>
        #map59 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 16)>
        #map60 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 17)>
        #map61 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 18)>
        #map62 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 19)>
        #map63 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 32)>
        #map64 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 33)>
        #map65 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 34)>
        #map66 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 35)>
        #map67 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 48)>
        #map68 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 49)>
        #map69 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 50)>
        #map70 = affine_map<()[s0] -> ((s0 floordiv 64) * 64 + ((s0 mod 64) floordiv 16) * 4 + 51)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm {
            stream.executable.export public @gemm workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> (index, index, index) {
            %c1 = arith.constant 1 : index
            %0 = affine.apply #map()[%arg0]
            %1 = affine.apply #map1()[%arg1]
            stream.return %0, %1, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c4294967295_i64 = arith.constant 4294967295 : i64
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c0_i32 = arith.constant 0 : i32
                %c32_i64 = arith.constant 32 : i64
                %c2147483646_i64 = arith.constant 2147483646 : i64
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c2 = arith.constant 2 : index
                %c1 = arith.constant 1 : index
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
                %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
                %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<bf16>
                %block_id_x = gpu.block_id  x upper_bound 2147483647
                %block_id_y = gpu.block_id  y upper_bound 2147483647
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %5 = affine.apply #map2()[%arg7]
                %reinterpret_cast = memref.reinterpret_cast %1 to offset: [0], sizes: [%arg5, %5], strides: [%arg9, 1] : memref<i8> to memref<?x?xi8, strided<[?, 1]>>
                %reinterpret_cast_0 = memref.reinterpret_cast %3 to offset: [0], sizes: [%arg6, %5], strides: [%arg11, 1] : memref<i8> to memref<?x?xi8, strided<[?, 1]>>
                %reinterpret_cast_1 = memref.reinterpret_cast %4 to offset: [0], sizes: [%arg5, %arg6], strides: [%arg12, 1] : memref<bf16> to memref<?x?xbf16, strided<[?, 1]>>
                %alloc = memref.alloc() : memref<192x128xi8, #gpu.address_space<workgroup>>
                %alloc_2 = memref.alloc() : memref<192x128xi8, #gpu.address_space<workgroup>>
                %alloc_3 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %alloc_4 = memref.alloc() : memref<256x128xi8, #gpu.address_space<workgroup>>
                %6 = affine.apply #map3()[%thread_id_x, %thread_id_y]
                %7 = gpu.subgroup_broadcast %6,  first_active_lane : index
                %8 = gpu.subgroup_broadcast %c0,  first_active_lane : index
                %9 = affine.apply #map4()[%arg7]
                %10 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_x]
                %11 = affine.apply #map6()[%thread_id_x]
                %12 = affine.apply #map7()[%thread_id_x]
                %13 = arith.xori %12, %11 : index
                %14 = affine.apply #map8()[%13]
                %15 = arith.muli %10, %9 overflow<nsw> : index
                %16 = arith.addi %15, %14 overflow<nsw> : index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast = memref.cast %reinterpret_cast_5 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %17 = arith.index_cast %9 : index to i14
                %18 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%17) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %18[%16], %alloc_4[%7, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %19 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                %20 = gpu.subgroup_broadcast %19,  first_active_lane : index
                %21 = affine.apply #map10()[%thread_id_x, %thread_id_y, %block_id_x]
                %22 = arith.muli %21, %9 overflow<nsw> : index
                %23 = arith.addi %22, %14 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%23], %alloc_4[%20, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %24 = affine.apply #map11()[%thread_id_x, %thread_id_y]
                %25 = gpu.subgroup_broadcast %24,  first_active_lane : index
                %26 = affine.apply #map12()[%thread_id_x, %thread_id_y, %block_id_x]
                %27 = arith.muli %26, %9 overflow<nsw> : index
                %28 = arith.addi %27, %14 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%28], %alloc_4[%25, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %29 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                %30 = gpu.subgroup_broadcast %29,  first_active_lane : index
                %31 = affine.apply #map14()[%thread_id_x, %thread_id_y, %block_id_x]
                %32 = arith.muli %31, %9 overflow<nsw> : index
                %33 = arith.addi %32, %14 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%33], %alloc_4[%30, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %34 = affine.apply #map15()[%thread_id_x, %thread_id_y]
                %35 = gpu.subgroup_broadcast %34,  first_active_lane : index
                %36 = affine.apply #map16()[%13, %thread_id_x]
                %37 = arith.index_cast %9 : index to i32
                %38 = arith.extui %37 : i32 to i64
                %39 = arith.addi %38, %c4294967295_i64 : i64
                %40 = arith.divui %39, %38 : i64
                %41 = arith.trunci %40 : i64 to i32
                %42 = arith.index_cast %36 : index to i32
                %43 = arith.extui %42 : i32 to i64
                %44 = arith.extui %41 : i32 to i64
                %45 = arith.muli %43, %44 : i64
                %46 = arith.shrui %45, %c32_i64 : i64
                %47 = arith.trunci %46 : i64 to i32
                %48 = arith.muli %47, %37 : i32
                %49 = arith.subi %42, %48 : i32
                %50 = arith.cmpi uge, %49, %37 : i32
                %51 = arith.extui %50 : i1 to i32
                %52 = arith.subi %47, %51 : i32
                %53 = arith.index_cast %52 : i32 to index
                %54 = affine.apply #map17()[%53, %block_id_y, %thread_id_x, %thread_id_y]
                %55 = arith.select %50, %37, %c0_i32 : i32
                %56 = arith.addi %49, %55 : i32
                %57 = arith.index_cast %56 : i32 to index
                %58 = arith.muli %54, %9 overflow<nsw> : index
                %59 = arith.addi %58, %57 overflow<nsw> : index
                %reinterpret_cast_6 = memref.reinterpret_cast %2 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
                %cast_7 = memref.cast %reinterpret_cast_6 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
                %60 = amdgpu.fat_raw_buffer_cast %cast_7 validBytes(%c2147483646_i64) cacheSwizzleStride(%17) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                amdgpu.gather_to_lds %60[%59], %alloc_2[%35, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                %61 = affine.apply #map18()[%thread_id_x, %thread_id_y]
                %62 = gpu.subgroup_broadcast %61,  first_active_lane : index
                %63 = affine.apply #map19()[%53, %block_id_y, %thread_id_x, %thread_id_y]
                %64 = arith.muli %63, %9 overflow<nsw> : index
                %65 = arith.addi %64, %57 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%65], %alloc_2[%62, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                %66 = affine.apply #map20()[%thread_id_x, %thread_id_y]
                %67 = gpu.subgroup_broadcast %66,  first_active_lane : index
                %68 = affine.apply #map21()[%53, %block_id_y, %thread_id_x, %thread_id_y]
                %69 = arith.muli %68, %9 overflow<nsw> : index
                %70 = arith.addi %69, %57 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%70], %alloc_2[%67, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                %71 = arith.index_cast %arg7 : index to i32
                %72 = arith.extui %71 : i32 to i64
                %73 = arith.addi %72, %c4294967295_i64 : i64
                %74 = arith.divui %73, %72 : i64
                %75 = arith.trunci %74 : i64 to i32
                %76 = affine.apply #map22()[%block_id_x]
                %77 = affine.apply #map23()[%thread_id_x]
                %78 = arith.index_cast %77 : index to i32
                %79 = arith.extui %78 : i32 to i64
                %80 = arith.extui %75 : i32 to i64
                %81 = arith.muli %79, %80 : i64
                %82 = arith.shrui %81, %c32_i64 : i64
                %83 = arith.trunci %82 : i64 to i32
                %84 = arith.muli %83, %71 : i32
                %85 = arith.subi %78, %84 : i32
                %86 = arith.cmpi uge, %85, %71 : i32
                %87 = arith.extui %86 : i1 to i32
                %88 = arith.subi %83, %87 : i32
                %89 = arith.index_cast %88 : i32 to index
                %90 = affine.apply #map24()[%89, %thread_id_x]
                %91 = affine.apply #map25()[%thread_id_x]
                %92 = arith.index_cast %91 : index to i32
                %93 = arith.extui %92 : i32 to i64
                %94 = arith.muli %93, %80 : i64
                %95 = arith.shrui %94, %c32_i64 : i64
                %96 = arith.trunci %95 : i64 to i32
                %97 = arith.muli %96, %71 : i32
                %98 = arith.subi %92, %97 : i32
                %99 = arith.cmpi uge, %98, %71 : i32
                %100 = arith.select %99, %71, %c0_i32 : i32
                %101 = arith.addi %98, %100 : i32
                %102 = arith.index_cast %101 : i32 to index
                %103 = affine.apply #map2()[%102]
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<?x?xi8, strided<[?, 1]>> -> memref<i8>, index, index, index, index, index
                %104 = arith.muli %76, %strides#0 overflow<nsw> : index
                %105 = arith.muli %90, %strides#0 overflow<nsw> : index
                %106 = arith.addi %105, %103 overflow<nsw> : index
                %reinterpret_cast_8 = memref.reinterpret_cast %1 to offset: [%104], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1], offset: ?>>
                %cast_9 = memref.cast %reinterpret_cast_8 : memref<2147483646xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
                %107 = arith.index_cast %strides#0 : index to i14
                %108 = amdgpu.fat_raw_buffer_cast %cast_9 validBytes(%c2147483646_i64) cacheSwizzleStride(%107) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                %109 = vector.load %108[%106] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %110 = affine.apply #map26()[%89, %thread_id_x]
                %111 = arith.muli %110, %strides#0 overflow<nsw> : index
                %112 = arith.addi %111, %103 overflow<nsw> : index
                %113 = vector.load %108[%112] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %114 = affine.apply #map27()[%block_id_y]
                %115 = affine.apply #map28()[%89, %thread_id_y]
                %base_buffer_10, %offset_11, %sizes_12:2, %strides_13:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<?x?xi8, strided<[?, 1]>> -> memref<i8>, index, index, index, index, index
                %116 = arith.muli %114, %strides_13#0 overflow<nsw> : index
                %117 = arith.muli %115, %strides_13#0 overflow<nsw> : index
                %118 = arith.addi %117, %103 overflow<nsw> : index
                %reinterpret_cast_14 = memref.reinterpret_cast %3 to offset: [%116], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1], offset: ?>>
                %cast_15 = memref.cast %reinterpret_cast_14 : memref<2147483646xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
                %119 = arith.index_cast %strides_13#0 : index to i14
                %120 = amdgpu.fat_raw_buffer_cast %cast_15 validBytes(%c2147483646_i64) cacheSwizzleStride(%119) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
                %121 = vector.load %120[%118] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %122 = affine.apply #map29()[%89, %thread_id_y]
                %123 = arith.muli %122, %strides_13#0 overflow<nsw> : index
                %124 = arith.addi %123, %103 overflow<nsw> : index
                %125 = vector.load %120[%124] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %126 = affine.apply #map30()[%89, %thread_id_y]
                %127 = arith.muli %126, %strides_13#0 overflow<nsw> : index
                %128 = arith.addi %127, %103 overflow<nsw> : index
                %129 = vector.load %120[%128] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                amdgpu.memory_counter_wait load(5)
                rocdl.s.barrier
                %130 = affine.apply #map31()[%thread_id_x, %thread_id_y]
                %131 = arith.index_cast %130 : index to i32
                %132 = arith.cmpi sge, %131, %c4_i32 : i32
                %133 = arith.cmpi slt, %131, %c4_i32 : i32
                scf.if %132 {
                rocdl.s.barrier
                }
                %134 = affine.apply #map32()[%arg7]
                %135 = arith.remsi %134, %c2 : index
                %136 = arith.subi %134, %135 : index
                %reinterpret_cast_16 = memref.reinterpret_cast %alloc_4 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
                %137 = affine.apply #map33()[%thread_id_x]
                %138 = arith.xori %137, %12 : index
                %139 = affine.apply #map34()[%thread_id_x, %138]
                %140 = affine.apply #map35()[%thread_id_x, %138]
                %141 = affine.apply #map36()[%thread_id_x, %138]
                %142 = affine.apply #map37()[%thread_id_x, %138]
                %reinterpret_cast_17 = memref.reinterpret_cast %alloc_2 to offset: [0], sizes: [24576], strides: [1] : memref<192x128xi8, #gpu.address_space<workgroup>> to memref<24576xi8, #gpu.address_space<workgroup>>
                %143 = affine.apply #map38()[%thread_id_y, %thread_id_x, %138]
                %144 = affine.apply #map39()[%thread_id_y, %thread_id_x, %138]
                %145 = affine.apply #map40()[%thread_id_y, %thread_id_x, %138]
                %146 = affine.apply #map41()[%thread_id_y, %thread_id_x, %138]
                %147 = affine.apply #map42()[%thread_id_y, %thread_id_x, %138]
                %148 = affine.apply #map43()[%thread_id_y, %thread_id_x, %138]
                %149 = affine.apply #map44()[%thread_id_x]
                %150 = arith.xori %149, %12 : index
                %151 = affine.apply #map34()[%thread_id_x, %150]
                %152 = affine.apply #map35()[%thread_id_x, %150]
                %153 = affine.apply #map36()[%thread_id_x, %150]
                %154 = affine.apply #map37()[%thread_id_x, %150]
                %155 = affine.apply #map38()[%thread_id_y, %thread_id_x, %150]
                %156 = affine.apply #map39()[%thread_id_y, %thread_id_x, %150]
                %157 = affine.apply #map40()[%thread_id_y, %thread_id_x, %150]
                %158 = affine.apply #map41()[%thread_id_y, %thread_id_x, %150]
                %159 = affine.apply #map42()[%thread_id_y, %thread_id_x, %150]
                %160 = affine.apply #map43()[%thread_id_y, %thread_id_x, %150]
                %reinterpret_cast_18 = memref.reinterpret_cast %alloc_3 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
                %reinterpret_cast_19 = memref.reinterpret_cast %alloc to offset: [0], sizes: [24576], strides: [1] : memref<192x128xi8, #gpu.address_space<workgroup>> to memref<24576xi8, #gpu.address_space<workgroup>>
                %161:29 = scf.for %arg13 = %c0 to %136 step %c2 iter_args(%arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %109, %arg39 = %113, %arg40 = %121, %arg41 = %125, %arg42 = %129) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>) {
                %514 = vector.bitcast %arg39 : vector<4xi8> to vector<4xf8E8M0FNU>
                %515 = vector.bitcast %arg42 : vector<4xi8> to vector<4xf8E8M0FNU>
                %516 = vector.bitcast %arg41 : vector<4xi8> to vector<4xf8E8M0FNU>
                %517 = vector.bitcast %arg40 : vector<4xi8> to vector<4xf8E8M0FNU>
                %518 = vector.bitcast %arg38 : vector<4xi8> to vector<4xf8E8M0FNU>
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(5)
                rocdl.s.barrier
                %519 = affine.apply #map45()[%arg13, %13]
                %520 = arith.addi %15, %519 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%520], %alloc_3[%7, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %521 = arith.addi %22, %519 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%521], %alloc_3[%20, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %522 = arith.addi %27, %519 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%522], %alloc_3[%25, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %523 = arith.addi %32, %519 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%523], %alloc_3[%30, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %524 = affine.apply #map46()[%arg13, %13, %thread_id_x]
                %525 = arith.index_cast %524 : index to i32
                %526 = arith.extui %525 : i32 to i64
                %527 = arith.muli %526, %44 : i64
                %528 = arith.shrui %527, %c32_i64 : i64
                %529 = arith.trunci %528 : i64 to i32
                %530 = arith.muli %529, %37 : i32
                %531 = arith.subi %525, %530 : i32
                %532 = arith.cmpi uge, %531, %37 : i32
                %533 = arith.extui %532 : i1 to i32
                %534 = arith.subi %529, %533 : i32
                %535 = arith.index_cast %534 : i32 to index
                %536 = affine.apply #map17()[%535, %block_id_y, %thread_id_x, %thread_id_y]
                %537 = arith.select %532, %37, %c0_i32 : i32
                %538 = arith.addi %531, %537 : i32
                %539 = arith.index_cast %538 : i32 to index
                %540 = arith.muli %536, %9 overflow<nsw> : index
                %541 = arith.addi %540, %539 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%541], %alloc[%35, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                %542 = affine.apply #map19()[%535, %block_id_y, %thread_id_x, %thread_id_y]
                %543 = arith.muli %542, %9 overflow<nsw> : index
                %544 = arith.addi %543, %539 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%544], %alloc[%62, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                %545 = affine.apply #map21()[%535, %block_id_y, %thread_id_x, %thread_id_y]
                %546 = arith.muli %545, %9 overflow<nsw> : index
                %547 = arith.addi %546, %539 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%547], %alloc[%67, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                rocdl.sched.barrier 0
                %548 = vector.load %reinterpret_cast_16[%139] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %549 = vector.load %reinterpret_cast_16[%140] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %550 = vector.load %reinterpret_cast_16[%141] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %551 = vector.load %reinterpret_cast_16[%142] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %552 = vector.load %reinterpret_cast_17[%143] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %553 = vector.load %reinterpret_cast_17[%144] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %554 = vector.load %reinterpret_cast_17[%145] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %555 = vector.load %reinterpret_cast_17[%146] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %556 = vector.load %reinterpret_cast_17[%147] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %557 = vector.load %reinterpret_cast_17[%148] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %558 = vector.bitcast %548 : vector<16xi8> to vector<32xf4E2M1FN>
                %559 = vector.bitcast %549 : vector<16xi8> to vector<32xf4E2M1FN>
                %560 = vector.bitcast %550 : vector<16xi8> to vector<32xf4E2M1FN>
                %561 = vector.bitcast %551 : vector<16xi8> to vector<32xf4E2M1FN>
                %562 = vector.bitcast %552 : vector<16xi8> to vector<32xf4E2M1FN>
                %563 = vector.bitcast %553 : vector<16xi8> to vector<32xf4E2M1FN>
                %564 = vector.bitcast %554 : vector<16xi8> to vector<32xf4E2M1FN>
                %565 = vector.bitcast %555 : vector<16xi8> to vector<32xf4E2M1FN>
                %566 = vector.bitcast %556 : vector<16xi8> to vector<32xf4E2M1FN>
                %567 = vector.bitcast %557 : vector<16xi8> to vector<32xf4E2M1FN>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %568 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%517[0] * %562) + %arg14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %569 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%517[1] * %563) + %arg15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %570 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%516[0] * %564) + %arg16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %571 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%516[1] * %565) + %arg17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %572 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%515[0] * %566) + %arg18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %573 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%515[1] * %567) + %arg19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %574 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%517[0] * %562) + %arg20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %575 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%517[1] * %563) + %arg21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %576 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%516[0] * %564) + %arg22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %577 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%516[1] * %565) + %arg23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %578 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%515[0] * %566) + %arg24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %579 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%515[1] * %567) + %arg25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %580 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%517[0] * %562) + %arg26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %581 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%517[1] * %563) + %arg27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %582 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%516[0] * %564) + %arg28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %583 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%516[1] * %565) + %arg29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %584 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%515[0] * %566) + %arg30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %585 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%515[1] * %567) + %arg31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %586 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%517[0] * %562) + %arg32 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %587 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%517[1] * %563) + %arg33 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %588 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%516[0] * %564) + %arg34 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %589 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%516[1] * %565) + %arg35 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %590 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%515[0] * %566) + %arg36 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %591 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%515[1] * %567) + %arg37 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.sched.barrier 0
                %592 = affine.apply #map47()[%arg13, %thread_id_x]
                %593 = arith.index_cast %592 : index to i32
                %594 = arith.extui %593 : i32 to i64
                %595 = arith.muli %594, %80 : i64
                %596 = arith.shrui %595, %c32_i64 : i64
                %597 = arith.trunci %596 : i64 to i32
                %598 = arith.muli %597, %71 : i32
                %599 = arith.subi %593, %598 : i32
                %600 = arith.cmpi uge, %599, %71 : i32
                %601 = arith.extui %600 : i1 to i32
                %602 = arith.subi %597, %601 : i32
                %603 = arith.index_cast %602 : i32 to index
                %604 = affine.apply #map24()[%603, %thread_id_x]
                %605 = affine.apply #map48()[%thread_id_x, %arg13]
                %606 = arith.index_cast %605 : index to i32
                %607 = arith.extui %606 : i32 to i64
                %608 = arith.muli %607, %80 : i64
                %609 = arith.shrui %608, %c32_i64 : i64
                %610 = arith.trunci %609 : i64 to i32
                %611 = arith.muli %610, %71 : i32
                %612 = arith.subi %606, %611 : i32
                %613 = arith.cmpi uge, %612, %71 : i32
                %614 = arith.select %613, %71, %c0_i32 : i32
                %615 = arith.addi %612, %614 : i32
                %616 = arith.index_cast %615 : i32 to index
                %617 = affine.apply #map2()[%616]
                %618 = arith.muli %604, %strides#0 overflow<nsw> : index
                %619 = arith.addi %618, %617 overflow<nsw> : index
                %620 = vector.load %108[%619] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %621 = vector.bitcast %620 : vector<4xi8> to vector<4xf8E8M0FNU>
                %622 = affine.apply #map26()[%603, %thread_id_x]
                %623 = arith.muli %622, %strides#0 overflow<nsw> : index
                %624 = arith.addi %623, %617 overflow<nsw> : index
                %625 = vector.load %108[%624] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %626 = vector.bitcast %625 : vector<4xi8> to vector<4xf8E8M0FNU>
                %627 = affine.apply #map28()[%603, %thread_id_y]
                %628 = arith.muli %627, %strides_13#0 overflow<nsw> : index
                %629 = arith.addi %628, %617 overflow<nsw> : index
                %630 = vector.load %120[%629] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %631 = vector.bitcast %630 : vector<4xi8> to vector<4xf8E8M0FNU>
                %632 = affine.apply #map29()[%603, %thread_id_y]
                %633 = arith.muli %632, %strides_13#0 overflow<nsw> : index
                %634 = arith.addi %633, %617 overflow<nsw> : index
                %635 = vector.load %120[%634] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %636 = vector.bitcast %635 : vector<4xi8> to vector<4xf8E8M0FNU>
                %637 = affine.apply #map30()[%603, %thread_id_y]
                %638 = arith.muli %637, %strides_13#0 overflow<nsw> : index
                %639 = arith.addi %638, %617 overflow<nsw> : index
                %640 = vector.load %120[%639] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %641 = vector.bitcast %640 : vector<4xi8> to vector<4xf8E8M0FNU>
                %642 = vector.load %reinterpret_cast_16[%151] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %643 = vector.load %reinterpret_cast_16[%152] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %644 = vector.load %reinterpret_cast_16[%153] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %645 = vector.load %reinterpret_cast_16[%154] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %646 = vector.load %reinterpret_cast_17[%155] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %647 = vector.load %reinterpret_cast_17[%156] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %648 = vector.load %reinterpret_cast_17[%157] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %649 = vector.load %reinterpret_cast_17[%158] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %650 = vector.load %reinterpret_cast_17[%159] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %651 = vector.load %reinterpret_cast_17[%160] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %652 = vector.bitcast %642 : vector<16xi8> to vector<32xf4E2M1FN>
                %653 = vector.bitcast %643 : vector<16xi8> to vector<32xf4E2M1FN>
                %654 = vector.bitcast %644 : vector<16xi8> to vector<32xf4E2M1FN>
                %655 = vector.bitcast %645 : vector<16xi8> to vector<32xf4E2M1FN>
                %656 = vector.bitcast %646 : vector<16xi8> to vector<32xf4E2M1FN>
                %657 = vector.bitcast %647 : vector<16xi8> to vector<32xf4E2M1FN>
                %658 = vector.bitcast %648 : vector<16xi8> to vector<32xf4E2M1FN>
                %659 = vector.bitcast %649 : vector<16xi8> to vector<32xf4E2M1FN>
                %660 = vector.bitcast %650 : vector<16xi8> to vector<32xf4E2M1FN>
                %661 = vector.bitcast %651 : vector<16xi8> to vector<32xf4E2M1FN>
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(7)
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %662 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %652) * (%517[2] * %656) + %568 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %663 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %652) * (%517[3] * %657) + %569 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %664 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %652) * (%516[2] * %658) + %570 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %665 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %652) * (%516[3] * %659) + %571 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %666 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %652) * (%515[2] * %660) + %572 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %667 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %652) * (%515[3] * %661) + %573 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %668 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %653) * (%517[2] * %656) + %574 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %669 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %653) * (%517[3] * %657) + %575 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %670 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %653) * (%516[2] * %658) + %576 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %671 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %653) * (%516[3] * %659) + %577 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %672 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %653) * (%515[2] * %660) + %578 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %673 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %653) * (%515[3] * %661) + %579 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %674 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %654) * (%517[2] * %656) + %580 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %675 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %654) * (%517[3] * %657) + %581 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %676 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %654) * (%516[2] * %658) + %582 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %677 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %654) * (%516[3] * %659) + %583 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %678 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %654) * (%515[2] * %660) + %584 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %679 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %654) * (%515[3] * %661) + %585 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %680 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %655) * (%517[2] * %656) + %586 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %681 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %655) * (%517[3] * %657) + %587 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %682 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %655) * (%516[2] * %658) + %588 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %683 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %655) * (%516[3] * %659) + %589 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %684 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %655) * (%515[2] * %660) + %590 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %685 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %655) * (%515[3] * %661) + %591 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                %686 = arith.addi %arg13, %c1 : index
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(5)
                rocdl.s.barrier
                %687 = affine.apply #map45()[%686, %13]
                %688 = arith.addi %15, %687 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%688], %alloc_4[%7, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %689 = arith.addi %22, %687 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%689], %alloc_4[%20, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %690 = arith.addi %27, %687 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%690], %alloc_4[%25, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %691 = arith.addi %32, %687 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%691], %alloc_4[%30, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %692 = affine.apply #map46()[%686, %13, %thread_id_x]
                %693 = arith.index_cast %692 : index to i32
                %694 = arith.extui %693 : i32 to i64
                %695 = arith.muli %694, %44 : i64
                %696 = arith.shrui %695, %c32_i64 : i64
                %697 = arith.trunci %696 : i64 to i32
                %698 = arith.muli %697, %37 : i32
                %699 = arith.subi %693, %698 : i32
                %700 = arith.cmpi uge, %699, %37 : i32
                %701 = arith.extui %700 : i1 to i32
                %702 = arith.subi %697, %701 : i32
                %703 = arith.index_cast %702 : i32 to index
                %704 = affine.apply #map17()[%703, %block_id_y, %thread_id_x, %thread_id_y]
                %705 = arith.select %700, %37, %c0_i32 : i32
                %706 = arith.addi %699, %705 : i32
                %707 = arith.index_cast %706 : i32 to index
                %708 = arith.muli %704, %9 overflow<nsw> : index
                %709 = arith.addi %708, %707 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%709], %alloc_2[%35, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                %710 = affine.apply #map19()[%703, %block_id_y, %thread_id_x, %thread_id_y]
                %711 = arith.muli %710, %9 overflow<nsw> : index
                %712 = arith.addi %711, %707 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%712], %alloc_2[%62, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                %713 = affine.apply #map21()[%703, %block_id_y, %thread_id_x, %thread_id_y]
                %714 = arith.muli %713, %9 overflow<nsw> : index
                %715 = arith.addi %714, %707 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%715], %alloc_2[%67, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                rocdl.sched.barrier 0
                %716 = vector.load %reinterpret_cast_18[%139] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %717 = vector.load %reinterpret_cast_18[%140] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %718 = vector.load %reinterpret_cast_18[%141] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %719 = vector.load %reinterpret_cast_18[%142] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %720 = vector.load %reinterpret_cast_19[%143] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %721 = vector.load %reinterpret_cast_19[%144] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %722 = vector.load %reinterpret_cast_19[%145] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %723 = vector.load %reinterpret_cast_19[%146] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %724 = vector.load %reinterpret_cast_19[%147] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %725 = vector.load %reinterpret_cast_19[%148] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %726 = vector.bitcast %716 : vector<16xi8> to vector<32xf4E2M1FN>
                %727 = vector.bitcast %717 : vector<16xi8> to vector<32xf4E2M1FN>
                %728 = vector.bitcast %718 : vector<16xi8> to vector<32xf4E2M1FN>
                %729 = vector.bitcast %719 : vector<16xi8> to vector<32xf4E2M1FN>
                %730 = vector.bitcast %720 : vector<16xi8> to vector<32xf4E2M1FN>
                %731 = vector.bitcast %721 : vector<16xi8> to vector<32xf4E2M1FN>
                %732 = vector.bitcast %722 : vector<16xi8> to vector<32xf4E2M1FN>
                %733 = vector.bitcast %723 : vector<16xi8> to vector<32xf4E2M1FN>
                %734 = vector.bitcast %724 : vector<16xi8> to vector<32xf4E2M1FN>
                %735 = vector.bitcast %725 : vector<16xi8> to vector<32xf4E2M1FN>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %736 = amdgpu.scaled_mfma 16x16x128 (%621[0] * %726) * (%631[0] * %730) + %662 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %737 = amdgpu.scaled_mfma 16x16x128 (%621[0] * %726) * (%631[1] * %731) + %663 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %738 = amdgpu.scaled_mfma 16x16x128 (%621[0] * %726) * (%636[0] * %732) + %664 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %739 = amdgpu.scaled_mfma 16x16x128 (%621[0] * %726) * (%636[1] * %733) + %665 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %740 = amdgpu.scaled_mfma 16x16x128 (%621[0] * %726) * (%641[0] * %734) + %666 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %741 = amdgpu.scaled_mfma 16x16x128 (%621[0] * %726) * (%641[1] * %735) + %667 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %742 = amdgpu.scaled_mfma 16x16x128 (%621[1] * %727) * (%631[0] * %730) + %668 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %743 = amdgpu.scaled_mfma 16x16x128 (%621[1] * %727) * (%631[1] * %731) + %669 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %744 = amdgpu.scaled_mfma 16x16x128 (%621[1] * %727) * (%636[0] * %732) + %670 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %745 = amdgpu.scaled_mfma 16x16x128 (%621[1] * %727) * (%636[1] * %733) + %671 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %746 = amdgpu.scaled_mfma 16x16x128 (%621[1] * %727) * (%641[0] * %734) + %672 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %747 = amdgpu.scaled_mfma 16x16x128 (%621[1] * %727) * (%641[1] * %735) + %673 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %748 = amdgpu.scaled_mfma 16x16x128 (%626[0] * %728) * (%631[0] * %730) + %674 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %749 = amdgpu.scaled_mfma 16x16x128 (%626[0] * %728) * (%631[1] * %731) + %675 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %750 = amdgpu.scaled_mfma 16x16x128 (%626[0] * %728) * (%636[0] * %732) + %676 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %751 = amdgpu.scaled_mfma 16x16x128 (%626[0] * %728) * (%636[1] * %733) + %677 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %752 = amdgpu.scaled_mfma 16x16x128 (%626[0] * %728) * (%641[0] * %734) + %678 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %753 = amdgpu.scaled_mfma 16x16x128 (%626[0] * %728) * (%641[1] * %735) + %679 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %754 = amdgpu.scaled_mfma 16x16x128 (%626[1] * %729) * (%631[0] * %730) + %680 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %755 = amdgpu.scaled_mfma 16x16x128 (%626[1] * %729) * (%631[1] * %731) + %681 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %756 = amdgpu.scaled_mfma 16x16x128 (%626[1] * %729) * (%636[0] * %732) + %682 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %757 = amdgpu.scaled_mfma 16x16x128 (%626[1] * %729) * (%636[1] * %733) + %683 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %758 = amdgpu.scaled_mfma 16x16x128 (%626[1] * %729) * (%641[0] * %734) + %684 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %759 = amdgpu.scaled_mfma 16x16x128 (%626[1] * %729) * (%641[1] * %735) + %685 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.sched.barrier 0
                %760 = affine.apply #map47()[%686, %thread_id_x]
                %761 = arith.index_cast %760 : index to i32
                %762 = arith.extui %761 : i32 to i64
                %763 = arith.muli %762, %80 : i64
                %764 = arith.shrui %763, %c32_i64 : i64
                %765 = arith.trunci %764 : i64 to i32
                %766 = arith.muli %765, %71 : i32
                %767 = arith.subi %761, %766 : i32
                %768 = arith.cmpi uge, %767, %71 : i32
                %769 = arith.extui %768 : i1 to i32
                %770 = arith.subi %765, %769 : i32
                %771 = arith.index_cast %770 : i32 to index
                %772 = affine.apply #map24()[%771, %thread_id_x]
                %773 = affine.apply #map48()[%thread_id_x, %686]
                %774 = arith.index_cast %773 : index to i32
                %775 = arith.extui %774 : i32 to i64
                %776 = arith.muli %775, %80 : i64
                %777 = arith.shrui %776, %c32_i64 : i64
                %778 = arith.trunci %777 : i64 to i32
                %779 = arith.muli %778, %71 : i32
                %780 = arith.subi %774, %779 : i32
                %781 = arith.cmpi uge, %780, %71 : i32
                %782 = arith.select %781, %71, %c0_i32 : i32
                %783 = arith.addi %780, %782 : i32
                %784 = arith.index_cast %783 : i32 to index
                %785 = affine.apply #map2()[%784]
                %786 = arith.muli %772, %strides#0 overflow<nsw> : index
                %787 = arith.addi %786, %785 overflow<nsw> : index
                %788 = vector.load %108[%787] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %789 = affine.apply #map26()[%771, %thread_id_x]
                %790 = arith.muli %789, %strides#0 overflow<nsw> : index
                %791 = arith.addi %790, %785 overflow<nsw> : index
                %792 = vector.load %108[%791] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %793 = affine.apply #map28()[%771, %thread_id_y]
                %794 = arith.muli %793, %strides_13#0 overflow<nsw> : index
                %795 = arith.addi %794, %785 overflow<nsw> : index
                %796 = vector.load %120[%795] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %797 = affine.apply #map29()[%771, %thread_id_y]
                %798 = arith.muli %797, %strides_13#0 overflow<nsw> : index
                %799 = arith.addi %798, %785 overflow<nsw> : index
                %800 = vector.load %120[%799] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %801 = affine.apply #map30()[%771, %thread_id_y]
                %802 = arith.muli %801, %strides_13#0 overflow<nsw> : index
                %803 = arith.addi %802, %785 overflow<nsw> : index
                %804 = vector.load %120[%803] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %805 = vector.load %reinterpret_cast_18[%151] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %806 = vector.load %reinterpret_cast_18[%152] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %807 = vector.load %reinterpret_cast_18[%153] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %808 = vector.load %reinterpret_cast_18[%154] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %809 = vector.load %reinterpret_cast_19[%155] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %810 = vector.load %reinterpret_cast_19[%156] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %811 = vector.load %reinterpret_cast_19[%157] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %812 = vector.load %reinterpret_cast_19[%158] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %813 = vector.load %reinterpret_cast_19[%159] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %814 = vector.load %reinterpret_cast_19[%160] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %815 = vector.bitcast %805 : vector<16xi8> to vector<32xf4E2M1FN>
                %816 = vector.bitcast %806 : vector<16xi8> to vector<32xf4E2M1FN>
                %817 = vector.bitcast %807 : vector<16xi8> to vector<32xf4E2M1FN>
                %818 = vector.bitcast %808 : vector<16xi8> to vector<32xf4E2M1FN>
                %819 = vector.bitcast %809 : vector<16xi8> to vector<32xf4E2M1FN>
                %820 = vector.bitcast %810 : vector<16xi8> to vector<32xf4E2M1FN>
                %821 = vector.bitcast %811 : vector<16xi8> to vector<32xf4E2M1FN>
                %822 = vector.bitcast %812 : vector<16xi8> to vector<32xf4E2M1FN>
                %823 = vector.bitcast %813 : vector<16xi8> to vector<32xf4E2M1FN>
                %824 = vector.bitcast %814 : vector<16xi8> to vector<32xf4E2M1FN>
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(7)
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %825 = amdgpu.scaled_mfma 16x16x128 (%621[2] * %815) * (%631[2] * %819) + %736 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %826 = amdgpu.scaled_mfma 16x16x128 (%621[2] * %815) * (%631[3] * %820) + %737 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %827 = amdgpu.scaled_mfma 16x16x128 (%621[2] * %815) * (%636[2] * %821) + %738 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %828 = amdgpu.scaled_mfma 16x16x128 (%621[2] * %815) * (%636[3] * %822) + %739 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %829 = amdgpu.scaled_mfma 16x16x128 (%621[2] * %815) * (%641[2] * %823) + %740 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %830 = amdgpu.scaled_mfma 16x16x128 (%621[2] * %815) * (%641[3] * %824) + %741 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %831 = amdgpu.scaled_mfma 16x16x128 (%621[3] * %816) * (%631[2] * %819) + %742 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %832 = amdgpu.scaled_mfma 16x16x128 (%621[3] * %816) * (%631[3] * %820) + %743 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %833 = amdgpu.scaled_mfma 16x16x128 (%621[3] * %816) * (%636[2] * %821) + %744 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %834 = amdgpu.scaled_mfma 16x16x128 (%621[3] * %816) * (%636[3] * %822) + %745 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %835 = amdgpu.scaled_mfma 16x16x128 (%621[3] * %816) * (%641[2] * %823) + %746 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %836 = amdgpu.scaled_mfma 16x16x128 (%621[3] * %816) * (%641[3] * %824) + %747 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %837 = amdgpu.scaled_mfma 16x16x128 (%626[2] * %817) * (%631[2] * %819) + %748 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %838 = amdgpu.scaled_mfma 16x16x128 (%626[2] * %817) * (%631[3] * %820) + %749 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %839 = amdgpu.scaled_mfma 16x16x128 (%626[2] * %817) * (%636[2] * %821) + %750 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %840 = amdgpu.scaled_mfma 16x16x128 (%626[2] * %817) * (%636[3] * %822) + %751 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %841 = amdgpu.scaled_mfma 16x16x128 (%626[2] * %817) * (%641[2] * %823) + %752 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %842 = amdgpu.scaled_mfma 16x16x128 (%626[2] * %817) * (%641[3] * %824) + %753 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %843 = amdgpu.scaled_mfma 16x16x128 (%626[3] * %818) * (%631[2] * %819) + %754 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %844 = amdgpu.scaled_mfma 16x16x128 (%626[3] * %818) * (%631[3] * %820) + %755 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %845 = amdgpu.scaled_mfma 16x16x128 (%626[3] * %818) * (%636[2] * %821) + %756 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %846 = amdgpu.scaled_mfma 16x16x128 (%626[3] * %818) * (%636[3] * %822) + %757 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %847 = amdgpu.scaled_mfma 16x16x128 (%626[3] * %818) * (%641[2] * %823) + %758 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %848 = amdgpu.scaled_mfma 16x16x128 (%626[3] * %818) * (%641[3] * %824) + %759 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                scf.yield %825, %826, %827, %828, %829, %830, %831, %832, %833, %834, %835, %836, %837, %838, %839, %840, %841, %842, %843, %844, %845, %846, %847, %848, %788, %792, %796, %800, %804 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>
                }
                %162:33 = scf.for %arg13 = %136 to %134 step %c1 iter_args(%arg14 = %161#0, %arg15 = %161#1, %arg16 = %161#2, %arg17 = %161#3, %arg18 = %161#4, %arg19 = %161#5, %arg20 = %161#6, %arg21 = %161#7, %arg22 = %161#8, %arg23 = %161#9, %arg24 = %161#10, %arg25 = %161#11, %arg26 = %161#12, %arg27 = %161#13, %arg28 = %161#14, %arg29 = %161#15, %arg30 = %161#16, %arg31 = %161#17, %arg32 = %161#18, %arg33 = %161#19, %arg34 = %161#20, %arg35 = %161#21, %arg36 = %161#22, %arg37 = %161#23, %arg38 = %161#24, %arg39 = %161#25, %arg40 = %161#26, %arg41 = %161#27, %arg42 = %161#28, %arg43 = %alloc_4, %arg44 = %alloc_3, %arg45 = %alloc_2, %arg46 = %alloc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<192x128xi8, #gpu.address_space<workgroup>>, memref<192x128xi8, #gpu.address_space<workgroup>>) {
                %514 = vector.bitcast %arg39 : vector<4xi8> to vector<4xf8E8M0FNU>
                %515 = vector.bitcast %arg42 : vector<4xi8> to vector<4xf8E8M0FNU>
                %516 = vector.bitcast %arg41 : vector<4xi8> to vector<4xf8E8M0FNU>
                %517 = vector.bitcast %arg40 : vector<4xi8> to vector<4xf8E8M0FNU>
                %518 = vector.bitcast %arg38 : vector<4xi8> to vector<4xf8E8M0FNU>
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(5)
                rocdl.s.barrier
                %519 = affine.apply #map45()[%arg13, %13]
                %520 = arith.addi %15, %519 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%520], %arg44[%7, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %521 = arith.addi %22, %519 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%521], %arg44[%20, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %522 = arith.addi %27, %519 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%522], %arg44[%25, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %523 = arith.addi %32, %519 overflow<nsw> : index
                amdgpu.gather_to_lds %18[%523], %arg44[%30, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<256x128xi8, #gpu.address_space<workgroup>>
                %524 = affine.apply #map46()[%arg13, %13, %thread_id_x]
                %525 = arith.index_cast %524 : index to i32
                %526 = arith.extui %525 : i32 to i64
                %527 = arith.muli %526, %44 : i64
                %528 = arith.shrui %527, %c32_i64 : i64
                %529 = arith.trunci %528 : i64 to i32
                %530 = arith.muli %529, %37 : i32
                %531 = arith.subi %525, %530 : i32
                %532 = arith.cmpi uge, %531, %37 : i32
                %533 = arith.extui %532 : i1 to i32
                %534 = arith.subi %529, %533 : i32
                %535 = arith.index_cast %534 : i32 to index
                %536 = affine.apply #map17()[%535, %block_id_y, %thread_id_x, %thread_id_y]
                %537 = arith.select %532, %37, %c0_i32 : i32
                %538 = arith.addi %531, %537 : i32
                %539 = arith.index_cast %538 : i32 to index
                %540 = arith.muli %536, %9 overflow<nsw> : index
                %541 = arith.addi %540, %539 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%541], %arg46[%35, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                %542 = affine.apply #map19()[%535, %block_id_y, %thread_id_x, %thread_id_y]
                %543 = arith.muli %542, %9 overflow<nsw> : index
                %544 = arith.addi %543, %539 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%544], %arg46[%62, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                %545 = affine.apply #map21()[%535, %block_id_y, %thread_id_x, %thread_id_y]
                %546 = arith.muli %545, %9 overflow<nsw> : index
                %547 = arith.addi %546, %539 overflow<nsw> : index
                amdgpu.gather_to_lds %60[%547], %arg46[%67, %8] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                rocdl.sched.barrier 0
                %reinterpret_cast_28 = memref.reinterpret_cast %arg43 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
                %548 = vector.load %reinterpret_cast_28[%139] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %549 = vector.load %reinterpret_cast_28[%140] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %550 = vector.load %reinterpret_cast_28[%141] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %551 = vector.load %reinterpret_cast_28[%142] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %reinterpret_cast_29 = memref.reinterpret_cast %arg45 to offset: [0], sizes: [24576], strides: [1] : memref<192x128xi8, #gpu.address_space<workgroup>> to memref<24576xi8, #gpu.address_space<workgroup>>
                %552 = vector.load %reinterpret_cast_29[%143] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %553 = vector.load %reinterpret_cast_29[%144] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %554 = vector.load %reinterpret_cast_29[%145] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %555 = vector.load %reinterpret_cast_29[%146] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %556 = vector.load %reinterpret_cast_29[%147] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %557 = vector.load %reinterpret_cast_29[%148] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %558 = vector.bitcast %548 : vector<16xi8> to vector<32xf4E2M1FN>
                %559 = vector.bitcast %549 : vector<16xi8> to vector<32xf4E2M1FN>
                %560 = vector.bitcast %550 : vector<16xi8> to vector<32xf4E2M1FN>
                %561 = vector.bitcast %551 : vector<16xi8> to vector<32xf4E2M1FN>
                %562 = vector.bitcast %552 : vector<16xi8> to vector<32xf4E2M1FN>
                %563 = vector.bitcast %553 : vector<16xi8> to vector<32xf4E2M1FN>
                %564 = vector.bitcast %554 : vector<16xi8> to vector<32xf4E2M1FN>
                %565 = vector.bitcast %555 : vector<16xi8> to vector<32xf4E2M1FN>
                %566 = vector.bitcast %556 : vector<16xi8> to vector<32xf4E2M1FN>
                %567 = vector.bitcast %557 : vector<16xi8> to vector<32xf4E2M1FN>
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %568 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%517[0] * %562) + %arg14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %569 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%517[1] * %563) + %arg15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %570 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%516[0] * %564) + %arg16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %571 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%516[1] * %565) + %arg17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %572 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%515[0] * %566) + %arg18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %573 = amdgpu.scaled_mfma 16x16x128 (%518[0] * %558) * (%515[1] * %567) + %arg19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %574 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%517[0] * %562) + %arg20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %575 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%517[1] * %563) + %arg21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %576 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%516[0] * %564) + %arg22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %577 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%516[1] * %565) + %arg23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %578 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%515[0] * %566) + %arg24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %579 = amdgpu.scaled_mfma 16x16x128 (%518[1] * %559) * (%515[1] * %567) + %arg25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %580 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%517[0] * %562) + %arg26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %581 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%517[1] * %563) + %arg27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %582 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%516[0] * %564) + %arg28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %583 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%516[1] * %565) + %arg29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %584 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%515[0] * %566) + %arg30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %585 = amdgpu.scaled_mfma 16x16x128 (%514[0] * %560) * (%515[1] * %567) + %arg31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %586 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%517[0] * %562) + %arg32 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %587 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%517[1] * %563) + %arg33 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %588 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%516[0] * %564) + %arg34 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %589 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%516[1] * %565) + %arg35 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %590 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%515[0] * %566) + %arg36 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %591 = amdgpu.scaled_mfma 16x16x128 (%514[1] * %561) * (%515[1] * %567) + %arg37 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.sched.barrier 0
                %592 = affine.apply #map47()[%arg13, %thread_id_x]
                %593 = arith.index_cast %592 : index to i32
                %594 = arith.extui %593 : i32 to i64
                %595 = arith.muli %594, %80 : i64
                %596 = arith.shrui %595, %c32_i64 : i64
                %597 = arith.trunci %596 : i64 to i32
                %598 = arith.muli %597, %71 : i32
                %599 = arith.subi %593, %598 : i32
                %600 = arith.cmpi uge, %599, %71 : i32
                %601 = arith.extui %600 : i1 to i32
                %602 = arith.subi %597, %601 : i32
                %603 = arith.index_cast %602 : i32 to index
                %604 = affine.apply #map24()[%603, %thread_id_x]
                %605 = affine.apply #map48()[%thread_id_x, %arg13]
                %606 = arith.index_cast %605 : index to i32
                %607 = arith.extui %606 : i32 to i64
                %608 = arith.muli %607, %80 : i64
                %609 = arith.shrui %608, %c32_i64 : i64
                %610 = arith.trunci %609 : i64 to i32
                %611 = arith.muli %610, %71 : i32
                %612 = arith.subi %606, %611 : i32
                %613 = arith.cmpi uge, %612, %71 : i32
                %614 = arith.select %613, %71, %c0_i32 : i32
                %615 = arith.addi %612, %614 : i32
                %616 = arith.index_cast %615 : i32 to index
                %617 = affine.apply #map2()[%616]
                %618 = arith.muli %604, %strides#0 overflow<nsw> : index
                %619 = arith.addi %618, %617 overflow<nsw> : index
                %620 = vector.load %108[%619] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %621 = affine.apply #map26()[%603, %thread_id_x]
                %622 = arith.muli %621, %strides#0 overflow<nsw> : index
                %623 = arith.addi %622, %617 overflow<nsw> : index
                %624 = vector.load %108[%623] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %625 = affine.apply #map28()[%603, %thread_id_y]
                %626 = arith.muli %625, %strides_13#0 overflow<nsw> : index
                %627 = arith.addi %626, %617 overflow<nsw> : index
                %628 = vector.load %120[%627] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %629 = affine.apply #map29()[%603, %thread_id_y]
                %630 = arith.muli %629, %strides_13#0 overflow<nsw> : index
                %631 = arith.addi %630, %617 overflow<nsw> : index
                %632 = vector.load %120[%631] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %633 = affine.apply #map30()[%603, %thread_id_y]
                %634 = arith.muli %633, %strides_13#0 overflow<nsw> : index
                %635 = arith.addi %634, %617 overflow<nsw> : index
                %636 = vector.load %120[%635] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
                %637 = vector.load %reinterpret_cast_28[%151] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %638 = vector.load %reinterpret_cast_28[%152] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %639 = vector.load %reinterpret_cast_28[%153] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %640 = vector.load %reinterpret_cast_28[%154] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %641 = vector.load %reinterpret_cast_29[%155] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %642 = vector.load %reinterpret_cast_29[%156] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %643 = vector.load %reinterpret_cast_29[%157] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %644 = vector.load %reinterpret_cast_29[%158] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %645 = vector.load %reinterpret_cast_29[%159] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %646 = vector.load %reinterpret_cast_29[%160] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %647 = vector.bitcast %637 : vector<16xi8> to vector<32xf4E2M1FN>
                %648 = vector.bitcast %638 : vector<16xi8> to vector<32xf4E2M1FN>
                %649 = vector.bitcast %639 : vector<16xi8> to vector<32xf4E2M1FN>
                %650 = vector.bitcast %640 : vector<16xi8> to vector<32xf4E2M1FN>
                %651 = vector.bitcast %641 : vector<16xi8> to vector<32xf4E2M1FN>
                %652 = vector.bitcast %642 : vector<16xi8> to vector<32xf4E2M1FN>
                %653 = vector.bitcast %643 : vector<16xi8> to vector<32xf4E2M1FN>
                %654 = vector.bitcast %644 : vector<16xi8> to vector<32xf4E2M1FN>
                %655 = vector.bitcast %645 : vector<16xi8> to vector<32xf4E2M1FN>
                %656 = vector.bitcast %646 : vector<16xi8> to vector<32xf4E2M1FN>
                rocdl.sched.barrier 0
                amdgpu.memory_counter_wait load(7)
                rocdl.s.barrier
                rocdl.sched.barrier 0
                rocdl.s.setprio 1
                %657 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %647) * (%517[2] * %651) + %568 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %658 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %647) * (%517[3] * %652) + %569 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %659 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %647) * (%516[2] * %653) + %570 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %660 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %647) * (%516[3] * %654) + %571 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %661 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %647) * (%515[2] * %655) + %572 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %662 = amdgpu.scaled_mfma 16x16x128 (%518[2] * %647) * (%515[3] * %656) + %573 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %663 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %648) * (%517[2] * %651) + %574 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %664 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %648) * (%517[3] * %652) + %575 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %665 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %648) * (%516[2] * %653) + %576 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %666 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %648) * (%516[3] * %654) + %577 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %667 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %648) * (%515[2] * %655) + %578 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %668 = amdgpu.scaled_mfma 16x16x128 (%518[3] * %648) * (%515[3] * %656) + %579 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %669 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %649) * (%517[2] * %651) + %580 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %670 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %649) * (%517[3] * %652) + %581 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %671 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %649) * (%516[2] * %653) + %582 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %672 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %649) * (%516[3] * %654) + %583 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %673 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %649) * (%515[2] * %655) + %584 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %674 = amdgpu.scaled_mfma 16x16x128 (%514[2] * %649) * (%515[3] * %656) + %585 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %675 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %650) * (%517[2] * %651) + %586 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %676 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %650) * (%517[3] * %652) + %587 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %677 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %650) * (%516[2] * %653) + %588 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %678 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %650) * (%516[3] * %654) + %589 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %679 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %650) * (%515[2] * %655) + %590 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %680 = amdgpu.scaled_mfma 16x16x128 (%514[3] * %650) * (%515[3] * %656) + %591 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                rocdl.s.setprio 0
                rocdl.sched.barrier 0
                scf.yield %657, %658, %659, %660, %661, %662, %663, %664, %665, %666, %667, %668, %669, %670, %671, %672, %673, %674, %675, %676, %677, %678, %679, %680, %620, %624, %628, %632, %636, %arg44, %arg43, %arg46, %arg45 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, vector<4xi8>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<256x128xi8, #gpu.address_space<workgroup>>, memref<192x128xi8, #gpu.address_space<workgroup>>, memref<192x128xi8, #gpu.address_space<workgroup>>
                }
                %163 = vector.bitcast %162#25 : vector<4xi8> to vector<4xf8E8M0FNU>
                %164 = vector.bitcast %162#28 : vector<4xi8> to vector<4xf8E8M0FNU>
                %165 = vector.bitcast %162#27 : vector<4xi8> to vector<4xf8E8M0FNU>
                %166 = vector.bitcast %162#26 : vector<4xi8> to vector<4xf8E8M0FNU>
                %167 = vector.bitcast %162#24 : vector<4xi8> to vector<4xf8E8M0FNU>
                scf.if %133 {
                rocdl.s.barrier
                }
                amdgpu.lds_barrier
                %reinterpret_cast_20 = memref.reinterpret_cast %162#31 to offset: [0], sizes: [24576], strides: [1] : memref<192x128xi8, #gpu.address_space<workgroup>> to memref<24576xi8, #gpu.address_space<workgroup>>
                %168 = vector.load %reinterpret_cast_20[%143] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %169 = vector.load %reinterpret_cast_20[%155] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %170 = vector.load %reinterpret_cast_20[%144] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %171 = vector.load %reinterpret_cast_20[%156] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %172 = vector.load %reinterpret_cast_20[%145] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %173 = vector.load %reinterpret_cast_20[%157] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %174 = vector.load %reinterpret_cast_20[%146] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %175 = vector.load %reinterpret_cast_20[%158] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %176 = vector.load %reinterpret_cast_20[%147] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %177 = vector.load %reinterpret_cast_20[%159] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %178 = vector.load %reinterpret_cast_20[%148] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %179 = vector.load %reinterpret_cast_20[%160] : memref<24576xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %reinterpret_cast_21 = memref.reinterpret_cast %162#29 to offset: [0], sizes: [32768], strides: [1] : memref<256x128xi8, #gpu.address_space<workgroup>> to memref<32768xi8, #gpu.address_space<workgroup>>
                %180 = vector.load %reinterpret_cast_21[%139] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %181 = vector.load %reinterpret_cast_21[%151] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %182 = vector.load %reinterpret_cast_21[%140] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %183 = vector.load %reinterpret_cast_21[%152] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %184 = vector.load %reinterpret_cast_21[%141] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %185 = vector.load %reinterpret_cast_21[%153] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %186 = vector.load %reinterpret_cast_21[%142] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %187 = vector.load %reinterpret_cast_21[%154] : memref<32768xi8, #gpu.address_space<workgroup>>, vector<16xi8>
                %188 = vector.bitcast %180 : vector<16xi8> to vector<32xf4E2M1FN>
                %189 = vector.bitcast %181 : vector<16xi8> to vector<32xf4E2M1FN>
                %190 = vector.bitcast %182 : vector<16xi8> to vector<32xf4E2M1FN>
                %191 = vector.bitcast %183 : vector<16xi8> to vector<32xf4E2M1FN>
                %192 = vector.bitcast %184 : vector<16xi8> to vector<32xf4E2M1FN>
                %193 = vector.bitcast %185 : vector<16xi8> to vector<32xf4E2M1FN>
                %194 = vector.bitcast %186 : vector<16xi8> to vector<32xf4E2M1FN>
                %195 = vector.bitcast %187 : vector<16xi8> to vector<32xf4E2M1FN>
                %196 = vector.bitcast %168 : vector<16xi8> to vector<32xf4E2M1FN>
                %197 = vector.bitcast %169 : vector<16xi8> to vector<32xf4E2M1FN>
                %198 = vector.bitcast %170 : vector<16xi8> to vector<32xf4E2M1FN>
                %199 = vector.bitcast %171 : vector<16xi8> to vector<32xf4E2M1FN>
                %200 = vector.bitcast %172 : vector<16xi8> to vector<32xf4E2M1FN>
                %201 = vector.bitcast %173 : vector<16xi8> to vector<32xf4E2M1FN>
                %202 = vector.bitcast %174 : vector<16xi8> to vector<32xf4E2M1FN>
                %203 = vector.bitcast %175 : vector<16xi8> to vector<32xf4E2M1FN>
                %204 = vector.bitcast %176 : vector<16xi8> to vector<32xf4E2M1FN>
                %205 = vector.bitcast %177 : vector<16xi8> to vector<32xf4E2M1FN>
                %206 = vector.bitcast %178 : vector<16xi8> to vector<32xf4E2M1FN>
                %207 = vector.bitcast %179 : vector<16xi8> to vector<32xf4E2M1FN>
                %208 = amdgpu.scaled_mfma 16x16x128 (%167[0] * %188) * (%166[0] * %196) + %162#0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %209 = amdgpu.scaled_mfma 16x16x128 (%167[2] * %189) * (%166[2] * %197) + %208 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %210 = amdgpu.scaled_mfma 16x16x128 (%167[0] * %188) * (%166[1] * %198) + %162#1 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %211 = amdgpu.scaled_mfma 16x16x128 (%167[2] * %189) * (%166[3] * %199) + %210 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %212 = amdgpu.scaled_mfma 16x16x128 (%167[0] * %188) * (%165[0] * %200) + %162#2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %213 = amdgpu.scaled_mfma 16x16x128 (%167[2] * %189) * (%165[2] * %201) + %212 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %214 = amdgpu.scaled_mfma 16x16x128 (%167[0] * %188) * (%165[1] * %202) + %162#3 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %215 = amdgpu.scaled_mfma 16x16x128 (%167[2] * %189) * (%165[3] * %203) + %214 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %216 = amdgpu.scaled_mfma 16x16x128 (%167[0] * %188) * (%164[0] * %204) + %162#4 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %217 = amdgpu.scaled_mfma 16x16x128 (%167[2] * %189) * (%164[2] * %205) + %216 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %218 = amdgpu.scaled_mfma 16x16x128 (%167[0] * %188) * (%164[1] * %206) + %162#5 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %219 = amdgpu.scaled_mfma 16x16x128 (%167[2] * %189) * (%164[3] * %207) + %218 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %220 = amdgpu.scaled_mfma 16x16x128 (%167[1] * %190) * (%166[0] * %196) + %162#6 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %221 = amdgpu.scaled_mfma 16x16x128 (%167[3] * %191) * (%166[2] * %197) + %220 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %222 = amdgpu.scaled_mfma 16x16x128 (%167[1] * %190) * (%166[1] * %198) + %162#7 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %223 = amdgpu.scaled_mfma 16x16x128 (%167[3] * %191) * (%166[3] * %199) + %222 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %224 = amdgpu.scaled_mfma 16x16x128 (%167[1] * %190) * (%165[0] * %200) + %162#8 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %225 = amdgpu.scaled_mfma 16x16x128 (%167[3] * %191) * (%165[2] * %201) + %224 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %226 = amdgpu.scaled_mfma 16x16x128 (%167[1] * %190) * (%165[1] * %202) + %162#9 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %227 = amdgpu.scaled_mfma 16x16x128 (%167[3] * %191) * (%165[3] * %203) + %226 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %228 = amdgpu.scaled_mfma 16x16x128 (%167[1] * %190) * (%164[0] * %204) + %162#10 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %229 = amdgpu.scaled_mfma 16x16x128 (%167[3] * %191) * (%164[2] * %205) + %228 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %230 = amdgpu.scaled_mfma 16x16x128 (%167[1] * %190) * (%164[1] * %206) + %162#11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %231 = amdgpu.scaled_mfma 16x16x128 (%167[3] * %191) * (%164[3] * %207) + %230 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %232 = amdgpu.scaled_mfma 16x16x128 (%163[0] * %192) * (%166[0] * %196) + %162#12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %233 = amdgpu.scaled_mfma 16x16x128 (%163[2] * %193) * (%166[2] * %197) + %232 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %234 = amdgpu.scaled_mfma 16x16x128 (%163[0] * %192) * (%166[1] * %198) + %162#13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %235 = amdgpu.scaled_mfma 16x16x128 (%163[2] * %193) * (%166[3] * %199) + %234 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %236 = amdgpu.scaled_mfma 16x16x128 (%163[0] * %192) * (%165[0] * %200) + %162#14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %237 = amdgpu.scaled_mfma 16x16x128 (%163[2] * %193) * (%165[2] * %201) + %236 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %238 = amdgpu.scaled_mfma 16x16x128 (%163[0] * %192) * (%165[1] * %202) + %162#15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %239 = amdgpu.scaled_mfma 16x16x128 (%163[2] * %193) * (%165[3] * %203) + %238 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %240 = amdgpu.scaled_mfma 16x16x128 (%163[0] * %192) * (%164[0] * %204) + %162#16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %241 = amdgpu.scaled_mfma 16x16x128 (%163[2] * %193) * (%164[2] * %205) + %240 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %242 = amdgpu.scaled_mfma 16x16x128 (%163[0] * %192) * (%164[1] * %206) + %162#17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %243 = amdgpu.scaled_mfma 16x16x128 (%163[2] * %193) * (%164[3] * %207) + %242 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %244 = amdgpu.scaled_mfma 16x16x128 (%163[1] * %194) * (%166[0] * %196) + %162#18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %245 = amdgpu.scaled_mfma 16x16x128 (%163[3] * %195) * (%166[2] * %197) + %244 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %246 = amdgpu.scaled_mfma 16x16x128 (%163[1] * %194) * (%166[1] * %198) + %162#19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %247 = amdgpu.scaled_mfma 16x16x128 (%163[3] * %195) * (%166[3] * %199) + %246 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %248 = amdgpu.scaled_mfma 16x16x128 (%163[1] * %194) * (%165[0] * %200) + %162#20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %249 = amdgpu.scaled_mfma 16x16x128 (%163[3] * %195) * (%165[2] * %201) + %248 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %250 = amdgpu.scaled_mfma 16x16x128 (%163[1] * %194) * (%165[1] * %202) + %162#21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %251 = amdgpu.scaled_mfma 16x16x128 (%163[3] * %195) * (%165[3] * %203) + %250 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %252 = amdgpu.scaled_mfma 16x16x128 (%163[1] * %194) * (%164[0] * %204) + %162#22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %253 = amdgpu.scaled_mfma 16x16x128 (%163[3] * %195) * (%164[2] * %205) + %252 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %254 = amdgpu.scaled_mfma 16x16x128 (%163[1] * %194) * (%164[1] * %206) + %162#23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %255 = amdgpu.scaled_mfma 16x16x128 (%163[3] * %195) * (%164[3] * %207) + %254 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
                %256 = arith.truncf %209 : vector<4xf32> to vector<4xbf16>
                %257 = arith.truncf %211 : vector<4xf32> to vector<4xbf16>
                %258 = arith.truncf %213 : vector<4xf32> to vector<4xbf16>
                %259 = arith.truncf %215 : vector<4xf32> to vector<4xbf16>
                %260 = arith.truncf %217 : vector<4xf32> to vector<4xbf16>
                %261 = arith.truncf %219 : vector<4xf32> to vector<4xbf16>
                %262 = arith.truncf %221 : vector<4xf32> to vector<4xbf16>
                %263 = arith.truncf %223 : vector<4xf32> to vector<4xbf16>
                %264 = arith.truncf %225 : vector<4xf32> to vector<4xbf16>
                %265 = arith.truncf %227 : vector<4xf32> to vector<4xbf16>
                %266 = arith.truncf %229 : vector<4xf32> to vector<4xbf16>
                %267 = arith.truncf %231 : vector<4xf32> to vector<4xbf16>
                %268 = arith.truncf %233 : vector<4xf32> to vector<4xbf16>
                %269 = arith.truncf %235 : vector<4xf32> to vector<4xbf16>
                %270 = arith.truncf %237 : vector<4xf32> to vector<4xbf16>
                %271 = arith.truncf %239 : vector<4xf32> to vector<4xbf16>
                %272 = arith.truncf %241 : vector<4xf32> to vector<4xbf16>
                %273 = arith.truncf %243 : vector<4xf32> to vector<4xbf16>
                %274 = arith.truncf %245 : vector<4xf32> to vector<4xbf16>
                %275 = arith.truncf %247 : vector<4xf32> to vector<4xbf16>
                %276 = arith.truncf %249 : vector<4xf32> to vector<4xbf16>
                %277 = arith.truncf %251 : vector<4xf32> to vector<4xbf16>
                %278 = arith.truncf %253 : vector<4xf32> to vector<4xbf16>
                %279 = arith.truncf %255 : vector<4xf32> to vector<4xbf16>
                %ep_shuffle_offset = arith.constant 1 : i32
                %ep_shuffle_width = arith.constant 64 : i32
                %ep_one = arith.constant 1 : index
                %ep_zero = arith.constant 0 : index
                %ep_lane_parity = arith.andi %thread_id_x, %ep_one : index
                %ep_is_even = arith.cmpi eq, %ep_lane_parity, %ep_zero : index

                %ep_base_buffer, %ep_offset, %ep_size0, %ep_size1, %ep_output_stride, %ep_col_stride = memref.extract_strided_metadata %reinterpret_cast_1 : memref<?x?xbf16, strided<[?, 1]>> -> memref<bf16>, index, index, index, index, index
                %ep_wg_row_off = arith.muli %76, %ep_output_stride overflow<nsw> : index
                %ep_wg_offset = arith.addi %ep_wg_row_off, %114 overflow<nsw> : index
                %ep_reinterpret = memref.reinterpret_cast %4 to offset: [%ep_wg_offset], sizes: [1073741822], strides: [1] : memref<bf16> to memref<1073741822xbf16, strided<[1], offset: ?>>
                %ep_cast = memref.cast %ep_reinterpret : memref<1073741822xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
                %ep_stride_i14 = arith.index_cast %ep_output_stride : index to i14
                %ep_buffer = amdgpu.fat_raw_buffer_cast %ep_cast validBytes(%c2147483645_i64) cacheSwizzleStride(%ep_stride_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>

                %ep_rb0_row0 = affine.apply #map49()[%thread_id_x]
                %ep_rb0_row1 = affine.apply #map51()[%thread_id_x]
                %ep_rb0_row2 = affine.apply #map52()[%thread_id_x]
                %ep_rb0_row3 = affine.apply #map53()[%thread_id_x]
                %ep_rb0_store_row_a = arith.select %ep_is_even, %ep_rb0_row0, %ep_rb0_row2 : index
                %ep_rb0_store_row_b = arith.select %ep_is_even, %ep_rb0_row1, %ep_rb0_row3 : index
                %ep_rb0_store_off_a = arith.muli %ep_rb0_store_row_a, %ep_output_stride overflow<nsw> : index
                %ep_rb0_store_off_b = arith.muli %ep_rb0_store_row_b, %ep_output_stride overflow<nsw> : index
                %ep_rb1_row0 = affine.apply #map59()[%thread_id_x]
                %ep_rb1_row1 = affine.apply #map60()[%thread_id_x]
                %ep_rb1_row2 = affine.apply #map61()[%thread_id_x]
                %ep_rb1_row3 = affine.apply #map62()[%thread_id_x]
                %ep_rb1_store_row_a = arith.select %ep_is_even, %ep_rb1_row0, %ep_rb1_row2 : index
                %ep_rb1_store_row_b = arith.select %ep_is_even, %ep_rb1_row1, %ep_rb1_row3 : index
                %ep_rb1_store_off_a = arith.muli %ep_rb1_store_row_a, %ep_output_stride overflow<nsw> : index
                %ep_rb1_store_off_b = arith.muli %ep_rb1_store_row_b, %ep_output_stride overflow<nsw> : index
                %ep_rb2_row0 = affine.apply #map63()[%thread_id_x]
                %ep_rb2_row1 = affine.apply #map64()[%thread_id_x]
                %ep_rb2_row2 = affine.apply #map65()[%thread_id_x]
                %ep_rb2_row3 = affine.apply #map66()[%thread_id_x]
                %ep_rb2_store_row_a = arith.select %ep_is_even, %ep_rb2_row0, %ep_rb2_row2 : index
                %ep_rb2_store_row_b = arith.select %ep_is_even, %ep_rb2_row1, %ep_rb2_row3 : index
                %ep_rb2_store_off_a = arith.muli %ep_rb2_store_row_a, %ep_output_stride overflow<nsw> : index
                %ep_rb2_store_off_b = arith.muli %ep_rb2_store_row_b, %ep_output_stride overflow<nsw> : index
                %ep_rb3_row0 = affine.apply #map67()[%thread_id_x]
                %ep_rb3_row1 = affine.apply #map68()[%thread_id_x]
                %ep_rb3_row2 = affine.apply #map69()[%thread_id_x]
                %ep_rb3_row3 = affine.apply #map70()[%thread_id_x]
                %ep_rb3_store_row_a = arith.select %ep_is_even, %ep_rb3_row0, %ep_rb3_row2 : index
                %ep_rb3_store_row_b = arith.select %ep_is_even, %ep_rb3_row1, %ep_rb3_row3 : index
                %ep_rb3_store_off_a = arith.muli %ep_rb3_store_row_a, %ep_output_stride overflow<nsw> : index
                %ep_rb3_store_off_b = arith.muli %ep_rb3_store_row_b, %ep_output_stride overflow<nsw> : index
                %ep_col0 = affine.apply #map50()[%thread_id_x, %thread_id_y]
                %ep_col0_adj = arith.subi %ep_col0, %ep_lane_parity : index
                %ep_col1 = affine.apply #map54()[%thread_id_x, %thread_id_y]
                %ep_col1_adj = arith.subi %ep_col1, %ep_lane_parity : index
                %ep_col2 = affine.apply #map55()[%thread_id_x, %thread_id_y]
                %ep_col2_adj = arith.subi %ep_col2, %ep_lane_parity : index
                %ep_col3 = affine.apply #map56()[%thread_id_x, %thread_id_y]
                %ep_col3_adj = arith.subi %ep_col3, %ep_lane_parity : index
                %ep_col4 = affine.apply #map57()[%thread_id_x, %thread_id_y]
                %ep_col4_adj = arith.subi %ep_col4, %ep_lane_parity : index
                %ep_col5 = affine.apply #map58()[%thread_id_x, %thread_id_y]
                %ep_col5_adj = arith.subi %ep_col5, %ep_lane_parity : index

                %ep_v209_row0 = vector.extract %209[0] : f32 from vector<4xf32>
                %ep_v209_row1 = vector.extract %209[1] : f32 from vector<4xf32>
                %ep_v209_row2 = vector.extract %209[2] : f32 from vector<4xf32>
                %ep_v209_row3 = vector.extract %209[3] : f32 from vector<4xf32>
                %ep_v209_row0_nbr, %ep_v209_row0_valid = gpu.shuffle xor %ep_v209_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v209_row1_nbr, %ep_v209_row1_valid = gpu.shuffle xor %ep_v209_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v209_row2_nbr, %ep_v209_row2_valid = gpu.shuffle xor %ep_v209_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v209_row3_nbr, %ep_v209_row3_valid = gpu.shuffle xor %ep_v209_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v209_row0_lo = arith.select %ep_is_even, %ep_v209_row0, %ep_v209_row0_nbr : f32
                %ep_v209_row0_hi = arith.select %ep_is_even, %ep_v209_row0_nbr, %ep_v209_row0 : f32
                %ep_v209_row1_lo = arith.select %ep_is_even, %ep_v209_row1, %ep_v209_row1_nbr : f32
                %ep_v209_row1_hi = arith.select %ep_is_even, %ep_v209_row1_nbr, %ep_v209_row1 : f32
                %ep_v209_row2_lo = arith.select %ep_is_even, %ep_v209_row2, %ep_v209_row2_nbr : f32
                %ep_v209_row2_hi = arith.select %ep_is_even, %ep_v209_row2_nbr, %ep_v209_row2 : f32
                %ep_v209_row3_lo = arith.select %ep_is_even, %ep_v209_row3, %ep_v209_row3_nbr : f32
                %ep_v209_row3_hi = arith.select %ep_is_even, %ep_v209_row3_nbr, %ep_v209_row3 : f32
                %ep_v209_store_a_lo = arith.select %ep_is_even, %ep_v209_row0_lo, %ep_v209_row2_lo : f32
                %ep_v209_store_a_hi = arith.select %ep_is_even, %ep_v209_row0_hi, %ep_v209_row2_hi : f32
                %ep_v209_store_b_lo = arith.select %ep_is_even, %ep_v209_row1_lo, %ep_v209_row3_lo : f32
                %ep_v209_store_b_hi = arith.select %ep_is_even, %ep_v209_row1_hi, %ep_v209_row3_hi : f32
                %ep_v209_pair_a_0 = vector.broadcast %ep_v209_store_a_lo : f32 to vector<2xf32>
                %ep_v209_pair_a = vector.insert %ep_v209_store_a_hi, %ep_v209_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v209_store_a = arith.truncf %ep_v209_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v209_pair_b_0 = vector.broadcast %ep_v209_store_b_lo : f32 to vector<2xf32>
                %ep_v209_pair_b = vector.insert %ep_v209_store_b_hi, %ep_v209_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v209_store_b = arith.truncf %ep_v209_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v209_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col0_adj overflow<nsw> : index
                %ep_v209_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col0_adj overflow<nsw> : index
                vector.store %ep_v209_store_a, %ep_buffer[%ep_v209_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v209_store_b, %ep_buffer[%ep_v209_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v211_row0 = vector.extract %211[0] : f32 from vector<4xf32>
                %ep_v211_row1 = vector.extract %211[1] : f32 from vector<4xf32>
                %ep_v211_row2 = vector.extract %211[2] : f32 from vector<4xf32>
                %ep_v211_row3 = vector.extract %211[3] : f32 from vector<4xf32>
                %ep_v211_row0_nbr, %ep_v211_row0_valid = gpu.shuffle xor %ep_v211_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v211_row1_nbr, %ep_v211_row1_valid = gpu.shuffle xor %ep_v211_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v211_row2_nbr, %ep_v211_row2_valid = gpu.shuffle xor %ep_v211_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v211_row3_nbr, %ep_v211_row3_valid = gpu.shuffle xor %ep_v211_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v211_row0_lo = arith.select %ep_is_even, %ep_v211_row0, %ep_v211_row0_nbr : f32
                %ep_v211_row0_hi = arith.select %ep_is_even, %ep_v211_row0_nbr, %ep_v211_row0 : f32
                %ep_v211_row1_lo = arith.select %ep_is_even, %ep_v211_row1, %ep_v211_row1_nbr : f32
                %ep_v211_row1_hi = arith.select %ep_is_even, %ep_v211_row1_nbr, %ep_v211_row1 : f32
                %ep_v211_row2_lo = arith.select %ep_is_even, %ep_v211_row2, %ep_v211_row2_nbr : f32
                %ep_v211_row2_hi = arith.select %ep_is_even, %ep_v211_row2_nbr, %ep_v211_row2 : f32
                %ep_v211_row3_lo = arith.select %ep_is_even, %ep_v211_row3, %ep_v211_row3_nbr : f32
                %ep_v211_row3_hi = arith.select %ep_is_even, %ep_v211_row3_nbr, %ep_v211_row3 : f32
                %ep_v211_store_a_lo = arith.select %ep_is_even, %ep_v211_row0_lo, %ep_v211_row2_lo : f32
                %ep_v211_store_a_hi = arith.select %ep_is_even, %ep_v211_row0_hi, %ep_v211_row2_hi : f32
                %ep_v211_store_b_lo = arith.select %ep_is_even, %ep_v211_row1_lo, %ep_v211_row3_lo : f32
                %ep_v211_store_b_hi = arith.select %ep_is_even, %ep_v211_row1_hi, %ep_v211_row3_hi : f32
                %ep_v211_pair_a_0 = vector.broadcast %ep_v211_store_a_lo : f32 to vector<2xf32>
                %ep_v211_pair_a = vector.insert %ep_v211_store_a_hi, %ep_v211_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v211_store_a = arith.truncf %ep_v211_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v211_pair_b_0 = vector.broadcast %ep_v211_store_b_lo : f32 to vector<2xf32>
                %ep_v211_pair_b = vector.insert %ep_v211_store_b_hi, %ep_v211_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v211_store_b = arith.truncf %ep_v211_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v211_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col1_adj overflow<nsw> : index
                %ep_v211_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col1_adj overflow<nsw> : index
                vector.store %ep_v211_store_a, %ep_buffer[%ep_v211_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v211_store_b, %ep_buffer[%ep_v211_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v213_row0 = vector.extract %213[0] : f32 from vector<4xf32>
                %ep_v213_row1 = vector.extract %213[1] : f32 from vector<4xf32>
                %ep_v213_row2 = vector.extract %213[2] : f32 from vector<4xf32>
                %ep_v213_row3 = vector.extract %213[3] : f32 from vector<4xf32>
                %ep_v213_row0_nbr, %ep_v213_row0_valid = gpu.shuffle xor %ep_v213_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v213_row1_nbr, %ep_v213_row1_valid = gpu.shuffle xor %ep_v213_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v213_row2_nbr, %ep_v213_row2_valid = gpu.shuffle xor %ep_v213_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v213_row3_nbr, %ep_v213_row3_valid = gpu.shuffle xor %ep_v213_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v213_row0_lo = arith.select %ep_is_even, %ep_v213_row0, %ep_v213_row0_nbr : f32
                %ep_v213_row0_hi = arith.select %ep_is_even, %ep_v213_row0_nbr, %ep_v213_row0 : f32
                %ep_v213_row1_lo = arith.select %ep_is_even, %ep_v213_row1, %ep_v213_row1_nbr : f32
                %ep_v213_row1_hi = arith.select %ep_is_even, %ep_v213_row1_nbr, %ep_v213_row1 : f32
                %ep_v213_row2_lo = arith.select %ep_is_even, %ep_v213_row2, %ep_v213_row2_nbr : f32
                %ep_v213_row2_hi = arith.select %ep_is_even, %ep_v213_row2_nbr, %ep_v213_row2 : f32
                %ep_v213_row3_lo = arith.select %ep_is_even, %ep_v213_row3, %ep_v213_row3_nbr : f32
                %ep_v213_row3_hi = arith.select %ep_is_even, %ep_v213_row3_nbr, %ep_v213_row3 : f32
                %ep_v213_store_a_lo = arith.select %ep_is_even, %ep_v213_row0_lo, %ep_v213_row2_lo : f32
                %ep_v213_store_a_hi = arith.select %ep_is_even, %ep_v213_row0_hi, %ep_v213_row2_hi : f32
                %ep_v213_store_b_lo = arith.select %ep_is_even, %ep_v213_row1_lo, %ep_v213_row3_lo : f32
                %ep_v213_store_b_hi = arith.select %ep_is_even, %ep_v213_row1_hi, %ep_v213_row3_hi : f32
                %ep_v213_pair_a_0 = vector.broadcast %ep_v213_store_a_lo : f32 to vector<2xf32>
                %ep_v213_pair_a = vector.insert %ep_v213_store_a_hi, %ep_v213_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v213_store_a = arith.truncf %ep_v213_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v213_pair_b_0 = vector.broadcast %ep_v213_store_b_lo : f32 to vector<2xf32>
                %ep_v213_pair_b = vector.insert %ep_v213_store_b_hi, %ep_v213_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v213_store_b = arith.truncf %ep_v213_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v213_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col2_adj overflow<nsw> : index
                %ep_v213_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col2_adj overflow<nsw> : index
                vector.store %ep_v213_store_a, %ep_buffer[%ep_v213_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v213_store_b, %ep_buffer[%ep_v213_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v215_row0 = vector.extract %215[0] : f32 from vector<4xf32>
                %ep_v215_row1 = vector.extract %215[1] : f32 from vector<4xf32>
                %ep_v215_row2 = vector.extract %215[2] : f32 from vector<4xf32>
                %ep_v215_row3 = vector.extract %215[3] : f32 from vector<4xf32>
                %ep_v215_row0_nbr, %ep_v215_row0_valid = gpu.shuffle xor %ep_v215_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v215_row1_nbr, %ep_v215_row1_valid = gpu.shuffle xor %ep_v215_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v215_row2_nbr, %ep_v215_row2_valid = gpu.shuffle xor %ep_v215_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v215_row3_nbr, %ep_v215_row3_valid = gpu.shuffle xor %ep_v215_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v215_row0_lo = arith.select %ep_is_even, %ep_v215_row0, %ep_v215_row0_nbr : f32
                %ep_v215_row0_hi = arith.select %ep_is_even, %ep_v215_row0_nbr, %ep_v215_row0 : f32
                %ep_v215_row1_lo = arith.select %ep_is_even, %ep_v215_row1, %ep_v215_row1_nbr : f32
                %ep_v215_row1_hi = arith.select %ep_is_even, %ep_v215_row1_nbr, %ep_v215_row1 : f32
                %ep_v215_row2_lo = arith.select %ep_is_even, %ep_v215_row2, %ep_v215_row2_nbr : f32
                %ep_v215_row2_hi = arith.select %ep_is_even, %ep_v215_row2_nbr, %ep_v215_row2 : f32
                %ep_v215_row3_lo = arith.select %ep_is_even, %ep_v215_row3, %ep_v215_row3_nbr : f32
                %ep_v215_row3_hi = arith.select %ep_is_even, %ep_v215_row3_nbr, %ep_v215_row3 : f32
                %ep_v215_store_a_lo = arith.select %ep_is_even, %ep_v215_row0_lo, %ep_v215_row2_lo : f32
                %ep_v215_store_a_hi = arith.select %ep_is_even, %ep_v215_row0_hi, %ep_v215_row2_hi : f32
                %ep_v215_store_b_lo = arith.select %ep_is_even, %ep_v215_row1_lo, %ep_v215_row3_lo : f32
                %ep_v215_store_b_hi = arith.select %ep_is_even, %ep_v215_row1_hi, %ep_v215_row3_hi : f32
                %ep_v215_pair_a_0 = vector.broadcast %ep_v215_store_a_lo : f32 to vector<2xf32>
                %ep_v215_pair_a = vector.insert %ep_v215_store_a_hi, %ep_v215_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v215_store_a = arith.truncf %ep_v215_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v215_pair_b_0 = vector.broadcast %ep_v215_store_b_lo : f32 to vector<2xf32>
                %ep_v215_pair_b = vector.insert %ep_v215_store_b_hi, %ep_v215_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v215_store_b = arith.truncf %ep_v215_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v215_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col3_adj overflow<nsw> : index
                %ep_v215_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col3_adj overflow<nsw> : index
                vector.store %ep_v215_store_a, %ep_buffer[%ep_v215_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v215_store_b, %ep_buffer[%ep_v215_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v217_row0 = vector.extract %217[0] : f32 from vector<4xf32>
                %ep_v217_row1 = vector.extract %217[1] : f32 from vector<4xf32>
                %ep_v217_row2 = vector.extract %217[2] : f32 from vector<4xf32>
                %ep_v217_row3 = vector.extract %217[3] : f32 from vector<4xf32>
                %ep_v217_row0_nbr, %ep_v217_row0_valid = gpu.shuffle xor %ep_v217_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v217_row1_nbr, %ep_v217_row1_valid = gpu.shuffle xor %ep_v217_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v217_row2_nbr, %ep_v217_row2_valid = gpu.shuffle xor %ep_v217_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v217_row3_nbr, %ep_v217_row3_valid = gpu.shuffle xor %ep_v217_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v217_row0_lo = arith.select %ep_is_even, %ep_v217_row0, %ep_v217_row0_nbr : f32
                %ep_v217_row0_hi = arith.select %ep_is_even, %ep_v217_row0_nbr, %ep_v217_row0 : f32
                %ep_v217_row1_lo = arith.select %ep_is_even, %ep_v217_row1, %ep_v217_row1_nbr : f32
                %ep_v217_row1_hi = arith.select %ep_is_even, %ep_v217_row1_nbr, %ep_v217_row1 : f32
                %ep_v217_row2_lo = arith.select %ep_is_even, %ep_v217_row2, %ep_v217_row2_nbr : f32
                %ep_v217_row2_hi = arith.select %ep_is_even, %ep_v217_row2_nbr, %ep_v217_row2 : f32
                %ep_v217_row3_lo = arith.select %ep_is_even, %ep_v217_row3, %ep_v217_row3_nbr : f32
                %ep_v217_row3_hi = arith.select %ep_is_even, %ep_v217_row3_nbr, %ep_v217_row3 : f32
                %ep_v217_store_a_lo = arith.select %ep_is_even, %ep_v217_row0_lo, %ep_v217_row2_lo : f32
                %ep_v217_store_a_hi = arith.select %ep_is_even, %ep_v217_row0_hi, %ep_v217_row2_hi : f32
                %ep_v217_store_b_lo = arith.select %ep_is_even, %ep_v217_row1_lo, %ep_v217_row3_lo : f32
                %ep_v217_store_b_hi = arith.select %ep_is_even, %ep_v217_row1_hi, %ep_v217_row3_hi : f32
                %ep_v217_pair_a_0 = vector.broadcast %ep_v217_store_a_lo : f32 to vector<2xf32>
                %ep_v217_pair_a = vector.insert %ep_v217_store_a_hi, %ep_v217_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v217_store_a = arith.truncf %ep_v217_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v217_pair_b_0 = vector.broadcast %ep_v217_store_b_lo : f32 to vector<2xf32>
                %ep_v217_pair_b = vector.insert %ep_v217_store_b_hi, %ep_v217_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v217_store_b = arith.truncf %ep_v217_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v217_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col4_adj overflow<nsw> : index
                %ep_v217_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col4_adj overflow<nsw> : index
                vector.store %ep_v217_store_a, %ep_buffer[%ep_v217_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v217_store_b, %ep_buffer[%ep_v217_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v219_row0 = vector.extract %219[0] : f32 from vector<4xf32>
                %ep_v219_row1 = vector.extract %219[1] : f32 from vector<4xf32>
                %ep_v219_row2 = vector.extract %219[2] : f32 from vector<4xf32>
                %ep_v219_row3 = vector.extract %219[3] : f32 from vector<4xf32>
                %ep_v219_row0_nbr, %ep_v219_row0_valid = gpu.shuffle xor %ep_v219_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v219_row1_nbr, %ep_v219_row1_valid = gpu.shuffle xor %ep_v219_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v219_row2_nbr, %ep_v219_row2_valid = gpu.shuffle xor %ep_v219_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v219_row3_nbr, %ep_v219_row3_valid = gpu.shuffle xor %ep_v219_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v219_row0_lo = arith.select %ep_is_even, %ep_v219_row0, %ep_v219_row0_nbr : f32
                %ep_v219_row0_hi = arith.select %ep_is_even, %ep_v219_row0_nbr, %ep_v219_row0 : f32
                %ep_v219_row1_lo = arith.select %ep_is_even, %ep_v219_row1, %ep_v219_row1_nbr : f32
                %ep_v219_row1_hi = arith.select %ep_is_even, %ep_v219_row1_nbr, %ep_v219_row1 : f32
                %ep_v219_row2_lo = arith.select %ep_is_even, %ep_v219_row2, %ep_v219_row2_nbr : f32
                %ep_v219_row2_hi = arith.select %ep_is_even, %ep_v219_row2_nbr, %ep_v219_row2 : f32
                %ep_v219_row3_lo = arith.select %ep_is_even, %ep_v219_row3, %ep_v219_row3_nbr : f32
                %ep_v219_row3_hi = arith.select %ep_is_even, %ep_v219_row3_nbr, %ep_v219_row3 : f32
                %ep_v219_store_a_lo = arith.select %ep_is_even, %ep_v219_row0_lo, %ep_v219_row2_lo : f32
                %ep_v219_store_a_hi = arith.select %ep_is_even, %ep_v219_row0_hi, %ep_v219_row2_hi : f32
                %ep_v219_store_b_lo = arith.select %ep_is_even, %ep_v219_row1_lo, %ep_v219_row3_lo : f32
                %ep_v219_store_b_hi = arith.select %ep_is_even, %ep_v219_row1_hi, %ep_v219_row3_hi : f32
                %ep_v219_pair_a_0 = vector.broadcast %ep_v219_store_a_lo : f32 to vector<2xf32>
                %ep_v219_pair_a = vector.insert %ep_v219_store_a_hi, %ep_v219_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v219_store_a = arith.truncf %ep_v219_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v219_pair_b_0 = vector.broadcast %ep_v219_store_b_lo : f32 to vector<2xf32>
                %ep_v219_pair_b = vector.insert %ep_v219_store_b_hi, %ep_v219_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v219_store_b = arith.truncf %ep_v219_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v219_addr_a = arith.addi %ep_rb0_store_off_a, %ep_col5_adj overflow<nsw> : index
                %ep_v219_addr_b = arith.addi %ep_rb0_store_off_b, %ep_col5_adj overflow<nsw> : index
                vector.store %ep_v219_store_a, %ep_buffer[%ep_v219_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v219_store_b, %ep_buffer[%ep_v219_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v221_row0 = vector.extract %221[0] : f32 from vector<4xf32>
                %ep_v221_row1 = vector.extract %221[1] : f32 from vector<4xf32>
                %ep_v221_row2 = vector.extract %221[2] : f32 from vector<4xf32>
                %ep_v221_row3 = vector.extract %221[3] : f32 from vector<4xf32>
                %ep_v221_row0_nbr, %ep_v221_row0_valid = gpu.shuffle xor %ep_v221_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v221_row1_nbr, %ep_v221_row1_valid = gpu.shuffle xor %ep_v221_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v221_row2_nbr, %ep_v221_row2_valid = gpu.shuffle xor %ep_v221_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v221_row3_nbr, %ep_v221_row3_valid = gpu.shuffle xor %ep_v221_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v221_row0_lo = arith.select %ep_is_even, %ep_v221_row0, %ep_v221_row0_nbr : f32
                %ep_v221_row0_hi = arith.select %ep_is_even, %ep_v221_row0_nbr, %ep_v221_row0 : f32
                %ep_v221_row1_lo = arith.select %ep_is_even, %ep_v221_row1, %ep_v221_row1_nbr : f32
                %ep_v221_row1_hi = arith.select %ep_is_even, %ep_v221_row1_nbr, %ep_v221_row1 : f32
                %ep_v221_row2_lo = arith.select %ep_is_even, %ep_v221_row2, %ep_v221_row2_nbr : f32
                %ep_v221_row2_hi = arith.select %ep_is_even, %ep_v221_row2_nbr, %ep_v221_row2 : f32
                %ep_v221_row3_lo = arith.select %ep_is_even, %ep_v221_row3, %ep_v221_row3_nbr : f32
                %ep_v221_row3_hi = arith.select %ep_is_even, %ep_v221_row3_nbr, %ep_v221_row3 : f32
                %ep_v221_store_a_lo = arith.select %ep_is_even, %ep_v221_row0_lo, %ep_v221_row2_lo : f32
                %ep_v221_store_a_hi = arith.select %ep_is_even, %ep_v221_row0_hi, %ep_v221_row2_hi : f32
                %ep_v221_store_b_lo = arith.select %ep_is_even, %ep_v221_row1_lo, %ep_v221_row3_lo : f32
                %ep_v221_store_b_hi = arith.select %ep_is_even, %ep_v221_row1_hi, %ep_v221_row3_hi : f32
                %ep_v221_pair_a_0 = vector.broadcast %ep_v221_store_a_lo : f32 to vector<2xf32>
                %ep_v221_pair_a = vector.insert %ep_v221_store_a_hi, %ep_v221_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v221_store_a = arith.truncf %ep_v221_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v221_pair_b_0 = vector.broadcast %ep_v221_store_b_lo : f32 to vector<2xf32>
                %ep_v221_pair_b = vector.insert %ep_v221_store_b_hi, %ep_v221_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v221_store_b = arith.truncf %ep_v221_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v221_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col0_adj overflow<nsw> : index
                %ep_v221_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col0_adj overflow<nsw> : index
                vector.store %ep_v221_store_a, %ep_buffer[%ep_v221_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v221_store_b, %ep_buffer[%ep_v221_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v223_row0 = vector.extract %223[0] : f32 from vector<4xf32>
                %ep_v223_row1 = vector.extract %223[1] : f32 from vector<4xf32>
                %ep_v223_row2 = vector.extract %223[2] : f32 from vector<4xf32>
                %ep_v223_row3 = vector.extract %223[3] : f32 from vector<4xf32>
                %ep_v223_row0_nbr, %ep_v223_row0_valid = gpu.shuffle xor %ep_v223_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v223_row1_nbr, %ep_v223_row1_valid = gpu.shuffle xor %ep_v223_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v223_row2_nbr, %ep_v223_row2_valid = gpu.shuffle xor %ep_v223_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v223_row3_nbr, %ep_v223_row3_valid = gpu.shuffle xor %ep_v223_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v223_row0_lo = arith.select %ep_is_even, %ep_v223_row0, %ep_v223_row0_nbr : f32
                %ep_v223_row0_hi = arith.select %ep_is_even, %ep_v223_row0_nbr, %ep_v223_row0 : f32
                %ep_v223_row1_lo = arith.select %ep_is_even, %ep_v223_row1, %ep_v223_row1_nbr : f32
                %ep_v223_row1_hi = arith.select %ep_is_even, %ep_v223_row1_nbr, %ep_v223_row1 : f32
                %ep_v223_row2_lo = arith.select %ep_is_even, %ep_v223_row2, %ep_v223_row2_nbr : f32
                %ep_v223_row2_hi = arith.select %ep_is_even, %ep_v223_row2_nbr, %ep_v223_row2 : f32
                %ep_v223_row3_lo = arith.select %ep_is_even, %ep_v223_row3, %ep_v223_row3_nbr : f32
                %ep_v223_row3_hi = arith.select %ep_is_even, %ep_v223_row3_nbr, %ep_v223_row3 : f32
                %ep_v223_store_a_lo = arith.select %ep_is_even, %ep_v223_row0_lo, %ep_v223_row2_lo : f32
                %ep_v223_store_a_hi = arith.select %ep_is_even, %ep_v223_row0_hi, %ep_v223_row2_hi : f32
                %ep_v223_store_b_lo = arith.select %ep_is_even, %ep_v223_row1_lo, %ep_v223_row3_lo : f32
                %ep_v223_store_b_hi = arith.select %ep_is_even, %ep_v223_row1_hi, %ep_v223_row3_hi : f32
                %ep_v223_pair_a_0 = vector.broadcast %ep_v223_store_a_lo : f32 to vector<2xf32>
                %ep_v223_pair_a = vector.insert %ep_v223_store_a_hi, %ep_v223_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v223_store_a = arith.truncf %ep_v223_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v223_pair_b_0 = vector.broadcast %ep_v223_store_b_lo : f32 to vector<2xf32>
                %ep_v223_pair_b = vector.insert %ep_v223_store_b_hi, %ep_v223_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v223_store_b = arith.truncf %ep_v223_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v223_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col1_adj overflow<nsw> : index
                %ep_v223_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col1_adj overflow<nsw> : index
                vector.store %ep_v223_store_a, %ep_buffer[%ep_v223_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v223_store_b, %ep_buffer[%ep_v223_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v225_row0 = vector.extract %225[0] : f32 from vector<4xf32>
                %ep_v225_row1 = vector.extract %225[1] : f32 from vector<4xf32>
                %ep_v225_row2 = vector.extract %225[2] : f32 from vector<4xf32>
                %ep_v225_row3 = vector.extract %225[3] : f32 from vector<4xf32>
                %ep_v225_row0_nbr, %ep_v225_row0_valid = gpu.shuffle xor %ep_v225_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v225_row1_nbr, %ep_v225_row1_valid = gpu.shuffle xor %ep_v225_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v225_row2_nbr, %ep_v225_row2_valid = gpu.shuffle xor %ep_v225_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v225_row3_nbr, %ep_v225_row3_valid = gpu.shuffle xor %ep_v225_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v225_row0_lo = arith.select %ep_is_even, %ep_v225_row0, %ep_v225_row0_nbr : f32
                %ep_v225_row0_hi = arith.select %ep_is_even, %ep_v225_row0_nbr, %ep_v225_row0 : f32
                %ep_v225_row1_lo = arith.select %ep_is_even, %ep_v225_row1, %ep_v225_row1_nbr : f32
                %ep_v225_row1_hi = arith.select %ep_is_even, %ep_v225_row1_nbr, %ep_v225_row1 : f32
                %ep_v225_row2_lo = arith.select %ep_is_even, %ep_v225_row2, %ep_v225_row2_nbr : f32
                %ep_v225_row2_hi = arith.select %ep_is_even, %ep_v225_row2_nbr, %ep_v225_row2 : f32
                %ep_v225_row3_lo = arith.select %ep_is_even, %ep_v225_row3, %ep_v225_row3_nbr : f32
                %ep_v225_row3_hi = arith.select %ep_is_even, %ep_v225_row3_nbr, %ep_v225_row3 : f32
                %ep_v225_store_a_lo = arith.select %ep_is_even, %ep_v225_row0_lo, %ep_v225_row2_lo : f32
                %ep_v225_store_a_hi = arith.select %ep_is_even, %ep_v225_row0_hi, %ep_v225_row2_hi : f32
                %ep_v225_store_b_lo = arith.select %ep_is_even, %ep_v225_row1_lo, %ep_v225_row3_lo : f32
                %ep_v225_store_b_hi = arith.select %ep_is_even, %ep_v225_row1_hi, %ep_v225_row3_hi : f32
                %ep_v225_pair_a_0 = vector.broadcast %ep_v225_store_a_lo : f32 to vector<2xf32>
                %ep_v225_pair_a = vector.insert %ep_v225_store_a_hi, %ep_v225_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v225_store_a = arith.truncf %ep_v225_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v225_pair_b_0 = vector.broadcast %ep_v225_store_b_lo : f32 to vector<2xf32>
                %ep_v225_pair_b = vector.insert %ep_v225_store_b_hi, %ep_v225_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v225_store_b = arith.truncf %ep_v225_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v225_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col2_adj overflow<nsw> : index
                %ep_v225_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col2_adj overflow<nsw> : index
                vector.store %ep_v225_store_a, %ep_buffer[%ep_v225_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v225_store_b, %ep_buffer[%ep_v225_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v227_row0 = vector.extract %227[0] : f32 from vector<4xf32>
                %ep_v227_row1 = vector.extract %227[1] : f32 from vector<4xf32>
                %ep_v227_row2 = vector.extract %227[2] : f32 from vector<4xf32>
                %ep_v227_row3 = vector.extract %227[3] : f32 from vector<4xf32>
                %ep_v227_row0_nbr, %ep_v227_row0_valid = gpu.shuffle xor %ep_v227_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v227_row1_nbr, %ep_v227_row1_valid = gpu.shuffle xor %ep_v227_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v227_row2_nbr, %ep_v227_row2_valid = gpu.shuffle xor %ep_v227_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v227_row3_nbr, %ep_v227_row3_valid = gpu.shuffle xor %ep_v227_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v227_row0_lo = arith.select %ep_is_even, %ep_v227_row0, %ep_v227_row0_nbr : f32
                %ep_v227_row0_hi = arith.select %ep_is_even, %ep_v227_row0_nbr, %ep_v227_row0 : f32
                %ep_v227_row1_lo = arith.select %ep_is_even, %ep_v227_row1, %ep_v227_row1_nbr : f32
                %ep_v227_row1_hi = arith.select %ep_is_even, %ep_v227_row1_nbr, %ep_v227_row1 : f32
                %ep_v227_row2_lo = arith.select %ep_is_even, %ep_v227_row2, %ep_v227_row2_nbr : f32
                %ep_v227_row2_hi = arith.select %ep_is_even, %ep_v227_row2_nbr, %ep_v227_row2 : f32
                %ep_v227_row3_lo = arith.select %ep_is_even, %ep_v227_row3, %ep_v227_row3_nbr : f32
                %ep_v227_row3_hi = arith.select %ep_is_even, %ep_v227_row3_nbr, %ep_v227_row3 : f32
                %ep_v227_store_a_lo = arith.select %ep_is_even, %ep_v227_row0_lo, %ep_v227_row2_lo : f32
                %ep_v227_store_a_hi = arith.select %ep_is_even, %ep_v227_row0_hi, %ep_v227_row2_hi : f32
                %ep_v227_store_b_lo = arith.select %ep_is_even, %ep_v227_row1_lo, %ep_v227_row3_lo : f32
                %ep_v227_store_b_hi = arith.select %ep_is_even, %ep_v227_row1_hi, %ep_v227_row3_hi : f32
                %ep_v227_pair_a_0 = vector.broadcast %ep_v227_store_a_lo : f32 to vector<2xf32>
                %ep_v227_pair_a = vector.insert %ep_v227_store_a_hi, %ep_v227_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v227_store_a = arith.truncf %ep_v227_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v227_pair_b_0 = vector.broadcast %ep_v227_store_b_lo : f32 to vector<2xf32>
                %ep_v227_pair_b = vector.insert %ep_v227_store_b_hi, %ep_v227_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v227_store_b = arith.truncf %ep_v227_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v227_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col3_adj overflow<nsw> : index
                %ep_v227_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col3_adj overflow<nsw> : index
                vector.store %ep_v227_store_a, %ep_buffer[%ep_v227_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v227_store_b, %ep_buffer[%ep_v227_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v229_row0 = vector.extract %229[0] : f32 from vector<4xf32>
                %ep_v229_row1 = vector.extract %229[1] : f32 from vector<4xf32>
                %ep_v229_row2 = vector.extract %229[2] : f32 from vector<4xf32>
                %ep_v229_row3 = vector.extract %229[3] : f32 from vector<4xf32>
                %ep_v229_row0_nbr, %ep_v229_row0_valid = gpu.shuffle xor %ep_v229_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v229_row1_nbr, %ep_v229_row1_valid = gpu.shuffle xor %ep_v229_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v229_row2_nbr, %ep_v229_row2_valid = gpu.shuffle xor %ep_v229_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v229_row3_nbr, %ep_v229_row3_valid = gpu.shuffle xor %ep_v229_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v229_row0_lo = arith.select %ep_is_even, %ep_v229_row0, %ep_v229_row0_nbr : f32
                %ep_v229_row0_hi = arith.select %ep_is_even, %ep_v229_row0_nbr, %ep_v229_row0 : f32
                %ep_v229_row1_lo = arith.select %ep_is_even, %ep_v229_row1, %ep_v229_row1_nbr : f32
                %ep_v229_row1_hi = arith.select %ep_is_even, %ep_v229_row1_nbr, %ep_v229_row1 : f32
                %ep_v229_row2_lo = arith.select %ep_is_even, %ep_v229_row2, %ep_v229_row2_nbr : f32
                %ep_v229_row2_hi = arith.select %ep_is_even, %ep_v229_row2_nbr, %ep_v229_row2 : f32
                %ep_v229_row3_lo = arith.select %ep_is_even, %ep_v229_row3, %ep_v229_row3_nbr : f32
                %ep_v229_row3_hi = arith.select %ep_is_even, %ep_v229_row3_nbr, %ep_v229_row3 : f32
                %ep_v229_store_a_lo = arith.select %ep_is_even, %ep_v229_row0_lo, %ep_v229_row2_lo : f32
                %ep_v229_store_a_hi = arith.select %ep_is_even, %ep_v229_row0_hi, %ep_v229_row2_hi : f32
                %ep_v229_store_b_lo = arith.select %ep_is_even, %ep_v229_row1_lo, %ep_v229_row3_lo : f32
                %ep_v229_store_b_hi = arith.select %ep_is_even, %ep_v229_row1_hi, %ep_v229_row3_hi : f32
                %ep_v229_pair_a_0 = vector.broadcast %ep_v229_store_a_lo : f32 to vector<2xf32>
                %ep_v229_pair_a = vector.insert %ep_v229_store_a_hi, %ep_v229_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v229_store_a = arith.truncf %ep_v229_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v229_pair_b_0 = vector.broadcast %ep_v229_store_b_lo : f32 to vector<2xf32>
                %ep_v229_pair_b = vector.insert %ep_v229_store_b_hi, %ep_v229_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v229_store_b = arith.truncf %ep_v229_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v229_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col4_adj overflow<nsw> : index
                %ep_v229_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col4_adj overflow<nsw> : index
                vector.store %ep_v229_store_a, %ep_buffer[%ep_v229_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v229_store_b, %ep_buffer[%ep_v229_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v231_row0 = vector.extract %231[0] : f32 from vector<4xf32>
                %ep_v231_row1 = vector.extract %231[1] : f32 from vector<4xf32>
                %ep_v231_row2 = vector.extract %231[2] : f32 from vector<4xf32>
                %ep_v231_row3 = vector.extract %231[3] : f32 from vector<4xf32>
                %ep_v231_row0_nbr, %ep_v231_row0_valid = gpu.shuffle xor %ep_v231_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v231_row1_nbr, %ep_v231_row1_valid = gpu.shuffle xor %ep_v231_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v231_row2_nbr, %ep_v231_row2_valid = gpu.shuffle xor %ep_v231_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v231_row3_nbr, %ep_v231_row3_valid = gpu.shuffle xor %ep_v231_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v231_row0_lo = arith.select %ep_is_even, %ep_v231_row0, %ep_v231_row0_nbr : f32
                %ep_v231_row0_hi = arith.select %ep_is_even, %ep_v231_row0_nbr, %ep_v231_row0 : f32
                %ep_v231_row1_lo = arith.select %ep_is_even, %ep_v231_row1, %ep_v231_row1_nbr : f32
                %ep_v231_row1_hi = arith.select %ep_is_even, %ep_v231_row1_nbr, %ep_v231_row1 : f32
                %ep_v231_row2_lo = arith.select %ep_is_even, %ep_v231_row2, %ep_v231_row2_nbr : f32
                %ep_v231_row2_hi = arith.select %ep_is_even, %ep_v231_row2_nbr, %ep_v231_row2 : f32
                %ep_v231_row3_lo = arith.select %ep_is_even, %ep_v231_row3, %ep_v231_row3_nbr : f32
                %ep_v231_row3_hi = arith.select %ep_is_even, %ep_v231_row3_nbr, %ep_v231_row3 : f32
                %ep_v231_store_a_lo = arith.select %ep_is_even, %ep_v231_row0_lo, %ep_v231_row2_lo : f32
                %ep_v231_store_a_hi = arith.select %ep_is_even, %ep_v231_row0_hi, %ep_v231_row2_hi : f32
                %ep_v231_store_b_lo = arith.select %ep_is_even, %ep_v231_row1_lo, %ep_v231_row3_lo : f32
                %ep_v231_store_b_hi = arith.select %ep_is_even, %ep_v231_row1_hi, %ep_v231_row3_hi : f32
                %ep_v231_pair_a_0 = vector.broadcast %ep_v231_store_a_lo : f32 to vector<2xf32>
                %ep_v231_pair_a = vector.insert %ep_v231_store_a_hi, %ep_v231_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v231_store_a = arith.truncf %ep_v231_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v231_pair_b_0 = vector.broadcast %ep_v231_store_b_lo : f32 to vector<2xf32>
                %ep_v231_pair_b = vector.insert %ep_v231_store_b_hi, %ep_v231_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v231_store_b = arith.truncf %ep_v231_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v231_addr_a = arith.addi %ep_rb1_store_off_a, %ep_col5_adj overflow<nsw> : index
                %ep_v231_addr_b = arith.addi %ep_rb1_store_off_b, %ep_col5_adj overflow<nsw> : index
                vector.store %ep_v231_store_a, %ep_buffer[%ep_v231_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v231_store_b, %ep_buffer[%ep_v231_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v233_row0 = vector.extract %233[0] : f32 from vector<4xf32>
                %ep_v233_row1 = vector.extract %233[1] : f32 from vector<4xf32>
                %ep_v233_row2 = vector.extract %233[2] : f32 from vector<4xf32>
                %ep_v233_row3 = vector.extract %233[3] : f32 from vector<4xf32>
                %ep_v233_row0_nbr, %ep_v233_row0_valid = gpu.shuffle xor %ep_v233_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v233_row1_nbr, %ep_v233_row1_valid = gpu.shuffle xor %ep_v233_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v233_row2_nbr, %ep_v233_row2_valid = gpu.shuffle xor %ep_v233_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v233_row3_nbr, %ep_v233_row3_valid = gpu.shuffle xor %ep_v233_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v233_row0_lo = arith.select %ep_is_even, %ep_v233_row0, %ep_v233_row0_nbr : f32
                %ep_v233_row0_hi = arith.select %ep_is_even, %ep_v233_row0_nbr, %ep_v233_row0 : f32
                %ep_v233_row1_lo = arith.select %ep_is_even, %ep_v233_row1, %ep_v233_row1_nbr : f32
                %ep_v233_row1_hi = arith.select %ep_is_even, %ep_v233_row1_nbr, %ep_v233_row1 : f32
                %ep_v233_row2_lo = arith.select %ep_is_even, %ep_v233_row2, %ep_v233_row2_nbr : f32
                %ep_v233_row2_hi = arith.select %ep_is_even, %ep_v233_row2_nbr, %ep_v233_row2 : f32
                %ep_v233_row3_lo = arith.select %ep_is_even, %ep_v233_row3, %ep_v233_row3_nbr : f32
                %ep_v233_row3_hi = arith.select %ep_is_even, %ep_v233_row3_nbr, %ep_v233_row3 : f32
                %ep_v233_store_a_lo = arith.select %ep_is_even, %ep_v233_row0_lo, %ep_v233_row2_lo : f32
                %ep_v233_store_a_hi = arith.select %ep_is_even, %ep_v233_row0_hi, %ep_v233_row2_hi : f32
                %ep_v233_store_b_lo = arith.select %ep_is_even, %ep_v233_row1_lo, %ep_v233_row3_lo : f32
                %ep_v233_store_b_hi = arith.select %ep_is_even, %ep_v233_row1_hi, %ep_v233_row3_hi : f32
                %ep_v233_pair_a_0 = vector.broadcast %ep_v233_store_a_lo : f32 to vector<2xf32>
                %ep_v233_pair_a = vector.insert %ep_v233_store_a_hi, %ep_v233_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v233_store_a = arith.truncf %ep_v233_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v233_pair_b_0 = vector.broadcast %ep_v233_store_b_lo : f32 to vector<2xf32>
                %ep_v233_pair_b = vector.insert %ep_v233_store_b_hi, %ep_v233_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v233_store_b = arith.truncf %ep_v233_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v233_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col0_adj overflow<nsw> : index
                %ep_v233_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col0_adj overflow<nsw> : index
                vector.store %ep_v233_store_a, %ep_buffer[%ep_v233_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v233_store_b, %ep_buffer[%ep_v233_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v235_row0 = vector.extract %235[0] : f32 from vector<4xf32>
                %ep_v235_row1 = vector.extract %235[1] : f32 from vector<4xf32>
                %ep_v235_row2 = vector.extract %235[2] : f32 from vector<4xf32>
                %ep_v235_row3 = vector.extract %235[3] : f32 from vector<4xf32>
                %ep_v235_row0_nbr, %ep_v235_row0_valid = gpu.shuffle xor %ep_v235_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v235_row1_nbr, %ep_v235_row1_valid = gpu.shuffle xor %ep_v235_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v235_row2_nbr, %ep_v235_row2_valid = gpu.shuffle xor %ep_v235_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v235_row3_nbr, %ep_v235_row3_valid = gpu.shuffle xor %ep_v235_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v235_row0_lo = arith.select %ep_is_even, %ep_v235_row0, %ep_v235_row0_nbr : f32
                %ep_v235_row0_hi = arith.select %ep_is_even, %ep_v235_row0_nbr, %ep_v235_row0 : f32
                %ep_v235_row1_lo = arith.select %ep_is_even, %ep_v235_row1, %ep_v235_row1_nbr : f32
                %ep_v235_row1_hi = arith.select %ep_is_even, %ep_v235_row1_nbr, %ep_v235_row1 : f32
                %ep_v235_row2_lo = arith.select %ep_is_even, %ep_v235_row2, %ep_v235_row2_nbr : f32
                %ep_v235_row2_hi = arith.select %ep_is_even, %ep_v235_row2_nbr, %ep_v235_row2 : f32
                %ep_v235_row3_lo = arith.select %ep_is_even, %ep_v235_row3, %ep_v235_row3_nbr : f32
                %ep_v235_row3_hi = arith.select %ep_is_even, %ep_v235_row3_nbr, %ep_v235_row3 : f32
                %ep_v235_store_a_lo = arith.select %ep_is_even, %ep_v235_row0_lo, %ep_v235_row2_lo : f32
                %ep_v235_store_a_hi = arith.select %ep_is_even, %ep_v235_row0_hi, %ep_v235_row2_hi : f32
                %ep_v235_store_b_lo = arith.select %ep_is_even, %ep_v235_row1_lo, %ep_v235_row3_lo : f32
                %ep_v235_store_b_hi = arith.select %ep_is_even, %ep_v235_row1_hi, %ep_v235_row3_hi : f32
                %ep_v235_pair_a_0 = vector.broadcast %ep_v235_store_a_lo : f32 to vector<2xf32>
                %ep_v235_pair_a = vector.insert %ep_v235_store_a_hi, %ep_v235_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v235_store_a = arith.truncf %ep_v235_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v235_pair_b_0 = vector.broadcast %ep_v235_store_b_lo : f32 to vector<2xf32>
                %ep_v235_pair_b = vector.insert %ep_v235_store_b_hi, %ep_v235_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v235_store_b = arith.truncf %ep_v235_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v235_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col1_adj overflow<nsw> : index
                %ep_v235_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col1_adj overflow<nsw> : index
                vector.store %ep_v235_store_a, %ep_buffer[%ep_v235_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v235_store_b, %ep_buffer[%ep_v235_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v237_row0 = vector.extract %237[0] : f32 from vector<4xf32>
                %ep_v237_row1 = vector.extract %237[1] : f32 from vector<4xf32>
                %ep_v237_row2 = vector.extract %237[2] : f32 from vector<4xf32>
                %ep_v237_row3 = vector.extract %237[3] : f32 from vector<4xf32>
                %ep_v237_row0_nbr, %ep_v237_row0_valid = gpu.shuffle xor %ep_v237_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v237_row1_nbr, %ep_v237_row1_valid = gpu.shuffle xor %ep_v237_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v237_row2_nbr, %ep_v237_row2_valid = gpu.shuffle xor %ep_v237_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v237_row3_nbr, %ep_v237_row3_valid = gpu.shuffle xor %ep_v237_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v237_row0_lo = arith.select %ep_is_even, %ep_v237_row0, %ep_v237_row0_nbr : f32
                %ep_v237_row0_hi = arith.select %ep_is_even, %ep_v237_row0_nbr, %ep_v237_row0 : f32
                %ep_v237_row1_lo = arith.select %ep_is_even, %ep_v237_row1, %ep_v237_row1_nbr : f32
                %ep_v237_row1_hi = arith.select %ep_is_even, %ep_v237_row1_nbr, %ep_v237_row1 : f32
                %ep_v237_row2_lo = arith.select %ep_is_even, %ep_v237_row2, %ep_v237_row2_nbr : f32
                %ep_v237_row2_hi = arith.select %ep_is_even, %ep_v237_row2_nbr, %ep_v237_row2 : f32
                %ep_v237_row3_lo = arith.select %ep_is_even, %ep_v237_row3, %ep_v237_row3_nbr : f32
                %ep_v237_row3_hi = arith.select %ep_is_even, %ep_v237_row3_nbr, %ep_v237_row3 : f32
                %ep_v237_store_a_lo = arith.select %ep_is_even, %ep_v237_row0_lo, %ep_v237_row2_lo : f32
                %ep_v237_store_a_hi = arith.select %ep_is_even, %ep_v237_row0_hi, %ep_v237_row2_hi : f32
                %ep_v237_store_b_lo = arith.select %ep_is_even, %ep_v237_row1_lo, %ep_v237_row3_lo : f32
                %ep_v237_store_b_hi = arith.select %ep_is_even, %ep_v237_row1_hi, %ep_v237_row3_hi : f32
                %ep_v237_pair_a_0 = vector.broadcast %ep_v237_store_a_lo : f32 to vector<2xf32>
                %ep_v237_pair_a = vector.insert %ep_v237_store_a_hi, %ep_v237_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v237_store_a = arith.truncf %ep_v237_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v237_pair_b_0 = vector.broadcast %ep_v237_store_b_lo : f32 to vector<2xf32>
                %ep_v237_pair_b = vector.insert %ep_v237_store_b_hi, %ep_v237_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v237_store_b = arith.truncf %ep_v237_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v237_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col2_adj overflow<nsw> : index
                %ep_v237_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col2_adj overflow<nsw> : index
                vector.store %ep_v237_store_a, %ep_buffer[%ep_v237_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v237_store_b, %ep_buffer[%ep_v237_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v239_row0 = vector.extract %239[0] : f32 from vector<4xf32>
                %ep_v239_row1 = vector.extract %239[1] : f32 from vector<4xf32>
                %ep_v239_row2 = vector.extract %239[2] : f32 from vector<4xf32>
                %ep_v239_row3 = vector.extract %239[3] : f32 from vector<4xf32>
                %ep_v239_row0_nbr, %ep_v239_row0_valid = gpu.shuffle xor %ep_v239_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v239_row1_nbr, %ep_v239_row1_valid = gpu.shuffle xor %ep_v239_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v239_row2_nbr, %ep_v239_row2_valid = gpu.shuffle xor %ep_v239_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v239_row3_nbr, %ep_v239_row3_valid = gpu.shuffle xor %ep_v239_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v239_row0_lo = arith.select %ep_is_even, %ep_v239_row0, %ep_v239_row0_nbr : f32
                %ep_v239_row0_hi = arith.select %ep_is_even, %ep_v239_row0_nbr, %ep_v239_row0 : f32
                %ep_v239_row1_lo = arith.select %ep_is_even, %ep_v239_row1, %ep_v239_row1_nbr : f32
                %ep_v239_row1_hi = arith.select %ep_is_even, %ep_v239_row1_nbr, %ep_v239_row1 : f32
                %ep_v239_row2_lo = arith.select %ep_is_even, %ep_v239_row2, %ep_v239_row2_nbr : f32
                %ep_v239_row2_hi = arith.select %ep_is_even, %ep_v239_row2_nbr, %ep_v239_row2 : f32
                %ep_v239_row3_lo = arith.select %ep_is_even, %ep_v239_row3, %ep_v239_row3_nbr : f32
                %ep_v239_row3_hi = arith.select %ep_is_even, %ep_v239_row3_nbr, %ep_v239_row3 : f32
                %ep_v239_store_a_lo = arith.select %ep_is_even, %ep_v239_row0_lo, %ep_v239_row2_lo : f32
                %ep_v239_store_a_hi = arith.select %ep_is_even, %ep_v239_row0_hi, %ep_v239_row2_hi : f32
                %ep_v239_store_b_lo = arith.select %ep_is_even, %ep_v239_row1_lo, %ep_v239_row3_lo : f32
                %ep_v239_store_b_hi = arith.select %ep_is_even, %ep_v239_row1_hi, %ep_v239_row3_hi : f32
                %ep_v239_pair_a_0 = vector.broadcast %ep_v239_store_a_lo : f32 to vector<2xf32>
                %ep_v239_pair_a = vector.insert %ep_v239_store_a_hi, %ep_v239_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v239_store_a = arith.truncf %ep_v239_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v239_pair_b_0 = vector.broadcast %ep_v239_store_b_lo : f32 to vector<2xf32>
                %ep_v239_pair_b = vector.insert %ep_v239_store_b_hi, %ep_v239_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v239_store_b = arith.truncf %ep_v239_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v239_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col3_adj overflow<nsw> : index
                %ep_v239_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col3_adj overflow<nsw> : index
                vector.store %ep_v239_store_a, %ep_buffer[%ep_v239_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v239_store_b, %ep_buffer[%ep_v239_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v241_row0 = vector.extract %241[0] : f32 from vector<4xf32>
                %ep_v241_row1 = vector.extract %241[1] : f32 from vector<4xf32>
                %ep_v241_row2 = vector.extract %241[2] : f32 from vector<4xf32>
                %ep_v241_row3 = vector.extract %241[3] : f32 from vector<4xf32>
                %ep_v241_row0_nbr, %ep_v241_row0_valid = gpu.shuffle xor %ep_v241_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v241_row1_nbr, %ep_v241_row1_valid = gpu.shuffle xor %ep_v241_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v241_row2_nbr, %ep_v241_row2_valid = gpu.shuffle xor %ep_v241_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v241_row3_nbr, %ep_v241_row3_valid = gpu.shuffle xor %ep_v241_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v241_row0_lo = arith.select %ep_is_even, %ep_v241_row0, %ep_v241_row0_nbr : f32
                %ep_v241_row0_hi = arith.select %ep_is_even, %ep_v241_row0_nbr, %ep_v241_row0 : f32
                %ep_v241_row1_lo = arith.select %ep_is_even, %ep_v241_row1, %ep_v241_row1_nbr : f32
                %ep_v241_row1_hi = arith.select %ep_is_even, %ep_v241_row1_nbr, %ep_v241_row1 : f32
                %ep_v241_row2_lo = arith.select %ep_is_even, %ep_v241_row2, %ep_v241_row2_nbr : f32
                %ep_v241_row2_hi = arith.select %ep_is_even, %ep_v241_row2_nbr, %ep_v241_row2 : f32
                %ep_v241_row3_lo = arith.select %ep_is_even, %ep_v241_row3, %ep_v241_row3_nbr : f32
                %ep_v241_row3_hi = arith.select %ep_is_even, %ep_v241_row3_nbr, %ep_v241_row3 : f32
                %ep_v241_store_a_lo = arith.select %ep_is_even, %ep_v241_row0_lo, %ep_v241_row2_lo : f32
                %ep_v241_store_a_hi = arith.select %ep_is_even, %ep_v241_row0_hi, %ep_v241_row2_hi : f32
                %ep_v241_store_b_lo = arith.select %ep_is_even, %ep_v241_row1_lo, %ep_v241_row3_lo : f32
                %ep_v241_store_b_hi = arith.select %ep_is_even, %ep_v241_row1_hi, %ep_v241_row3_hi : f32
                %ep_v241_pair_a_0 = vector.broadcast %ep_v241_store_a_lo : f32 to vector<2xf32>
                %ep_v241_pair_a = vector.insert %ep_v241_store_a_hi, %ep_v241_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v241_store_a = arith.truncf %ep_v241_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v241_pair_b_0 = vector.broadcast %ep_v241_store_b_lo : f32 to vector<2xf32>
                %ep_v241_pair_b = vector.insert %ep_v241_store_b_hi, %ep_v241_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v241_store_b = arith.truncf %ep_v241_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v241_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col4_adj overflow<nsw> : index
                %ep_v241_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col4_adj overflow<nsw> : index
                vector.store %ep_v241_store_a, %ep_buffer[%ep_v241_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v241_store_b, %ep_buffer[%ep_v241_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v243_row0 = vector.extract %243[0] : f32 from vector<4xf32>
                %ep_v243_row1 = vector.extract %243[1] : f32 from vector<4xf32>
                %ep_v243_row2 = vector.extract %243[2] : f32 from vector<4xf32>
                %ep_v243_row3 = vector.extract %243[3] : f32 from vector<4xf32>
                %ep_v243_row0_nbr, %ep_v243_row0_valid = gpu.shuffle xor %ep_v243_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v243_row1_nbr, %ep_v243_row1_valid = gpu.shuffle xor %ep_v243_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v243_row2_nbr, %ep_v243_row2_valid = gpu.shuffle xor %ep_v243_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v243_row3_nbr, %ep_v243_row3_valid = gpu.shuffle xor %ep_v243_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v243_row0_lo = arith.select %ep_is_even, %ep_v243_row0, %ep_v243_row0_nbr : f32
                %ep_v243_row0_hi = arith.select %ep_is_even, %ep_v243_row0_nbr, %ep_v243_row0 : f32
                %ep_v243_row1_lo = arith.select %ep_is_even, %ep_v243_row1, %ep_v243_row1_nbr : f32
                %ep_v243_row1_hi = arith.select %ep_is_even, %ep_v243_row1_nbr, %ep_v243_row1 : f32
                %ep_v243_row2_lo = arith.select %ep_is_even, %ep_v243_row2, %ep_v243_row2_nbr : f32
                %ep_v243_row2_hi = arith.select %ep_is_even, %ep_v243_row2_nbr, %ep_v243_row2 : f32
                %ep_v243_row3_lo = arith.select %ep_is_even, %ep_v243_row3, %ep_v243_row3_nbr : f32
                %ep_v243_row3_hi = arith.select %ep_is_even, %ep_v243_row3_nbr, %ep_v243_row3 : f32
                %ep_v243_store_a_lo = arith.select %ep_is_even, %ep_v243_row0_lo, %ep_v243_row2_lo : f32
                %ep_v243_store_a_hi = arith.select %ep_is_even, %ep_v243_row0_hi, %ep_v243_row2_hi : f32
                %ep_v243_store_b_lo = arith.select %ep_is_even, %ep_v243_row1_lo, %ep_v243_row3_lo : f32
                %ep_v243_store_b_hi = arith.select %ep_is_even, %ep_v243_row1_hi, %ep_v243_row3_hi : f32
                %ep_v243_pair_a_0 = vector.broadcast %ep_v243_store_a_lo : f32 to vector<2xf32>
                %ep_v243_pair_a = vector.insert %ep_v243_store_a_hi, %ep_v243_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v243_store_a = arith.truncf %ep_v243_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v243_pair_b_0 = vector.broadcast %ep_v243_store_b_lo : f32 to vector<2xf32>
                %ep_v243_pair_b = vector.insert %ep_v243_store_b_hi, %ep_v243_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v243_store_b = arith.truncf %ep_v243_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v243_addr_a = arith.addi %ep_rb2_store_off_a, %ep_col5_adj overflow<nsw> : index
                %ep_v243_addr_b = arith.addi %ep_rb2_store_off_b, %ep_col5_adj overflow<nsw> : index
                vector.store %ep_v243_store_a, %ep_buffer[%ep_v243_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v243_store_b, %ep_buffer[%ep_v243_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v245_row0 = vector.extract %245[0] : f32 from vector<4xf32>
                %ep_v245_row1 = vector.extract %245[1] : f32 from vector<4xf32>
                %ep_v245_row2 = vector.extract %245[2] : f32 from vector<4xf32>
                %ep_v245_row3 = vector.extract %245[3] : f32 from vector<4xf32>
                %ep_v245_row0_nbr, %ep_v245_row0_valid = gpu.shuffle xor %ep_v245_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v245_row1_nbr, %ep_v245_row1_valid = gpu.shuffle xor %ep_v245_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v245_row2_nbr, %ep_v245_row2_valid = gpu.shuffle xor %ep_v245_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v245_row3_nbr, %ep_v245_row3_valid = gpu.shuffle xor %ep_v245_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v245_row0_lo = arith.select %ep_is_even, %ep_v245_row0, %ep_v245_row0_nbr : f32
                %ep_v245_row0_hi = arith.select %ep_is_even, %ep_v245_row0_nbr, %ep_v245_row0 : f32
                %ep_v245_row1_lo = arith.select %ep_is_even, %ep_v245_row1, %ep_v245_row1_nbr : f32
                %ep_v245_row1_hi = arith.select %ep_is_even, %ep_v245_row1_nbr, %ep_v245_row1 : f32
                %ep_v245_row2_lo = arith.select %ep_is_even, %ep_v245_row2, %ep_v245_row2_nbr : f32
                %ep_v245_row2_hi = arith.select %ep_is_even, %ep_v245_row2_nbr, %ep_v245_row2 : f32
                %ep_v245_row3_lo = arith.select %ep_is_even, %ep_v245_row3, %ep_v245_row3_nbr : f32
                %ep_v245_row3_hi = arith.select %ep_is_even, %ep_v245_row3_nbr, %ep_v245_row3 : f32
                %ep_v245_store_a_lo = arith.select %ep_is_even, %ep_v245_row0_lo, %ep_v245_row2_lo : f32
                %ep_v245_store_a_hi = arith.select %ep_is_even, %ep_v245_row0_hi, %ep_v245_row2_hi : f32
                %ep_v245_store_b_lo = arith.select %ep_is_even, %ep_v245_row1_lo, %ep_v245_row3_lo : f32
                %ep_v245_store_b_hi = arith.select %ep_is_even, %ep_v245_row1_hi, %ep_v245_row3_hi : f32
                %ep_v245_pair_a_0 = vector.broadcast %ep_v245_store_a_lo : f32 to vector<2xf32>
                %ep_v245_pair_a = vector.insert %ep_v245_store_a_hi, %ep_v245_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v245_store_a = arith.truncf %ep_v245_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v245_pair_b_0 = vector.broadcast %ep_v245_store_b_lo : f32 to vector<2xf32>
                %ep_v245_pair_b = vector.insert %ep_v245_store_b_hi, %ep_v245_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v245_store_b = arith.truncf %ep_v245_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v245_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col0_adj overflow<nsw> : index
                %ep_v245_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col0_adj overflow<nsw> : index
                vector.store %ep_v245_store_a, %ep_buffer[%ep_v245_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v245_store_b, %ep_buffer[%ep_v245_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v247_row0 = vector.extract %247[0] : f32 from vector<4xf32>
                %ep_v247_row1 = vector.extract %247[1] : f32 from vector<4xf32>
                %ep_v247_row2 = vector.extract %247[2] : f32 from vector<4xf32>
                %ep_v247_row3 = vector.extract %247[3] : f32 from vector<4xf32>
                %ep_v247_row0_nbr, %ep_v247_row0_valid = gpu.shuffle xor %ep_v247_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v247_row1_nbr, %ep_v247_row1_valid = gpu.shuffle xor %ep_v247_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v247_row2_nbr, %ep_v247_row2_valid = gpu.shuffle xor %ep_v247_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v247_row3_nbr, %ep_v247_row3_valid = gpu.shuffle xor %ep_v247_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v247_row0_lo = arith.select %ep_is_even, %ep_v247_row0, %ep_v247_row0_nbr : f32
                %ep_v247_row0_hi = arith.select %ep_is_even, %ep_v247_row0_nbr, %ep_v247_row0 : f32
                %ep_v247_row1_lo = arith.select %ep_is_even, %ep_v247_row1, %ep_v247_row1_nbr : f32
                %ep_v247_row1_hi = arith.select %ep_is_even, %ep_v247_row1_nbr, %ep_v247_row1 : f32
                %ep_v247_row2_lo = arith.select %ep_is_even, %ep_v247_row2, %ep_v247_row2_nbr : f32
                %ep_v247_row2_hi = arith.select %ep_is_even, %ep_v247_row2_nbr, %ep_v247_row2 : f32
                %ep_v247_row3_lo = arith.select %ep_is_even, %ep_v247_row3, %ep_v247_row3_nbr : f32
                %ep_v247_row3_hi = arith.select %ep_is_even, %ep_v247_row3_nbr, %ep_v247_row3 : f32
                %ep_v247_store_a_lo = arith.select %ep_is_even, %ep_v247_row0_lo, %ep_v247_row2_lo : f32
                %ep_v247_store_a_hi = arith.select %ep_is_even, %ep_v247_row0_hi, %ep_v247_row2_hi : f32
                %ep_v247_store_b_lo = arith.select %ep_is_even, %ep_v247_row1_lo, %ep_v247_row3_lo : f32
                %ep_v247_store_b_hi = arith.select %ep_is_even, %ep_v247_row1_hi, %ep_v247_row3_hi : f32
                %ep_v247_pair_a_0 = vector.broadcast %ep_v247_store_a_lo : f32 to vector<2xf32>
                %ep_v247_pair_a = vector.insert %ep_v247_store_a_hi, %ep_v247_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v247_store_a = arith.truncf %ep_v247_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v247_pair_b_0 = vector.broadcast %ep_v247_store_b_lo : f32 to vector<2xf32>
                %ep_v247_pair_b = vector.insert %ep_v247_store_b_hi, %ep_v247_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v247_store_b = arith.truncf %ep_v247_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v247_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col1_adj overflow<nsw> : index
                %ep_v247_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col1_adj overflow<nsw> : index
                vector.store %ep_v247_store_a, %ep_buffer[%ep_v247_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v247_store_b, %ep_buffer[%ep_v247_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v249_row0 = vector.extract %249[0] : f32 from vector<4xf32>
                %ep_v249_row1 = vector.extract %249[1] : f32 from vector<4xf32>
                %ep_v249_row2 = vector.extract %249[2] : f32 from vector<4xf32>
                %ep_v249_row3 = vector.extract %249[3] : f32 from vector<4xf32>
                %ep_v249_row0_nbr, %ep_v249_row0_valid = gpu.shuffle xor %ep_v249_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v249_row1_nbr, %ep_v249_row1_valid = gpu.shuffle xor %ep_v249_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v249_row2_nbr, %ep_v249_row2_valid = gpu.shuffle xor %ep_v249_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v249_row3_nbr, %ep_v249_row3_valid = gpu.shuffle xor %ep_v249_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v249_row0_lo = arith.select %ep_is_even, %ep_v249_row0, %ep_v249_row0_nbr : f32
                %ep_v249_row0_hi = arith.select %ep_is_even, %ep_v249_row0_nbr, %ep_v249_row0 : f32
                %ep_v249_row1_lo = arith.select %ep_is_even, %ep_v249_row1, %ep_v249_row1_nbr : f32
                %ep_v249_row1_hi = arith.select %ep_is_even, %ep_v249_row1_nbr, %ep_v249_row1 : f32
                %ep_v249_row2_lo = arith.select %ep_is_even, %ep_v249_row2, %ep_v249_row2_nbr : f32
                %ep_v249_row2_hi = arith.select %ep_is_even, %ep_v249_row2_nbr, %ep_v249_row2 : f32
                %ep_v249_row3_lo = arith.select %ep_is_even, %ep_v249_row3, %ep_v249_row3_nbr : f32
                %ep_v249_row3_hi = arith.select %ep_is_even, %ep_v249_row3_nbr, %ep_v249_row3 : f32
                %ep_v249_store_a_lo = arith.select %ep_is_even, %ep_v249_row0_lo, %ep_v249_row2_lo : f32
                %ep_v249_store_a_hi = arith.select %ep_is_even, %ep_v249_row0_hi, %ep_v249_row2_hi : f32
                %ep_v249_store_b_lo = arith.select %ep_is_even, %ep_v249_row1_lo, %ep_v249_row3_lo : f32
                %ep_v249_store_b_hi = arith.select %ep_is_even, %ep_v249_row1_hi, %ep_v249_row3_hi : f32
                %ep_v249_pair_a_0 = vector.broadcast %ep_v249_store_a_lo : f32 to vector<2xf32>
                %ep_v249_pair_a = vector.insert %ep_v249_store_a_hi, %ep_v249_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v249_store_a = arith.truncf %ep_v249_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v249_pair_b_0 = vector.broadcast %ep_v249_store_b_lo : f32 to vector<2xf32>
                %ep_v249_pair_b = vector.insert %ep_v249_store_b_hi, %ep_v249_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v249_store_b = arith.truncf %ep_v249_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v249_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col2_adj overflow<nsw> : index
                %ep_v249_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col2_adj overflow<nsw> : index
                vector.store %ep_v249_store_a, %ep_buffer[%ep_v249_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v249_store_b, %ep_buffer[%ep_v249_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v251_row0 = vector.extract %251[0] : f32 from vector<4xf32>
                %ep_v251_row1 = vector.extract %251[1] : f32 from vector<4xf32>
                %ep_v251_row2 = vector.extract %251[2] : f32 from vector<4xf32>
                %ep_v251_row3 = vector.extract %251[3] : f32 from vector<4xf32>
                %ep_v251_row0_nbr, %ep_v251_row0_valid = gpu.shuffle xor %ep_v251_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v251_row1_nbr, %ep_v251_row1_valid = gpu.shuffle xor %ep_v251_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v251_row2_nbr, %ep_v251_row2_valid = gpu.shuffle xor %ep_v251_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v251_row3_nbr, %ep_v251_row3_valid = gpu.shuffle xor %ep_v251_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v251_row0_lo = arith.select %ep_is_even, %ep_v251_row0, %ep_v251_row0_nbr : f32
                %ep_v251_row0_hi = arith.select %ep_is_even, %ep_v251_row0_nbr, %ep_v251_row0 : f32
                %ep_v251_row1_lo = arith.select %ep_is_even, %ep_v251_row1, %ep_v251_row1_nbr : f32
                %ep_v251_row1_hi = arith.select %ep_is_even, %ep_v251_row1_nbr, %ep_v251_row1 : f32
                %ep_v251_row2_lo = arith.select %ep_is_even, %ep_v251_row2, %ep_v251_row2_nbr : f32
                %ep_v251_row2_hi = arith.select %ep_is_even, %ep_v251_row2_nbr, %ep_v251_row2 : f32
                %ep_v251_row3_lo = arith.select %ep_is_even, %ep_v251_row3, %ep_v251_row3_nbr : f32
                %ep_v251_row3_hi = arith.select %ep_is_even, %ep_v251_row3_nbr, %ep_v251_row3 : f32
                %ep_v251_store_a_lo = arith.select %ep_is_even, %ep_v251_row0_lo, %ep_v251_row2_lo : f32
                %ep_v251_store_a_hi = arith.select %ep_is_even, %ep_v251_row0_hi, %ep_v251_row2_hi : f32
                %ep_v251_store_b_lo = arith.select %ep_is_even, %ep_v251_row1_lo, %ep_v251_row3_lo : f32
                %ep_v251_store_b_hi = arith.select %ep_is_even, %ep_v251_row1_hi, %ep_v251_row3_hi : f32
                %ep_v251_pair_a_0 = vector.broadcast %ep_v251_store_a_lo : f32 to vector<2xf32>
                %ep_v251_pair_a = vector.insert %ep_v251_store_a_hi, %ep_v251_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v251_store_a = arith.truncf %ep_v251_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v251_pair_b_0 = vector.broadcast %ep_v251_store_b_lo : f32 to vector<2xf32>
                %ep_v251_pair_b = vector.insert %ep_v251_store_b_hi, %ep_v251_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v251_store_b = arith.truncf %ep_v251_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v251_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col3_adj overflow<nsw> : index
                %ep_v251_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col3_adj overflow<nsw> : index
                vector.store %ep_v251_store_a, %ep_buffer[%ep_v251_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v251_store_b, %ep_buffer[%ep_v251_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v253_row0 = vector.extract %253[0] : f32 from vector<4xf32>
                %ep_v253_row1 = vector.extract %253[1] : f32 from vector<4xf32>
                %ep_v253_row2 = vector.extract %253[2] : f32 from vector<4xf32>
                %ep_v253_row3 = vector.extract %253[3] : f32 from vector<4xf32>
                %ep_v253_row0_nbr, %ep_v253_row0_valid = gpu.shuffle xor %ep_v253_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v253_row1_nbr, %ep_v253_row1_valid = gpu.shuffle xor %ep_v253_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v253_row2_nbr, %ep_v253_row2_valid = gpu.shuffle xor %ep_v253_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v253_row3_nbr, %ep_v253_row3_valid = gpu.shuffle xor %ep_v253_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v253_row0_lo = arith.select %ep_is_even, %ep_v253_row0, %ep_v253_row0_nbr : f32
                %ep_v253_row0_hi = arith.select %ep_is_even, %ep_v253_row0_nbr, %ep_v253_row0 : f32
                %ep_v253_row1_lo = arith.select %ep_is_even, %ep_v253_row1, %ep_v253_row1_nbr : f32
                %ep_v253_row1_hi = arith.select %ep_is_even, %ep_v253_row1_nbr, %ep_v253_row1 : f32
                %ep_v253_row2_lo = arith.select %ep_is_even, %ep_v253_row2, %ep_v253_row2_nbr : f32
                %ep_v253_row2_hi = arith.select %ep_is_even, %ep_v253_row2_nbr, %ep_v253_row2 : f32
                %ep_v253_row3_lo = arith.select %ep_is_even, %ep_v253_row3, %ep_v253_row3_nbr : f32
                %ep_v253_row3_hi = arith.select %ep_is_even, %ep_v253_row3_nbr, %ep_v253_row3 : f32
                %ep_v253_store_a_lo = arith.select %ep_is_even, %ep_v253_row0_lo, %ep_v253_row2_lo : f32
                %ep_v253_store_a_hi = arith.select %ep_is_even, %ep_v253_row0_hi, %ep_v253_row2_hi : f32
                %ep_v253_store_b_lo = arith.select %ep_is_even, %ep_v253_row1_lo, %ep_v253_row3_lo : f32
                %ep_v253_store_b_hi = arith.select %ep_is_even, %ep_v253_row1_hi, %ep_v253_row3_hi : f32
                %ep_v253_pair_a_0 = vector.broadcast %ep_v253_store_a_lo : f32 to vector<2xf32>
                %ep_v253_pair_a = vector.insert %ep_v253_store_a_hi, %ep_v253_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v253_store_a = arith.truncf %ep_v253_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v253_pair_b_0 = vector.broadcast %ep_v253_store_b_lo : f32 to vector<2xf32>
                %ep_v253_pair_b = vector.insert %ep_v253_store_b_hi, %ep_v253_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v253_store_b = arith.truncf %ep_v253_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v253_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col4_adj overflow<nsw> : index
                %ep_v253_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col4_adj overflow<nsw> : index
                vector.store %ep_v253_store_a, %ep_buffer[%ep_v253_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v253_store_b, %ep_buffer[%ep_v253_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                %ep_v255_row0 = vector.extract %255[0] : f32 from vector<4xf32>
                %ep_v255_row1 = vector.extract %255[1] : f32 from vector<4xf32>
                %ep_v255_row2 = vector.extract %255[2] : f32 from vector<4xf32>
                %ep_v255_row3 = vector.extract %255[3] : f32 from vector<4xf32>
                %ep_v255_row0_nbr, %ep_v255_row0_valid = gpu.shuffle xor %ep_v255_row0, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v255_row1_nbr, %ep_v255_row1_valid = gpu.shuffle xor %ep_v255_row1, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v255_row2_nbr, %ep_v255_row2_valid = gpu.shuffle xor %ep_v255_row2, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v255_row3_nbr, %ep_v255_row3_valid = gpu.shuffle xor %ep_v255_row3, %ep_shuffle_offset, %ep_shuffle_width : f32
                %ep_v255_row0_lo = arith.select %ep_is_even, %ep_v255_row0, %ep_v255_row0_nbr : f32
                %ep_v255_row0_hi = arith.select %ep_is_even, %ep_v255_row0_nbr, %ep_v255_row0 : f32
                %ep_v255_row1_lo = arith.select %ep_is_even, %ep_v255_row1, %ep_v255_row1_nbr : f32
                %ep_v255_row1_hi = arith.select %ep_is_even, %ep_v255_row1_nbr, %ep_v255_row1 : f32
                %ep_v255_row2_lo = arith.select %ep_is_even, %ep_v255_row2, %ep_v255_row2_nbr : f32
                %ep_v255_row2_hi = arith.select %ep_is_even, %ep_v255_row2_nbr, %ep_v255_row2 : f32
                %ep_v255_row3_lo = arith.select %ep_is_even, %ep_v255_row3, %ep_v255_row3_nbr : f32
                %ep_v255_row3_hi = arith.select %ep_is_even, %ep_v255_row3_nbr, %ep_v255_row3 : f32
                %ep_v255_store_a_lo = arith.select %ep_is_even, %ep_v255_row0_lo, %ep_v255_row2_lo : f32
                %ep_v255_store_a_hi = arith.select %ep_is_even, %ep_v255_row0_hi, %ep_v255_row2_hi : f32
                %ep_v255_store_b_lo = arith.select %ep_is_even, %ep_v255_row1_lo, %ep_v255_row3_lo : f32
                %ep_v255_store_b_hi = arith.select %ep_is_even, %ep_v255_row1_hi, %ep_v255_row3_hi : f32
                %ep_v255_pair_a_0 = vector.broadcast %ep_v255_store_a_lo : f32 to vector<2xf32>
                %ep_v255_pair_a = vector.insert %ep_v255_store_a_hi, %ep_v255_pair_a_0 [1] : f32 into vector<2xf32>
                %ep_v255_store_a = arith.truncf %ep_v255_pair_a : vector<2xf32> to vector<2xbf16>
                %ep_v255_pair_b_0 = vector.broadcast %ep_v255_store_b_lo : f32 to vector<2xf32>
                %ep_v255_pair_b = vector.insert %ep_v255_store_b_hi, %ep_v255_pair_b_0 [1] : f32 into vector<2xf32>
                %ep_v255_store_b = arith.truncf %ep_v255_pair_b : vector<2xf32> to vector<2xbf16>
                %ep_v255_addr_a = arith.addi %ep_rb3_store_off_a, %ep_col5_adj overflow<nsw> : index
                %ep_v255_addr_b = arith.addi %ep_rb3_store_off_b, %ep_col5_adj overflow<nsw> : index
                vector.store %ep_v255_store_a, %ep_buffer[%ep_v255_addr_a] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>
                vector.store %ep_v255_store_b, %ep_buffer[%ep_v255_addr_b] {alignment = 4 : i64} : memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<2xbf16>

                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: !hal.fence, %arg11: !hal.fence) -> !hal.buffer_view {
            %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
            %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
            %2 = hal.tensor.import wait(%arg10) => %arg0 : !hal.buffer_view -> tensor<?x?xi8>{%0, %1}
            %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
            %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
            %5 = hal.tensor.import wait(%arg10) => %arg1 : !hal.buffer_view -> tensor<?x?xi8>{%3, %4}
            %6 = hal.buffer_view.dim<%arg2 : !hal.buffer_view>[0] : index
            %7 = hal.buffer_view.dim<%arg2 : !hal.buffer_view>[1] : index
            %8 = hal.tensor.import wait(%arg10) => %arg2 : !hal.buffer_view -> tensor<?x?xi8>{%6, %7}
            %9 = hal.buffer_view.dim<%arg3 : !hal.buffer_view>[0] : index
            %10 = hal.buffer_view.dim<%arg3 : !hal.buffer_view>[1] : index
            %11 = hal.tensor.import wait(%arg10) => %arg3 : !hal.buffer_view -> tensor<?x?xi8>{%9, %10}
            %12 = hal.buffer_view.dim<%arg4 : !hal.buffer_view>[0] : index
            %13 = hal.buffer_view.dim<%arg4 : !hal.buffer_view>[1] : index
            %14 = hal.tensor.import wait(%arg10) => %arg4 : !hal.buffer_view -> tensor<?x?xbf16>{%12, %13}
            %c0 = arith.constant 0 : index
            %dim = tensor.dim %2, %c0 : tensor<?x?xi8>
            %c0_0 = arith.constant 0 : index
            %dim_1 = tensor.dim %8, %c0_0 : tensor<?x?xi8>
            %c1 = arith.constant 1 : index
            %dim_2 = tensor.dim %2, %c1 : tensor<?x?xi8>
            %c2 = arith.constant 2 : index
            %15 = arith.muli %dim_2, %c2 : index
            %c32 = arith.constant 32 : index
            %16 = arith.divsi %15, %c32 : index
            %c2_3 = arith.constant 2 : index
            %17 = arith.divsi %15, %c2_3 : index
            %18 = flow.dispatch @gemm::@gemm[%dim, %dim_1, %15, %arg5, %arg6, %arg7, %arg8, %arg9](%2, %5, %8, %11, %14, %arg5, %arg6, %arg7, %arg8, %arg9, %dim, %dim_1, %15) : (tensor<?x?xi8>{%dim, %17}, tensor<?x?xi8>{%dim, %16}, tensor<?x?xi8>{%dim_1, %17}, tensor<?x?xi8>{%dim_1, %16}, tensor<?x?xbf16>{%dim, %dim_1}, index, index, index, index, index, index, index, index) -> %14{%dim, %dim_1}
            %19 = hal.tensor.barrier join(%18 : tensor<?x?xbf16>) => %arg11 : !hal.fence
            %c0_4 = arith.constant 0 : index
            %dim_5 = tensor.dim %18, %c0_4 : tensor<?x?xbf16>
            %c1_6 = arith.constant 1 : index
            %dim_7 = tensor.dim %18, %c1_6 : tensor<?x?xbf16>
            %20 = hal.tensor.export %19 : tensor<?x?xbf16>{%dim_5, %dim_7} -> !hal.buffer_view
            return %20 : !hal.buffer_view
        }
        }

    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape,
        block,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = False
    options.linearize_shared_access = True
    options.wave_runtime = True
    options.dump_intermediates = "intermediates2"
    options.override_mlir = mlir_256x192
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds(
        use_stagger=True, shape=shape, block=block
    )
    UNROLL_FACTOR = tkl.sym.UNROLL_FACTOR
    options.subs[UNROLL_FACTOR] = 2
    options.postprocess = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
            %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.loop.unroll %0 { factor = %%UNROLL_FACTOR%% } : !transform.any_op
            transform.yield
        }
    }
    """

    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True, output_dtype=torch.bfloat16)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scales and B shuffling and B->LDS ({mode}) test passed!"
    )


def test_dbuf_8wave_mixed_pingpong_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 8 waves, with stagger.

    A variant of the ping-pong schedule that hides the latency of the extra
    WorkgroupBarrier required for large shapes. With staggering, the two
    clusters of waves write to LDS at different times.
    When the bus becomes congested, memory operations loaded by the later cluster may not arrive
    in LDS before the other cluster attempts to read from it. In this case,
    we add a second workgroup barrier to fix the timing and prevent incorrect output results.

    This schedule overlaps that barrier with useful work by splitting LDS loads:
      - "Safe" loads: rows this wave wrote itself — readable immediately after
        memory_counter_wait, before the global WorkgroupBarrier.
      - "Dependent" loads: rows written by other waves — deferred until after
        the global WorkgroupBarrier.

    This lets the MFMAs on the safe operands start firing as soon as the
    barrier releases, effectively hiding the second barrier's latency behind
    the early loads and compute.
    """
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(4, 2))
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_mixed_pingpong_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 8-wave mixed ping pong test passed!")


def test_dbuf_8wave_mixed_pingpong_shuffle_mxfp_gemm(
    is_debug=False, shape=(16384, 16384, 16384), block=(256, 256, 256)
):
    """Like :func:`test_dbuf_8wave_mixed_pingpong_mxfp_gemm` but with A_scale & B_scale
    preshuffled and prefetched to VGPRs.

    Note: preshuffling B and loading it directly to VGPRs combined with prefetching
    consumes too many VGPRs and causes spilling.
    """

    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape, block, wave_shape=(4, 2)
    )

    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_mixed_pingpong_shuffle_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, only_scale=True)
    print("MXFP GEMM double-buffer 8-wave mixed ping pong with shuffling test passed!")


def test_dbuf_4wave_mxfp_asymmetric_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Asymmetric-prefetch MXFP4 GEMM: A through LDS (2x prefetch), B direct from global."""
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric.mlir"
    options.print_mlir = True
    options.dump_binaries = "build/binaries"
    options.dump_intermediates = "build/intermediates"
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.use_water_backend = True
    schedule = get_mxfp4_asymmetric_schedule()

    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM asymmetric-prefetch 4-wave test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 256, 256),
    eliminate_epilogue=True,
):
    """Asymmetric MXFP4 GEMM with preshuffled B data and B scales."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )

    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print("MXFP GEMM preshuffle-B 4-wave test passed!")


def test_dbuf_4wave_mxfp_asymmetric_gemm_cpp(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """Asymmetric MXFP4 GEMM using C++ WaveASM backend (no preshuffle)."""
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    options.backend = "asm"
    options.wave_runtime = True
    options.dump_intermediates = "build/intermediates"
    schedule = get_mxfp4_asymmetric_schedule()
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM asymmetric 4-wave (WaveASM backend) test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm_cpp(
    is_debug=False,
    shape=(512, 1024, 8192),  # 4*T0, 4*T1, 8192
    block=(128, 256, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM using C++ WaveASM backend."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    options.backend = "asm"
    options.use_buffer_ops = True
    options.wave_runtime = True
    options.use_wave_asm_backend = True
    options.dump_intermediates = "build/intermediates"
    options.eliminate_epilogue = eliminate_epilogue
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print(
        f"MXFP GEMM preshuffle-B 4-wave (WaveASM) epilogue elimination={eliminate_epilogue} PASSED"
    )


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 256, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    # Make M, N, K dynamic so the compiler does not specialize on problem size.
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "llvm"
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print("MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (LLVM backend) test passed!")


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_wide_stores(
    is_debug=False,
    shape=(1024, 3072, 8192),
    block=(256, 192, 256),
    eliminate_epilogue=False,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K and wide epilogue stores.

    Uses the wide_store variant to swap MFMA operands (B as LHS, A as RHS),
    aligning the accumulator's contiguous values with the output's stride-1
    dimension. The coalesce_wide_stores pass emits v_permlane16_swap_b32
    + buffer_store_dwordx4 (8 bf16 per store) instead of buffer_store_short.
    """
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b_wide_store(
        shape,
        block,
        wave_shape=(2, 2),
        reorder_workgroups=True,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "llvm"
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True, output_dtype=torch.bfloat16)
    print("MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (wide stores) test passed!")


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 256, 256),
    eliminate_epilogue=False,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(1, 4), reorder_workgroups=False
    )
    # Make M, N, K dynamic so the compiler does not specialize on problem size.
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print(
        "MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (WaveASM backend) test passed!"
    )


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
        args.test,
        globals(),
        args.debug,
        args.repeat,
        args.shape,
        args.block,
        args.eliminate_epilogue,
    )
    exit(0 if success else 1)
