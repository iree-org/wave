# RUN: python %s | FileCheck %s


import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
)
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def get_wave_compile_options(
    canonicalize: bool = False,
    dynamic_symbols=[],
    additional_symbols={},
    location_capture_config=LocationCaptureConfig(
        level=LocationCaptureLevel.FILE_LINE_COL
    ),
    drop_debug_info_before_mlir=True,
):
    bindings = {
        M: 16,
        N: 16,
        K: 16,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
    }
    bindings.update(additional_symbols)

    # Remove dynamic symbols from the bindings.
    for sym in dynamic_symbols:
        if sym in bindings:
            del bindings[sym]

    return WaveCompileOptions(
        subs=bindings,
        canonicalize=canonicalize,
        dynamic_symbols=dynamic_symbols,
        compile_to_mlir=True,
        location_capture_config=location_capture_config,
        drop_debug_info_before_mlir=drop_debug_info_before_mlir,
    )


@run_test
def test_read_write():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 16, N: 16})
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def read_write(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    options = get_wave_compile_options(canonicalize=True)
    # Compile to AMDGCN assembly (for lit tests, no amdclang++)
    options.compile_to_asm = True

    read_write = wave_compile(options, read_write)
    print(read_write.asm)

    # CHECK-LABEL:    test_read_write
    # CHECK:          .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
    # CHECK:          .text
    # CHECK:          .protected read_write
    # CHECK:          .globl read_write
    # CHECK:          .p2align 8
    # CHECK:          .type read_write,@function
    # CHECK:          .section .rodata,#alloc
    # CHECK:          .p2align 6
    # CHECK:          .amdhsa_kernel read_write
    # CHECK:          .amdhsa_user_sgpr_kernarg_segment_ptr 1
    # CHECK:          .amdhsa_accum_offset 12
    # CHECK:          .amdhsa_next_free_vgpr 12
    # CHECK:          .amdhsa_next_free_sgpr 16
    # CHECK:          .amdhsa_group_segment_fixed_size 0
    # CHECK:          .amdhsa_private_segment_fixed_size 0
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_x 1
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_y 1
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_z 1
    # CHECK:          .amdhsa_system_vgpr_workitem_id 0
    # CHECK:          .amdhsa_float_denorm_mode_32 3
    # CHECK:          .amdhsa_float_denorm_mode_16_64 3
    # CHECK:          .end_amdhsa_kernel
    # CHECK:          .text
    # CHECK:          # SRD upper word (gfx9xx): data_format=4 => 0x20000
    # CHECK:          .set Srd127_96, 0x20000
    # CHECK:          read_write:
    # CHECK:              s_load_dwordx2 s[2:3], s[0:1], 0x0
    # CHECK:              s_load_dwordx2 s[4:5], s[0:1], 0x8
    # CHECK:              s_waitcnt lgkmcnt(0)
    # CHECK:              # SRD for Value(%reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<f16> to memref<16x16xf16, strided<[16, 1]>>) (arg0)
    # CHECK:              s_mov_b32 s8, s2
    # CHECK:              s_mov_b32 s9, s3
    # CHECK:              s_mov_b32 s10, 2048
    # CHECK:              s_mov_b32 s11, Srd127_96
    # CHECK:              # SRD for Value(%reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<f16> to memref<16x16xf16, strided<[16, 1]>>) (arg1)
    # CHECK:              s_mov_b32 s12, s4
    # CHECK:              s_mov_b32 s13, s5
    # CHECK:              s_mov_b32 s14, 2048
    # CHECK:              s_mov_b32 s15, Srd127_96
    # CHECK:              # lane id (0..63)
    # CHECK:              v_mbcnt_lo_u32_b32 v0, -1, 0
    # CHECK:              v_mbcnt_hi_u32_b32 v0, -1, v0
    # CHECK:              v_lshlrev_b32 v2, 5, v0
    # CHECK:              # load 32B from Value(%reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<f16> to memref<16x16xf16, strided<[16, 1]>>)
    # CHECK:              buffer_load_dwordx4  v[4:7], v2, s[8:11], 0 offen offset:0
    # CHECK:              s_waitcnt vmcnt(0)
    # CHECK:              buffer_load_dwordx4  v[8:11], v2, s[8:11], 0 offen offset:16
    # CHECK:              s_waitcnt vmcnt(0)
    # CHECK:              v_lshlrev_b32 v2, 5, v0
    # CHECK:              # store 32B to Value(%reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<f16> to memref<16x16xf16, strided<[16, 1]>>)
    # CHECK:              buffer_store_dwordx4 v[4:7], v2, s[12:15], 0 offen offset:0
    # CHECK:              buffer_store_dwordx4 v[8:11], v2, s[12:15], 0 offen offset:16
    # CHECK:              s_endpgm
