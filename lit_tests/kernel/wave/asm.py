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
    # CHECK:          .amdhsa_accum_offset {{[0-9]+}}
    # CHECK:          .amdhsa_next_free_vgpr {{[0-9]+}}
    # CHECK:          .amdhsa_next_free_sgpr {{[0-9]+}}
    # CHECK:          .amdhsa_group_segment_fixed_size 0
    # CHECK:          .amdhsa_private_segment_fixed_size 0
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_x 0
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_y 0
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_z 0
    # CHECK:          .amdhsa_system_vgpr_workitem_id 1
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
    # CHECK:              s_mov_b32 s10, {{[0-9]+}}
    # CHECK:              s_mov_b32 s11, Srd127_96
    # CHECK:              # SRD for Value(%reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<f16> to memref<16x16xf16, strided<[16, 1]>>) (arg1)
    # CHECK:              s_mov_b32 s12, s4
    # CHECK:              s_mov_b32 s13, s5
    # CHECK:              s_mov_b32 s14, {{[0-9]+}}
    # CHECK:              s_mov_b32 s15, Srd127_96
    # CHECK:              # lane id (0..63)
    # CHECK:              v_mbcnt_lo_u32_b32 v{{[0-9]+}}, -1
    # CHECK:              v_mbcnt_hi_u32_b32 v{{[0-9]+}}, -1, v{{[0-9]+}}
    # CHECK:              v_lshlrev_b32 v{{[0-9]+}}, 5, v{{[0-9]+}}
    # CHECK:              # load 32B from {{.*}}
    # CHECK:              buffer_load_dwordx4  v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[8:11], 0 offen offset:0
    # CHECK:              buffer_load_dwordx4  v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[8:11], 0 offen offset:16
    # CHECK:              s_waitcnt vmcnt(0)
    # CHECK:              # store 32B to {{.*}}
    # CHECK:              buffer_store_dwordx4 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[12:15], 0 offen offset:0
    # CHECK:              buffer_store_dwordx4 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[12:15], 0 offen offset:16
    # CHECK:              s_endpgm


@run_test
def test_mma():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

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
            M: 16,
            N: 16,
            K: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    # Compile to AMDGCN assembly (for lit tests, no amdclang++)
    compile_options.compile_to_asm = True
    mma = wave_compile(compile_options, mma)
    print(mma.asm)

    # CHECK-LABEL:    test_mma
    # CHECK:          .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
    # CHECK:          .text
    # CHECK:          .protected mma
    # CHECK:          .globl mma
    # CHECK:          .p2align 8
    # CHECK:          .type mma,@function
    # CHECK:          .section .rodata,#alloc
    # CHECK:          .p2align 6
    # CHECK:          .amdhsa_kernel mma
    # CHECK:          .amdhsa_user_sgpr_kernarg_segment_ptr 1
    # CHECK:          .amdhsa_accum_offset {{[0-9]+}}
    # CHECK:          .amdhsa_next_free_vgpr {{[0-9]+}}
    # CHECK:          .amdhsa_next_free_sgpr {{[0-9]+}}
    # CHECK:          .amdhsa_group_segment_fixed_size {{[0-9]+}}
    # CHECK:          .amdhsa_private_segment_fixed_size 0
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_x 0
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_y 0
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_z 0
    # CHECK:          .amdhsa_system_vgpr_workitem_id 1
    # CHECK:          .amdhsa_float_denorm_mode_32 3
    # CHECK:          .amdhsa_float_denorm_mode_16_64 3
    # CHECK:          .end_amdhsa_kernel
    # CHECK:          .text
    # CHECK:          # SRD upper word (gfx9xx): data_format=4 => 0x20000
    # CHECK:          .set Srd127_96, 0x20000
    # CHECK:          mma:
    # CHECK:              s_load_dwordx2 s[2:3], s[0:1], 0x0
    # CHECK:              s_load_dwordx2 s[4:5], s[0:1], 0x8
    # CHECK:              s_load_dwordx2 s[6:7], s[0:1], 0x10
    # CHECK:              s_waitcnt lgkmcnt(0)
    # CHECK:              # SRD for {{.*}} (arg0)
    # CHECK:              s_mov_b32 s8, s2
    # CHECK:              s_mov_b32 s9, s3
    # CHECK:              s_mov_b32 s10, {{[0-9]+}}
    # CHECK:              s_mov_b32 s11, Srd127_96
    # CHECK:              # SRD for {{.*}} (arg1)
    # CHECK:              s_mov_b32 s12, s4
    # CHECK:              s_mov_b32 s13, s5
    # CHECK:              s_mov_b32 s14, {{[0-9]+}}
    # CHECK:              s_mov_b32 s15, Srd127_96
    # CHECK:              # SRD for {{.*}} (arg2)
    # CHECK:              s_mov_b32 s16, s6
    # CHECK:              s_mov_b32 s17, s7
    # CHECK:              s_mov_b32 s18, {{[0-9]+}}
    # CHECK:              s_mov_b32 s19, Srd127_96
    # CHECK:              # lane id (0..63)
    # CHECK:              v_mbcnt_lo_u32_b32 v{{[0-9]+}}, -1
    # CHECK:              v_mbcnt_hi_u32_b32 v{{[0-9]+}}, -1, v{{[0-9]+}}
    # CHECK:              # load 8B from {{.*}}
    # CHECK:              buffer_load_dwordx2 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[8:11], 0 offen offset:0
    # CHECK:              s_waitcnt vmcnt(0)
    # CHECK:              ds_write_b64 v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}]
    # CHECK:              # load 8B from {{.*}}
    # CHECK:              buffer_load_dwordx2 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[12:15], 0 offen offset:0
    # CHECK:              s_waitcnt vmcnt(0)
    # CHECK:              ds_write_b64 v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}]
    # CHECK:              s_barrier
    # CHECK:              ds_read_b64 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}
    # CHECK:              ds_read_b64 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}
    # CHECK:              v_mfma_f32_16x16x16_f16 a[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0
    # CHECK:              s_nop 6
    # CHECK:              v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
    # CHECK:              v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
    # CHECK:              v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
    # CHECK:              v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
    # CHECK-DAG:          v_lshrrev_b32 v{{[0-9]+}}, 4, v{{[0-9]+}}
    # CHECK-DAG:          v_and_b32 v{{[0-9]+}}, 15, v{{[0-9]+}}
    # CHECK-DAG:          v_lshlrev_b32 v{{[0-9]+}}, {{[0-9]+}}, v{{[0-9]+}}
    # CHECK-DAG:          v_add_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
    # CHECK:              # store 4B to {{.*}}
    # CHECK:              buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[16:19], 0 offen offset:{{[0-9]+}}
    # CHECK:              # store 4B to {{.*}}
    # CHECK:              buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[16:19], 0 offen offset:{{[0-9]+}}
    # CHECK:              # store 4B to {{.*}}
    # CHECK:              buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[16:19], 0 offen offset:{{[0-9]+}}
    # CHECK:              # store 4B to {{.*}}
    # CHECK:              buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[16:19], 0 offen offset:{{[0-9]+}}
    # CHECK:              s_endpgm
