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
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_x {{[01]}}
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_y {{[01]}}
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_z {{[01]}}
    # CHECK:          .amdhsa_system_vgpr_workitem_id {{[0-9]+}}
    # CHECK:          .amdhsa_float_denorm_mode_32 3
    # CHECK:          .amdhsa_float_denorm_mode_16_64 3
    # CHECK:          .end_amdhsa_kernel
    # CHECK:          .text
    # CHECK:          # SRD upper word (gfx9xx): data_format=4 => 0x20000
    # CHECK:          .set Srd127_96, 131072
    # CHECK:          read_write:
    # CHECK:              s_load_dwordx2 s[{{[0-9]+}}:{{[0-9]+}}], s[0:1], {{[0-9]+}}
    # CHECK:              s_load_dwordx2 s[{{[0-9]+}}:{{[0-9]+}}], s[0:1], {{[0-9]+}}
    # CHECK:              s_waitcnt lgkmcnt(0)
    # CHECK:              s_mov_b32 s{{[0-9]+}}, {{.*}}
    # CHECK:              s_mov_b32 s{{[0-9]+}}, {{.*}}
    # CHECK:              s_mov_b32 s{{[0-9]+}}, {{.*}}
    # CHECK:              s_mov_b32 s{{[0-9]+}}, {{.*}}
    # CHECK:              v_mbcnt_lo_u32_b32 v{{[0-9]+}}, -1
    # CHECK:              v_mbcnt_hi_u32_b32 v{{[0-9]+}}, -1, v{{[0-9]+}}
    # CHECK:              v_lshlrev_b32 v{{[0-9]+}}, {{[0-9]+}}, v{{[0-9]+}}
    # CHECK:              buffer_load_dwordx4 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen
    # CHECK:              buffer_load_dwordx4 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen offset:16
    # CHECK:              s_waitcnt vmcnt(0)
    # CHECK:              buffer_store_dwordx4 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen
    # CHECK:              buffer_store_dwordx4 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen offset:16
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
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_x {{[01]}}
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_y {{[01]}}
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_z {{[01]}}
    # CHECK:          .amdhsa_system_vgpr_workitem_id {{[0-9]+}}
    # CHECK:          .amdhsa_float_denorm_mode_32 3
    # CHECK:          .amdhsa_float_denorm_mode_16_64 3
    # CHECK:          .end_amdhsa_kernel
    # CHECK:          .text
    # CHECK:          # SRD upper word (gfx9xx): data_format=4 => 0x20000
    # CHECK:          .set Srd127_96, 131072
    # CHECK:          mma:
    # CHECK:              s_load_dwordx2 s[{{[0-9]+}}:{{[0-9]+}}], s[0:1], {{[0-9x]+}}
    # CHECK:              s_load_dwordx2 s[{{[0-9]+}}:{{[0-9]+}}], s[0:1], {{[0-9x]+}}
    # CHECK:              s_load_dwordx2 s[{{[0-9]+}}:{{[0-9]+}}], s[0:1], {{[0-9x]+}}
    # CHECK:              s_waitcnt lgkmcnt(0)
    # CHECK:              s_mov_b32 s{{[0-9]+}}, {{.*}}
    # CHECK:              s_mov_b32 s{{[0-9]+}}, {{.*}}
    # CHECK:              v_mbcnt_lo_u32_b32 v{{[0-9]+}}, -1
    # CHECK:              v_mbcnt_hi_u32_b32 v{{[0-9]+}}, -1, v{{[0-9]+}}
    # CHECK:              buffer_load_dwordx2 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen
    # CHECK:              s_waitcnt vmcnt(0)
    # CHECK:              ds_write_b64 v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}]
    # CHECK:              buffer_load_dwordx2 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen
    # CHECK:              s_waitcnt vmcnt(0)
    # CHECK:              ds_write_b64 v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}]
    # CHECK:              s_barrier
    # CHECK:              ds_read_b64 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}
    # CHECK:              ds_read_b64 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}
    # CHECK:              v_mfma_f32_16x16x16_f16 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 0
    # CHECK:              buffer_store_dword v{{.*}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen
    # CHECK:              buffer_store_dword v{{.*}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen
    # CHECK:              buffer_store_dword v{{.*}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen
    # CHECK:              buffer_store_dword v{{.*}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen
    # CHECK:              s_endpgm


@run_test
def test_mma_multi_workgroup_multi_wave():
    """
    Test MMA with multi-workgroup, multi-wave constraint structure.

    Uses 1024x1024x16 shape to test multi-workgroup code generation.
    With BLOCK_M=16, BLOCK_N=16, this creates a 64x64 grid of workgroups.

    Note: This test verifies that the constraint structure and assembly
    generation work correctly for the multi-workgroup case. The centralized
    ticketing system ensures uniform VMEM/LGKM ticket issuance across all
    workgroups and waves.
    """
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
    def mma_multi(
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
            M: 1024,
            N: 1024,
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
    mma_multi = wave_compile(compile_options, mma_multi)
    print(mma_multi.asm)

    # CHECK-LABEL:    test_mma_multi_workgroup_multi_wave
    # CHECK:          .protected mma_multi
    # CHECK:          .amdhsa_kernel mma_multi
    # CHECK:          .amdhsa_user_sgpr_kernarg_segment_ptr 1
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_x 1
    # CHECK:          .amdhsa_system_sgpr_workgroup_id_y 1
    # CHECK:          mma_multi:
    # CHECK:          v_mfma_f32_16x16x16_f16
    # CHECK:          buffer_store_dword
    # CHECK:          s_endpgm


@run_test
def test_gemm_multi_wave_k_loop():
    """
    Test multi-wave GEMM with K-loop (BLOCK_K=64).

    Uses 4 waves per workgroup (BLOCK_M=32, BLOCK_N=32, WAVE_M=16, WAVE_N=16)
    with BLOCK_K=64 to test loop generation with chained MFMA accumulators.

    Verifies:
    - Loop induction variable initialization
    - Multiple MFMA instructions with accumulator chaining
    - Loop increment and branch back
    """
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M // 2),  # 2 waves in M dimension
        tkw.WaveConstraint(N, BLOCK_N // 2),  # 2 waves in N dimension
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def gemm_multi_wave(
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

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    compile_options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            K: 128,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 64,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    compile_options.compile_to_asm = True
    gemm_multi_wave = wave_compile(compile_options, gemm_multi_wave)
    print(gemm_multi_wave.asm)

    # CHECK-LABEL:    test_gemm_multi_wave_k_loop
    # CHECK:          .protected gemm_multi_wave
    # CHECK:          .amdhsa_kernel gemm_multi_wave
    # CHECK:          .amdhsa_system_vgpr_workitem_id {{[0-9]+}}
    # CHECK:          gemm_multi_wave:

    # Verify loop structure - header with comparison and conditional branch
    # CHECK:          loop_0_header:
    # CHECK:          s_cmp_lt_u32 s{{[0-9]+}}, s{{[0-9]+}}
    # CHECK:          s_cbranch_scc1 loop_0_body

    # Verify loop body has MFMA instructions
    # CHECK:          loop_0_body:
    # CHECK:          v_mfma_f32_16x16x16_f16

    # Verify loop latch - increment and branch back
    # CHECK:          loop_0_latch:
    # CHECK:          s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
    # CHECK:          s_branch loop_0_header

    # Verify loop exit and result stores
    # CHECK:          loop_0_exit:
    # CHECK:          buffer_store_dword
    # CHECK:          s_endpgm


@run_test
def test_gemm_gather_to_lds():
    """
    Test GEMM with gather_to_lds (global_to_shared) enabled.

    When use_global_to_shared=True, the compiler generates buffer_load_dword...lds
    instructions that load directly from global memory to LDS, bypassing VGPRs.

    Verifies:
    - buffer_load_dword ... lds instructions are emitted
    - M0 register setup for LDS addressing
    - Proper barrier synchronization (vmcnt + lgkmcnt + s_barrier)
    """
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def gemm_g2s(
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

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    compile_options = WaveCompileOptions(
        subs={
            M: 32,
            N: 32,
            K: 32,
            BLOCK_M: 16,
            BLOCK_N: 16,
            BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        use_global_to_shared=True,  # Enable gather_to_lds
    )
    compile_options.compile_to_asm = True
    gemm_g2s = wave_compile(compile_options, gemm_g2s)
    print(gemm_g2s.asm)

    # CHECK-LABEL:    test_gemm_gather_to_lds
    # CHECK:          .protected gemm_g2s
    # CHECK:          .amdhsa_kernel gemm_g2s
    # CHECK:          gemm_g2s:

    # Verify loop structure for K-loop
    # CHECK:          loop_0_header:
    # CHECK:          s_cmp_lt_u32 s{{[0-9]+}}, s{{[0-9]+}}
    # CHECK:          s_cbranch_scc1 loop_0_body

    # Verify loop body has MFMA instruction
    # CHECK:          loop_0_body:
    # CHECK:          v_mfma_f32_16x16x16_f16

    # Verify loop latch
    # CHECK:          loop_0_latch:
    # CHECK:          s_branch loop_0_header

    # Verify loop exit and stores
    # CHECK:          loop_0_exit:
    # CHECK:          buffer_store_dword
    # CHECK:          s_endpgm
