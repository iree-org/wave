.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected gemm_kernel
.globl gemm_kernel
.p2align 8
.type gemm_kernel,@function

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel gemm_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_count 2
  .amdhsa_accum_offset 32
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
  .amdhsa_group_segment_fixed_size 8192
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_vgpr_workitem_id 1
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text

# SRD upper word (gfx9xx): data_format=4 => 0x20000
.set Srd127_96, 0x20000

gemm_kernel:
    s_load_dwordx2 s[4:5], s[0:1], 0  // Load base addr for arg0
    s_load_dwordx2 s[8:9], s[0:1], 8  // Load base addr for arg1
    s_load_dwordx2 s[12:13], s[0:1], 16  // Load base addr for arg2
    s_waitcnt lgkmcnt(0)
    s_mov_b32 s6, 0x7ffffffc  // SRD size for arg0
    s_mov_b32 s7, 0x20000  // SRD stride for arg0
    s_mov_b32 s10, 0x7ffffffc  // SRD size for arg1
    s_mov_b32 s11, 0x20000  // SRD stride for arg1
    s_mov_b32 s14, 0x40000  // SRD size for arg2
    s_mov_b32 s15, 0x20000  // SRD stride for arg2
    s_mov_b32 s16, s4  // SRD word0
    s_and_b32 s17, s5, 0xffff
    s_or_b32 s17, s17, 0x40800000  // cache swizzle
    s_mov_b32 s18, 0x7ffffffd  // SRD word2
    s_mov_b32 s19, 0x27000  // SRD word3
    s_mov_b32 s4, s8  // SRD word0
    s_and_b32 s5, s9, 0xffff
    s_or_b32 s5, s5, 0x40800000  // cache swizzle
    s_mov_b32 s6, 0x7ffffffd  // SRD word2
    s_mov_b32 s7, 0x27000  // SRD word3
    // Initialize loop 0
    s_mov_b32 s24, 0  // loop 0 counter = 0
    s_mov_b32 s25, 1  // loop 0 step = 1
    s_mov_b32 s26, 2  // loop 0 upper = 2
    v_mov_b32 v4, 0
    v_mov_b32 v5, 0
    v_mov_b32 v6, 0
    v_mov_b32 v7, 0  // Initialize accumulator 0 to 0.0
loop_0_header:
    s_cmp_lt_u32 s24, s26  // compare loop 0 counter < upper
    s_cbranch_scc1 loop_0_body
    s_branch loop_0_exit
loop_0_body:
    s_waitcnt vmcnt(0)
    s_waitcnt lgkmcnt(0)
    s_barrier  // LDS barrier
    // gather_to_lds: 16B from Value(%reinterpret_cast_1 = memref.reinterpret_cast %0 to offset: [0], sizes: [1073741822], strides: [1] : memref<f16> to memref<1073741822xf16, strided<[1]>>) to LDS
    v_bfe_u32 v1, v0, 10, 10  // extract tid_y from flat_tid
    v_bfe_u32 v2, v0, 0, 10  // extract tid_x from flat_tid
    v_lshrrev_b32 v3, 3, v2  // floor div by 8 (shift)
    v_lshl_add_u32 v8, v1, 4, v3  // fused: (kv4 << 4) + kv7
    v_lshrrev_b32 v9, 5, v8  // floor add by 32 (shift)
    v_mov_b32 v8, 0xffffe000  // materialize -8192
    v_mul_lo_u32 v10, v9, v8  // floor(tid_y/2 + floor(tid_x/8)/32) * -8192
    v_lshlrev_b32 v8, 4, v2  // tid_x << 4
    v_add_u32 v9, v10, v8  // add
    v_mov_b32 v11, s24  // broadcast s24 to VGPR
    v_lshlrev_b32 v12, 7, v11  // s24 << 7
    v_add_u32 v11, v9, v12  // add
    v_lshlrev_b32 v9, 7, v3  // floor(tid_x/8) << 7
    v_add_u32 v3, v11, v9  // add
    v_lshlrev_b32 v11, 12, v1  // tid_y << 12
    v_add_u32 v13, v3, v11  // add
    v_mov_b32 v3, s2  // wgid_x from s2
    v_lshl_add_u32 v14, v3, 13, v13  // fused: (kv21 << 13) + kv20
    v_mov_b32 v13, 0x1000  // imm = 4096
    v_lshrrev_b32 v15, 6, v2  // floor div by 64 (shift)
    v_lshl_add_u32 v16, v1, 1, v15  // fused: (kv4 << 1) + kv26
    v_lshrrev_b32 v17, 2, v16  // floor add by 4 (shift)
    v_mov_b32 v16, 0xfffff000  // materialize -4096
    v_mul_lo_u32 v18, v17, v16  // floor(tid_y/2 + floor(tid_x/64)/4) * -4096
    v_add_u32 v16, v13, v18  // add
    v_lshlrev_b32 v17, 10, v15  // floor(tid_x/64) << 10
    v_add_u32 v18, v16, v17  // add
    v_lshlrev_b32 v16, 11, v1  // tid_y << 11
    v_add_u32 v19, v18, v16  // add
    s_nop 0  // hazard mitigation
    v_readfirstlane_b32 s8, v19
    s_mov_b32 m0, s8
    buffer_load_dwordx4 v14, s[16:19], 0 offen lds  // gather 16B
    // gather_to_lds: 16B from Value(%reinterpret_cast_2 = memref.reinterpret_cast %1 to offset: [0], sizes: [1073741822], strides: [1] : memref<f16> to memref<1073741822xf16, strided<[1]>>) to LDS
    v_add_u32 v14, v10, v8  // add
    v_add_u32 v8, v14, v12  // add
    v_add_u32 v10, v8, v9  // add
    v_add_u32 v8, v10, v11  // add
    v_mov_b32 v9, s3  // wgid_y from s3
    v_lshl_add_u32 v10, v9, 13, v8  // fused: (kv40 << 13) + kv39
    v_add_u32 v8, v17, v16  // add
    v_lshrrev_b32 v11, 1, v1  // tid_y >> 1 (div by 2)
    v_lshrrev_b32 v12, 2, v15  // floor(tid_x/64) >> 2 (div by 4)
    v_add_u32 v14, v11, v12  // add
    v_and_b32 v11, 0, v14  // mod 1 (and)
    v_lshl_add_u32 v12, v11, 12, v8  // fused: (kv47 << 12) + kv43
    s_nop 0  // hazard mitigation
    v_readfirstlane_b32 s8, v12
    s_mov_b32 m0, s8
    buffer_load_dwordx4 v10, s[4:7], 0 offen lds  // gather 16B
    s_waitcnt vmcnt(0)
    s_waitcnt lgkmcnt(0)
    s_barrier  // LDS barrier
    v_and_b32 v8, 63, v2  // mod 64 (and)
    v_lshrrev_b32 v10, 4, v8  // floor div by 16 (shift)
    v_lshlrev_b32 v8, 3, v10  // floor((Mod(tid_x, 64))/16) << 3
    v_and_b32 v11, 15, v2  // mod 16 (and)
    v_lshlrev_b32 v2, 7, v11  // Mod(tid_x, 16) << 7
    v_add_u32 v12, v8, v2  // add
    v_add_u32 v14, v12, v16  // add
    ds_read_b64 v[16:17], v14 offset:0  // LDS load 8B @ offset 0
    ds_read_b64 v[18:19], v14 offset:32  // LDS load 8B @ offset 32
    ds_read_b64 v[20:21], v14 offset:64  // LDS load 8B @ offset 64
    ds_read_b64 v[22:23], v14 offset:96  // LDS load 8B @ offset 96
    v_add_u32 v12, v13, v8  // add
    v_add_u32 v13, v12, v2  // add
    v_lshlrev_b32 v12, 11, v15  // floor(tid_x/64) << 11
    v_add_u32 v14, v13, v12  // add
    ds_read_b64 v[24:25], v14 offset:0  // LDS load 8B @ offset 0
    v_mov_b32 v13, 0x1020  // imm = 4128
    v_add_u32 v14, v13, v8  // add
    v_add_u32 v13, v14, v2  // add
    v_add_u32 v14, v13, v12  // add
    ds_read_b64 v[26:27], v14 offset:0  // LDS load 8B @ offset 0
    v_mov_b32 v13, 0x1040  // imm = 4160
    v_add_u32 v14, v13, v8  // add
    v_add_u32 v13, v14, v2  // add
    v_add_u32 v14, v13, v12  // add
    ds_read_b64 v[28:29], v14 offset:0  // LDS load 8B @ offset 0
    v_mov_b32 v13, 0x1060  // imm = 4192
    v_add_u32 v14, v13, v8  // add
    v_add_u32 v8, v14, v2  // add
    v_add_u32 v2, v8, v12  // add
    ds_read_b64 v[12:13], v2 offset:0  // LDS load 8B @ offset 0
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x16_f16 v[4:7], v[24:25], v[16:17], v[4:7]  // MFMA with accumulator (in-place)
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x16_f16 v[4:7], v[26:27], v[18:19], v[4:7]  // MFMA with accumulator (in-place)
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x16_f16 v[4:7], v[28:29], v[20:21], v[4:7]  // MFMA with accumulator (in-place)
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x16_f16 v[4:7], v[12:13], v[22:23], v[4:7]  // MFMA with accumulator (in-place)
loop_0_latch:
    s_add_u32 s24, s24, s25  // loop 0 counter += step
    s_branch loop_0_header
loop_0_exit:
    v_lshlrev_b32 v2, 6, v1  // tid_y << 6
    v_lshl_add_u32 v1, v11, 2, v2  // fused: (kv53 << 2) + kv91
    v_lshl_add_u32 v2, v9, 7, v1  // fused: (kv40 << 7) + kv92
    v_lshl_add_u32 v1, v10, 12, v2  // fused: (kv51 << 12) + kv94
    v_lshl_add_u32 v2, v15, 14, v1  // fused: (kv26 << 14) + kv96
    v_lshl_add_u32 v1, v3, 15, v2  // fused: (kv21 << 15) + kv98
    s_waitcnt vmcnt(0)
    buffer_store_dword v[4:4], v1, s[12:15], 0 offen  // store 4B @ offset 0
    s_waitcnt vmcnt(0)
    buffer_store_dword v[5:5], v1, s[12:15], 0 offen offset:1024  // store 4B @ offset 1024
    s_waitcnt vmcnt(0)
    buffer_store_dword v[6:6], v1, s[12:15], 0 offen offset:2048  // store 4B @ offset 2048
    s_waitcnt vmcnt(0)
    buffer_store_dword v[7:7], v1, s[12:15], 0 offen offset:3072  // store 4B @ offset 3072
    s_endpgm

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
amdhsa.kernels:
  - .name: gemm_kernel
    .symbol: 'gemm_kernel.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name: arg0_ptr
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .value_type: i8*
      - .name: arg1_ptr
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .value_type: i8*
      - .name: arg2_ptr
        .size: 8
        .offset: 16
        .value_kind: global_buffer
        .value_type: i8*
    .group_segment_fixed_size:   8192
    .kernarg_segment_align:      8
    .kernarg_segment_size:       24
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 2
      - 1
    .sgpr_count:                 32
    .sgpr_spill_count:           0
    .uniform_work_group_size:    1
    .vgpr_count:                 32
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata