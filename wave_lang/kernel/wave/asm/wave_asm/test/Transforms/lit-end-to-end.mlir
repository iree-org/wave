// RUN: waveasm-translate --waveasm-scoped-cse --waveasm-linear-scan --waveasm-hazard-mitigation --emit-assembly %s | FileCheck %s
//
// Test: End-to-end pipeline: CSE -> RegAlloc -> Hazard Mitigation -> Assembly

// Full assembly structure check
// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK: .text
// CHECK: .globl gemm_kernel
// CHECK: gemm_kernel:

waveasm.program @gemm_kernel target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<tid = 0, kernarg = 0> {
  // Thread ID
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Kernarg pointer
  %kptr = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>

  // Load buffer descriptors
  %offset0 = waveasm.constant 0 : !waveasm.imm<0>
  %srd_a = waveasm.s_load_dwordx4 %kptr, %offset0 : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<4>

  // Duplicate constant (should be eliminated by CSE)
  %offset0_dup = waveasm.constant 0 : !waveasm.imm<0>

  %offset16 = waveasm.constant 16 : !waveasm.imm<16>
  %srd_b = waveasm.s_load_dwordx4 %kptr, %offset16 : !waveasm.psreg<0, 2>, !waveasm.imm<16> -> !waveasm.sreg<4>

  // Calculate global offset
  // CHECK: v_lshlrev_b32
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %byte_offset = waveasm.v_lshlrev_b32 %c4, %tid : !waveasm.imm<4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // Load A and B tiles
  // CHECK: buffer_load_dwordx4
  %tile_a = waveasm.buffer_load_dwordx4 %srd_a, %byte_offset : !waveasm.sreg<4>, !waveasm.vreg -> !waveasm.vreg<4, 4>

  // CHECK: buffer_load_dwordx4
  %tile_b = waveasm.buffer_load_dwordx4 %srd_b, %byte_offset : !waveasm.sreg<4>, !waveasm.vreg -> !waveasm.vreg<4, 4>

  // Initialize accumulator
  %zero = waveasm.constant 0 : !waveasm.imm<0>

  // MFMA instruction (matrix multiply-accumulate)
  // CHECK: v_mfma
  %mfma = waveasm.v_mfma_f32_32x32x8_f16 %tile_a, %tile_b, %zero : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.imm<0> -> !waveasm.vreg<16>

  // Store result
  %offset32 = waveasm.constant 32 : !waveasm.imm<32>
  %srd_c = waveasm.s_load_dwordx4 %kptr, %offset32 : !waveasm.psreg<0, 2>, !waveasm.imm<32> -> !waveasm.sreg<4>

  // Note: This would be buffer_store, but we'll end with endpgm for simplicity
  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// CHECK: .amdhsa_kernel gemm_kernel
// CHECK: .amdhsa_next_free_vgpr
// CHECK: .amdhsa_next_free_sgpr
// CHECK: .end_amdhsa_kernel

// CHECK: .amdgpu_metadata
// CHECK: .name: gemm_kernel
// CHECK: .wavefront_size: 64
// CHECK: .end_amdgpu_metadata
