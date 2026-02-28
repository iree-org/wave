// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: Assembly emission with proper AMDGCN format

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK: .text
// CHECK: .globl assembly_test
// CHECK: .p2align 8
// CHECK: assembly_test:

waveasm.program @assembly_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Test: precolored registers are resolved correctly
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %s0 = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>

  // CHECK: v_add_u32 v2, v0, v1
  %r1 = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Test: comment emission
  // CHECK: ; This is a test comment
  waveasm.comment "This is a test comment"

  // Test: constant immediates
  %c64 = waveasm.constant 64 : !waveasm.imm<64>

  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// CHECK: .amdhsa_kernel assembly_test
// CHECK: .amdhsa_group_segment_fixed_size 0
// CHECK: .amdhsa_user_sgpr_kernarg_segment_ptr 1
// CHECK: .amdhsa_next_free_vgpr
// CHECK: .amdhsa_next_free_sgpr
// CHECK: .end_amdhsa_kernel

// CHECK: .amdgpu_metadata
// CHECK: amdhsa.version:
// CHECK: amdhsa.kernels:
// CHECK: .name: assembly_test
// CHECK: .symbol: assembly_test.kd
// CHECK: .wavefront_size: 64
// CHECK: .end_amdgpu_metadata
