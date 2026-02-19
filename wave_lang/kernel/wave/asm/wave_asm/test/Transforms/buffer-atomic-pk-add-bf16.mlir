// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: buffer_atomic_pk_add_bf16 assembly emission

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK: .text
// CHECK: .globl atomic_bf16_test
// CHECK: atomic_bf16_test:

waveasm.program @atomic_bf16_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %s0 = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // CHECK: buffer_atomic_pk_add_bf16 v0, v1, s[0:3], 0 offen
  waveasm.buffer_atomic_pk_add_bf16 %v0, %s0, %v1 : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.pvreg<1>

  // CHECK: buffer_atomic_pk_add_bf16 v0, v1, s[0:3], 0 offen offset:256
  waveasm.buffer_atomic_pk_add_bf16 %v0, %s0, %v1 offset : 256 : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.pvreg<1>

  // CHECK: buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen
  waveasm.buffer_atomic_add_f32 %v0, %s0, %v1 : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.pvreg<1>

  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// CHECK: .amdhsa_kernel atomic_bf16_test
// CHECK: .end_amdhsa_kernel
