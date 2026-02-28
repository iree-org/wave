// RUN: waveasm-translate %s --waveasm-linear-scan --emit-assembly -o %t.s
// RUN: FileCheck %s < %t.s
// RUN: %rocm_clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -c %t.s -o %t.o
// RUN: %rocm_lld -shared -o %t.hsaco %t.o
// RUN: file %t.hsaco | FileCheck --check-prefix=CHECK-HSACO %s
//
// REQUIRES: rocm-toolchain
//
// This test verifies that the generated assembly can be assembled
// by clang into valid GPU object code and linked into an HSACO.

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK: .globl simple_add
// CHECK: simple_add:
// CHECK: v_add_u32
// CHECK: s_endpgm
// CHECK: .amdhsa_kernel simple_add

// CHECK-HSACO: ELF 64-bit LSB shared object, AMD GPU

waveasm.program @simple_add
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  // Get some input values
  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %v1 = waveasm.precolored.vreg 1 : !waveasm.vreg

  // Add them
  %v2 = waveasm.v_add_u32 %v0, %v1 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
