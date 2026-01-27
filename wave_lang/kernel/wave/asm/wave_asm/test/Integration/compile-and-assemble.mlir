// RUN: waveasm-translate %s --emit-assembly -o %t.s
// RUN: FileCheck %s < %t.s
// RUN: %rocm_clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -c %t.s -o %t.o
// RUN: %rocm_lld -shared -o %t.hsaco %t.o
// RUN: file %t.hsaco | FileCheck --check-prefix=CHECK-HSACO %s
//
// REQUIRES: rocm-toolchain
// XFAIL: *
// Assembly emission pass not yet integrated into CLI
//
// This test verifies that the generated assembly can be assembled
// by clang into valid GPU object code and linked into an HSACO.

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK: .globl simple_add
// CHECK: simple_add:
// CHECK: v_add_u32
// CHECK: s_endpgm
// CHECK: .amdhsa_kernel simple_add
// CHECK: .amdhsa_accum_offset

// CHECK-HSACO: ELF 64-bit LSB shared object, AMD GPU

waveasm.program @simple_add
  abi = #waveasm.abi<flatTidVReg = 0, kernargPtrSRegLo = 0, kernargPtrSRegHi = 1>
  target = #waveasm.target<"gfx942", 5>
  attributes {maxVGPRs = 256, maxSGPRs = 104, workgroupSize = [64, 1, 1]} {

  // Physical registers
  %v0 = waveasm.def_pvreg : !waveasm.pvreg<0>
  %v1 = waveasm.def_pvreg : !waveasm.pvreg<1>
  %v2 = waveasm.def_pvreg : !waveasm.pvreg<2>

  waveasm.instr "v_add_u32" defs(%v0 : !waveasm.pvreg<0>)
                           uses(%v1, %v2 : !waveasm.pvreg<1>, !waveasm.pvreg<2>)

  waveasm.instr "s_endpgm" defs() uses()
}
