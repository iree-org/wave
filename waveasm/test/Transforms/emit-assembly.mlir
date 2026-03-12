// RUN: waveasm-translate %s
// Assembly emission pass not yet integrated into CLI

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK: .text
// CHECK: .globl emit_test
// CHECK: emit_test:
// CHECK: v_add_u32
// CHECK: s_endpgm
// CHECK: .amdhsa_kernel emit_test

waveasm.program @emit_test
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  // Precolored physical registers (already allocated)
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v2 = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  waveasm.comment "Add two vectors"

  %result = waveasm.v_add_u32 %v1, %v2 : !waveasm.pvreg<1>, !waveasm.pvreg<2> -> !waveasm.vreg

  waveasm.s_endpgm
}
