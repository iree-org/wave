// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: v_cndmask_b32 uses VOP2 (no vcc) when both sources are VGPRs,
// and VOP3 (explicit vcc) when a source is a non-VGPR.

// CHECK-LABEL: vcndmask_vop2:

waveasm.program @vcndmask_vop2 target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>

  // Both sources are VGPRs -- VOP2, no explicit vcc.
  // CHECK: v_cndmask_b32 v{{[0-9]+}}, v0, v1{{$}}
  %r = waveasm.v_cndmask_b32 %v0, %v1, %vcc : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// CHECK-LABEL: vcndmask_vop3:

waveasm.program @vcndmask_vop3 target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>

  // src0 is an inline constant -- forces VOP3, explicit vcc.
  // CHECK: v_cndmask_b32 v{{[0-9]+}}, 0, v0, vcc
  %r = waveasm.v_cndmask_b32 %c0, %v0, %vcc : !waveasm.imm<0>, !waveasm.pvreg<0>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
