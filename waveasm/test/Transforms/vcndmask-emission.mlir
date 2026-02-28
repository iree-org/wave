// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: v_cndmask_b32 drops explicit VCC operand

// CHECK-LABEL: vcndmask_test:

waveasm.program @vcndmask_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>

  // VCC operand should be dropped (implicit in VOP2)
  // CHECK: v_cndmask_b32 v{{[0-9]+}}, v0, v1
  // CHECK-NOT: vcc
  %r = waveasm.v_cndmask_b32 %v0, %v1, %vcc : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
