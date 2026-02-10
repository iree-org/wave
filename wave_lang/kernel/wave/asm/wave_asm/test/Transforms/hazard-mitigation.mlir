// RUN: waveasm-translate --waveasm-hazard-mitigation %s 2>&1 | FileCheck %s
//
// Test the hazard mitigation pass for VALU -> v_readfirstlane hazard

// CHECK-LABEL: waveasm.program @valu_readfirstlane_hazard
waveasm.program @valu_readfirstlane_hazard target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Define input registers
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // VALU instruction that writes to a VGPR
  // CHECK: waveasm.v_add_u32
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // The hazard pass should insert s_nop between VALU and v_readfirstlane
  // when the same VGPR is written then immediately read
  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %sum : !waveasm.vreg -> !waveasm.sreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

// Test case: No hazard when different VGPRs are used
// CHECK-LABEL: waveasm.program @no_hazard_different_vgpr
waveasm.program @no_hazard_different_vgpr target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v2 = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  // VALU writes to result from v0, v1
  // CHECK: waveasm.v_add_u32
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // v_readfirstlane reads from v2 (different from the VALU result)
  // No hazard expected - should NOT insert s_nop
  // CHECK-NOT: waveasm.s_nop
  // CHECK: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v2 : !waveasm.pvreg<2> -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test case: gfx1250 should also need hazard mitigation
// CHECK-LABEL: waveasm.program @gfx1250_hazard
waveasm.program @gfx1250_hazard target = #waveasm.target<#waveasm.gfx1250, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: waveasm.v_add_u32
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Should insert s_nop for gfx1250 as well
  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %sum : !waveasm.vreg -> !waveasm.sreg

  waveasm.s_endpgm
}
