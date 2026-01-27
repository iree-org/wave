// RUN: waveasm-translate --waveasm-hazard-mitigation --target=gfx942 %s | FileCheck %s
//
// Test: Hazard mitigation pass inserts s_nop for VALU -> v_readfirstlane hazard

// CHECK-LABEL: waveasm.program @readfirstlane_hazard
waveasm.program @readfirstlane_hazard target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // VALU instruction that writes to VGPR
  // CHECK: waveasm.v_add_u32
  %valu_result = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // v_readfirstlane reading the VALU result requires NOP hazard mitigation
  // CHECK: waveasm.s_nop
  // CHECK: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %valu_result : !waveasm.vreg -> !waveasm.sreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @no_hazard_different_reg
waveasm.program @no_hazard_different_reg target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v2 = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  // VALU instruction writes to one register
  %valu_result = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // v_readfirstlane reading a DIFFERENT register - no hazard
  // CHECK: waveasm.v_readfirstlane_b32 %{{.*}} : !waveasm.pvreg<2>
  // CHECK-NOT: waveasm.s_nop
  %scalar = waveasm.v_readfirstlane_b32 %v2 : !waveasm.pvreg<2> -> !waveasm.sreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @no_hazard_with_gap
waveasm.program @no_hazard_with_gap target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>

  // VALU instruction
  %valu_result = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Some intervening instructions
  %temp1 = waveasm.v_add_u32 %v0, %c10 : !waveasm.pvreg<0>, !waveasm.imm<10> -> !waveasm.vreg
  %temp2 = waveasm.v_add_u32 %v1, %c10 : !waveasm.pvreg<1>, !waveasm.imm<10> -> !waveasm.vreg
  %temp3 = waveasm.v_add_u32 %temp1, %temp2 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  %temp4 = waveasm.v_add_u32 %temp3, %c10 : !waveasm.vreg, !waveasm.imm<10> -> !waveasm.vreg

  // Now reading valu_result - enough instructions passed, no extra NOP needed
  // (The pass should track instruction distance)
  %scalar = waveasm.v_readfirstlane_b32 %valu_result : !waveasm.vreg -> !waveasm.sreg

  waveasm.s_endpgm
}
