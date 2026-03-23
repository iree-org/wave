// RUN: waveasm-translate %s --waveasm-arith-legalization | FileCheck %s
// Verify arith.readfirstlane legalization.

// CHECK-LABEL: waveasm.program @test_readfirstlane
waveasm.program @test_readfirstlane
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %s0 = waveasm.precolored.sreg 0 : !waveasm.sreg

  // VGPR i32 -> v_readfirstlane_b32.
  // CHECK: waveasm.v_readfirstlane_b32 %{{.*}} : !waveasm.vreg
  %rfl_v = waveasm.arith.readfirstlane %v0 : (!waveasm.vreg) -> !waveasm.sreg

  // SGPR -> no-op (forwarded).
  // CHECK-NOT: waveasm.v_readfirstlane_b32 %{{.*}} : !waveasm.sreg
  %rfl_s = waveasm.arith.readfirstlane %s0 : (!waveasm.sreg) -> !waveasm.sreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_readfirstlane_i64
waveasm.program @test_readfirstlane_i64
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %v1 = waveasm.precolored.vreg 1 : !waveasm.vreg
  %vpair = waveasm.pack %v0, %v1 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2, 2>

  // VGPR i64 -> two v_readfirstlane_b32 ops + pack.
  // CHECK: [[LO:%.*]] = waveasm.v_readfirstlane_b32
  // CHECK: [[HI:%.*]] = waveasm.v_readfirstlane_b32
  // CHECK: waveasm.pack [[LO]], [[HI]]
  %rfl_64 = waveasm.arith.readfirstlane %vpair : (!waveasm.vreg<2, 2>) -> !waveasm.sreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}
