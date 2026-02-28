// RUN: waveasm-translate --waveasm-linear-scan %s | FileCheck %s
//
// Test readlane operations: v_readlane_b32 and v_readfirstlane_b32
// These are used for subgroup broadcasts from a specific lane to all lanes.

//===----------------------------------------------------------------------===//
// Test: v_readfirstlane_b32 - read from lane 0
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_readfirstlane
waveasm.program @test_readfirstlane target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Create a VGPR value to read from
  %c42 = waveasm.constant 42 : !waveasm.imm<42>
  %vgpr = waveasm.v_mov_b32 %c42 : !waveasm.imm<42> -> !waveasm.vreg

  // Read from lane 0 into an SGPR
  // CHECK: waveasm.v_readfirstlane_b32
  %sgpr = waveasm.v_readfirstlane_b32 %vgpr : !waveasm.vreg -> !waveasm.sreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: v_readlane_b32 - read from specific lane
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_readlane
waveasm.program @test_readlane target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Create a VGPR value to read from
  %c100 = waveasm.constant 100 : !waveasm.imm<100>
  %vgpr = waveasm.v_mov_b32 %c100 : !waveasm.imm<100> -> !waveasm.vreg

  // Create lane index
  %lane = waveasm.constant 7 : !waveasm.imm<7>

  // Read from lane 7 into an SGPR
  // CHECK: waveasm.v_readlane_b32
  %sgpr = waveasm.v_readlane_b32 %vgpr, %lane : !waveasm.vreg, !waveasm.imm<7> -> !waveasm.sreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: v_readlane_b32 with SGPR lane index
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_readlane_sgpr_lane
waveasm.program @test_readlane_sgpr_lane target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Create a VGPR value
  %c50 = waveasm.constant 50 : !waveasm.imm<50>
  %vgpr = waveasm.v_mov_b32 %c50 : !waveasm.imm<50> -> !waveasm.vreg

  // Lane from an SGPR (e.g., computed at runtime)
  %lane_sreg = waveasm.precolored.sreg 0, 1 : !waveasm.psreg<0, 1>

  // Read from lane specified by SGPR
  // CHECK: waveasm.v_readlane_b32
  %sgpr = waveasm.v_readlane_b32 %vgpr, %lane_sreg : !waveasm.vreg, !waveasm.psreg<0, 1> -> !waveasm.sreg

  waveasm.s_endpgm
}
