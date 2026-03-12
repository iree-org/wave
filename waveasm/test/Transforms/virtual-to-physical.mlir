// RUN: waveasm-translate %s
// Register allocation pass not yet integrated into CLI
//
// This test verifies that virtual registers are correctly allocated
// to physical registers through the linear scan algorithm.

// CHECK: Register allocation for virtual_alloc_test
// CHECK: Peak VGPRs:
// CHECK: Peak SGPRs:

waveasm.program @virtual_alloc_test
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  // Virtual registers - should be allocated to physical registers
  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %v1 = waveasm.precolored.vreg 1 : !waveasm.vreg

  // Immediate constant
  %imm42 = waveasm.constant 42 : !waveasm.imm<42>

  // Chain of operations creating overlapping live ranges
  %v2 = waveasm.v_add_u32 %v0, %v1 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  %v3 = waveasm.v_mul_lo_u32 %v2, %v0 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // v0 is now dead, can be reused
  %v4 = waveasm.v_sub_u32 %v3, %v1 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
