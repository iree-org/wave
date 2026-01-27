// RUN: waveasm-translate %s
// Register allocation pass not yet integrated into CLI

// CHECK: Register allocation for regalloc_test
// CHECK: Peak VGPRs:
// CHECK: Peak SGPRs:

waveasm.program @regalloc_test
  target = #waveasm.target<"gfx942", 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  // Virtual registers to be allocated
  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %v1 = waveasm.precolored.vreg 1 : !waveasm.vreg

  // Instructions with overlapping live ranges
  %v2 = waveasm.v_add_u32 %v0, %v1 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  %v3 = waveasm.v_mov_b32 %v2 : !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
