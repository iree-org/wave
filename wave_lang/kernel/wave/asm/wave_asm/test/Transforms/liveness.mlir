// RUN: waveasm-translate %s
// Liveness pass not yet integrated into CLI

// The liveness output appears before the IR
// CHECK: Liveness analysis for simple_kernel
// CHECK: VGPRs:
// CHECK: SGPRs:

waveasm.program @simple_kernel
  target = #waveasm.target<"gfx942", 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  // Define virtual registers for liveness tracking
  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %v1 = waveasm.precolored.vreg 1 : !waveasm.vreg
  %v2 = waveasm.precolored.vreg 2 : !waveasm.vreg

  // Use v1 and v2, def v0
  %sum = waveasm.v_add_u32 %v1, %v2 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // Use v0, def v1
  %moved = waveasm.v_mov_b32 %sum : !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
