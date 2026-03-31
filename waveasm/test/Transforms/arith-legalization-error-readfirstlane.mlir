// RUN: not waveasm-translate %s --waveasm-arith-legalization 2>&1 | FileCheck %s
// Verify that readfirstlane from unsupported width produces a diagnostic.

waveasm.program @test_readfirstlane_bad_width
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v_wide = waveasm.precolored.vreg 0, 4 : !waveasm.vreg<4, 4>

  // CHECK: unsupported readfirstlane width (got 4 dwords)
  %rfl = waveasm.arith.readfirstlane %v_wide : (!waveasm.vreg<4, 4>) -> !waveasm.sreg

  waveasm.s_endpgm
}
