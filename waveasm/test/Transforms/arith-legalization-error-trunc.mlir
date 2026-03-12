// RUN: not waveasm-translate %s --waveasm-arith-legalization 2>&1 | FileCheck %s
// Verify that trunc from unsupported width produces a diagnostic.

waveasm.program @test_trunc_bad_width
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s_wide = waveasm.precolored.sreg 0, 4 : !waveasm.sreg<4, 4>

  // CHECK: unsupported operand width (expected i32 or i64, got 4 dwords)
  %trunc = waveasm.arith.trunc %s_wide : (!waveasm.sreg<4, 4>) -> !waveasm.sreg

  waveasm.s_endpgm
}
