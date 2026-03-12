// RUN: not waveasm-translate %s --waveasm-arith-legalization 2>&1 | FileCheck %s
// Verify that sext from non-i32 source produces a diagnostic.

waveasm.program @test_sext_bad_source
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s_wide = waveasm.precolored.sreg 0, 2 : !waveasm.sreg<2, 2>

  // CHECK: sext source must be i32 (got 2 dwords)
  %sext = waveasm.arith.sext %s_wide : (!waveasm.sreg<2, 2>) -> !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}
