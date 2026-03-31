// RUN: not waveasm-translate %s --waveasm-arith-legalization 2>&1 | FileCheck %s
// Verify that unsupported operand widths produce diagnostics.

// Unsupported width: sreg<4> is not i32 or i64.
waveasm.program @test_add_unsupported_width
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s_wide = waveasm.precolored.sreg 0, 4 : !waveasm.sreg<4, 4>
  %s_wide2 = waveasm.precolored.sreg 4, 4 : !waveasm.sreg<4, 4>

  // CHECK: unsupported operand width (expected i32 or i64, got 4 dwords)
  %add = waveasm.arith.add %s_wide, %s_wide2 : (!waveasm.sreg<4, 4>, !waveasm.sreg<4, 4>) -> !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}
