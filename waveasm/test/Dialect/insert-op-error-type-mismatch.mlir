// RUN: not waveasm-translate %s 2>&1 | FileCheck %s

// CHECK: error{{.*}}result type '!waveasm.sreg' must match source type '!waveasm.sreg<4, 4>'
waveasm.program @insert_type_mismatch
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  %s0 = waveasm.precolored.sreg 0 {size = 4, alignment = 4} : !waveasm.sreg<4, 4>
  %word = waveasm.precolored.sreg 10 : !waveasm.sreg
  %bad = waveasm.insert %word into %s0[2] : !waveasm.sreg, !waveasm.sreg<4, 4> -> !waveasm.sreg
  waveasm.s_endpgm
}
