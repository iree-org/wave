// RUN: waveasm-translate %s

// Test InsertOp roundtrip: replace a single word in a wide register.
waveasm.program @insert_sgpr
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  %s0 = waveasm.precolored.sreg 0 {size = 4, alignment = 4} : !waveasm.sreg<4, 4>
  %word = waveasm.precolored.sreg 10 : !waveasm.sreg

  // Replace word 2 (num_records) in an SRD.
  %updated = waveasm.insert %word into %s0[2] : !waveasm.sreg, !waveasm.sreg<4, 4> -> !waveasm.sreg<4, 4>
  waveasm.dce_protect %updated : !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}
