// RUN: not waveasm-translate %s --waveasm-linear-scan 2>&1 | FileCheck %s

// CHECK: error{{.*}}multi-word insert is not yet supported
waveasm.program @insert_multiword_unsupported
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  %src = waveasm.precolored.sreg 0 {size = 4} : !waveasm.sreg<4, 4>
  %val = waveasm.precolored.sreg 10 {size = 2} : !waveasm.sreg<2>
  %updated = waveasm.insert %val into %src[1] : !waveasm.sreg<2>, !waveasm.sreg<4, 4> -> !waveasm.sreg<4, 4>
  waveasm.dce_protect %updated : !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}
