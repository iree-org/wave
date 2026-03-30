// RUN: not waveasm-translate %s --waveasm-linear-scan 2>&1 | FileCheck %s

// CHECK: error{{.*}}insert with AGPR operands is not yet supported
waveasm.program @insert_agpr_unsupported
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  %avec = waveasm.precolored.areg 0 {size = 4} : !waveasm.areg<4>
  %a10 = waveasm.precolored.areg 10 : !waveasm.areg
  %updated = waveasm.insert %a10 into %avec[2] : !waveasm.areg, !waveasm.areg<4> -> !waveasm.areg<4>
  waveasm.dce_protect %updated : !waveasm.areg<4>

  waveasm.s_endpgm
}
