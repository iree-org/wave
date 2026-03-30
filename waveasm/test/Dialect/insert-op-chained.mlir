// RUN: waveasm-translate %s --waveasm-linear-scan 2>&1 | FileCheck %s

// Test chained InsertOps: replace multiple words in sequence.
// The liveness pass must walk the chain to find the root source
// since intermediate insert results are erased from the worklist.

// CHECK-LABEL: waveasm.program @insert_chained
waveasm.program @insert_chained
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  %srd = waveasm.precolored.sreg 0 {size = 4, alignment = 4} : !waveasm.sreg<4, 4>
  %w2 = waveasm.precolored.sreg 10 : !waveasm.sreg
  %w3 = waveasm.precolored.sreg 11 : !waveasm.sreg

  // First insert: mov into slot 2, then insert.
  // CHECK: waveasm.s_mov_b32 {{.*}} -> !waveasm.psreg<2>
  // CHECK: waveasm.insert {{.*}} into {{.*}}[2] {{.*}} -> !waveasm.psreg<0, 4>
  %step1 = waveasm.insert %w2 into %srd[2] : !waveasm.sreg, !waveasm.sreg<4, 4> -> !waveasm.sreg<4, 4>

  // Second insert: chains from %step1, mov into slot 3.
  // CHECK: waveasm.s_mov_b32 {{.*}} -> !waveasm.psreg<3>
  // CHECK: waveasm.insert {{.*}} into {{.*}}[3] {{.*}} -> !waveasm.psreg<0, 4>
  %step2 = waveasm.insert %w3 into %step1[3] : !waveasm.sreg, !waveasm.sreg<4, 4> -> !waveasm.sreg<4, 4>
  waveasm.dce_protect %step2 : !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}
