// RUN: waveasm-translate %s --waveasm-linear-scan 2>&1 | FileCheck %s

// Test: same value inserted into two different vectors.
// Each InsertOp emits its own mov into the target slot at its position.
// The inserted value is independently allocated, not aliased to the
// source, so no early clobber can occur.

// CHECK-LABEL: waveasm.program @insert_shared_value
waveasm.program @insert_shared_value
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  %srd_a = waveasm.precolored.sreg 0 {size = 4, alignment = 4} : !waveasm.sreg<4, 4>
  %srd_b = waveasm.precolored.sreg 4 {size = 4, alignment = 4} : !waveasm.sreg<4, 4>
  %word = waveasm.precolored.sreg 10 : !waveasm.sreg

  // Each insert emits a mov from %word into the target slot.
  // CHECK: waveasm.s_mov_b32 {{.*}} -> !waveasm.psreg<2>
  // CHECK: waveasm.insert
  // CHECK: waveasm.s_mov_b32 {{.*}} -> !waveasm.psreg<6>
  // CHECK: waveasm.insert
  %a = waveasm.insert %word into %srd_a[2] : !waveasm.sreg, !waveasm.sreg<4, 4> -> !waveasm.sreg<4, 4>
  %b = waveasm.insert %word into %srd_b[2] : !waveasm.sreg, !waveasm.sreg<4, 4> -> !waveasm.sreg<4, 4>
  waveasm.dce_protect %a : !waveasm.sreg<4, 4>
  waveasm.dce_protect %b : !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}
