// RUN: waveasm-translate %s --waveasm-linear-scan 2>&1 | FileCheck %s

// Test: inserted value defined before a use of the source SRD.
// The mov into the target slot must appear at the InsertOp position,
// not at the inserted value's def. Otherwise the intervening extract
// from the source SRD would see the new value in slot [2] prematurely.

// CHECK-LABEL: waveasm.program @insert_no_early_clobber
waveasm.program @insert_no_early_clobber
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  %srd = waveasm.precolored.sreg 0 {size = 4, alignment = 4} : !waveasm.sreg<4, 4>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg

  // Define the new word value.
  %c42 = waveasm.constant 42 : !waveasm.imm<42>
  // CHECK: [[NEW:%.*]] = waveasm.s_mov_b32 {{.*}} : !waveasm.imm<42>
  %new_word = waveasm.s_mov_b32 %c42 : !waveasm.imm<42> -> !waveasm.sreg

  // Extract word 2 from source SRD -- must see original value, not 42.
  // CHECK: waveasm.extract {{.*}}[2]
  // CHECK-NEXT: waveasm.dce_protect
  %orig_w2 = waveasm.extract %srd[2] : !waveasm.sreg<4, 4> -> !waveasm.sreg
  waveasm.dce_protect %orig_w2 : !waveasm.sreg

  // The mov into srd[2] appears here, after the extract.
  // CHECK: waveasm.s_mov_b32 [[NEW]] {{.*}} -> !waveasm.psreg<2>
  // CHECK-NEXT: waveasm.dce_protect
  // CHECK-NEXT: waveasm.insert
  %updated = waveasm.insert %new_word into %srd[2] : !waveasm.sreg, !waveasm.sreg<4, 4> -> !waveasm.sreg<4, 4>
  waveasm.dce_protect %updated : !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}
