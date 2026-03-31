// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test: Duplicate pack inputs are resolved by inserting copies during
// liveness analysis, so register allocation succeeds.

waveasm.program @pack_duplicate_input target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %imm0 = waveasm.constant 0 : !waveasm.imm<0>
  %v0 = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg

  // CHECK: waveasm.v_mov_b32
  // CHECK: waveasm.pack
  %packed = waveasm.pack %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  waveasm.buffer_store_dwordx2 %packed, %srd, %voff : !waveasm.vreg<2>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}
