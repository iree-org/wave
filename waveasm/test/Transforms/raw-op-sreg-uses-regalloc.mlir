// RUN: waveasm-translate --waveasm-linear-scan %s | FileCheck %s
// Verify that sreg_uses on RawOp reserves physical SGPRs so the
// linear-scan allocator does not assign them to virtual registers.

// Without sreg_uses the allocator would assign s[2:3] to %val.
// With sreg_uses = [2, 3] it must skip to s[4:5].
// CHECK-LABEL: waveasm.program @sreg_uses_reserve
waveasm.program @sreg_uses_reserve target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>

  // Reserve s2 and s3 via sreg_uses.
  waveasm.raw "s_load_dword s2, s[0:1], 0" sreg_uses = [2, 3]

  // CHECK: waveasm.s_load_dwordx2 {{.*}} -> !waveasm.psreg<4, 2>
  %val = waveasm.s_load_dwordx2 %s0, %c0 : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<2>

  waveasm.s_endpgm
}

// Without sreg_uses the same virtual register gets s[2:3].
// CHECK-LABEL: waveasm.program @no_sreg_uses_baseline
waveasm.program @no_sreg_uses_baseline target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>

  // No sreg_uses -- allocator is free to use s2.
  waveasm.raw "s_load_dword s2, s[0:1], 0"

  // CHECK: waveasm.s_load_dwordx2 {{.*}} -> !waveasm.psreg<2, 2>
  %val = waveasm.s_load_dwordx2 %s0, %c0 : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<2>

  waveasm.s_endpgm
}

// Multiple raw ops accumulate reservations.
// CHECK-LABEL: waveasm.program @sreg_uses_multiple
waveasm.program @sreg_uses_multiple target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>

  // Reserve s2, s3, s4, s5.
  waveasm.raw "s_nop 0" sreg_uses = [2, 3]
  waveasm.raw "s_nop 0" sreg_uses = [4, 5]

  // Must skip to s6.
  // CHECK: waveasm.s_load_dwordx2 {{.*}} -> !waveasm.psreg<6, 2>
  %val = waveasm.s_load_dwordx2 %s0, %c0 : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<2>

  waveasm.s_endpgm
}
