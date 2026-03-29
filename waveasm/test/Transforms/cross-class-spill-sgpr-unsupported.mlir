// RUN: not waveasm-translate --waveasm-linear-scan="max-sgprs=4 max-vgprs=8" %s 2>&1 | FileCheck %s
//
// Test: SGPR overflow triggers a hard error because SGPR -> VGPR spilling
// is not yet implemented.  The allocator does not attempt cross-class
// eviction for SGPRs (altPool is nullptr).

// CHECK: error: 'waveasm.program' op Failed to allocate SGPR
waveasm.program @sgpr_overflow target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>
  %s2 = waveasm.precolored.sreg 2 : !waveasm.psreg<2>
  %s3 = waveasm.precolored.sreg 3 : !waveasm.psreg<3>

  // All 4 SGPRs precolored.  Virtual SGPRs have nowhere to go.
  %r1 = waveasm.s_mul_i32 %s0, %s1 : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.sreg
  %r2 = waveasm.s_mul_i32 %s2, %s3 : !waveasm.psreg<2>, !waveasm.psreg<3> -> !waveasm.sreg
  %r3 = waveasm.s_mul_i32 %r1, %r2 : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg

  waveasm.s_endpgm
}
