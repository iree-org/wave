// RUN: not waveasm-translate --waveasm-linear-scan="max-vgprs=4 max-agprs=0" %s 2>&1 | FileCheck %s
//
// Test: When both VGPR and AGPR pools are exhausted, allocation fails
// with a diagnostic.  max-agprs=0 leaves no spare AGPRs for eviction.

// CHECK: error: 'waveasm.program' op Failed to allocate VGPR
waveasm.program @both_exhausted target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %r1 = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  %r2 = waveasm.v_add_u32 %r1, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %r3 = waveasm.v_add_u32 %r2, %v1 : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg
  %sum = waveasm.v_add_u32 %r1, %r3 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  waveasm.s_endpgm
}
