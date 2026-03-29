// RUN: waveasm-translate --waveasm-linear-scan="max-vgprs=8 max-agprs=4" %s | FileCheck %s
//
// Test: No spill needed when VGPRs fit within the limit.
// With max-vgprs=8 and only 3 virtual VGPRs needed, no spill should occur.

// CHECK-LABEL: waveasm.program @no_spill_needed
// CHECK-NOT: v_accvgpr_write_b32
// CHECK-NOT: v_accvgpr_read_b32
waveasm.program @no_spill_needed target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %r1 = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  %r2 = waveasm.v_add_u32 %r1, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %r3 = waveasm.v_add_u32 %r2, %v1 : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg
  %sum = waveasm.v_add_u32 %r1, %r3 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  waveasm.s_endpgm
}
