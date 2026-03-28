// RUN: waveasm-translate --waveasm-linear-scan="max-vgprs=4 max-agprs=4" %s | FileCheck %s
//
// Test: Tied values (MFMA accumulator -> result) are NOT spill candidates.
// The spill should pick an untied value instead.
//
// With max-vgprs=4: v0, v1 precolored; v14, v15 reserved.  Only v2, v3
// allocatable.  Three values are live simultaneously -> one must spill.
// The spill must NOT be a tied value.

// CHECK-LABEL: waveasm.program @no_tied_spill
waveasm.program @no_tied_spill target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Long-lived untied value.
  %addr = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Second value creating pressure.
  %tmp = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Third value that triggers the spill.
  %extra = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Use all three -> one must be spilled.  Verify that spill/reload appear.
  // CHECK: waveasm.v_accvgpr_write_b32
  // CHECK: waveasm.v_accvgpr_read_b32
  %s1 = waveasm.v_add_u32 %addr, %tmp : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  %s2 = waveasm.v_add_u32 %s1, %extra : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
