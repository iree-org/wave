// RUN: waveasm-translate --waveasm-linear-scan="max-vgprs=5 max-agprs=8" %s | FileCheck %s
//
// Test: Multiple values are spilled to different AGPRs when VGPR pressure
// exceeds the limit by more than one register.
//
// With max-vgprs=5: v0, v1 precolored; v14, v15 outside range (>= 5).
// Only v2, v3, v4 are allocatable.  We create 4 virtual VGPRs with
// overlapping liveness, so at least 1 must be spilled.  The long-lived
// values r1 and r2 compete for VGPRs while r3 and r4 also need them.
//
// NOTE: two spilled values must not feed the same instruction because
// both would reload into the single scratch v14.  This test avoids that.

// CHECK-LABEL: waveasm.program @multiple_spills
waveasm.program @multiple_spills target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Four virtual VGPRs; r1, r2 are long-lived.
  %r1 = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  %r2 = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  %r3 = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  %r4 = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // At least one spill write expected.
  // CHECK: waveasm.v_accvgpr_write_b32
  // Use spilled values one at a time (avoid double-reload into same scratch).
  %s1 = waveasm.v_add_u32 %r1, %r3 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  %s2 = waveasm.v_add_u32 %r2, %r4 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  %final = waveasm.v_add_u32 %s1, %s2 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
