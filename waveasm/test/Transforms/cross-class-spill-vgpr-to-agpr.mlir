// RUN: waveasm-translate --waveasm-linear-scan="max-vgprs=4 max-agprs=4" %s | FileCheck %s
//
// Test: VGPR -> AGPR cross-class spilling.
//
// This program needs 3 virtual VGPRs but only 2 are allocatable (v2, v3)
// given max-vgprs=4 with v0/v1 precolored and v14/v15 reserved as scratch.
// The allocator evicts the longest-lived value (r1) to AGPR a0 and inserts
// v_accvgpr_write/read pairs around its def and uses.

// CHECK-LABEL: waveasm.program @spill_vgpr_to_agpr
waveasm.program @spill_vgpr_to_agpr target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // r1 is live across r2 and r3, but only 2 VGPRs are free.
  // The allocator should spill r1 to AGPR.

  // CHECK: [[R1:%.*]] = waveasm.v_add_u32 {{.*}} -> !waveasm.pvreg<[[R1IDX:[0-9]+]]>
  %r1 = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // CHECK: waveasm.v_accvgpr_write_b32 [[R1]], {{.*}} : !waveasm.pvreg<[[R1IDX]]>, !waveasm.pareg<[[ASPILL:[0-9]+]]>
  // CHECK: [[RELOAD1:%.*]] = waveasm.v_accvgpr_read_b32 {{.*}} : !waveasm.pareg<[[ASPILL]]> -> !waveasm.pvreg<14>

  // CHECK: waveasm.v_add_u32 [[RELOAD1]], {{.*}} -> !waveasm.pvreg<
  %r2 = waveasm.v_add_u32 %r1, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.v_add_u32 {{.*}} -> !waveasm.pvreg<
  %r3 = waveasm.v_add_u32 %r2, %v1 : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg

  // r1 is used again here, so a second reload is needed.
  // CHECK: [[RELOAD2:%.*]] = waveasm.v_accvgpr_read_b32 {{.*}} : !waveasm.pareg<[[ASPILL]]> -> !waveasm.pvreg<14>
  // CHECK: waveasm.v_add_u32 [[RELOAD2]], {{.*}} -> !waveasm.pvreg<
  %sum1 = waveasm.v_add_u32 %r1, %r3 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}
