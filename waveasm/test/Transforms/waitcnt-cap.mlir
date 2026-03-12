// RUN: waveasm-translate --waveasm-insert-waitcnt %s | FileCheck %s
//
// Test: Waitcnt values are capped to hardware limits.
// GFX9 has 4-bit lgkmcnt (max 15) and 6-bit vmcnt (max 63).
// When the computed threshold exceeds the hardware limit, the pass
// must fall back to 0 (full drain) for correctness.

// CHECK-LABEL: waveasm.program @lgkmcnt_overflow_cap
waveasm.program @lgkmcnt_overflow_cap target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %addr = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Issue 17 LDS reads to make the first load's lgkmcnt threshold = 16,
  // which exceeds the 4-bit hardware max of 15.
  %ld0  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld1  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld2  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld3  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld4  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld5  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld6  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld7  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld8  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld9  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld10 = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld11 = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld12 = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld13 = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld14 = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld15 = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  %ld16 = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg

  // Using ld0 requires lgkmcnt(16) but max is 15.
  // The pass must cap this to lgkmcnt(0) instead of emitting an invalid value.
  // CHECK: waveasm.s_waitcnt
  // CHECK-NOT: lgkmcnt(16)
  // CHECK-NOT: lgkmcnt(17)
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %result = waveasm.v_add_u32 %ld0, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  waveasm.s_endpgm
}
