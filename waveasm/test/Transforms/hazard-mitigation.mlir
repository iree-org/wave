// RUN: waveasm-translate --waveasm-hazard-mitigation %s 2>&1 | FileCheck %s
//
// Test the hazard mitigation pass:
// - VALU -> v_readfirstlane hazard (gfx940+)
// - Trans -> non-Trans VALU forwarding hazard (gfx940+)

// CHECK-LABEL: waveasm.program @valu_readfirstlane_hazard
waveasm.program @valu_readfirstlane_hazard target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Define input registers
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // VALU instruction that writes to a VGPR
  // CHECK: waveasm.v_add_u32
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // The hazard pass should insert s_nop between VALU and v_readfirstlane
  // when the same VGPR is written then immediately read
  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %sum : !waveasm.vreg -> !waveasm.sreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

// Test case: No hazard when different VGPRs are used
// CHECK-LABEL: waveasm.program @no_hazard_different_vgpr
waveasm.program @no_hazard_different_vgpr target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v2 = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  // VALU writes to result from v0, v1
  // CHECK: waveasm.v_add_u32
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // v_readfirstlane reads from v2 (different from the VALU result)
  // No hazard expected - should NOT insert s_nop
  // CHECK-NOT: waveasm.s_nop
  // CHECK: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v2 : !waveasm.pvreg<2> -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test case: gfx1250 should also need hazard mitigation
// CHECK-LABEL: waveasm.program @gfx1250_hazard
waveasm.program @gfx1250_hazard target = #waveasm.target<#waveasm.gfx1250, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: waveasm.v_add_u32
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Should insert s_nop for gfx1250 as well.
  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %sum : !waveasm.vreg -> !waveasm.sreg

  waveasm.s_endpgm
}

// -----------------------------------------------------------------------
// Trans -> non-Trans VALU forwarding hazard tests (gfx940+).
// Transcendental ops (v_rcp_f32, v_rsq_f32, ...) have a one-cycle
// forwarding hazard when a non-Trans VALU immediately consumes the
// result. See LLVM TransDefWaitstates = 1.
// -----------------------------------------------------------------------

// Test: v_rcp_f32 result consumed by v_mul_f32 -- needs s_nop.
// CHECK-LABEL: waveasm.program @trans_valu_hazard_rcp
waveasm.program @trans_valu_hazard_rcp target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: waveasm.v_rcp_f32
  %rcp = waveasm.v_rcp_f32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_mul_f32
  %result = waveasm.v_mul_f32 %rcp, %v1 : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test: v_rsq_f32 result consumed by v_add_f32 -- needs s_nop.
// CHECK-LABEL: waveasm.program @trans_valu_hazard_rsq
waveasm.program @trans_valu_hazard_rsq target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: waveasm.v_rsq_f32
  %rsq = waveasm.v_rsq_f32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_add_f32
  %result = waveasm.v_add_f32 %rsq, %rsq : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test: v_sqrt_f32 result consumed by v_mul_f32 -- needs s_nop.
// CHECK-LABEL: waveasm.program @trans_valu_hazard_sqrt
waveasm.program @trans_valu_hazard_sqrt target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: waveasm.v_sqrt_f32
  %sqrt = waveasm.v_sqrt_f32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_mul_f32
  %result = waveasm.v_mul_f32 %sqrt, %v1 : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test: Trans result NOT consumed by next VALU -- no s_nop needed.
// CHECK-LABEL: waveasm.program @trans_no_hazard_different_vgpr
waveasm.program @trans_no_hazard_different_vgpr target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v2 = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  // v_rcp_f32 writes to its result, but the next VALU reads different regs.
  // CHECK: waveasm.v_rcp_f32
  %rcp = waveasm.v_rcp_f32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK-NOT: waveasm.s_nop
  // CHECK: waveasm.v_add_u32
  %result = waveasm.v_add_u32 %v1, %v2 : !waveasm.pvreg<1>, !waveasm.pvreg<2> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test: Trans followed by non-VALU op -- no hazard.
// CHECK-LABEL: waveasm.program @trans_no_hazard_non_valu_consumer
waveasm.program @trans_no_hazard_non_valu_consumer target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: waveasm.v_rcp_f32
  %rcp = waveasm.v_rcp_f32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // v_readfirstlane is not a VALU op, so no Trans hazard.
  // (It IS a readfirstlane hazard since v_rcp_f32 is VALU, but that's
  // a different hazard type.)
  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %rcp : !waveasm.vreg -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test: Trans -> Trans (same VGPR) -- no s_nop needed.
// The forwarding hazard only applies when a non-Trans VALU consumes
// the Trans result. See LLVM: !SIInstrInfo::isTRANS(*VALU) guard.
// CHECK-LABEL: waveasm.program @trans_trans_no_hazard
waveasm.program @trans_trans_no_hazard target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: waveasm.v_rcp_f32
  %rcp = waveasm.v_rcp_f32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // Trans consuming Trans result -- no forwarding hazard.
  // CHECK-NOT: waveasm.s_nop
  // CHECK: waveasm.v_sqrt_f32
  %sqrt = waveasm.v_sqrt_f32 %rcp : !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
