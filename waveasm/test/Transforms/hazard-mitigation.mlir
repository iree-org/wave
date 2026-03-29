// RUN: waveasm-translate --waveasm-hazard-mitigation %s 2>&1 | FileCheck %s
//
// Test the hazard mitigation pass:
// - VALU -> v_readfirstlane hazard (gfx940+)
// - Trans -> non-Trans VALU forwarding hazard (gfx940+)

// Physical-register conflict: VALU writes pvreg<2>, readfirstlane reads it.
// This is the realistic post-regalloc form.
// CHECK-LABEL: waveasm.program @valu_readfirstlane_hazard
waveasm.program @valu_readfirstlane_hazard target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Define input registers.
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // VALU instruction that writes to a VGPR.
  // CHECK: waveasm.v_add_u32
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.pvreg<2>

  // The hazard pass should insert s_nop between VALU and v_readfirstlane
  // when the same VGPR is written then immediately read.
  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %sum : !waveasm.pvreg<2> -> !waveasm.sreg

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

// SSA-identity conflict: VALU writes vreg (virtual), readfirstlane reads the
// same SSA value. Tests the fallback path for pre-regalloc or mixed-type IR.
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

// -----------------------------------------------------------------------
// Post-regalloc hazard detection tests.
// After register allocation, different SSA values may share the same
// physical VGPR.  The hazard checker must compare physical register
// indices, not just SSA value identity.
// -----------------------------------------------------------------------

// Test: Different SSA values at the same physical VGPR.
// The VALU writes to pvreg<12> via one SSA value; v_readfirstlane reads
// pvreg<12> via a different SSA value.  The hazard must still be detected.
// CHECK-LABEL: waveasm.program @phys_reg_alias_hazard
waveasm.program @phys_reg_alias_hazard target = #waveasm.target<#waveasm.gfx950, 5> abi = #waveasm.abi<> {
  %v12 = waveasm.precolored.vreg 12 : !waveasm.pvreg<12>
  %v26 = waveasm.precolored.vreg 26 : !waveasm.pvreg<26>
  %imm12 = waveasm.constant 12 : !waveasm.imm<12>

  // VALU writes to pvreg<12> (result is a new SSA value, not %v12).
  // CHECK: waveasm.v_lshlrev_b32
  %shifted = waveasm.v_lshlrev_b32 %imm12, %v26 : !waveasm.imm<12>, !waveasm.pvreg<26> -> !waveasm.pvreg<12>

  // v_readfirstlane reads %v12 -- same physical register, different SSA value.
  // CHECK-NEXT: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v12 : !waveasm.pvreg<12> -> !waveasm.sreg

  waveasm.s_endpgm
}

// -----------------------------------------------------------------------
// Non-emitting ops between VALU and v_readfirstlane.
// Ops like extract, constant, and precolored.sreg do not produce assembly
// instructions.  The hazard checker must look past them to find the
// nearest real predecessor.
// -----------------------------------------------------------------------

// Test: Extract ops between VALU write and v_readfirstlane read.
// CHECK-LABEL: waveasm.program @non_emitting_extract_hazard
waveasm.program @non_emitting_extract_hazard target = #waveasm.target<#waveasm.gfx950, 5> abi = #waveasm.abi<> {
  %v26 = waveasm.precolored.vreg 26 : !waveasm.pvreg<26>
  %s24 = waveasm.precolored.sreg 24, 4 : !waveasm.psreg<24, 4>
  %imm12 = waveasm.constant 12 : !waveasm.imm<12>

  // VALU writes to pvreg<12>.
  // CHECK: waveasm.v_lshlrev_b32
  %shifted = waveasm.v_lshlrev_b32 %imm12, %v26 : !waveasm.imm<12>, !waveasm.pvreg<26> -> !waveasm.pvreg<12>

  // Non-emitting extract ops (no assembly emitted).
  %w0 = waveasm.extract %s24[0] : !waveasm.psreg<24, 4> -> !waveasm.psreg<24>
  %w1 = waveasm.extract %s24[1] : !waveasm.psreg<24, 4> -> !waveasm.psreg<25>

  // v_readfirstlane reads the VALU result.  Despite the intervening
  // extracts, a NOP is required because no real instruction separates
  // the VALU write from the readfirstlane.
  // CHECK: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %shifted : !waveasm.pvreg<12> -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test: Both issues combined -- alias + non-emitting gap.
// CHECK-LABEL: waveasm.program @alias_through_non_emitting
waveasm.program @alias_through_non_emitting target = #waveasm.target<#waveasm.gfx950, 5> abi = #waveasm.abi<> {
  %v12 = waveasm.precolored.vreg 12 : !waveasm.pvreg<12>
  %v26 = waveasm.precolored.vreg 26 : !waveasm.pvreg<26>
  %s24 = waveasm.precolored.sreg 24, 4 : !waveasm.psreg<24, 4>
  %imm12 = waveasm.constant 12 : !waveasm.imm<12>

  // VALU writes to pvreg<12>.
  // CHECK: waveasm.v_lshlrev_b32
  %shifted = waveasm.v_lshlrev_b32 %imm12, %v26 : !waveasm.imm<12>, !waveasm.pvreg<26> -> !waveasm.pvreg<12>

  // Non-emitting ops in between.
  %w0 = waveasm.extract %s24[0] : !waveasm.psreg<24, 4> -> !waveasm.psreg<24>
  %w1 = waveasm.extract %s24[1] : !waveasm.psreg<24, 4> -> !waveasm.psreg<25>

  // v_readfirstlane reads pvreg<12> via a different SSA value (%v12)
  // through non-emitting ops.
  // CHECK: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v12 : !waveasm.pvreg<12> -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test: A real instruction between VALU and v_readfirstlane -- no NOP needed.
// CHECK-LABEL: waveasm.program @real_instruction_gap_no_hazard
waveasm.program @real_instruction_gap_no_hazard target = #waveasm.target<#waveasm.gfx950, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v12 = waveasm.precolored.vreg 12 : !waveasm.pvreg<12>
  %imm12 = waveasm.constant 12 : !waveasm.imm<12>

  // VALU writes to pvreg<12>.
  %shifted = waveasm.v_lshlrev_b32 %imm12, %v0 : !waveasm.imm<12>, !waveasm.pvreg<0> -> !waveasm.pvreg<12>

  // A real emitting instruction provides the needed delay.
  // CHECK: waveasm.v_add_u32
  %gap = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.pvreg<2>

  // No NOP needed because v_add_u32 provided the delay.
  // CHECK-NOT: waveasm.s_nop
  // CHECK: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v12 : !waveasm.pvreg<12> -> !waveasm.sreg

  waveasm.s_endpgm
}

// -----------------------------------------------------------------------
// Region boundary tests.
// The backward scan must not cross region boundaries. Structured
// control flow ops (waveasm.if, waveasm.loop) are not in the
// isNonEmittingOp list, so they act as barriers.
// -----------------------------------------------------------------------

// Test: VALU before an if, v_readfirstlane inside the if body.
// The waveasm.if op sits between them and acts as a barrier, so the
// backward scan stops there and does not see the VALU.
// CHECK-LABEL: waveasm.program @region_boundary_blocks_hazard_scan
waveasm.program @region_boundary_blocks_hazard_scan target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %cond = waveasm.precolored.sreg 0 : !waveasm.psreg<0>

  // VALU writes pvreg<2>.
  // CHECK: waveasm.v_add_u32
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.pvreg<2>

  // waveasm.if is an emitting op -- the backward scan stops here.
  // CHECK: waveasm.if
  waveasm.if %cond : !waveasm.psreg<0> {
    // No s_nop needed: the scan sees waveasm.if as the predecessor,
    // not the v_add_u32 above.
    // CHECK-NOT: waveasm.s_nop
    // CHECK: waveasm.v_readfirstlane_b32
    %scalar = waveasm.v_readfirstlane_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.sreg
    waveasm.yield
  }

  waveasm.s_endpgm
}

// Test: VALU inside an if body, v_readfirstlane after the if.
// The backward scan looks into the region to find the last emitting op.
// CHECK-LABEL: waveasm.program @region_exit_hazard
waveasm.program @region_exit_hazard target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %cond = waveasm.precolored.sreg 0 : !waveasm.psreg<0>

  // CHECK: waveasm.if
  waveasm.if %cond : !waveasm.psreg<0> {
    // VALU writes pvreg<0> -- same register that readfirstlane reads.
    %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.pvreg<0>
    waveasm.yield
  }

  // The scan skips yield and sees v_add_u32 as the predecessor.
  // CHECK: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test: if/else -- VALU in the else branch, readfirstlane after the if.
// The flattened list has the else branch last, so the scan finds its VALU.
// CHECK-LABEL: waveasm.program @if_else_exit_hazard
waveasm.program @if_else_exit_hazard target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %cond = waveasm.precolored.sreg 0 : !waveasm.psreg<0>

  waveasm.if %cond : !waveasm.psreg<0> {
    // Then branch: SALU only, no VGPR hazard.
    %s = waveasm.s_mov_b32 %cond : !waveasm.psreg<0> -> !waveasm.sreg
    waveasm.yield
  } else {
    // Else branch: VALU writes pvreg<0>.
    // CHECK: waveasm.v_add_u32
    %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.pvreg<0>
    waveasm.yield
  }

  // The scan finds v_add_u32 from the else branch.
  // CHECK: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test: if/else -- VALU only in the then branch (not the else).
// The flattened scan only sees the else branch (last in the list) and
// finds s_mov_b32 -- not a VALU, so no hazard is inserted.
// This is accidentally safe: when the then-path is taken, the implicit
// s_branch (skip else) provides the required 1-cycle gap. But the pass
// does not model implicit branches -- it gets the right answer for the
// wrong reason.
// FIXME: for correctness the scan should check all if/else exit paths.
// CHECK-LABEL: waveasm.program @if_else_then_only_valu
waveasm.program @if_else_then_only_valu target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %cond = waveasm.precolored.sreg 0 : !waveasm.psreg<0>

  waveasm.if %cond : !waveasm.psreg<0> {
    // Then branch: VALU writes pvreg<0>.
    %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.pvreg<0>
    waveasm.yield
  } else {
    // Else branch: SALU only.
    %s = waveasm.s_mov_b32 %cond : !waveasm.psreg<0> -> !waveasm.sreg
    waveasm.yield
  }

  // The scan finds s_mov_b32 from the else branch -- not a VALU.
  // Safe in practice due to the implicit branch after the then body.
  // CHECK-NOT: waveasm.s_nop
  // CHECK: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test: VALU inside a loop body, readfirstlane after the loop.
// The condition op (emits s_cbranch) sits between them, acting as a
// real instruction gap. No hazard needed.
// CHECK-LABEL: waveasm.program @loop_exit_no_hazard
waveasm.program @loop_exit_no_hazard target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %limit = waveasm.constant 4 : !waveasm.imm<4>
  %init = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // CHECK: waveasm.loop
  %final = waveasm.loop(%i = %init) : (!waveasm.sreg) -> (!waveasm.sreg) {
    // VALU writes pvreg<0>.
    %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.pvreg<0>
    %cond = waveasm.s_cmp_lt_u32 %i, %limit : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%i) : !waveasm.sreg
  }

  // condition (s_cbranch) is the last emitting op -- not a VALU.
  // CHECK-NOT: waveasm.s_nop
  // CHECK: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test: nested if -- VALU in the inner if, readfirstlane after the outer if.
// The scan skips both yields and finds the VALU.
// CHECK-LABEL: waveasm.program @nested_if_exit_hazard
waveasm.program @nested_if_exit_hazard target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %cond = waveasm.precolored.sreg 0 : !waveasm.psreg<0>

  // CHECK: waveasm.if
  waveasm.if %cond : !waveasm.psreg<0> {
    waveasm.if %cond : !waveasm.psreg<0> {
      // VALU in the innermost if.
      %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.pvreg<0>
      waveasm.yield
    }
    waveasm.yield
  }

  // The scan skips both yields and finds v_add_u32.
  // CHECK: waveasm.s_nop 0
  // CHECK-NEXT: waveasm.v_readfirstlane_b32
  %scalar = waveasm.v_readfirstlane_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.sreg

  waveasm.s_endpgm
}
