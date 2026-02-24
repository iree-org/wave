// RUN: waveasm-translate --waveasm-loop-address-promotion %s 2>&1 | FileCheck %s
//
// Tests for the LoopAddressPromotion pass.
//
// The pass finds V_ADD_U32 ops inside LoopOps that:
//   1. Feed a DS_READ (LDS load).
//   2. Have one loop-invariant VGPR operand and one rotating SGPR block arg.
// It replaces them with precomputed rotating VGPR iter_args, eliminating
// per-iteration V_ADD_U32 address computation from the loop body.

// ---- Basic: double-buffered SGPR rotation, one LDS read ----
// Two SGPR block args rotate each iteration (s0->s1, s1->s0).
// A V_ADD_U32(tid, s0) feeds a DS_READ_B128. After promotion:
//   - Two precomputed VGPRs (v_add(tid,init_s0), v_add(tid,init_s1)) become iter_args.
//   - The V_ADD_U32 inside the loop is eliminated.
//   - The new VGPR iter_args rotate each iteration.

// CHECK-LABEL: @basic_double_buffer
waveasm.program @basic_double_buffer
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 8 : !waveasm.imm<8>

  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s0 = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s1 = waveasm.s_mov_b32 %one  : !waveasm.imm<1> -> !waveasm.sreg

  // After promotion the loop should have extra VGPR iter_args for the
  // precomputed addresses.
  // CHECK: waveasm.loop
  // CHECK-SAME: !waveasm.vreg
  %r:3 = waveasm.loop(%iv = %init_iv, %s0 = %init_s0, %s1 = %init_s1)
      : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
     -> (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg) {

    // V_ADD_U32 feeding DS_READ — should be promoted out of the loop.
    // CHECK-NOT: waveasm.v_add_u32 {{.*}} !waveasm.pvreg<0>, !waveasm.sreg
    %addr = waveasm.v_add_u32 %tid, %s0 : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg

    // The DS_READ should still be present, using a VGPR (iter_arg) address.
    // CHECK: waveasm.ds_read_b128
    // CHECK-SAME: !waveasm.vreg
    %val = waveasm.ds_read_b128 %addr : !waveasm.vreg -> !waveasm.vreg<4, 4>

    // IV bump + condition.
    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    // Rotation: s0->s1, s1->s0.
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %s1, %s0) : !waveasm.sreg, !waveasm.sreg, !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- No transformation: V_ADD_U32 does not feed DS_READ ----
// The V_ADD_U32 result is used by a non-LDS op, so the pass should skip it.

// CHECK-LABEL: @no_transform_not_lds
waveasm.program @no_transform_not_lds
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 8 : !waveasm.imm<8>

  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s0 = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s1 = waveasm.s_mov_b32 %one  : !waveasm.imm<1> -> !waveasm.sreg

  // Loop should NOT gain extra iter_args.
  // CHECK: waveasm.loop
  // CHECK-NOT: !waveasm.vreg
  // CHECK-SAME: {
  %r:3 = waveasm.loop(%iv = %init_iv, %s0 = %init_s0, %s1 = %init_s1)
      : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
     -> (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg) {

    // V_ADD_U32 does NOT feed a DS_READ — feeds a regular V_ADD_U32 instead.
    // CHECK: waveasm.v_add_u32 {{.*}} !waveasm.pvreg<0>, !waveasm.sreg
    %addr = waveasm.v_add_u32 %tid, %s0 : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %dummy = waveasm.v_add_u32 %addr, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %s1, %s0) : !waveasm.sreg, !waveasm.sreg, !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- No transformation: no rotation group ----
// Block args are passed through unchanged (identity permutation), so no
// rotation group is detected.

// CHECK-LABEL: @no_transform_no_rotation
waveasm.program @no_transform_no_rotation
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 8 : !waveasm.imm<8>

  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s0 = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // CHECK: waveasm.loop
  // CHECK-NOT: !waveasm.vreg
  // CHECK-SAME: {
  %r:2 = waveasm.loop(%iv = %init_iv, %s0 = %init_s0)
      : (!waveasm.sreg, !waveasm.sreg)
     -> (!waveasm.sreg, !waveasm.sreg) {

    // s0 is passed through unchanged — no rotation, no promotion.
    // CHECK: waveasm.v_add_u32 {{.*}} !waveasm.pvreg<0>, !waveasm.sreg
    %addr = waveasm.v_add_u32 %tid, %s0 : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %val = waveasm.ds_read_b128 %addr : !waveasm.vreg -> !waveasm.vreg<4, 4>

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    // Identity: s0->s0, no rotation.
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %s0) : !waveasm.sreg, !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- Triple-buffered rotation ----
// Three SGPR block args rotate (s0->s1->s2->s0). The V_ADD_U32 feeding
// DS_READ should be promoted with 3 precomputed VGPR iter_args.

// CHECK-LABEL: @triple_buffer
waveasm.program @triple_buffer
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %two = waveasm.constant 2 : !waveasm.imm<2>
  %limit = waveasm.constant 8 : !waveasm.imm<8>

  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s0 = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s1 = waveasm.s_mov_b32 %one  : !waveasm.imm<1> -> !waveasm.sreg
  %init_s2 = waveasm.s_mov_b32 %two  : !waveasm.imm<2> -> !waveasm.sreg

  // 3 extra VGPR iter_args for the promoted addresses.
  // CHECK: waveasm.loop
  // CHECK-SAME: !waveasm.vreg
  %r:4 = waveasm.loop(%iv = %init_iv, %s0 = %init_s0, %s1 = %init_s1, %s2 = %init_s2)
      : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
     -> (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg, !waveasm.sreg) {

    // CHECK-NOT: waveasm.v_add_u32 {{.*}} !waveasm.pvreg<0>, !waveasm.sreg
    %addr = waveasm.v_add_u32 %tid, %s0 : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg

    // CHECK: waveasm.ds_read_b128
    %val = waveasm.ds_read_b128 %addr : !waveasm.vreg -> !waveasm.vreg<4, 4>

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    // Rotation: s0->s1->s2->s0.
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %s1, %s2, %s0) : !waveasm.sreg, !waveasm.sreg, !waveasm.sreg, !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- Two independent LDS reads in same rotation group ----
// Both V_ADD_U32 ops use different loop-invariant VGPRs but the same rotating
// SGPR. Both should be promoted independently.

// CHECK-LABEL: @two_reads_same_group
waveasm.program @two_reads_same_group
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 8 : !waveasm.imm<8>

  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %base = waveasm.v_mov_b32 %one : !waveasm.imm<1> -> !waveasm.vreg
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s0 = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s1 = waveasm.s_mov_b32 %one  : !waveasm.imm<1> -> !waveasm.sreg

  // Should gain 2*2=4 extra VGPR iter_args (2 per promotable add, group size 2).
  // CHECK: waveasm.loop
  // CHECK-SAME: !waveasm.vreg
  %r:3 = waveasm.loop(%iv = %init_iv, %s0 = %init_s0, %s1 = %init_s1)
      : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
     -> (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg) {

    // First read: v_add(tid, s0).
    // CHECK-NOT: waveasm.v_add_u32 {{.*}} !waveasm.pvreg<0>, !waveasm.sreg
    %addr0 = waveasm.v_add_u32 %tid, %s0 : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    // CHECK: waveasm.ds_read_b128
    %val0 = waveasm.ds_read_b128 %addr0 : !waveasm.vreg -> !waveasm.vreg<4, 4>

    // Second read: v_add(base, s0) — same rotation group, different invariant VGPR.
    // CHECK-NOT: waveasm.v_add_u32 {{.*}} !waveasm.vreg, !waveasm.sreg
    %addr1 = waveasm.v_add_u32 %base, %s0 : !waveasm.vreg, !waveasm.sreg -> !waveasm.vreg
    // CHECK: waveasm.ds_read_b32
    %val1 = waveasm.ds_read_b32 %addr1 : !waveasm.vreg -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %s1, %s0) : !waveasm.sreg, !waveasm.sreg, !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- Operand order: invariant is src1 ----
// v_add_u32(s0, tid) — rotating SGPR is src0, invariant VGPR is src1.
// The pass should handle both orderings.

// CHECK-LABEL: @operand_order_reversed
waveasm.program @operand_order_reversed
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 8 : !waveasm.imm<8>

  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s0 = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_s1 = waveasm.s_mov_b32 %one  : !waveasm.imm<1> -> !waveasm.sreg

  // CHECK: waveasm.loop
  // CHECK-SAME: !waveasm.vreg
  %r:3 = waveasm.loop(%iv = %init_iv, %s0 = %init_s0, %s1 = %init_s1)
      : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
     -> (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg) {

    // Reversed operand order: s0 first, tid second.
    // CHECK-NOT: waveasm.v_add_u32 {{.*}} !waveasm.sreg, !waveasm.pvreg<0>
    %addr = waveasm.v_add_u32 %s0, %tid : !waveasm.sreg, !waveasm.pvreg<0> -> !waveasm.vreg

    // CHECK: waveasm.ds_read_b64
    %val = waveasm.ds_read_b64 %addr : !waveasm.vreg -> !waveasm.vreg<2>

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %s1, %s0) : !waveasm.sreg, !waveasm.sreg, !waveasm.sreg
  }

  waveasm.s_endpgm
}
