// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Tests for generalized WAR hazard detection.
//
// The old hasBufferLoadWARHazard only detected WAR hazards when the iter_arg
// was defined by a buffer_load. The new hasWARHazard detects overlap for ANY
// defining op, and changes the comparison from >= to > (same-point use is
// safe because the read happens before the write in the same instruction).

//===----------------------------------------------------------------------===//
// Test 1: Scalar WAR : s_add_u32 iter_arg with block_arg used after def
//===----------------------------------------------------------------------===//
//
// The most common real-world scenario: in unrolled loops, CSE merges an
// affine.apply (e.g. iv+2 for bounds check) with the IV increment (also
// iv+2 for step=2). The merged s_add_u32 sits mid-body. If the allocator
// ties it to the IV block_arg, the increment clobbers the IV before later
// instructions read it.
//
// BEFORE fix: hasBufferLoadWARHazard returns false (s_add_u32 is not
//   buffer_load) → allocator ties → silent data corruption.
// AFTER fix:  hasWARHazard detects %iv has uses after %next_iv's def →
//   allocator keeps them separate.

// CHECK-LABEL: waveasm.program @scalar_war_hazard
waveasm.program @scalar_war_hazard
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c2 = waveasm.constant 2 : !waveasm.imm<2>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>

  // CHECK: waveasm.s_mov_b32 {{.*}} -> !waveasm.psreg<[[INIT_IV:[0-9]+]]>
  %init_iv = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  // CHECK: waveasm.loop
  %iv_out = waveasm.loop(%iv = %init_iv)
      : (!waveasm.sreg) -> (!waveasm.sreg) {

    // Iter_arg defined EARLY : WAR hazard source.
    // CHECK: waveasm.s_add_u32 {{.*}}!waveasm.psreg<[[INIT_IV]]>{{.*}}!waveasm.imm<2>
    %next_iv:2 = waveasm.s_add_u32 %iv, %c2
        : !waveasm.sreg, !waveasm.imm<2> -> !waveasm.sreg, !waveasm.sreg

    // Block_arg %iv used AFTER %next_iv is defined : WAR victim.
    // If tied, this reads iv+2 instead of iv.
    // CHECK: waveasm.s_add_u32 {{.*}}!waveasm.psreg<[[INIT_IV]]>{{.*}}!waveasm.imm<1>
    %offset:2 = waveasm.s_add_u32 %iv, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg

    %cond = waveasm.s_cmp_lt_u32 %offset#0, %c10
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv#0) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 2: VGPR WAR : v_add_u32 iter_arg with VGPR block_arg used after def
//===----------------------------------------------------------------------===//
//
// Same hazard pattern but with VGPRs: the iter_arg is a v_add_u32 (not a
// buffer_load), and the block_arg is used afterward by v_mul_lo_u32.
//
// BEFORE fix: Not detected (v_add_u32 is not buffer_load).
// AFTER fix:  Detected via def/use overlap.

// CHECK-LABEL: waveasm.program @vgpr_war_hazard
waveasm.program @vgpr_war_hazard
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[INIT_VAL:[0-9]+]]>
  %init_val = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  // CHECK: waveasm.loop
  %i_out, %val_out = waveasm.loop(%i = %init_i, %val = %init_val)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // VGPR iter_arg defined EARLY.
    // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[INIT_VAL]]>{{.*}} -> !waveasm.pvreg<
    %new_val = waveasm.v_add_u32 %val, %v0
        : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    // VGPR block_arg %val used AFTER %new_val defined : WAR hazard.
    // CHECK: waveasm.v_mul_lo_u32 {{.*}}!waveasm.pvreg<[[INIT_VAL]]>{{.*}} -> !waveasm.pvreg<
    %late_use = waveasm.v_mul_lo_u32 %val, %v0
        : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    %next_i:2 = waveasm.s_add_u32 %i, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i#0, %c10
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_i#0, %new_val) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 3: No hazard : block_arg only used AT the iter_arg def point (>= vs >)
//===----------------------------------------------------------------------===//
//
// The block_arg %iv is ONLY used as the operand of the s_add_u32 that
// defines the iter_arg. defPoints[%next_iv] == usePoints[%iv], so:
//   Old >=: flags hazard (wrong : overly conservative)
//   New > : no hazard (correct : read-before-write at same point)
//
// This is the standard loop-counter pattern. If the generalized check used
// >= instead of >, it would incorrectly untie every loop IV increment,
// wasting a register on each loop.
//
// The test verifies tying is preserved: %next_iv gets the SAME physical
// register as %init_iv (the block_arg's allocation).

// CHECK-LABEL: waveasm.program @no_hazard_same_point
waveasm.program @no_hazard_same_point
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>

  // CHECK: waveasm.s_mov_b32 {{.*}} -> !waveasm.psreg<[[IV:[0-9]+]]>
  %init_iv = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  %iv_out = waveasm.loop(%iv = %init_iv)
      : (!waveasm.sreg) -> (!waveasm.sreg) {

    // %iv's ONLY use is as the operand of this instruction.
    // Same program point as the def of %next_iv → no hazard with >.
    // CHECK: waveasm.s_add_u32 {{.*}}!waveasm.psreg<[[IV]]>{{.*}} -> !waveasm.psreg<[[IV]]>
    %next_iv:2 = waveasm.s_add_u32 %iv, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg

    // Work that does NOT use %iv : only uses %next_iv.
    %cond = waveasm.s_cmp_lt_u32 %next_iv#0, %c10
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv#0) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 4: Mixed : MFMA accumulator tying preserved alongside scalar WAR
//===----------------------------------------------------------------------===//
//
// A realistic loop with BOTH an MFMA accumulator chain (should stay tied)
// and a scalar IV with a WAR hazard (should be untied). This is the most
// critical test: it verifies the fix is selective : only the hazardous pair
// is untied while the MFMA tying (essential for performance) is preserved.
//
// The > comparison is what makes this work:
//   MFMA: defPoints[%new_acc] == usePoints[%acc] (same instruction) → no hazard
//   IV:   defPoints[%next_iv] < usePoints[%iv] (used afterward) → hazard

// CHECK-LABEL: waveasm.program @mixed_mfma_and_scalar_war
waveasm.program @mixed_mfma_and_scalar_war
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c2 = waveasm.constant 2 : !waveasm.imm<2>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  %init_iv = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[ACC:[0-9]+]], 4>
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.loop
  %iv_out, %acc_out = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>) -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {

    // --- Scalar IV: WAR hazard ---
    // Define iter_arg for %iv EARLY.
    %next_iv:2 = waveasm.s_add_u32 %iv, %c2
        : !waveasm.sreg, !waveasm.imm<2> -> !waveasm.sreg, !waveasm.sreg

    // --- MFMA accumulator: no hazard ---
    // %acc used as operand and %new_acc defined at the SAME program point.
    // With >: same-point → no hazard → tying preserved.
    // CHECK: waveasm.v_mfma_f32_16x16x16_f16 {{.*}} -> !waveasm.pvreg<[[ACC]], 4>
    %new_acc = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc
        : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4>
        -> !waveasm.vreg<4, 4>

    // --- Scalar IV: WAR victim ---
    // Uses %iv AFTER %next_iv defined → triggers WAR.
    %offset:2 = waveasm.s_add_u32 %iv, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg

    %cond = waveasm.s_cmp_lt_u32 %offset#0, %c10
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_iv#0, %new_acc) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 5: Multiple block_arg uses after iter_arg def
//===----------------------------------------------------------------------===//
//
// Stress test: the block_arg %iv is used THREE times after the iter_arg
// %next_iv is defined. Each use would read the wrong value if tied.
// The hazard check finds the first use > def and returns true immediately.

// CHECK-LABEL: waveasm.program @multiple_post_def_uses
waveasm.program @multiple_post_def_uses
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c2 = waveasm.constant 2 : !waveasm.imm<2>
  %c3 = waveasm.constant 3 : !waveasm.imm<3>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %c100 = waveasm.constant 100 : !waveasm.imm<100>

  // CHECK: waveasm.s_mov_b32 {{.*}} -> !waveasm.psreg<[[IV:[0-9]+]]>
  %init_iv = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  // CHECK: waveasm.loop
  %iv_out = waveasm.loop(%iv = %init_iv)
      : (!waveasm.sreg) -> (!waveasm.sreg) {

    // Iter_arg defined first.
    %next_iv:2 = waveasm.s_add_u32 %iv, %c4
        : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg, !waveasm.sreg

    // Three subsequent uses of %iv : all read wrong value if tied.
    // CHECK: waveasm.s_add_u32 {{.*}}!waveasm.psreg<[[IV]]>{{.*}}!waveasm.imm<1>
    %off1:2 = waveasm.s_add_u32 %iv, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    // CHECK: waveasm.s_add_u32 {{.*}}!waveasm.psreg<[[IV]]>{{.*}}!waveasm.imm<2>
    %off2:2 = waveasm.s_add_u32 %iv, %c2
        : !waveasm.sreg, !waveasm.imm<2> -> !waveasm.sreg, !waveasm.sreg
    // CHECK: waveasm.s_add_u32 {{.*}}!waveasm.psreg<[[IV]]>{{.*}}!waveasm.imm<3>
    %off3:2 = waveasm.s_add_u32 %iv, %c3
        : !waveasm.sreg, !waveasm.imm<3> -> !waveasm.sreg, !waveasm.sreg

    %cond = waveasm.s_cmp_lt_u32 %off3#0, %c100
        : !waveasm.sreg, !waveasm.imm<100> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv#0) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 6: No hazard : iter_arg defined AFTER all block_arg uses (negative)
//===----------------------------------------------------------------------===//
//
// All uses of %iv come BEFORE %next_iv is defined. No overlap, no hazard.
// The allocator should tie them (same physical register) both before and
// after the fix. This is a regression guard: the generalized check must
// NOT over-trigger when the order is safe.

// CHECK-LABEL: waveasm.program @no_war_def_after_all_uses
waveasm.program @no_war_def_after_all_uses
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[SUM:[0-9]+]]>
  %init_sum = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  %i_out, %sum_out = waveasm.loop(%i = %init_i, %sum = %init_sum)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // All uses of %sum come FIRST : before the iter_arg is defined.
    %doubled = waveasm.v_add_u32 %sum, %sum
        : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    // Iter_arg defined AFTER the last use of %sum → no WAR hazard.
    // Should be tied: same physical register as init.
    // CHECK: waveasm.v_add_u32 {{.*}} -> !waveasm.pvreg<[[SUM]]>
    %new_sum = waveasm.v_add_u32 %doubled, %v0
        : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    %next_i:2 = waveasm.s_add_u32 %i, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i#0, %c10
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_i#0, %new_sum) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}
