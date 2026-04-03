// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Emit-assembly tests for generalized WAR hazard detection.
//
// When the allocator untied an iter_arg from its block_arg (WAR hazard), the
// emitter inserts s_mov_b32/v_mov_b32 back-edge copies before s_cbranch_scc1.
// When tied (no hazard), no copy is needed : the iter_arg already lives in
// the block_arg register.
//
// Tests A and B FAIL on main (no copy : iter_arg incorrectly tied to block_arg).
// Test C PASSES on both (no hazard : iter_arg correctly tied, no copy).
// Test D FAILS on main (no scalar copy : all incorrectly tied).

//===----------------------------------------------------------------------===//
// Test A: Scalar WAR -> s_mov_b32 back-edge copy
//===----------------------------------------------------------------------===//
//
// s_add_u32 iter_arg defined mid-body, block_arg %iv used afterward.
// After fix: untied -> emitter inserts s_mov_b32 before s_cbranch_scc1.
// On main:   tied   -> no copy -> CHECK for s_mov_b32 FAILS.

// CHECK-LABEL: emit_scalar_war_copy:
waveasm.program @emit_scalar_war_copy
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c2 = waveasm.constant 2 : !waveasm.imm<2>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>

  %init_iv = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  %iv_out = waveasm.loop(%iv = %init_iv)
      : (!waveasm.sreg) -> (!waveasm.sreg) {

    // Iter_arg defined EARLY : WAR hazard source.
    %next_iv:2 = waveasm.s_add_u32 %iv, %c2
        : !waveasm.sreg, !waveasm.imm<2> -> !waveasm.sreg, !waveasm.scc

    // Block_arg %iv used AFTER %next_iv defined.
    %offset:2 = waveasm.s_add_u32 %iv, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc

    %cond = waveasm.s_cmp_lt_u32 %offset#0, %c10
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.scc
    waveasm.condition %cond : !waveasm.scc iter_args(%next_iv#0) : !waveasm.sreg
  }

  // CHECK:      s_cmp_lt_u32
  // Back-edge copy: iter_arg register -> block_arg register (untied).
  // On main this copy is absent because both share the same register.
  // CHECK:      s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
  // CHECK-NEXT: s_cbranch_scc1

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test B: VGPR WAR -> v_mov_b32 back-edge copy
//===----------------------------------------------------------------------===//
//
// v_add_u32 iter_arg defined early, VGPR block_arg %val used afterward.
// After fix: untied -> emitter inserts v_mov_b32 before s_cbranch_scc1.
// On main:   tied   -> no copy -> CHECK for v_mov_b32 FAILS.

// CHECK-LABEL: emit_vgpr_war_copy:
waveasm.program @emit_vgpr_war_copy
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_val = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  %i_out, %val_out = waveasm.loop(%i = %init_i, %val = %init_val)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // VGPR iter_arg defined EARLY : WAR hazard source.
    %new_val = waveasm.v_add_u32 %val, %v0
        : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    // VGPR block_arg %val used AFTER %new_val defined.
    %late_use = waveasm.v_mul_lo_u32 %val, %v0
        : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    %next_i:2 = waveasm.s_add_u32 %i, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
    %cond = waveasm.s_cmp_lt_u32 %next_i#0, %c10
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.scc
    waveasm.condition %cond : !waveasm.scc
        iter_args(%next_i#0, %new_val) : !waveasm.sreg, !waveasm.vreg
  }

  // CHECK:      s_cmp_lt_u32
  // Back-edge copy: VGPR iter_arg register -> block_arg register (untied).
  // On main this copy is absent because both share the same register.
  // CHECK:      v_mov_b32 v{{[0-9]+}}, v{{[0-9]+}}
  // CHECK-NEXT: s_cbranch_scc1

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test C: No hazard - no back-edge copy (passes on both main and fix)
//===----------------------------------------------------------------------===//
//
// Block_arg %iv is ONLY used as the operand of the s_add_u32 that defines
// the iter_arg. Same program point: with > no hazard - tied - no copy.
// This also passes on main (buffer_load filter skips s_add_u32 - tied).
//
// Regression guard: the generalized check must NOT untie every loop counter.

// CHECK-LABEL: emit_no_copy_same_point:
waveasm.program @emit_no_copy_same_point
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>

  %init_iv = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  %iv_out = waveasm.loop(%iv = %init_iv)
      : (!waveasm.sreg) -> (!waveasm.sreg) {

    // %iv's ONLY use is as the operand of this instruction.
    // Same point as the def of %next_iv -> says no hazard - tied.
    %next_iv:2 = waveasm.s_add_u32 %iv, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc

    %cond = waveasm.s_cmp_lt_u32 %next_iv#0, %c10
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.scc
    waveasm.condition %cond : !waveasm.scc iter_args(%next_iv#0) : !waveasm.sreg
  }

  // No back-edge copy: iter_arg and block_arg share the same register.
  // CHECK:      s_cmp_lt_u32
  // CHECK-NOT:  s_mov_b32 s{{[0-9]+}}, s
  // CHECK-NOT:  v_mov_b32
  // CHECK:      s_cbranch_scc1

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test D: Mixed MFMA + scalar : selective back-edge copy
//===----------------------------------------------------------------------===//
//
// MFMA accumulator iter_arg: same-point def/use -> says no hazard -> tied.
// Scalar IV iter_arg: block_arg used AFTER def -> hazard -> untied.
//
// After fix: ONLY the scalar iter_arg gets an s_mov_b32 copy.
//            The MFMA accumulator has NO v_mov_b32 copies (4-wide tied).
// On main:   BOTH are tied -> no copies at all -> s_mov_b32 CHECK FAILS.
//
// This is the most critical test: it proves the fix is selective
// it catches the scalar WAR while preserving MFMA tying.

// CHECK-LABEL: emit_mixed_selective:
waveasm.program @emit_mixed_selective
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c2 = waveasm.constant 2 : !waveasm.imm<2>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  %init_iv = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %iv_out, %acc_out = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>) -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {

    // Scalar IV iter_arg defined EARLY : WAR hazard source.
    %next_iv:2 = waveasm.s_add_u32 %iv, %c2
        : !waveasm.sreg, !waveasm.imm<2> -> !waveasm.sreg, !waveasm.scc

    // MFMA: %acc used and %new_acc defined at SAME point -> no hazard.
    %new_acc = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc
        : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4>
        -> !waveasm.vreg<4, 4>

    // Block_arg %iv used AFTER %next_iv defined : WAR victim.
    %offset:2 = waveasm.s_add_u32 %iv, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc

    %cond = waveasm.s_cmp_lt_u32 %offset#0, %c10
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.scc
    waveasm.condition %cond : !waveasm.scc
        iter_args(%next_iv#0, %new_acc) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  // CHECK:      s_cmp_lt_u32
  // No VGPR copies : MFMA accumulator is correctly tied (same-point, > check).
  // CHECK-NOT:  v_mov_b32
  // Only scalar copy : WAR hazard untied the IV iter_arg from its block_arg.
  // On main this copy is absent because the scalar iter_arg is also tied.
  // CHECK:      s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
  // CHECK-NEXT: s_cbranch_scc1

  waveasm.s_endpgm
}
