// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: Scratch VGPR literal cache eliminates redundant v_mov_b32
// materializations when the same non-inline constant is used consecutively.
// The cache must be invalidated at control flow merge points (labels, loop
// back-edges, if/else joins) and when kScratchVGPR is written with a
// non-literal value.

// -----

// Test 1: Consecutive v_cndmask_b32 with the same literal in src0.
// The second materialization should be elided.
// CHECK-LABEL: cndmask_same_literal:
waveasm.program @cndmask_same_literal
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v2 = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>
  %c999 = waveasm.constant 999 : !waveasm.imm<999>

  // First use: must materialize.
  // CHECK:     v_mov_b32 v15, 999
  // CHECK:     v_cndmask_b32 [[R1:v[0-9]+]], v15, v0
  %r1 = waveasm.v_cndmask_b32 %c999, %v0, %vcc : !waveasm.imm<999>, !waveasm.pvreg<0>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // Second use with same literal: should NOT rematerialize.
  // CHECK-NOT: v_mov_b32 v15
  // CHECK:     v_cndmask_b32 [[R2:v[0-9]+]], v15, v1
  %r2 = waveasm.v_cndmask_b32 %c999, %v1, %vcc : !waveasm.imm<999>, !waveasm.pvreg<1>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // Third use, still same literal: should NOT rematerialize.
  // CHECK-NOT: v_mov_b32 v15
  // CHECK:     v_cndmask_b32 {{v[0-9]+}}, v15, v2
  %r3 = waveasm.v_cndmask_b32 %c999, %v2, %vcc : !waveasm.imm<999>, !waveasm.pvreg<2>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// -----

// Test 2: Different literal values must rematerialize.
// CHECK-LABEL: cndmask_different_literals:
waveasm.program @cndmask_different_literals
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>
  %c100 = waveasm.constant 100 : !waveasm.imm<100>
  %c200 = waveasm.constant 200 : !waveasm.imm<200>

  // First literal.
  // CHECK:     v_mov_b32 v15, 100
  // CHECK:     v_cndmask_b32 {{v[0-9]+}}, v15, v0
  %r1 = waveasm.v_cndmask_b32 %c100, %v0, %vcc : !waveasm.imm<100>, !waveasm.pvreg<0>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // Different literal: must rematerialize.
  // CHECK:     v_mov_b32 v15, 200
  // CHECK:     v_cndmask_b32 {{v[0-9]+}}, v15, v1
  %r2 = waveasm.v_cndmask_b32 %c200, %v1, %vcc : !waveasm.imm<200>, !waveasm.pvreg<1>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// -----

// Test 3: Cache invalidated at loop back-edge.
// The literal must be rematerialized at the top of each iteration because
// the back-edge merges two control flow paths.
// CHECK-LABEL: loop_invalidates_cache:
waveasm.program @loop_invalidates_cache
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c777 = waveasm.constant 777 : !waveasm.imm<777>

  %init = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  // Pre-loop materialization.
  // CHECK:     v_mov_b32 v15, 777
  // CHECK:     v_cndmask_b32 {{v[0-9]+}}, v15, v0
  %pre = waveasm.v_cndmask_b32 %c777, %v0, %vcc : !waveasm.imm<777>, !waveasm.pvreg<0>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // CHECK: L_loop_0:
  // Inside the loop: cache was invalidated at the label, so must rematerialize.
  // CHECK:     v_mov_b32 v15, 777
  // CHECK:     v_cndmask_b32 {{v[0-9]+}}, v15, v1
  %out = waveasm.loop(%iv = %init) : (!waveasm.vreg) -> (!waveasm.vreg) {
    %body = waveasm.v_cndmask_b32 %c777, %v1, %vcc : !waveasm.imm<777>, !waveasm.pvreg<1>, !waveasm.psreg<106, 2> -> !waveasm.vreg
    %next = waveasm.v_add_u32 %iv, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %cond_s = waveasm.v_readfirstlane_b32 %next : !waveasm.vreg -> !waveasm.sreg
    %ub = waveasm.v_readfirstlane_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %cond_s, %ub : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next) : !waveasm.vreg
  }

  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// -----

// Test 4: Cache survives within a straight-line loop body.
// Two consecutive uses of the same literal inside the loop: second is elided.
// CHECK-LABEL: loop_body_cache_reuse:
waveasm.program @loop_body_cache_reuse
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c888 = waveasm.constant 888 : !waveasm.imm<888>

  %init = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  // CHECK: L_loop_0:
  // First use in loop body: materialized (cache was invalidated at label).
  // CHECK:     v_mov_b32 v15, 888
  // CHECK:     v_cndmask_b32 {{v[0-9]+}}, v15, v0
  // Second use in same iteration: cache hit, no rematerialization.
  // CHECK-NOT: v_mov_b32 v15
  // CHECK:     v_cndmask_b32 {{v[0-9]+}}, v15, v1
  %out = waveasm.loop(%iv = %init) : (!waveasm.vreg) -> (!waveasm.vreg) {
    %a = waveasm.v_cndmask_b32 %c888, %v0, %vcc : !waveasm.imm<888>, !waveasm.pvreg<0>, !waveasm.psreg<106, 2> -> !waveasm.vreg
    %b = waveasm.v_cndmask_b32 %c888, %v1, %vcc : !waveasm.imm<888>, !waveasm.pvreg<1>, !waveasm.psreg<106, 2> -> !waveasm.vreg
    %next = waveasm.v_add_u32 %iv, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %cond_s = waveasm.v_readfirstlane_b32 %next : !waveasm.vreg -> !waveasm.sreg
    %ub = waveasm.v_readfirstlane_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %cond_s, %ub : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next) : !waveasm.vreg
  }

  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// -----

// Test 5: Cache invalidated at if/else merge point.
// CHECK-LABEL: if_else_invalidates_cache:
waveasm.program @if_else_invalidates_cache
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>
  %c555 = waveasm.constant 555 : !waveasm.imm<555>

  // Materialize in the then block.
  // CHECK:     v_mov_b32 v15, 555
  // CHECK:     v_cndmask_b32
  // The else block must also materialize (entered from the branch, not then).
  // CHECK:     v_mov_b32 v15, 555
  // CHECK:     v_cndmask_b32
  waveasm.if %s0 : !waveasm.psreg<0> {
    %t = waveasm.v_cndmask_b32 %c555, %v0, %vcc : !waveasm.imm<555>, !waveasm.pvreg<0>, !waveasm.psreg<106, 2> -> !waveasm.vreg
    waveasm.yield
  } else {
    %e = waveasm.v_cndmask_b32 %c555, %v1, %vcc : !waveasm.imm<555>, !waveasm.pvreg<1>, !waveasm.psreg<106, 2> -> !waveasm.vreg
    waveasm.yield
  }

  // After merge: cache was invalidated, must rematerialize.
  // CHECK:     v_mov_b32 v15, 555
  // CHECK:     v_cndmask_b32
  %post = waveasm.v_cndmask_b32 %c555, %v0, %vcc : !waveasm.imm<555>, !waveasm.pvreg<0>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// -----

// Test 6: Cache works across different instruction types.
// A literal materialized for v_cndmask_b32 can be reused by v_cmp (via
// materializeLiteralOperand) and vice versa.
// CHECK-LABEL: cross_instruction_cache:
waveasm.program @cross_instruction_cache
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>
  %c4096 = waveasm.constant 4096 : !waveasm.imm<4096>

  // v_cmp with literal: materialize.
  // CHECK:     v_mov_b32 v15, 4096
  // CHECK:     v_cmp_lt_u32 v0, v15
  waveasm.v_cmp_lt_u32 %v0, %c4096 : !waveasm.pvreg<0>, !waveasm.imm<4096>

  // v_cndmask with same literal: reuse from cache.
  // CHECK-NOT: v_mov_b32 v15
  // CHECK:     v_cndmask_b32 {{v[0-9]+}}, v15, v1
  %r = waveasm.v_cndmask_b32 %c4096, %v1, %vcc : !waveasm.imm<4096>, !waveasm.pvreg<1>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
