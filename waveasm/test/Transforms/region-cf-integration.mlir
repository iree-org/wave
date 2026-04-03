// RUN: waveasm-translate %s | FileCheck %s
//
// Integration test: verifies that already-formed WaveASM region-based control
// flow survives a round-trip through waveasm-translate with correct SSA
// threading, iter_arg types, and structural fidelity.

//===----------------------------------------------------------------------===//
// LoopOp integration
//===----------------------------------------------------------------------===//

// --- Basic loop: single sreg iter_arg ---
// CHECK-LABEL: @test_loop_structure
waveasm.program @test_loop_structure
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 16 : i64, sgprs = 16 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %ten = waveasm.constant 10 : !waveasm.imm<10>
  %init = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // CHECK:      %[[INIT:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK-NEXT: %{{.*}} = waveasm.loop (%[[IV:.*]] = %[[INIT]]) : (!waveasm.sreg) -> !waveasm.sreg {
  %counter = waveasm.loop(%i = %init) : (!waveasm.sreg) -> (!waveasm.sreg) {
    // CHECK-NEXT:   %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %next:2 = waveasm.s_add_u32 %i, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    // CHECK-NEXT:   %[[COND:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next#0, %ten : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    // CHECK-NEXT:   waveasm.condition %[[COND]] : !waveasm.sreg iter_args(%[[NEXT]]) : !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next#0) : !waveasm.sreg
  // CHECK-NEXT: }
  }

  waveasm.s_endpgm
}

// --- Loop with accumulator: sreg + vreg iter_args ---
// CHECK-LABEL: @test_loop_accumulator
waveasm.program @test_loop_accumulator
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 16 : i64, sgprs = 16 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 16 : !waveasm.imm<16>
  %init_i = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_sum = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // CHECK:      %[[I0:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK-NEXT: %[[S0:.*]] = waveasm.v_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.vreg
  // CHECK-NEXT: %{{.*}}:2 = waveasm.loop (%[[IV:.*]] = %[[I0]], %[[SUM:.*]] = %[[S0]]) : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
  %i_final, %result = waveasm.loop(%i = %init_i, %sum = %init_sum)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
    // CHECK-NEXT:   %[[NEWSUM:.*]] = waveasm.v_add_u32 %[[SUM]], %[[IV]] : !waveasm.vreg, !waveasm.sreg -> !waveasm.vreg
    %new_sum = waveasm.v_add_u32 %sum, %i
        : !waveasm.vreg, !waveasm.sreg -> !waveasm.vreg
    // CHECK-NEXT:   %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %next:2 = waveasm.s_add_u32 %i, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    // CHECK-NEXT:   %[[COND:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<16> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next#0, %limit : !waveasm.sreg, !waveasm.imm<16> -> !waveasm.sreg
    // CHECK-NEXT:   waveasm.condition %[[COND]] : !waveasm.sreg iter_args(%[[NEXT]], %[[NEWSUM]]) : !waveasm.sreg, !waveasm.vreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next#0, %new_sum) : !waveasm.sreg, !waveasm.vreg
  // CHECK-NEXT: }
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// IfOp integration
//===----------------------------------------------------------------------===//

// --- If-then-else producing vreg ---
// CHECK-LABEL: @test_if_then_else
waveasm.program @test_if_then_else
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 16 : i64, sgprs = 16 : i64} {

  %tid = waveasm.precolored.sreg 0 : !waveasm.sreg
  %a = waveasm.precolored.vreg 0 : !waveasm.vreg
  %b = waveasm.precolored.vreg 1 : !waveasm.vreg
  %threshold = waveasm.constant 100 : !waveasm.imm<100>

  // CHECK:      %[[CMP:.*]] = waveasm.s_cmp_lt_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<100> -> !waveasm.sreg
  %cmp = waveasm.s_cmp_lt_u32 %tid, %threshold : !waveasm.sreg, !waveasm.imm<100> -> !waveasm.sreg

  // CHECK-NEXT: %{{.*}} = waveasm.if %[[CMP]] : !waveasm.sreg -> !waveasm.vreg {
  %result = waveasm.if %cmp : !waveasm.sreg -> !waveasm.vreg {
    // CHECK-NEXT:   %{{.*}} = waveasm.v_add_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %add = waveasm.v_add_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    // CHECK-NEXT:   waveasm.yield %{{.*}} : !waveasm.vreg
    waveasm.yield %add : !waveasm.vreg
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT:   %{{.*}} = waveasm.v_sub_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %sub = waveasm.v_sub_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    // CHECK-NEXT:   waveasm.yield %{{.*}} : !waveasm.vreg
    waveasm.yield %sub : !waveasm.vreg
  // CHECK-NEXT: }
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Nested control flow integration
//===----------------------------------------------------------------------===//

// --- Nested loops: accumulator threads from outer to inner and back ---
// CHECK-LABEL: @test_nested_loops
waveasm.program @test_nested_loops
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %outer_lim = waveasm.constant 4 : !waveasm.imm<4>
  %inner_lim = waveasm.constant 8 : !waveasm.imm<8>
  %init_i = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // CHECK:      %{{.*}}:2 = waveasm.loop (%[[OI:.*]] = %{{.*}}, %[[OACC:.*]] = %{{.*}}) : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
  %i_final, %outer_result = waveasm.loop(%i = %init_i, %acc_outer = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %init_j = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

    // Inner loop inherits accumulator from outer block arg
    // CHECK:      %[[INNER:.*]]:2 = waveasm.loop (%{{.*}} = %{{.*}}, %{{.*}} = %[[OACC]]) : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
    %j_final, %inner_result = waveasm.loop(%j = %init_j, %acc_inner = %acc_outer)
        : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
      // Inner body: multiply outer * inner counters, accumulate
      // CHECK:      waveasm.v_mul_lo_u32 %[[OI]], %{{.*}} : !waveasm.sreg, !waveasm.sreg -> !waveasm.vreg
      %product = waveasm.v_mul_lo_u32 %i, %j
          : !waveasm.sreg, !waveasm.sreg -> !waveasm.vreg
      %new_acc = waveasm.v_add_u32 %acc_inner, %product
          : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
      %next_j:2 = waveasm.s_add_u32 %j, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
      %cond_j = waveasm.s_cmp_lt_u32 %next_j#0, %inner_lim : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
      waveasm.condition %cond_j : !waveasm.sreg iter_args(%next_j#0, %new_acc) : !waveasm.sreg, !waveasm.vreg
    }

    %next_i:2 = waveasm.s_add_u32 %i, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cond_i = waveasm.s_cmp_lt_u32 %next_i#0, %outer_lim : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg

    // Outer condition: inner loop's vreg result (#1) becomes outer accumulator
    // CHECK:      waveasm.condition %{{.*}} : !waveasm.sreg iter_args(%{{.*}}, %[[INNER]]#1) : !waveasm.sreg, !waveasm.vreg
    waveasm.condition %cond_i : !waveasm.sreg iter_args(%next_i#0, %inner_result) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// --- If inside a loop ---
// CHECK-LABEL: @test_if_in_loop
waveasm.program @test_if_in_loop
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 16 : i64, sgprs = 16 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %two = waveasm.constant 2 : !waveasm.imm<2>
  %limit = waveasm.constant 10 : !waveasm.imm<10>
  %init = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // CHECK:      %{{.*}} = waveasm.loop (%[[IV:.*]] = %{{.*}}) : (!waveasm.sreg) -> !waveasm.sreg {
  %final = waveasm.loop(%i = %init) : (!waveasm.sreg) -> (!waveasm.sreg) {
    // CHECK:      %[[REM:.*]] = waveasm.s_and_b32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %rem = waveasm.s_and_b32 %i, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    // CHECK-NEXT: %[[EVEN:.*]] = waveasm.s_cmp_eq_u32 %[[REM]], %{{.*}} : !waveasm.sreg, !waveasm.imm<0> -> !waveasm.sreg
    %is_even = waveasm.s_cmp_eq_u32 %rem, %zero : !waveasm.sreg, !waveasm.imm<0> -> !waveasm.sreg

    // CHECK-NEXT: %[[STEP:.*]] = waveasm.if %[[EVEN]] : !waveasm.sreg -> !waveasm.sreg {
    %step = waveasm.if %is_even : !waveasm.sreg -> !waveasm.sreg {
      // CHECK:      waveasm.yield %{{.*}} : !waveasm.sreg
      %step_val = waveasm.s_mov_b32 %two : !waveasm.imm<2> -> !waveasm.sreg
      waveasm.yield %step_val : !waveasm.sreg
    // CHECK:      } else {
    } else {
      // CHECK:      waveasm.yield %{{.*}} : !waveasm.sreg
      %step_val = waveasm.s_mov_b32 %one : !waveasm.imm<1> -> !waveasm.sreg
      waveasm.yield %step_val : !waveasm.sreg
    }

    // If result feeds loop counter update
    // CHECK:      %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %[[IV]], %[[STEP]] : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg, !waveasm.sreg
    %next:2 = waveasm.s_add_u32 %i, %step : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg, !waveasm.sreg
    // CHECK-NEXT: %[[CONT:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    %cont = waveasm.s_cmp_lt_u32 %next#0, %limit : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    // CHECK-NEXT: waveasm.condition %[[CONT]] : !waveasm.sreg iter_args(%[[NEXT]]) : !waveasm.sreg
    waveasm.condition %cont : !waveasm.sreg iter_args(%next#0) : !waveasm.sreg
  }

  waveasm.s_endpgm
}
