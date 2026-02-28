// RUN: waveasm-translate %s | waveasm-translate | FileCheck %s
// RUN: waveasm-translate %s
//
// Round-trip (parse -> print -> parse) tests for region-based control flow ops.
// Verifies that LoopOp, IfOp, ConditionOp, and YieldOp survive round-trip
// with correct types, block arguments, iter_args threading, and SSA structure.

//===----------------------------------------------------------------------===//
// LoopOp Tests
//===----------------------------------------------------------------------===//

// --- Simple loop: single iter_arg (SGPR counter) ---
// Verifies init flows into loop, block arg used in body, condition feeds back.
// CHECK-LABEL: @simple_loop
waveasm.program @simple_loop
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 10 : !waveasm.imm<10>
  %init = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // CHECK:      %[[INIT:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK-NEXT: %{{.*}} = waveasm.loop (%[[IV:.*]] = %[[INIT]]) : (!waveasm.sreg) -> !waveasm.sreg {
  %final = waveasm.loop(%i = %init) : (!waveasm.sreg) -> (!waveasm.sreg) {
    // CHECK-NEXT:   %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %next:2 = waveasm.s_add_u32 %i, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    // CHECK-NEXT:   %[[COND:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next#0, %limit : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    // CHECK-NEXT:   waveasm.condition %[[COND]] : !waveasm.sreg iter_args(%[[NEXT]]) : !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next#0) : !waveasm.sreg
  // CHECK-NEXT: }
  }

  waveasm.s_endpgm
}

// --- Loop with two iter_args: SGPR counter + VGPR accumulator ---
// Verifies multi-result (:2), mixed register types, accumulator uses both
// block args, and condition iter_args feed both back.
// CHECK-LABEL: @loop_with_accumulator
waveasm.program @loop_with_accumulator
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 4 : !waveasm.imm<4>
  %init_i = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // CHECK:      %[[I0:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK-NEXT: %[[A0:.*]] = waveasm.v_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.vreg
  // CHECK-NEXT: %{{.*}}:2 = waveasm.loop (%[[IV:.*]] = %[[I0]], %[[ACC:.*]] = %[[A0]]) : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
  %final_i, %result = waveasm.loop(%i = %init_i, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
    // Accumulator adds vreg block arg + sreg block arg
    // CHECK-NEXT:   %[[NEWACC:.*]] = waveasm.v_add_u32 %[[ACC]], %[[IV]] : !waveasm.vreg, !waveasm.sreg -> !waveasm.vreg
    %new_acc = waveasm.v_add_u32 %acc, %i
        : !waveasm.vreg, !waveasm.sreg -> !waveasm.vreg
    // CHECK-NEXT:   %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %next:2 = waveasm.s_add_u32 %i, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    // CHECK-NEXT:   %[[COND:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next#0, %limit : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
    // CHECK-NEXT:   waveasm.condition %[[COND]] : !waveasm.sreg iter_args(%[[NEXT]], %[[NEWACC]]) : !waveasm.sreg, !waveasm.vreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next#0, %new_acc) : !waveasm.sreg, !waveasm.vreg
  // CHECK-NEXT: }
  }

  waveasm.s_endpgm
}

// --- MFMA accumulation loop with vreg<4> iter_arg ---
// Verifies vector register types round-trip and MFMA reads accumulator
// block arg and feeds its result back through condition iter_args.
// CHECK-LABEL: @loop_mfma_accumulation
waveasm.program @loop_mfma_accumulation
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 64 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %init_k = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg<4>

  // CHECK:      %{{.*}}:2 = waveasm.loop (%[[K:.*]] = %{{.*}}, %[[ACC:.*]] = %{{.*}}) : (!waveasm.sreg, !waveasm.vreg<4>) -> (!waveasm.sreg, !waveasm.vreg<4>) {
  %final_k, %final_acc = waveasm.loop(%k = %init_k, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg<4>) -> (!waveasm.sreg, !waveasm.vreg<4>) {
    %a_tile = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg<2>
    %b_tile = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg<2>

    // MFMA reads accumulator block arg (%ACC) and produces new accumulator
    // CHECK:      %[[MFMA:.*]] = waveasm.v_mfma_f32_16x16x16_f16 %{{.*}}, %{{.*}}, %[[ACC]] : !waveasm.vreg<2>, !waveasm.vreg<2>, !waveasm.vreg<4> -> !waveasm.vreg<4>
    %new_acc = waveasm.v_mfma_f32_16x16x16_f16 %a_tile, %b_tile, %acc
        : !waveasm.vreg<2>, !waveasm.vreg<2>, !waveasm.vreg<4> -> !waveasm.vreg<4>

    %next_k:2 = waveasm.s_add_u32 %k, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %continue = waveasm.s_cmp_lt_u32 %next_k#0, %four : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg

    // MFMA result fed back as accumulator iter_arg (vreg<4>)
    // CHECK:      waveasm.condition %{{.*}} : !waveasm.sreg iter_args(%{{.*}}, %[[MFMA]]) : !waveasm.sreg, !waveasm.vreg<4>
    waveasm.condition %continue : !waveasm.sreg iter_args(%next_k#0, %new_acc) : !waveasm.sreg, !waveasm.vreg<4>
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// IfOp Tests
//===----------------------------------------------------------------------===//

// --- If-then-else with single vreg result ---
// Verifies condition operand flows into if, result types, yield in both branches.
// CHECK-LABEL: @simple_if_then
waveasm.program @simple_if_then
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %a = waveasm.precolored.vreg 0 : !waveasm.vreg
  %b = waveasm.precolored.vreg 1 : !waveasm.vreg
  %cond_val = waveasm.precolored.sreg 2 : !waveasm.sreg

  // CHECK:      %[[COND:.*]] = waveasm.precolored.sreg 2 : !waveasm.sreg
  // CHECK-NEXT: %{{.*}} = waveasm.if %[[COND]] : !waveasm.sreg -> !waveasm.vreg {
  %result = waveasm.if %cond_val : !waveasm.sreg -> !waveasm.vreg {
    // CHECK-NEXT:   %[[SUM:.*]] = waveasm.v_add_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %sum = waveasm.v_add_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    // CHECK-NEXT:   waveasm.yield %[[SUM]] : !waveasm.vreg
    waveasm.yield %sum : !waveasm.vreg
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT:   waveasm.yield %{{.*}} : !waveasm.vreg
    waveasm.yield %a : !waveasm.vreg
  // CHECK-NEXT: }
  }

  waveasm.s_endpgm
}

// --- If with computed condition (s_cmp_lt_u32 -> if) ---
// Verifies comparison result feeds into if, then/else each yield correctly.
// CHECK-LABEL: @if_then_else
waveasm.program @if_then_else
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %tid = waveasm.precolored.sreg 0 : !waveasm.sreg
  %a = waveasm.precolored.vreg 0 : !waveasm.vreg
  %b = waveasm.precolored.vreg 1 : !waveasm.vreg
  %limit = waveasm.constant 10 : !waveasm.imm<10>

  // CHECK:      %[[CMP:.*]] = waveasm.s_cmp_lt_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
  %cmp = waveasm.s_cmp_lt_u32 %tid, %limit : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg

  // CHECK-NEXT: %{{.*}} = waveasm.if %[[CMP]] : !waveasm.sreg -> !waveasm.vreg {
  %result = waveasm.if %cmp : !waveasm.sreg -> !waveasm.vreg {
    // CHECK-NEXT:   %{{.*}} = waveasm.v_add_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %sum = waveasm.v_add_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    // CHECK-NEXT:   waveasm.yield %{{.*}} : !waveasm.vreg
    waveasm.yield %sum : !waveasm.vreg
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT:   %{{.*}} = waveasm.v_sub_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %diff = waveasm.v_sub_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    // CHECK-NEXT:   waveasm.yield %{{.*}} : !waveasm.vreg
    waveasm.yield %diff : !waveasm.vreg
  // CHECK-NEXT: }
  }

  waveasm.s_endpgm
}

// --- If with multiple results (:2) ---
// Verifies multi-result if, both branches yield two values with matching types.
// CHECK-LABEL: @if_multiple_results
waveasm.program @if_multiple_results
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %a = waveasm.precolored.vreg 0 : !waveasm.vreg
  %b = waveasm.precolored.vreg 1 : !waveasm.vreg
  %cond = waveasm.precolored.sreg 2 : !waveasm.sreg

  // CHECK:      %{{.*}}:2 = waveasm.if %{{.*}} : !waveasm.sreg -> !waveasm.vreg, !waveasm.vreg {
  %r1, %r2 = waveasm.if %cond : !waveasm.sreg -> !waveasm.vreg, !waveasm.vreg {
    // CHECK:      %{{.*}} = waveasm.v_add_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %sum = waveasm.v_add_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    // CHECK-NEXT: %{{.*}} = waveasm.v_mul_lo_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %prod = waveasm.v_mul_lo_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    // CHECK-NEXT: waveasm.yield %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg
    waveasm.yield %sum, %prod : !waveasm.vreg, !waveasm.vreg
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: %{{.*}} = waveasm.v_sub_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %diff = waveasm.v_sub_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    // CHECK-NEXT: %{{.*}} = waveasm.v_and_b32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %mask = waveasm.v_and_b32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    // CHECK-NEXT: waveasm.yield %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg
    waveasm.yield %diff, %mask : !waveasm.vreg, !waveasm.vreg
  // CHECK-NEXT: }
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Nested Control Flow Tests
//===----------------------------------------------------------------------===//

// --- Nested loops: accumulator threading across loop nesting ---
// Verifies outer block arg flows into inner loop init, inner loop result (#1)
// feeds back to outer condition iter_args.
// CHECK-LABEL: @nested_loops
waveasm.program @nested_loops
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %outer_limit = waveasm.constant 4 : !waveasm.imm<4>
  %inner_limit = waveasm.constant 8 : !waveasm.imm<8>
  %init_i = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // Outer loop: two iter_args (counter + accumulator)
  // CHECK:      %{{.*}}:2 = waveasm.loop (%[[OI:.*]] = %{{.*}}, %[[OACC:.*]] = %{{.*}}) : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
  %final_i, %outer_result = waveasm.loop(%i = %init_i, %outer_acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %init_j = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

    // Inner loop: outer accumulator (%OACC) is the init for inner accumulator
    // CHECK:      %[[INNER:.*]]:2 = waveasm.loop (%{{.*}} = %{{.*}}, %[[IACC:.*]] = %[[OACC]]) : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
    %final_j, %inner_result = waveasm.loop(%j = %init_j, %inner_acc = %outer_acc)
        : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
      // CHECK:      %{{.*}} = waveasm.v_add_u32 %[[IACC]], %{{.*}} : !waveasm.vreg, !waveasm.sreg -> !waveasm.vreg
      %val = waveasm.v_add_u32 %inner_acc, %j
          : !waveasm.vreg, !waveasm.sreg -> !waveasm.vreg
      %next_j:2 = waveasm.s_add_u32 %j, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
      %cond_j = waveasm.s_cmp_lt_u32 %next_j#0, %inner_limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
      // CHECK:      waveasm.condition %{{.*}} : !waveasm.sreg iter_args(%{{.*}}, %{{.*}}) : !waveasm.sreg, !waveasm.vreg
      waveasm.condition %cond_j : !waveasm.sreg iter_args(%next_j#0, %val) : !waveasm.sreg, !waveasm.vreg
    }

    %next_i:2 = waveasm.s_add_u32 %i, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cond_i = waveasm.s_cmp_lt_u32 %next_i#0, %outer_limit : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg

    // Outer condition: accumulator comes from inner loop's result #1
    // CHECK:      waveasm.condition %{{.*}} : !waveasm.sreg iter_args(%{{.*}}, %[[INNER]]#1) : !waveasm.sreg, !waveasm.vreg
    waveasm.condition %cond_i : !waveasm.sreg iter_args(%next_i#0, %inner_result) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// --- If nested inside a loop ---
// Verifies if result (sreg) feeds into s_add_u32 which feeds condition iter_args.
// CHECK-LABEL: @loop_with_if
waveasm.program @loop_with_if
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %two = waveasm.constant 2 : !waveasm.imm<2>
  %limit = waveasm.constant 10 : !waveasm.imm<10>
  %init = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // CHECK:      %{{.*}} = waveasm.loop (%[[IV:.*]] = %{{.*}}) : (!waveasm.sreg) -> !waveasm.sreg {
  %result = waveasm.loop(%i = %init) : (!waveasm.sreg) -> (!waveasm.sreg) {
    // Block arg used by s_and to check parity
    // CHECK:      %[[REM:.*]] = waveasm.s_and_b32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %rem = waveasm.s_and_b32 %i, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    // CHECK-NEXT: %[[EVEN:.*]] = waveasm.s_cmp_eq_u32 %[[REM]], %{{.*}} : !waveasm.sreg, !waveasm.imm<0> -> !waveasm.sreg
    %is_even = waveasm.s_cmp_eq_u32 %rem, %zero : !waveasm.sreg, !waveasm.imm<0> -> !waveasm.sreg

    // If branches on the comparison result, producing an sreg step value
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

    // If result (%STEP) used to update counter, which feeds condition
    // CHECK:      %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %[[IV]], %[[STEP]] : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg, !waveasm.sreg
    %next:2 = waveasm.s_add_u32 %i, %step : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg, !waveasm.sreg
    // CHECK-NEXT: %[[CONT:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    %continue = waveasm.s_cmp_lt_u32 %next#0, %limit : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    // CHECK-NEXT: waveasm.condition %[[CONT]] : !waveasm.sreg iter_args(%[[NEXT]]) : !waveasm.sreg
    waveasm.condition %continue : !waveasm.sreg iter_args(%next#0) : !waveasm.sreg
  }

  waveasm.s_endpgm
}
