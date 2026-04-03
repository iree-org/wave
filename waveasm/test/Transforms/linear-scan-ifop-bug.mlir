// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Verify that IfOp yield type mismatches after register allocation are
// accepted.  The then- and else-branches may yield values allocated to
// different physical registers (e.g. pvreg<3> vs pvreg<1>).
// RegionBranchOpInterface::areTypesCompatible() treats structurally
// compatible physical register types as equivalent, so this is valid.

//===----------------------------------------------------------------------===//
// Test 1: Standalone IfOp — branches yield different physical registers.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @if_yield_type_mismatch
// CHECK:       waveasm.if
// CHECK:         waveasm.yield {{.*}} : !waveasm.pvreg<3>
// CHECK:       } else {
// CHECK:         waveasm.yield {{.*}} : !waveasm.pvreg<1>

waveasm.program @if_yield_type_mismatch
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %val = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %s_zero = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %cmp = waveasm.s_cmp_lt_u32 %s_zero, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg

  %result = waveasm.if %cmp : !waveasm.sreg -> !waveasm.vreg {
    %added = waveasm.v_add_u32 %val, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
    waveasm.yield %added : !waveasm.vreg
  } else {
    waveasm.yield %val : !waveasm.vreg
  }

  %use = waveasm.v_add_u32 %result, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 2: IfOp inside loop body.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @if_in_loop_type_mismatch
// CHECK:       waveasm.loop
// CHECK:         waveasm.if
// CHECK:           waveasm.yield {{.*}} : !waveasm.pvreg<3>
// CHECK:         } else {
// CHECK:           waveasm.yield {{.*}} : !waveasm.pvreg<1>

waveasm.program @if_in_loop_type_mismatch
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_val = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  %i_out, %val_out = waveasm.loop(%i = %init_i, %val = %init_val)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %cmp = waveasm.s_cmp_lt_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg

    %new_val = waveasm.if %cmp : !waveasm.sreg -> !waveasm.vreg {
      %added = waveasm.v_add_u32 %val, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
      waveasm.yield %added : !waveasm.vreg
    } else {
      waveasm.yield %val : !waveasm.vreg
    }

    %new_sum = waveasm.v_add_u32 %new_val, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

    %next_i:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %loop_cond = waveasm.s_cmp_lt_u32 %next_i#0, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %loop_cond : !waveasm.sreg iter_args(%next_i#0, %new_sum) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 3: Multiple IfOps in sequence inside a loop.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @multiple_ifs_compound
// CHECK:       waveasm.loop
// CHECK:         waveasm.if
// CHECK:           waveasm.yield
// CHECK:         } else {
// CHECK:           waveasm.yield

waveasm.program @multiple_ifs_compound
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_val = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  %i_out, %val_out = waveasm.loop(%i = %init_i, %val = %init_val)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %cmp1 = waveasm.s_cmp_lt_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %if1_result = waveasm.if %cmp1 : !waveasm.sreg -> !waveasm.vreg {
      %a1 = waveasm.v_add_u32 %val, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
      waveasm.yield %a1 : !waveasm.vreg
    } else {
      waveasm.yield %val : !waveasm.vreg
    }

    %cmp2 = waveasm.s_cmp_lt_u32 %i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    %if2_result = waveasm.if %cmp2 : !waveasm.sreg -> !waveasm.vreg {
      %b1 = waveasm.v_add_u32 %if1_result, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
      waveasm.yield %b1 : !waveasm.vreg
    } else {
      waveasm.yield %if1_result : !waveasm.vreg
    }

    %next_i:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %loop_cond = waveasm.s_cmp_lt_u32 %next_i#0, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %loop_cond : !waveasm.sreg iter_args(%next_i#0, %if2_result) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}
