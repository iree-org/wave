// RUN: waveasm-translate --disable-pass-verifier --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test that IfOp results are properly tied to their yield operands via
// the liveness analysis (Pass 3c).  Without tied classes, the allocator
// may assign different physical registers to the yield operand and the
// IfOp result, causing verification failures or incorrect codegen.

//===----------------------------------------------------------------------===//
// Test 1: IfOp with vreg result - tied to then-yield and else-yield
//===----------------------------------------------------------------------===//

// Register allocation should succeed (no "Failed to allocate" error).
// The IfOp result should get the same physical register as the yield operands.
// CHECK-LABEL: waveasm.program @ifop_vreg_tying
// CHECK-NOT: Failed to allocate
// CHECK: waveasm.if
// CHECK: waveasm.s_endpgm

waveasm.program @ifop_vreg_tying
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %cond = waveasm.precolored.sreg 2 : !waveasm.sreg

  %a = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %b = waveasm.v_mov_b32 %c1 : !waveasm.imm<1> -> !waveasm.vreg

  %result = waveasm.if %cond : !waveasm.sreg -> !waveasm.vreg {
    %sum = waveasm.v_add_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    waveasm.yield %sum : !waveasm.vreg
  } else {
    waveasm.yield %a : !waveasm.vreg
  }

  %out = waveasm.v_add_u32 %result, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 2: IfOp inside a loop -- tests that IfOp result def points are
// placed after the IfOp body, not at the IfOp itself
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @ifop_in_loop
// CHECK-NOT: Failed to allocate
// CHECK: waveasm.loop
// CHECK:   waveasm.if
// CHECK: waveasm.s_endpgm

waveasm.program @ifop_in_loop
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %cond = waveasm.precolored.sreg 2 : !waveasm.sreg

  %final:2 = waveasm.loop(%i = %init_i, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %v0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
    %v1 = waveasm.v_mov_b32 %c1 : !waveasm.imm<1> -> !waveasm.vreg

    %step = waveasm.if %cond : !waveasm.sreg -> !waveasm.vreg {
      waveasm.yield %v1 : !waveasm.vreg
    } else {
      waveasm.yield %v0 : !waveasm.vreg
    }

    %new_acc = waveasm.v_add_u32 %acc, %step : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %next:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cont = waveasm.s_cmp_lt_u32 %next#0, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cont : !waveasm.sreg iter_args(%next#0, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 3: IfOp with wide (vreg<4,4>) accumulator results
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @ifop_wide_accum
// CHECK-NOT: Failed to allocate
// CHECK: waveasm.if
// CHECK: waveasm.s_endpgm

waveasm.program @ifop_wide_accum
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %cond = waveasm.precolored.sreg 2 : !waveasm.sreg

  %init = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %result = waveasm.if %cond : !waveasm.sreg -> !waveasm.vreg<4, 4> {
    waveasm.yield %init : !waveasm.vreg<4, 4>
  } else {
    %zero = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
    waveasm.yield %zero : !waveasm.vreg<4, 4>
  }

  waveasm.s_endpgm
}
