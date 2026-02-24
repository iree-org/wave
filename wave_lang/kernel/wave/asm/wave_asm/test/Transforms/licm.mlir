// RUN: waveasm-translate --waveasm-peephole --waveasm-licm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: LICM — hoist chain of VALU address ops out of loop.
//
// Both v_lshlrev_b32 and v_add_u32 use only loop-external values.
// After LICM pass 1 hoists the shift, pass 2 hoists the add (iterative).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_licm_chain
waveasm.program @test_licm_chain
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // Peephole fuses shift+add into v_lshl_add_u32, then LICM hoists it.
  // CHECK: waveasm.v_lshl_add_u32
  // CHECK: waveasm.loop
  // CHECK-NOT: waveasm.v_lshl_add_u32
  // CHECK: waveasm.buffer_store_dword
  %r = waveasm.loop(%iv = %init_iv)
      : (!waveasm.sreg) -> (!waveasm.sreg) {

    // All operands are loop-external — both should be hoisted.
    %shifted = waveasm.v_lshlrev_b32 %c4, %v0 : !waveasm.imm<4>, !waveasm.pvreg<0> -> !waveasm.vreg
    %addr = waveasm.v_add_u32 %shifted, %v1 : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg

    waveasm.buffer_store_dword %addr, %srd, %v0 : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: LICM — no hoist when operand is a loop-carried block argument.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_licm_no_hoist
waveasm.program @test_licm_no_hoist
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // v_add_u32 uses block arg %acc — must stay inside the loop.
  // CHECK: waveasm.loop
  // CHECK: waveasm.v_add_u32
  %r:2 = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %addr = waveasm.v_add_u32 %v0, %acc : !waveasm.pvreg<0>, !waveasm.vreg -> !waveasm.vreg
    waveasm.buffer_store_dword %addr, %srd, %v0 : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %addr) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}
