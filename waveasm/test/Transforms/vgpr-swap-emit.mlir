// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: VGPR swap cycle in loop iter_args during assembly emission.
//
// The loop swaps two VGPR iter_args each iteration: the condition passes
// %b as the new %a and %a as the new %b, creating a 2-element swap cycle
// that the emitter must resolve with a temporary VGPR.

// CHECK-LABEL: vgpr_swap_iter_args:
waveasm.program @vgpr_swap_iter_args
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c42 = waveasm.constant 42 : !waveasm.imm<42>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %init_iv = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_a = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %init_b = waveasm.v_mov_b32 %c42 : !waveasm.imm<42> -> !waveasm.vreg

  // Loop that swaps %a and %b each iteration.
  // CHECK: v_mov_b32 v
  // CHECK: v_mov_b32 v
  // CHECK: v_mov_b32 v
  // CHECK: s_cbranch_scc1
  %r:3 = waveasm.loop(%iv = %init_iv, %a = %init_a, %b = %init_b)
      : (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg)
      -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg) {

    // Use both so they're not dead.
    %sum = waveasm.v_add_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    %next_iv:2 = waveasm.s_add_u32 %iv, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv#0, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    // Swap: pass %b as new %a, %a as new %b.
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv#0, %b, %a) : !waveasm.sreg, !waveasm.vreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}
