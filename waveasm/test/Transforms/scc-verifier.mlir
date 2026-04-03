// RUN: not waveasm-translate --waveasm-scc-verifier %s 2>&1 | FileCheck %s
//
// Verify that the SCC verifier detects hazards:
// 1. SCC-clobbering op between SCC producer and condition consumer.
// 2. SCC-clobbering op between SCC producer and if consumer.

// CHECK: error: SCC hazard: 1 SCC-clobbering op(s) between SCC producer 'waveasm.s_cmp_lt_u32' and consumer 'waveasm.condition'
waveasm.program @clobber_before_condition
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 10 : !waveasm.imm<10>
  %init = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  %final = waveasm.loop(%i = %init) : (!waveasm.sreg) -> (!waveasm.sreg) {
    // SCC producer.
    %cond = waveasm.s_cmp_lt_u32 %i, %limit
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.scc
    // SCC clobber between producer and consumer.
    %next:2 = waveasm.s_add_u32 %i, %one
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
    // Consumer reads stale SCC.
    waveasm.condition %cond : !waveasm.scc iter_args(%next#0) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

// CHECK: error: SCC hazard: 1 SCC-clobbering op(s) between SCC producer 'waveasm.s_cmp_lt_u32' and consumer 'waveasm.if'
waveasm.program @clobber_before_if
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %tid = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %limit = waveasm.constant 10 : !waveasm.imm<10>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // SCC producer.
  %cmp = waveasm.s_cmp_lt_u32 %tid, %limit
      : !waveasm.psreg<0>, !waveasm.imm<10> -> !waveasm.scc
  // Clobber.
  %dummy:2 = waveasm.s_add_u32 %tid, %one
      : !waveasm.psreg<0>, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
  // Consumer reads stale SCC.
  %result = waveasm.if %cmp : !waveasm.scc -> !waveasm.vreg {
    %sum = waveasm.v_add_u32 %a, %b
        : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
    waveasm.yield %sum : !waveasm.vreg
  } else {
    %diff = waveasm.v_sub_u32 %a, %b
        : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
    waveasm.yield %diff : !waveasm.vreg
  }

  waveasm.s_endpgm
}
