// RUN: waveasm-translate --waveasm-scc-spill-reload %s | FileCheck %s
//
// Verify that SCC spill/reload inserts s_cselect_b32 (spill) after the SCC
// producer and s_cmp_ne_u32 (reload) before each consumer when an
// SCC-clobbering op intervenes.

// CHECK-LABEL: waveasm.program @spill_reload_condition
waveasm.program @spill_reload_condition
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
    // Spill: s_cselect_b32 saves SCC to SGPR right after the producer.
    // CHECK: waveasm.s_cmp_lt_u32
    // CHECK-NEXT: waveasm.constant 1
    // CHECK-NEXT: waveasm.constant 0
    // CHECK-NEXT: waveasm.s_cselect_b32
    // SCC-clobbering op.
    %next:2 = waveasm.s_add_u32 %i, %one
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
    // Reload: s_cmp_ne_u32 restores SCC from SGPR before the consumer.
    // CHECK: waveasm.s_add_u32
    // CHECK: waveasm.s_cmp_ne_u32
    // CHECK: waveasm.condition
    waveasm.condition %cond : !waveasm.scc iter_args(%next#0) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

// Verify no spill/reload when there is no intervening clobber.
// CHECK-LABEL: waveasm.program @no_spill_when_adjacent
waveasm.program @no_spill_when_adjacent
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 10 : !waveasm.imm<10>
  %init = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  %final = waveasm.loop(%i = %init) : (!waveasm.sreg) -> (!waveasm.sreg) {
    // Increment first, then compare.  No clobber between cmp and condition.
    %next:2 = waveasm.s_add_u32 %i, %one
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
    %cond = waveasm.s_cmp_lt_u32 %next#0, %limit
        : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.scc
    // No spill/reload should be inserted.
    // CHECK: waveasm.s_cmp_lt_u32
    // CHECK-NOT: waveasm.s_cselect_b32
    // CHECK-NOT: waveasm.s_cmp_ne_u32
    // CHECK: waveasm.condition
    waveasm.condition %cond : !waveasm.scc iter_args(%next#0) : !waveasm.sreg
  }

  waveasm.s_endpgm
}
