// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: SGPR swap cycle in loop iter_args with precolored VCC present.
//
// The loop swaps two SGPR iter_args each iteration, which the emitter must
// resolve using a temporary SGPR.  A precolored VCC (s[106:107]) is kept live
// across the loop so it appears in the peak-SGPR scan.  VCC is an
// architectural register emitted as "vcc" in assembly; it must not contribute
// to the general-purpose SGPR peak used to pick the swap temp.  Otherwise the
// temp would be allocated at s108 or higher -- past the last user SGPR
// (s105 on gfx942) -- which the assembler rejects as out-of-range.

// CHECK-LABEL: sgpr_swap_with_vcc:
waveasm.program @sgpr_swap_with_vcc
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 4 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c42 = waveasm.constant 42 : !waveasm.imm<42>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %init_iv = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_a = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_b = waveasm.s_mov_b32 %c42 : !waveasm.imm<42> -> !waveasm.sreg

  // Keep a precolored VCC alive across the loop via a pre-body v_cndmask user.
  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>
  %vreg0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %vreg1 = waveasm.v_mov_b32 %c1 : !waveasm.imm<1> -> !waveasm.vreg

  // Loop that swaps %a and %b each iteration.  The emitter resolves the
  // swap cycle with a three-instruction sequence (temp = a; a = b; b = temp)
  // using a scratch SGPR.  The scratch must be an in-range user SGPR.  Any
  // three-digit `s1XX` would be past the last user SGPR on gfx942 (s105 is
  // the last user SGPR; s106-s107 are VCC; s108+ is TTMP / out-of-range).
  // CHECK: s_cmp_lt_u32
  // CHECK-NOT: s_mov_b32 s1{{[0-9][0-9]}},
  // CHECK-NOT: s_mov_b32 s{{[0-9]+}}, s1{{[0-9][0-9]}}
  // CHECK: s_cbranch_scc1
  %r:3 = waveasm.loop(%iv = %init_iv, %a = %init_a, %b = %init_b)
      : (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg)
      -> (!waveasm.sreg, !waveasm.sreg, !waveasm.sreg) {

    // Use both iter_args so they stay live.
    %sum:2 = waveasm.s_add_u32 %a, %b : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg, !waveasm.scc

    // Keep VCC consumed inside the loop so it participates in the peak scan.
    %sel = waveasm.v_cndmask_b32 %vreg0, %vreg1, %vcc
         : !waveasm.vreg, !waveasm.vreg, !waveasm.psreg<106, 2> -> !waveasm.vreg

    %next_iv:2 = waveasm.s_add_u32 %iv, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
    %cond = waveasm.s_cmp_lt_u32 %next_iv#0, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.scc
    // Swap: pass %b as new %a, %a as new %b.
    waveasm.condition %cond : !waveasm.scc iter_args(%next_iv#0, %b, %a) : !waveasm.sreg, !waveasm.sreg, !waveasm.sreg
  }

  waveasm.s_endpgm
}
