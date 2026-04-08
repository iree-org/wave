// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: Inline AGPR constants. When writing an inline constant ([-16, 64])
// to an AGPR, emit v_accvgpr_write_b32 directly without a scratch VGPR.
// Non-inline literals still require v_mov_b32 to scratch VGPR first.

// CHECK-LABEL: agpr_inline_test:

waveasm.program @agpr_inline_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  // Inline constant 0 -> direct v_accvgpr_write_b32, no scratch VGPR
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  // CHECK: v_accvgpr_write_b32 a{{[0-9]+}}, 0
  %a0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.areg

  // Inline constant 42 -> direct v_accvgpr_write_b32
  %c42 = waveasm.constant 42 : !waveasm.imm<42>
  // CHECK-NEXT: v_accvgpr_write_b32 a{{[0-9]+}}, 42
  %a1 = waveasm.v_mov_b32 %c42 : !waveasm.imm<42> -> !waveasm.areg

  // Non-inline literal 999 -> must use scratch VGPR
  %c999 = waveasm.constant 999 : !waveasm.imm<999>
  // CHECK-NEXT: v_mov_b32 v15, 999
  // CHECK-NEXT: v_accvgpr_write_b32 a{{[0-9]+}}, v15
  %a2 = waveasm.v_mov_b32 %c999 : !waveasm.imm<999> -> !waveasm.areg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
