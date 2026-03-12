// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: v_add_u32 commutes non-inline literal from src1 to src0

// CHECK-LABEL: vadd_commute_test:

waveasm.program @vadd_commute_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Non-inline literal in src1 should be commuted to src0
  %c256 = waveasm.constant 256 : !waveasm.imm<256>
  // CHECK-NOT: v_mov_b32
  // CHECK: v_add_u32 v{{[0-9]+}}, 256, v0
  %r1 = waveasm.v_add_u32 %v0, %c256 : !waveasm.pvreg<0>, !waveasm.imm<256> -> !waveasm.vreg

  // Inline constant should work without commutation
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  // CHECK: v_add_u32 v{{[0-9]+}}, v{{[0-9]+}}, 1
  %r2 = waveasm.v_add_u32 %r1, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
