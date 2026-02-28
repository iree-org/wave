// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: ceildiv pattern: (x + N-1) >> log2(N)

// CHECK-LABEL: ceildiv_pattern_test:

waveasm.program @ceildiv_pattern_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Simulate ceildiv(x, 16): add bias 15, then shift right by 4
  %c15 = waveasm.constant 15 : !waveasm.imm<15>
  // CHECK: v_add_u32 v{{[0-9]+}}, 15, v0
  %biased = waveasm.v_add_u32 %c15, %v0 : !waveasm.imm<15>, !waveasm.pvreg<0> -> !waveasm.vreg

  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  // CHECK: v_lshrrev_b32 v{{[0-9]+}}, 4, v{{[0-9]+}}
  %result = waveasm.v_lshrrev_b32 %c4, %biased : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
