// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: v_cmp_* assembly emission with VCC and literal handling

// CHECK-LABEL: vcmp_test:

waveasm.program @vcmp_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: v_cmp_eq_u32 vcc, v0, v1
  waveasm.v_cmp_eq_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1>

  // Inline constant - no scratch VGPR needed
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  // CHECK: v_cmp_lt_u32 vcc, v0, 0
  waveasm.v_cmp_lt_u32 %v0, %c0 : !waveasm.pvreg<0>, !waveasm.imm<0>

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
