// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: GFX9 carry ops emit implicit VCC as explicit operand.
// v_add_co_u32/v_sub_co_u32: dst, vcc, src0, src1
// v_addc_co_u32/v_subb_co_u32: dst, vcc, src0, src1, vcc (carry-in).

// CHECK-LABEL: carry_add:
waveasm.program @carry_add target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  // CHECK: v_add_co_u32 v{{[0-9]+}}, vcc, v0, v1
  %r:2 = waveasm.v_add_co_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg, !waveasm.sreg
  waveasm.s_endpgm
}

// CHECK-LABEL: carry_sub:
waveasm.program @carry_sub target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  // CHECK: v_sub_co_u32 v{{[0-9]+}}, vcc, v0, v1
  %r:2 = waveasm.v_sub_co_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg, !waveasm.sreg
  waveasm.s_endpgm
}

// CHECK-LABEL: carry_addc:
waveasm.program @carry_addc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  // CHECK: v_addc_co_u32 v{{[0-9]+}}, vcc, v0, v1, vcc
  %r:2 = waveasm.v_addc_co_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg, !waveasm.sreg
  waveasm.s_endpgm
}

// CHECK-LABEL: carry_subb:
waveasm.program @carry_subb target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  // CHECK: v_subb_co_u32 v{{[0-9]+}}, vcc, v0, v1, vcc
  %r:2 = waveasm.v_subb_co_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg, !waveasm.sreg
  waveasm.s_endpgm
}

// Inline constant source: materialized to scratch VGPR via v_mov_b32.
// CHECK-LABEL: carry_add_literal:
waveasm.program @carry_add_literal target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c = waveasm.constant 12345 : !waveasm.imm<12345>
  // Non-inline literal in src1 gets materialized to a scratch VGPR.
  // CHECK:      v_mov_b32 v{{[0-9]+}}, 12345
  // CHECK-NEXT: v_add_co_u32 v{{[0-9]+}}, vcc, v0, v
  %r:2 = waveasm.v_add_co_u32 %v0, %c : !waveasm.pvreg<0>, !waveasm.imm<12345> -> !waveasm.vreg, !waveasm.sreg
  waveasm.s_endpgm
}

// Inline constant (0) passes through without materialization.
// CHECK-LABEL: carry_add_inline:
waveasm.program @carry_add_inline target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  // CHECK: v_add_co_u32 v{{[0-9]+}}, vcc, 0, v0
  %r:2 = waveasm.v_add_co_u32 %c0, %v0 : !waveasm.imm<0>, !waveasm.pvreg<0> -> !waveasm.vreg, !waveasm.sreg
  waveasm.s_endpgm
}
