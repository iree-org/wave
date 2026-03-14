// RUN: waveasm-translate --waveasm-sgpr-promotion %s 2>&1 | FileCheck %s
//
// Test SGPR promotion: VALU ops with all-SGPR/imm operands should be
// promoted to SALU equivalents.

// Test 1: V_ADD_U32 with two SGPR operands -> S_ADD_U32
// CHECK-LABEL: waveasm.program @promote_add
waveasm.program @promote_add target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>

  // SGPR + SGPR -> should become S_ADD_U32
  // CHECK: waveasm.s_add_u32
  // CHECK-NOT: waveasm.v_add_u32
  %v0 = waveasm.v_add_u32 %s0, %s1 : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 2: V_MUL_LO_U32 with SGPR operands -> S_MUL_I32
// CHECK-LABEL: waveasm.program @promote_mul
waveasm.program @promote_mul target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>

  // CHECK: waveasm.s_mul_i32
  // CHECK-NOT: waveasm.v_mul_lo_u32
  %v0 = waveasm.v_mul_lo_u32 %s0, %s1 : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 3: V_AND_B32 with SGPR + immediate -> S_AND_B32
// CHECK-LABEL: waveasm.program @promote_and_imm
waveasm.program @promote_and_imm target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %c255 = waveasm.constant 255 : !waveasm.imm<255>

  // CHECK: waveasm.s_and_b32
  // CHECK-NOT: waveasm.v_and_b32
  %v0 = waveasm.v_and_b32 %s0, %c255 : !waveasm.psreg<0>, !waveasm.imm<255> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 4: V_LSHLREV_B32 operand reversal: (shift_amt, value) -> S_LSHL_B32(value, shift_amt)
// CHECK-LABEL: waveasm.program @promote_shift_rev
waveasm.program @promote_shift_rev target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %c3 = waveasm.constant 3 : !waveasm.imm<3>

  // CHECK: waveasm.s_lshl_b32
  // CHECK-NOT: waveasm.v_lshlrev_b32
  %v0 = waveasm.v_lshlrev_b32 %c3, %s0 : !waveasm.imm<3>, !waveasm.psreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 5: No promotion when a VGPR operand is present
// CHECK-LABEL: waveasm.program @no_promote_vgpr
waveasm.program @no_promote_vgpr target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>

  // VGPR + SGPR -> should remain VALU
  // CHECK: waveasm.v_add_u32
  // CHECK-NOT: waveasm.s_add_u32
  %v1 = waveasm.v_add_u32 %v0, %s0 : !waveasm.pvreg<0>, !waveasm.psreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 6: Cascading promotion - result of promoted op feeds another VALU
// CHECK-LABEL: waveasm.program @cascade_promote
waveasm.program @cascade_promote target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>
  %s2 = waveasm.precolored.sreg 2 : !waveasm.psreg<2>

  // First add promoted: SGPR + SGPR
  // CHECK: waveasm.s_add_u32
  %v0 = waveasm.v_add_u32 %s0, %s1 : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.vreg

  // Second mul: result of promoted add (now SGPR) + SGPR -> also promoted
  // CHECK: waveasm.s_mul_i32
  // CHECK-NOT: waveasm.v_mul_lo_u32
  %v1 = waveasm.v_mul_lo_u32 %v0, %s2 : !waveasm.vreg, !waveasm.psreg<2> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 7: V_READFIRSTLANE_B32 elimination when source is SGPR
// CHECK-LABEL: waveasm.program @readfirstlane_elim
waveasm.program @readfirstlane_elim target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>

  // Add is promoted to SALU (SGPR result)
  // CHECK: waveasm.s_add_u32
  %v0 = waveasm.v_add_u32 %s0, %s1 : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.vreg

  // V_READFIRSTLANE of promoted-to-SGPR value -> should be eliminated
  // CHECK-NOT: waveasm.v_readfirstlane_b32
  %s3 = waveasm.v_readfirstlane_b32 %v0 : !waveasm.vreg -> !waveasm.sreg

  waveasm.s_endpgm
}

// Test 8: V_LSHL_OR_B32 decomposition to S_LSHL_B32 + S_OR_B32
// CHECK-LABEL: waveasm.program @promote_lshl_or
waveasm.program @promote_lshl_or target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %c3 = waveasm.constant 3 : !waveasm.imm<3>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>

  // CHECK: waveasm.s_lshl_b32
  // CHECK: waveasm.s_or_b32
  // CHECK-NOT: waveasm.v_lshl_or_b32
  %v0 = waveasm.v_lshl_or_b32 %s0, %c3, %s1 : !waveasm.psreg<0>, !waveasm.imm<3>, !waveasm.psreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}
