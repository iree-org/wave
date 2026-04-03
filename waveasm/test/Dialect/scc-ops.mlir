// RUN: waveasm-translate %s | FileCheck %s
//
// Roundtrip tests for SCC-producing and SCC-consuming ops.

// CHECK-LABEL: waveasm.program @scc_binary_with_carry
waveasm.program @scc_binary_with_carry
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>
  %s2 = waveasm.precolored.sreg 2 : !waveasm.psreg<2>
  %s3 = waveasm.precolored.sreg 3 : !waveasm.psreg<3>

  // SALUBinaryWithCarryOp: produces dst + SCC carry-out.
  // CHECK: [[LO:%.*]], [[SCC1:%.*]] = waveasm.s_add_u32
  // CHECK-SAME: -> !waveasm.sreg, !waveasm.scc
  %lo:2 = waveasm.s_add_u32 %s0, %s2
      : !waveasm.psreg<0>, !waveasm.psreg<2> -> !waveasm.sreg, !waveasm.scc

  // SALUBinaryWithCarryInOp: reads SCC carry-in, produces dst + SCC carry-out.
  // CHECK: [[HI:%.*]], [[SCC2:%.*]] = waveasm.s_addc_u32 [[SCC1]]
  // CHECK-SAME: -> !waveasm.sreg, !waveasm.scc
  %hi:2 = waveasm.s_addc_u32 %lo#1, %s1, %s3
      : !waveasm.scc, !waveasm.psreg<1>, !waveasm.psreg<3>
      -> !waveasm.sreg, !waveasm.scc

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @scc_binary_with_scc_clobber
waveasm.program @scc_binary_with_scc_clobber
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>

  // SALUBinaryWithSCCOp: bitwise ops clobber SCC.
  // CHECK: [[AND:%.*]], %{{.*}} = waveasm.s_and_b32
  // CHECK-SAME: -> !waveasm.sreg, !waveasm.scc
  %and:2 = waveasm.s_and_b32 %s0, %s1
      : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.sreg, !waveasm.scc

  // CHECK: [[OR:%.*]], %{{.*}} = waveasm.s_or_b32
  %or:2 = waveasm.s_or_b32 %s0, %s1
      : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.sreg, !waveasm.scc

  // CHECK: [[LSHL:%.*]], %{{.*}} = waveasm.s_lshl_b32
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %lshl:2 = waveasm.s_lshl_b32 %s0, %c4
      : !waveasm.psreg<0>, !waveasm.imm<4> -> !waveasm.sreg, !waveasm.scc

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @scc_cmp_and_cselect
waveasm.program @scc_cmp_and_cselect
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>

  // SALUCmpOp: produces SCC.
  // CHECK: [[CMP:%.*]] = waveasm.s_cmp_lt_u32
  // CHECK-SAME: -> !waveasm.scc
  %cmp = waveasm.s_cmp_lt_u32 %s0, %c10
      : !waveasm.psreg<0>, !waveasm.imm<10> -> !waveasm.scc

  // S_CSELECT_B32: consumes SCC.
  // CHECK: waveasm.s_cselect_b32 [[CMP]]
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %sel = waveasm.s_cselect_b32 %cmp, %one, %zero
      : !waveasm.scc, !waveasm.imm<1>, !waveasm.imm<0> -> !waveasm.sreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @scc_unary_with_scc
waveasm.program @scc_unary_with_scc
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>

  // SALUUnaryWithSCCOp: clobbers SCC.
  // CHECK: [[NOT:%.*]], %{{.*}} = waveasm.s_not_b32
  // CHECK-SAME: -> !waveasm.sreg, !waveasm.scc
  %not:2 = waveasm.s_not_b32 %s0
      : !waveasm.psreg<0> -> !waveasm.sreg, !waveasm.scc

  // CHECK: [[BREV:%.*]], %{{.*}} = waveasm.s_brev_b32
  %brev:2 = waveasm.s_brev_b32 %s0
      : !waveasm.psreg<0> -> !waveasm.sreg, !waveasm.scc

  waveasm.s_endpgm
}
