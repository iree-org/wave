// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Verify assembly emission for SCC-related ops:
// - s_cselect_b32 emits dst, src0, src1 (no SCC operand).
// - s_cmp_ne_u32 emits as s_cmp_lg_u32 (hardware mnemonic).
// - s_add_u32 / s_addc_u32 skip SCC operands in emission.

// CHECK-LABEL: scc_cselect_and_cmp_ne:
waveasm.program @scc_cselect_and_cmp_ne
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>

  // s_cmp_ne_u32 -> s_cmp_lg_u32 in assembly.
  // CHECK: s_cmp_lg_u32 s0, 0
  %cmp = waveasm.s_cmp_ne_u32 %s0, %c0
      : !waveasm.psreg<0>, !waveasm.imm<0> -> !waveasm.scc

  // s_cselect_b32: SCC operand not emitted.
  // CHECK: s_cselect_b32 s{{[0-9]+}}, 1, 0
  %sel = waveasm.s_cselect_b32 %cmp, %c1, %c0
      : !waveasm.scc, !waveasm.imm<1>, !waveasm.imm<0> -> !waveasm.sreg

  waveasm.s_endpgm
}

// CHECK-LABEL: scc_carry_chain:
waveasm.program @scc_carry_chain
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>
  %s2 = waveasm.precolored.sreg 2 : !waveasm.psreg<2>
  %s3 = waveasm.precolored.sreg 3 : !waveasm.psreg<3>

  // s_add_u32: SCC result not emitted.
  // CHECK: s_add_u32 s{{[0-9]+}}, s0, s2
  %lo:2 = waveasm.s_add_u32 %s0, %s2
      : !waveasm.psreg<0>, !waveasm.psreg<2> -> !waveasm.sreg, !waveasm.scc

  // s_addc_u32: SCC operand and SCC result not emitted.
  // CHECK: s_addc_u32 s{{[0-9]+}}, s1, s3
  %hi:2 = waveasm.s_addc_u32 %lo#1, %s1, %s3
      : !waveasm.scc, !waveasm.psreg<1>, !waveasm.psreg<3>
      -> !waveasm.sreg, !waveasm.scc

  waveasm.s_endpgm
}

// CHECK-LABEL: scc_bitwise_skip:
waveasm.program @scc_bitwise_skip
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>

  // SALUBinaryWithSCCOp: SCC result not emitted.
  // CHECK: s_and_b32 s{{[0-9]+}}, s0, s1
  %and:2 = waveasm.s_and_b32 %s0, %s1
      : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.sreg, !waveasm.scc

  waveasm.s_endpgm
}
