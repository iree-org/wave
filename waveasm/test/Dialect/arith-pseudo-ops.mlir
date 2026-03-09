// RUN: waveasm-translate %s | waveasm-translate | FileCheck %s
// Verify that generic arithmetic pseudo-ops roundtrip through parsing and printing.

// CHECK: waveasm.program @test_arith_pseudo_ops
waveasm.program @test_arith_pseudo_ops
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %s0 = waveasm.precolored.sreg 0 : !waveasm.sreg
  %c42 = waveasm.constant 42 : !waveasm.imm<42>

  // CHECK: waveasm.arith.add
  %add_vv = waveasm.arith.add %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // CHECK: waveasm.arith.add
  %add_sv = waveasm.arith.add %s0, %v0 : (!waveasm.sreg, !waveasm.vreg) -> !waveasm.vreg

  // CHECK: waveasm.arith.add
  %add_ss = waveasm.arith.add %s0, %s0 : (!waveasm.sreg, !waveasm.sreg) -> !waveasm.sreg

  // CHECK: waveasm.arith.mul
  %mul = waveasm.arith.mul %v0, %c42 : (!waveasm.vreg, !waveasm.imm<42>) -> !waveasm.vreg

  // CHECK: waveasm.arith.cmp eq
  %cmp_eq = waveasm.arith.cmp eq, %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // CHECK: waveasm.arith.cmp slt
  %cmp_slt = waveasm.arith.cmp slt, %s0, %c42 : (!waveasm.sreg, !waveasm.imm<42>) -> !waveasm.sreg

  // CHECK: waveasm.arith.cmp ult
  %cmp_ult = waveasm.arith.cmp ult, %v0, %s0 : (!waveasm.vreg, !waveasm.sreg) -> !waveasm.vreg

  // CHECK: waveasm.arith.select
  %sel = waveasm.arith.select %cmp_eq, %v0, %add_vv : (!waveasm.vreg, !waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // CHECK: waveasm.arith.trunc
  %s_wide = waveasm.precolored.sreg 4, 2 : !waveasm.sreg<2, 2>
  %trunc = waveasm.arith.trunc %s_wide : (!waveasm.sreg<2, 2>) -> !waveasm.sreg

  // CHECK: waveasm.arith.sext
  %sext = waveasm.arith.sext %s0 : (!waveasm.sreg) -> !waveasm.sreg<2, 2>

  // CHECK: waveasm.arith.zext
  %zext = waveasm.arith.zext %v0 : (!waveasm.vreg) -> !waveasm.vreg<2>

  waveasm.s_endpgm
}
