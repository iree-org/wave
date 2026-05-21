// RUN: waveasm-translate %s --waveasm-arith-legalization | FileCheck %s
// Verify that arith.or and arith.and are lowered to concrete SALU/VALU ops.

//===----------------------------------------------------------------------===//
// i32 OR
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_or_i32
waveasm.program @test_or_i32
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %s0 = waveasm.precolored.sreg 0 : !waveasm.sreg
  %s1 = waveasm.precolored.sreg 1 : !waveasm.sreg
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c42 = waveasm.constant 42 : !waveasm.imm<42>

  // VGPR | VGPR -> v_or_b32.
  // CHECK: waveasm.v_or_b32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg
  %or_vv = waveasm.arith.or %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // SGPR | SGPR -> s_or_b32.
  // CHECK: waveasm.s_or_b32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.sreg
  %or_ss = waveasm.arith.or %s0, %s1 : (!waveasm.sreg, !waveasm.sreg) -> !waveasm.sreg

  // SGPR | VGPR -> v_mov_b32 + v_or_b32.
  // CHECK: waveasm.v_mov_b32
  // CHECK: waveasm.v_or_b32
  %or_sv = waveasm.arith.or %s0, %v0 : (!waveasm.sreg, !waveasm.vreg) -> !waveasm.vreg

  // SGPR | imm -> s_or_b32.
  // CHECK: waveasm.s_or_b32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<42>
  %or_si = waveasm.arith.or %s0, %c42 : (!waveasm.sreg, !waveasm.imm<42>) -> !waveasm.sreg

  // imm | SGPR -> s_or_b32 with the SGPR normalized to operand 0.
  // CHECK: waveasm.s_or_b32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<1>
  %or_is = waveasm.arith.or %c1, %s0 : (!waveasm.imm<1>, !waveasm.sreg) -> !waveasm.sreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// i32 AND
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_and_i32
waveasm.program @test_and_i32
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %s0 = waveasm.precolored.sreg 0 : !waveasm.sreg
  %s1 = waveasm.precolored.sreg 1 : !waveasm.sreg
  %c1 = waveasm.constant 1 : !waveasm.imm<1>

  // VGPR & VGPR -> v_and_b32.
  // CHECK: waveasm.v_and_b32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg
  %and_vv = waveasm.arith.and %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // SGPR & SGPR -> s_and_b32.
  // CHECK: waveasm.s_and_b32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.sreg
  %and_ss = waveasm.arith.and %s0, %s1 : (!waveasm.sreg, !waveasm.sreg) -> !waveasm.sreg

  // SGPR & VGPR -> v_mov_b32 + v_and_b32.
  // CHECK: waveasm.v_mov_b32
  // CHECK: waveasm.v_and_b32
  %and_sv = waveasm.arith.and %s0, %v0 : (!waveasm.sreg, !waveasm.vreg) -> !waveasm.vreg

  // imm & SGPR -> s_and_b32 with the SGPR normalized to operand 0.
  // CHECK: waveasm.s_and_b32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<1>
  %and_is = waveasm.arith.and %c1, %s0 : (!waveasm.imm<1>, !waveasm.sreg) -> !waveasm.sreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// i64 OR
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_or_i64_salu
waveasm.program @test_or_i64_salu
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %a = waveasm.precolored.sreg 0, 2 : !waveasm.sreg<2, 2>
  %b = waveasm.precolored.sreg 2, 2 : !waveasm.sreg<2, 2>

  // Native s_or_b64 for i64 SALU.
  // CHECK: waveasm.s_or_b64 %{{.*}}, %{{.*}} : !waveasm.sreg<2, 2>, !waveasm.sreg<2, 2>
  %or = waveasm.arith.or %a, %b : (!waveasm.sreg<2, 2>, !waveasm.sreg<2, 2>) -> !waveasm.sreg<2, 2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_or_i64_valu
waveasm.program @test_or_i64_valu
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %v1 = waveasm.precolored.vreg 1 : !waveasm.vreg
  %va = waveasm.pack %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>
  %vb = waveasm.pack %v1, %v1 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  // Native v_or_b64 for i64 VALU.
  // CHECK: waveasm.v_or_b64 %{{.*}}, %{{.*}} : !waveasm.vreg<2>, !waveasm.vreg<2>
  %or = waveasm.arith.or %va, %vb : (!waveasm.vreg<2>, !waveasm.vreg<2>) -> !waveasm.vreg<2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// i64 AND
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_and_i64_salu
waveasm.program @test_and_i64_salu
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %a = waveasm.precolored.sreg 0, 2 : !waveasm.sreg<2, 2>
  %b = waveasm.precolored.sreg 2, 2 : !waveasm.sreg<2, 2>

  // Native s_and_b64 for i64 SALU.
  // CHECK: waveasm.s_and_b64 %{{.*}}, %{{.*}} : !waveasm.sreg<2, 2>, !waveasm.sreg<2, 2>
  %and = waveasm.arith.and %a, %b : (!waveasm.sreg<2, 2>, !waveasm.sreg<2, 2>) -> !waveasm.sreg<2, 2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}
