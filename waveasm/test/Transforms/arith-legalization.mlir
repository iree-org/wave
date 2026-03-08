// RUN: waveasm-translate %s --waveasm-arith-legalization | FileCheck %s
// Verify that generic arithmetic pseudo-ops are lowered to concrete SALU/VALU ops.

// CHECK-LABEL: waveasm.program @test_add
waveasm.program @test_add
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %s0 = waveasm.precolored.sreg 0 : !waveasm.sreg
  %s1 = waveasm.precolored.sreg 1 : !waveasm.sreg
  %c42 = waveasm.constant 42 : !waveasm.imm<42>

  // VGPR + VGPR → v_add_u32.
  // CHECK: waveasm.v_add_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg
  %add_vv = waveasm.arith.add %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // SGPR + SGPR → s_add_u32.
  // CHECK: waveasm.s_add_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.sreg
  %add_ss = waveasm.arith.add %s0, %s1 : (!waveasm.sreg, !waveasm.sreg) -> !waveasm.sreg

  // SGPR + VGPR → v_add_u32 (SGPR broadcast, 1 SGPR allowed).
  // CHECK: waveasm.v_add_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.vreg
  %add_sv = waveasm.arith.add %s0, %v0 : (!waveasm.sreg, !waveasm.vreg) -> !waveasm.vreg

  // SGPR + imm → s_add_u32.
  // CHECK: waveasm.s_add_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<42>
  %add_si = waveasm.arith.add %s0, %c42 : (!waveasm.sreg, !waveasm.imm<42>) -> !waveasm.sreg

  // No arith pseudo-ops should remain.
  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_mul
waveasm.program @test_mul
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %s0 = waveasm.precolored.sreg 0 : !waveasm.sreg
  %s1 = waveasm.precolored.sreg 1 : !waveasm.sreg
  %c42 = waveasm.constant 42 : !waveasm.imm<42>

  // VGPR × imm → v_mul_lo_u32.
  // CHECK: waveasm.v_mul_lo_u32
  %mul_vi = waveasm.arith.mul %v0, %c42 : (!waveasm.vreg, !waveasm.imm<42>) -> !waveasm.vreg

  // SGPR × SGPR → s_mul_i32.
  // CHECK: waveasm.s_mul_i32
  %mul_ss = waveasm.arith.mul %s0, %s1 : (!waveasm.sreg, !waveasm.sreg) -> !waveasm.sreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_cmp
waveasm.program @test_cmp
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %s0 = waveasm.precolored.sreg 0 : !waveasm.sreg
  %c10 = waveasm.constant 10 : !waveasm.imm<10>

  // VGPR, VGPR → v_cmp_lt_i32 (sets VCC).
  // CHECK: waveasm.v_cmp_lt_i32
  %cmp_vv = waveasm.arith.cmp slt, %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // SGPR, imm → s_cmp_lt_u32 (sets SCC).
  // CHECK: waveasm.s_cmp_lt_u32
  %cmp_si = waveasm.arith.cmp ult, %s0, %c10 : (!waveasm.sreg, !waveasm.imm<10>) -> !waveasm.sreg

  // SGPR, VGPR → v_mov_b32 (constant bus) + v_cmp_eq_i32.
  // CHECK: waveasm.v_mov_b32
  // CHECK: waveasm.v_cmp_eq_i32
  %cmp_sv = waveasm.arith.cmp eq, %s0, %v0 : (!waveasm.sreg, !waveasm.vreg) -> !waveasm.vreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_select
waveasm.program @test_select
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %v1 = waveasm.precolored.vreg 1 : !waveasm.vreg

  // CHECK: waveasm.v_cmp_lt_i32
  %cmp = waveasm.arith.cmp slt, %v0, %v1 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // CHECK: waveasm.v_cndmask_b32
  %sel = waveasm.arith.select %cmp, %v0, %v1 : (!waveasm.vreg, !waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_wide_narrowing
waveasm.program @test_wide_narrowing
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %s_wide = waveasm.precolored.sreg 4, 2 : !waveasm.sreg<2, 2>

  // Wide SGPR narrowed to low sub-register before VALU add.
  // CHECK: waveasm.precolored.sreg 4 : !waveasm.sreg
  // CHECK: waveasm.v_add_u32
  %add = waveasm.arith.add %s_wide, %v0 : (!waveasm.sreg<2, 2>, !waveasm.vreg) -> !waveasm.vreg

  // Explicit trunc of wide SGPR.
  // CHECK: waveasm.precolored.sreg 4 : !waveasm.sreg
  %trunc = waveasm.arith.trunc %s_wide : (!waveasm.sreg<2, 2>) -> !waveasm.sreg

  // Wide SGPR in cmp: narrowed, then moved to VGPR for constant bus.
  // CHECK: waveasm.precolored.sreg 4 : !waveasm.sreg
  // CHECK: waveasm.v_mov_b32
  // CHECK: waveasm.v_cmp_lt_i32
  %cmp = waveasm.arith.cmp slt, %s_wide, %v0 : (!waveasm.sreg<2, 2>, !waveasm.vreg) -> !waveasm.vreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}
