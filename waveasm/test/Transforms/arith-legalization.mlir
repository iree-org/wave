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

  // VGPR + VGPR -> v_add_u32.
  // CHECK: waveasm.v_add_u32 %{{.*}}, %{{.*}} : !waveasm.vreg, !waveasm.vreg
  %add_vv = waveasm.arith.add %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // SGPR + SGPR -> s_add_u32.
  // CHECK: waveasm.s_add_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.sreg
  %add_ss = waveasm.arith.add %s0, %s1 : (!waveasm.sreg, !waveasm.sreg) -> !waveasm.sreg

  // SGPR + VGPR -> v_add_u32 (SGPR broadcast via v_mov_b32).
  // CHECK: waveasm.v_mov_b32
  // CHECK: waveasm.v_add_u32
  %add_sv = waveasm.arith.add %s0, %v0 : (!waveasm.sreg, !waveasm.vreg) -> !waveasm.vreg

  // SGPR + imm -> s_add_u32.
  // CHECK: waveasm.s_add_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<42>
  %add_si = waveasm.arith.add %s0, %c42 : (!waveasm.sreg, !waveasm.imm<42>) -> !waveasm.sreg

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

  // VGPR * imm -> v_mul_lo_u32.
  // CHECK: waveasm.v_mul_lo_u32
  %mul_vi = waveasm.arith.mul %v0, %c42 : (!waveasm.vreg, !waveasm.imm<42>) -> !waveasm.vreg

  // SGPR * SGPR -> s_mul_i32.
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

  // VGPR, VGPR -> v_cmp_lt_i32 (sets VCC).
  // CHECK: waveasm.v_cmp_lt_i32
  %cmp_vv = waveasm.arith.cmp slt, %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // SGPR, imm -> s_cmp_lt_u32 (sets SCC).
  // CHECK: waveasm.s_cmp_lt_u32
  %cmp_si = waveasm.arith.cmp ult, %s0, %c10 : (!waveasm.sreg, !waveasm.imm<10>) -> !waveasm.sreg

  // SGPR, VGPR -> v_mov_b32 (constant bus) + v_cmp_eq_i32.
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

  // Trunc of precolored wide SGPR -> precolored lo half.
  // CHECK: waveasm.precolored.sreg 4 : !waveasm.sreg
  %trunc = waveasm.arith.trunc %s_wide : (!waveasm.sreg<2, 2>) -> !waveasm.sreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// i64 tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_add_i64_salu
waveasm.program @test_add_i64_salu
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %a = waveasm.precolored.sreg 0, 2 : !waveasm.sreg<2, 2>
  %b = waveasm.precolored.sreg 2, 2 : !waveasm.sreg<2, 2>

  // Precolored SGPR pairs: split into lo/hi via precolored.sreg at known indices.
  // CHECK-DAG: waveasm.precolored.sreg 0 : !waveasm.sreg
  // CHECK-DAG: waveasm.precolored.sreg 1 : !waveasm.sreg
  // CHECK-DAG: waveasm.precolored.sreg 2 : !waveasm.sreg
  // CHECK-DAG: waveasm.precolored.sreg 3 : !waveasm.sreg
  // Carry chain: s_add_u32 (lo) then s_addc_u32 (hi).
  // CHECK: waveasm.s_add_u32
  // CHECK-NEXT: waveasm.s_addc_u32
  // CHECK: waveasm.pack
  %add = waveasm.arith.add %a, %b : (!waveasm.sreg<2, 2>, !waveasm.sreg<2, 2>) -> !waveasm.sreg<2, 2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_add_i64_valu
waveasm.program @test_add_i64_valu
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %a = waveasm.precolored.vreg 0 : !waveasm.vreg
  %b = waveasm.precolored.vreg 1 : !waveasm.vreg
  %va = waveasm.pack %a, %a : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>
  %vb = waveasm.pack %b, %b : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  // VGPR i64 add: splitI64 looks through pack, uses originals directly.
  // CHECK: waveasm.v_add_co_u32
  // CHECK-NEXT: waveasm.v_addc_co_u32
  // CHECK: waveasm.pack
  %add = waveasm.arith.add %va, %vb : (!waveasm.vreg<2>, !waveasm.vreg<2>) -> !waveasm.vreg<2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_mul_i64_salu
waveasm.program @test_mul_i64_salu
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %a = waveasm.precolored.sreg 0, 2 : !waveasm.sreg<2, 2>
  %b = waveasm.precolored.sreg 2, 2 : !waveasm.sreg<2, 2>

  // Schoolbook i64 multiply: lo = mul_lo, hi = mul_hi + cross1 + cross2.
  // CHECK: waveasm.s_mul_i32
  // CHECK: waveasm.s_mul_hi_u32
  // CHECK: waveasm.s_mul_i32
  // CHECK: waveasm.s_mul_i32
  // CHECK: waveasm.s_add_u32
  // CHECK: waveasm.s_add_u32
  // CHECK: waveasm.pack
  %mul = waveasm.arith.mul %a, %b : (!waveasm.sreg<2, 2>, !waveasm.sreg<2, 2>) -> !waveasm.sreg<2, 2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_mul_i64_valu
waveasm.program @test_mul_i64_valu
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %a = waveasm.precolored.vreg 0 : !waveasm.vreg
  %b = waveasm.precolored.vreg 1 : !waveasm.vreg
  %va = waveasm.pack %a, %a : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>
  %vb = waveasm.pack %b, %b : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  // VALU schoolbook: mul_lo, mul_hi, 2x mul_lo, v_add3_u32.
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_add3_u32
  // CHECK: waveasm.pack
  %mul = waveasm.arith.mul %va, %vb : (!waveasm.vreg<2>, !waveasm.vreg<2>) -> !waveasm.vreg<2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_sext
waveasm.program @test_sext
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s0 = waveasm.precolored.sreg 0 : !waveasm.sreg
  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg

  // SGPR sext: hi = s_ashr_i32(lo, 31), then merge.
  // CHECK: waveasm.s_ashr_i32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<31>
  // CHECK: waveasm.pack
  %sext_s = waveasm.arith.sext %s0 : (!waveasm.sreg) -> !waveasm.sreg<2, 2>

  // VGPR sext: hi = v_ashrrev_i32(31, lo), then pack.
  // CHECK: waveasm.v_ashrrev_i32 %{{.*}}, %{{.*}} : !waveasm.imm<31>, !waveasm.vreg
  // CHECK: waveasm.pack
  %sext_v = waveasm.arith.sext %v0 : (!waveasm.vreg) -> !waveasm.vreg<2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_zext
waveasm.program @test_zext
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %s0 = waveasm.precolored.sreg 0 : !waveasm.sreg
  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg

  // SGPR zext: hi = s_mov_b32(0), then merge.
  // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0>
  // CHECK: waveasm.pack
  %zext_s = waveasm.arith.zext %s0 : (!waveasm.sreg) -> !waveasm.sreg<2, 2>

  // VGPR zext: hi = v_mov_b32(0), then pack.
  // CHECK: waveasm.v_mov_b32 %{{.*}} : !waveasm.imm<0>
  // CHECK: waveasm.pack
  %zext_v = waveasm.arith.zext %v0 : (!waveasm.vreg) -> !waveasm.vreg<2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_cmp_i64_eq
waveasm.program @test_cmp_i64_eq
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %a = waveasm.precolored.sreg 0, 2 : !waveasm.sreg<2, 2>
  %b = waveasm.precolored.sreg 2, 2 : !waveasm.sreg<2, 2>

  // i64 eq: XOR each half, OR, compare to 0.
  // CHECK: waveasm.s_xor_b32
  // CHECK: waveasm.s_xor_b32
  // CHECK: waveasm.s_or_b32
  // CHECK: waveasm.s_cmp_eq_i32
  %cmp = waveasm.arith.cmp eq, %a, %b : (!waveasm.sreg<2, 2>, !waveasm.sreg<2, 2>) -> !waveasm.sreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_cmp_i64_slt_valu
waveasm.program @test_cmp_i64_slt_valu
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %v1 = waveasm.precolored.vreg 1 : !waveasm.vreg
  %va = waveasm.pack %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>
  %vb = waveasm.pack %v1, %v1 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  // Ordered i64 slt (VALU): hi signed + lo unsigned + select on hi equality.
  // CHECK: waveasm.v_cmp_lt_i32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_cmp_lt_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_cmp_eq_i32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_cmp_ne_i32
  %cmp = waveasm.arith.cmp slt, %va, %vb : (!waveasm.vreg<2>, !waveasm.vreg<2>) -> !waveasm.vreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_cmp_i64_slt_salu
waveasm.program @test_cmp_i64_slt_salu
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %a = waveasm.precolored.sreg 0, 2 : !waveasm.sreg<2, 2>
  %b = waveasm.precolored.sreg 2, 2 : !waveasm.sreg<2, 2>

  // Ordered i64 slt (SALU): hiLt | (hiEq & loLt).
  // CHECK: waveasm.s_cmp_lt_i32
  // CHECK: waveasm.s_cmp_eq_i32
  // CHECK: waveasm.s_cmp_lt_u32
  // CHECK: waveasm.s_and_b32
  // CHECK: waveasm.s_or_b32
  %cmp = waveasm.arith.cmp slt, %a, %b : (!waveasm.sreg<2, 2>, !waveasm.sreg<2, 2>) -> !waveasm.sreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_select_i64
waveasm.program @test_select_i64
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %v1 = waveasm.precolored.vreg 1 : !waveasm.vreg
  %va = waveasm.pack %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>
  %vb = waveasm.pack %v1, %v1 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  // Set up VCC with a compare first.
  // CHECK: waveasm.v_cmp_lt_i32
  %cmp = waveasm.arith.cmp slt, %v0, %v1 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg

  // i64 select: split, select each half, pack.
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.pack
  %sel = waveasm.arith.select %cmp, %va, %vb : (!waveasm.vreg<2>, !waveasm.vreg<2>, !waveasm.vreg) -> !waveasm.vreg<2>

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @test_trunc_i64
waveasm.program @test_trunc_i64
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 16 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
  %va = waveasm.pack %v0, %v0 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  // Trunc i64 VGPR from pack -> returns pack input directly (no extract).
  // CHECK-NOT: waveasm.extract
  %trunc = waveasm.arith.trunc %va : (!waveasm.vreg<2>) -> !waveasm.vreg

  // CHECK-NOT: waveasm.arith.
  waveasm.s_endpgm
}
