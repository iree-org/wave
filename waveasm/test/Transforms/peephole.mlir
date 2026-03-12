// RUN: waveasm-translate --waveasm-peephole %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: lshl + add -> v_mad_u32_u24 fusion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_lshl_add_fusion
waveasm.program @test_lshl_add_fusion target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %base = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %offset = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // Shift by 2 (multiply by 4)
  %c2 = waveasm.constant 2 : !waveasm.imm<2>
  %shifted = waveasm.v_lshlrev_b32 %c2, %base : !waveasm.imm<2>, !waveasm.pvreg<0> -> !waveasm.vreg

  // Add offset - this should be fused with the shift into v_lshl_add_u32
  // CHECK: waveasm.v_lshl_add_u32
  %result = waveasm.v_add_u32 %shifted, %offset : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg

  // Use the result so it's not eliminated as dead code
  waveasm.buffer_store_dword %result, %srd, %offset : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<1>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Add zero elimination
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_add_zero
waveasm.program @test_add_zero target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %src = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // Add zero should be eliminated, result should just use %src directly
  // CHECK-NOT: waveasm.v_add_u32
  // CHECK: waveasm.buffer_store_dword
  %result = waveasm.v_add_u32 %src, %c0 : !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg

  // Use the result
  waveasm.buffer_store_dword %result, %srd, %src : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Multiply by one elimination
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_mul_one
waveasm.program @test_mul_one target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %src = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // Multiply by 1 should be eliminated
  // CHECK-NOT: waveasm.v_mul_lo_u32
  // CHECK: waveasm.buffer_store_dword
  %result = waveasm.v_mul_lo_u32 %src, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg

  // Use the result
  waveasm.buffer_store_dword %result, %srd, %src : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Multiply by power of 2 -> shift
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_mul_pow2
waveasm.program @test_mul_pow2 target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %src = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // Multiply by 8 should become shift by 3
  // CHECK: waveasm.v_lshlrev_b32
  // CHECK-NOT: waveasm.v_mul_lo_u32
  %result = waveasm.v_mul_lo_u32 %src, %c8 : !waveasm.pvreg<0>, !waveasm.imm<8> -> !waveasm.vreg

  // Use the result
  waveasm.buffer_store_dword %result, %srd, %src : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Multiply by zero -> zero constant
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_mul_zero
waveasm.program @test_mul_zero target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %src = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // Multiply by 0 should become a move of 0
  // CHECK: waveasm.v_mov_b32
  // CHECK-NOT: waveasm.v_mul_lo_u32
  %result = waveasm.v_mul_lo_u32 %src, %c0 : !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg

  // Use the result
  waveasm.buffer_store_dword %result, %srd, %src : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Multiply by negative power of 2 -> shift + negate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_mul_neg_pow2
waveasm.program @test_mul_neg_pow2 target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %src = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // First move -64 to VGPR (this is how translation emits it)
  %cn64_imm = waveasm.constant -64 : !waveasm.imm<-64>
  %cn64 = waveasm.v_mov_b32 %cn64_imm : !waveasm.imm<-64> -> !waveasm.vreg

  // Multiply by -64 should become: shift by 6 + negate
  // CHECK: waveasm.v_lshlrev_b32
  // CHECK: waveasm.v_sub_u32
  // CHECK-NOT: waveasm.v_mul_lo_u32
  %result = waveasm.v_mul_lo_u32 %src, %cn64 : !waveasm.pvreg<0>, !waveasm.vreg -> !waveasm.vreg

  // Use the result
  waveasm.buffer_store_dword %result, %srd, %src : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Multiply by negative power of 2 (-16) -> shift + negate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_mul_neg16
waveasm.program @test_mul_neg16 target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %src = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // Negative constant directly as operand
  %cn16 = waveasm.constant -16 : !waveasm.imm<-16>

  // Multiply by -16 should become: shift by 4 + negate
  // CHECK: waveasm.v_lshlrev_b32
  // CHECK: waveasm.v_sub_u32
  // CHECK-NOT: waveasm.v_mul_lo_u32
  %result = waveasm.v_mul_lo_u32 %src, %cn16 : !waveasm.pvreg<0>, !waveasm.imm<-16> -> !waveasm.vreg

  // Use the result
  waveasm.buffer_store_dword %result, %srd, %src : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: (x >> N) << N -> x & ~((1 << N) - 1)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_shr_shl_to_and
waveasm.program @test_shr_shl_to_and target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %src = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // floor(x / 16) * 16 = (x >> 4) << 4 should become x & 0xFFFFFFF0
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %shr = waveasm.v_lshrrev_b32 %c4, %src : !waveasm.imm<4>, !waveasm.pvreg<0> -> !waveasm.vreg
  // CHECK: waveasm.v_and_b32
  // CHECK-NOT: waveasm.v_lshlrev_b32
  %shl = waveasm.v_lshlrev_b32 %c4, %shr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

  waveasm.buffer_store_dword %shl, %srd, %src : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: a + (0 - b) -> a - b
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_add_neg_to_sub
waveasm.program @test_add_neg_to_sub target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // a + (0 - b) should become a - b (single v_sub_u32)
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %neg = waveasm.v_sub_u32 %c0, %b : !waveasm.imm<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  // CHECK: waveasm.v_sub_u32 %{{.*}}, %{{.*}} : !waveasm.pvreg<0>, !waveasm.pvreg<1>
  // CHECK-NOT: waveasm.v_add_u32
  %result = waveasm.v_add_u32 %a, %neg : !waveasm.pvreg<0>, !waveasm.vreg -> !waveasm.vreg

  waveasm.buffer_store_dword %result, %srd, %a : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: BufferLoadLDSSoffsetPattern — direct scalar shift in voffset.
//
//   v_lshlrev_b32(4, sgpr) + v_add_u32(shifted, vbase)
//   → s_lshl_b32(sgpr, 4) into soffset, vbase as voffset.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_lds_soffset_direct
waveasm.program @test_lds_soffset_direct target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %vbase = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %sgpr = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  %shifted = waveasm.v_lshlrev_b32 %c4, %sgpr : !waveasm.imm<4>, !waveasm.sreg -> !waveasm.vreg
  %combined = waveasm.v_add_u32 %shifted, %vbase : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  // Scalar shift extracted into soffset; vbase used directly.
  // CHECK: waveasm.s_lshl_b32
  // CHECK-NOT: waveasm.v_lshlrev_b32
  // CHECK-NOT: waveasm.v_add_u32
  // CHECK: waveasm.buffer_load_dword_lds {{.*}} : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.sreg
  waveasm.buffer_load_dword_lds %combined, %srd, %c0 : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: BufferLoadLDSSoffsetPattern — nested v_add_u32 (Case 2).
//
//   v_add_u32(v_add_u32(thread, v_lshlrev(4, sgpr)), row)
//   → s_lshl_b32(sgpr, 4) into soffset, v_add_u32(thread, row) as voffset.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_lds_soffset_nested
waveasm.program @test_lds_soffset_nested target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %thread = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %row = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %sgpr = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  %shifted = waveasm.v_lshlrev_b32 %c4, %sgpr : !waveasm.imm<4>, !waveasm.sreg -> !waveasm.vreg
  %inner = waveasm.v_add_u32 %thread, %shifted : !waveasm.pvreg<0>, !waveasm.vreg -> !waveasm.vreg
  %outer = waveasm.v_add_u32 %inner, %row : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg

  // Scalar shift extracted; remaining VGPR bases combined into new voffset.
  // CHECK: waveasm.s_lshl_b32
  // CHECK: waveasm.v_add_u32 {{.*}} : !waveasm.pvreg<0>, !waveasm.pvreg<1>
  // CHECK: waveasm.buffer_load_dwordx4_lds {{.*}} : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.sreg
  waveasm.buffer_load_dwordx4_lds %outer, %srd, %c0 : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: BufferLoadLDSSoffsetPattern — shift in v_lshl_add_u32 addend (Case 3).
//
//   v_add_u32(v_lshl_add_u32(vgpr, 2, v_lshlrev(4, sgpr)), row)
//   → s_lshl_b32 into soffset, v_lshl_add_u32(vgpr, 2, 0)+row as voffset.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_lds_soffset_lshl_add
waveasm.program @test_lds_soffset_lshl_add target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %vgpr = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %row = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c2 = waveasm.constant 2 : !waveasm.imm<2>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %sgpr = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  %shifted = waveasm.v_lshlrev_b32 %c4, %sgpr : !waveasm.imm<4>, !waveasm.sreg -> !waveasm.vreg
  %fused = waveasm.v_lshl_add_u32 %vgpr, %c2, %shifted : !waveasm.pvreg<0>, !waveasm.imm<2>, !waveasm.vreg -> !waveasm.vreg
  %outer = waveasm.v_add_u32 %fused, %row : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg

  // Scalar shift extracted; v_lshl_add_u32 gets zero as addend.
  // CHECK: waveasm.s_lshl_b32
  // CHECK-NOT: waveasm.v_lshlrev_b32
  // CHECK: waveasm.v_lshl_add_u32 {{.*}} : !waveasm.pvreg<0>, !waveasm.imm<2>, !waveasm.imm<0>
  // CHECK: waveasm.buffer_load_dword_lds {{.*}} : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.sreg
  waveasm.buffer_load_dword_lds %outer, %srd, %c0 : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: BufferLoadLDSSoffsetPattern — no transform when soffset non-zero.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_lds_soffset_no_transform
waveasm.program @test_lds_soffset_no_transform target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %vbase = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %sgpr = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %soff = waveasm.s_mov_b32 %c4 : !waveasm.imm<4> -> !waveasm.sreg

  %shifted = waveasm.v_lshlrev_b32 %c4, %sgpr : !waveasm.imm<4>, !waveasm.sreg -> !waveasm.vreg
  %combined = waveasm.v_add_u32 %shifted, %vbase : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  // soffset already non-zero — pattern should not fire.
  // Also, LshlAddPattern is blocked by the LDS guard.
  // CHECK-NOT: waveasm.s_lshl_b32
  // CHECK-NOT: waveasm.v_lshl_add_u32
  // CHECK: waveasm.buffer_load_dword_lds
  waveasm.buffer_load_dword_lds %combined, %srd, %soff : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.sreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: BFEPackIdentityPattern — pack of 4 byte BFEs from same source = nop.
//
//   lshl_or(bfe(x,24,8), 24, lshl_or(bfe(x,16,8), 16,
//     lshl_or(bfe(x,8,8), 8, bfe(x,0,8)))) → x
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_bfe_pack_identity
waveasm.program @test_bfe_pack_identity target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %src = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %c16 = waveasm.constant 16 : !waveasm.imm<16>
  %c24 = waveasm.constant 24 : !waveasm.imm<24>

  // Extract 4 bytes from the same dword.
  %b0 = waveasm.v_bfe_u32 %src, %c0, %c8 : !waveasm.pvreg<0>, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
  %b1 = waveasm.v_bfe_u32 %src, %c8, %c8 : !waveasm.pvreg<0>, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
  %b2 = waveasm.v_bfe_u32 %src, %c16, %c8 : !waveasm.pvreg<0>, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
  %b3 = waveasm.v_bfe_u32 %src, %c24, %c8 : !waveasm.pvreg<0>, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

  // Pack them back — should be replaced with %src.
  %inner = waveasm.v_lshl_or_b32 %b1, %c8, %b0 : !waveasm.vreg, !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg
  %middle = waveasm.v_lshl_or_b32 %b2, %c16, %inner : !waveasm.vreg, !waveasm.imm<16>, !waveasm.vreg -> !waveasm.vreg
  %packed = waveasm.v_lshl_or_b32 %b3, %c24, %middle : !waveasm.vreg, !waveasm.imm<24>, !waveasm.vreg -> !waveasm.vreg

  // The entire BFE+pack chain should be eliminated.
  // CHECK-NOT: waveasm.v_bfe_u32
  // CHECK-NOT: waveasm.v_lshl_or_b32
  // CHECK: waveasm.buffer_store_dword {{.*}} : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.buffer_store_dword %packed, %srd, %src : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: BFEPackIdentityPattern — no transform when BFEs have different sources.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_bfe_pack_no_transform
waveasm.program @test_bfe_pack_no_transform target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %src1 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %src2 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %c16 = waveasm.constant 16 : !waveasm.imm<16>
  %c24 = waveasm.constant 24 : !waveasm.imm<24>

  // BFEs extract from two different source dwords.
  %b0 = waveasm.v_bfe_u32 %src1, %c0, %c8 : !waveasm.pvreg<0>, !waveasm.imm<0>, !waveasm.imm<8> -> !waveasm.vreg
  %b1 = waveasm.v_bfe_u32 %src2, %c8, %c8 : !waveasm.pvreg<1>, !waveasm.imm<8>, !waveasm.imm<8> -> !waveasm.vreg
  %b2 = waveasm.v_bfe_u32 %src1, %c16, %c8 : !waveasm.pvreg<0>, !waveasm.imm<16>, !waveasm.imm<8> -> !waveasm.vreg
  %b3 = waveasm.v_bfe_u32 %src1, %c24, %c8 : !waveasm.pvreg<0>, !waveasm.imm<24>, !waveasm.imm<8> -> !waveasm.vreg

  %inner = waveasm.v_lshl_or_b32 %b1, %c8, %b0 : !waveasm.vreg, !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg
  %middle = waveasm.v_lshl_or_b32 %b2, %c16, %inner : !waveasm.vreg, !waveasm.imm<16>, !waveasm.vreg -> !waveasm.vreg
  %packed = waveasm.v_lshl_or_b32 %b3, %c24, %middle : !waveasm.vreg, !waveasm.imm<24>, !waveasm.vreg -> !waveasm.vreg

  // Pack chain should remain — different source dwords.  All 3 lshl_or survive.
  // CHECK: waveasm.v_lshl_or_b32
  // CHECK: waveasm.v_lshl_or_b32
  // CHECK: waveasm.v_lshl_or_b32
  // CHECK: waveasm.buffer_store_dword
  waveasm.buffer_store_dword %packed, %srd, %src1 : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: BufferLoadLDSSoffsetPattern — reversed operand order in v_add_u32.
//
// Same as Case 1 but with v_add_u32(%vbase, %shifted) instead of
// v_add_u32(%shifted, %vbase).  Verifies commutative operand matching.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_lds_soffset_reversed
waveasm.program @test_lds_soffset_reversed target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %vbase = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %sgpr = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  %shifted = waveasm.v_lshlrev_b32 %c4, %sgpr : !waveasm.imm<4>, !waveasm.sreg -> !waveasm.vreg
  // Reversed operand order: vbase first, shifted second.
  %combined = waveasm.v_add_u32 %vbase, %shifted : !waveasm.pvreg<0>, !waveasm.vreg -> !waveasm.vreg

  // CHECK: waveasm.s_lshl_b32
  // CHECK-NOT: waveasm.v_lshlrev_b32
  // CHECK-NOT: waveasm.v_add_u32
  // CHECK: waveasm.buffer_load_dword_lds {{.*}} : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.sreg
  waveasm.buffer_load_dword_lds %combined, %srd, %c0 : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: BufferLoadLDSSoffsetPattern — no transform when shift base is VGPR.
//
// The pattern requires the shift base to be SGPR (scalar) so it can become
// soffset.  A VGPR shift base should be left alone.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_lds_soffset_vgpr_base
waveasm.program @test_lds_soffset_vgpr_base target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>

  // Shift base is VGPR, not SGPR — cannot go into soffset.
  %shifted = waveasm.v_lshlrev_b32 %c4, %v0 : !waveasm.imm<4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %combined = waveasm.v_add_u32 %shifted, %v1 : !waveasm.vreg, !waveasm.pvreg<1> -> !waveasm.vreg

  // No soffset extraction — shift+add fused by LshlAddPattern instead.
  // CHECK-NOT: waveasm.s_lshl_b32
  // CHECK: waveasm.v_lshl_add_u32
  // CHECK: waveasm.buffer_load_dword_lds
  waveasm.buffer_load_dword_lds %combined, %srd, %c0 : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  waveasm.s_endpgm
}
