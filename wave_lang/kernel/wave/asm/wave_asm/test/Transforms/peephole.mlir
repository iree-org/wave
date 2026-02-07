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
