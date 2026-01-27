// RUN: waveasm-translate --waveasm-peephole %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: lshl + add -> v_mad_u32_u24 fusion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_lshl_add_fusion
waveasm.program @test_lshl_add_fusion target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
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
waveasm.program @test_add_zero target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
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
waveasm.program @test_mul_one target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
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
waveasm.program @test_mul_pow2 target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
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
waveasm.program @test_mul_zero target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
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
