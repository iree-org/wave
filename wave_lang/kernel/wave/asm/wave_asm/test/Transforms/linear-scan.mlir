// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test linear scan register allocation with multi-register ranges and alignment

// Test 1: Basic allocation - single registers
// CHECK-LABEL: waveasm.program @basic_alloc
waveasm.program @basic_alloc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Precolored registers should keep their physical assignment
  // CHECK: waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  // CHECK: waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Virtual register should be allocated to physical
  // CHECK: waveasm.v_add_u32 {{.*}} -> !waveasm.pvreg<
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 2: Multi-register allocation (4 consecutive VGPRs with 4-alignment)
// CHECK-LABEL: waveasm.program @multi_reg_alloc
waveasm.program @multi_reg_alloc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Vector of 4 registers with 4-alignment - must start at v0, v4, v8, etc.
  // Since v0 is precolored, should allocate at v4
  // CHECK: -> !waveasm.pvreg<4, 4>
  %vec = waveasm.v_mov_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// Test 3: SGPR multi-register allocation (2 consecutive SGPRs for 64-bit pointer)
// CHECK-LABEL: waveasm.program @sgpr_pair_alloc
waveasm.program @sgpr_pair_alloc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Precolored SGPR pair (kernarg pointer at s[0:1])
  // CHECK: waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %karg = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>

  // Another SGPR pair - should be allocated after s[0:1]
  // The s_load returns an SGPR, testing allocation
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  // CHECK: waveasm.s_load_dwordx2 {{.*}} -> !waveasm.psreg<{{[0-9]+}}, 2>
  %ptr = waveasm.s_load_dwordx2 %karg, %c0 : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<2>

  waveasm.s_endpgm
}

// Test 4: MFMA accumulator (16 consecutive VGPRs)
// CHECK-LABEL: waveasm.program @mfma_acc_alloc
waveasm.program @mfma_acc_alloc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %v4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>

  // MFMA with 16-register accumulator
  // CHECK: waveasm.v_mfma_f32_32x32x8_f16 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 16>
  %acc = waveasm.v_mfma_f32_32x32x8_f16 %v0, %v4, %c0 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.imm<0> -> !waveasm.vreg<16>

  waveasm.s_endpgm
}

// Test 5: Multiple overlapping ranges
// CHECK-LABEL: waveasm.program @overlapping_ranges
waveasm.program @overlapping_ranges target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Create several values with overlapping lifetimes
  // CHECK: -> !waveasm.pvreg<
  %a = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  // CHECK: -> !waveasm.pvreg<
  %b = waveasm.v_mul_lo_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  // Both %a and %b are live here - need different registers
  // CHECK: -> !waveasm.pvreg<
  %c = waveasm.v_add_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
