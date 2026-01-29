// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test MFMA tied operand handling: result should get same physical register as accumulator

// Test 1: MFMA with VGPR accumulator - result tied to acc
// CHECK-LABEL: waveasm.program @mfma_tied_acc
waveasm.program @mfma_tied_acc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Input operands for MFMA
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  // Initial accumulator - will be allocated a physical register
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[ACC:[0-9]+]], 16>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %acc0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<16>

  // MFMA - result should be tied to acc0, same physical register
  // CHECK: waveasm.v_mfma_f32_32x32x8_f16 {{.*}} -> !waveasm.pvreg<[[ACC]], 16>
  %acc1 = waveasm.v_mfma_f32_32x32x8_f16 %a, %b, %acc0 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<16> -> !waveasm.vreg<16>

  // Second MFMA in chain - result tied to acc1 (same physical register)
  // CHECK: waveasm.v_mfma_f32_32x32x8_f16 {{.*}} -> !waveasm.pvreg<[[ACC]], 16>
  %acc2 = waveasm.v_mfma_f32_32x32x8_f16 %a, %b, %acc1 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<16> -> !waveasm.vreg<16>

  waveasm.s_endpgm
}

// Test 2: MFMA with immediate zero accumulator - no tying needed
// CHECK-LABEL: waveasm.program @mfma_imm_acc
waveasm.program @mfma_imm_acc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %zero = waveasm.constant 0 : !waveasm.imm<0>

  // MFMA with immediate acc - result gets a fresh allocation
  // CHECK: waveasm.v_mfma_f32_32x32x8_f16 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 16>
  %acc = waveasm.v_mfma_f32_32x32x8_f16 %a, %b, %zero : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.imm<0> -> !waveasm.vreg<16>

  waveasm.s_endpgm
}

// Test 3: Multiple independent MFMA chains - each tied within its chain
// CHECK-LABEL: waveasm.program @mfma_two_chains
waveasm.program @mfma_two_chains target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>

  // First chain accumulator
  // CHECK: %[[C1ACC:.*]] = waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[CHAIN1:[0-9]+]], 16>
  %chain1_acc0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<16>

  // Second chain accumulator - different physical register
  // CHECK: %[[C2ACC:.*]] = waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[CHAIN2:[0-9]+]], 16>
  %chain2_acc0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<16>

  // First chain MFMA - tied to chain1_acc0
  // CHECK: waveasm.v_mfma_f32_32x32x8_f16 {{.*}}%[[C1ACC]]{{.*}} -> !waveasm.pvreg<[[CHAIN1]], 16>
  %chain1_acc1 = waveasm.v_mfma_f32_32x32x8_f16 %a, %b, %chain1_acc0 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<16> -> !waveasm.vreg<16>

  // Second chain MFMA - tied to chain2_acc0
  // CHECK: waveasm.v_mfma_f32_32x32x8_f16 {{.*}}%[[C2ACC]]{{.*}} -> !waveasm.pvreg<[[CHAIN2]], 16>
  %chain2_acc1 = waveasm.v_mfma_f32_32x32x8_f16 %a, %b, %chain2_acc0 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<16> -> !waveasm.vreg<16>

  waveasm.s_endpgm
}

// Test 4: 16x16 MFMA variant (4 register accumulator)
// CHECK-LABEL: waveasm.program @mfma_16x16
waveasm.program @mfma_16x16 target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>

  // 4-register accumulator for 16x16 MFMA
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[ACC16:[0-9]+]], 4>
  %acc0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // MFMA 16x16 - result tied to acc0
  // CHECK: waveasm.v_mfma_f32_16x16x16_f16 {{.*}} -> !waveasm.pvreg<[[ACC16]], 4>
  %acc1 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc0 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}
