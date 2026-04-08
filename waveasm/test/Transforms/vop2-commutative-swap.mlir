// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: VOP2 commutative literal swap. For VOP2 instructions (v_and_b32,
// v_or_b32, v_xor_b32), when a non-inline literal appears in src1, the
// emitter swaps operands to place it in src0, avoiding scratch VGPR
// materialization. Non-commutative ops still need materialization.

// CHECK-LABEL: vop2_commute_swap_test:

waveasm.program @vop2_commute_swap_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // v_and_b32 with literal in src1: swap to src0 (commutative)
  %c4096 = waveasm.constant 4096 : !waveasm.imm<4096>
  // CHECK-NOT: v_mov_b32
  // CHECK: v_and_b32 v{{[0-9]+}}, 4096, v0
  %r1 = waveasm.v_and_b32 %v0, %c4096 : !waveasm.pvreg<0>, !waveasm.imm<4096> -> !waveasm.vreg

  // v_or_b32 with literal in src1: swap to src0 (commutative)
  %c256 = waveasm.constant 256 : !waveasm.imm<256>
  // CHECK-NOT: v_mov_b32
  // CHECK: v_or_b32 v{{[0-9]+}}, 256, v0
  %r2 = waveasm.v_or_b32 %v0, %c256 : !waveasm.pvreg<0>, !waveasm.imm<256> -> !waveasm.vreg

  // v_xor_b32 with literal in src1: swap to src0 (commutative)
  %c128 = waveasm.constant 128 : !waveasm.imm<128>
  // CHECK-NOT: v_mov_b32
  // CHECK: v_xor_b32 v{{[0-9]+}}, 128, v0
  %r3 = waveasm.v_xor_b32 %v0, %c128 : !waveasm.pvreg<0>, !waveasm.imm<128> -> !waveasm.vreg

  // v_lshlrev_b32 with literal in src0: literal already in correct position
  %c200 = waveasm.constant 200 : !waveasm.imm<200>
  // CHECK-NOT: v_mov_b32
  // CHECK: v_lshlrev_b32 v{{[0-9]+}}, 200, v0
  %r4 = waveasm.v_lshlrev_b32 %c200, %v0 : !waveasm.imm<200>, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
