// RUN: not waveasm-translate --waveasm-linear-scan='max-vgprs=32 max-sgprs=104 max-agprs=32' %s 2>&1 | FileCheck %s
//
// Verify that exceeding the register budget produces a single clear error
// message with the budget and the amount used, instead of many repeated
// "register index is out of range" assembler errors.
//
// This kernel deliberately creates many concurrent live vreg<4,4> values
// (10 x 4 = 40 VGPRs minimum) against a 32-VGPR budget.

// CHECK: error: Failed to allocate VGPR: kernel requires
// CHECK-SAME: but only 32 are available
// CHECK-NOT: register index is out of range
// CHECK-NOT: Failed to allocate

waveasm.program @overflow_test
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  // 10 accumulator values, each 4 VGPRs wide -> 40 VGPRs needed.
  // With the budget set to 32, allocation must fail.
  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %z0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z2 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z3 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z4 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z5 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z6 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z7 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z8 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z9 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %ri, %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9 = waveasm.loop(
      %i = %init_i,
      %acc0 = %z0, %acc1 = %z1, %acc2 = %z2, %acc3 = %z3,
      %acc4 = %z4, %acc5 = %z5, %acc6 = %z6, %acc7 = %z7,
      %acc8 = %z8, %acc9 = %z9)
      : (!waveasm.sreg,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    %n0 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc0 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n1 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc1 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n2 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc2 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n3 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc3 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n4 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc4 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n5 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc5 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n6 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc6 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n7 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc7 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n8 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc8 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n9 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc9 : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i#0, %c8 : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i#0, %n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7, %n8, %n9)
        : !waveasm.sreg,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  %use = waveasm.v_add_u32 %r0, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}
