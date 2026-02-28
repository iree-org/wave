// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Regression test: Verify that loop results do not inflate peak VGPR pressure.
//
// This test creates a reduced double-buffered GEMM pattern with enough
// accumulator iter_args and loop body data to exceed 256 VGPRs if loop
// results incorrectly overlap the loop body. With the fix (loop result
// def points after the loop), the accumulators are only counted once
// and regalloc succeeds.
//
// Without the fix: 30 vreg<4,4> loop results (120 VGPRs) + 30 vreg<4,4>
// block args (120 VGPRs) + body data (~40 VGPRs) = ~280 peak > 256 limit.
// With the fix: 30 vreg<4,4> block args (120 VGPRs) + body data (~40 VGPRs)
// = ~160 peak < 256 limit.

// CHECK-LABEL: waveasm.program @double_buffer_pressure
// CHECK-NOT: Failed to allocate
waveasm.program @double_buffer_pressure
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c16 = waveasm.constant 16 : !waveasm.imm<16>

  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %v8 = waveasm.precolored.vreg 8 : !waveasm.pvreg<8>

  // Initialize 30 accumulator vreg<4,4> = 120 VGPRs
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
  %z10 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z11 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z12 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z13 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z14 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z15 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z16 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z17 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z18 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z19 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z20 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z21 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z22 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z23 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z24 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z25 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z26 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z27 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z28 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z29 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.loop
  %ri, %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10, %r11, %r12, %r13, %r14, %r15, %r16, %r17, %r18, %r19, %r20, %r21, %r22, %r23, %r24, %r25, %r26, %r27, %r28, %r29 = waveasm.loop(
      %i = %init_i,
      %acc0 = %z0, %acc1 = %z1, %acc2 = %z2, %acc3 = %z3,
      %acc4 = %z4, %acc5 = %z5, %acc6 = %z6, %acc7 = %z7,
      %acc8 = %z8, %acc9 = %z9, %acc10 = %z10, %acc11 = %z11,
      %acc12 = %z12, %acc13 = %z13, %acc14 = %z14, %acc15 = %z15,
      %acc16 = %z16, %acc17 = %z17, %acc18 = %z18, %acc19 = %z19,
      %acc20 = %z20, %acc21 = %z21, %acc22 = %z22, %acc23 = %z23,
      %acc24 = %z24, %acc25 = %z25, %acc26 = %z26, %acc27 = %z27,
      %acc28 = %z28, %acc29 = %z29)
      : (!waveasm.sreg,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    // MFMA chains: accumulate into each accumulator
    %n0 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc0 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n1 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc1 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n2 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc2 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n3 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc3 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n4 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc4 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n5 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc5 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n6 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc6 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n7 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc7 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n8 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc8 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n9 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc9 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n10 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc10 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n11 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc11 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n12 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc12 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n13 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc13 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n14 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc14 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n15 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc15 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n16 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc16 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n17 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc17 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n18 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc18 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n19 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc19 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n20 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc20 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n21 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc21 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n22 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc22 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n23 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc23 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n24 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc24 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n25 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc25 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n26 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc26 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n27 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc27 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n28 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc28 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n29 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc29 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    // Loop counter
    %next_i:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i#0, %c16 : !waveasm.sreg, !waveasm.imm<16> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i#0,
        %n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7,
        %n8, %n9, %n10, %n11, %n12, %n13, %n14, %n15,
        %n16, %n17, %n18, %n19, %n20, %n21, %n22, %n23,
        %n24, %n25, %n26, %n27, %n28, %n29)
        : !waveasm.sreg,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  // Use a few loop results after the loop (forces them to be live in epilogue)
  // CHECK: waveasm.v_add_u32
  %use0 = waveasm.v_add_u32 %r0, %v8 : !waveasm.vreg<4, 4>, !waveasm.pvreg<8> -> !waveasm.vreg
  %use1 = waveasm.v_add_u32 %r15, %v8 : !waveasm.vreg<4, 4>, !waveasm.pvreg<8> -> !waveasm.vreg
  %use2 = waveasm.v_add_u32 %r29, %v8 : !waveasm.vreg<4, 4>, !waveasm.pvreg<8> -> !waveasm.vreg

  waveasm.s_endpgm
}
