// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test that loop results do NOT inflate register pressure inside the loop body.
// Loop results are only available after the loop exits, so their live ranges
// should start after the loop, not at the LoopOp itself.

//===----------------------------------------------------------------------===//
// Test 1: Loop results should not inflate body pressure
//===----------------------------------------------------------------------===//

// This test creates a loop with vreg<4,4> iter_args (accumulators) plus
// additional vreg values inside the loop body. If loop results incorrectly
// have their def point at the LoopOp index, they would double-count the
// accumulator VGPRs during the loop body, inflating pressure.
//
// The key check is that regalloc succeeds (no "Failed to allocate" error)
// and that the loop result gets a physical register via tying.

// CHECK-LABEL: waveasm.program @loop_result_no_body_pressure
waveasm.program @loop_result_no_body_pressure
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  // Initialize loop counter and accumulator
  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[ACC:[0-9]+]], 4>
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // Loop: iter_args carry (counter, accumulator)
  // CHECK: waveasm.loop
  %i_out, %acc_out = waveasm.loop(%i = %init_i, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>) -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {

    // Some work inside the loop body using VGPRs
    %tmp1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg
    %tmp2 = waveasm.v_add_u32 %tmp1, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

    // MFMA: result tied to accumulator iter_arg
    // CHECK: waveasm.v_mfma_f32_16x16x16_f16 {{.*}} -> !waveasm.pvreg<[[ACC]], 4>
    %new_acc = waveasm.v_mfma_f32_16x16x16_f16 %v4, %v4, %acc
        : !waveasm.pvreg<4, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    // Loop bookkeeping
    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i, %new_acc) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  // Use the loop result after the loop -- should get same phys reg as accumulator
  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[ACC]], 4>{{.*}} -> !waveasm.pvreg<
  %use = waveasm.v_add_u32 %acc_out, %v0
      : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 2: Loop result tied to block arg (MFMA accumulator chain)
//===----------------------------------------------------------------------===//

// Verify the full tying chain: init_acc -> block_arg -> MFMA result ->
// condition iter_arg -> loop result all share the same physical register.

// CHECK-LABEL: waveasm.program @loop_result_tied_to_block_arg
waveasm.program @loop_result_tied_to_block_arg
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  // Init accumulator
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[TIED_ACC:[0-9]+]], 4>
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  // Loop with MFMA accumulator
  %i_out, %acc_out = waveasm.loop(%i = %init_i, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>) -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {

    // MFMA with tied accumulator -- result MUST get same phys reg
    // CHECK: waveasm.v_mfma_f32_16x16x16_f16 {{.*}} -> !waveasm.pvreg<[[TIED_ACC]], 4>
    %new_acc = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc
        : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c8 : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i, %new_acc) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  // Post-loop use of accumulator result -- same physical register
  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[TIED_ACC]], 4>{{.*}} -> !waveasm.pvreg<
  %final = waveasm.v_add_u32 %acc_out, %a
      : !waveasm.vreg<4, 4>, !waveasm.pvreg<0, 4> -> !waveasm.vreg

  waveasm.s_endpgm
}
