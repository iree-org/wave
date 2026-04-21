// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test: waveasm.if results feeding waveasm.loop init args.
//
// When an if-op result is used as a loop init arg, the register allocator
// ties the if result and the loop block arg to the same physical register.
// The LinearScanPass must set the if-op result type from the allocation
// mapping (not from the then-yield operand type), because the then-yield
// operand may have been allocated to a different physical register (e.g.
// from an inner loop).  Without the fix, the LoopLikeOpInterface verifier
// rejects the mismatch between init arg and region iter_arg types.

// CHECK-LABEL: waveasm.program @if_result_feeds_loop

// The if-op result must get its physical register from the allocation mapping.
// CHECK:       waveasm.if {{.*}} -> !waveasm.pareg<[[IF_REG:[0-9]+]], 4>

// The then-yield carries the if-result register.
// CHECK:         waveasm.v_mfma_scale_f32_16x16x128_f8f6f4 {{.*}} -> !waveasm.pareg<[[IF_REG]], 4>
// CHECK-NEXT:    waveasm.yield {{.*}} : !waveasm.pareg<[[IF_REG]], 4>

// The loop init arg type must exactly match the if-result type (IF_REG).
// CHECK:       waveasm.loop ({{.*}}!waveasm.pareg<[[IF_REG]], 4>) -> ({{.*}}!waveasm.pareg<[[LOOP_REG:[0-9]+]], 4>)

// Inside the loop body, the MFMA uses and produces the loop iter-arg register.
// CHECK:         waveasm.v_mfma_scale_f32_16x16x128_f8f6f4 {{.*}}!waveasm.pareg<[[LOOP_REG]], 4>{{.*}} -> !waveasm.pareg<[[LOOP_REG]], 4>

waveasm.program @if_result_feeds_loop
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %v0 = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %v4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %vs = waveasm.precolored.vreg 8 : !waveasm.pvreg<8>

  %s_zero = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %cmp = waveasm.s_cmp_lt_u32 %s_zero, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.scc

  %if_result = waveasm.if %cmp : !waveasm.scc -> !waveasm.areg<4, 4> {
    %acc_init = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.areg<4, 4>
    %mfma = waveasm.v_mfma_scale_f32_16x16x128_f8f6f4 %v0, %v4, %acc_init, %vs, %vs
        : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.areg<4, 4>, !waveasm.pvreg<8>, !waveasm.pvreg<8> -> !waveasm.areg<4, 4>
    waveasm.yield %mfma : !waveasm.areg<4, 4>
  } else {
    %zero_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.areg<4, 4>
    waveasm.yield %zero_acc : !waveasm.areg<4, 4>
  }

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  %i_out, %acc_out = waveasm.loop(%i = %init_i, %acc = %if_result)
      : (!waveasm.sreg, !waveasm.areg<4, 4>) -> (!waveasm.sreg, !waveasm.areg<4, 4>) {

    %new_mfma = waveasm.v_mfma_scale_f32_16x16x128_f8f6f4 %v0, %v4, %acc, %vs, %vs
        : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.areg<4, 4>, !waveasm.pvreg<8>, !waveasm.pvreg<8> -> !waveasm.areg<4, 4>

    %next_i:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
    %loop_cond = waveasm.s_cmp_lt_u32 %next_i#0, %c4 : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.scc
    waveasm.condition %loop_cond : !waveasm.scc iter_args(%next_i#0, %new_mfma) : !waveasm.sreg, !waveasm.areg<4, 4>
  }

  waveasm.s_endpgm
}
