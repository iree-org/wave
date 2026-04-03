// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: scf.for loop translation to WaveASM loop with induction variable
// handling, body translation, and condition terminator.

module {
  gpu.module @test_loop {
    // CHECK-LABEL: waveasm.program @loop_kernel
    gpu.func @loop_kernel() kernel {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index

      // Induction variable init via s_mov_b32, loop carries sreg
      // CHECK:      %[[INIT:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
      // CHECK-NEXT: %{{.*}} = waveasm.loop (%[[IV:.*]] = %[[INIT]]) : (!waveasm.sreg) -> !waveasm.sreg {
      scf.for %i = %c0 to %c4 step %c1 {
        // Body: arith.addi %i, %c1 -> v_add_u32 using block arg
        // CHECK:      waveasm.v_add_u32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.vreg
        %sum = arith.addi %i, %c1 : index
      }
      // Increment, compare, condition
      // CHECK:      %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
      // CHECK-NEXT: %[[CMP:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
      // CHECK-NEXT: waveasm.condition %[[CMP]] : !waveasm.sreg iter_args(%[[NEXT]]) : !waveasm.sreg

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
