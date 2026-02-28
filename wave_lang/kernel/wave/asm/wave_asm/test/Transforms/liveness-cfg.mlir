// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s
//
// Test: CFG-based liveness analysis with loop.
// Verifies scf.for translates to waveasm.loop with correct induction variable
// threading and condition terminator structure.

module {
  gpu.module @test_liveness_loop {
    // CHECK-LABEL: waveasm.program @liveness_loop_kernel
    gpu.func @liveness_loop_kernel() kernel {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index

      // Init via s_mov_b32, loop carries single sreg
      // CHECK:      %[[INIT:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
      // CHECK-NEXT: %{{.*}} = waveasm.loop (%[[IV:.*]] = %[[INIT]]) : (!waveasm.sreg) -> !waveasm.sreg {
      scf.for %i = %c0 to %c4 step %c1 {
        // Body uses block arg
        // CHECK:      waveasm.v_add_u32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.vreg
        %sum = arith.addi %i, %c1 : index
      }
      // Increment, compare, condition with iter_arg
      // CHECK:      %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
      // CHECK-NEXT: %[[CMP:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
      // CHECK-NEXT: waveasm.condition %[[CMP]] : !waveasm.sreg iter_args(%[[NEXT]]) : !waveasm.sreg

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
