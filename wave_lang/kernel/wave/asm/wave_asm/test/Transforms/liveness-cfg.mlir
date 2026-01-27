// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s
//
// Test CFG-based liveness analysis with loop

module {
  gpu.module @test_liveness_loop {
    // CHECK: waveasm.program @liveness_loop_kernel
    gpu.func @liveness_loop_kernel() kernel {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index

      // CHECK: waveasm.s_mov_b32
      // CHECK: waveasm.label
      scf.for %i = %c0 to %c4 step %c1 {
        // Use the induction variable inside loop
        // CHECK: waveasm.v_add_u32
        %sum = arith.addi %i, %c1 : index
      }
      // CHECK: waveasm.s_cbranch_scc1

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
