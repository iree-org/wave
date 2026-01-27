// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test scf.for loop translation to WAVEASM instructions

module {
  gpu.module @test_loop {
    // CHECK: waveasm.program @loop_kernel
    gpu.func @loop_kernel() kernel {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index

      // CHECK: waveasm.s_mov_b32
      // CHECK: waveasm.label @L_loop_
      scf.for %i = %c0 to %c4 step %c1 {
        // Loop body: arith.addi uses SGPR loop counter + immediate
        // CHECK: waveasm.v_add_u32 {{.*}} : !waveasm.psreg{{.*}}, !waveasm.imm
        %sum = arith.addi %i, %c1 : index
      }
      // CHECK: waveasm.s_add_u32
      // CHECK: waveasm.s_cmp_lt_u32
      // CHECK: waveasm.s_cbranch_scc1

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
