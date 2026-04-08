// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: Scalar cmpi + scalar select fusion into s_cmp + s_cselect_b32.
// When both comparison operands are scalar and the select's true/false values
// are also scalar, the backend fuses the pair into a single s_cmp + s_cselect
// sequence, avoiding the VALU v_cmp + v_cndmask path.

module {
  gpu.module @test_select_fusion {

    // CHECK-LABEL: waveasm.program @cmpi_select_scalar_fusion
    gpu.func @cmpi_select_scalar_fusion() kernel {
      %wg_x = gpu.block_id x upper_bound 16
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c200 = arith.constant 200 : index

      // Scalar cmpi + scalar select -> s_cmp_lt_i32 + s_cselect_b32
      // CHECK: waveasm.s_cmp_lt_i32
      // CHECK: waveasm.s_cselect_b32
      // CHECK-NOT: waveasm.v_cmp
      // CHECK-NOT: waveasm.v_cndmask
      %cmp = arith.cmpi slt, %wg_x, %c10 : index
      %sel = arith.select %cmp, %c100, %c200 : index

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
