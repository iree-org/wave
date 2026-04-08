// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: SALU promotion of scalar arithmetic. When both operands of an arith op
// are in SGPRs (e.g. workgroup_id), the auto-select emit helpers route through
// SALU instructions instead of VALU.

module {
  gpu.module @test_salu_promotion {

    // CHECK-LABEL: waveasm.program @salu_mul_add
    gpu.func @salu_mul_add() kernel {
      %wg_x = gpu.block_id x upper_bound 4
      %wg_y = gpu.block_id y upper_bound 4
      %c128 = arith.constant 128 : index

      // Scalar (SGPR) * immediate -> s_mul_i32
      // CHECK: waveasm.s_mul_i32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<128> -> !waveasm.sreg
      %prod = arith.muli %wg_x, %c128 : index

      // Scalar (SGPR) + scalar (SGPR) -> s_add_u32
      // CHECK: waveasm.s_add_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg, !waveasm.scc
      %sum = arith.addi %prod, %wg_y : index

      // CHECK: waveasm.s_endpgm
      gpu.return
    }

    // CHECK-LABEL: waveasm.program @salu_cmpi
    gpu.func @salu_cmpi() kernel {
      %wg_x = gpu.block_id x upper_bound 16
      %c10 = arith.constant 10 : index

      // Scalar cmpi -> s_cmp (SCC result). The immediate is first moved to SGPR.
      // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<10> -> !waveasm.sreg
      // CHECK: waveasm.s_cmp_lt_i32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.sreg -> !waveasm.scc
      %cmp = arith.cmpi slt, %wg_x, %c10 : index

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
