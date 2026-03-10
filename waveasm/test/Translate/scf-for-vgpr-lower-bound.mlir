// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: scf.for with a VGPR lower bound (from arith.minsi/select).
//
// In split-K kernels the loop lower bound can come from arith.select
// (v_cndmask_b32), producing a VGPR.  The loop induction variable must
// be an SGPR (s_add_u32 / s_cmp_lt_u32), so the backend must insert a
// v_readfirstlane_b32 to convert the VGPR lower bound to SGPR.

module {
  gpu.module @test_vgpr_lower_bound {

    // CHECK-LABEL: waveasm.program @vgpr_lower_bound_loop
    gpu.func @vgpr_lower_bound_loop() kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index

      // arith.minsi produces a VGPR (v_min_i32)
      %lb = arith.minsi %c8, %c16 : index

      // Lower bound is VGPR -> must be converted to SGPR via readfirstlane
      // CHECK:      waveasm.v_min_i32
      // CHECK:      %[[LB:.*]] = waveasm.v_readfirstlane_b32 %{{.*}} : !waveasm.vreg -> !waveasm.sreg
      // CHECK:      waveasm.loop (%{{.*}} = %[[LB]]
      %init = arith.constant 0 : i32
      %result = scf.for %i = %lb to %c16 step %c1
          iter_args(%acc = %init) -> (i32) {
        %i_i32 = arith.index_cast %i : index to i32
        %new_acc = arith.addi %acc, %i_i32 : i32
        scf.yield %new_acc : i32
      }

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
