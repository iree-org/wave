// RUN: waveasm-translate %s | FileCheck %s
//
// Test: Translation from SCF dialect to region-based WaveASM control flow.
// Verifies scf.for -> waveasm.loop and scf.if -> waveasm.if with correct
// SSA threading, iter_args, and condition patterns.
//
// With SALU promotion, arith.cmpi on scalar operands emits s_cmp (SCC result),
// so waveasm.if gets an SCC condition and the output is in custom form.

module {
  gpu.module @test_scf_translation {

    // --- scf.for(0, 16, 1) -> waveasm.loop with SGPR induction variable ---
    // CHECK-LABEL: waveasm.program @scf_for_to_loop
    gpu.func @scf_for_to_loop() kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index

      // Init materialised via s_mov_b32, loop carries single sreg
      // CHECK:      waveasm.s_mov_b32
      // CHECK:      waveasm.loop
      scf.for %i = %c0 to %c16 step %c1 {
        %i_i32 = arith.index_cast %i : index to i32
      }
      // Induction variable incremented, compared, condition terminates
      // CHECK:      %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
      // CHECK-NEXT: %[[CMP:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<16> -> !waveasm.scc
      // CHECK-NEXT: waveasm.condition %[[CMP]] : !waveasm.scc iter_args(%[[NEXT]]) : !waveasm.sreg

      // CHECK: waveasm.s_endpgm
      gpu.return
    }

    // --- scf.for with iter_args -> waveasm.loop with two iter_args ---
    // CHECK-LABEL: waveasm.program @scf_for_with_iter_args
    gpu.func @scf_for_with_iter_args() kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %init = arith.constant 0 : i32

      // Two inits: sreg counter + vreg accumulator
      // CHECK:      waveasm.s_mov_b32
      // CHECK:      waveasm.v_mov_b32
      // CHECK:      waveasm.loop
      %result = scf.for %i = %c0 to %c16 step %c1
          iter_args(%acc = %init) -> (i32) {
        %i_i32 = arith.index_cast %i : index to i32
        %new_acc = arith.addi %acc, %i_i32 : i32
        scf.yield %new_acc : i32
      }
      // Body accumulates: vreg + sreg
      // CHECK:      waveasm.v_add_u32
      // Induction variable incremented, compared, condition with both iter_args
      // CHECK:      %[[NEXT:.*]], %{{.*}} = waveasm.s_add_u32 %{{.*}}, %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
      // CHECK-NEXT: %[[CMP:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<16> -> !waveasm.scc
      // CHECK-NEXT: waveasm.condition %[[CMP]] : !waveasm.scc iter_args(%[[NEXT]], %{{.*}}) : !waveasm.sreg, !waveasm.vreg

      // CHECK: waveasm.s_endpgm
      gpu.return
    }

    // --- scf.if -> waveasm.if with then/else branches ---
    // CHECK-LABEL: waveasm.program @scf_if_to_wave_if
    gpu.func @scf_if_to_wave_if() kernel {
      %arg0 = arith.constant 5 : i32
      %arg1 = arith.constant 3 : i32
      %c10 = arith.constant 10 : i32
      %cond_i32 = arith.cmpi slt, %arg0, %c10 : i32
      %cond_ext = arith.extui %cond_i32 : i1 to i32

      // SALU promotion: scalar cmpi produces SCC directly
      // CHECK:      waveasm.s_cmp_lt_i32
      // CHECK:      %{{.*}} = waveasm.if %{{.*}} : !waveasm.scc -> !waveasm.vreg {
      %result = scf.if %cond_i32 -> i32 {
        // CHECK:      waveasm.v_add_u32
        %sum = arith.addi %arg0, %arg1 : i32
        // CHECK:      waveasm.yield
        scf.yield %sum : i32
      } else {
        // CHECK:      waveasm.v_sub_u32
        %diff = arith.subi %arg0, %arg1 : i32
        // CHECK:      waveasm.yield
        scf.yield %diff : i32
      }

      // CHECK: waveasm.s_endpgm
      gpu.return
    }

    // --- Nested scf.for -> nested waveasm.loop ---
    // CHECK-LABEL: waveasm.program @nested_scf_loops
    gpu.func @nested_scf_loops() kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index

      // Outer loop: sreg counter
      // CHECK:      waveasm.loop
      scf.for %i = %c0 to %c4 step %c1 {
        // Inner loop: sreg counter
        // CHECK:      waveasm.loop
        scf.for %j = %c0 to %c8 step %c1 {
          // Body uses both outer and inner IVs
          // CHECK:      waveasm.s_add_u32
          %sum = arith.addi %i, %j : index
        }
        // Inner condition
        // CHECK:      waveasm.condition %{{.*}} : !waveasm.scc iter_args(%{{.*}}) : !waveasm.sreg
      }
      // Outer condition
      // CHECK:      waveasm.condition %{{.*}} : !waveasm.scc iter_args(%{{.*}}) : !waveasm.sreg

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
