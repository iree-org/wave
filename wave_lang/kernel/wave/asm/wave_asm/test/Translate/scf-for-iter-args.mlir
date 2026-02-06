// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: scf.for with iter_args (loop-carried values) translated to
// waveasm.loop with correct multi-iter_arg handling.

module {
  gpu.module @test_loop_iter_args {

    // --- Single iter_arg: accumulator pattern (sum += i) ---
    // CHECK-LABEL: waveasm.program @accumulator_loop
    gpu.func @accumulator_loop() kernel {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %init_sum = arith.constant 0 : i32

      // Two inits: sreg counter (from scf.for IV) + vreg accumulator
      // CHECK:      %[[I0:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
      // CHECK-NEXT: %[[A0:.*]] = waveasm.v_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.vreg
      // CHECK-NEXT: %{{.*}}:2 = waveasm.loop (%[[IV:.*]] = %[[I0]], %[[ACC:.*]] = %[[A0]]) : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
      %final_sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%sum = %init_sum) -> (i32) {
        %i_i32 = arith.index_cast %i : index to i32
        // Accumulate: vreg acc + sreg IV
        // CHECK:      %[[NEWSUM:.*]] = waveasm.v_add_u32 %[[ACC]], %[[IV]] : !waveasm.vreg, !waveasm.sreg -> !waveasm.vreg
        %new_sum = arith.addi %sum, %i_i32 : i32
        scf.yield %new_sum : i32
      }
      // Increment, compare, condition with both iter_args
      // CHECK:      %[[NEXT:.*]] = waveasm.s_add_u32 %[[IV]], %{{.*}} : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
      // CHECK-NEXT: %[[CMP:.*]] = waveasm.s_cmp_lt_u32 %[[NEXT]], %{{.*}} : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
      // CHECK-NEXT: waveasm.condition %[[CMP]] : !waveasm.sreg iter_args(%[[NEXT]], %[[NEWSUM]]) : !waveasm.sreg, !waveasm.vreg

      // CHECK: waveasm.s_endpgm
      gpu.return
    }

    // --- Two iter_args: a' = a + b, b' = a << 1 ---
    // CHECK-LABEL: waveasm.program @multi_iter_args
    gpu.func @multi_iter_args() kernel {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %init_a = arith.constant 1 : i32
      %init_b = arith.constant 2 : i32

      // Three inits: sreg counter + vreg a + vreg b
      // CHECK:      %{{.*}}:3 = waveasm.loop (%{{.*}} = %{{.*}}, %[[A:.*]] = %{{.*}}, %[[B:.*]] = %{{.*}}) : (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg) {
      %final_a, %final_b = scf.for %i = %c0 to %c8 step %c1
          iter_args(%a = %init_a, %b = %init_b) -> (i32, i32) {
        // a' = a + b (vreg + vreg)
        // CHECK:      %[[NEWA:.*]] = waveasm.v_add_u32 %[[A]], %[[B]] : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
        %new_a = arith.addi %a, %b : i32
        // b' = a << 1 (shift left by 1)
        // CHECK:      %[[NEWB:.*]] = waveasm.v_lshlrev_b32 %{{.*}}, %[[A]] : !waveasm.imm<1>, !waveasm.vreg -> !waveasm.vreg
        %two = arith.constant 1 : i32
        %new_b = arith.shli %a, %two : i32
        scf.yield %new_a, %new_b : i32, i32
      }
      // Condition feeds back all three iter_args
      // CHECK:      waveasm.condition %{{.*}} : !waveasm.sreg iter_args(%{{.*}}, %[[NEWA]], %[[NEWB]]) : !waveasm.sreg, !waveasm.vreg, !waveasm.vreg

      // CHECK: waveasm.s_endpgm
      gpu.return
    }

    // --- Nested loops with iter_args threading ---
    // CHECK-LABEL: waveasm.program @nested_loops
    gpu.func @nested_loops() kernel {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %init = arith.constant 0 : i32

      // Outer loop: sreg counter + vreg accumulator
      // CHECK:      %{{.*}}:2 = waveasm.loop (%{{.*}} = %{{.*}}, %[[OACC:.*]] = %{{.*}}) : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
      %outer_result = scf.for %i = %c0 to %c2 step %c1 iter_args(%outer_acc = %init) -> (i32) {
        // Inner loop: accumulator threads from outer -> inner init
        // CHECK:      %[[INNER:.*]]:2 = waveasm.loop (%{{.*}} = %{{.*}}, %{{.*}} = %[[OACC]]) : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
        %inner_result = scf.for %j = %c0 to %c2 step %c1 iter_args(%inner_acc = %outer_acc) -> (i32) {
          %one = arith.constant 1 : i32
          %next = arith.addi %inner_acc, %one : i32
          scf.yield %next : i32
        }
        // Inner condition
        // CHECK:      waveasm.condition %{{.*}} : !waveasm.sreg iter_args(%{{.*}}, %{{.*}}) : !waveasm.sreg, !waveasm.vreg
        scf.yield %inner_result : i32
      }
      // Outer condition uses inner loop result (#1) for accumulator
      // CHECK:      waveasm.condition %{{.*}} : !waveasm.sreg iter_args(%{{.*}}, %[[INNER]]#1) : !waveasm.sreg, !waveasm.vreg

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
