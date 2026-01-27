// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test scf.for loop translation with iter_args (loop-carried values)

module {
  gpu.module @test_loop_iter_args {
    // Test 1: Simple accumulator loop with iter_args
    // CHECK: waveasm.program @accumulator_loop
    gpu.func @accumulator_loop() kernel {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %init_sum = arith.constant 0 : i32

      // Loop with single iter_arg (accumulator pattern)
      // CHECK: waveasm.label @L_loop_
      %final_sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%sum = %init_sum) -> (i32) {
        // Accumulate: sum += i
        %i_i32 = arith.index_cast %i : index to i32
        // CHECK: waveasm.v_add_u32
        %new_sum = arith.addi %sum, %i_i32 : i32
        scf.yield %new_sum : i32
      }
      // CHECK: waveasm.s_add_u32
      // CHECK: waveasm.s_cmp_lt_u32
      // CHECK: waveasm.s_cbranch_scc1

      // CHECK: waveasm.s_endpgm
      gpu.return
    }

    // Test 2: Loop with multiple iter_args
    // CHECK: waveasm.program @multi_iter_args
    gpu.func @multi_iter_args() kernel {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %init_a = arith.constant 1 : i32
      %init_b = arith.constant 2 : i32

      // Loop with two iter_args
      // CHECK: waveasm.label @L_loop_
      %final_a, %final_b = scf.for %i = %c0 to %c8 step %c1
          iter_args(%a = %init_a, %b = %init_b) -> (i32, i32) {
        // a' = a + b
        // b' = a * 2
        // CHECK: waveasm.v_add_u32
        %new_a = arith.addi %a, %b : i32
        // CHECK: waveasm.v_lshlrev_b32
        %two = arith.constant 1 : i32
        %new_b = arith.shli %a, %two : i32
        scf.yield %new_a, %new_b : i32, i32
      }
      // CHECK: waveasm.s_add_u32
      // CHECK: waveasm.s_cbranch_scc1

      // CHECK: waveasm.s_endpgm
      gpu.return
    }

    // Test 3: Nested loops with iter_args
    // CHECK: waveasm.program @nested_loops
    gpu.func @nested_loops() kernel {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %init = arith.constant 0 : i32

      // Outer loop
      // CHECK: waveasm.label @L_loop_
      %outer_result = scf.for %i = %c0 to %c2 step %c1 iter_args(%outer_acc = %init) -> (i32) {
        // Inner loop
        // CHECK: waveasm.label @L_loop_
        %inner_result = scf.for %j = %c0 to %c2 step %c1 iter_args(%inner_acc = %outer_acc) -> (i32) {
          %one = arith.constant 1 : i32
          %next = arith.addi %inner_acc, %one : i32
          scf.yield %next : i32
        }
        scf.yield %inner_result : i32
      }

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
