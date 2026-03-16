// RUN: water-opt %s --water-test-wave-infer-index-exprs --split-input-file --verify-diagnostics | FileCheck %s

//
// This file contains tests for index expression dataflow analyses that require
// injecting specific lattice states via attributes. These are predominantly
// checking the index conflict detection when propagating across a single
// operation, i.e., when a lattice first reaches the top state from non-top
// states.
//

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @add_then_mul(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@M, @K] of f16>,
    %c: !wave.tensor<[@M, @K] of f16>
  ) -> !wave.tensor<[@M, @K] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %add = wave.add %a, %b {wave_test.override_result_index = [[1,{
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
      K = #wave.index_mapping<[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    // expected-error @below {{conflict when propagating index expressions from result #0 to operand #0}}
    // expected-note @below {{original operand lattice}}
    // expected-note @below {{result #0 lattice}}
    %mul = wave.mul %add, %c {wave_test.override_result_index = [[1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>,
      K = #wave.index_mapping<[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %mul : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  func.func @add_then_mul(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@M, @K] of f16>,
    %c: !wave.tensor<[@M, @K] of f16>
  ) -> !wave.tensor<[@M, @K] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %add = wave.add %a, %b {wave_test.override_result_index = [[1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
      K = #wave.index_mapping<[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    // expected-error @below {{conflict when propagating index expressions from operand #0 to result #0}}
    // expected-note @below {{original result lattice}}
    // expected-note @below {{operand #0 lattice}}
    %mul = wave.mul %add, %c {wave_test.override_result_index = [[1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>,
      K = #wave.index_mapping<[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %mul : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  func.func @operand_conflict(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@M, @K] of f16>,
    %c: !wave.tensor<[@M, @K] of f16>
  ) -> !wave.tensor<[@M, @K] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{operand #1 lattice}}
    // expected-note @below {{original result lattice}}
    %add = wave.add %a, %b {wave_test.override_operand_index = [[1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
      K = #wave.index_mapping<[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }], [1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 44, 1, 1)>,
      K = #wave.index_mapping<[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    return %add : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

// Generic error message when reached top somehow without detecting the conflict before.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict detected in index expressions for result #0}}
    // expected-note @below {{PLEASE REPORT}}
    %result = wave.add %a, %b {wave_test.override_result_index = ["<top>"]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %result : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// Joining with the same expression results in that expression.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_with_same
  func.func @join_with_same(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining with null (uninitialized) doesn't crash and gives the other expression.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_with_null
  func.func @join_with_null(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (<NULL>, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining with bottom (denoted as unit) gives the other expression.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_with_bottom
  func.func @join_with_bottom(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }],
       unit  // will default-initialize to bottom.
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining with zero is gives the other expression.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_with_zero
  func.func @join_with_zero(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (0, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Additional constant summand makes expressions join to top.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{original result lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40 + 1, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different constant summands join to top.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{original result lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40 + 2, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40 + 1, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different constant values other than zero join to top.
// Also, difference may be in the step.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{original result lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1,{
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 3, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 2, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Difference in stride joins to top.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{original result lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 2)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 3)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Stride 1 joins with the other constant stride to become that stride.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 2)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 2)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Step 1 joins with the other non-constant step to become that step.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 40, T0, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, T0, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different expressions in step join to top even if they would have resulted in a sum for start.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{original result lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0, T0, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (T0, WG0, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different expressions involving threads join to top.
// Note that here the underlying affine expression is the same, but symbols
// are different, we should be able to catch that.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{original result lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T1>] -> (T1 * 40, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different expressions involving workgroups join to top.
// Note that there are unused symbols in mappings.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @simple_add(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{original result lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>] -> (WG0, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>] -> (WG1, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}


// -----

// Joining thread and block components is fine. Note that some symbols are unused in mappings.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_threads_workgroups
  func.func @join_threads_workgroups(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 + T0, 1, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (T0, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Identical constant summands don't sum up when symbols do.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @same_constant_summands
  func.func @same_constant_summands(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 + T0 + 2, 1, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<WG0>] -> (WG0 + 2, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 + 2 , 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining thread and block components is fine, this requires aligning symbols in mappings.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_threads_workgroups_align
  func.func @join_threads_workgroups_align(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 + T0, 1, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<WG0>] -> (WG0, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0, 1, 1)>
    }]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining iter symbols and blocks is fine and results in an add.
// TODO: Also check that iter symbols don't leak form the loop to results.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_iter_workgroups
  func.func @join_iter_workgroups(
    %a: !wave.tensor<[@M, @K] of f32>,
    %b: !wave.tensor<[@M, @K] of f32>
  ) -> !wave.tensor<[@M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %result = wave.iterate @K iter_args(%a) {
    ^bb0(%a_arg: !wave.tensor<[@M, @K] of f32>):
      // CHECK: wave.add
      // CHECK-SAME: index
      // CHECK-SAME: M : <[#wave.index_symbol<WG0>, #wave.iter<"K">] -> (WG0 + _Iter_K, 1, 1)>
      %partial_result = wave.add %a_arg, %b {wave_test.override_operand_index = [[1, {
        M = #wave.index_mapping<[#wave.index_symbol<WG0>] -> (WG0, 1, 1)>
      }], [1, {
        M = #wave.index_mapping<[#wave.iter<"K">] -> (_Iter_K, 1, 1)>
      }]]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Joining iter symbols with themselves is fine.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_iter_same
  func.func @join_iter_same(
    %a: !wave.tensor<[@M, @K] of f32>,
    %b: !wave.tensor<[@M, @K] of f32>
  ) -> !wave.tensor<[@M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %result = wave.iterate @K iter_args(%a) {
    ^bb0(%a_arg: !wave.tensor<[@M, @K] of f32>):
      // CHECK: wave.add
      // CHECK-SAME: index
      // CHECK-SAME: M : <[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)
      %partial_result = wave.add %a_arg, %b {wave_test.override_operand_index = [[1, {
        M = #wave.index_mapping<[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }], [1, {
        M = #wave.index_mapping<[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }]]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Joining different iter symbols with is fine and results in a sum.
// Also check that we are not leaking iter symbols to operations after the loop
// by checking that they are not used in expressions for loop results.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_iters
  func.func @join_iters(
    %a: !wave.tensor<[@M, @K] of f32>,
    %b: !wave.tensor<[@M, @K] of f32>
  ) -> !wave.tensor<[@M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.iterate @M
    // CHECK-SAME: index
    // CHECK-SAME: M = #wave.index_mapping<[] -> (0, 1, 1)>
    %result = wave.iterate @M iter_args(%b) {
    ^bb0(%b_arg: !wave.tensor<[@M, @K] of f32>):
      // CHECK: wave.iterate @K
      // CHECK-SAME: index
      // CHECK-SAME: M = #wave.index_mapping<[#wave.iter<"M">] -> (_Iter_M, 1, 1)>
      %inner_result = wave.iterate @K iter_args(%a) {
      ^bb1(%a_arg: !wave.tensor<[@M, @K] of f32>):
        // CHECK: wave.add
        // CHECK-SAME: index
        // CHECK-SAME: M : <[#wave.iter<"K">, #wave.iter<"M">] -> (_Iter_K + _Iter_M, 1, 1)>
        %partial_result = wave.add %a_arg, %b_arg {wave_test.override_operand_index = [[1, {
          M = #wave.index_mapping<[#wave.iter<"M">] -> (_Iter_M, 1, 1)>
        }], [1, {
          M = #wave.index_mapping<[#wave.iter<"K">] -> (_Iter_K, 1, 1)>
        }]]}
        : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

        wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
      } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %inner_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Otherwise iter symbols behave like any other component, e.g., different
// expressions involving the same symbol join to top.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @join_iters(
    %a: !wave.tensor<[@M, @K] of f32>,
    %b: !wave.tensor<[@M, @K] of f32>
  ) -> !wave.tensor<[@M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %result = wave.iterate @K iter_args(%a) {
    ^bb0(%a_arg: !wave.tensor<[@M, @K] of f32>):
      // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
      // expected-note @below {{original result lattice:}}
      // expected-note @below {{operand #1 lattice:}}
      %partial_result = wave.add %a_arg, %b {wave_test.override_operand_index = [[1, {
        M = #wave.index_mapping<[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }], [1, {
        M = #wave.index_mapping<[#wave.iter<"K">] -> (_Iter_K * 2, 1, 1)>
      }]]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Check that we don't leak iter symbols to values before the loop.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @do_not_leak_above
  func.func @do_not_leak_above(
    %a: !wave.tensor<[@M, @K] of f32>,
    %b: !wave.tensor<[@M, @K] of f32>
  ) -> !wave.tensor<[@M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Note that indexing over K is still present, but it uses the default (0, 1, 1) value
    // and none of the index expressions use _Iter_K
    // CHECK: wave.read
    // CHECK-DAG: M : <[] -> (42, 1, 1)>
    // CHECK-DAG: K : <[] -> (0, 1, 1)>
    // CHECK: wave.iterate
    %b_reg = wave.read %b : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    %result = wave.iterate @K iter_args(%a) {
    ^bb0(%a_arg: !wave.tensor<[@M, @K] of f32>):
      %partial_result = wave.add %a_arg, %b_reg {wave_test.override_operand_index = [[1, {
        M = #wave.index_mapping<[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }], [1, {
        M = #wave.index_mapping<[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }]]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Check that we propagate lattices between adjacent operands of a write.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @write_sideways_propagation
  func.func @write_sideways_propagation(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>,
    %c: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[] -> (42)>, workgroup_dim = <x>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 1024}>
  } {
    // CHECK: index
    // CHECK: M : <[] -> (42, 1, <NULL>)>
    wave.write %a, %b {wave_test.override_operand_index = [
      unit, [1, {
      M = #wave.index_mapping<[#wave.iter<"K">] -> (<NULL>, 1, <NULL>)>
    }]]
    } : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>
    %c_reg = wave.read %c {wave_test.override_result_index = [[1, {
      M = #wave.index_mapping<[#wave.iter<"K">] -> (42, <NULL>, <NULL>)>
    }]]} : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    wave.write %c_reg, %b : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @priority_join
  func.func @priority_join(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Low priority lattice.
    %a_reg = wave.read %a {wave_test.override_result_index = [[0, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
    }]]} : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    // Higher priority lattice on the other operand. Selected without causing a conflict.
    // CHECK: wave.add
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 20, 1, 1)>
    %sum = wave.add %a_reg, %b {wave_test.override_operand_index = [
      unit,
      [3, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 20, 1, 1)>
      }]
    ]} : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  func.func @same_priority_conflict(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Both operands have same priority but different expressions - should conflict
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{original result lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %sum = wave.add %a, %b {wave_test.override_operand_index = [[1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
    }], [1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 20, 1, 1)>
    }]]} : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    return
  }
}

// -----

// Test that higher priority from write propagates backward through multiple operations.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @priority_backward_through_chain
  func.func @priority_backward_through_chain(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>,
    %output: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.read
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 30, 1, 1)>
    %a_reg = wave.read %a : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    // The override has low priority, it only matters at initialization and is
    // then itself overridden by the higher-priority lattice in backpropagation.
    // CHECK: wave.add
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 30, 1, 1)>
    %sum1 = wave.add %a_reg, %b {wave_test.override_operand_index = [
      unit,
      [0, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 5, 1, 1)>
      }]
    ]} : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    // CHECK: wave.add
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 30, 1, 1)>
    %sum2 = wave.add %sum1, %a_reg : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    // CHECK: wave.write
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 30, 1, 1)>
    wave.write %sum2, %output {wave_test.override_operand_index = [
      unit,
      [3, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 30, 1, 1)>
      }]
    ]} : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>

    return
  }
}

// -----

// Check that sideways propagation between operands of a write that would
// lead to a conflict is not happening.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @write_sideways_no_conflicting_propagation
  func.func @write_sideways_no_conflicting_propagation(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>,
    %c: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [4, 1, 1]>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
      #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 4)>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64 : i64, M = 128}>
  } {
    // CHECK: wave.write
    // CHECK: index
    // CHECK: M : <[] -> (1, <NULL>, <NULL>)>
    wave.write %a, %b {wave_test.override_operand_index = [
      unit, [1, {
      M = #wave.index_mapping<[#wave.iter<"K">] -> (1, <NULL>, <NULL>)>
    }]]
    } : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>
    // CHECK: wave.read
    // CHECK: index
    // CHECK: M : <[] -> (42, <NULL>, <NULL>)>
    %c_reg = wave.read %c {wave_test.override_result_index = [[1, {
      M = #wave.index_mapping<[#wave.iter<"K">] -> (42, <NULL>, <NULL>)>
    }]]} : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    wave.write %c_reg, %b : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>
    return
  }
}

// -----

// Test that priority is propagated when index expressions are
// equal. Both read and add using its result are initialized
// to the same value, read with priority 0 and add with prioity 3.
// Check that this value overrides the one the write is initialized
// with give its priority is only 1.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @propagate_priority_equal_values
  func.func @propagate_priority_equal_values(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>,
    %output: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Read result is at the lowest priority.
    %a_reg = wave.read %a {wave_test.override_result_index = [[0, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
    }]]} : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    // Same value expression but at higher priority.
    // CHECK: wave.add
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
    %sum = wave.add %a_reg, %b {wave_test.override_operand_index = [
      unit,
      [3, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }]
    ]} : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    // Write's own initialization has priority 1 (kWritePriority), but the sum's
    // lattice has priority 3, so it dominates and the write retains T0 * 10.
    // CHECK: wave.write
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
    wave.write %a_reg, %output {wave_test.override_operand_index = [
      unit, [1, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 30, 1, 1)>
      }]
    ]} : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @forward_joins_with_backward
  func.func @forward_joins_with_backward(
    %a: !wave.tensor<[@M] of f32>,
    %output: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1],
                                mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 32}>
    ]
  } {
    // Only provide the start expression for "read". It will have to join with the whole
    // index expression for write during initialization and set non-null step and stride.
    // CHECK: wave.read
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 16, 1, 1)>
    %a_reg = wave.read %a {wave_test.override_result_index = [[1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 16, <NULL>, <NULL>)>
    }]]} : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    // Add a write operation for %a_reg to the %output function argument
    // CHECK: wave.write
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 16, 1, 1)>
    wave.write %a_reg, %output : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>

    return %a_reg : !wave.tensor<[@M] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @forward_backward_conflict(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1],
                                mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 32}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[] -> (64)>, workgroup_dim = <x>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128}>
  } {
    // Set the result lattice to one with a different offset than what is implied by
    // write, which should trigger an error.
    %a_reg = wave.read %a {wave_test.override_result_index = [[1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 16, 32, 32)>
    }]]} : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    // expected-error @below {{conflict for value to store index expression when propagating from implied by write operation lattice}}
    // expected-note @below {{original value to store lattice}}
    // expected-note @below {{implied by write operation lattice}}
    wave.write %a_reg, %b : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @write_memory_vs_implied(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1],
                                mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 32}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[] -> (64)>, workgroup_dim = <x>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128}>
  } {
    %a_reg = wave.read %a : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    %b_bad = wave.read %b {wave_test.override_result_index = [[1, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 16, 32, 32)>
    }]]} : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>

    // Override the value-to-store operand with the index expression implied by
    // write to avoid sideways propagation. This will hit an error during
    // initialization rather than during propagation.
    // expected-error @below {{conflict for memory index expression when propagating from implied by write operation lattice}}
    // expected-note @below {{original memory lattice}}
    // expected-note @below {{implied by write operation lattice}}
    wave.write %a_reg, %b_bad {wave_test.override_operand_index = [[1, {
      M = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (T0 mod 64 + WG0 * 64, 1, 1)>
      }]]} : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @mma_lhs_vs_implied(
    %lhs: !wave.tensor<[@M, @K] of f16>,
    %rhs: !wave.tensor<[@N, @K] of f16>,
    %acc: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1],
                                mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                vector_shapes = {M = 16, N = 16, K = 16}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[] -> (64)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[] -> (64)>, workgroup_dim = <y>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, K = 128}>
  } {
    %lhs_bad = wave.read %lhs {wave_test.override_result_index = [[3, {
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 16, 32, 32)>
    }]]} : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    // expected-error @below {{conflict for LHS index expression when propagating from implied by MMA kind lattice}}
    // expected-note @below {{original LHS lattice}}
    // expected-note @below {{implied by MMA kind lattice}}
    %r = wave.mma %lhs_bad, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>}
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @mma_lhs_vs_implied_vector_shape(
    %lhs: !wave.tensor<[@M, @K] of f16>,
    %rhs: !wave.tensor<[@N, @K] of f16>,
    %acc: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1],
                                mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                vector_shapes = {M = 16, N = 16, K = 16}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[] -> (64)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[] -> (64)>, workgroup_dim = <y>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, K = 128}>
  } {
    %lhs_bad = wave.read %lhs {wave_test.override_result_index = [[3, {
      K = #wave.index_mapping<[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>,
      M = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (T0 mod 16 + WG0 * 64, 1, 1)>
    }, {M = 42, K = 42}]]} : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    // expected-error @below {{conflict for LHS vector shape when propagating from implied by MMA kind lattice}}
    // expected-note @below {{original LHS lattice}}
    // expected-note @below {{implied by MMA kind lattice}}
    %r = wave.mma %lhs_bad, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>}
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @mma_lhs_vs_implied(
    %lhs: !wave.tensor<[@M, @K] of f16>,
    %rhs: !wave.tensor<[@N, @K] of f16>,
    %acc: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1],
                                mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                vector_shapes = {M = 16, N = 16, K = 16}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[] -> (64)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[] -> (64)>, workgroup_dim = <y>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, K = 128}>
  } {
    // Forcibly setting the result lattice to top to check index attribute generation.
    // This triggers an additional error because reaching top through the normal process
    // would have been reported earlier and would have aborted the inference before
    // reaching this point.
    // expected-error @below {{conflict detected in index expressions for mma result}}
    // expected-note @below {{PLEASE REPORT}}
    %r = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>,
      wave_test.override_result_index = ["<top>"]
    }
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// Joining two lattices with the same vectorShape works.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_with_same_vector_shape
  func.func @join_with_same_vector_shape(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    // CHECK-SAME: vector_shape [{M : 4 : i64}]
    // The vectorShape should be preserved and match.
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 4 : i64}],
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 4 : i64}]
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining a lattice with vectorShape with one without vectorShape takes the non-null one.

// TODO: we need to capture the vector shape on the op

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @join_with_one_vector_shape
  func.func @join_with_one_vector_shape(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: M : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    // CHECK-SAME: vector_shape [{M : 8 : i64}]
    // The vectorShape from operand 0 should be preserved.
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 8 : i64}],
      {M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining two lattices with different vectorShapes results in top and emits a diagnostic.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @vector_shape_conflict(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating vector shapes from operand #1 to result #0}}
    // expected-note @below {{original result lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 4 : i64}],
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 8 : i64}]
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

//
// shouldPropagateIndexExprs tests
//
// shouldPropagateIndexExprs(from, to) (WaveInterfaces.cpp) returns false when
// the destination lattice's concrete index keys omit some non-unit dimension
// from the source lattice's vector shape; identityIndexExprsPropagate then
// skips the join for that edge instead of driving the lattice to top.
//

// Forward skip: reciprocal A produces a result with vecShape {M=4, K=4}
// (K is extra, absent from the 1D tensor shape [@M]). Reciprocal B's result
// has only M in its concrete keys (since the tensor is 1D). Forward propagation
// from A.result to B.result: from.vecShape non-unit = {M, K}, to keys = {M},
// K is missing => shouldPropagateIndexExprs returns false => skip.
// B.result keeps M = T0 * 32 from its override, never gets T0 * 99 from A.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  // CHECK-LABEL: @propagation_skip_forward
  func.func @propagation_skip_forward(
    %a: !wave.tensor<[@M] of f16>
  ) -> !wave.tensor<[@M] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.reciprocal
    // CHECK-SAME: T0 * 99
    %t = wave.reciprocal %a {
      wave_test.override_result_index = [
        [1, {M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>},
         {M = 4 : i64, K = 4 : i64}]
      ]
    } : (!wave.tensor<[@M] of f16>) -> !wave.tensor<[@M] of f16>
    // CHECK: wave.reciprocal
    // CHECK-SAME: T0 * 32
    // CHECK-NOT: T0 * 99
    %r = wave.reciprocal %t {
      wave_test.override_result_index = [
        [1, {M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>},
         {M = 4 : i64}]
      ]
    } : (!wave.tensor<[@M] of f16>) -> !wave.tensor<[@M] of f16>
    return %r : !wave.tensor<[@M] of f16>
  }
}

// -----

// Forward propagation through broadcast: same source lattice as above (vecShape
// {M=4, K=4}), but the broadcast result is 2D [@M, @K] so its concrete keys
// include both M and K. shouldPropagateIndexExprs finds all non-unit dims
// covered and returns true, so M = T0 * 99 propagates into the broadcast
// result.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  // CHECK-LABEL: @propagation_through_broadcast
  func.func @propagation_through_broadcast(
    %a: !wave.tensor<[@M] of f16>
  ) -> !wave.tensor<[@M, @K] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.reciprocal
    // CHECK-SAME: T0 * 99
    %t = wave.reciprocal %a {
      wave_test.override_result_index = [
        [1, {M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>},
         {M = 4 : i64, K = 4 : i64}]
      ]
    } : (!wave.tensor<[@M] of f16>) -> !wave.tensor<[@M] of f16>
    // CHECK: wave.broadcast
    // CHECK-SAME: T0 * 99
    %r = wave.broadcast %t
      : (!wave.tensor<[@M] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %r : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

// Backward skip: reciprocal operands have M-only concrete keys with vecShape
// {M=4, K=4}. The add result has both M and K as non-unit dims in its vecShape.
// Backward propagation from the add result into each reciprocal operand:
// from.vecShape non-unit = {M, K}, to keys = {M}, K missing => skip.
// Reciprocals keep M = T0 * 32; the add result retains M = T0 * 99.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes {wave_test.disable_forward} {
  // CHECK-LABEL: @propagation_skip_backward_partial_operand
  func.func @propagation_skip_backward_partial_operand(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@M, @K] of f16>
  ) -> !wave.tensor<[@M, @K] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.reciprocal
    // CHECK-SAME: M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>
    %a_rp = wave.reciprocal %a {
      wave_test.override_operand_index = [
        [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>},
         {M = 4 : i64, K = 4 : i64}]
      ]} : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    // CHECK: wave.add
    // CHECK-SAME: M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
    %result = wave.add %a_rp, %a_rp {
      wave_test.override_result_index = [
        [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>},
         {M = 4 : i64, K = 4 : i64}]
      ]
    } : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %result : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

//
// Tests for reduction operations (wave.max_element, wave.min_element, etc.)
//

// Test that reduction forward propagation keeps only symbols present in result shape
// and propagates init unchanged.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  // CHECK-LABEL: @reduction_forward_propagation
  func.func @reduction_forward_propagation(
    %input: !wave.tensor<[@N, @M] of f32>,
    %init: !wave.tensor<[@N] of f32>
  ) -> !wave.tensor<[@N] of f32, <register>> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Reduction result has shape [@N]; init has shape [@N]; input has [@N, @M].
    // Forward propagation should keep only N from input, and propagate init fully.
    // CHECK: wave.max_element
    // CHECK-SAME: index
    // CHECK-SAME: N : <[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>
    %result = wave.max_element %input init(%init) <warp> {wave_test.override_operand_index = [
      {N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
       M = #wave.index_mapping<[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>},
      {N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>}
    ]} : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32, <register>>
    return %result : !wave.tensor<[@N] of f32, <register>>
  }
}

// -----

// Test that reduction backward propagation propagates to init keeping only init symbols,
// and propagates to reduced value as identity.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @reduction_backward_propagation
  func.func @reduction_backward_propagation(
    %input: !wave.tensor<[@N, @M] of f32>,
    %init: !wave.tensor<[@N] of f32>
  ) -> !wave.tensor<[@N] of f32, <register>> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Backward propagation should propagate result to both operands,
    // but for init, keep only symbols in init shape.
    // CHECK: wave.max_element
    // CHECK-SAME: index
    // CHECK-SAME: N : <[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>
    %result = wave.max_element %input init(%init) <warp> {wave_test.override_result_index = [{
      N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>
    }]} : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32, <register>>
    return %result : !wave.tensor<[@N] of f32, <register>>
  }
}

// -----

// Test reduction with conflicting index expressions in forward propagation

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  func.func @reduction_forward_conflict(
    %input: !wave.tensor<[@N, @M] of f32>,
    %init: !wave.tensor<[@N] of f32>
  ) -> !wave.tensor<[@N] of f32, <register>> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Input and init have different N expressions - should conflict
    // expected-error @below {{conflict when propagating index expressions from init #0 to result #0}}
    // expected-note @below {{original result lattice}}
    // expected-note @below {{init #0 lattice}}
    %result = wave.max_element %input init(%init) <warp> {wave_test.override_operand_index = [
      {N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
       M = #wave.index_mapping<[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>},
      {N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}
    ]} : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32, <register>>
    return %result : !wave.tensor<[@N] of f32, <register>>
  }
}

// -----

// Per-key priority: when two operands have different per-key priorities,
// each key independently takes the higher-priority mapping.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  // CHECK-LABEL: @per_key_priority_independent_keys
  func.func @per_key_priority_independent_keys(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Operand 0 has M at priority 3 and N at priority 1.
    // Operand 1 has M at priority 1 and N at priority 3.
    // Result should pick M from operand 0 and N from operand 1.
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 * 20, 1, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = 3 : i32, N = 1 : i32}, {
         M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
         N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }],
      [{M = 1 : i32, N = 3 : i32}, {
         M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 16, 1, 1)>,
         N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 20, 1, 1)>
      }]
    ]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %result : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// Per-key priority: a low-priority key can be overridden by backward
// propagation from a higher-priority source without affecting the high-priority
// key already present.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @per_key_priority_backward_override
  func.func @per_key_priority_backward_override(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>,
    %output: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Read result: M at pri=3 (high), N at pri=0 (low).
    // N should be overridden by backward propagation from write.
    // CHECK: wave.read
    // CHECK-SAME: index
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    %a_reg = wave.read %a {wave_test.override_result_index = [
      [{M = 3 : i32, N = 0 : i32}, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }]
    ]} : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    %sum = wave.add %a_reg, %b : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

    // Write: M at pri=1 (write default), N at pri=3.
    // N (pri=3) overrides read's N (pri=0) via backpropagation.
    // M (pri=1) does NOT override read's M (pri=3).
    // CHECK: wave.write
    // CHECK-SAME: index
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    wave.write %sum, %output {wave_test.override_operand_index = [
      unit,
      [{M = 1 : i32, N = 3 : i32}, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 50, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
      }]
    ]} : !wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>

    return
  }
}

// -----

// Bitcast forward: i8 (8-bit) to f4 (4-bit) doubles the step on the last
// dimension. Leading dimensions pass through unchanged.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @bitcast_forward_scales_step
  func.func @bitcast_forward_scales_step(
    %src: !wave.tensor<[@M, @K2] of i8, <global>>,
    %dst: !wave.tensor<[@M, @K] of f4E2M1FN, <global>>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 16, K = 128, K2 = #wave.expr_list<[#wave.symbol<"K">] -> (K ceildiv 2)>}>
  } {
    // Inject K2 step=16 on the read result.
    %data = wave.read %src {wave_test.override_result_index = [[0, {
        "K2" = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 2, 16, 1)>,
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    }]]} : (!wave.tensor<[@M, @K2] of i8, <global>>) -> !wave.tensor<[@M, @K2] of i8, <register>>

    // After bitcast i8->f4 (8-bit to 4-bit), the K2 step of 16 should become
    // K step of 32 (doubled). M passes through unchanged.
    // CHECK: wave.bitcast
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (T0 * 2, 32, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %bitcasted = wave.bitcast %data
      : !wave.tensor<[@M, @K2] of i8, <register>> to !wave.tensor<[@M, @K] of f4E2M1FN, <register>>

    // CHECK: wave.write
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (T0 * 2, 32, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    wave.write %bitcasted, %dst
      : !wave.tensor<[@M, @K] of f4E2M1FN, <register>>, !wave.tensor<[@M, @K] of f4E2M1FN, <global>>

    return
  }
}

// -----

// Per-key priority: same priority, same expression for some keys but different
// for others - the differing key should cause a conflict.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  func.func @per_key_same_priority_partial_conflict(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // M matches between operands, but N differs - should conflict.
    // expected-error @below {{conflict when propagating index expressions from operand #1 to result #0}}
    // expected-note @below {{original result lattice}}
    // expected-note @below {{operand #1 lattice}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
       N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
    }], [1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
       N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 20, 1, 1)>
    }]]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %result : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// Bitcast backward: f4 (4-bit) result index expressions should scale down
// to i8 (8-bit) operand index expressions (step halved).
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @bitcast_backward_scales_step
  func.func @bitcast_backward_scales_step(
    %src: !wave.tensor<[@M, @K2] of i8, <global>>,
    %dst: !wave.tensor<[@M, @K] of f4E2M1FN, <global>>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 16, K = 128, K2 = #wave.expr_list<[#wave.symbol<"K">] -> (K ceildiv 2)>}>
  } {
    // CHECK: wave.read
    // CHECK-DAG: K2 : <[#wave.index_symbol<T0>] -> (T0 * 2, 16, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %data = wave.read %src
      : (!wave.tensor<[@M, @K2] of i8, <global>>) -> !wave.tensor<[@M, @K2] of i8, <register>>

    // Inject K step=32 on the bitcast result. Backward propagation should
    // produce K2 step=16 (halved) on the read.
    %bitcasted = wave.bitcast %data {wave_test.override_result_index = [[0, {
      K = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 2, 32, 1)>,
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    }]]} : !wave.tensor<[@M, @K2] of i8, <register>> to !wave.tensor<[@M, @K] of f4E2M1FN, <register>>

    wave.write %bitcasted, %dst
      : !wave.tensor<[@M, @K] of f4E2M1FN, <register>>, !wave.tensor<[@M, @K] of f4E2M1FN, <global>>

    return
  }
}

// -----

// Per-key priority: higher priority for one key should override even when the
// other key would conflict at equal priority. The conflicting key is resolved
// by priority, not by structural merge.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_backward } {
  // CHECK-LABEL: @per_key_priority_resolves_conflict
  func.func @per_key_priority_resolves_conflict(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Operand 0: M at pri=1 with T0*10, N at pri=1 with T0*20.
    // Operand 1: M at pri=3 with T0*30, N at pri=3 with T0*40.
    // Higher priority wins for both keys, no structural merge conflict.
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 * 30, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    %result = wave.add %a, %b {wave_test.override_operand_index = [[1, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>,
       N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 20, 1, 1)>
    }], [3, {
       M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 30, 1, 1)>,
       N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }]]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %result : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// Bitcast with f4 (4-bit) to i8 (8-bit) halves the step on the last dimension.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @bitcast_narrow_to_wide_halves_step
  func.func @bitcast_narrow_to_wide_halves_step(
    %src: !wave.tensor<[@M, @K] of f4E2M1FN, <global>>,
    %dst: !wave.tensor<[@M, @K2] of i8, <global>>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 16, K = 128, K2 = #wave.expr_list<[#wave.symbol<"K">] -> (K ceildiv 2)>}>
  } {
    %data = wave.read %src {wave_test.override_result_index = [[0, {
      K = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 2, 32, 1)>,
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    }]]} : (!wave.tensor<[@M, @K] of f4E2M1FN, <global>>) -> !wave.tensor<[@M, @K] of f4E2M1FN, <register>>

    // After bitcast f4->i8 (4-bit to 8-bit), the K step of 32 should become
    // K2 step of 16 (halved). M passes through unchanged.
    // CHECK: wave.bitcast
    // CHECK-DAG: K2 : <[#wave.index_symbol<T0>] -> (T0 * 2, 16, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %bitcasted = wave.bitcast %data
      : !wave.tensor<[@M, @K] of f4E2M1FN, <register>> to !wave.tensor<[@M, @K2] of i8, <register>>

    wave.write %bitcasted, %dst
      : !wave.tensor<[@M, @K2] of i8, <register>>, !wave.tensor<[@M, @K2] of i8, <global>>

    return
  }
}

// -----

// Bitcast with same bitwidth (i8 to f8E8M0FNU): step is preserved.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @bitcast_same_bitwidth_identity
  func.func @bitcast_same_bitwidth_identity(
    %src: !wave.tensor<[@M, @K] of i8, <global>>,
    %dst: !wave.tensor<[@M, @K] of f8E8M0FNU, <global>>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 16, K = 128}>
  } {
    %data = wave.read %src {wave_test.override_result_index = [[0, {
      K = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 2, 16, 1)>,
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    }]]} : (!wave.tensor<[@M, @K] of i8, <global>>) -> !wave.tensor<[@M, @K] of i8, <register>>

    // Same bitwidth: step is preserved, symbol name stays K.
    // CHECK: wave.bitcast
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (T0 * 2, 16, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %bitcasted = wave.bitcast %data
      : !wave.tensor<[@M, @K] of i8, <register>> to !wave.tensor<[@M, @K] of f8E8M0FNU, <register>>

    wave.write %bitcasted, %dst
      : !wave.tensor<[@M, @K] of f8E8M0FNU, <register>>, !wave.tensor<[@M, @K] of f8E8M0FNU, <global>>

    return
  }
}

// -----

// Backward propagation from an elementwise operation should not modify
// the MMA result's index expressions, even when the elementwise result
// has a higher priority lattice.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @no_backward_into_mma_from_elementwise
  func.func @no_backward_into_mma_from_elementwise(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@N, @K] of f16>,
    %c: !wave.tensor<[@M, @N] of f32>,
    %d: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.mma
    // CHECK: }, {
    // CHECK: }, {
    // CHECK: }, {
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %mma = wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
      -> !wave.tensor<[@M, @N] of f32>

    // CHECK: wave.add
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
    %add = wave.add %mma, %d {wave_test.override_result_index = [
      [100, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
      }, {M = 16 : i64, N = 16 : i64}]
    ]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

    return
  }
}

// -----

// Backward propagation from a permute operation should not modify
// the MMA result's index expressions.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @no_backward_into_mma_from_permute
  func.func @no_backward_into_mma_from_permute(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@N, @K] of f16>,
    %c: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.mma
    // CHECK: }, {
    // CHECK: }, {
    // CHECK: }, {
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %mma = wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
      -> !wave.tensor<[@M, @N] of f32>

    // CHECK: wave.permute
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
    %permuted = wave.permute %mma {wave_test.override_result_index = [
      [100, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
      }, {M = 16 : i64, N = 16 : i64}]
    ]}
    : !wave.tensor<[@M, @N] of f32> to !wave.tensor<[@N, @M] of f32>

    return
  }
}

// -----

// Backward propagation from a write operation should not modify
// the MMA result's index expressions.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @no_backward_into_mma_from_write
  func.func @no_backward_into_mma_from_write(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@N, @K] of f16>,
    %c: !wave.tensor<[@M, @N] of f32>,
    %output: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.mma
    // CHECK: }, {
    // CHECK: }, {
    // CHECK: }, {
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %mma = wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
      -> !wave.tensor<[@M, @N] of f32>

    // The write's memory operand has a high-priority override with different
    // expressions. Sideways propagation within the write should not modify the
    // MMA result (the value-to-store).
    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    wave.write %mma, %output {wave_test.override_operand_index = [
      unit,
      [100, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
      }, {M = 16 : i64, N = 16 : i64}]
    ]}
    : !wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>

    return
  }
}

// -----

// Backward propagation from a yield operation in a loop should not modify
// the MMA result's index expressions. The iterate's result gets a high-priority
// lattice from a downstream add; that lattice flows back through the yield but
// should not reach the MMA result.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @no_backward_into_mma_from_yield
  func.func @no_backward_into_mma_from_yield(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@N, @K] of f16>,
    %c: !wave.tensor<[@M, @N] of f32>,
    %d: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.iterate
    %result = wave.iterate @K iter_args(%c) {
    ^bb0(%acc: !wave.tensor<[@M, @N] of f32>):
      // CHECK: wave.mma
      // CHECK: }, {
      // CHECK: }, {
      // CHECK: }, {
      // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
      // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
      %mma = wave.mma %a, %b, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>}
        : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
        -> !wave.tensor<[@M, @N] of f32>

      wave.yield %mma : !wave.tensor<[@M, @N] of f32>
    } : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

    // High-priority override on the iterate result's downstream use.
    // CHECK: wave.add
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
    %add = wave.add %result, %d {wave_test.override_result_index = [
      [100, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 99, 1, 1)>
      }, {M = 16 : i64, N = 16 : i64}]
    ]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

    return
  }
}

// -----

// Overriding result lattice without specifying vector shape results in an error.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] attributes { wave_test.disable_forward } {
  func.func @mma_missing_result_vector_shape(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@N, @K] of f16>,
    %c: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<
        threads_per_wave = 64,
        waves_per_block = [2, 3, 4],
        mma_type = #wave.mma_kind<f32_16x16x16_f16>,
        vector_shapes = {M = 16 : i64, N = 16 : i64, K = 16 : i64}>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, K = 64}>
  } {
    // expected-error @below {{missing vector shape for result}}
    %m = wave.mma %a, %b, %c {
      kind = #wave.mma_kind<f32_16x16x16_f16>,
      wave_test.override_result_index = [[1, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
      }]]
    } : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %m : !wave.tensor<[@M, @N] of f32>
  }
}


// -----

// The test infer pass writes `vector_shape` from lattice storage (via overrides).
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @pass_emits_vector_shape
  func.func @pass_emits_vector_shape(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> !wave.tensor<[@M] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128}>
  } {
    // CHECK: wave.add
    // CHECK-SAME: vector_shape [{M : 7 : i64}]
    %result = wave.add %a, %b {wave_test.override_result_index = [[{
      M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }, {M = 7 : i64}]]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}
