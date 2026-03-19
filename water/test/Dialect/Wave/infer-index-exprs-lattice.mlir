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

    // expected-error @below {{conflict when propagating index expressions from result to operand #0}}
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

    // expected-error @below {{conflict when propagating index expressions from operand to result #0}}
    // expected-note @below {{original result lattice}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
      // expected-error @below {{incompatible operand lattices when propagating from those to result}}
      // expected-note @below {{operand #0 lattice}}
      // expected-note @below {{operand #1 lattice}}
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
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
    // The vectorShape should be preserved and match.
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 4 : i32}],
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 4 : i32}]
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
    // The vectorShape from operand 0 should be preserved.
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 8 : i32}],
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
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice:}}
    // expected-note @below {{operand #1 lattice:}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 4 : i32}],
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 8 : i32}]
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
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
    // expected-error @below {{conflict when propagating index expressions from init to result #0}}
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
