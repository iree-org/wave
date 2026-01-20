// RUN: water-opt %s --water-test-wave-infer-index-exprs --split-input-file --verify-diagnostics

//
// This file contains tests for index expression dataflow analyses that require
// injecting specific lattice states via attributes. These are predominantly
// checking the index conflict detection when propagating across a single
// operation, i.e., when a lattice first reaches the top state from non-top
// states.
//

normalform.module [#wave.normal_form<full_types>] {
  func.func @simple_mma(
    %lhs: !wave.tensor<[@M, @K] of f16>,
    %rhs: !wave.tensor<[@N, @K] of f16>,
    %acc: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %lhs_override = wave.read %lhs { wave_test.override_result_index = [
        #wave.index_exprs<[<"N"> : [#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)]>
    ]} : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    // expected-error @below {{conflict when propagating forward to the result lattice in MmaOp}}
    // expected-note @below {{Result lattice}}
    // expected-note @below {{LHS lattice}}
    // expected-note @below {{RHS lattice}}
    // expected-note @below {{Accumulator lattice}}
    %r = wave.mma %lhs_override, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>}
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @simple_mma(
    %lhs: !wave.tensor<[@M, @K] of f16>,
    %rhs: !wave.tensor<[@N, @K] of f16>,
    %acc: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating to LHS from result in MmaOp}}
    // expected-note @below {{LHS lattice}}
    // expected-note @below {{result lattice}}
    %r = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>,
      wave_test.override_result_index = [
        #wave.index_exprs<[<"K"> : [#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)]>
      ]
    }
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @simple_mma(
    %lhs: !wave.tensor<[@M, @K] of f16>,
    %rhs: !wave.tensor<[@N, @K] of f16>,
    %acc: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating to RHS from result in MmaOp}}
    // expected-note @below {{RHS lattice}}
    // expected-note @below {{result lattice}}
    %r = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>,
      wave_test.override_operand_index = [
        unit,
        #wave.index_exprs<[<"N"> : [#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)]>
      ]
    }
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @simple_mma(
    %lhs: !wave.tensor<[@M, @K] of f16>,
    %rhs: !wave.tensor<[@N, @K] of f16>,
    %acc: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error @below {{conflict when propagating to accumulator from result in MmaOp}}
    // expected-note @below {{accumulator lattice}}
    // expected-note @below {{result lattice}}
    %r = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>,
      wave_test.override_operand_index = [
        unit,
        unit,
        #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)]>
      ]
    }
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @add_then_mul(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@M, @K] of f16>,
    %c: !wave.tensor<[@M, @K] of f16>
  ) -> !wave.tensor<[@M, @K] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %add = wave.add %a, %b {wave_test.override_result_index = [
      #wave.index_exprs<[
        <"M"> : [#wave.index_symbol<T0>] -> (T0 * 32, 1, 1),
        <"K"> : [#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)
      ]>
    ]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    // expected-error @below {{conflict when propagating index expressions from result to operand #0}}
    // expected-note @below {{original operand lattice}}
    // expected-note @below {{result #0 lattice}}
    %mul = wave.mul %add, %c {wave_test.override_result_index = [
      #wave.index_exprs<[
        <"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1),
        <"K"> : [#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)
      ]>
    ]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %mul : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] attributes { wave_test.disable_backward } {
  func.func @add_then_mul(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@M, @K] of f16>,
    %c: !wave.tensor<[@M, @K] of f16>
  ) -> !wave.tensor<[@M, @K] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %add = wave.add %a, %b {wave_test.override_result_index = [
      #wave.index_exprs<[
        <"M"> : [#wave.index_symbol<T0>] -> (T0 * 32, 1, 1),
        <"K"> : [#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)
      ]>
    ]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    // expected-error @below {{conflict when propagating index expressions from operand to result #0}}
    // expected-note @below {{original result lattice}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
    %mul = wave.mul %add, %c {wave_test.override_result_index = [
      #wave.index_exprs<[
        <"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1),
        <"K"> : [#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)
      ]>
    ]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %mul : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] attributes { wave_test.disable_backward } {
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
    %add = wave.add %a, %b {wave_test.override_operand_index = [
      #wave.index_exprs<[
        <"M"> : [#wave.index_symbol<T0>] -> (T0 * 32, 1, 1),
        <"K"> : [#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)
      ]>,
      #wave.index_exprs<[
        <"M"> : [#wave.index_symbol<T0>] -> (T0 * 44, 1, 1),
        <"K"> : [#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)
      ]>
    ]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    return %add : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

// Generic error message when reached top somehow without detecting the conflict before.

normalform.module [#wave.normal_form<full_types>] {
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

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: <"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining with null (uninitialized) doesn't crash and gives the other expression.

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: <"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (<NULL>, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining with bottom (denoted as unit) gives the other expression.

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: <"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)]>,
       unit  // will default-initialize to bottom.
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining with zero is gives the other expression.

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: <"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (0, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Additional constant summand makes expressions join to top.

normalform.module [#wave.normal_form<full_types>] {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40 + 1, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different constant summands join to top.

normalform.module [#wave.normal_form<full_types>] {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40 + 2, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40 + 1, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different constant values other than zero join to top.
// Also, difference may be in the step.

normalform.module [#wave.normal_form<full_types>] {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 3, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 2, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Difference in stride joins to top.

normalform.module [#wave.normal_form<full_types>] {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 2)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 3)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Stride 1 joins with the other constant stride to become that stride.

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: <"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 2)
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 2)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Step 1 joins with the other non-constant step to become that step.

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: <"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, T0, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, T0, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different expressions in step join to top even if they would have resulted in a sum for start.

normalform.module [#wave.normal_form<full_types>] {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0, T0, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (T0, WG0, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different expressions involving threads join to top.
// Note that here the underlying affine expression is the same, but symbols
// are different, we should be able to catch that.

normalform.module [#wave.normal_form<full_types>] {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T1>] -> (T1 * 40, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different expressions involving workgroups join to top.
// Note that there are unused symbols in mappings.

normalform.module [#wave.normal_form<full_types>] {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<WG0>, #wave.index_symbol<WG1>] -> (WG0, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<WG0>, #wave.index_symbol<WG1>] -> (WG1, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}


// -----

// Joining thread and block components is fine. Note that some symbols are unused in mappings.

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: index =
    // CHECK-SAME: <"M"> : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 + T0, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (T0, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Identical constant summands don't sum up when symbols do.

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: index =
    // CHECK-SAME: <"M"> : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 + T0 + 2, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<WG0>] -> (WG0 + 2, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0 + 2 , 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining thread and block components is fine, this requires aligning symbols in mappings.

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: index =
    // CHECK-SAME: <"M"> : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 + T0, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<WG0>] -> (WG0, 1, 1)]>,
       #wave.index_exprs<[<"M"> : [#wave.index_symbol<T0>] -> (T0, 1, 1)]>
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining iter symbols and blocks is fine and results in an add.
// TODO: Also check that iter symbols don't leak form the loop to results.

normalform.module [#wave.normal_form<full_types>] {
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
      // CHECK-SAME: <"M"> : [#wave.index_symbol<WG0>, #wave.iter<"K">] -> (WG0 + _Iter_K, 1, 1)
      %partial_result = wave.add %a_arg, %b {wave_test.override_operand_index = [
        #wave.index_exprs<[<"M"> : [#wave.index_symbol<WG0>] -> (WG0, 1, 1)]>,
        #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (_Iter_K, 1, 1)]>
      ]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Joining iter symbols with themselves is fine.

normalform.module [#wave.normal_form<full_types>] {
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
      // CHECK-SAME: <"M"> : [#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)
      %partial_result = wave.add %a_arg, %b {wave_test.override_operand_index = [
        #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)]>,
        #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)]>
      ]}
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

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-SAME: M = #wave<index_mapping[] -> (0, 1, 1)>
    %result = wave.iterate @M iter_args(%b) {
    ^bb0(%b_arg: !wave.tensor<[@M, @K] of f32>):
      // CHECK: wave.iterate @K
      // CHECK-SAME: index
      // CHECK-SAME: M = #wave<index_mapping[#wave.iter<"M">] -> (_Iter_M, 1, 1)>
      %inner_result = wave.iterate @K iter_args(%a) {
      ^bb1(%a_arg: !wave.tensor<[@M, @K] of f32>):
        // CHECK: wave.add
        // CHECK-SAME: index
        // CHECK-SAME: <"M"> : [#wave.iter<"K">, #wave.iter<"M">] -> (_Iter_K + _Iter_M, 1, 1)
        %partial_result = wave.add %a_arg, %b_arg {wave_test.override_operand_index = [
          #wave.index_exprs<[<"M"> : [#wave.iter<"M">] -> (_Iter_M, 1, 1)]>,
          #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (_Iter_K, 1, 1)]>
        ]}
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

normalform.module [#wave.normal_form<full_types>] {
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
      %partial_result = wave.add %a_arg, %b {wave_test.override_operand_index = [
        #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)]>,
        #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (_Iter_K * 2, 1, 1)]>
      ]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Check that we don't leak iter symbols to values before the loop.

normalform.module [#wave.normal_form<full_types>] {
  // CHECK-LABEL: @do_not_leak_above
  func.func @do_not_leak_above(
    %a: !wave.tensor<[@M, @K] of f32>,
    %b: !wave.tensor<[@M, @K] of f32>
  ) -> !wave.tensor<[@M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.read
    // CHECK-SAME: index =
    // CHECK-SAME: <"M"> : [] -> (42, 1, 1)
    %b_reg = wave.read %b : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    %result = wave.iterate @K iter_args(%a) {
    ^bb0(%a_arg: !wave.tensor<[@M, @K] of f32>):
      %partial_result = wave.add %a_arg, %b_reg {wave_test.override_operand_index = [
        #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)]>,
        #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)]>
      ]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Check that we propagate lattices between adjacent operands of a write.

normalform.module [#wave.normal_form<full_types>] {
  // CHECK-LABEL: @write_sideways_propagation
  func.func @write_sideways_propagation(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>,
    %c: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: index
    // CHECK: <"M"> : [] -> (42, 1, <NULL>)
    wave.write %a, %b {wave_test.override_operand_index = [
      unit,
      #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (<NULL>, 1, <NULL>)]>
    ]
    } : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>
    %c_reg = wave.read %c {wave_test.override_result_index = [
      #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (42, <NULL>, <NULL>)]>
    ]} : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    wave.write %c_reg, %b : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>
    return
  }
}

// -----

// Check that sideways propagation between operands of a write that would
// lead to a conflict is not happening.

normalform.module [#wave.normal_form<full_types>] {
  // CHECK-LABEL: @write_sideways_no_conflicting_propagation
  func.func @write_sideways_no_conflicting_propagation(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>,
    %c: !wave.tensor<[@M] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // CHECK: wave.write
    // CHECK: index
    // CHECK: <"M"> : [] -> (1, <NULL>, <NULL>)
    wave.write %a, %b {wave_test.override_operand_index = [
      unit,
      #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (1, <NULL>, <NULL>)]>
    ]
    } : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>
    // CHECK: wave.read
    // CHECK: index
    // CHECK: <"M"> : [] -> (42, <NULL>, <NULL>)
    %c_reg = wave.read %c {wave_test.override_result_index = [
      #wave.index_exprs<[<"M"> : [#wave.iter<"K">] -> (42, <NULL>, <NULL>)]>
    ]} : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    wave.write %c_reg, %b : !wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>
    return
  }
}

// -----

// Check that dimension ordering is preserved after join() operations.
// With DictionaryAttr, M and K would be alphabetically sorted to K, M.
// WaveIndexExprsAttr must preserve the original M, K order.

normalform.module [#wave.normal_form<full_types>] {
  // CHECK-LABEL: @join_preserves_dimension_order
  func.func @join_preserves_dimension_order(
    %a: !wave.tensor<[@M, @K] of f32>,
    %b: !wave.tensor<[@M, @K] of f32>
  ) -> !wave.tensor<[@M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Join two lattices with M, K ordering - the result must preserve M before K.
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: {M : {{.*}}, K : {{.*}}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1),
         <"K"> : [#wave.index_symbol<T0>] -> ((T0 floordiv 16) * 4, 4, 1)
       ]>,
       #wave.index_exprs<[
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1),
         <"K"> : [#wave.index_symbol<T0>] -> ((T0 floordiv 16) * 4, 4, 1)
       ]>
    ]}
    : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Join with three dimensions - verifies ordering is preserved for larger tuples.
// Both operands have B, M, K order matching the tensor type [@B, @M, @K].

normalform.module [#wave.normal_form<full_types>] {
  // CHECK-LABEL: @join_preserves_three_dim_order
  func.func @join_preserves_three_dim_order(
    %a: !wave.tensor<[@B, @M, @K] of f32>,
    %b: !wave.tensor<[@B, @M, @K] of f32>
  ) -> !wave.tensor<[@B, @M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Both operands have B, M, K order. Result must preserve B, M, K order.
    // With DictionaryAttr this would be sorted to B, K, M (alphabetical).
    // CHECK: wave.add
    // CHECK-SAME: index
    // CHECK-SAME: {B : {{.*}}, M : {{.*}}, K : {{.*}}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[
         <"B"> : [#wave.index_symbol<WG2>] -> (WG2, 1, 1),
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1),
         <"K"> : [#wave.index_symbol<T0>] -> ((T0 floordiv 16) * 4, 4, 1)
       ]>,
       #wave.index_exprs<[
         <"B"> : [#wave.index_symbol<WG2>] -> (WG2, 1, 1),
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1),
         <"K"> : [#wave.index_symbol<T0>] -> ((T0 floordiv 16) * 4, 4, 1)
       ]>
    ]}
    : (!wave.tensor<[@B, @M, @K] of f32>, !wave.tensor<[@B, @M, @K] of f32>) -> !wave.tensor<[@B, @M, @K] of f32>
    return %result : !wave.tensor<[@B, @M, @K] of f32>
  }
}

// -----

// Join conflict: operands have same dimensions but in different order with
// conflicting mappings. The dimension ordering differs (M,K vs K,M) AND the
// mappings for K conflict, causing the join to reach top.

normalform.module [#wave.normal_form<full_types>] {
  func.func @join_different_order_conflicting_mappings(
    %a: !wave.tensor<[@M, @K] of f32>,
    %b: !wave.tensor<[@M, @K] of f32>
  ) -> !wave.tensor<[@M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // LHS: M, K order with K mapping using T0 floordiv 16
    // RHS: K, M order with K mapping using T0 floordiv 32 (different!)
    // The K mappings conflict, causing join to reach top.
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1),
         <"K"> : [#wave.index_symbol<T0>] -> ((T0 floordiv 16) * 4, 4, 1)
       ]>,
       #wave.index_exprs<[
         <"K"> : [#wave.index_symbol<T0>] -> ((T0 floordiv 32) * 4, 4, 1),
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)
       ]>
    ]}
    : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Join conflict: one operand has subset of dimensions with conflicting mapping.
// LHS has M and K, RHS has only M but with a different mapping.

normalform.module [#wave.normal_form<full_types>] {
  func.func @join_subset_conflicting_mapping(
    %a: !wave.tensor<[@M, @K] of f32>,
    %b: !wave.tensor<[@M, @K] of f32>
  ) -> !wave.tensor<[@M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // LHS: M uses T0 mod 16
    // RHS: M uses T0 mod 32 (different!)
    // The M mappings conflict, causing join to reach top.
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1),
         <"K"> : [#wave.index_symbol<T0>] -> ((T0 floordiv 16) * 4, 4, 1)
       ]>,
       #wave.index_exprs<[
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)
       ]>
    ]}
    : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Join conflict: disjoint dimensions with different symbols that conflict
// when mappings are incompatible.

normalform.module [#wave.normal_form<full_types>] {
  func.func @join_three_dims_conflicting_shared_dim(
    %a: !wave.tensor<[@B, @M, @K] of f32>,
    %b: !wave.tensor<[@B, @M, @K] of f32>
  ) -> !wave.tensor<[@B, @M, @K] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Both have B, M, K but B has conflicting mappings:
    // LHS: B uses WG2
    // RHS: B uses WG0 (different workgroup dimension!)
    // expected-error @below {{incompatible operand lattices when propagating from those to result}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [
       #wave.index_exprs<[
         <"B"> : [#wave.index_symbol<WG2>] -> (WG2, 1, 1),
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1),
         <"K"> : [#wave.index_symbol<T0>] -> ((T0 floordiv 16) * 4, 4, 1)
       ]>,
       #wave.index_exprs<[
         <"B"> : [#wave.index_symbol<WG0>] -> (WG0, 1, 1),
         <"M"> : [#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1),
         <"K"> : [#wave.index_symbol<T0>] -> ((T0 floordiv 16) * 4, 4, 1)
       ]>
    ]}
    : (!wave.tensor<[@B, @M, @K] of f32>, !wave.tensor<[@B, @M, @K] of f32>) -> !wave.tensor<[@B, @M, @K] of f32>
    return %result : !wave.tensor<[@B, @M, @K] of f32>
  }
}
