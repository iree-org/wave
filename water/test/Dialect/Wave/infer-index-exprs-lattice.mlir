// RUN: water-opt %s --water-test-wave-infer-index-exprs --split-input-file --verify-diagnostics

//
// This file contains tests for index expression dataflow analyses that require
// injecting specific lattice states via attributes. These are predominantly
// checking the index conflict detection when propagating across a single
// operation, i.e., when a lattice first reaches the top state from non-top
// states.
//

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  func.func @simple_mma(
    %lhs: !wave.tensor<[@M, @K] of f16>,
    %rhs: !wave.tensor<[@N, @K] of f16>,
    %acc: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %lhs_override = wave.read %lhs { wave_test.override_result_index = [{
        N = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>
    }]} : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
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

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
        {K = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>}
      ]
    }
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
        {N = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>}
      ]
    }
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
        {M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>}
      ]
    }
         : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
         -> !wave.tensor<[@M, @N] of f32>
    return %r : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  func.func @add_then_mul(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@M, @K] of f16>,
    %c: !wave.tensor<[@M, @K] of f16>
  ) -> !wave.tensor<[@M, @K] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %add = wave.add %a, %b {wave_test.override_result_index = [{
      M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
      K = #wave<index_mapping[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    // expected-error @below {{conflict when propagating index expressions from result to operand #0}}
    // expected-note @below {{original operand lattice}}
    // expected-note @below {{result #0 lattice}}
    %mul = wave.mul %add, %c {wave_test.override_result_index = [{
      M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>,
      K = #wave<index_mapping[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %mul : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types>, wave_test.disable_backward } {
  func.func @add_then_mul(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@M, @K] of f16>,
    %c: !wave.tensor<[@M, @K] of f16>
  ) -> !wave.tensor<[@M, @K] of f16> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %add = wave.add %a, %b {wave_test.override_result_index = [{
      M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
      K = #wave<index_mapping[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    // expected-error @below {{conflict when propagating index expressions from operand to result #0}}
    // expected-note @below {{original result lattice}}
    // expected-note @below {{operand #0 lattice}}
    // expected-note @below {{operand #1 lattice}}
    %mul = wave.mul %add, %c {wave_test.override_result_index = [{
      M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>,
      K = #wave<index_mapping[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %mul : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types>, wave_test.disable_backward } {
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
    %add = wave.add %a, %b {wave_test.override_operand_index = [{
      M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
      K = #wave<index_mapping[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }, {
      M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 44, 1, 1)>,
      K = #wave<index_mapping[#wave.index_symbol<T1>] -> (T1 * 16, 1, 1)>
    }]}: (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>

    return %add : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

// Generic error message when reached top somehow without detecting the conflict before.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: M : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining with null (uninitialized) doesn't crash and gives the other expression.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: M : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (<NULL>, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining with bottom (denoted as unit) gives the other expression.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: M : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    },
       unit  // will default-initialize to bottom.
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining with zero is gives the other expression.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: M : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (0, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Additional constant summand makes expressions join to top.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40 + 1, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different constant summands join to top.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40 + 2, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40 + 1, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different constant values other than zero join to top.
// Also, difference may be in the step.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 3, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 2, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Difference in stride joins to top.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 2)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 3)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Stride 1 joins with the other constant stride to become that stride.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: M : [#wave.index_symbol<T0>] -> (T0 * 40, 1, 2)
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 2)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Step 1 joins with the other non-constant step to become that step.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: M : [#wave.index_symbol<T0>] -> (T0 * 40, T0, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, T0, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different expressions in step join to top even if they would have resulted in a sum for start.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0, T0, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (T0, WG0, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different expressions involving threads join to top.
// Note that here the underlying affine expression is the same, but symbols
// are different, we should be able to catch that.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T1>] -> (T1 * 40, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Different expressions involving workgroups join to top.
// Note that there are unused symbols in mappings.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>] -> (WG0, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>] -> (WG1, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}


// -----

// Joining thread and block components is fine. Note that some symbols are unused in mappings.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: {M : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 + T0, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (T0, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Identical constant summands don't sum up when symbols do.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: {M : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 + T0 + 2, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<WG0>] -> (WG0 + 2, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0 + 2 , 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining thread and block components is fine, this requires aligning symbols in mappings.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: {M : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 + T0, 1, 1)
    %result = wave.add %a, %b {wave_test.override_operand_index = [{
       M = #wave<index_mapping[#wave.index_symbol<WG0>] -> (WG0, 1, 1)>
    }, {
       M = #wave<index_mapping[#wave.index_symbol<T0>] -> (T0, 1, 1)>
    }]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
    return %result : !wave.tensor<[@M] of f32>
  }
}

// -----

// Joining iter symbols and blocks is fine and results in an add.
// TODO: Also check that iter symbols don't leak form the loop to results.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
      // CHECK-SAME: M : [#wave.index_symbol<WG0>, #wave.iter<"K">] -> (WG0 + _Iter_K, 1, 1)
      %partial_result = wave.add %a_arg, %b {wave_test.override_operand_index = [{
        M = #wave<index_mapping[#wave.index_symbol<WG0>] -> (WG0, 1, 1)>
      }, {
        M = #wave<index_mapping[#wave.iter<"K">] -> (_Iter_K, 1, 1)>
      }]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Joining iter symbols with themselves is fine.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
      // CHECK-SAME: M : [#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)
      %partial_result = wave.add %a_arg, %b {wave_test.override_operand_index = [{
        M = #wave<index_mapping[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }, {
        M = #wave<index_mapping[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }]}
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

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
        // CHECK-SAME: M : [#wave.iter<"M">, #wave.iter<"K">] -> (_Iter_M + _Iter_K, 1, 1)
        %partial_result = wave.add %a_arg, %b_arg {wave_test.override_operand_index = [{
          M = #wave<index_mapping[#wave.iter<"M">] -> (_Iter_M, 1, 1)>
        }, {
          M = #wave<index_mapping[#wave.iter<"K">] -> (_Iter_K, 1, 1)>
        }]}
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

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
      %partial_result = wave.add %a_arg, %b {wave_test.override_operand_index = [{
        M = #wave<index_mapping[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }, {
        M = #wave<index_mapping[#wave.iter<"K">] -> (_Iter_K * 2, 1, 1)>
      }]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}

// -----

// Check that we don't leak iter symbols to values before the loop.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
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
    // CHECK-SAME: {M : [] -> (42, 1, 1)
    %b_reg = wave.read %b : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    %result = wave.iterate @K iter_args(%a) {
    ^bb0(%a_arg: !wave.tensor<[@M, @K] of f32>):
      %partial_result = wave.add %a_arg, %b_reg {wave_test.override_operand_index = [{
        M = #wave<index_mapping[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }, {
        M = #wave<index_mapping[#wave.iter<"K">] -> (_Iter_K + 42, 1, 1)>
      }]}
      : (!wave.tensor<[@M, @K] of f32>, !wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>

      wave.yield %partial_result : !wave.tensor<[@M, @K] of f32>
    } : (!wave.tensor<[@M, @K] of f32>) -> !wave.tensor<[@M, @K] of f32>
    return %result : !wave.tensor<[@M, @K] of f32>
  }
}
