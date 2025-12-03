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
    // expected-error @below {{incompatible operand when propagating from those to result}}
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
