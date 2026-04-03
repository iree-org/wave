// RUN: water-opt %s -split-input-file --verify-diagnostics --water-test-wave-infer-index-exprs='attach-source-vector-shapes' | FileCheck %s

// sourceVectorShape: two operands with the same vector shape {M=16, N=16}
// and same priority should join cleanly.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @source_non_batch_dims_same_dims_same_priority
  func.func @source_non_batch_dims_same_dims_same_priority(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Both operands have sourceVectorShape {M=16, N=16}, priority 3.
    // Same priority, same shape -> should join cleanly without conflict.
    // CHECK: wave.add
    // CHECK-SAME: source_vector_shape_priorities = [3 : i32]
    // CHECK-SAME: source_vector_shapes = [#wave.symbol_mapping<@M = 16 : i64, @N = 16 : i64>]
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [3, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }, #wave.symbol_mapping<@M = 16 : i64, @N = 16 : i64>],
      [3, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }, #wave.symbol_mapping<@M = 16 : i64, @N = 16 : i64>]
    ]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %result : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// sourceVectorShape: two operands with different source vector shapes and same
// sourceVectorShapePriority should cause a conflict (top).
// Per-key priorities differ so the joinable vector shapes merge successfully,
// but sourceVectorShapePriority = max(per-key priorities) is the same for both.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @source_non_batch_dims_different_dims_same_priority(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Operand 0: sourceVectorShape {M=16, N=16}, sourceVectorShapePriority=5
    // Operand 1: sourceVectorShape {M=16, N=0}, sourceVectorShapePriority=5
    // Per-key vector shape join succeeds (higher priority wins for each key)
    // but sourceVectorShapes differ at equal priority 5 -> conflict.
    // expected-error @below {{conflict when propagating source vector shapes from operand #1 to result #0}}
    // expected-note @below {{original result lattice}}
    // expected-note @below {{operand #1 lattice}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = 3 : i32, N = 5 : i32}, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }, #wave.symbol_mapping<@M = 16 : i64, @N = 16 : i64>],
      [{M = 5 : i32, N = 3 : i32}, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }, #wave.symbol_mapping<@M = 16 : i64, @N = 0 : i64>]
    ]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %result : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// sourceVectorShape: two operands with different source vector shapes but
// different sourceVectorShapePriority should resolve by picking the higher one.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @source_non_batch_dims_different_dims_different_priority
  func.func @source_non_batch_dims_different_dims_different_priority(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Operand 0: sourceVectorShape {M=16, N=16}, sourceVectorShapePriority=5
    // Operand 1: sourceVectorShape {M=16, N=0}, sourceVectorShapePriority=3
    // Per-key vector shape join succeeds (M=16 match, N: pri 5>3 -> lhs wins).
    // sourceVectorShapePriority 5>3 -> no conflict, picks operand 0.
    // CHECK: wave.add
    // CHECK-SAME: source_vector_shape_priorities = [5 : i32]
    // CHECK-SAME: source_vector_shapes = [#wave.symbol_mapping<@M = 16 : i64, @N = 16 : i64>]
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = 5 : i32, N = 5 : i32}, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }, #wave.symbol_mapping<@M = 16 : i64, @N = 16 : i64>],
      [{M = 3 : i32, N = 3 : i32}, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }, #wave.symbol_mapping<@M = 16 : i64, @N = 0 : i64>]
    ]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %result : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// sourceVectorShape: one operand has a source vector shape, the other doesn't
// (no vectorShape). Should propagate the one that has it.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @source_non_batch_dims_one_present_one_absent
  func.func @source_non_batch_dims_one_present_one_absent(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Operand 0 has no vectorShape -> no sourceVectorShape.
    // Operand 1 has vectorShape {M=16, N=16} -> sourceVectorShape set.
    // Should join cleanly, picking operand 1's sourceVectorShape.
    // CHECK: wave.add
    // CHECK-SAME: source_vector_shape_priorities = [1 : i32]
    // CHECK-SAME: source_vector_shapes = [#wave.symbol_mapping<@M = 16 : i64, @N = 16 : i64>]
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [1, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }],
      [1, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }, #wave.symbol_mapping<@M = 16 : i64, @N = 16 : i64>]
    ]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %result : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// sourceVectorShape: two operands with identical vector shapes containing only
// batch dims (values <= 1) should join cleanly.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @source_non_batch_dims_all_batch
  func.func @source_non_batch_dims_all_batch(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // Both operands have sourceVectorShape {M=0, N=0} with same priority.
    // Identical shapes -> should join cleanly.
    // CHECK: wave.add
    // CHECK-SAME: source_vector_shape_priorities = [3 : i32]
    // CHECK-SAME: source_vector_shapes = [#wave.symbol_mapping<@M = 0 : i64, @N = 0 : i64>]
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [3, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }, #wave.symbol_mapping<@M = 0 : i64, @N = 0 : i64>],
      [3, {
        M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 32, 1, 1)>,
        N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 10, 1, 1)>
      }, #wave.symbol_mapping<@M = 0 : i64, @N = 0 : i64>]
    ]}
    : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return %result : !wave.tensor<[@M, @N] of f32>
  }
}
