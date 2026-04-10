// RUN: water-opt %s --water-wave-expand-variadic-reductions --split-input-file | FileCheck %s

// Variadic reductions are expanded into chained single-input reductions.

// CHECK-LABEL: @expand_variadic_sum
// CHECK-SAME: (%[[A:.*]]: {{.*}}, %[[B:.*]]: {{.*}}, %[[C:.*]]: {{.*}}, %[[INIT:.*]]: {{.*}})
func.func @expand_variadic_sum(%a: !wave.tensor<[@N, @M] of f32>, %b: !wave.tensor<[@N, @M] of f32>, %c: !wave.tensor<[@N, @M] of f32>, %init: !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32> {
  // CHECK: %[[R0:.*]] = wave.sum %[[A]] init(%[[INIT]]) <warp>
  // CHECK: %[[R1:.*]] = wave.sum %[[B]] init(%[[R0]]) <warp>
  // CHECK: %[[R2:.*]] = wave.sum %[[C]] init(%[[R1]]) <warp>
  // CHECK: return %[[R2]]
  %result = wave.sum %a, %b, %c init(%init) <warp> : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32>
  return %result : !wave.tensor<[@N] of f32>
}

// -----

// CHECK-LABEL: @expand_variadic_max_element
// CHECK-SAME: (%[[A:.*]]: {{.*}}, %[[B:.*]]: {{.*}}, %[[INIT:.*]]: {{.*}})
func.func @expand_variadic_max_element(%a: !wave.tensor<[@N, @M] of f32>, %b: !wave.tensor<[@N, @M] of f32>, %init: !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32> {
  // CHECK: %[[R0:.*]] = wave.max_element %[[A]] init(%[[INIT]]) <warp>
  // CHECK: %[[R1:.*]] = wave.max_element %[[B]] init(%[[R0]]) <warp>
  // CHECK: return %[[R1]]
  %result = wave.max_element %a, %b init(%init) <warp> : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32>
  return %result : !wave.tensor<[@N] of f32>
}

// -----

// Axis attribute is preserved on each chained reduction.
// CHECK-LABEL: @expand_preserves_axis
// CHECK-SAME: (%[[A:.*]]: {{.*}}, %[[B:.*]]: {{.*}}, %[[INIT:.*]]: {{.*}})
func.func @expand_preserves_axis(%a: !wave.tensor<any of f32>, %b: !wave.tensor<any of f32>, %init: !wave.tensor<any of f32>) -> !wave.tensor<any of f32> {
  // CHECK: %[[R0:.*]] = wave.sum %[[A]] init(%[[INIT]]) along @K <warp>
  // CHECK: %[[R1:.*]] = wave.sum %[[B]] init(%[[R0]]) along @K <warp>
  // CHECK: return %[[R1]]
  %result = wave.sum %a, %b init(%init) along @K <warp> : (!wave.tensor<any of f32>, !wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<any of f32>
  return %result : !wave.tensor<any of f32>
}

// -----

// Index attribute is preserved on each chained reduction.
// CHECK-LABEL: @expand_preserves_index
// CHECK-SAME: (%[[A:.*]]: {{.*}}, %[[B:.*]]: {{.*}}, %[[INIT:.*]]: {{.*}})
func.func @expand_preserves_index(%a: !wave.tensor<[@N, @M] of f32>, %b: !wave.tensor<[@N, @M] of f32>, %init: !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32> {
  // CHECK: %[[R0:.*]] = wave.sum %[[A]] init(%[[INIT]]) <warp> index [#wave.symbol_mapping<@N =
  // CHECK: %[[R1:.*]] = wave.sum %[[B]] init(%[[R0]]) <warp> index [#wave.symbol_mapping<@N =
  // CHECK: return %[[R1]]
  %result = wave.sum %a, %b init(%init) <warp> index [#wave.symbol_mapping<@N = #wave.index_mapping<[] -> (42, 1, 1)>>] : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32>
  return %result : !wave.tensor<[@N] of f32>
}

// -----

// Single-input reductions are left unchanged.
// CHECK-LABEL: @single_input_sum_unchanged
// CHECK-SAME: (%[[A:.*]]: {{.*}}, %[[INIT:.*]]: {{.*}})
func.func @single_input_sum_unchanged(%a: !wave.tensor<[@N, @M] of f32>, %init: !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32> {
  // CHECK: wave.sum %[[A]] init(%[[INIT]]) <warp>
  // CHECK-NOT: wave.sum
  %result = wave.sum %a init(%init) <warp> : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32>
  return %result : !wave.tensor<[@N] of f32>
}
