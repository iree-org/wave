// RUN: water-opt %s --split-input-file --verify-diagnostics

func.func @mismatch_element_lhs_acc(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@B, @C] of f32>, %acc: !wave.tensor<[@A, @C] of bf16>) {
  // expected-error @below {{unexpected accumulator/result elemental type 'bf16' for MMA kind f32_16x16x16_f16}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@B, @C] of f32>, !wave.tensor<[@A, @C] of bf16>) -> !wave.tensor<[@A, @C] of bf16>
}

// -----

func.func @mismatch_element_int(%lhs: !wave.tensor<[@A, @B] of i32>, %rhs: !wave.tensor<[@B, @C] of i32>, %acc: !wave.tensor<[@A, @C] of i32>) {
  // expected-error @below {{unexpected lhs/rhs elemental type 'i32' for MMA kind i32_32x32x16_i8}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<i32_32x32x16_i8>} : (!wave.tensor<[@A, @B] of i32>, !wave.tensor<[@B, @C] of i32>, !wave.tensor<[@A, @C] of i32>) -> !wave.tensor<[@A, @C] of i32>
}

// -----

func.func @mismatch_element_lhs_rhs(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@B, @C] of bf16>, %acc: !wave.tensor<[@A, @C] of f32>) {
  // expected-error @below {{expected LHS and RHS elemental types to match, got 'f32', 'bf16'}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@B, @C] of bf16>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
}

// -----

func.func @mismatch_dim_lhs_rhs(%lhs: !wave.tensor<[@A, @B] of f16>, %rhs: !wave.tensor<[@C, @B] of f16>, %acc: !wave.tensor<[@A, @C] of f32>) {
  // expected-error @below {{expected LHS dimension #1 (#wave.symbol<"B">) to match RHS dimension #0 (#wave.symbol<"C">)}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f16>, !wave.tensor<[@C, @B] of f16>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
}

// -----

func.func @mismatch_dim_lhs_acc(%lhs: !wave.tensor<[@A, @B] of f16>, %rhs: !wave.tensor<[@B, @C] of f16>, %acc: !wave.tensor<[@E, @D] of f32>) {
  // expected-error @below {{expected LHS dimension #0 (#wave.symbol<"A">) to match accumulator dimension #0 (#wave.symbol<"E">)}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f16>, !wave.tensor<[@B, @C] of f16>, !wave.tensor<[@E, @D] of f32>) -> !wave.tensor<[@E, @D] of f32>
}

// -----

func.func @mismatch_dim_rhs_acc(%lhs: !wave.tensor<[@A, @B] of f16>, %rhs: !wave.tensor<[@B, @C] of f16>, %acc: !wave.tensor<[@A, @D] of f32>) {
  // expected-error @below {{expected RHS dimension #1 (#wave.symbol<"C">) to match accumulator dimension #1 (#wave.symbol<"D">)}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f16>, !wave.tensor<[@B, @C] of f16>, !wave.tensor<[@A, @D] of f32>) -> !wave.tensor<[@A, @D] of f32>
}

// -----

func.func @invalid_register_type(%arg0: f32) {
  // expected-error @below {{expected wave tensor type, got 'f32'}}
  wave.register %arg0 : f32
}

// -----

func.func @register_type_mismatch(%arg0: f32) {
  // expected-error @below {{expected the type of the init value to match the elemental type of the tensor}}
  "wave.register"(%arg0) : (f32) -> !wave.tensor<[@A, @B] of bf16>
}

// -----

func.func @register_underspecified(%arg0: f32) {
  // expected-error @below {{expected fully-specified tensor type}}
  wave.register %arg0 : !wave.tensor<any of f32>
}

// -----

func.func @iterate_mismatching_results(%arg0: !wave.tensor<any of f32>, %arg1: !wave.tensor<any of f32>) {
  // expected-error @below {{expects the same number of iter_args (2) and results (1)}}
  wave.iterate @I iter_args(%arg0, %arg1) {
  ^bb0(%arg2: !wave.tensor<any of f32>, %arg3: !wave.tensor<any of f32>):
    wave.yield %arg2 : !wave.tensor<any of f32>
  } : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<any of f32>
}

// -----

func.func @iterate_mismatching_results(%arg0: !wave.tensor<[@A] of f32>, %arg1: !wave.tensor<any of f32>) {
  // expected-error @below {{along control flow edge from parent operands to Region #0: source type #0 '!wave.tensor<[@A] of f32>' should match input type #0 '!wave.tensor<[@B] of f32>'}}
  wave.iterate @I iter_args(%arg0, %arg1) {
  ^bb0(%arg2: !wave.tensor<[@B] of f32>, %arg3: !wave.tensor<any of f32>):
    wave.yield %arg2, %arg3 : !wave.tensor<[@B] of f32>, !wave.tensor<any of f32>
  } : (!wave.tensor<[@A] of f32>, !wave.tensor<any of f32>) -> (!wave.tensor<any of f32>, !wave.tensor<any of f32>)
}
