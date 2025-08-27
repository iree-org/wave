// RUN: water-opt %s --split-input-file --verify-diagnostics

func.func @mismatch_element_lhs_acc(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@B, @C] of f32>, %acc: !wave.tensor<[@A, @C] of bf16>) {
  // expected-error @below {{expected LHS and accumulator elemental types to match, got 'f32', 'bf16'}}
  wave.mma %lhs, %rhs, %acc : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@B, @C] of f32>, !wave.tensor<[@A, @C] of bf16>) -> !wave.tensor<[@A, @C] of bf16>
}

// -----

func.func @mismatch_element_lhs_rhs(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@B, @C] of bf16>, %acc: !wave.tensor<[@A, @C] of f32>) {
  // expected-error @below {{expected LHS and RHS elemental types to match, got 'f32', 'bf16'}}
  wave.mma %lhs, %rhs, %acc : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@B, @C] of bf16>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
}

// -----

func.func @mismatch_dim_lhs_rhs(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@C, @B] of f32>, %acc: !wave.tensor<[@A, @C] of f32>) {
  // expected-error @below {{expected LHS dimension #1 (#wave.symbol<"B">) to match RHS dimension #0 (#wave.symbol<"C">)}}
  wave.mma %lhs, %rhs, %acc : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@C, @B] of f32>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
}

// -----

func.func @mismatch_dim_lhs_acc(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@B, @C] of f32>, %acc: !wave.tensor<[@E, @D] of f32>) {
  // expected-error @below {{expected LHS dimension #0 (#wave.symbol<"A">) to match accumulator dimension #0 (#wave.symbol<"E">)}}
  wave.mma %lhs, %rhs, %acc : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@B, @C] of f32>, !wave.tensor<[@E, @D] of f32>) -> !wave.tensor<[@E, @D] of f32>
}

// -----

func.func @mismatch_dim_rhs_acc(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@B, @C] of f32>, %acc: !wave.tensor<[@A, @D] of f32>) {
  // expected-error @below {{expected RHS dimension #1 (#wave.symbol<"C">) to match accumulator dimension #1 (#wave.symbol<"D">)}}
  wave.mma %lhs, %rhs, %acc : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@B, @C] of f32>, !wave.tensor<[@A, @D] of f32>) -> !wave.tensor<[@A, @D] of f32>
}
