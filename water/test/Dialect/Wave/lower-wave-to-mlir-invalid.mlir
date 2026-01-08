// RUN: water-opt %s -allow-unregistered-dialect -lower-wave-to-mlir --split-input-file --verify-diagnostics

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types,resolved_allocations>} {
  func.func @binary_ops_pattern_failure() {
    %cst = arith.constant 1.0 : f32
    // expected-error @below {{wave dialect operation with no hyperparameters provided by any ancestor}}
    %lhs = wave.register %cst : vector<4xf32>
    return
  }
}

// -----


module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types,resolved_allocations>} {
  // expected-error @+1 {{failed to convert starting at this operation}}
  func.func @read_pattern_failure(%mem: !wave.tensor<[@M, @N] of f16, <global>>)
    attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128}>} {

    // ReadOpLoweringPattern requires 'index' attribute
    // expected-error @+1 {{failed to legalize operation 'wave.read' that was explicitly marked illegal}}
    %result = wave.read %mem : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<8xf16>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types,resolved_allocations>} {
  // expected-error @+1 {{failed to convert starting at this operation}}
  func.func @write_pattern_failure(%mem: !wave.tensor<[@M, @N] of f16, <global>>)
    attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128}>} {

    %cst = arith.constant 0.0 : f16
    %reg = wave.register %cst : vector<8xf16>

    // WriteOpLoweringPattern requires 'index' attribute
    // expected-error @+1 {{failed to legalize operation 'wave.write' that was explicitly marked illegal}}
    wave.write %reg, %mem : vector<8xf16>, !wave.tensor<[@M, @N] of f16, <global>>
    return
  }
}

// -----

// Should not crash on null stride.

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types,resolved_allocations>} {
// expected-error @below {{failed to convert starting at this operation}}
func.func @lower_read_non_innermost_dim(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128, N = 128}>}  {
  // expected-error @below {{failed to legalize}}
  %0 = wave.read %mem index [{
    M : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 * 8 , <NULL>, 64),
    N : [#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 8) * T1, 1, 1)}]
    : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<8xf16>
  return
  }
}
