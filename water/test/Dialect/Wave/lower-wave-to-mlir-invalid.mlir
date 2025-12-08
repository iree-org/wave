// RUN: water-opt %s -allow-unregistered-dialect -lower-wave-to-mlir --split-input-file --verify-diagnostics

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @binary_ops_pattern_failure() {
    %cst = arith.constant 1.0 : f32
    // expected-error @below {{wave dialect operation with no hyperparameters provided by any ancestor}}
    %lhs = wave.register %cst : vector<4xf32>
    return
  }
}

// -----


module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
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

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
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
