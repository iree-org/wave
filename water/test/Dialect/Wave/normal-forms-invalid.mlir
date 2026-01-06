// RUN: water-opt %s --split-input-file --verify-diagnostics

module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {
  // expected-error @below {{normal form requires tensor types to be fully specified at function boundaries}}
  func.func private @foo(!wave.tensor<any of f32>)
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_op_types>} {
  func.func @foo() {
    %0 = arith.constant 0.0 : f32
    // expected-error @below {{normal form requires tensor types to be fully specified}}
    wave.register %0 : !wave.tensor<any of f32, <register>>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<index_exprs>} {
  func.func @bar() {
    %0 = arith.constant 0.0 : f32
    // expected-error @below {{normal form requires index expressions to be provided for all supported wave dialect operations}}
    wave.register %0 : !wave.tensor<[@X] of f32, <register>>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<resolved_allocations>} {
  func.func @unresolved_allocate() {
    // expected-error @below {{normal form requires all wave.allocate operations to have memref result type}}
    %0 = wave.allocate {distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)>}
      : !wave.tensor<[@M] of f32, <shared>>
    return
  }
}
