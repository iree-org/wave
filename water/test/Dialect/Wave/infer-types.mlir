// RUN: water-opt %s --water-wave-infer-types --split-input-file --verify-diagnostics | FileCheck %s

// CHECK: #wave.normal_form<full_types>
module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {

// CHECK-LABEL: @propagate_forward
func.func @propagate_forward(%a: !wave.tensor<[@A, @B] of bf16>) {
  // CHECK: (!wave.tensor<[@A, @B] of bf16>, !wave.tensor<[@A, @B] of bf16>) -> !wave.tensor<[@A, @B] of bf16>
  wave.add %a, %a : (!wave.tensor<[@A, @B] of bf16>, !wave.tensor<[@A, @B] of bf16>) -> !wave.tensor<any of bf16>
  return
}

// CHECK-LABEL: @propagate_backward
func.func @propagate_backward() {
  %b = water_test.wave_tensor : !wave.tensor<any of bf16>
  // CHECK: (!wave.tensor<[@A, @B] of bf16>, !wave.tensor<[@A, @B] of bf16>) -> !wave.tensor<[@A, @B] of bf16>
  wave.add %b, %b : (!wave.tensor<any of bf16>, !wave.tensor<any of bf16>) -> !wave.tensor<[@A, @B] of bf16>
  return
}

// CHECK-LABEL: @propagate_control_flow
func.func @propagate_control_flow(%a: !wave.tensor<[@A] of f32>, %b: !wave.tensor<[@A] of f32>, %c: i1) {
  cf.cond_br %c, ^bb0(%a: !wave.tensor<[@A] of f32>), ^bb1(%b: !wave.tensor<[@A] of f32>)

^bb0(%arg0: !wave.tensor<[@A] of f32>):
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  wave.exp2 %arg0: (!wave.tensor<[@A] of f32>) -> !wave.tensor<any of f32>
  return

^bb1(%arg1: !wave.tensor<[@A] of f32>):
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  wave.exp2 %arg1: (!wave.tensor<[@A] of f32>) -> !wave.tensor<any of f32>
  return
}

// CHECK-LABEL: @propagate_control_flow_block
func.func @propagate_control_flow_block(%a: !wave.tensor<[@A] of f32>) {
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  %b = wave.exp2 %a: (!wave.tensor<[@A] of f32>) -> !wave.tensor<any of f32>
  // CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@A] of f32>)
  cf.br ^bb0(%b: !wave.tensor<any of f32>)

// CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@A] of f32>):
^bb0(%arg0: !wave.tensor<any of f32>):
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  wave.exp2 %arg0: (!wave.tensor<any of f32>) -> !wave.tensor<any of f32>
  return
}

// CHECK-LABEL: @propagate_control_flow_backward
func.func @propagate_control_flow_backward() {
  // CHECK: !wave.tensor<[@C] of f32>
  %b = water_test.wave_tensor : !wave.tensor<any of f32>
  // CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@C] of f32>)
  cf.br ^bb0(%b: !wave.tensor<any of f32>)

// CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@C] of f32>):
^bb0(%arg0: !wave.tensor<any of f32>):
  // CHECK: (!wave.tensor<[@C] of f32>) -> !wave.tensor<[@C] of f32>
  wave.exp2 %arg0: (!wave.tensor<any of f32>) -> !wave.tensor<[@C] of f32>
  return
}

// CHECK-LABEL: @propagate_structured_control_flow
func.func @propagate_structured_control_flow(%a: !wave.tensor<[@A] of f32>) {
  %0 = wave.iterate @I iter_args(%a) {
  // CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@A] of f32>)
  ^bb0(%arg0: !wave.tensor<any of f32>):
    // CHECK: (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
    %m = wave.mul %arg0, %arg0 : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<any of f32>
    // CHECK: !wave.tensor<[@A] of f32>
    wave.yield %m : !wave.tensor<any of f32>
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  } : (!wave.tensor<[@A] of f32>) -> !wave.tensor<any of f32>
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  wave.exp2 %0 : (!wave.tensor<any of f32>) -> !wave.tensor<any of f32>
  return
}

// CHECK-LABEL: @propagate_structured_control_partial
func.func @propagate_structured_control_partial(%a: !wave.tensor<[@A] of f32>) {
  %0 = wave.iterate @I iter_args(%a) {
  // CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@A] of f32>)
  ^bb0(%arg0: !wave.tensor<any of f32>):
    // CHECK: (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
    %m = wave.mul %arg0, %arg0 : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<[@A] of f32>
    // CHECK: !wave.tensor<[@A] of f32>
    wave.yield %m : !wave.tensor<[@A] of f32>
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  } : (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  wave.exp2 %0 : (!wave.tensor<[@A] of f32>) -> !wave.tensor<any of f32>
  return
}

// CHECK-LABEL: @propagate_structured_control_pre
func.func @propagate_structured_control_pre(%a: !wave.tensor<[@A] of f32>) {
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  %b = wave.exp2 %a : (!wave.tensor<[@A] of f32>) -> !wave.tensor<any of f32>
  %0 = wave.iterate @I iter_args(%b) {
  // CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@A] of f32>)
  ^bb0(%arg0: !wave.tensor<any of f32>):
    // CHECK: (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
    %m = wave.mul %arg0, %arg0 : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<any of f32>
    // CHECK: !wave.tensor<[@A] of f32>
    wave.yield %m : !wave.tensor<any of f32>
  } : (!wave.tensor<any of f32>) -> !wave.tensor<any of f32>
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  wave.exp2 %0 : (!wave.tensor<any of f32>) -> !wave.tensor<any of f32>
  return
}

// CHECK-LABEL: @propagate_structured_control_backward
func.func @propagate_structured_control_backward() {
  // CHECK: !wave.tensor<[@A] of f32>
  %b = water_test.wave_tensor : !wave.tensor<any of f32>
  %0 = wave.iterate @I iter_args(%b) {
  // CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@A] of f32>)
  ^bb0(%arg0: !wave.tensor<any of f32>):
    // CHECK: (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
    %m = wave.mul %arg0, %arg0 : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<any of f32>
    // CHECK: !wave.tensor<[@A] of f32>
    wave.yield %m : !wave.tensor<any of f32>
  } : (!wave.tensor<any of f32>) -> !wave.tensor<any of f32>
  // CHECK: (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
  wave.exp2 %0 : (!wave.tensor<any of f32>) -> !wave.tensor<[@A] of f32>
  return
}

} // module

// -----

module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {
  func.func @conflict(%arg0: !wave.tensor<[@A] of f32>) {
    // expected-error @below {{failed to propagate type information backward: irreconcilable types during type inference from results(!wave.tensor<[@B] of f32>) to operands(!wave.tensor<[@A] of f32>) for results #0}}
    %0 = wave.exp2 %arg0 : (!wave.tensor<[@A] of f32>) -> !wave.tensor<any of f32>
    wave.exp2 %0 : (!wave.tensor<any of f32>) -> !wave.tensor<[@B] of f32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {
  func.func @iterate_conflict(%arg0: !wave.tensor<[@A] of f32>) {
    // expected-error @below {{type conflict was detected for result #0}}
    %0 = wave.iterate @I iter_args(%arg0) {
    ^bb0(%arg1: !wave.tensor<[@A] of f32>):
      wave.yield %arg1 : !wave.tensor<[@A] of f32>
    } : (!wave.tensor<[@A] of f32>) -> (!wave.tensor<any of f32>)
    wave.iterate @I iter_args(%0) {
    ^bb0(%arg1: !wave.tensor<any of f32>):
      wave.yield %arg1 : !wave.tensor<any of f32>
    } : (!wave.tensor<any of f32>) -> (!wave.tensor<[@B] of f32>)
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {
  func.func @force_fail_forward(%arg0: !wave.tensor<[@A] of f32>) {
    // expected-error @below {{failed to propagate type information forward: intentionally failed to propagate forward}}
    water_test.wave_fail_propagation %arg0, %arg0 {forward}
      : (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>)
      -> (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>)
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {
  func.func @force_fail_backward(%arg0: !wave.tensor<[@A] of f32>) {
    // expected-error @below {{failed to propagate type information backward: intentionally failed to propagate backward}}
    water_test.wave_fail_propagation %arg0, %arg0 {backward}
      : (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>)
      -> (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>)
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {
  func.func @type_mismatch_operands(%arg0: !wave.tensor<[@A] of f32>, %arg1: !wave.tensor<[@B] of f32>) {
    // This op intentionally doesn't have a verifier for shape matching so we can
    // see the failure coming from the analysis/interface trait.
    // expected-error @below {{ailed to propagate type information backward: irreconcilable types during type inference from results(!wave.tensor<[@A] of f32>) to operands(!wave.tensor<[@B] of f32>) for results #1}}
    water_test.wave_fail_propagation %arg0, %arg1
      : (!wave.tensor<[@A] of f32>, !wave.tensor<[@B] of f32>)
      -> (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>)
    return
  }
}

// -----

// expected-error @below {{water-wave-infer-types pass expects the root operation or its ancestor to guarantee the full_func_boundary normal form}}
module {
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {
  func.func @iterate_mismatching_results(%arg0: !wave.tensor<[@A] of f32>, %arg1: !wave.tensor<[@B] of f32>) {
    %read = wave.read %arg1 : (!wave.tensor<[@B] of f32>) -> !wave.tensor<any of f32>
    // expected-error @below {{expected iter arg #0 dimension #0 (#wave.symbol<"A">) to match block iter arg #0 dimension #0 (#wave.symbol<"B">)}}
    wave.iterate @I iter_args(%arg0, %read) {
    ^bb0(%arg2: !wave.tensor<[@B] of f32>, %arg3: !wave.tensor<any of f32>):
      wave.yield %arg2, %arg3 : !wave.tensor<[@B] of f32>, !wave.tensor<any of f32>
    } : (!wave.tensor<[@A] of f32>, !wave.tensor<any of f32>) -> (!wave.tensor<any of f32>, !wave.tensor<any of f32>)
    return
  }
}
