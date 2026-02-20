// RUN: water-opt %s --water-wave-infer-types='partial=true' --split-input-file --verify-diagnostics | FileCheck %s

// CHECK: #wave.normal_form<full_func_boundary>
normalform.module [#wave.normal_form<full_func_boundary>] {

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

// CHECK-LABEL: @propagate_reduction_input_to_others
func.func @propagate_reduction_input_to_others() {
  %input = water_test.wave_tensor : !wave.tensor<[@M, @N, @K, @L] of f32>
  %init = water_test.wave_tensor : !wave.tensor<any of f32>
  // CHECK: wave.sum
  // CHECK-SAME: (!wave.tensor<[@M, @N, @K, @L] of f32>, !wave.tensor<[@M, @N, @K] of f32>) -> !wave.tensor<[@M, @N, @K] of f32>
  wave.sum %input init(%init) <warp>
    : (!wave.tensor<[@M, @N, @K, @L] of f32>, !wave.tensor<any of f32>) -> !wave.tensor<any of f32>
  return
}

// CHECK-LABEL: @propagate_reduction_init_to_result
func.func @propagate_reduction_init_to_result() {
  %input = water_test.wave_tensor : !wave.tensor<any of f32>
  %init = water_test.wave_tensor : !wave.tensor<[@M, @N, @K] of f32>
  // CHECK: wave.sum
  // CHECK-SAME: (!wave.tensor<[@M, @N, @K, @L] of f32>, !wave.tensor<[@M, @N, @K] of f32>) -> !wave.tensor<[@M, @N, @K] of f32>
  wave.sum %input init(%init) along @L <warp>
    : (!wave.tensor<any of f32>, !wave.tensor<[@M, @N, @K] of f32>) -> !wave.tensor<any of f32>
  return
}

func.func @propagate_reduction_result_to_init() {
  %input = water_test.wave_tensor : !wave.tensor<any of f32>
  %init = water_test.wave_tensor : !wave.tensor<any of f32>
  // CHECK: wave.sum
  // CHECK-SAME: (!wave.tensor<[@N, @M, @L, @K] of f32>, !wave.tensor<[@N, @M, @L] of f32>) -> !wave.tensor<[@N, @M, @L] of f32>
  wave.sum %input init(%init) along @K <warp>
    : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<[@N, @M, @L] of f32>
  return
}

// CHECK-LABEL: @broadcast_forward
func.func @broadcast_forward(%src: !wave.tensor<[@M] of f32, <register>>) {
  // CHECK: wave.broadcast
  // CHECK-SAME: (!wave.tensor<[@M] of f32, <register>>) -> !wave.tensor<[@M, @N, @K] of f32, <register>>
  wave.broadcast %src dims [@N, @K] : (!wave.tensor<[@M] of f32, <register>>) -> !wave.tensor<any of f32, <register>>
  return
}

// CHECK-LABEL: @broadcast_backward
func.func @broadcast_backward() {
  // CHECK: wave.broadcast
  // CHECK-SAME: (!wave.tensor<[@A, @B] of bf16, <register>>) -> !wave.tensor<[@A, @B, @C] of bf16, <register>>
  %src = water_test.wave_tensor : !wave.tensor<any of bf16, <register>>
  wave.broadcast %src dims [@C] : (!wave.tensor<any of bf16, <register>>) -> !wave.tensor<[@A, @B, @C] of bf16, <register>>
  return
}

} // normalform.module

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
  func.func @conflict(%arg0: !wave.tensor<[@A] of f32>) {
    // expected-error @below {{failed to propagate type information backward: irreconcilable types during type inference from results(!wave.tensor<[@B] of f32>) to operands(!wave.tensor<[@A] of f32>) for results #0}}
    %0 = wave.exp2 %arg0 : (!wave.tensor<[@A] of f32>) -> !wave.tensor<any of f32>
    wave.exp2 %0 : (!wave.tensor<any of f32>) -> !wave.tensor<[@B] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
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

normalform.module [#wave.normal_form<full_func_boundary>] {
  func.func @force_fail_forward(%arg0: !wave.tensor<[@A] of f32>) {
    // expected-error @below {{failed to propagate type information forward: intentionally failed to propagate forward}}
    water_test.wave_fail_propagation %arg0, %arg0 {forward}
      : (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>)
      -> (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>)
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
  func.func @force_fail_backward(%arg0: !wave.tensor<[@A] of f32>) {
    // expected-error @below {{failed to propagate type information backward: intentionally failed to propagate backward}}
    water_test.wave_fail_propagation %arg0, %arg0 {backward}
      : (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>)
      -> (!wave.tensor<[@A] of f32>, !wave.tensor<[@A] of f32>)
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
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

// CHECK: #wave.normal_form<full_func_boundary>
normalform.module [#wave.normal_form<full_func_boundary>] {

// CHECK-LABEL: @propagate_read_forward
// ReadOp: forward propagation from memory to result (value).
func.func @propagate_read_forward(%mem: !wave.tensor<[@M, @N] of f32, <global>>) {
  // CHECK: (!wave.tensor<[@M, @N] of f32, <global>>) -> !wave.tensor<[@M, @N] of f32, <register>>
  %0 = wave.read %mem : (!wave.tensor<[@M, @N] of f32, <global>>) -> !wave.tensor<any of f32, <register>>
  return
}

// CHECK-LABEL: @propagate_read_backward
// ReadOp: backward propagation from result to memory operand.
func.func @propagate_read_backward() {
  %mem = water_test.wave_tensor : !wave.tensor<any of f32, <global>>
  // CHECK: (!wave.tensor<[@M, @N] of f32, <global>>) -> !wave.tensor<[@M, @N] of f32, <register>>
  %0 = wave.read %mem : (!wave.tensor<any of f32, <global>>) -> !wave.tensor<[@M, @N] of f32, <register>>
  return
}

// CHECK-LABEL: @propagate_write_backward_memory_to_value
// WriteOp: backward propagation from memory to value operand.
func.func @propagate_write_backward_memory_to_value(%mem: !wave.tensor<[@M, @N] of f32, <global>>) {
  %val = water_test.wave_tensor : !wave.tensor<any of f32, <register>>
  // CHECK: wave.write %{{.*}}, %{{.*}} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
  wave.write %val, %mem : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
  return
}

// CHECK-LABEL: @propagate_write_backward_value_to_memory
// WriteOp: backward propagation from value to memory operand.
func.func @propagate_write_backward_value_to_memory(%val: !wave.tensor<[@M, @N] of f32, <register>>) {
  %mem = water_test.wave_tensor : !wave.tensor<any of f32, <global>>
  // CHECK: wave.write %{{.*}}, %{{.*}} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
  wave.write %val, %mem : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<any of f32, <global>>
  return
}

}

// -----

// CHECK: #wave.normal_form<full_func_boundary>
normalform.module [#wave.normal_form<full_func_boundary>] {

// CHECK-LABEL: @propagate_permute_forward
func.func @propagate_permute_forward(%a: !wave.tensor<[@B, @M, @N] of f32, <register>>) {
  // Result type is specified at parse time (required to be fully-specified).
  // CHECK: !wave.tensor<[@B, @M, @N] of f32, <register>> to !wave.tensor<[@M, @N, @B] of f32, <register>>
  wave.permute %a : !wave.tensor<[@B, @M, @N] of f32, <register>> to !wave.tensor<[@M, @N, @B] of f32, <register>>
  return
}

// CHECK-LABEL: @propagate_permute_2d
func.func @propagate_permute_2d(%a: !wave.tensor<[@M, @N] of f16, <register>>) {
  // CHECK: !wave.tensor<[@M, @N] of f16, <register>> to !wave.tensor<[@N, @M] of f16, <register>>
  wave.permute %a : !wave.tensor<[@M, @N] of f16, <register>> to !wave.tensor<[@N, @M] of f16, <register>>
  return
}

// CHECK-LABEL: @infer_permute_input_from_previous_op
// Test that permute input type can be inferred when it comes from a previous operation.
func.func @infer_permute_input_from_previous_op(%a: !wave.tensor<[@M, @N] of f32, <register>>) {
  // Input comes from exp2 which propagates the shape forward.
  // CHECK: wave.exp2 %{{.*}} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
  %b = wave.exp2 %a : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<any of f32, <register>>
  // Permute input type is inferred from exp2 output.
  // CHECK: wave.permute %{{.*}} : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[@N, @M] of f32, <register>>
  wave.permute %b : !wave.tensor<any of f32, <register>> to !wave.tensor<[@N, @M] of f32, <register>>
  return
}

// CHECK-LABEL: @infer_result_from_next_op
// Test that result type can be inferred from a permute consumer.
func.func @infer_result_from_next_op(%a: !wave.tensor<[@M, @N] of f32, <register>>) {
  // The exp2 result type is inferred from the permute which requires [@M, @N] input.
  // CHECK: wave.exp2 %{{.*}} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
  %b = wave.exp2 %a : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
  // CHECK: wave.permute %{{.*}} : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[@N, @M] of f32, <register>>
  %c = wave.permute %b : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<any of f32, <register>>
  // The add uses the permuted result.
  // CHECK: wave.add %{{.*}}, %{{.*}} : (!wave.tensor<[@N, @M] of f32, <register>>, !wave.tensor<[@N, @M] of f32, <register>>) -> !wave.tensor<[@N, @M] of f32, <register>>
  wave.add %c, %c : (!wave.tensor<any of f32, <register>>, !wave.tensor<any of f32, <register>>) -> !wave.tensor<[@N, @M] of f32, <register>>
  return
}

}

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
  func.func @permute_mismatching_symbols(%arg0: !wave.tensor<[@A, @B, @C] of f32, <register>>) {
    %0 = wave.exp2 %arg0 : (!wave.tensor<[@A, @B, @C] of f32, <register>>) -> !wave.tensor<any of f32, <register>>
    // expected-error @below {{failed to propagate type information forward: input dimension 'A' is not present in result shape}}
    wave.permute %0 : !wave.tensor<any of f32, <register>> to !wave.tensor<[@M, @N, @K] of f32, <register>>
    return
  }
}

// -----

// expected-error @below {{water-wave-infer-types pass expects the root operation or its ancestor to guarantee the full_func_boundary normal form}}
normalform.module [] {
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
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

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
  // CHECK-LABEL: @read_with_mapping_forward
  func.func @read_with_mapping_forward(%mem: !wave.tensor<[@A, @B, @C, @D] of f32, <global>>) {
    // CHECK: (!wave.tensor<[@A, @B, @C, @D] of f32, <global>>) -> !wave.tensor<[@D, @B, @A, @C] of f32, <register>
    wave.read %mem { mapping = #wave.expr_list<[](d0,d1,d2,d3)->(d3,d1,d0,d2)> }
      : (!wave.tensor<[@A, @B, @C, @D] of f32, <global>>) -> !wave.tensor<any of f32, <register>>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
  // CHECK-LABEL: @read_with_mapping_backward
  func.func @read_with_mapping_backward() {
    %mem = water_test.wave_tensor : !wave.tensor<any of f32, <global>>
    // CHECK: (!wave.tensor<[@C, @B, @D, @A] of f32, <global>>) -> !wave.tensor<[@A, @B, @C, @D] of f32, <register>>
    wave.read %mem { mapping = #wave.expr_list<[](d0,d1,d2,d3)->(d3,d1,d0,d2)> }
      : (!wave.tensor<any of f32, <global>>) -> !wave.tensor<[@A, @B, @C, @D] of f32, <register>>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
  // CHECK-LABEL: @write_with_mapping_backward_memory_to_value
  func.func @write_with_mapping_backward_memory_to_value(%mem: !wave.tensor<[@A, @B, @C, @D] of f32, <global>>) {
    %val = water_test.wave_tensor : !wave.tensor<any of f32, <register>>
    // CHECK: wave.write %{{.*}}, %{{.*}} : !wave.tensor<[@D, @B, @A, @C] of f32, <register>>, !wave.tensor<[@A, @B, @C, @D] of f32, <global>>
    wave.write %val, %mem { mapping = #wave.expr_list<[](d0,d1,d2,d3)->(d3,d1,d0,d2)> }
      : !wave.tensor<any of f32, <register>>, !wave.tensor<[@A, @B, @C, @D] of f32, <global>>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>] {
  // CHECK-LABEL: @write_with_mapping_backward_value_to_memory
  func.func @write_with_mapping_backward_value_to_memory(%val: !wave.tensor<[@A, @B, @C, @D] of f32, <register>>) {
    %mem = water_test.wave_tensor : !wave.tensor<any of f32, <global>>
    // CHECK: wave.write %{{.*}}, %{{.*}} : !wave.tensor<[@A, @B, @C, @D] of f32, <register>>, !wave.tensor<[@C, @B, @D, @A] of f32, <global>>
    wave.write %val, %mem { mapping = #wave.expr_list<[](d0,d1,d2,d3)->(d3,d1,d0,d2)> }
      : !wave.tensor<[@A, @B, @C, @D] of f32, <register>>, !wave.tensor<any of f32, <global>>
    return
  }
}
