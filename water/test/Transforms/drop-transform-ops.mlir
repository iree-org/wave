// RUN: water-opt %s --water-drop-transform-ops | FileCheck %s

// CHECK-LABEL: module
module {
  // CHECK: func.func @regular_function
  func.func @regular_function(%arg0: f32) -> f32 {
    return %arg0 : f32
  }

  // CHECK-NOT: transform.named_sequence
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.yield
    }
  }

  // CHECK: func.func @another_regular_function
  func.func @another_regular_function(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %result = arith.addi %arg0, %c1 : i32
    return %result : i32
  }
}
