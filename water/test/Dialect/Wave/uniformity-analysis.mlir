// RUN: water-opt %s --water-wave-uniformity-analysis --split-input-file | FileCheck %s

// CHECK-LABEL: @simple_uniform
func.func @simple_uniform(%a: i32, %b: i32) -> i32 {
  // CHECK: arith.addi
  %c = arith.addi %a, %b : i32
  return %c : i32
}

// -----

// CHECK-LABEL: @simple_arithmetic
func.func @simple_arithmetic(%a: f32, %b: f32) -> f32 {
  // CHECK: arith.addf
  %c = arith.addf %a, %b : f32
  // CHECK: arith.mulf
  %d = arith.mulf %c, %b : f32
  return %d : f32
}

// -----

// CHECK-LABEL: @control_flow
func.func @control_flow(%cond: i1, %a: f32, %b: f32) -> f32 {
  // CHECK: cf.cond_br
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // CHECK: arith.addf
  %sum = arith.addf %a, %b : f32
  cf.br ^bb3(%sum : f32)

^bb2:
  // CHECK: arith.mulf
  %prod = arith.mulf %a, %b : f32
  cf.br ^bb3(%prod : f32)

^bb3(%result: f32):
  return %result : f32
}
