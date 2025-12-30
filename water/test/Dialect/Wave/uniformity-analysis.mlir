// RUN: water-opt %s --water-wave-uniformity-analysis | FileCheck %s

// CHECK-LABEL: @constant_uniform
func.func @constant_uniform() -> i32 {
  // CHECK: arith.constant {wave.uniform}
  %c = arith.constant 42 : i32
  return %c : i32
}
