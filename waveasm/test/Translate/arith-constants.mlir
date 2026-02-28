// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s
//
// Test arith constant translation

module {
  // CHECK: arith.constant
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c42 = arith.constant 42 : i32
  %cf = arith.constant 3.14 : f32
}
