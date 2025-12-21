// RUN: water-opt %s --water-memref-lowering | FileCheck %s

// CHECK-LABEL: func.func @test
func.func @test(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: memref.load
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>
  return %0 : f32
}
