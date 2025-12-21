// RUN: water-opt %s --water-memref-lowering | FileCheck %s

// CHECK-LABEL: func.func @test
func.func @test(%arg0: memref<?xi8>) -> memref<?xi8> {
  // CHECK: return
  return %arg0 : memref<?xi8>
}
