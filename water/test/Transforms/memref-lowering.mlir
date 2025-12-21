// RUN: water-opt %s --water-memref-lowering | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
func.func @test(%arg0: memref<?xi8>) -> memref<?xi8> {
  // CHECK: return %[[ARG]] : !llvm.ptr
  return %arg0 : memref<?xi8>
}
