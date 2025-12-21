// RUN: water-opt %s --water-memref-lowering | FileCheck %s

// CHECK-LABEL: func @test_signature
//  CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
func.func @test_signature(%arg0: memref<?xi8>) -> memref<?xi8> {
  // CHECK: return %[[ARG]] : !llvm.ptr
  return %arg0 : memref<?xi8>
}

// CHECK-LABEL: func @test_load
//  CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr)
func.func @test_load(%ptr: memref<f32>) -> f32 {
  // CHECK: %[[VAL:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> f32
  // CHECK: return %[[VAL]]
  %0 = memref.load %ptr[] : memref<f32>
  return %0 : f32
}

// CHECK-LABEL: func @test_store
//  CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[VAL:.*]]: f32)
func.func @test_store(%ptr: memref<f32>, %val: f32) {
  // CHECK: llvm.store %[[VAL]], %[[PTR]] : f32, !llvm.ptr
  memref.store %val, %ptr[] : memref<f32>
  return
}
