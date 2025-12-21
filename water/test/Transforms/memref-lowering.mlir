// RUN: water-opt %s --water-memref-lowering | FileCheck %s

// CHECK-LABEL: func @test_signature_0d
//  CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
func.func @test_signature_0d(%arg0: memref<i32>) -> memref<i32> {
  // CHECK: return %[[ARG]] : !llvm.ptr
  return %arg0 : memref<i32>
}

// CHECK-LABEL: func @test_signature_1d
//  CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
func.func @test_signature_1d(%arg0: memref<?xi8>) -> memref<?xi8> {
  // CHECK: return %[[ARG]] : !llvm.ptr
  return %arg0 : memref<?xi8>
}

// CHECK-LABEL: func @test_unrealized_cast
//  CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
func.func @test_unrealized_cast(%arg0: memref<?xi8>) -> memref<i32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : memref<?xi8> to memref<i32>
  // CHECK: return %[[ARG]] : !llvm.ptr
  return %0 : memref<i32>
}

// CHECK-LABEL: func @test_cast
//  CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
func.func @test_cast(%arg0: memref<?xi8>) -> memref<2xi8> {
  %0 = memref.cast %arg0 : memref<?xi8> to memref<2xi8>
  // CHECK: return %[[ARG]] : !llvm.ptr
  return %0 : memref<2xi8>
}

// CHECK-LABEL: func @test_view
//  CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
func.func @test_view(%arg0: memref<?xi8>) -> memref<10xi8> {
  %c64 = arith.constant 64 : index
  %0 = memref.view %arg0[%c64][] : memref<?xi8> to memref<10xi8>
  // CHECK: %[[C64:.*]] = arith.constant 64 : index
  // CHECK: %[[C64I:.*]] = arith.index_cast %[[C64]] : index to i64
  // CHECK: %[[RES:.*]] = llvm.getelementptr %[[ARG]][%[[C64I]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: return %[[RES]] : !llvm.ptr
  return %0 : memref<10xi8>
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

// CHECK-LABEL: func @test_load_aligned
//  CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr)
func.func @test_load_aligned(%ptr: memref<f32>) -> f32 {
  // CHECK: %[[VAL:.*]] = llvm.load %[[PTR]] {alignment = 16 : i64} : !llvm.ptr -> f32
  // CHECK: return %[[VAL]]
  %0 = memref.load %ptr[] {alignment = 16 : i64} : memref<f32>
  return %0 : f32
}

// CHECK-LABEL: func @test_store_aligned
//  CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[VAL:.*]]: f32)
func.func @test_store_aligned(%ptr: memref<f32>, %val: f32) {
  // CHECK: llvm.store %[[VAL]], %[[PTR]] {alignment = 16 : i64} : f32, !llvm.ptr
  memref.store %val, %ptr[] {alignment = 16 : i64} : memref<f32>
  return
}

// CHECK-LABEL: func @test_load_nontemporal
//  CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr)
func.func @test_load_nontemporal(%ptr: memref<f32>) -> f32 {
  // CHECK: %[[VAL:.*]] = llvm.load %[[PTR]] {nontemporal} : !llvm.ptr -> f32
  // CHECK: return %[[VAL]]
  %0 = memref.load %ptr[] {nontemporal = true} : memref<f32>
  return %0 : f32
}

// CHECK-LABEL: func @test_store_nontemporal
//  CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[VAL:.*]]: f32)
func.func @test_store_nontemporal(%ptr: memref<f32>, %val: f32) {
  // CHECK: llvm.store %[[VAL]], %[[PTR]] {nontemporal} : f32, !llvm.ptr
  memref.store %val, %ptr[] {nontemporal = true} : memref<f32>
  return
}

// CHECK-LABEL: func @test_load_aligned_nontemporal
//  CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr)
func.func @test_load_aligned_nontemporal(%ptr: memref<f32>) -> f32 {
  // CHECK: %[[VAL:.*]] = llvm.load %[[PTR]] {alignment = 8 : i64, nontemporal} : !llvm.ptr -> f32
  // CHECK: return %[[VAL]]
  %0 = memref.load %ptr[] {alignment = 8 : i64, nontemporal = true} : memref<f32>
  return %0 : f32
}

// CHECK-LABEL: func @test_store_aligned_nontemporal
//  CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[VAL:.*]]: f32)
func.func @test_store_aligned_nontemporal(%ptr: memref<f32>, %val: f32) {
  // CHECK: llvm.store %[[VAL]], %[[PTR]] {alignment = 8 : i64, nontemporal} : f32, !llvm.ptr
  memref.store %val, %ptr[] {alignment = 8 : i64, nontemporal = true} : memref<f32>
  return
}
