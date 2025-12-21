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
