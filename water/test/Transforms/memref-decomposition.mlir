// RUN: water-opt %s -water-memref-decomposition | FileCheck %s

// CHECK-LABEL: func.func @simple_load
func.func @simple_load(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: memref.load
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>
  return %0 : f32
}

// CHECK-LABEL: func.func @simple_store
func.func @simple_store(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: f32) {
  // CHECK: memref.store
  memref.store %val, %arg0[%i, %j] : memref<10x20xf32>
  return
}
