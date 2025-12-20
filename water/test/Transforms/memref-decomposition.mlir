// RUN: water-opt %s --water-memref-decomposition | FileCheck %s

// CHECK-LABEL: func.func @load_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xi8>, %[[SIZE0:.*]]: index, %[[SIZE1:.*]]: index, %[[STRIDE0:.*]]: index, %[[STRIDE1:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index)
func.func @load_2d(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[VIEW:.*]] = memref.view %[[ARG0]]
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW]][] : memref<f32>
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>
  return %0 : f32
}

// CHECK-LABEL: func.func @store_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xi8>, %[[SIZE0:.*]]: index, %[[SIZE1:.*]]: index, %[[STRIDE0:.*]]: index, %[[STRIDE1:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[VAL:.*]]: f32)
func.func @store_2d(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: f32) {
  // CHECK: %[[VIEW:.*]] = memref.view %[[ARG0]]
  // CHECK: memref.store %[[VAL]], %[[VIEW]][] : memref<f32>
  memref.store %val, %arg0[%i, %j] : memref<10x20xf32>
  return
}

// CHECK-LABEL: func.func @load_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xi8>, %[[SIZE0:.*]]: index, %[[SIZE1:.*]]: index, %[[SIZE2:.*]]: index, %[[STRIDE0:.*]]: index, %[[STRIDE1:.*]]: index, %[[STRIDE2:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
func.func @load_3d(%arg0: memref<4x8x16xf16>, %i: index, %j: index, %k: index) -> f16 {
  // CHECK: %[[VIEW:.*]] = memref.view %[[ARG0]]
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW]][] : memref<f16>
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j, %k] : memref<4x8x16xf16>
  return %0 : f16
}

// CHECK-LABEL: func.func @multiple_loads
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xi8>, %[[SIZE0:.*]]: index, %[[SIZE1:.*]]: index, %[[STRIDE0:.*]]: index, %[[STRIDE1:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index)
func.func @multiple_loads(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[VIEW0:.*]] = memref.view %[[ARG0]]
  // CHECK: %[[LOAD0:.*]] = memref.load %[[VIEW0]][] : memref<f32>
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>

  // CHECK: %[[VIEW1:.*]] = memref.view %[[ARG0]]
  // CHECK: %[[LOAD1:.*]] = memref.load %[[VIEW1]][] : memref<f32>
  %1 = memref.load %arg0[%j, %i] : memref<10x20xf32>

  // CHECK: %[[ADD:.*]] = arith.addf %[[LOAD0]], %[[LOAD1]]
  %2 = arith.addf %0, %1 : f32
  // CHECK: return %[[ADD]]
  return %2 : f32
}

// CHECK-LABEL: func.func @different_types
func.func @different_types(%arg0: memref<8x16xi32>, %arg1: memref<4x8xf64>, %i: index, %j: index) {
  // CHECK: memref.view {{.*}} : memref<?xi8> to memref<i32>
  // CHECK: memref.load {{.*}} : memref<i32>
  %0 = memref.load %arg0[%i, %j] : memref<8x16xi32>

  // CHECK: memref.view {{.*}} : memref<?xi8> to memref<f64>
  // CHECK: memref.load {{.*}} : memref<f64>
  %1 = memref.load %arg1[%i, %j] : memref<4x8xf64>
  return
}
