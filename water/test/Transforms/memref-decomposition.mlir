// RUN: water-opt %s --water-memref-decomposition --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @load_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @load_2d(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][] : memref<?xi8> to memref<800xi8>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[VIEW]] : memref<800xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[CAST2]][%[[IDX]]][] : memref<?xi8> to memref<f32>
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW2]][] : memref<f32>
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>
  return %0 : f32
}

// CHECK-LABEL: func.func @store_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
func.func @store_2d(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: f32) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][] : memref<?xi8> to memref<800xi8>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[VIEW]] : memref<800xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[CAST2]][%[[IDX]]][] : memref<?xi8> to memref<f32>
  // CHECK: memref.store %[[ARG3]], %[[VIEW2]][] : memref<f32>
  memref.store %val, %arg0[%i, %j] : memref<10x20xf32>
  return
}

// CHECK-LABEL: func.func @load_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8x16xf16>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @load_3d(%arg0: memref<4x8x16xf16>, %i: index, %j: index, %k: index) -> f16 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<4x8x16xf16> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][] : memref<?xi8> to memref<1024xi8>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[VIEW]] : memref<1024xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[CAST2]][%[[IDX]]][] : memref<?xi8> to memref<f16>
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW2]][] : memref<f16>
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j, %k] : memref<4x8x16xf16>
  return %0 : f16
}

// CHECK-LABEL: func.func @vector_load_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @vector_load_2d(%arg0: memref<10x20xf32>, %i: index, %j: index) -> vector<4xf32> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][] : memref<?xi8> to memref<800xi8>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[VIEW]] : memref<800xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[CAST2]][%[[IDX]]][] : memref<?xi8> to memref<vector<4xf32>>
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW2]][] : memref<vector<4xf32>>
  // CHECK: return %[[LOAD]]
  %0 = vector.load %arg0[%i, %j] : memref<10x20xf32>, vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: vector<4xf32>)
func.func @vector_store_2d(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: vector<4xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][] : memref<?xi8> to memref<800xi8>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[VIEW]] : memref<800xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[CAST2]][%[[IDX]]][] : memref<?xi8> to memref<vector<4xf32>>
  // CHECK: memref.store %[[ARG3]], %[[VIEW2]][] : memref<vector<4xf32>>
  vector.store %val, %arg0[%i, %j] : memref<10x20xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @vector_load_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8x16xf16>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @vector_load_3d(%arg0: memref<4x8x16xf16>, %i: index, %j: index, %k: index) -> vector<8xf16> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<4x8x16xf16> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][] : memref<?xi8> to memref<1024xi8>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[VIEW]] : memref<1024xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[CAST2]][%[[IDX]]][] : memref<?xi8> to memref<vector<8xf16>>
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW2]][] : memref<vector<8xf16>>
  // CHECK: return %[[LOAD]]
  %0 = vector.load %arg0[%i, %j, %k] : memref<4x8x16xf16>, vector<8xf16>
  return %0 : vector<8xf16>
}

// CHECK-LABEL: func.func @multiple_loads
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @multiple_loads(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][] : memref<?xi8> to memref<800xi8>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[VIEW]] : memref<800xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[CAST2]][%[[IDX]]][] : memref<?xi8> to memref<f32>
  // CHECK: %[[LOAD0:.*]] = memref.load %[[VIEW2]][] : memref<f32>
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>

  // CHECK: %[[IDX3:.*]] = affine.apply
  // CHECK: %[[VIEW3:.*]] = memref.view %[[CAST2]][%[[IDX3]]][] : memref<?xi8> to memref<f32>
  // CHECK: %[[LOAD1:.*]] = memref.load %[[VIEW3]][] : memref<f32>
  %1 = memref.load %arg0[%j, %i] : memref<10x20xf32>

  // CHECK: %[[ADD:.*]] = arith.addf %[[LOAD0]], %[[LOAD1]]
  %2 = arith.addf %0, %1 : f32
  // CHECK: return %[[ADD]]
  return %2 : f32
}

// CHECK-LABEL: func.func @different_types
// CHECK-SAME: (%[[ARG0:.*]]: memref<8x16xi32>, %[[ARG1:.*]]: memref<4x8xf64>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @different_types(%arg0: memref<8x16xi32>, %arg1: memref<4x8xf64>, %i: index, %j: index) -> (i32, f64) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // Canonicalizer processes arg1 first
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : memref<4x8xf64> to memref<?xi8>
  // CHECK: %[[VIEW1:.*]] = memref.view %[[CAST1]][%[[C0]]][] : memref<?xi8> to memref<256xi8>
  // CHECK: %[[CAST1_2:.*]] = memref.cast %[[VIEW1]] : memref<256xi8> to memref<?xi8>
  // Then arg0
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<8x16xi32> to memref<?xi8>
  // CHECK: %[[VIEW0:.*]] = memref.view %[[CAST0]][%[[C0]]][] : memref<?xi8> to memref<512xi8>
  // CHECK: %[[CAST0_2:.*]] = memref.cast %[[VIEW0]] : memref<512xi8> to memref<?xi8>
  // CHECK: %[[IDX0:.*]] = affine.apply
  // CHECK: %[[VIEW0_2:.*]] = memref.view %[[CAST0_2]][%[[IDX0]]][] : memref<?xi8> to memref<i32>
  // CHECK: %[[LOAD0:.*]] = memref.load %[[VIEW0_2]][] : memref<i32>
  %0 = memref.load %arg0[%i, %j] : memref<8x16xi32>

  // CHECK: %[[IDX1:.*]] = affine.apply
  // CHECK: %[[VIEW1_2:.*]] = memref.view %[[CAST1_2]][%[[IDX1]]][] : memref<?xi8> to memref<f64>
  // CHECK: %[[LOAD1:.*]] = memref.load %[[VIEW1_2]][] : memref<f64>
  %1 = memref.load %arg1[%i, %j] : memref<4x8xf64>
  // CHECK: return %[[LOAD0]], %[[LOAD1]]
  return %0, %1 : i32, f64
}

// CHECK-LABEL: func.func @reinterpret_cast_0d
// CHECK-SAME: (%[[BASE:.*]]: memref<f32>)
func.func @reinterpret_cast_0d(%base: memref<f32>) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<f32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][] : memref<?xi8> to memref<4xi8>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[VIEW]] : memref<4xi8> to memref<?xi8>
  // CHECK: %[[VIEW2:.*]] = memref.view %[[CAST2]][%[[C0]]][] : memref<?xi8> to memref<8xi8>
  // CHECK: %[[CAST3:.*]] = memref.cast %[[VIEW2]] : memref<8xi8> to memref<?xi8>
  // CHECK: %[[CAST4:.*]] = builtin.unrealized_conversion_cast %[[CAST3]] : memref<?xi8> to memref<f32>
  // CHECK: %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[CAST4]] to offset: [0], sizes: [1, 2], strides: [3, 4]
  // CHECK: memref.cast %[[REINTERPRET]]
  %0 = memref.reinterpret_cast %base to offset: [0], sizes: [1, 2], strides: [3, 4] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}


// CHECK-LABEL: func.func @reinterpret_cast
// CHECK-SAME: (%[[BASE:.*]]: memref<100xf32>, %[[OFFSET:.*]]: index, %[[SIZE0:.*]]: index, %[[SIZE1:.*]]: index, %[[STRIDE0:.*]]: index, %[[STRIDE1:.*]]: index)
func.func @reinterpret_cast(%base: memref<100xf32>, %offset: index, %size0: index, %size1: index, %stride0: index, %stride1: index) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<100xf32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][] : memref<?xi8> to memref<400xi8>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[VIEW]] : memref<400xi8> to memref<?xi8>
  // CHECK: %[[SIZE:.*]] = affine.apply
  // CHECK: %[[OFF:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[CAST2]][%[[OFF]]][%[[SIZE]]] : memref<?xi8> to memref<?xi8>
  // CHECK: %[[CAST3:.*]] = builtin.unrealized_conversion_cast %[[VIEW2]] : memref<?xi8> to memref<f32>
  // CHECK: %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[CAST3]] to offset: [0], sizes: [%[[SIZE0]], %[[SIZE1]]], strides: [%[[STRIDE0]], %[[STRIDE1]]]
  // CHECK: memref.cast %[[REINTERPRET]]
  %0 = memref.reinterpret_cast %base to offset: [%offset], sizes: [%size0, %size1], strides: [%stride0, %stride1] : memref<100xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}
