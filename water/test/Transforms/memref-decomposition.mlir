// RUN: water-opt %s --water-memref-decomposition | FileCheck %s

// CHECK-LABEL: func.func @load_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @load_2d(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG0]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C20:.*]] = arith.constant 20 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[SIZE:.*]] = affine.apply
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<f32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][%[[SIZE]]] : memref<?xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: affine.max
  // CHECK: %[[IDX2:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[VIEW]][%[[IDX2]]][] : memref<?xi8> to memref<f32>
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW2]][] : memref<f32>
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>
  return %0 : f32
}

// CHECK-LABEL: func.func @store_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
func.func @store_2d(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: f32) {
  // CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG0]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C20:.*]] = arith.constant 20 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[SIZE:.*]] = affine.apply
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<f32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][%[[SIZE]]] : memref<?xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: affine.max
  // CHECK: %[[IDX2:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[VIEW]][%[[IDX2]]][] : memref<?xi8> to memref<f32>
  // CHECK: memref.store %[[ARG3]], %[[VIEW2]][] : memref<f32>
  memref.store %val, %arg0[%i, %j] : memref<10x20xf32>
  return
}

// CHECK-LABEL: func.func @load_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8x16xf16>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @load_3d(%arg0: memref<4x8x16xf16>, %i: index, %j: index, %k: index) -> f16 {
  // CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[ARG0]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C128:.*]] = arith.constant 128 : index
  // CHECK: %[[C16:.*]] = arith.constant 16 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[SIZE:.*]] = affine.apply
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<f16> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][%[[SIZE]]] : memref<?xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: affine.max
  // CHECK: %[[IDX2:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[VIEW]][%[[IDX2]]][] : memref<?xi8> to memref<f16>
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW2]][] : memref<f16>
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j, %k] : memref<4x8x16xf16>
  return %0 : f16
}

// CHECK-LABEL: func.func @vector_load_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @vector_load_2d(%arg0: memref<10x20xf32>, %i: index, %j: index) -> vector<4xf32> {
  // CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG0]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C20:.*]] = arith.constant 20 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[SIZE:.*]] = affine.apply
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<f32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][%[[SIZE]]] : memref<?xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: affine.max
  // CHECK: %[[IDX2:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[VIEW]][%[[IDX2]]][] : memref<?xi8> to memref<vector<4xf32>>
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW2]][] : memref<vector<4xf32>>
  // CHECK: return %[[LOAD]]
  %0 = vector.load %arg0[%i, %j] : memref<10x20xf32>, vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: vector<4xf32>)
func.func @vector_store_2d(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: vector<4xf32>) {
  // CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG0]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C20:.*]] = arith.constant 20 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[SIZE:.*]] = affine.apply
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<f32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][%[[SIZE]]] : memref<?xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: affine.max
  // CHECK: %[[IDX2:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[VIEW]][%[[IDX2]]][] : memref<?xi8> to memref<vector<4xf32>>
  // CHECK: memref.store %[[ARG3]], %[[VIEW2]][] : memref<vector<4xf32>>
  vector.store %val, %arg0[%i, %j] : memref<10x20xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @vector_load_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8x16xf16>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @vector_load_3d(%arg0: memref<4x8x16xf16>, %i: index, %j: index, %k: index) -> vector<8xf16> {
  // CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[ARG0]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C128:.*]] = arith.constant 128 : index
  // CHECK: %[[C16:.*]] = arith.constant 16 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[SIZE:.*]] = affine.apply
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<f16> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][%[[SIZE]]] : memref<?xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: affine.max
  // CHECK: %[[IDX2:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[VIEW]][%[[IDX2]]][] : memref<?xi8> to memref<vector<8xf16>>
  // CHECK: %[[LOAD:.*]] = memref.load %[[VIEW2]][] : memref<vector<8xf16>>
  // CHECK: return %[[LOAD]]
  %0 = vector.load %arg0[%i, %j, %k] : memref<4x8x16xf16>, vector<8xf16>
  return %0 : vector<8xf16>
}

// CHECK-LABEL: func.func @multiple_loads
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @multiple_loads(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG0]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C20:.*]] = arith.constant 20 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[SIZE:.*]] = affine.apply
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<f32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]][%[[C0]]][%[[SIZE]]] : memref<?xi8> to memref<?xi8>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: affine.max
  // CHECK: %[[IDX2:.*]] = affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[VIEW]][%[[IDX2]]][] : memref<?xi8> to memref<f32>
  // CHECK: %[[LOAD0:.*]] = memref.load %[[VIEW2]][] : memref<f32>
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>

  // CHECK: %[[IDX3:.*]] = affine.apply
  // CHECK: affine.max
  // CHECK: %[[IDX4:.*]] = affine.apply
  // CHECK: %[[VIEW3:.*]] = memref.view %[[VIEW]][%[[IDX4]]][] : memref<?xi8> to memref<f32>
  // CHECK: %[[LOAD1:.*]] = memref.load %[[VIEW3]][] : memref<f32>
  %1 = memref.load %arg0[%j, %i] : memref<10x20xf32>

  // CHECK: %[[ADD:.*]] = arith.addf %[[LOAD0]], %[[LOAD1]]
  %2 = arith.addf %0, %1 : f32
  // CHECK: return %[[ADD]]
  return %2 : f32
}

// CHECK-LABEL: func.func @different_types
// CHECK-SAME: (%[[ARG0:.*]]: memref<8x16xi32>, %[[ARG1:.*]]: memref<4x8xf64>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @different_types(%arg0: memref<8x16xi32>, %arg1: memref<4x8xf64>, %i: index, %j: index) {
  // CHECK: memref.extract_strided_metadata %[[ARG1]]
  // CHECK: arith.constant 0 : index
  // CHECK: arith.constant 8 : index
  // CHECK: arith.constant 1 : index
  // CHECK: builtin.unrealized_conversion_cast {{.*}} : memref<f64> to memref<?xi8>
  // CHECK: memref.view {{.*}} : memref<?xi8> to memref<?xi8>
  // CHECK: memref.extract_strided_metadata %[[ARG0]]
  // CHECK: arith.constant 0 : index
  // CHECK: arith.constant 16 : index
  // CHECK: arith.constant 1 : index
  // CHECK: builtin.unrealized_conversion_cast {{.*}} : memref<i32> to memref<?xi8>
  // CHECK: memref.view {{.*}} : memref<?xi8> to memref<?xi8>
  // CHECK: memref.view {{.*}} : memref<?xi8> to memref<i32>
  // CHECK: memref.load {{.*}} : memref<i32>
  %0 = memref.load %arg0[%i, %j] : memref<8x16xi32>

  // CHECK: memref.view {{.*}} : memref<?xi8> to memref<f64>
  // CHECK: memref.load {{.*}} : memref<f64>
  %1 = memref.load %arg1[%i, %j] : memref<4x8xf64>
  return
}

// CHECK-LABEL: func.func @reinterpret_cast
// CHECK-SAME: (%[[BASE:.*]]: memref<f32>)
func.func @reinterpret_cast_0d(%base: memref<f32>) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK: %[[BASE_BUF:.*]], %[[BASE_OFF:.*]] = memref.extract_strided_metadata %[[BASE]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE_BUF]] : memref<f32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]]
  // CHECK: %[[VIEW2:.*]] = memref.view %[[VIEW]]
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %[[VIEW2]] : memref<?xi8> to memref<f32>
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: memref.reinterpret_cast %[[CAST2]] to offset: [%[[C0_1]]], sizes: [%[[C1]], %[[C2]]], strides: [%[[C3]], %[[C4]]]
  %0 = memref.reinterpret_cast %base to offset: [0], sizes: [1, 2], strides: [3, 4] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}


// CHECK-LABEL: func.func @reinterpret_cast
// CHECK-SAME: (%[[BASE:.*]]: memref<100xf32>, %[[OFFSET:.*]]: index, %[[SIZE0:.*]]: index, %[[SIZE1:.*]]: index, %[[STRIDE0:.*]]: index, %[[STRIDE1:.*]]: index)
func.func @reinterpret_cast(%base: memref<100xf32>, %offset: index, %size0: index, %size1: index, %stride0: index, %stride1: index) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK: %[[BASE_BUF:.*]], %[[BASE_OFF:.*]], %[[BASE_SIZE:.*]], %[[BASE_STRIDE:.*]] = memref.extract_strided_metadata %[[BASE]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: affine.apply
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BASE_BUF]] : memref<f32> to memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[CAST]]
  // CHECK: affine.apply
  // CHECK: %[[VIEW2:.*]] = memref.view %[[VIEW]]
  // CHECK: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %[[VIEW2]] : memref<?xi8> to memref<f32>
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: memref.reinterpret_cast %[[CAST2]] to offset: [%[[C0_1]]], sizes: [%[[SIZE0]], %[[SIZE1]]], strides: [%[[STRIDE0]], %[[STRIDE1]]]
  %0 = memref.reinterpret_cast %base to offset: [%offset], sizes: [%size0, %size1], strides: [%stride0, %stride1] : memref<100xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}
