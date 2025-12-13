// RUN: water-opt %s --water-materialize-reg-copy | FileCheck %s

// CHECK-LABEL: func @test_simple_load
func.func @test_simple_load(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %arg0[%arg1, %arg2] [1, 1] [1, 1]
  // CHECK-SAME: memref<10x20xf32> to memref<1x1xf32, strided<[20, 1], offset: ?>>
  // CHECK: %[[TEMP:.*]] = memref.alloca() : memref<1x1xf32, 128 : i32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[TEMP]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = memref.load %[[TEMP]][%[[C0]], %[[C0_1]]]
  // CHECK: return %[[RESULT]]
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>
  return %0 : f32
}

// CHECK-LABEL: func @test_1d_load
func.func @test_1d_load(%arg0: memref<100xf16>, %i: index) -> f16 {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %arg0[%arg1] [1] [1]
  // CHECK-SAME: memref<100xf16> to memref<1xf16, strided<[1], offset: ?>>
  // CHECK: %[[TEMP:.*]] = memref.alloca() : memref<1xf16, 128 : i32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[TEMP]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = memref.load %[[TEMP]][%[[C0]]]
  // CHECK: return %[[RESULT]]
  %0 = memref.load %arg0[%i] : memref<100xf16>
  return %0 : f16
}

// CHECK-LABEL: func @test_3d_load
func.func @test_3d_load(%arg0: memref<8x16x32xi32>, %i: index, %j: index, %k: index) -> i32 {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %arg0[%arg1, %arg2, %arg3] [1, 1, 1] [1, 1, 1]
  // CHECK-SAME: memref<8x16x32xi32> to memref<1x1x1xi32, strided<[512, 32, 1], offset: ?>>
  // CHECK: %[[TEMP:.*]] = memref.alloca() : memref<1x1x1xi32, 128 : i32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[TEMP]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[C0_2:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = memref.load %[[TEMP]][%[[C0]], %[[C0_1]], %[[C0_2]]]
  // CHECK: return %[[RESULT]]
  %0 = memref.load %arg0[%i, %j, %k] : memref<8x16x32xi32>
  return %0 : i32
}

// CHECK-LABEL: func @test_multiple_loads
func.func @test_multiple_loads(%arg0: memref<10x10xf32>, %i: index, %j: index) -> f32 {
  // CHECK: memref.subview
  // CHECK: memref.alloca() : memref<1x1xf32, 128 : i32>
  // CHECK: memref.copy
  // CHECK: memref.load
  %0 = memref.load %arg0[%i, %j] : memref<10x10xf32>

  // CHECK: memref.subview
  // CHECK: memref.alloca() : memref<1x1xf32, 128 : i32>
  // CHECK: memref.copy
  // CHECK: memref.load
  %1 = memref.load %arg0[%j, %i] : memref<10x10xf32>

  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}
