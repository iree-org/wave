// RUN: water-opt %s --water-materialize-reg-copy | FileCheck %s

// CHECK-LABEL: func @test_simple_load
func.func @test_simple_load(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %arg0[%arg1, %arg2] [1, 1] [1, 1]
  // CHECK-SAME: memref<10x20xf32> to memref<1x1xf32, strided<[20, 1], offset: ?>>
  // CHECK: %[[TEMP:.*]] = memref.alloca() : memref<1x1xf32, 128 : i32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[TEMP]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = memref.load %[[TEMP]][%[[C0]], %[[C0]]]
  // CHECK: return %[[RESULT]]
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>
  return %0 : f32
}

// CHECK-LABEL: func @test_simple_vector_load
func.func @test_simple_vector_load(%arg0: memref<10x20xf32>, %i: index, %j: index) -> vector<4xf32> {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %arg0[%arg1, %arg2] [1, 4] [1, 1]
  // CHECK-SAME: memref<10x20xf32> to memref<1x4xf32, strided<[20, 1], offset: ?>>
  // CHECK: %[[TEMP:.*]] = memref.alloca() : memref<1x4xf32, 128 : i32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[TEMP]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = vector.load %[[TEMP]][%[[C0]], %[[C0]]]
  // CHECK: return %[[RESULT]]
  %0 = vector.load %arg0[%i, %j] : memref<10x20xf32>, vector<4xf32>
  return %0 : vector<4xf32>
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
  // CHECK: %[[RESULT:.*]] = memref.load %[[TEMP]][%[[C0]], %[[C0]], %[[C0]]]
  // CHECK: return %[[RESULT]]
  %0 = memref.load %arg0[%i, %j, %k] : memref<8x16x32xi32>
  return %0 : i32
}

// CHECK-LABEL: func @test_multiple_loads
func.func @test_multiple_loads(%arg0: memref<10x10xf32>, %i: index, %j: index) -> f32 {
  // First load: subview, alloca, copy
  // CHECK: memref.subview
  // CHECK: memref.alloca() : memref<1x1xf32, 128 : i32>
  // CHECK: memref.copy
  %0 = memref.load %arg0[%i, %j] : memref<10x10xf32>

  // Second load: subview, alloca, copy
  // CHECK: memref.subview
  // CHECK: memref.alloca() : memref<1x1xf32, 128 : i32>
  // CHECK: memref.copy
  %1 = memref.load %arg0[%j, %i] : memref<10x10xf32>

  // Now the actual loads happen right before the addf (late as possible)
  // CHECK: memref.load
  // CHECK: memref.load
  // CHECK: arith.addf
  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}

// CHECK-LABEL: func @test_skip_memspace_128
func.func @test_skip_memspace_128(%arg0: memref<10xf32>, %arg1: memref<5xf32, 128 : i32>, %i: index) -> f32 {
  // This load should be transformed (from default memspace)
  // First: subview, alloca, copy
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %arg0[%arg2] [1] [1]
  // CHECK: %[[TEMP:.*]] = memref.alloca() : memref<1xf32, 128 : i32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[TEMP]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  %0 = memref.load %arg0[%i] : memref<10xf32>

  // This load should NOT be transformed (already from memspace 128)
  // It stays in place
  // CHECK: %[[VAL1:.*]] = memref.load %arg1[%arg2] : memref<5xf32, 128 : i32>
  %1 = memref.load %arg1[%i] : memref<5xf32, 128 : i32>

  // The load from temp happens late (right before addf)
  // CHECK: %[[VAL0:.*]] = memref.load %[[TEMP]][%[[C0]]]
  // Note: operands may be reordered
  // CHECK: arith.addf
  %result = arith.addf %0, %1 : f32
  // CHECK: return
  return %result : f32
}

// CHECK-LABEL: func @test_control_flow
func.func @test_control_flow(%arg0: memref<10xf32>, %cond: i1, %i: index) -> f32 {
  // Load happens once, but value is used in multiple blocks
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %arg0[%arg2] [1] [1]
  // CHECK: %[[TEMP:.*]] = memref.alloca() : memref<1xf32, 128 : i32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[TEMP]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  %val = memref.load %arg0[%i] : memref<10xf32>

  // CHECK: cf.cond_br
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // First block: load happens here before the addf
  // CHECK: ^bb1:
  // CHECK: %[[CONST1:.*]] = arith.constant 1.0
  // CHECK: %[[LOAD1:.*]] = memref.load %[[TEMP]][%[[C0]]]
  // CHECK: %[[ADD1:.*]] = arith.addf %[[LOAD1]], %[[CONST1]]
  %c1 = arith.constant 1.0 : f32
  %sum1 = arith.addf %val, %c1 : f32
  // CHECK: cf.br ^bb3(%[[ADD1]]
  cf.br ^bb3(%sum1 : f32)

^bb2:
  // Second block: another load happens here before the mulf
  // CHECK: ^bb2:
  // CHECK: %[[CONST2:.*]] = arith.constant 2.0
  // CHECK: %[[LOAD2:.*]] = memref.load %[[TEMP]][%[[C0]]]
  // CHECK: %[[MUL:.*]] = arith.mulf %[[LOAD2]], %[[CONST2]]
  %c2 = arith.constant 2.0 : f32
  %prod = arith.mulf %val, %c2 : f32
  // CHECK: cf.br ^bb3(%[[MUL]]
  cf.br ^bb3(%prod : f32)

^bb3(%result: f32):
  // CHECK: ^bb3(%[[RESULT:.*]]: f32):
  // CHECK: return %[[RESULT]]
  return %result : f32
}

// CHECK-LABEL: func @test_loop_hoist
func.func @test_loop_hoist(%arg0: memref<100xf32>, %lb: index, %ub: index, %step: index, %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<1xf32, 128 : i32>
  // CHECK: arith.constant 0 : index
  // CHECK: memref.store %arg4, %[[ALLOCA]]
  // CHECK: scf.for %[[IV:.*]] = %arg1 to %arg2 step %arg3 iter_args(%[[ITER_ARG:.*]] = %arg4)
  %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> (f32) {
    // CHECK: memref.load %[[ALLOCA]]
    // CHECK: memref.store %{{.*}}, %arg0[%c0]
    memref.store %arg, %arg0[%c0] : memref<100xf32>
    %alloca = memref.alloca() : memref<1xf32, 128 : i32>
    %subview = memref.subview %arg0[%iv] [1] [1] : memref<100xf32> to memref<1xf32, strided<[1], offset: ?>>
    memref.copy %subview, %alloca : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, 128 : i32>
    %val = memref.load %alloca[%c0] : memref<1xf32, 128 : i32>
    // CHECK: memref.subview
    // CHECK: memref.copy
    // CHECK: memref.load %[[ALLOCA]]
    // CHECK: scf.yield
    scf.yield %val : f32
  }
  // CHECK: memref.load %[[ALLOCA]]
  // CHECK: return
  return %result : f32
}
