// RUN: water-opt %s -allow-unregistered-dialect -lower-wave-to-mlir -split-input-file --verify-diagnostics | FileCheck %s

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @no_hyperparams() {
    %cst = arith.constant 0.0 : f32
    // expected-error @below {{wave dialect operation with no hyperparameters provided by any ancestor}}
    wave.register %cst : vector<4xf32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_register
  func.func @lower_register() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.register
    // CHECK:     arith.constant dense<0.000000e+00> : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    wave.register %cst : vector<4xf32>

    // CHECK:     arith.constant dense<1.000000e+00> : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    wave.register %cst1 : vector<4xf32>

    // CHECK:     arith.constant dense<2> : vector<8xi32>
    %cst2 = arith.constant 2 : i32
    wave.register %cst2 : vector<8xi32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_register_with_unrealized_cast
  func.func @lower_register_with_unrealized_cast() attributes {wave.hyperparameters = #wave.hyperparameters<{Y = 10, Z = 1, BLOCK_M = 1, BLOCK_K = 2}>} {
    // CHECK-NOT: wave.register
    // CHECK:     %[[ALLOC:.*]] = memref.alloc() : memref<1x6xf32, #gpu.address_space<workgroup>>
    // CHECK:     %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]] : memref<1x6xf32, #gpu.address_space<workgroup>> to !wave.tensor<[@Y, @Z] of f32, <shared>>
    // CHECK:     "test.foo"(%[[CAST]])
    %0 = wave.allocate
    { distributed_shape = #wave.distributed_shape<[BLOCK_M, BLOCK_K] -> (BLOCK_M, BLOCK_K + 4)>}
    : !wave.tensor<[@Y, @Z] of f32, <shared>>
    "test.foo"(%0) : (!wave.tensor<[@Y, @Z] of f32, <shared>>) -> ()
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_nested_register
  func.func @lower_nested_register(%cond: i1) attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.register
    scf.if %cond {
      // CHECK:     arith.constant dense<0.000000e+00> : vector<1xf32>
      %cst = arith.constant 0.0 : f32
      wave.register %cst : vector<1xf32>
      scf.yield
    } else {
      // CHECK:     arith.constant dense<1.000000e+00> : vector<1xf32>
      %cst1 = arith.constant 1.0 : f32
      wave.register %cst1 : vector<1xf32>
      scf.yield
    }
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_add
  func.func @lower_add() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.add
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.addf %[[LHS]], %[[RHS]] : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %addf = wave.add %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<2> : vector<2xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<2xi32>
    // CHECK:     arith.addi %[[LHSI]], %[[RHSI]] : vector<2xi32>
    %cst2 = arith.constant 2 : i32
    %lhsi = wave.register %cst2 : vector<2xi32>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : vector<2xi32>
    %addi = wave.add %lhsi, %rhsi : (vector<2xi32>, vector<2xi32>) -> vector<2xi32>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_mul
  func.func @lower_mul() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.mul
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.mulf %[[LHS]], %[[RHS]] : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %mulf = wave.mul %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<2> : vector<4xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<4xi32>
    // CHECK:     arith.muli %[[LHSI]], %[[RHSI]] : vector<4xi32>
    %cst2 = arith.constant 2 : i32
    %lhsi = wave.register %cst2 : vector<4xi32>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : vector<4xi32>
    %muli = wave.mul %lhsi, %rhsi : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_div
  func.func @lower_div() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.div
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.divf %[[LHS]], %[[RHS]] : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %divf = wave.div %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<-2> : vector<4xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<4xi32>
    // CHECK:     arith.divsi %[[LHSI]], %[[RHSI]] : vector<4xi32>
    %cst2 = arith.constant -2 : i32
    %lhsi = wave.register %cst2 : vector<4xi32>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : vector<4xi32>
    %divsi = wave.div %lhsi, %rhsi : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
// CHECK-LABEL: func.func @lower_alloc_view
func.func @lower_alloc_view() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 4, BLOCK_K = 28, M = 128, N=128, K= 128}>}  {
  // CHECK: %[[BUFF:.*]] = memref.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  %parent = wave.allocate { distributed_shape = #wave.distributed_shape<[] -> (256)> }
    : !wave.tensor<[@M,@N,@K] of i8, <shared>>

  // CHECK: %[[OFF:.*]] = arith.constant 128 : index
  // CHECK: %[[VIEW:.*]] = memref.view %[[BUFF]][%[[OFF]]][]
  // CHECK-SAME: : memref<256xi8, #gpu.address_space<workgroup>> to memref<4x32xbf16, #gpu.address_space<workgroup>>
  %buf = wave.allocate in %parent : !wave.tensor<[@M,@N,@K] of i8, <shared>>
    { distributed_shape = #wave.distributed_shape<[BLOCK_M, BLOCK_K] -> (BLOCK_M, BLOCK_K + 4)>, offset = 128}
    : !wave.tensor<[@M, @K] of bf16, <shared>>
  return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
// CHECK-LABEL: @lower_alloc
func.func @lower_alloc() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 4, BLOCK_K = 28, M = 128, K= 128}>}  {
  // CHECK: memref.alloc() : memref<4x32xbf16, #gpu.address_space<workgroup>>
  %buf = wave.allocate
    { distributed_shape = #wave.distributed_shape<[BLOCK_M, BLOCK_K] -> (BLOCK_M, BLOCK_K + 4)>}
    : !wave.tensor<[@M, @K] of bf16, <shared>>
  return
  }
}
