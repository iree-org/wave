// RUN: water-opt %s -allow-unregistered-dialect -lower-wave-to-mlir -split-input-file | FileCheck %s

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  // CHECK-LABEL: func.func @lower_register
  func.func @lower_register() attributes {wave.hyperparameters = #wave.hyperparameters<{X = 100, Y = 10, Z = 1}>} {
    // CHECK-NOT: wave.register
    // CHECK:     arith.constant dense<0.000000e+00> : vector<10x1xf32>
    %cst = arith.constant 0.0 : f32
    wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>

    // CHECK:     arith.constant dense<1.000000e+00> : vector<10x1x100xf32>
    %cst1 = arith.constant 1.0 : f32
    wave.register %cst1 : !wave.tensor<[@Y, @Z, @X] of f32, <register>>

    // CHECK:     arith.constant dense<2> : vector<10x1x100xi32>
    %cst2 = arith.constant 2 : i32
    wave.register %cst2 : !wave.tensor<[@Y, @Z, @X] of i32, <register>>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  // CHECK-LABEL: func.func @lower_register_with_unrealized_cast
  func.func @lower_register_with_unrealized_cast() attributes {wave.hyperparameters = #wave.hyperparameters<{X = 100, Y = 10, Z = 1}>} {
    // CHECK-NOT: wave.register
    // CHECK:     %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<10x1xf32>
    // CHECK:     %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[CST]] : vector<10x1xf32> to !wave.tensor<[@Y, @Z] of f32, <register>>
    // CHECK:     "test.foo"(%[[CAST]])
    %cst = arith.constant 0.0 : f32
    %0 = wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>
    "test.foo"(%0) : (!wave.tensor<[@Y, @Z] of f32, <register>>) -> ()
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  // CHECK-LABEL: func.func @lower_nested_register
  func.func @lower_nested_register(%cond: i1) attributes {wave.hyperparameters = #wave.hyperparameters<{X = 100, Y = 10, Z = 1}>} {
    // CHECK-NOT: wave.register
    scf.if %cond {
      // CHECK:     arith.constant dense<0.000000e+00> : vector<10x1xf32>
      %cst = arith.constant 0.0 : f32
      wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>
      scf.yield
    } else {
      // CHECK:     arith.constant dense<1.000000e+00> : vector<10x1x100xf32>
      %cst1 = arith.constant 1.0 : f32
      wave.register %cst1 : !wave.tensor<[@Y, @Z, @X] of f32, <register>>
      scf.yield
    }
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  // CHECK-LABEL: func.func @lower_add
  func.func @lower_add() attributes {wave.hyperparameters = #wave.hyperparameters<{X = 100, Y = 10, Z = 1}>} {
    // CHECK-NOT: wave.add
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<100x10x1xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<100x10x1xf32>
    // CHECK:     arith.addf %[[LHS]], %[[RHS]] : vector<100x10x1xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : !wave.tensor<[@X, @Y, @Z] of f32, <register>>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : !wave.tensor<[@X, @Y, @Z] of f32, <register>>
    %addf = wave.add %lhs, %rhs : (!wave.tensor<[@X, @Y, @Z] of f32, <register>>, !wave.tensor<[@X, @Y, @Z] of f32, <register>>) -> !wave.tensor<[@X, @Y, @Z] of f32, <register>>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<2> : vector<10x1x100xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<10x1x100xi32>
    // CHECK:     arith.addi %[[LHSI]], %[[RHSI]] : vector<10x1x100xi32>
    %cst2 = arith.constant 2 : i32
    %lhsi = wave.register %cst2 : !wave.tensor<[@Y, @Z, @X] of i32, <register>>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : !wave.tensor<[@Y, @Z, @X] of i32, <register>>
    %addi = wave.add %lhsi, %rhsi : (!wave.tensor<[@Y, @Z, @X] of i32, <register>>, !wave.tensor<[@Y, @Z, @X] of i32, <register>>) -> !wave.tensor<[@Y, @Z, @X] of i32, <register>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  // CHECK-LABEL: func.func @lower_mul
  func.func @lower_mul() attributes {wave.hyperparameters = #wave.hyperparameters<{X = 100, Y = 10, Z = 1}>} {
    // CHECK-NOT: wave.mul
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<100x10x1xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<100x10x1xf32>
    // CHECK:     arith.mulf %[[LHS]], %[[RHS]] : vector<100x10x1xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : !wave.tensor<[@X, @Y, @Z] of f32, <register>>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : !wave.tensor<[@X, @Y, @Z] of f32, <register>>
    %mulf = wave.mul %lhs, %rhs : (!wave.tensor<[@X, @Y, @Z] of f32, <register>>, !wave.tensor<[@X, @Y, @Z] of f32, <register>>) -> !wave.tensor<[@X, @Y, @Z] of f32, <register>>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<2> : vector<10x1x100xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<10x1x100xi32>
    // CHECK:     arith.muli %[[LHSI]], %[[RHSI]] : vector<10x1x100xi32>
    %cst2 = arith.constant 2 : i32
    %lhsi = wave.register %cst2 : !wave.tensor<[@Y, @Z, @X] of i32, <register>>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : !wave.tensor<[@Y, @Z, @X] of i32, <register>>
    %muli = wave.mul %lhsi, %rhsi : (!wave.tensor<[@Y, @Z, @X] of i32, <register>>, !wave.tensor<[@Y, @Z, @X] of i32, <register>>) -> !wave.tensor<[@Y, @Z, @X] of i32, <register>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  // CHECK-LABEL: func.func @lower_div
  func.func @lower_div() attributes {wave.hyperparameters = #wave.hyperparameters<{X = 100, Y = 10, Z = 1}>} {
    // CHECK-NOT: wave.div
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<100x10x1xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<100x10x1xf32>
    // CHECK:     arith.divf %[[LHS]], %[[RHS]] : vector<100x10x1xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : !wave.tensor<[@X, @Y, @Z] of f32, <register>>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : !wave.tensor<[@X, @Y, @Z] of f32, <register>>
    %divf = wave.div %lhs, %rhs : (!wave.tensor<[@X, @Y, @Z] of f32, <register>>, !wave.tensor<[@X, @Y, @Z] of f32, <register>>) -> !wave.tensor<[@X, @Y, @Z] of f32, <register>>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<-2> : vector<10x1x100xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<10x1x100xi32>
    // CHECK:     arith.divsi %[[LHSI]], %[[RHSI]] : vector<10x1x100xi32>
    %cst2 = arith.constant -2 : i32
    %lhsi = wave.register %cst2 : !wave.tensor<[@Y, @Z, @X] of i32, <register>>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : !wave.tensor<[@Y, @Z, @X] of i32, <register>>
    %divsi = wave.div %lhsi, %rhsi : (!wave.tensor<[@Y, @Z, @X] of i32, <register>>, !wave.tensor<[@Y, @Z, @X] of i32, <register>>) -> !wave.tensor<[@Y, @Z, @X] of i32, <register>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
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

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
// CHECK-LABEL: @lower_alloc
func.func @lower_alloc() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 4, BLOCK_K = 28, M = 128, N=128, K= 128}>}  {
  // CHECK: memref.alloc() : memref<4x32xbf16, #gpu.address_space<workgroup>>
  %buf = wave.allocate
    { distributed_shape = #wave.distributed_shape<[BLOCK_M, BLOCK_K] -> (BLOCK_M, BLOCK_K + 4)>}
    : !wave.tensor<[@M, @K] of bf16, <shared>>
  wave.read %buf : (!wave.tensor<[@M, @K] of bf16, <shared>>) -> !wave.tensor<[@M, @K] of bf16, <register>>
  return
  }
}
