// RUN: water-opt %s --water-wave-resolve-distributed-allocations --split-input-file | FileCheck %s

// Test basic shared memory allocation resolution.
// CHECK-LABEL: func.func @resolve_basic_alloc
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @resolve_basic_alloc() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_K = 32, M = 128, K = 64}>} {
    // CHECK: wave.allocate {distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K)>}
    // CHECK-SAME: memref<64x32xbf16, #gpu.address_space<workgroup>>
    %buf = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K)>}
      : !wave.tensor<[@M, @K] of bf16, <shared>>
    return
  }
}

// -----

// Test allocation with expression in distributed shape.
// CHECK-LABEL: func.func @resolve_alloc_with_expr
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @resolve_alloc_with_expr() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_K = 28, M = 128, K = 64}>} {
    // CHECK: wave.allocate {distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>}
    // CHECK-SAME: memref<64x32xbf16, #gpu.address_space<workgroup>>
    %buf = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>}
      : !wave.tensor<[@M, @K] of bf16, <shared>>
    return
  }
}

// -----

// Test that child allocation is also resolved.
// CHECK-LABEL: func.func @resolve_child_alloc
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @resolve_child_alloc() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_K = 32, M = 128, K = 64, SIZE = 8192}>} {
    // CHECK: %[[PARENT:.*]] = wave.allocate {distributed_shape = #wave.expr_list<[] -> (8192)>}
    // CHECK-SAME: memref<8192xi8, #gpu.address_space<workgroup>>
    %parent = wave.allocate { distributed_shape = #wave.expr_list<[] -> (8192)> }
      : !wave.tensor<[@SIZE] of i8, <shared>>

    // CHECK: wave.allocate in %[[PARENT]] : {{.*}} {distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K)>
    // CHECK-SAME: memref<64x32xbf16, #gpu.address_space<workgroup>>
    %buf = wave.allocate in %parent : !wave.tensor<[@SIZE] of i8, <shared>>
      { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K)>, offset = 0 }
      : !wave.tensor<[@M, @K] of bf16, <shared>>
    return
  }
}
