// RUN: water-opt %s --water-wave-resolve-distributed-allocations --split-input-file | FileCheck %s

// Test basic shared memory allocation resolution.
// CHECK: normalform.module [#wave.normal_form<full_types,resolved_allocations,ordered_syms>]
// CHECK-LABEL: func.func @resolve_basic_alloc
normalform.module [#wave.normal_form<full_types>] {
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
// CHECK: normalform.module [#wave.normal_form<full_types,resolved_allocations,ordered_syms>]
// CHECK-LABEL: func.func @resolve_alloc_with_expr
normalform.module [#wave.normal_form<full_types>] {
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
// CHECK: normalform.module [#wave.normal_form<full_types,resolved_allocations,ordered_syms>]
// CHECK-LABEL: func.func @resolve_child_alloc
normalform.module [#wave.normal_form<full_types>] {
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

// -----

// Test that resolved_allocations is set even when there are no allocations.
// CHECK: normalform.module [#wave.normal_form<full_types,resolved_allocations,ordered_syms>]
normalform.module [#wave.normal_form<full_types>] {
  func.func @no_allocations() {
    return
  }
}

// -----

// Test that resolved_allocations is set on a module without existing normal form.
// CHECK: normalform.module [#wave.normal_form<resolved_allocations,ordered_syms>]
normalform.module [] {
  func.func @resolve_without_existing_normal_form() attributes {wave.hyperparameters = #wave.hyperparameters<{M = 32}>} {
    // CHECK: wave.allocate
    // CHECK-SAME: memref<32xf32, #gpu.address_space<workgroup>>
    %0 = wave.allocate {distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)>}
      : !wave.tensor<[@M] of f32, <shared>>
    return
  }
}

// -----

// Test that ordered_syms is set on wave.read ops.
// The tensor has shape [@M, @K, @N] - ordered_syms should preserve this order,
// NOT the alphabetical order (K, M, N) that DictionaryAttr would produce.
// CHECK-LABEL: func.func @read_ordered_syms
normalform.module [#wave.normal_form<full_types>] {
  func.func @read_ordered_syms(%mem: !wave.tensor<[@M, @K, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64, K = 32, N = 128, T0 = 64}>} {
    // CHECK: wave.read
    // CHECK-SAME: ordered_syms = [#wave.symbol<"M">, #wave.symbol<"K">, #wave.symbol<"N">]
    %0 = wave.read %mem index [{
        M : [#wave.index_symbol<T0>] -> (T0, 1, 1),
        K : [#wave.index_symbol<T0>] -> (T0 * 2, 1, 1),
        N : [#wave.index_symbol<T0>] -> (T0 * 3, 8, 1)
    }] {elements_per_thread = 8 : i64} : (!wave.tensor<[@M, @K, @N] of f16, <global>>) -> vector<8xf16>
    return
  }
}

// -----

// Test that ordered_syms is set on wave.write ops.
// CHECK-LABEL: func.func @write_ordered_syms
normalform.module [#wave.normal_form<full_types>] {
  func.func @write_ordered_syms(%val: vector<8xf16>, %mem: !wave.tensor<[@M, @K, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64, K = 32, N = 128, T0 = 64}>} {
    // CHECK: wave.write
    // CHECK-SAME: ordered_syms = [#wave.symbol<"M">, #wave.symbol<"K">, #wave.symbol<"N">]
    wave.write %val, %mem index [{
        M : [#wave.index_symbol<T0>] -> (T0, 1, 1),
        K : [#wave.index_symbol<T0>] -> (T0 * 2, 1, 1),
        N : [#wave.index_symbol<T0>] -> (T0 * 3, 8, 1)
    }] {elements_per_thread = 8 : i64} : vector<8xf16>, !wave.tensor<[@M, @K, @N] of f16, <global>>
    return
  }
}
