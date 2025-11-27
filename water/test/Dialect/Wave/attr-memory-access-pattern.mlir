// RUN: water-opt %s -split-input-file | water-opt | FileCheck %s
// RUN: water-opt %s -split-input-file --mlir-print-op-generic | water-opt | FileCheck %s

// Test basic memory access pattern parsing without LDS promotion
// CHECK-LABEL: @memory_access_pattern_basic_no_promotion
func.func @memory_access_pattern_basic_no_promotion(%value: !wave.tensor<any of f32, <register>>, %mem: !wave.tensor<[@M] of f32, <global>>) {
  // CHECK: wave.write
  // CHECK-SAME: memory_access_pattern = #wave.memory_access_pattern<
  // CHECK-SAME:   use_lds_promotion = false,
  // CHECK-SAME:   group_id = "basic_test"
  // CHECK-SAME: >
  wave.write %value, %mem {
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = false,
      group_id = "basic_test"
    >
  } : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M] of f32, <global>>
  return
}

// -----

// Test memory access pattern with complete LDS promotion - 1D case with symbolic vector size
// CHECK-LABEL: @memory_access_pattern_1d_lds_promotion
func.func @memory_access_pattern_1d_lds_promotion(%value: !wave.tensor<any of f32, <register>>, %mem: !wave.tensor<[@M] of f32, <global>>) {
  // CHECK: wave.write
  // CHECK-SAME: memory_access_pattern = #wave.memory_access_pattern<
  // CHECK-SAME:   use_lds_promotion = true,
  // CHECK-SAME:   group_id = "lds_1d",
  // CHECK-SAME:   lds_block_global_base = <[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
  // CHECK-SAME:   lds_block_shape = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
  // CHECK-SAME:   lds_load_indices = <[#wave.index_symbol<T0>, #wave.symbol<"VEC_SIZE">] -> (T0 * VEC_SIZE)>,
  // CHECK-SAME:   lds_load_vector_sizes = <[#wave.symbol<"VEC_SIZE">] -> (VEC_SIZE)>,
  // CHECK-SAME:   global_store_indices = <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">, #wave.symbol<"VEC_SIZE">] -> (WG0 * BLOCK_M + T0 * VEC_SIZE)>,
  // CHECK-SAME:   global_store_vector_sizes = <[#wave.symbol<"VEC_SIZE">] -> (VEC_SIZE)>
  // CHECK-SAME: >
  wave.write %value, %mem {
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = true,
      group_id = "lds_1d",
      lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
      lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
      lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>, #wave.symbol<"VEC_SIZE">] -> (T0 * VEC_SIZE)>,
      lds_load_vector_sizes = #wave.expr_list<[#wave.symbol<"VEC_SIZE">] -> (VEC_SIZE)>,
      global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">, #wave.symbol<"VEC_SIZE">] -> (WG0 * BLOCK_M + T0 * VEC_SIZE)>,
      global_store_vector_sizes = #wave.expr_list<[#wave.symbol<"VEC_SIZE">] -> (VEC_SIZE)>
    >
  } : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M] of f32, <global>>
  return
}

// -----

// Test memory access pattern with constant vector size - 1D case
// CHECK-LABEL: @memory_access_pattern_1d_constant_vector_size
func.func @memory_access_pattern_1d_constant_vector_size(%value: !wave.tensor<any of f32, <register>>, %mem: !wave.tensor<[@M] of f32, <global>>) {
  // CHECK: wave.write
  // CHECK-SAME: memory_access_pattern = #wave.memory_access_pattern<
  // CHECK-SAME:   use_lds_promotion = true,
  // CHECK-SAME:   group_id = "lds_1d_const",
  // CHECK-SAME:   lds_block_global_base = <[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
  // CHECK-SAME:   lds_block_shape = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
  // CHECK-SAME:   lds_load_indices = <[#wave.index_symbol<T0>] -> (T0 * 64)>,
  // CHECK-SAME:   lds_load_vector_sizes = <[] -> (64)>,
  // CHECK-SAME:   global_store_indices = <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64)>,
  // CHECK-SAME:   global_store_vector_sizes = <[] -> (64)>
  // CHECK-SAME: >
  wave.write %value, %mem {
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = true,
      group_id = "lds_1d_const",
      lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
      lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
      lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
      lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
      global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64)>,
      global_store_vector_sizes = #wave.expr_list<[] -> (64)>
    >
  } : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M] of f32, <global>>
  return
}

// -----

// Test memory access pattern with complete LDS promotion - 2D case
// CHECK-LABEL: @memory_access_pattern_2d_lds_promotion
func.func @memory_access_pattern_2d_lds_promotion(%value: !wave.tensor<any of f32, <register>>, %mem: !wave.tensor<[@M, @N] of f32, <global>>) {
  // CHECK: wave.write
  // CHECK-SAME: memory_access_pattern = #wave.memory_access_pattern<
  // CHECK-SAME:   use_lds_promotion = true,
  // CHECK-SAME:   group_id = "lds_2d",
  // CHECK-SAME:   lds_block_global_base = <[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">] -> (WG0 * BLOCK_M, WG1 * BLOCK_N)>,
  // CHECK-SAME:   lds_block_shape = <[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">] -> (BLOCK_M, BLOCK_N)>,
  // CHECK-SAME:   lds_load_indices = <[#wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"VEC_SIZE_M">, #wave.symbol<"VEC_SIZE_N">] -> (T0 * VEC_SIZE_M, T1 * VEC_SIZE_N)>,
  // CHECK-SAME:   lds_load_vector_sizes = <[#wave.symbol<"VEC_SIZE_M">, #wave.symbol<"VEC_SIZE_N">] -> (VEC_SIZE_M, VEC_SIZE_N)>,
  // CHECK-SAME:   global_store_indices = <[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">, #wave.symbol<"VEC_SIZE_M">, #wave.symbol<"VEC_SIZE_N">] -> (WG0 * BLOCK_M + T0 * VEC_SIZE_M, WG1 * BLOCK_N + T1 * VEC_SIZE_N)>,
  // CHECK-SAME:   global_store_vector_sizes = <[#wave.symbol<"VEC_SIZE_M">, #wave.symbol<"VEC_SIZE_N">] -> (VEC_SIZE_M, VEC_SIZE_N)>
  // CHECK-SAME: >
  wave.write %value, %mem {
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = true,
      group_id = "lds_2d",
      lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">] -> (WG0 * BLOCK_M, WG1 * BLOCK_N)>,
      lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">] -> (BLOCK_M, BLOCK_N)>,
      lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"VEC_SIZE_M">, #wave.symbol<"VEC_SIZE_N">] -> (T0 * VEC_SIZE_M, T1 * VEC_SIZE_N)>,
      lds_load_vector_sizes = #wave.expr_list<[#wave.symbol<"VEC_SIZE_M">, #wave.symbol<"VEC_SIZE_N">] -> (VEC_SIZE_M, VEC_SIZE_N)>,
      global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">, #wave.symbol<"VEC_SIZE_M">, #wave.symbol<"VEC_SIZE_N">] -> (WG0 * BLOCK_M + T0 * VEC_SIZE_M, WG1 * BLOCK_N + T1 * VEC_SIZE_N)>,
      global_store_vector_sizes = #wave.expr_list<[#wave.symbol<"VEC_SIZE_M">, #wave.symbol<"VEC_SIZE_N">] -> (VEC_SIZE_M, VEC_SIZE_N)>
    >
  } : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
  return
}