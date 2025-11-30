// RUN: water-opt %s -split-input-file -verify-diagnostics

// Test: empty group_id should fail
func.func @memory_access_pattern_empty_group_id(%value: !wave.tensor<any of f32, <register>>, %mem: !wave.tensor<[@M] of f32, <global>>) {
  wave.write %value, %mem {
    // expected-error @+1 {{group_id cannot be empty}}
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = false,
      group_id = ""
    >
  } : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M] of f32, <global>>
  return
}

// -----

// Test: LDS parameters specified when use_lds_promotion=false should fail
func.func @memory_access_pattern_lds_params_when_disabled(%value: !wave.tensor<any of f32, <register>>, %mem: !wave.tensor<[@M] of f32, <global>>) {
  wave.write %value, %mem {
    // expected-error @+1 {{LDS promotion parameters should not be specified when use_lds_promotion=false}}
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = false,
      group_id = "test",
      lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>] -> (WG0)>
    >
  } : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M] of f32, <global>>
  return
}

// -----

// Test: Partial LDS specification when use_lds_promotion=true should fail
func.func @memory_access_pattern_partial_lds_specification(%value: !wave.tensor<any of f32, <register>>, %mem: !wave.tensor<[@M] of f32, <global>>) {
  wave.write %value, %mem {
    // expected-error @+1 {{when LDS promotion is enabled, all LDS parameters must be specified: lds_block_global_base, lds_block_shape, lds_load_indices, lds_load_vector_sizes, global_store_indices}}
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = true,
      group_id = "test",
      lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>] -> (WG0)>
    >
  } : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M] of f32, <global>>
  return
}

// -----

// Test: Mismatched ranks between lds_block_global_base and lds_block_shape should fail
func.func @memory_access_pattern_mismatched_base_shape_ranks(%value: !wave.tensor<any of f32, <register>>, %mem: !wave.tensor<[@M, @N] of f32, <global>>) {
  wave.write %value, %mem {
    // expected-error @+1 {{lds_block_global_base rank (1) must match lds_block_shape rank (2)}}
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = true,
      group_id = "test",
      lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>] -> (WG0)>,
      lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">] -> (BLOCK_M, BLOCK_N)>,
      lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0)>,
      lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
      global_store_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0)>
    >
  } : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
  return
}

// -----

// Test: Mismatched ranks between lds_load_indices and lds_load_vector_sizes should fail
func.func @memory_access_pattern_mismatched_lds_load_ranks(%value: !wave.tensor<any of f32, <register>>, %mem: !wave.tensor<[@M] of f32, <global>>) {
  wave.write %value, %mem {
    // expected-error @+1 {{lds_load_indices rank (1) must match lds_load_vector_sizes rank (2)}}
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = true,
      group_id = "test",
      lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>] -> (WG0)>,
      lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
      lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0)>,
      lds_load_vector_sizes = #wave.expr_list<[#wave.symbol<"VEC_M">, #wave.symbol<"VEC_N">] -> (VEC_M, VEC_N)>,
      global_store_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0)>
    >
  } : !wave.tensor<any of f32, <register>>, !wave.tensor<[@M] of f32, <global>>
  return
}
