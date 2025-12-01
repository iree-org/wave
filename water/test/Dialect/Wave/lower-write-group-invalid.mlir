// RUN: water-opt %s --split-input-file --lower-wave-to-mlir --verify-diagnostics

// Test: Operations with the same group_id in different scopes should fail lowering
module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
// expected-error @+1 {{failed to convert starting at this operation}}
func.func @test_different_scope_same_group(%cond: i1, %mem1: !wave.tensor<[@M] of f32, <global>>, %mem2: !wave.tensor<[@M] of f32, <global>>) attributes {
  wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1024, M = 1024}>
} {
  %cst = arith.constant 0.0 : f32
  %waveValue1 = wave.register %cst : vector<4xf32>
  %waveValue2 = wave.register %cst : vector<4xf32>

  scf.if %cond {
    // This operation is inside the scf.if scope
    // expected-error @+1 {{failed to legalize operation 'wave.write' that was explicitly marked illegal}}
    wave.write %waveValue1, %mem1 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "scope_violation_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 4)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (4)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 4)>
        >
    } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>
  }

  // expected-error @+1 {{LDS promotion group 'scope_violation_group' contains operations in different scopes. All operations with the same group_id must be in the same block for correct LDS barrier semantics.}}
  wave.write %waveValue2, %mem2 index [{
      M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4 + 256, 4, 1)
    }] {
    memory_access_pattern = #wave.memory_access_pattern<
      use_lds_promotion = true,
      group_id = "scope_violation_group",
      lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
      lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
      lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 4)>,
      lds_load_vector_sizes = #wave.expr_list<[] -> (4)>,
      global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 4 + 256)>
      >
  } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

  return
}
}
