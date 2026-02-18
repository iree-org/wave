// RUN: water-opt %s | FileCheck %s

// CHECK: #wave.symbol<"A">
func.func private @attr() attributes { test.foo = #wave.symbol<"A"> }

// CHECK: !wave.tensor<[@A, @B] of bf16>
func.func private @foo() -> !wave.tensor<[@A, @B] of bf16>

// CHECK: !wave.tensor<any of bf16>
func.func private @unspecified_tensor() -> !wave.tensor<any of bf16>

// CHECK: !wave.tensor<any of i32, <global>>
func.func private @address_space() -> !wave.tensor<any of i32, <global>>

// CHECK: !wave.tensor<any of i32, <global>>
func.func private @address_space_full() -> !wave.tensor<any of i32, #wave.address_space<global>>

// CHECK: !wave.tensor<any of i8>
func.func private @address_space_default() -> !wave.tensor<any of i8, <unspecified>>

// CHECK: #wave.index_mapping<[#wave.iter<"A">, #wave.symbol<"B">] -> (_Iter_A + B, 2, 2)>
func.func private @iter_symbol_in_mapping() attributes { wave_test.index = #wave.index_mapping<[#wave.iter<"A">, #wave.symbol<"B">] -> (_Iter_A + B, 2, 2)>}

// CHECK: #wave.index_mapping<[#wave.symbol<"A">] -> (<NULL>, A, <NULL>)>
func.func private @null_exprs_in_mapping() attributes { wave_test.index = #wave.index_mapping<[#wave.symbol<"A">] -> (<NULL>, A, <NULL>)>}

// CHECK: #wave.index_mapping<[#wave.symbol<"B">] -> (B, 0, 0)>
func.func private @mapping_with_unused_symbols() attributes { wave_test.index = #wave.index_mapping<[#wave.symbol<"A">, #wave.symbol<"B">, #wave.symbol<"C">] -> (B, 0, 0)>}

// Nit: we do ceildiv 2 on the first value to avoid having a subexpression directly
// adding two symbols since there is a pesky "canonicalization" that would reorder them
// based on commutativity of addition inside the affine expression constructor.
// CHECK: #wave.index_mapping<
// CHECK:   #wave.index_symbol<WG0>
// CHECK:   #wave.index_symbol<T0>
// CHECK:   #wave.index_symbol<T1>
// CHECK:   #wave.index_symbol<T2>
// CHECK:   #wave.iter<"I">
// CHECK:   #wave.iter<"J">
// CHECK:   #wave.symbol<"A">
// CHECK:   #wave.symbol<"B">
// CHECK: B ceildiv 2 + A + _Iter_J + _Iter_I + T2 + T0 + T1 + WG0
func.func private @mapping_reorder_symbols() attributes { wave_test.index = #wave.index_mapping<[
    #wave.symbol<"B">, #wave.symbol<"A">,
    #wave.iter<"J">, #wave.iter<"I">,
    #wave.index_symbol<T2>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.index_symbol<WG0>] -> (B ceildiv 2 + A + _Iter_J + _Iter_I + T2 + T0 + T1 + WG0, 1, 1)>}

// CHECK: #wave.index_mapping<
// CHECK:   #wave.index_symbol<WG0>
// CHECK:   #wave.index_symbol<T2>
// CHECK:   #wave.iter<"I">
// CHECK:   #wave.symbol<"B">
// CHECK: B ceildiv 2 + _Iter_I + T2 + WG0
func.func private @mapping_reorder_and_unused_symbols() attributes { wave_test.index = #wave.index_mapping<[
    #wave.symbol<"B">, #wave.symbol<"A">,
    #wave.iter<"J">, #wave.iter<"I">,
    #wave.index_symbol<T2>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.index_symbol<WG0>] -> ((B ceildiv 2) + _Iter_I + T2 + WG0, 1, 1)>}


// CHECK: #wave.index_mapping<
// CHECK:   #wave.symbol<"A">
// CHECK:   #wave.symbol<"B">
// CHECK: A ceildiv B + B - A
func.func private @mapping_duplicate_symbols() attributes { wave_test.index = #wave.index_mapping<[
    #wave.symbol<"B">, #wave.symbol<"A">,
    #wave.symbol<"A">, #wave.iter<"B">] -> (A ceildiv B + B - A, 1, 1)> }

// CHECK: #wave.expr_list<[#wave.iter<"A">, #wave.symbol<"B">] -> (_Iter_A + B)>
func.func private @iter_symbol_in_expr() attributes { wave_test.index = #wave.expr_list<[#wave.iter<"A">, #wave.symbol<"B">] -> (_Iter_A + B)>}

// CHECK-LABEL: func.func private @empty_mapping
// CHECK: test.map = #wave.symbol_mapping<>
func.func private @empty_mapping() attributes {
  test.map = #wave.symbol_mapping<>
}

// CHECK-LABEL: func.func private @single_entry
// CHECK: #wave.symbol_mapping<@M = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>>
func.func private @single_entry() attributes {
  test.map = #wave.symbol_mapping<
    @M = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>
  >
}

// CHECK-LABEL: func.func private @two_entries_ordering
// CHECK: #wave.symbol_mapping<@M = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, @N = #wave.expr_list<[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>>
func.func private @two_entries_ordering() attributes {
  test.map = #wave.symbol_mapping<
    @M = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
    @N = #wave.expr_list<[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>
  >
}

// CHECK-LABEL: func.func private @three_results_per_value
// CHECK: #wave.symbol_mapping<@M = #wave.expr_list<[#wave.symbol<"A">] -> (A, A + 1, A + 2)>>
func.func private @three_results_per_value() attributes {
  test.map = #wave.symbol_mapping<
    @M = #wave.expr_list<[#wave.symbol<"A">] -> (A, A + 1, A + 2)>
  >
}

// CHECK-LABEL: @index_mapping
// CHECK: #wave.symbol_mapping<@M = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M, 1, BLOCK_M)>, @N = #wave.index_mapping<[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, BLOCK_N)>>
func.func private @index_mapping() {
  water_test.wave_symbol_mapping {index_mapping = #wave.symbol_mapping<
    @M = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M, 1, BLOCK_M)>,
    @N = #wave.index_mapping<[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, BLOCK_N)>
  >}
  return
}
