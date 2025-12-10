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

// CHECK: #wave<index_mapping[#wave.iter<"A">, #wave.symbol<"B">] -> (_Iter_A + B, 2, 2)>
func.func private @iter_symbol_in_mapping() attributes { wave_test.index = #wave<index_mapping[#wave.iter<"A">, #wave.symbol<"B">] -> (_Iter_A + B, 2, 2)>}

// CHECK: #wave<index_mapping[#wave.symbol<"A">] -> (<NULL>, A, <NULL>)>
func.func private @null_exprs_in_mapping() attributes { wave_test.index = #wave<index_mapping[#wave.symbol<"A">] -> (<NULL>, A, <NULL>)>}

// CHECK: #wave.expr_list<[#wave.iter<"A">, #wave.symbol<"B">] -> (_Iter_A + B)>
func.func private @iter_symbol_in_expr() attributes { wave_test.index = #wave.expr_list<[#wave.iter<"A">, #wave.symbol<"B">] -> (_Iter_A + B)>}
