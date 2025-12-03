// RUN: water-opt %s | water-opt | FileCheck %s
// RUN: water-opt %s --mlir-print-op-generic | water-opt | FileCheck %s

//===---------------------------------------------------------------------------
// The following tests basically check the presence of an operation in the
// dialect. We don't need to check the syntax as it is auto-generated and the
// generator is tested extensively.
//===---------------------------------------------------------------------------

// CHECK-LABEL: @mma
func.func @mma(%lhs: !wave.tensor<[@A, @B] of f16>, %rhs: !wave.tensor<[@C, @B] of f16>, %acc: !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32> {
  // CHECK: wave.mma
  %0 = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f16>, !wave.tensor<[@C, @B] of f16>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
  return %0 : !wave.tensor<[@A, @C] of f32>
}

// CHECK-LABEL: @extract_slice
func.func @extract_slice(%memory: !wave.tensor<[@A, @B] of f16>) -> !wave.tensor<[@A, @B] of f16> {
  // CHECK: wave.extract_slice
  %0 = wave.extract_slice %memory {offset = #wave.expr_list<[] -> (3)>, size = #wave.expr_list<[] -> (32)>, stride = #wave.expr_list<[] -> (2)>} : (!wave.tensor<[@A, @B] of f16>) -> !wave.tensor<[@A, @B] of f16>
  return %0 : !wave.tensor<[@A, @B] of f16>
}

// CHECK-LABEL: @unary
func.func @unary(%value: !wave.tensor<[@A, @B] of bf16>) -> !wave.tensor<[@A, @B] of bf16> {
  // CHECK: wave.exp2
  %0 = wave.exp2 %value : (!wave.tensor<[@A, @B] of bf16>) -> !wave.tensor<[@A, @B] of bf16>
  return %0 : !wave.tensor<[@A, @B] of bf16>
}

// CHECK-LABEL: @binary
func.func @binary(%lhs: !wave.tensor<[@A, @B] of bf16>, %rhs: !wave.tensor<any of bf16>) -> !wave.tensor<[@A, @B] of bf16> {
  // CHECK: wave.add
  %0 = wave.add %lhs, %rhs : (!wave.tensor<[@A, @B] of bf16>, !wave.tensor<any of bf16>) -> !wave.tensor<[@A, @B] of bf16>
  // CHECK: wave.mul
  %1 = wave.mul %0, %rhs : (!wave.tensor<[@A, @B] of bf16>, !wave.tensor<any of bf16>) -> !wave.tensor<[@A, @B] of bf16>
  // CHECK: wave.div
  %2 = wave.div %1, %lhs : (!wave.tensor<[@A, @B] of bf16>, !wave.tensor<[@A, @B] of bf16>) -> !wave.tensor<[@A, @B] of bf16>
  return %2 : !wave.tensor<[@A, @B] of bf16>
}

// CHECK-LABEL: @memory
func.func @memory(%mem: !wave.tensor<[@X] of f32, <global>>) {
  // CHECK: wave.read
  %0 = wave.read %mem : (!wave.tensor<[@X] of f32, <global>>) -> !wave.tensor<any of f32, <register>>
  // CHECK: wave.write
  wave.write %0, %mem : !wave.tensor<any of f32, <register>>, !wave.tensor<[@X] of f32, <global>>
  return
}

//===---------------------------------------------------------------------------
// Operations with custom syntax elements.
//===---------------------------------------------------------------------------

// CHECK-LABEL: @iterate
func.func @iterate(%input0: !wave.tensor<any of f32>, %input1: !wave.tensor<any of bf16>)
    -> (!wave.tensor<any of f32>, !wave.tensor<any of bf16>) {
  // Note that it is okay to have compatible but not equal types along internal
  // control flow edges, i.e., the types of block arguments don't need to
  // exactly match those of operands and the types of the terminator results
  // don't need to exactly match those of iterate results as long as one of them
  // is underspecified.
  //
  // CHECK: wave.iterate @I iter_args
  // CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@A] of f32>, %{{.*}}: !wave.tensor<[@B] of bf16>):
  %iter_result:2 = wave.iterate @I iter_args(%input0, %input1) {
  ^bb0(%in_arg0: !wave.tensor<[@A] of f32>, %in_arg1: !wave.tensor<[@B] of bf16>):
    // CHECK: wave.yield %{{.*}}, %{{.*}} : !wave.tensor<[@A] of f32>, !wave.tensor<[@B] of bf16>
    wave.yield %in_arg0, %in_arg1 : !wave.tensor<[@A] of f32>, !wave.tensor<[@B] of bf16>
  } : (!wave.tensor<any of f32>, !wave.tensor<any of bf16>) -> (!wave.tensor<any of f32>, !wave.tensor<any of bf16>)
  return %iter_result#0, %iter_result#1 : !wave.tensor<any of f32>, !wave.tensor<any of bf16>
}

// CHECK-LABEL: @register
func.func @register() {
  %0 = arith.constant 0.0 : f32
  // CHECK: wave.register %{{.*}} : !wave.tensor<[@Y, @Z] of f32>
  %register = wave.register %0 : !wave.tensor<[@Y, @Z] of f32>
  return
}

// CHECK-LABEL: @register_with_symbols
func.func @register_with_symbols() {
  %0 = arith.constant 0.0 : f32
  // CHECK: wave.register
  %register = wave.register %0
    index [{
      M : [#wave.symbol<"THREAD_ID">, #wave.symbol<"BLOCK_SIZE">] -> (THREAD_ID floordiv BLOCK_SIZE, 1, 1),
      N : [#wave.symbol<"THREAD_ID">, #wave.symbol<"BLOCK_SIZE">] -> (THREAD_ID * BLOCK_SIZE + 42, 1, 1)
    }]
    : !wave.tensor<[@M, @N] of f32, <register>>
  return
}

// CHECK-LABEL: @register_with_symbols_complex_index
func.func @register_with_symbols_complex_index() {
  %0 = arith.constant 0.0 : f32
  // CHECK: wave.register
  %register = wave.register %0
    index [{
      B : [#wave.index_symbol<WG2>, #wave.symbol<"BLOCK_B">] -> (WG2 * (BLOCK_B+BLOCK_B), BLOCK_B * (WG2+WG2), WG2 * BLOCK_B),
      M : [#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">, #wave.index_symbol<T0>] -> (WG0 * BLOCK_M + BLOCK_M * ((T0 floordiv 64) floordiv 2) + T0 mod 32, 1, 1),
      N : [#wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">, #wave.index_symbol<WG1>, #wave.index_symbol<GPR_NUM>, #wave.index_symbol<T0>] -> (T1 * (BLOCK_N floordiv 2) + BLOCK_N * WG1 + GPR_NUM mod 4 + ((GPR_NUM floordiv 4) mod 4) * 8 + ((T0 mod 64) floordiv 32) * 4, 1, 1)
    }]
    : !wave.tensor<[@B, @N, @M] of f32, <register>>
  return
}

// CHECK-LABEL: @register_with_symbols_empty_symbol_list
func.func @register_with_symbols_empty_symbol_list() {
  %0 = arith.constant 0.0 : f32
  // CHECK: wave.register
  %register = wave.register %0 index [{B : [] -> (0, 1, 1)}]
    : !wave.tensor<[@B] of f32, <register>>
  return
}

// CHECK-LABEL: @register_with_hyperparameter
// CHECK-SAME:  wave.hyperparameters<{A = 100 : i64, B = 10 : i64}>
func.func @register_with_hyperparameter() attributes {hyperparameters = #wave.hyperparameters<{A = 100, B = 10}>} {
  %0 = arith.constant 0.0 : f32
  // CHECK: wave.register
  %register = wave.register %0
    : !wave.tensor<[@A, @B] of f32, <register>>
  return
}

// CHECK-LABEL: @allocate
func.func @allocate() -> !wave.tensor<[@M, @N] of bf16, <shared>> {
  // CHECK: wave.allocate
  %parent = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>}
    : !wave.tensor<[@M, @N] of bf16, <shared>>

  %buf = wave.allocate in %parent : !wave.tensor<[@M, @N] of bf16, <shared>>
    { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>, offset = 64}
    : !wave.tensor<[@M, @N] of bf16, <shared>>

  return %buf : !wave.tensor<[@M, @N] of bf16, <shared>>
}

// CHECK-LABEL: @index_magic_symbols
func.func @index_magic_symbols(%mem: !wave.tensor<[@M] of f16, <global>>)
attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 32, BLOCK_N = 32, M = 128, N = 256}>}  {
  // CHECK: wave.read
  // CHECK: index
  // CHECK: #wave.index_symbol<WG0>
  // CHECK: #wave.index_symbol<T0>
  %0 = wave.read %mem index [{
      M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 64, 1, 64),
      N : [#wave.index_symbol<T1>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 2) * T1, BLOCK_N ceildiv 2, 1)}]
    : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
  return
}

// CHECK-LABEL: @write_with_bounds
func.func @write_with_bounds(%memo: !wave.tensor<[@M] of f32>, %val: !wave.tensor<[@M] of f32, <register>>) {
  // CHECK:       wave.read_write_bounds
  wave.write %val, %memo { bounds = #wave.read_write_bounds<{ M = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M * 64)>}> } : !wave.tensor<[@M] of f32, <register>>, !wave.tensor<[@M] of f32>
  return
}

// CHECK-LABEL: @cast_wave_tensor
func.func @cast_wave_tensor(%arg0: !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of bf16> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : !wave.tensor<[@M, @N] of f32> to !wave.tensor<[@M, @N] of bf16>
  return %0 : !wave.tensor<[@M, @N] of bf16>
}

// CHECK-LABEL: @cast_wave_tensor_int_to_float
func.func @cast_wave_tensor_int_to_float(%arg0: !wave.tensor<[@A, @B] of i32>) -> !wave.tensor<[@A, @B] of f32> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : !wave.tensor<[@A, @B] of i32> to !wave.tensor<[@A, @B] of f32>
  return %0 : !wave.tensor<[@A, @B] of f32>
}

// CHECK-LABEL: @cast_vector
func.func @cast_vector(%arg0: vector<4xf32>) -> vector<4xbf16> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : vector<4xf32> to vector<4xbf16>
  return %0 : vector<4xbf16>
}

// CHECK-LABEL: @cast_wave_tensor_underspecified
func.func @cast_wave_tensor_underspecified(%arg0: !wave.tensor<any of f32>) -> !wave.tensor<any of bf16> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : !wave.tensor<any of f32> to !wave.tensor<any of bf16>
  return %0 : !wave.tensor<any of bf16>
}

// CHECK-LABEL: @cast_wave_tensor_with_index
func.func @cast_wave_tensor_with_index(%arg0: !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f16> {
  // CHECK: wave.cast
  // CHECK-SAME: index
  %0 = wave.cast %arg0 index [{
    M : [#wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (T0 * BLOCK_M, 1, 1),
    N : [#wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (T1 * BLOCK_N, 1, 1)
  }] : !wave.tensor<[@M, @N] of f32> to !wave.tensor<[@M, @N] of f16>
  return %0 : !wave.tensor<[@M, @N] of f16>
}

// CHECK-LABEL: @cast_int_types
func.func @cast_int_types(%arg0: !wave.tensor<[@X, @Y] of i8>) -> !wave.tensor<[@X, @Y] of i32> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : !wave.tensor<[@X, @Y] of i8> to !wave.tensor<[@X, @Y] of i32>
  return %0 : !wave.tensor<[@X, @Y] of i32>
}

// CHECK-LABEL: @cast_float_types
func.func @cast_float_types(%arg0: !wave.tensor<[@A] of f64>) -> !wave.tensor<[@A] of f16> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : !wave.tensor<[@A] of f64> to !wave.tensor<[@A] of f16>
  return %0 : !wave.tensor<[@A] of f16>
}

// CHECK-LABEL: @cast_vector_int
func.func @cast_vector_int(%arg0: vector<8xi32>) -> vector<8xi8> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : vector<8xi32> to vector<8xi8>
  return %0 : vector<8xi8>
}

// CHECK-LABEL: @cast_single_element_vector
func.func @cast_single_element_vector(%arg0: vector<1xf32>) -> vector<1xbf16> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : vector<1xf32> to vector<1xbf16>
  return %0 : vector<1xbf16>
}

// CHECK-LABEL: @cast_wave_tensor_1d
func.func @cast_wave_tensor_1d(%arg0: !wave.tensor<[@K] of f32>) -> !wave.tensor<[@K] of i32> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : !wave.tensor<[@K] of f32> to !wave.tensor<[@K] of i32>
  return %0 : !wave.tensor<[@K] of i32>
}

// CHECK-LABEL: @cast_wave_tensor_3d
func.func @cast_wave_tensor_3d(%arg0: !wave.tensor<[@X, @Y, @Z] of bf16>) -> !wave.tensor<[@X, @Y, @Z] of f32> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : !wave.tensor<[@X, @Y, @Z] of bf16> to !wave.tensor<[@X, @Y, @Z] of f32>
  return %0 : !wave.tensor<[@X, @Y, @Z] of f32>
}

// CHECK-LABEL: @cast_mixed_specified
func.func @cast_mixed_specified(%arg0: !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<any of bf16> {
  // CHECK: wave.cast
  %0 = wave.cast %arg0 : !wave.tensor<[@M, @N] of f32> to !wave.tensor<any of bf16>
  return %0 : !wave.tensor<any of bf16>
}
