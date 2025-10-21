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
    index {
      M : [THREAD_ID, BLOCK_SIZE] -> (THREAD_ID floordiv BLOCK_SIZE, 1, 1),
      N : [THREAD_ID, BLOCK_SIZE] -> (THREAD_ID * BLOCK_SIZE + 42, 1, 1)
    }
    : !wave.tensor<[@M, @N] of f32, <register>>
  return
}

// CHECK-LABEL: @register_with_symbols_complex_index
func.func @register_with_symbols_complex_index() {
  %0 = arith.constant 0.0 : f32
  // CHECK: wave.register
  %register = wave.register %0
    index {
      B : [WG2, BLOCK_B] -> (WG2 * (BLOCK_B+BLOCK_B), BLOCK_B * (WG2+WG2), WG2 * BLOCK_B),
      M : [_WG0, BLOCK_M, _T0] -> (_WG0 * BLOCK_M + BLOCK_M * ((_T0 floordiv 64) floordiv 2) + _T0 mod 32, 1, 1),
      N : [_T1, BLOCK_N, _WG1, GPR_NUM, _T0] -> (_T1 * (BLOCK_N floordiv 2) + BLOCK_N * _WG1 + GPR_NUM mod 4 + ((GPR_NUM floordiv 4) mod 4) * 8 + ((_T0 mod 64) floordiv 32) * 4, 1, 1)
    }
    : !wave.tensor<[@B, @N, @M] of f32, <register>>
  return
}

// CHECK-LABEL: @register_with_symbols_empty_symbol_list
func.func @register_with_symbols_empty_symbol_list() {
  %0 = arith.constant 0.0 : f32
  // CHECK: wave.register
  %register = wave.register %0 index {B : [] -> (0, 1, 1)}
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
  %parent = wave.allocate { distributed_shape = #wave.expr<[BLOCK_M, BLOCK_K] -> (BLOCK_M, BLOCK_K + 4)>}
    : !wave.tensor<[@M, @N] of bf16, <shared>>

  %buf = wave.allocate in %parent : !wave.tensor<[@M, @N] of bf16, <shared>>
    { distributed_shape = #wave.expr<[BLOCK_M, BLOCK_K] -> (BLOCK_M, BLOCK_K + 4)>, offset = 64}
    : !wave.tensor<[@M, @N] of bf16, <shared>>

  return %buf : !wave.tensor<[@M, @N] of bf16, <shared>>
}

// CHECK-LABEL: @index_magic_symbols
func.func @index_magic_symbols(%mem: !wave.tensor<[@M] of f16, <global>>)
attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 32, BLOCK_N = 32, M = 128, N = 256}>}  {
  // CHECK: wave.read
  // CHECK: index
  // CHECK: _WG0
  // CHECK: _T0
  %0 = wave.read %mem index {
      M : [BLOCK_M, _WG0, _T0] -> (BLOCK_M * _WG0 + (BLOCK_M floordiv 2) * (_T0 floordiv 64) + _T0 mod 64, 1, 64),
      N : [_T1, _WG1, BLOCK_N] -> (_WG1 * BLOCK_N + (BLOCK_N floordiv 2) * _T1, BLOCK_N ceildiv 2, 1)}
    : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
  return
}

// CHECK-LABEL: @write_with_bounds
func.func @write_with_bounds(%memo: !wave.tensor<[@M] of f32>, %val: !wave.tensor<[@M] of f32, <register>>) {
  // CHECK:       wave.read_write_bounds
  wave.write %val, %memo { bounds = #wave.read_write_bounds<{ M = #wave.expr<[BLOCK_M] -> (BLOCK_M * 64)>}> } : !wave.tensor<[@M] of f32, <register>>, !wave.tensor<[@M] of f32>
  return
}
