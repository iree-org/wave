// RUN: water-opt %s | water-opt | FileCheck %s
// RUN: water-opt %s --mlir-print-op-generic | water-opt | FileCheck %s

//===---------------------------------------------------------------------------
// The following tests basically check the presence of an operation in the
// dialect. We don't need to check the syntax as it is auto-generated and the
// generator is tested extensively.
//===---------------------------------------------------------------------------

// CHECK-LABEL: @mma
func.func @mma(%lhs: !wave.tensor<[@A, @B] of f16>, %rhs: !wave.tensor<[@B, @C] of f16>, %acc: !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32> {
  // CHECK: wave.mma
  %0 = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f16>, !wave.tensor<[@B, @C] of f16>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
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
  wave.register %0 : !wave.tensor<[@Y, @Z] of f32>
  return
}
