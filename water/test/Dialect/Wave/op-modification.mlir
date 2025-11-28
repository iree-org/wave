// RUN: water-opt %s --water-test-wave-dialect-functions | FileCheck %s

// Technically these are matrix multiplications, but we really care about the iterators.

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  // CHECK-LABEL: @make_isolated
  // CHECK-SAME:  %[[ARG_A:.+]]: !wave.tensor<[@M, @K] of bf16, <shared>>
  // CHECK-SAME:  %[[ARG_B:.+]]: !wave.tensor<[@N, @K] of bf16, <shared>>
  func.func @make_isolated(%a: !wave.tensor<[@M, @K] of bf16, <shared>>,
                           %b: !wave.tensor<[@N, @K] of bf16, <shared>>,
                           %c: !wave.tensor<[@M, @N] of f32, <global>>) {

    %0 = arith.constant 0.0 : f32
    %c_reg = wave.register %0 : !wave.tensor<[@M, @N] of f32>

    // CHECK:      wave.iterate
    // CHECK-SAME: captures(%[[ARG_A]], %[[ARG_B]])
    %mma_result = wave.iterate @K iter_args(%c_reg) attributes {wave_test.make_isolated} {
    // CHECK:      ^{{.*}}(%[[ACC:.+]]: !wave.tensor<[@M, @N] of f32>
    // CHECK-SAME: %[[INNER_A:.+]]: !wave.tensor<[@M, @K] of bf16, <shared>>
    // CHECK-SAME: %[[INNER_B:.+]]: !wave.tensor<[@N, @K] of bf16, <shared>>
    ^bb0(%acc: !wave.tensor<[@M, @N] of f32>):

      // CHECK: wave.read %[[INNER_A]]
      %a_reg = wave.read %a : (!wave.tensor<[@M, @K] of bf16, <shared>>) -> !wave.tensor<[@M, @K] of bf16>
      // CHECK: wave.read %[[INNER_B]]
      %b_reg = wave.read %b : (!wave.tensor<[@N, @K] of bf16, <shared>>) -> !wave.tensor<[@N, @K] of bf16>

      %inner_acc = wave.mma %a_reg, %b_reg, %acc {kind = #wave.mma_kind<f32_32x32x16_bf16>} :
        (!wave.tensor<[@M, @K] of bf16>, !wave.tensor<[@N, @K] of bf16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

      wave.yield %inner_acc : !wave.tensor<[@M, @N] of f32>
    } : (!wave.tensor<[@M, @N] of f32>)-> (!wave.tensor<[@M, @N] of f32>)

    wave.write %mma_result, %c : !wave.tensor<[@M, @N] of f32> , !wave.tensor<[@M, @N] of f32, <global>>

    return
  }
}

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  // CHECK-LABEL: @make_non_isolated
  // CHECK-SAME:  %[[ARG_A:.+]]: !wave.tensor<[@M, @K] of bf16, <shared>>
  // CHECK-SAME:  %[[ARG_B:.+]]: !wave.tensor<[@N, @K] of bf16, <shared>>
  func.func @make_non_isolated(%arg0: !wave.tensor<[@M, @K] of bf16, <shared>>, %arg1: !wave.tensor<[@N, @K] of bf16, <shared>>, %arg2: !wave.tensor<[@M, @N] of f32, <global>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = wave.register %cst : !wave.tensor<[@M, @N] of f32>

    // CHECK:     wave.iterate
    // CHECK-NOT: captures
    %1 = wave.iterate @K iter_args(%0) captures(%arg0, %arg1) attributes {wave_test.make_non_isolated} {
    // CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@M, @N] of f32>):
    ^bb0(%arg3: !wave.tensor<[@M, @N] of f32>, %arg4: !wave.tensor<[@M, @K] of bf16, <shared>>, %arg5: !wave.tensor<[@N, @K] of bf16, <shared>>):

      // CHECK: wave.read %[[ARG_A]]
      %2 = wave.read %arg4 : (!wave.tensor<[@M, @K] of bf16, <shared>>) -> !wave.tensor<[@M, @K] of bf16>
      // CHECK: wave.read %[[ARG_B]]
      %3 = wave.read %arg5 : (!wave.tensor<[@N, @K] of bf16, <shared>>) -> !wave.tensor<[@N, @K] of bf16>

      %4 = wave.mma %2, %3, %arg3 {kind = #wave.mma_kind<f32_32x32x16_bf16>} : (!wave.tensor<[@M, @K] of bf16>, !wave.tensor<[@N, @K] of bf16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
      wave.yield %4 : !wave.tensor<[@M, @N] of f32>
    } : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @K] of bf16, <shared>>, !wave.tensor<[@N, @K] of bf16, <shared>>) -> !wave.tensor<[@M, @N] of f32>
    wave.write %1, %arg2 : !wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32, <global>>
    return
  }
}
