// RUN: water-opt %s --water-test-wave-dialect-functions | FileCheck %s

// Technically these are matrix multiplications, but we really care about the iterators.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @scaled_mma_make_isolated
  // CHECK-SAME:  %[[ARG_A:.+]]: !wave.tensor<[@M, @K] of f4E2M1FN, <shared>>
  // CHECK-SAME:  %[[ARG_AS:.+]]: !wave.tensor<[@M, @K32] of f8E8M0FNU, <shared>>
  // CHECK-SAME:  %[[ARG_B:.+]]: !wave.tensor<[@N, @K] of f4E2M1FN, <shared>>
  // CHECK-SAME:  %[[ARG_BS:.+]]: !wave.tensor<[@N, @K32] of f8E8M0FNU, <shared>>
  func.func @scaled_mma_make_isolated(
      %a: !wave.tensor<[@M, @K] of f4E2M1FN, <shared>>,
      %a_scale: !wave.tensor<[@M, @K32] of f8E8M0FNU, <shared>>,
      %b: !wave.tensor<[@N, @K] of f4E2M1FN, <shared>>,
      %b_scale: !wave.tensor<[@N, @K32] of f8E8M0FNU, <shared>>,
      %c: !wave.tensor<[@M, @N] of f32, <global>>) {

    %0 = arith.constant 0.0 : f32
    %c_reg = wave.register %0 : !wave.tensor<[@M, @N] of f32>

    // CHECK:      wave.iterate
    // CHECK-SAME: captures(%[[ARG_A]], %[[ARG_AS]], %[[ARG_B]], %[[ARG_BS]])
    %mma_result = wave.iterate @K iter_args(%c_reg) attributes {wave_test.make_isolated} {
    // CHECK:      ^{{.*}}(%[[ACC:.+]]: !wave.tensor<[@M, @N] of f32>
    // CHECK-SAME: %[[INNER_A:.+]]: !wave.tensor<[@M, @K] of f4E2M1FN, <shared>>
    // CHECK-SAME: %[[INNER_AS:.+]]: !wave.tensor<[@M, @K32] of f8E8M0FNU, <shared>>
    // CHECK-SAME: %[[INNER_B:.+]]: !wave.tensor<[@N, @K] of f4E2M1FN, <shared>>
    // CHECK-SAME: %[[INNER_BS:.+]]: !wave.tensor<[@N, @K32] of f8E8M0FNU, <shared>>
    ^bb0(%acc: !wave.tensor<[@M, @N] of f32>):

      // CHECK: wave.read %[[INNER_A]]
      %a_reg = wave.read %a : (!wave.tensor<[@M, @K] of f4E2M1FN, <shared>>) -> !wave.tensor<[@M, @K] of f4E2M1FN>
      // CHECK: wave.read %[[INNER_AS]]
      %as_reg = wave.read %a_scale : (!wave.tensor<[@M, @K32] of f8E8M0FNU, <shared>>) -> !wave.tensor<[@M, @K32] of f8E8M0FNU>
      // CHECK: wave.read %[[INNER_B]]
      %b_reg = wave.read %b : (!wave.tensor<[@N, @K] of f4E2M1FN, <shared>>) -> !wave.tensor<[@N, @K] of f4E2M1FN>
      // CHECK: wave.read %[[INNER_BS]]
      %bs_reg = wave.read %b_scale : (!wave.tensor<[@N, @K32] of f8E8M0FNU, <shared>>) -> !wave.tensor<[@N, @K32] of f8E8M0FNU>

      %inner_acc = wave.scaled_mma %a_reg, %as_reg, %b_reg, %bs_reg, %acc
        {kind = #wave.mma_kind<f32_16x16x128_f8f6f4>}
        : (!wave.tensor<[@M, @K] of f4E2M1FN>, !wave.tensor<[@M, @K32] of f8E8M0FNU>,
           !wave.tensor<[@N, @K] of f4E2M1FN>, !wave.tensor<[@N, @K32] of f8E8M0FNU>,
           !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

      wave.yield %inner_acc : !wave.tensor<[@M, @N] of f32>
    } : (!wave.tensor<[@M, @N] of f32>)-> (!wave.tensor<[@M, @N] of f32>)

    wave.write %mma_result, %c : !wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32, <global>>

    return
  }
}

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @scaled_mma_make_non_isolated
  // CHECK-SAME:  %[[ARG_A:.+]]: !wave.tensor<[@M, @K] of f4E2M1FN, <shared>>
  // CHECK-SAME:  %[[ARG_AS:.+]]: !wave.tensor<[@M, @K32] of f8E8M0FNU, <shared>>
  // CHECK-SAME:  %[[ARG_B:.+]]: !wave.tensor<[@N, @K] of f4E2M1FN, <shared>>
  // CHECK-SAME:  %[[ARG_BS:.+]]: !wave.tensor<[@N, @K32] of f8E8M0FNU, <shared>>
  func.func @scaled_mma_make_non_isolated(
      %arg0: !wave.tensor<[@M, @K] of f4E2M1FN, <shared>>,
      %arg1: !wave.tensor<[@M, @K32] of f8E8M0FNU, <shared>>,
      %arg2: !wave.tensor<[@N, @K] of f4E2M1FN, <shared>>,
      %arg3: !wave.tensor<[@N, @K32] of f8E8M0FNU, <shared>>,
      %arg4: !wave.tensor<[@M, @N] of f32, <global>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = wave.register %cst : !wave.tensor<[@M, @N] of f32>

    // CHECK:     wave.iterate
    // CHECK-NOT: captures
    %1 = wave.iterate @K iter_args(%0) captures(%arg0, %arg1, %arg2, %arg3) attributes {wave_test.make_non_isolated} {
    // CHECK: ^{{.*}}(%{{.*}}: !wave.tensor<[@M, @N] of f32>):
    ^bb0(%acc: !wave.tensor<[@M, @N] of f32>,
         %cap_a: !wave.tensor<[@M, @K] of f4E2M1FN, <shared>>,
         %cap_as: !wave.tensor<[@M, @K32] of f8E8M0FNU, <shared>>,
         %cap_b: !wave.tensor<[@N, @K] of f4E2M1FN, <shared>>,
         %cap_bs: !wave.tensor<[@N, @K32] of f8E8M0FNU, <shared>>):

      // CHECK: wave.read %[[ARG_A]]
      %2 = wave.read %cap_a : (!wave.tensor<[@M, @K] of f4E2M1FN, <shared>>) -> !wave.tensor<[@M, @K] of f4E2M1FN>
      // CHECK: wave.read %[[ARG_AS]]
      %3 = wave.read %cap_as : (!wave.tensor<[@M, @K32] of f8E8M0FNU, <shared>>) -> !wave.tensor<[@M, @K32] of f8E8M0FNU>
      // CHECK: wave.read %[[ARG_B]]
      %4 = wave.read %cap_b : (!wave.tensor<[@N, @K] of f4E2M1FN, <shared>>) -> !wave.tensor<[@N, @K] of f4E2M1FN>
      // CHECK: wave.read %[[ARG_BS]]
      %5 = wave.read %cap_bs : (!wave.tensor<[@N, @K32] of f8E8M0FNU, <shared>>) -> !wave.tensor<[@N, @K32] of f8E8M0FNU>

      %6 = wave.scaled_mma %2, %3, %4, %5, %acc
        {kind = #wave.mma_kind<f32_16x16x128_f8f6f4>}
        : (!wave.tensor<[@M, @K] of f4E2M1FN>, !wave.tensor<[@M, @K32] of f8E8M0FNU>,
           !wave.tensor<[@N, @K] of f4E2M1FN>, !wave.tensor<[@N, @K32] of f8E8M0FNU>,
           !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
      wave.yield %6 : !wave.tensor<[@M, @N] of f32>
    } : (!wave.tensor<[@M, @N] of f32>,
         !wave.tensor<[@M, @K] of f4E2M1FN, <shared>>, !wave.tensor<[@M, @K32] of f8E8M0FNU, <shared>>,
         !wave.tensor<[@N, @K] of f4E2M1FN, <shared>>, !wave.tensor<[@N, @K32] of f8E8M0FNU, <shared>>) -> !wave.tensor<[@M, @N] of f32>
    wave.write %1, %arg4 : !wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32, <global>>
    return
  }
}
