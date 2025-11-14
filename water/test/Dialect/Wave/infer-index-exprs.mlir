// RUN: water-opt %s --water-wave-infer-index-exprs --split-input-file --verify-diagnostics | FileCheck %s

// expected-error @below {{expects the root operation or its ancestor to guarantee the full_types normal for}}
module {
  func.func @normal_form() {
    return
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  func.func @simple_mma(%a: !wave.tensor<[@M, @K] of f16>,
                        %b: !wave.tensor<[@N, @K] of f16>,
                        %c: !wave.tensor<[@M, @N] of f32>) {
    // expected-error @below {{wave dialect operation without constraints on an ancestor}}
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  // expected-error @below {{expected a hardware constraint}}
  func.func @simple_mma(%a: !wave.tensor<[@M, @K] of f16>,
                        %b: !wave.tensor<[@N, @K] of f16>,
                        %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = []} {
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  // expected-error @below {{expected a waves_per_block entry with three elements in the hardware constraint}}
  func.func @simple_mma(%a: !wave.tensor<[@M, @K] of f16>,
                        %b: !wave.tensor<[@N, @K] of f16>,
                        %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64>
  ]} {
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}


// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  // CHECK: @simple_mma
  func.func @simple_mma(%a: !wave.tensor<[@M, @K] of f16>,
                        %b: !wave.tensor<[@N, @K] of f16>,
                        %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.mma
    // Left-hand side
    // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    // CHECK-DAG:  K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 1)
    // CHECK: }, {
    // Right-hand side
    // CHECK-DAG:  K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 1)
    // CHECK-DAG:  N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    // CHECK: }, {
    // Accumulator
    // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG:  N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    // CHECK: }, {
    // Result (matches the accumulator)
    // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG:  N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)}
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  // CHECK: @simple_mma_with_reads_and_write
  func.func @simple_mma_with_reads_and_write(%a: !wave.tensor<[@M, @K] of f16>,
                                             %b: !wave.tensor<[@N, @K] of f16>,
                                             %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.read
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    // CHECK-DAG: K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 1)
    %a_read = wave.read %a : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16, <register>>
    // CHECK: wave.read
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    // CHECK-DAG: K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 1)
    %b_read = wave.read %b : (!wave.tensor<[@N, @K] of f16>) -> !wave.tensor<[@N, @K] of f16, <register>>
    %cst = arith.constant 0.0 : f32
    // CHECK: wave.register
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    %c_reg = wave.register %cst : !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.mma
    // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    // CHECK-DAG:  K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 1)
    // CHECK: }, {
    // CHECK-DAG:  K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 1)
    // CHECK-DAG:  N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    // CHECK: }, {
    // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG:  N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    // CHECK: }, {
    // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG:  N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    %mma = wave.mma %a_read, %b_read, %c_reg {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.write
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 * 2 + _T2 * 6) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 * 2 + _T2 * 6) mod 16, 1, 1)
    wave.write %mma, %c : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  // CHECK-LABEL: @mma_chain
  func.func @mma_chain(%a: !wave.tensor<[@M, @K] of f16>,
                       %b: !wave.tensor<[@N, @K] of f16>,
                       %c: !wave.tensor<[@M, @P] of f32>,
                       %d: !wave.tensor<[@P, @N] of f16>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [1, 2, 2]>
  ]} {
    // CHECK: wave.read
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK-DAG: K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 1)
    %a_read = wave.read %a
      : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16, <register>>
    // CHECK: wave.read
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK-DAG: K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 1)
    %b_read = wave.read %b
      : (!wave.tensor<[@N, @K] of f16>) -> !wave.tensor<[@N, @K] of f16, <register>>
    %cst_0 = arith.constant 0.0 : f32
    // CHECK: wave.register
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    %c_reg = wave.register %cst_0
      : !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.mma
    // CHECK-DAG: K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 1)
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK: }, {
    // CHECK-DAG: K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 1)
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK: }, {
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK: }, {
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    %mma1 = wave.mma %a_read, %b_read, %c_reg {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.cast
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 1)
    %mma1_casted = wave.cast %mma1 { target_element_type = f16 }
      : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[@M, @N] of f16, <register>>

    // Second read and register
    // CHECK: wave.read
    // CHECK-DAG: P : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 1)
    %d_read = wave.read %d
      : (!wave.tensor<[@P, @N] of f16>) -> !wave.tensor<[@P, @N] of f16, <register>>
    %cst_1 = arith.constant 0.0 : f32
    // CHECK: wave.register
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG: P : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    %c_reg2 = wave.register %cst_1
      : !wave.tensor<[@M, @P] of f32, <register>>
    // CHECK: wave.mma
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 1)
    // CHECK: }, {
    // CHECK-DAG: N : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 1)
    // CHECK-DAG: P : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK: }, {
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG: P : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    // CHECK: }, {
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG: P : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    %mma2 = wave.mma %mma1_casted, %d_read, %c_reg2 {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @N] of f16, <register>>, !wave.tensor<[@P, @N] of f16, <register>>, !wave.tensor<[@M, @P] of f32, <register>>) -> !wave.tensor<[@M, @P] of f32, <register>>

    // CHECK: wave.write
    // CHECK-DAG: M : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 16) * 4, 4, 16)
    // CHECK-DAG: P : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 16, 1, 1)
    wave.write %mma2, %c : !wave.tensor<[@M, @P] of f32, <register>>, !wave.tensor<[@M, @P] of f32>
    return
  }
}

// -----

module attributes { wave.normal_form = #wave.normal_form<full_types> } {
  // Technically this is a matrix multiplication, but we really care about the iterators.
  func.func @iterate(%a: !wave.tensor<[@M, @K] of bf16, <shared>>,
                     %b: !wave.tensor<[@N, @K] of bf16, <shared>>,
                     %c: !wave.tensor<[@M, @N] of f32, <global>>)
    attributes { wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64,
                               waves_per_block = [1, 2, 2]>
    ]} {

    // CHECK: %[[CST:.*]] = arith.constant
    %0 = arith.constant 0.0 : f32

    // CHECK:      wave.register
    // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> (((_GPR_NUM floordiv 4) * 8) mod 32 + (((_T0 + _T1 + _T2 * 2) mod 64) floordiv 32) * 4 + _GPR_NUM mod 4, 16, 32)
    // CHECK-DAG:  N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 32, 1, 1)
    %c_reg = wave.register %0 : !wave.tensor<[@M, @N] of f32>

    // CHECK:      wave.iterate
    // CHECK-SAME: iter_args
    %mma_result = wave.iterate @K iter_args(%c_reg) {
      ^bb0(%acc: !wave.tensor<[@M, @N] of f32>):

        // CHECK:      wave.read
        // CHECK-DAG:  K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 32) * 8, 8, 1)
        // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 32, 1, 1)
        %a_reg = wave.read %a : (!wave.tensor<[@M, @K] of bf16, <shared>>) -> !wave.tensor<[@M, @K] of bf16>

        // CHECK:      wave.read
        // CHECK-DAG:  K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 32) * 8, 8, 1)
        // CHECK-DAG:  N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 32, 1, 1)
        %b_reg = wave.read %b : (!wave.tensor<[@N, @K] of bf16, <shared>>) -> !wave.tensor<[@N, @K] of bf16>

        // CHECK:      wave.mma
        // CHECK-DAG:  K : [_T0, _T1, _T2, _GPR_NUM] -> ((((_T0 + _T1 + _T2 * 2) mod 64) floordiv 32) * 8, 8, 1)
        // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 32, 1, 1)
        %inner_acc = wave.mma %a_reg, %b_reg, %acc {kind = #wave.mma_kind<f32_32x32x16_bf16>} :
          (!wave.tensor<[@M, @K] of bf16>, !wave.tensor<[@N, @K] of bf16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

        // CHECK:      wave.yield
        wave.yield %inner_acc : !wave.tensor<[@M, @N] of f32>
    } : (!wave.tensor<[@M, @N] of f32>)-> (!wave.tensor<[@M, @N] of f32>)

    // CHECK:      wave.write
    // CHECK-DAG:  M : [_T0, _T1, _T2, _GPR_NUM] -> (((_GPR_NUM floordiv 4) * 8) mod 32 + (((_T0 + _T1 + _T2 * 2) mod 64) floordiv 32) * 4 + _GPR_NUM mod 4, 16, 32)
    // CHECK-DAG:  N : [_T0, _T1, _T2, _GPR_NUM] -> ((_T0 + _T1 + _T2 * 2) mod 32, 1, 1)
    wave.write %mma_result, %c : !wave.tensor<[@M, @N] of f32> , !wave.tensor<[@M, @N] of f32, <global>>

    return
  }
}
