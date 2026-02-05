// RUN: water-opt %s --water-wave-infer-index-exprs --allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s

// expected-error @below {{expects the root operation or its ancestor to guarantee the full_types normal form}}
normalform.module [] {
  func.func @normal_form() {
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
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

normalform.module [#wave.normal_form<full_types>] {
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

normalform.module [#wave.normal_form<full_types>] {
  // expected-error @below {{expected either waves_per_block in the hardware constraint or wave constraints on an ancestor op}}
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

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK: }, {
    // Right-hand side
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Accumulator
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Result (matches the accumulator)
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  // CHECK: @simple_mma_with_reads_and_write
  func.func @simple_mma_with_reads_and_write(%a: !wave.tensor<[@M, @K] of f16>,
                                             %b: !wave.tensor<[@N, @K] of f16>,
                                             %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.read
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    %a_read = wave.read %a : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16, <register>>
    // CHECK: wave.read
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    %b_read = wave.read %b : (!wave.tensor<[@N, @K] of f16>) -> !wave.tensor<[@N, @K] of f16, <register>>
    %cst = arith.constant 0.0 : f32
    // CHECK: wave.register
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %c_reg = wave.register %cst : !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.mma
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %mma = wave.mma %a_read, %b_read, %c_reg {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    wave.write %mma, %c : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

// MMA only, with wave constraints and workgroup constraints for M, N, K.

normalform.module [#wave.normal_form<full_types>] {
  // CHECK: @mma_with_workgroup_constraints
  func.func @mma_with_workgroup_constraints(%a: !wave.tensor<[@M, @K] of f16>,
                                            %b: !wave.tensor<[@N, @K] of f16>,
                                            %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64>,
    #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
    #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>,
    #wave.workgroup_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>, workgroup_dim = <z>>,
    #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 4)>>,
    #wave.wave_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N floordiv 4)>>,
    #wave.wave_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K floordiv 4)>>
  ], wave.hyperparameters = #wave.hyperparameters<{M = 256 : i64, N = 256 : i64, K = 128 : i64, BLOCK_M = 64 : i64, BLOCK_N = 64 : i64, BLOCK_K = 32 : i64}>} {
    // CHECK: wave.mma
    // CHECK-SAME: index
    // CHECK-DAG: K : <[#wave.index_symbol<WG2>, #wave.index_symbol<T0>, #wave.index_symbol<T2>, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 16) * 4 + WG2 * BLOCK_K + T2 * (BLOCK_K floordiv 4), 4, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (T0 mod 16 + WG0 * BLOCK_M + (T0 floordiv 64) * (BLOCK_M floordiv 4), 1, 1)>
    // CHECK: }, {
    // CHECK-DAG: K : <[#wave.index_symbol<WG2>, #wave.index_symbol<T0>, #wave.index_symbol<T2>, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 16) * 4 + WG2 * BLOCK_K + T2 * (BLOCK_K floordiv 4), 4, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (T0 mod 16 + WG1 * BLOCK_N + T1 * (BLOCK_N floordiv 4), 1, 1)>
    // CHECK: }, {
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (((T0 mod 64) floordiv 16) * 4 + WG0 * BLOCK_M + (T0 floordiv 64) * (BLOCK_M floordiv 4), 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (T0 mod 16 + WG1 * BLOCK_N + T1 * (BLOCK_N floordiv 4), 1, 1)>
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----


// Two MMAs in a row. We need to store to the temporary storage and
// load back because of the index (layout) change.

normalform.module [#wave.normal_form<full_types>] {
  // CHECK-LABEL: @mma_chain
  func.func @mma_chain(%a: !wave.tensor<[@M, @K] of f16>,
                       %b: !wave.tensor<[@N, @K] of f16>,
                       %c: !wave.tensor<[@M, @P] of f32>,
                       %d: !wave.tensor<[@P, @N] of f16>,
                       %storage: !wave.tensor<[@M, @N] of f16>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [1, 2, 2]>
  ]} {
    // CHECK: wave.read
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %a_read = wave.read %a
      : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16, <register>>
    // CHECK: wave.read
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %b_read = wave.read %b
      : (!wave.tensor<[@N, @K] of f16>) -> !wave.tensor<[@N, @K] of f16, <register>>
    %cst_0 = arith.constant 0.0 : f32
    // CHECK: wave.register
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %c_reg = wave.register %cst_0
      : !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.mma
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %mma1 = wave.mma %a_read, %b_read, %c_reg {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // CHECK: wave.cast
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %mma1_casted = wave.cast %mma1
      : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[@M, @N] of f16, <register>>

    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    wave.write %mma1_casted, %storage : !wave.tensor<[@M, @N] of f16, <register>>, !wave.tensor<[@M, @N] of f16>
    // CHECK: wave.read
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    %reloaded = wave.read %storage : (!wave.tensor<[@M, @N] of f16>) -> !wave.tensor<[@M, @N] of f16, <register>>

    // Second read and register
    // CHECK: wave.read
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: P : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %d_read = wave.read %d
      : (!wave.tensor<[@P, @N] of f16>) -> !wave.tensor<[@P, @N] of f16, <register>>
    %cst_1 = arith.constant 0.0 : f32
    // CHECK: wave.register
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: P : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %c_reg2 = wave.register %cst_1
      : !wave.tensor<[@M, @P] of f32, <register>>
    // CHECK: wave.mma
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK: }, {
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: P : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: P : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: P : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %mma2 = wave.mma %reloaded, %d_read, %c_reg2 {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @N] of f16, <register>>, !wave.tensor<[@P, @N] of f16, <register>>, !wave.tensor<[@M, @P] of f32, <register>>) -> !wave.tensor<[@M, @P] of f32, <register>>

    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: P : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    wave.write %mma2, %c : !wave.tensor<[@M, @P] of f32, <register>>, !wave.tensor<[@M, @P] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  // CHECK: @mma_32x32x8_f16
  func.func @mma_32x32x8_f16(%a: !wave.tensor<[@M, @K] of f16>,
                             %b: !wave.tensor<[@N, @K] of f16>,
                             %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.mma
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4, 4, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4, 4, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_32x32x8_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  // CHECK: @mma_16x16x32_f16
  func.func @mma_16x16x32_f16(%a: !wave.tensor<[@M, @K] of f16>,
                              %b: !wave.tensor<[@N, @K] of f16>,
                              %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.mma
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 8, 8, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 8, 8, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x32_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  // CHECK: @mma_16x16x32_k4_f8
  func.func @mma_16x16x32_k4_f8(%a: !wave.tensor<[@M, @K] of f8E5M2>,
                                %b: !wave.tensor<[@N, @K] of f8E5M2>,
                                %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.mma
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> ((GPR_NUM floordiv 4) * 16 + ((T0 mod 64) floordiv 16) * 4 + GPR_NUM mod 4, 8, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> ((GPR_NUM floordiv 4) * 16 + ((T0 mod 64) floordiv 16) * 4 + GPR_NUM mod 4, 8, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x32_k4_f8>}
      : (!wave.tensor<[@M, @K] of f8E5M2>, !wave.tensor<[@N, @K] of f8E5M2>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  // CHECK: @mma_32x32x16_f16
  func.func @mma_32x32x16_f16(%a: !wave.tensor<[@M, @K] of f16>,
                              %b: !wave.tensor<[@N, @K] of f16>,
                              %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.mma
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 8, 8, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 8, 8, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_32x32x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  // CHECK: @mma_32x32x16_k4_f8
  func.func @mma_32x32x16_k4_f8(%a: !wave.tensor<[@M, @K] of f8E5M2>,
                                %b: !wave.tensor<[@N, @K] of f8E5M2>,
                                %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.mma
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> ((GPR_NUM floordiv 4) * 8 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 8, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> ((GPR_NUM floordiv 4) * 8 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 8, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_32x32x16_k4_f8>}
      : (!wave.tensor<[@M, @K] of f8E5M2>, !wave.tensor<[@N, @K] of f8E5M2>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
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
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    %c_reg = wave.register %0 : !wave.tensor<[@M, @N] of f32>

    // CHECK:      wave.iterate
    // CHECK-SAME: iter_args
    // CHECK-SAME: index
    // CHECK-DAG:  M = #wave.index_mapping<[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    %mma_result = wave.iterate @K iter_args(%c_reg) {
      ^bb0(%acc: !wave.tensor<[@M, @N] of f32>):

        // CHECK:      wave.read
        // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 8, 8, 1)>
        // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
        %a_reg = wave.read %a : (!wave.tensor<[@M, @K] of bf16, <shared>>) -> !wave.tensor<[@M, @K] of bf16>

        // CHECK:      wave.read
        // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 8, 8, 1)>
        // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
        %b_reg = wave.read %b : (!wave.tensor<[@N, @K] of bf16, <shared>>) -> !wave.tensor<[@N, @K] of bf16>

        // CHECK:      wave.mma
        // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 8, 8, 1)>
        // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
        %inner_acc = wave.mma %a_reg, %b_reg, %acc {kind = #wave.mma_kind<f32_32x32x16_bf16>} :
          (!wave.tensor<[@M, @K] of bf16>, !wave.tensor<[@N, @K] of bf16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

        // CHECK:      wave.yield
        wave.yield %inner_acc : !wave.tensor<[@M, @N] of f32>
    } : (!wave.tensor<[@M, @N] of f32>)-> (!wave.tensor<[@M, @N] of f32>)

    // CHECK:      wave.write
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    wave.write %mma_result, %c : !wave.tensor<[@M, @N] of f32> , !wave.tensor<[@M, @N] of f32, <global>>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @unregistered_noprop(%a: !wave.tensor<[@M, @K] of f16>,
                             %b: !wave.tensor<[@N, @K] of f16>,
                             %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                             waves_per_block = [1, 1, 1]>
  ]} {
    // expected-error @below {{cannot propagate index expressions across an operation not implementing the wave infer index expressions interface}}
    %modified_a = "test.unregistered"(%a) : (!wave.tensor<[@M, @K] of f16>) ->  !wave.tensor<[@M, @K] of f16>
    %result = wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>} :
        (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
      -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

// Cannot propagate for only pure operations in absence of MMA/writes/reductions.

normalform.module [#wave.normal_form<full_types>] {
  func.func @simple_add(%a: !wave.tensor<[@M, @K] of f16>,
                        %b: !wave.tensor<[@M, @K] of f16>)
      -> !wave.tensor<[@M, @K] of f16>
      attributes {
        wave.constraints = [
          #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
        ]
      } {

    // expected-error @below {{failed to infer index expressions for result #0}}
    %add = wave.add %a, %b : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %add : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

// There is no inference source here so we can't infer.

normalform.module [#wave.normal_form<full_types>] {
  func.func @failed_to_infer_write(%src: !wave.tensor<[@M, @N] of f32>, %dst: !wave.tensor<[@M, @N] of f32, <global>>)
    attributes { wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]} {

    // expected-error @below {{failed to infer index expressions for value to store}}
    wave.write %src, %dst : !wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32, <global>>

    return
  }
}

// -----

// There is no inference source here so we can't infer.

normalform.module [#wave.normal_form<full_types>] {
  func.func @failed_to_infer_binop(%a: !wave.tensor<[@M, @N] of f32>, %b: !wave.tensor<[@M, @N] of f32>)
    -> !wave.tensor<[@M, @N] of f32>
    attributes { wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]} {

    // expected-error @below {{failed to infer index expressions for result #0}}
    %sum = wave.add %a, %b : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

    return %sum : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  // expected-error @below {{unsupported constraint type: #wave.device_constraint}}
  func.func @empty() attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>,
      #wave.device_constraint<dim = <"M">, tile_size = <[] -> (42)>, device_dim = 0>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 100}>
  } {
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @allocate_register_index(
    %a: !wave.tensor<[@M, @K] of f16>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    %cst = arith.constant 1.0 : f16
    // CHECK: wave.register
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    %b_reg = wave.register %cst : !wave.tensor<[@N, @K] of f16, <register>>
    %cst0 = arith.constant 0.0 : f32
    %c_reg = wave.register %cst0 : !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.allocate
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %alloc = wave.allocate {distributed_shape = #wave.expr_list<[] -> (42)>}
      : !wave.tensor<[@M, @N] of f32, <shared>>
    %c = wave.mma %a, %b_reg, %c_reg { kind = #wave.mma_kind<f32_16x16x16_f16> }
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %c, %alloc : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <shared>>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @tiling_constraints_in_iter(
    %x: !wave.tensor<[@M, @K] of f16>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>,
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>,
      #wave.tiling_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{K = 128 : i64, M = 256 : i64, N = 1024 : i64, BLOCK_K = 32 : i64, BLOCK_M = 64 : i64}>
  } {
    // The computation below is bogus and is generally dead code (writing to a local allocation),
    // its point is to only access the values defined inside the loop so no propagation of
    // tiling constraints through block arguments is possible.
    wave.iterate @K iter_args() {
      %cst = arith.constant 1.0 : f16
      // CHECK: wave.register
      // CHECK: K : <[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 16) * 4 + _Iter_K * BLOCK_K, 4, 1)>
      %a_reg = wave.register %cst : !wave.tensor<[@M, @K] of f16, <register>>
      %b_reg = wave.register %cst : !wave.tensor<[@N, @K] of f16, <register>>
      // CHECK: wave.register
      // CHECK: K : <[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 16) * 4 + _Iter_K * BLOCK_K, 4, 1)>
      %cst0 = arith.constant 0.0 : f32
      %c_reg = wave.register %cst0 : !wave.tensor<[@M, @N] of f32, <register>>
      %alloc = wave.allocate {distributed_shape = #wave.expr_list<[] -> (42)>}
        : !wave.tensor<[@M, @N] of f32, <shared>>
      // CHECK: wave.mma
      // CHECK: K : <[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 16) * 4 + _Iter_K * BLOCK_K, 4, 1)>
      %c = wave.mma %a_reg, %b_reg, %c_reg { kind = #wave.mma_kind<f32_16x16x16_f16> }
        : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
      wave.write %c, %alloc : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <shared>>
      wave.yield
    } : () -> ()

    // Tiling constraints must not appear outside of the loop.
    %cst = arith.constant 0.0 : f16
    // CHECK: wave.register
    // CHECK-NOT: wave.iter<"K">
    // CHECK-NOT: _Iter_K * BLOCK_K
    %y = wave.register %cst : !wave.tensor<[@N, @K] of f16, <register>>
    %cst0 = arith.constant 0.0 : f32
    %z_reg = wave.register %cst0 : !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.mma
    // CHECK-NOT: wave.iter<"K">
    // CHECK-NOT: _Iter_K * BLOCK_K
    %z = wave.mma %x, %y, %z_reg {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32>
    return %z : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @tiling_constraints_in_nested_iter(
    %x: !wave.tensor<[@M, @K] of f16>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>,
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>,
      #wave.tiling_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{K = 128 : i64, M = 256 : i64, N = 1024 : i64, BLOCK_K = 32 : i64, BLOCK_M = 64 : i64}>
  } {
    wave.iterate @K iter_args() {
      wave.iterate @M iter_args() {
        %cst = arith.constant 1.0 : f16
        // CHECK: wave.register
        // CHECK-DAG: K : <[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 16) * 4 + _Iter_K * BLOCK_K, 4, 1)>
        // CHECK-DAG: M : <[#wave.index_symbol<T0>, #wave.iter<"M">, #wave.symbol<"BLOCK_M">] -> (T0 mod 16 + _Iter_M * BLOCK_M, 1, 1)>
        %a_reg = wave.register %cst : !wave.tensor<[@M, @K] of f16, <register>>
        %b_reg = wave.register %cst : !wave.tensor<[@N, @K] of f16, <register>>
        // CHECK: wave.register
        // CHECK: K : <[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 16) * 4 + _Iter_K * BLOCK_K, 4, 1)>
        %cst0 = arith.constant 0.0 : f32
        %c_reg = wave.register %cst0 : !wave.tensor<[@M, @N] of f32, <register>>
        %alloc = wave.allocate {distributed_shape = #wave.expr_list<[] -> (42)>}
          : !wave.tensor<[@M, @N] of f32, <shared>>
        // CHECK: wave.mma
        // CHECK-DAG: K : <[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 16) * 4 + _Iter_K * BLOCK_K, 4, 1)>
        // CHECK-DAG: M : <[#wave.index_symbol<T0>, #wave.iter<"M">, #wave.symbol<"BLOCK_M">] -> (T0 mod 16 + _Iter_M * BLOCK_M, 1, 1)>
        %c = wave.mma %a_reg, %b_reg, %c_reg { kind = #wave.mma_kind<f32_16x16x16_f16> }
          : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
        wave.write %c, %alloc : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <shared>>
        wave.yield
      } : () -> ()

      // Tiling constraints must not appear outside of the, but this concerns only the inner loop here.
      %cst = arith.constant 0.0 : f16
      // CHECK: wave.register
      // CHECK: wave.iter<"K">
      // CHECK: _Iter_K * BLOCK_K
      // CHECK-NOT: wave.iter<"M">
      // CHECK-NOT: _Iter_M * BLOCK_M
      %y = wave.register %cst : !wave.tensor<[@N, @K] of f16, <register>>
      %cst0 = arith.constant 0.0 : f32
      %z_reg = wave.register %cst0 : !wave.tensor<[@M, @N] of f32, <register>>
      // CHECK: wave.mma
      // CHECK: wave.iter<"K">
      // CHECK: _Iter_K * BLOCK_K
      // CHECK-NOT: wave.iter<"M">
      // CHECK-NOT: _Iter_M * BLOCK_M
      %z = wave.mma %x, %y, %z_reg {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32>
      wave.yield
    } : () -> ()
    return
  }
}

// -----

// Test broadcast propagates index expressions (identity propagation).
normalform.module [#wave.normal_form<full_types>] {
  // CHECK-LABEL: @broadcast_index_exprs
  func.func @broadcast_index_exprs(
      %a: !wave.tensor<[@M, @K] of f16>,
      %b: !wave.tensor<[@N, @K] of f16>,
      %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // First do an MMA to establish index expressions.
    // CHECK: wave.mma
    %mma = wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // Broadcast the MMA result from [@M, @N] to [@M, @N, @P] - should propagate M and N's index exprs.
    // CHECK: wave.broadcast
    // CHECK-DAG: M : <[#wave.index_symbol<T0>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>]
    %broadcasted = wave.broadcast %mma
      : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N, @P] of f32, <register>>

    return
  }
}

// -----

// Test broadcast gets index expression for broadcasted dimension via backward propagation.
normalform.module [#wave.normal_form<full_types>] {
  // CHECK-LABEL: @broadcast_index_exprs_backward
  func.func @broadcast_index_exprs_backward(
      %a: !wave.tensor<[@M, @K] of f16>,
      %b: !wave.tensor<[@N, @K] of f16>,
      %c: !wave.tensor<[@M, @N] of f32>,
      %d: !wave.tensor<[@P, @K] of f16>,
      %e: !wave.tensor<[@M, @P] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // First MMA establishes index expressions for @M and @N.
    %mma1 = wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // Second MMA establishes index expressions for @M and @P.
    %mma2 = wave.mma %a, %d, %e {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@P, @K] of f16>, !wave.tensor<[@M, @P] of f32>) -> !wave.tensor<[@M, @P] of f32, <register>>

    // Broadcast mma1 result from [@M, @N] to [@M, @N, @P].
    // @M and @N get index exprs from mma1 (forward propagation).
    // @P gets index expr from mma2 via the add operation (backward propagation).
    // CHECK: wave.broadcast
    // CHECK-DAG: M : <[#wave.index_symbol<T0>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>]
    // CHECK-DAG: P : <[#wave.index_symbol<T0>]
    %broadcasted = wave.broadcast %mma1
      : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N, @P] of f32, <register>>

    // Broadcast mma2 result from [@M, @P] to [@M, @N, @P] to match.
    %broadcasted2 = wave.broadcast %mma2
      : (!wave.tensor<[@M, @P] of f32, <register>>) -> !wave.tensor<[@M, @N, @P] of f32, <register>>

    // Add requires matching shapes - index exprs propagate between operands.
    %add = wave.add %broadcasted, %broadcasted2
      : (!wave.tensor<[@M, @N, @P] of f32, <register>>, !wave.tensor<[@M, @N, @P] of f32, <register>>) -> !wave.tensor<[@M, @N, @P] of f32, <register>>

    return
  }
}
