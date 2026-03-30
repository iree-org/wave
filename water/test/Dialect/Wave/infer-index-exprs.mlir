// RUN: water-opt %s --water-wave-infer-index-exprs --allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s

// expected-error @below {{expects the root operation or its ancestor to guarantee the full_func_boundary normal form}}
normalform.module [] {
  func.func @normal_form() {
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

// MMA is the only Wave op with four per-value index / vector_shape slots.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @mma_emits_four_vector_shape_slots
  func.func @mma_emits_four_vector_shape_slots(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@N, @K] of f16>,
    %c: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<
        threads_per_wave = 64,
        waves_per_block = [2, 3, 4],
        mma_type = #wave.mma_kind<f32_16x16x16_f16>,
        vector_shapes = {M = 1 : i64, N = 1 : i64, K = 8 : i64}>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, K = 64}>
  } {
    // CHECK: wave.mma
    // Four per-value dicts: lhs (M,K), rhs (K,N), acc (M,N), result (M,N); MMA
    // kind fixes M/N/K vector sizes at 16 : i64 regardless of hardware hints.
    // CHECK-SAME: vector_shape [{K : 16 : i64, M : 16 : i64}, {K : 16 : i64, N : 16 : i64}, {M : 16 : i64, N : 16 : i64}, {M : 16 : i64, N : 16 : i64}]
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

// Batched (3D) MMA: batch dimension B is leading; M, N, K indexing is as in 2D.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK: @batched_mma
  func.func @batched_mma(%a: !wave.tensor<[@B, @M, @K] of f16>,
                         %b: !wave.tensor<[@B, @N, @K] of f16>,
                         %c: !wave.tensor<[@B, @M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.mma
    // LHS
    // CHECK-DAG:  B : <[] -> (0, 1, 1)>
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK: }, {
    // RHS
    // CHECK-DAG:  B : <[] -> (0, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Accumulator
    // CHECK-DAG:  B : <[] -> (0, 1, 1)>
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Result
    // CHECK-DAG:  B : <[] -> (0, 1, 1)>
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@B, @M, @K] of f16>, !wave.tensor<[@B, @N, @K] of f16>, !wave.tensor<[@B, @M, @N] of f32>) -> !wave.tensor<[@B, @M, @N] of f32>
    return
  }
}

// -----

// Make sure the tiling constraints apply to batch dimensions. Note that there
// are tests below for propagation across `wave.iterate`, here it is not the
// point.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK: @batched_mma_in_a_loop
  func.func @batched_mma_in_a_loop(%a: !wave.tensor<[@B, @M, @K] of f16>,
                                   %b: !wave.tensor<[@B, @N, @K] of f16>,
                                   %c: !wave.tensor<[@B, @M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>,
    #wave.tiling_constraint<dim = <"B">, tile_size = <[#wave.symbol<"BLOCK_B">] -> (BLOCK_B)>>
  ], wave.hyperparameters = #wave.hyperparameters<{BLOCK_B = 2 : i64, B = 10, M = 16, N = 16, K = 16}>} {
    // CHECK: wave.mma
    // LHS
    // CHECK-DAG:  B : <[#wave.iter<"B">, #wave.symbol<"BLOCK_B">] -> (_Iter_B * BLOCK_B, 1, 1)>
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK: }, {
    // RHS
    // CHECK-DAG:  B : <[#wave.iter<"B">, #wave.symbol<"BLOCK_B">] -> (_Iter_B * BLOCK_B, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Accumulator
    // CHECK-DAG:  B : <[#wave.iter<"B">, #wave.symbol<"BLOCK_B">] -> (_Iter_B * BLOCK_B, 1, 1)>
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Result
    // CHECK-DAG:  B : <[#wave.iter<"B">, #wave.symbol<"BLOCK_B">] -> (_Iter_B * BLOCK_B, 1, 1)>
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }
    wave.iterate @B {
    ^bb0:
      wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
        : (!wave.tensor<[@B, @M, @K] of f16>, !wave.tensor<[@B, @N, @K] of f16>, !wave.tensor<[@B, @M, @N] of f32>) -> !wave.tensor<[@B, @M, @N] of f32>
      wave.yield
    } : () -> ()
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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


// Two MMAs in a row. We store to the temporary storage and
// load back because of the index (layout) change.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @mma_chain_load_store
  func.func @mma_chain_load_store(%a: !wave.tensor<[@M, @K] of f16>,
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

// Test that index expressions don't propagate through chained MMA operations.
// Each MMA should maintain its own index expressions set during initialization,
// regardless of the index expressions from the previous MMA in the chain.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @mma_chain_without_propagation
  func.func @mma_chain_without_propagation(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@N, @K] of f16>,
    %c1: !wave.tensor<[@M, @N] of f32>,
    %d: !wave.tensor<[@M, @K] of f16>,
    %e: !wave.tensor<[@K, @N] of f16>,
    %c2: !wave.tensor<[@M, @K] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64,
                                waves_per_block = [2, 3, 4]>
    ]
  } {
    // First MMA operation
    // CHECK: wave.mma
    // LHS
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK: }, {
    // RHS
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Accumulator
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Result (matches the accumulator)
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %result1 = wave.mma %a, %b, %c1 {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
      -> !wave.tensor<[@M, @N] of f32>

    // Cast should have index expressions from the earlier mma because it is
    // given higher priority.
    // CHECK: wave.cast
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %result1_casted = wave.cast %result1
      : !wave.tensor<[@M, @N] of f32> to !wave.tensor<[@M, @N] of f16>

    // Second MMA operation using result from first MMA as operand.
    // The index expressions should be independently initialized for this MMA,
    // not propagated from %result1_casted
    // CHECK: wave.mma
    // LHS
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK: }, {
    // RHS
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Accumulator (using %result1 from first MMA)
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // Result (matches the accumulator)
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %result2 = wave.mma %result1_casted, %e, %c2 {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @N] of f16>, !wave.tensor<[@K, @N] of f16>, !wave.tensor<[@M, @K] of f32>)
      -> !wave.tensor<[@M, @K] of f32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

// Check that an unmapped dimension gets the default (0, 1, 1) index expression.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK: @mma_32x32x8_f16_3d
  func.func @mma_32x32x8_f16_3d(%a: !wave.tensor<[@B, @M, @K] of f16>,
                                %b: !wave.tensor<[@B, @N, @K] of f16>,
                                %c: !wave.tensor<[@B, @M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.mma
    // CHECK-DAG:  B : <[] -> (0, 1, 1)>
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4, 4, 1)>
    // CHECK: }, {
    // CHECK-DAG:  B : <[] -> (0, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4, 4, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  B : <[] -> (0, 1, 1)>
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  B : <[] -> (0, 1, 1)>
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> (((GPR_NUM floordiv 4) * 8) mod 32 + ((T0 mod 64) floordiv 32) * 4 + GPR_NUM mod 4, 16, 32)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 32, 1, 1)>
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_32x32x8_f16>}
      : (!wave.tensor<[@B, @M, @K] of f16>, !wave.tensor<[@B, @N, @K] of f16>, !wave.tensor<[@B, @M, @N] of f32>) -> !wave.tensor<[@B, @M, @N] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK: @scaled_mma_16x16x128_f8
  func.func @scaled_mma_16x16x128_f8(%a: !wave.tensor<[@M, @K] of f8E5M2>,
                              %a_scale: !wave.tensor<[@M, @K32] of f8E8M0FNU>,
                              %b: !wave.tensor<[@N, @K] of f8E5M2>,
                              %b_scale: !wave.tensor<[@N, @K32] of f8E8M0FNU>,
                              %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.scaled_mma
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> ((GPR_NUM floordiv 16) * 64 + ((T0 mod 64) floordiv 16) * 16 + GPR_NUM mod 16, 32, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K32 : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> ((GPR_NUM floordiv 16) * 64 + ((T0 mod 64) floordiv 16) * 16 + GPR_NUM mod 16, 32, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K32 : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 32, 1, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: vector_shape
    // CHECK-DAG:  K : 128
    // CHECK-DAG:  M : 16
    // CHECK: }, {
    // CHECK-DAG:  K32 : 4
    // CHECK-DAG:  M : 16
    // CHECK: }, {
    // CHECK-DAG:  K : 128
    // CHECK-DAG:  N : 16
    // CHECK: }, {
    // CHECK-DAG:  K32 : 4
    // CHECK-DAG:  N : 16
    // CHECK: }, {
    // CHECK-DAG:  M : 16
    // CHECK-DAG:  N : 16
    // CHECK: }, {
    // CHECK-DAG:  M : 16
    // CHECK-DAG:  N : 16
    %r = wave.scaled_mma %a, %a_scale, %b, %b_scale, %c {kind = #wave.mma_kind<f32_16x16x128_f8f6f4>}
      : (!wave.tensor<[@M, @K] of f8E5M2>, !wave.tensor<[@M, @K32] of f8E8M0FNU>,
         !wave.tensor<[@N, @K] of f8E5M2>, !wave.tensor<[@N, @K32] of f8E8M0FNU>,
         !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK: @scaled_mma_16x16x128_f4
  func.func @scaled_mma_16x16x128_f4(%a: !wave.tensor<[@M, @K] of f4E2M1FN>,
                              %a_scale: !wave.tensor<[@M, @K32] of f8E8M0FNU>,
                              %b: !wave.tensor<[@N, @K] of f4E2M1FN>,
                              %b_scale: !wave.tensor<[@N, @K32] of f8E8M0FNU>,
                              %c: !wave.tensor<[@M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // CHECK: wave.scaled_mma
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> ((GPR_NUM floordiv 16) * 64 + ((T0 mod 64) floordiv 16) * 16 + GPR_NUM mod 16, 32, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG:  K32 : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 32, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K : <[#wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>] -> ((GPR_NUM floordiv 16) * 64 + ((T0 mod 64) floordiv 16) * 16 + GPR_NUM mod 16, 32, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  K32 : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 32, 1, 1)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG:  M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG:  N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: vector_shape
    // CHECK-DAG:  K : 128
    // CHECK-DAG:  M : 16
    // CHECK: }, {
    // CHECK-DAG:  K32 : 4
    // CHECK-DAG:  M : 16
    // CHECK: }, {
    // CHECK-DAG:  K : 128
    // CHECK-DAG:  N : 16
    // CHECK: }, {
    // CHECK-DAG:  K32 : 4
    // CHECK-DAG:  N : 16
    // CHECK: }, {
    // CHECK-DAG:  M : 16
    // CHECK-DAG:  N : 16
    // CHECK: }, {
    // CHECK-DAG:  M : 16
    // CHECK-DAG:  N : 16
    %r = wave.scaled_mma %a, %a_scale, %b, %b_scale, %c {kind = #wave.mma_kind<f32_16x16x128_f8f6f4>}
      : (!wave.tensor<[@M, @K] of f4E2M1FN>, !wave.tensor<[@M, @K32] of f8E8M0FNU>,
         !wave.tensor<[@N, @K] of f4E2M1FN>, !wave.tensor<[@N, @K32] of f8E8M0FNU>,
         !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

// Pure operations get default thread-independent index exprs when no MMA/writes/reductions.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @simple_add
  func.func @simple_add(%a: !wave.tensor<[@M, @K] of f16>,
                        %b: !wave.tensor<[@M, @K] of f16>)
      -> !wave.tensor<[@M, @K] of f16>
      attributes {
        wave.constraints = [
          #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
        ]
      } {

    // CHECK: wave.add
    // CHECK-SAME: index [
    %add = wave.add %a, %b : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16>
    return %add : !wave.tensor<[@M, @K] of f16>
  }
}

// -----

// There is no inference source here so we can't infer.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @failed_to_infer_write(%src: !wave.tensor<[@M, @N] of f32>, %dst: !wave.tensor<[@M, @N] of f32, <global>>)
    attributes { wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]} {

    // expected-error @below {{failed to infer index expressions for value to store}}
    // expected-error @below {{the error above may be caused by the following: couldn't find vector shapes in the contiguity check}}
    wave.write %src, %dst : !wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32, <global>>

    return
  }
}

// -----

// Pure operations get default thread-independent index exprs.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @failed_to_infer_binop
  func.func @failed_to_infer_binop(%a: !wave.tensor<[@M, @N] of f32>, %b: !wave.tensor<[@M, @N] of f32>)
    -> !wave.tensor<[@M, @N] of f32>
    attributes { wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]} {

    // CHECK: wave.add
    // CHECK-SAME: index [
    %sum = wave.add %a, %b : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

    return %sum : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

// Tiling for iterator K must not be joined into index lattices of [@M, @N] tensors used
// inside `wave.iterate @K` (whether as explicit captures or as outer SSA values).
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @iterate_k_capture_without_k_dim
  func.func @iterate_k_capture_without_k_dim(
    %mask: !wave.tensor<[@M, @N] of f32, <global>>
  ) -> !wave.tensor<[@M, @N] of f32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>,
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{K = 128 : i64, M = 256 : i64, N = 1024 : i64, BLOCK_K = 32 : i64}>
  } {
    %cst = arith.constant 0.0 : f32
    %acc_init = wave.register %cst : !wave.tensor<[@M, @N] of f32>

    // CHECK: wave.iterate @K iter_args({{.*}}) attributes {index = [{M = #wave.index_mapping<[] -> (0, 1, 1)>, N = #wave.index_mapping<[] -> (0, 1, 1)>}]}
    %out = wave.iterate @K iter_args(%acc_init) captures(%mask) {
    ^bb0(%acc: !wave.tensor<[@M, @N] of f32>, %m: !wave.tensor<[@M, @N] of f32, <global>>):
      // CHECK: wave.read{{.*}}index [{M : <[] -> (0, 1, 1)>, N : <[] -> (0, 1, 1)>}]
      // CHECK-NOT: wave.iter<"K">
      // CHECK-NOT: _Iter_K
      %m_read = wave.read %m : (!wave.tensor<[@M, @N] of f32, <global>>) -> !wave.tensor<[@M, @N] of f32>
      // CHECK: wave.add{{.*}}index [{M : <[] -> (0, 1, 1)>, N : <[] -> (0, 1, 1)>}]
      %sum = wave.add %acc, %m_read : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
      wave.yield %sum : !wave.tensor<[@M, @N] of f32>
    } : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32, <global>>) -> (!wave.tensor<[@M, @N] of f32>)

    return %out : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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

    // Propagation-stopper heuristic prevents propagation from above because the
    // broadcast contains a dimension @P not included in the source vector shape.
    // CHECK: wave.broadcast
    // CHECK-DAG: M : <[] -> (0, 1, 1)>
    // CHECK-DAG: N : <[] -> (0, 1, 1)>
    %broadcasted = wave.broadcast %mma
      : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N, @P] of f32, <register>>

    return
  }
}

// -----

// Test broadcast gets index expression for broadcasted dimension via backward propagation.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
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
    // CHECK-DAG: M : <[] -> (0, 1, 1)>
    // CHECK-DAG: N : <[] -> (0, 1, 1)>
    // CHECK-DAG: P : <[] -> (0, 1, 1)>
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

// -----

// Backward propagation from the MMA should give the register only M's index
// expr, not N (the broadcast dimension).
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @broadcast_mma_back_to_register
  func.func @broadcast_mma_back_to_register(
      %a: !wave.tensor<[@M, @K] of f16>,
      %b: !wave.tensor<[@N, @K] of f16>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [2, 3, 4]>
  ]} {
    // Register has shape [M] only. Broadcast to [M, N] to use as MMA accumulator.
    %cst = arith.constant 0.0 : f32
    // CHECK: wave.register
    // Backward propagation from MMA: register gets M index expr only (not N).
    // CHECK:     M :
    // CHECK-NOT: N :
    %reg = wave.register %cst : !wave.tensor<[@M] of f32, <register>>

    // CHECK: wave.broadcast
    %acc = wave.broadcast %reg
      : (!wave.tensor<[@M] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // Broadcast result is the accumulator operand; index exprs propagate back
    // through it to the register.
    %mma = wave.mma %a, %b, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    return
  }
}

// -----

// Test permute forward propagation: index expressions flow from MMA result
// through permute to write. The permute swaps M and N dimensions, which should
// swap their strides.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @permute_propagation
  func.func @permute_propagation(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@N, @K] of f16>,
    %c: !wave.tensor<[@M, @N] of f32>,
    %dst: !wave.tensor<[@N, @M] of f32, <global>>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // MMA produces index expressions with:
    //   M: (((T0 mod 64) floordiv 16) * 4, 4, 16)  - stride 16
    //   N: (T0 mod 16, 1, 1)                        - stride 1
    // CHECK: wave.mma
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %mma_result = wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16>, !wave.tensor<[@N, @K] of f16>, !wave.tensor<[@M, @N] of f32>)
      -> !wave.tensor<[@M, @N] of f32>

    // After permute from [M, N] to [N, M], strides should be swapped:
    //   M: (((T0 mod 64) floordiv 16) * 4, 4, 1)   - stride becomes 1 (was N's stride)
    //   N: (T0 mod 16, 1, 16)                       - stride becomes 16 (was M's stride)
    // CHECK: wave.permute
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 16)>
    %permuted = wave.permute %mma_result
      : !wave.tensor<[@M, @N] of f32> to !wave.tensor<[@N, @M] of f32>

    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 16)>
    wave.write %permuted, %dst : !wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N, @M] of f32, <global>>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @propagate_from_write
  func.func @propagate_from_write(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>,
    %output: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 4 : i64, N = 1 : i64}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, BLOCK_M = 64 : i64, BLOCK_N = 64 : i64}>
  } {
    // CHECK: wave.read
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK-DAG: vector_shape [{M : 4 : i64, N : 1 : i64}]
    %a_reg = wave.read %a : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.read
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK-DAG: vector_shape [{M : 4 : i64, N : 1 : i64}]
    %b_reg = wave.read %b : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // CHECK: wave.add
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK-DAG: vector_shape [{M : 4 : i64, N : 1 : i64}]
    %sum = wave.add %a_reg, %b_reg : (!wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK-DAG: vector_shape [{M : 4 : i64, N : 1 : i64}]
    wave.write %sum, %output : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32>

    return
  }
}

// -----

// Elements per thread provided on the op used instead of the value inferred from workgroup constraints.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @propagate_from_write_explicit_ept
  func.func @propagate_from_write_explicit_ept(
    %output: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 4 : i64, N = 1 : i64}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, BLOCK_M = 64 : i64, BLOCK_N = 64 : i64}>
  } {
    %cst = arith.constant 0.0 : f32
    %reg = wave.register %cst : !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * 8 + WG0 * BLOCK_M, 8, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK-DAG: vector_shape [{M : 4 : i64, N : 1 : i64}]
    wave.write %reg, %output {elements_per_thread = 8} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32>

    return
  }
}

// -----

// Elements per thread is used for the trailing dimension because its vector shape is no longer 1.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @propagate_from_write_vector_shape
  func.func @propagate_from_write_vector_shape(
    %output: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 4 : i64, N = 16 : i64}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, BLOCK_M = 64 : i64, BLOCK_N = 64 : i64}>
  } {
    %cst = arith.constant 0.0 : f32
    %reg = wave.register %cst : !wave.tensor<[@M, @N] of f32, <register>>
    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (T0 mod 64 + WG0 * BLOCK_M, 1, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 8, 1)>
    // CHECK-SAME: vector_shape [{M : 4 : i64, N : 16 : i64}]
    wave.write %reg, %output {elements_per_thread = 8} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32>

    return
  }
}

// -----

// Test that unmapped dimensions get default (0, 1, 1) index expressions
// when there are no workgroup/wave/tiling constraints for them.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @unmapped_dimension_default
  func.func @unmapped_dimension_default(
    %a: !wave.tensor<[@B, @M, @N] of f32>,
    %output: !wave.tensor<[@B, @M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 4 : i64, N = 1 : i64}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{B = 8, M = 128, N = 128, BLOCK_M = 64 : i64, BLOCK_N = 64 : i64}>
  } {
    // Read should get default index expression for B dimension (no constraints)
    // and computed expressions for M and N dimensions
    // CHECK: wave.read
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK: vector_shape [{M : 4 : i64, N : 1 : i64}]
    %a_reg = wave.read %a : (!wave.tensor<[@B, @M, @N] of f32>) -> !wave.tensor<[@B, @M, @N] of f32, <register>>

    // Write should preserve the same index expressions
    // CHECK: wave.write
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK: vector_shape [{M : 4 : i64, N : 1 : i64}]
    wave.write %a_reg, %output : !wave.tensor<[@B, @M, @N] of f32, <register>>, !wave.tensor<[@B, @M, @N] of f32>

    return
  }
}

// -----

// Test priority-based propagation with multiple write operations.
// All writes should establish index expressions with the same priority,
// and the join should succeed since they agree.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @multiple_writes_consistent
  func.func @multiple_writes_consistent(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>,
    %out1: !wave.tensor<[@M, @N] of f32>,
    %out2: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 4 : i64, N = 1 : i64}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, BLOCK_M = 64 : i64, BLOCK_N = 64 : i64}>
  } {
    // CHECK: wave.read
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK: vector_shape [{M : 4 : i64, N : 1 : i64}]
    %a_reg = wave.read %a : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // CHECK: wave.read
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK: vector_shape [{M : 4 : i64, N : 1 : i64}]
    %b_reg = wave.read %b : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // CHECK: wave.add
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK: vector_shape [{M : 4 : i64, N : 1 : i64}]
    %sum = wave.add %a_reg, %b_reg : (!wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // Both writes establish the same index expressions
    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK: vector_shape [{M : 4 : i64, N : 1 : i64}]
    wave.write %sum, %out1 : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32>

    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((T0 mod 64) * (BLOCK_M ceildiv 64) + WG0 * BLOCK_M, BLOCK_M ceildiv 64, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    // CHECK: vector_shape [{M : 4 : i64, N : 1 : i64}]
    wave.write %a_reg, %out2 : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32>

    return
  }
}

// -----

// Test write when all dimension symbols are absent from constraints.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @write_all_dimensions_unmapped
  func.func @write_all_dimensions_unmapped(
    %a: !wave.tensor<[@P, @Q] of f32>,
    %output: !wave.tensor<[@P, @Q] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {P = 1 : i64, Q = 1 : i64}>
    ],
    wave.hyperparameters = #wave.hyperparameters<{P = 8 : i64, Q = 16 : i64}>
  } {
    // CHECK: wave.read
    // CHECK-DAG: P : <[] -> (0, 1, 1)>
    // CHECK-DAG: Q : <[] -> (0, 1, 1)>
    // CHECK: vector_shape [{P : 1 : i64, Q : 1 : i64}]
    %a_reg = wave.read %a : (!wave.tensor<[@P, @Q] of f32>) -> !wave.tensor<[@P, @Q] of f32, <register>>

    // CHECK: wave.write
    // CHECK-DAG: P : <[] -> (0, 1, 1)>
    // CHECK-DAG: Q : <[] -> (0, 1, 1)>
    // CHECK: vector_shape [{P : 1 : i64, Q : 1 : i64}]
    wave.write %a_reg, %output : !wave.tensor<[@P, @Q] of f32, <register>>, !wave.tensor<[@P, @Q] of f32>

    return
  }
}


// -----

// MMa index expression has higher priority than write.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @write_after_mma_priority
  func.func @write_after_mma_priority(
    %a: !wave.tensor<[@M, @K] of f16>,
    %b: !wave.tensor<[@N, @K] of f16>,
    %c: !wave.tensor<[@M, @N] of f32>,
    %output: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 4 : i64, N = 1 : i64}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>,
      #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>>,
      #wave.wave_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, K = 64, BLOCK_M = 64 : i64, BLOCK_N = 64 : i64}>
  } {
    // CHECK: wave.read
    // CHECK: vector_shape [{K : 16 : i64, M : 16 : i64}]
    %a_reg = wave.read %a : (!wave.tensor<[@M, @K] of f16>) -> !wave.tensor<[@M, @K] of f16, <register>>
    // CHECK: wave.read
    // CHECK: vector_shape [{K : 16 : i64, N : 16 : i64}]
    %b_reg = wave.read %b : (!wave.tensor<[@N, @K] of f16>) -> !wave.tensor<[@N, @K] of f16, <register>>
    %mma = wave.mma %a_reg, %b_reg, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32, <register>>
    // Write in isolation would have inferred step=floordiv(BLOCK_M, 64) since M is mapped
    // to workgroup X with 64 threads in it, but we obtain step=4 propagate from the mma
    // above, because that has higher priority.
    // CHECK: wave.write
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (((T0 mod 64) floordiv 16) * 4 + WG0 * BLOCK_M
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (T0 mod 16 + WG1 * BLOCK_N
    // CHECK: vector_shape [{M : 16 : i64, N : 16 : i64}]
    wave.write %mma, %output : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @batched_mma_with_reads_and_write
  func.func @batched_mma_with_reads_and_write(%a: !wave.tensor<[@B, @M, @K] of f16>,
                                              %b: !wave.tensor<[@B, @N, @K] of f16>,
                                              %c: !wave.tensor<[@B, @M, @N] of f32>)
  attributes { wave.constraints = [
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [1, 2, 2]>
  ]} {
    // CHECK: wave.read
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    %a_read = wave.read %a
      : (!wave.tensor<[@B, @M, @K] of f16>) -> !wave.tensor<[@B, @M, @K] of f16, <register>>
    // CHECK: wave.read
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %b_read = wave.read %b
      : (!wave.tensor<[@B, @N, @K] of f16>) -> !wave.tensor<[@B, @N, @K] of f16, <register>>
    %cst = arith.constant 0.0 : f32
    // CHECK: wave.register
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %c_reg = wave.register %cst
      : !wave.tensor<[@B, @M, @N] of f32, <register>>
    // CHECK: wave.mma
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK: }, {
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: K : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    // CHECK: }, {
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    %mma = wave.mma %a_read, %b_read, %c_reg {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@B, @M, @K] of f16, <register>>, !wave.tensor<[@B, @N, @K] of f16, <register>>, !wave.tensor<[@B, @M, @N] of f32, <register>>) -> !wave.tensor<[@B, @M, @N] of f32, <register>>
    // The write should get its index expressions from the MMA result via backward propagation.
    // CHECK: wave.write
    // CHECK-DAG: B : <[] -> (0, 1, 1)>
    // CHECK-DAG: M : <[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 16) * 4, 4, 16)>
    // CHECK-DAG: N : <[#wave.index_symbol<T0>] -> (T0 mod 16, 1, 1)>
    wave.write %mma, %c : !wave.tensor<[@B, @M, @N] of f32, <register>>, !wave.tensor<[@B, @M, @N] of f32>
    return
  }
}

// -----

// Make sure we write index expr initialization doesn't crash
// on rank-0 tensors.

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @write_rank0_tensor
  func.func @write_rank0_tensor(
    %src: !wave.tensor<[] of f32>,
    %dst: !wave.tensor<[] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1],
                                mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                vector_shapes = {M = 4 : i64}>
    ]
  } {
    // CHECK: wave.write
    wave.write %src, %dst : !wave.tensor<[] of f32>, !wave.tensor<[] of f32>
    return
  }
}

// -----

// Test that identity ops (e.g. add) get index expressions initialized from
// thread-independent constraints during forward init.
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @identity_init_forward
  func.func @identity_init_forward(
    %a: !wave.tensor<[@M, @N] of f32>,
    %b: !wave.tensor<[@M, @N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 4 : i64, N = 1 : i64}>,
      #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128, BLOCK_M = 64 : i64, BLOCK_N = 64 : i64}>
  } {
    // Add (identity trait) gets result index exprs from init (thread-independent constraints).
    // CHECK: wave.add
    // CHECK-DAG: M : <[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M, 1, 1)>
    // CHECK-DAG: N : <[#wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N, 1, 1)>
    %sum = wave.add %a, %b : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32, <register>>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  func.func @wrong_type_graceful_failure(
    %a: !wave.tensor<[@M] of f32>,
    %b: !wave.tensor<[@M] of f32>
  ) -> vector<4xf32> attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>
    ]
  } {
    // expected-error@below {{failed to infer index expressions for result #0}}
    %result = wave.add %a, %b {wave_test.override_operand_index = [
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 4 : i32}],
      [{M = #wave.index_mapping<[#wave.index_symbol<T0>] -> (T0 * 40, 1, 1)>}, {M = 4 : i32}]
    ]}
    : (!wave.tensor<[@M] of f32>, !wave.tensor<[@M] of f32>) -> vector<4xf32>
    return %result : vector<4xf32>
  }
}


// -----

// Test that reduction ops get index expressions initialized from thread-independent
// constraints for both result (forward) and operands (backward init).
normalform.module [#wave.normal_form<full_func_boundary>, #wave.normal_form<full_op_types>] {
  // CHECK-LABEL: @reduction_init
  func.func @reduction_init(
    %input: !wave.tensor<[@N, @M] of f32>,
    %init: !wave.tensor<[@N] of f32>
  ) attributes {
    wave.constraints = [
      #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>,
      #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <x>>
    ],
    wave.hyperparameters = #wave.hyperparameters<{N = 128, M = 64, BLOCK_N = 32 : i64}>
  } {
    // Reduction result has shape [@N]; init has shape [@N]; input has [@N, @M].
    // CHECK: wave.max_element
    // CHECK-DAG: N : <[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_N">] -> (WG0 * BLOCK_N, 1, 1)>
    %result = wave.max_element %input init(%init) <warp> : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32, <register>>
    return
  }
}
